# ChunkFwdH 设计

方案设计规则版本：V2。本次增量为 A5 双 head 跨 chunk 输入 lookahead，接口沿用 [api.md](api.md)。旧 Stage/数学/任务分工保持，以下第 10 节单独给出受影响的搬运、资源与同步证明；不将本次局部流水改动宣称为全算子任务映射迁移。

## 1. 目标与边界

`ChunkFwdH` 是独立的 Ascend C 算子，符号为 `ChunkFwdH`、`aclnnChunkFwdH` 和
`fla_npu.ops.ascendc.chunk_fwd_h`。它不复用、不修改 `ChunkGatedDeltaRuleFwdH`，也不提供
legacy `torch.ops.npu` 兼容入口。

当前支持 A2、A3、A5，固定 `K=V=128`、`chunk_size=64`，`k/w/u` 为 BF16。gate 和
state 分别支持 BF16/FP32，`state_v_first=true/false` 均由 kernel 原生处理。公开 tensor
descriptor 使用非私有 ND；输入可由 host 连续化，直接写入的输出必须连续。

## 2. 计算语义

记 `E(x)=exp(x)`，`use_exp2=true` 时为 `exp2(x)`。每个 value head、每个 chunk 执行：

```text
H_c = cast_BF16(R_c)
Pacc_c = W_c @ H_c
P_c = cast_PType(Pacc_c)  # StateT=BF16 时 PType=BF16，否则为 FP32
V_new_fp32_c = fp32(U_c) - fp32(P_c)
V_new_c = cast_BF16(V_new_fp32_c)

g-only:
  V_new_g_c[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32_c[i,:])
  D_c = k_raw_c^T @ V_new_g_c
  R_next = E(g_last) * R_c + D_c

gk-only:
  D_c = kg_c^T @ V_new_c
  R_next[k,v] = E(gk_last[k]) * R_c[k,v] + D_c[k,v]
```

最终 chunk 不请求 `final_state` 时，只生成当前 `H` 和 `V_new`，跳过 Stage2/Stage3。

## 3. head round 规划

规划在 chunk 循环前完成。每个 AIC round 最多四个 head：AIV0 处理 round head 0/2，
AIV1 处理 1/3；两个 AIV 各有 local slot 0/1。

g-only 先按 `group_size=HV/HK` 划分共享 raw K 的 value-head 组：

- `HK:HV=1:3`：每轮三个 value head，一个 kg slot，只读一次 raw K。
- `HK:HV=1:2`：每轮四个 value head，两个 kg slot，各 raw K 只读一次。
- `HK:HV=1:6`：拆为 4+2 两轮；第二轮重新读取同一个 raw K，不跨 round 保留。

gk-only 每个 value head 对应自己的 prepared kg；本轮按 active head 数读取 1..4 份。
`FwdHHeadRoundPlan` 显式记录 `head -> hv -> kh -> kgSlot -> aiv -> localSlot`，Stage2 只遍历
`requiredKhCount`，不预留或加载 16 份 kg。

## 4. 分阶段实现

### S-1：FP32 初态转换

仅 `initial_state` 为 FP32 时执行 `H0=cast_BF16(initial_state)`。A2/A3 逐 tile 处理；A5
每个 head 使用一次完整 RegBase VF，并为两个 local head 分配独立 FP32 input/BF16 output
bank，以“预取当前 head、计算前一 head”的循环实现 ping-pong。全部 active H MTE3 drain
后统一发布本 phase 的 `H_READY`。

### Stage0：入口状态与第一次矩阵乘

AIC 将当前 `W[M,K]` 与 `H[K,V]` 从 GM 搬到四个 round-head L1 槽，随后执行
`P=W@H`。Stage0 不读取 kg/k_raw。tail chunk 先用 MTE2 `InitConstValue` 清零当前 W 槽，
再覆盖有效 ND 行；Cube M 取 `AlignUp(valid_tokens,16)`。

`W`、`U` 和 `g/gk` 的 GM 数据在一次算子执行中只消费一次，GM 读取使用 L2 cache bypass，
避免流式数据挤占递推数据的缓存空间。若当前 work unit 完整覆盖一个 key head 的全部
value-head consumer，则该份 `K` 只从 GM 读取一次并使用 bypass；consumer 跨 work unit 时
保留默认 L2 策略。`initial_state`、递推 `H` 和 Stage1 `right` 存在跨 AIV/AIC 的重复读取
或刚写即读关系，保留默认 L2 策略。

- A2/A3：L0C 经 Fixpipe 将对齐后的 M 行写入 P GM scratch，补齐行恒为零；AIV 再以
  MTE2 只读取 `valid_tokens` 个有效行。
- A5：L0C 经 Fixpipe 直接将有效行写入配对 AIV 的 local UB slot。

### Stage1：向量修正

AIV 读取 `U` 和 P，计算 `V_new`；g-only 同时生成带相对 gate 的 BF16 right，gk-only
直接以 BF16 `V_new` 作为 right。`V_new` 写 GM。right 必须经 MTE3 写 GM workspace，禁止
UB 直搬 L1。

A5 从 MTE2 后开始的全部向量计算由 RegBase VF 完成。g-only 的完整 chunk 先将 64 个
`exp(g_last-g_i)` 一次性向量化写入 gate-scale bank，Stage1 逐行广播该结果，不再重复执行
64 次向量 Exp；tail chunk 保留标量 gate 读取和单元素写回，避免读取 DataCopyPad 未初始化的
UB 余量。VF 只使用 `if constexpr` 模板分支和明确类型的 `for` 循环，并以两组 FP32 寄存器
覆盖 128 列。

### Stage2：第二次矩阵乘

AIC 此时才读取本 round 实际需要的 kg/k_raw，并等待每个 head 独立的 `RIGHT_READY`，把
GM ND right 搬为 L1 Cube 输入。计算 `D=k^T@right`；`state_v_first=true` 时使用等价转置式
`D_physical=right^T@k`，直接生成物理 `[V,K]`。tail chunk 对当前实际 kg/k_raw 槽和各 head
right 槽分别清零后再覆盖有效数据，Cube K 取 `AlignUp(valid_tokens,16)`。

- A2/A3：L0C 经 Fixpipe 写 D GM scratch，AIV MTE2 读取。
- A5：L0C 经 Fixpipe 直接写配对 AIV UB。

### Stage3：递推状态更新

AIV 用 FP32 算术执行 `R_next=decay*R+D`，按 StateT 保存 BF16 或 FP32 rolling state，并按需
写下一 chunk 的 H 或 final_state。A5 使用独立 RegBase VF，不调用 A2/A3 向量实现。

## 5. 存储布局

历史 H 使用 chunk-major：dense `[B,C,HV,K,V]`，公开 packed `[1,total_chunks,HV,K,V]`。
矩阵基址为 `((b*C+c)*HV+hv)*K*V`，packed 为 `(global_chunk*HV+hv)*K*V`；
`state_v_first` 继续只控制矩阵内部 K/V 顺序。A2/A3/A5 的 Cube/Vector 共用
`FwdHHOffset`，初始写出、下块写出和读取同时切换，状态递推计算顺序不变。

AIC L1 固定分区：W `[0,64) KiB`，保留空洞 `[64,128) KiB`，H/right `[128,256) KiB`，
kg `[256,320) KiB`。kg 区最多四个 16 KiB slot；每个 round 只占用 `requiredKhCount` 个。

A5 的 FP32 state 路径将物理数据槽和逻辑 head slot 分开管理。每个 AIV 的静态 UB 布局为：

| 地址范围 | 用途 |
| --- | --- |
| `[0,64) KiB` | 两个 local head 串行复用的 P/D 数据槽；P 占 `[0,32) KiB`，right 占 `[32,48) KiB`，D 占整个 64 KiB |
| `[64,128) KiB` | local slot 0 的 FP32 rolling state |
| `[128,192) KiB` | local slot 1 的 FP32 rolling state |
| `[192,208) KiB` / `[208,224) KiB` | 两个 head 的 BF16 U/V_new work bank |
| `[224,226) KiB` | gate、gate-scale 和 alpha bank |

四 head work unit 中，AIV0 处理 round head 0/2，AIV1 处理 1/3。每个 AIV 的两份 64 KiB
state 分别按逻辑 local slot 0/1 绑定，并在整个 chunk 循环内常驻；仅首块读取 initial state、
末块写 final state，不再逐 chunk 经 GM workspace 回写和恢复。P 与 D 只使用物理数据槽 0，
按 `P0 -> P1 -> D0 -> D1` 的生产消费顺序串行复用，不增加 head round。

`StateBf16Slot(0)` 的 `[128,160) KiB` 地址与第二份 FP32 state 的低半区重叠，但两个生命周期
互斥：它只在“首 chunk、无 initial state”时临时生成全零 H0；该次 H0 MTE3 完成后，Stage3
才初始化两份 FP32 state。存在 FP32 initial state 时先在 `[64,192) KiB` 完成 S-1 转换，
Stage1 的 `ZERO_STATE=false`，不会访问该 BF16 地址别名。

A5 的 BF16 state 路径维持两个原始 local slot。A2/A3 使用两个 32 KiB tile local slot 和
两个 32 KiB BF16 state slot，P/D 使用每核每 round-head 独立的 GM scratch。

A5 的 `gk` 路径在 `state_v_first=true` 时，物理 state 为 `[V,K]`，所有 V 行使用
同一条 128 维 `E(gk_last)`。Stage3 在行循环前按原有指数公式计算一次，保存在一对
FP32 向量寄存器中供 128 行复用；不改变 state 更新、BF16 舍入和写回顺序。
`state_v_first=false` 仍按当前 K 行加载对应 gate，scalar-g 路径保持原有 alpha 复用。

## 6. 同步协议

A5 每个 local slot 分别维护 `P_READY/P_FREE`、`D_READY/D_FREE`、
`RIGHT_READY/RIGHT_FREE`、`H_READY`。A2/A3 mode2 是 `AIC + 2*AIV` 集合同步：每个
pair 由 AIC set/wait 一次，两个 AIV 对同一 ID 各 wait/set 一次；尾 pair 缺 head 的 AIV
执行 dummy 同步但不访问数据。ready 由真实生产 pipe 发布，free 由最后消费者发布；同一
slot/pair 的事件复用前必须完成上一代 wait。

A5 FP32 state 的两个 local head 虽然共享一个 P/D 物理槽，跨核 flag 仍按逻辑 local slot
0/1 区分，ready/free 的 ID 和 set/wait 次数均不改变。Stage1 可以在 V pipe 完成后尽早发布
`P_FREE`，因为后一个 P 只覆盖低 32 KiB；但 D 会覆盖完整 64 KiB，必须等第二个 head 的
right MTE3 读完高 32 KiB。为此，同一 AIV 的两次 `RIGHT_READY` 保持原 ID 和次数，由
`PIPE_MTE3` 在两个 Stage1 都下发后统一发布。AIC 在搬入第一个 right 前等待 local slot 0 的
`RIGHT_READY`，因此首个 D 的 Fixpipe 不会与第二个 right 的 MTE3 形成 WAR。单 head 和
BF16 state 路径仍在各自 right 写回后立即发布 ready。

Stage 内按核内 head id 统一写一套流程，`headId&1` 选择 ping/pong L0 slot。当前 slot 的
MTE2 完成即可启动该 slot 的 VEC/Cube，不等待另一 slot；Cube->Fixpipe 和 VEC->MTE3 同理。

A5 的 FP32 scalar-g 单 head work unit额外启用跨 chunk lookahead。AIC 将 W 放在 L1 slot
0/1、K 放在 slot 2/3，按 chunk 奇偶轮转；当前 right 的 GM->L1 已下发后，立即预取下一
chunk 的 W/K，使下一块 MTE2 与当前 MTE1/MMAD/Fixpipe 重叠。AIV 同样以两个独立 input
bank 轮转 U/g，在消费当前 bank 前先下发下一 bank；Work bank 由 `MTE3_MTE2` free credit
保护，gate/alpha bank 由最终 Stage3 V consumer 发布 `V_MTE2` free credit。P/D/right/state
仍使用原 local slot 和跨核 ready/free 协议，递推 H 仍严格等待上一 chunk 的 `H_READY`。

A2/A3 在无初态、多 chunk 场景为第二个 chunk 的 P scratch 预置一次 free credit，因为首
chunk 没有 Stage0/P；后续 credit 由前一 chunk Stage1 产生。round 结束时 P 与 D 两条独立
scratch 链分别回收，不能用互斥分支漏掉其中一条。

跨 work unit 使用双向收口：两个 AIV 等本 unit 的 MTE3 全部完成后发布 `ROUND_DONE`；AIC 收到
完成信号并回收 P/D/right 后才回 ACK。A5 的 DONE 由 `PIPE_MTE3` 发布，DONE wait 和 ACK
使用 `PIPE_S` 作为 scalar/control gate；下一 unit 的 kg/H/W MTE2 因而不能越过 ACK，
不会与上一 unit 的未完成写回交叠。A2/A3 用 mode2 collective 完成同等的收口。

Host 将 `(sequence, value_head)` 展平为连续 head task。设有效 sequence 数为 `N`、物理 AIC
核数为 `C`，则 `totalHeadTasks=N*HV`、`headsPerCore=ceil(totalHeadTasks/C)`，实际启动核数为
`blockDim=ceil(totalHeadTasks/headsPerCore)`。第 `coreIdx` 个 AIC 处理连续区间
`[coreIdx*headsPerCore, min((coreIdx+1)*headsPerCore, totalHeadTasks))`，不再把同一 sequence
的全部 head round 绑定到一个核。核内区间跨越 sequence 边界时必须拆分 work unit；同一
sequence 内也按连续 value head 每四个一组拆分，因此每个 unit 最多四个 head。每个 head
的全部 chunk 递推仍固定在同一核并顺序执行，不依赖 block 启动顺序建立状态依赖。

## 7. 变长序列

变长模式要求 BNSD 容器 `B=1`。`cu_seqlens` 从 0 开始、以 T 结束且严格递增；
`chunk_indices` 若存在，必须是 sequence-major canonical `(seq_id, chunk_id)`。连续 head task
区间跨 sequence 时按边界拆为独立 work unit；同一 sequence 的不同 value head 也可分配到
不同核。state/final_state 的首维是 sequence 数，H 的 chunk 维使用全局 chunk 前缀，各 head
的 GM 输出区间互不重叠。

## 8. TilingData

`ChunkFwdHTilingData` 只保存 host 已校验并由 kernel 直接消费的数据，不在 tiling 中保存每个
head round 的展开数组。kernel 在进入 chunk 循环前，按 `kNumHead/vNumHead` 生成
`FwdHHeadRoundPlan`，因此每个 round 的 active head 数、实际 Hk 数和 head-to-kg-slot 映射
不会随 chunk 改变。

| 字段 | 语义 |
| --- | --- |
| `batch` | dense 时为逻辑 batch 数，varlen 时为 sequence 数 |
| `seqlen` | k/w/u 容器的 token 维长度 |
| `kNumHead` / `vNumHead` | Hk/Hv 数；g-only 要求 `vNumHead % kNumHead == 0` |
| `kHeadDim` / `vHeadDim` | 当前均固定为 128 |
| `chunkSize` | 当前固定为 64 |
| `useInitialState` | 是否读取 initial_state |
| `storeFinalState` | 是否生成并写 final_state |
| `isVariedLen` | 是否启用 varlen 调度 |
| `shapeBatch` | dense 的物理 batch；varlen 固定为 1 |
| `tokenBatch` | varlen 的 sequence 数；dense 固定为 1 |
| `vWorkspaceOffset` | A2/A3 P 的 GM scratch，按 FP32 slot stride 预留 `[blockDim,4,64,128]`；实际元素为 PType |
| `vUpdateWorkspaceOffset` | Stage1 BF16 right 的 GM workspace，形状为 `[blockDim,4,64,128]` |
| `kDecayWorkspaceOffset` | FP32 rolling state 的 GM workspace，形状为 `[blockDim,4,128,128]`；A5 使用每 AIV 两份常驻 state，不访问该段 |
| `hWorkspaceOffset` | A2/A3 D 的 FP32 GM scratch，形状为 `[blockDim,4,128,128]` |

workspace 的 core 维使用实际 `blockDim`，各段按 512 Byte 对齐，并在 CANN lib-api workspace
之后额外保留运行时安全区。
A5 的 P/D 走 L0C->AIV UB，不消费对应的 P/D GM scratch；为保持跨架构统一 tiling，offset
仍由 host 生成，但 A5 kernel 不访问这些地址。

## 9. 模板参数与 TilingKey

`ChunkFwdH` 的 kernel 全局入口按以下顺序声明六个编译期模板参数：

1. `D_T_G`：g/gk dtype，支持 BF16、FP32；
2. `V_DIM`：value head dim，当前仅注册 128；
3. `USE_GK`：scalar-g 或 key-gk；
4. `USE_EXP2`：exp 或 exp2；
5. `STATE_FP32`：rolling/final state 为 BF16 或 FP32；
6. `STATE_V_FIRST`：state 布局为 `[K,V]` 或 `[V,K]`。

模板声明共注册 `2 x 1 x 2 x 2 x 2 x 2 = 32` 个可达实例。host 在完成 dtype、shape、
gate 模式和 state 输出校验后，以完全相同的参数顺序调用 `GET_TPL_TILING_KEY`；kernel 直接由
对应实例构造 `GateT`、`FwdHCompilePolicy` 和 state offset，不再读取 tiling 数据做 gate mode、
指数函数、state dtype 或 state layout 的运行时分派。

state dtype 的模板选择保持运行语义优先级：存在 `initial_state` 时取 initial dtype；无 initial
但写 final state 时取 final output dtype；二者都不存在时 rolling state 固定选择 FP32。
`STATE_V_FIRST` 同时用于编译期 state 地址映射，避免热路径中的 layout 条件分支。

代码不维护手写数值 key，也不在 kernel 内使用 `TILING_KEY_IS` 做二次分派。TilingKey 数值由
目标 CANN 根据模板声明编码，只用于 host、binary metadata 和 runtime 之间的选择闭环，不作为
公开接口或跨版本稳定语义。

## 10. A5 双 head 跨 chunk 输入 lookahead

本节为已编码、待设备验收的实验候选，尚未正式采用。

### 目标、选择条件与预算

基线 codex/gdn-profiling：809bce6887639e831caa856b8ac3a197a6a94294。用户授权本候选设计与编码，正式采用仍需设备验收。GDN case 0 的普通 H mean=588.803 us，七项 kernel 时间和=1708.391 us；至少 5% 整体收益要求节省 85.420 us，在其余项不变时 H 不高于 503.383 us（下降 14.51%）。timer warmup=5，普通 ATK 原预热与 20 条记录保持一致；A/B 使用相同冻结输入、配置、环境与 wheel 校验。case 1 与共享调用的其它支持场景要求无不可接受回退，实测值待补。

唯一主要变量是跨 chunk 独立输入的预取距离；不改变公开 ABI、tiling、workspace、Cube/VF 算术、指数、BF16 cast、state layout 或状态递推。A5 FP32 state、scalar-g 模板中，只有 activeHeadCount=2、G=HV/HK=2、requiredKhCount=1 且两 head 绑定同一 kh 的 work unit 启用新路径。AIC/AIV 使用同一选择函数。一个/三/四 head、G1/G3/G4、不完整 G2 pair、gk、BF16 state 和 A2/A3 沿用旧路径；现有单 head lookahead 原样保留。

case 0：N=1、HV=32、HK=16、T=11274、BT64、FP32 初态/终态，177 chunks（最后 M=10）。按可用 C=28 条件代入现有 tiling 得到 blockDim=16、每 AIC 两 head；全部 chunk 仍属于同一 head 链并顺序更新。AIV0/1 各持有一个 head/localSlot0，两 head 共用一个 K。每个完整 chunk/AIC 独立输入 W 两份、U 两份、K 一份各 16 KiB，g 两份各 256 B，合计约 80.5 KiB；GM 字节数不增加。达到 5% 约需每次 chunk 循环减少 0.483 us 暴露时长。当前缺 pipe 数据，不能证明该时长可隐藏；可达性暂定，以硬件 A/B 决定，不能将整个独立输入搬运量当成可消除耗时。

### Stage 0–3 受影响行为

符号：c 为本 unit 的 chunk 序号，p=c&1，r=0/1 为 roundHead；W 的物理槽为 2p+r，共享 K 物理槽为 p；H/right 槽为 r；AIV 的 U/g input bank 为 p。两 head 的实际 hv/kh、GM offset、tail 有效行 M、状态矩阵内部 layout 继续由原 scheduler/helper 产生。

| Stage | 原数学、物理落点与本次搬运 |
| --- | --- |
| 0，Cube | H=cast_BF16(R)，P=W[M,128] @ H[128,128]，BF16 输入/FP32 累加，FP32 P Fixpipe→所属 AIV UB。W 在前一 chunk 预取；递推 H 仍经原 H_READY 后 GM→L1。两 head 的 WH 顺序不变 |
| 1，Vector | V_new_fp32=fp32(U)−P，V_new=cast_BF16(V_new_fp32)，right=cast_BF16(exp*(g_last−g_i)×V_new_fp32)。U/g 在当前处理前预取下一 bank；vNew/right 写回和原 ready/free 不变 |
| 2，Cube | D=K[M,128]^T @ right[M,128]，BF16 输入/FP32 累加、FP32 D Fixpipe→AIV UB；STATE_V_FIRST 使用原等价转置式。两 head 共享同一预取 K，分别读取各自 right，KV 顺序不变 |
| 3，Vector | R_next=exp*(g_last)×R+D，FP32 state 常驻；按原顺序生成 BF16 H_next 或 final_state。使用当前 p 的 gate/alpha，V 消费后才归还该 gate bank |

R21：输入值域、FP32 指数/乘加、BF16 舍入、初态转换与 tail VF 掩码原样复用 CPU 标杆对应实现，不作代数或累加次序变换。尾块仍清零 W/K/right 的无效 Cube 区域，只读取 M 个有效 U/g；first 无初态跳过 WH，last 不请求 final_state 跳过 KV/state update。

### L1/UB 绝对地址与生命周期

以下单位 KiB，半开区间。L1 属于同一 AIC，UB 区间以每个 AIV 的 allocator 基址为零点；runtime UB 起始位置及高端 VF 栈约束保持。

| 存储地址 | 对象、dtype/layout、大小及生命周期 |
| --- | --- |
| L1 [0,16) / [16,32) | p0 的 W r0/r1，BF16[64,128]、原 Cube layout，各16；MTE2 首写，Stage0 MTE1 末读；整组 free 后覆盖 |
| L1 [32,48) / [48,64) | p1 的 W r0/r1，同上；p0/p1 可同时存活 |
| L1 [64,128) | 空闲，无数据，未新增占用 |
| L1 [128,160) / [160,192) | H/right r0/r1，BF16[128,128] H 或 [64,128] right；GM MTE2 首写，Stage0/2 MTE1 末读，原事件4/5保护同槽复用 |
| L1 [192,256) | 原 H/right 槽2/3未激活；空闲，跨 unit 仍按原 credits 复用 |
| L1 [256,272) / [272,288) | K p0/p1，BF16[64,128]原 Cube layout，各16；组 MTE2 首写，Stage2 r1 的 MTE1 末读后整组 free |
| L1 [288,320) | 原 K 槽2/3未激活；空闲，无新增分配；静态 L1 上限仍320 |
| UB [0,64) | 原 P/D 数据槽0：P低32、right [32,48)、D整个64；Fixpipe写P/D，V/MTE3末读，原跨核 P/D/right/H ready/free保护 |
| UB [64,128) | localSlot0 的 FP32[128,128] rolling state，64；初态首次搬入/首块置零，Stage3 原位递推，整个 unit 常驻 |
| UB [128,192) | localSlot1 state 未激活；first无初态时 [128,160) 暂存零BF16 H0，MTE3末读后释放，不与有效FP32 state重叠 |
| UB [192,208) / [208,224) | U/V_new p0/p1，BF16[64,128]，各16；MTE2首写、VF原位更新、MTE3输出末读，以 WorkFree 2/3 复用 |
| UB [224,224.25) / [224.5,224.75) | scalar g p0/p1，最多FP32[64]，各256 B；MTE2首写，Stage1 gate VF末读；沿原gate bank协议保护 |
| UB [224.25,224.5) / [224.75,225) | gateScale p0/p1，FP32[64]，各256 B；Stage1 gate VF首写、Stage1 right VF末读，同bank不提前覆盖 |
| UB [225,225.03125) / [225.03125,225.0625) | alpha p0/p1，FP32标量按32 B对齐；Stage1首写、Stage3末读，GateFree 0/1在最终V消费者之后归还 |
| UB [225.0625,226) | 既有预留空闲区域；无新增分配 |

有效 L1 最高288、每个 AIV 静态区域最高226；原静态上限、L0 A/B/C 双槽和 GM workspace 不增加。每个 p 的 W 与 K 作为一个不可分割的 L1 所有权组，以避免四个 W 加两个 K 再加 H/right 需要过多 HardEvent ID。W/K 不能独立提前归还；组在最后共享 K 消费结束之后释放。

### 搬运和事件闭环

完整 chunk/AIC：W读GM两次共32 KiB，共享K读一次16 KiB；两个AIV读U共32 KiB、g共512 B。每head仍写vNew16 KiB、right16 KiB并由Cube读right16 KiB，写/读H32 KiB，P/D A5直接UB；state只首块读初态、末块写终态。No-final末块沿原逻辑省略right/K/状态更新。L2 bypass/默认策略及所有GM offset使用旧helper，时机之外不变。

| 资源 | 源→目的、ID、初态与循环 |
| --- | --- |
| 输入组 free | MTE1→MTE2，WReadyEvent(p)=0/1；原 InitEvents 各置1 credit。PrefetchPair先Wait，最后K MTE1消费者后Set；无Stage2末块则最后W读后Set；DrainEvents消费最终credit |
| 输入组 ready | MTE2→MTE1，WDoneEvent(p)=0/1；组内有效W/K全部下发后Set一次，当前chunk使用前Wait一次；个别W/K helper不另Set/Wait组事件 |
| 未使用W事件 | MTE1→MTE2 2/3保持原初始credit，退出仍按原Drain消费；MTE2→MTE1 2/3不发布 |
| H/right L1 | MTE1↔MTE2 4/5，原初态、Wait/Set不变；最高ID5，仍不用静态tensor保留的6/7 |
| U/V_new | MTE3→MTE2 WorkFree 2/3，MTE2→V WorkReady 2/3，V→MTE3 WorkDone 2/3；每chunk WaitFree→Copy→SetReady→WaitReady→VF→Set/WaitDone→MTE3→SetFree |
| g/gateScale/alpha | V→MTE2 GateFree 0/1，初态1；仅需要right时Wait，当前Stage3的最终V消费后Set；末块no-final不取得也不归还不存在的credit |
| P/D/right/H与Io | 原localSlot0跨核与Io事件不变，各AIV拥有不同peer ID；H_READY仍在H的MTE3之后，P/D释放和状态更新次序不变 |

组控制由 helper 模板 `MANAGE_INPUT_EVENTS=false` 在新路径关闭 W/K 的独立取得、ready、归还；旧路径默认true，保持既有事件顺序。H/right与L0/Fixpipe事件不受该模板控制。整组ready只等待一次后可消费两个W和共享K；free延迟到整个chunk最后消费者，不引入新的全流水barrier或事件ID。

调度伪代码：

```text
AIC Init原事件；PrefetchPair(chunk0,p0)：Wait组free→按需求Copy W0/W1/K→Set组ready
for c:
  Wait组ready[p]
  原H_READY/GM→L1；WH head0、head1（first无初态省略）
  if NeedStage2:
    原RIGHT_READY→LoadRight head0、head1
    if存在next: PrefetchPair(next,p^1)        # 已把当前right排在next预取之前
    KV head0、head1，保持共享K消费与P/D ready/free
  Set组free[p]                             # last no-final也必须归还
原terminal P/D free、ROUND_DONE/ACK、DrainEvents

每个AIV：原S-1；预取自身head的chunk0 U/g input bank0
for c:
  取得数据槽0并先发布IoReady                 # 当前处理不额外等待next输入复制
  if存在next: 预取next U/g到p^1
  原right free；Stage1消费当前p，发布当前right ready
  if NeedStage3: Stage3消费当前p的alpha并归还gate bank
原末块/ROUND_DONE/ACK/DrainLocalEvents
```

p回绕只覆盖c−2已释放的输入；prefetch不越过work-unit边界。最末无next不发空预取，空unit保持原Process过滤，terminal无Stage2分支仍归还L1输入组，first无初态仍等待共享K ready。当前right排在未来输入前，且未来输入组free来自c−1的已完成MTE1消费，不依赖当前待下发KV，避免MTE2队列等待形成循环。

### 增量评审与验证

按R02/R08/R10/R11/R13/R17检查依赖、容量、GM一次读取、双bank及别名；R21保持原数值门禁。全局task分核、slot0的P/D直接UB、rolling state常驻及ROUND协议沿用，不改支持范围。自检结论：地址和所有权组可实施、事件数合法、首块/尾块/回绕/退出在离线模型中闭环；性能可达性及真实同步仍需设备验证。本候选已按以上设计编码；测试证据、参考CHUNK_DEPENDENT V1（a14961e9dfe8c8264768d67fb447101873da4ecf）的复用范围与待验项记录到 [validation.md](validation.md)。

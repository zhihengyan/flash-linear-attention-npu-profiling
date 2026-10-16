# ChunkFwdO：raw QK 共享候选设计

方案设计规则版本：`V2`。本轮依据既有 [接口](api.md)、CPU 标杆和 K32 当前实现，
补齐原先缺失的 A5 设计记录。继续使用历史 AIV head-owner 映射和 A-prime UB->L1 路径，
未进行 R01/R06 的结构迁移；具体差异列在末尾，不声称整算子已完成 V2 迁移。

## 1. 目标

基线为 `codex/gdn-prepare-k32` / `d0cb628df7c8c23c0342392dfaf4ec910aba9947`。
在 ChunkFwdO 的 Stage2 中，每个 task group 内同一 key head 的 FP32 raw QK 只计算一次，
从同一 L0C 结果分别 Fixpipe 给各 owner AIV；QH、gate、A-prime、AV、输出均继续逐 head
执行。不改变 taskGroupSize（G3 为 3，其余为 4）、host 分核、workspace、公开输出或算法。
不新增 GM 中转、Vector 重排、全局同步或私有 probe，普通/Profile 入口共用本实现。

主模型 case0：Ascend950PR_957c，B=1、HK=16、HV=32、T=11274、K=V=128、BT=64，
q/k/v 为 BF16、g 为 FP32、varlen、FP32 initial_state、输出 final_state。O 使用 28 AIC；
177 chunks（176 full，tail=10）、每 chunk 8 个四 head group，共 1416 工作项，
grid-stride 下 16 个核各 51 项、12 个核各 50 项。shape 和入口保持冻结性能 JSON。

10/16 K32 的七项均值为 1610.88620 us，O 为 358.29505 us；O MAC 96.50610 us、
MTE2 331.40270 us，仅作瓶颈线索。若全部从 O 取得约 2.7% 整链收益，需净省 43.49393 us，
即 O 降低约 12.14%。用户未将 2.7%/3% 设为硬门槛；实际收益未知。

完整 chunk 每 head 的 MAC 为 QK=64*64*128，QH=64*128*128，AV=64*128*64；
G2 的 QK 次数减半意味着 O 总 MAC 少 12.5%，不是 O 时长少 12.5%。case0 的 raw QK
调用从 5664 降至 2832，QH/AV 及 raw QK Fixpipe 分发次数保持。
Q/K GM 加载本来已共享，所以 GM 逻辑搬运不减少；省去的是重复 QK MMAD 和它的
Q/K L1->L0 读取。更长的 L0C 占用、对另一槽的 QH 复用及 Scalar 分支可能抵消收益。
原 H/Prepare 实验的 MAC/搬运减量 proxy 不用于本轮预测。精度先行，普通七项同口径
K32/B/K32 交错、多进程采样；诊断预热、样本配置和验收步骤见 [验证记录](validation.md)。

## 2. Stage 完整详设

### 符号、任务和共享边界

单位为 Byte，空间表另标 KiB；M=64、K=V=128，n 为有效 token 数。G=HV/HK，
CG=taskGroupSize。work=chunk*ceil(HV/CG)+headGroup，core 按 work+=blockDim 跨步。
head group 起点 hvBase=headGroup*CG，count=min(CG,HV-hvBase)，r=0..count-1，
hv=hvBase+r，hk=hv/G，owner=r%2，localSlot=r/2。两个 AIV 都参与每 head 的有序握手，
只有 owner 执行该 head 的 VF 和写回。

同一 HK 在当前 group 内的第一 head 为 max(hk*G,hvBase)。`loadQK` 标记该第一 head；
最后 head 的条件为 `r+1==count || (hv+1)/G != hk`，命名 lastQKConsumer。
共享键含当前 chunk、当前 group 及 hk；缓存不跨 work 复用。每个 group 首 head 必定
loadQK=true，故 tail、head group 跨 HK 或 G1 都不依赖上一 work 的残留值。

| Stage | 类型 | 公式和结果 | 主要空间 |
| --- | --- | --- | --- |
| 1 | Vector | gateA=E(g_i-g_j)，gateO=E(g_i) | AIV gate banks |
| 2 | Cube | raw QK=Q@K^T；QH=Q@H | L1 Q/K/H、L0C raw 与 QH、AIV raw banks |
| 3 | Vector | BF16 A-prime=causal(raw QK*gateA)，O_s'=QH*gateO | owner AIV UB、原 L1 A-prime |
| 4 | Cube | O_l=A-prime@V | L1 A-prime/V、L0、owner AIV O_l |
| 5 | Vector | O=BF16(scale*(O_s'+O_l)) | owner AIV UB 与 GM O |

这里 E 为 exp，use_exp2 时按现有 VF 的 ln2 换底序列；无新增代数变换。

### 全局空间与生命周期

L1 仍按四 head 预留，所有 Cube 源 layout/transpose 保持：

| L1 区间 / KiB | 对象、份数和生命周期 |
| --- | --- |
| [0,64) | Q：4*16 KiB，BF16 [64,128]；每共享子组只激活首 head 槽，最后 QH 读完才 free |
| [64,128) | K：4*16 KiB，BF16 [128,64] ColumnMajor 视图；同一 qkEvent 保护 |
| [128,256) | H：4*32 KiB，BF16 [128,128]，state_v_first 选 RM/CM；当前 QH 读完即 free |
| [256,288) | A-prime：4*8 KiB，BF16 zN；原 Stage3 MTE3 写，Stage4 消费后按原组协议复用 |
| [288,352) | V：4*16 KiB，BF16 [64,128]；保留一 head lookahead 和原 H/V event 复用 |

最高 L1 地址仍为 352 KiB，以现有 `ArchTag::L1_SIZE` static_assert 为容量依据。
L0A/L0B 各两个 32 KiB 槽；L0C 为两个 128 KiB 槽，总 256 KiB，仍由现有架构断言限制。
共享 raw QK 占 C 槽 s 内起始 16 KiB（FP32 [64,64]）；QH 固定使用另一个槽 s^1 的
起始 32 KiB（FP32 [64,128]）。raw 内容从首次 MMAD 到最后一次 Fixpipe 之间保持。
Stage4 仍使用原双槽轮转，仅在 raw 已发布 free 后开始复用；不新分配 L0C。

每 AIV 的 UB 沿用以下绝对半开区间；各对象仅由自己的 owner 生产/消费：

| UB 区间 / KiB | 对象、份数和生命周期 |
| --- | --- |
| [0,0.5) | 两个 gateO bank，各 64 FP32；Stage1 写，Stage3 消费 |
| [0.5,32.5) | 两个 gateA bank，各 [64,64] FP32；Stage1 写，Stage3 消费 |
| [32.5,33) | 两份 g scratch，各 256 B；原 MTE2/V 事件保护 |
| [33,97) | 两个 QH/O_s' bank，各 32 KiB FP32；Stage2 Fixpipe、Stage3 VF、Stage5 VF |
| [97,129) | 两个 raw QK bank，各 16 KiB FP32；每 head 仍独立 Fixpipe，Stage3 VF 末次读 |
| [129,193) | 两个 O_l/O bank，各 32 KiB；Stage4 FP32 写、Stage5 原地 BF16 输出 |
| [193,208) | 原保留空间，不新使用 |
| [208,224) | 原 A-prime FP32 预留，不新增使用 |
| [224,240) | 两个 BF16 A-prime bank，各 8 KiB；Stage3 VF 写，MTE3 读 L1 后 free |
| [240,248) | 原尾部保留空间，不新使用 |

UB 总预留仍为 248 KiB；原 mte3ToVStream、输入就绪和最终排空保持。两个 head 的 gate、
QH、A-prime 和 O_l 不共享数值。GM Q/K/V/O 的 BHTD/BSHD offset 及 chunk-major H offset
沿用 chunk_fwd_o_common.h。原 aPrimeWorkspaceOffset 和每核双窗口八个 8 KiB record 的
workspace 预留不变；本次新增 workspace=0，raw QK 不写 GM。

### Stage 1：Vector，gate

每 owner 按原顺序一次读取自己的 g，计算 gateO/gateA；Full/tail、exp/exp2 和原 FP32
指数序列保持。该 Stage 可与 Cube 准备重叠，UB gate 区与 raw/QH 目标互不重叠。

### Stage 2：Cube，共享 raw QK 与逐 head QH

1. loadQK 时等 qkEvent free，将 Q/K 的有效 n 行搬入原 L1 槽，并发布 MTE2_MTE1。
   每 head 仍独立读 H 到自己的 L1 槽；L2 bypass、padding 和物理 layout 保持。
2. group 首 head 在任何 L0/UB 访问前，仍等上一 group Stage5 的有序完成 token。
3. 仅 loadQK 时取 l0ASlot/l0BSlot/l0CSlot，消耗各自 free，执行原 Q/K L1->L0 及
   `raw=Q[64,128]@K^T[128,64]`，BF16 输入、FP32 累加、初始化 true。
   将 C 槽号保存为 qkCSlot。M_FIX set/wait 仅随这次真实 MMAD 产生一次。
4. 每 head 都从 qkCSlot 原位 FP32 raw 以原 NO_SPLIT CopyQktToUb 写 owner/localSlot。
   PIPE_FIX 本身有序；后续 head 不再等待不存在的 M_FIX credit。
   仅 lastQKConsumer 发布 FIX_M(qkCSlot)，此前不允许该槽成为任何 MMAD 目的地。
5. QH 仍执行 `Q[64,128]@H[128,128]`，BF16/FP32 原序；C 使用 qkCSlot^1，
   自己的 FIX_M/M_FIX 每 head 完整闭环。A/B 槽按真实发出的 MMAD 次数轮转。
   MTE1_M 的 set/wait ID 改为该次 A 槽号（A/B 始终同号），不再跟 C 槽号；
   去重后 C 与 A/B 奇偶会分离，A 槽的 M_MTE1 free 才能保证同一 MTE1_M 通知已被消费。
   仅最后 consumer 的 QH L1 读取完成后发布 qkEvent 的 MTE1_MTE2 free，保护 Q 的末次读取。
6. 原 QH FP32 Fixpipe 完成后，每 head 发布一次 Cube->Vector ready；两个 AIV 按原序消费。

G1 的 producer=lastConsumer，每 head 仍 QK/QH 两次 MMAD，C 槽与基线同序。
G2/3/4 的共享只消除重复 raw 生产；未就绪的 H 仍按原 hEvent 等待。
tail 的 raw/QH M 仍为 64，有效 n 和原 padding 保持，不把 tail 改成另一计算形状。

### Stage 3：Vector，逐 head gate/mask

原 Stage3Gate64VF 不变：使用自己的 gateA/gateO 和独立 FP32 raw/QH，生成自己的 BF16
A-prime 和 FP32 O_s'。保留原 causal mask、有效行、CAST_RINT 舍入及 A-prime UB->L1
四段上传。两个 AIV 都参与每 head ready，non-owner 仅握手，不新增广播 VF。

### Stage 4：Cube，逐 head AV

原 `A-prime[64,64]@V[64,128]` 和 tail n*n 形状保持，FP32 累加；V 的一 head lookahead
和两个 L0 槽轮转保持。所有 raw Fixpipe 均已入队释放信用后才进入此循环，实际覆盖仍等
FIX_M 完成，不因最后 consumer 的 Scalar 函数返回而跳过 wait。
Stage4 的 MTE1_M 同样按 A 槽编号，使 QK/QH/AV 统一保护输入通知的复用；只改已有 ID
的选择，不增加事件。G1 的 A/B/C 同步轮转，实际 ID 值仍与原实现相同。

### Stage 5：Vector，输出

原 Stage5Fuse64VF、scale、BF16 cast、BSND/TND strided DataCopyPad 和 MTE3 free 保持。
O 不共享 gate/H 相关结果；group 结束的集体 token 保持一次，末 group 仍按原入口消费。

### 搬运汇总与同步

| 数据 / 每完整四 head G2 group | 基线 | 候选 |
| --- | --- | --- |
| Q/K GM->L1 | 各 2*16 KiB | 相同 |
| Q/K 为 raw QK 的 L1->L0 | 各 4*16 KiB | 各 2*16 KiB |
| Q 为 QH 的 L1->L0 | 4*16 KiB | 相同 |
| H GM->L1 | 4*32 KiB | 相同 |
| raw QK L0C->owner UB | 4*16 KiB | 相同，无 GM 中转 |
| QH L0C->UB | 4*32 KiB | 相同 |
| gate/A-prime/V/AV/O | 原各 head 路径 | 相同 |

核内资源数量不变：L1 MTE1_MTE2/MTE2_MTE1 event ID 为 5..12，共八份；A/B 的
M_MTE1 free 为 0/2 与 1/3，共四份；MTE1_M 使用 A 槽号 0/1，M_FIX、FIX_M 使用 C 槽号 0/1。
FIX_M 和 A/B free 在 Init 各置一份。raw 的 FIX_M 由最后 Fixpipe 发回，QH/AV 由各自
Fixpipe 发回，下一 MMAD 必须消费；L1 的剩余 free 在原 Process 尾循环消费。
不改变原 L0 退出处理、不增加 PIPE_ALL 排空。全部 ID 上界保持既有实现。

```text
每 work 首 head: Q/K/H 预取 -> 原上一组 Stage5 token wait（若有）
每共享 HK 子组:
  loadQK: qkEvent free wait -> GM Q/K -> qk ready -> L0 A/B/C free wait
          -> MTE1_M(A) set/wait -> 原 raw MMAD -> M_FIX -> 首次 Fixpipe wait
  每 head: raw Fixpipe 到 owner UB
           最后 head: FIX_M(rawC) set
           QH: otherC free wait -> MTE1_M(A) set/wait -> 原 MMAD -> Fixpipe -> otherC free set
           最后 head: qkEvent free set（Q 已完成最后 L1 读取）
           原 Cube->AIV ready
原每 head Stage3 -> 原 Stage4 -> 原 Stage5；下一 work 沿原 group token 复用 UB
末 work: 原 pending Stage5 token wait -> 原 L1 free 收口
```

空核不进入 group 循环，head tail 只遍历 count；每个非空子组恰好有一个 producer 和一个
last consumer，因此 raw credit 不泄漏、不重复释放。两个共享组连续、奇数 CG、跨 core
grid-stride 和 slot 回绕均通过相同规则处理，不能缓存跨 chunk 的 raw。

### 规则审查和验证门槛

R02/R03/R08/R11/R13/R17：空间上界不增加，raw 与 QH 物理隔离，末消费者 free 明确。
R04/R05/R09：保留四 head 与每 AIV 两份结果区，不压缩驻留。
R10/R12/R15/R16/R18/R19：保持原输入、VF、Stage 和逐 head 输出，只复用重复 Cube 结果。
R07 的 Cube->Cube 规则不新增适用对象（raw 的消费者是 Vector）；R14 不新增 GM 中转。
R20 的 work、CG、分核和波数保持，组大小调整不在本轮。
R01/R06 的历史 head-owner/UB->L1 与当前 V2 仍有结构差异，本轮不迁移、不声称满足这些
历史差异；本次共享本身不引入新的 Vector->Cube 数据路径。

R21：每个 head 原来相同 Q/K 的 raw 使用相同 K=128 FP32 累加次序；现在复用同一结果，
不改变 reduction tree、dtype、scale、gate 指数、mask 或 BF16 cast 时机，不引入新数值
变换或新的输入值域。CPU 标杆/混合容差保持。同步错误仍可能破坏数值，必须独立设备验收。

阶段04采用 CHUNK_INDEPENDENT V1，来源 `a78efa9512dfa64c20b2a58a75e84e88a054d694`，
采用有序逐 head 握手、真实生产 pipe ready 和末消费者 free；具体布局以本算子为准。
用户已明确允许跳过逐 Stage 设备验证编码门槛，直接完成整链接入；不等于精度或性能通过。
离线测试必须覆盖 G1/2/3/4、不同 count、slot 回绕、跨组、tail 和错误释放负例；正式构建、
ATK、MSS、确定性、case0/1、共享调用和 Example/ST 待设备执行，详见 validation.md。

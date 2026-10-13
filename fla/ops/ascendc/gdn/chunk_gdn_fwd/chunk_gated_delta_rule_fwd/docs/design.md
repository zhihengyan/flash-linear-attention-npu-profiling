# GDN 计时接入设计

当前规则版本：V2。本设计仅描述诊断增量；基线为 main `90176b44321ebdbf3ae2a9057243157d7085c6e1`。按仓库阶段 03→04→05 路由读取现有实现、ChunkFwdH 设计、原 ATK generator/executor/reference。采用 CHUNK_DEPENDENT V1 开发参考（来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf）的现有 H 状态依赖链，不改算子调度。host/kernel/调用层接入技术写法参考本地 timer_by_skill 的已存在诊断实现；本轮新增完整关键运算与循环观测、布局 v2 和独立源码审核，未沿用旧审核结论。

计算链保持：KKT=KKᵀ 的 gate/下三角系数 → solve → W/U；H 内 WH、vNew=U−WH、KV、状态更新；O 内 QK、QH、causal mask/gate、Attn×vNew、输出融合。原 fixed/varlen、head ratio、BT64/128、V128/256、initial/final state、B30 和 H/O idle/pipeline 分支均保持。原图、关键步骤与源码配对清单由 [timer_flow.md](timer_flow.md) 定义。

## 新增 GM 布局与所有权

设 F=6、D=28、I=32、Gcap=256、C=3Gcap=768。每核 timing 槽数 F+D×I，两个 int64/槽；每行 words 按 16 对齐。每核实际调用计数 D 个 int64，按 16 对齐；计数保存总数，事件仅采样前 I 次。一个 kernel slice = 16-word header + C×timing row + C×count row；四个 slice 对应 Phase6/Prepare/H/O。具体派生常量只维护在 canonical C++ 头，parser 读取其枚举与 constexpr 算术，host 按 TOTAL_BUFFER_SIZE 校验。

地址半开区间（单位 int64 words）：slice s 基址 s×BUFFER_WORDS_PER_KERNEL；header=[base,base+16)；核心 c timing 行=[base+16+c×stride,base+16+(c+1)×stride)；counts 基址=base+16+N_TIMING_COUNTER，c 行=[counts+c×countStride,counts+(c+1)×countStride)。AIC group g 使用 c=3g，AIV0/1 使用 c=3g+1/2。各行和 header 独占完整 cache line；没有其他写者。header 由 AIC0 写实际 blockDim、arch、layout/version/开关/容量。超容量不会转写 core0，解析时拒绝。

新增 timer 张量为独立 GM 区域，生命周期从采集端 torch.zeros 到 launch 同步、导出之后；不占原 workspace，不分配 UB/L1/L0，不改变原 task/output 地址。每 scope 的局部 start/iter 独立，嵌套与重复不会覆盖；每类的 per-core actual counter 在开始处更新。诊断末尾本核局部 PIPE_ALL 只执行一次，结束戳写完后 DataCacheCleanAndInvalid 发布本核 timing/count 行，适配 Prepare 关闭 kernel-end 自动 DCache 写回的已有选项。该发布和新增标量打点会扰动诊断调用，不能宣称零开销。

## 同步与 R01–R21 增量审查

所有原同步资源、ready/free、数据驻留份数、CG、task/work/head/chunk 映射、tail/padding 和 workspace 生命周期保持。新增诊断行只有本核读写，header 唯一写者；不增加跨核依赖、event/flag/SyncAll 或 UB/L1 资源。因此 R01–R21 中业务 Stage、空间和调度条目沿用基线，仅补充上述独立 GM 所有权与容量。源码校验按顺序比较原 WaitFlag/SetFlag/CrossCore*/SyncAll/PipeBarrier 调用；编译、实际模板选择、同步、精度和内存验收仍需设备证据。

所有细分 scope 使用 NoBarrier，等待项以原 wait 返回为锚点；matmul/VF/epilogue 调用窗含原内部同步，不能称纯计算完成时延。固定总窗包含末尾诊断 drain；GM 发布在结束戳之后。ENABLE_TIMER=0 移除实际 scope/Init/Finish 操作；正常 null timer 不读时钟、不写 GM、不 drain。开关 A/B 的真实打点开销留待设备测量。

验证入口及状态统一见 [timer_run.md](timer_run.md)、[timer_report.md](timer_report.md)。上述计时接入没有性能优化目标；下面记录在其基础上的 Stage6 实验候选，原性能 case 的 shape 与比较标准沿用。

## Prepare Stage6：K 输入提前搬运实验

方案设计规则版本：V2。编码分支 `codex/gdn-stage6-overlap` 从 `codex/gdn-profiling` 的 `809bce6887639e831caa856b8ac3a197a6a94294` 创建。用户已确认进行该候选编码；是否正式采用仍以设备精度和性能结果为准。公开 GDN 和独立 Prepare 的接口沿用原实现与 [Prepare README](../../chunk_gated_delta_rule_fwd_prepare/README.md)，不增加公开开关、参数或 TilingKey。

### 目标与实验范围

本次只调整共享 Prepare Arch35 `Stage6_AivOne` 内 K 输入 DataCopy 的下发位置。目标是用户 case 0：`B=1,Hk=16,Hv=32,T=11274,K=V=128,BT=64`，varlen、初态 FP32、输出最终状态；基线见 [case 0 分析](timer_analysis_case0_2026-10-13.md)。普通 Prepare/H/O 均值为 715.090/588.803/359.376 us，七项 kernel 时间和均值为 1708.391 us。Stage6 采样包含原等待，无法据此定量预估可隐藏的 K 搬运时间；本次是待实测候选，不填写预计必达提升。

提前搬运条件为 `V==128 && HRatio==2 && HV%4==0`，其余参数组合使用原顺序，包括 V256、G=1/3/4 及 Hv 不满足完整 GVA2 pack 的情况。支持域保持原样；该条件只选择局部指令顺序。普通/诊断调用复用同一函数。A2/A3 的 Prepare Arch22 实现和其它融合路径不修改。

### Stage6 的数据与地址所有权

原计算维持：`Vb=cast_BF16(fp32(V)*beta_eff)`；`Kbg` 使用原 `ScaleRowsBetaExp2gVF` 的指数、乘法与 cast 顺序，不作代数变换。g_prime/beta_eff 来自 Stage1，K_hat 为 Stage1 产生或直接引用的 K，V 为输入。Stage6 将 Vb/Kbg 上传 L1，Stage7 消费它们与 Stage5 的 A，计算 `W=A@Kbg`、`U=A@Vb`。

令 `db=(taskIdx>>1)&1`，每个 AIV 在完整四任务 pack 中分别执行 task 0/2 或 1/3。GVA2 的 `PackWorkId` 将实际 head 顺序映射为 owner0/owner1/sibling0/sibling1，因此同一 AIV 的两个任务共享自己的 source K。Hv 为 4 的倍数保证 pack 不跨 chunk 的 head 边界且没有不满 pack；case 0 为每 chunk 8 个 pack，若总 chunk 数为 177，则共 1416 pack，28 AIC 下每组 50～51 pack。这里 chunk 数来自模型输入推导，varlen 的实际边界以设备冻结输入为准。

每个 AIV 的相关 UB 地址半开区间（KiB）：

| 对象 | ping db=0 | pong db=1 | 大小 / 生命周期 |
| --- | --- | --- | --- |
| g_prime / beta_eff | [9,9.25) / [9.25,9.5) | [9.5,9.75) / [9.75,10) | 各 256 B，FP32[64]，Stage1 产生，Stage6 末次消费 |
| K_hat / Kbg | [32,48) | [48,64) | 各 16 KiB，BF16[64,128]；MTE2 首次写入，VF 原位缩放，MTE3 最后上传 L1 |
| V / Vb（V128） | [64,80) | [80,96) | 各 16 KiB，BF16[64,128]；MTE2 首次写入，VF 原位缩放，MTE3 最后上传 L1 |
| Stage7 W | [172,188) | [188,204) | 各 16 KiB，Cube Fixpipe 写入，AIV 的 DrainPackWuToGm 最后消费 |
| Stage7 U | [204,220) | [220,236) | 各 16 KiB，Cube Fixpipe 写入，AIV 的 DrainPackWuToGm 最后消费 |

K/V、系数、Stage7 写入区互不重叠。K/V 区域与旧 Stage1/3 区域重叠，复用前仍经过 ProcessAiv 的 `MTE3_V` 和 Stage3 完成处理，以及 Stage6 循环前的 `MTE3_MTE2` Set/Wait；没有把 K 预取移出这个边界。同一 AIV 的 Stage1 K_hat GM 写回经过现有 MTE3 完成边界后才进入 Stage6；本实验不扩大到跨 AIV 的 K producer/consumer 组合。

实际 GM 地址保持 `OffsetBHTD(batch,hv,tokenStart,HV,T,V)` 和 `OffsetBHTD(batch,hk,tokenStart,HK,T,K)`，读有效 M 行；L1 仍为 `l1Vb[taskIdx]`、`l1Kbg[taskIdx]` 的原 NZ 布局/偏移。每个有效任务读 K/V 各一次，V128 完整块为各 16 KiB，上传 L1 各一次；字节数、L2 策略、workspace、UB/L1/L0 峰值及全局地址分配均不增加。原 UB 峰值 236 KiB、原 L1 图及分配沿用源码常量，不以此新增容量。

### 操作顺序与同步闭环

原顺序：

```text
tail 清零及 V→MTE2 保护（需要时）
MTE2 Copy V → Set V-ready(0) → Wait V-ready(0)
VF scale V → V→MTE3 Set/Wait(0) → Upload Vb
MTE2 Copy K → Set K-ready(1) → Wait K-ready(1)
VF scale K → V→MTE3 Set/Wait(1) → Upload Kbg
原 PIPE_MTE3 Stage6-ready → Cube Stage7
```

候选仅在符合条件时改为：

```text
tail 清零及 V→MTE2 保护（需要时）
MTE2 Copy V → Set V-ready(0) → MTE2 Copy K → Wait V-ready(0)
VF scale V → V→MTE3 Set/Wait(0) → Upload Vb
Set K-ready(1) → Wait K-ready(1)
VF scale K → V→MTE3 Set/Wait(1) → Upload Kbg
原 PIPE_MTE3 Stage6-ready → Cube Stage7
```

所有原 Set/Wait、event ID、跨核 ready/free 的次数与顺序保持；提前的是 K 的 DataCopy，K-ready 的 Set/Wait 仍在 Vb 上传下发之后。V-ready 必须发布在 K 搬运之前，否则 V 的等待会额外覆盖 K 的传输而失去目标重叠。SetFlag 在源流水前序完成后设置标志，不阻塞源流水后续指令；WaitFlag 约束目的流水。这项 API 语义依据 [Ascend C 核内同步说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/advanced_kernel_programming/intra_core_sync/mutex_multi_pipeline_sync.html) 核对，具体生成指令和实际重叠仍需本轮 CANN 构建及设备 profiling。

| 同步资源 | 原生产/等待与保护对象 | 本次处理 |
| --- | --- | --- |
| MTE3→MTE2，kHeUb0，pack 入口 | 原 Stage1/3/前 pack 写出完成后，才覆盖其别名 UB | 保留，K 预取仍在其后 |
| tail 的 MTE3→V、V→MTE2，kHeUb0 | 尾块清零前保护复用、清零完成后允许 GM 搬入 | 保留，K/V 仍只覆盖有效行 |
| MTE2→V，kHeUb0 | V 搬入完成后允许 V 缩放 | Set 在提前 K Copy 之前，Wait 在其之后，事件仍只发布前序 V 数据 |
| V→MTE3，kHeUb0 | V 缩放完成后允许上传 Vb | 保留 |
| MTE2→V，kHeUb1 | K 搬入完成后允许 K 缩放 | Set/Wait 保留原位置；提前 Copy 已位于它们之前 |
| V→MTE3，kHeUb1 | K 缩放完成后允许上传 Kbg | 保留 |
| Stage6-ready，PIPE_MTE3 | 同一 task 的 Vb/Kbg 上传后发布，Cube 既有等待后消费 | 保留原 Notify/Wait；跨 task、pack、尾任务计数不变 |

无新增 event、barrier、buffer 或 slot。两个 task 的 db 分别为 0/1，同一 pack 内不重用；下一 pack 的覆盖仍经原释放闭环。Stage4/5/7 的代码和 L1 上传位置不修改，现有相互重叠关系保持。

### 增量评审与验证界限

本次按 R02/R08/R11/R13/R17 复核输入就绪、地址区间、别名释放和可能并行的 K/V；按 R10 复核每个动态分支恰好读取一次 K；R21 下指数/乘法/cast 与尾块的零值处理维持原调用。既有任务分工、直接 UB→L1 数据路径及其它 Stage 设计在本次不迁移或重写，不将此局部修改宣称为全算子规则迁移。

已确认的唯一主要变量是 K DataCopy 的位置。预取可能受 MTE2 队列、带宽或原 Cube 流水影响，不能从静态顺序宣称实际 overlap 已发生。代码审查结论为可实施实验；正式采用条件为 CPU 标杆/受影响路径精度通过、普通七项 kernel 时间和跨批次下降超过波动、case 1 及其它支持场景无不可接受回退。开发期结果与设备待验证矩阵见 [validation.md](validation.md)。

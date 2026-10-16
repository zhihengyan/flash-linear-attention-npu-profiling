# Prepare Stage3 NZ8 批量上传实验

方案设计规则版本：V2。分支 `codex/gdn-stage3-nz-upload`，直接基于 `codex/gdn-prepare-k32` / `d0cb628df7c8c23c0342392dfaf4ec910aba9947`。接口见 [api.md](api.md)，K32 求解见 [prepare_k32_design.md](prepare_k32_design.md)。本轮仅改变 Stage3 结果的 UB 布局与上传粒度，不继承 N32/Block32 的计算或调度改动。

## 1. 目标与实验边界

沿用原公开前向调用链和普通/Profile共用的Prepare kernel。原Stage3每task上传−L需8次DataCopy，两leaf再需8次；case0有5664tasks、90624次上传调用。实验先在UB用Gather将结果打包为NZ8，再以1次完整−L和2次leaf矩形拷贝上传：每task16→3次，case0减少73632次，降幅81.25%。这些是源码调用数，不是已测硬件指令数。

搬运的数学结果、FP32位模式、L1地址、跨核ready/free以及Stage4/5的K32计算保持。代价是每task额外96组Gather/StoreAlign、48KiB UB读写流量、两次Vector store→load本地屏障，以及一次V→MTE3 event对；不得只按调用数减少宣称加速。新增Vector成本若抵消收益，则停止本版，不叠加leaf求解算法变化。

模型case0：Ascend950PR_957c，B1/HK16/HV32/T11274/K=V128/BT64，BF16 q/k/v和beta、FP32 g/初态，G2、varlen及末态输出。177 chunks，5664 tasks，pack4/1416 packs，28 AIC/56 AIV，各AIC50/51 packs；每AIV100/102 tasks。case1为原B2/HK=HV32/T8192/K=V128定长模型，同样须回归；没有其本轮基线数据，不推算收益。

以2026-10-16用户提供的K32普通case0为基线：七项1610.88620us，Prepare620.28335us；约2.7%参考对应43.49393us，若全来自Prepare则需降低7.01195%。2.7%不是硬门槛。旧K64 NoBarrier上传窗口只提示开销位置，不能乘task数推导节省；尚无证据证明本实验可获得上述净收益。

主指标仍为每次 `Cast→Transpose(g)→Transpose(beta)→Prepare→H→O→Transpose(gCumsum)` 七项Task Duration(us)之和。最后转置为gCumsum `[B,H,T]→[B,T,H]`。普通ATK原采样配置保持（历史每轮20组）；warmup5只控制诊断重放。保留输入、golden、精度阈值、公开输出和支持范围。

## 2. Stage详设与资源

沿用CHUNK_INDEPENDENT V1参考（来源 `a78efa9512dfa64c20b2a58a75e84e88a054d694`）的独立chunk、任务隔离和原ready/free闭环。工程参考仅当前K32自身源码；不据本轮声称完成R01/R06全量迁移。既有Stage3的UB→L1边界保持，本轮未引入新的Vector→Cube边界。公开H仍有chunk间依赖，不改H/O。

### 布局与绝对地址

原Stage3−L在UB是64×64 ND；leaf结果在UB是32×64 ND，两32×32物理leaf并排。两leaf物理内容是逻辑逆矩阵的转置，实验仅重排其字节，不再次转置或重新求解。

| 区域（KiB，半开） | 用途、生产与最后消费 |
| --- | --- |
| UB[0,0.25) | 新增常驻64个uint32 Gather索引；Stage0一次生成，Stage3只读；原[0,0.5)为空洞 |
| UB[0.5,9) | 原VCS索引及单位阵保持 |
| UB[9,10) | 两db的g/beta resident保持 |
| UB[10,50)/[50,90) | 原Stage3 LPacked/ResVcs/LFull两槽保持；含同时存活的pong gate，不能用于新scratch |
| UB[90,106)/[114,130) | db0/1新的16KiB −L NZ8 scratch，Stage3打包→MTE3末次读取 |
| UB[106,114)/[130,138) | db0/1新的8KiB双leaf NZ8 scratch，VCS后打包→MTE3末次读取 |
| UB[138,140) | 本阶段空闲，未使用 |
| UB[140,172) | 原KKT ping/pong保留，GVA共享输入不能当scratch |
| UB[172,236)/V256的既有区域 | 原WU输出及其它资源保持；物理UB分配上限不变 |
| L1[64,128) | 原四个NegL槽，仍为64×64 FP32 NZ8；与khat跨pack别名闭环保持 |
| L1[128,192)/[192,256) | 原四个Right/Left槽，TL/BR象限更新，其余象限保持Stage0零 |
| 其它L1、L0、GM/workspace | K32全部保持，最高L1地址496KiB；无新增GM或host分配 |

新scratch复用的是S1的hat/rstd等已经释放的UB[90,138)，不是增加UB容量。ProcessAiv完成本pack全部Stage1后，原MTE3_V事件已经保证S1输出写回完成，才进入Stage3；Stage3当前slot和另一slot的LFull均不被覆盖。S3完成后，原ProcessAiv中的MTE3_MTE2 Set/Wait保证上传消费完毕，再进入Stage6覆盖相关区域。下一pack Stage1仍受既有同步保护。两个AIV各有私有UB，各db独立scratch。

### Stage0：Vector，生成常驻Gather索引

用int32 Arange、ShiftRights/Muls/Add产生 `index[j]=(j//8)*64+j%8=(j//8)*56+j`，j=0..63；以相同b32位模式存入uint32索引区。全部值非负且最大455，无溢出或舍入。紧随原FillVcsIdentity，复用其已有V→S/S→V顺序；新pack VF入口另有Vector store→load本地依赖屏障。没有新跨核event或全pipe barrier。

### Stage3：Vector/MTE3，打包并上传

每次Gather读取8行×8列（64个float），StoreAlign写连续256B。ND源行跨度固定64。令rows=64/groups=1处理−L，rows=32/groups=2处理双leaf，cols=64/groups；循环g/fc/rb的地址为：

```text
srcBase = rb*8*64 + g*cols + fc*8
dstBase = g*rows*cols + fc*rows*8 + rb*64
dst[dstBase+j] = src[srcBase + (j//8)*64 + j%8]
```

负L输出恰覆盖4096个float，leaf输出两个1024-float NZ8 tile，不改变元素值。每个Gather base为8元素倍数（32B），最大读偏移在合法源矩阵内；所有输出每64元素一段、不重叠。

−L上传 `DataCopyParams(1,512,0,0)`，共16KiB。每个leaf上传 `DataCopyParams(4,32,0,32)`：4段，每段1KiB，目的段间隔1KiB。Right目的偏移0，Left目的偏移2304 float（9KiB），恰覆盖原64×64槽的TL/BR象限；未写部分保持原零。两leaf源相隔1024 float（4KiB）。总上传字节仍24KiB/task，即case0的132.75MiB，减少的是调用粒度而非有效字节量。

完整顺序：

```text
原Wait KKT → 原MulKktGateVF → 原V_MTE3 Set/Wait
→ 原PackDiagLeavesFromUb → 原MTE3_V(kHeS3Pack) Set/Wait
→ 新Pack<64,1>(−L) → 新V_MTE3(kHeUb0) Set/Wait → 新1次−L上传
→ 原MulReduceScatterVF32（算法/循环/归约均不改）
→ 新Pack<32,2>(leaf，入口Vector STORE→LOAD本地屏障)
→ 原V_MTE3 Set/Wait → 新2次leaf上传 → 原Stage3 ready
```

负L MTE3上传仍可与VCS并行；新增−L和leaf scratch不重叠，无需等待前者结束才打包leaf。Stage3 ready仍在原MTE3上按序发布，Stage4消费、NegL free及下一pack khat复用规则保持。S6/S7和ProcessAic/Aiv调度不变，含G1/2/3/4、pack3/4、tail、partial pack及最终drain。

### 数值、资源与API依据

只新增Gather/StoreAlign位复制，不新增浮点运算或改变FP32归约/BF16 cast时机。−L和leaf的NaN payload、±0等位模式应原样重排；离线检查使用uint32地址模型验证，不能据此宣称硬件指令或正式设备精度通过。

官方[UB Gather](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/Gather_discrete.md)以元素为索引，float配uint32，base需32B对齐。[Arange](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/index_operations/Arange.md)不支持uint32，因此索引生成采用int32，写入后按uint32读取。

索引算术另按官方[ShiftRights](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/ShiftRights.md)使用int32向量和int16位移量；[Muls](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Muls-27.md)使用int32向量/标量。API文档支持关系不能替代目标环境实际编译。

未选择的快捷路径：[LoopMode](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move_aux_config/SetLoopModePara.md)支持GM↔UB而非此UB→L1；[UB ND2NZ](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%85%A5/DataCopy_GMToL1_ND2NZ.md)说明其UB接口借Matmul workspace做GM中转，不能当作无成本直接搬运。

## 3. 验证与采用条件

离线核对真实C++参数、NZ8排列、目的象限、零区/NaN位模式及全部valid1..64；检查两槽别名边界，并以生产调度与K32逐方法对照保持作为不同G/V、partial pack及连续pack生命周期依据，不将其当作这些设备case已经执行。对比K32验证VCS/MMAD/host/ABI及未受影响阶段保持。现有K32历史测试冻结为d0cb628d，新的Stage3检查单独验证当前代码。

本机无CANN/NPU，不能执行编译、设备精度或真实性能验证。按既有用户授权完成整链接入，保持原wheel→安装→精度→profile入口；不因缺逐Stage设备结果留下未接入的helper。设备精度、MSS、确定性及完整支持范围仍须原标准通过。最终采用以同环境独立重复的普通七项净收益决定，辅看Vector/MTE3/Scalar字段，不能用调用数或旧timer窗口代替。

设备命令与结果统一写入 [validation.md](validation.md)。

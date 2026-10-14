# Prepare Stage4/5 半区 K 候选

规则版本：V2；现有算子性能优化，按阶段 03 → 04 → 05 执行。基线为 `codex/gdn-profiling` 的 `809bce6887639e831caa856b8ac3a197a6a94294`，候选分支为 `codex/gdn-prepare-k32`。这是待设备验证的实验方案，尚未采用为性能达标方案。逐阶段开发沿用 CHUNK_DEPENDENT V1 参考（来源 commit `a14961e9dfe8c8264768d67fb447101873da4ecf`）的原有 H 依赖链。

本轮唯一主变量是 Prepare arch35 Stage4/5 中三个 FP32 MMAD 的归约区间。Stage3 布局/上传、Stage4 的 I@I 预填、Y 的 GM 中转、Stage5 的 BF16 转换/输出、Stage6/7、H/O、host、TilingKey、公共 ABI、计时布局和 ATK 输入/CPU 标杆/容差沿用基线。Stage 编号指现有源码中的函数编号；本轮不重新划分整个算子的 Stage。

## 输入、基线与目标

目标为 Ascend950PR_957c、ATK 前向性能 case 0：q/k `[1,16,11274,128]` BF16，v `[1,32,11274,128]` BF16，g `[1,11274,32]` FP32，beta 同形 BF16；BT64、BNSD、GVA=2、varlen、FP32 initial_state `[1,32,128,128]`、输出 final_state，scale=1/sqrt(128)。精确生成规则由既有 case JSON 和 executor 定义，不改值域。

A0 普通调用 20 组报告：Prepare=715.08995us，H=588.8026us，O=359.37605us，Cast=4.97105us，三次 Transpose 合计40.15175us；七个 kernel 的 Task Duration 合计均值1708.3914us。验收要求七个 kernel 合计至少下降5%，即相对这份历史基线需节省85.41957us、达到1622.97183us以下。实际采用判断必须使用新鲜 A/B/A 重复测量，历史数据仅作预算。

`D:/Code/results/msopprof_r0` 的 Prepare 单次 kernel replay 为708.430969us，28 AIC/56 AIV、1650MHz。AIC Cube 活跃时间均值541.390136us、比例77.56251%，MTE1 比例23.2370%；各 pipe 并行，比例不是耗时份额。该采集与普通 msprof 的缓存/采样方式不同，不能据此计算优化收益，也没有已提供的 PipeTimeline 可证明唯一关键路径。它支持尝试减少 Cube/MTE1 工作量，但是否处于暴露关键路径仍待普通 A/B 和新 msOpProf 证据。

## 代数依据

令 R00=(I32+L00)^(-1)，R11=(I32+L11)^(-1)。逻辑矩阵 Right=diag(R00,0)，Left=diag(0,R11)，N=-L。Stage0 清零完整 leaf 槽；Stage3 只覆盖相应32×32对角象限，跨 pack 保持其它象限为零。因此对有限的操作数：

| 源码操作 | 基线 M,N,K | 候选归约区间 | 初始化 C |
| --- | --- | --- | --- |
| Stage4_PrefillI：I@I | 64,64,64 | 保持 K64 | true |
| Stage4_AccIssue：Left@N | 64,64,64 | K=[32,64)，K32 | false，累加到 I |
| Stage5_Compute 第一次：I@Left | 64,64,64 | K=[32,64)，K32 | true |
| Stage5_Compute 第二次：Y@Right | 64,64,64 | K=[0,32)，K32 | false |

Y=I+Left@N，A=Left+Y@Right 不变。即使 chunk.M<32，计算仍使用已补齐的64×64 tile；尾块的零元素及 leaf 对角单位元素由基线生产，不能把 chunk.M 用作本轮 K 长度。M/N 都保持64，未裁剪输出。

## 物理布局和加载描述符

FP32 L1 槽仍为64×64 NZ，C0=8。已存入 L1 的 ND 矩阵 S 的地址（float 元素）为 `(col//8)*512 + row*8 + col%8`。leaf 的物理 S 是逻辑 leaf 的转置；I、N、Y 的物理 S 是逻辑矩阵本身。不能只给通用方形加载函数传 K32：Stage4 需要选择物理源的32行，再转置；Stage5 则选择物理源的32列。

下表适用于同一次 MMAD 的 A、B 两次 LoadData；起始地址仍使用原 tensor 基址：

| 调用 | 物理源切片 | mStartPosition | kStartPosition | mStep | kStep | srcStride | dstStride | ifTranspose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage4 Left/N | S[32:64,0:64] | 2 | 0 | 2 | 8 | 4 | 4 | true |
| Stage5 I/Left | S[0:64,32:64] | 0 | 4 | 4 | 4 | 4 | 4 | false |
| Stage5 Y/Right | S[0:64,0:32] | 0 | 0 | 4 | 4 | 4 | 4 | false |

mStart/mStep 单位16元素，kStart/kStep 单位32字节，stride 单位512字节；FP32 转置的 kStep=8 满足偶数约束。源仍为完整64行，srcStride=4。L0A 目标为64×32 NZ，L0B 为32×64 Zn，其目标步长均为4；Stage4 的 dstStride **不是**源 mStep=2。转置后目标仍有64行（A）或64列（B）。此推导依据官方 [LoadData 2D V2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/Load2DV2.md) 和 [3510 转置 A 的配置示例](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cube_matrix_computation.html#l12l0-数据搬运)。

每个源切片最大地址不超过4095个float，每次写入 L0A/B `[0,2048)` float（8KiB），不读取槽内后8KiB旧数据。原每银行16KiB槽及 tensor.SetSize 保持，银行0/1及P银行的偏移、别名和生命周期不变。L0A/B 起始地址原已满足512B对齐；源起始选择分别位于1024B或8192B偏移，满足32B对齐。L0C 仍写完整64×64 FP32。

## 同步、资源与适用范围

专用模板仅用于以上三处调用。两次 LoadData 后仍按原顺序 Set/Wait MTE1_M，再 Wait FIX_M 后发 MMAD；外层 Wait M_MTE1、Set M_FIX、Set M_MTE1 保持。Stage5 两次 MMAD 之间原 Set FIX_M 保持。Stage4_PrefillI 和所有 Dump/notify/跨 pack ready/free 均保持。

| 区域 | 本轮分配增量 | 生命周期/所有权 |
| --- | --- | --- |
| L1 | 0 | 每 task 原 Y/NegL/Leaf 槽；Stage3 ready 后消费，原 Stage4/7 放行复用 |
| UB | 0 | Stage0/3/6 原生产区、AIV task 分工保持 |
| L0A/B | 0 | 原四个16KiB银行；候选只写前8KiB，原消费完成后复用 |
| L0C | 0 | 原 task 独立结果槽；完整64×64输出，原 Fixpipe 完成后复用 |
| GM/workspace | 0 | 原每 AIC 64KiB Y workspace；没有增加输出写者 |
| event/flag | 0 | 原编号、发布/等待次数和顺序保持 |

host 已强制 Prepare BT64、K128、V128/256、GVA1..4，因此所有 arch35 现有合法 Prepare 路径都使用本候选；不用新 runtime 分支，也不声称只影响 case0。G=3、部分 pack、varlen、tail、output_a=False、FP32/BF16 gate 等都必须回归。其它架构的实现不变。R01–R20 涉及的已有任务映射/资源/同步继续按源码校准，不借本次局部优化宣称全实现完成V2迁移。

## 数值和收益边界（R21）

计算仍为 FP32→FP32，A 仍按原 Fixpipe 转 BF16；没有 HF32、新近似、新指数或 cast/mask 顺序变化。删除的是 leaf 结构零对应的乘加，不增加中间值的数学幅度。若 `|N|<=l`，严格下三角32块的逆可按 `|Rij|<=l*(1+l)^(i-j-1)`、对角1上界检查；取 `r=max(1,l*(1+l)^30)`，Y 的有效单元上界为 `1+32*r*l`，A 为 `r+32*(1+32*r*l)*r`。例如 l<=2 时 r<=4.118e14、Y<=2.636e16、A<=3.474e32，低于FP32/BF16最大有限值。此条件界限不是新增输入约束；原值域是否满足及较大中间值的有限性仍需按原 Stage3 采数确认。

这不是对任意非有限输入的位级等价证明：IEEE 0×Inf/NaN 不等于有限零，硬件改变归约长度也可能改变舍入/有符号零。既有 ATK 使用有限值检查及原精度标准；设备必须验证现有完整值域、逐输出精度和确定性。不得为通过修改 generator/CPU golden/容差；若出现溢出或精度失败，按精度定位路由退回本设计，不能把 CPU 模型通过当作设备通过。

四次 FP32 MMAD 的每 task MAC 从4×64³=1048576降到64³+3×64²×32=655360（下降37.5%）；三次 MMAD 各两次加载从共96KiB降到48KiB。Stage4 prefill 的共享加载仍为每 pack 32KiB，其它 GM、UB→L1、Fixpipe 字节不变；MMAD 和加载调用次数不变。case0 的5664 tasks 理论少2227175424 MAC、48KiB/task L1→L0传输。该比例只针对这些 FP32操作，不能作为端到端收益。

达到历史5%目标，若全部节省仅来自 Prepare，则 Prepare 至少下降11.9453%。以本次 Cube 均值做理想估算，85.41957us约为 Cube 活跃时间的15.778%，三处裁剪所在 FP32部分需占 Cube 活跃时间约42.07%且节省完全暴露才够；当前报告没有分离这一份额。故可达性为“有计算减量空间、未证明5%”，候选值得一次精度和普通性能实验，具体计划见 [validation.md](validation.md)。

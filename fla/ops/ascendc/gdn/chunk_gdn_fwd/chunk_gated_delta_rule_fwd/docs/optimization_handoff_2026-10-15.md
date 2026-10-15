# GDN 优化任务交接摘要

更新日期：2026-10-15。本文供新会话继续定位和实验使用，汇总已完成的尝试与证据，不代表完整设备验收完成。

## 1. 接手目标与当前决定

用户希望继续为 Ascend950 的 GDN 寻找值得采用的优化，重点为前向模型性能 case0。此前明确认为整体用时降低至少 5% 的改动才值得投入。

当前应保留 **Prepare K32**，以其作为后续优化的业务基线，在独立新分支研究新候选。Stage6 提前搬运 K 和 H 双 head lookahead 均未采用，不叠加进 K32。K32 相对同日原始 A 的七个 kernel 合计均值降低 **5.424%**，该项收益复核已经完成，不要再将“补测新鲜 A”作为当前待办。

K32 的完整支持范围、case1、MSS、确定性等正式回归尚未全部有报告。应保留当前候选代码与 wheel，完成回归；可以继续做只读定位和方案分析，进一步实验须从 K32 单独分支开展，不覆盖已验证候选。

用户此次要求的是交接摘要；本摘要未实施新的性能优化，也未授权推送、合并或删除历史分支。

## 2. 仓库、分支和工作树

| 用途 | 分支/关键提交 | 本机位置 |
| --- | --- | --- |
| 桌面会话主目录，历史 Stage6 实验 | `codex/gdn-stage6-overlap`，HEAD `dddd7bce` | `D:/Code/flash-linear-attention-npu` |
| 原始业务基线 | `codex/gdn-profiling`，`809bce6887639e831caa856b8ac3a197a6a94294` | 当前挂在 `C:/Users/zhihengyan/.codex/worktrees/gdn-prepare-stage3-5-timer/flash-linear-attention-npu`；目录名与当前分支名不同 |
| **应保留的 K32 候选** | `codex/gdn-prepare-k32`；优化提交 `dd707730627b29b1862fd335dfbc76e7f0e2666d` | `C:/Users/zhihengyan/.codex/worktrees/gdn-prepare-k32/flash-linear-attention-npu` |
| Stage3–5 诊断工具及分析 | `codex/gdn-prepare-stage3-5-timer`，`c4c46cc9` | 当前没有工作树检出该分支，可先用 `git show` 阅读 |
| 历史 H 双 head 实验 | `codex/gdn-h-dual-head-lookahead`，`f1d084d4` | `C:/Users/zhihengyan/.codex/worktrees/gdn-h-dual-head-lookahead/flash-linear-attention-npu` |

K32 优化提交直接基于 809bce68；之后 `666cd98e`、`de28a027` 仅补充性能分析文档，实测内核源码仍为 dd707730。本交接文件新增前，主目录和 K32 工作树均无未提交改动。接手时重新检查 `git status`、`git worktree list` 和实际 HEAD，不能把主目录的 Stage6 分支当作 K32。

已有合适 K32 工作树，应优先复用其源码和资料；如用户要求新实验分支，从 K32 出发，使用 `codex/` 前缀，保留候选分支用于回归和追溯。

## 3. 必读资料与流程

首先读取当前工作树的根 `AGENTS.md` 和 `docs/agents/README.md`。内部性能优化按阶段 **03 方案设计 → 04 算子开发 → 05 算子测试**，读取当前接口、CPU 标杆、设计、实现、测试和开发期验证记录。优化保持公开接口、语义、支持范围、CPU golden、输入生成与精度阈值。

K32 工作树内的入口资料（下面路径相对该工作树根目录）：

```text
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/docs/
  timer_analysis_k32_same_day_ab_2026-10-15.md   # 最新收益判断，优先阅读
  validation.md                               # 当前验证状态、执行步骤、待验项
  prepare_k32_design.md                        # K32 代数、物理布局、同步和资源依据
  design.md                                   # 当前算子设计入口
  timer_analysis_k32_b1_b3_2026-10-15.md         # 历史 A0 对比，已由同日复核更新
  timer_analysis_case0_2026-10-13.md            # 原始 case0 分析和原 brief
  timer_analysis.md / timer_flow.md / timer_run.md
```

部分早期设计段落保留“待验证”的原始预算表述；**实测状态以最新同日分析和 validation 为准**。历史文档中“新实验从 gdn-profiling 出发”的建议属于 K32 之前的阶段，下一轮不应因此丢弃 K32。

历史分析分别在对应分支：

```text
codex/gdn-stage6-overlap:
  .../docs/timer_analysis_stage6_b1_2026-10-13.md
codex/gdn-h-dual-head-lookahead:
  .../docs/timer_analysis_h_dual_head_2026-10-14.md
  .../docs/timer_optimization_direction_2026-10-14.md
codex/gdn-prepare-stage3-5-timer:
  .../docs/timer_analysis_stage35_2026-10-14.md
  .../docs/timer_prepare_stage35_design.md
  .../docs/timer_prepare_stage35_audit.md
  .../docs/timer_prepare_stage35_report.md
```

这里 `...` 为 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd`。无需切换分支即可使用 `git show 分支:完整仓库相对路径` 读取。

## 4. 模型与性能口径

设备为 Ascend950PR_957c，内部环境 Python3.11.10，已提供摘要的 torch 为 2.10.0+cpu。主要 case0：

- q/k：BF16 `[1,16,11274,128]`；v：BF16 `[1,32,11274,128]`。
- g：FP32 `[1,11274,32]`；beta：同形 BF16。
- chunk_size=64，BNSD，GVA=2，varlen=true，scale=1/sqrt(128)。
- FP32 initial_state `[1,32,128,128]`，output_final_state=true。

既定验收口径是普通调用 msprof 的七个 kernel **`Task Duration(us)` 逐调用求和后的均值**：

```text
Cast → Transpose → Transpose → Prepare → H → O → Transpose
```

这是设备任务时长之和，不包含 host 和 launch 间隔，不应称为完整墙钟端到端时延。每轮普通采集有 140 行，即 20 次调用。K32 同日 A/B 各三轮，共各 60 次；不能称为各 60 个独立环境实验。

保留全部样本，不挑最快一轮；总分位数从逐次七项和计算，不能相加各 kernel 分位数。最新普通性能 p95 为排序位置 `0.95*(n-1)` 线性插值，部分旧 timer/报告使用 nearest-rank，引用时区分。

所有主要实验的 case JSON SHA256 相同：
`33530343cec97fc0d7ea7257229221c1f429d547d6e86699ccb23696f39858c0`。
输入快照 SHA256 相同：
`1f4a29b6a40feb83a424979370638c0b1fb6ee51fa80dd0e513e5d22de73ef38`。

## 5. 已做尝试及取舍

| 尝试 | 主变量与实测 | 当前判断 |
| --- | --- | --- |
| 原始 A0 | 七项 1708.39140μs；Prepare715.08995、H588.80260、O359.37605μs | 历史预算与回溯基线；当前收益优先用同日 A |
| Prepare Stage6 提前搬运 K | 把 K 搬运提前，尝试与 V 处理重叠；七项约1708.209μs，仅约下降0.0107%；Prepare反而增加约0.0983% | 停止、未采用；不再要求为该候选补测 A |
| H 双 head lookahead | 联合调度两条 head 的 W/K 输入与 bank/credit；H降至580.68085μs（-1.379%），七项1699.62250μs（-0.513%） | 远低于5%，停止、未采用；不能把其收益直接加到 K32 |
| Prepare Stage3–5 细打点 | 多窗 coarse/detail、两轮重复，共16 captures；普通七项1711.27750μs，Prepare717.34270μs | 定位工具，不是优化版本；获得上传/求解/中转的下发路径信息 |
| 原始实现 msopprof replay | Prepare708.430969μs，28AIC/56AIV，1650MHz；AIC Cube均值541.390136μs/77.56251%，MTE1约23.2370% | 支持研究计算和加载减量，不证明唯一关键路径 |
| **Prepare K32** | 三处 FP32 MMAD 裁去结构零对应半区；三轮 B 七项合并1612.33015μs，Prepare620.64430μs | 相对同日 A，七项-5.424%、Prepare-13.186%；保留并完成正式回归 |

Stage6、H 的收益均来自单批历史对照，不能将小变化解释为已确认的稳定增益，也不能由这两项失败推断整个算子已无空间。K32 已表明结构性计算减量可以达到既定目标。

## 6. Stage 编号和 K32 的具体改动

Stage3/4/5/6/7 指 Prepare 内核源码中的函数阶段，不是七个 launch 的编号，也不等于 profiler stage ID。

主要实现位于：

```text
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/
  chunk_gated_delta_rule_fwd_prepare.h
  chunk_gated_delta_rule_fwd_prepare_matmul.h
  chunk_gated_delta_rule_fwd_prepare_datacopy.h
  chunk_gated_delta_rule_fwd_prepare_vf.h
  chunk_gated_delta_rule_fwd_prepare_common.h
```

| 阶段 | 源码操作 |
| --- | --- |
| Stage3 | KKT 后 gate 处理，完整 −L 上传、两个32×32对角叶块的打包/求解/上传，发布下游 ready |
| Stage4 | `Stage4_PrefillI` 和 `Stage4_AccIssue` 计算 Y；`Stage4_Dump` 将 FP32 Y 经 GM 回读 L1 |
| Stage5 | `Stage5_Compute` 合成 A；`Stage5_Dump` 输出/转换 A，并向下游提供 L1 A |
| Stage6 | `Stage6_AivOne` 搬运 V/K，施加 beta/g 的逐行缩放，上传供 W/U 使用的输入 |
| Stage7 | `Stage7_AicOne` 做 W/U 矩阵乘和原输出流水 |

设 `Right=diag(R00,0)`、`Left=diag(0,R11)`，`Rii=(I32+Lii)^-1`，`N=-L`。Stage0 清零完整 leaf 槽，Stage3 仅写对应对角象限，因而可以使用结构零裁剪：

| 调用 | K32 归约区间 | initC |
| --- | --- | --- |
| Stage4 I@I 预填 | 保持 K64 | true |
| Stage4 Left@N | K=[32,64) | false，累加到 I |
| Stage5 I@Left | K=[32,64) | true |
| Stage5 Y@Right | K=[0,32) | false |

公式 `Y=I+Left@N`、`A=Left+Y@Right` 保持。实现只改 `.h` 和 `_matmul.h`，添加 `FillSolveHalfKLoad`、`SolveHalfKMatmulToL0C` 专用 helper。原通用 helper、事件次序、缓冲分配、task映射、Stage3上传、Y/A中转、Stage6/7、H/O、host和ABI均沿用基线。

每 task 四次 FP32 MMAD 的总 MAC 理论减少37.5%；三处 A/B 的 L1→L0 加载总量96KiB→48KiB。该比例不等于整个 Prepare 或前向收益。没有减少 Stage3 UB→L1 字节或调用数。

布局敏感点：FP32 L1 为64×64 NZ、C0=8，leaf 物理矩阵为逻辑转置。Stage4 选择物理源行32..63并转置，`mStart=2,mStep=2,kStep=8,srcStride=4,dstStride=4`；**dstStride不是2**。Stage5 选择物理列半区，`mStep=4,kStep=4,srcStride=dstStride=4`，kStart=4或0。L0每个原16KiB槽只写/读前8KiB，L0C仍完整64×64。

该 helper 对 arch35 原合法 Prepare 路径生效，影响范围不只case0：GVA1..4、V128/256、tail、varlen、output_a等均需回归。FP32和既有BF16转换保持；改变归约长度仍需设备精度/确定性验证，不能宣称对任意 NaN/Inf 位级等价。

## 7. 最新同日 A/B：当前应引用的数字

| 轮次 | 七项均值/μs | Prepare均值/μs |
| --- | ---: | ---: |
| A1 | 1705.45645 | 715.13815 |
| A2 | 1704.92515 | 714.81085 |
| A3 | 1704.02030 | 714.79455 |
| A合并 | **1704.80063** | **714.91452** |
| B1 K32 | 1613.28590 | 623.60130 |
| B2 K32 | 1612.17790 | 620.26910 |
| B3 K32 | 1611.52665 | 618.06250 |
| B合并 | **1612.33015** | **620.64430** |

- 七项净节省92.47048μs，**降低5.424%**；Prepare节省94.27022μs，**降低13.186%**。
- H：586.75883→588.33007μs，O：357.92022→358.23867μs，合计慢约1.88968μs，部分抵消Prepare收益。
- 已观测最保守轮均值组合：最快A3对最慢B1，仍降低5.325%。九种轮均值组合不是九次实际配对实验，更不是置信下界。
- 七项中位数1705.04800→1611.23100μs，降低5.502%；经验p95为1714.38475→1629.42160μs，**降低4.956%，没有达到5%**。既定均值目标已满足，不扩写为所有统计量或每次调用均满足。
- 七项样本标准差A5.87320、B9.50346μs；B逐调用波动较大，但三轮均值极差只有1.75925μs（0.1091%）。
- 当前5%目标线1619.56060μs，B均值余量7.23045μs。新A比历史A0快0.2102%，故原先历史对比5.623%更新为同日5.424%。

六份普通CSV均为140行，名称/耗时与各自metadata完全一致，输入hash相同；实际顺序、时间戳和任务编号已核查。采集是10:36左右三轮B，11:40左右三轮A，**不是实际交错A/B/A**。

版本身份边界：B metadata提交为dd707730。A由用户明确说明安装原始A0 wheel复测；其metadata的checkout提交为本地不存在的 `3b0664fc5680aac3745e945f34c62a0ac1014ccc`。采集器的 `source_commit()` 记录采集目录HEAD，不能直接当作wheel编译提交。接受用户已说明的A身份，正式归档保留wheel文件/hash和安装校验日志，不反复要求用户重新确认已说明的安装动作。

同为普通CSV的Prepare管线字段：MAC573.62250→390.68353μs（-31.892%）；MTE1 164.22017→161.29248；Scalar144.44062→144.97368；Vector347.08343→348.95947；AIV MTE3 322.58048→269.25022。支持计算减量解释，但pipe重叠，字段不能相加或直接充当关键路径时长；MTE3下降不证明Stage3已优化。原msopprof replay的541.39μs不能直接与这组普通CSV的390.68μs作收益比。

## 8. 旧打点已经解决和仍未解决的问题

原始 timer 每核每stage只保留前32次，984个core/stage计数截断；scope和是core-cycles，不是launch耗时份额。窗口可能包含下发、内部等待与队列背压，未观测区域不是已证明的硬件气泡。原核间窗口较均衡，不能因此断言无其它瓶颈。

Stage3–5诊断使用pack起点0/16/32/48，每窗最多8pack，coarse/detail各两轮，共16 captures。用户带出的brief确认为 `stage35_analysis/brief.txt` 的跨capture汇总，无需再要求同内容的share_brief或巨大逐task表。

原K64稳定区段：−L上传median738.5–752cycles，leaf上传387–415；源码每task共16次UB→L1调用，case0共5664tasks、90624次调用，逻辑有效上传量132.75MiB。这个计数是优化线索，不能把median×task数当可节省时长。UB约236/248KiB、L1约496/512KiB，不能默认新增布局scratch有空间。

**关键语义修正**：GetSystemCycle属于PIPE_S；Ascend950非PIPE_S的CrossCoreWaitFlag只阻塞指定流水。NoBarrier包围wait API得到的是标量观测窗，不能用S3_KKT_WAIT约4cycles、S4_INPUT_WAIT约34cycles证明硬件只等这么久，也不能用S3_LEAF_SOLVE约75cycles证明Vector求解很便宜。相关修正在Stage35分析中。不要为测量擅加逐阶段PIPE_ALL或把业务wait改PIPE_S，这会改变被测流水。

目前没有K32对应的完整细阶段硬件timeline，也没有证明其剩余唯一关键路径。K64的旧Stage排序不能直接沿用为K32的优化优先级。

## 9. 下一轮如何继续寻找候选

1. 先读取K32当前代码和最新数据，确认回归状态。使用现有普通CSV做筛选，必要时仅补能区分候选的K32 pipe/依赖证据；不要默认再重跑整套16capture打点。
2. 优先复查Prepare剩余的计算、布局、输出链。其均值620.64μs，仍是最大单项；H588.33、O358.24μs也不能忽略。最大单项本身不证明唯一关键瓶颈。
3. 未实施的结构候选包括：Stage3条带上传/地址下发减量，Vector leaf求解实际指令与依赖，Stage4 FP32 Y的GM中转，Stage5在保留公开GM A输出条件下给W/U供数的路径。先核实实际完成依赖、合法格式、容量、生命周期及事件，再选一项。
4. Y当前依靠GM ChannelSplit形成消费者使用的C0=8形态，不能直接把Fixpipe目的地址替换成L1。A是H真实输入，不能省掉应输出的A；tail/output_a=false的直写路径也不能未经布局证明照搬。H的FP32 state已常驻，P/D已直接写UB，不要把这些当成尚可消除的GM往返。
5. 新实验相对K32测增量，同时保留与原始A的整体收益。旧实验百分比不能线性累加；每轮只设置一个主要变量，并计入新增布局/同步/转换的成本。

原始5%目标已经由K32满足。若下一轮仍按“相对当前K32再降低5%”作为投入门槛，则需要再节省80.61651μs；其它项不变时，Prepare需约下降12.99%、H约13.70%、O约22.50%。这是条件预算，不是已找到空间，也不是用户已经单独确认的新验收要求。方案中应清楚区分“相对K32的增量”和“相对原始A的累计收益”。

若证据显示候选成本已被隐藏，或可实施节省不足预算，就停止该候选；不能只因某个timer窗缩短采用改动。

## 10. 正确性和正式回归状态

用户报告K32“精度测试全过”，但未带出完整case清单/报告，不能扩大为2416条、全部seed、MSS及确定性全通过。

已完成本机离线检查：`test_prepare_k32.py` 4项、`test_timer_contract.py` 14项，共18项通过；包括物理NZ地址/半区/转置描述符、错误stride负例、9有效长度×4seed×3幅度的108组数值模型、原helper和事件契约检查。这不是Ascend指令仿真，不替代设备验收。

K32开发期精度筛查为8批×12=96条，CASE_START为24/36/216/228/1224/1236/1416/1428，CASE_END=start+12且不含右端。正式Prepare共2416条，还需覆盖GVA1/3/4、V256、B>1、部分pack、varlen、tail、output_a及gate类型。前向双标杆500条分片、case1与Prepare六个perf case、MSS/确定性和Example/ST按现有验证计划与ATK README完成。

每批accuracy结束有独立XLSX及summary页；`check_atk_result.py`默认只看最新mtime报告，不自动汇总全部批次。不要把最新一批通过称为8批或全量通过。

旧 `check_timer_source.py` 比较固定main/文件内同步文本；K32新增专用helper会多一份Set/Wait源码文本，因此其原文本规则不适用。不要放宽旧计时分支规则或声称其通过。正式验收完成前保留validation；阶段05完成后才按仓库流程将有效结论收敛至ATK README并清理开发期记录。

## 11. 采集入口与二进制匹配注意点

K32继承809基线入口，支持：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0 --output-dir NEW_DIRECTORY
```

**此版本没有** `--reference-wheel`、`--ordinary-only`、`--prepare-stage35`、`--timer-repeats`；不要把其它诊断分支的新参数直接用于K32入口。

入口先采普通ATK/msprof，再采timer诊断；`--warmup`默认5仅控制诊断重放，不能据此说普通性能也只预热5次。编译、安装、校验二进制在精度/性能之前。

A/B可以使用对应预编译wheel，不必每次重编；但原入口安装检查对照当前目录build/中的.so及.o。仅切换Git/替换wheel不会恢复匹配build。使用各自匹配源码/build目录，或固定已验证支持reference-wheel的同一新版采集器用于两者；必须把采集器版本与实际wheel身份分开记录。新版采集器曾修复wheel内部归档路径识别，复用前阅读其实现，不盲目搬参数。

最小带出数据为每轮run_metadata.json和普通op_summary.csv；输入/raw大文件无需传出。若需要内部trace，只带可支持关键判断的小摘要、采集版本/模式和关键依赖说明。B与A均使用新进程、相同输入/采样条件，优先同环境临近/交错重复。

复制多行Bash曾因续行拆断导致 `--aic-metrics: command not found`；后续命令使用完整单行或Bash数组，参数按内部已安装msopprof版本help核对。

## 12. 证据位置与本机能力

所有用户结果均位于本机可读的 `D:/Code/results`，不要改写：

| 目录 | 内容 |
| --- | --- |
| `atk_output_profiling` | 原始A0，metadata内含普通140条耗时；timer摘要/三个CSV/log |
| `atk_output_stage6_overlap` | Stage6首轮B1 |
| `atk_output_h_dual_head_lookahead` | H双head首轮 |
| `atk_output_stage3-5` | 跨16capture的brief和run_metadata |
| `msopprof_r0` | 原实现replay的OpBasicInfo/PipeUtilization/Memory及log；没有已提供的PipeTimeline |
| `atk_output_k32/k32_B_1_20261015`、`k32_B_2_20261015`、`k32_B_3_20261015` | 每轮run_metadata.json及op_summary.csv |
| `atk_output_profiling_1015/A_1_20261015`、`A_2_20261015`、`A_3_20261015` | 用户安装原始A0 wheel的同日三轮复测，metadata及普通CSV |

本地可复用的分析脚本与忽略输出：

```text
D:/Code/flash-linear-attention-npu/tests/atk/chunk_gated_delta_rule_fwd/atk_output/k32_analysis/
  compare_k32.py
  summary_fresh_a.json
  launches_fresh_a.csv
  summary.json / launches.csv  # 较早的历史A0对比
```

脚本已经核对六份CSV与metadata、行顺序、每次七项分组、输入hash，并用Decimal独立对账。报告含源文件SHA256。无需重新索取已提供的新A数据。

本机Windows没有CANN/NPU/AscendC编译器，不能本地宣称完成设备构建/精度/性能。用户在内部Linux执行设备测试。可用Python位于：

```text
C:/Users/zhihengyan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe
```

Python3.12.14/NumPy2.3.5可做离线分析。K32工作树在主工作区外，写入或Git操作可能需要工具沙箱提权；不要因此另建一份不明身份的候选。新会话按当前权限操作，保留准确工作目录。

## 13. 给新会话的起始任务

先核对K32工作树和最新同日A/B报告，整理尚未完成的回归证据；随后基于K32的源码和现有pipe数据，为剩余Prepare/H/O路径提出有预算、有合法布局/同步依据的候选。先说明哪些结论已有证据、哪些需要内部补测，再选择一个主要变量设计下一轮实验。保留K32现有实现及其5.424%的已测均值收益，不自动合并Stage6、双H或Stage35诊断代码，不重复要求新鲜A，也不把旧K64打点排序直接当新基线瓶颈。

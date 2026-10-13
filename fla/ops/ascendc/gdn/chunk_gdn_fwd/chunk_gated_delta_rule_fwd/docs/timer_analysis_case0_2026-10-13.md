# GDN case 0：内部设备 timer 摘要分析

整理日期：2026-10-13。首轮基于用户提供的 `brief.txt`；第二轮读取三个汇总 CSV 和 `profiling.log`；第三轮读取 `run_metadata.json` 中从普通调用 msprof 报告提取的 Task Duration(us)。未读取 raw、完整 parsed.csv、trace 或原始 op_summary 文件。数据目录已从 `atk_output_timer_case0` 重命名为 `atk_output`，原四个文件哈希保持一致。

## 当前结论

本次实际采集到 **PREPARE → FWD_H → FWD_O** 三个 launch。各 launch、各核型的最长诊断总窗比中位数高 **0.65%～1.53%**，这一轮没有明显的核总窗长尾。下一步优先检查 Prepare 的 `WU_PREPARE`、`KKT_EPILOGUE`，以及 O 的 AIC/AIV 交接等待；这些是排查线索，尚不能确定关键路径瓶颈。

第二轮数据进一步支持：**首轮局部实验可优先考虑 Prepare Stage6 的 K/V 搬运与处理流水**，保持原数值运算及数据就绪约束；这是按现有证据和改动范围选择的候选，尚未证明收益最大。初始化已测窗口很小，最长 SOLVE 窗口集中在每核首个调用，O 等待普遍存在，直接改分核、求解公式或删同步都缺少充分依据。详细证据和候选顺序见下文“第二轮：四个文件补充分析”。

第三轮已取得 Prepare/H/O 各 20 条普通调用耗时记录：均值分别为 **715.090 / 588.803 / 359.376 us**，占同批七项 kernel 时间和均值的 **41.86% / 34.47% / 21.04%**。Prepare 是最大的单项，已有普通性能基线支持进入 Stage6 首轮单变量实验，不需要为了选择首个候选继续补相同的耗时摘要。20 条来自同一份采集，实际收益仍需同条件交替 A/B 复跑确认。

限制主要来自采样：**984 个 kernel/core/stage 计数项截断，每项只保留前 32 次调用**。H 的已采样窗口并集中位覆盖率只有 AIC 1.57%、AIV 0.95%，不能据此认为 H 的计算很轻或其余时间在空闲。单次诊断采集也不能证明性能回归或优化收益。

## Stage 编号和 A/B 的含义

本文的 Stage4/5/6/7 沿用 **Prepare Arch35 kernel 内部的阶段编号**，属于 PREPARE 这一个 launch；它们不是公开 GDN 调用链的四个独立 launch。H/O 实现也有各自的阶段编号，阅读时以所属 kernel 为准。

| Prepare 阶段 | 执行单元与操作 | 源码函数 | 对应 timer 名称 |
| --- | --- | --- | --- |
| Stage4 | AIC：三角求解的中间步骤。预填单位矩阵 I，计算 `Y = I + LeafLeft @ (-L)`，写出 Y 并搬回 L1，供 Stage5 使用 | Stage4_PrefillI、Stage4_AccIssue、Stage4_Dump | PREPARE_SOLVE_PREFILL、PREPARE_SOLVE_ISSUE、PREPARE_SOLVE_DUMP |
| Stage5 | AIC：计算求解结果 `A = LeafLeft + Y @ LeafRight`，转换并保存 A，供 W/U 矩阵乘使用；按 outputA/tail 分支选择 GM/L1 写出路径 | Stage5_Compute、Stage5_Dump | SOLVE、PREPARE_SOLVE_DUMP |
| Stage6 | AIV：把 V、K 从 GM 搬到 UB，按 beta/gate 缩放为 `Vb = V * beta_eff`、`Kbg = K_hat * beta_eff * E(g_prime)`，再转换布局并上传 L1；包含尾块处理及原事件等待 | Stage6_AivOne | WU_PREPARE |
| Stage7 | AIC：执行 `W = A @ Kbg`、`U = A @ Vb`，经 Fixpipe 将结果送到配对 AIV 的 UB；随后由 AIV 的 DrainPackWuToGm 写回 GM | Stage7_AicOne；后续 DrainPackWuToGm | W_MM、U_MM；二者只覆盖其中的矩阵乘调用窗口 |

这里 `@` 表示矩阵乘，I 为单位矩阵；L、LeafLeft、LeafRight 是前序 KKT/gate 和 Stage3 准备的求解矩阵。`g_prime` 是 chunk 内累加后的 gate，E 根据 useExp2 取 exp 或 exp2；K_hat 按配置为处理后的 K 或原 K。GM 是设备全局内存，UB 是向量侧片上缓冲，L1 是矩阵运算使用的片上存储。

编号和公式可在 [Prepare Arch35 源码](../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h) 的 Stage 4～7 注释及函数中核对；总体调用链、打点名称和代码位置见 [timer_flow.md](timer_flow.md) 的“改造前执行流程”“原图节点与观测名称”和“逐点配对清单”。

“Stage4/5/7 的已有重叠”指不同执行单元或不同 task 之间的流水重叠。Stage4/5 生成 A，Stage6 准备 Kbg/Vb；两路数据就绪后 Stage7 才能消费。同一 task 的 Stage7 仍须遵守这些依赖。Stage6 不需要等该 task 的 Stage5 完成，所以它的输入准备可与 Cube 求解重叠；处理不同 task 时，Stage6 也可能与 Stage7 交错。调整 Stage6 要保护这些既有关系。

**本轮 A/B 就是修改前后对照实验：A 为当前未优化的普通调用，B 为仅包含 Stage6 候选改动的普通调用。** 使用同输入、同设备与软件环境、相同预热和采样次数，先验证精度，再比较 msprof Task Duration(us)。本轮主要看整个公开调用的七项 kernel 时间和，同时检查 Prepare 单项及 H/O 是否发生代价转移；不以诊断 scope 变短代替实际提速。

建议 A→B→A 交替重复：最后恢复 A 是为了检查设备状态或测量条件的变化是否造成表面收益，不是第三个优化方案。第一轮只改变 K 输入搬运的下发位置，不同时修改求解公式、分核或其它流水；这样才能把收益或回退归因到这项改动。

## 采集条件与证据范围

用户报告的运行命令，在仓库根目录执行：

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cases 0
```

| 项目 | brief 中的记录 |
| --- | --- |
| 设备 / torch | Ascend950PR_957c / 2.10.0+cpu |
| 采集 / warmup | 1 次 / 5 次 |
| 已解析 CSV 校验 / decode | passed / complete |
| 诊断与普通调用输出对照 | 十项输出精确一致 |
| q、k | BF16，`[1,16,11274,128]` |
| v | BF16，`[1,32,11274,128]` |
| g | FP32，`[1,11274,32]` |
| beta | BF16，`[1,11274,32]` |
| initial_state | FP32，`[1,32,128,128]` |
| 选项 | chunk_size=64，BNSD，scale=0.0883883461356163，output_final_state=true，varlen=true |

这些状态是用户提供的摘要记录，说明该样本通过了工具中的相应检查，并且诊断与普通调用结果一致；它们不替代独立 CPU 标杆精度验收或全路径设备回归。第三轮 metadata 补充了 source_commit、OPP vendor 路径、成功性能用例记录和普通 kernel 耗时；尚未提供 CANN 版本、已加载产物的校验记录、采集时的变长序列边界或经核实的计时频率。

摘要没有 PHASE6 行。当前 [host 路由](../op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp) 的 `UsePreparePath` 条件与本次 BF16、K/V=128、chunk_size=64、Hv/Hq=2 的输入相符，普通 msprof 记录也确认 Prepare/H/O 三项。源码对应关系以当前仓库为依据；采集 source_commit 已核对相关 host/kernel/runtime 未改变，已加载构建产物仍需内部记录核实。

## 核总窗与采样覆盖

所有 duration 都是 cycles。表中“最长偏离”按 `(max / median - 1) × 100%` 由 brief 的整数重算；等待比例和采样覆盖是各核比例的中位数，不是所有核窗口合并后的比例。

| launch / 核型 | 参与核数 | 总窗 median / p95 / max | 最长偏离 | 已观测等待并集 median | 采样窗口并集 median |
| --- | ---: | --- | ---: | ---: | ---: |
| PREPARE / AIC | 28 | 707744 / 713403 / 713692 | 0.84% | 0.09% | 14.08% |
| PREPARE / AIV | 56 | 707500 / 714996 / 715347 | 1.11% | 0.02% | 27.61% |
| FWD_H / AIC | 16 | 591794 / 595664 / 595664 | 0.65% | 0.25% | 1.57% |
| FWD_H / AIV | 32 | 592286 / 596309 / 596597 | 0.73% | 0.03% | 0.95% |
| FWD_O / AIC | 28 | 354782 / 358707 / 360200 | 1.53% | 7.54% | 9.43% |
| FWD_O / AIV | 56 | 353770 / 357742 / 358149 | 1.24% | 5.80% | 7.02% |

同一核型的诊断总窗中位数呈 Prepare > H > O。这个顺序可用于安排排查，但不能把各核中位数相加当成端到端延迟，或算出三个 launch 的真实耗时占比；正式耗时应使用普通调用的重复 msprof Task Duration(us)。缺少频率证据时不换算微秒。

核总窗相近只说明这次总窗分布较集中。同步等待可能让各核结束时间趋近，不能据此证明计算负载完全均衡。Prepare 与 O 为 28 AIC + 56 AIV，H 为 16 AIC + 32 AIV；这是参与本次 launch 的核数，不是芯片物理核数。H 使用更少核也不能直接认定为利用不足，需要结合状态递推依赖、任务分配和资源约束。

`issue-scope-cores` 等于各行核数，表示所有这些核都有对应计算阶段观测点，不代表计算单元持续忙。

## Prepare：先拆分输入准备和 KKT 后处理

| 阶段 / 核型 | 已采样窗口和，core-cycles | 单次 median / p95 / max | 实际调用次数 / 核 |
| --- | ---: | --- | --- |
| WU_PREPARE / AIV | 4865939 | 3074.5 / 5092 / 7775 | 100～102 |
| KKT_EPILOGUE / AIV | 4288095 | 2451 / 3069 / 7353 | 100～102 |
| PREPARE_NORM_GATE / AIV | 1578437 | 1066 / 1417 / 1903 | 100～102 |
| SOLVE / AIC | 1573992 | 1982 / 2775 / 7937 | 200～204 |
| W_MM / AIC | 897917 | 1021 / 1432 / 2455 | 200～204 |
| PREPARE_SOLVE_ISSUE / AIC | 100828 | 47 / 54 / 4324 | 200～204 |

以上窗口和跨核相加，单位是 core-cycles；阶段之间也可能重叠，不能用该排行计算延迟贡献。Prepare 的实际调用次数在各核之间仅有约 2% 的差异，支持“调用计数较接近”，不证明每次任务的数据量和代价相同。

当前 [Prepare Arch35 实现](../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h) 中几个 scope 的范围直接影响解释：

- `WU_PREPARE` 对应 `Stage6_AivOne`，覆盖 K/V 的 GM→UB 搬运、beta/exp 缩放、UB→L1 搬运、已有事件等待及尾块处理。这条路径的窗口按单个 task 记录。它是较大的输入准备窗口，不是 W/U 矩阵乘的纯计算耗时。其 p95 约为 median 的 1.66 倍，优先核对搬运、缩放和流水就绪。
- `KKT_EPILOGUE` 对应 `Stage3_AivOne`，计时开始在 `WaitCubeKktDone` 之前，还覆盖 gate、求解叶子块准备、向量操作及搬运。max 是 median 的 3 倍；大窗口既可能来自后处理，也可能含上游 Cube 结果等待。应先区分就绪等待与后处理，不能仅据标签归因给 Vector 运算。Prepare 的显式等待比例低，也不排除这类数学 scope 内部的等待。
- `SOLVE` 对应 `Stage5_Compute` 的部分求解运算及内部等待；完整求解还包括 `PREPARE_SOLVE_PREFILL`、`PREPARE_SOLVE_ISSUE`、`PREPARE_SOLVE_DUMP` 等阶段。不能用一个 SOLVE 行代表整个求解过程。
- `PREPARE_SOLVE_ISSUE` 的 median=47、p95=54，但 max=4324，约为 median 的 92 倍。主体样本很短而最大窗口突出，更适合定位极端调用，而不是认定整阶段持续慢。scope 包含已有 `M_MTE1` 等待；需查看对应核、调用序号及前后窗口，才能判断流水资源就绪等候选原因。
- `W_MM` 主要包围矩阵乘下发，前面的 `M_MTE1` 等待及后面的 `M_FIX` 等待、Fixpipe 写回在窗口外。该行不能代表完整 W 生成延迟。brief 只列前 12 项，`U_MM` 未上榜也不说明没有执行或比 W 更快。

首轮内部查看时，先核对 `WU_PREPARE` 和 `KKT_EPILOGUE` 的长调用；对 `PREPARE_SOLVE_ISSUE` 的极端 max 单独找时间位置，避免把少数长窗口当成稳定的每次开销。

## O：等待交接是可见线索

O 的已观测等待并集中位比例为 AIC **7.54%**、AIV **5.80%**，高于本次 Prepare 和 H 的对应比例。它仅统计已打点、已保存的等待窗口，不能解释为完整运行等待占比或硬件空闲率。

| O_WAIT 核型 | 单次 median / p95 / max，cycles | p95 / median | max / median | 实际调用次数 / 核 |
| --- | --- | ---: | ---: | --- |
| AIC | 857.5 / 1700 / 2639 | 1.98 | 3.08 | 250～255 |
| AIV | 366.5 / 1897 / 4039 | 5.18 | 11.02 | 400～408 |

AIV 等待的长尾更明显，但 AIC/AIV 单次窗口属于不同等待点，不能当成同一工作的直接对照。当前 [O Cube](../../chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h) 和 [O Vector](../../chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_vector.h) 把多个交接等待合并为 `O_WAIT`：包括 Cube/Vector 间 mask、Attn×V、输出及阶段完成交接。一个任务可以触发多次 O_WAIT，不能把等待调用数当成任务数，也不能只凭 iteration 判断等待的是哪个 flag。

在内部先查看以下核，再沿同一核的时间戳检查相邻 `O_QK`、`O_QH`、`O_MASK`、`O_ATTN_V`、`O_OUTPUT` 等窗口：

| 核型 | 最大已观测等待比例的 core_id | 最长总窗的 core_id |
| --- | --- | --- |
| AIC | 81、36、72 | 27、42、45 |
| AIV | 13、41、58 | 29、11、10 |

这两组排行不同，不能自动把等待比例最大的核当作整个 launch 最慢的核。brief 只给出候选 ID，实际比例、时间位置和职责需在 `core_summary.csv`、`longest_windows.csv` 与原 parsed.csv 中核对。`core_id` 是 timer 编码的逻辑编号；不同 launch 中相同编号不保证对应同一工作。

若要对齐 AIC/AIV 的时间位置，应先核实共同时钟；仅凭各核独立的相对 offset 不能建立跨核因果关系。

## H 与截断：当前证据较薄

H 的两项上榜 Vector scope 为 `H_VNEW`（median/p95/max=89/92/1104）和 `H_STATE_UPDATE`（73/90/730），每核实际调用数均为 177。`ceil(11274/64)=177` 与这一计数相符，但未提供 varlen 的实际边界，不能据此恢复所有 head/chunk/task 的映射。

这类短 issue scope 只观察指令下发附近的窗口，存在测量开销且不保证包含异步执行完成；不能据此认为真正的 Vnew 或 state update 计算开销只有几十 cycles。H 的窗口并集覆盖仅约 1%～1.6%，其余区域可能包含初始化、未打点操作、异步运算及第 32 次之后的调用，不能标成“等待气泡”。

以下是保留调用数占实际调用数的比例，与前表的时间窗口覆盖率含义不同：

| 阶段集合 | 实际调用次数 / 核 | 前 32 次的调用采样比例 |
| --- | --- | --- |
| Prepare 上榜 AIV 阶段 | 100～102 | 31.37%～32.00% |
| Prepare 上榜 AIC 阶段 | 200～204 | 15.69%～16.00% |
| H_VNEW / H_STATE_UPDATE | 177 | 18.08% |
| O_WAIT / AIC | 250～255 | 12.55%～12.80% |
| O_WAIT / AIV | 400～408 | 7.84%～8.00% |

984 表示被截断的 kernel/core/stage 计数项数，不是 984 个硬件核。前缀样本不能覆盖循环中段和收尾特征，不按 `actual_calls / 32` 放大已采样 duration 来估计全量。

## 第二轮：四个文件补充分析

### 数据来源与一致性

用户提供的四个文件位于仓库的 [atk_output](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/)（原名 atk_output_timer_case0）。本文只做文件间交叉检查，不把它当作重新完成 raw decode 或正式精度验收：

- core_summary 共 216 行：Prepare 84 核、H 48 核、O 84 核；逻辑 core_id、group_id、sub_id 对应关系正确，等待和覆盖比例与各自 cycles 相除结果一致。
- stage_summary 共 27 个 launch/stage/核型分组，合计 31488 个动态样本；与 core_summary 的 sample_events 按 launch/核型汇总一致，截断计数两侧均为 984。
- 27 组动态调用的样本数和 min/median/max 均与 profiling.log 一致，六组 KERNEL_TIMING 的核数和 min/median/max 也一致。日志保留 `.5` 中位数；首轮 brief 因格式化显示为整数时属于舍入差异。
- longest_windows 的 12 条记录按 duration 降序排列，调用序号均在 0～31 内，offset+duration 均不超过对应核总窗。

输入文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| [core_summary.csv](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/core_summary.csv) | `2f7f47f02256e46b712bc24e90b05d8d2b01c415278e4798d9ec109e046b2573` |
| [longest_windows.csv](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/longest_windows.csv) | `2bf21155e40e174d784a6e3e8d487f6c7a4015a1015d37c71a7566924b5ec715` |
| [stage_summary.csv](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/stage_summary.csv) | `9540796367236b436fcc317c16f29f1af835e986bff40d840553b82a8fc23d31` |
| [profiling.log](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/profiling.log) | `e45ebadfe2959e76df4633addc7fd48411e6c3e856cd369f0d002ab67c549049` |

### 1. 初始化窗口很小，H 的主体执行仍需关注

profiling.log 补充了固定窗口。以下都是 cycles；最后一列是两个分组中位数之比，不是逐核比例的中位数。

| launch / 核型 | KERNEL_TIMING median | MAIN_INIT median / max | MAIN_BODY median | INIT median / 总窗 median |
| --- | ---: | --- | ---: | ---: |
| PREPARE / AIC | 707743.5 | 1696.5 / 2718 | 705410 | 0.240% |
| PREPARE / AIV | 707500 | 1984 / 4036 | 705556.5 | 0.280% |
| FWD_H / AIC | 591794 | 621.5 / 1122 | 591330.5 | 0.105% |
| FWD_H / AIV | 592285.5 | 45 / 933 | 591773.5 | 0.008% |
| FWD_O / AIC | 354782 | 1364 / 2323 | 353245 | 0.384% |
| FWD_O / AIV | 353770 | 1215.5 / 2703 | 352168.5 | 0.344% |

用每组 INIT max / KERNEL_TIMING min 给出保守界，各核已测初始化窗口均小于各自总窗的 0.8%。压缩这段已测窗口的直接空间较小，首轮不优先优化初始化。INIT 是 issue window，异步初始化指令的后续影响仍可能落入主体，不能把这当成全部初始化硬件成本的上界。

H 的 MAIN_BODY 中位数与总窗接近，说明大窗口主要落在主体执行。其动态 scope 的覆盖依旧不足；新增完整阶段表也没有把它解释为硬件空闲。尤其 H_WH/H_KV 每核实际 354 次，但仅保留 32 次，调用采样比例为 9.04%；AIC H_WAIT 为 32/1416=2.26%，AIV H_WAIT 为 32/531=6.03%。这些比例不同，不能由 H 的低已观测等待比例判断整个递推链没有等待。

### 2. 最长窗口集中在启动早段，不能据 max 重写求解

12 条最长样本全部来自 Prepare：

| 阶段 | 条数 | 调用序号 | start offset 范围，cycles | duration 范围，cycles |
| --- | ---: | --- | --- | --- |
| SOLVE / AIC | 9 | 全部为 0 | 4351～6858 | 7189～7937 |
| WU_PREPARE / AIV | 2 | 全部为 4 | 43694～44386 | 7452～7775 |
| KKT_EPILOGUE / AIV | 1 | 0 | 5537 | 7353 |

SOLVE[0] 的九条窗口均在对应核总窗前 2% 内结束；全部 12 条中最晚结束的窗口也只到对应核总窗的 7.45%。这支持“最大的已采样窗口集中在启动早段”，不支持“稳态每次求解都要约 8000 cycles”。warmup=5 已执行，首个调用仍然可能体现每次 launch 自身的流水启动；不能直接称为未预热或冷缓存问题。

当前 Prepare 完整 G=2 pack 每个 AIV 按 task 0/2 或 1/3 执行两个 Stage6 调用。按当前 scheduler 推导，WU_PREPARE 调用序号 4 位于该核第三个 pack，仍是早段；这是当前源码映射的推断，不是 CSV 中直接记录的 task 标识。

longest_windows 是所有阶段合并后的全局前 12 条，因此它没有 O_WAIT、H 或 PREPARE_SOLVE_ISSUE 的定位记录。缺席不表示这些阶段没有长窗口；若要判断 SOLVE 的稳定开销，需在内部比较调用 0 与调用 1～31 的分布。

### 3. W/U 下发窗口差异很大，先核对资源等待

| Prepare / AIC 阶段 | 单次 median / p95 / max | 每核调用 median 的 min～max | 实际调用次数 / 核 |
| --- | --- | --- | --- |
| W_MM | 1021 / 1432 / 2455 | 942.5～1039 | 200～204 |
| U_MM | 29 / 177 / 1174 | 29～32 | 200～204 |
| SOLVE | 1982 / 2775 / 7937 | 1493.5～2045.5 | 200～204 |

本 case 的 W/U 两次矩阵乘均为 BF16 输入、64×64 与 64×128 的乘法，调用计数一致，但 W_MM median 约为 U_MM 的 35.21 倍。[WuMatmulToL0C](../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare_matmul.h) 内部包含 LoadData、`MTE1_M` 和 `FIX_M` 等待；外层 Stage7 先执行 W、再执行 U，使用不同 bank，并在两者之间执行已有完成等待及 Fixpipe。因此算术形状相同不能推导窗口代价相同，更不能推导 U 的纯计算快 35 倍。

需要区分操作数搬运就绪、L0C 复用和两次下发所处流水位置。当前数据不能定位到底是哪条等待导致差异；不把它直接归因为 Cube 算力，也不根据这两项 scope 的比值决定改求解公式或 W/U 算法。

Prepare 的 PREPARE_NORM_GATE 每核调用 median 范围为 422.5～1324.5。Stage1 包含 q/k source head 的 owner/sibling 分工和已有交接，scope 名称也不证明核内 L2Norm 开关开启。当前每核聚合没有分离具体任务职责，不能把范围差异直接称为严重分核不均或归一化瓶颈。

### 4. O 的等待普遍存在，未见明显 AIV0/AIV1 偏斜

| O 核型 | 已观测等待比例 min / median / max | 总窗 min / median / max，cycles |
| --- | --- | --- |
| AIC，28 核 | 6.33% / 7.54% / 8.53% | 342970 / 354782 / 360200 |
| AIV，56 核 | 4.97% / 5.80% / 6.77% | 340722 / 353770 / 358149 |

这部分已采样等待存在于所有参与核，未表现为仅少数异常核承担大部分等待。AIV0/AIV1 的等待比例中位数分别为 5.817% / 5.798%，总窗中位数分别为 353932.5 / 353646.5 cycles，本次未见明显的两类 AIV 系统性偏斜。

例如 O/AIC core 36 的总窗最短（342970），等待比例却较高（8.49%）；总窗最长的 core 27 为 360200，等待比例为 7.13%。这进一步说明，按等待比例排行调分核不能保证缩短关键路径。原 O_WAIT 还混合 owner/non-owner 以及不同阶段交接，单次调用 median 的核间差异也可能受到等待职责和样本组成影响。

后续应定位 producer 的完成与发布路径、消费者的就绪和 slot 复用，优先减少生产方暴露出来的延迟或改善合法预取。等待本身保护数据所有权，不把删除 wait/flag 当成优化候选。

### 5. 候选顺序、风险与最小实验

下面是第二轮数据支持的**排查和实验顺序**，不表示已经证明收益排序。第三轮已补充普通调用的三个 launch 耗时及整体时间预算，见下文；具体 Stage6 候选收益仍不能由 scope 外推，源码候选也未在设备实施。

| 优先级 | 候选 / 唯一主要变量 | 证据与预期作用 | 主要风险 / 采用条件 |
| --- | --- | --- | --- |
| 1 | Prepare Stage6：只调整 K/V 输入的搬运下发与处理重叠 | WU_PREPARE median=3074.5、p95=5092；当前代码先完成 V 的搬运/缩放/上传，再发起 K 的搬运。可评估同一 task 内提前下发 K，让其与 V 处理重叠 | 相对局部，不改指数、乘法、cast 顺序。仍须证明 UB 别名、tail 清零、MTE2/MTE3 slot 复用与 Stage6 ready 闭环；已有事件和完成条件不可直接省略。普通 Prepare 及整个调用耗时都需实测 |
| 2 | Prepare Stage3：先分离 KKT 就绪等待，再选择一个后处理搬运环节 | KKT_EPILOGUE median=2451，但窗口从 WaitCubeKktDone 前开始。先确认主要时间来自 producer 等待还是 leaf/gate/布局搬运，才能选择有效改动 | 涉及 leaf 数据、UB/L1 复用和下一 pack 的 NegL 别名；改动范围与风险高于 Stage6。未拆分来源前保持待选，不直接改求解公式 |
| 并行补证；暂不排收益 | H 的主体路径：先看普通耗时及 WH/KV、right/state 搬运的覆盖 | MAIN_BODY 约 591k cycles，已测初始化短，而动态覆盖只有约 1%～1.6%。不能因短 issue scope 忽略 H | 状态跨 chunk 递推及 FP32/BF16 更新顺序使改动风险较高。若普通 H 是主要耗时，应把这条路径提升为首要排查项；现有数据不足以指定具体改动 |
| 3 | O：选一个 producer/consumer 交接，验证就绪前可合法重叠的搬运 | 已采样等待普遍存在，AIV 两类分布接近；应追踪共同流水交接，而不是仅处理个别核 | 需要按等待点区分 Stage3/Stage5 和 owner/non-owner，并证明 ready/free 次数和回绕；不删除同步。O 的普通耗时较小时，端到端收益预算也较小 |
| 暂不优先 | MAIN_INIT、按最慢核直接改分核、仅因 SOLVE max 改求解公式 | 初始化已测窗口小，核总窗集中，最大的 SOLVE 样本均为首个调用 | 尚无足够证据支持作为首轮优化；不能把更多核或更短某个 scope 自动当成真实提速 |

Stage6 候选只是一项可验证假设。提前下发的 K 搬运可能因同一 MTE2 队列、带宽或资源占用而无法有效隐藏，还可能挤压已有 Stage4/5/7 重叠；必须以设备 A/B 选择，而不是按 source 顺序认定一定加速。若普通调用证明 H 或 O 占主要耗时，以上实验顺序应相应调整。

H 的 16 个 AIC 与当前 [tiling](../../chunk_fwd_h/op_host/chunk_fwd_h_tiling_processor.h) 的均匀 head 分配相符：`headsPerCore=ceil(N*HV/C)`，`blockDim=ceil(N*HV/headsPerCore)`。以 N=1、HV=32、可用 C=28 为条件代入，得到每核两 head、16 核。这里 N/C 代入仍需采集版本和实际 tiling 核实；直接启动 28 核不会创造更多相互独立的 chunk 链，也可能引入一 head/两 head 不均及共享 K 的重复读取，因此不能仅凭“16 少于 28”选择增核。

正式进入优化实现时按仓库阶段 03→04→05 建立设计和实验记录。首轮仅改一个主要变量；先做同输入输出对照、CPU 标杆精度以及受影响路径的 tail/varlen/状态回归，再比较普通调用的重复 msprof 数据。候选需要使整个算子调用耗时的收益超过重复测量波动，并且没有其它支持场景回退，才可采用；timer 窗口变短只用来验证假设。

### 后续局部定位所需的小数据

普通性能摘要已经在第三轮补齐。需要进一步确定具体阻塞点时，可从内部 parsed.csv 导出 SOLVE/WU_PREPARE/KKT_EPILOGUE 按调用序号 0～31、核型分组的样本数和 min/median/p95/max，区分首轮与后续前缀；从 O_WAIT 提取每阶段最长几条及同核前后阶段。原始绝对时间戳可留在内部，提供相对本核 offset 即可。O 等待点的精确对应仍需内部时间顺序与源码，而非单独的 iteration 编号。这些数据不是进入 Stage6 首轮实验的前置条件。

现有文件已足以提出上述候选，不需要继续传出大 raw/trace。循环中后段未被当前前 32 次采样覆盖；若后续收益判断依赖这些区域，需另行设计采样并做普通/诊断对照，不能盲目增大容量或外推。通用指标定义见 [timer_analysis.md](timer_analysis.md)，采集约束见 [timer_run.md](timer_run.md)，打点位置见 [timer_flow.md](timer_flow.md)。

## 第三轮：普通调用 msprof 性能基线

### 来源、记录数量与版本

来源：[run_metadata.json](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/run_metadata.json)，SHA256=`1fc92659f6a7b3ba782606646518c280e9836fc767ed2a1aff1cd4df3be20d48`。metadata 的 status=complete、SoC=ascend950、device=0、case_ids=[0]；报告统计记录该性能 case 执行成功。它不替代完整精度验收。

source_commit 为 `29422b8fb49094110208479f7692bd5a2651ea2c`。与分析时 HEAD `1bc6b6d9ef22ac4e0f6a9617cfe77329b6eec3c5` 比较，变化只涉及 timer 文档、离线汇总工具及其测试，相关 kernel/host/调用 runtime 源码未变。metadata 中 OPP vendor 路径已提供，但路径本身不能证明已加载二进制与 source_commit 完全匹配；正式 A/B 仍保留原环境/安装产物校验记录。

另有一项复跑时要核对：采集 case_json_sha256 为 `33530343cec97fc0d7ea7257229221c1f429d547d6e86699ccb23696f39858c0`，与本地当前性能用例文件及采集 source_commit 中该文件的字节哈希均不同。换行归一化后仍不相同，尚不能确认仅是格式差异。A/B 应沿用内部本次保存的 performance_cases.json 和冻结输入，或先核对其语义差异；不用本地同名文件自动替换此次用例。metadata 记录的 case 0 输入快照 SHA256 为 `1f4a29b6a40feb83a424979370638c0b1fb6ee51fa80dd0e513e5d22de73ef38`，本机未读取该快照。

performance_evidence 含来自同一个 DUT op_summary 的 140 条记录：Prepare、H、O 各 20 条，Cast 20 条，Transpose 60 条。全部 duration 为有限正数，行序呈 20 次重复的七项模式：

```text
Cast → 前置 Transpose 1 → 前置 Transpose 2 → Prepare → H → O → 末置 Transpose
```

本次三个目标 kernel 均带 aclnnChunkGatedDeltaRuleFwd 前缀，因此名称筛选实际包含 H/O，无需补缺失行。七项均属于该公开调用记录；Cast/Transpose 纳入完整 kernel 时间预算，但不同位置的三个 Transpose 分开统计，不混成一个分布。

这 20 条/项是**单次 profiler 采集内的重复记录**，不是 20 次独立启动测试。metadata 不含 Task ID、Stream ID 或开始时间；下文时间和按连续七项模式构造 20 个候选调用组，不声称构建了完整跨 launch 时间轴。正式 A/B 保留原报告中的任务身份及原统计配置。metadata 的 warmup=5 是诊断采集配置，普通 ATK 性能测试的预热配置需沿用原入口，不能从该字段推断。

### 单项耗时和时间预算

单位 us；p95 使用 nearest-rank；CV 为本批样本总体标准差 / 均值，不代表跨测试批次稳定性。占比使用各项均值 / 七项时间和均值，避免相加各项分位数。

| 项目 | 样本数 | min / median / p95 / max | mean | 本批 CV | 七项时间和均值占比 |
| --- | ---: | --- | ---: | ---: | ---: |
| Prepare | 20 | 713.217 / 714.836 / 717.501 / 717.848 | 715.090 | 0.157% | 41.86% |
| H | 20 | 573.946 / 589.859 / 597.398 / 599.280 | 588.803 | 1.106% | 34.47% |
| O | 20 | 356.778 / 358.993 / 363.401 / 364.413 | 359.376 | 0.542% | 21.04% |
| Cast | 20 | 4.111 / 4.860 / 6.056 / 6.319 | 4.971 | 11.331% | 0.29% |
| 前置 Transpose 1 | 20 | 15.696 / 15.885 / 16.177 / 16.388 | 15.911 | 1.069% | 0.93% |
| 前置 Transpose 2 | 20 | 14.043 / 14.295 / 14.566 / 15.320 | 14.345 | 1.834% | 0.84% |
| 末置 Transpose | 20 | 9.020 / 9.871 / 10.451 / 10.746 | 9.895 | 4.915% | 0.58% |

| 按模式分组的时间和 | 组数 | min / median / p95 / max，us | mean，us | 本批 CV |
| --- | ---: | --- | ---: | ---: |
| Prepare + H + O | 20 | 1647.409 / 1662.174 / 1675.177 / 1676.718 | 1663.269 | 0.465% |
| 全部七项 kernel | 20 | 1692.236 / 1707.583 / 1721.140 / 1721.705 | 1708.391 | 0.453% |

七项 kernel 时间和的均值约 1.708 ms。该时间和不包含 host、launch 间隙或未纳入 metadata 的任务，不称为 CPU 侧端到端 wall time。Cast 和三个 Transpose 合计约 2.64%，首轮不优先为这部分改接口或融合边界。

Prepare 是最大且本批波动较小的单项，支持继续选择 Stage6 的局部候选；H 占比仍有约 34.47%，不应因内部采样窗口很短而忽略。为了理解收益预算，若 **整个 Prepare** 平均耗时下降 10%，且其余六项不变，则七项时间和均值可减少约 71.509 us、下降约 4.19%。这只是条件计算，不能把 WU_PREPARE 采样窗口下降 10% 等同于整个 Prepare 下降 10%。

### 现在可以进入首轮实验

已有普通基线足够选择首个候选并开展 A/B，无需继续补同类普通耗时摘要。首轮主要变量仍是 Stage6 同一 task 内 K 输入搬运的下发位置，尝试与 V 的处理重叠；实现前核对 UB 别名、tail 清零、事件闭环及已有 Stage4/5/7 重叠。保持现有数学运算、dtype、cast 和输出语义，先按仓库阶段 03 建立设计和验证记录，再按 04/05 实施与回归。本轮只完成分析归档，未修改算子或执行设备实验。

设备验证使用同环境、同输入和原性能入口，普通调用 A/B 的七项 Task Duration 时间和为主要收益口径，Prepare 单项为直接验证，H/O 及转换项用于检查代价转移。各项分位数分别保留，总分位数在每次调用的时间和上计算。timer 仅用于核对候选是否减少预期窗口，不作为采用依据。

建议交替执行基线 A、候选 B、恢复基线 A，并重复若干批次，保持预热/次数/频率及负载条件一致。按本 metadata 提取顺序，前十组/后十组的七项均值为 1711.478 / 1705.305 us，相差约 0.36%；单批内存在差异，不能把一次很小的下降直接视为稳定收益。收益需要跨批复现并超过波动，同时通过精度和受影响支持场景回归。尚无优化后数据，当前不能填写已获收益。

## 原始 brief 留档

下面保留用户提供的摘要，便于后续对照；未补写内部运行数据。

```text
GDN timer summary (cycles; one capture)
CSV=passed decode=complete comparison=ten outputs exactly match uninstrumented call
device=Ascend950PR_957c torch=2.10.0+cpu captures=1 warmup=5
inputs={"q":{"shape":[1,16,11274,128],"dtype":"torch.bfloat16"},"k":{"shape":[1,16,11274,128],"dtype":"torch.bfloat16"},"v":{"shape":[1,32,11274,128],"dtype":"torch.bfloat16"},"g":{"shape":[1,11274,32],"dtype":"torch.float32"},"beta":{"shape":[1,11274,32],"dtype":"torch.bfloat16"}}
options={"chunk_size":64,"layout":"BNSD","scale":0.0883883461356163,"output_final_state":true,"initial_state":{"shape":[1,32,128,128],"dtype":"torch.float32"},"varlen":true}
sample_limit=32 truncated_core_stage_counters=984
Per-core kernel windows: kernel/type cores issue-scope-cores median/p95/max cycles max/median wait-union median% sampled-window median%
FWD_H/AIC 16 16 591794/595664/595664 1.007 0.25% 1.57%
FWD_H/AIV 32 32 592286/596309/596597 1.007 0.03% 0.95%
FWD_O/AIC 28 28 354782/358707/360200 1.015 7.54% 9.43%
FWD_O/AIV 56 56 353770/357742/358149 1.012 5.80% 7.02%
PREPARE/AIC 28 28 707744/713403/713692 1.008 0.09% 14.08%
PREPARE/AIV 56 56 707500/714996/715347 1.011 0.02% 27.61%
Largest sampled scope sums (NOT latency shares): kernel/stage/type sample-core-cycles call-median/p95/max actual-calls/core[min,max] truncated-cores
PREPARE/WU_PREPARE/AIV 4865939 3074.5/5092/7775 [100,102] 56
PREPARE/KKT_EPILOGUE/AIV 4288095 2451/3069/7353 [100,102] 56
PREPARE/PREPARE_NORM_GATE/AIV 1578437 1066/1417/1903 [100,102] 56
PREPARE/SOLVE/AIC 1573992 1982/2775/7937 [200,204] 28
FWD_O/O_WAIT/AIV 1142887 366.5/1897/4039 [400,408] 56
PREPARE/W_MM/AIC 897917 1021/1432/2455 [200,204] 28
FWD_O/O_WAIT/AIC 742663 857.5/1700/2639 [250,255] 28
PREPARE/PREPARE_KKT_GATE/AIV 209699 100.5/198/846 [100,102] 56
PREPARE/PREPARE_SOLVE_ISSUE/AIC 100828 47/54/4324 [200,204] 28
FWD_O/O_GATE/AIV 100643 42/46/1306 [100,102] 56
FWD_H/H_VNEW/AIV 97177 89/92/1104 [177,177] 32
FWD_H/H_STATE_UPDATE/AIV 83794 73/90/730 [177,177] 32
Cores to inspect locally: kernel/type longest-window core IDs; largest observed wait-fraction core IDs
FWD_H/AIC [42, 36, 21] [21, 45, 12]
FWD_H/AIV [38, 44, 23] [14, 16, 17]
FWD_O/AIC [27, 42, 45] [81, 36, 72]
FWD_O/AIV [29, 11, 10] [13, 41, 58]
PREPARE/AIC [42, 45, 39] [48, 75, 63]
PREPARE/AIV [43, 47, 44] [59, 47, 79]
WARNINGS
984 core/stage counters are truncated; each keeps only the first 32 calls. Do not extrapolate full-work duration.
INTERPRETATION
Sample sums across cores are core-cycles, not elapsed launch time or bottleneck contribution.
Most math scopes are issue windows and may include original internal waits; they are not pure compute duration.
Wait unions use a single core's timestamps; they do not measure hardware idle cycles.
Unobserved timeline regions can be uninstrumented work, asynchronous execution or truncation; they are not proven bubbles.
Scope coverage includes coarse Process windows; high coverage does not imply complete fine-stage sampling.
Iteration means per-stage call ordinal, not a shared head/chunk/task id.
Each launch is analyzed separately; this tool does not construct a cross-core or cross-launch timeline.
One capture cannot establish a stable performance regression; use repeated ordinary msprof measurements.
```

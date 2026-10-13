# GDN case 0：内部设备 timer 摘要分析

整理日期：2026-10-13。首轮基于用户提供的 `brief.txt`；第二轮读取用户拷出的 `core_summary.csv`、`longest_windows.csv`、`stage_summary.csv` 和 `profiling.log`，结合当前仓库的打点位置分析。未读取 raw、完整 parsed.csv、trace 或普通调用 msprof 报告。日期表示分析整理时间，构建版本未提供。

## 当前结论

本次实际采集到 **PREPARE → FWD_H → FWD_O** 三个 launch。各 launch、各核型的最长诊断总窗比中位数高 **0.65%～1.53%**，这一轮没有明显的核总窗长尾。下一步优先检查 Prepare 的 `WU_PREPARE`、`KKT_EPILOGUE`，以及 O 的 AIC/AIV 交接等待；这些是排查线索，尚不能确定关键路径瓶颈。

第二轮数据进一步支持：**首轮局部实验可优先考虑 Prepare Stage6 的 K/V 搬运与处理流水**，保持原数值运算及数据就绪约束；这是按现有证据和改动范围选择的候选，尚未证明收益最大。初始化已测窗口很小，最长 SOLVE 窗口集中在每核首个调用，O 等待普遍存在，直接改分核、求解公式或删同步都缺少充分依据。详细证据和候选顺序见下文“第二轮：四个文件补充分析”。

限制主要来自采样：**984 个 kernel/core/stage 计数项截断，每项只保留前 32 次调用**。H 的已采样窗口并集中位覆盖率只有 AIC 1.57%、AIV 0.95%，不能据此认为 H 的计算很轻或其余时间在空闲。单次诊断采集也不能证明性能回归或优化收益。

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

这些状态是用户提供的摘要记录，说明该样本通过了工具中的相应检查，并且诊断与普通调用结果一致；它们不替代独立 CPU 标杆精度验收或全路径设备回归。本文没有完整 ATK/msprof 成功报告、CANN 版本、构建 commit、实际加载的 wheel/OPP 信息、采集时的变长序列边界或经核实的计时频率。

摘要没有 PHASE6 行。当前 [host 路由](../op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp) 的 `UsePreparePath` 条件与本次 BF16、K/V=128、chunk_size=64、Hv/Hq=2 的输入相符，后续按 Prepare/H/O 路径的 scope 解释。源码对应关系以当前仓库为依据，运行构建版本仍需在内部核对。

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

用户提供的四个文件位于仓库的 [atk_output_timer_case0](../../../../../../../tests/atk/chunk_gated_delta_rule_fwd/atk_output/)。本文只做文件间交叉检查，不把它当作重新完成 raw decode 或正式精度验收：

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

下面是当前数据支持的**排查和实验顺序**，不表示已经证明收益排序。尚无普通调用的三个 launch 耗时，收益大小不能量化；源码候选也未在设备实施。

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

### 当前仍需的小数据

1. **普通调用的性能基线最关键。** 提供同一版本、case 0 的实际 Prepare、H、O kernel 名称及 Task Duration(us)；重复测量时保留运行/调用编号、次数和分布，以及构建/加载版本。run_metadata.json 的 performance_evidence 可先提供，但当前脚本按名称筛选的证据不保证含 H/O，缺失时补对应 op_summary 行即可。三 launch 总耗时按同一次普通调用的实际三项相加，再比较重复调用分布；不相加各项 p95，也不用诊断 cycles 代替。
2. **需要确定具体阻塞点时再补局部摘要。** 从内部 parsed.csv 导出 SOLVE/WU_PREPARE/KKT_EPILOGUE 按调用序号 0～31、核型分组的样本数和 min/median/p95/max，区分首轮与后续前缀；从 O_WAIT 提取每阶段最长几条及同核前后阶段。原始绝对时间戳可留在内部，提供相对本核 offset 即可。O 等待点的精确对应仍需内部时间顺序与源码，而非单独的 iteration 编号。

四个文件已足以提出上述候选，不需要继续传出大 raw/trace。循环中后段未被当前前 32 次采样覆盖；若后续收益判断依赖这些区域，需另行设计采样并做普通/诊断对照，不能盲目增大容量或外推。通用指标定义见 [timer_analysis.md](timer_analysis.md)，采集约束见 [timer_run.md](timer_run.md)，打点位置见 [timer_flow.md](timer_flow.md)。

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

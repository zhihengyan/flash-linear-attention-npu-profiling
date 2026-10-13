# 在内部环境分析 GDN timer

raw、完整 CSV、输入张量与 trace 留在原环境。先在内部汇总，再使用 brief.txt 里的计数、分位数与核编号讨论问题。汇总工具只读已解析 CSV 和 JSON 元数据，使用 Python 标准库；复制一个 summarize_timer.py 即可运行，不需要重编 kernel、安装 torch/CANN 或访问外网。

## 生成小摘要

RUN_DIR 指向主采集入口生成的 gdn_timer_<时间戳>_<标识> 目录。在仓库根目录运行：

```bash
UT=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut
python "$UT/summarize_timer.py" "$RUN_DIR/timer/case_0"
python "$UT/summarize_timer.py" "$RUN_DIR/timer/case_1"
```

也可直接传 gdn_npu_time_parsed.csv；timer_layout.json 必须在它旁边，capture_metadata.json 存在时一并读取。工具检查核/launch 完整性、时间戳配对、每阶段计数、重复/缺失记录及总窗口边界。只校验已解析数据，不重新读取 raw 或替代原解析器的 raw/layout 校验。

默认输出 case_N/analysis/，保存现有目录并拒绝覆盖；需要再次分析时用 `--output-dir "$RUN_DIR/analysis_case0_second"` 指定新目录。默认单位 cycles，原元数据的 cycles_per_us 仅作为核实频率的记录保留，不猜频率。

| 文件 | 用途 | 是否需要带出 |
| --- | --- | --- |
| brief.txt | 一般几 KB；状态、输入 shape/dtype、关键选项、核总窗、前 12 个阶段窗口统计、待查核编号及截断说明 | 首轮只需要这份文本 |
| summary.json | 所有阶段汇总、核型分组、极端窗口与校验警告，无输入值及原始时间戳大数组 | 需要更细分析时使用 |
| stage_summary.csv | 按 kernel/stage/AIC-AIV 分组的采样调用统计与实际次数范围 | 留在内部查看 |
| core_summary.csv | 每核总窗口、已观察等待窗口并集、采样窗口并集与截断类型数 | 留在内部筛选异常核 |
| longest_windows.csv | 12 个最长动态调用窗口；含核号、阶段、调用序号及相对本核总窗的起始 offset | 用于内部定位时间段 |

只有 raw 时，先用已有 profile_gdn.py 的 `--parse-only "$RAW_CSV" --output-dir "$PARSED_DIR"`，再对解析后的 CSV 运行 summarize_timer.py。解析失败时先处理错误，不跳过缺核、配对或布局检查。

## 按顺序读结果

### 1. 先确认采集有效性

查看 run_metadata.json 的 status、capture_metadata.json 的 decode_status 和 output_comparison。完整主入口应分别是 complete、complete、`ten outputs exactly match uninstrumented call`。这意味着普通/诊断十项输出精确一致，不等于独立 CPU 标杆正式精度验收。

brief 中出现 unknown 表示对应证据未提供；例如离线 parse-only 没有 capture_metadata.json。不要把 unknown 当通过。精度对照失败、缺 kernel 或负 duration 时，先按 timer_run 及仓库精度路由定位。核实本轮 wheel/OPP、真实 kernel 选择和原 ATK/msprof 成功报告。

### 2. 看 kernel 和核型，再看阶段

实际路由是单独 PHASE6，或 PREPARE/FWD_H/FWD_O 三个 launch。逐 launch 分析；各 launch 的计时原点独立，不用它们拼全链时间轴。AIC/AIV 分开比较。

brief 的 `Per-core kernel windows` 行显示各核总窗 median/p95/max、max/median、已观测等待比例和采样窗口覆盖比例。较长的核窗口是排查线索；总窗包含初始化、已有等待和诊断末尾本核 drain，不等于正式 msprof Task Duration(us)。空闲参与核也可能在原同步上等待，要结合 issue-scope-cores 与实际任务计数判断。该字段仅表示存在计算阶段调用观测点，不证明计算单元持续忙。

max/median 明显偏大时，在 core_summary.csv 筛出同一 kernel、同一核型的长窗口核，再比较它们的 stage_summary、actual_calls 和 longest_windows。实际调用次数不同、尾块大小不同或职责不同都可能造成差异；不能仅凭一张排行断言分核不均。

### 3. 找大窗口及等待的来源

阶段表有每次调用的 median/p95/max，以及各核实际调用次数范围。p95 采用 nearest-rank 定义。Per-core-call-median 的 min/max 可进一步比较核间单次调用窗口，而非只比较各核总量。

| 阶段 | 先检查的内容 |
| --- | --- |
| KKT_MM / KKT_EPILOGUE / PREPARE_KKT_GATE | KKT 矩阵乘、gate/mask 与系数处理；Arch22 fused cumsum 合在 KKT_EPILOGUE 中，prepared cumsum 才有独立 CUMSUM |
| SOLVE / PREPARE_SOLVE_* / SOLVE_WAIT | 求解、预填充/下发/写回以及原交接等待；长等待需追到前序 producer 与就绪标志 |
| WU_PREPARE / W_MM / U_MM / WU_WAIT | 输入预处理、W/U 矩阵乘、ring ready/free；WU_PREPARE 单窗可能包含本核多个 task |
| H_WH / H_VNEW / H_KV / H_STATE_UPDATE / H_WAIT | 状态递推链与 producer/consumer 等待；小尾块可能由 AIV 替代 WH/KV，核型要按实际分支解释 |
| O_QK / O_QH / O_GATE / O_MASK / O_ATTN_V / O_OUTPUT / O_WAIT | 输出路径的矩阵乘、预建 gate、causal mask、融合/写回与等待 |
| FRONT_WAIT / PHASE_WAIT / PHASE6_HO_WAIT | 原 front-batch ACK、全核会合和串行 H/O 交接；可能包含某些核等待其它核的时间 |

`Largest sampled scope sums` 只按“各核已采样调用窗口之和”排序，单位为 core-cycles。不能把它除以某个总窗当耗时占比，也不能作为关键路径贡献；多个 scope 可以嵌套或重叠。多数数学 scope 是 issue window，包含原内部搬运/等待，或仅测指令下发，不能称为纯矩阵计算耗时。阶段的大窗口用于确定下一步在内部看哪个调用段、等待标志或流水指标。

已观察等待比例 = 同一个核上被打点等待窗口的并集长度 / 本核 KERNEL_TIMING。固定 H/O wait 与动态 PHASE_WAIT 重叠时只算一次。它不是硬件空闲率；未采样的等待未计入，异步 pipe 也可能仍在执行。采样窗口覆盖率同样不是 pipe 利用率；粗粒度 Process 窗可覆盖未记录的细分运算。

### 4. 对截断和循环序号保持明确解释

当前只保存每核每阶段前 32 次调用，但记录实际调用总数。brief 的 truncated_core_stage_counters 非零，或 stage_summary 的 truncated_cores 非零，都表示该阶段只是前缀样本。不要用“前 32 次耗时 × actual/32”外推全量，首轮缓存、tail、状态依赖及流水启动/收尾可能不同。

iteration 是每阶段自己的调用序号；H_WH[5] 和 H_KV[5] 不保证是同一个 chunk/head/task。可以在同一核上按时间戳/offset 看先后，跨阶段对应关系需对照真实 scheduler。不同阶段的样本次数也不能直接当相同粒度工作量比较。

## 在内部查看等待与流水

若已有经时钟核实的 trace，使用内部可访问的 Perfetto 或 MindStudio Insight，定位 brief 列出的核和 longest_windows 的阶段/序号。对一个长等待，查看本核等待开始/返回以及上游就绪发布之前的工作。只有确认 AIC/AIV 共同时钟后，才进行跨核时间位置比较。

没有内部 trace viewer 也可直接用 longest_windows.csv 与原 parsed.csv 在同一核按 start_cycle 排序。brief/逐核比例分析不需要外网，也不需要生成 trace。曲线上的空白只能说明未记录窗口，可能是未打点操作、异步工作或采样截断；不能据此宣告等待气泡。是否成为性能瓶颈，仍需结合普通调用的重复 msprof Task Duration(us)、pipe/搬运指标与后续有对照的实验。

## 首轮讨论所需的小数据

逐 case 粘贴 brief.txt 即可，注明 case0/1。它包含设备、torch、shape/dtype、已保存关键选项、实际 launch、状态和截断；不包含输入张量值或完整 raw/trace。若只能手动提供，给出以下四组内容：采集/输出对照状态与截断；各 kernel/AIC-AIV 总窗 median/p95/max；最大的计算与等待阶段调用 median/p95/max 和实际次数；最长窗口核与最大等待比例核编号。

要判定性能回归，再追加同条件普通调用 msprof Task Duration(us) 的重复测量摘要和版本信息。一份诊断样本用于提出候选解释，不能单独建立稳定的性能或负载均衡结论。

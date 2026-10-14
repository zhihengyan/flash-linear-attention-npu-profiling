# GDN timer 工具

主要采集入口位于原 ATK 测试目录：`tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py`，复用两个原 A5 性能 case 的真实输入值与参数。运行步骤见 [timer_run.md](../../docs/timer_run.md)，测量含义与完整清单见 [timer_flow.md](../../docs/timer_flow.md)，源码/设备状态见 [timer_report.md](../../docs/timer_report.md)。

本目录的 profile_gdn.py 是保存输入的重放/离线 raw 解析入口；parse_timer_csv.py 严格验证布局、完整 launch、配对、总窗、实际/采样计数和work ID；trace_parser.py 仅在明确核实频率与共同时钟后导出 Perfetto。默认单位cycles，各类型最多32槽，部分采样在metadata标记。

本分支升级为布局v3：Prepare按每核选定pack窗口，H/O/Phase6保留前32调用；iteration为密集采样槽，新Prepare细分stage有原work ID。主入口追加 --prepare-stage35 --timer-repeats 2，自动粗/细成对采样并运行analyze_prepare_stage35.py，输出跨窗口brief、perturbation、stage及同核任务/pack汇总。旧v2 raw使用gdn-profiling分支解析。设计与当前验证见 [诊断设计](../../docs/timer_prepare_stage35_design.md)、[本轮记录](../../docs/timer_prepare_stage35_report.md)。

test_timer_contract.py 的离线合同测试与 check_timer_source.py 的源码/同步/ABI 检查无需 NPU。它们不能替代 CANN 构建、ATK 精度/确定性/内存、Example/ST 或真实打点开销验证。

数据不能带出内部环境时，运行 `python summarize_timer.py /实际case目录`，在 analysis/ 生成可粘贴的 brief.txt 和留在内部的阶段/逐核表。只读取原 parsed.csv 与元数据，不需要 raw、torch/CANN 或重编。具体分析顺序及等待/重叠/截断解释见 [timer_analysis.md](../../docs/timer_analysis.md)。

# GDN timer 工具

主要采集入口位于原 ATK 测试目录：`tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py`，复用两个原 A5 性能 case 的真实输入值与参数。运行步骤见 [timer_run.md](../../docs/timer_run.md)，测量含义与完整清单见 [timer_flow.md](../../docs/timer_flow.md)，源码/设备状态见 [timer_report.md](../../docs/timer_report.md)。

本目录的 profile_gdn.py 是保存输入的重放/离线 raw 解析入口；parse_timer_csv.py 严格验证布局、完整 launch、配对、总窗和实际循环计数；trace_parser.py 仅在明确核实频率与共同时钟后导出 Perfetto。默认单位 cycles，采样前 32 次/类型/核，超容量会告警并在 metadata 标记。

test_timer_contract.py 的离线合同测试与 check_timer_source.py 的源码/同步/ABI 检查无需 NPU。它们不能替代 CANN 构建、ATK 精度/确定性/内存、Example/ST 或真实打点开销验证。

数据不能带出内部环境时，运行 `python summarize_timer.py /实际case目录`，在 analysis/ 生成可粘贴的 brief.txt 和留在内部的阶段/逐核表。只读取原 parsed.csv 与元数据，不需要 raw、torch/CANN 或重编。具体分析顺序及等待/重叠/截断解释见 [timer_analysis.md](../../docs/timer_analysis.md)。

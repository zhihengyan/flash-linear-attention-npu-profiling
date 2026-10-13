# GDN profiling 接入状态

分支：codex/gdn-profiling，从 main `90176b44321ebdbf3ae2a9057243157d7085c6e1` 创建。接入范围为 ChunkGatedDeltaRuleFwd 的融合 Arch22/Arch35 路径与 A5 Prepare/H/O 路径；叶子算子 ChunkGatedDeltaRuleFwdPrepare、ChunkFwdH、ChunkFwdO 的普通调用也需设备回归。

已接入独立 ACLNN Profile 符号、可选 L0 timer 输入、Python 私有诊断调用、6 个固定阶段和 28 个动态阶段。当前 195 个动态 scope 覆盖关键运算及已有等待。每阶段每核采样前 32 次，实际总数另存并标记截断；四个 launch 的缓冲分开。原同步顺序与公开生产 ABI 保持，timer 布局 v2 是新增诊断协议。

原图/每点源码证据见 [timer_flow.md](timer_flow.md)，操作见 [timer_run.md](timer_run.md)。主采集复用原 ATK case 的实际输入与参数，导出保留原 raw、CSV、metadata、日志；频率及共同时钟经确认后才生成 Perfetto。

## 验证证据

源码检查 `tests/ut/check_timer_source.py`：195 个 scope 与文档逐点对应，28 个动态枚举全部使用；与固定 main 基点逐文件比较业务同步调用顺序、旧 C 声明和 Python 生产签名。完整命令可复现逐点 JSON，不作为设备证明。

离线 decoder 单元测试：布局/对齐、重复记录、三 launch、采样截断、错误长度/版本、缺 kernel、混合路由、缺配对、超总窗、超计数、计数 padding、时钟转换与大整数精度。14 项离线测试已通过（包含原 ATK 成功 case 与真实 kernel profiling 证据检查）；3 项既有 GDN 生产 wrapper 契约定向测试通过。Python 语法与 git diff --check 通过。

全量 test_aclnn_ctypes_abi.py 尝试运行 11 项：9 项通过、1 项设备测试跳过、1 项无关 backward DHU 测试因本机缺少 torch 导入失败；没有将全量测试记为通过。

## 独立审核

按 op-timer-ascendc Step 9，由 fresh-context 独立 agent 对照原始流程图和最终代码完成审核。**修复后独立源码审核通过**：195 个 scope 与清单双向一致，28 个动态类型全部对应；固定/动态配对、timer 参数传递、模式/核型及普通调用兼容性未发现剩余已确认缺陷。详见 [本轮独立审核](timer_audit.md)，其中保留初审发现、修复复查、命令结果和每个 scope 的源码证据。本轮使用新审核结果，不沿用旧分支报告。

## 设备状态与限制

当前 Windows 环境只发现 Python，未发现 Docker、CANN/NPU 编译器、npu-smi 或 C++ 编译器。本轮未进行 CANN 构建、安装、符号真实加载、A2/A3/A5 模板选择、设备核映射/GM 发布、精度/确定性/内存验收、Example/ST、ENABLE_TIMER=0 编译 A/B 或真实 raw/CSV/trace 采集。没有阶段耗时或性能提升结论。

阶段 05 的设备验收尚未完成；源码离线通过不能写成算子正式验收通过。诊断打点、末尾局部 drain 和 GM 发布有开销；需按 timer_run 的原 case 做设备对照。动态 iter 是每阶段的调用序号，不能将不同阶段相同 iter 自动视为同一 task，超过采样容量的后续调用没有时间戳。

## 2026-10-13 内部分析工具

新增 summarize_timer.py：直接从原 parsed.csv、timer_layout.json 和可选 capture_metadata.json 生成 brief.txt、summary.json、逐阶段/逐核表及长窗口表。等待比例采用本核等待窗口并集，不重复相加；实际次数与已采样次数分开记录，不外推截断样本。11 项小型合成数据测试通过，覆盖重叠等待、大整数 offset、截断、缺核/缺样本/重复/越界/不一致计数、未知验证状态及 CLI 产物/保留现有结果。未改 kernel、host 或采集输入；未使用真实设备数据，不能据此给出阶段耗时结论。操作和解释见 [timer_analysis.md](timer_analysis.md)。

# AscendTimerV2 迁移指南

这是实现参考；任务顺序、用户方案与交付按 [SKILL.md](../SKILL.md)。先完成环境/原用例发现和原流程方案，用户选择后再修改代码。历史案例不作为当前流程命令。

## 1. 环境与原用例

确认一个公开入口和原性能测试实现语言、用例文件、生成器/执行器、比较标准。芯片未知先询问，默认建议 A5；已有用户选择不重复问。核对实际设备与构建 SoC/CANN，记录 Docker、源码、Python 和设备路径。
本仓 A2/A3/A5 编包参数分别为 ascend910b/ascend910_93/ascend950；其他工程读取其构建实现。完整 wheel 可以包含未打点算子，不因此扩大任务范围。

原用例可能启动多个 kernel，例：GDN 前向有融合路由和 Prepare/H/O 路由。原图必须追到实际分支，按公开入口而非某个文件名确定任务边界。对每个实际 kernel 使用独立 timer slice；不把未启动 slice 的零记录当失败。

## 2. A5 优先检查：timer 到 GM 的发布

### 2.1 本次经验及适用条件

GDN 的 A5/Ascend950PR Prepare 编译配置含 `-cce-aicore-dcci-before-kernel-end=false`，旧计时器用标量 GlobalTensor 写 timer，却没有显式发布。用户 raw 中 Prepare 全零、H/O 有记录。源码修复增加 timer 定向发布；独立重放另补 OPP 初始化。这个例子支持把检查提前，不能推导所有 A5 都需要同一种修复，或其他芯片不存在风险。

先查实际写入指令与缓存路径、目标编译选项（包括公共 CMake/继承参数）、已有写回机制，以及当前 CANN 中 API 的支持。标量缓存写入且自动发布关闭/缺失时，设计显式发布。PIPE_ALL 或主机 synchronize 本身不能作为此写入路径已有写回机制的证据。
若 buffer 没有非零记录，先核对实际路由/启用打点/加载产物/布局，再调查发布；不能根据全零直接关闭解析校验。

### 2.2 模板发布接口

默认 `ASCEND_TIMER_EXPLICIT_GM_PUBLISH=0` 不引用缓存发布 API。确认目标支持后在 kernel 构建配置设 1；不要单凭产品档位强行开启。模板用以下模式，缓存语义与支持以目标版本文档为准：

```cpp
AscendC::DataCacheCleanAndInvalid<int64_t,
    AscendC::CacheLine::SINGLE_CACHE_LINE,
    AscendC::DcciDst::CACHELINE_OUT>(data[word]);
```

参见 [DataCacheCleanAndInvalid 官方文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.md)。固定 64B 对齐是本模板布局合同，采用其他布局前核实平台的 cache line。
每核事件行和动态计数区按 128B 对齐，基址按 64B 对齐，写者不共享 cache line。host 验证地址、大小、连续性和设备；kernel 对无效地址/核映射拒绝记录，而非写到其他核。
所有计时对象的最后 Tok 写完后，仅一个对象调用 Publish；它发布本核完整事件行及动态计数区域，不刷新整个 DCache，也不增加 SyncAll。模板无共享 metadata；定制 header 必须另设唯一写者和独占行，在需观测起点之外发布。
事件行可能较大，定向发布仍有开销。发布时间不包含在结束时间戳中，实际诊断 launch 承担该开销，不将它解释为零扰动。

## 3. OPP 初始化与本轮产物

所有使用动态 OPP 的入口，包括独立重放，必须在可能初始化 CANN 注册表的 torch_npu 导入/设备调用前设置路径。按当前解释器定位安装包，不盲用源码路径或旧 FLA_NPU_OPP_PATH。不要 import 一个会提前初始化后端的包来查位置；可以先 find_spec 顶层包并核对其 origin/vendor。

```python
from importlib.util import find_spec
from pathlib import Path
import os

spec = find_spec("实际安装包名")
if spec is None or spec.origin is None:
    raise RuntimeError("当前解释器未找到已安装算子包")
vendor = Path(spec.origin).resolve().parent / "实际OPP vendor相对路径"
if not vendor.is_dir():
    raise RuntimeError(f"缺少安装包OPP: {vendor}")
# 路径按工程约定设置，并保留所需系统 OPP。
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
# 之后再 import torch_npu / 转移 tensor / 调用设备。
```

这是按工程填写的片段，不是通用包名/布局。已经初始化的进程不能靠改变量重新注册，使用新进程。
构建必须识别本轮唯一归档，验证归档 OPP 与本轮 staging 内容匹配后提供准确安装命令；提交 hash 字典序不代表时间顺序。保持 Python 环境一致；采集预检还要核对实际加载的 host、所有受影响 kernel 和适配层，不能只凭 host.so 一致判定新 kernel 已安装。
检查器 exit 0 不等于比对发生：缺少 staging、归档或 runtime kernel 等对象的 WARN 必须标记未验证，先补齐可对照的本轮产物证据，不能当成功继续正式采集。
wheel=installed≠staging 指向所选归档与当前构建不一致；wheel=staging≠installed 指向安装覆盖或环境选择。不要通过跳过 kernel MD5 检查继续采集。
工程有追加式构建日志时，显示进度不能作为重编成功证据；隔离本轮输出/记录，避免把旧阶段日志当本轮。

## 4. 布局、枚举与核映射

assets 四个计时/解析文件供复制适配；仅有 doc 运行模板不等于已有真实采集入口。
模板每核存储两段：

```text
[0, N_TIMING_COUNTER) : core × 对齐事件行，每项 [start,end] INT64
[N_TIMING_COUNTER, TOTAL_BUFFER_SIZE) : core × 对齐动态迭代计数
```

每核事件项数为 FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM × MAX_DYNAMIC_ITER，每项两个 word，行对齐到 16 word；动态计数也对齐到 16 word。修改枚举、迭代上限或容量时 kernel/host/parser 同步。
`ASCEND_TIMER_CORE_CAPACITY` 默认 96，仅是模板容量。按实际 launch 和映射配置，Python N_CORE_COUNT 同步，必要时给采集 metadata 记录布局版本/头文件 hash，不用新解析器解释旧二进制 raw。
默认 mixed 映射为 group×3 的 AIC 行和 group×3+subBlock+1 的两 AIV 行。模板拒绝其他 AIV task ratio，纯 Vector/其他混合比例须按真实调度改映射和解析。容量覆盖最高 row id，不能仅拿 blockDim 与总行数比较。

固定/动态枚举名不能重复，阶段按原图定制，保留 KERNEL_TIMING 第 0 槽；解析名称按枚举顺序对应。无需动态项时去除占位动态成员仅留哨兵；模板 state 数组仍有合法大小。

## 5. host 与普通行为

保留生产公开 ABI，沿原框架增加可选诊断能力或独立 Profile ABI，不自动追加公开参数。ctypes/Stable ABI/C++ 入口按项目接口、设备/stream、tensor mutation 规则处理。
普通调用传 null，不偷偷分配 timer。诊断调用创建独立清零 INT64，存活到 stream 同步/导出完成；校验 contiguous、正确设备、总 word 数和对齐。可选诊断 tensor 的校验示例：

```cpp
if (timer.has_value() && timer->defined()) {
    TORCH_CHECK(timer->is_contiguous(), "timer must be contiguous");
    TORCH_CHECK(timer->scalar_type() == at::kLong, "timer must be int64");
    TORCH_CHECK(timer->numel() >= AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE,
                "timer is too small");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(timer->data_ptr()) % AscendTimer::CACHE_LINE_BYTES == 0,
                "timer must be cache-line aligned");
    // 同时确认与业务输入相同的设备，按原诊断 ABI 传入。
} // absent: 原普通 ABI/null timer，不能自动创建计时缓冲。
```

## 6. kernel 打点与关闭

```cpp
TIMER_BLOCK(timer.Init(timerAddr));
TIMER_BLOCK(timerAll.Init(timerAddr));
TIMER_BLOCK(timerAll.TikNoBarrier());
TIMER_BLOCK(timer.TikNoBarrier());
// 原业务，不为时间基准新增 SyncAll。
// 与原流程配对的阶段/动态 Tok 已执行。
TIMER_BLOCK(timerAll.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX));
TIMER_BLOCK(timerAll.Publish()); // 全部对象结束戳已写，发布本核事件+计数。
```

模板保留 barrier Tok 作为显式工具，不在每个裸 matmul/VF 后自动调用。没有安全完成锚点时优先记录 issue window/瞬时点；经方案确认的完成窗口才使用局部 drain。原 WaitFlag/CrossCoreWaitFlag/SyncAll 必须核对作用 pipe/对象，名字不是完成证明。
固定项一次一个在途起点，重叠窗口用独立对象；动态项每 type 一次一个在途 iter，Tik/Tok 同 iter，同类型不可重入。每个早退/idle 分支也要最终收尾。
同槽重复写用 Accumulate：start 为首起点，end 为 start+累计duration，可能超过总墙钟窗。将该事件/动态前缀加入 parser 的 ACCUMULATED_EVENTS，只统计 duration，trace 跳过。
索引/iter 越界不写。计数区保存真实总 iter，事件记录最多 MAX 条；CSV 给出 IterCount、RecordedIterLimit、Truncated，parsed.metadata.json 报告截断，分析不能假装覆盖全部迭代。此计数语义需与新模板配套，旧会截断计数的二进制 raw 无法恢复真实总 iter。停测同时摘 Tik/Tok/计数，业务同步保留。

```cpp
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1
#endif
#if ENABLE_TIMER
#define TIMER_BLOCK(...) do { __VA_ARGS__; } while (0)
#else
#define TIMER_BLOCK(...) do { } while (0)
#endif
```

不能用 ifdef 判断 0/1。关闭时宏移除调用，模板 Init 也拒绝绑定地址，使误用未包宏 API 时仍无计时工作；null timer 同样无时钟/屏障/GM 发布。性能 A/B 使用原输入与测量方式，并注明计时与发布扰动。

## 7. 原用例入口与默认 cycles

新入口与原测试实现语言一致，参数沿工程约定，主入口完成预检、原性能测试、诊断、解析和汇总；默认 one operator，多个 case/内部 kernel 不是多个任务。原系统性能采集可保留，不另造 benchmark。
在原 DUT 初始化/计时循环外保存实际输入值及全部参数、用例 ID、输入 hash，再用同输入独立运行普通对照/诊断；遵守工程比较规则，不普遍强制 bitwise equal。导出同步、文件写入、解析/汇总不进原性能循环。
每轮清零独立 timer；先保存 raw、环境/加载/输出比较状态，再解析，即使解析失败也保留证据。原性能时延与内部诊断窗口分别报告。

Python 工程对已有 raw 的解析示例（替换实际文件路径）：

```python
parsed = parse_timer_csv.parse_and_dump(raw_path, parsed_path,
    expected_core_ids=actual_launched_core_ids)
parse_timer_csv.print_summary(parsed)
if cycles_per_us is not None and shared_clock_verified:
    events = trace_parser.generate_trace_json(parsed, cycles_per_us=cycles_per_us,
        shared_clock=True)
```

CSV 默认保留整数 start/end/duration_cycles，无默认频率。cycles_per_us 由平台核实后显式提供，增加 duration_us；Perfetto 还需要共同时钟确认。trace 接受所有定制名称及瞬时点，跳过 Accumulate；每个 kernel/capture 单独 origin，不合并推算跨 launch 绝对间隔。
parser 的 expected_core_ids 由真实路由/launch 提供，用于拒绝缺/额外核心；容量空行允许为零。多 slice 实现还要外层按路径校验 kernel slice 完整性。默认模板无路由 metadata，不能凭 parse 成功就声称所有 launch 已记录。按阶段约定检查必需事件，稀疏/条件事件不强制每个槽非零。

## 8. 运行文档与集中报告

以 assets/timer_run.md 生成 docs/timer_run.md，与 timer_flow 同级。填写真实 docker exec、仓库、镜像环境激活、CANN、SoC 构建、准确 wheel/run 安装、一主入口命令、结果和故障路径。最后至多一个可选重放/离线入口。生成后检查占位已替换、入口/参数存在、命令语法与路径正确；模板自身保留参数占位供适配。
文档唯一维护当前操作，原测试/算子 README 导航链接。默认 raw 集中 profiling_results/<算子>/<run_id>/ 并忽略，不将海量数据塞进 docs 或提交仓库；已有工程结果约定可保留。
timer_report 汇总当前结论与独立审核/设备状态，记录版本/输入/加载/布局证据及原数据链接。保留历史引用，详细失败日志按轮次存放，不把长排障过程堆进当前报告。

## 9. 迁移核对清单

- [ ] 芯片/设备/CANN/Python/单公开入口/原用例与参数明确。
- [ ] 原图与用户方案确认后实施，覆盖实际多 kernel/早退路径。
- [ ] ABI、null/ENABLE_TIMER=0、容量/映射/对齐/写者均检查。
- [ ] GM 发布按真实配置选择，覆盖事件/计数/metadata，不共享 cache line、不加业务同步。
- [ ] enum/host/parser 同套，expected 核/路径完整性与阶段锚点核对；Accumulate 仅 duration。
- [ ] 主入口复用原用例/真实输入，输出比较与失败 raw 保留，重放独立准备 OPP。
- [ ] 本轮归档/staging/加载产物一致，原性能循环与诊断分开。
- [ ] 三份文档同 docs，完整运行命令和集中结果可找到。
- [ ] 独立审核如实给证据，源码/模拟/CANN/NPU验收状态分开。

模板回归 `python scripts/validate_assets.py` 会执行真实解析/trace 的合成输入检查；若有 C++17 compiler，额外编译执行 AscendC host stub 验证开关/映射/发布范围。host stub 不验证真实 DCache/指令/API兼容，仍需目标 CANN 编译和设备验收。

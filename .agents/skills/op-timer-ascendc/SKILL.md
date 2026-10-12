---
name: op-timer-ascendc
description: "给现有 CANN 算子（AscendC kernel + aclnn host + 现有调用适配层）接入 AscendTimerV2 内核计时：kernel 内 Tik/Tok 打点写入 GM int64 缓冲，host 侧创建/校验 timer 张量，python 导出解析成每核每阶段 CSV 并用 perfetto 可视化，定位 kernel 内部各阶段耗时、核间负载均衡与等待气泡。覆盖 host / kernel / python UT 三处改动。触发关键词：内核计时、kernel 计时、AscendTimer、Tik、Tok、打点、perfetto、核间等待、等待气泡、负载均衡、kernel 内部耗时。"
---

## 角色

AscendC 内核计时工程师。给一个**已有 CANN 算子**接入 AscendTimerV2 内核计时能力，把 kernel 内各阶段耗时、AIC/AIV 核间负载均衡与等待气泡变成可观测、可量化的数据（每核每阶段 CSV + perfetto trace）。

**质量优先级：正确性 > 计时不扰动流水 > 可读性。** 计时是**观测工具**，不得改变算子数值语义、不得破坏原有同步（WaitFlag / CrossCoreWaitFlag / SyncAll / matmul 等）与流水结构；为计时新增的 barrier 会把被测流水本身打乱，必须避免。

**输出语言**：打点说明、迁移报告、Checklist 默认用中文；枚举名、`Tik`/`Tok`、`TIMER_BLOCK`、脚本名、CSV/JSON key 保持原始形式，必要时补中文解释。

## 适用边界（本 skill 做/不做）

**做**：
- 给已有算子接入计时框架并打点（host / kernel / python UT 三侧）。
- 在既有计时框架上新增/停测某阶段的打点、调整 Tok 选型（Overwrite / Accumulate、barrier / NoBarrier）。
- 导出 raw、解析 CSV、生成 perfetto trace，并把结果转成耗时/负载均衡/等待气泡结论。

**不做**：
- 不做算子本身的性能优化动作（那是 `op-perf` 的事）；本 skill 只负责**观测**，把瓶颈数据摆出来。
- 不从 0 开发算子（那是 `op-dev` / `op-code-gen` 的事）。前提是算子已存在。
- 不另造 msprof 等外部采集流程；工程原性能用例自带的系统 profile 可以复用。本 skill 聚焦 kernel 内计时。
- 不保证计时器对性能零影响：`TIMER_BLOCK` 打点本身有少量开销，结论需在 `ENABLE_TIMER` 开关下做 A/B 校验。

## 必读资源

- **迁移指南（本 skill 私有）**：`references/migration-guide.md` — 计时原理（缓冲布局 / Tik/Tok / 动态迭代 / Mode）+ host / kernel / python UT 三侧迁移步骤 + 核对清单
- **实战案例（本 skill 私有）**：`references/worked-example-fused-attention-op.md` — 12 步完整落地记录（算子名与内部符号已匿名化），含各步 ref、【坑】、停测/恢复、验证项
- **模板（可分发）**：`assets/AscendTimerV2.hpp`、`assets/AscendTimerV2_device.hpp`（拷到目标 `op_kernel/`）、`assets/parse_timer_csv.py`、`assets/trace_parser.py`（拷到目标 `tests/ut/`）

## 前置校验（MANDATORY，第一步）

| 条件 | 缺失时的响应 |
|------|------------|
| 目标算子已有 AscendC kernel + host + 调用适配（如 `_torch_adpt.h`、ctypes、Stable ABI 或 C++） | "算子工程不完整，请先完成算子本身开发再接入计时" |
| 目标算子工程有编译/原测试通道 | 缺原用例时确认来源，不自造用例替代；设备不可用时可完成已授权源码接入，标记设备验证待完成 |
| 计时框架文件是否已存在 | 已存在 → 直接复用并跳过迁移步骤；不存在 → 从本 skill `assets/` 拷贝 |

## 工作流

### Step 0: 芯片、单算子与原用例确认（新增）

- 先确认芯片，推荐/默认 A5 (`ascend950`)；已有上下文明确时直接沿用。未知可按注明的默认假设分析，实际编译/运行前核对设备。SoC 映射以工程为准，本仓 A2/A3/A5 为 ascend910b/ascend910_93/ascend950。
- 一次仅处理一个公开算子入口，其内部实际多 kernel 路由仍需覆盖。记录容器、仓库、CANN/Python 环境和设备号。
- 找到工程原性能入口、实现语言、用例文件、输入生成器、执行器与比较规则；新采集入口语言一致，复用真实输入值与全部参数。
- A5 优先检查自动 kernel-end DCache 写回选项与 timer 发布；所有相关平台检查 OPP 初始化和本轮产物/实际加载一致性。细节见迁移指南 §10。

**执行约束**：保留原 Step 1 → Step 9；每步完成后记录结果（success / 坑），迁移到新算子时逐项对照 `references/migration-guide.md` 的「9. 迁移核对清单」做 DoD。

### Step 1: 文件准备与前置确认

从本 skill `assets/` 把四文件拷进目标算子工程：
- `AscendTimerV2.hpp`、`AscendTimerV2_device.hpp` → 目标算子 `op_kernel/`（如 vllm-ascend 的 `csrc/attention/<算子>/op_kernel/`）
- `parse_timer_csv.py`、`trace_parser.py` → `tests/ut/`

确认目标算子已有 host / kernel / python UT 三侧结构；若框架文件已在目标工程（上次已迁移过），跳到 Step 2 起按需增删打点。

### Step 2: host 侧接入

- 按现有接口接入可选诊断 timer（如 `Input("timer")`）；稳定公开 ABI 不强行追加参数，必要时沿工程规则使用独立 Profile 入口。
- PyTorch 适配层（`_torch_adpt.h`）创建/校验 timer 张量：`contiguous`、`dtype=int64`、`numel ≥ AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE`（**用常量校验，不硬编码**）；仅诊断调用创建全 0 缓冲；普通调用无 timer 时保持原行为。代码模板见 `references/migration-guide.md` §6.2。
- kernel 主入口增加 `__gm__ uint8_t *timerAddr` 参数，host 从 timer 张量取地址传入。

### Step 3: 扫描总体流程 → 画原始流程图 → 规划打桩方案 → 修改枚举

**3.1 扫描算子整体流程并画出「原始流程图」（必做，不可跳过）**

在改任何枚举 / 打点之前，先通读目标算子的 **host 侧原本流程**（aclnn host / tiling / kernel 下发的编排）与 kernel 主流程、子模块，建立总体流程认知：

- 从 host 侧入口与 kernel 主入口追调用链：aclnn 接口 → tiling → kernel 下发，再到 `GmmNoQuantAswt` / `GMMProcess::Process` / `matmulService` / `vectorService` 之类，明确 AIC/AIV 编排、各子模块（service）职责与同步点（`SyncAll` / `WaitFlag` / `CrossCoreWaitFlag`）。
- 划出三类阶段：**固定项**（整个 kernel 只跑一次：init、整体 mm、epilogue、同步基准等）；**动态项**（循环内按 `iter` 重复：m-split 外层、K 循环、逐块处理等）；每个阶段对应的**代码位置与所属 service**。
- 记录每个阶段的**完成锚点**（末尾是否有 `WaitFlag` / `CrossCoreWaitFlag` / `SyncAll`），为后面 Step 5/6 的 Tok 选型（`TokNoBarrier` vs barrier `Tok`）提供依据。
- **产出「原始流程图」**：先画出**算子改造前的原始执行流程图**（未加任何计时），以 **host 侧流程为主线**（aclnn 入口 → tiling → kernel 下发）串起 kernel 内的固定项 / 动态项、AIC/AIV 分支、循环嵌套与同步锚点；用 mermaid flowchart 或 ASCII 图（择清晰者）。图中每个节点标注：**阶段名 / 固定或动态 / 所属 service / 完成锚点**，动态项标出所属循环层级（iter 轴）。连同「总体流程说明」落到工程内一份 markdown（如 `docs/timer_flow.md`），便于用户以后查看与复盘（可作计时迁移报告素材）。

**3.2 直接按算子原理中的每个关键步骤设计打点（不选择粒度方案）**

根据 3.1 的真实实现生成一份完整打点设计，直接实施，不提供 A/B/C 粒度选项或要求用户手动选方案。

- 从算子原理逐项映射到代码。例如 GDN 核对 KKT、下三角求逆/求解、W/U；继续展开 H 内部 WH、vNew、KV、状态更新，以及 O 内部 QK、QH、causal mask、Attn×Vnew、输出融合。以实际原理和可达实现为准，包含融合/拆分路由，不能仅用 INIT/BODY/H/O 大窗口代替关键运算。
- 每个关键步骤在原图标出 Tik/Tok、独立阶段名、代码位置、核型、固定/动态、循环轴和完成锚点。步骤在循环内重复就使用动态项，避免不同 head/chunk 覆盖同一槽；说明采样容量与截断。
- 融合或交错的异步步骤没有安全独立完成边界时，仍记录对应 issue window/瞬时点并说明与完成锚点的关系，不为分开测量强行增加同步或排空，也不能称为纯计算耗时。
- 保留总窗口及关键已有等待，用于理解覆盖关系；关键步骤可能重叠，不能简单相加当算子总时间。
- 打点清单：`原理步骤 | 阶段名 | 文件/代码段 | 固定/动态/iter | AIC/AIV/service | Tik/Tok | 锚点/模式 | 解释`，与图一起写入 docs/timer_flow.md。

**3.3 修改枚举变量（必须按算子定制，不可直接使用模板枚举）**

- 在 `AscendTimerV2.hpp` 的 `FixedTiming` / `DynamicTimingType` 中，**用 3.2 关键步骤的阶段名替换模板示例**（模板里的 `FIXED_STAGE_1` / `DYN_STAGE_1` 只是占位示例，直接使用会导致打点与实际流程对不上、导出全是未命名列）；`FIXED_TIMING_COUNT` / `DYNAMIC_TYPE_COUNT_ENUM` 作哨兵自动计数。
- **【坑】固定/动态两组枚举成员名必须两两不重复**，否则编译器报 `redefinition of enumerator`。
- 缓冲布局常量由枚举数量 + `MAX_DYNAMIC_ITER` 自动推导，不需手改。
- 同步改 `tests/ut/parse_timer_csv.py` 的 `FIXED_TIMING_NAMES` / `DYNAMIC_TIMING_NAMES` 命名映射（与枚举索引一一对应，且与 3.2 打点清单同名），常量用 `len(...)` 自动跟随。

**3.4 关键步骤覆盖检查（直接继续实施）**

核对原理步骤、实际调用链、图上打点、枚举与解析映射逐项对应，再进入 Step 4。每个关键步骤都要有观察点和明确含义；未执行的分支说明适用条件，不臆造时间戳。仅目标范围/环境信息缺失或真实外部权限不足时请求澄清，不再设置粒度/方案选择 gate。

### Step 4: kernel 主流程初始化

```cpp
TIMER_BLOCK(timer.Init(timerAddr));        // 普通计时器
TIMER_BLOCK(timerAll.Init(timerAddr));     // 整体 kernel 计时器（KERNEL_TIMING）
<service>.SetTimerPtr(&timer);             // 把 timer 指针交给需要打点的子模块（如 vector service）
// 复用原有同步；不为统一时间基准新增 SyncAll。
TIMER_BLOCK(timerAll.TikNoBarrier());      // 固定项起点一律 NoBarrier
TIMER_BLOCK(timer.TikNoBarrier());
```

### Step 5: 固定项打点

固定项起点统一 `TikNoBarrier()`；Tok 按是否自带完成锚点选型：
- 代码段末尾有完成锚点（`WaitFlag<...>` / `CrossCoreWaitFlag` / `SyncAll`）→ `TokNoBarrier<Overwrite>`（锚点之后异步工作已完成，读到真实完成时刻）。
- 裸下发、末尾无安全完成锚点 → 优先用 NoBarrier 记录 issue window/瞬时点；只有明确需要且已说明扰动的完成窗口才用 barrier `Tok<Overwrite>`。不逐运算强制排空原流水。

末尾整体收尾：`timerAll.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX)`（局部 drain，注明扰动）。确认需显式发布时，全部对象结束戳写完后调用一次 `TIMER_BLOCK(timerAll.Publish())`；仅发布本核 timer 行/动态计数，定制 metadata 另由唯一写者发布。

### Step 6: 动态项打点（循环内）

- 循环开始前：`setDynamicActualIter(type, loopTimes)` 记录实际迭代次数（超过 `MAX_DYNAMIC_ITER` 自动截断）；调用时机放在想要计时的迭代真正开始的地方（tiling 算出 loopTimes 之后、循环体之前）。
- 循环体内：`TikNoBarrier(type, iter)` 起，`Tok<Mode>(type, iter)` 止。
- **Mode 选型**：同一 `(type, iter)` 槽位单轮只写一次 → `Overwrite`；单轮写多次（如 m-split 内层循环每块同一 `stageIdx`）→ **`Accumulate`**（否则 Overwrite 只剩最后一块）。`Accumulate` 的 start/end 是合成值，**只做 duration 分析，不做绝对时间轴**。
- **barrier 选型**同 Step 5（Tok 前有完成锚点 → `TokNoBarrier`；无安全锚点 → issue/瞬时 NoBarrier，必要完成窗口才用 `Tok`）。
- **细粒度打点**：把一段代码拆成相邻子段（如 `CrossCoreWaitFlag` / `matmulService.ComputeStage2` / `CrossCoreSetFlag` 各成一段），每段一个 `(TikNoBarrier, TokNoBarrier<Accumulate>)` 对，定位单条同步指令耗时。
- **时刻点法（跨函数复杂时序，无需匹配 Tik/Tok）**：执行路径复杂、跨函数 Tik/Tok 配对难以保证完全匹配时，在两个关心的时刻各打一个「瞬时对」——`TikNoBarrier` 紧接 `TokNoBarrier<Overwrite>`（相邻两行、中间无业务代码），取该对 start/end 的**平均数**作时刻基准点，两基准点相减即两点间耗时（`(B.start+B.end)/2 - (A.start+A.end)/2`）。两时刻必须用**不同槽位**（各定义一个动态类型）。只记发生时刻、不打乱原流水、不要求配对；本方法未在 worked-example 中使用，属备用观测手段（详见 `migration-guide.md` §7.4）。

### Step 7: 子模块打点 + 停测管理

- 打点在独立 service 类里：类内加 `AscendTimerDevice *timerPtr_` 成员 + `SetTimerPtr(&timer)` 注入；函数签名如 `void Foo(const StageInfo &info)`，槽位索引从 `info` 取（如 `info.stageIdx`）；include `AscendTimerV2_device.hpp`。
- **【坑】`Tok`/`TokNoBarrier` 是模板方法，指针调用必须写 `timerPtr_->template TokNoBarrier<Mode>(...)`**，对象调用写 `timer.template ...`。漏写报 `comparison` / `expected primary-expression` 编译错误。
- **停测某阶段**：注释掉它的 Tik/Tok 对 + 对应的 `setDynamicActualIter`；**等待/同步调用本身（`CrossCoreWaitFlag` / `WaitFlag`）是功能代码必须保留**，只摘打点。停测一律用 `//` 注释而非删除，便于恢复。

### Step 8: python UT 导出 + 可视化 + 验证

- UT 创建 int64 timer 张量（`torch.zeros(_timer_parser.TOTAL_BUFFER_SIZE, dtype=torch.int64, device="npu")`）作可选输入传算子；调用后 `torch.npu.synchronize()`。
- 导出 raw（`np.savetxt`）、`parse_and_dump` 解析每核每阶段 CSV、`print_summary` 终端汇总、`generate_trace_json` 生成 perfetto trace。
- 可视化：浏览器打开 https://ui.perfetto.dev/ 加载 trace JSON（也可用 MindStudio Insight）。各核轨道（`group_X_AIC_X` / `group_X_AIV_X`）色块即各阶段事件，`args` 含 `start_cycle/end_cycle/dur`，直接看 AIC/AIV 每核每阶段耗时与等待气泡。
- 默认输出 cycles，频率已核实才输出 us；Perfetto 还需确认相关核型共同时钟。Accumulate 只分析 duration，不放入绝对时间轴。
- **验证项**：raw 长度 = `TOTAL_BUFFER_SIZE`；无负 duration；实际时间轴事件落在 `KERNEL_TIMING` 时间窗内（Accumulate 合成 end 除外）；IterCount 与预期一致。

**8.1 工程用例与完整操作说明（新增）**

- 在原测试目录生成同语言主要采集入口，自动进行环境/加载检查、原性能用例、同输入普通对照/诊断、raw/CSV/metadata/日志/汇总。沿原比较规则；输入保存、文件导出和解析在原性能计时循环之外。
- 安装后以一个主要入口完成，至多第二个重放/离线入口。OPP 在可能初始化注册表的导入/设备调用前准备，独立重放自行准备；预检不仅看 exit 0，还要确认相关 kernel 比对实际执行，缺证据不能当成功。
- 用 assets/timer_run.md 生成 docs/timer_run.md，与 timer_flow 同级：Docker exec → 镜像 Python/CANN 激活 → 目标 SoC 构建 → 本轮准确包安装 → 主入口 → 结果阅读。填真实路径与选项，不能按提交号字典序选择旧 wheel。
- 当前结论/独立审核/设备状态集中 docs/timer_report.md，原 README 提供导航。原始数据默认 profiling_results/<算子>/<run_id>/ 并忽略；已有工程输出约定可保留并直接链接。源码检查/模拟不能写成设备通过。

### Step 9: 收尾独立校验（对照原始流程图，由独立 agent 执行）

打点与导出全部完成后，**另起一个独立子 agent**（fresh context，不复用打点实现的上下文，避免自证）拿着 3.1 的原始流程图（含 3.2 关键步骤的打桩标注）与最终落地代码做交叉校验，**不得由打点实现者自己校验**。主判据：

- **关键运算覆盖**：逐项核对原理中的关键步骤、实际分支和观察点；不能只记录总窗口，不能漏掉融合/拆分路由。
- **打点位置 vs 流程图一致性**：代码里每个实际 Tik/Tok 的位置，能否在原始流程图上找到对应节点；流程图标注的打桩节点与代码是否**完全匹配**（无漏标、无多标、无错位）——含循环层级（iter 轴）与 AIC/AIV 分支是否对得上。
- **配对完备性**：每个 `Tik`（含 `TikNoBarrier`）是否有对应的 `Tok`/`TokNoBarrier` 收尾；每个被计时 / 被 `setDynamicActualIter` 的动态类型是否 Tik/Tok 成对；时刻点法（瞬时对）是否成对——逐一核对，列出任何**无收尾的 tik** 或**无起点的 tok**。
- **槽位 / 枚举 / 命名 / 清单四方一致**：`AscendTimerV2.hpp` 枚举、`parse_timer_csv.py` 命名映射、3.2 打点清单、流程图标注四者逐项对齐（数量、顺序、名称）。
- **锚点与 Tok 选型**：Tok 是否按各段完成锚点正确选了 `TokNoBarrier<Overwrite>` vs `Tok<Overwrite>`；`Accumulate` 槽位是否确为单轮写多次。

**如实上报**：任何不一致、缺失配对、图上与代码不符，都必须**如实列出**（文件:行 + 现象 + 与流程图哪一处对不上），不得隐瞒、不得美化、不得以「大体一致」带过；即使无问题，也要给出**逐项对照的证据**（执行的校验命令 + 输出，逐节点/逐 tik 的匹配结果）。校验未通过时，修复后**重跑本步**直到通过，才可交付。

## 产出

| 文件 | 位置 | 说明 |
|------|------|------|
| 打点后的 kernel / host / UT 代码 | 目标算子工程 | 三侧改动（若复用已有框架则只有打点改动） |
| `rank_*_npu_time_parsed.csv` | 目标 UT 导出目录 | 每核每阶段 start/end/duration |
| `*_npu_time.json` / `all_rank_trace.json` | 目标 UT 导出目录 | perfetto trace |
| `timer_flow.md` | 目标工程 `docs/` | 原始流程图 + 各关键步骤的打点/清单/锚点 |
| `timer_run.md` | 与 timer_flow 同级 | Docker 到安装/原用例采集的完整步骤 |
| `timer_report.md` | 与 timer_flow 同级 | 当前结论、独立审核、设备状态和数据链接 |
| 独立校验报告（Step 9） | timer_report 中 | 打点 vs 原始流程图逐项对照 + 每个 Tik/Tok 配对核对，问题如实列出（含证据） |
| 计时结论（可选） | 分析输出 | 各阶段耗时表 + AIC/AIV 负载均衡 / 等待气泡结论 |

## DoD 检查点

- [ ] 四模板文件已拷入目标工程（或已确认复用），host / kernel / python UT 三侧改动齐全
- [ ] 固定/动态枚举名无重名；`AscendTimerV2.hpp` 与 `parse_timer_csv.py` 常量一致
- [ ] 3.1 **原始流程图**已输出到文件（如 `docs/timer_flow.md`，只画改造前流程、host 侧为主线）；3.2 覆盖每个关键原理步骤并标注测量含义；不要求用户选择粒度方案
- [ ] host 尺寸校验用常量；kernel 入口有 `timerAddr` 参数；主流程 Init/Tik/Tok 配对，复用原同步不新增 SyncAll
- [ ] 起点一律 `TikNoBarrier`；Tok 按锚点选型；模板调用写全 `.template`
- [ ] 单轮写多次的槽位用 `Accumulate` 且只做 duration 分析
- [ ] 每个实际打点的动态类型有 `setDynamicActualIter`；停测类型两者都注释（等待调用保留）
- [ ] 原用例主入口、同目录三文档及本轮产物检查齐备；raw/CSV 成功，trace 仅已确认时钟时生成；实测状态如实记录
- [ ] Step 9 独立子 agent（非实现者）已对照原始流程图校验：打点位置逐项匹配、每个 Tik 有对应 Tok、枚举/命名/清单/图四方一致；结论**如实上报**（含逐项证据）
- [ ] （可选）`ENABLE_TIMER=0` 对照，确认打点开销可接受

## 失败处置

- **编译错误 `redefinition of enumerator`** → 检查固定/动态枚举重名，改前缀区分。
- **编译错误 `comparison` / `expected primary-expression`** → 检查 `Tok`/`TokNoBarrier` 调用是否漏写 `.template`（尤其指针调用 `timerPtr_->template ...`）。
- **解析出恒 0 列** → 该类型未 `setDynamicActualIter`（或已停测但没注释 setDynamicActualIter）。要么补 setDynamicActualIter，要么接受冗余兜底列。
- **耗时被严重低估 / 只剩最后一块** → 该槽位单轮写多次却用了 `Overwrite`，改 `Accumulate`；若只测到少量迭代，检查 `setDynamicActualIter` 是否在循环真正开始前调用。
- **NoBarrier Tok 测到的时间不合理地小（如异步 matmul 只 0.0x us）** → 这是指令下发时刻而非完成时刻，需配合完成锚点（WaitFlag/CrossCoreWaitFlag/SyncAll）或 barrier Tok 才能反映真实耗时。
- **计时器本身拖慢性能** → 检查是否误用了带 barrier 的 Tik（起点必须 `TikNoBarrier`）；必要时 `ENABLE_TIMER=0` 关计时做对照。

## 归档摘要（给 orchestrator）

```
op-timer-ascendc → 已为 <算子名> 接入 AscendTimerV2（host/kernel/python UT 三侧，N 个固定项 / M 个动态项），
导出每核每阶段 CSV + perfetto trace；关键结论：<阶段> 耗时 X us / <核间> 等待气泡 Y us
```

---
name: op-timer-ascendc
description: "给现有 CANN 算子（AscendC kernel + aclnn host + PyTorch 适配层）接入 AscendTimerV2 内核计时：kernel 内 Tik/Tok 打点写入 GM int64 缓冲，host 侧创建/校验 timer 张量，python 导出解析成每核每阶段 CSV 并用 perfetto 可视化，定位 kernel 内部各阶段耗时、核间负载均衡与等待气泡。覆盖 host / kernel / python UT 三处改动。触发关键词：内核计时、kernel 计时、AscendTimer、Tik、Tok、打点、perfetto、核间等待、等待气泡、负载均衡、kernel 内部耗时。"
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
- 不执行 msprof 等外部 profiler 采集（那是 `op-perf` / `op-ascend-env` 的 msprof 路径）。本 skill 聚焦 kernel 内计时。
- 不保证计时器对性能零影响：`TIMER_BLOCK` 打点本身有少量开销，结论需在 `ENABLE_TIMER` 开关下做 A/B 校验。

## 必读资源

- **迁移指南（本 skill 私有）**：`references/migration-guide.md` — 计时原理（缓冲布局 / Tik/Tok / 动态迭代 / Mode）+ host / kernel / python UT 三侧迁移步骤 + 核对清单
- **实战案例（本 skill 私有）**：`references/worked-example-fused-attention-op.md` — 12 步完整落地记录（算子名与内部符号已匿名化），含各步 ref、【坑】、停测/恢复、验证项
- **模板（可分发）**：`assets/AscendTimerV2.hpp`、`assets/AscendTimerV2_device.hpp`（拷到目标 `op_kernel/`）、`assets/parse_timer_csv.py`、`assets/trace_parser.py`（拷到目标 `tests/ut/`）

## 前置校验（MANDATORY，第一步）

| 条件 | 缺失时的响应 |
|------|------------|
| 目标算子已有 AscendC kernel + aclnn host 适配 + PyTorch 适配层（`_torch_adpt.h`） | "算子工程不完整，请先完成算子本身开发再接入计时" |
| 目标算子工程有编译/UT 通道（能跑一个最小调用） | "先保证算子能编译运行，再接入计时，否则无法验证" |
| 计时框架文件是否已存在 | 已存在 → 直接复用并跳过迁移步骤；不存在 → 从本 skill `assets/` 拷贝 |

## 工作流

**执行约束**：Step 1 → Step 9 按序执行；每步完成后记录结果（success / 坑），迁移到新算子时逐项对照 `references/migration-guide.md` 的「9. 迁移核对清单」做 DoD。

### Step 1: 文件准备与前置确认

从本 skill `assets/` 把四文件拷进目标算子工程：
- `AscendTimerV2.hpp`、`AscendTimerV2_device.hpp` → 目标算子 `op_kernel/`（如 vllm-ascend 的 `csrc/attention/<算子>/op_kernel/`）
- `parse_timer_csv.py`、`trace_parser.py` → `tests/ut/`

确认目标算子已有 host / kernel / python UT 三侧结构；若框架文件已在目标工程（上次已迁移过），跳到 Step 2 起按需增删打点。

### Step 2: host 侧接入

- 算子输入定义增加 `Input("timer")`（可选输入）。
- PyTorch 适配层（`_torch_adpt.h`）创建/校验 timer 张量：`contiguous`、`dtype=int64`、`numel ≥ AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE`（**用常量校验，不硬编码**）；无 timer 输入时自动创建全 0 缓冲。代码模板见 `references/migration-guide.md` §6.2。
- kernel 主入口增加 `__gm__ uint8_t *timerAddr` 参数，host 从 timer 张量取地址传入。

### Step 3: 扫描总体流程 → 画原始流程图 → 规划打桩方案 → 修改枚举

**3.1 扫描算子整体流程并画出「原始流程图」（必做，不可跳过）**

在改任何枚举 / 打点之前，先通读目标算子的 **host 侧原本流程**（aclnn host / tiling / kernel 下发的编排）与 kernel 主流程、子模块，建立总体流程认知：

- 从 host 侧入口与 kernel 主入口追调用链：aclnn 接口 → tiling → kernel 下发，再到 `GmmNoQuantAswt` / `GMMProcess::Process` / `matmulService` / `vectorService` 之类，明确 AIC/AIV 编排、各子模块（service）职责与同步点（`SyncAll` / `WaitFlag` / `CrossCoreWaitFlag`）。
- 划出三类阶段：**固定项**（整个 kernel 只跑一次：init、整体 mm、epilogue、同步基准等）；**动态项**（循环内按 `iter` 重复：m-split 外层、K 循环、逐块处理等）；每个阶段对应的**代码位置与所属 service**。
- 记录每个阶段的**完成锚点**（末尾是否有 `WaitFlag` / `CrossCoreWaitFlag` / `SyncAll`），为后面 Step 5/6 的 Tok 选型（`TokNoBarrier` vs barrier `Tok`）提供依据。
- **产出「原始流程图」**：先画出**算子改造前的原始执行流程图**（未加任何计时），以 **host 侧流程为主线**（aclnn 入口 → tiling → kernel 下发）串起 kernel 内的固定项 / 动态项、AIC/AIV 分支、循环嵌套与同步锚点；用 mermaid flowchart 或 ASCII 图（择清晰者）。图中每个节点标注：**阶段名 / 固定或动态 / 所属 service / 完成锚点**，动态项标出所属循环层级（iter 轴）。连同「总体流程说明」落到工程内一份 markdown（如 `docs/timer_flow.md`），便于用户以后查看与复盘（可作计时迁移报告素材）。

**3.2 基于原始流程图给出若干初步打桩方案（图上标注打桩位置 + 选取理由）**

在**不改任何代码**的前提下，结合 3.1 的原始流程图给出**若干个（≥2）初步打桩方案**，供用户比较选择。方案通常按粒度/覆盖分层，例如：

- 方案 A（粗粒度，少而稳）：只打固定项 + 动态项顶层循环边界，开销最小；
- 方案 B（中粒度）：固定项 + 顶层循环 + 关键子模块段（如 matmul / vector service 各成一段）；
- 方案 C（细粒度）：在 B 基础上进一步拆到同步指令级子段（`CrossCoreWaitFlag` / `matmulService.ComputeStage2` / `CrossCoreSetFlag` 各一段），定位单条同步开销。

对**每个方案**都要：

- **在原始流程图上画出打桩位置**：复制 3.1 的图，在对应节点上用醒目标记（如 `[Tik]`/`[Tok]` 节点、`▲/▼`、着色加粗）标出 Tik/Tok 起止与槽位；打点若挂在该节点控制的控制流之外，须标清其挂靠的节点。
- **简要总结选取理由**：说明为何选这些边界（对应哪条流水 / 哪个完成锚点 / 想回答哪个性能问题），以及粒度取舍（开销 vs 分辨率）、Tok 模式（Overwrite / Accumulate）与 barrier 选型依据。
- 附该方案的打点清单（先不动代码）：`阶段名 | 位置（文件:行 / 代码段）| 固定/动态 | 所属 service | 起点/终点打点 | Tok 模式（Overwrite/Accumulate）与 barrier 选型`。

**打点位置必须贴合算子真实流水**，不要照搬模板里的示例阶段。将各方案（图上标注 + 理由 + 清单）与 3.3 的枚举方案一并提交用户**决策**（见 3.4），用户选定后才动手改代码。

**3.3 修改枚举变量（必须按算子定制，不可直接使用模板枚举）**

- 在 `AscendTimerV2.hpp` 的 `FixedTiming` / `DynamicTimingType` 中，**用 3.2 选定方案的阶段名替换模板示例**（模板里的 `FIXED_STAGE_1` / `DYN_STAGE_1` 只是占位示例，直接使用会导致打点与实际流程对不上、导出全是未命名列）；`FIXED_TIMING_COUNT` / `DYNAMIC_TYPE_COUNT_ENUM` 作哨兵自动计数。
- **【坑】固定/动态两组枚举成员名必须两两不重复**，否则编译器报 `redefinition of enumerator`。
- 缓冲布局常量由枚举数量 + `MAX_DYNAMIC_ITER` 自动推导，不需手改。
- 同步改 `tests/ut/parse_timer_csv.py` 的 `FIXED_TIMING_NAMES` / `DYNAMIC_TIMING_NAMES` 命名映射（与枚举索引一一对应，且与 3.2 打点清单同名），常量用 `len(...)` 自动跟随。

**3.4 方案决策（用户 gate）**

将 3.1 的原始流程图、3.2 的若干打桩方案（含图上标注与选取理由）、3.3 的枚举修改方案（含 `parse_timer_csv.py` 命名映射）一并提交用户**决策**：用户选定一个方案（或要求组合 / 修订）后方可进入 Step 4 起实施。如需调整，回到 3.1/3.2 修订后再确认。

### Step 4: kernel 主流程初始化

```cpp
TIMER_BLOCK(timer.Init(timerAddr));        // 普通计时器
TIMER_BLOCK(timerAll.Init(timerAddr));     // 整体 kernel 计时器（KERNEL_TIMING）
<service>.SetTimerPtr(&timer);             // 把 timer 指针交给需要打点的子模块（如 vector service）
TIMER_BLOCK(if (timerAddr != nullptr) { AscendC::SyncAll<false>(); });  // 统一各核时间基准
TIMER_BLOCK(timerAll.TikNoBarrier());      // 固定项起点一律 NoBarrier
TIMER_BLOCK(timer.TikNoBarrier());
```

### Step 5: 固定项打点

固定项起点统一 `TikNoBarrier()`；Tok 按是否自带完成锚点选型：
- 代码段末尾有完成锚点（`WaitFlag<...>` / `CrossCoreWaitFlag` / `SyncAll`）→ `TokNoBarrier<Overwrite>`（锚点之后异步工作已完成，读到真实完成时刻）。
- 裸下发、末尾无任何等待/同步（如 `matmulService.ComputeStage2(...)` 后不跟 wait）→ 带 barrier 的 `Tok<Overwrite>`（先排空再读时钟兜底）。

末尾整体收尾：`timerAll.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX)`（barrier 排空）。

### Step 6: 动态项打点（循环内）

- 循环开始前：`setDynamicActualIter(type, loopTimes)` 记录实际迭代次数（超过 `MAX_DYNAMIC_ITER` 自动截断）；调用时机放在想要计时的迭代真正开始的地方（tiling 算出 loopTimes 之后、循环体之前）。
- 循环体内：`TikNoBarrier(type, iter)` 起，`Tok<Mode>(type, iter)` 止。
- **Mode 选型**：同一 `(type, iter)` 槽位单轮只写一次 → `Overwrite`；单轮写多次（如 m-split 内层循环每块同一 `stageIdx`）→ **`Accumulate`**（否则 Overwrite 只剩最后一块）。`Accumulate` 的 start/end 是合成值，**只做 duration 分析，不做绝对时间轴**。
- **barrier 选型**同 Step 5（Tok 前有完成锚点 → `TokNoBarrier`；无锚点 → `Tok`）。
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
- **验证项**：raw 长度 = `TOTAL_BUFFER_SIZE`；无负 duration；各阶段事件落在 `KERNEL_TIMING` 时间窗内；IterCount 与预期一致。

### Step 9: 收尾独立校验（对照原始流程图，由独立 agent 执行）

打点与导出全部完成后，**另起一个独立子 agent**（fresh context，不复用打点实现的上下文，避免自证）拿着 3.1 的原始流程图（含 3.2 选定方案的打桩标注）与最终落地代码做交叉校验，**不得由打点实现者自己校验**。主判据：

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
| `timer_flow.md` | 目标工程 `docs/` | 算子**原始流程图**（改造前，host 侧为主线）+ 固定项/动态项清单与完成锚点 + 各打桩方案的**图上标注与选取理由** |
| 独立校验报告（Step 9） | 分析输出 | 打点 vs 原始流程图逐项对照 + 每个 Tik/Tok 配对核对，问题如实列出（含证据） |
| 计时结论（可选） | 分析输出 | 各阶段耗时表 + AIC/AIV 负载均衡 / 等待气泡结论 |

## DoD 检查点

- [ ] 四模板文件已拷入目标工程（或已确认复用），host / kernel / python UT 三侧改动齐全
- [ ] 固定/动态枚举名无重名；`AscendTimerV2.hpp` 与 `parse_timer_csv.py` 常量一致
- [ ] 3.1 **原始流程图**已输出到文件（如 `docs/timer_flow.md`，只画改造前流程、host 侧为主线）；3.2 的若干打桩方案（含图上标注 + 选取理由）经用户**决策**选定；3.3 枚举方案确认后再实施
- [ ] host 尺寸校验用常量；kernel 入口有 `timerAddr` 参数；主流程 Init + SyncAll 基准已建
- [ ] 起点一律 `TikNoBarrier`；Tok 按锚点选型；模板调用写全 `.template`
- [ ] 单轮写多次的槽位用 `Accumulate` 且只做 duration 分析
- [ ] 每个实际打点的动态类型有 `setDynamicActualIter`；停测类型两者都注释（等待调用保留）
- [ ] UT 导出 raw / parsed CSV / trace 成功；验证项全过
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

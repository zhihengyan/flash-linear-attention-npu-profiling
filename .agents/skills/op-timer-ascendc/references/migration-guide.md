# AscendTimerV2 计时功能迁移指南（op-timer-ascendc skill 参考）

> **skill 内模板位置**：本 skill 的 `assets/` 已内置计时框架四个文件，迁移时直接拷贝到目标算子工程对应位置，无需从其它算子工程溯源：
> - `assets/AscendTimerV2.hpp` → 目标 `op_kernel/AscendTimerV2.hpp`
> - `assets/AscendTimerV2_device.hpp` → 目标 `op_kernel/AscendTimerV2_device.hpp`
> - `assets/parse_timer_csv.py` → 目标 `tests/ut/parse_timer_csv.py`
> - `assets/trace_parser.py` → 目标 `tests/ut/trace_parser.py`
>
> 12 步实战案例（算子名与内部符号已匿名化）见 `worked-example-fused-attention-op.md`。

本文档说明两件事：
1. **AscendTimerV2 计时功能的工作原理**（缓冲布局、Tik/Tok、动态迭代、分核映射）；
2. **如何把该计时功能迁移到某个现有 CANN 算子的工程里**，覆盖 host 侧、kernel 侧、python UT 测试侧三处需要改动的代码。

文档以一个长序列注意力类融合算子（下称 `example_fused_attn_op`）为落地示例，但迁移方法对任意 CANN 算子（AscendC kernel + aclnn host 适配 + PyTorch 适配层）通用。请把下文中的 `你的算子名` / `你的阶段` 替换为你自己的名字。

---

# 第一部分：计时原理

## 1. 设计目标与总体思路

AscendTimerV2 用于在 **AI Core kernel 内部** 对指定代码段打点，记录各阶段占用的 system cycle 数，从而分析 kernel 各阶段耗时、核间负载均衡、等待气泡等。它同时支持：

- **固定项（fixed）**：整个 kernel 内只打一次、不区分迭代的阶段，如整体 kernel 时长、某段同步、某段主循环。
- **动态项（dynamic）**：在循环内按 `iter` 重复计时的阶段，并额外记录每类动态项的**实际迭代次数**，便于分析循环展开后的逐段耗时。

实现上是：kernel 内用 `Tik`（start）与 `Tok`（end）读取 `AscendC::GetSystemCycle()` 时间戳，把 `[start_cycle, end_cycle]` 写入一个由 host 传入的 **GM 大缓冲（int64 张量）**；kernel 跑完后 host 把缓冲读回、解析成「每核每阶段」的耗时 CSV，再转成 perfetto 可用的 trace JSON 可视化。

计时功能由三个层组成：

| 层 | 文件 | 作用 | 运行环境 |
|---|---|---|---|
| 公共定义 | `op_kernel/AscendTimerV2.hpp` | 枚举（固定/动态）、缓冲布局常量、`TIMER_BLOCK` 开关宏 | CPU/NPU 共用 |
| NPU 端 | `op_kernel/AscendTimerV2_device.hpp` | `AscendTimerDevice` 类：`Tik`/`Tok`/`setDynamicActualIter` 等 | AI Core kernel |
| Host 接入 | `你的算子_torch_adpt.h` | 创建/校验 timer 张量，作为算子输入传入 | Host（PyTorch 适配层） |

另有 Python 侧解析与可视化（非编译必需，用于分析）：
- `parse_timer_csv.py`：raw 缓冲 → 每核每阶段 CSV + 终端汇总
- `trace_parser.py`：parsed CSV → perfetto JSON
- `test_你的算子_precisions.py`：UT，调用算子并导出 timer


## 2. 缓冲布局（核心）

所有核把计时结果写进**一个 GM 大缓冲**（`int64` 数组），分成两个区域：

```
[0, N_TIMING_COUNTER)            : 基础区。每核 [start_cycle, end_cycle] 成对存储
[N_TIMING_COUNTER, TOTAL_SIZE)   : 动态迭代次数区。每核记录各类动态项实际迭代次数
```

基础区里每个「计时项」占 2 个 `int64` 槽位：`[start_cycle, end_cycle]`。布局常量定义在 `AscendTimerV2.hpp`：

```
N_TIMING_ITEM_PER_CORE       = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER   // 每核计时项数
N_TIMING_COUNTER_PER_CORE    = N_TIMING_ITEM_PER_CORE * 2                                        // 每核原始计数器数（每项占2个int64）
N_TIMING_COUNTER_PER_CORE_ALIGN = 16 对齐                                                        // 128B对齐，避免多核写同一cache line
N_CORE_COUNT                 = 96                                                                // AIC+AIV 总核数（按你的 SoC 调整）
N_TIMING_COUNTER             = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT                    // 基础区总长
DYNAMIC_ITER_PER_CORE_ALIGN  = 16 对齐
DYNAMIC_ITER_TOTAL_SIZE      = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT                        // 动态迭代次数区总长
TOTAL_BUFFER_SIZE            = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE                        // 缓冲总长（int64 个数）
```

**动态项槽位索引**（`AscendTimerV2_device.hpp`）：

```
timing_idx = FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER + iter
per-core GM 槽位 = base + timing_idx*2     // base 按 core 分派（AIC/AIV 不同区域）
```

各常量**由枚举数量和 `MAX_DYNAMIC_ITER` 自动推导**，改一处（如把迭代上限调大）只需同步改 `MAX_DYNAMIC_ITER`，其余自动跟随。

## 3. 枚举定义

> **⚠️ 改枚举之前，必须先做「扫描总体流程 → 规划打点」两步，枚举按算子定制，不可直接使用模板**：
> 1. **先扫描算子整体流程**：通读目标算子 kernel 主入口 → 各子模块（service）的调用链，划出三类阶段 —— **固定项**（整个 kernel 只跑一次：init、整体 mm、epilogue、同步基准等）、**动态项**（循环内按 `iter` 重复：m-split 外层、K 循环、逐块处理等）；记录每个阶段所属 service 与**完成锚点**（末尾是否有 `WaitFlag` / `CrossCoreWaitFlag` / `SyncAll`，为后文 Tok 选型提供依据）。把扫描结果整理成「总体流程说明」输出到工程内一份 markdown（如 `docs/timer_flow.md`），便于用户以后查看与复盘。
> 2. **根据总体流程安排打点位置（产出打点规划）**：列出打点清单 `阶段名 | 位置（文件:行/代码段）| 固定/动态 | 所属 service | 起点/终点打点 | Tok 模式（Overwrite/Accumulate）与 barrier 选型`。打点位置必须贴合算子真实流水。
> 3. **按规划修改枚举**：把规划出的阶段名**替换掉下方模板里的占位示例**。`FIXED_STAGE_1` / `DYN_STAGE_1` 只是占位符，直接使用会导致打点与实际流程对不上、导出全是未命名列。
> 4. **方案确认（用户 gate）**：把打点规划与枚举修改方案（含 `parse_timer_csv.py` 命名映射）一并提交用户确认，确认通过后再进入后续打点实施；如需调整则回到第 1/2 步修订。

### 3.1 固定计时项 `FixedTiming`

单个 kernel 内只打一次、不分迭代的阶段。`FIXED_TIMING_COUNT` 是末尾哨兵（自动计数量）。

```cpp
enum FixedTiming
{
    KERNEL_TIMING_IDX = 0,      // 整体 kernel 时长（保留）
    MAIN_INIT,                  // 用 §3 引导块第 2 步规划出的固定阶段名替换占位示例 ...
    MAIN_GEMM,                  // ... 每个固定阶段一个成员
    // ... 按需添加 ...
    FIXED_TIMING_COUNT          // 哨兵（自动计数量，勿手改）
};
```

> 📐 **加固定项的零迁移条件（布局不变性）**：每核计数 `N_TIMING_COUNTER_PER_CORE` 经 16 对齐后才计入总布局。
> 新增 k 个固定项后，若 `(新 FIXED_TIMING_COUNT + DYN×MAX)×2` 与旧值 16 对齐后相同（例 516→520，同对齐到 528），
> `TOTAL_BUFFER_SIZE` 不变，host/app 侧张量与分配**零迁移**。但**动态项槽位基址** `FIXED_TIMING_COUNT + type×MAX + iter`
> 随之平移——旧二进制的数据不可用新解析器读（反之亦然），二进制与解析器必须**同套**使用。

### 3.2 动态计时项 `DynamicTimingType`

在循环内按 `iter` 重复计时、可统计实际迭代数的阶段。`DYNAMIC_TYPE_COUNT_ENUM` 是末尾哨兵。

```cpp
enum DynamicTimingType
{
    DYN_MSPLIT_LOOP,            // 用 §3 引导块第 2 步规划出的动态阶段名替换占位示例 ...
    DYN_K_LOOP,                 // ... 每个动态阶段一个成员
    // ... 按需添加 ...
    DYNAMIC_TYPE_COUNT_ENUM     // 哨兵（自动计数量，勿手改）
};
```

> ⚠️ **命名冲突（易踩坑）**：固定与动态两个枚举共处 `namespace AscendTimer`。若两组成员名有重复，编译器直接报 `redefinition of enumerator`。**务必保证两组枚举成员名两两不重复**（可用前缀区分，如固定项加 `MAIN_`、动态项用无前缀短名）。

### 3.3 常量与开关宏

```cpp
static constexpr int MAX_DYNAMIC_ITER = 64;   // 每类动态项最大迭代次数

#define ENABLE_TIMER 1                         // 计时总开关
#ifdef ENABLE_TIMER
#define TIMER_BLOCK(code) do { code ; } while (0)
#else
#define TIMER_BLOCK(code) do { } while (0)
#endif
```

用 `TIMER_BLOCK(...)` 包裹打点代码，可在 `ENABLE_TIMER=0` 时整体摘除，避免计时影响流水、便于开关验证。

## 4. NPU 端 API（AscendTimerV2_device.hpp）

`AscendTimerDevice` 提供以下 API：

| API | 说明 |
|---|---|
| `Init(addr)` | 绑定 GM 缓冲地址；设置 `block_id/group_id/subblockid/core_id` 等核信息，初始化各动态项状态 |
| `Tik()` / `Tok()` | 固定项（默认 `KERNEL_TIMING_IDX`）起止，带 `PipeBarrier<PIPE_ALL>` 排空 |
| `TikNoBarrier()` / `TokNoBarrier()` | 固定项起止，**不排空流水**，记录指令下发时刻 |
| `Tik(type, iter)` / `TikNoBarrier(type, iter)` | 动态项起始。**守卫：`iter<0 || iter>=MAX_DYNAMIC_ITER` 直接 return**（越界静默丢弃，不污染其它槽位）。带 barrier 版本先排空再读时钟，NoBarrier 版本不排空 |
| `Tok<Mode>(type, iter)` / `TokNoBarrier<Mode>(type, iter)` | 动态项结束。**守卫：`dynamic_valid[type]==false` 或起始 iter 不匹配则 return**（防错配） |
| `setDynamicActualIter(type, iter)` | 记录某类型动态项的实际迭代次数（超过 `MAX_DYNAMIC_ITER` 自动截断） |
| `zeroDynamicTimingAll()` / `zeroDynamicActualIterAll()` | 清零本核全部动态计时槽位 / 动态迭代计数（防跨 launch 脏累加） |

**Mode 标签**：`Overwrite`（覆盖写）与 `Accumulate`（累加）两种写回方式：

- `Overwrite`：把 `[start, end]` 直接写进槽位。**同一 (type, iter) 槽位被多次写入时，只保留最后一次**。
- `Accumulate`：首次写入 `[start, start+elapsed]`，之后每次只把 `elapsed` 累进 end 字段（start 保持第一次）。**用于「循环内多次写同一 (type, iter) 槽位」的场景**（例如 m-split 内层循环每次都以同一个 `stageIdx` 写同一槽位）——此时用 Overwrite 会被覆盖成只留最后一块，用 Accumulate 得到的是所有子块耗时之和。
- ⚠️ **Accumulate 的 `start` 是第一次打点时刻、`end` 是合成累加值，都不是真实墙钟绝对时刻**：只能用于耗时（duration）分析，不能拿绝对时间戳做 overlap/时间轴分析。

### 关键实现细节

- **Tik 一律用 `TikNoBarrier(type, iter)`**：起点记录指令下发时刻、不排空流水，避免计时器本身拖慢流水。**Tok 按代码段是否自带「完成锚点」选型**：
  - 代码段末尾有完成锚点（`WaitFlag<...>` / `CrossCoreWaitFlag(...)` / `SyncAll()`，锚点之后该段的异步工作确定已完成）→ 用 `TokNoBarrier<Mode>`，锚点之后立即读时钟，拿到真实完成时刻；
  - 代码段只是裸下发、末尾没有任何等待/同步（例如 `matmulService.ComputeStage2(...)` 后不跟 wait）→ 用带 barrier 的 `Tok<Mode>`，先 `PipeBarrier<PIPE_ALL>()` 排空再读时钟兜底。
- `Tok`/`TokNoBarrier` 是模板方法（`Mode` 是模板实参），**调用必须写 `template` 关键字**：`timer.template Tok<Overwrite>(...)`（对象）或 `timerPtr_->template TokNoBarrier<Accumulate>(...)`（指针）。漏写会报 `comparison` / `expected primary-expression` 类编译错误。
- `dynamic_start_cycles[type]`、`dynamic_start_iters[type]`、`dynamic_valid[type]` 三个数组实现「动态项一次只能有一个测量在途」：`Tik` 写 start 并置 valid，`Tok` 校验 valid 与 iter 匹配后写 end 并清 valid。不同 type 之间可重叠，同一 type 不可。
- `calculate_index` 依据 `group_id`/`subblockid` 把本核映射到 GM 基础区的具体区域（AIC 与 AIV 分属不同块），避免多核写同一 cache line。

---

# 第二部分：迁移到现有算子的工程

下面按 host 侧、kernel 侧、python UT 测试侧三处，给出迁移步骤与代码模板。

## 5. 文件准备

把下面几个文件拷贝进你的算子工程（通常放在 `csrc/attention/你的算子/op_kernel/` 下）：

- `AscendTimerV2.hpp`
- `AscendTimerV2_device.hpp`
- Python 侧：`parse_timer_csv.py`、`trace_parser.py`（放在 `tests/ut/`）

> 本 skill 的 `assets/` 已内置这四个文件，直接拷贝即可。

## 6. host 侧接入

### 6.1 算子输入里加一个 timer 张量

在算子的输入定义中增加 `Input("timer")`（以 aclnn 算子定义 `.cpp` 为例）：

```cpp
this->Input("timer")   // 可选输入：AscendTimerV2 缓冲
```

> 若你的算子走 PyTorch 适配层（`_torch_adpt.h`），在 aclnn 调用参数里把 timer 张量作为可选参数传下去。

### 6.2 PyTorch 适配层创建/校验 timer 张量

算子通过一个**可选输入 tensor** 接收 timer 缓冲。若无 timer 输入则创建一个全 0 缓冲，保证 kernel 侧 `t_addr != nullptr` 时能写：

```cpp
at::Tensor timer_tensor;
if (timer.has_value() && timer->defined() && timer->numel() > 0) {
    timer_tensor = *timer;
    TORCH_CHECK(timer_tensor.is_contiguous(), "timer tensor must be contiguous");
    TORCH_CHECK(timer_tensor.scalar_type() == at::kLong, "timer tensor dtype must be int64");
    TORCH_CHECK(timer_tensor.numel() >=
        static_cast<int64_t>(AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE),
        "timer tensor numel < required");
} else {
    timer_tensor = at::zeros(
        {static_cast<int64_t>(AscendTimer::N_TIMING_COUNTER) +
         static_cast<int64_t>(AscendTimer::DYNAMIC_ITER_TOTAL_SIZE)},
        query.options().dtype(at::kLong));   // 用你自己的输入张量 options
}
```

> 尺寸校验用**常量**（`N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE`）而非硬编码字面量，这样改 `MAX_DYNAMIC_ITER` 后 host 侧自动跟随。

把 `timer_tensor` 作为可选参数传入 aclnn 调用（`EXEC_NPU_CMD(aclnn你的算子, ..., timer_tensor, ...)`）。

### 6.3 kernel 入口接收 timer 地址

kernel 的主入口函数增加一个 `__gm__ uint8_t *timerAddr` 参数，由 host 从 timer 张量取数据地址传进来。传入后即可在 kernel 内 `Init`。

## 7. kernel 侧接入

### 7.1 在 kernel 主流程中声明并初始化 timer

在 kernel 主函数里创建计时器对象并绑定缓冲，同时把 timer 指针交给需要打点的子模块（如 vector service）：

```cpp
// kernel 主流程初始化处
TIMER_BLOCK(timer.Init(timerAddr));        // 普通计时器
TIMER_BLOCK(timerAll.Init(timerAddr));     // 整体 kernel 计时器（KERNEL_TIMING）
vectorService.SetTimerPtr(&timer);          // 把 timer 指针交给子模块（如需要）
TIMER_BLOCK(
    if (timerAddr != nullptr) { AscendC::SyncAll<false>(); }   // 确保各核拿到统一时间基准
);
TIMER_BLOCK(timerAll.TikNoBarrier());        // 整体 kernel 起点：固定项起点一律 NoBarrier
TIMER_BLOCK(timer.TikNoBarrier());
```

> 若你的算子把部分逻辑拆到独立的 service 类（如 `<YourVectorService>`），需要在那个类里加一个 `timerPtr_` 成员和 `SetTimerPtr` 方法，并 include `AscendTimerV2_device.hpp`。

### 7.2 固定项打点

固定项只在 kernel 内执行一次。起点统一用 `TikNoBarrier()`；终点按是否自带完成锚点选 `Tok<Overwrite>`（无锚点，barrier 排空兜底）或 `TokNoBarrier<Overwrite>`（末尾有 `SyncAll` / `CrossCoreWaitFlag` 等锚点）：

```cpp
TIMER_BLOCK(timer.TikNoBarrier());                    // 段落起始（NoBarrier）
// ... 你的阶段代码（裸下发、无锚点） ...
TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::MAIN_INIT));
TIMER_BLOCK(timer.TikNoBarrier());                    // 下一段落
// ... 你的阶段代码，末尾有 SyncAll / CrossCoreWaitFlag 等完成锚点 ...
TIMER_BLOCK(timer.template TokNoBarrier<Overwrite>(AscendTimer::MAIN_GEMM));
// 末尾整体收尾（带 barrier 排空）
TIMER_BLOCK(timerAll.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX));
```

> **operator() 之外的 kernel 前相**（如 AIV 预建表 → `SyncAll` → AIC 消费的三段式入口分发）：主流程 timer
> 对象要进 operator() 才能初始化，前相看不见。在 kernel 入口另建**入口级 timer 对象**
> （`AscendTimerDevice entryTimer; TIMER_BLOCK(entryTimer.Init(timerAddr));`），按核型分别打固定窗
> （AIC 打"到达分发点→栅栏释放"、AIV 打"建表入口→返回"；AIC/AIV 的 core_id 槽位天然错开，互不踩写，
> 且与 operator() 内主 timer 用不同 fixed 槽）。入口先 `entryTimer.zero(新固定项)` 清零：不走该路径的
> 变体不写这些槽，同一计时缓冲混跑会话的上一 launch 残值会污染解析。

### 7.3 动态项打点（循环内）

在循环开始前用 tiling 算出的实际迭代次数调用 `setDynamicActualIter`，循环体内用 `TikNoBarrier(type, iter)` / `Tok<Mode>(type, iter)` 包住要计时的代码段。**动态项起点一律 `TikNoBarrier`**；Tok 的 `Mode` 与是否带 barrier 按下面两条规则选：

- **Mode**：同一 `(type, iter)` 槽位在单次迭代里只写一次 → `Overwrite`；单次迭代里会被写多次（例如 m-split 内层循环每块都以同一个 `stageIdx` 写同一槽位）→ `Accumulate`（累加所有子块耗时，否则 Overwrite 只剩最后一块）。
- **barrier 与否**：Tok 前有完成锚点（`WaitFlag<...>` / `CrossCoreWaitFlag`）→ `TokNoBarrier`；裸下发无锚点 → `Tok`（barrier 排空兜底）。

```cpp
// 记录实际迭代数（在 tiling 计算出 loopTimes 后，仅当 >0）
if (loopTimes > 0) {
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::DYN_MSPLIT_LOOP, loopTimes));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::DYN_K_LOOP, loopTimes));
}

// 循环体内打点
TIMER_BLOCK(timer.TikNoBarrier(AscendTimer::DYN_MSPLIT_LOOP, iterIdx));
DoStage1(...);                               // 只统计这一个函数的耗时
TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::DYN_MSPLIT_LOOP, iterIdx));

// 若只想测某个函数本身，把调用前的同步/等待、调用后的置位放到计时范围外：
CrossCoreWaitFlag(...);                      // 不在计时内（且是 DYN_K_LOOP 的完成锚点）
TIMER_BLOCK(timer.TikNoBarrier(AscendTimer::DYN_K_LOOP, iterIdx));
DoStage2(...);
TIMER_BLOCK(timer.template TokNoBarrier<Overwrite>(AscendTimer::DYN_K_LOOP, iterIdx));
CrossCoreSetFlag<...>(...);                  // 不在计时内
```

**细粒度打点**：可以在一条语句前后单独打点，把一段代码拆成多个相邻的子段（如 `matmulService.ComputeStage2(...)` 前的 `CrossCoreWaitFlag`、调用本身、随后的 `CrossCoreSetFlag` 各成一段），每段一个 `(TikNoBarrier, TokNoBarrier<Accumulate>)` 对，从而定位到单条同步指令的耗时。这些子段共用同一个 `stageIdx`、单轮迭代写同一槽位多次，所以**必须用 `Accumulate`**。

**通过 service 子模块打点**：打点点若在独立的 service 类里，用的是 `AscendTimerDevice *timerPtr_` 指针（通过 `SetTimerPtr(&timer)` 注入），函数签名形如 `void Foo(const StageInfo &info)`，槽位索引从 `info` 取（如 `info.stageIdx`），调用写成 `timerPtr_->TikNoBarrier(AscendTimer::DYN_X, info.stageIdx)` / `timerPtr_->template TokNoBarrier<Accumulate>(AscendTimer::DYN_X, info.stageIdx)`（指针调用同样要 `->template`）。

### 7.4 时刻点法（跨函数复杂时序，无需匹配 Tik/Tok）

当算子执行路径较复杂、跨函数的 Tik/Tok 配对难以保证完全匹配（如起点在函数 A、终点在函数 B，或存在分支导致起止配不上）时，可以用**时刻点法**：不追求一对 Tik/Tok 包住整段代码，而是在关心的两个时刻各打一个「瞬时对」，把发生时刻记下来，事后用两个时刻的差做耗时分析。

- 在时刻点 A：`TikNoBarrier` 紧接 `TokNoBarrier<Overwrite>`（两行相邻、中间无业务代码）。`start`/`end` 是同一时刻的两次相邻读时钟，其**平均数**作为该时刻的基准点，可抵消单次读时钟抖动。
- 在时刻点 B：同样打一个瞬时对。
- 耗时 = `(B.start + B.end)/2 - (A.start + A.end)/2`。
- **两个时刻必须用不同槽位**：在枚举里各定义一个动态类型（如 `EVENT_A` / `EVENT_B`），同一 `(type, iter)` 槽位只存一个时刻，否则 `Overwrite` 会被第二次覆盖。
- 优点：**不打乱原算子的流水顺序**（全程 NoBarrier）、**不要求跨函数 Tik/Tok 配对**、只记录发生时刻，具体分析事后做。代价是拿不到单段时长的绝对时间轴，只适合两点间耗时的时序分析。

```cpp
// 时刻点 A：TikNoBarrier 紧接 TokNoBarrier<Overwrite>（无业务代码夹在中间）
TIMER_BLOCK(timer.TikNoBarrier(AscendTimer::EVENT_A, iterIdx));
TIMER_BLOCK(timer.template TokNoBarrier<Overwrite>(AscendTimer::EVENT_A, iterIdx));
// ... 业务代码（可能跨多个函数，无需配对） ...
// 时刻点 B
TIMER_BLOCK(timer.TikNoBarrier(AscendTimer::EVENT_B, iterIdx));
TIMER_BLOCK(timer.template TokNoBarrier<Overwrite>(AscendTimer::EVENT_B, iterIdx));
// 事后：elapsed = (EVENT_B.start + EVENT_B.end)/2 - (EVENT_A.start + EVENT_A.end)/2
```

> 本方法未在 worked-example 案例中使用，属备用观测手段；案例里全部采用成对 Tik/Tok 的匹配测量。

### 7.5 停测某个阶段

直接注释掉它的 Tik/Tok 对，并**同步注释对应的 `setDynamicActualIter`**。以停测一段 WAIT 为例——等待调用本身是功能代码，**必须保留**，只摘掉打点：

```cpp
//        TIMER_BLOCK(timer.TikNoBarrier(AscendTimer::SOME_WAIT, iterIdx));
        SomeWait(...);                        // CrossCoreWaitFlag / WaitFlag：功能必需，保留
//        TIMER_BLOCK(timer.template TokNoBarrier<Accumulate>(AscendTimer::SOME_WAIT, iterIdx));
```

若不注释对应的 `setDynamicActualIter`，该类型迭代计数仍会被写入，但无实际 start/end 数据，解析时会得到恒 0 列（冗余噪音）。停测一律用 `//` 注释而非删除，便于后续恢复。

## 8. python UT 测试侧接入

### 8.1 常量与命名映射（parse_timer_csv.py）

Python 侧常量必须与 `AscendTimerV2.hpp` **完全一致**：

```python
FIXED_TIMING_COUNT = len(FIXED_TIMING_NAMES)          # 与枚举哨兵一致
DYNAMIC_TYPE_COUNT_ENUM = len(DYNAMIC_TIMING_NAMES)   # 与枚举哨兵一致
MAX_DYNAMIC_ITER = 64
N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER
N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2
N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) // 16 * 16
N_CORE_COUNT = 96
N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT
DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT_ENUM + 15) // 16 * 16
DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT
TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE
```

> 这两个计数从命名映射字典 `len(...)` 自动推导，与 `AscendTimerV2.hpp` 的枚举数量保持一致；新增/删除枚举后无需手动改数字。

命名映射字典（`FIXED_TIMING_NAMES` / `DYNAMIC_TIMING_NAMES`）与枚举**按索引一一对应**：

```python
FIXED_TIMING_NAMES = {
    0: "KERNEL_TIMING",
    1: "MAIN_INIT",
    2: "MAIN_GEMM",
    # ...
}
DYNAMIC_TIMING_NAMES = {
    0: "DYN_MSPLIT_LOOP",
    1: "DYN_K_LOOP",
    # ...
}
```

> 兜底列数说明：`get_timing_names` 对「未 `setDynamicActualIter` 的类型」（IterCount=0）默认输出 `MAX_DYNAMIC_ITER` 个零值列，冗余但无害。若想瘦身，把兜底 `else MAX_DYNAMIC_ITER` 改成 `else 0`。

### 8.2 UT 侧导出流程（test_你的算子_precisions.py）

在 UT 里调用算子并携带 timer 张量，跑完后同步、导出 raw / parsed CSV / trace JSON：

```python
timer = torch.zeros(_timer_parser.TOTAL_BUFFER_SIZE, dtype=torch.int64, device="npu")
# 调用你的算子，把 timer 作为可选输入传入
call_your_op(inp, timer=timer)
torch.npu.synchronize()

np.savetxt(raw_path, timer.cpu().numpy().reshape(-1, 1), fmt="%d")   # raw 缓冲
_timer_parser.parse_and_dump(str(raw_path), str(parsed_path))        # parsed CSV（每核每阶段）
_timer_parser.print_summary(str(parsed_path))                        # 终端汇总

base_cycle = _trace_parser.find_min_start(parsed_path)
events = _trace_parser.generate_trace_json(parsed_path, 0, base_cycle)
with open(trace_json_path, "w", encoding="utf-8") as f:
    json.dump(events, f, indent=2, ensure_ascii=False, allow_nan=False)
```

### 8.3 可视化

浏览器打开 **https://ui.perfetto.dev/**，拖入 `trace_json/all_rank_trace.json`（或 `rank_0_npu_time.json`）。左侧轨道为各核（`group_0_AIC_0` / `group_0_AIV_0`...），色块即各阶段事件，`args` 里含 `start_cycle/end_cycle/dur`。也可以使用 MindStudio Insight 查看这个文件。

---

## 9. 迁移核对清单

迁移到新算子时逐项确认：

- [ ] 固定/动态枚举成员名**两两不重复**（避免 redefinition）
- [ ] `AscendTimerV2.hpp` 与 `parse_timer_csv.py` 的常量（`FIXED_TIMING_COUNT` / `DYNAMIC_TYPE_COUNT_ENUM` / `MAX_DYNAMIC_ITER` 等）一致
- [ ] host 侧尺寸校验用常量（非硬编码）
- [ ] 起点一律 `TikNoBarrier`；终点按锚点选 `Tok`（无锚点）还是 `TokNoBarrier`（有锚点）；模板调用写全 `.template`（`timer.template` / `timerPtr_->template`）
- [ ] 同一 `(type, iter)` 槽位单轮只写一次用 `Overwrite`，单轮写多次（如 m-split）用 `Accumulate`；用 `Accumulate` 的点只做 duration 分析，不做绝对时间轴
- [ ] 每个 `Tik(type, iter)` 都有对应 `Tok(type, iter)`，且 iter 一致
- [ ] 每个实际打点的动态类型都调用了 `setDynamicActualIter`；停测的类型则两者都注释（等待调用本身保留）
- [ ] 需要打点的子模块（如 vector service）已调用 `SetTimerPtr`、函数从 `info` 取 `stageIdx`、用 `timerPtr_->` 打点、include 头文件
- [ ] 停测阶段用 `//` 注释而非删除（便于恢复）
- [ ] 验证：raw 长度 = `TOTAL_BUFFER_SIZE`，无负 duration，各阶段事件落在 `KERNEL_TIMING` 时间窗内，IterCount 与预期一致

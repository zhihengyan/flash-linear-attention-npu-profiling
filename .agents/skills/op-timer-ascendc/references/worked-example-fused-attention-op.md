# 实战案例：某融合算子迁移 AscendTimerV2（12 步）

> **历史案例**：本文保留原实战过程，不是当前任务执行流程。当前流程与模板以 [SKILL.md](../SKILL.md) 和 [迁移指南](migration-guide.md) 为准。旧自动分配 timer、默认新增 SyncAll、固定 cycles/us 与 ifdef 开关示例不可直接照搬。

> 本文件是把 AscendTimerV2 迁移进一个**长序列注意力类融合算子**（下称 `example_fused_attn_op`：AIC cube + AIV vector 异构、含多层嵌套循环与跨核同步）的完整实战记录，含每步之间的 `ref` 依赖、实际踩过并已解决的【坑】、打点选型依据与验证项。**算子名、业务域、内部符号均已匿名化**，仅保留可教学的技术骨架。迁移到新算子时可对照本文按清单逐项执行，详见 `migration-guide.md` 的「9. 迁移核对清单」。

## 场景要素

- **硬件**：昇腾 AI Core（AIC+AIV 异构 AI Core），多卡
- **软件**：某 vllm-ascend 定制分支；CANN（AscendC 开发）/ torch_npu
- **业务**：长序列注意力类算子（含稀疏选择），超长序列输入
- **目标**：kernel 内各阶段耗时 / 核间负载均衡 / 等待气泡的逐核定位与 perfetto 可视化

## 背景

`example_fused_attn_op` 是长序列注意力类融合算子：AIC（Cube）侧执行 matmul（两个 cube 阶段，下称 cube1 / cube2），AIV（Vector）侧执行 softmax 与 selection（状态）更新，两侧通过 MTE2/MTE3 搬移、核内 WaitFlag、跨核 CrossCoreSetFlag / CrossCoreWaitFlag mailbox 同步构成深度重叠的异步流水；主循环按 m-split 分块、按 seq 分块多层嵌套。外部 profiler（如 msprof）只能看到 kernel 粒度时间线，看不到 kernel 内部各阶段耗时、核间等待气泡，也无法定位瓶颈是 matmul、vector、搬移还是某条同步指令。

> 说明：算子名与各内部函数、同步 flag、字段均为**占位示例**，仅示意流程结构；真实迁移时请对照目标算子自身的主流程与命名。

---

## 步骤 1：文件准备

把 `AscendTimerV2.hpp`、`AscendTimerV2_device.hpp` 拷贝到算子工程 `csrc/<算子>/op_kernel/` 下；把 `parse_timer_csv.py`、`trace_parser.py` 放到 `tests/ut/`。

结果：success。

## 步骤 2（ref 步骤1）：host 侧接入 timer 张量

① 算子输入定义增加 `Input("timer")`（可选输入）；② PyTorch 适配层（`_torch_adpt.h`）创建/校验 timer 张量：`contiguous`、`dtype=int64`、`numel ≥ N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE`（用常量校验，不硬编码），无 timer 输入时自动创建全 0 缓冲；③ kernel 主入口增加 `__gm__ uint8_t *timerAddr` 参数。

```cpp
at::Tensor timer_tensor;
if (timer.has_value() && timer->defined() && timer->numel() > 0) {
    timer_tensor = *timer;
    TORCH_CHECK(timer_tensor.is_contiguous(), "timer must be contiguous");
    TORCH_CHECK(timer_tensor.scalar_type() == at::kLong, "timer dtype must be int64");
    TORCH_CHECK(timer_tensor.numel() >= (int64_t)(AscendTimer::N_TIMING_COUNTER +
        AscendTimer::DYNAMIC_ITER_TOTAL_SIZE), "timer numel < required");
} else {
    timer_tensor = at::zeros({(int64_t)AscendTimer::N_TIMING_COUNTER +
        (int64_t)AscendTimer::DYNAMIC_ITER_TOTAL_SIZE}, x.options().dtype(at::kLong));
}
```

结果：success。

## 步骤 3（ref 步骤2）：扫描总体流程 → 规划打点 → 枚举与常量定义【坑：枚举重名】

改枚举之前，先按新工作流把算子的总体流程扫一遍、产出打点规划：

**3.1 扫描算子总体流程**：主循环 `MainLoop`（m-split 分块 + seq 分块多层嵌套）；AIC（Cube）侧执行 matmul（cube1 / cube2），AIV（Vector）侧执行 softmax 与 selection 更新（`VectorStage1` / `VectorStage2` / `MergeStage`），两侧经 MTE2/MTE3 搬移、核内 `WaitFlag`、跨核 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 构成深度重叠异步流水。
- **固定项**（整个 kernel 只跑一次）：Init 后的状态更新（`INIT_UPDATE`）、`SyncAll<false>` 同步基准（`SYNC_ALL`）、主循环整体阶段（`MAIN_CUBE1` / `MAIN_VEC1` / `MAIN_CUBE2` / `MAIN_VEC2`）。
- **动态项**（循环内按 `iter` 重复）：主循环每迭代的 cube1 / cube2 / vec1 / vec2 总时长、各自 WAIT 等待段、cube2 内部细粒度段（MMSERVICE / CCFLAG）、`MergeStage` 的 PLAN / COPY / MTE2 / MTE3 / TAIL / SELUPDATE 细粒度段。
- **完成锚点**：WAIT 段末尾是 `CrossCoreWaitFlag`（可 `TokNoBarrier`）；裸下发的 `CUBE2_MMSERVICE` / `CUBE2_CCFLAG_*` 无锚点（需 barrier `Tok` 兜底）。
- 扫描结果整理成总体流程说明输出到 `docs/timer_flow.md`，便于复盘（另建议配原始流程图，见 SKILL.md Step 3）。

**3.2 枚举按算子定制**：固定 7 项、动态 19 项，全部用 3.1 规划出的阶段名命名，模板里的 `FIXED_STAGE_1` / `DYN_STAGE_1` 占位符一律不用：

```cpp
enum FixedTiming
{
    KERNEL_TIMING_IDX = 0,
    INIT_UPDATE,            // AIV: Init 后的状态更新阶段
    SYNC_ALL,               // SyncAll<false> 同步
    MAIN_CUBE1,             // cube: 主循环 matmul 阶段 1
    MAIN_VEC1,              // vector: 主循环 stage1 阶段
    MAIN_CUBE2,             // cube: 主循环 matmul 阶段 2
    MAIN_VEC2,              // vector: 主循环 stage2 阶段
    FIXED_TIMING_COUNT      // 哨兵（自动计数量）
};
enum DynamicTimingType
{
    VEC_MERGE,              // AIV: MergeStage
    VEC1_WAIT,              // AIV: 等上游 cube 阶段（输入就绪）
    VEC1,                   // AIV: VectorStage1（softmax→中间结果）
    VEC2_WAIT,              // AIV: 等上游 cube 阶段（输出就绪）
    VEC2,                   // AIV: VectorStage2（结果写回）
    CUBE1_WAIT,             // AIC: 等上游搬移就绪
    CUBE1,                  // AIC: cube 阶段 1 总时长
    CUBE2_WAIT,             // AIC: cube 阶段 2 等上游 vector 就绪
    CUBE2_MMSERVICE,        // AIC: cube 阶段 2 的 matmulService 耗时
    CUBE2_CCFLAG_A,         // cube 阶段 2 的 CrossCoreSetFlag（下游 A 就绪）
    CUBE2_CCFLAG_B,         // cube 阶段 2 的 CrossCoreSetFlag（下游 B 就绪）
    CUBE2,                  // AIC: cube 阶段 2 总时长
    CUBE_CCFLAG_3,          // cube 阶段 2 之后的 CrossCoreSetFlag
    MERGE_PLAN,             // MergeStage 的 Plan 部分
    MERGE_COPY,             // MergeStage 数据搬移（MTE2/MTE3）
    MERGE_MTE2,             // MergeStage 主循环 MTE2 搬入
    MERGE_MTE3,             // MergeStage 主循环 MTE3 搬出
    MERGE_TAIL,             // MergeStage 尾块处理
    MERGE_SELUPDATE,        // MergeStage 的状态更新导出
    DYNAMIC_TYPE_COUNT_ENUM // 哨兵（自动计数量）
};
```

缓冲布局常量（`N_TIMING_ITEM_PER_CORE`、`N_CORE_COUNT`、`TOTAL_BUFFER_SIZE` 等）由枚举数量 + `MAX_DYNAMIC_ITER` 自动推导；python 侧 `parse_timer_csv.py` 的 `FIXED_TIMING_NAMES` / `DYNAMIC_TIMING_NAMES` 命名映射与枚举**按索引一一对应**（0 起、同名、`len(...)` 自动跟随）。

【坑】固定/动态两组枚举成员名必须两两不重复，否则编译器报 `redefinition of enumerator`。规避成功。

打点规划与枚举方案一并提交用户确认，确认通过后再进入步骤 4 实施（用户 gate）。

结果：success。

## 步骤 4（ref 步骤3）：kernel 主流程初始化

`timer`/`timerAll.Init(timerAddr)` 绑定缓冲；`SyncAll<false>()` 统一各核时间基准；固定项起点一律用 `TikNoBarrier()`（记录指令下发时刻，不排空流水）。

```cpp
TIMER_BLOCK(timer.Init(timerAddr));
TIMER_BLOCK(timerAll.Init(timerAddr));
vectorService.SetTimerPtr(&timer);
TIMER_BLOCK(if (timerAddr != nullptr) { AscendC::SyncAll<false>(); });
TIMER_BLOCK(timerAll.TikNoBarrier());
TIMER_BLOCK(timer.TikNoBarrier());
```

结果：success。

## 步骤 5（ref 步骤4）：固定项打点

`INIT_UPDATE`（barrier `Tok` 收掉 Init 里的那次 Tik）、`SYNC_ALL`（SyncAll 之后用 `TokNoBarrier`）、`KERNEL_TIMING`（timerAll 用 barrier `Tok` 整体收尾）。选型规则：代码段末尾自带完成锚点（WaitFlag/CrossCoreWaitFlag/SyncAll）用 `TokNoBarrier`，裸下发无锚点用带 barrier 的 `Tok`。

结果：success。

## 步骤 6（ref 步骤5）：动态项打点（MainLoop 主循环）

主循环每迭代打一次：`CUBE1` / `CUBE2`（两个 cube 阶段总时长）、`VEC1` / `VEC2`（两个 vector stage）、`VEC_MERGE`、`MERGE_SELUPDATE`、`CUBE_CCFLAG_3` 等；等待段（`VEC1_WAIT` / `VEC2_WAIT` / `CUBE1_WAIT` / `CUBE2_WAIT`）与 cube2 内部细粒度段（`CUBE2_MMSERVICE` / `CUBE2_CCFLAG_A` / `CUBE2_CCFLAG_B`）各占一个槽位。循环体内用 `TikNoBarrier(type, iter)` 起、`Tok<Mode>(type, iter)` / `TokNoBarrier<Mode>(type, iter)` 止；循环前调用 `setDynamicActualIter` 记录实际迭代次数（超过 `MAX_DYNAMIC_ITER` 自动截断）。

结果：success。

## 步骤 7（ref 步骤6）：子模块（vector service）打点【坑：.template 关键字】

打点点在独立 service 类里时，类内加 `AscendTimerDevice *timerPtr_` 成员 + `SetTimerPtr(&timer)` 注入；打点函数签名如 `void Foo(const StageInfo &info)`，槽位索引从 `info.stageIdx` 取。

【坑】`Tok`/`TokNoBarrier` 是模板方法，指针调用必须写 `timerPtr_->template TokNoBarrier<Mode>(...)`，漏写会报 `comparison`/`expected primary-expression` 编译错误。规避成功。

```cpp
TIMER_BLOCK(timerPtr_->TikNoBarrier(AscendTimer::VEC1, info.stageIdx));
VectorStage1SingleBuf(...);
TIMER_BLOCK(timerPtr_->template TokNoBarrier<Accumulate>(AscendTimer::VEC1, info.stageIdx));
```

结果：success。

## 步骤 8（ref 步骤7）：cube 阶段 2 细粒度打点【坑：Overwrite vs Accumulate】

把一段代码拆成多个相邻子段（CrossCoreWaitFlag / matmulService / CrossCoreSetFlag 各成一段），每段一个 `(TikNoBarrier, TokNoBarrier<Accumulate>)` 对，定位到单条同步指令耗时。

【坑】这些子段共用同一 `stageIdx`、单轮迭代写同一 `(type, iter)` 槽位多次：用 `Overwrite` 会被覆盖成只剩最后一块，必须用 `Accumulate`（首写 `[start, start+elapsed]`，之后把 elapsed 累进 end 字段），得到所有子块耗时之和。规避成功。

结果：success。

## 步骤 9（ref 步骤8）：m-split 内层循环的 Overwrite 数据丢失

m-split 内层循环每块都以同一 `stageIdx` 写同一动态槽位，最初用 `TokNoBarrier<Overwrite>`，解析只剩最后一块、耗时被严重低估；全部改为 `Accumulate` 后得到各子块耗时之和。

【注意】`Accumulate` 的 `start` 是第一次打点时刻、`end` 是合成累加值，都不是真实墙钟时刻，只能做 duration 分析，不能拿绝对时间戳做 overlap/时间轴分析。

结果：success。

## 步骤 10（ref 步骤9）：停测 WAIT 段与 VEC_MERGE 收尾

把 `VEC1_WAIT` / `VEC2_WAIT` / `CUBE1_WAIT` / `CUBE2_WAIT` 等 WAIT 段的打点注释掉，但保留 `CrossCoreWaitFlag` 等等待调用本身（它是功能代码）；同时注释对应的 `setDynamicActualIter`，避免解析出恒 0 列噪音；`VEC_MERGE` 的 Tok 改回带 barrier 的普通 `Tok`（该段裸下发无锚点，用 barrier 兜底）。停测一律用 `//` 注释而非删除，便于恢复。

结果：success。

## 步骤 11（ref 步骤10）：python UT 导出

UT 中创建 int64 timer 张量并作为可选输入传给算子，调用后 `torch.npu.synchronize()`；导出 raw 缓冲（`np.savetxt`）、`parse_and_dump` 解析每核每阶段 CSV、`print_summary` 终端汇总、`generate_trace_json` 生成 perfetto trace。

```python
timer = torch.zeros(_timer_parser.TOTAL_BUFFER_SIZE, dtype=torch.int64, device="npu")
call_your_op(inp, timer=timer)
torch.npu.synchronize()
np.savetxt(raw_path, timer.cpu().numpy().reshape(-1, 1), fmt="%d")
_timer_parser.parse_and_dump(str(raw_path), str(parsed_path))
_timer_parser.print_summary(str(parsed_path))
```

结果：success。

## 步骤 12（ref 步骤11）：可视化与验证

用浏览器打开 https://ui.perfetto.dev/ 加载 `all_rank_trace.json`：左侧轨道为各核（`group_AIC_X` / `group_AIV_X`），色块即各阶段事件，`args` 里含 `start_cycle/end_cycle/dur`，可直观看到 AIC/AIV 每核每阶段耗时与等待气泡。

验证项：raw 长度 = `TOTAL_BUFFER_SIZE`、无负 duration、各阶段事件落在 `KERNEL_TIMING` 时间窗内、IterCount 与预期一致。

结果：success。

---

## 效果/价值

- kernel 内部各阶段（matmul / vector / 搬移 / 跨核同步）耗时逐核可见，AIC/AIV 核间负载均衡与等待气泡可量化分析；
- 细粒度打点可把耗时定位到单条同步指令（CrossCoreSetFlag/CrossCoreWaitFlag），指导 m-split 分块与跨核同步策略调优；
- 计时框架由三部分解耦组成（公共定义 `AscendTimerV2.hpp`、NPU 端 `AscendTimerV2_device.hpp`、Python 解析工具），迁移方法对任意 CANN 算子（AscendC kernel + aclnn host 适配 + PyTorch 适配层）通用。

## 经验教训

- 固定/动态两组枚举成员名必须两两不重复，避免 redefinition 编译错误；常量由枚举数量自动推导，host 与 python 侧用常量校验，不硬编码字面量。
- Tik 一律用 `TikNoBarrier`，避免计时器本身排空流水、拖慢性能；Tok 按代码段是否自带完成锚点选型——有锚点（WaitFlag/CrossCoreWaitFlag/SyncAll）用 `TokNoBarrier`，裸下发无锚点用 barrier Tok 兜底。
- 对异步 matmul 用 NoBarrier Tok 测到的是指令下发时刻而非执行完成时刻，需配合锚点或 barrier 才能反映真实耗时。
- 同一 `(type, iter)` 槽位单轮只写一次用 `Overwrite`，单轮写多次（如 m-split 内层循环）必须用 `Accumulate`；`Accumulate` 的 start/end 是合成值，只做 duration 分析，不做绝对时间轴。
- `Tok`/`TokNoBarrier` 是模板方法，对象调用写 `timer.template`，指针调用写 `timerPtr_->template`，漏写必编译报错。
- 停测 WAIT 段时，等待调用本身（CrossCoreWaitFlag/WaitFlag）必须保留，只注释打点与对应 `setDynamicActualIter`；停测用 `//` 注释而非删除，便于恢复。
- 解析侧兜底列（未 setDynamicActualIter 的类型）会输出 `MAX_DYNAMIC_ITER` 个零值列，冗余但无害，可自行瘦身。
- 备用方法：`时刻点法`（在两个时刻各打一个 `TikNoBarrier` 紧接 `TokNoBarrier<Overwrite>` 的瞬时对，用对内平均值之差近似两点耗时）适用于跨函数 Tik/Tok 配对困难的复杂时序分析，**本案例未使用**，详见 `migration-guide.md` §7.4。

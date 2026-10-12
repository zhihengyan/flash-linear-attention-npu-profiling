#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// ========== 1. 枚举定义（CPU/NPU共用） ==========
// 【模板占位】下面两组枚举名仅为占位示例，**必须按你的算子主流程替换为你自己的阶段名**
// （否则导出列全部未命名、与实际流程对不上）。先按 SKILL.md Step 1 确认方案，再按迁移指南第 4 节适配布局。
// 固定计时项（整个 kernel 只跑一次）
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX = 0,     // 固定槽位：整体 kernel 计时（勿删）
    FIXED_STAGE_1,             // 占位：如 Init 后的整体阶段
    FIXED_STAGE_2,             // 占位：如同步基准（SyncAll）
    FIXED_STAGE_3,             // 占位：如主循环 cube 阶段
    FIXED_STAGE_4,             // 占位：如主循环 vector 阶段
    FIXED_STAGE_5,             // 占位：如主循环第二 cube 阶段
    FIXED_STAGE_6,             // 占位：如主循环第二 vector 阶段
    FIXED_TIMING_COUNT
};

// 动态计时项类型（循环内按 iter 重复；常见构成：总时长 / WAIT 等待段 / 细粒度子段）
enum DynamicTimingType : int32_t
{
    DYN_STAGE_1,    // 占位
    DYN_STAGE_2,    // 占位
    DYN_STAGE_3,    // 占位
    DYN_STAGE_4,    // 占位
    DYN_STAGE_5,    // 占位
    DYN_STAGE_6,    // 占位
    DYN_STAGE_7,    // 占位
    DYN_STAGE_8,    // 占位
    DYN_STAGE_9,    // 占位
    DYN_STAGE_10,   // 占位
    DYN_STAGE_11,   // 占位
    DYN_STAGE_12,   // 占位
    DYN_STAGE_13,   // 占位
    DYN_STAGE_14,   // 占位
    DYN_STAGE_15,   // 占位
    DYN_STAGE_16,   // 占位
    DYN_STAGE_17,   // 占位
    DYN_STAGE_18,   // 占位
    DYN_STAGE_19,   // 占位
    DYNAMIC_TYPE_COUNT_ENUM
};

// ========== 2. 公共常量定义（CPU/NPU共用） ==========
// 所有动态项默认最大迭代次数
static constexpr int MAX_DYNAMIC_ITER = 64;

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每个计时项占 2 个 int64 槽位: [start_cycle, end_cycle]
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;                                     // 每核原始计数器数
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;                // 16倍对齐后每核计数器数 128B对齐 每个核写入N_TIMING_COUNTER_PER_CORE_ALIGN个数
// Capacity of timer rows, not a physical core count. Match the parser and launch.
#ifndef ASCEND_TIMER_CORE_CAPACITY
#define ASCEND_TIMER_CORE_CAPACITY 96
#endif
static constexpr int N_CORE_COUNT = ASCEND_TIMER_CORE_CAPACITY;
static constexpr int CACHE_LINE_BYTES = 64;
static constexpr int CACHE_LINE_WORDS = CACHE_LINE_BYTES / sizeof(int64_t);
static_assert(N_CORE_COUNT > 0, "timer capacity must be positive");
static_assert(N_TIMING_COUNTER_PER_CORE_ALIGN % CACHE_LINE_WORDS == 0);
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;                           // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;                         // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;    // 每核动态迭代次数区16对齐 128B对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT; // 动态迭代次数总存储区大小，每个核cacheline不重叠
static_assert(DYNAMIC_ITER_PER_CORE_ALIGN % CACHE_LINE_WORDS == 0);
}
// Override from the target build; 0 must remove instrumentation.
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1
#endif
// Enable only after checking GM visibility, target API and cache-line ownership.
#ifndef ASCEND_TIMER_EXPLICIT_GM_PUBLISH
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 0
#endif
#if ENABLE_TIMER
#define TIMER_BLOCK(...) do { __VA_ARGS__; } while (0)
#else
#define TIMER_BLOCK(...) do { } while (0)
#endif

#endif // ASCEND_TIMER_COMMON_H

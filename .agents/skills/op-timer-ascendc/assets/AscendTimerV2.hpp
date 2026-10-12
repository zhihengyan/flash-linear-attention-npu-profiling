#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// ========== 1. 枚举定义（CPU/NPU共用） ==========
// 【模板占位】下面两组枚举名仅为占位示例，**必须按你的算子主流程替换为你自己的阶段名**
// （否则导出列全部未命名、与实际流程对不上）。替换步骤见 SKILL.md Step 3.3 / migration-guide.md。
// 固定计时项（整个 kernel 只跑一次）
enum FixedTiming
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
enum DynamicTimingType
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
static constexpr int N_CORE_COUNT = 96; // 计时缓冲预留的核数上限（AIC+AIV），按目标硬件型号与分核策略调整，这里保留为96
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;                           // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;                         // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;    // 每核动态迭代次数区16对齐 128B对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT; // 动态迭代次数总存储区大小，每个核cacheline不重叠
}
// ========== 4. 计时开关宏 ==========
 #define ENABLE_TIMER 1 // 计时开关，因为计时会打乱流水，开关验证计时器对流水的影响。
#ifdef ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
        code ;            \
    } while (0);
#else
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
    } while (0);
#endif

#endif // ASCEND_TIMER_COMMON_H

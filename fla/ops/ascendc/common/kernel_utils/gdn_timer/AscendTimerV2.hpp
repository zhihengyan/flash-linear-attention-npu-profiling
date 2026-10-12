// Derived from op-timer-ascendc, commit 52e396e599a0101b00757ab575c029cb4284ad24.
// Copyright (c) 2026 Send_Tanks. MIT; see AscendTimerV2_LICENSE.txt.
#ifndef GDN_ASCEND_TIMER_V2_HPP
#define GDN_ASCEND_TIMER_V2_HPP

#include <stdint.h>

namespace AscendTimer {

// Scheme A: fixed events only. Keep the order identical to the parser.
enum FixedTiming {
    KERNEL_TIMING_IDX = 0,
    MAIN_INIT,
    MAIN_BODY,
    PHASE6_FRONT_DONE,
    PHASE6_H_DISPATCH_DONE,
    PHASE6_HO_WAIT,
    FIXED_TIMING_COUNT
};

enum KernelSlice {
    PHASE6 = 0,
    PREPARE,
    FWD_H,
    FWD_O,
    KERNEL_SLICE_COUNT
};

static constexpr int LAYOUT_VERSION = 1;
static constexpr int HEADER_MAGIC = 0x47444e54;
static constexpr int HEADER_WORDS = 16;
static constexpr int CACHE_LINE_BYTES = 64;
static constexpr int CACHE_LINE_WORDS = CACHE_LINE_BYTES / 8;
// Reserved capacity, NOT a claim about the physical core count of any SoC.
// Init records actual AIC blockDim; the parser rejects capacity overflow.
static constexpr int CORE_GROUP_CAPACITY = 256;
static constexpr int CORES_PER_GROUP = 3;
static constexpr int N_CORE_COUNT = CORE_GROUP_CAPACITY * CORES_PER_GROUP;
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int BUFFER_WORDS_PER_KERNEL = HEADER_WORDS + N_TIMING_COUNTER;
static constexpr int TOTAL_BUFFER_SIZE = BUFFER_WORDS_PER_KERNEL * KERNEL_SLICE_COUNT;

} // namespace AscendTimer

#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1
#endif

#if ENABLE_TIMER
#define TIMER_BLOCK(code) do { code; } while (0)
#else
#define TIMER_BLOCK(code) do {} while (0)
#endif

#endif // GDN_ASCEND_TIMER_V2_HPP

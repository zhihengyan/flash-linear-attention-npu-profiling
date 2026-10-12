#ifndef ASCEND_TIMER_DEVICE_H
#define ASCEND_TIMER_DEVICE_H

#include "AscendTimerV2.hpp"
#include <type_traits>

#define ASCEND_TIMER_TOTAL_BUFFER_SIZE (AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE)
struct Accumulate {};
struct Overwrite {};

// Mixed AIC + two AIV example. Adapt mapping and parser together for other ratios.
class AscendTimerDevice {
public:
    static constexpr int STATE_COUNT = AscendTimer::DYNAMIC_TYPE_COUNT > 0 ? AscendTimer::DYNAMIC_TYPE_COUNT : 1;
    GM_ADDR t_addr = nullptr;
    int64_t start_cycle = 0;
    int64_t dynamic_start_cycles[STATE_COUNT] = {};
    int dynamic_start_iters[STATE_COUNT] = {};
    bool dynamic_valid[STATE_COUNT] = {};
    int block_id = 0, r = 0, group_id = 0, subblockid = 0, core_id = -1;

    __aicore__ inline AscendTimerDevice() = default;
    __aicore__ inline explicit AscendTimerDevice(GM_ADDR addr) { Init(addr); }
    __aicore__ inline void Init(GM_ADDR addr) {
        t_addr = nullptr;
        core_id = -1;
        start_cycle = 0;
        for (int type = 0; type < STATE_COUNT; ++type) {
            dynamic_start_cycles[type] = 0;
            dynamic_start_iters[type] = -1;
            dynamic_valid[type] = false;
        }
#if ENABLE_TIMER
        if (addr == nullptr || reinterpret_cast<uint64_t>(addr) % AscendTimer::CACHE_LINE_BYTES) { return; }
        block_id = AscendC::GetBlockIdx();
        group_id = block_id;
        subblockid = 0;
        if ASCEND_IS_AIV {
            r = AscendC::GetTaskRation();
            if (r != 2) { return; } // Reject unsupported mapping; never alias rows.
            group_id = block_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            if (subblockid < 0 || subblockid >= 2) { return; }
            core_id = group_id * 3 + subblockid + 1;
        } else {
            core_id = group_id * 3;
        }
        if (group_id < 0 || core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT) {
            core_id = -1;
            return;
        }
        t_addr = addr;
#endif
    }
    __aicore__ inline void Init(GM_ADDR addr, int64_t start) { Init(addr); if (t_addr) { start_cycle = start; } }
    __aicore__ inline GM_ADDR GetTAddr() { return t_addr; }
    __aicore__ inline int64_t GetStartCycle() { return start_cycle; }
    __aicore__ inline int GetCoreId() { return core_id; }

    __aicore__ inline void zero(int32_t idx) {
        if (!ValidFixed(idx)) { return; }
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(t_addr));
        int offset = calculate_index(idx);
        data(offset) = 0;
        data(offset + 1) = 0;
    }
    __aicore__ inline void zero() { zero(AscendTimer::KERNEL_TIMING_IDX); }
    __aicore__ inline void setDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t count) {
        if (!t_addr || (int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || count < 0) { return; }
        // Store the true total so the parser can report samples clipped at MAX_DYNAMIC_ITER.
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(t_addr));
        data(AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type) = count;
    }
    __aicore__ inline void zeroDynamicActualIterAll() {
        for (int type = 0; type < AscendTimer::DYNAMIC_TYPE_COUNT; ++type) {
            setDynamicActualIter(static_cast<AscendTimer::DynamicTimingType>(type), 0);
        }
    }
    __aicore__ inline void zeroDynamicTimingAll() {
        if (!t_addr) { return; }
        for (int idx = AscendTimer::FIXED_TIMING_COUNT; idx < AscendTimer::N_TIMING_ITEM_PER_CORE; ++idx) { zero(idx); }
    }

    __aicore__ inline void TikNoBarrier() { if (t_addr) { start_cycle = AscendC::GetSystemCycle(); } }
    __aicore__ inline void Tik() {
        if (!t_addr) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        TikNoBarrier();
    }
    __aicore__ inline void Tik(int64_t start) { if (t_addr) { start_cycle = start; } }
    __aicore__ inline void TikNoBarrier(AscendTimer::DynamicTimingType type, int iter) {
        if (!ValidDynamic(type, iter)) { return; }
        dynamic_start_cycles[(int)type] = AscendC::GetSystemCycle();
        dynamic_start_iters[(int)type] = iter;
        dynamic_valid[(int)type] = true;
    }
    __aicore__ inline void Tik(AscendTimer::DynamicTimingType type, int iter) {
        if (!ValidDynamic(type, iter)) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        TikNoBarrier(type, iter);
    }

    template <typename Mode = Accumulate>
    __aicore__ inline void TokNoBarrier(int32_t idx) {
        if (!ValidFixed(idx)) { return; }
        Write<Mode>(idx, start_cycle, AscendC::GetSystemCycle());
    }
    template <typename Mode = Accumulate>
    __aicore__ inline void Tok(int32_t idx) {
        if (!ValidFixed(idx)) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        TokNoBarrier<Mode>(idx);
    }
    template <typename Mode = Overwrite>
    __aicore__ inline void Tok() { Tok<Mode>(AscendTimer::KERNEL_TIMING_IDX); }
    template <typename Mode = Overwrite>
    __aicore__ inline void TokNoBarrier(AscendTimer::DynamicTimingType type, int iter) {
        if (!ValidDynamic(type, iter) || !dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) { return; }
        int idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;
        Write<Mode>(idx, dynamic_start_cycles[(int)type], AscendC::GetSystemCycle());
        dynamic_valid[(int)type] = false;
        dynamic_start_iters[(int)type] = -1;
    }
    template <typename Mode = Overwrite>
    __aicore__ inline void Tok(AscendTimer::DynamicTimingType type, int iter) {
        if (!ValidDynamic(type, iter) || !dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        TokNoBarrier<Mode>(type, iter);
    }

    // Call ONCE after every timer object's final Tok; publication is outside measured windows.
    __aicore__ inline void Publish() {
#if ENABLE_TIMER && ASCEND_TIMER_EXPLICIT_GM_PUBLISH
        if (!t_addr) { return; }
        Flush(core_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN, AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN);
        Flush(AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN,
              AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN);
#endif
    }
    __aicore__ inline void print() {
        if (t_addr) { AscendC::printf("AscendTimer core=%d group=%d addr=%p\n", core_id, group_id, t_addr); }
    }

private:
    __aicore__ inline bool ValidFixed(int idx) const {
        return t_addr && idx >= 0 && idx < AscendTimer::N_TIMING_ITEM_PER_CORE;
    }
    __aicore__ inline bool ValidDynamic(AscendTimer::DynamicTimingType type, int iter) const {
        return t_addr && (int)type >= 0 && (int)type < AscendTimer::DYNAMIC_TYPE_COUNT &&
               iter >= 0 && iter < AscendTimer::MAX_DYNAMIC_ITER;
    }
    __aicore__ inline int calculate_index(int idx) const {
        return core_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN + idx * 2;
    }
    template <typename Mode>
    __aicore__ inline void Write(int idx, int64_t start, int64_t end) {
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(t_addr));
        int offset = calculate_index(idx);
        if constexpr (std::is_same_v<Mode, Accumulate>) {
            if (data(offset) == 0 && data(offset + 1) == 0) { data(offset) = start; data(offset + 1) = end; }
            else { data(offset + 1) += end - start; }
        } else { data(offset) = start; data(offset + 1) = end; }
    }
#if ENABLE_TIMER && ASCEND_TIMER_EXPLICIT_GM_PUBLISH
    __aicore__ inline void Flush(int offset, int words) {
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(t_addr) + offset);
        for (int word = 0; word < words; word += AscendTimer::CACHE_LINE_WORDS) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                            AscendC::DcciDst::CACHELINE_OUT>(data[word]);
        }
    }
#endif
};
#endif

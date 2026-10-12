// Fixed-event adaptation of op-timer-ascendc AscendTimerV2 (scheme A).
// Copyright (c) 2026 Send_Tanks. MIT; see AscendTimerV2_LICENSE.txt.
#ifndef GDN_ASCEND_TIMER_V2_DEVICE_HPP
#define GDN_ASCEND_TIMER_V2_DEVICE_HPP

#include "kernel_operator.h"
#include "AscendTimerV2.hpp"
#include <type_traits>

struct Overwrite {};
struct Accumulate {};

class AscendTimerDevice {
public:
    __aicore__ inline AscendTimerDevice() = default;

    __aicore__ inline void Init(GM_ADDR timerAddr, AscendTimer::KernelSlice kernel)
    {
        tAddr_ = nullptr;
        valid_ = false;
        if (timerAddr == nullptr) {
            return;
        }
        uint32_t group = AscendC::GetBlockIdx();
        uint32_t member = 0;
        if ASCEND_IS_AIV {
            group /= AscendC::GetSubBlockNum();
            member = AscendC::GetSubBlockIdx() + 1;
        }
        coreId_ = group * AscendTimer::CORES_PER_GROUP + member;
        GM_ADDR slice = timerAddr + static_cast<uint64_t>(kernel) *
            AscendTimer::BUFFER_WORDS_PER_KERNEL * sizeof(int64_t);
        // AIC 0 is the sole metadata writer. No new cross-core synchronization.
        if ASCEND_IS_AIC {
            if (group == 0) {
                AscendC::GlobalTensor<int64_t> header;
                header.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slice));
                header(0) = AscendTimer::HEADER_MAGIC;
                header(1) = AscendTimer::LAYOUT_VERSION;
                header(2) = static_cast<int64_t>(kernel);
                header(3) = AscendC::GetBlockNum();
                header(4) = AscendTimer::CORE_GROUP_CAPACITY;
                header(5) = AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN;
                header(6) = AscendTimer::FIXED_TIMING_COUNT;
                header(7) = ENABLE_TIMER;
#if defined(__CCE_AICORE__)
                header(8) = __CCE_AICORE__;
#endif
            }
        }
        // Do not redirect an out-of-range core to core zero.
        if (group >= AscendTimer::CORE_GROUP_CAPACITY || member >= AscendTimer::CORES_PER_GROUP) {
            return;
        }
        tAddr_ = slice + AscendTimer::HEADER_WORDS * sizeof(int64_t);
    }

    __aicore__ inline void BindFrom(const AscendTimerDevice &source)
    {
        tAddr_ = source.tAddr_;
        coreId_ = source.coreId_;
        startCycle_ = 0;
        endCycle_ = 0;
        valid_ = false;
    }

    __aicore__ inline void TikNoBarrier()
    {
        if (tAddr_ != nullptr) {
            startCycle_ = AscendC::GetSystemCycle();
            valid_ = true;
        }
    }

    template <typename Mode = Overwrite>
    __aicore__ inline void Tok(int32_t timingIdx)
    {
        if (tAddr_ == nullptr || !valid_) {
            return;
        }
        // Scheme A drains only at the very end of the actual kernel.
        AscendC::PipeBarrier<PIPE_ALL>();
        this->template TokNoBarrier<Mode>(timingIdx);
    }

    template <typename Mode = Overwrite>
    __aicore__ inline void TokNoBarrier(int32_t timingIdx)
    {
        if (tAddr_ != nullptr && valid_) {
            this->template TokNoBarrier<Mode>(timingIdx, AscendC::GetSystemCycle());
        }
    }

    // Reuse the final kernel timestamp for MAIN_BODY, without a second drain
    // or an end timestamp outside KERNEL_TIMING.
    template <typename Mode = Overwrite>
    __aicore__ inline void TokNoBarrier(int32_t timingIdx, int64_t endCycle)
    {
        if (tAddr_ == nullptr || !valid_ || timingIdx < 0 ||
            timingIdx >= AscendTimer::FIXED_TIMING_COUNT) {
            return;
        }
        endCycle_ = endCycle;
        const int32_t index = static_cast<int32_t>(coreId_) *
            AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN + timingIdx * 2;
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tAddr_));
        if constexpr (std::is_same<Mode, Accumulate>::value) {
            if (data(index) == 0 && data(index + 1) == 0) {
                data(index) = startCycle_;
                data(index + 1) = endCycle;
            } else {
                data(index + 1) += endCycle - startCycle_;
            }
        } else {
            data(index) = startCycle_;
            data(index + 1) = endCycle;
        }
        valid_ = false;
    }

    __aicore__ inline int64_t GetEndCycle() const { return endCycle_; }

private:
    GM_ADDR tAddr_ = nullptr;
    uint32_t coreId_ = 0;
    int64_t startCycle_ = 0;
    int64_t endCycle_ = 0;
    bool valid_ = false;
};

// Independent starts prevent INIT/BODY, whole-kernel and Phase6 boundaries
// from overwriting one another's in-flight timestamp.
class GdnKernelTimer {
public:
    __aicore__ inline GdnKernelTimer() = default;

    __aicore__ inline void Init(GM_ADDR addr, AscendTimer::KernelSlice kernel)
    {
#if ENABLE_TIMER
        all_.Init(addr, kernel);
        stage_.BindFrom(all_);
        boundary_.BindFrom(all_);
        all_.TikNoBarrier();
        stage_.TikNoBarrier();
#endif
    }

    __aicore__ inline void InitDone()
    {
#if ENABLE_TIMER
        stage_.template TokNoBarrier<Overwrite>(AscendTimer::MAIN_INIT);
        stage_.TikNoBarrier();
#endif
    }

    __aicore__ inline void Point(AscendTimer::FixedTiming timing)
    {
#if ENABLE_TIMER
        boundary_.TikNoBarrier();
        boundary_.template TokNoBarrier<Overwrite>(timing);
#endif
    }

    __aicore__ inline void HoWaitBegin()
    {
#if ENABLE_TIMER
        boundary_.TikNoBarrier();
#endif
    }

    __aicore__ inline void HoWaitEnd()
    {
#if ENABLE_TIMER
        boundary_.template TokNoBarrier<Overwrite>(AscendTimer::PHASE6_HO_WAIT);
#endif
    }

    __aicore__ inline void Finish()
    {
#if ENABLE_TIMER
        all_.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
        stage_.template TokNoBarrier<Overwrite>(AscendTimer::MAIN_BODY, all_.GetEndCycle());
#endif
    }

private:
#if ENABLE_TIMER
    AscendTimerDevice all_;
    AscendTimerDevice stage_;
    AscendTimerDevice boundary_;
#endif
};

#endif // GDN_ASCEND_TIMER_V2_DEVICE_HPP

// GDN adaptation of op-timer-ascendc AscendTimerV2.
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
        // Rows must own whole cache lines; never flush a neighbour's storage.
        if (timerAddr == nullptr || reinterpret_cast<uint64_t>(timerAddr) % AscendTimer::CACHE_LINE_BYTES != 0) {
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
                header(9) = AscendTimer::DYNAMIC_TYPE_COUNT_ENUM;
                header(10) = AscendTimer::MAX_DYNAMIC_ITER;
                header(11) = AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
                FlushCacheLines(header, AscendTimer::HEADER_WORDS);
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
        // Drain only at the very end of a diagnostic kernel.
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
    __aicore__ inline bool Enabled() const { return tAddr_ != nullptr; }

    __aicore__ inline void setDynamicActualIter(int type, int64_t count)
    {
        if (!Enabled() || type < 0 || type >= AscendTimer::DYNAMIC_TYPE_COUNT_ENUM) return;
        AscendC::GlobalTensor<int64_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tAddr_) +
                               AscendTimer::N_TIMING_COUNTER);
        counts(coreId_ * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + type) = count;
    }

    __aicore__ inline void TokNoBarrier(int type, int iter, int64_t start)
    {
        if (!Enabled() || type < 0 || type >= AscendTimer::DYNAMIC_TYPE_COUNT_ENUM ||
            iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER) return;
        const int64_t end = AscendC::GetSystemCycle();
        const int index = coreId_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN +
            2 * (AscendTimer::FIXED_TIMING_COUNT + type * AscendTimer::MAX_DYNAMIC_ITER + iter);
        AscendC::GlobalTensor<int64_t> data;
        data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tAddr_));
        data(index) = start;
        data(index + 1) = end;
    }

    __aicore__ inline void FlushRow()
    {
        if (tAddr_ == nullptr) {
            return;
        }
        AscendC::GlobalTensor<int64_t> row;
        row.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tAddr_) +
                           coreId_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN);
        FlushCacheLines(row, AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN);
        AscendC::GlobalTensor<int64_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tAddr_) +
            AscendTimer::N_TIMING_COUNTER + coreId_ * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN);
        FlushCacheLines(counts, AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN);
    }

private:
    __aicore__ static inline void FlushCacheLines(const AscendC::GlobalTensor<int64_t> &data, int words)
    {
        static_assert(AscendTimer::CACHE_LINE_WORDS * sizeof(int64_t) == AscendTimer::CACHE_LINE_BYTES);
        static_assert(AscendTimer::HEADER_WORDS % AscendTimer::CACHE_LINE_WORDS == 0);
        static_assert(AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN % AscendTimer::CACHE_LINE_WORDS == 0);
        // Scalar GM stores stay in DCache. Prepare explicitly disables the
        // compiler's end-of-kernel DCCI, so PIPE_ALL alone cannot publish them.
        for (int word = 0; word < words; word += AscendTimer::CACHE_LINE_WORDS) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                            AscendC::DcciDst::CACHELINE_OUT>(data[word]);
        }
    }

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
        if (!all_.Enabled()) return;
        for (int type = 0; type < AscendTimer::DYNAMIC_TYPE_COUNT_ENUM; ++type) next_[type] = 0;
        all_.TikNoBarrier();
        stage_.TikNoBarrier();
#endif
    }

    __aicore__ inline bool Enabled() const
    {
#if ENABLE_TIMER
        return all_.Enabled();
#else
        return false;
#endif
    }

    // Each stage has its own call ordinal; nested scopes cannot overwrite starts.
    __aicore__ inline int Begin(AscendTimer::DynamicTimingType type)
    {
#if ENABLE_TIMER
        if (!Enabled()) return -1;
        const int iter = next_[type]++;
        all_.setDynamicActualIter(type, next_[type]);
        return iter < AscendTimer::MAX_DYNAMIC_ITER ? iter : -1;
#else
        return -1;
#endif
    }

    __aicore__ inline void End(AscendTimer::DynamicTimingType type, int iter, int64_t start)
    {
#if ENABLE_TIMER
        all_.TokNoBarrier(type, iter, start);
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
        // Publish only this core's timer row, after both measured end stamps.
        // No whole-cache flush or extra business/cross-core synchronization.
        all_.FlushRow();
#endif
    }

private:
#if ENABLE_TIMER
    AscendTimerDevice all_;
    AscendTimerDevice stage_;
    AscendTimerDevice boundary_;
    int next_[AscendTimer::DYNAMIC_TYPE_COUNT_ENUM];
#endif
};

// The destructor pairs every scope, including return/continue paths. All stage
// windows are NoBarrier issue windows or existing waits; only Finish drains.
class GdnTimerScope {
public:
    __aicore__ inline GdnTimerScope(GdnKernelTimer *timer, AscendTimer::DynamicTimingType type)
        : timer_(timer), type_(type)
    {
        if (timer_ != nullptr && timer_->Enabled()) {
            iter_ = timer_->Begin(type_);
            if (iter_ >= 0) start_ = AscendC::GetSystemCycle(); // TikNoBarrier
        }
    }
    __aicore__ inline ~GdnTimerScope()
    {
        if (iter_ >= 0) timer_->End(type_, iter_, start_); // TokNoBarrier<Overwrite>
    }
    GdnTimerScope(const GdnTimerScope &) = delete;
    GdnTimerScope &operator=(const GdnTimerScope &) = delete;
private:
    GdnKernelTimer *timer_;
    AscendTimer::DynamicTimingType type_;
    int iter_ = -1;
    int64_t start_ = 0;
};

#define GDN_TIMER_JOIN_IMPL(a, b) a##b
#define GDN_TIMER_JOIN(a, b) GDN_TIMER_JOIN_IMPL(a, b)
#if ENABLE_TIMER
#define GDN_TIMER_SCOPE(timer, stage) GdnTimerScope GDN_TIMER_JOIN(gdnTimerScope_, __LINE__)(timer, AscendTimer::stage)
#else
#define GDN_TIMER_SCOPE(timer, stage) do {} while (0)
#endif

#endif // GDN_ASCEND_TIMER_V2_DEVICE_HPP

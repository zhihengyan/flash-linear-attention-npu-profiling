/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Cube MMAD: C[m, n] = A[m, k] @ B[k, n] into L0C. L1 is cube NZ.
 *   fp32 C0=8, bf16 C0=16.
 *   transposeB=false: B NZ is [n, k] (kkt: same k' tile as A, n=m).
 *   transposeB=true:  B NZ is [k, n] (I / -L / Y). Leaf tiles from 8-col
 *   ND->NZ are NZ(inv.T): pass transposeA / !transposeB for those Loads.
 * fp32 L1 slots are 64x64, so A/B srcStride is kNumMFracs64.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

__aicore__ inline void FillLoad2D(AscendC::LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep,
                                  uint16_t srcStride, uint16_t dstStride, bool transpose)
{
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mStep;
    p.kStep = kStep;
    p.srcStride = srcStride;
    p.dstStride = dstStride;
    p.ifTranspose = transpose;
    p.sid = 0;
}

template <typename InDtype>
__aicore__ inline void LoadL1NzToL0AB(AscendC::LocalTensor<InDtype> l1A,
                                      AscendC::LocalTensor<InDtype> l1B,
                                      AscendC::LocalTensor<InDtype> l0A,
                                      AscendC::LocalTensor<InDtype> l0B,
                                      int32_t m, int32_t n, int32_t k,
                                      bool transposeB, bool transposeA, uint8_t evt)
{
    constexpr uint32_t c0 = 32 / sizeof(InDtype);
    constexpr bool kFp32 = sizeof(InDtype) == sizeof(float);
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    const uint16_t aSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : mFracs;

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, aKFracs, aSrc, mFracs, transposeA);
    AscendC::LoadData(l0A, l1A, loadA);

    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16)
                                        : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0)
                                        : static_cast<uint16_t>(k / c0);
    const uint16_t bSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : bMFracs;

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, bMFracs, bKFracs, bSrc, bMFracs, transposeB);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
}

template <typename InDtype>
__aicore__ inline void MmadL0ABToL0C(AscendC::LocalTensor<InDtype> l0A,
                                     AscendC::LocalTensor<InDtype> l0B,
                                     AscendC::LocalTensor<float> l0C,
                                     int32_t m, int32_t n, int32_t k, bool initC, uint8_t evt)
{
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = k;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

template <typename InDtype>
__aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                   AscendC::LocalTensor<InDtype> l1B,
                                   AscendC::LocalTensor<InDtype> l0A,
                                   AscendC::LocalTensor<InDtype> l0B,
                                   AscendC::LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k,
                                   bool initC, bool transposeB, bool transposeA = false,
                                   uint8_t evt = 0)
{
    LoadL1NzToL0AB<InDtype>(l1A, l1B, l0A, l0B, m, n, k, transposeB, transposeA, evt);
    MmadL0ABToL0C<InDtype>(l0A, l0B, l0C, m, n, k, initC, evt);
}

// Solve-only FP32 tiles: logical M=N=64, K=32, backed by full 64x64 L1 NZ.
// With transpose, select source rows (32x64 -> 64x32); without transpose,
// select source columns (64x32). Both A and B use the same descriptor here:
// B's packed NZ view is the transpose of logical [K,N] in L0B Zn.
// dstStride stays 4 even when source mStep=2: the destination retains all
// 64 output rows/columns. Each load fills exactly the first 8 KiB of its
// existing 16 KiB L0 bank. See docs/prepare_k32_design.md in the fused op.
template <bool TransposeInputs, bool SecondHalf>
__aicore__ inline void FillSolveHalfKLoad(AscendC::LoadData2DParamsV2 &p)
{
    if constexpr (TransposeInputs) {
        FillLoad2D(p, 2, 8, 4, 4, true);
        p.mStartPosition = SecondHalf ? 2 : 0;
    } else {
        FillLoad2D(p, 4, 4, 4, 4, false);
        p.kStartPosition = SecondHalf ? 4 : 0;
    }
}

template <bool TransposeInputs, bool SecondHalf>
__aicore__ inline void SolveHalfKMatmulToL0C(AscendC::LocalTensor<float> l1A,
                                            AscendC::LocalTensor<float> l1B,
                                            AscendC::LocalTensor<float> l0A,
                                            AscendC::LocalTensor<float> l0B,
                                            AscendC::LocalTensor<float> l0C,
                                            bool initC, uint8_t evt)
{
    AscendC::LoadData2DParamsV2 load;
    FillSolveHalfKLoad<TransposeInputs, SecondHalf>(load);
    AscendC::LoadData(l0A, l1A, load);
    AscendC::LoadData(l0B, l1B, load);
    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    MmadL0ABToL0C<float>(l0A, l0B, l0C, kChunk64, kChunk64, kVcs32, initC, evt);
}

// X = R11 @ (-L10). Both inputs retain full64 FP32 NZ images.
// Left stores R11.T in BR; NegL stores -L10 in BL. Each Load fills only
// the first 4 KiB of its 16 KiB bank. The result is a standalone32 tile.
__aicore__ inline void FillStage4Block32Loads(AscendC::LoadData2DParamsV2 &loadA,
                                             AscendC::LoadData2DParamsV2 &loadB)
{
    FillLoad2D(loadA, 2, 4, 4, 2, true);
    loadA.mStartPosition = 2;
    loadA.kStartPosition = 4;
    FillLoad2D(loadB, 2, 4, 4, 2, true);
    loadB.mStartPosition = 2;
}

__aicore__ inline void Stage4Block32MatmulToL0C(AscendC::LocalTensor<float> l1LeafLeft,
                                               AscendC::LocalTensor<float> l1NegL,
                                               AscendC::LocalTensor<float> l0A,
                                               AscendC::LocalTensor<float> l0B,
                                               AscendC::LocalTensor<float> l0C, uint8_t evt)
{
    AscendC::LoadData2DParamsV2 loadA;
    AscendC::LoadData2DParamsV2 loadB;
    FillStage4Block32Loads(loadA, loadB);
    AscendC::LoadData(l0A, l1LeafLeft, loadA);
    AscendC::LoadData(l0B, l1NegL, loadB);
    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    MmadL0ABToL0C<float>(l0A, l0B, l0C, kVcs32, kVcs32, kVcs32, true, evt);
}

// C10 = X @ R00. X is tight32 NZ; Right stores R00.T in full64 TL.
// An untransposed L0B load is interpreted as Zn by the MMAD.
__aicore__ inline void FillStage5Block32Loads(AscendC::LoadData2DParamsV2 &loadA,
                                             AscendC::LoadData2DParamsV2 &loadB)
{
    FillLoad2D(loadA, 2, 4, 2, 2, false);
    FillLoad2D(loadB, 2, 4, 4, 2, false);
}

__aicore__ inline void Stage5Block32MatmulToL0C(AscendC::LocalTensor<float> l1X,
                                               AscendC::LocalTensor<float> l1LeafRight,
                                               AscendC::LocalTensor<float> l0A,
                                               AscendC::LocalTensor<float> l0B,
                                               AscendC::LocalTensor<float> l0C, uint8_t evt)
{
    AscendC::LoadData2DParamsV2 loadA;
    AscendC::LoadData2DParamsV2 loadB;
    FillStage5Block32Loads(loadA, loadB);
    AscendC::LoadData(l0A, l1X, loadA);
    AscendC::LoadData(l0B, l1LeafRight, loadB);
    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    MmadL0ABToL0C<float>(l0A, l0B, l0C, kVcs32, kVcs32, kVcs32, true, evt);
}

// C = A[m,k] @ B[k,n]. One MMAD. A is 64x64 (kStep=4). B is 64x128
// (isTranspose, kStep=8, dstStride=8). MMAD k stays kDim.
template <typename InDtype>
__aicore__ inline void WuMatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                     AscendC::LocalTensor<InDtype> l1B,
                                     AscendC::LocalTensor<InDtype> l0A,
                                     AscendC::LocalTensor<InDtype> l0B,
                                     AscendC::LocalTensor<float> l0C,
                                     int32_t m, int32_t n, int32_t kDim,
                                     uint8_t evt)
{
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t nFracs = static_cast<uint16_t>(n / 16);
    const uint16_t kFracs = static_cast<uint16_t>(kDim / 16);

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, kFracs, mFracs, mFracs, false);
    AscendC::LoadData(l0A, l1A, loadA);

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, kFracs, nFracs, kFracs, nFracs, true);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = kDim;
    mmad.cmatrixInitVal = true;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif

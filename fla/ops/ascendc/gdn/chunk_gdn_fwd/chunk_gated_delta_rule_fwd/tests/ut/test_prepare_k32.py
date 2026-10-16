"""Historical K32 FP32 NZ checks; not a CANN instruction emulator.

Read the frozen K32 commit's literal load descriptors and call sites, then
compare independently decoded operands against dense matrix slices. CANN
compilation, hardware rounding and synchronization still need device tests.
Requires NumPy; run this file directly from any directory.
The active Stage5 experiment is checked by test_prepare_stage5_direct_l1.py.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import unittest

import numpy as np


OP = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[8]
ARCH = OP.parent / "chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35"
MATMUL = ARCH / "chunk_gated_delta_rule_fwd_prepare_matmul.h"
KERNEL = ARCH / "chunk_gated_delta_rule_fwd_prepare.h"
BASELINE = "809bce6887639e831caa856b8ac3a197a6a94294"
K32_COMMIT = "d0cb628df7c8c23c0342392dfaf4ec910aba9947"


@lru_cache(maxsize=None)
def k32_source(path):
    return subprocess.check_output(
        ["git", "show", f"{K32_COMMIT}:{path.relative_to(ROOT).as_posix()}"],
        cwd=ROOT, encoding="utf-8",
    )


def body(source, name):
    start = source.index("{", source.index("inline void " + name + "("))
    level = 1
    end = start + 1
    while level:
        level += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


def tokens(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S))


@dataclass(frozen=True)
class Load:
    m_start: int
    k_start: int
    m_step: int
    k_step: int
    src_stride: int
    dst_stride: int
    transpose: bool


def source_load(transpose, second_half):
    # Deliberately limited to the two literal descriptor branches, not a C++
    # interpreter. Fail if the source no longer has this form; don't silently
    # test a Python copy of stale production parameters.
    source = body(k32_source(MATMUL), "FillSolveHalfKLoad")
    branches = re.fullmatch(
        r"\s*if constexpr \(TransposeInputs\) \{(.*?)\} else \{(.*?)\}\s*",
        source, re.S,
    )
    if branches is None:
        raise AssertionError("update descriptor extraction for changed C++ structure")
    branch = branches.group(1 if transpose else 2)
    params = re.search(r"FillLoad2D\(p, (\d+), (\d+), (\d+), (\d+), (true|false)\);", branch)
    if params is None:
        raise AssertionError("expected literal LoadData descriptor")
    m_step, k_step, src_stride, dst_stride = map(int, params.groups()[:4])
    field, high, low = re.search(
        r"p\.(mStartPosition|kStartPosition) = SecondHalf \? (\d+) : (\d+);", branch
    ).groups()
    offset = int(high if second_half else low)
    return Load(offset if field == "mStartPosition" else 0,
                offset if field == "kStartPosition" else 0,
                m_step, k_step, src_stride, dst_stride, params.group(5) == "true")


def call_modes(stage):
    source = body(k32_source(KERNEL), stage)
    return [(a == "true", b == "true") for a, b in re.findall(
        r"SolveHalfKMatmulToL0C<(true|false), (true|false)>", source
    )]


def nz_encode(matrix):
    """Independent ND -> NZ encoder: [columns/8, rows/16, 16, 8]."""
    rows, cols = matrix.shape
    return matrix.reshape(rows // 16, 16, cols // 8, 8).transpose(2, 0, 1, 3).reshape(-1)


def load_model(source, desc):
    """Source-fractal addressing and optional transpose, into a poisoned slot.

    A target is NZ [64,32]; the same bytes are B's Zn [32,64]. A transpose
    groups two FP32 16x8 source fractals to transpose their 16x16 square.
    The destination is addressed with dst_stride, independently of m_step.
    """
    target = np.full(4096, np.nan, dtype=source.dtype)
    written = set()
    source_indices = set()
    for m in range(desc.m_step * 16):
        for k in range(desc.k_step * 8):
            row = desc.m_start * 16 + m
            col = desc.k_start * 8 + k
            src = (col // 8 * desc.src_stride + row // 16) * 128 + row % 16 * 8 + col % 8
            dr, dc = (k, m) if desc.transpose else (m, k)
            dst = (dc // 8 * desc.dst_stride + dr // 16) * 128 + dr % 16 * 8 + dc % 8
            if src >= len(source) or dst >= len(target) or dst in written:
                raise ValueError("out of bounds or overlapping load")
            target[dst] = source[src]
            written.add(dst)
            source_indices.add(src)
    return target, written, source_indices


def decode_operands(l1_a, l1_b, mode):
    desc = source_load(*mode)
    # Decode by the target MMAD's fixed M=64,K=32,N=64, not source step sizes.
    row, col = np.indices((64, 32))
    addresses = (col // 8 * 4 + row // 16) * 128 + row % 16 * 8 + col % 8
    a = load_model(nz_encode(l1_a), desc)[0][addresses]
    b = load_model(nz_encode(l1_b), desc)[0][addresses].T
    return a, b


class PrepareK32(unittest.TestCase):
    def test_loaded_operands_and_slot_bounds(self):
        row, col = np.indices((64, 64))
        a_source = (10000 * row + col).astype(np.float32)
        b_source = (-100 * row + 3 * col + 7).astype(np.float32)
        modes = call_modes("Stage4_AccIssue") + call_modes("Stage5_Compute")
        self.assertEqual(modes, [(True, True), (False, True), (False, False)])
        for mode in modes:
            with self.subTest(mode=mode):
                a, b = decode_operands(a_source, b_source, mode)
                if mode == (True, True):
                    expected_a, expected_b = a_source[32:, :].T, b_source[32:, :]
                elif mode == (False, True):
                    expected_a, expected_b = a_source[:, 32:], b_source[:, 32:].T
                else:
                    expected_a, expected_b = a_source[:, :32], b_source[:, :32].T
                np.testing.assert_array_equal(a, expected_a)
                np.testing.assert_array_equal(b, expected_b)
                target, written, read = load_model(nz_encode(a_source), source_load(*mode))
                self.assertEqual(written, set(range(2048)))
                self.assertEqual(len(read), 2048)
                self.assertLess(max(read), 4096)
                self.assertTrue(np.isnan(target[2048:]).all())

    def test_model_rejects_wrong_transpose_stride_and_half(self):
        desc = source_load(True, True)
        source = nz_encode(np.arange(4096, dtype=np.float32).reshape(64, 64))
        with self.assertRaises(ValueError):
            load_model(source, replace(desc, dst_stride=2))
        good = load_model(source, desc)[0][:2048]
        for wrong in (replace(desc, m_start=0), replace(desc, src_stride=2)):
            self.assertFalse(np.array_equal(good, load_model(source, wrong)[0][:2048]))

    def test_block_inverse_full_and_tail(self):
        mode4, = call_modes("Stage4_AccIssue")
        mode51, mode52 = call_modes("Stage5_Compute")
        eye = np.eye(64, dtype=np.float32)
        # Non-symmetric leaf inverses catch confusing logical leaves with their
        # transposed physical L1 images. Boundary tails span both leaf blocks.
        for valid in (1, 10, 16, 31, 32, 33, 48, 63, 64):
            for seed in range(4):
                for scale in (0.0, 0.02, 0.2):
                    with self.subTest(valid=valid, seed=seed, scale=scale):
                        rng = np.random.default_rng(seed)
                        lower = np.tril(rng.uniform(-scale, scale, (64, 64)), -1).astype(np.float32)
                        lower[valid:, :] = 0
                        lower[:, valid:] = 0
                        left, right = np.zeros_like(lower), np.zeros_like(lower)
                        left[32:, 32:] = np.linalg.inv(eye[32:, 32:] + lower[32:, 32:])
                        right[:32, :32] = np.linalg.inv(eye[:32, :32] + lower[:32, :32])
                        dense_y = eye + left @ (-lower)
                        a4, b4 = decode_operands(left.T, -lower, mode4)
                        candidate_y = eye + a4 @ b4
                        a51, b51 = decode_operands(eye, left.T, mode51)
                        a52, b52 = decode_operands(candidate_y, right.T, mode52)
                        candidate_a = a51 @ b51 + a52 @ b52
                        dense_a = eye @ left + dense_y @ right
                        golden_a = np.linalg.solve(eye.astype(np.float64) + lower, eye)
                        np.testing.assert_allclose(candidate_y, dense_y, rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(candidate_a, dense_a, rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(candidate_a, golden_a, rtol=2e-6, atol=2e-7)
                        self.assertTrue(np.isfinite(candidate_a).all())
                        np.testing.assert_array_equal(candidate_a[valid:, valid:], eye[valid:, valid:])

    def test_fixed_mmad_and_unchanged_baseline_regions(self):
        source = k32_source(MATMUL)
        common = k32_source(ARCH / "chunk_gated_delta_rule_fwd_prepare_common.h")
        for name, value in (("kChunk64", 64), ("kVcs32", 32)):
            self.assertRegex(common, rf"constexpr uint32_t {name} = {value};")
        helper = tokens(body(source, "SolveHalfKMatmulToL0C"))
        self.assertIn("MmadL0ABToL0C<float>(l0A,l0B,l0C,kChunk64,kChunk64,kVcs32,initC,evt);", helper)
        self.assertIn("LoadData(l0A,l1A,load);AscendC::LoadData(l0B,l1B,load);", helper)
        self.assertIn("SetFlag<AscendC::HardEvent::MTE1_M>(evt);WaitFlag<AscendC::HardEvent::MTE1_M>(evt);", helper)
        default = tokens(body(source, "FillLoad2D"))
        self.assertIn("p.mStartPosition=0;p.kStartPosition=0;", default)
        for field in ("mStep", "kStep", "srcStride", "dstStride"):
            self.assertIn(f"p.{field}={field};", default)
        self.assertIn("p.ifTranspose=transpose;p.sid=0;", default)
        def baseline(path):
            return subprocess.check_output(
                ["git", "show", f"{BASELINE}:{path.relative_to(ROOT).as_posix()}"],
                cwd=ROOT, encoding="utf-8",
            )
        previous_matmul = baseline(MATMUL)
        for name in ("FillLoad2D", "LoadL1NzToL0AB", "MmadL0ABToL0C", "MatmulToL0C", "WuMatmulToL0C"):
            self.assertEqual(tokens(body(source, name)), tokens(body(previous_matmul, name)))
        previous = baseline(KERNEL)
        current = k32_source(KERNEL)
        inventory_doc = k32_source(OP / "docs/timer_flow.md")
        relative = KERNEL.relative_to(ROOT).as_posix()
        for match in re.finditer(r"GDN_TIMER_SCOPE\(timer_,\s*(\w+)\)", current):
            line = current.count("\n", 0, match.start()) + 1
            self.assertIn(f"| {match[1]} | `{relative}:{line}`", inventory_doc)
        # Outside the two call-site bodies, preserve all code tokens; inside
        # them preserve ordered event calls and the complete timer scopes.
        for name in ("Stage4_AccIssue", "Stage5_Compute"):
            before, after = body(previous, name), body(current, name)
            pattern = r"(?:WaitFlag|SetFlag)<[^>]+>\([^;]+;|GDN_TIMER_SCOPE\([^;]+;"
            self.assertEqual([tokens(s) for s in re.findall(pattern, before)],
                             [tokens(s) for s in re.findall(pattern, after)])
            previous = previous.replace(before, "")
            current = current.replace(after, "")
        self.assertEqual(tokens(previous), tokens(current))


if __name__ == "__main__":
    unittest.main(verbosity=2)

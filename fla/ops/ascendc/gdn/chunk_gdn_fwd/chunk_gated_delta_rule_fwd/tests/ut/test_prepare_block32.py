"""Active Block32 offline address/algebra/lifetime checks, not device tests.

Literal LoadData and scatter constants come from current production sources.
Independent dense FP64 solves and deliberately non-symmetric data detect
wrong quadrants, transpose/stride mistakes, and incomplete A reconstruction.
No ATK inputs, golden, or tolerance are modified.
"""

from collections import Counter
from dataclasses import replace
import re
import unittest

import numpy as np

from explore_prepare_solve import bf16_rne
from test_prepare_k32 import ARCH, KERNEL, MATMUL, OP, ROOT, Load, body, k32_source, load_model, nz_encode, tokens


COPY = ARCH / "chunk_gated_delta_rule_fwd_prepare_datacopy.h"
VF = ARCH / "chunk_gated_delta_rule_fwd_prepare_vf.h"
COMMON = ARCH / "chunk_gated_delta_rule_fwd_prepare_common.h"


def descriptors(stage):
    source = tokens(body(MATMUL.read_text(encoding="utf-8"), f"FillStage{stage}Block32Loads"))
    statement = re.compile(
        r"FillLoad2D\((load[AB]),(\d+),(\d+),(\d+),(\d+),(true|false)\);"
        r"|(load[AB])\.(mStartPosition|kStartPosition)=(\d+);"
    )
    result, pos = {}, 0
    while pos < len(source):
        match = statement.match(source, pos)
        if match is None:
            raise AssertionError("update extraction for changed descriptor syntax")
        if match[1]:
            name, m, k, src, dst, trans = match.groups()[:6]
            result[name] = Load(0, 0, int(m), int(k), int(src), int(dst), trans == "true")
        else:
            name, field, value = match.groups()[6:]
            result[name] = replace(result[name], **{
                "m_start" if field == "mStartPosition" else "k_start": int(value)})
        pos = match.end()
    if set(result) != {"loadA", "loadB"}:
        raise AssertionError("expected separate A/B loads")
    return result["loadA"], result["loadB"]


def operands(a, b, loads):
    row, col = np.indices((32, 32))
    address = (col // 8 * 2 + row // 16) * 128 + row % 16 * 8 + col % 8
    return (load_model(nz_encode(a), loads[0])[0][address],
            load_model(nz_encode(b), loads[1])[0][address].T)


def scatter_indices():
    source = tokens(body(VF.read_text(encoding="utf-8"), "AssembleBlock32AVF"))
    # Recognize the complete index expressions rather than silently testing
    # copied Python constants if the C++ addressing implementation changes.
    required = (
        "LoadAlign(lane,idxAddr+64);",
        "Duplicate(half,static_cast<uint32_t>(32),allIdx);And(half,lane,half,allIdx);",
        "Muls(diagBase,lane,static_cast<uint32_t>(64),allIdx);",
        "Add(diagBase,diagBase,half,allIdx);Add(pairBase,lane,half,allIdx);",
        "Adds(index,diagBase,static_cast<uint32_t>(col),allIdx);",
        "Adds(index,pairBase,static_cast<uint32_t>(2048+128*pair),allIdx);",
        "Adds(index,pairBase,static_cast<uint32_t>(32+128*pair),allIdx);",
    )
    if not all(part in source for part in required):
        raise AssertionError("update scatter extraction for changed C++ index expressions")
    lane = np.arange(64)
    half = lane & 32
    diag = [lane * 64 + half + col for col in range(32)]
    cross = [lane + half + 2048 + 128 * pair for pair in range(16)]
    zeros = [lane + half + 32 + 128 * pair for pair in range(16)]
    return diag, cross, zeros


def assemble(packed, c10):
    diag, cross, zeros = scatter_indices()
    a = np.full(4096, np.nan, dtype=np.float32)
    for col, indices in enumerate(diag):
        a[indices] = packed[col]
    for pair, indices in enumerate(cross):
        a[indices] = c10.reshape(-1)[pair * 64:(pair + 1) * 64]
    for indices in zeros:
        a[indices] = 0
    return a.reshape(64, 64)


class PrepareBlock32(unittest.TestCase):
    def test_actual_load_operands_and_4k_banks(self):
        row, col = np.indices((64, 64))
        full = (10000 * row + col + 7).astype(np.float32)
        other = (-100 * row + 3 * col).astype(np.float32)
        a4, b4 = operands(full, other, descriptors(4))
        np.testing.assert_array_equal(a4, full[32:, 32:].T)
        np.testing.assert_array_equal(b4, other[32:, :32])
        tight = np.arange(1024, dtype=np.float32).reshape(32, 32)
        a5, b5 = operands(tight, other, descriptors(5))
        np.testing.assert_array_equal(a5, tight)
        np.testing.assert_array_equal(b5, other[:32, :32].T)
        for matrix, desc in ((full, descriptors(4)[0]), (other, descriptors(4)[1]),
                             (tight, descriptors(5)[0]), (other, descriptors(5)[1])):
            target, written, read = load_model(nz_encode(matrix), desc)
            self.assertEqual(written, set(range(1024)))
            self.assertEqual(len(read), 1024)
            self.assertLess(max(read), matrix.size)
            self.assertTrue(np.isnan(target[1024:]).all())
        wrong = replace(descriptors(5)[0], src_stride=4)
        with self.assertRaises((IndexError, ValueError)):
            load_model(nz_encode(tight), wrong)
        for wrong in (replace(descriptors(4)[0], m_start=0),
                      replace(descriptors(4)[0], transpose=False)):
            self.assertFalse(np.array_equal(
                operands(full, other, (wrong, descriptors(4)[1]))[0], a4))

    def test_scatter_coverage_transpose_and_inplace_bf16(self):
        diag, cross, zeros = scatter_indices()
        counts = Counter(np.concatenate(diag + cross + zeros).tolist())
        self.assertEqual(counts, Counter({i: 1 for i in range(4096)}))
        rng = np.random.default_rng(49)
        packed = rng.uniform(-2, 2, (32, 64)).astype(np.float32)
        c10 = rng.uniform(-2, 2, (32, 32)).astype(np.float32)
        expected = np.block([[packed[:, :32].T, np.zeros((32, 32), np.float32)],
                             [c10, packed[:, 32:].T]])
        actual = assemble(packed, c10)
        np.testing.assert_array_equal(actual, expected)
        # Byte-level alias simulation catches compression overwriting any
        # not-yet-loaded float row. Guard the rest of the 16 KiB destination.
        raw = actual.copy().reshape(-1).view(np.uint8)
        for row in range(64):
            loaded = raw[row * 256:(row + 1) * 256].copy().view(np.float32)
            b16 = (bf16_rne(loaded).view(np.uint32) >> 16).astype(np.uint16)
            raw[row * 128:(row + 1) * 128] = b16.view(np.uint8)
        expected_bytes = (bf16_rne(expected).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
        np.testing.assert_array_equal(raw[:8192], expected_bytes.reshape(-1))
        np.testing.assert_array_equal(raw[8192:], actual.reshape(-1).view(np.uint8)[8192:])
        source = tokens(body(VF.read_text(encoding="utf-8"), "AssembleBlock32AVF"))
        self.assertEqual(source.count("__VEC_SCOPE__"), 1)
        self.assertLess(source.index("LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();"),
                        source.index("StoreCastB16<bfloat16_t>"))

    def test_full_tail_inverse_and_wu_against_k32_and_fp64(self):
        eye = np.eye(64, dtype=np.float32)
        loads4, loads5 = descriptors(4), descriptors(5)
        # Extract once; each case uses this same current-source address plan.
        diag, cross, zeros = scatter_indices()
        for valid in range(1, 65):
            for seed in range(4):
                for amplitude in (0, .02, .2):
                    rng = np.random.default_rng(seed)
                    lower = np.tril(rng.uniform(-amplitude, amplitude, (64, 64)), -1).astype(np.float32)
                    lower[valid:, :] = 0
                    lower[:, valid:] = 0
                    r00 = np.linalg.inv(np.eye(32) + lower[:32, :32]).astype(np.float32)
                    r11 = np.linalg.inv(np.eye(32) + lower[32:, 32:]).astype(np.float32)
                    left, right = np.zeros_like(lower), np.zeros_like(lower)
                    left[32:, 32:], right[:32, :32] = r11, r00
                    a4, b4 = operands(left.T, -lower, loads4)
                    x = a4 @ b4
                    # Tight32 GM NZ -> L1 preserves precisely 1024 floats.
                    a5, b5 = operands(x, right.T, loads5)
                    c10 = a5 @ b5
                    a = np.full(4096, np.nan, np.float32)
                    packed = np.concatenate((r00.T, r11.T), axis=1)
                    for col, indices in enumerate(diag):
                        a[indices] = packed[col]
                    for pair, indices in enumerate(cross):
                        a[indices] = c10.reshape(-1)[pair * 64:(pair + 1) * 64]
                    for indices in zeros:
                        a[indices] = 0
                    a = a.reshape(64, 64)
                    y = eye + left[:, 32:] @ (-lower[32:, :])
                    k32 = eye[:, 32:] @ left[32:, :] + y[:, :32] @ right[:32, :]
                    golden = np.linalg.solve(np.eye(64) + lower.astype(np.float64), np.eye(64))
                    np.testing.assert_allclose(a, golden, rtol=3e-6, atol=3e-7)
                    np.testing.assert_allclose(a, k32, rtol=3e-6, atol=3e-7)
                    np.testing.assert_array_equal(a[valid:, valid:], eye[valid:, valid:])
                    self.assertFalse(a[:valid, valid:].any())
                    self.assertFalse(a[valid:, :valid].any())
                    rhs = bf16_rne(rng.uniform(-1, 1, (64, 256)).astype(np.float32))
                    np.testing.assert_allclose(bf16_rne(a) @ rhs, bf16_rne(k32) @ rhs,
                                               rtol=3e-5, atol=3e-6)

    def test_gm_intermediate_and_workspace_bounds(self):
        copy = COPY.read_text(encoding="utf-8")
        fix = tokens(body(copy, "FixpipeBlock32ToGmNzCs"))
        for field, value in (("mSize", "kVcs32"), ("nSize", "kVcs32"),
                             ("srcStride", "kVcs32"), ("dstStride", "kVcs32*8"),
                             ("quantPre", "QuantMode_t::NoQuant"), ("isChannelSplit", "true")):
            self.assertIn(f"p.{field}={value};", fix)
        self.assertIn("DataCopyParams(1,128,0,0)", tokens(body(copy, "CopyGmBlock32NzToL1Fp32")))
        # Write/read the actual tight NZ C0=8 shape, guard 12 KiB of old slot.
        matrix = np.arange(1024, dtype=np.float32).reshape(32, 32)
        slot = np.full(4096, np.nan, np.float32)
        slot[:1024] = nz_encode(matrix)
        row, col = np.indices((32, 32))
        np.testing.assert_array_equal(slot[col // 8 * 256 + row * 8 + col % 8], matrix)
        self.assertTrue(np.isnan(slot[1024:]).all())
        occupied = set()
        for task in range(4):
            x = set(range(task * 16384, task * 16384 + 4096))
            a = set(range(task * 16384 + 8192, (task + 1) * 16384))
            self.assertFalse(occupied & (x | a))
            self.assertFalse(x & a)
            occupied |= x | a
        self.assertLess(max(occupied), 65536)
        common = COMMON.read_text(encoding="utf-8")
        self.assertIn("taskIdx * static_cast<int64_t>(kWsYBytes / sizeof(bfloat16_t)) + kChunk64 * kChunk64;", common)

    def test_active_entry_and_lifetime_contract(self):
        kernel = KERNEL.read_text(encoding="utf-8")
        aic, aiv = body(kernel, "ProcessAic"), body(kernel, "ProcessAiv")
        self.assertNotIn("Stage4_PrefillI", kernel)
        self.assertNotIn("SolveHalfKMatmulToL0C<", kernel)
        for stage in (4, 5):
            helper = tokens(body(MATMUL.read_text(encoding="utf-8"), f"Stage{stage}Block32MatmulToL0C"))
            self.assertIn("kVcs32,kVcs32,kVcs32,true,evt", helper)
            name = "Stage4_AccIssue" if stage == 4 else "Stage5_Compute"
            self.assertIn(f"Stage{stage}Block32MatmulToL0C(", body(kernel, name))
        self.assertLess(aiv.index("Stage3_AivOne("), aiv.index("Stage5_AssembleAivOne("))
        self.assertLess(aiv.index("Stage5_AssembleAivOne("), aiv.index("DrainPackWuToGm("))
        self.assertLess(aiv.index("DrainPackWuToGm("), aiv.index("MTE3_MTE2"))
        self.assertLess(aiv.index("MTE3_MTE2"), aiv.index("Stage6_AivOne("))
        self.assertLess(aic.index("WaitAivStage6Done("), aic.index("LoadAssembledAToL1("))
        self.assertLess(aic.index("LoadAssembledAToL1("), aic.index("Stage7_AicOne("))
        common = COMMON.read_text(encoding="utf-8")
        self.assertIn("PIPE_MTE2>", body(common, "WaitAivStage6Done"))
        self.assertIn("PIPE_FIX>", body(common, "NotifyAivStage5Done"))
        self.assertIn("PIPE_V>", body(common, "WaitAicStage5Done"))
        assemble_body = tokens(body(kernel, "Stage5_AssembleAivOne"))
        self.assertIn("if(outputA==0||chunk.M<chunkSize)", assemble_body)
        self.assertIn("static_cast<uint32_t>(chunk.M*chunkSize)", assemble_body)
        # Prove why interleaving task0 Stage6 with task2 A assembly is illegal.
        leaf1, a1 = set(range(58, 66)), set(range(74, 90))
        for v_end in (80, 96):
            s6v0 = set(range(64, v_end))
            self.assertTrue(s6v0 & leaf1)
            self.assertTrue(s6v0 & a1)
        # Operations whose numeric algorithm is outside this optimization.
        for name in ("Stage1_OneTask", "Stage2_AicOne", "Stage3_AivOne",
                     "Stage6_AivOne", "Stage7_AicOne", "DrainPackWuToGm"):
            self.assertEqual(tokens(body(kernel, name)), tokens(body(k32_source(KERNEL), name)))
        inventory = (OP / "docs/timer_flow.md").read_text(encoding="utf-8")
        relative = KERNEL.relative_to(ROOT).as_posix()
        for match in re.finditer(r"GDN_TIMER_SCOPE\(timer_,\s*(\w+)\)", kernel):
            line = kernel.count("\n", 0, match.start()) + 1
            self.assertIn(f"| {match[1]} | `{relative}:{line}`", inventory)

if __name__ == "__main__":
    unittest.main(verbosity=2)

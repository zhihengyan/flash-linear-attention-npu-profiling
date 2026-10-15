"""Active N32 layout/algebra/source checks, without CANN or an NPU.

Literal descriptors come from production C++; independent NZ addressing and
dense FP64 solves check the interpretation. This is not instruction emulation
or evidence of device rounding, synchronization, precision or performance.
"""

from dataclasses import replace
from functools import lru_cache
import re
import unittest

import numpy as np

from test_prepare_k32 import (
    ARCH, KERNEL, MATMUL, OP, ROOT, Load, body, k32_source, load_model,
    nz_encode, source_load, tokens,
)


COPY = ARCH / "chunk_gated_delta_rule_fwd_prepare_datacopy.h"


def literal_loads(source):
    """Accept only the two literal FillLoad2D calls and optional row starts."""
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    loads = {}
    pattern = r"FillLoad2D\((load[AB]), (\d+), (\d+), (\d+), (\d+), (true|false)\);"
    for match in re.finditer(pattern, source):
        name, m, k, src, dst, trans = match.groups()
        if name in loads:
            raise AssertionError("duplicate descriptor")
        loads[name] = Load(0, 0, int(m), int(k), int(src), int(dst), trans == "true")
    source = re.sub(pattern, "", source)
    for name, value in re.findall(r"(load[AB])\.mStartPosition = (\d+);", source):
        loads[name] = replace(loads[name], m_start=int(value))
    source = re.sub(r"load[AB]\.mStartPosition = \d+;", "", source)
    if source.strip() or set(loads) != {"loadA", "loadB"}:
        raise AssertionError("update extraction for changed production descriptor structure")
    return loads["loadA"], loads["loadB"]


@lru_cache(maxsize=None)
def descriptors(stage):
    source = MATMUL.read_text(encoding="utf-8")
    if stage == "prefill":
        return literal_loads(body(source, "FillN32PrefillLoads"))
    branches = re.fullmatch(
        r"\s*if constexpr \(TransposeInputs\) \{(.*?)\} else \{(.*?)\}\s*",
        body(source, "FillN32SolveLoads"), re.S,
    )
    if branches is None or stage not in ("acc", "update"):
        raise AssertionError("unexpected N32 solve descriptor branches")
    return literal_loads(branches.group(1 if stage == "acc" else 2))


def nz_decode(storage, rows, cols, c0=8):
    # Decode by the MMAD/Fixpipe consumer's dimensions, not descriptor steps.
    row, col = np.indices((rows, cols))
    return storage[(col // c0 * rows + row) * c0 + col % c0]


def c_encode(matrix):
    rows, cols = matrix.shape
    return matrix.reshape(rows // 16, 16, cols // 16, 16).transpose(2, 0, 1, 3).reshape(-1)


def operands(a, b, stage):
    ad, bd = descriptors(stage)
    k = 64 if stage == "prefill" else 32
    av = load_model(nz_encode(a), ad)[0]
    bv = load_model(nz_encode(b), bd)[0]
    return nz_decode(av, 64, k), nz_decode(bv, 32, k).T


def bf16(value):
    """Software round-to-nearest-even BF16, returning FP32 values."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


class PrepareN32(unittest.TestCase):
    def test_actual_operands_and_bank_bounds(self):
        row, col = np.indices((64, 64))
        a = (row * 10000 + col + 1).astype(np.float32)
        b = (-row * 100 + col * 3 + 7).astype(np.float32)
        for stage in ("prefill", "acc", "update"):
            a_source = a[:, :32].copy() if stage == "update" else a
            expected = {
                "prefill": (a, b[:, :32]),
                "acc": (a[32:, :].T, b[32:, :32]),
                "update": (a[:, :32], b[:32, :32].T),
            }[stage]
            with self.subTest(stage=stage):
                actual = operands(a_source, b, stage)
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)
                sizes = (4096, 2048) if stage == "prefill" else (2048, 1024)
                for matrix, desc, size in zip((a_source, b), descriptors(stage), sizes):
                    loaded, written, read = load_model(nz_encode(matrix), desc)
                    self.assertEqual(written, set(range(size)))
                    self.assertEqual(len(read), size)
                    self.assertLess(max(read), matrix.size)
                    self.assertTrue(np.isnan(loaded[size:]).all())

    def test_wrong_stride_half_and_unwritten_y_are_detected(self):
        source = nz_encode(np.arange(4096, dtype=np.float32).reshape(64, 64))
        ad, bd = descriptors("acc")
        good = load_model(source, bd)[0][:1024]
        for wrong in (replace(bd, m_start=0), replace(bd, src_stride=2), replace(bd, transpose=False)):
            candidate = load_model(source, wrong)[0][:1024]
            self.assertFalse(np.array_equal(good, candidate))
        with self.assertRaises(ValueError):
            load_model(source, replace(ad, dst_stride=2))
        y = source[:2048]
        with self.assertRaises(ValueError):
            load_model(y, replace(descriptors("update")[0], k_step=8))
        wrong_b = load_model(source, replace(descriptors("prefill")[1], dst_stride=4))[0]
        self.assertTrue(np.isnan(nz_decode(wrong_b, 32, 64)).any())

    def test_rectangular_y_transfer_and_full_c_prefix(self):
        source = COPY.read_text(encoding="utf-8")
        fix = tokens(body(source, "FixpipeN32YToGmNzCs"))
        for field in ("nSize=kVcs32", "mSize=kChunk64", "srcStride=kChunk64",
                      "dstStride=kChunk64*8", "quantPre=QuantMode_t::NoQuant",
                      "isChannelSplit=true", "unitFlag=0"):
            self.assertIn("p." + field + ";", fix)
        self.assertIn("AscendC::Fixpipe<float,float,CFG_NZ_L1>(gmTensor,l0CTensor,p);", fix)
        self.assertIn("DataCopyParams(1,256,0,0)", tokens(body(source, "CopyGmN32YToL1Fp32")))
        self.assertEqual(256 * 32, 64 * 32 * 4)
        full = np.arange(4096, dtype=np.float32).reshape(64, 64)
        c = c_encode(full)
        np.testing.assert_array_equal(nz_decode(c[:2048], 64, 32, 16), full[:, :32])
        original_right = c[2048:].copy()
        c[:2048] += 10000
        np.testing.assert_array_equal(c[2048:], original_right)
        np.testing.assert_array_equal(nz_decode(c, 64, 64, 16)[:, 32:], full[:, 32:])
        # Four original 16KiB slots/core; Fixpipe/readback touch only 8KiB.
        gm = np.full(4 * 4096, np.nan, dtype=np.float32)
        for task in range(4):
            gm[task * 4096:task * 4096 + 2048] = nz_encode(full[:, :32] + task)
        for task in range(4):
            slot = gm[task * 4096:(task + 1) * 4096]
            self.assertTrue(np.isnan(slot[2048:]).all())
            np.testing.assert_array_equal(nz_decode(slot[:2048].copy(), 64, 32), full[:, :32] + task)

    def test_all_tails_inverse_and_bf16_wu(self):
        eye = np.eye(64, dtype=np.float32)
        pa, pb = operands(eye, eye, "prefill")
        prefill = pa @ pb
        # Original full-A initialization helper is retained verbatim in C++.
        init_desc = source_load(False, True)
        for valid in range(1, 65):
            for seed in range(4):
                for amplitude in (0.0, 0.02, 0.2):
                    with self.subTest(valid=valid, seed=seed, amplitude=amplitude):
                        rng = np.random.default_rng(seed)
                        lower = np.tril(rng.uniform(-amplitude, amplitude, (64, 64)), -1).astype(np.float32)
                        lower[valid:, :] = 0
                        lower[:, valid:] = 0
                        left, right = np.zeros_like(lower), np.zeros_like(lower)
                        left[32:, 32:] = np.linalg.inv(np.eye(32) + lower[32:, 32:]).astype(np.float32)
                        right[:32, :32] = np.linalg.inv(np.eye(32) + lower[:32, :32]).astype(np.float32)
                        acc_a, acc_b = operands(left.T, -lower, "acc")
                        y = prefill + acc_a @ acc_b
                        # Explicit L0C(C0=16) -> GM/L1(C0=8) -> LoadData path.
                        y = nz_decode(c_encode(y), 64, 32, 16)
                        gm = np.full(4096, np.nan, dtype=np.float32)
                        gm[:2048] = nz_encode(y)
                        l1_y = nz_decode(gm[:2048].copy(), 64, 32)
                        init_a = nz_decode(load_model(nz_encode(eye), init_desc)[0], 64, 32)
                        init_b = nz_decode(load_model(nz_encode(left.T), init_desc)[0], 64, 32).T
                        c = c_encode(init_a @ init_b)
                        high_half = c[2048:].copy()
                        ua, ub = operands(l1_y, right.T, "update")
                        c[:2048] += c_encode(ua @ ub)
                        np.testing.assert_array_equal(c[2048:], high_half)
                        actual = nz_decode(c, 64, 64, 16)
                        k32 = eye @ left + (eye + left @ (-lower)) @ right
                        golden = np.linalg.solve(np.eye(64) + lower.astype(np.float64), np.eye(64))
                        np.testing.assert_allclose(y, (eye + left @ (-lower))[:, :32], rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(actual, k32, rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(actual, golden, rtol=3e-6, atol=3e-7)
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_array_equal(actual[valid:, valid:], eye[valid:, valid:])
                        np.testing.assert_array_equal(actual[:valid, valid:], np.zeros((valid, 64 - valid)))
                        np.testing.assert_array_equal(actual[valid:, :valid], np.zeros((64 - valid, valid)))
                        # W(K128) and U(V128/V256): software BF16 models only.
                        for width in (128, 256):
                            rhs = bf16(rng.uniform(-0.25, 0.25, (64, width)))
                            candidate = bf16(bf16(actual) @ rhs)
                            reference = bf16(bf16(golden) @ rhs)
                            np.testing.assert_allclose(candidate, reference, rtol=0.008, atol=0.004)

    def test_production_entry_and_original_event_sequences(self):
        current = KERNEL.read_text(encoding="utf-8")
        previous = k32_source(KERNEL)
        changed = ("Stage4_PrefillI", "Stage4_AccIssue", "Stage4_Dump", "Stage5_Compute")
        event = r"(?:WaitFlag|SetFlag)<[^>]+>\([^;]+;|GDN_TIMER_SCOPE\([^;]+;"
        for name in changed:
            before, after = body(previous, name), body(current, name)
            self.assertEqual([tokens(s) for s in re.findall(event, before)],
                             [tokens(s) for s in re.findall(event, after)])
            previous, current = previous.replace(before, ""), current.replace(after, "")
        self.assertEqual(tokens(previous), tokens(current))
        kernel = KERNEL.read_text(encoding="utf-8")
        self.assertIn("LoadN32PrefillToL0AB(l1I,l0Af[0],l0Bf[0],0);", tokens(body(kernel, changed[0])))
        self.assertIn("bt,kVcs32,bt,true,bank", tokens(body(kernel, changed[0])))
        self.assertIn("N32SolveMatmulToL0C<true>", body(kernel, changed[1]))
        self.assertIn("FixpipeN32YToGmNzCs", body(kernel, changed[2]))
        self.assertIn("CopyGmN32YToL1Fp32", body(kernel, changed[2]))
        compute = tokens(body(kernel, changed[3]))
        self.assertIn("SolveHalfKMatmulToL0C<false,true>", compute)
        self.assertIn("l0CTask[taskIdx],true,bank);", compute)
        self.assertIn("N32SolveMatmulToL0C<false>", compute)
        matmul = MATMUL.read_text(encoding="utf-8")
        for name in ("FillLoad2D", "LoadL1NzToL0AB", "MmadL0ABToL0C", "MatmulToL0C",
                     "FillSolveHalfKLoad", "SolveHalfKMatmulToL0C", "WuMatmulToL0C"):
            self.assertEqual(tokens(body(matmul, name)), tokens(body(k32_source(MATMUL), name)))
        update = tokens(body(matmul, "N32SolveMatmulToL0C"))
        self.assertIn("MmadL0ABToL0C<float>(l0A,l0B,l0C,kChunk64,kVcs32,kVcs32,false,evt);", update)
        for name in ("LoadN32PrefillToL0AB", "N32SolveMatmulToL0C"):
            helper = tokens(body(matmul, name))
            self.assertIn("SetFlag<AscendC::HardEvent::MTE1_M>(evt);WaitFlag<AscendC::HardEvent::MTE1_M>(evt);", helper)
            self.assertIn("AscendC::LoadData(l0B,", helper)
        for path in (ARCH / "chunk_gated_delta_rule_fwd_prepare_common.h",
                     ARCH / "chunk_gated_delta_rule_fwd_prepare_vf.h",
                     ARCH.parent / "chunk_gated_delta_rule_fwd_prepare.cpp",
                     OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp"):
            self.assertEqual(tokens(path.read_text(encoding="utf-8")), tokens(k32_source(path)))
        entry = (ARCH.parent / "chunk_gated_delta_rule_fwd_prepare.cpp").read_text(encoding="utf-8")
        self.assertIn("op.Process();", entry)
        host = (OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp").read_text(encoding="utf-8")
        self.assertEqual(host.count("return ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl("), 2)
        self.assertIn("aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize(", host)
        self.assertIn("auto prepareResult = l0op::ChunkGatedDeltaRuleFwdPrepare(", host)

    def test_active_timer_inventory_lines(self):
        source = KERNEL.read_text(encoding="utf-8")
        inventory = (OP / "docs/timer_flow.md").read_text(encoding="utf-8")
        relative = KERNEL.relative_to(ROOT).as_posix()
        for match in re.finditer(r"GDN_TIMER_SCOPE\(timer_,\s*(\w+)\)", source):
            line = source.count("\n", 0, match.start()) + 1
            self.assertIn(f"| {match[1]} | `{relative}:{line}`", inventory)


if __name__ == "__main__":
    unittest.main(verbosity=2)

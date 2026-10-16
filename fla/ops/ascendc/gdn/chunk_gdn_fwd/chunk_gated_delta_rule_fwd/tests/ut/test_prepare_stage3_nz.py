"""Stage3 address/bit-copy checks against independent dense NZ8 operands.

Extract production gather expressions and literal DataCopy descriptors. These
checks cover layout and source integration, not CANN codegen, races or timing.
Run directly or with unittest discovery; NumPy and local K32 git history needed.
"""

import ast
from collections import Counter
import re
import unittest

import numpy as np

from test_prepare_k32 import ARCH, KERNEL, MATMUL, OP, ROOT, body, k32_source, nz_encode, tokens

VF = ARCH / "chunk_gated_delta_rule_fwd_prepare_vf.h"
COPY = ARCH / "chunk_gated_delta_rule_fwd_prepare_datacopy.h"
COMMON = ARCH / "chunk_gated_delta_rule_fwd_prepare_common.h"


def arithmetic(expression, **values):
    """Only integer arithmetic from the narrowly checked C++ address expressions."""
    node = ast.parse(expression.strip(), mode="eval").body

    def visit(item):
        if isinstance(item, ast.Constant) and isinstance(item.value, int):
            return item.value
        if isinstance(item, ast.Name):
            return values[item.id]
        if isinstance(item, ast.BinOp):
            left, right = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Mult):
                return left * right
            if isinstance(item.op, ast.Div):
                return left // right
        raise AssertionError(f"unsupported source expression: {expression}")

    return visit(node)


def gather_plan(rows, groups):
    source = tokens(body(VF.read_text(encoding="utf-8"), "PackStage3Nz8VF"))
    loops = re.findall(r"for\(uint16_t(\w+)=0;\1<([^;]+);\+\+\1\)", source)
    if [name for name, _ in loops] != ["group", "fc", "rb"]:
        raise AssertionError("update extraction for changed gather loop structure")
    if "Gather(value,srcAddr+srcOff,index,all);StoreAlign(dstAddr+dstOff,value,all);" not in source:
        raise AssertionError("gather/store operands changed")
    cols = re.search(r"constexpruint32_tcols=([^;]+);", source).group(1)
    expressions = [re.search(r"constuint32_t" + name + r"=([^;]+);", source).group(1)
                   for name in ("srcOff", "dstOff")]
    values = dict(Rows=rows, Groups=groups, cols=arithmetic(cols, Groups=groups))
    plan = []
    for group in range(arithmetic(loops[0][1], **values)):
        for fc in range(arithmetic(loops[1][1], **values)):
            for rb in range(arithmetic(loops[2][1], **values)):
                context = dict(values, group=group, fc=fc, rb=rb)
                plan.append(tuple(arithmetic(expr, **context) for expr in expressions))
    return plan


def gather_indices():
    source = tokens(body(VF.read_text(encoding="utf-8"), "FillS3Nz8GatherIndexVF"))
    match = re.fullmatch(
        r"__ubuf__int32_t\*dst=\(__ubuf__int32_t\*\)indices.GetPhyAddr\(\);"
        r"__VEC_SCOPE__\{MaskRegall=CreateMask<int32_t,MaskPattern::ALL>\(\);"
        r"RegTensor<int32_t>lane,row,offset;Arange\(lane,(\d+)\);"
        r"ShiftRights\(row,lane,int16_t\((\d+)\),all\);"
        r"Muls\(row,row,int32_t\((\d+)\),all\);Add\(offset,row,lane,all\);"
        r"StoreAlign\(dst,offset,all\);\}", source,
    )
    if match is None:
        raise AssertionError("update extraction for changed index producer")
    start, shift, scale = map(int, match.groups())
    lanes = np.arange(start, start + 64, dtype=np.int32)
    return (lanes >> shift) * scale + lanes


def packed_model(matrix, groups, indices=None):
    indices = gather_indices() if indices is None else indices
    src = matrix.reshape(-1)
    dst = np.full(src.size, 0xDEADBEEF, dtype=np.uint32)
    written, read = [], []
    for source, target in gather_plan(matrix.shape[0], groups):
        addresses = source + indices
        if source % 8 or target % 8 or addresses.min() < 0 or addresses.max() >= src.size:
            raise ValueError("invalid gather base or read bounds")
        dst[target:target + 64] = src[addresses]
        written.extend(range(target, target + 64))
        read.extend(addresses.tolist())
    if sorted(written) != list(range(src.size)) or sorted(read) != list(range(src.size)):
        raise ValueError("gather read/write coverage is not a permutation")
    return dst


def uploads(name):
    source = tokens(body(COPY.read_text(encoding="utf-8"), name))
    matches = list(re.finditer(
        r"DataCopy\((\w+)(?:\[(\d+)\])?,packedNz(?:\[(\d+)\])?,"
        r"DataCopyParams\((\d+),(\d+),(\d+),(\d+)\)\);", source,
    ))
    if "".join(m.group() for m in matches) != source:
        raise AssertionError("expected literal upload descriptors only")
    return [(m[1], *map(lambda x: int(x or 0), m.groups()[1:])) for m in matches]


def copy_blocks(destination, source, dst_off, src_off, count, length, src_gap, dst_gap):
    # DataCopyParams are 32-byte blocks; operands here are uint32 bit patterns.
    touched = []
    for block in range(count):
        dst = dst_off + block * (length + dst_gap) * 8
        src = src_off + block * (length + src_gap) * 8
        if dst + length * 8 > len(destination) or src + length * 8 > len(source):
            raise ValueError("DataCopy out of bounds")
        destination[dst:dst + length * 8] = source[src:src + length * 8]
        touched.extend(range(dst, dst + length * 8))
    if len(touched) != len(set(touched)):
        raise ValueError("DataCopy overlaps its destination")
    return set(touched)


def uploaded_operands(neg_l, leaves, leaf_override=None):
    # Nonzero guard words surround the real slots; Stage0 zeros only the slots.
    targets = {name: np.full(4112, 0xBAADF00D, dtype=np.uint32)
               for name in ("l1", "l1LeafRight", "l1LeafLeft")}
    for target in targets.values():
        target[8:-8] = 0
    written = {}
    for name, source in (("UploadStage3NegLNzToL1", packed_model(neg_l, 1)),
                         ("UploadStage3LeafNzToL1", packed_model(leaves, 2))):
        descriptors = leaf_override if leaf_override is not None and "Leaf" in name else uploads(name)
        for target, dst, src, *params in descriptors:
            written[target] = copy_blocks(targets[target][8:-8], source, dst, src, *params)
    return targets, written


class PrepareStage3Nz(unittest.TestCase):
    def test_bitwise_layout_and_leaf_quadrants_all_tails(self):
        special = np.array([0, 0x80000000, 0x7FC12345, 0xFFCABCDE, 0x7F800000,
                            0xFF800000, 1, 0x007FFFFF], dtype=np.uint32)
        for seed in (17, 29):
            rng = np.random.default_rng(seed)
            for valid in range(1, 65):
                with self.subTest(seed=seed, valid=valid):
                    neg_l = rng.integers(0, 2**32, (64, 64), dtype=np.uint32)
                    # Matrix-level tail padding, with adversarial bits in live rows.
                    neg_l[valid:, :] = 0
                    neg_l[:, valid:] = 0
                    neg_l.flat[:min(8, valid)] = special[:min(8, valid)]
                    leaves = rng.integers(0, 2**32, (32, 64), dtype=np.uint32)
                    leaves[0, :8] = special
                    expected_right = np.zeros((64, 64), dtype=np.uint32)
                    expected_left = expected_right.copy()
                    expected_right[:32, :32] = leaves[:, :32]
                    expected_left[32:, 32:] = leaves[:, 32:]
                    targets, written = uploaded_operands(neg_l, leaves)
                    for name, expected in (("l1", neg_l), ("l1LeafRight", expected_right),
                                           ("l1LeafLeft", expected_left)):
                        np.testing.assert_array_equal(targets[name][8:-8], nz_encode(expected))
                        self.assertTrue(np.all(targets[name][:8] == 0xBAADF00D))
                        self.assertTrue(np.all(targets[name][-8:] == 0xBAADF00D))
                    self.assertEqual(len(written["l1"]), 4096)
                    self.assertEqual(len(written["l1LeafRight"]), 1024)
                    self.assertEqual(len(written["l1LeafLeft"]), 1024)

    def test_gather_bounds_and_models_reject_wrong_units_and_quadrant_stride(self):
        np.testing.assert_array_equal(gather_indices(), np.arange(64) // 8 * 64 + np.arange(64) % 8)
        matrix = np.arange(4096, dtype=np.uint32).reshape(64, 64)
        leaves = matrix[:32, :]
        with self.assertRaisesRegex(ValueError, "bounds"):
            packed_model(matrix, 1, gather_indices() * 4)  # byte indices would be wrong
        wrong = [list(item) for item in uploads("UploadStage3LeafNzToL1")]
        wrong[1][1] -= 256  # BR origin loses one 32-row strip
        targets, _ = uploaded_operands(matrix, leaves, wrong)
        expected = np.zeros((64, 64), dtype=np.uint32)
        expected[32:, 32:] = leaves[:, 32:]
        self.assertFalse(np.array_equal(targets["l1LeafLeft"][8:-8], nz_encode(expected)))
        wrong[1][1] += 256
        wrong[0][-1] = 0  # compact output has to skip the unpopulated L1 rows
        targets, _ = uploaded_operands(matrix, leaves, wrong)
        expected[:] = 0
        expected[:32, :32] = leaves[:, :32]
        self.assertFalse(np.array_equal(targets["l1LeafRight"][8:-8], nz_encode(expected)))

    def test_unaffected_math_scheduler_host_and_normal_profile_entry(self):
        # Whole original helpers remain identical; new helpers are additions only.
        additions = {VF: ("FillS3Nz8GatherIndexVF", "PackStage3Nz8VF"),
                     COPY: ("UploadStage3NegLNzToL1", "UploadStage3LeafNzToL1")}
        for path, names in additions.items():
            current = path.read_text(encoding="utf-8")
            for name in names:
                start = current.index("__aicore__ inline void " + name + "(")
                end = start + len(current[start:].split("{", 1)[0]) + len(body(current, name)) + 2
                if name == "PackStage3Nz8VF":
                    start = current.rindex("template <", 0, start)
                current = current[:start] + current[end:]
            self.assertEqual(tokens(current), tokens(k32_source(path)), path.name)
        # Every existing method except the three explicit wiring changes is frozen.
        source = KERNEL.read_text(encoding="utf-8")
        original = k32_source(KERNEL)
        for name in re.findall(r"inline void (\w+)\(", original):
            if name not in ("InitOnChip", "Stage0_GenerateResidentAux", "Stage3_AivOne"):
                self.assertEqual(tokens(body(source, name)), tokens(body(original, name)), name)
        paths = [MATMUL, OP.parent / "chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp",
                 OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp"]
        for path in paths:
            self.assertEqual(tokens(path.read_text(encoding="utf-8")), tokens(k32_source(path)), path.name)

    def test_ub_alias_lifetimes_and_shared_kkt_are_preserved(self):
        common = COMMON.read_text(encoding="utf-8")
        pairs = {}
        for name in ("kUbS3UploadNz", "kUbS3LFull", "kUbS3LPacked", "kUbS3ResVcs"):
            values = re.search(name + r"\[2\] = \{([^}]+)\}", common).group(1).split(",")
            pairs[name] = [arithmetic(value, kPrepareKb=1024) for value in values]
        resident = re.search(r"kUbS3UploadIdx = (\d+);", common).group(1)
        self.assertEqual(int(resident), 0)
        self.assertLessEqual(int(resident) + 256, 512)
        self.assertEqual(pairs["kUbS3UploadNz"], [90 * 1024, 114 * 1024])
        for db, start in enumerate(pairs["kUbS3UploadNz"]):
            end = start + (4096 + 2048) * 4
            self.assertLessEqual(end, 140 * 1024)  # never overwrite shared-owner KKT
            for name, size in (("kUbS3LFull", 16384), ("kUbS3LPacked", 8192),
                               ("kUbS3ResVcs", 8192)):
                for other in pairs[name]:
                    self.assertFalse(start < other + size and other < end)
            if db == 0:
                self.assertLessEqual(end, pairs["kUbS3UploadNz"][1])
        source = tokens(body(KERNEL.read_text(encoding="utf-8"), "ProcessAiv"))
        # This order applies to every G, V, nThis, and consecutive pack through
        # the original loops (also compared in full to K32 in the preceding test).
        anchors = ["Stage1_OneTask(", "SetFlag<HardEvent::MTE3_V>(kHeUb0);",
                   "Stage3_PrepareGate(", "Stage3_AivOne(",
                   "WaitFlag<HardEvent::MTE3_MTE2>(kHeUb0);", "Stage6_AivOne("]
        pos = -1
        for anchor in anchors:
            pos = source.index(anchor, pos + 1)

    def test_stage3_event_publication_and_live_buffers(self):
        source = KERNEL.read_text(encoding="utf-8")
        stage0 = tokens(body(source, "Stage0_GenerateResidentAux"))
        self.assertIn("FillS3Nz8GatherIndexVF(ubS3UploadIdx);FillVcsIdentity(ubIVcs,ubVcsIdx);", stage0)
        s3 = tokens(body(source, "Stage3_AivOne"))
        original = tokens(body(k32_source(KERNEL), "Stage3_AivOne"))
        event = r"(?:SetFlag|WaitFlag)<HardEvent::\w+>\(\w+\);"
        added = Counter(re.findall(event, s3)) - Counter(re.findall(event, original))
        self.assertEqual(added, Counter({"SetFlag<HardEvent::V_MTE3>(kHeUb0);": 1,
                                        "WaitFlag<HardEvent::V_MTE3>(kHeUb0);": 1}))
        full = "PackStage3Nz8VF<64,1>(ubLFull[db],ubS3UploadNz[db],ubS3UploadIdx);"
        leaf = "PackStage3Nz8VF<32,2>(ubResVcs[db],leafNz,ubS3UploadIdx);"
        order = ["WaitCubeKktDone(taskIdx);", "MulKktGateVF(", "PackDiagLeavesFromUb(",
                 "WaitFlag<HardEvent::MTE3_V>(kHeS3Pack);", full,
                 "WaitFlag<HardEvent::V_MTE3>(kHeUb0);", "UploadStage3NegLNzToL1(",
                 "MulReduceScatterVF32(", "leafNz=ubS3UploadNz[db][kS3NegLNzElems];", leaf,
                 "WaitFlag<HardEvent::V_MTE3>(kHeUb0);", "UploadStage3LeafNzToL1(",
                 "NotifyAicStage3Done(taskIdx);"]
        pos = -1
        for anchor in order:
            pos = s3.index(anchor, pos + 1)
        pack = tokens(body(VF.read_text(encoding="utf-8"), "PackStage3Nz8VF"))
        self.assertLess(pack.index("LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();"),
                        pack.index("LoadAlign(index,idxAddr);"))
        init = tokens(body(source, "InitOnChip"))
        self.assertIn("ubS3UploadNz[db].SetSize(kS3UploadNzElems);", init)

    def test_upload_call_count_bytes_and_timer_inventory(self):
        full = uploads("UploadStage3NegLNzToL1")
        leaf = uploads("UploadStage3LeafNzToL1")
        self.assertEqual(len(full) + len(leaf), 3)
        self.assertEqual(sum(count * length * 32 for _, _, _, count, length, _, _ in full + leaf), 24576)
        self.assertEqual(len(gather_plan(64, 1)) + len(gather_plan(32, 2)), 96)
        source = KERNEL.read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"GDN_TIMER_SCOPE\([^;]+", source),
                         re.findall(r"GDN_TIMER_SCOPE\([^;]+", k32_source(KERNEL)))
        inventory = (OP / "docs/timer_flow.md").read_text(encoding="utf-8")
        relative = KERNEL.relative_to(ROOT).as_posix()
        rows = [int(number) for number in re.findall(re.escape(relative) + r":(\d+)", inventory)]
        actual = [i for i, line in enumerate(source.splitlines(), 1) if "GDN_TIMER_SCOPE(" in line]
        self.assertEqual(rows, actual)


if __name__ == "__main__":
    unittest.main()

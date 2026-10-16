"""Offline H handoff ordering/ownership checks, not an AscendC emulator.

Read the candidate's actual writeback branches and UB constants. Check the
resulting partial order for conflicting accesses, including unsafe mutations.
No device timing, CANN compilation or numerical accuracy is inferred.
"""

from collections import defaultdict
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[8]
OP = Path(__file__).resolve().parents[2]
H = OP.parent / "chunk_fwd_h"
VEC = H / "op_kernel/arch35/chunk_fwd_h_vec.h"
POLICY = H / "op_kernel/chunk_fwd_h_policy.h"
BASELINE = "d0cb628df7c8c23c0342392dfaf4ec910aba9947"


def clean(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S))


def braced(source, start):
    begin = source.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return begin, end


def body(source, name):
    begin, end = braced(source, source.index(" " + name + "("))
    return source[begin + 1:end - 1]


def writeback(source):
    code = body(source, "ConsumeStage1Head")
    begin = code.index("            const bool earlyRightReady")
    end = code.index("            if constexpr (!CompilePolicy::STATE_FP32)",
                     code.index("AscendC::DataCopy(vNew", begin))
    # That first occurrence belongs to the fallback right branch; its closing
    # braces precede the unchanged state/H0 block with the same condition.
    end = code.index("            if constexpr (!CompilePolicy::STATE_FP32)", end + 1)
    return code[begin:end]


def specialize(code, *, early, write_right, fp32):
    """Only the three boolean branches in the writeback region are supported."""
    choices = {"WRITE_RIGHT": write_right, "earlyRightReady": early,
               "!earlyRightReady": not early, "!CompilePolicy::STATE_FP32": not fp32}
    match = re.search(r"\bif(?: constexpr)?\s*\(([^()]*)\)\s*\{", code)
    if match is None:
        return code
    begin, end = braced(code, match.start())
    condition = match[1].strip()
    if condition not in choices:
        raise AssertionError("unrecognized writeback condition: " + condition)
    chosen = specialize(code[begin + 1:end - 1], early=early,
                        write_right=write_right, fp32=fp32) if choices[condition] else ""
    return code[:match.start()] + chosen + specialize(
        code[end:], early=early, write_right=write_right, fp32=fp32)


def operations(code):
    pattern = r"DataCopy\((rightGm|vNew)\[|FwdHAivLocalFlag\((FWD_H_RIGHT_READY_FLAG),"
    return ["ready" if item[2] else item[1] for item in re.finditer(pattern, code)]


def early_policy(source, fp32, scalar_g, heads):
    guard = clean(body(source, "UseEarlyRightReady"))
    expected = ("ifconstexpr(CompilePolicy::STATE_FP32&&"
                "CompilePolicy::GATE_MODE==FwdHGateMode::SCALAR_G){"
                "returnunit.headRound.activeHeadCount<=FWD_H_AIV_COUNT;}returnfalse;")
    if guard != expected:
        raise AssertionError("review the ownership model for a changed policy")
    count = int(re.search(r"FWD_H_AIV_COUNT = (\d+);", POLICY.read_text(encoding="utf-8"))[1])
    return fp32 and scalar_g and heads <= count


class Order:
    def __init__(self):
        self.edges = defaultdict(set)

    def chain(self, *nodes):
        for first, second in zip(nodes, nodes[1:]):
            self.edges[first].add(second)

    def before(self, first, second):
        pending, seen = [first], set()
        while pending:
            node = pending.pop()
            if node == second:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self.edges[node])
        return False


def handoff_order(per_head_ops, early, *, ready_before_copy=False, free_before_vnew=False):
    """MTE3 completion order + the cross-core read/compute dependency chain.

    VF of a subsequent local head can issue before its MTE3 completes. Its
    right aliases D, so every right DMA read must precede the first D write.
    The public V_new work banks must remain owned until their DMA reads end.
    """
    order, mte3, rights, vnews = Order(), [], [], []
    for head, ops in enumerate(per_head_ops):
        names = [f"{head}:{op}" for op in ops]
        if ready_before_copy and f"{head}:ready" in names:
            names.remove(f"{head}:ready")
            names.insert(0, f"{head}:ready")
        free = f"{head}:work_free"
        if free_before_vnew:
            names.insert(names.index(f"{head}:vNew"), free)
        else:
            names.append(free)
        mte3.extend(names)
        rights.append(f"{head}:rightGm")
        vnews.append(f"{head}:vNew")
        order.chain(free, f"{head}:next_input_write")
    if not early:
        mte3.extend(f"{head}:ready" for head in range(len(per_head_ops)))
    order.chain(*mte3)
    # Wait RIGHT_READY, read GM -> L1 -> L0, execute MMAD, then Fixpipe D.
    order.chain("0:ready", "right_gm_read", "right_l1_read", "mmad", "D_write")
    return order, rights, vnews


class HRightHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VEC.read_text(encoding="utf-8")
        cls.previous = subprocess.check_output(
            ["git", "show", f"{BASELINE}:{VEC.relative_to(ROOT).as_posix()}"],
            cwd=ROOT, encoding="utf-8")
        cls.region = writeback(cls.source)

    def test_only_writeback_and_publish_owners_change(self):
        current, previous = self.source, self.previous
        # All math, Stage3, input prefetch, offsets, local free events and
        # final drain remain literal K32 code outside these narrow regions.
        for name in ("ConsumeStage1Head", "RunStage1", "ProcessSingleHeadPipeline"):
            a, b = body(current, name), body(previous, name)
            if name == "ConsumeStage1Head":
                old_begin = b.index("            AscendC::DataCopy(vNew")
                old_end = b.index("            if constexpr (!CompilePolicy::STATE_FP32)", old_begin)
                old_end = b.index("            if constexpr (!CompilePolicy::STATE_FP32)", old_end + 1)
                self.assertEqual(clean(a.replace(self.region, "")),
                                 clean(b[:old_begin] + b[old_end:]))
            else:
                a = a.replace("writeRight && !UseEarlyRightReady(unit)", "writeRight")
                self.assertEqual(clean(a), clean(b))
            current = current.replace(body(current, name), "")
            previous = previous.replace(b, "")
        start = current.index("    __aicore__ inline bool UseEarlyRightReady(")
        _, end = braced(current, start)
        current = current[:start] + current[end:]
        self.assertEqual(clean(current), clean(previous))

    def test_actual_branches_and_exactly_one_ready(self):
        for fp32 in (False, True):
            for scalar_g in (False, True):
                for heads in range(1, 5):
                    early = early_policy(self.source, fp32, scalar_g, heads)
                    for write_right in (False, True):
                        with self.subTest(fp32=fp32, scalar_g=scalar_g, heads=heads,
                                          write_right=write_right):
                            ops = operations(specialize(self.region, early=early,
                                                       write_right=write_right, fp32=fp32))
                            expected = ["rightGm", "ready", "vNew"] if early else ["vNew", "rightGm"]
                            if not fp32:
                                expected.append("ready")
                            self.assertEqual(ops, expected if write_right else ["vNew"])
                            outer_ready = int(fp32 and not early and write_right)
                            self.assertEqual(ops.count("ready") + outer_ready, int(write_right))
        for name in ("RunStage1", "ProcessSingleHeadPipeline"):
            code = body(self.source, name)
            self.assertRegex(code, r"if \(writeRight && !UseEarlyRightReady\(unit\)\)\s*\{")
            self.assertEqual(code.count("FWD_H_RIGHT_READY_FLAG"), 1)

    def test_buffers_are_disjoint_for_early_D_and_public_write(self):
        policy = POLICY.read_text(encoding="utf-8")
        def literal_kib(name):
            return int(re.search(rf"{name} = (\d+) \* 1024;", policy)[1]) * 1024
        d_end = literal_kib("FWD_H_UB_LOCAL_SLOT_BYTES")
        work_begin = literal_kib("FWD_H_UB_BF16_WORK_BASE")
        gate_begin = literal_kib("FWD_H_UB_GATE_BASE")
        self.assertLessEqual(d_end, work_begin)
        self.assertEqual(gate_begin - work_begin, 2 * 64 * 128 * 2)
        # source-address formulas are part of the ownership proof
        self.assertIn("FWD_H_TOKEN_MATRIX_FP32_BYTES", body(self.source, "RightSlot"))
        self.assertIn("FWD_H_UB_BF16_WORK_BASE", body(self.source, "WorkBf16Slot"))

    def test_early_handoff_orders_conflicting_accesses(self):
        ops = operations(specialize(self.region, early=True, write_right=True, fp32=True))
        order, rights, vnews = handoff_order([ops], True)
        self.assertTrue(all(order.before(right, "D_write") for right in rights))
        self.assertTrue(order.before(vnews[0], "0:next_input_write"))
        # This dependency is deliberately absent; different UB regions allow overlap.
        self.assertFalse(order.before(vnews[0], "D_write"))
        self.assertFalse(order.before("D_write", vnews[0]))

    def test_unsafe_ready_and_free_mutations_are_detected(self):
        ops = operations(specialize(self.region, early=True, write_right=True, fp32=True))
        order, rights, _ = handoff_order([ops], True, ready_before_copy=True)
        self.assertFalse(order.before(rights[0], "D_write"))
        order, _, vnews = handoff_order([ops], True, free_before_vnew=True)
        self.assertFalse(order.before(vnews[0], "0:next_input_write"))

    def test_shared_right_requires_aggregate_ready(self):
        fallback = operations(specialize(self.region, early=False, write_right=True, fp32=True))
        order, rights, _ = handoff_order([fallback, fallback], False)
        self.assertTrue(all(order.before(right, "D_write") for right in rights))
        early = operations(specialize(self.region, early=True, write_right=True, fp32=True))
        wrong, rights, _ = handoff_order([early, early], True)
        self.assertFalse(wrong.before(rights[-1], "D_write"))

    def test_terminal_and_long_sequences_keep_credit_counts(self):
        for chunks in (1, 2, 5, 177):
            for final_state in (False, True):
                totals = defaultdict(int)
                for chunk in range(chunks):
                    need_right = final_state or chunk + 1 < chunks
                    ops = operations(specialize(self.region, early=True,
                                               write_right=need_right, fp32=True))
                    for op in ops:
                        totals[op] += 1
                self.assertEqual(totals["vNew"], chunks)
                self.assertEqual(totals["ready"], chunks - int(not final_state))
                self.assertEqual(totals["rightGm"], totals["ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

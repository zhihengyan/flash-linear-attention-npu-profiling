"""Offline QK reuse/credit/ownership checks; not CANN or a hardware emulator.

Use actual C++ subgroup predicates and slot expressions in a small serial
resource model. An independent consumer oracle rejects premature release.
Negative mutations demonstrate that bad reuse, stale keys and missing event
credits are observable. Numerical checks only validate reuse semantics.
"""

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re
import struct
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[8]
OP = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o"
CUBE = OP / "op_kernel/arch35/chunk_fwd_o_cube.h"
COMMON = OP / "op_kernel/arch35/chunk_fwd_o_common.h"
BASE = "d0cb628df7c8c23c0342392dfaf4ec910aba9947"


def stripped(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def tokens(text):
    return re.sub(r"\s+", "", stripped(text))


def braced(text, start):
    begin = text.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return begin, end


def body(text, name):
    declaration = re.search(r"\bvoid\s+" + re.escape(name) + r"\(", text)
    if declaration is None:
        raise AssertionError("missing function definition: " + name)
    begin, end = braced(text, declaration.start())
    return text[begin + 1:end - 1]


def branches(text, condition):
    pattern = r"\bif\s*\(" + re.escape(condition) + r"\)\s*\{"
    return [text[braced(text, m.start())[0] + 1:braced(text, m.start())[1] - 1]
            for m in re.finditer(pattern, text)]


def expression(text, variable):
    match = re.search(r"const (?:bool|uint32_t) " + variable + r"\s*=\s*([^;]+);", text)
    if match is None:
        raise AssertionError("missing source expression: " + variable)
    return match[1]


def evaluate(expr, values):
    expr = re.sub(r"(\d+)U\b", r"\1", expr)
    expr = expr.replace("tiling_.hvPerHk", "G").replace("||", " or ").replace("&&", " and ")
    expr = expr.replace("/", "//")
    expr = " ".join(expr.split())
    # This only reads numeric/bool expressions from this repository's source.
    return eval(expr, {"__builtins__": {}}, values)


@dataclass
class Policy:
    load: str
    last: str
    qh_slot: str
    next_slot: str

    @classmethod
    def read(cls, source):
        process = body(source, "Process")
        stage = body(source, "ProcessStage2Head")
        update = re.search(r"l0CSlot_ = ([^;]+);", stage)[1]
        return cls(expression(process, "loadQK"), expression(process, "lastQKConsumer"),
                   expression(stage, "qhCSlot"), update)


def schedule(hv, group, chunks=1, cores=1):
    count = (hv + group - 1) // group
    result = [[] for _ in range(cores)]
    for chunk in range(chunks):
        for head_group in range(count):
            begin = head_group * group
            result[(chunk * count + head_group) % cores].append(
                (chunk, list(range(begin, min(begin + group, hv)))))
    return result


def replay(policy, work, G, seed=0, mutation=None):
    c_free, l1_free = [1, 1], [1] * 4
    c_data, pinned = [None, None], None
    c_next, raw_slot = seed, None
    counts, copies, trace = Counter(), [], []

    def produce(slot, value):
        nonlocal pinned
        if c_free[slot] != 1:
            raise AssertionError("missing FIX_M credit")
        if pinned is not None and pinned[0] == slot:
            raise AssertionError("MMAD overwrites pinned raw")
        c_free[slot] = 0
        c_data[slot] = value
        trace.append((value[-1], slot))

    def release(slot):
        if c_free[slot] != 0:
            raise AssertionError("duplicate FIX_M credit")
        c_free[slot] = 1

    for work_id, (chunk, heads) in enumerate(work):
        # Oracle groups consumers by their input identity, independently of the
        # production first/last predicates and current cache slot.
        remaining = Counter(head // G for head in heads)
        consumed_ready = 0
        for offset, hv in enumerate(heads):
            hk = hv // G
            first = max(hk * G, heads[0])
            q_slot = first - heads[0]
            values = dict(hv=hv, hk=hk, firstHvInGroup=first, headOffset=offset,
                          taskCount=len(heads), G=G)
            load = evaluate(policy.load, values)
            last = evaluate(policy.last, values)
            if load:
                if l1_free[q_slot] != 1:
                    raise AssertionError("missing Q/K L1 free")
                l1_free[q_slot] = 0
                raw_slot = c_next
                produce(raw_slot, (chunk, work_id, hk, "QK"))
                pinned = (raw_slot, hk)
                c_next ^= 1
                counts["QK"] += 1
                counts["M_FIX_raw"] += 1
            elif mutation == "wait_raw_every_head":
                raise AssertionError("M_FIX wait without a new raw MMAD")
            if raw_slot is None or c_data[raw_slot] != (chunk, work_id, hk, "QK"):
                raise AssertionError("stale or wrong raw QK identity")
            owner, local = offset % 2, offset // 2
            copies.append((chunk, hv, hk, owner, local))
            counts["raw_copy"] += 1
            remaining[hk] -= 1
            if last or mutation == "release_raw_every_head":
                if remaining[hk]:
                    raise AssertionError("release before final raw consumer")
                pinned = None
                release(raw_slot)
                counts["FIX_M_raw"] += 1
            qh_slot = evaluate(policy.qh_slot, dict(qkCSlot_=raw_slot))
            if mutation == "qh_uses_raw":
                qh_slot = raw_slot
            produce(qh_slot, (chunk, work_id, hv, "QH"))
            counts["QH"] += 1
            release(qh_slot)
            c_next = evaluate(policy.next_slot, dict(qhCSlot=qh_slot))
            if last or mutation == "release_q_every_head":
                if remaining[hk]:
                    raise AssertionError("release before final QH L1 read")
                if l1_free[q_slot] != 0:
                    raise AssertionError("duplicate L1 free")
                l1_free[q_slot] = 1
            counts["ready"] += 1
            consumed_ready += 1
        if pinned is not None or any(remaining.values()):
            raise AssertionError("raw not released at work boundary")
        if consumed_ready != len(heads):
            raise AssertionError("missing owner/non-owner head notification")
        # Original Stage4 alternates free C slots after Stage2, including the
        # odd group that flips the next work's initial slot.
        for hv in heads:
            produce(c_next, (chunk, work_id, hv, "AV"))
            counts["AV"] += 1
            release(c_next)
            c_next ^= 1
        if c_free != [1, 1] or l1_free != [1] * 4:
            raise AssertionError("unbalanced credit at work boundary")
    return counts, copies, trace


def bf16(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def mm(left, right):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*right)] for row in left]


def output(raw, q, h, v, g):
    n = len(q)
    a = [[bf16(raw[i][j] * math.exp(g[i] - g[j])) if j <= i else 0.0
          for j in range(n)] for i in range(n)]
    qh, av = mm(q, h), mm(a, v)
    return [[bf16(.125 * (qh[i][j] * math.exp(g[i]) + av[i][j]))
             for j in range(len(v[0]))] for i in range(n)]


class OSharedQK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = CUBE.read_text(encoding="utf-8")
        cls.previous = subprocess.check_output(
            ["git", "show", f"{BASE}:{CUBE.relative_to(ROOT).as_posix()}"],
            cwd=ROOT, encoding="utf-8")
        cls.policy = Policy.read(cls.source)

    def test_subgroups_credits_and_slot_wrap(self):
        for G in range(1, 5):
            for hv in (G, 2 * G, 5 * G):
                for group in (3, 4):
                    for seed in (0, 1):
                        for work in schedule(hv, group, chunks=5, cores=3):
                            with self.subTest(G=G, hv=hv, group=group, seed=seed):
                                counts, copies, _ = replay(self.policy, work, G, seed)
                                expected = sum(len({h // G for h in heads}) for _, heads in work)
                                self.assertEqual(counts["QK"], expected)
                                self.assertEqual(counts["M_FIX_raw"], counts["FIX_M_raw"])
                                for kind in ("QH", "AV", "ready", "raw_copy"):
                                    self.assertEqual(counts[kind], len(copies))

    def test_case0_full_and_tail_workload(self):
        totals = Counter()
        for work in schedule(32, 4, chunks=177, cores=28):
            counts, copies, _ = replay(self.policy, work, 2)
            totals.update(counts)
            self.assertTrue(all(owner in (0, 1) and slot in (0, 1)
                                for _, _, _, owner, slot in copies))
        self.assertEqual(totals["QK"], 2832)
        for kind in ("QH", "AV", "raw_copy", "ready"):
            self.assertEqual(totals[kind], 5664)

    def test_negative_lifetimes_and_missing_producer(self):
        work = schedule(12, 4, chunks=3)[0]
        for mutation, reason in (("release_raw_every_head", "final raw"),
                                 ("release_q_every_head", "final QH"),
                                 ("qh_uses_raw", "missing FIX_M|overwrites pinned"),
                                 ("wait_raw_every_head", "without a new")):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(AssertionError, reason):
                replay(self.policy, work, 3, mutation=mutation)
        wrong = Policy(self.policy.load, "True", self.policy.qh_slot, self.policy.next_slot)
        with self.assertRaisesRegex(AssertionError, "final raw"):
            replay(wrong, work, 3)
        wrong = Policy("False", self.policy.last, self.policy.qh_slot, self.policy.next_slot)
        with self.assertRaisesRegex(AssertionError, "stale"):
            replay(wrong, work, 3)

    def test_g1_slot_order_matches_original(self):
        for group in (3, 4):
            work = schedule(7, group, chunks=3)[0]
            for seed in (0, 1):
                _, _, actual = replay(self.policy, work, 1, seed)
                expected, slot = [], seed
                for _, heads in work:
                    for _ in heads:
                        expected.extend((("QK", slot), ("QH", slot ^ 1)))
                    for _ in heads:
                        expected.append(("AV", slot))
                        slot ^= 1
                self.assertEqual(actual, expected)

    def test_numerical_reuse_keeps_head_specific_gate_and_H(self):
        rng = random.Random(20261016)
        def matrix(rows, cols):
            return [[bf16(rng.uniform(-.2, .2)) for _ in range(cols)] for _ in range(rows)]
        for G in (1, 2, 3, 4):
            for valid in (1, 3, 8):
                q = [matrix(valid, 4) for _ in range(3)]
                k = [matrix(valid, 4) for _ in range(3)]
                h = [matrix(4, 4) for _ in range(3 * G)]
                v = [matrix(valid, 4) for _ in range(3 * G)]
                g = [[-.01 * (hv + 1) * i for i in range(valid)] for hv in range(3 * G)]
                raw_cache = {hk: mm(q[hk], list(zip(*k[hk]))) for hk in range(3)}
                _, copies, _ = replay(self.policy, schedule(3 * G, 3 if G == 3 else 4)[0], G)
                for _, hv, raw_hk, _, _ in copies:
                    baseline = output(mm(q[hv // G], list(zip(*k[hv // G]))),
                                      q[hv // G], h[hv], v[hv], g[hv])
                    candidate = output(raw_cache[raw_hk], q[hv // G], h[hv], v[hv], g[hv])
                    self.assertEqual(candidate, baseline)

    def test_source_ownership_guards_and_unchanged_paths(self):
        stage = body(self.source, "ProcessStage2Head")
        previous_stage = body(self.previous, "ProcessStage2Head")
        self.assertEqual(tokens(stage.split("\n        LocalTensor<Element> l1Q =", 1)[0]),
                         tokens(previous_stage.split("// Move Q/K", 1)[0]))
        producer, = [b for b in branches(stage, "loadQK") if "QkTileMmad{}(" in b]
        for marker in ("qkCSlot_ = qktCSlot", "WaitFlag<HardEvent::FIX_M>(qktCSlot)",
                       "SetFlag<HardEvent::M_FIX>(qktCSlot)", "WaitFlag<HardEvent::M_FIX>(qktCSlot)"):
            self.assertIn(marker, producer)
        previous_raw = previous_stage[previous_stage.index("const uint32_t qktASlot"):
                                      previous_stage.index("// Publish Q @ K^T")]
        previous_raw = re.sub(r"LocalTensor<Element> l1Q\s*=\s*.*?;", "", previous_raw, count=1, flags=re.S)
        current_raw = producer.replace("qkCSlot_ = qktCSlot;", "")
        current_raw = current_raw.replace("SetFlag<HardEvent::M_FIX>(qktCSlot);", "")
        current_raw = current_raw.replace("WaitFlag<HardEvent::M_FIX>(qktCSlot);", "")
        current_raw = current_raw.replace("<HardEvent::MTE1_M>(qktASlot)", "<HardEvent::MTE1_M>(qktCSlot)")
        self.assertEqual(tokens(current_raw), tokens(previous_raw))
        for marker in ("SetFlag<HardEvent::FIX_M>(qktCSlot)",
                       "SetFlag<HardEvent::MTE1_MTE2>(qkEvent)"):
            self.assertEqual(stage.count(marker), 1)
            self.assertTrue(any(marker in b for b in branches(stage, "lastQKConsumer")))
        self.assertLess(stage.index("copyQktToUb(tensorARawUb"),
                        stage.index("SetFlag<HardEvent::FIX_M>(qktCSlot)"))
        self.assertLess(stage.index("CrossCoreWaitFlag(vecToCubeFlag_)"), stage.index("QkTileMmad{}("))
        before_qh = body(self.previous, "ProcessStage2Head").split("// Move Q/H", 1)[1]
        after_qh = stage.split("// Move Q/H", 1)[1]
        after_qh = after_qh.replace("const uint32_t qhCSlot = qkCSlot_ ^ 1U;", "const uint32_t qhCSlot = l0CSlot_;")
        after_qh = after_qh.replace("l0CSlot_ = qhCSlot ^ 1U;", "l0CSlot_ ^= 1U;")
        after_qh = after_qh.replace("if (lastQKConsumer)", "if (loadQK)")
        after_qh = after_qh.replace("<HardEvent::MTE1_M>(qhASlot)", "<HardEvent::MTE1_M>(qhCSlot)")
        self.assertEqual(tokens(after_qh), tokens(before_qh))
        current, previous = self.source, self.previous
        for name in ("Init", "Process", "ProcessStage2Head"):
            a, b = body(current, name), body(previous, name)
            if name == "Init":
                self.assertEqual(tokens(a.replace("qkCSlot_ = 0U;", "")), tokens(b))
            if name == "Process":
                a = re.sub(r"const bool lastQKConsumer = .*?;", "", a, flags=re.S)
                self.assertEqual(tokens(a.replace("loadQK, lastQKConsumer,", "loadQK,")), tokens(b))
            current = current.replace(a if name == "ProcessStage2Head" else body(current, name), "")
            previous = previous.replace(b, "")
        current = current.replace("uint32_t qkCSlot_ = 0U;", "")
        current = current.replace("bool lastQKConsumer, bool waitPreviousStage5", "bool waitPreviousStage5")
        current = current.replace("<HardEvent::MTE1_M>(avASlot)", "<HardEvent::MTE1_M>(avCSlot)")
        self.assertEqual(tokens(current), tokens(previous))
        for path in (COMMON, OP / "op_kernel/arch35/chunk_fwd_o_vector.h"):
            baseline = subprocess.check_output(["git", "show", f"{BASE}:{path.relative_to(ROOT).as_posix()}"],
                                               cwd=ROOT, encoding="utf-8")
            self.assertEqual(tokens(path.read_text(encoding="utf-8")), tokens(baseline))

    def test_actual_buffers_fit_and_owner_destinations_are_distinct(self):
        common = COMMON.read_text(encoding="utf-8")
        c_bytes = int(re.search(r"CHUNK_FWD_O_L0_C_BYTES = (\d+)U \* 1024U;", common)[1]) * 1024
        self.assertLessEqual(64 * 64 * 4, c_bytes)
        self.assertLessEqual(64 * 128 * 4, c_bytes)
        base = int(re.search(r"CHUNK_FWD_O_UB_LAYOUT_ARAW_BASE = (\d+)U;", common)[1])
        stride = int(re.search(r"CHUNK_FWD_O_UB_LAYOUT_ARAW_SLOT_BYTES = (\d+)U \* 1024U;", common)[1]) * 1024
        destinations = {(head % 2, base + (head // 2) * stride) for head in range(4)}
        self.assertEqual(len(destinations), 4)
        self.assertEqual(stride, 64 * 64 * 4)

    def test_mte1_ready_is_protected_by_operand_slot_free(self):
        stage = body(self.source, "ProcessStage2Head")
        av = body(self.source, "ProcessStage4Head")
        sources = {
            "QK": re.findall(r"SetFlag<HardEvent::MTE1_M>\((qkt\w+)\)", stage)[0],
            "QH": re.findall(r"SetFlag<HardEvent::MTE1_M>\((qh\w+)\)", stage)[0],
            "AV": re.findall(r"SetFlag<HardEvent::MTE1_M>\((av\w+)\)", av)[0],
        }
        def check(trace, expressions, seed):
            last_set, operand = {}, seed
            for kind, c_slot in trace:
                prefix = {"QK": "qkt", "QH": "qh", "AV": "av"}[kind]
                event = evaluate(expressions[kind], {prefix + "ASlot": operand,
                                                     prefix + "CSlot": c_slot})
                # Re-setting an ID is protected by the previous MMAD only if
                # MTE1 has to acquire that MMAD's same A/B slot free credit.
                if event in last_set and last_set[event] != operand:
                    raise AssertionError("ready ID can be reused without its previous operand free")
                last_set[event] = operand
                operand ^= 1
        for seed in (0, 1):
            for G in (1, 2, 3, 4):
                _, _, trace = replay(self.policy, schedule(5 * G, 3 if G == 3 else 4, 5)[0], G, seed)
                check(trace, sources, seed)
        for name, prefix in (("QK", "qkt"), ("QH", "qh"), ("AV", "av")):
            target = stage if name != "AV" else av
            self.assertEqual(target.count(f"SetFlag<HardEvent::MTE1_M>({prefix}ASlot)"),
                             target.count(f"WaitFlag<HardEvent::MTE1_M>({prefix}ASlot)"))
        _, _, trace = replay(self.policy, schedule(12, 4, 3)[0], 3)
        wrong = {kind: prefix + "CSlot" for kind, prefix in (("QK", "qkt"), ("QH", "qh"), ("AV", "av"))}
        with self.assertRaisesRegex(AssertionError, "previous operand free"):
            check(trace, wrong, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

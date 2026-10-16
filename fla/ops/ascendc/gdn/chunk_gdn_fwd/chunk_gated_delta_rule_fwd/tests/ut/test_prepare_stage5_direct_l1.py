"""Offline Stage5 layout/event contracts, not an AscendC instruction emulator.

Read production Fixpipe fields and branch bodies; compare direct NZ16 output
with an independent GM-ND -> NZ encoder. Hardware rounding, runtime ordering
and performance still require the original device tests.
"""

from collections import Counter
from dataclasses import dataclass, replace
import re
import unittest

import numpy as np

from test_prepare_k32 import ARCH, KERNEL, OP, ROOT, body, k32_source, tokens

COPY = ARCH / "chunk_gated_delta_rule_fwd_prepare_datacopy.h"


def branches(source):
    """Parse only braces, if/else and statements in the real copyout body."""
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)

    def space(pos):
        while pos < len(source) and source[pos].isspace():
            pos += 1
        return pos

    def sequence(pos, nested=False):
        result = []
        while (pos := space(pos)) < len(source):
            if source[pos] == "}":
                if not nested:
                    raise AssertionError("unexpected closing brace")
                return result, pos + 1
            condition = re.match(r"if\s*\(([^()]+)\)\s*\{", source[pos:])
            if condition:
                yes, pos = sequence(pos + condition.end(), True)
                pos = space(pos)
                no = []
                otherwise = re.match(r"else\s*\{", source[pos:])
                if otherwise:
                    no, pos = sequence(pos + otherwise.end(), True)
                result.append(("if", condition[1], yes, no))
            else:
                end = source.index(";", pos)
                statement = source[pos:end + 1]
                if "{" in statement or "}" in statement:
                    raise AssertionError("update parser for new source control flow")
                result.append(("statement", tokens(statement)))
                pos = end + 1
        if nested:
            raise AssertionError("unclosed branch")
        return result, pos

    return sequence(0)[0]


def source_trace(source, valid, output_a, task):
    values = {"m": valid, "n": 64, "outputA": int(output_a), "chunk.M": valid,
              "chunkSize": 64, "0": 0, "bank": task & 1, "taskIdx": task, "t": task}

    def condition(expression):
        for item in expression.split("&&"):
            match = re.fullmatch(r"\s*([\w.]+)\s*(==|!=)\s*([\w.]+)\s*", item)
            if match is None:
                raise AssertionError(f"unsupported condition: {item}")
            equal = values[match[1]] == values[match[3]]
            if equal != (match[2] == "=="):
                return False
        return True

    def walk(nodes):
        result = []
        for node in nodes:
            if node[0] == "if":
                result.extend(walk(node[2] if condition(node[1]) else node[3]))
                continue
            statement = node[1]
            event = re.fullmatch(r"(SetFlag|WaitFlag)<HardEvent::(\w+)>\((\w+)\);", statement)
            if event:
                result.append((event[1], event[2], values[event[3]]))
            elif statement.startswith("FixpipeL0cToGmNd<InDtype>("):
                result.append(("GM", task))
            elif statement.startswith("DumpAL0cToL1("):
                result.append(("L1", task))
            elif statement.startswith("CopyGmNdToL1Nz<InDtype>("):
                result.append(("Reload", task))
            elif not statement.startswith(("const", "GDN_TIMER_SCOPE(")):
                raise AssertionError(f"unrecognized statement: {statement}")
        return result

    return walk(branches(source))


def consumer(source):
    schedule = body(source, "ProcessAic")
    return schedule.rsplit("DecodeTask(cur.base, cur.nThis, t, chunk, hv);", 1)[1].split("Stage7_AicOne(", 1)[0]


@dataclass(frozen=True)
class Fix:
    src_stride: int
    dst_stride: int
    quant: str


def direct_descriptor():
    source = tokens(body(KERNEL.read_text(encoding="utf-8"), "DumpAL0cToL1"))
    fields = dict(re.findall(r"aFixpipeParams\.(\w+)=([^;]+);", source))
    if fields != dict(nSize="n", mSize="n", srcStride="n", dstStride="n*16",
                      quantPre="QuantMode_t::F322BF16", unitFlag="0", isChannelSplit="false"):
        raise AssertionError("review changed Fixpipe descriptor, including quant/format")
    return Fix(64, 64 * 16, fields["quantPre"])


def bf16_rne(matrix):
    bits = np.asarray(matrix, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def nz16(matrix):
    rows, cols = matrix.shape
    return matrix.reshape(rows // 16, 16, cols // 16, 16).transpose(2, 0, 1, 3).reshape(-1)


def direct_model(l0, descriptor):
    if descriptor.quant != "QuantMode_t::F322BF16":
        raise ValueError("unexpected quantization")
    target = np.full(4096 + 32, 0x5A5A, dtype=np.uint16)
    read, written = [], []
    for row in range(64):
        for col in range(64):
            source = col // 16 * descriptor.src_stride * 16 + row * 16 + col % 16
            dest = col // 16 * descriptor.dst_stride + row * 16 + col % 16
            if source >= len(l0) or dest >= 4096 or source < 0 or dest < 0:
                raise ValueError("copy bounds")
            target[16 + dest] = bf16_rne(l0[source:source + 1])[0]
            read.append(source)
            written.append(dest)
    if sorted(read) != list(range(4096)) or sorted(written) != list(range(4096)):
        raise ValueError("copy is not a full nonoverlapping tile")
    return target


def run_ready_protocol(source, pack_sizes, output_a, valid):
    """Token/ownership model; it cannot prove real asynchronous pipe timing."""
    credits = Counter({("FIX_MTE1", 0): 1})
    l1_owner = {}
    for pack, size in enumerate(pack_sizes):
        if credits["FIX_MTE1", 0] != 1:
            raise ValueError("missing or duplicate pack entry credit")
        credits["FIX_MTE1", 0] -= 1
        # Stage4 Y uses the same MTE2_MTE1 id as the old full-A path.
        # It must be consumed by Stage5 before any A-ready wait is attempted.
        for task in range(size):
            credits["MTE2_MTE1", task] += 1
            credits["MTE2_MTE1", task] -= 1
            rows = valid(pack, task) if callable(valid) else valid
            produced = source_trace(body(source, "Stage5_Dump"), rows, output_a, task)
            reads = []
            for action in produced:
                if action[0] in ("GM", "L1"):
                    reads.append(action[0])
                    if action[0] == "L1":
                        l1_owner[task] = pack
                if action[0] == "SetFlag" and action[1] in ("FIX_MTE1", "MTE2_MTE1"):
                    if l1_owner.get(task) != pack:
                        raise ValueError("ready before L1 write")
                    credits[action[1], action[2]] += 1
                if action == ("SetFlag", "FIX_M", task & 1):
                    if reads != (["GM", "L1"] if output_a else ["L1"]):
                        raise ValueError("L0C released before both outputs")
        for task in range(size):
            rows = valid(pack, task) if callable(valid) else valid
            for action in source_trace(consumer(source), rows, output_a, task):
                if action[0] == "WaitFlag":
                    key = action[1:]
                    if credits[key] != 1:
                        raise ValueError("unmatched A-ready wait")
                    credits[key] -= 1
            if l1_owner[task] != pack:
                raise ValueError("Stage7 read stale L1 A")
        if any(credits.values()):
            raise ValueError("orphan A/Y credit before pack exit")
        # Original pack-exit signal, after the unchanged Stage7 last consumer.
        credits["FIX_MTE1", 0] += 1
    return credits


class PrepareStage5DirectL1(unittest.TestCase):
    def test_fp32_to_bf16_nz16_and_gm_valid_row_bounds(self):
        descriptor = direct_descriptor()
        rng = np.random.default_rng(516)
        # Non-symmetric values plus exact tie/zero/subnormal bit patterns.
        matrix = rng.normal(size=(64, 64)).astype(np.float32)
        matrix[0, :8] = np.array([0, 0x80000000, 0x3F808000, 0x3F818000,
                                  0xBF808000, 0xBF818000, 1, 0x007FFFFF], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(bf16_rne(matrix[0, :6]),
                                      [0, 0x8000, 0x3F80, 0x3F82, 0xBF80, 0xBF82])
        actual = direct_model(nz16(matrix), descriptor)
        expected = nz16(bf16_rne(matrix))
        np.testing.assert_array_equal(actual[16:-16], expected)
        self.assertTrue(np.all(actual[:16] == 0x5A5A) and np.all(actual[-16:] == 0x5A5A))
        # The existing GM-ND path uses the same conversion, but only M rows.
        # Decode L0C by fixed physical fractals, independently of the NZ encoder.
        l0 = nz16(matrix)
        rows, cols = np.indices((64, 64))
        source = cols // 16 * 1024 + rows // 16 * 256 + rows % 16 * 16 + cols % 16
        for valid in range(1, 65):
            gm = np.full(64 * 64 + 32, 0xA55A, dtype=np.uint16)
            gm[16:16 + valid * 64] = bf16_rne(l0[source[:valid]]).reshape(-1)
            np.testing.assert_array_equal(gm[16:16 + valid * 64], bf16_rne(matrix[:valid]).reshape(-1))
            self.assertTrue(np.all(gm[16 + valid * 64:] == 0xA55A))
            if valid == 64:
                np.testing.assert_array_equal(nz16(gm[16:-16].reshape(64, 64)), actual[16:-16])

    def test_negative_layout_and_old_ready_wait_are_detected(self):
        matrix = np.arange(4096, dtype=np.float32).reshape(64, 64)
        desc = direct_descriptor()
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            direct_model(nz16(matrix), replace(desc, dst_stride=512))
        with self.assertRaisesRegex(ValueError, "bounds"):
            direct_model(nz16(matrix), replace(desc, src_stride=128))
        self.assertFalse(np.array_equal(direct_model(nz16(matrix.T), desc)[16:-16], nz16(bf16_rne(matrix))))
        source = KERNEL.read_text(encoding="utf-8")
        wait = consumer(source)
        wrong = source.replace(wait, wait.replace("FIX_MTE1", "MTE2_MTE1"))
        with self.assertRaisesRegex(ValueError, "unmatched A-ready"):
            run_ready_protocol(wrong, [4, 3, 1], True, 64)

    def test_source_branches_preserve_tail_and_a0_and_dual_output_order(self):
        current = KERNEL.read_text(encoding="utf-8")
        original = k32_source(KERNEL)
        for valid in range(1, 65):
            for output_a in (False, True):
                for task in range(4):
                    trace = source_trace(body(current, "Stage5_Dump"), valid, output_a, task)
                    old = source_trace(body(original, "Stage5_Dump"), valid, output_a, task)
                    self.assertNotIn(("Reload", task), trace)
                    self.assertEqual([a for a in trace if a[0] in ("GM", "L1")],
                                     [("GM", task), ("L1", task)] if output_a else [("L1", task)])
                    self.assertEqual(trace[-3:], [("L1", task), ("SetFlag", "FIX_MTE1", task),
                                                 ("SetFlag", "FIX_M", task & 1)])
                    if not output_a or valid < 64:
                        self.assertEqual(trace, old)
                    self.assertEqual(source_trace(consumer(current), valid, output_a, task),
                                     [("WaitFlag", "FIX_MTE1", task)])

    def test_y_and_a_event_tokens_across_partial_and_consecutive_packs(self):
        source = KERNEL.read_text(encoding="utf-8")
        for group in (1, 2, 3, 4):
            capacity = 3 if group == 3 else 4
            for last in range(1, capacity + 1):
                for output_a in (False, True):
                    for valid in (1, 10, 31, 32, 33, 63, 64):
                        with self.subTest(group=group, last=last, output_a=output_a, valid=valid):
                            credits = run_ready_protocol(source, [capacity] * 4 + [last], output_a, valid)
                            self.assertEqual(+credits, Counter({("FIX_MTE1", 0): 1}))
                # Mix full/tail tasks and alternate them between generations.
                for output_a in (False, True):
                    mixed = lambda pack, task: (64, 10, 32, 63)[(pack + task) % 4]
                    credits = run_ready_protocol(source, [capacity] * 4 + [last], output_a, mixed)
                    self.assertEqual(+credits, Counter({("FIX_MTE1", 0): 1}))

    def test_only_copyout_and_its_ready_consumer_change_from_k32(self):
        current = KERNEL.read_text(encoding="utf-8")
        original = k32_source(KERNEL)
        for name in ("Stage5_Dump", "ProcessAic"):
            before, after = body(original, name), body(current, name)
            if name == "ProcessAic":
                self.assertEqual(tokens(before.replace(consumer(original), "")),
                                 tokens(after.replace(consumer(current), "")))
            else:
                # GM parameters, offset computation and source operand are preserved.
                gm = r"FixpipeL0cToGmNd<InDtype>\([^;]+;"
                self.assertEqual(re.findall(gm, tokens(before)), re.findall(gm, tokens(after)))
                self.assertEqual(tokens(before.split("WaitFlag<HardEvent::M_FIX>")[0]),
                                 tokens(after.split("WaitFlag<HardEvent::M_FIX>")[0]))
            original = original.replace(before, "")
            current = current.replace(after, "")
        self.assertEqual(tokens(original), tokens(current))
        for path in ARCH.glob("*.h"):
            if path != KERNEL:
                self.assertEqual(tokens(path.read_text(encoding="utf-8")), tokens(k32_source(path)), path.name)
        for path in (OP.parent / "chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp",
                     OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp"):
            self.assertEqual(tokens(path.read_text(encoding="utf-8")), tokens(k32_source(path)), path.name)

    def test_timer_inventory_and_remaining_y_dependency(self):
        source = KERNEL.read_text(encoding="utf-8")
        schedule = tokens(body(source, "ProcessAic"))
        self.assertIn("WaitFlag<HardEvent::MTE2_MTE1>(t);Stage5_Compute(t);", schedule)
        self.assertIn("WaitFlag<HardEvent::FIX_MTE1>(t);Stage7_AicOne(chunk,hv,t);", schedule)
        self.assertEqual(re.findall(r"GDN_TIMER_SCOPE\([^;]+", source),
                         re.findall(r"GDN_TIMER_SCOPE\([^;]+", k32_source(KERNEL)))
        inventory = (OP / "docs/timer_flow.md").read_text(encoding="utf-8")
        relative = KERNEL.relative_to(ROOT).as_posix()
        rows = [int(number) for number in re.findall(re.escape(relative) + r":(\d+)", inventory)]
        actual = [i for i, line in enumerate(source.splitlines(), 1) if "GDN_TIMER_SCOPE(" in line]
        self.assertEqual(rows, actual)
        full_tasks = 11274 // 64 * 32
        self.assertEqual(full_tasks * 64 * 64 * 2, 44 * 1024**2)


if __name__ == "__main__":
    unittest.main()

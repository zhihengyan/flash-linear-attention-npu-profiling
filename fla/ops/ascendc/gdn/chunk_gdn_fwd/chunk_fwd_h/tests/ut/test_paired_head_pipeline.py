"""Offline pipe-queue ownership model; not a CANN timing/numeric emulator."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import importlib.util
import json
import random
import re
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[8]
OP = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h"
BASELINE = "809bce6887639e831caa856b8ac3a197a6a94294"


def function_body(text, name):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    match = re.search(r"\b" + name + r"\s*\([^;{}]*\)\s*(?:const\s*)?\{", text)
    if not match:
        raise AssertionError("Missing function: " + name)
    begin = match.end()
    end, depth = begin, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end - 1]


class QueueModel:
    """MTE2/MTE1/MMAD/Fixpipe plus two independent AIV MTE2/V/MTE3 queues."""
    def __init__(self, chunks, initial, final, fault=None, force_cube=False):
        self.chunks, self.initial, self.final, self.fault = chunks, initial, final, fault
        self.streams = defaultdict(deque)
        self.credits = Counter()
        self.memory = {}
        self.reads = Counter()
        self.busy = set()
        self.consumed = defaultdict(set)
        self.expected = {}
        self.state = {0: 0, 1: 0}
        self.has_cube = force_cube or initial or final or chunks > 1
        if self.has_cube:
            for p in range(2):
                self.credits[("group_free", p)] = 1
            for h in range(2):
                self.credits[("hr_free", h)] = 1
        for h in range(2):
            if initial:
                self.memory[("H", h)] = 0
                self.credits[("H_ready", h)] = 1
            for p in range(2):
                for kind in ("work_free", "gate_free"):
                    self.credits[(kind, h, p)] = 1
            self.credits[("io_free", h)] = 1
        self.build()

    def add(self, stream, kind, *args):
        self.streams[stream].append((kind, args))

    def wait(self, stream, *event):
        self.add(stream, "wait", event)

    def send(self, stream, *event):
        self.add(stream, "send", event)

    def has_p(self, c):
        return self.initial or c > 0

    def needs_d(self, c):
        return self.final or c + 1 < self.chunks

    def prefetch_group(self, c):
        p = c & 1
        self.wait("c.mte2", "group_free", p)
        self.add("c.mte2", "acquire", p, c)
        required = {("W", h) for h in range(2)} if self.has_p(c) else set()
        if self.needs_d(c):
            required.update(("K", h) for h in range(2))
        self.expected[c] = required
        if self.has_p(c):
            for h in range(2):
                self.add("c.mte2", "input", "W", 2 * p + h, c)
        if self.needs_d(c):
            self.add("c.mte2", "input", "K", p, c)
        self.send("c.mte2", "group_ready", p)

    def prefetch_vector(self, h, c):
        p = 0 if self.fault == "wrong_input_bank" else c & 1
        stream = f"v{h}.mte2"
        self.wait(stream, "work_free", h, p)
        if self.needs_d(c):
            self.wait(stream, "gate_free", h, p)
        self.add(stream, "input", "U" + str(h), p, c)
        if self.needs_d(c):
            self.add(stream, "input", "g" + str(h), p, c)
        self.send(stream, "input_ready", h, p)

    def build(self):
        if self.has_cube:
            self.prefetch_group(0)
        for c in range(self.chunks):
            p = c & 1
            if self.has_cube:
                self.wait("c.mte1", "group_ready", p)
            if self.has_p(c):
                for h in range(2):
                    if self.fault != "missing_H_wait":
                        self.wait("c.mte2", "H_ready", h)
                    self.wait("c.mte2", "hr_free", h)
                    self.add("c.mte2", "check", "H", h, c)
                    self.send("c.mte2", "hr_done", h)
                    self.wait("c.mte1", "hr_done", h)
                    self.add("c.mte1", "consume", "W", 2 * p + h, c, h)
                    self.send("c.mte1", "hr_free", h)
                    self.send("c.mte1", "p_request", h)
                    self.wait("c.m", "p_request", h)
                    self.send("c.m", "p_done", h)
                    self.wait("c.fix", "p_done", h)
                    if c:
                        self.wait("c.fix", "d_free", h)
                    self.add("c.fix", "write", "P", h, c)
                    self.send("c.fix", "p_ready", h)
            if self.needs_d(c):
                for h in range(2):
                    self.wait("c.mte2", "right_ready", h)
                    self.wait("c.mte2", "hr_free", h)
                    self.add("c.mte2", "check", "right", h, c)
                    self.send("c.mte2", "hr_done", h)
                if c + 1 < self.chunks:
                    self.prefetch_group(c + 1)
                for h in range(2):
                    self.wait("c.mte1", "hr_done", h)
                    self.add("c.mte1", "consume", "K", p, c, h)
                    self.send("c.mte1", "hr_free", h)
                    self.send("c.mte1", "right_free", h)
                    self.send("c.mte1", "d_request", h)
                    self.wait("c.m", "d_request", h)
                    self.send("c.m", "d_done", h)
                    self.wait("c.fix", "d_done", h)
                    if self.has_p(c):
                        self.wait("c.fix", "p_free", h)
                    self.add("c.fix", "write", "D", h, c)
                    self.send("c.fix", "d_ready", h)
            if self.has_cube and not (self.fault == "omit_terminal_release" and not self.needs_d(c)):
                self.add("c.mte1", "release", p, c)
                self.send("c.mte1", "group_free", p)
        if self.has_cube:
            for p in range(2):
                self.wait("c.mte1", "group_free", p)
            for h in range(2):
                self.wait("c.mte1", "hr_free", h)
                if self.final:
                    self.wait("c.fix", "d_free", h)
                elif self.has_p(self.chunks - 1):
                    self.wait("c.fix", "p_free", h)
        for h in range(2):
            m2, vec, m3 = f"v{h}.mte2", f"v{h}.v", f"v{h}.mte3"
            self.prefetch_vector(h, 0)
            for c in range(self.chunks):
                p = c & 1
                self.wait(m2, "io_free", h)
                self.send(m2, "io_ready", h)
                if c + 1 < self.chunks:
                    self.prefetch_vector(h, c + 1)
                self.wait(vec, "input_ready", h, p)
                self.wait(vec, "io_ready", h)
                if c:
                    self.wait(vec, "right_free", h)
                if self.has_p(c):
                    self.wait(vec, "p_ready", h)
                    self.add(vec, "check", "P", h, c)
                self.add(vec, "check", "U" + str(h), p, c)
                if self.needs_d(c):
                    self.add(vec, "check", "g" + str(h), p, c)
                self.add(vec, "write", "right_ub", h, c)
                if self.has_p(c):
                    self.send(vec, "p_free", h)
                self.send(vec, "s1_done", h)
                self.wait(m3, "s1_done", h)
                self.add(m3, "check", "U" + str(h), p, c)
                self.add(m3, "write", "vNew", h, c)
                if self.needs_d(c):
                    self.add(m3, "check", "right_ub", h, c)
                    self.add(m3, "write", "right", h, c)
                    self.send(m3, "right_ready", h)
                self.send(m3, "work_free", h, p)
                self.send(m3, "io_free", h)
                if self.needs_d(c):
                    # Stage3's existing Io probe returns then reacquires the S1 credit.
                    self.wait(m2, "io_free", h)
                    self.send(m3, "io_free", h)
                    self.wait(m2, "io_free", h)
                    self.wait(vec, "d_ready", h)
                    self.add(vec, "check", "D", h, c)
                    self.add(vec, "check", "g" + str(h), p, c)
                    self.add(vec, "state", h, c)
                    self.send(vec, "gate_free", h, p)
                    self.send(vec, "s3_done", h)
                    self.wait(m3, "s3_done", h)
                    if c + 1 < self.chunks:
                        self.add(m3, "write", "H", h, c + 1)
                        self.send(m3, "H_ready", h)
                    self.send(m3, "d_free", h)
                    self.send(m3, "io_free", h)
            if self.final:
                self.wait(vec, "right_free", h)
            self.wait(m2, "io_free", h)
            for p in range(2):
                self.wait(m2, "work_free", h, p)
                self.wait(m2, "gate_free", h, p)

    def execute(self, kind, args):
        if kind == "wait":
            self.credits[args[0]] -= 1
        elif kind == "send":
            self.credits[args[0]] += 1
        elif kind == "acquire":
            p, c = args
            assert p not in self.busy, "overwriting live input group"
            self.busy.add(p)
            self.consumed[c] = set()
        elif kind == "input":
            name, slot, c = args
            self.memory[(name, slot)] = c
            self.reads[(name, c)] += 1
        elif kind in ("check", "consume"):
            name, slot, c = args[:3]
            assert self.memory.get((name, slot)) == c, f"stale {name} bank at chunk {c}"
            if kind == "consume":
                self.consumed[c].add((name, args[3]))
        elif kind == "write":
            name, slot, c = args
            self.memory[(name, slot)] = c
        elif kind == "release":
            p, c = args
            assert self.consumed[c] == self.expected[c], "release before last input consumer"
            self.busy.remove(p)
        elif kind == "state":
            h, c = args
            assert self.state[h] == c, "state recurrence reordered"
            self.state[h] = c + 1

    def run(self, seed):
        rng = random.Random(seed)
        while any(self.streams.values()):
            ready = [name for name, queue in self.streams.items() if queue and
                     (queue[0][0] != "wait" or self.credits[queue[0][1][0]] > 0)]
            assert ready, "deadlock: " + str({name: q[0] for name, q in self.streams.items() if q})
            self.execute(*self.streams[rng.choice(ready)].popleft())
        assert not self.busy, "live input group at exit"
        assert all(n == 0 for n in self.credits.values()), "unconsumed credit at exit"
        for c in range(self.chunks):
            assert self.reads[("K", c)] == int(self.needs_d(c)), "shared K not read exactly once"
            assert self.reads[("W", c)] == 2 * int(self.has_p(c))
            for h in range(2):
                assert self.reads[("U" + str(h), c)] == 1
                assert self.reads[("g" + str(h), c)] == int(self.needs_d(c))
        assert self.state[0] == self.state[1] == self.chunks - int(not self.final)


class PairedHeadContract(unittest.TestCase):
    def test_random_pipe_interleavings_and_exit_credits(self):
        for chunks in (1, 2, 3, 17, 177):
            for initial in (False, True):
                for final in (False, True):
                    for seed in range(12):
                        with self.subTest(chunks=chunks, initial=initial, final=final, seed=seed):
                            QueueModel(chunks, initial, final).run(seed)

    def test_detects_missing_terminal_free(self):
        with self.assertRaisesRegex(AssertionError, "deadlock"):
            QueueModel(3, True, False, "omit_terminal_release").run(1)

    def test_short_varlen_unit_in_a_globally_active_cube_launch(self):
        # Global T can exceed 64 although this sequence needs neither WH nor KV.
        # Its empty group must still balance credits for later work units.
        for seed in range(12):
            QueueModel(1, False, False, force_cube=True).run(seed)

    def test_detects_wrong_input_bank(self):
        with self.assertRaisesRegex(AssertionError, "deadlock|stale"):
            QueueModel(3, True, True, "wrong_input_bank").run(2)

    def test_detects_unprotected_recurrence_input(self):
        caught = False
        for seed in range(20):
            try:
                QueueModel(3, True, True, "missing_H_wait").run(seed)
            except AssertionError:
                caught = True
                break
        self.assertTrue(caught)

    def test_shared_production_selector_accepts_only_complete_gva2_pair(self):
        body = function_body((OP / "op_kernel/chunk_fwd_h_utils.h").read_text(encoding="utf-8"),
                             "FwdHUsePairedHeadLookahead")
        expression = re.search(r"return\s+(.*?);", body, re.S).group(1)
        expression = re.sub(r"static_cast<uint64_t>\(([^)]+)\)", r"\1", expression)
        expression = " ".join(expression.replace("2ULL", "2").replace("&&", " and ").split())
        for group in (1, 2, 3, 4):
            for count in (1, 2, 3, 4):
                for start in range(16):
                    khs = [h // group for h in range(start, start + count)]
                    tiling = SimpleNamespace(vNumHead=16 * group, kNumHead=16)
                    plan = SimpleNamespace(activeHeadCount=count, requiredKhCount=len(set(khs)),
                                           heads=[SimpleNamespace(kh=kh) for kh in khs])
                    actual = eval(expression, {"__builtins__": {}}, {"tiling": tiling, "plan": plan})
                    self.assertEqual(actual, group == 2 and count == 2 and start % 2 == 0)

    def test_unmodified_helpers_with_default_input_event_policy(self):
        path = "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h"
        before = subprocess.check_output(
            ["git", "show", BASELINE + ":" + path], cwd=ROOT, text=True, encoding="utf-8")
        after = (ROOT / path).read_text(encoding="utf-8")
        for name in ("LoadStage0W", "ComputeStage0Head", "LoadKg", "ComputeStage2Head"):
            body = function_body(after, name)
            marker = "if constexpr (MANAGE_INPUT_EVENTS)"
            while marker in body:
                begin = body.index(marker)
                brace = body.index("{", begin)
                end, depth = brace + 1, 1
                while depth:
                    depth += (body[end] == "{") - (body[end] == "}")
                    end += 1
                body = body[:begin] + body[brace + 1:end - 1] + body[end:]
            self.assertEqual(re.sub(r"\s+", "", body),
                             re.sub(r"\s+", "", function_body(before, name)), name)

    def test_source_uses_joint_credits_and_current_right_before_lookahead(self):
        cube = (OP / "op_kernel/arch35/chunk_fwd_h_cube.h").read_text(encoding="utf-8")
        vec = (OP / "op_kernel/arch35/chunk_fwd_h_vec.h").read_text(encoding="utf-8")
        body = function_body(cube, "ProcessPairedHeadPipeline")
        self.assertLess(body.index("LoadRight("), body.index("parity ^ 1U"))
        self.assertIn("ComputeStage0Head<false>", body)
        self.assertIn("ComputeStage2Head<false>", body)
        self.assertLess(body.index("ComputeStage2Head<false>"), body.index("WReadyEvent(parity)"))
        self.assertIn("LoadStage0W<false>", function_body(cube, "PrefetchPairedHeadInputs"))
        self.assertIn("LoadKg<false>", function_body(cube, "PrefetchPairedHeadInputs"))
        for text in (cube, vec):
            self.assertIn("FwdHUsePairedHeadLookahead(args_.tiling, unit.headRound)", text)
        self.assertIn("unit.headRound.heads[aiv_]", function_body(vec, "ProcessSingleHeadPipeline"))

    def test_vec_arithmetic_and_existing_single_head_pipeline_are_unchanged(self):
        path = "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h"
        before = subprocess.check_output(
            ["git", "show", BASELINE + ":" + path], cwd=ROOT, text=True, encoding="utf-8")
        after = (ROOT / path).read_text(encoding="utf-8")
        for name in ("RunSMinusOne", "ConsumeStage1Head", "DispatchStage1", "ConsumeStage3Head",
                     "RunStage3Impl", "FwdHStage1Arch35Vf", "FwdHStage3Arch35Vf",
                     "ProcessSingleHeadPipeline"):
            self.assertEqual(re.sub(r"\s+", "", function_body(before, name)),
                             re.sub(r"\s+", "", function_body(after, name)), name)


class FrozenPairedCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case_dir = ROOT / "tests/atk/chunk_fwd_h"
        spec = importlib.util.spec_from_file_location("paired_case_generator", cls.case_dir / "gen_chunk_fwd_h.py")
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)

    def test_existing_cases_seeds_and_standards_are_preserved(self):
        for filename, count in (("atk_chunk_fwd_h.json", 200),
                                ("atk_chunk_fwd_h_perf.json", 44), ("atk_chunk_fwd_h_mss.json", 32)):
            path = "tests/atk/chunk_fwd_h/" + filename
            old = json.loads(subprocess.check_output(
                ["git", "show", BASELINE + ":" + path], cwd=ROOT, text=True, encoding="utf-8"))
            new = json.loads((self.case_dir / filename).read_text(encoding="utf-8"))
            self.assertEqual(new[:count], old)

    def test_frozen_json_matches_generator(self):
        for filename, build in (("atk_chunk_fwd_h.json", self.gen.build_accuracy_specs),
                                ("atk_chunk_fwd_h_perf.json", self.gen.build_perf_specs),
                                ("atk_chunk_fwd_h_mss.json", self.gen.build_mss_specs)):
            self.assertEqual(json.loads((self.case_dir / filename).read_text(encoding="utf-8")),
                             self.gen._payloads(build()))

    def test_branch_templates_lifecycle_and_task_ranges_are_covered(self):
        all_specs = self.gen.build_accuracy_specs()[200:]
        self.assertEqual(len(all_specs), 144)
        specs = all_specs[:48]
        self.assertEqual(len(specs), 48)
        for offset, first in enumerate(specs):
            copies = [all_specs[offset + repetition * 48] for repetition in range(3)]
            self.assertEqual(len({s["seed"] for s in copies}), 3)
            for copy in copies:
                self.assertEqual({k: v for k, v in copy.items() if k not in {"seed", "case_id", "case_key"}},
                                 {k: v for k, v in first.items() if k not in {"seed", "case_id", "case_key"}})
        matrix = {(s["gate_dtype"], s["use_exp2"], s["state_v_first"],
                   s["state_dtype"] == "fp32", s["output_final_state"]) for s in specs[:32]}
        self.assertEqual(len(matrix), 32)
        self.assertEqual(len(self.gen.build_mss_specs()[32:]), 8)
        for s in specs:
            self.assertEqual((s["input_scale"], s["gate_step_scale"], s["state_scale"]), (0.05, 0.02, 0.05))
            self.assertEqual(s["mode"], "g")
            self.assertEqual(s["HV"], 2 * s["HK"])
            sequences = len(s["seqlens"].split(",")) if s["seqlens"] else s["B"]
            total = sequences * s["HV"]
            heads_per_core = (total + 27) // 28
            blocks = (total + heads_per_core - 1) // heads_per_core
            stride = (total + blocks - 1) // blocks
            selected, multi_unit = 0, False
            for core in range(blocks):
                cursor, end = core * stride, min((core + 1) * stride, total)
                units = 0
                while cursor < end:
                    start = cursor % s["HV"]
                    count = min(4, end - cursor, s["HV"] - start)
                    selected += int(count == 2 and start % 2 == 0)
                    cursor += count
                    units += 1
                multi_unit |= units > 1
            self.assertGreater(selected, 0, s["case_key"])
            if "unit_transition" in s["case_key"]:
                self.assertTrue(multi_unit, s["case_key"])


if __name__ == "__main__":
    unittest.main()

"""Diagnostic protocol and same-core task correlation tests; no device evidence."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_prepare_stage35 as a
import parse_timer_csv as p
from test_timer_contract import sample


def detail_raw(start=16, sampled=2):
    raw = sample((1, 2, 3))
    l = p.LAYOUT
    offset = l["BUFFER_WORDS_PER_KERNEL"]
    raw[offset + 12:offset + 15] = [start, 8, 1]
    core, stage = 1, l["PREPARE_S3_KKT_WAIT"]
    counts = offset + l["HEADER_WORDS"] + l["N_TIMING_COUNTER"] + core * l["DYNAMIC_ITER_PER_CORE_ALIGN"]
    raw[counts + stage] = 100
    raw[counts + l["DYNAMIC_ITER_TOTAL_SIZE"] + stage] = sampled
    tags = offset + l["HEADER_WORDS"] + l["N_TIMING_COUNTER"] + l["DYNAMIC_ITER_TOTAL_SIZE"] * 2
    tags += core * l["DETAIL_TAG_PER_CORE_ALIGN"] + (stage - l["FIRST_DETAIL_TYPE"]) * l["MAX_DYNAMIC_ITER"]
    row = offset + l["HEADER_WORDS"] + core * l["N_TIMING_COUNTER_PER_CORE_ALIGN"]
    row += 2 * (l["FIXED_TIMING_COUNT"] + stage * l["MAX_DYNAMIC_ITER"])
    for i in range(sampled):
        raw[row + i * 2:row + i * 2 + 2] = [2**54 + 101 + i * 3, 2**54 + 102 + i * 3]
        raw[tags + i] = start * 4 + i * 4 + 1
    return raw, tags


def task_fixture(branch="full", start=0):
    # One original G=2 four-task pack. AIV0 tasks0/2 map to work0/1.
    capture = dict(inputs=dict(v=dict(shape=[1, 4, 64, 128]),
                               k=dict(shape=[1, 2, 64, 128])),
                   options=dict(layout="BNSD"), header_sha256="header", input_snapshot_sha256="input")
    if branch == "tail":
        capture["inputs"]["v"]["shape"][2] = 63
        capture["inputs"]["k"]["shape"][2] = 63
    if branch == "no_output":
        capture["options"]["disable_recompute"] = False
    prepare = dict(block_dim=1, sampling=dict(mode="pack_window", pack_start=start, pack_count=8,
                                            detail_enabled=True),
                   iterations=[])
    aic = a.S4 + a.S5 + a.S7 + ["S4_Y_TRANSFER", "S5_A_TRANSFER", "S5_A_READY_PUBLISH", "S5_READY_PUBLISH"]
    aic += (["S5_A_GM_WRITE", "S5_A_GM_DONE_WAIT"] if branch != "no_output" else [])
    aic += ["S5_A_L1_READ" if branch == "full" else "S5_A_L1_WRITE"]
    for core in range(3):
        prepare["iterations"].append(dict(core_id=core, stage="PREPARE_PACK", actual=1))
        for name in aic if core == 0 else a.S3 + ["S3_POST_KKT"]:
            prepare["iterations"].append(dict(core_id=core, stage="PREPARE_" + name, actual=4 if core == 0 else 2))
    events = []
    next_slot = {}

    def add(core, work, stage, begin, end):
        key = core, stage
        slot = next_slot.get(key, 0)
        next_slot[key] = slot + 1
        events.append(dict(kernel="PREPARE", core_id=core, group_id=0,
                           core_type="AIC" if core == 0 else "AIV", sub_id=core - 1,
                           stage="PREPARE_" + stage, iteration=slot, work_id=work,
                           kind="wait" if stage.endswith("WAIT") else "issue",
                           start_cycle=begin + 1000, end_cycle=end + 1000, duration_cycles=end - begin))
    if start:
        return events, prepare, capture
    for core in range(3):
        add(core, 0, "PACK", 0, 990)
    for work in range(4):
        core = 1 if work in (0, 1) else 2
        for i, stage in enumerate(a.S3):
            add(core, work, stage, i * 10 + 1, i * 10 + 5)
        add(core, work, "S3_POST_KKT", 10, 99)
        seq = a.S4 + a.S5
        seq += (["S5_A_GM_WRITE", "S5_A_GM_DONE_WAIT"] if branch != "no_output" else [])
        seq += ["S5_A_L1_READ" if branch == "full" else "S5_A_L1_WRITE",
                "S5_A_READY_PUBLISH", "S5_READY_PUBLISH"] + a.S7
        for i, stage in enumerate(seq):
            add(0, work, stage, i * 10 + 101, i * 10 + 105)
        add(0, work, "S4_Y_TRANSFER", 120, 169)
        a_transfer_start = 100 + seq.index("S5_CUBE_DONE_WAIT") * 10
        a_transfer_end = 100 + seq.index("S5_READY_PUBLISH") * 10 + 9
        add(0, work, "S5_A_TRANSFER", a_transfer_start, a_transfer_end)
    return events, prepare, capture


def combined_fixture(detail=True):
    return dict(capture="detail" if detail else "coarse",
                sampling=dict(mode="pack_window", pack_start=0, pack_count=8, detail_enabled=detail),
                stages=[], task_rows=[], pack_rows=[],
                kernel_windows=[dict(core_type="AIC", median=110 if detail else 100),
                                dict(core_type="AIV", median=220 if detail else 200)],
                inputs={}, options={}, header_sha256="header", input_snapshot_sha256="input", block_dim=1)


class SelectedPackDecode(unittest.TestCase):
    def test_late_dense_slots_preserve_actual_and_work_zero(self):
        raw, _ = detail_raw(start=0)
        events, metadata = p.parse_timer(raw)
        events = [e for e in events if e["stage"] == "PREPARE_S3_KKT_WAIT"]
        self.assertEqual([(e["iteration"], e["work_id"]) for e in events], [(0, 0), (1, 4)])
        count = next(c for m in metadata if m["kernel"] == "PREPARE" for c in m["iterations"]
                     if c["stage"] == "PREPARE_S3_KKT_WAIT")
        self.assertEqual((count["actual"], count["sampled"]), (100, 2))
        raw, _ = detail_raw(start=16)
        events, metadata = p.parse_timer(raw)
        self.assertEqual([e["work_id"] for e in events if e["stage"] == "PREPARE_S3_KKT_WAIT"], [64, 68])
        self.assertEqual(metadata[0]["sampling"]["pack_start"], 16)

    def test_empty_late_window_valid(self):
        raw, _ = detail_raw(start=48, sampled=0)
        events, metadata = p.parse_timer(raw)
        self.assertFalse(any(e["stage"] == "PREPARE_S3_KKT_WAIT" for e in events))
        count = next(c for m in metadata for c in m["iterations"] if c["stage"] == "PREPARE_S3_KKT_WAIT")
        self.assertEqual(count["sampled"], 0)
        self.assertTrue(count["truncated"])

    def test_bad_tag_and_duplicate_rejected(self):
        raw, tag = detail_raw()
        for bad in (0, raw[tag + 1]):
            with self.subTest(tag=bad):
                raw[tag] = bad
                with self.assertRaisesRegex(ValueError, "work_id"):
                    p.parse_timer(raw)

    def test_detail_disabled_rejected(self):
        raw, _ = detail_raw()
        raw[p.LAYOUT["BUFFER_WORDS_PER_KERNEL"] + 14] = 0
        with self.assertRaisesRegex(ValueError, "detail disabled"):
            p.parse_timer(raw)

    def test_controls_and_sample_padding_rejected(self):
        raw, _ = detail_raw()
        base = p.LAYOUT["BUFFER_WORDS_PER_KERNEL"]
        raw[base + 13] = 9
        with self.assertRaisesRegex(ValueError, "sampling controls"):
            p.parse_timer(raw)
        raw[base + 13] = 8
        padding = base + p.LAYOUT["HEADER_WORDS"] + p.LAYOUT["N_TIMING_COUNTER"] + p.LAYOUT["DYNAMIC_ITER_TOTAL_SIZE"] + len(p.DYNAMIC_TIMING_NAMES)
        raw[padding] = 1
        with self.assertRaisesRegex(ValueError, "count padding"):
            p.parse_timer(raw)


class TaskCorrelation(unittest.TestCase):
    def test_full_tail_and_no_output_paths(self):
        for branch in ("full", "tail", "no_output"):
            with self.subTest(branch=branch):
                rows, packs = a.task_rows(*task_fixture(branch))
                self.assertEqual((len(rows), len(packs)), (8, 3))
                self.assertEqual({r["work_id"] for r in rows}, set(range(4)))

    def test_empty_requested_tail_valid(self):
        self.assertEqual(a.task_rows(*task_fixture(start=48)), ([], []))

    def test_all_new_counts_and_events_missing_rejected(self):
        events, prepare, capture = task_fixture()
        prepare["iterations"] = []
        with self.assertRaisesRegex(ValueError, "real input work count"):
            a.task_rows([], prepare, capture)

    def test_fixed_and_varlen_work_count(self):
        _, _, capture = task_fixture()
        self.assertEqual(a.expected_task_count(capture), 4)
        capture["inputs"]["v"]["shape"][2] = 70
        capture["options"].update(cu_seqlens=[0, 0, 10, 70], chunk_indices=[1, 0, 2, 0])
        self.assertEqual(a.expected_task_count(capture), 8)
        capture["options"]["chunk_indices"] = []
        with self.assertRaisesRegex(ValueError, "chunk_indices"):
            a.expected_task_count(capture)

    def test_per_core_actual_loss_rejected(self):
        events, prepare, capture = task_fixture()
        c = next(c for c in prepare["iterations"] if c["stage"] == "PREPARE_S3_LEAF_SOLVE")
        c["actual"] -= 1
        with self.assertRaisesRegex(ValueError, "mandatory actual"):
            a.task_rows(events, prepare, capture)

    def test_branch_must_match_real_output_and_tail(self):
        events, prepare, capture = task_fixture("no_output")
        capture["options"]["disable_recompute"] = True
        with self.assertRaisesRegex(ValueError, "actual A branch"):
            a.task_rows(events, prepare, capture)
        events, prepare, capture = task_fixture("tail")
        capture["inputs"]["v"]["shape"][2] = 64
        with self.assertRaisesRegex(ValueError, "actual A branch"):
            a.task_rows(events, prepare, capture)

    def test_four_rank4_layout_aliases(self):
        for layout in ("BNSD", "NTD", "BSND", "TND"):
            events, prepare, capture = task_fixture("tail")
            capture["options"]["layout"] = layout
            if layout in ("BSND", "TND"):
                for name in ("v", "k"):
                    b, h, t, d = capture["inputs"][name]["shape"]
                    capture["inputs"][name]["shape"] = [b, t, h, d]
            with self.subTest(layout=layout):
                self.assertEqual(a.expected_task_count(capture), 4)
                rows, _ = a.task_rows(events, prepare, capture)
                self.assertTrue(all(row["valid_tokens"] == 63 for row in rows))

    def test_missing_entire_task_and_single_stage_rejected(self):
        events, prepare, capture = task_fixture()
        for filtered in ([e for e in events if not (e["core_id"] == 1 and e["work_id"] == 0 and e["stage"] != "PREPARE_PACK")],
                         [e for e in events if e["stage"] != "PREPARE_S3_KKT_WAIT"]):
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    a.task_rows(filtered, prepare, capture)

    def test_duplicate_business_id_and_order_rejected(self):
        events, prepare, capture = task_fixture()
        item = next(e for e in events if e["stage"] == "PREPARE_S3_KKT_WAIT")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            a.task_rows(events + [item], prepare, capture)
        item.update(end_cycle=1990, duration_cycles=1990-item["start_cycle"])
        with self.assertRaisesRegex(ValueError, "order"):
            a.task_rows(events, prepare, capture)

    def test_wrong_window_or_core_group_rejected(self):
        events, prepare, capture = task_fixture()
        prepare["sampling"]["pack_start"] = 16
        with self.assertRaisesRegex(ValueError, "outside selected"):
            a.task_rows(events, prepare, capture)
        prepare["sampling"]["pack_start"] = 0
        prepare["block_dim"] = 2
        for e in events:
            e["work_id"] += 4
        with self.assertRaisesRegex(ValueError, "another core"):
            a.task_rows(events, prepare, capture)

    def test_pairing_and_provenance_required(self):
        with self.assertRaisesRegex(ValueError, "matching coarse"):
            a.combine([combined_fixture()])
        c, d = combined_fixture(False), combined_fixture(True)
        result = a.combine([c, d])
        self.assertAlmostEqual(result["controls"][0]["detailed_window_change_percent"], 10)
        d["input_snapshot_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "input_snapshot"):
            a.combine([c, d])

    def test_cli_preserves_outputs_and_writes_small_files(self):
        c, d = combined_fixture(False), combined_fixture(True)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(a, "analyze_capture", side_effect=lambda name: c if str(name) == "coarse" else d), mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            out = Path(directory) / "analysis"
            self.assertEqual(a.main(["coarse", "detail", "--output-dir", str(out)]), 0)
            self.assertTrue((out / "perturbation.csv").is_file())
            self.assertTrue((out / "share_brief.txt").is_file())
            self.assertLess((out / "brief.txt").stat().st_size, 5000)
            self.assertEqual(a.main(["coarse", "detail", "--output-dir", str(out)]), 1)

    def test_real_csv_layout_capture_pipeline_and_controls(self):
        events, prepare, capture = task_fixture()
        prepare["arch"] = 310
        for count in prepare["iterations"]:
            count.update(sampled=count["actual"], truncated=False)
        for kernel in ("PREPARE", "FWD_H", "FWD_O"):
            for core in range(3):
                for stage, begin, end, kind in (("KERNEL_TIMING", 990, 2000, "window"),
                                               ("MAIN_INIT", 990, 1000, "issue"),
                                               ("MAIN_BODY", 1000, 2000, "window")):
                    events.append(dict(kernel=kernel, core_id=core, group_id=0,
                                       core_type="AIC" if core == 0 else "AIV", sub_id=core - 1,
                                       stage=stage, iteration=-1, work_id=-1, kind=kind,
                                       start_cycle=begin, end_cycle=end, duration_cycles=end - begin))
        layout = dict(layout=p.LAYOUT, kernels=[prepare] + [
            dict(kernel=kernel, block_dim=1, arch=310, iterations=[]) for kernel in ("FWD_H", "FWD_O")])
        prepare["kernel"] = "PREPARE"
        capture.update(decode_status="complete", output_comparison="ten outputs exactly match uninstrumented call",
                       prepare_sampling={k: prepare["sampling"][k] for k in ("pack_start", "pack_count", "detail_enabled")})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with (directory / "gdn_npu_time_parsed.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=p.FIELDS)
                writer.writeheader()
                writer.writerows(events)
            (directory / "timer_layout.json").write_text(json.dumps(layout), encoding="utf-8")
            path = directory / "capture_metadata.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            result = a.analyze_capture(directory)
            self.assertEqual(len(result["task_rows"]), 8)
            capture["prepare_sampling"]["pack_start"] = 1
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controls differ"):
                a.analyze_capture(directory)


class CollectorPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[8]
        spec = importlib.util.spec_from_file_location("stage35_collector", root / "tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py")
        cls.collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.collector)

    def test_default_ordinary_and_matched_plans(self):
        c = self.collector
        self.assertEqual(len(c.diagnostic_plan(c.parse_args([]))), 1)
        self.assertEqual(c.diagnostic_plan(c.parse_args(["--ordinary-only"])), [])
        args = c.parse_args(["--prepare-stage35", "--timer-repeats", "2"])
        plan = c.diagnostic_plan(args)
        self.assertEqual(len(plan), 16)
        self.assertEqual({w["pack_start"] for w in plan}, {0, 16, 32, 48})
        self.assertTrue(all(w["pack_count"] == 8 for w in plan))

    def test_bad_combinations_and_bounds(self):
        c = self.collector
        for argv in (["--ordinary-only", "--prepare-stage35"], ["--prepare-pack-starts", "-1"],
                     ["--prepare-pack-count", "9"], ["--timer-repeats", "0"]):
            with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                c.parse_args(argv)

    def test_ordinary_only_never_probes_or_captures_timer(self):
        c = self.collector
        commands = []
        def logged(command, log, env):
            commands.append(command)
            if "-scope=performance" in command:
                path = Path(env["GDN_PROFILE_INPUT_DIR"])
                path.mkdir(parents=True)
                (path / "case_0.pt").write_bytes(b"frozen-input")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            args = c.parse_args(["--ordinary-only", "--cases", "0", "--output-dir", str(out)])
            with mock.patch.dict(c.os.environ, {}, clear=True), mock.patch.object(c, "run_logged", side_effect=logged), mock.patch.object(c, "installed_vendor", return_value=Path(directory) / "vendor"), mock.patch.object(c, "verify_atk_launcher", return_value=Path(directory) / "atk"), mock.patch.object(c.shutil, "which", return_value="available"), mock.patch.object(c, "verify_profile_artifacts", return_value={"mode": "build_staging"}), mock.patch.object(c, "verify_performance_evidence", return_value={"executed_case_ids": [0]}), mock.patch.object(c, "source_commit", return_value="checkout"), mock.patch("sys.stdout"):
                self.assertEqual(c.run_profile(args), out.resolve())
            result = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertFalse((out / "timer").exists())
            self.assertFalse(any(c.SYMBOL_PROBE in cmd or any("profile_gdn.py" in str(arg) for arg in cmd) for cmd in commands))


if __name__ == "__main__":
    unittest.main()

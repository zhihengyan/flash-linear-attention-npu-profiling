"""CPU-only regression for complete-call latency aggregation (no torch/ATK)."""

import csv
import contextlib
import io
import json
from pathlib import Path
import shlex
import types
import tempfile
import unittest

import profile_gdn_latency as latency


def fixture_csv(durations):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Op Name", "OP Type", "Task Type", "Task Start Time(us)", "Task Duration(us)", "Device ID"])
    names = ["Transpose", "Cast", "ChunkGatedDeltaRuleFwdPrepare", "ChunkGatedDeltaRuleFwdH", "ChunkFwdO", "Transpose"]
    position = 0
    for call, values in enumerate(durations):
        for name, value in zip(names, values):
            # Per-launch Op Name suffixes differ; OP Type is stable.
            writer.writerow([f"{name}_{call}", name, "AI_CORE", str(position), value, "0"])
            position += 100
    stream.seek(0)
    return stream


def sample_rows(durations):
    return latency.parse_csv(fixture_csv(durations), "fixture.csv")


class GdnLatencyTest(unittest.TestCase):
    def test_counts_auxiliary_kernels_before_and_after_prepare(self):
        rows = sample_rows([[20, 30, 100, 40, 50, 60], [2, 3, 10, 4, 5, 6], [4, 6, 20, 8, 10, 12]])
        result = latency.summarize_rows(rows, warmup=1, iterations=2)
        self.assertEqual(result["kernels_per_call"], 6)
        self.assertEqual(result["prepare_us"]["mean"], 15)
        self.assertEqual(result["gdn_forward_total_us"]["mean"], 45)
        self.assertEqual([item["gdn_forward_total_us"] for item in result["samples"]], [30, 60])

    def test_unassigned_setup_kernel_is_rejected(self):
        rows = sample_rows([[20, 30, 100, 40, 50, 60], [2, 3, 10, 4, 5, 6], [2, 3, 10, 4, 5, 6]])
        extra = dict(rows[0], name="OneTimeInit", start=-100, duration=999, prepare=False, signature=("init", "aicore"))
        with self.assertRaisesRegex(ValueError, "unexpected setup kernels"):
            latency.summarize_rows([extra, *rows], warmup=1, iterations=2)

    def test_missing_call_is_rejected(self):
        rows = sample_rows([[2, 3, 10, 4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "Prepare launches"):
            latency.summarize_rows(rows, warmup=1, iterations=2)

    def test_missing_trailing_auxiliary_kernel_is_rejected(self):
        rows = sample_rows([[2, 3, 10, 4, 5, 6]] * 3)
        with self.assertRaisesRegex(ValueError, "kernel sequence"):
            latency.summarize_rows(rows[:-1], warmup=1, iterations=2)

    def test_inconsistent_measured_graph_is_rejected(self):
        rows = sample_rows([[2, 3, 10, 4, 5, 6]] * 3)
        rows[-2]["signature"] = ("unexpectedkernel", "aicore")
        with self.assertRaisesRegex(ValueError, "kernel sequence"):
            latency.summarize_rows(rows, warmup=1, iterations=2)

    def test_nonfinite_timing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sample_rows([[2, 3, float("nan"), 4, 5, 6]])

    def test_no_partial_kernel_name_filter_or_replay_in_command(self):
        options = types.SimpleNamespace(
            case_json=Path("case with spaces.json"), case_id=0, device=0,
            warmup=10, iterations=20, seed=42,
        )
        command = latency.build_msprof_command(options, Path("output directory"), "msprof")
        self.assertNotIn("op", command)
        self.assertFalse(any("kernel-name" in value or "mstx" in value for value in command))
        application = shlex.split(next(value.split("=", 1)[1] for value in command if value.startswith("--application=")))
        self.assertIn("--worker", application)
        self.assertIn(str(options.case_json.resolve()), application)
        self.assertIn("--task-time=on", command)

    def test_single_call_can_be_summarized(self):
        result = latency.summarize_rows(sample_rows([[2, 3, 10, 4, 5, 6]]), warmup=0, iterations=1)
        self.assertEqual(result["prepare_us"]["mean"], 10)
        self.assertEqual(result["gdn_forward_total_us"]["mean"], 30)

    def test_report_discovery_and_output(self):
        base = Path(__file__).resolve().parent / "runs"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="latency_test_", dir=base) as name:
            output = Path(name).resolve()
            self.assertTrue(output.is_relative_to(base.resolve()))
            report_dir = output / "profile/PROF_test/mindstudio_profiler_output"
            report_dir.mkdir(parents=True)
            (report_dir / "op_summary_0.csv").write_text(
                fixture_csv([[2, 3, 10, 4, 5, 6]] * 3).getvalue(), encoding="utf-8",
            )
            (output / "run_manifest.json").write_text(json.dumps({
                "warmup": 1, "iterations": 2, "completed_calls": 3,
                "case_id": 0, "case_name": "test", "case_json_sha256": "test",
            }), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                latency.summarize(output)
            saved = json.loads((output / "latency_summary.json").read_text())
            self.assertEqual(saved["prepare_us"]["mean"], 10)
            self.assertEqual(saved["gdn_forward_total_us"]["mean"], 30)
            self.assertIn("Prepare mean:", stdout.getvalue())
            self.assertIn("GDN forward total mean:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)

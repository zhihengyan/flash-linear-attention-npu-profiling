"""Compact exports preserve repeat variation and do not require device data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compact_prepare_stage35 as c


def fixture():
    controls = []
    stages = []
    for start in (0, 16, 32, 48):
        for side in ("AIC", "AIV"):
            controls.append(dict(pack_start=start, pack_count=8, core_type=side,
                                 coarse_median_cycles=100, detail_median_cycles=110,
                                 coarse_min_cycles=99, coarse_max_cycles=101,
                                 detail_min_cycles=109, detail_max_cycles=111,
                                 coarse_captures=2, detail_captures=2))
        for repeat, (median, calls) in enumerate(((10, 1), (100, 100))):
            stages.append(dict(capture=f"detail_pack{start}_r{repeat}", pack_start=start,
                               pack_count=8, detail_enabled=True, core_type="AIV",
                               stage="PREPARE_S3_LEAF_SOLVE", sampled_calls=calls,
                               call_median_cycles=median, call_p95_cycles=median*2,
                               call_max_cycles=median*3))
    return dict(schema="gdn-prepare-stage35/v1", controls=controls, stages=stages,
                inputs={"v": {"shape": [1, 32, 11274, 128]}}, options={"layout": "BNSD"},
                input_snapshot_sha256="input", header_sha256="header")


def run_metadata():
    rows = []
    for label, values in (("ChunkGatedDeltaRuleFwdPrepare", [10, 10]),
                          ("ChunkFwdH", [20, 20]), ("ChunkFwdO", [30, 30]),
                          ("Cast", [1, 1]), ("Transpose", [2, 3, 4, 2, 3, 4])):
        rows += [dict(name=["", "aclnnChunkGatedDeltaRuleFwd_" + label, "", label, ""],
                      task_duration_us=value) for value in values]
    return dict(case_ids=[0], input_sha256={"0": "input"},
                performance_evidence=dict(executed_case_ids=[0], target_kernels=rows),
                reference_wheel=dict(sha256="wheel"))


class CompactExport(unittest.TestCase):
    def test_repeat_statistics_are_not_pooled_and_input_is_unchanged(self):
        result = fixture()
        before = copy.deepcopy(result)
        text = c.render_compact(result)
        self.assertIn("m:55[10..100],p:200,x:300,n:101", text)
        self.assertIn("not pooled quantiles", text)
        self.assertEqual(result, before)
        result["stages"] = [r for r in result["stages"] if r["pack_start"] != 48]
        self.assertTrue(next(line for line in c.render_compact(result).splitlines() if line.startswith("S3_LEAF_SOLVE/")).endswith(" / -"))

    def test_ordinary_counts_and_mean_sum(self):
        text = c.render_compact(fixture(), run_metadata())
        self.assertIn("Prepare 2 10/10/10/10/10", text)
        self.assertIn("Seven-kernel mean sum=70us", text)
        self.assertIn("no total p95", text)
        self.assertIn("wheel_sha256=wheel", text)

    def test_ordinary_identity_and_incomplete_counts(self):
        meta = run_metadata()
        meta["input_sha256"]["0"] = "wrong"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            c.render_compact(fixture(), meta)
        meta = run_metadata()
        meta["performance_evidence"]["target_kernels"].pop()
        self.assertIn("mean sum unavailable", c.render_compact(fixture(), meta))
        meta = run_metadata()
        meta["case_ids"] = [0, 1]
        self.assertIn("multi-case", c.render_compact(fixture(), meta))

    def test_invalid_schema_and_nonfinite_values_rejected(self):
        result = fixture()
        result["schema"] = "unknown"
        with self.assertRaisesRegex(ValueError, "summary.json"):
            c.render_compact(result)
        for bad in (float("nan"), float("inf"), -1):
            result = fixture()
            result["stages"][0]["call_median_cycles"] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "finite nonnegative"):
                c.render_compact(result)

    def test_full_default_stage_catalog_stays_small(self):
        result = fixture()
        template = result["stages"][0]
        result["stages"] = []
        for stage in range(32):
            for start in (0, 16, 32, 48):
                for repeat in range(2):
                    result["stages"].append(dict(template, stage=f"PREPARE_STAGE{stage}",
                                                 pack_start=start, capture=f"detail_{start}_{repeat}"))
        text = c.render_compact(result, run_metadata())
        self.assertLessEqual(len(text.splitlines()), 70)
        self.assertLess(len(text.encode("utf-8")), 12000)

    def test_existing_summary_only_cli_and_preserved_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "summary.json").write_text(json.dumps(fixture()), encoding="utf-8")
            meta = directory / "run_metadata.json"
            meta.write_text(json.dumps(run_metadata()), encoding="utf-8")
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(c.main([str(directory), "--run-metadata", str(meta)]), 0)
                self.assertEqual(c.main([str(directory)]), 1)
            path = directory / "share_brief.txt"
            self.assertIn("mean sum=70us", path.read_text(encoding="utf-8"))
            self.assertFalse((directory / "gdn_npu_time_raw.csv").exists())


if __name__ == "__main__":
    unittest.main()

"""Check the A5 profiling entry's orchestration with fake device commands."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py"
spec = importlib.util.spec_from_file_location("gdn_profile_performance", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@contextmanager
def fixture_directory():
    path = ROOT / (".gdn_profile_runner_test_" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().parent == ROOT.resolve()
        shutil.rmtree(path)


class GdnProfilePerformanceTest(unittest.TestCase):
    def fake_pipeline(self, args, fixture, *, atk_failure=False, missing_input=False, wrong_atk=False):
        self.commands = []
        vendor = fixture / "vendor"
        vendor.mkdir()
        self.child_env = None

        def run_logged(command, log_path, env):
            self.commands.append(command)
            self.child_env = env.copy()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("mock device command\n", encoding="utf-8")
            if command[0] == "bash":
                if atk_failure:
                    raise subprocess.CalledProcessError(7, command)
                if not missing_input:
                    inputs = Path(env["GDN_PROFILE_INPUT_DIR"])
                    inputs.mkdir()
                    for case_id in args.cases:
                        (inputs / f"case_{case_id}.pt").write_bytes(b"mock ATK tensor snapshot")
            if "--inputs" in command:
                output = Path(command[command.index("--output-dir") + 1])
                names = ["gdn_npu_time_raw.csv", "gdn_npu_time_parsed.csv",
                         "timer_layout.json", "capture_metadata.json"]
                if "--shared-clock" in command:
                    names.append("gdn_npu_time.json")
                for name in names:
                    (output / name).write_text("mock capture result\n", encoding="utf-8")

        with patch.object(runner, "run_logged", side_effect=run_logged), \
             patch.object(runner, "verify_atk_launcher", side_effect=RuntimeError("ATK launcher must use the current Python environment") if wrong_atk else None,
                          return_value=Path(sys.executable).parent / "atk"), \
             patch.object(runner, "installed_vendor", return_value=vendor), \
             patch.object(runner, "source_commit", return_value="test-source-commit"), \
             patch.object(runner.shutil, "which", return_value="mock-tool"), \
             patch.dict(os.environ, {"PATH": "", "ATK_ENV": sys.prefix,
                 "FLA_NPU_SOC": "ascend910b", "FLA_NPU_ENV": "old-opp-env", "FLA_NPU_OPP_PATH": "different-debug-opp",
                 "CANN_ENV": "old-cann-env", "CASE_START": "400", "CASE_END": "500",
                 "PERFORMANCE_START": "400", "PERFORMANCE_END": "500"}, clear=True):
            return runner.run_profile(args)

    def test_default_a5_two_cases_and_invalid_clock_arguments(self):
        args = runner.parse_args([])
        self.assertEqual((runner.SOC, args.cases, args.device, args.warmup),
                         ("ascend950", [0, 1], 0, 5))
        self.assertIsNone(args.cycles_per_us)
        for argv in (["--device", "-1"], ["--warmup", "-1"], ["--shared-clock"],
                     ["--cycles-per-us", "0"], ["--cycles-per-us", "nan"],
                     ["--cycles-per-us", "inf"], ["--cases", "2"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_two_cases_have_new_inputs_logs_and_exact_capture_calls(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output), "--device", "3"])
            self.assertEqual(self.fake_pipeline(args, fixture), output)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual((metadata["status"], metadata["soc"], metadata["device"]),
                             ("complete", "ascend950", 3))
            self.assertEqual(metadata["case_ids"], [0, 1])
            self.assertEqual(set(metadata["input_sha256"]), {"0", "1"})
            self.assertEqual((output / "performance_cases.json").read_bytes(),
                             (runner.OP_DIR / "atk_chunk_gated_delta_rule_fwd_perf.json").read_bytes())
            captures = [command for command in self.commands if "--inputs" in command]
            self.assertEqual(len(captures), 2)
            for case_id, command in enumerate(captures):
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[command.index("--inputs") + 1], str(output / "inputs" / f"case_{case_id}.pt"))
                self.assertIn("--check-output", command)
                self.assertNotIn("--shared-clock", command)
                self.assertFalse((output / "timer" / f"case_{case_id}" / "gdn_npu_time.json").exists())
            self.assertEqual(self.child_env["FLA_NPU_SOC"], "ascend950")
            self.assertEqual(self.child_env["PERFORMANCE_START"], "")
            self.assertEqual(self.child_env["CASE_START"], "")
            self.assertEqual(self.child_env["GDN_PROFILE_INPUT_DIR"], str(output / "inputs"))
            for name in ("ATK_ENV", "CANN_ENV", "FLA_NPU_ENV", "FLA_NPU_OPP_PATH"):
                self.assertNotIn(name, self.child_env)
            atk = next(command for command in self.commands if command[0] == "bash")
            self.assertIn("-soc=ascend950", atk)
            self.assertIn("-scope=performance", atk)
            self.assertTrue(any(command[:2] == [sys.executable, "-c"] and command[2] == runner.ATK_PROBE
                                for command in self.commands))

    def test_atk_launcher_accepts_current_python_and_env_shebang(self):
        with fixture_directory() as fixture:
            python = fixture / "python"
            python.touch()
            launcher = fixture / "atk"
            for shebang in (f"#!{python.as_posix()}", "#!/usr/bin/env python3", "#!/usr/bin/env -S python3"):
                with self.subTest(shebang=shebang):
                    launcher.write_text(shebang + "\n", encoding="utf-8")
                    with patch.object(runner.sys, "executable", str(python)), \
                         patch.object(runner.shutil, "which", side_effect=lambda name, **kwargs: str(launcher if name == "atk" else python)):
                        self.assertEqual(runner.verify_atk_launcher({"PATH": ""}), launcher)

    def test_atk_launcher_rejects_other_environment_or_unknown_interpreter(self):
        with fixture_directory() as fixture:
            current = fixture / "current"
            other = fixture / "other"
            current.mkdir()
            other.mkdir()
            python = current / "python"
            python.touch()
            for directory, shebang in ((other, f"#!{python.as_posix()}"), (current, f"#!{(other / 'python').as_posix()}"),
                                       (current, "#!/bin/sh"), (current, "no shebang")):
                with self.subTest(directory=directory, shebang=shebang):
                    launcher = directory / "atk"
                    launcher.write_text(shebang + "\n", encoding="utf-8")
                    with patch.object(runner.sys, "executable", str(python)), \
                         patch.object(runner.shutil, "which", return_value=str(launcher)), \
                         self.assertRaisesRegex(RuntimeError, "current Python environment"):
                        runner.verify_atk_launcher({"PATH": ""})

    def test_wrong_atk_launcher_stops_before_device_checks(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output)])
            with self.assertRaisesRegex(RuntimeError, "ATK launcher"):
                self.fake_pipeline(args, fixture, wrong_atk=True)
            self.assertEqual(self.commands, [])
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["status"], "failed")

    def test_one_case_and_verified_clock_are_forwarded(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output), "--cases", "1",
                                     "--cycles-per-us", "123", "--shared-clock", "--warmup", "2"])
            self.fake_pipeline(args, fixture)
            self.assertEqual((self.child_env["PERFORMANCE_START"], self.child_env["PERFORMANCE_END"]), ("1", "2"))
            captures = [command for command in self.commands if "--inputs" in command]
            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0][captures[0].index("--cycles-per-us") + 1], "123.0")
            self.assertEqual(captures[0][captures[0].index("--warmup") + 1], "2")
            self.assertTrue((output / "timer/case_1/gdn_npu_time.json").is_file())

    def test_atk_failure_stops_before_capture_and_records_failure(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output)])
            with self.assertRaises(subprocess.CalledProcessError):
                self.fake_pipeline(args, fixture, atk_failure=True)
            self.assertFalse(any("--inputs" in command for command in self.commands))
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["status"], "failed")

    def test_missing_snapshot_stops_before_capture(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output)])
            with self.assertRaisesRegex(RuntimeError, "ATK did not export"):
                self.fake_pipeline(args, fixture, missing_input=True)
            self.assertFalse(any("--inputs" in command for command in self.commands))

    def test_existing_directory_is_preserved(self):
        with fixture_directory() as fixture:
            sentinel = fixture / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            args = runner.parse_args(["--output-dir", str(fixture)])
            with patch.object(runner, "run_logged") as run, self.assertRaisesRegex(ValueError, "new output directory"):
                runner.run_profile(args)
            run.assert_not_called()
            self.assertEqual(sentinel.read_text(), "keep")

    def test_different_atk_environment_is_rejected_before_device_checks(self):
        with fixture_directory() as fixture:
            output = fixture / "result"
            args = runner.parse_args(["--output-dir", str(output)])
            with patch.dict(os.environ, {"ATK_ENV": str(fixture / "other-python")}), \
                 patch.object(runner, "run_logged") as run, \
                 patch.object(runner, "source_commit", return_value=None), \
                 self.assertRaisesRegex(RuntimeError, "Activate ATK_ENV"):
                runner.run_profile(args)
            run.assert_not_called()
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["status"], "failed")

    def test_file_hash_streams_multiple_chunks(self):
        with fixture_directory() as fixture:
            path = fixture / "input.pt"
            data = b"x" * (2 * 1024 * 1024 + 17)
            path.write_bytes(data)
            self.assertEqual(runner.file_sha256(path), hashlib.sha256(data).hexdigest())

    def test_logged_command_failure_keeps_output_and_nonzero_exit(self):
        with fixture_directory() as fixture:
            log = fixture / "failure.log"
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                runner.run_logged([sys.executable, "-c", "print('diagnostic evidence'); raise SystemExit(7)"], log, os.environ.copy())
            self.assertEqual(raised.exception.returncode, 7)
            self.assertIn("diagnostic evidence", log.read_text())


if __name__ == "__main__":
    unittest.main()

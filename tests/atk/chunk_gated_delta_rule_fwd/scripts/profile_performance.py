#!/usr/bin/env python3
"""Run the existing A5 ATK performance cases and capture their per-core GDN timers."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[4]
OP_DIR = REPO_ROOT / "tests/atk/chunk_gated_delta_rule_fwd"
UT_DIR = REPO_ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut"
SOC = "ascend950"

ATK_PROBE = """
import atk
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
print('ATK module:', atk.__file__)
print('Executor ATK imports: OK')
"""

SYMBOL_PROBE = """
import fla_npu
from fla_npu.ops.ascendc import _aclnn_ctypes
print('fla_npu:', fla_npu.__file__)
print('adapter:', _aclnn_ctypes.__file__)
assert hasattr(_aclnn_ctypes, '_profile_chunk_gated_delta_rule_fwd'), 'Python adapter is outdated'
libraries = fla_npu.load_ascendc_opapi_libraries()
for library in libraries:
    print('library:', library._name)
for name in ('aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize', 'aclnnChunkGatedDeltaRuleFwdProfile'):
    assert any(hasattr(library, name) for library in libraries), 'Missing symbol: ' + name
print('Profile symbols: OK')
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cases", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, help="a new directory; defaults to ignored atk_output/")
    parser.add_argument("--cycles-per-us", type=float, help="verified GetSystemCycle frequency")
    parser.add_argument("--shared-clock", action="store_true", help="assert verified common AIC/AIV clock")
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0:
        parser.error("device and warmup must be nonnegative")
    if args.cycles_per_us is not None and (not math.isfinite(args.cycles_per_us) or args.cycles_per_us <= 0):
        parser.error("cycles-per-us must be a finite positive verified value")
    if args.shared_clock and args.cycles_per_us is None:
        parser.error("shared-clock requires cycles-per-us")
    args.cases = sorted(set(args.cases))
    return args


def run_logged(command, log_path, env):
    print("+", shlex.join(str(arg) for arg in command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
    if result:
        raise subprocess.CalledProcessError(result, command)


def installed_vendor(env):
    # Locate the package without initializing CANN before the OPP path is set.
    probe = """
from importlib.util import find_spec
from pathlib import Path
spec = find_spec('fla_npu')
if spec is None or spec.origin is None:
    raise SystemExit('Current Python environment has no installed fla_npu')
print(Path(spec.origin).resolve().parent / 'opp/vendors/fla_npu_transformer')
"""
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
                            text=True, capture_output=True, check=True)
    vendor = Path(result.stdout.strip())
    if not vendor.is_dir():
        raise RuntimeError(f"Installed wheel has no OPP vendor directory: {vendor}")
    return vendor


def source_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_atk_launcher(env):
    launcher = Path(shutil.which("atk", path=env["PATH"]))
    python = Path(sys.executable)
    mismatch = "ATK launcher must use the current Python environment; activate it or reinstall ATK there."
    if launcher.resolve().parent != python.parent.resolve():
        raise RuntimeError(mismatch)
    with launcher.open("rb") as stream:
        first_line = stream.readline(4096).decode("utf-8").strip()
    if not first_line.startswith("#!"):
        raise RuntimeError(mismatch)
    command = shlex.split(first_line[2:])
    if command and Path(command[0]).name == "env":
        command = command[1:]
        if command and command[0] == "-S":
            command = command[1:]
        interpreter = shutil.which(command[0], path=env["PATH"]) if command else None
    else:
        interpreter = command[0] if command else None
    if not interpreter or Path(interpreter).parent.resolve() != python.parent.resolve() or Path(interpreter).resolve() != python.resolve():
        raise RuntimeError(mismatch)
    return launcher


def run_profile(args):
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    output = (args.output_dir or OP_DIR / "atk_output" / ("gdn_timer_" + tag)).resolve()
    if output.exists():
        raise ValueError(f"Choose a new output directory; existing results are preserved: {output}")
    output.mkdir(parents=True)
    metadata_path = output / "run_metadata.json"
    metadata = {"status": "running", "soc": SOC, "python": sys.executable,
                "device": args.device, "case_ids": args.cases, "warmup": args.warmup,
                "cycles_per_us": args.cycles_per_us, "shared_clock_asserted": args.shared_clock,
                "source_commit": source_commit()}

    def write_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_metadata()
    print("Result directory:", output, flush=True)
    try:
        env = os.environ.copy()
        atk_env = env.get("ATK_ENV")
        if atk_env and Path(atk_env).resolve() != Path(sys.prefix).resolve():
            raise RuntimeError("Activate ATK_ENV before building, installing and running this script; Python must match.")
        # Use the already activated container environment and the installed wheel.
        for name in ("ATK_ENV", "CANN_ENV", "FLA_NPU_ENV", "FLA_NPU_OPP_ENV", "FLA_NPU_OPP_PATH"):
            env.pop(name, None)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(OP_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["FLA_NPU_SOC"] = SOC
        env["GDN_PROFILE_INPUT_DIR"] = str(output / "inputs")
        env["ATK_OUTPUT_ROOT"] = str(output / "atk")
        for name in ("CASE_START", "CASE_END", "PERFORMANCE_START", "PERFORMANCE_END"):
            env[name] = ""
        if len(args.cases) == 1:
            env["PERFORMANCE_START"] = str(args.cases[0])
            env["PERFORMANCE_END"] = str(args.cases[0] + 1)
        for executable in ("bash", "atk", "npu-smi"):
            if shutil.which(executable, path=env["PATH"]) is None:
                raise RuntimeError(f"Missing {executable} in the activated container environment")
        metadata["atk_launcher"] = str(verify_atk_launcher(env))
        vendor = installed_vendor(env)
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + ":" + str(vendor / "op_api/lib") + ":" + env.get("ASCEND_CUSTOM_OPP_PATH", "")
        metadata["opp_vendor"] = str(vendor)
        case_json = OP_DIR / "atk_chunk_gated_delta_rule_fwd_perf.json"
        shutil.copyfile(case_json, output / "performance_cases.json")
        metadata["case_json_sha256"] = file_sha256(case_json)
        write_metadata()
        checks = [
            (["atk", "--version"], "atk_version.log"),
            ([sys.executable, "-c", ATK_PROBE], "atk_environment.log"),
            (["npu-smi", "info"], "npu_info.log"),
            ([sys.executable, str(REPO_ROOT / "scripts/check_npu_env.py")], "environment.log"),
            ([sys.executable, str(REPO_ROOT / "scripts/verify_libcust_opapi_md5.py"), "--python"], "installed_artifacts.log"),
            ([sys.executable, "-c", SYMBOL_PROBE], "profile_symbols.log"),
        ]
        for command, filename in checks:
            run_logged(command, output / filename, env)
        run_logged(["bash", str(REPO_ROOT / "tests/atk/run_test_cpu.sh"),
                    "-op=chunk_gated_delta_rule_fwd", f"-npu_device_id={args.device}",
                    f"-soc={SOC}", "-scope=performance"], output / "atk_performance.log", env)
        snapshots = {case_id: output / "inputs" / f"case_{case_id}.pt" for case_id in args.cases}
        for path in snapshots.values():
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"ATK did not export {path}; check worker environment and executor version")
        metadata["input_sha256"] = {str(case_id): file_sha256(path)
                                    for case_id, path in snapshots.items()}
        write_metadata()
        for case_id, snapshot in snapshots.items():
            case_output = output / "timer" / f"case_{case_id}"
            command = [sys.executable, str(UT_DIR / "profile_gdn.py"), "--inputs", str(snapshot),
                       "--device", str(args.device), "--warmup", str(args.warmup), "--check-output",
                       "--output-dir", str(case_output)]
            if args.cycles_per_us is not None:
                command += ["--cycles-per-us", str(args.cycles_per_us)]
            if args.shared_clock:
                command += ["--shared-clock"]
            run_logged(command, case_output / "profiling.log", env)
            expected = ["gdn_npu_time_raw.csv", "gdn_npu_time_parsed.csv", "timer_layout.json", "capture_metadata.json"]
            if args.shared_clock:
                expected.append("gdn_npu_time.json")
            for filename in expected:
                path = case_output / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Profiling output is missing or empty: {path}")
        metadata["status"] = "complete"
        write_metadata()
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        write_metadata()
        raise
    print("Profiling complete:", output, flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    try:
        run_profile(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("Profiling failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Profile each frozen A5 model case with CANN msopprof, without ATK.

The parent process runs msopprof once per case; the worker only launches GDN.
Statistics use OpBasicInfo*.csv Task Duration(us), never Python wall time.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from accuracy_a5 import (
    CASE_DIR, PERF_JSON, add_case_arguments, build_cpu_inputs, case_description,
    case_seed, collect_outputs, npu_runtime, prepare_call, select_cases,
    sha256, source_metadata, write_json,
)


KERNEL_FILTER = "chunk_gated_delta_rule_fwd*|ChunkGatedDeltaRuleFwd*"


def matches_kernel(name, patterns):
    # The public prepare / state operators share the fused kernel's prefix.
    # Never report their durations as fused GDN, even under a broad filter.
    for sibling in ("chunk_gated_delta_rule_fwd_prepare", "chunk_gated_delta_rule_fwd_h",
                    "ChunkGatedDeltaRuleFwdPrepare", "ChunkGatedDeltaRuleFwdH"):
        if name == sibling or name.startswith(sibling + "_"):
            return False
    return any(fnmatch.fnmatchcase(name, part if "*" in part else part + "*")
               for part in patterns.split("|"))


def parse_durations(output_dir, kernel_filter, expected_count):
    """Support single-task and per-task CSVs, including timestamp suffixes.

    Unknown schemas, empty collections and extra/missing samples are errors.
    Keep the source row for every sample so the reported statistic is auditable.
    """
    samples = []
    csv_files = [path for path in Path(output_dir).rglob("*.csv")
                 if path.stem == "OpBasicInfo" or path.stem.startswith("OpBasicInfo_")]
    for path in sorted(csv_files):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {field.strip(): field for field in (reader.fieldnames or [])}
            required = ("Op Name", "Task Duration(us)")
            if not all(field in fields for field in required):
                raise RuntimeError(f"unsupported OpBasicInfo schema in {path}: {reader.fieldnames}")
            for line, raw in enumerate(reader, start=2):
                row = {key.strip(): value for key, value in raw.items() if key is not None}
                if not matches_kernel(row["Op Name"].strip(), kernel_filter):
                    continue
                duration = float(row["Task Duration(us)"].strip())
                if not math.isfinite(duration) or duration <= 0:
                    raise RuntimeError(f"invalid Task Duration(us) in {path}:{line}")
                samples.append({"duration_us": duration, "csv": str(path.resolve()),
                                "line": line, "raw": row})
    if len(samples) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} target samples, found {len(samples)} in {output_dir}; "
            "check msopprof.log, kernel filter, actual CSV schema and worker.json"
        )
    values = [sample["duration_us"] for sample in samples]
    return {
        "sample_count": len(values), "mean_us": statistics.mean(values),
        "median_us": statistics.median(values), "min_us": min(values), "max_us": max(values),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": samples,
    }


def build_command(args, record, case_dir, executable):
    application = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--case-file", str(args.case_file.resolve()), "--case-ids", str(record["id"]),
        "--device", str(args.device), "--seed-base", str(args.seed_base),
        "--iterations", str(args.iterations), "--output-dir", str(case_dir.resolve()),
    ]
    # msopprof --application takes a single POSIX command string. shlex.join
    # quotes each application argument; the parent never invokes a shell.
    return [
        executable, "--application=" + shlex.join(application),
        "--output=" + str((case_dir / "profile").resolve()),
        "--kernel-name=" + args.kernel_name, "--launch-count=" + str(args.iterations),
        "--warm-up=" + str(args.warm_up), "--aic-metrics=" + args.aic_metrics, "--kill=off",
    ]


def run_worker(args, records):
    if len(records) != 1:
        raise ValueError("a profiling worker must select exactly one case")
    torch, ascendc, environment = npu_runtime(args.device)
    torch.set_num_threads(1)
    record = records[0]
    seed = case_seed(record, args.seed_base)
    with torch.no_grad():
        case, inputs, state = build_cpu_inputs(record, seed)
        call, npu_inputs = prepare_call(ascendc, case, inputs, state, args.device)
        # Input generation/transfers and host chunk indices precede the measured
        # launches. Kernel-name filtering excludes their auxiliary device work.
        torch.npu.synchronize()
        outputs = None
        for _ in range(args.iterations):
            outputs = call()
            torch.npu.synchronize()
        actual = collect_outputs(outputs, case, npu_inputs)
        finite = {name: bool(torch.isfinite(value).all()) for name, value in actual.items()}
    metadata = {
        **source_metadata(args.case_file), **environment,
        "perf_runner_sha256": sha256(__file__), "case": case_description(record),
        "seed": seed, "launches": args.iterations, "finite": finite,
        "disable_recompute": True, "layout": "BNSD", "use_exp2": False,
    }
    write_json(args.output_dir / "worker.json", metadata)
    if not all(finite.values()):
        raise RuntimeError("GDN returned NaN/Inf; see worker.json")
    print(f"GDN_WORKER_DONE case={record['id']} launches={args.iterations}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser, PERF_JSON)
    parser.add_argument("--iterations", type=int, default=10, help="target launches / samples per case")
    parser.add_argument("--warm-up", type=int, default=10, help="msopprof's internal warm-up, not Python launches")
    parser.add_argument("--aic-metrics", default="BasicInfo", help="BasicInfo for latency; Default for bottlenecks")
    parser.add_argument("--kernel-name", default=KERNEL_FILTER)
    parser.add_argument("--msopprof", default="msopprof", help="tool executable or absolute path")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "runs" / "standalone_perf")
    parser.add_argument("--dry-run", action="store_true", help="print commands; no torch, NPU or tool needed")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 5000 or not 0 <= args.warm_up <= 500:
        parser.error("--iterations must be 1..5000 and --warm-up must be 0..500")
    records = select_cases(args)
    if args.list_cases:
        for record in records:
            print(json.dumps(case_description(record), ensure_ascii=False))
        return 0
    if args.worker:
        return run_worker(args, records)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_dir.resolve() / run_id
    if args.dry_run:
        for record in records:
            command = build_command(args, record, run_dir / f"case_{record['id']}", args.msopprof)
            print(shlex.join(command))
        return 0
    if os.name != "posix":
        parser.error("actual msopprof collection requires the Linux Ascend host")
    executable = shutil.which(args.msopprof)
    if executable is None:
        parser.error("msopprof not found; source CANN set_env.sh or use --msopprof /absolute/path/msopprof")
    run_dir.mkdir(parents=True, exist_ok=False)
    version = subprocess.run([executable, "--version"], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    write_json(run_dir / "environment.json", {
        **source_metadata(args.case_file), "perf_runner_sha256": sha256(__file__),
        "msopprof_version": version.stdout, "metric": "Task Duration(us)",
        "warm_up": args.warm_up, "iterations": args.iterations,
        "aic_metrics": args.aic_metrics, "kernel_filter": args.kernel_name,
    })
    results = []
    for record in records:
        case_dir = run_dir / f"case_{record['id']}"
        case_dir.mkdir()
        command = build_command(args, record, case_dir, executable)
        write_json(case_dir / "command.json", {"argv": command, "case": case_description(record)})
        print(f"Profiling case={record['id']} -> {case_dir}", flush=True)
        result = {"case": case_description(record), "directory": str(case_dir)}
        with (case_dir / "msopprof.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        try:
            if completed.returncode:
                raise RuntimeError(f"msopprof exited {completed.returncode}; see {case_dir / 'msopprof.log'}")
            worker_path = case_dir / "worker.json"
            if not worker_path.is_file():
                raise RuntimeError("worker did not finish; see msopprof.log")
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            if worker["launches"] != args.iterations or not all(worker["finite"].values()):
                raise RuntimeError("worker launch count or finite-output check failed")
            result.update(parse_durations(case_dir / "profile", args.kernel_name, args.iterations))
            result["collected"] = True
            print(f"case={record['id']} samples={result['sample_count']} "
                  f"mean={result['mean_us']:.3f} us median={result['median_us']:.3f} us", flush=True)
        except Exception as exc:
            result.update(collected=False, error=f"{type(exc).__name__}: {exc}")
            print(result["error"], file=sys.stderr, flush=True)
        results.append(result)
        write_json(run_dir / "performance.json", {"results": results})
    fields = ("case_id", "name", "collected", "sample_count", "mean_us", "median_us", "min_us", "max_us", "stdev_us")
    with (run_dir / "performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"case_id": result["case"]["id"], "name": result["case"]["name"],
                             **{key: result.get(key, "") for key in fields[2:]}})
    print(f"Reports: {run_dir}")
    return int(any(not result["collected"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect one GDN forward case and report Prepare and full-chain kernel time.

Uses ordinary msprof, not msprof op replay. The repository's total-time metric
is the sum of Task Duration(us) for ALL kernels in one forward API call.
No kernel-name filter or MSTX is used. Input construction reuses the frozen
forward performance case builder in scripts/profile_msprof.py.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys

OP_DIR = Path(__file__).resolve().parent
PREPARE = "chunkgateddeltarulefwdprepare"


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def parse_csv(stream, source):
    reader = csv.DictReader(stream)
    columns = {normalized(name): name for name in reader.fieldnames or ()}
    required = ("opname", "taskstarttimeus", "taskdurationus")
    if any(name not in columns for name in required):
        raise ValueError(f"{source}: expected Op Name, Task Start Time(us), Task Duration(us)")
    rows = []
    for line, record in enumerate(reader, 2):
        if not any(record.values()):
            continue
        try:
            start = Decimal(record[columns["taskstarttimeus"]].strip())
            duration = float(record[columns["taskdurationus"]])
        except (ValueError, InvalidOperation, TypeError, AttributeError) as exc:
            raise ValueError(f"{source}:{line}: invalid task timing") from exc
        if not start.is_finite() or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{source}:{line}: non-finite or negative task timing")
        name = record[columns["opname"]]
        op_type = record.get(columns.get("optype", ""), "")
        task_type = record.get(columns.get("tasktype", ""), "")
        # Op names can have per-launch numeric suffixes; Op Type is stable.
        type_name = normalized(op_type)
        signature = type_name if type_name not in ("", "na", "none", "null") else re.sub(r"\d+", "#", name.lower())
        rows.append({
            "name": name, "type": op_type, "start": start, "duration": duration,
            "signature": (signature, normalized(task_type)),
            "prepare": PREPARE in normalized(name + op_type),
            "device": record.get(columns.get("deviceid", ""), ""),
        })
    return rows


def summarize_rows(rows, warmup, iterations):
    """Validate whole repeated calls, retaining auxiliary kernels at BOTH ends.

    Prepare occurs once per public forward call in this A5 implementation.
    All captured compute kernels must form complete repeated call sequences;
    unexpected setup kernels or changing call graphs are rejected, not trimmed.
    """
    rows = sorted(rows, key=lambda item: item["start"])
    if not rows:
        raise ValueError("empty op_summary: no latency result can be computed")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    devices = {row["device"] for row in rows if row["device"]}
    if len(devices) > 1:
        raise ValueError("multiple devices in op_summary; use one fresh run directory per case")
    positions = [index for index, row in enumerate(rows) if row["prepare"]]
    expected = warmup + iterations
    if len(positions) != expected:
        raise ValueError(
            f"expected {expected} Prepare launches, found {len(positions)}; "
            "the worker failed, data is incomplete, or this kernel graph is unsupported"
        )
    if len(rows) % expected:
        raise ValueError("incomplete full-call kernel sequence or unexpected setup kernels; refusing a partial total")
    kernels_per_call = len(rows) // expected
    reference = [row["signature"] for row in rows[-kernels_per_call:]]
    samples = []
    for index in range(expected):
        group = rows[index * kernels_per_call:(index + 1) * kernels_per_call]
        if [row["signature"] for row in group] != reference or sum(row["prepare"] for row in group) != 1:
            raise ValueError("inconsistent full-call kernel sequence; refusing a partial total")
        if index >= warmup:
            samples.append({
                "prepare_us": sum(row["duration"] for row in group if row["prepare"]),
                "gdn_forward_total_us": sum(row["duration"] for row in group),
            })
    result = {
        "metric": "sum of op_summary Task Duration(us) per complete forward API call",
        "warmup": warmup, "iterations": iterations, "kernels_per_call": kernels_per_call,
        "samples": samples,
    }
    for name in ("prepare_us", "gdn_forward_total_us"):
        values = [sample[name] for sample in samples]
        result[name] = {
            "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "stdev": statistics.pstdev(values),
        }
    return result


def summarize(output):
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["warmup"] + manifest["iterations"]
    if manifest.get("completed_calls") != expected:
        raise ValueError("worker did not complete every requested API call")
    paths = sorted((output / "profile").rglob("op_summary*.csv"))
    if not paths:
        raise ValueError(f"no op_summary CSV under {output / 'profile'}; see msprof.log")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(parse_csv(stream, str(path)))
    # Exporting the same profile twice must not silently double the total.
    identities = [(row["device"], row["start"], row["name"], row["type"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate task rows across CSVs; use a fresh profiling output directory")
    result = summarize_rows(rows, manifest["warmup"], manifest["iterations"])
    result.update({"case_id": manifest["case_id"], "case_name": manifest["case_name"],
                   "case_json_sha256": manifest["case_json_sha256"],
                   "op_summary_files": [str(path) for path in paths]})
    (output / "latency_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Prepare mean:          {result['prepare_us']['mean']:.3f} us")
    print(f"GDN forward total mean: {result['gdn_forward_total_us']['mean']:.3f} us")
    print(f"Samples: {manifest['iterations']} (excluded {manifest['warmup']} warmup calls)")
    print(f"Result: {output / 'latency_summary.json'}")
    return result


def worker(options, output):
    # The existing builder generates tensor data and initial state on CPU;
    # H2D copies are not compute kernels in op_summary. No inputs are rebuilt
    # and no output copies/check kernels are launched between API calls.
    helper_path = OP_DIR / "scripts/profile_msprof.py"
    spec = importlib.util.spec_from_file_location("gdn_latency_inputs", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = options.case_json.read_bytes()
    selected = [row for row in json.loads(raw) if int(row["id"]) == options.case_id]
    if len(selected) != 1:
        raise ValueError(f"case id {options.case_id} must select exactly one performance case")
    row = selected[0]
    config = helper.case_config(row)
    seed = config["seed"] if options.seed is None else options.seed

    import torch
    import torch_npu
    import fla_npu
    from fla_npu.ops import ascendc

    torch.npu.set_device(options.device)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    manifest = {
        "case_id": options.case_id, "case_name": row["name"],
        "case_json": str(options.case_json.resolve()),
        "case_json_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": seed, "device": options.device, "warmup": options.warmup,
        "iterations": options.iterations, "completed_calls": 0, "pid": os.getpid(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "fla_npu_path": str(fla_npu.__file__), "config": config,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with torch.no_grad():
        args, kwargs = helper.build_call(config, torch, torch.device(f"npu:{options.device}"), seed)
        torch.npu.synchronize()
        outputs = None
        for _ in range(options.warmup + options.iterations):
            outputs = ascendc.chunk_gated_delta_rule_fwd(*args, **kwargs)
            torch.npu.synchronize()
            manifest["completed_calls"] += 1
        del outputs
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_msprof_command(options, output, executable):
    application = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--case-json", str(options.case_json.resolve()), "--case-id", str(options.case_id),
                   "--device", str(options.device), "--warmup", str(options.warmup),
                   "--iterations", str(options.iterations), "--output", str(output)]
    if options.seed is not None:
        application.extend(("--seed", str(options.seed)))
    return [executable, "--application=" + shlex.join(application),
            "--output=" + str(output / "profile"), "--task-time=on", "--aicpu=on"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=0, help="forward _perf.json id: 0=inference, 1=training")
    parser.add_argument("--case-json", type=Path, default=OP_DIR / "atk_chunk_gated_delta_rule_fwd_perf.json")
    parser.add_argument("--device", type=nonnegative, default=0)
    parser.add_argument("--warmup", type=nonnegative, default=10)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--seed", type=nonnegative)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--msprof", default="msprof", help="ordinary msprof executable")
    parser.add_argument("--summarize-only", action="store_true", help="summarize an existing --output run")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.summarize_only and options.output is None:
        parser.error("--summarize-only requires --output")
    output = (options.output or OP_DIR / "runs" / datetime.now().strftime("gdn_latency_%Y%m%d_%H%M%S_%f")).resolve()
    if options.worker:
        worker(options, output)
        return
    if options.summarize_only:
        summarize(output)
        return
    executable = shutil.which(options.msprof)
    if executable is None:
        parser.error("msprof not found; source the CANN environment first")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty; use --summarize-only to reread an existing run")
    output.mkdir(parents=True, exist_ok=True)
    command = build_msprof_command(options, output, executable)
    print(f"Collecting complete GDN forward calls; log: {output / 'msprof.log'}", flush=True)
    with (output / "msprof.log").open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"msprof exited {completed.returncode}; see {output / 'msprof.log'}")
        if not list((output / "profile").rglob("op_summary*.csv")):
            # Some CANN installations keep raw PROF data without auto-export.
            exported = subprocess.run(
                [executable, "--export=on", "--output=" + str(output / "profile")],
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            if exported.returncode:
                raise RuntimeError(f"msprof export failed; see {output / 'msprof.log'}")
    summarize(output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

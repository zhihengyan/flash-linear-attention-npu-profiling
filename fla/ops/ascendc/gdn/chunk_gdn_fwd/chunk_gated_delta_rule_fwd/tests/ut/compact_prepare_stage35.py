"""Export one small text from existing Stage3--5 summary.json; standard library only.

No raw/parsed CSV, torch, CANN or recapture is needed. Per-capture distributions
are retained as repeat ranges; quantiles are never pooled from other quantiles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("summary metrics must be finite nonnegative numbers")
    return value


def cell(rows):
    if not rows:
        return "-"
    medians = [number(r["call_median_cycles"]) for r in rows]
    p95 = max(number(r["call_p95_cycles"]) for r in rows)
    maximum = max(number(r["call_max_cycles"]) for r in rows)
    calls = [r["sampled_calls"] for r in rows]
    if any(type(n) is not int or n <= 0 for n in calls):
        raise ValueError("sampled_calls must be positive integers")
    return (f"m:{statistics.median(medians):g}[{min(medians):g}..{max(medians):g}]"
            f",p:{p95:g},x:{maximum:g},n:{sum(calls)}")


def ordinary_lines(metadata, expected_input_sha):
    if metadata is None:
        return ["Ordinary msprof: not supplied; diagnostic cycles alone cannot quantify benefit."]
    cases = metadata.get("case_ids", [])
    if len(cases) != 1:
        return ["Ordinary msprof: omitted; multi-case metadata lacks per-case kernel attribution."]
    case = str(cases[0])
    if metadata.get("input_sha256", {}).get(case) != expected_input_sha:
        raise ValueError("run_metadata input SHA256 does not match diagnostic summary")
    evidence = metadata.get("performance_evidence", {})
    if cases[0] not in evidence.get("executed_case_ids", []):
        raise ValueError("run_metadata lacks successful ordinary case evidence")
    grouped = defaultdict(list)
    unknown = 0
    for row in evidence.get("target_kernels", []):
        names = row.get("name", [])
        label = " ".join(names) if isinstance(names, list) else str(names)
        label = label.lower().replace("_", "")
        if "chunkgateddeltarulefwdprepare" in label:
            key = "Prepare"
        elif "chunkfwdh" in label:
            key = "H"
        elif "chunkfwdo" in label:
            key = "O"
        elif "transpose" in label:
            key = "Transpose"
        elif "cast" in label:
            key = "Cast"
        else:
            unknown += 1
            continue
        duration = number(row["task_duration_us"])
        if duration <= 0:
            raise ValueError("ordinary task duration must be positive")
        grouped[key].append(duration)
    lines = [f"Ordinary msprof case={case}: n mean/median/p95/min/max (us)"]
    for key in ("Prepare", "H", "O", "Cast", "Transpose"):
        values = sorted(grouped[key])
        if values:
            label = "Transpose(all 3)" if key == "Transpose" else key
            lines.append(f"{label} {len(values)} {statistics.mean(values):.6g}/"
                         f"{statistics.median(values):.6g}/{values[math.ceil(len(values)*.95)-1]:.6g}/"
                         f"{values[0]:.6g}/{values[-1]:.6g}")
    n = len(grouped["Prepare"])
    if n and not unknown and all(len(grouped[k]) == n for k in ("H", "O", "Cast")) and len(grouped["Transpose"]) == 3*n:
        total = sum(sum(v) for v in grouped.values()) / n
        lines.append(f"Seven-kernel mean sum={total:.6g}us (includes three Transposes; not host end-to-end latency).")
    else:
        lines.append("Seven-kernel mean sum unavailable: missing, extra or inconsistent kernel counts.")
    lines.append("No aligned launch IDs in this metadata: no total p95 or per-iteration regression claim.")
    artifact = metadata.get("reference_wheel", {})
    if artifact.get("sha256"):
        lines.append("wheel_sha256=" + artifact["sha256"])
    return lines


def render_compact(result, metadata=None):
    if result.get("schema") != "gdn-prepare-stage35/v1":
        raise ValueError("expected existing gdn-prepare-stage35/v1 summary.json")
    controls = result["controls"]
    windows = sorted({(r["pack_start"], r["pack_count"]) for r in controls})
    if not windows:
        raise ValueError("summary has no paired windows")
    stages = defaultdict(lambda: defaultdict(list))
    for row in result["stages"]:
        if row["detail_enabled"]:
            window = row["pack_start"], row["pack_count"]
            if window not in windows:
                raise ValueError("stage window lacks a coarse/detail control")
            stages[row["core_type"], row["stage"]][window].append(row)
    selected_options = {k: result["options"][k] for k in ("chunk_size", "layout", "disable_recompute", "output_final_state")
                        if k in result["options"]}
    lines = ["Prepare Stage3--5 share summary (cycles; existing analyzed captures)",
             "Export reads summary only; capture validity was checked by the source analyzer.",
             "inputs=" + json.dumps(result["inputs"], separators=(",", ":")),
             "options=" + json.dumps(selected_options, separators=(",", ":")),
             "input_snapshot_sha256=" + result["input_snapshot_sha256"],
             "header_sha256=" + result["header_sha256"]]
    lines.extend(ordinary_lines(metadata, result["input_snapshot_sha256"]))
    lines.extend(["", "Perturbation: per-capture kernel medians; ranges across repeats; publication after end stamps excluded."])
    for row in sorted(controls, key=lambda r: (r["pack_start"], r["pack_count"], r["core_type"])):
        a, b = number(row["coarse_median_cycles"]), number(row["detail_median_cycles"])
        if a == 0:
            raise ValueError("coarse kernel median must be positive")
        delta = 100*(b/a-1)
        lines.append(f"W{row['pack_start']}+{row['pack_count']} {row['core_type']} "
                     f"coarse:{a:g}[{number(row['coarse_min_cycles']):g}..{number(row['coarse_max_cycles']):g}] "
                     f"detail:{b:g}[{number(row['detail_min_cycles']):g}..{number(row['detail_max_cycles']):g}] "
                     f"delta:{delta:+.2f}% repeats:{row['coarse_captures']}/{row['detail_captures']}")
    lines.extend(["", "Stage / " + " / ".join(f"W{s}+{n}" for s, n in windows),
                  "Cell m=median of per-capture call medians [repeat min..max]; p=max per-capture p95; x=max call; n=sample calls across repeats.",
                  "These are not pooled quantiles; '-' means no observed scope in that window, not zero duration."])
    for (core_type, stage), by_window in sorted(stages.items(), key=lambda item: (item[0][1], item[0][0])):
        label = stage.removeprefix("PREPARE_")
        lines.append(f"{label}/{core_type} / " + " / ".join(cell(by_window[w]) for w in windows))
    if not stages:
        lines.append("No detailed stage samples in the selected windows.")
    lines.extend(["", "Scopes may nest and contain original internal waits; do not add them into latency shares.",
                  "Compare startup/stable/tail and repeat ranges before choosing a candidate; diagnostic deltas are not optimization benefit.",
                  "PACK includes previous-pack AIV drain; final current-pack drain is outside the loop.",
                  "No cross-core timestamp subtraction; no claim of hardware idle cycles or a 5% opportunity."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="existing stage35_analysis directory or summary.json")
    parser.add_argument("--run-metadata", type=Path, help="optional same run's ordinary msprof metadata")
    parser.add_argument("--output", type=Path, help="new file; defaults to summary directory/share_brief.txt")
    args = parser.parse_args(argv)
    source = args.summary / "summary.json" if args.summary.is_dir() else args.summary
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
        metadata = json.loads(args.run_metadata.read_text(encoding="utf-8")) if args.run_metadata else None
        text = render_compact(result, metadata)
        path = args.output or source.parent / "share_brief.txt"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(text, end="")
        print(f"File: {path} ({len(text.splitlines())} lines; {len(text.encode('utf-8'))} bytes)")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Compact export failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize an existing GDN capture inside its environment; no torch/CANN needed.

Reads parsed CSV and timer_layout.json, never raw tensors or raw timer storage.
Sums across cores are observations of sampled scopes, not launch latency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def stats(values):
    ordered = sorted(values)
    return dict(count=len(ordered), min=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(len(ordered) * .95) - 1], max=ordered[-1])


def union_cycles(intervals):
    """Merge intervals on ONE core; nested waiting windows are not counted twice."""
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def read_events(path):
    fields = {"kernel", "core_id", "group_id", "core_type", "sub_id", "stage", "kind",
              "iteration", "start_cycle", "end_cycle", "duration_cycles"}
    events = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("CSV lacks required GDN fields; decode it with the matching parser first")
        for line, row in enumerate(reader, 2):
            event = {key: row[key] for key in fields}
            event["work_id"] = int(row.get("work_id", -1))
            if event["work_id"] < -1:
                raise ValueError(f"CSV line {line}: invalid work_id")
            for key in ("core_id", "group_id", "sub_id", "iteration", "start_cycle", "end_cycle", "duration_cycles"):
                event[key] = int(event[key])
            core = event["core_id"]
            expected_type = "AIC" if core % 3 == 0 else "AIV"
            if core < 0 or event["group_id"] != core // 3 or event["core_type"] != expected_type or event["sub_id"] != core % 3 - 1:
                raise ValueError(f"CSV line {line}: invalid MIX core identity")
            start, end = event["start_cycle"], event["end_cycle"]
            if start <= 0 or end < start or end >= 2**63 or end-start != event["duration_cycles"]:
                raise ValueError(f"CSV line {line}: invalid timing pair/duration")
            if event["iteration"] < -1 or event["kind"] not in {"issue", "wait", "window", "point"}:
                raise ValueError(f"CSV line {line}: invalid iteration/kind")
            if event["iteration"] >= 0 and event["kind"] not in {"issue", "wait"}:
                raise ValueError(f"CSV line {line}: invalid dynamic event kind")
            if event["iteration"] >= 0 and ((event["kind"] == "wait") != event["stage"].endswith("WAIT")):
                raise ValueError(f"CSV line {line}: inconsistent wait stage")
            key = (event["kernel"], core, event["stage"], event["iteration"])
            if key in seen:
                raise ValueError(f"CSV line {line}: duplicate stage/core/iteration")
            seen.add(key)
            events.append(event)
    if not events:
        raise ValueError("empty parsed CSV")
    return events


def analyze(events, layout, capture=None):
    capture = capture or {}
    members = layout["layout"]["CORES_PER_GROUP"]
    capacity = layout["layout"]["CORE_GROUP_CAPACITY"]
    sample_limit = layout["layout"]["MAX_DYNAMIC_ITER"]
    if members != 3 or not isinstance(sample_limit, int) or sample_limit <= 0:
        raise ValueError("unsupported core mapping or sample limit")
    metadata = {item["kernel"]: item for item in layout["kernels"]}
    if len(metadata) != len(layout["kernels"]):
        raise ValueError("duplicate kernel metadata")
    kernels = set(metadata)
    if kernels not in ({"PHASE6"}, {"PREPARE", "FWD_H", "FWD_O"}):
        raise ValueError("incomplete or mixed GDN launch route")
    expected_cores = set()
    counts = {}
    for kernel, item in metadata.items():
        groups = item["block_dim"]
        if type(groups) is not int or not 0 < groups <= capacity:
            raise ValueError(f"{kernel}: invalid block_dim")
        expected_cores.update((kernel, core) for core in range(groups * members))
        for count in item["iterations"]:
            core, actual = count["core_id"], count["actual"]
            if type(core) is not int or not 0 <= core < groups * members or type(actual) is not int or actual <= 0:
                raise ValueError(f"{kernel}: invalid dynamic count")
            key = kernel, core, count["stage"]
            if key in counts:
                raise ValueError("duplicate dynamic count metadata")
            sampled = count["sampled"]
            pack_window = item.get("sampling", {}).get("mode") == "pack_window"
            if type(sampled) is not int or not 0 <= sampled <= min(actual, sample_limit) or count["truncated"] != (actual > sampled):
                raise ValueError("inconsistent sampled/actual/truncated metadata")
            if not pack_window and sampled != min(actual, sample_limit):
                raise ValueError("inconsistent sampled/actual/truncated metadata")
            counts[key] = count
    core_events = defaultdict(list)
    sampled_iters = defaultdict(set)
    for event in events:
        core_events[event["kernel"], event["core_id"]].append(event)
        if event["iteration"] >= 0:
            sampled_iters[event["kernel"], event["core_id"], event["stage"]].add(event["iteration"])
    if set(core_events) != expected_cores:
        raise ValueError("CSV cores/launches do not match timer_layout.json")
    if set(sampled_iters) != {key for key, count in counts.items() if count["sampled"]}:
        raise ValueError("CSV dynamic stages do not match iteration metadata")
    for key, count in counts.items():
        if sampled_iters[key] != set(range(count["sampled"])):
            raise ValueError(f"{key}: missing samples or iterations beyond sampled count")

    core_rows = []
    by_stage = defaultdict(list)
    longest = []
    for (kernel, core), observed in sorted(core_events.items()):
        fixed = {e["stage"]: e for e in observed if e["iteration"] == -1}
        if not {"KERNEL_TIMING", "MAIN_INIT", "MAIN_BODY"}.issubset(fixed):
            raise ValueError(f"{kernel}/core {core}: missing fixed total/init/body")
        parent = fixed["KERNEL_TIMING"]
        begin, end = parent["start_cycle"], parent["end_cycle"]
        if end == begin:
            raise ValueError("zero kernel window")
        for event in observed:
            if event["start_cycle"] < begin or event["end_cycle"] > end:
                raise ValueError(f"{kernel}/core {core}: event outside KERNEL_TIMING")
        dynamic = [e for e in observed if e["iteration"] >= 0]
        waits = [e for e in dynamic if e["kind"] == "wait"]
        # The fixed serial H/O wait overlaps PHASE_WAIT; merge rather than add.
        if "PHASE6_HO_WAIT" in fixed:
            waits.append(fixed["PHASE6_HO_WAIT"])
        wait_union = union_cycles((e["start_cycle"], e["end_cycle"]) for e in waits)
        sampled_union = union_cycles((e["start_cycle"], e["end_cycle"]) for e in dynamic + waits)
        core_counts = [c for (k, c_id, _), c in counts.items() if k == kernel and c_id == core]
        row = dict(kernel=kernel, core_id=core, group_id=core // 3, core_type=parent["core_type"],
                   sub_id=parent["sub_id"], kernel_cycles=end-begin,
                   observed_wait_union_cycles=wait_union, observed_wait_fraction=wait_union/(end-begin),
                   sampled_scope_union_cycles=sampled_union, sampled_scope_fraction=sampled_union/(end-begin),
                   has_issue_scope=any(e["kind"] == "issue" for e in dynamic),
                   truncated_types=sum(c["truncated"] for c in core_counts),
                   sample_events=len(dynamic))
        core_rows.append(row)
        for event in dynamic:
            by_stage[event["kernel"], event["stage"], event["core_type"]].append(event)
            longest.append(dict(kernel=kernel, core_id=core, group_id=core//3, core_type=event["core_type"],
                stage=event["stage"], iteration=event["iteration"], kind=event["kind"],
                work_id=event.get("work_id", -1),
                duration_cycles=event["duration_cycles"], start_offset_cycles=event["start_cycle"]-begin))
    stage_rows = []
    for (kernel, stage, core_type), observed in sorted(by_stage.items()):
        per_core = defaultdict(list)
        for event in observed:
            per_core[event["core_id"]].append(event["duration_cycles"])
        actual = [counts[kernel, core, stage]["actual"] for core in per_core]
        sampled = [counts[kernel, core, stage]["sampled"] for core in per_core]
        row = dict(kernel=kernel, stage=stage, core_type=core_type, kind=observed[0]["kind"],
            observed_cores=len(per_core), sampled_calls=len(observed),
            actual_calls_per_core_min=min(actual), actual_calls_per_core_max=max(actual),
            sampled_calls_per_core_min=min(sampled), sampled_calls_per_core_max=max(sampled),
            truncated_cores=sum(counts[kernel, core, stage]["truncated"] for core in per_core),
            sample_sum_core_cycles=sum(e["duration_cycles"] for e in observed),
            **{f"call_{key}_cycles": value for key, value in stats([e["duration_cycles"] for e in observed]).items() if key != "count"})
        medians = [statistics.median(values) for values in per_core.values()]
        row["per_core_call_median_min_cycles"] = min(medians)
        row["per_core_call_median_max_cycles"] = max(medians)
        stage_rows.append(row)
    kernel_rows = []
    for kernel in sorted(kernels):
        for core_type in ("AIC", "AIV"):
            rows = [r for r in core_rows if r["kernel"] == kernel and r["core_type"] == core_type]
            durations = stats([r["kernel_cycles"] for r in rows])
            kernel_rows.append(dict(kernel=kernel, core_type=core_type, cores=len(rows),
                cores_with_issue_scopes=sum(r["has_issue_scope"] for r in rows),
                window_cycles=durations, max_over_median=durations["max"]/durations["median"],
                observed_wait_fraction=stats([r["observed_wait_fraction"] for r in rows]),
                sampled_scope_fraction=stats([r["sampled_scope_fraction"] for r in rows]),
                longest_window_cores=sorted(rows,key=lambda r:r["kernel_cycles"],reverse=True)[:3],
                highest_observed_wait_cores=sorted(rows,key=lambda r:r["observed_wait_fraction"],reverse=True)[:3]))
    frequency = layout.get("cycles_per_us")
    if frequency is not None and (not isinstance(frequency, (int,float)) or not math.isfinite(frequency) or frequency <= 0):
        raise ValueError("invalid saved cycles_per_us")
    warnings = []
    if not stage_rows:
        warnings.append("No dynamic scopes are recorded; key-stage analysis is unavailable.")
    truncated = sum(c["truncated"] for c in counts.values())
    if truncated:
        warnings.append(f"{truncated} core/stage counters are partially sampled (at most {sample_limit} calls). Consult sampling metadata. Do not extrapolate full-work duration.")
    if capture.get("decode_status", "unknown") != "complete":
        warnings.append("Capture decode_status is not recorded as complete; this summary validates parsed data, not original raw.")
    comparison = capture.get("output_comparison", "unknown")
    if not isinstance(comparison, str):
        raise ValueError("output_comparison must be a string")
    if not comparison.startswith("ten outputs exactly match"):
        warnings.append("Ordinary/profile numerical comparison is not recorded as passed.")
    options = capture.get("options", {})
    selected_options = {key: options[key] for key in ("chunk_size", "layout", "scale", "use_exp2",
        "use_qk_l2norm_in_kernel", "state_v_first", "output_final_state", "disable_recompute",
        "return_intermediate_states", "use_gate_in_kernel", "use_beta_sigmoid_in_kernel",
        "allow_neg_eigval", "initial_state") if key in options}
    if "cu_seqlens" in options:
        selected_options["varlen"] = options["cu_seqlens"] is not None
    summary = dict(schema="gdn-timer-summary/v1", parsed_csv_validation="passed",
        capture_decode_status=capture.get("decode_status","unknown"), output_comparison=comparison,
        device_name=capture.get("device_name","unknown"), torch=capture.get("torch","unknown"),
        inputs=capture.get("inputs",{}), options=selected_options,
        header_sha256=capture.get("header_sha256","unknown"),
        sample_count=capture.get("sample_count","unknown"),
        warmup=capture.get("warmup","unknown"), cycles_per_us=frequency,
        sample_limit=sample_limit, truncated_core_stage_counters=truncated,
        sampling={key: value.get("sampling", {"mode": "prefix"}) for key, value in metadata.items()},
        warnings=warnings, kernels=kernel_rows, stages=stage_rows,
        longest_dynamic_windows=sorted(longest,key=lambda e:e["duration_cycles"],reverse=True)[:12],
        semantics=["Sample sums across cores are core-cycles, not elapsed launch time or bottleneck contribution.",
            "Most math scopes are issue windows and may include original internal waits; they are not pure compute duration.",
            "Wait unions use a single core's timestamps; they do not measure hardware idle cycles.",
            "Unobserved timeline regions can be uninstrumented work, asynchronous execution or truncation; they are not proven bubbles.",
            "Scope coverage includes coarse Process windows; high coverage does not imply complete fine-stage sampling.",
            "Iteration is a dense per-stage sample slot; use work_id for Prepare detail task joins.",
            "Each launch is analyzed separately; this tool does not construct a cross-core or cross-launch timeline.",
            "One capture cannot establish a stable performance regression; use repeated ordinary msprof measurements."])
    return summary, core_rows


def brief(summary, top=12):
    lines = ["GDN timer summary (cycles; one capture)",
        f"CSV={summary['parsed_csv_validation']} decode={summary['capture_decode_status']} comparison={summary['output_comparison']}",
        f"device={summary['device_name']} torch={summary['torch']} captures={summary['sample_count']} warmup={summary['warmup']}",
        "inputs="+json.dumps(summary["inputs"],separators=(",",":")),
        "options="+json.dumps(summary["options"],separators=(",",":")),
        "sampling="+json.dumps(summary["sampling"],separators=(",",":")),
        f"sample_limit={summary['sample_limit']} truncated_core_stage_counters={summary['truncated_core_stage_counters']}",
        "", "Per-core kernel windows: kernel/type cores issue-scope-cores median/p95/max cycles max/median wait-union median% sampled-window median%"]
    for row in summary["kernels"]:
        w=row["window_cycles"]
        lines.append(f"{row['kernel']}/{row['core_type']} {row['cores']} {row['cores_with_issue_scopes']} "
            f"{w['median']:g}/{w['p95']:g}/{w['max']:g} {row['max_over_median']:.3f} "
            f"{100*row['observed_wait_fraction']['median']:.2f}% {100*row['sampled_scope_fraction']['median']:.2f}%")
    lines += ["", "Largest sampled scope sums (NOT latency shares): kernel/stage/type sample-core-cycles call-median/p95/max actual-calls/core[min,max] truncated-cores"]
    for row in sorted(summary["stages"],key=lambda r:r["sample_sum_core_cycles"],reverse=True)[:top]:
        lines.append(f"{row['kernel']}/{row['stage']}/{row['core_type']} {row['sample_sum_core_cycles']} "
            f"{row['call_median_cycles']:g}/{row['call_p95_cycles']:g}/{row['call_max_cycles']:g} "
            f"[{row['actual_calls_per_core_min']},{row['actual_calls_per_core_max']}] {row['truncated_cores']}")
    lines += ["", "Cores to inspect locally: kernel/type longest-window core IDs; largest observed wait-fraction core IDs"]
    for row in summary["kernels"]:
        lines.append(f"{row['kernel']}/{row['core_type']} "
            f"{[r['core_id'] for r in row['longest_window_cores']]} "
            f"{[r['core_id'] for r in row['highest_observed_wait_cores']]}")
    lines += ["", "WARNINGS"] + (summary["warnings"] or ["none from metadata; device validity still requires original capture evidence"])
    lines += ["", "INTERPRETATION"] + summary["semantics"]
    return "\n".join(lines)+"\n"


def dump_csv(path, rows):
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ["kernel","stage","core_type"])
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path,help="case_N directory, or gdn_npu_time_parsed.csv")
    parser.add_argument("--output-dir",type=Path,help="defaults to capture/analysis; existing outputs are preserved")
    parser.add_argument("--top",type=int,default=12,help="top sampled stages in pasteable brief (1-40)")
    args=parser.parse_args(argv)
    if not 1<=args.top<=40:parser.error("top must be in 1..40")
    csv_path=args.capture/"gdn_npu_time_parsed.csv" if args.capture.is_dir() else args.capture
    directory=csv_path.parent
    try:
        layout=json.loads((directory/"timer_layout.json").read_text(encoding="utf-8"))
        capture_path=directory/"capture_metadata.json"
        capture=json.loads(capture_path.read_text(encoding="utf-8")) if capture_path.exists() else None
        summary,cores=analyze(read_events(csv_path),layout,capture)
        digest=hashlib.sha256()
        with csv_path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
        summary["parsed_csv_sha256"]=digest.hexdigest()
        output=args.output_dir or directory/"analysis"
        output.mkdir(parents=True,exist_ok=False)
        (output/"brief.txt").write_text(brief(summary,args.top),encoding="utf-8")
        (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        dump_csv(output/"stage_summary.csv",summary["stages"])
        dump_csv(output/"core_summary.csv",cores)
        dump_csv(output/"longest_windows.csv",summary["longest_dynamic_windows"])
        print(brief(summary,args.top),end="")
        print(f"Files: {output.resolve()}\nPaste only brief.txt; keep raw/CSV/trace in your environment.")
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f"Analysis failed: {error}",file=sys.stderr);return 1
    return 0


if __name__=="__main__":raise SystemExit(main())

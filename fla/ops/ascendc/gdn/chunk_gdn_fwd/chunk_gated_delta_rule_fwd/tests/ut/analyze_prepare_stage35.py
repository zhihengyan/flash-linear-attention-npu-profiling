"""Join Prepare Stage3--5 observations by business work_id on one core.

Only reads parsed captures. No cross-core timestamp subtraction or latency attribution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import statistics
import sys

from summarize_timer import analyze, dump_csv, read_events, stats, union_cycles
from compact_prepare_stage35 import render_compact

S3 = ["S3_KKT_WAIT", "S3_GATE_MUL", "S3_L_READY_WAIT", "S3_LEAF_PACK",
      "S3_PACK_READY_WAIT", "S3_NEG_L_UPLOAD", "S3_LEAF_SOLVE",
      "S3_SOLVE_READY_WAIT", "S3_LEAF_UPLOAD", "S3_READY_PUBLISH"]
S4 = ["S4_INPUT_WAIT", "S4_ACC_ISSUE", "S4_CUBE_DONE_WAIT", "S4_Y_GM_WRITE",
      "S4_Y_GM_DONE_WAIT", "S4_Y_L1_READ", "S4_READY_PUBLISH"]
S5 = ["S5_Y_READY_WAIT", "S5_COMPUTE", "S5_CUBE_DONE_WAIT"]
S7 = ["S7_INPUT_WAIT", "S7_A_READY_WAIT"]
PREFIX = "PREPARE_"

def expected_task_count(capture):
    """Compute work count from actual input contract, independently of timer counters."""
    options = capture["options"]
    shape = capture["inputs"]["v"]["shape"]
    head_axis = 1 if options.get("layout", "BNSD") in ("BNSD", "NTD") else 2
    heads, tokens, batch = shape[head_axis], shape[3 - head_axis], shape[0]
    bt = options.get("chunk_size", 64)
    if type(bt) is not int or bt <= 0:
        raise ValueError("invalid chunk_size")
    cu = options.get("cu_seqlens")
    if cu is None:
        chunks = batch * ((tokens + bt - 1) // bt)
    else:
        if not isinstance(cu, (list, tuple)) or len(cu) < 2 or batch != 1:
            raise ValueError("varlen task count needs exported cu_seqlens values")
        if any(type(v) is not int for v in cu) or cu[0] != 0 or cu[-1] != tokens or any(a > b for a, b in zip(cu, cu[1:])):
            raise ValueError("invalid cu_seqlens")
        chunks = sum((b - a + bt - 1) // bt for a, b in zip(cu, cu[1:]))
        indices = options.get("chunk_indices")
        if not isinstance(indices, (list, tuple)) or len(indices) != 2 * chunks:
            raise ValueError("chunk_indices count differs from sequence chunk count")
    return chunks * heads

def a_transfer_contract(capture, global_chunk):
    """Original A output flag and GetChunkRange valid M, from captured inputs."""
    options = capture["options"]
    value = options.get("disable_recompute")
    output_a = True if value is None else bool(value)
    bt = options.get("chunk_size", 64)
    cu = options.get("cu_seqlens")
    if cu is None:
        shape = capture["inputs"]["v"]["shape"]
        token_axis = 2 if options.get("layout", "BNSD") in ("BNSD", "NTD") else 1
        tokens = shape[token_axis]
        chunks_per_batch = (tokens + bt - 1) // bt
        m = min(bt, tokens - (global_chunk % chunks_per_batch) * bt)
    else:
        indices = options["chunk_indices"]
        seq, local_chunk = indices[global_chunk * 2:global_chunk * 2 + 2]
        if type(seq) is not int or type(local_chunk) is not int or not 0 <= seq < len(cu) - 1 or local_chunk < 0:
            raise ValueError("invalid chunk_indices task")
        m = min(bt, cu[seq + 1] - cu[seq] - local_chunk * bt)
    if not 0 < m <= bt:
        raise ValueError("invalid task chunk length")
    return output_a, m, bt

def task_rows(events, prepare, capture):
    """Validate identities, branch coverage and original same-core order."""
    selected = [e for e in events if e["kernel"] == "PREPARE" and e["stage"].startswith(PREFIX)
                and (re.fullmatch(r"PREPARE_S[3457]_\w+", e["stage"]) or e["stage"] == "PREPARE_PACK")]
    sampling = prepare["sampling"]
    if not sampling["detail_enabled"]:
        if selected:
            raise ValueError("detail events in a coarse capture")
        return [], []
    grouped = defaultdict(dict)
    pack_events = []
    inputs = capture["inputs"]
    head_axis = 1 if capture["options"].get("layout", "BNSD") in ("BNSD", "NTD") else 2
    hv = inputs["v"]["shape"][head_axis]
    hk = inputs["k"]["shape"][head_axis]
    if hv <= 0 or hk <= 0 or hv % hk:
        raise ValueError("invalid head ratio in capture")
    pack_size = 3 if hv // hk == 3 else 4
    groups = prepare["block_dim"]
    total_tasks = sum(c["actual"] for c in prepare["iterations"]
                      if c["stage"] == "PREPARE_S4_INPUT_WAIT" and c["core_id"] % 3 == 0)
    if total_tasks != expected_task_count(capture):
        raise ValueError("Stage4 actual task count differs from real input work count")
    actual = {(c["core_id"], c["stage"][len(PREFIX):]): c["actual"] for c in prepare["iterations"]}
    aic_required = S4 + S5 + S7 + ["S4_Y_TRANSFER", "S5_A_TRANSFER",
                                  "S5_A_READY_PUBLISH", "S5_READY_PUBLISH"]
    n_packs = (total_tasks + pack_size - 1) // pack_size
    for group in range(groups):
        pack_lengths = [min(pack_size, total_tasks - pack * pack_size) for pack in range(group, n_packs, groups)]
        for member in range(3):
            core = group * 3 + member
            task_count = sum(pack_lengths) if member == 0 else sum(len(range(member - 1, n, 2)) for n in pack_lengths)
            required = aic_required if member == 0 else S3 + ["S3_POST_KKT"]
            forbidden = S3 + ["S3_POST_KKT"] if member == 0 else aic_required
            if actual.get((core, "PACK"), 0) != len(pack_lengths):
                raise ValueError("per-core actual pack count differs from original grid stride")
            if any(actual.get((core, s), 0) != task_count for s in required) or any(actual.get((core, s), 0) for s in forbidden):
                raise ValueError("per-core mandatory actual stage counts differ from original tasks")
            if member == 0:
                if actual.get((core, "S5_A_L1_READ"), 0) + actual.get((core, "S5_A_L1_WRITE"), 0) != task_count:
                    raise ValueError("actual A transfer branches do not cover all tasks")
                if actual.get((core, "S5_A_GM_WRITE"), 0) != actual.get((core, "S5_A_GM_DONE_WAIT"), 0):
                    raise ValueError("actual A GM write/readiness counts differ")
                group_work = [work for pack in range(group, n_packs, groups)
                              for work in range(pack * pack_size, min(total_tasks, (pack + 1) * pack_size))]
                gm_count = read_count = 0
                for work in group_work:
                    output_a, m, bt = a_transfer_contract(capture, work // hv)
                    gm_count += int(output_a)
                    read_count += int(output_a and m == bt)
                if actual.get((core, "S5_A_GM_WRITE"), 0) != gm_count or actual.get((core, "S5_A_L1_READ"), 0) != read_count:
                    raise ValueError("actual A branch counts differ from input/output contract")
    expected_tasks, expected_packs = set(), set()
    for group in range(groups):
        for ordinal in range(sampling["pack_start"], sampling["pack_start"] + sampling["pack_count"]):
            base = (group + ordinal * groups) * pack_size
            n = min(pack_size, total_tasks - base)
            if n <= 0:
                continue
            for member in range(3):
                core = group * 3 + member
                expected_packs.add((core, ordinal))
                local_tasks = range(n) if member == 0 else range(member - 1, n, 2)
                for t in local_tasks:
                    offset = 3 - t if hv // hk == 2 and n == 4 and t in (1, 2) else t
                    expected_tasks.add((core, base + offset))
    for e in selected:
        work = e.get("work_id", -1)
        if work < 0:
            raise ValueError("Prepare detail needs a business work_id")
        global_pack = work // pack_size
        ordinal = global_pack // groups
        if global_pack % groups != e["group_id"]:
            raise ValueError("work_id belongs to another core group")
        if not sampling["pack_start"] <= ordinal < sampling["pack_start"] + sampling["pack_count"]:
            raise ValueError("work_id outside selected pack window")
        if e["stage"] == "PREPARE_PACK":
            pack_events.append(dict(core_id=e["core_id"], core_type=e["core_type"],
                                    pack_ordinal=ordinal, first_work_id=work,
                                    issue_cycles=e["duration_cycles"]))
            continue
        key = e["core_id"], work
        stage = e["stage"][len(PREFIX):]
        if stage in grouped[key]:
            raise ValueError("duplicate same-core/stage/work_id")
        grouped[key][stage] = e
    observed_packs = [(p["core_id"], p["pack_ordinal"]) for p in pack_events]
    if set(grouped) != expected_tasks or set(observed_packs) != expected_packs or len(set(observed_packs)) != len(observed_packs):
        raise ValueError("incomplete task/pack coverage for selected window")

    def ordered(stages, names):
        for a, b in zip(names, names[1:]):
            if stages[a]["end_cycle"] > stages[b]["start_cycle"]:
                raise ValueError(f"invalid original order: {a} then {b}")

    def contained(stages, parent, children):
        p = stages[parent]
        for name in children:
            child = stages[name]
            if not p["start_cycle"] <= child["start_cycle"] <= child["end_cycle"] <= p["end_cycle"]:
                raise ValueError(f"{name} outside {parent}")

    result = []
    for (core, work), stages in sorted(grouped.items()):
        core_type = "AIC" if core % 3 == 0 else "AIV"
        if core_type == "AIV":
            required = S3 + ["S3_POST_KKT"]
            if set(stages) != set(required):
                raise ValueError("incomplete/unexpected Stage3 task scopes")
            ordered(stages, S3)
            contained(stages, "S3_POST_KKT", S3[1:])
            if stages["S3_KKT_WAIT"]["end_cycle"] > stages["S3_POST_KKT"]["start_cycle"]:
                raise ValueError("post-KKT scope precedes KKT readiness")
            parent_names = ["S3_POST_KKT"]
        else:
            required = S4 + S5 + S7 + ["S4_Y_TRANSFER", "S5_A_TRANSFER",
                                      "S5_A_READY_PUBLISH", "S5_READY_PUBLISH"]
            if not set(required).issubset(stages):
                raise ValueError("incomplete Stage4/5 task scopes")
            has_gm = "S5_A_GM_WRITE" in stages
            if has_gm != ("S5_A_GM_DONE_WAIT" in stages):
                raise ValueError("unpaired A GM write/readiness scope")
            paths = [s for s in ("S5_A_L1_READ", "S5_A_L1_WRITE") if s in stages]
            if len(paths) != 1 or (paths[0] == "S5_A_L1_READ" and not has_gm):
                raise ValueError("invalid full/tail/no-output A transfer branch")
            output_a, m, bt = a_transfer_contract(capture, work // hv)
            expected_path = "S5_A_L1_READ" if output_a and m == bt else "S5_A_L1_WRITE"
            if has_gm != output_a or paths[0] != expected_path:
                raise ValueError("A transfer branch differs from real input/output contract")
            allowed = set(required) | {"S5_A_GM_WRITE", "S5_A_GM_DONE_WAIT", *paths}
            if set(stages) != allowed - ({"S5_A_GM_WRITE", "S5_A_GM_DONE_WAIT"} if not has_gm else set()):
                raise ValueError("unexpected Stage4/5 task scope")
            transfer = ["S5_CUBE_DONE_WAIT"] + (["S5_A_GM_WRITE", "S5_A_GM_DONE_WAIT"] if has_gm else []) + paths + ["S5_A_READY_PUBLISH", "S5_READY_PUBLISH"]
            ordered(stages, S4 + S5 + transfer[1:] + S7)
            contained(stages, "S4_Y_TRANSFER", S4[2:])
            contained(stages, "S5_A_TRANSFER", transfer)
            parent_names = ["S4_Y_TRANSFER", "S5_A_TRANSFER", "S4_ACC_ISSUE", "S5_COMPUTE"]
        waits = [e for e in stages.values() if e["kind"] == "wait"]
        # The envelope includes scheduling gaps, original waits and diagnostic stores.
        begin = min(e["start_cycle"] for e in stages.values())
        end = max(e["end_cycle"] for e in stages.values())
        row = dict(core_id=core, core_type=core_type, work_id=work, hv=work % hv,
                   global_chunk=work // hv, pack_ordinal=(work // pack_size) // groups,
                   task_envelope_cycles=end - begin,
                   observed_wait_union_cycles=union_cycles((e["start_cycle"], e["end_cycle"]) for e in waits))
        row["valid_tokens"] = a_transfer_contract(capture, work // hv)[1]
        row.update({name + "_cycles": e["duration_cycles"] for name, e in stages.items()})
        row["parent_scope_names"] = ",".join(parent_names)
        result.append(row)
    # An empty requested late window is a valid observation, not a fabricated sample.
    return result, pack_events


def analyze_capture(directory):
    directory = Path(directory)
    layout = json.loads((directory / "timer_layout.json").read_text(encoding="utf-8"))
    capture = json.loads((directory / "capture_metadata.json").read_text(encoding="utf-8"))
    if layout["layout"]["LAYOUT_VERSION"] != 3:
        raise ValueError("Stage3--5 analysis needs matching layout v3")
    if capture.get("decode_status") != "complete" or not capture.get("output_comparison", "").startswith("ten outputs exactly match"):
        raise ValueError("capture must pass strict decoding and the ten-output comparison")
    events = read_events(directory / "gdn_npu_time_parsed.csv")
    summary, _ = analyze(events, layout, capture)
    prepare = next((k for k in layout["kernels"] if k["kernel"] == "PREPARE"), None)
    if prepare is None:
        raise ValueError("selected route has no standalone Prepare; Stage3--5 diagnostics unavailable")
    controls = capture.get("prepare_sampling")
    if controls != {k: prepare["sampling"][k] for k in ("pack_start", "pack_count", "detail_enabled")}:
        raise ValueError("capture controls differ from kernel sampling header")
    tasks, packs = task_rows(events, prepare, capture)
    stages = [s for s in summary["stages"] if s["kernel"] == "PREPARE" and
              (re.fullmatch(r"PREPARE_S[3457]_\w+", s["stage"]) or s["stage"] == "PREPARE_PACK")]
    kernels = [dict(core_type=k["core_type"], **k["window_cycles"]) for k in summary["kernels"] if k["kernel"] == "PREPARE"]
    return dict(capture=directory.name, sampling=prepare["sampling"], stages=stages,
                kernel_windows=kernels, task_rows=tasks, pack_rows=packs,
                header_sha256=capture.get("header_sha256"),
                input_snapshot_sha256=capture.get("input_snapshot_sha256"),
                inputs=capture["inputs"], options=capture["options"], block_dim=prepare["block_dim"])


def combine(captures):
    if not captures:
        raise ValueError("no captures")
    first = captures[0]
    for c in captures:
        for name in ("inputs", "options", "header_sha256", "input_snapshot_sha256", "block_dim"):
            if c[name] != first[name] or (name.endswith("sha256") and not c[name]):
                raise ValueError(f"captures differ or lack evidence: {name}")
    pairs = defaultdict(lambda: {False: [], True: []})
    stages, tasks, packs = [], [], []
    for c in captures:
        sampling = c["sampling"]
        key = sampling["pack_start"], sampling["pack_count"]
        pairs[key][sampling["detail_enabled"]].append(c)
        for stage in c["stages"]:
            stages.append(dict(capture=c["capture"], **sampling, **stage))
        tasks.extend(dict(capture=c["capture"], **r) for r in c["task_rows"])
        packs.extend(dict(capture=c["capture"], **r) for r in c["pack_rows"])
    controls = []
    for (start, count), modes in sorted(pairs.items()):
        if not modes[False] or not modes[True]:
            raise ValueError("each detailed window needs a matching coarse control")
        for core_type in ("AIC", "AIV"):
            def medians(mode):
                return [next(k["median"] for k in c["kernel_windows"] if k["core_type"] == core_type)
                        for c in modes[mode]]
            coarse, detail = medians(False), medians(True)
            a, b = statistics.median(coarse), statistics.median(detail)
            controls.append(dict(pack_start=start, pack_count=count, core_type=core_type,
                                 coarse_captures=len(coarse), detail_captures=len(detail),
                                 coarse_median_cycles=a, detail_median_cycles=b,
                                 detailed_window_change_percent=100 * (b / a - 1),
                                 coarse_min_cycles=min(coarse), coarse_max_cycles=max(coarse),
                                 detail_min_cycles=min(detail), detail_max_cycles=max(detail)))
    return dict(schema="gdn-prepare-stage35/v1", controls=controls, stages=stages,
                tasks=tasks, packs=packs, inputs=first["inputs"], options=first["options"],
                input_snapshot_sha256=first["input_snapshot_sha256"],
                header_sha256=first["header_sha256"])


def brief(result):
    lines = ["Prepare Stage3--5 diagnosis (cycles; separate captures)",
             "Validated: layout v3, selected pack work IDs, same-core order, ten outputs.",
             "Detailed vs coarse Prepare kernel windows (perturbation check, NOT optimization benefit):"]
    for r in result["controls"]:
        lines.append(f"pack={r['pack_start']}+{r['pack_count']} {r['core_type']} "
                     f"coarse/detail median={r['coarse_median_cycles']:g}/{r['detail_median_cycles']:g} "
                     f"change={r['detailed_window_change_percent']:+.2f}% "
                     f"captures={r['coarse_captures']}/{r['detail_captures']}")
    lines += ["", "Detailed stage call median/p95/max cycles (NOT latency shares):"]
    for r in result["stages"]:
        if r["detail_enabled"]:
            lines.append(f"{r['capture']} {r['core_type']} {r['stage']} "
                         f"n={r['sampled_calls']} {r['call_median_cycles']:g}/{r['call_p95_cycles']:g}/{r['call_max_cycles']:g}")
    lines += ["", "Interpretation:",
              "Compare stable pack windows before ranking candidates. Empty tail windows are valid for smaller shapes.",
              "S3_KKT_WAIT vs S3_POST_KKT separates readiness from postprocessing; inspect S3_LEAF_SOLVE and original waits.",
              "S4_INPUT_WAIT tests pressure on AIC. Y write/read issue and completion waits are separate observations.",
              "S5_A_TRANSFER includes its waits; S7_A_READY_WAIT observes remaining A readiness after S7_INPUT_WAIT.",
              "Scopes contain original internal waits and may nest; never add scope medians or sum cores into launch latency.",
              "Task envelope is an elapsed same-core observation including scheduling gaps and timer stores.",
              "No cross-core/cross-launch subtraction; work_id identifies tasks without assuming clock alignment.",
              "Large or inconsistent detailed/coarse changes require lighter instrumentation or pipe measurements.",
              "Perturbation compares measured kernel windows; count/tag publication and DCCI after the end stamps are excluded.",
              "PREPARE_PACK includes previous-pack AIV drain; the last current-pack output drain is outside the loop.",
              "One pair cannot quantify stable perturbation. This diagnostic does not prove a 5% performance opportunity.",
              "Acceptance uses repeated ordinary msprof total and ATK precision/memory checks."]
    return "\n".join(lines) + "\n"


def dump_union_csv(path, rows):
    # Full/tail branches and AIC/AIV naturally have different columns.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["capture", "core_id", "work_id"]
    import csv
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, help="optional same run's ordinary msprof evidence for share_brief")
    args = parser.parse_args(argv)
    try:
        result = combine([analyze_capture(c) for c in args.captures])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "brief.txt").write_text(brief(result), encoding="utf-8")
        dump_csv(args.output_dir / "perturbation.csv", result["controls"])
        dump_union_csv(args.output_dir / "window_stage_summary.csv", result["stages"])
        dump_union_csv(args.output_dir / "task_summary.csv", result["tasks"])
        dump_union_csv(args.output_dir / "pack_summary.csv", result["packs"])
        small = {k: v for k, v in result.items() if k not in ("tasks", "packs")}
        (args.output_dir / "summary.json").write_text(json.dumps(small, indent=2, allow_nan=False), encoding="utf-8")
        metadata = json.loads(args.run_metadata.read_text(encoding="utf-8")) if args.run_metadata else None
        (args.output_dir / "share_brief.txt").write_text(render_compact(small, metadata), encoding="utf-8")
        print(brief(result), end="")
        print("Files:", args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Stage3--5 analysis failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

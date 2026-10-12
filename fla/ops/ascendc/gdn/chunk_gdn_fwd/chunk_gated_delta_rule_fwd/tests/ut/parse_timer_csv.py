"""Scheme A decoder. Layout and names come from the installed C++ timer header.

Adapted from op-timer-ascendc (52e396e); MIT, Copyright (c) 2026 Send_Tanks.
See ../../op_kernel/AscendTimerV2_LICENSE.txt. Raw timestamps stay integer cycles.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import operator
import re
import statistics
from pathlib import Path

HEADER_PATH = Path(__file__).resolve().parents[2].parents[2] / "common/kernel_utils/gdn_timer/AscendTimerV2.hpp"


def read_layout(path=HEADER_PATH):
    """Evaluate only integer constexpr arithmetic, never execute header text."""
    source = re.sub(r"//[^\n]*", "", Path(path).read_text(encoding="utf-8"))
    values, enums = {}, {}
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.FloorDiv: operator.floordiv}

    def integer(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](integer(node.left), integer(node.right))
        raise ValueError(f"unsupported layout expression: {ast.dump(node)}")

    for enum in ("FixedTiming", "KernelSlice"):
        match = re.search(r"enum\s+" + enum + r"\s*\{(.*?)\}", source, re.S)
        if match is None:
            raise ValueError(f"missing enum {enum}")
        names, index = [], 0
        for member in match[1].split(","):
            name, *expr = member.strip().split("=", 1)
            name = name.strip()
            if not name:
                continue
            if expr:
                index = integer(ast.parse(expr[0].strip(), mode="eval").body)
            if index != len(names) or name in values:
                raise ValueError("timer enums must be unique and contiguous")
            values[name] = index
            names.append(name)
            index += 1
        enums[enum] = names[:-1]
    for name, expr in re.findall(r"static constexpr int (\w+)\s*=\s*(.*?);", source, re.S):
        values[name] = integer(ast.parse(expr.strip().replace("/", "//"), mode="eval").body)
    return values, enums


LAYOUT, ENUMS = read_layout()
TOTAL_BUFFER_SIZE = LAYOUT["TOTAL_BUFFER_SIZE"]
FIXED_TIMING_NAMES = {i: "KERNEL_TIMING" if n == "KERNEL_TIMING_IDX" else n
                      for i, n in enumerate(ENUMS["FixedTiming"])}
KERNEL_NAMES = dict(enumerate(ENUMS["KernelSlice"]))
DYNAMIC_TIMING_NAMES = {}  # Scheme A has no loop events or iteration counters.
POINTS = {"PHASE6_FRONT_DONE", "PHASE6_H_DISPATCH_DONE"}
FIELDS = ["kernel", "core_id", "group_id", "core_type", "sub_id", "stage", "kind",
          "start_cycle", "end_cycle", "duration_cycles"]


def validate_frequency(cycles_per_us):
    if cycles_per_us is not None and (not math.isfinite(cycles_per_us) or cycles_per_us <= 0):
        raise ValueError("cycles_per_us must be a verified positive finite frequency")


def read_raw(path):
    result = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            result.extend(int(value.strip()) for value in row if value.strip())
    return result


def summarize_raw(raw):
    """Keep missing-slice evidence even when strict decoding cannot complete."""
    summaries = []
    stride = LAYOUT["N_TIMING_COUNTER_PER_CORE_ALIGN"]
    for kernel_id, kernel in KERNEL_NAMES.items():
        offset = kernel_id * LAYOUT["BUFFER_WORDS_PER_KERNEL"]
        data = raw[offset:offset + LAYOUT["BUFFER_WORDS_PER_KERNEL"]]
        rows = data[LAYOUT["HEADER_WORDS"]:]
        summaries.append({"kernel": kernel, "nonzero_words": sum(value != 0 for value in data),
                          "header": data[:LAYOUT["HEADER_WORDS"]],
                          "nonzero_core_rows": sum(any(rows[i:i + stride]) for i in range(0, len(rows), stride))})
    return {"word_count": len(raw), "slices": summaries}


def parse_timer(raw):
    if len(raw) != TOTAL_BUFFER_SIZE:
        raise ValueError(f"raw length {len(raw)} != TOTAL_BUFFER_SIZE {TOTAL_BUFFER_SIZE}")
    if any(type(value) is not int or value < 0 or value >= 2**63 for value in raw):
        raise ValueError("raw values must be nonnegative int64 integers")
    stride = LAYOUT["N_TIMING_COUNTER_PER_CORE_ALIGN"]
    capacity = LAYOUT["CORE_GROUP_CAPACITY"]
    members = LAYOUT["CORES_PER_GROUP"]
    events, metadata = [], []
    for kernel_id, kernel in KERNEL_NAMES.items():
        offset = kernel_id * LAYOUT["BUFFER_WORDS_PER_KERNEL"]
        data = raw[offset:offset + LAYOUT["BUFFER_WORDS_PER_KERNEL"]]
        if not any(data):
            continue
        expected = {0: LAYOUT["HEADER_MAGIC"], 1: LAYOUT["LAYOUT_VERSION"], 2: kernel_id,
                    4: capacity, 5: stride, 6: len(FIXED_TIMING_NAMES), 7: 1}
        if any(data[i] != value for i, value in expected.items()):
            raise ValueError(f"{kernel}: invalid layout metadata (rebuild OPP and parser together)")
        groups = data[3]
        if not 0 < groups <= capacity:
            raise ValueError(f"{kernel}: blockDim {groups} exceeds reserved capacity {capacity}")
        metadata.append({"kernel": kernel, "block_dim": groups, "arch": data[8]})
        rows = data[LAYOUT["HEADER_WORDS"]:]
        if any(rows[groups * members * stride:]):
            raise ValueError(f"{kernel}: writes outside actual blockDim")
        for core_id in range(groups * members):
            row = rows[core_id * stride:(core_id + 1) * stride]
            parent = row[:2]
            if not all(parent) or parent[1] < parent[0]:
                raise ValueError(f"{kernel}/core {core_id}: missing or invalid KERNEL_TIMING")
            if any(row[len(FIXED_TIMING_NAMES) * 2:]):
                raise ValueError(f"{kernel}/core {core_id}: nonzero alignment padding")
            for stage_id, stage in FIXED_TIMING_NAMES.items():
                start, end = row[stage_id * 2:stage_id * 2 + 2]
                if start == end == 0:
                    if stage_id <= 2 or (kernel == "PHASE6" and stage_id in (3, 4)):
                        raise ValueError(f"{kernel}/core {core_id}: missing {stage}")
                    continue
                if not start or not end or end < start:
                    raise ValueError(f"{kernel}/core {core_id}/{stage}: unpaired or negative duration")
                if start < parent[0] or end > parent[1]:
                    raise ValueError(f"{kernel}/core {core_id}/{stage}: outside KERNEL_TIMING")
                if kernel != "PHASE6" and stage_id > 2:
                    raise ValueError(f"{kernel}: unexpected Phase6 event")
                member = core_id % members
                events.append(dict(kernel=kernel, core_id=core_id, group_id=core_id // members,
                                   core_type="AIC" if member == 0 else "AIV", sub_id=member - 1,
                                   stage=stage, kind="point" if stage in POINTS else
                                   "issue" if stage == "MAIN_INIT" else "window",
                                   start_cycle=start, end_cycle=end, duration_cycles=end - start))
            if not (row[2] <= row[3] <= row[4] <= row[5] == parent[1]):
                raise ValueError(f"{kernel}/core {core_id}: invalid INIT/BODY order")
            if kernel == "PHASE6":
                if not (row[4] <= row[6] <= row[7] <= row[8] <= row[9]):
                    raise ValueError(f"{kernel}/core {core_id}: invalid dispatch boundary order")
                if row[10] and row[10] < row[9]:
                    raise ValueError(f"{kernel}/core {core_id}: H/O wait precedes H dispatch boundary")
        if kernel == "PHASE6":
            waits = [bool(rows[core * stride + 10]) for core in range(groups * members)]
            if any(waits) and not all(waits):
                raise ValueError("PHASE6: inconsistent serial H/O wait branch across cores")
    if not metadata:
        raise ValueError("no captured kernels: empty input, timer disabled, or wrong loaded OPP")
    kernels = {item["kernel"] for item in metadata}
    if "PHASE6" in kernels and len(kernels) != 1:
        raise ValueError("mutually exclusive GDN routes appear in one buffer; zero before each call")
    if "PHASE6" not in kernels and kernels != {"PREPARE", "FWD_H", "FWD_O"}:
        missing = {"PREPARE", "FWD_H", "FWD_O"} - kernels
        raise ValueError(f"incomplete Prepare/H/O capture: captured={','.join(sorted(kernels))}; "
                         f"missing={','.join(sorted(missing))}; keep raw/metadata and check timer GM publication and loaded kernels")
    return events, metadata


def parse_and_dump(raw_path, output_dir, cycles_per_us=None):
    validate_frequency(cycles_per_us)
    events, metadata = parse_timer(read_raw(raw_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gdn_npu_time_parsed.csv"
    fields = FIELDS + (["duration_us"] if cycles_per_us is not None else [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            record = dict(event)
            if cycles_per_us is not None:
                record["duration_us"] = event["duration_cycles"] / cycles_per_us
            writer.writerow(record)
    (output_dir / "timer_layout.json").write_text(json.dumps({
        "layout": LAYOUT, "kernels": metadata, "cycles_per_us": cycles_per_us,
        "clock_alignment": "unverified; raw cycles only",
        "init_semantics": "issue window, may include asynchronous initialization",
        "point_semantics": "dispatch boundary, midpoint of adjacent Tik/Tok pair",
    }, indent=2), encoding="utf-8")
    return path, events, metadata


def print_summary(events):
    groups = {}
    for event in events:
        if event["kind"] != "point":
            groups.setdefault((event["kernel"], event["stage"], event["core_type"]), []).append(
                event["duration_cycles"])
    print("kernel stage core_type samples min/median/max (cycles; INIT is issue window)")
    for key, durations in sorted(groups.items()):
        print(*key, len(durations), min(durations), statistics.median(durations), max(durations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("timer_parsed"))
    parser.add_argument("--cycles-per-us", type=float)
    args = parser.parse_args()
    path, events, _ = parse_and_dump(args.raw, args.output_dir, args.cycles_per_us)
    print_summary(events)
    print(path)


if __name__ == "__main__":
    main()

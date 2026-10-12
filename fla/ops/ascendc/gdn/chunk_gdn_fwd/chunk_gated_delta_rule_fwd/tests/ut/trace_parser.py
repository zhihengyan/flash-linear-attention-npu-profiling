"""Perfetto export of validated Scheme A CSV. Frequency/common clock must be verified.

Adapted from op-timer-ascendc (52e396e); MIT, Copyright (c) 2026 Send_Tanks.
See ../../op_kernel/AscendTimerV2_LICENSE.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from parse_timer_csv import KERNEL_NAMES, validate_frequency


def generate_trace_json(events, output_path, cycles_per_us, *, shared_clock=False):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None or not shared_clock:
        raise ValueError("trace requires verified cycles_per_us and shared_clock=True")
    trace, origins, tracks = [], {}, set()
    pids = {name: index + 1 for index, name in KERNEL_NAMES.items()}
    for event in events:
        kernel = event["kernel"]
        origins[kernel] = min(origins.get(kernel, event["start_cycle"]), event["start_cycle"])
    for kernel, pid in pids.items():
        if kernel in origins:
            trace.append(dict(ph="M", name="process_name", pid=pid,
                              args={"name": kernel + " (separate kernel origin)"}))
    for event in events:
        kernel, core = event["kernel"], event["core_id"]
        pid = pids[kernel]
        if (pid, core) not in tracks:
            tracks.add((pid, core))
            name = f"group_{event['group_id']}_{event['core_type']}_{event['sub_id']}"
            trace.append(dict(ph="M", name="thread_name", pid=pid, tid=core, args={"name": name}))
        # Subtract integer origins before conversion; counters can exceed 2**53.
        start = event["start_cycle"] - origins[kernel]
        record = dict(name=event["stage"], cat=event["kind"], pid=pid, tid=core,
                      ts=start / cycles_per_us, args={k: event[k] for k in
                      ("start_cycle", "end_cycle", "duration_cycles", "kind")})
        if event["kind"] == "point":
            record.update(ph="i", s="t", ts=(start + event["duration_cycles"] / 2) / cycles_per_us)
        else:
            record.update(ph="X", dur=event["duration_cycles"] / cycles_per_us)
        trace.append(record)
    result = {"traceEvents": trace, "displayTimeUnit": "us", "metadata": {
        "cycles_per_us": cycles_per_us, "shared_clock_verified_by_caller": True,
        "alignment": "each kernel origin is independent; not a host end-to-end timeline",
        "MAIN_INIT": "issue window", "points": "dispatch boundaries, not pipeline completion"}}
    output_path = Path(output_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gdn_npu_time.json"))
    parser.add_argument("--cycles-per-us", type=float, required=True)
    parser.add_argument("--shared-clock", action="store_true", help="assert verified common AIC/AIV counter")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    for event in events:
        for key in ("core_id", "group_id", "sub_id", "start_cycle", "end_cycle", "duration_cycles"):
            event[key] = int(event[key])
        if event["end_cycle"] < event["start_cycle"] or event["duration_cycles"] != event["end_cycle"] - event["start_cycle"]:
            raise ValueError("invalid CSV duration")
    print(generate_trace_json(events, args.output, args.cycles_per_us, shared_clock=args.shared_clock))


if __name__ == "__main__":
    main()

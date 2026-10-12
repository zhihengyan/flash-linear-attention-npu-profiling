"""Parsed timer CSV to Perfetto; a verified clock and shared-clock assertion are required."""
import argparse
import csv
import json
from pathlib import Path
from parse_timer_csv import is_accumulated, validate_frequency

def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)

def collect_event_names(fields):
    return [name[:-len("_start_cycle")] for name in fields if name.endswith("_start_cycle")
            and name[:-len("_start_cycle")] + "_end_cycle" in fields]

def find_min_start(path):
    fields, rows = read_csv_rows(path)
    starts = [int(row[event + "_start_cycle"]) for event in collect_event_names(fields)
              for row in rows if not is_accumulated(event) and int(row[event + "_start_cycle"]) > 0]
    return min(starts) if starts else 0

def generate_trace_json(csv_path, rank=0, base_cycle=None, *, cycles_per_us=None, shared_clock=False):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None or not shared_clock:
        raise ValueError("trace requires verified cycles_per_us and shared_clock=True")
    fields, rows = read_csv_rows(csv_path)
    origin = find_min_start(csv_path) if base_cycle is None else base_cycle
    events = []
    for row in rows:
        for name in collect_event_names(fields):
            if is_accumulated(name):
                continue
            start, end = int(row[name + "_start_cycle"]), int(row[name + "_end_cycle"])
            if (start, end) == (0, 0):
                continue
            if start <= 0 or end < start or start < origin:
                raise ValueError(f"invalid trace interval: {name}")
            event = {"ph": "X", "cat": row["core_type"], "pid": f"capture_{rank}",
                     "tid": f"group_{row['group_id']}_{row['core_type']}_{row['sub_id']}",
                     "name": name, "ts": (start-origin) / cycles_per_us, "dur": (end-start) / cycles_per_us,
                     "args": {"start_cycle": start, "end_cycle": end, "origin_cycle": origin}}
            if start == end:
                event.update(ph="i", s="t")
                event.pop("dur")
            events.append(event)
    return events

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles-per-us", type=float, required=True)
    parser.add_argument("--shared-clock", action="store_true")
    args = parser.parse_args()
    events = generate_trace_json(args.csv, cycles_per_us=args.cycles_per_us, shared_clock=args.shared_clock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=False, allow_nan=False, indent=2)
    print(f"Trace: {args.output.resolve()} ({len(events)} events)")

if __name__ == "__main__":
    main()

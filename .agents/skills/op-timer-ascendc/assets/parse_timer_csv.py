"""Raw start/end timer to CSV; default unit cycles, no assumed clock frequency."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

FIXED_TIMING_NAMES = {
    0: "KERNEL_TIMING",
    1: "FIXED_STAGE_1",
    2: "FIXED_STAGE_2",
    3: "FIXED_STAGE_3",
    4: "FIXED_STAGE_4",
    5: "FIXED_STAGE_5",
    6: "FIXED_STAGE_6",
}

DYNAMIC_TIMING_NAMES = {
    0: "DYN_STAGE_1",
    1: "DYN_STAGE_2",
    2: "DYN_STAGE_3",
    3: "DYN_STAGE_4",
    4: "DYN_STAGE_5",
    5: "DYN_STAGE_6",
    6: "DYN_STAGE_7",
    7: "DYN_STAGE_8",
    8: "DYN_STAGE_9",
    9: "DYN_STAGE_10",
    10: "DYN_STAGE_11",
    11: "DYN_STAGE_12",
    12: "DYN_STAGE_13",
    13: "DYN_STAGE_14",
    14: "DYN_STAGE_15",
    15: "DYN_STAGE_16",
    16: "DYN_STAGE_17",
    17: "DYN_STAGE_18",
    18: "DYN_STAGE_19",
}

FIXED_TIMING_COUNT = len(FIXED_TIMING_NAMES)
DYNAMIC_TYPE_COUNT_ENUM = len(DYNAMIC_TIMING_NAMES)
MAX_DYNAMIC_ITER = 64
# 与AscendTimerV2.hpp一致
N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER
N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2
N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) // 16 * 16
N_CORE_COUNT = 96
N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT

DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM
DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) // 16 * 16
DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT

TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE

CORES_PER_GROUP = 3

# Customize with the kernel. Accumulate has a synthetic end; exclude it from time-window/trace checks.
ACCUMULATED_EVENTS = set()

def validate_frequency(cycles_per_us):
    if cycles_per_us is not None and (not math.isfinite(cycles_per_us) or cycles_per_us <= 0):
        raise ValueError("cycles_per_us must be a verified finite positive value")

def load_raw_data(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [int(value) for row in csv.reader(handle) for value in row if value.strip()]

def split_data(raw):
    if len(raw) != TOTAL_BUFFER_SIZE:
        raise ValueError(f"raw words {len(raw)} != layout {TOTAL_BUFFER_SIZE}; check installed kernel and parser layout")
    return raw[:N_TIMING_COUNTER], raw[N_TIMING_COUNTER:]

def compute_dynamic_max_iter(counts):
    result = [0] * DYNAMIC_TYPE_COUNT
    for core in range(N_CORE_COUNT):
        for kind in range(DYNAMIC_TYPE_COUNT):
            count = counts[core * DYNAMIC_ITER_PER_CORE_ALIGN + kind]
            if not 0 <= count < 2**63:
                raise ValueError(f"invalid dynamic count {count} at core {core}, type {kind}")
            result[kind] = min(MAX_DYNAMIC_ITER, max(result[kind], count))
    return result

def get_timing_names(max_iters):
    return list(FIXED_TIMING_NAMES.values()) + [f"{name}_{i}" for kind, name in DYNAMIC_TIMING_NAMES.items()
                                             for i in range(max_iters[kind])]

def parse_dynamic_name(name):
    prefix, _, number = name.rpartition("_")
    for kind, label in DYNAMIC_TIMING_NAMES.items():
        if prefix == label and number.isdigit():
            return kind, int(number)
    return None, None

def get_dynamic_timing_idx(kind, iteration):
    return FIXED_TIMING_COUNT + kind * MAX_DYNAMIC_ITER + iteration

def get_core_info(core):
    group, member = divmod(core, CORES_PER_GROUP)
    return (group, "AIC", 0) if member == 0 else (group, "AIV", member - 1)

def get_base_index(core):
    return core * N_TIMING_COUNTER_PER_CORE_ALIGN

def get_event_index(base, logical):
    return base + logical * 2

def timing_logical_idx(index, name):
    if index < FIXED_TIMING_COUNT:
        return index
    kind, iteration = parse_dynamic_name(name)
    return get_dynamic_timing_idx(kind, iteration)

def is_accumulated(name):
    prefix, _ = parse_dynamic_name(name)
    return name in ACCUMULATED_EVENTS or (prefix is not None and DYNAMIC_TIMING_NAMES[prefix] in ACCUMULATED_EVENTS)

def parse_and_dump(raw_csv_path, output_csv_path, cycles_per_us=None, expected_core_ids=None):
    validate_frequency(cycles_per_us)
    raw = load_raw_data(raw_csv_path)
    base, counts = split_data(raw)
    names = get_timing_names(compute_dynamic_max_iter(counts))
    expected = None if expected_core_ids is None else set(expected_core_ids)
    if expected is not None and (not expected or any(core < 0 or core >= N_CORE_COUNT for core in expected)):
        raise ValueError("expected core IDs are empty or exceed configured capacity")
    rows, active, truncated = [], set(), []
    for core in range(N_CORE_COUNT):
        group, kind, sub = get_core_info(core)
        offset = get_base_index(core)
        kernel_start, kernel_end = base[offset:offset + 2]
        core_counts = counts[core * DYNAMIC_ITER_PER_CORE_ALIGN:(core + 1) * DYNAMIC_ITER_PER_CORE_ALIGN]
        row_data = base[offset:offset + N_TIMING_COUNTER_PER_CORE_ALIGN]
        present = any(row_data) or any(core_counts)
        if present:
            if not 0 < kernel_start <= kernel_end:
                raise ValueError(f"incomplete KERNEL_TIMING at core {core}")
            active.add(core)
            for type_id, label in DYNAMIC_TIMING_NAMES.items():
                actual = core_counts[type_id]
                if actual > MAX_DYNAMIC_ITER:
                    truncated.append({"core_id": core, "event_type": label, "actual_iterations": actual,
                                      "recorded_iteration_limit": MAX_DYNAMIC_ITER})
                # Reject residual records beyond the declared count, including types omitted from CSV columns.
                first_unrecorded = min(actual, MAX_DYNAMIC_ITER)
                first = offset + get_dynamic_timing_idx(type_id, first_unrecorded) * 2
                end = offset + get_dynamic_timing_idx(type_id, MAX_DYNAMIC_ITER) * 2
                if any(base[first:end]):
                    raise ValueError(f"event exceeds declared iteration count at core {core}: {label}")
        row = [core, group, kind, sub]
        for index, name in enumerate(names):
            logical = timing_logical_idx(index, name)
            start, end = base[get_event_index(offset, logical):get_event_index(offset, logical) + 2]
            if (start, end) != (0, 0):
                if not 0 < start <= end:
                    raise ValueError(f"invalid start/end at core {core}, event {name}")
                if not is_accumulated(name) and not kernel_start <= start <= end <= kernel_end:
                    raise ValueError(f"event outside KERNEL_TIMING at core {core}: {name}")
                type_id, iteration = parse_dynamic_name(name)
                if type_id is not None and iteration >= core_counts[type_id]:
                    raise ValueError(f"event exceeds declared iteration count at core {core}: {name}")
            duration = end - start
            row.extend([start, end, duration])
            if cycles_per_us is not None:
                row.append(duration / cycles_per_us)
        for count in core_counts[:DYNAMIC_TYPE_COUNT]:
            row.extend([count, min(count, MAX_DYNAMIC_ITER), int(count > MAX_DYNAMIC_ITER)])
        rows.append(row)
    if not active:
        raise ValueError("no active timer rows; check actual route, enabled timer, loaded artifact and GM publication")
    if expected is not None and active != expected:
        raise ValueError(f"core capture mismatch: missing={sorted(expected-active)}, extra={sorted(active-expected)}")
    columns = ["core_id", "group_id", "core_type", "sub_id"]
    for name in names:
        columns += [name + "_start_cycle", name + "_end_cycle", name + "_duration_cycles"]
        if cycles_per_us is not None:
            columns.append(name + "_duration_us")
    for name in DYNAMIC_TIMING_NAMES.values():
        columns += [name + "_IterCount", name + "_RecordedIterLimit", name + "_Truncated"]
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    metadata = {"unit": "cycles", "cycles_per_us": cycles_per_us, "core_capacity": N_CORE_COUNT,
                "max_dynamic_iter": MAX_DYNAMIC_ITER, "active_core_ids": sorted(active),
                "expected_core_ids": sorted(expected) if expected is not None else None,
                "core_presence_verified": expected is not None, "truncated_events": truncated}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    if truncated:
        print(f"WARNING: {len(truncated)} dynamic counters exceed recording capacity; see {path.with_suffix('.metadata.json')}")
    print(f"CSV: {path.resolve()} ({len(active)} active rows; presence {'verified' if expected is not None else 'not verified against launch'})")
    return path

def print_summary(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames or []
    for column in fields:
        if not column.endswith("_duration_cycles"):
            continue
        event = column[:-len("_duration_cycles")]
        for kind in ("AIC", "AIV"):
            values = [int(row[column]) for row in rows if row["core_type"] == kind and int(row[event + "_start_cycle"]) > 0]
            if values:
                print(f"{event} {kind}: n={len(values)}, min/median/max={min(values)}/{median(values)}/{max(values)} cycles")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles-per-us", type=float)
    parser.add_argument("--expected-core-ids", type=int, nargs="+")
    args = parser.parse_args()
    print_summary(parse_and_dump(args.raw, args.output, args.cycles_per_us, args.expected_core_ids))

if __name__ == "__main__":
    main()

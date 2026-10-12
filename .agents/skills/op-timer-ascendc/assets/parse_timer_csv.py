"""
Parse AscendTimerV2 timer CSV data.

New timer layout stores one event as two int64 values:
  [start_cycle, end_cycle]

Current layout:
  [0, N_TIMING_COUNTER)           : per-core timing start/end pairs
  [N_TIMING_COUNTER, TOTAL_SIZE)  : per-core dynamic iteration counts

Usage:
  python parse_timer_csv.py --dir timer --rank-start 0 --rank-end 7
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# 与AscendTimerV2.hpp一致（模板占位，需按你的算子替换为同名阶段名，见 SKILL.md Step 3.3）
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
# 按实际方案填 Accumulate 阶段/动态类型前缀，合成end不进入trace。
ACCUMULATED_EVENTS = set()

def validate_frequency(cycles_per_us):
    if cycles_per_us is not None and (not math.isfinite(cycles_per_us) or cycles_per_us <= 0):
        raise ValueError("cycles_per_us must be a verified finite positive number")

def cycle_to_us(cycles, cycles_per_us):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None:
        raise ValueError("clock frequency is not confirmed")
    return cycles / cycles_per_us

def is_accumulated(name):
    kind, _ = parse_dynamic_name(name)
    return name in ACCUMULATED_EVENTS or (kind is not None and DYNAMIC_TIMING_NAMES[kind] in ACCUMULATED_EVENTS)


def load_raw_data(csv_path):
    data = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    data.append(int(item))
    return data


def split_data(raw_data):
    if len(raw_data) != TOTAL_BUFFER_SIZE:
        print(f"[ERROR] 数据量 {len(raw_data)} != 新版 start/end timer 格式预期 {TOTAL_BUFFER_SIZE}")
        print("[ERROR] 请确认目标算子的 kernel/host 与 torch_extension 已重新编译安装，旧 duration 格式不再支持。")
        sys.exit(1)
    base_cycle_data = raw_data[:N_TIMING_COUNTER]
    dynamic_iter_data = raw_data[N_TIMING_COUNTER:N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE]
    return base_cycle_data, dynamic_iter_data


def compute_dynamic_max_iter(dynamic_iter_data):
    max_iters = [0] * DYNAMIC_TYPE_COUNT
    for core_id in range(N_CORE_COUNT):
        core_offset = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
        for t in range(DYNAMIC_TYPE_COUNT):
            idx = core_offset + t
            if idx < len(dynamic_iter_data) and dynamic_iter_data[idx] > max_iters[t]:
                max_iters[t] = dynamic_iter_data[idx]
    return max_iters


def get_timing_names(dynamic_max_iters):
    names = [FIXED_TIMING_NAMES.get(i, f"UNKNOWN_FIXED_{i}") for i in range(FIXED_TIMING_COUNT)]
    for t in range(DYNAMIC_TYPE_COUNT):
        prefix = DYNAMIC_TIMING_NAMES.get(t, f"UNKNOWN_DYN_{t}")
        actual_max = dynamic_max_iters[t] if t < len(dynamic_max_iters) and dynamic_max_iters[t] > 0 else MAX_DYNAMIC_ITER
        actual_max = min(actual_max, MAX_DYNAMIC_ITER)   # 防止 iter 区脏数据导致槽位别名
        for iter_idx in range(actual_max):
            names.append(f"{prefix}_{iter_idx}")
    return names


def parse_dynamic_name(name):
    last_underscore = name.rfind("_")
    if last_underscore < 0:
        return None, None
    prefix = name[:last_underscore]
    iter_str = name[last_underscore + 1:]
    if not iter_str.isdigit():
        return None, None
    for type_id, type_name in DYNAMIC_TIMING_NAMES.items():
        if type_name == prefix:
            return type_id, int(iter_str)
    return None, None


def get_dynamic_timing_idx(type_id, iter_idx):
    return FIXED_TIMING_COUNT + type_id * MAX_DYNAMIC_ITER + iter_idx


def get_core_info(core_id):
    group_id = core_id // CORES_PER_GROUP
    sub_group_id = core_id % CORES_PER_GROUP
    if sub_group_id == 0:
        return group_id, "AIC", 0
    return group_id, "AIV", sub_group_id - 1


def get_base_index(core_id):
    group_id, core_type, sub_id = get_core_info(core_id)
    base = group_id * N_TIMING_COUNTER_PER_CORE_ALIGN * CORES_PER_GROUP
    if core_type == "AIV":
        base += N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + sub_id)
    return base


def get_event_index(base_index, logical_idx):
    return base_index + logical_idx * 2


def timing_logical_idx(name_idx, name):
    if name_idx < FIXED_TIMING_COUNT:
        return name_idx
    type_id, iter_idx = parse_dynamic_name(name)
    if type_id is None:
        return -1
    return get_dynamic_timing_idx(type_id, iter_idx)


def parse_and_dump(raw_csv_path, output_csv_path, cycles_per_us=None):
    validate_frequency(cycles_per_us)
    raw_data = load_raw_data(raw_csv_path)
    base_cycle_data, dynamic_iter_data = split_data(raw_data)
    if not any(base_cycle_data):
        raise ValueError("no timer records; check actual route, loaded kernel, enabled timer and GM publication")
    dynamic_max_iters = compute_dynamic_max_iter(dynamic_iter_data)

    print("[INFO] timer layout: start_end")
    print("[INFO] 动态项全局最大迭代次数:")
    for t in range(DYNAMIC_TYPE_COUNT):
        print(f"  {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}: {dynamic_max_iters[t]}")
        if dynamic_max_iters[t] > MAX_DYNAMIC_ITER :
            print(f"    [WARNING] {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')} 实际迭代次数 {dynamic_max_iters[t]} "
                    f"超过上限 {MAX_DYNAMIC_ITER}，超出部分迭代的计时已被截断/未记录")

    timing_names = get_timing_names(dynamic_max_iters)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name in timing_names:
            header += [f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration_cycles"]
            if cycles_per_us is not None:
                header.append(f"{name}_duration(us)")
        header += [f"{DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}_IterCount" for t in range(DYNAMIC_TYPE_COUNT)]
        writer.writerow(header)

        for core_id in range(N_CORE_COUNT):
            group_id, core_type, sub_id = get_core_info(core_id)
            base_index = get_base_index(core_id)
            kernel_start, kernel_end = base_cycle_data[base_index:base_index + 2]
            count_start = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
            active = any(base_cycle_data[base_index:base_index + N_TIMING_COUNTER_PER_CORE_ALIGN]) or any(
                dynamic_iter_data[count_start:count_start + DYNAMIC_ITER_PER_CORE_ALIGN])
            if active and not 0 < kernel_start <= kernel_end:
                raise ValueError(f"incomplete KERNEL_TIMING at core {core_id}")
            row = [core_id, group_id, core_type, sub_id]

            for name_idx, name in enumerate(timing_names):
                logical_idx = timing_logical_idx(name_idx, name)
                event_idx = get_event_index(base_index, logical_idx) if logical_idx >= 0 else -1

                if 0 <= event_idx + 1 < len(base_cycle_data):
                    start_cycle = base_cycle_data[event_idx]
                    end_cycle = base_cycle_data[event_idx + 1]
                    if (start_cycle, end_cycle) != (0, 0) and not 0 < start_cycle <= end_cycle:
                        raise ValueError(f"invalid interval at core {core_id}: {name}")
                    if start_cycle > 0 and (not kernel_start <= start_cycle <= kernel_end or
                                             (not is_accumulated(name) and end_cycle > kernel_end)):
                        raise ValueError(f"event outside KERNEL_TIMING at core {core_id}: {name}")
                    duration_cycles = end_cycle - start_cycle
                    row += [str(start_cycle), str(end_cycle), str(duration_cycles)]
                    if cycles_per_us is not None:
                        row.append(f"{cycle_to_us(duration_cycles, cycles_per_us):.2f}")
                else:
                    row += ["0", "0", "0"]
                    if cycles_per_us is not None:
                        row.append("0.00")

            for t in range(DYNAMIC_TYPE_COUNT):
                idx = core_id * DYNAMIC_ITER_PER_CORE_ALIGN + t
                row.append(str(dynamic_iter_data[idx]) if idx < len(dynamic_iter_data) else "0")

            writer.writerow(row)

    print(f"[INFO] 解析完成, 结果写入: {output_csv_path}")


def print_summary(output_csv_path):
    with open(output_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        kernel_col = "KERNEL_TIMING_duration(us)" if "KERNEL_TIMING_duration(us)" in header else "KERNEL_TIMING_duration_cycles"
        unit = "us" if kernel_col.endswith("(us)") else "cycles"
        if kernel_col not in header:
            print("[WARNING] 未找到 KERNEL_TIMING duration")
            return
        kernel_idx = header.index(kernel_col)

        print("\n[INFO] 有计时数据的核心 (KERNEL_TIMING > 0):")
        print(f"{'core_id':>8} {'group':>6} {'type':>5} {'sub':>4} {('KERNEL_TIMING('+unit+')'):>22}")
        print("-" * 50)
        count = 0
        for row in reader:
            try:
                val = float(row[kernel_idx])
            except (ValueError, IndexError):
                continue
            if val > 0:
                print(f"{row[0]:>8} {row[1]:>6} {row[2]:>5} {row[3]:>4} {val:>18.2f}")
                count += 1
        if count == 0:
            print("  (无有效计时数据)")
        print(f"\n[INFO] 共 {count} 个核心有计时数据")


if __name__ == "__main__":
    default_timer_dir = str(Path.cwd() / "timer")

    parser = argparse.ArgumentParser(description="解析 AscendTimerV2 格式的 timer CSV 数据")
    parser.add_argument("--dir", "-d", default=default_timer_dir, help="timer CSV 文件所在目录")
    parser.add_argument("--rank-start", type=int, default=0, help="起始 rank (包含)")
    parser.add_argument("--rank-end", type=int, default=7, help="结束 rank (包含)")
    parser.add_argument("--cycles-per-us", type=float, help="verified clock frequency; default exports cycles")
    args = parser.parse_args()

    timer_dir = Path(args.dir)
    for rank in range(args.rank_start, args.rank_end + 1):
        input_path = timer_dir / f"rank_{rank}_npu_time.csv"
        output_path = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if not input_path.exists():
            print(f"[SKIP] rank_{rank}: 文件不存在 {input_path}")
            continue
        print(f"\n{'='*60}")
        print(f"[INFO] 处理 rank_{rank}")
        print(f"{'='*60}")
        parse_and_dump(str(input_path), str(output_path), args.cycles_per_us)
        print_summary(str(output_path))

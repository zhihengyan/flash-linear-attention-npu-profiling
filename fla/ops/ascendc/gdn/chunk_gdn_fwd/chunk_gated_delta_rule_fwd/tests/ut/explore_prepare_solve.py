"""Offline Stage-03 screening; no ATK golden or device implementation is changed.

Run with NumPy and --results-root pointing to the supplied read-only captures.
The FP32/BF16 models do not simulate Ascend instructions or dynamic scheduling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import statistics

import numpy as np


ORDER = ["Cast", "Transpose", "Transpose", "Prepare", "H", "O", "Transpose"]
PIPE_FIELDS = [
    "aic_mac_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
    "aic_fixpipe_time(us)", "aic_scalar_time(us)", "aiv_vec_time(us)",
    "aiv_mte2_time(us)", "aiv_mte3_time(us)", "aiv_scalar_time(us)",
]


def role(name):
    for needle, label in (("ChunkGatedDeltaRuleFwdPrepare", "Prepare"),
                          ("ChunkFwdH", "H"), ("ChunkFwdO", "O"),
                          ("Transpose", "Transpose"), ("Cast", "Cast")):
        if needle in name:
            return label
    raise ValueError(f"Unknown kernel: {name}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_captures(root):
    all_samples = {"A": defaultdict(list), "K32": defaultdict(list)}
    pipe_samples = {label: defaultdict(list) for label in all_samples}
    sources, identities = {}, []
    for label, parent, prefix in (
        ("A", "atk_output_profiling_1015", "A"),
        ("K32", "atk_output_k32", "k32_B"),
    ):
        for run in (1, 2, 3):
            directory = root / parent / f"{prefix}_{run}_20261015"
            meta_path, csv_path = directory / "run_metadata.json", directory / "op_summary.csv"
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            assert meta["status"] == "complete" and meta["case_ids"] == [0]
            evidence = meta["performance_evidence"]
            assert evidence["executed_case_ids"] == [0]
            for report in evidence["reports"]:
                state = report["statistic"]
                assert state["case_ids"] == [0]
                assert not state["execution_failed_case_ids"]
                assert not state["accuracy_failed_case_ids"]
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            recorded = evidence["target_kernels"]
            assert len(rows) == len(recorded) == 140
            assert [(r["Op Name"], Decimal(r["Task Duration(us)"].strip())) for r in rows] == [
                (next(n for n in r["name"] if n), Decimal(str(r["task_duration_us"])))
                for r in recorded
            ]
            assert rows == sorted(rows, key=lambda r: (
                Decimal(r["Task Start Time(us)"].strip()), int(r["Task ID"])))
            identities.append({key: meta[key] for key in (
                "case_json_sha256", "input_sha256", "soc", "device", "python")})
            for first in range(0, 140, 7):
                group = rows[first:first + 7]
                assert [role(r["Op Name"]) for r in group] == ORDER
                durations = defaultdict(Decimal)
                for row in group:
                    duration = Decimal(row["Task Duration(us)"].strip())
                    assert duration.is_finite() and duration > 0
                    kernel = role(row["Op Name"])
                    durations[kernel] += duration
                    if kernel in ("Prepare", "H", "O"):
                        for field in PIPE_FIELDS:
                            pipe_samples[label][f"{kernel}/{field}"].append(float(row[field]))
                durations["Total"] = sum(durations.values())
                for kernel, value in durations.items():
                    all_samples[label][kernel].append(float(value))
            for path in (meta_path, csv_path):
                sources[str(path)] = sha256(path)
    assert all(identity == identities[0] for identity in identities)
    return {
        "samples_per_version": 60,
        "identity": identities[0],
        "sources_sha256": sources,
        "mean_us": {label: {key: statistics.mean(values) for key, values in data.items()}
                    for label, data in all_samples.items()},
        "pipe_mean_us_not_additive": {
            label: {key: statistics.mean(values) for key, values in data.items()}
            for label, data in pipe_samples.items()},
    }


def bf16_rne(values):
    """Finite FP32 -> BF16 round-to-nearest-even -> FP32, for model comparison."""
    values = np.asarray(values, dtype=np.float32)
    assert np.isfinite(values).all()
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1))
            & np.uint32(0xFFFF0000)).view(np.float32)


def numerical_screen():
    eye = np.eye(64, dtype=np.float32)
    maxima = defaultdict(float)
    mismatched_bf16, count = 0, 0
    negative_control_detected = False
    for valid in range(1, 65):
        for seed in range(4):
            for amplitude in (0.0, 0.02, 0.2):
                rng = np.random.default_rng(seed)
                lower = np.tril(rng.uniform(-amplitude, amplitude, (64, 64)), -1).astype(np.float32)
                lower[valid:, :] = 0
                lower[:, valid:] = 0
                r00 = np.linalg.inv(np.eye(32) + lower[:32, :32]).astype(np.float32)
                r11 = np.linalg.inv(np.eye(32) + lower[32:, 32:]).astype(np.float32)
                left, right = np.zeros_like(lower), np.zeros_like(lower)
                left[32:, 32:], right[:32, :32] = r11, r00
                # Current K32's live output dependencies, with FP32 operations.
                y = eye + left[:, 32:] @ (-lower[32:, :])
                current = eye[:, 32:] @ left[32:, :] + y[:, :32] @ right[:32, :]
                # Conservative N32: keep full A initialization and update left half.
                y_left = eye[:, :32] + left[:, 32:] @ (-lower[32:, :32])
                n32 = eye[:, 32:] @ left[32:, :]
                n32[:, :32] += y_left @ r00
                # Block solve: packed leaf representation is physically transposed.
                packed = np.concatenate((r00.T, r11.T), axis=1)
                x = r11 @ (-lower[32:, :32])
                cross = x @ r00
                block = np.zeros_like(lower)
                block[:32, :32] = packed[:, :32].T
                block[32:, 32:] = packed[:, 32:].T
                block[32:, :32] = cross
                golden = np.linalg.solve(np.eye(64) + lower.astype(np.float64), np.eye(64))
                for name, actual in (("n32", n32), ("block", block)):
                    np.testing.assert_allclose(actual, current, rtol=3e-6, atol=3e-7)
                    np.testing.assert_allclose(actual, golden, rtol=3e-6, atol=3e-7)
                    np.testing.assert_array_equal(actual[valid:, valid:], eye[valid:, valid:])
                    assert not np.any(actual[:valid, valid:])
                    assert not np.any(actual[valid:, :valid])
                    maxima[f"{name}_vs_k32_max_abs"] = max(
                        maxima[f"{name}_vs_k32_max_abs"], float(np.max(np.abs(actual - current))))
                    maxima[f"{name}_vs_fp64_max_abs"] = max(
                        maxima[f"{name}_vs_fp64_max_abs"], float(np.max(np.abs(actual - golden))))
                cast_current, cast_block = bf16_rne(current), bf16_rne(block)
                mismatched_bf16 += int(np.count_nonzero(cast_current != cast_block))
                # A feeds both W and U; two independent 128-column RHSs.
                rhs = bf16_rne(rng.uniform(-1, 1, (64, 256)).astype(np.float32))
                maxima["wu_fp32_max_abs"] = max(maxima["wu_fp32_max_abs"], float(
                    np.max(np.abs(cast_current @ rhs - cast_block @ rhs))))
                if not np.allclose(packed[:, :32], r00, rtol=1e-5, atol=1e-6):
                    negative_control_detected = True
                count += 1
    assert negative_control_detected, "The non-symmetric leaf must detect a missing transpose"
    # Logical quadrant assembly covers each NZ element exactly once and splits
    # on whole 32-byte blocks. This does not validate a LoadData/Fixpipe descriptor.
    row, col = np.indices((64, 64))
    quadrant_blocks = []
    for r0, c0 in ((0, 0), (0, 32), (32, 0), (32, 32)):
        mask = (row >= r0) & (row < r0 + 32) & (col >= c0) & (col < c0 + 32)
        offsets = ((col // 16) * 64 * 16 + row * 16 + col % 16)[mask]
        quadrant_blocks.append(set((offsets // 16).tolist()))
    assert set.union(*quadrant_blocks) == set(range(256))
    assert sum(map(len, quadrant_blocks)) == 256
    return {"matrix_cases": count, "valid_lengths": "1..64", "seeds": 4,
            "lower_amplitudes": [0.0, 0.02, 0.2], "max_errors": dict(maxima),
            "bf16_A_different_elements": mismatched_bf16,
            "wrong_leaf_transpose_detected": negative_control_detected,
            "logical_bf16_NZ_quadrants_cover_disjoint_32B_blocks": True}


def budgets(captures):
    a, k32 = captures["mean_us"]["A"], captures["mean_us"]["K32"]
    old_saved = a["Prepare"] - k32["Prepare"]
    # A proxy from the previous measured reduction; neither a bound nor a forecast.
    cases = {"K32": 64**3 + 3 * 64 * 64 * 32,
             "N32": 64 * 32 * 64 + 64 * 64 * 32 + 2 * 64 * 32 * 32,
             "block32": 2 * 32**3}
    # User screening floor, not a new measured acceptance requirement.
    screening_fraction = .03
    screening_saving = k32["Total"] * screening_fraction
    result = {}
    for name in ("N32", "block32"):
        removed = cases["K32"] - cases[name]
        proxy = old_saved * removed / (4 * 64**3 - cases["K32"])
        # Prefill is deliberately issued under Stage3 Vector work. Discount its
        # removed MAC completely in a second sensitivity scenario; neither case
        # establishes actual exposed cycles or brackets the measured benefit.
        removed_prefill = 64**3 - 64 * 32 * 64 if name == "N32" else 64**3
        hidden_prefill_proxy = old_saved * (removed - removed_prefill) / (3 * 64 * 64 * 32)
        result[name] = {"solve_MAC_per_task": cases[name],
                        "removed_vs_K32": removed,
                        "solve_MAC_reduction_pct": removed / cases["K32"] * 100,
                        "previous_marginal_proxy_us_NOT_prediction": proxy,
                        "proxy_total_reduction_pct_NOT_prediction": proxy / k32["Total"] * 100,
                        "proxy_less_screening_floor_us": proxy - screening_saving,
                        "removed_prefill_MAC_possibly_hidden": removed_prefill,
                        "prefill_fully_hidden_proxy_us_NOT_prediction": hidden_prefill_proxy,
                        "hidden_prefill_total_reduction_pct_NOT_prediction": hidden_prefill_proxy / k32["Total"] * 100,
                        "hidden_prefill_proxy_less_screening_floor_us": hidden_prefill_proxy - screening_saving}
    conversion_saving = k32["Cast"] + k32["Transpose"]
    return {"K32_solve_MAC_per_task": cases["K32"],
            "user_theoretical_screening_floor_pct": screening_fraction * 100,
            "screening_floor_saving_us": screening_saving,
            "equivalent_total_mean_us_not_acceptance_target": k32["Total"] - screening_saving,
            "single_kernel_reduction_equivalent_pct": {
                name: screening_saving / k32[name] * 100 for name in ("Prepare", "H", "O")},
            "conversion_only_if_other_kernels_unchanged": {
                "optimistic_removed_us": conversion_saving,
                "optimistic_total_reduction_pct": conversion_saving / k32["Total"] * 100,
                "decision": "discard_below_user_theoretical_floor"},
            "candidate_screen": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    captures = read_captures(args.results_root)
    report = {"kind": "offline_exploration_not_device_acceptance",
              "baseline_kernel_commit": "dd707730627b29b1862fd335dfbc76e7f0e2666d",
              "captures": captures, "numerical_screen": numerical_screen(),
              "budgets": budgets(captures)}
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()

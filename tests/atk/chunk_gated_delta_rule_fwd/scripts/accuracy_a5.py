#!/usr/bin/env python3
"""Run fused GDN against the existing CPU FP64 golden, without importing ATK.

The JSON supplies shapes, dtypes, attributes and input ranges. It does not supply
rtol/atol: its three ATK double-benchmark ratios are a different criterion.
See README_standalone_a5.md before choosing tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


CASE_DIR = Path(__file__).resolve().parents[1]
ACCURACY_JSON = CASE_DIR / "atk_chunk_gated_delta_rule_fwd.json"
PERF_JSON = CASE_DIR / "atk_chunk_gated_delta_rule_fwd_perf.json"
SCENARIOS = {
    "dense", "varlen", "state_initial_final", "state_initial_only", "state_zero_final"
}
DTYPE_NAMES = {"bf16", "fp16"}


def read_cases(path):
    """Read only ordinary JSON; neither the ATK generator nor executor is used."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("case file must contain a nonempty JSON array")
    ids = set()
    for record in records:
        case_id = int(record["id"])
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        specs = {item["name"]: item for item in record["inputs"]}
        if len(specs) != len(record["inputs"]):
            raise ValueError(f"case {case_id}: duplicate input names")
        attrs = {name: item["range_values"] for name, item in specs.items()
                 if item["type"] == "attr"}
        dtype = attrs["qkv_dtype"]
        if dtype not in DTYPE_NAMES:
            raise ValueError(f"case {case_id}: unsupported dtype {dtype}")
        b, hk, t, k = specs["q"]["shape"]
        vb, hv, vt, v = specs["v"]["shape"]
        if min(b, hk, hv, t) <= 0 or k != 128 or v not in (128, 256) or hv % hk:
            raise ValueError(f"case {case_id}: invalid GDN dimensions")
        if (vb, vt) != (b, t) or specs["k"]["shape"] != specs["q"]["shape"]:
            raise ValueError(f"case {case_id}: inconsistent q/k/v shapes")
        if attrs["chunk_size"] not in (64, 128) or not math.isfinite(attrs["scale"]):
            raise ValueError(f"case {case_id}: invalid chunk_size/scale")
        for name in ("q", "k", "v", "g", "beta"):
            spec = specs[name]
            if spec["type"] != "tensor":
                raise ValueError(f"case {case_id}: {name} is not a tensor")
            expected_dtype = "fp32" if name == "g" else dtype
            if spec["dtype"] != expected_dtype:
                raise ValueError(f"case {case_id}: {name} dtype must be {expected_dtype}")
            bounds = spec["range_values"]
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"case {case_id}: invalid {name} range")
            if spec.get("outlier_values"):
                raise ValueError("outlier sampling is not supported by this runner")
        for name in ("g", "beta"):
            if specs[name]["shape"] != [b, t, hv]:
                raise ValueError(f"case {case_id}: {name} must have shape [B,T,Hv]")
        scenario = attrs["scenario"]
        if scenario not in SCENARIOS or not isinstance(attrs["is_varlen"], bool):
            raise ValueError(f"case {case_id}: invalid scenario/is_varlen")
        cu = None
        if attrs["is_varlen"]:
            cu = tuple(int(x) for x in str(attrs["cu_seqlens_spec"]).split(","))
            if (b != 1 or len(cu) < 2 or cu[0] != 0 or cu[-1] != t
                    or any(left >= right for left, right in zip(cu, cu[1:]))):
                raise ValueError(f"case {case_id}: invalid cu_seqlens")
        elif str(attrs["cu_seqlens_spec"]).lower() != "none":
            raise ValueError(f"case {case_id}: dense metadata must be 'none'")
        if (scenario == "dense" and cu is not None) or (scenario == "varlen" and cu is None):
            raise ValueError(f"case {case_id}: scenario and cu_seqlens disagree")
        record["_specs"], record["_attrs"], record["_cu"] = specs, attrs, cu
    return records


def add_case_arguments(parser, default_json):
    parser.add_argument("--case-file", type=Path, default=default_json)
    parser.add_argument("--case-ids", help="comma-separated JSON ids; default: all cases")
    parser.add_argument("--dtype", choices=("bf16", "fp16"))
    parser.add_argument("--list-cases", action="store_true", help="requires only Python stdlib")
    parser.add_argument("--device", type=int, default=0, help="logical NPU id in this process")
    parser.add_argument("--seed-base", type=int, default=20260827)


def select_cases(args):
    records = read_cases(args.case_file)
    if args.case_ids is not None:
        wanted = {int(x.strip()) for x in args.case_ids.split(",") if x.strip()}
        missing = wanted - {int(record["id"]) for record in records}
        if missing:
            raise ValueError(f"case ids not found: {sorted(missing)}")
        records = [record for record in records if int(record["id"]) in wanted]
    if args.dtype:
        records = [record for record in records if record["_attrs"]["qkv_dtype"] == args.dtype]
    if not records:
        raise ValueError("case selection is empty")
    return records


def case_description(record):
    attrs, specs = record["_attrs"], record["_specs"]
    return {
        "id": record["id"], "name": record["name"],
        "shapes": {name: specs[name]["shape"] for name in ("q", "k", "v", "g", "beta")},
        "attributes": attrs,
        "ranges": {name: specs[name]["range_values"] for name in ("q", "k", "v", "g", "beta")},
    }


def case_seed(record, seed_base):
    seed = record.get("default_seed")
    return int(seed_base + int(record["id"]) if seed is None else seed)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_metadata(case_file):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CASE_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "case_file": str(Path(case_file).resolve()), "case_sha256": sha256(case_file),
        "runner_sha256": sha256(__file__),
        "reference_sha256": sha256(CASE_DIR / "gdn_reference.py"),
        "python": sys.version,
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "FLA_NPU_SOC",
        )},
    }


def numerical_modules():
    # These are existing, pure PyTorch helpers. Importing the ATK executor would
    # pull in ATK; deliberately import only gdn_reference instead.
    import torch

    if str(CASE_DIR) not in sys.path:
        sys.path.insert(0, str(CASE_DIR))
    reference = importlib.import_module("gdn_reference")
    return torch, reference


def build_cpu_inputs(record, seed):
    torch, reference = numerical_modules()
    specs, attrs = record["_specs"], record["_attrs"]
    b, hk, t, k = specs["q"]["shape"]
    _, hv, _, v = specs["v"]["shape"]
    case = reference.GdnCase(b, hk, hv, t, k, v, int(attrs["chunk_size"]),
                             float(attrs["scale"]), attrs["scenario"], record["_cu"])
    case.validate()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    inputs = []
    for name in ("q", "k", "v", "g", "beta"):
        spec = specs[name]
        low, high = map(float, spec["range_values"])
        tensor = torch.empty(spec["shape"], dtype=torch.float32, device="cpu")
        tensor.uniform_(low, high, generator=generator)
        inputs.append(tensor.to(dtypes[spec["dtype"]]).contiguous())
    # No L2 normalization, gate transform, sigmoid, clamping or input rescaling.
    initial_state = reference.deterministic_initial_state(case)
    return case, tuple(inputs), initial_state


def npu_runtime(device_id):
    import torch
    import torch_npu
    import fla_npu
    from fla_npu.ops import ascendc

    if device_id < 0 or not torch.npu.is_available():
        raise RuntimeError("a usable Ascend A5 NPU is required")
    torch.npu.set_device(device_id)
    device_name = str(torch.npu.get_device_name(device_id))
    if "950" not in device_name:
        raise RuntimeError(f"expected Ascend A5 / 950, found {device_name}")
    return torch, ascendc, {
        "device": device_id, "device_name": device_name, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "fla_npu_file": str(fla_npu.__file__),
        "fla_npu_version": getattr(fla_npu, "__version__", "unavailable"),
    }


def prepare_call(ascendc, case, inputs, initial_state, device_id):
    torch, reference = numerical_modules()
    device = torch.device(f"npu:{device_id}")
    npu_inputs = tuple(tensor.to(device).contiguous() for tensor in inputs)
    state = None if initial_state is None else initial_state.to(device).contiguous()
    kwargs = {
        "initial_state": state, "output_final_state": case.output_final_state,
        "chunk_size": case.chunk_size, "scale": case.scale,
        "cu_seqlens": None if case.cu_seqlens is None else list(case.cu_seqlens),
        "chunk_indices": reference.canonical_chunk_indices(case.cu_seqlens, case.chunk_size),
        "layout": "BNSD", "use_exp2": False, "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False, "state_v_first": False,
        "disable_recompute": True, "return_intermediate_states": False,
    }

    def call():
        return ascendc.chunk_gated_delta_rule_fwd(*npu_inputs, **kwargs)

    return call, npu_inputs


def collect_outputs(outputs, case, npu_inputs):
    torch, reference = numerical_modules()
    if len(outputs) != 10:
        raise RuntimeError(f"expected the current 10-output interface, got {len(outputs)} outputs")
    o, final_state, g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = outputs
    if any(item is not None for item in (beta_eff, h, q_rstd, k_rstd)):
        raise RuntimeError("disabled optional outputs must be None")
    if q_hat is not npu_inputs[0] or k_hat is not npu_inputs[1]:
        raise RuntimeError("with L2Norm disabled, q_hat/k_hat must alias the inputs")
    if not case.output_final_state and final_state is not None:
        raise RuntimeError("unexpected final_state")
    raw = {"o": o, "g_cumsum": g_cumsum, "A": a}
    if case.output_final_state:
        raw["final_state"] = final_state
    result = {}
    for name in reference.output_names(case):
        if not isinstance(raw[name], torch.Tensor):
            raise RuntimeError(f"missing tensor output: {name}")
        tensor = raw[name].detach().cpu().contiguous()
        result[name] = reference.mask_a_contract(tensor, case) if name == "A" else tensor
    return result


def golden_outputs(case, inputs):
    _, reference = numerical_modules()
    outputs = reference.run_golden_reference(*inputs, case, inputs[0].dtype)
    result = dict(zip(reference.output_names(case), outputs))
    result["o"] = result["o"].transpose(1, 2).contiguous()
    return result


def valid_values(name, tensor, case):
    torch, reference = numerical_modules()
    if name != "A":
        return tensor.reshape(-1)
    # Exclude undefined tail columns, including every logical varlen boundary.
    return torch.cat([
        tensor[b, :, begin:end, :end - begin].reshape(-1)
        for b, _, begin, end in reference.chunk_ranges(case)
    ])


def compare_output(name, actual, expected, case, rtol, atol):
    torch, _ = numerical_modules()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "error": "output shape/dtype mismatch",
                "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    actual = valid_values(name, actual, case).double()
    expected = valid_values(name, expected, case).double()
    if not bool(torch.isfinite(expected).all()):
        raise RuntimeError(f"CPU golden {name} contains NaN/Inf; invalid comparison")
    if not bool(torch.isfinite(actual).all()):
        return {"passed": False, "error": "NPU output contains NaN/Inf"}
    error = (actual - expected).abs()
    limit = atol + rtol * expected.abs()
    bad = error > limit
    expected_norm = float(torch.linalg.vector_norm(expected))
    error_norm = float(torch.linalg.vector_norm(error))
    relative_l2 = error_norm / expected_norm if expected_norm else (0.0 if not error_norm else None)
    return {
        "passed": not bool(bad.any()), "rtol": rtol, "atol": atol,
        "elements": actual.numel(), "mismatched": int(bad.sum()),
        "max_abs": float(error.max()), "mean_abs": float(error.mean()),
        "rmse": float(error.square().mean().sqrt()),
        "relative_l2": relative_l2 if relative_l2 is None or math.isfinite(relative_l2) else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser, ACCURACY_JSON)
    parser.add_argument("--rtol", type=float, help="required, explicitly chosen mixed tolerance")
    parser.add_argument("--atol", type=float, help="required, explicitly chosen mixed tolerance")
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "runs" / "standalone_accuracy")
    parser.add_argument("--save", choices=("failed", "all", "none"), default="failed")
    args = parser.parse_args(argv)
    records = select_cases(args)
    if args.list_cases:
        for record in records:
            print(json.dumps(case_description(record), ensure_ascii=False))
        return 0
    if any(x is None or not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
        parser.error("supply finite, nonnegative --rtol and --atol; ATK ratios cannot be converted to these")
    if args.seed_count < 1 or args.cpu_threads < 1:
        parser.error("--seed-count and --cpu-threads must be positive")
    torch, ascendc, environment = npu_runtime(args.device)
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {**source_metadata(args.case_file), **environment},
        "criterion": "abs(dut - golden) <= atol + rtol * abs(golden), every valid element",
        "rtol": args.rtol, "atol": args.atol, "seed_count": args.seed_count,
        "reference": "existing CPU FP64 recurrence, cast to public output dtype",
        "planned_comparisons": len(records) * args.seed_count, "completed": False,
        "results": [],
    }
    failures = 0
    with torch.no_grad():
        for record in records:
            for offset in range(args.seed_count):
                seed = case_seed(record, args.seed_base) + offset
                row = {"case": case_description(record), "seed": seed}
                # Drop references to device allocations before preparing the next case.
                case = inputs = state = call = npu_inputs = outputs = actual = expected = None
                try:
                    case, inputs, state = build_cpu_inputs(record, seed)
                    call, npu_inputs = prepare_call(ascendc, case, inputs, state, args.device)
                    outputs = call()
                    torch.npu.synchronize()
                    actual = collect_outputs(outputs, case, npu_inputs)
                    outputs = call = npu_inputs = None
                    expected = golden_outputs(case, inputs)
                    row["outputs"] = {
                        name: compare_output(name, actual[name], expected[name], case, args.rtol, args.atol)
                        for name in expected
                    }
                    row["passed"] = all(item["passed"] for item in row["outputs"].values())
                except Exception as exc:
                    row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                if inputs is not None and (args.save == "all" or (args.save == "failed" and not row["passed"])):
                    dump = args.output_dir / f"case_{record['id']}_seed_{seed}.pt"
                    # On a device failure, save only already available CPU data;
                    # do not access a possibly poisoned device context again.
                    torch.save({"case": case_description(record), "seed": seed,
                                "inputs": dict(zip(("q", "k", "v", "g", "beta"), inputs)),
                                "initial_state": state, "dut": actual, "golden": expected}, dump)
                    row["dump"] = str(dump.resolve())
                failures += not row["passed"]
                report["results"].append(row)
                report["failed"] = failures
                write_json(args.output_dir / "accuracy.json", report)
                print(f"case={record['id']} seed={seed} {'PASS' if row['passed'] else 'FAIL'} "
                      + json.dumps(row.get("outputs", row.get("error")), ensure_ascii=False), flush=True)
                if "error" in row:
                    # Device errors can poison the context. Preserve the report and
                    # require a fresh process instead of misreporting later cases.
                    return 1
    report["completed"] = True
    report["passed"] = failures == 0
    write_json(args.output_dir / "accuracy.json", report)
    print(f"{len(report['results'])} comparisons, {failures} failed; {args.output_dir / 'accuracy.json'}")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())

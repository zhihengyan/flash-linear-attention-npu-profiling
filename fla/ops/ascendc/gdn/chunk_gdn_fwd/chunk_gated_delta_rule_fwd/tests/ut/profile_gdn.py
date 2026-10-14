"""Capture one diagnostic GDN forward call, or decode an existing raw dump.

This is a profiling utility, not an ATK precision/performance acceptance test.
Requires a rebuilt OPP and fla_npu package with the diagnostic entry for capture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

from parse_timer_csv import HEADER_PATH, LAYOUT, TOTAL_BUFFER_SIZE, parse_and_dump, print_summary, summarize_raw, validate_frequency
from trace_parser import generate_trace_json


def prepare_capture_environment():
    """Register the current wheel before torch_npu initializes the kernel registry."""
    spec = find_spec("fla_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("Current Python has no installed fla_npu; install the complete A5 wheel first")
    vendor = Path(spec.origin).resolve().parent / "opp/vendors/fla_npu_transformer"
    if not vendor.is_dir():
        raise RuntimeError(f"Installed wheel has no OPP vendor directory: {vendor}")
    paths = [str(vendor), str(vendor / "op_api/lib")]
    paths.extend(part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
                 if part and part not in paths)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(paths)


def export(raw_path, output_dir, cycles_per_us=None, shared_clock=False):
    validate_frequency(cycles_per_us)
    if shared_clock and cycles_per_us is None:
        raise ValueError("--shared-clock requires --cycles-per-us")
    path, events, metadata = parse_and_dump(raw_path, output_dir, cycles_per_us)
    print_summary(events)
    print(f"CSV: {path}")
    if cycles_per_us is not None and shared_clock:
        print("Trace:", generate_trace_json(events, Path(output_dir) / "gdn_npu_time.json",
                                           cycles_per_us, shared_clock=True))
    else:
        print("Perfetto skipped: verify clock frequency and a common AIC/AIV counter before conversion.")
    return metadata


def capture(q, k, v, g, beta, *, output_dir, warmup=5, check_output=False,
            cycles_per_us=None, shared_clock=False, prepare_pack_start=0,
            prepare_pack_count=8, prepare_detail=False, input_snapshot_sha256=None, **kwargs):
    """Use real model tensors here. Keep each call's timer separate and synchronize before export."""
    import torch
    from fla_npu.ops.ascendc import _aclnn_ctypes as adapter
    from fla_npu.ops.ascendc import _runtime

    validate_frequency(cycles_per_us)
    if type(prepare_pack_start) is not int or not 0 <= prepare_pack_start < 2**63:
        raise ValueError("prepare_pack_start must be a nonnegative int64")
    if type(prepare_pack_count) is not int or not 1 <= prepare_pack_count <= LAYOUT["MAX_PREPARE_PACK_COUNT"]:
        raise ValueError("prepare_pack_count must be 1..8")
    if type(prepare_detail) is not bool:
        raise ValueError("prepare_detail must be bool")
    if warmup < 0 or q.device.type != "npu":
        raise ValueError("warmup must be nonnegative and inputs must be on an NPU")
    if shared_clock and cycles_per_us is None:
        raise ValueError("shared_clock requires cycles_per_us")
    if not hasattr(adapter, "_profile_chunk_gated_delta_rule_fwd"):
        raise RuntimeError("loaded fla_npu lacks the profiling adapter; rebuild/install the complete package")
    device = q.device
    with torch.npu.device(device), torch.no_grad():
        # Warm-up uses the production ABI, never overwrites the measured buffer.
        for _ in range(warmup):
            adapter.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kwargs)
        baseline = adapter.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kwargs) if check_output else None
        torch.npu.synchronize(device)
        timer = torch.zeros(TOTAL_BUFFER_SIZE, dtype=torch.int64, device=device)
        if timer.data_ptr() % LAYOUT["CACHE_LINE_BYTES"]:
            raise ValueError("timer storage must be cache-line aligned for per-core GM publication")
        control_offset = LAYOUT["PREPARE"] * LAYOUT["BUFFER_WORDS_PER_KERNEL"] + 12
        timer[control_offset:control_offset + 3].copy_(torch.tensor(
            [prepare_pack_start, prepare_pack_count, int(prepare_detail)], dtype=torch.int64, device=device))
        torch.npu.synchronize(device)  # Complete host controls before the diagnostic call.
        outputs = adapter._profile_chunk_gated_delta_rule_fwd(q, k, v, g, beta, timer=timer, **kwargs)
        torch.npu.synchronize(device)
        # Copy raw evidence before any further launch or output-comparison failure.
        raw = timer.cpu().tolist()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "gdn_npu_time_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows((value,) for value in raw)
    metadata = {
        "device": str(device), "device_name": torch.npu.get_device_name(device),
        "torch": torch.__version__, "adapter_path": str(Path(adapter.__file__).resolve()),
        "opapi_libraries": [str(lib._name) for lib in _runtime.runtime()._libraries],
        "environment": {name: os.environ.get(name) for name in
                        ("ASCEND_HOME_PATH", "ASCEND_CUSTOM_OPP_PATH")},
        "inputs": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in
                   zip(("q", "k", "v", "g", "beta"), (q, k, v, g, beta))},
        "options": {name: value.detach().cpu().reshape(-1).tolist()
                    if name in ("cu_seqlens", "chunk_indices") and hasattr(value, "shape")
                    else {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if hasattr(value, "shape") else value for name, value in kwargs.items()},
        "warmup": warmup, "sample_count": 1,
        "timer_data_ptr": timer.data_ptr(), "timer_words": TOTAL_BUFFER_SIZE,
        "capture_process_id": os.getpid(),
        "prepare_sampling": {"pack_start": prepare_pack_start, "pack_count": prepare_pack_count,
                             "detail_enabled": prepare_detail},
        "input_snapshot_sha256": input_snapshot_sha256,
        "cycles_per_us": cycles_per_us, "shared_clock_asserted": shared_clock,
        "header_sha256": hashlib.sha256(HEADER_PATH.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_summary": summarize_raw(raw), "decode_status": "pending",
        "output_comparison": "pending" if baseline is not None else "not requested",
    }
    meta_path = output_dir / "capture_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if baseline is not None:
        try:
            if len(baseline) != 10 or len(outputs) != 10:
                raise AssertionError("GDN profiling must preserve all ten outputs")
            for index, (expected, actual) in enumerate(zip(baseline, outputs)):
                if expected is None or actual is None:
                    if expected is not actual:
                        raise AssertionError(f"output[{index}] optional contract differs")
                else:
                    # Same inputs/kernel: exact equality detects changed numerical behavior.
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            metadata["output_comparison"] = "ten outputs exactly match uninstrumented call"
        except Exception:
            metadata["output_comparison"] = "failed; raw retained, run precision investigation"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise
    # Preserve numerical evidence even if decoding fails because a slice is absent.
    try:
        metadata["kernels"] = export(raw_path, output_dir, cycles_per_us, shared_clock)
        metadata["decode_status"] = "complete"
    except Exception as error:
        metadata["decode_status"] = "failed"
        metadata["decode_error"] = str(error)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parse-only", type=Path, help="no torch/CANN needed")
    parser.add_argument("--inputs", type=Path, help="torch.save dict: q/k/v/g/beta and optional kwargs")
    parser.add_argument("--output-dir", type=Path, default=Path("gdn_timer"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check-output", action="store_true")
    parser.add_argument("--cycles-per-us", type=float, help="verified GetSystemCycle frequency")
    parser.add_argument("--shared-clock", action="store_true", help="assert verified common AIC/AIV counter")
    parser.add_argument("--prepare-pack-start", type=int, default=0)
    parser.add_argument("--prepare-pack-count", type=int, default=8)
    parser.add_argument("--prepare-detail", action="store_true")
    args = parser.parse_args()
    if args.parse_only is not None:
        export(args.parse_only, args.output_dir, args.cycles_per_us, args.shared_clock)
        return
    if args.inputs is None:
        parser.error("capture requires --inputs exported from the original ATK case")
    prepare_capture_environment()
    import torch
    import torch_npu  # Registers torch.npu.

    del torch_npu
    device = torch.device(f"npu:{args.device}")
    if args.inputs is not None:
        saved = torch.load(args.inputs, map_location="cpu", weights_only=True)
        inputs = [saved[name].to(device).contiguous() for name in ("q", "k", "v", "g", "beta")]
        kwargs = saved.get("kwargs", {})
        kwargs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in kwargs.items()}
    capture(*inputs, output_dir=args.output_dir, warmup=args.warmup, check_output=args.check_output,
            cycles_per_us=args.cycles_per_us, shared_clock=args.shared_clock,
            prepare_pack_start=args.prepare_pack_start, prepare_pack_count=args.prepare_pack_count,
            prepare_detail=args.prepare_detail,
            input_snapshot_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(), **kwargs)


if __name__ == "__main__":
    main()

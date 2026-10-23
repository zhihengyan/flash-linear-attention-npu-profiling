"""Shared CLI and collection loop for standalone msprof op applications.

This module has no ATK dependency. Operator-specific input construction stays
in each operator's scripts/profile_msprof.py.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


class NativeMstx:
    """Call the CANN MSTX API without torch_npu profiler enable-state gating.

    ABI: Ascend/mstx docs/en/api_reference/Common/mstxRangeStartA.md.
    The native range id is uint64_t, and zero means failure.
    """

    def __init__(self, library="libms_tools_ext.so"):
        try:
            self.library = ctypes.CDLL(library)
            self.start = self.library.mstxRangeStartA
            self.end = self.library.mstxRangeEnd
        except (OSError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot load native MSTX from {library!r}: {exc}. "
                "Source the CANN environment or pass --mstx-lib /absolute/path/libms_tools_ext.so."
            ) from exc
        self.start.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.start.restype = ctypes.c_uint64
        self.end.argtypes = [ctypes.c_uint64]
        self.end.restype = None

    def range_start(self, message):
        range_id = self.start(message.encode("utf-8"), None)
        if not range_id:
            raise RuntimeError(
                "Native mstxRangeStartA returned 0: no collection range was created. "
                f"Run msprof op with --mstx=on --mstx-include={message}; "
                "check the CANN/msprof versions and loaded MSTX library. "
                "For a kernel-filter diagnostic use --no-mstx in the application "
                "and omit --mstx/--mstx-include from msprof."
            )
        return range_id

    def range_end(self, range_id):
        self.end(range_id)


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _positive(value):
    number = _nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def chunk_indices(cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return None
    return [
        value
        for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:]))
        for chunk in range((end - begin + chunk_size - 1) // chunk_size)
        for value in (sequence, chunk)
    ]


def main(op_name, script_path, case_config, build_call):
    """Load one frozen case, prepare once, warm up, then mark NPU calls only."""
    parser = argparse.ArgumentParser(
        description=f"Standalone {op_name} application for msprof op; no ATK required."
    )
    parser.add_argument(
        "--case-json", type=Path,
        default=Path(script_path).resolve().parents[1] / f"atk_{op_name}_perf.json",
    )
    parser.add_argument("--case-id", type=int, default=0, help="top-level id in the performance JSON")
    parser.add_argument("--device", type=_nonnegative, default=0, help="logical NPU device id")
    parser.add_argument("--warmup", type=_nonnegative, default=10)
    parser.add_argument("--iterations", type=_positive, default=1, help="public API calls inside MSTX range")
    parser.add_argument("--seed", type=_nonnegative, help="override the case's deterministic input seed")
    parser.add_argument("--mstx-lib", default="libms_tools_ext.so", help="native MSTX library name or absolute path")
    parser.add_argument("--no-mstx", action="store_true", help="diagnostic mode: omit msprof --mstx flags and select an explicit kernel name")
    parser.add_argument("--list-cases", action="store_true", help="list JSON ids without importing torch")
    parser.add_argument("--dry-run", action="store_true", help="validate and print configuration without torch/NPU")
    options = parser.parse_args()
    try:
        raw = options.case_json.read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows:
            raise ValueError("case JSON must be a nonempty list")
        ids = [int(row["id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("case JSON contains duplicate ids")
        if options.list_cases:
            print(json.dumps([{"id": row["id"], "name": row["name"]} for row in rows], indent=2))
            return
        selected = [row for row in rows if int(row["id"]) == options.case_id]
        if not selected:
            raise ValueError(f"unknown case id {options.case_id}; available ids: {ids}")
        row = selected[0]
        config = case_config(row)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    seed = int(config["seed"] if options.seed is None else options.seed)
    metadata = {
        "operator": op_name,
        "case_json": str(options.case_json.resolve()),
        "case_json_sha256": hashlib.sha256(raw).hexdigest(),
        "case_id": int(row["id"]),
        "case_name": row["name"],
        "seed": seed,
        "device": options.device,
        "warmup": options.warmup,
        "iterations": options.iterations,
        "mstx_range": None if options.no_mstx else op_name,
        "mstx_backend": "disabled" if options.no_mstx else "native_ctypes",
        "mstx_library": None if options.no_mstx else options.mstx_lib,
        "config": config,
    }
    if options.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    # Lazy imports let case inspection run on a CPU-only machine.
    import torch
    import torch_npu
    import fla_npu
    from fla_npu.ops import ascendc

    torch.npu.set_device(options.device)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    mstx = None if options.no_mstx else NativeMstx(options.mstx_lib)
    operation = getattr(ascendc, op_name)
    device = torch.device(f"npu:{options.device}")
    metadata.update({
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "fla_npu_path": str(fla_npu.__file__),
        "ascendc_path": str(ascendc.__file__),
        "device_name": torch.npu.get_device_name(options.device),
    })
    print(json.dumps(metadata, indent=2), flush=True)
    with torch.no_grad():
        args, kwargs = build_call(config, torch, device, seed)
        torch.npu.synchronize()
        outputs = None
        for _ in range(options.warmup):
            outputs = operation(*args, **kwargs)
        torch.npu.synchronize()
        # Pair this range with msprof op --mstx=on --kernel-name="*".
        # Include all forward subkernels, including its Cast/Transpose calls.
        range_id = None if mstx is None else mstx.range_start(op_name)
        if range_id is not None:
            print(f"[MSTX] native range started: id={range_id}, message={op_name}", flush=True)
        try:
            for _ in range(options.iterations):
                outputs = operation(*args, **kwargs)
                torch.npu.synchronize()
        finally:
            if mstx is not None:
                mstx.range_end(range_id)
        del outputs
    print(
        f"Completed {options.iterations} API call(s); "
        "collection succeeds only if msprof exports nonempty kernel data.", flush=True,
    )

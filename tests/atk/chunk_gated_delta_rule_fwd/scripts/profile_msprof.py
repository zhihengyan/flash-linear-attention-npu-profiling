#!/usr/bin/env python3
"""Profile the frozen GDN forward model cases with an external msprof op."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from msprof_runner import chunk_indices, main


def case_config(row):
    inputs = {item["name"]: item for item in row["inputs"]}
    attrs = {name: item["range_values"] for name, item in inputs.items() if item["type"] == "attr"}
    tensors = {
        name: {key: inputs[name][key] for key in ("shape", "dtype", "range_values")}
        for name in ("q", "k", "v", "g", "beta")
    }
    b, hk, t, k = tensors["q"]["shape"]
    vb, hv, vt, v = tensors["v"]["shape"]
    if tensors["k"]["shape"] != tensors["q"]["shape"] or (vb, vt) != (b, t):
        raise ValueError("q/k/v shapes do not match")
    if min(b, hk, hv, t) <= 0 or k != 128 or v not in (128, 256) or hv % hk:
        raise ValueError("invalid GDN B/H/T/K/V dimensions")
    if any(tensors[name]["shape"] != [b, t, hv] for name in ("g", "beta")):
        raise ValueError("forward g/beta must be BTH, not BHT")
    dtype = attrs["qkv_dtype"]
    if dtype not in ("bf16", "fp16") or any(tensors[name]["dtype"] != dtype for name in ("q", "k", "v", "beta")):
        raise ValueError("q/k/v/beta dtype must match qkv_dtype (bf16/fp16)")
    if tensors["g"]["dtype"] != "fp32":
        raise ValueError("forward g must be fp32")
    if attrs["chunk_size"] not in (64, 128):
        raise ValueError("forward chunk_size must be 64 or 128")
    cu = None
    if attrs["is_varlen"]:
        cu = [int(value.strip()) for value in attrs["cu_seqlens_spec"].split(",")]
        if b != 1 or len(cu) < 2 or cu[0] != 0 or cu[-1] != t or any(x >= y for x, y in zip(cu, cu[1:])):
            raise ValueError("invalid forward cu_seqlens")
    scenario = attrs["scenario"]
    if scenario not in {"dense", "varlen", "state_initial_final", "state_initial_only", "state_zero_final"}:
        raise ValueError(f"unsupported scenario: {scenario}")
    if (scenario == "dense" and cu is not None) or (scenario == "varlen" and cu is None):
        raise ValueError("scenario and is_varlen disagree")
    return {
        "tensors": tensors, "attrs": attrs, "cu_seqlens": cu,
        "seed": row["default_seed"] if row.get("default_seed") is not None else 20260817 + int(row["id"]),
    }


def build_call(config, torch, device, seed):
    # Reuse the executor's state contract without importing ATK or running a golden.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gdn_reference import GdnCase, deterministic_initial_state

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tensors = []
    for offset, name in enumerate(("q", "k", "v", "g", "beta"), start=1):
        spec = config["tensors"][name]
        low, high = spec["range_values"]
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        value = torch.rand(tuple(spec["shape"]), generator=generator, dtype=torch.float32)
        tensors.append((value * (high - low) + low).to(dtypes[spec["dtype"]]).to(device))
    q, k, v, g, beta = tensors
    attrs = config["attrs"]
    cu = config["cu_seqlens"]
    case = GdnCase(
        batch=q.shape[0], k_heads=q.shape[1], v_heads=v.shape[1], tokens=q.shape[2],
        key_dim=q.shape[3], value_dim=v.shape[3], chunk_size=attrs["chunk_size"],
        scale=attrs["scale"], scenario=attrs["scenario"],
        cu_seqlens=None if cu is None else tuple(cu),
    )
    case.validate()
    initial = deterministic_initial_state(case)
    return (q, k, v, g, beta), {
        "initial_state": None if initial is None else initial.to(device).contiguous(),
        "output_final_state": case.output_final_state,
        "chunk_size": case.chunk_size,
        "cu_seqlens": cu,
        "chunk_indices": chunk_indices(cu, case.chunk_size),
        "scale": case.scale,
    }


if __name__ == "__main__":
    main("chunk_gated_delta_rule_fwd", __file__, case_config, build_call)

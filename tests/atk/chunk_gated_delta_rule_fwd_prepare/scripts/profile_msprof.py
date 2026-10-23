#!/usr/bin/env python3
"""Profile Prepare cases using the frozen case_spec and executor distributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from msprof_runner import chunk_indices, main


def case_config(row):
    entries = [item["range_values"] for item in row["inputs"] if item["name"] == "case_spec"]
    if len(entries) != 1:
        raise ValueError("Prepare case must contain exactly one case_spec")
    raw = entries[0]
    spec = json.loads(raw) if isinstance(raw, str) else dict(raw)
    b, hk, hv, t, k, v = (int(spec[name]) for name in ("B", "HK", "HV", "T", "K", "V"))
    if min(b, hk, hv, t) <= 0 or k != 128 or v not in (128, 256) or hv % hk or hv // hk not in (1, 2, 3, 4):
        raise ValueError("invalid Prepare B/H/T/K/V dimensions")
    if spec.get("dtype", "bf16") != "bf16" or spec.get("chunk_size", 64) != 64:
        raise ValueError("Prepare requires bf16 q/k/v and chunk_size=64")
    gate_dtype = spec.get("g_dtype", "fp32")
    if gate_dtype not in ("fp32", "bf16") or spec.get("beta_dtype", "fp32") != gate_dtype:
        raise ValueError("Prepare g/beta must have the same fp32/bf16 dtype")
    if spec.get("use_gate_in_kernel", False) and any(spec.get(name, "fp32") != gate_dtype for name in ("a_log_dtype", "dt_bias_dtype")):
        raise ValueError("Prepare a_log/dt_bias dtype must match g/beta")
    if spec.get("allow_neg_eigval", False) and not spec.get("use_beta_sigmoid_in_kernel", True):
        raise ValueError("allow_neg_eigval requires beta sigmoid")
    lengths = spec.get("seqlens")
    if isinstance(lengths, str):
        lengths = json.loads(lengths) if lengths.strip().startswith("[") else [int(x) for x in lengths.split(",") if x.strip()]
    cu = None
    if lengths:
        lengths = [int(n) for n in lengths]
        if b != 1 or any(n <= 0 for n in lengths) or sum(lengths) != t:
            raise ValueError("invalid Prepare seqlens")
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + int(length))
    return {"spec": spec, "cu_seqlens": cu, "seed": int(spec.get("seed", 20260817))}


def build_call(config, torch, device, seed):
    from _ascendc_common_executor import _randn

    spec = config["spec"]
    b, hk, hv, t, k, v = (int(spec[name]) for name in ("B", "HK", "HV", "T", "K", "V"))
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}

    def randn(shape, name, offset, scale=0.05):
        return _randn(shape, name, dtypes[name], device, seed + offset, scale)

    q = randn((b, hk, t, k), "bf16", 1)
    key = randn((b, hk, t, k), "bf16", 2)
    if not spec.get("use_qk_l2norm_in_kernel", True):
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        key = torch.nn.functional.normalize(key.float(), p=2, dim=-1).to(key.dtype)
    value = randn((b, hv, t, v), "bf16", 3)
    g = randn((b, hv, t), spec.get("g_dtype", "fp32"), 4, 0.2)
    beta = randn((b, hv, t), spec.get("beta_dtype", "fp32"), 5, 0.5)
    use_gate = spec.get("use_gate_in_kernel", False)
    cu = config["cu_seqlens"]
    return (q, key, value, g, beta), {
        "chunk_size": int(spec.get("chunk_size", 64)),
        "use_qk_l2norm_in_kernel": spec.get("use_qk_l2norm_in_kernel", True),
        "use_gate_in_kernel": use_gate,
        "use_beta_sigmoid_in_kernel": spec.get("use_beta_sigmoid_in_kernel", True),
        "allow_neg_eigval": spec.get("allow_neg_eigval", False),
        "use_exp2": spec.get("use_exp2", True),
        "output_a": spec.get("output_a", True),
        "a_log": randn((hv,), spec.get("a_log_dtype", "fp32"), 6, 0.1) if use_gate else None,
        "dt_bias": randn((hv,), spec.get("dt_bias_dtype", "fp32"), 7, 0.1) if use_gate else None,
        "cu_seqlens": cu,
        "chunk_indices": chunk_indices(cu, int(spec.get("chunk_size", 64))),
    }


if __name__ == "__main__":
    main("chunk_gated_delta_rule_fwd_prepare", __file__, case_config, build_call)

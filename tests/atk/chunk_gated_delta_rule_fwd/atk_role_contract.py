#!/usr/bin/env python3
"""ATK ``mixed_tolerance_bm`` 的融合算子与六 ACLNN 对照角色合同。"""

from __future__ import annotations


RUNTIME_ROLE_ORDER = ("dut", "benchmark")
DUT_NODE_NAME = "phase6"
DUT_NODE_NAMES = (DUT_NODE_NAME, "npu_dut")
BENCHMARK_NODE_NAME = "gold"
METRIC_TENSOR_PAIRS = {"Actual": ("dut", "benchmark")}


def role_for_atk_task(
    device: str,
    node_name: str,
    is_benchmark_task: bool,
) -> str:
    """两个命名 NPU node 参与比较；CPU 参考分支只供独立排障。"""

    device = str(device).strip().lower()
    node_name = str(node_name).strip()
    if device == "cpu":
        if not is_benchmark_task:
            raise RuntimeError(
                "CPU 仅允许承载 ATK 自动创建的双标杆 golden 任务："
                f"name={node_name!r} is_benchmark_task={is_benchmark_task}"
            )
        return "golden"
    if device != "npu" or is_benchmark_task:
        raise RuntimeError(
            "无法识别 ATK 双标杆任务："
            f"device={device!r} name={node_name!r} "
            f"is_benchmark_task={is_benchmark_task}"
        )
    if node_name in DUT_NODE_NAMES:
        return "dut"
    if node_name == BENCHMARK_NODE_NAME:
        return "benchmark"
    raise RuntimeError(
        "NPU node.name 必须为 "
        f"{DUT_NODE_NAMES!r} 中任一名称或 {BENCHMARK_NODE_NAME!r}，实际为 {node_name!r}"
    )

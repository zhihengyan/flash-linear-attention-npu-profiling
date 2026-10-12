"""Check ATK input snapshots and timing-loop isolation without ATK, torch or an NPU."""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR = ROOT / "tests/atk/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py"


def load_nodes(names, namespace):
    tree = ast.parse(EXECUTOR.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(EXECUTOR), "exec"), namespace)
    return namespace


@contextmanager
def fixture_directory():
    # Default mkdir permissions allow the Windows sandbox SID to use the fixture.
    path = ROOT / (".gdn_profile_export_test_" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().parent == ROOT.resolve()
        shutil.rmtree(path)


class FakeTensor:
    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype

    def detach(self):
        return FakeTensor(self.values.copy(), self.dtype)

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def to(self, _device):
        return self


class GdnProfileInputExportTest(unittest.TestCase):
    def setUp(self):
        self.saved = []
        def save(payload, stream):
            self.saved.append(payload)
            stream.write(b"mock torch.save payload")
        self.torch = types.SimpleNamespace(save=Mock(side_effect=save))
        self.initial = FakeTensor([0.0128, -0.0128], "fp32")
        self.state = Mock(return_value=self.initial)
        self.indices = Mock(return_value=[0, 0, 0, 1, 1, 0])
        self.namespace = load_nodes({"export_profile_inputs"}, {
            "os": os, "Path": Path, "torch": self.torch,
            "deterministic_initial_state": self.state,
            "canonical_chunk_indices": self.indices,
        })
        self.export = self.namespace["export_profile_inputs"]
        self.inputs = [FakeTensor([index, index + 1], dtype) for index, dtype in
                       enumerate(("bf16", "bf16", "bf16", "fp32", "bf16"))]
        self.case = types.SimpleNamespace(scale=0.08838834764831843, chunk_size=64,
            cu_seqlens=(0, 65, 129), scenario="state_initial_final", output_final_state=True)

    def test_disabled_and_other_roles_do_not_touch_disk_or_inputs(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "mkdir") as mkdir:
            self.assertIsNone(self.export(self.inputs, self.case, 0, "dut"))
            mkdir.assert_not_called()
        with patch.dict(os.environ, {"GDN_PROFILE_INPUT_DIR": "must-not-be-used"}), \
             patch.object(Path, "mkdir") as mkdir:
            for role in ("benchmark", "golden"):
                self.assertIsNone(self.export(self.inputs, self.case, 0, role))
            mkdir.assert_not_called()
        self.state.assert_not_called()
        self.torch.save.assert_not_called()

    def test_varlen_state_case_preserves_actual_values_dtype_and_options(self):
        with fixture_directory() as directory, patch.dict(os.environ, {
            "GDN_PROFILE_INPUT_DIR": str(directory)}):
            path = self.export(self.inputs, self.case, 7, "dut")
            self.assertEqual(path, directory / "case_7.pt")
            self.assertEqual(path.read_bytes(), b"mock torch.save payload")
        payload = self.saved[0]
        for name, original in zip(("q", "k", "v", "g", "beta"), self.inputs):
            self.assertEqual(payload[name].values, original.values)
            self.assertEqual(payload[name].dtype, original.dtype)
            self.assertIsNot(payload[name], original)
        self.assertEqual(payload["kwargs"], {
            "layout": "BNSD", "scale": self.case.scale, "chunk_size": 64,
            "initial_state": self.initial, "output_final_state": True,
            "cu_seqlens": [0, 65, 129], "chunk_indices": [0, 0, 0, 1, 1, 0]})
        self.assertEqual((payload["case_id"], payload["scenario"], payload["profile_input_version"]),
                         (7, "state_initial_final", 1))
        self.indices.assert_called_once_with(self.case.cu_seqlens, 64)

    def test_dense_without_state_and_first_snapshot_is_not_overwritten(self):
        self.case.cu_seqlens = None
        self.case.scenario = "dense"
        self.case.output_final_state = False
        self.state.return_value = None
        self.indices.return_value = None
        with fixture_directory() as directory, patch.dict(os.environ, {
            "GDN_PROFILE_INPUT_DIR": str(directory)}):
            path = self.export(self.inputs, self.case, 1, "dut")
            before = path.read_bytes()
            self.inputs[0].values[0] = 999
            self.assertIsNone(self.export(self.inputs, self.case, 1, "dut"))
            self.assertEqual(path.read_bytes(), before)
        self.torch.save.assert_called_once()
        payload = self.saved[0]
        self.assertEqual(payload["q"].values[0], 0)
        self.assertIsNone(payload["kwargs"]["initial_state"])
        self.assertIsNone(payload["kwargs"]["cu_seqlens"])
        self.assertIsNone(payload["kwargs"]["chunk_indices"])
        self.assertFalse(payload["kwargs"]["output_final_state"])

    def test_relative_directory_is_rejected_before_writing(self):
        with patch.dict(os.environ, {"GDN_PROFILE_INPUT_DIR": "relative-dir"}), \
             patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "绝对路径"):
                self.export(self.inputs, self.case, 0, "dut")
            mkdir.assert_not_called()
        self.torch.save.assert_not_called()

    def test_export_is_only_in_dut_initialization_and_never_in_timing_loop(self):
        for role in ("dut", "benchmark", "golden"):
            with self.subTest(role=role):
                export = Mock()
                run = Mock(return_value=())
                torch = types.SimpleNamespace(no_grad=nullcontext)
                ns = load_nodes({"FunctionApi"}, {
                    "BaseApi": object, "torch": torch, "export_profile_inputs": export,
                    "build_inputs": lambda _values: (self.case, "bf16", tuple(self.inputs)),
                    "output_names": lambda _case: (), "role_for_atk_task": lambda *_args: role,
                    "_npu_device": lambda device_id: f"npu:{device_id}", "run_npu": run,
                    "run_cpu": run, "SIX_ACLNN_OPS": (),
                })
                executor = types.SimpleNamespace(device="npu", device_id=0, _task_name=role,
                    _is_benchmark_task=role=="benchmark", _case_id=0, _npu_inputs=None)
                ns["FunctionApi"].init_by_input_data(executor, types.SimpleNamespace(kwargs={}))
                if role == "dut":
                    export.assert_called_once_with(executor._inputs, self.case, 0, "dut")
                else:
                    export.assert_not_called()
                export.reset_mock()
                for _ in range(3):
                    self.assertIsNone(ns["FunctionApi"].__call__(executor, None, False))
                export.assert_not_called()
                self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()

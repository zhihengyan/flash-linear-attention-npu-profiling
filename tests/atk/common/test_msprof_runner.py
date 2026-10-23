"""CPU-only regressions for native MSTX ABI and collection boundaries."""

from __future__ import annotations

import contextlib
import ctypes
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import msprof_runner


class MsprofRunnerTest(unittest.TestCase):
    def native_library(self, range_id):
        events = []
        active = [False]

        def start(message, stream):
            events.append(("start", message, stream))
            active[0] = range_id != 0
            return range_id

        def end(value):
            events.append(("end", value))
            active[0] = False

        library = types.SimpleNamespace(
            mstxRangeStartA=ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_char_p, ctypes.c_void_p)(start),
            mstxRangeEnd=ctypes.CFUNCTYPE(None, ctypes.c_uint64)(end),
        )
        return library, events, active

    def run_application(self, active, *extra):
        calls = []
        setup = []
        tensor = object()

        def operation(*args, **kwargs):
            calls.append((active[0], args))
            return args

        def build_call(config, torch, device, seed):
            setup.append(active[0])
            return (tensor,), {}

        torch = types.ModuleType("torch")
        torch.__version__ = "test"
        torch.device = lambda value: value
        torch.no_grad = contextlib.nullcontext
        torch.npu = types.SimpleNamespace(
            config=types.SimpleNamespace(),
            set_device=lambda value: None,
            set_compile_mode=lambda **kwargs: None,
            synchronize=lambda: None,
            get_device_name=lambda value: "test-device",
        )
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.__version__ = "test"
        # No torch_npu.npu.mstx: using it again must fail this regression.
        fla_npu = types.ModuleType("fla_npu")
        fla_npu.__file__ = "test/fla_npu/__init__.py"
        ops = types.ModuleType("fla_npu.ops")
        ops.ascendc = types.SimpleNamespace(
            chunk_gated_delta_rule_fwd=operation, __file__="test/ascendc/__init__.py",
        )
        modules = {"torch": torch, "torch_npu": torch_npu, "fla_npu": fla_npu, "fla_npu.ops": ops}
        script = Path(__file__).resolve().parents[1] / "chunk_gated_delta_rule_fwd/scripts/profile_msprof.py"
        output = io.StringIO()
        argv = [str(script), "--warmup", "2", "--iterations", "3", *extra]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            msprof_runner.main(
                "chunk_gated_delta_rule_fwd", script, lambda row: {"seed": 123}, build_call,
            )
        return calls, setup, output.getvalue()

    def test_native_abi_preserves_uint64_range_id(self):
        large_id = (1 << 40) + 17
        library, events, _ = self.native_library(large_id)
        with patch.object(msprof_runner.ctypes, "CDLL", return_value=library):
            mstx = msprof_runner.NativeMstx()
            actual = mstx.range_start("gdn_test")
            mstx.range_end(actual)
        self.assertEqual(actual, large_id)
        self.assertEqual(events, [("start", b"gdn_test", None), ("end", large_id)])

    def test_setup_and_warmup_are_outside_native_range(self):
        library, events, active = self.native_library(9)
        with patch.object(msprof_runner.ctypes, "CDLL", return_value=library):
            calls, setup, output = self.run_application(active)
        self.assertEqual(setup, [False])
        self.assertEqual([item[0] for item in calls], [False, False, True, True, True])
        self.assertEqual(len({id(item[1][0]) for item in calls}), 1)
        self.assertEqual(events, [("start", b"chunk_gated_delta_rule_fwd", None), ("end", 9)])
        self.assertIn("[MSTX] native range started: id=9", output)

    def test_zero_range_id_aborts(self):
        library, events, active = self.native_library(0)
        with patch.object(msprof_runner.ctypes, "CDLL", return_value=library):
            with self.assertRaisesRegex(RuntimeError, "returned 0"):
                self.run_application(active)
        self.assertEqual(events, [("start", b"chunk_gated_delta_rule_fwd", None)])

    def test_no_mstx_does_not_load_library(self):
        with patch.object(msprof_runner.ctypes, "CDLL", side_effect=AssertionError("must not load MSTX")):
            calls, setup, output = self.run_application([False], "--no-mstx")
        self.assertEqual(setup, [False])
        self.assertEqual([item[0] for item in calls], [False] * 5)
        self.assertNotIn("[MSTX] native range started", output)

    def test_missing_library_has_actionable_error(self):
        with patch.object(msprof_runner.ctypes, "CDLL", side_effect=OSError("not found")):
            with self.assertRaisesRegex(RuntimeError, "--mstx-lib"):
                msprof_runner.NativeMstx()


if __name__ == "__main__":
    unittest.main(verbosity=2)

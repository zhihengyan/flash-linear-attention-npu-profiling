"""Offline contract tests. All raw fixtures are synthetic, never performance evidence."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import re
import sys
import types
import unittest
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import parse_timer_csv as parser
from trace_parser import generate_trace_json

OP = Path(__file__).resolve().parents[2]
ROOT = next(parent for parent in OP.parents if (parent / "torch_custom").is_dir())
STRIDE = parser.LAYOUT["N_TIMING_COUNTER_PER_CORE_ALIGN"]
HEADER = parser.LAYOUT["HEADER_WORDS"]
SLICE = parser.LAYOUT["BUFFER_WORDS_PER_KERNEL"]


@contextmanager
def test_directory():
    # tempfile's private 0700 mkdir denies the Windows sandbox identity.
    path = ROOT / (".gdn_timer_test_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("temporary test directory escaped the repository")
        shutil.rmtree(path)


def fixture(kernels=("PHASE6",), *, wait=True, origin=1000):
    raw = [0] * parser.TOTAL_BUFFER_SIZE
    for kernel in kernels:
        kernel_id = next(i for i, name in parser.KERNEL_NAMES.items() if name == kernel)
        offset = kernel_id * SLICE
        raw[offset:offset + 9] = [parser.LAYOUT["HEADER_MAGIC"], parser.LAYOUT["LAYOUT_VERSION"], kernel_id,
                                 1, parser.LAYOUT["CORE_GROUP_CAPACITY"], STRIDE,
                                 len(parser.FIXED_TIMING_NAMES), 1, 220]
        for core in range(3):
            base = offset + HEADER + core * STRIDE
            pairs = [(origin, origin + 100), (origin + 1, origin + 2), (origin + 3, origin + 100)]
            if kernel == "PHASE6":
                pairs.extend([(origin + 10, origin + 11), (origin + 20, origin + 21),
                              (origin + 22, origin + 30) if wait else (0, 0)])
            for slot, pair in enumerate(pairs):
                raw[base + slot * 2:base + slot * 2 + 2] = pair
    return raw


class TimerParserTest(unittest.TestCase):
    def test_shared_header_names_and_capacity(self):
        self.assertEqual(list(parser.FIXED_TIMING_NAMES.values()), ["KERNEL_TIMING", "MAIN_INIT", "MAIN_BODY",
                         "PHASE6_FRONT_DONE", "PHASE6_H_DISPATCH_DONE", "PHASE6_HO_WAIT"])
        self.assertEqual(parser.DYNAMIC_TIMING_NAMES, {})
        self.assertEqual(parser.TOTAL_BUFFER_SIZE, SLICE * len(parser.KERNEL_NAMES))
        self.assertGreaterEqual(STRIDE, len(parser.FIXED_TIMING_NAMES) * 2)
        self.assertEqual(STRIDE % 16, 0)

    def test_both_routes_and_subcore_mapping(self):
        for kernels in (("PHASE6",), ("PREPARE", "FWD_H", "FWD_O")):
            with self.subTest(kernels=kernels):
                events, meta = parser.parse_timer(fixture(kernels))
                self.assertEqual({item["kernel"] for item in meta}, set(kernels))
                totals = [event for event in events if event["stage"] == "KERNEL_TIMING"]
                self.assertEqual(len(totals), len(kernels) * 3)
                self.assertEqual([(e["core_type"], e["sub_id"]) for e in totals[:3]],
                                 [("AIC", -1), ("AIV", 0), ("AIV", 1)])

    def test_overlap_branch_does_not_fabricate_wait(self):
        events, _ = parser.parse_timer(fixture(wait=False))
        self.assertNotIn("PHASE6_HO_WAIT", {event["stage"] for event in events})

    def test_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "raw length"):
            parser.parse_timer(fixture()[:-1])

    def test_empty_disabled_capture(self):
        with self.assertRaisesRegex(ValueError, "no captured kernels"):
            parser.parse_timer([0] * parser.TOTAL_BUFFER_SIZE)

    def test_malformed_pairs_and_outside_parent(self):
        for pair in ((0, 1002), (1003, 1002), (999, 1002), (1001, 1101)):
            with self.subTest(pair=pair):
                raw = fixture()
                raw[HEADER + 2:HEADER + 4] = pair
                with self.assertRaises(ValueError):
                    parser.parse_timer(raw)

    def test_layout_mismatch_and_overflow(self):
        for index, value in ((0, 0), (1, 99), (3, 257), (4, 96), (5, 12), (6, 7), (7, 0)):
            with self.subTest(index=index):
                raw = fixture()
                raw[index] = value
                with self.assertRaises(ValueError):
                    parser.parse_timer(raw)

    def test_unused_core_write(self):
        raw = fixture()
        raw[HEADER + 3 * STRIDE] = 1
        with self.assertRaisesRegex(ValueError, "outside actual blockDim"):
            parser.parse_timer(raw)

    def test_missing_core_and_boundary(self):
        for offset in (HEADER, HEADER + 6):
            raw = fixture()
            raw[offset:offset + 2] = [0, 0]
            with self.assertRaises(ValueError):
                parser.parse_timer(raw)

    def test_bad_dispatch_order_and_mixed_wait_branch(self):
        raw = fixture()
        raw[HEADER + 6:HEADER + 8] = [1030, 1031]
        with self.assertRaisesRegex(ValueError, "dispatch boundary order"):
            parser.parse_timer(raw)
        raw = fixture()
        raw[HEADER + STRIDE + 10:HEADER + STRIDE + 12] = [0, 0]
        with self.assertRaisesRegex(ValueError, "inconsistent serial"):
            parser.parse_timer(raw)

    def test_incomplete_and_mixed_route(self):
        for kernels in (("PREPARE",), ("PHASE6", "FWD_H")):
            with self.subTest(kernels=kernels), self.assertRaises(ValueError):
                parser.parse_timer(fixture(kernels))

    def test_export_and_trace_integer_precision(self):
        origin = 2**55
        with test_directory() as temporary:
            directory = Path(temporary)
            raw_path = directory / "synthetic_raw.csv"
            raw_path.write_text("\n".join(map(str, fixture(origin=origin))), encoding="utf-8")
            csv_path, events, _ = parser.parse_and_dump(raw_path, directory, 10)
            self.assertIn("duration_us", csv_path.read_text(encoding="utf-8"))
            trace_path = generate_trace_json(events, directory / "synthetic_trace.json", 10, shared_clock=True)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
            point = next(event for event in trace if event.get("name") == "PHASE6_FRONT_DONE")
            self.assertEqual(point["ph"], "i")
            self.assertEqual(point["ts"], 1.05)
            total = next(event for event in trace if event.get("name") == "KERNEL_TIMING")
            self.assertEqual(total["dur"], 10)
            self.assertEqual(total["args"]["start_cycle"], origin)

    def test_trace_requires_verified_units_and_clock(self):
        events, _ = parser.parse_timer(fixture())
        for frequency, shared in ((None, True), (10, False), (0, True), (-1, True), (float("nan"), True)):
            with self.subTest(frequency=frequency, shared=shared), self.assertRaises(ValueError):
                generate_trace_json(events, "unused.json", frequency, shared_clock=shared)


def load_adapter():
    path = ROOT / "torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py"
    spec = importlib.util.spec_from_file_location("gdn_timer_existing_abi_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimerAbiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing = load_adapter()
        cls.adapter = cls.existing.ACLNN_CTYPES

    def test_profile_appends_one_descriptor_preserving_abi(self):
        old = self.adapter._GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleFwd"]
        new = self.adapter._GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleFwdProfile"]
        self.assertEqual(len(old), 29)
        self.assertEqual(new, old[:-2] + [ctypes.c_void_p] + old[-2:])
        fake_torch = types.ModuleType("torch")
        fake_torch.float32, fake_torch.bfloat16, fake_torch.int64 = object(), object(), object()
        bump = mock.Mock()
        fake_torch.autograd = types.SimpleNamespace(graph=types.SimpleNamespace(increment_version=bump))
        Tensor = self.existing.FakeTensor
        q = Tensor((1, 2, 65, 128), fake_torch.bfloat16)
        v = Tensor((1, 4, 65, 128), fake_torch.bfloat16)
        g = Tensor((1, 65, 4), fake_torch.float32)
        timer = Tensor((parser.TOTAL_BUFFER_SIZE,), fake_torch.int64)
        timer.device = q.device
        captured = []

        def launch(name, build_args, outputs):
            context = self.existing.FakeCallContext()
            captured.append((name, build_args(context), context.descriptor_metadata))
            return outputs

        def empty(shape, like, **kwargs):
            return Tensor(shape, kwargs.get("dtype", like.dtype))

        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(self.adapter, "_empty", side_effect=empty), \
                mock.patch.object(self.adapter, "_call_aclnn", side_effect=launch):
            old_outputs = self.adapter.npu_chunk_gated_delta_rule_fwd(q, q, v, g, g)
            new_outputs = self.adapter._profile_chunk_gated_delta_rule_fwd(q, q, v, g, g, timer=timer)
        self.assertEqual(captured[0][0], "aclnnChunkGatedDeltaRuleFwd")
        self.assertEqual(captured[1][0], "aclnnChunkGatedDeltaRuleFwdProfile")
        self.assertEqual(len(captured[0][1]), 27)
        self.assertEqual(len(captured[1][1]), 28)
        self.assertEqual(captured[1][2][-1][0], "timer")
        self.assertIs(captured[1][2][-1][1], timer)
        self.assertEqual(len(old_outputs), 10)
        self.assertEqual(len(new_outputs), 10)
        bump.assert_called_once_with((timer,))

    def test_profile_rejects_bad_timer_before_launch(self):
        torch = types.ModuleType("torch")
        torch.int64 = object()
        Tensor = self.existing.FakeTensor
        q = Tensor((1, 2, 65, 128))
        cases = [None, Tensor((10,), object()), Tensor((10, 1), torch.int64),
                 Tensor((10,), torch.int64, contiguous=False), Tensor((10,), torch.int64, device_type="cpu")]
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(self.adapter, "_call_aclnn") as launch:
            for timer in cases:
                with self.subTest(timer=timer), self.assertRaises(ValueError):
                    self.adapter._profile_chunk_gated_delta_rule_fwd(q, q, q, q, q, timer=timer)
            launch.assert_not_called()

    def test_legacy_forward_contract_regressions(self):
        case = self.existing.AclnnCtypesAbiTest()
        case.test_gdn_training_and_inference_output_contract()
        case.test_gdn_norm_outputs_and_input_aliases()

    def test_failed_launch_and_grad_timer_do_not_bump_version(self):
        torch = types.ModuleType("torch")
        torch.int64 = object()
        bump = mock.Mock()
        torch.autograd = types.SimpleNamespace(graph=types.SimpleNamespace(increment_version=bump))
        Tensor = self.existing.FakeTensor
        q = Tensor((1, 2, 65, 128))
        timer = Tensor((parser.TOTAL_BUFFER_SIZE,), torch.int64)
        timer.device = q.device
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(self.adapter, "_chunk_gated_delta_rule_fwd_impl", side_effect=RuntimeError("launch failed")) as launch:
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                self.adapter._profile_chunk_gated_delta_rule_fwd(q, q, q, q, q, timer=timer)
            bump.assert_not_called()
            timer.requires_grad = True
            with self.assertRaisesRegex(RuntimeError, "must not require gradients"):
                self.adapter._profile_chunk_gated_delta_rule_fwd(q, q, q, q, q, timer=timer)
            self.assertEqual(launch.call_count, 1)
            bump.assert_not_called()

    def test_mutation_metadata_names_private_timer(self):
        import ast
        tree = ast.parse((ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py").read_text(encoding="utf-8"))
        table = next(node.value for node in tree.body if isinstance(node, ast.Assign) and
                     any(isinstance(target, ast.Name) and target.id == "MUTATED_ARGUMENTS" for target in node.targets))
        self.assertEqual(ast.literal_eval(table)["_profile_chunk_gated_delta_rule_fwd"], ("timer",))

    def test_cpp_declarations_and_kernel_input_positions(self):
        text = (OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h").read_text(encoding="utf-8")
        def params(name):
            match = re.search(name + r"\((.*?)\);", text, re.S)
            return [re.sub(r"\s+", " ", item.strip()) for item in match[1].split(",")]
        old = params("aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize")
        new = params("aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize")
        self.assertEqual(len(old), 29)
        self.assertEqual(new, old[:-2] + ["const aclTensor *timer"] + old[-2:])
        for name in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_fwd_prepare", "chunk_fwd_h", "chunk_fwd_o"):
            with self.subTest(op=name):
                base = OP.parent / name
                definition = (base / "op_host" / (name + "_def.cpp")).read_text(encoding="utf-8")
                inputs = re.findall(r'this->Input\("(.*?)"\)', definition)
                self.assertEqual(inputs[-1], "timer")
                timer_definition = definition[definition.index('this->Input("timer")'):definition.index('this->Output(')]
                self.assertNotIn("AutoContiguous", timer_definition)
                launcher = (base / "op_host/op_api" / (name + ".cpp")).read_text(encoding="utf-8")
                args = re.search(r"OP_INPUT\((.*?)\)", launcher, re.S)[1].split(",")
                self.assertEqual(len(args), len(inputs))
                self.assertEqual(args[-1].strip(), "timerOptional")
                if name == "chunk_gated_delta_rule_fwd":
                    sources = [base / "op_kernel/internal" / arch / (name + "_" + arch + ".cpp")
                               for arch in ("arch22", "arch35")]
                else:
                    sources = [base / "op_kernel" / (name + ".cpp")]
                for source in sources:
                    raw = source.read_text(encoding="utf-8")
                    signature = re.search(r"void " + name + r"\((.*?)\)\s*\{", raw, re.S)[1]
                    raw_args = re.findall(r"GM_ADDR (\w+)", signature)
                    self.assertEqual(raw_args[len(inputs) - 1], "timerAddr")
                    self.assertEqual(raw_args.count("timerAddr"), 1)

    def test_packaged_common_include_is_independent_of_source_tree(self):
        shared = ROOT / "fla/ops/ascendc/common"
        for source in (OP.parent / "chunk_fwd_h/op_kernel/chunk_fwd_h.cpp",
                       OP.parent / "chunk_fwd_o/op_kernel/chunk_fwd_o.cpp",
                       OP.parent / "chunk_gated_delta_rule_fwd_prepare/op_kernel/chunk_gated_delta_rule_fwd_prepare.cpp"):
            include = re.search(r'#include "(kernel_utils/gdn_timer/.*?)"', source.read_text(encoding="utf-8"))[1]
            self.assertTrue((shared / include).is_file())
        for name in ("AscendTimerV2.hpp", "AscendTimerV2_device.hpp"):
            forwarding = (OP / "op_kernel" / name).read_text(encoding="utf-8")
            self.assertIn('#include "kernel_utils/gdn_timer/' + name + '"', forwarding)


if __name__ == "__main__":
    unittest.main()

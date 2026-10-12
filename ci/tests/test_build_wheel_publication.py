"""Synthetic wheel publication checks; no CANN build or pip install is executed."""
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_wheel_publication", ROOT / "scripts/build_wheel.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
DIST = "flash_linear_attention_npu_a5"
OLD = DIST + "-26.10.0+main.devf9ba043-py3-none-manylinux_2_34_x86_64.whl"
NEW = DIST + "-26.10.0+main.dev9b0ff19-py3-none-manylinux_2_34_x86_64.whl"
OPP = {
    "fla_npu/opp/vendors/config.ini": b"load_priority=fla_npu_transformer\n",
    "fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so": b"synthetic host",
    "fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel/ascend950/prepare/timer.o": b"new timer cache publication",
}


@contextmanager
def fixture_directory():
    root = ROOT / (".wheel_publication_test_" + uuid4().hex)
    root.mkdir()
    try:
        with patch.object(builder, "REPO_ROOT", root), patch.object(builder, "get_wheel_dist_name", return_value=DIST):
            yield root
    finally:
        assert root.resolve().parent == ROOT.resolve()
        shutil.rmtree(root)


def stage(root, dirname="lib"):
    site = root / "build" / dirname
    for name, content in OPP.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return site / "fla_npu"


def write_wheel(path, entries=None, scheme=None, stable=False):
    entries = OPP if entries is None else entries
    info = DIST + "-26.10.0.dist-info"
    prefix = DIST + "-26.10.0.data/" + scheme + "/" if scheme else ""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(prefix + name, content)
        if stable:
            archive.writestr(prefix + "fla_npu/libfla_npu_stable.so", b"synthetic stable launcher")
        archive.writestr(info + "/METADATA", "Metadata-Version: 2.1\nName: flash-linear-attention-npu-a5\nVersion: 26.10.0\n")
        archive.writestr(info + "/RECORD", info + "/METADATA,,\n" + info + "/RECORD,,\n")


class WheelPublicationTest(unittest.TestCase):
    def test_main_selects_this_invocation_even_when_old_hash_sorts_last(self):
        with fixture_directory() as root:
            output = root / "dist"
            output.mkdir()
            (output / OLD).write_bytes(b"preserve older wheel")
            stage(root)
            commands = []

            def build(command, env, progress, **kwargs):
                commands.append(command)
                current = Path(command[command.index("-w") + 1])
                self.assertNotEqual(current, output)
                self.assertEqual(list(current.iterdir()), [])
                write_wheel(current / NEW)
                return 0

            text = io.StringIO()
            with patch.object(builder, "_prepare_abi_free_launcher"), \
                 patch.object(builder, "_run_with_progress", side_effect=build), \
                 patch.object(sys, "argv", ["build_wheel.py", "--wheel-dir", str(output), "--no-progress"]), \
                 redirect_stdout(text):
                self.assertEqual(builder.main(), 0)
            self.assertLess(NEW, OLD)  # Reproduces the previous lexical-selection error.
            self.assertEqual((output / OLD).read_bytes(), b"preserve older wheel")
            self.assertTrue((output / NEW).is_file())
            self.assertIn(builder._install_command(output / NEW), text.getvalue())
            self.assertIn("Verified all 3 staged OPP files", text.getvalue())
            self.assertEqual(len(commands), 1)

    def test_repeated_filename_is_replaced_by_current_verified_payload(self):
        with fixture_directory() as root:
            stage(root)
            output, current = root / "dist", root / "dist/current"
            current.mkdir(parents=True)
            (output / NEW).write_bytes(b"old same-name wheel")
            write_wheel(current / NEW)
            with redirect_stdout(io.StringIO()):
                result = builder._publish_built_wheel(current, output)
            self.assertEqual(result, output / NEW)
            self.assertEqual(builder._manifest_from_wheel(result), builder._manifest_from_directory(root / "build/lib/fla_npu"))
            self.assertFalse(current.exists())

    def test_output_under_cleaned_intermediate_roots_survives_build_cleanup(self):
        for dirname in ("build", "build_out", "output"):
            with self.subTest(dirname=dirname), fixture_directory() as root:
                output = root / dirname / "wheel-results"

                def build(command, env, progress, **kwargs):
                    current = Path(command[command.index("-w") + 1])
                    self.assertTrue(current.is_relative_to(root / "dist"))
                    for name in ("build", "build_out", "output"):
                        target = root / name
                        assert target.resolve().parent == root.resolve()
                        if target.exists():
                            shutil.rmtree(target)
                        target.mkdir()
                    stage(root)
                    write_wheel(current / NEW)
                    return 0

                with patch.object(builder, "_prepare_abi_free_launcher"), \
                     patch.object(builder, "_run_with_progress", side_effect=build), \
                     patch.object(sys, "argv", ["build_wheel.py", "--wheel-dir", str(output), "--no-progress"]), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(builder.main(), 0)
                self.assertTrue((output / NEW).is_file())

    def test_changed_missing_or_extra_opp_refuses_publication_and_retains_archive(self):
        variants = [dict(OPP), dict(list(OPP.items())[1:]), {**OPP, "fla_npu/opp/extra.o": b"unexpected"}]
        variants[0][next(name for name in OPP if name.endswith(".o"))] = b"old timer"
        for entries in variants:
            with self.subTest(entries=list(entries)), fixture_directory() as root:
                stage(root)
                output, current = root / "dist", root / "dist/current"
                current.mkdir(parents=True)
                (output / NEW).write_bytes(b"preserve existing result")
                write_wheel(current / NEW, entries)
                with self.assertRaisesRegex(RuntimeError, "refusing installation command"):
                    builder._publish_built_wheel(current, output)
                self.assertTrue((current / NEW).is_file())
                self.assertEqual((output / NEW).read_bytes(), b"preserve existing result")

    def test_root_and_data_schemes_preserve_opp_while_injecting_runtime_pins(self):
        for scheme in (None, "purelib", "platlib"):
            with self.subTest(scheme=scheme), fixture_directory() as root:
                package = stage(root, "lib.linux-x86_64-cpython-311")
                current = root / "dist/current"
                current.mkdir(parents=True)
                write_wheel(current / NEW, scheme=scheme, stable=True)
                with redirect_stdout(io.StringIO()):
                    result = builder._publish_built_wheel(current, current.parent)
                self.assertEqual(builder._manifest_from_wheel(result), builder._manifest_from_directory(package))
                with zipfile.ZipFile(result) as archive:
                    metadata = archive.read(next(name for name in archive.namelist() if name.endswith("/METADATA"))).decode()
                self.assertIn("Requires-Dist: torch>=", metadata)
                self.assertIn("Requires-Dist: torch_npu>=", metadata)

    def test_no_multiple_or_wrong_distribution_outputs_are_rejected(self):
        for names in ([], [OLD, NEW], ["other-1.0-py3-none-any.whl"]):
            with self.subTest(names=names), fixture_directory() as root:
                current = root / "dist/current"
                current.mkdir(parents=True)
                for name in names:
                    write_wheel(current / name)
                with self.assertRaisesRegex(RuntimeError, "exactly one current-build wheel"):
                    builder._publish_built_wheel(current, current.parent)

    def test_missing_or_ambiguous_staging_is_rejected(self):
        for count in (0, 2):
            with self.subTest(count=count), fixture_directory() as root:
                for i in range(count):
                    stage(root, "lib" + str(i))
                current = root / "dist/current"
                current.mkdir(parents=True)
                write_wheel(current / NEW)
                with self.assertRaisesRegex(RuntimeError, "identify this build's staged OPP"):
                    builder._publish_built_wheel(current, current.parent)

    def test_failed_build_does_not_fall_back_to_an_old_wheel(self):
        with fixture_directory() as root:
            output = root / "dist"
            output.mkdir()
            (output / OLD).write_bytes(b"keep")
            text = io.StringIO()
            with patch.object(builder, "_prepare_abi_free_launcher"), \
                 patch.object(builder, "_run_with_progress", return_value=7), \
                 patch.object(sys, "argv", ["build_wheel.py", "--wheel-dir", str(output), "--no-progress"]), \
                 redirect_stdout(text), self.assertRaises(subprocess.CalledProcessError):
                builder.main()
            self.assertEqual((output / OLD).read_bytes(), b"keep")
            self.assertNotIn("Install command:", text.getvalue())


if __name__ == "__main__":
    unittest.main()

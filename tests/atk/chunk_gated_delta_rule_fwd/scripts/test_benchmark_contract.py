#!/usr/bin/env python3
"""标准库回归；设置 GDN_ATK_TEST_BASH 可额外运行模拟 ATK 的两个 shell 入口。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from xml.etree.ElementTree import Element, SubElement, tostring
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
REPO_DIR = OP_DIR.parents[2]
sys.path[:0] = [str(OP_DIR), str(SCRIPT_DIR)]

from aggregate_matrix_summaries import aggregate
from atk_role_contract import RUNTIME_ROLE_ORDER, role_for_atk_task
from summarize_atk_report import summarize_report
from validate_runtime_roles import audit_roles


def make_report(path, case_ids, *, error_column=False, reverse_columns=False,
                failed_case=None, accuracy_columns=True):
    headers = ["名称", "总用例数", "执行成功用例个数", "执行失败用例个数", "通过用例个数"]
    values = ["gold", len(case_ids), len(case_ids), 0, len(case_ids)]
    if error_column:
        headers.append("错误信息匹配用例个数")
        values.append(0)
    headers.extend(["通过率", "精度是否达标"])
    values.extend([100, "Pass"])
    if reverse_columns:
        headers.reverse()
        values.reverse()
    statistic = [["编号", "运行结果"] + (["gold.o_精度通过", "gold.A_精度通过"] if accuracy_columns else [])]
    statistic.extend(
        [case_id, "SUCCESS"] + (["False" if case_id == failed_case else "True", True] if accuracy_columns else [])
        for case_id in case_ids
    )
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    workbook = Element(f"{{{main}}}workbook")
    sheets = SubElement(workbook, f"{{{main}}}sheets")
    relations = Element(f"{{{pkg}}}Relationships")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, rows) in enumerate({"summary": [headers, values], "statistic": statistic}.items(), 1):
            SubElement(sheets, f"{{{main}}}sheet", {"name": name, f"{{{rel}}}id": f"rId{index}"})
            SubElement(relations, f"{{{pkg}}}Relationship", {"Id": f"rId{index}", "Target": f"worksheets/sheet{index}.xml"})
            sheet = Element(f"{{{main}}}worksheet")
            data = SubElement(sheet, f"{{{main}}}sheetData")
            for row_index, row_values in enumerate(rows, 1):
                row = SubElement(data, f"{{{main}}}row")
                for column, value in enumerate(row_values):
                    cell = SubElement(row, f"{{{main}}}c", {"r": f"{chr(65 + column)}{row_index}"})
                    if isinstance(value, str):
                        cell.set("t", "inlineStr")
                        SubElement(SubElement(cell, f"{{{main}}}is"), f"{{{main}}}t").text = value
                    else:
                        if isinstance(value, bool):
                            cell.set("t", "b")
                        SubElement(cell, f"{{{main}}}v").text = str(int(value))
            archive.writestr(f"xl/worksheets/sheet{index}.xml", tostring(sheet))
        archive.writestr("xl/workbook.xml", tostring(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", tostring(relations))


def fake_atk():
    arguments = sys.argv[2:]
    if arguments == ["--version"]:
        print("ATK 26.8.8")
        return
    def option(name, default=None):
        return arguments[arguments.index(name) + 1] if name in arguments else default
    assert arguments[0] == "task" and option("--task") == "accuracy"
    assert "--syc_dataset" in arguments and "-bd" not in arguments
    nodes = Path(option("-n")).read_text(encoding="utf-8")
    assert nodes.count("backend: npu") == 2 and "backend: cpu" not in nodes
    cases = json.loads(Path(option("-c")).read_text(encoding="utf-8"))
    assert all(case["standard"]["acc"] == "mixed_tolerance_bm" for case in cases)
    assert all(item["range_values"] == [-1, 0]
               for case in cases for item in case["inputs"] if item["name"] == "g")
    selected = cases[int(option("-s", 0)):int(option("-e", len(cases)))]
    ids = [case["id"] for case in selected]
    mode = os.environ.get("GDN_FAKE_ATK_MODE", "pass")
    for case_id in ids:
        for role in RUNTIME_ROLE_ORDER:
            if mode == "missing_role" and role == "benchmark":
                continue
            print(f"[gdn-double-atk] case={case_id} role={role}")
    make_report(Path("atk_output/mock/report/mock.xlsx"), ids,
                failed_case=ids[0] if mode == "accuracy_fail" else None,
                accuracy_columns=mode != "no_comparison")


class TemporaryWorkspace(unittest.TestCase):
    def setUp(self):
        output = OP_DIR / "atk_output"
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="benchmark_contract_", dir=output)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)


class BenchmarkContractTest(TemporaryWorkspace):
    def test_case_configuration_and_counts(self):
        for name, count in (("", 500), ("_mss", 6), ("_perf", 2)):
            cases = json.loads((OP_DIR / f"atk_chunk_gated_delta_rule_fwd{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cases), count)
            self.assertTrue(all(case["standard"]["acc"] == "mixed_tolerance_bm" for case in cases))

    def test_all_case_files_use_nonpositive_g(self):
        paths = [*sorted(OP_DIR.glob("*.json")), *sorted((SCRIPT_DIR / "cases").glob("*.json"))]
        self.assertEqual(len(paths), 4)
        for path in paths:
            cases = json.loads(path.read_text(encoding="utf-8"))
            for case in cases:
                with self.subTest(file=path.name, case=case["id"]):
                    gates = [item for item in case["inputs"] if item["name"] == "g"]
                    self.assertEqual(len(gates), 1)
                    self.assertEqual(gates[0]["range_values"], [-1, 0])

    def test_two_roles_per_smoke_case(self):
        self.assertEqual(RUNTIME_ROLE_ORDER, ("dut", "benchmark"))
        self.assertEqual(role_for_atk_task("npu", "phase6", False), "dut")
        self.assertEqual(role_for_atk_task("npu", "gold", False), "benchmark")
        log = "\n".join(f"[gdn-double-atk] case={case_id} role={role}" for case_id in range(6) for role in RUNTIME_ROLE_ORDER)
        self.assertTrue(audit_roles(log, 6)["passed"])
        self.assertFalse(audit_roles(log.replace("role=benchmark", "ignored"), 6)["passed"])
        self.assertFalse(audit_roles(log + "\n[gdn-double-atk] case=0 role=dut", 6)["passed"])

    def test_seven_eight_and_reordered_summary_columns(self):
        for error_column, reverse in ((False, False), (True, False), (False, True)):
            path = self.directory / "report.xlsx"
            make_report(path, [0, 1], error_column=error_column, reverse_columns=reverse)
            result = summarize_report(path, 2)
            self.assertTrue(result["passed"], result["errors"])

    def test_string_false_is_accuracy_failure(self):
        path = self.directory / "report.xlsx"
        make_report(path, [0, 1], failed_case=1)
        result = summarize_report(path, 2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["statistic"]["accuracy_failed_case_ids"], [1])

    def test_no_comparison_is_rejected(self):
        path = self.directory / "report.xlsx"
        make_report(path, [0], accuracy_columns=False)
        with self.assertRaisesRegex(ValueError, "精度通过"):
            summarize_report(path, 1)

    def test_incomplete_and_duplicate_cases_fail(self):
        path = self.directory / "report.xlsx"
        make_report(path, [0])
        self.assertFalse(summarize_report(path, 2)["passed"])
        make_report(path, [0, 0])
        self.assertFalse(summarize_report(path, 2)["passed"])

    def test_failed_or_old_shard_never_becomes_pass(self):
        path = self.directory / "shard_0_1/summary.json"
        path.parent.mkdir()
        summary = {"expected_cases": 1, "accuracy_standard": "mixed_tolerance_bm", "passed": False,
                   "statistic": {"case_ids": [0]}}
        path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertFalse(aggregate(self.directory, 1)["passed"])
        summary["passed"] = True
        path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertTrue(aggregate(self.directory, 1)["passed"])
        summary.pop("accuracy_standard")
        path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertFalse(aggregate(self.directory, 1)["passed"])


def bash_path(path):
    value = str(path).replace("\\", "/")
    return f"/{value[0].lower()}{value[2:]}" if len(value) > 1 and value[1] == ":" else value


@unittest.skipUnless(os.environ.get("GDN_ATK_TEST_BASH"), "设置 GDN_ATK_TEST_BASH 启用模拟 shell 入口")
class ShellEntryTest(TemporaryWorkspace):
    def setUp(self):
        super().setUp()
        self.bin_dir = self.directory / "bin"
        self.bin_dir.mkdir()
        python = shlex.quote(bash_path(sys.executable))
        for name, command in (("python3", f"{python} \"$@\""),
                              ("atk-stub", f"{python} {shlex.quote(bash_path(Path(__file__).resolve()))} --fake-atk \"$@\"")):
            path = self.bin_dir / name
            path.write_bytes(f"#!/usr/bin/env bash\nexec {command}\n".encode())
            path.chmod(0o755)

    def run_entry(self, script, **settings):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("GDN_ATK_", "DN_ATK_", "ACCURACY_")) and key not in {"ATK_BIN", "ATK_OUTPUT_ROOT"}}
        env.update(ATK_BIN=bash_path(self.bin_dir / "atk-stub"), PYTHONIOENCODING="utf-8", **settings)
        command = f"export PATH={shlex.quote(bash_path(self.bin_dir))}:$PATH\nexec bash {shlex.quote(bash_path(SCRIPT_DIR / script))} 0"
        return subprocess.run([os.environ["GDN_ATK_TEST_BASH"], "-c", command], cwd=REPO_DIR,
                              env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)

    def test_smoke_using_dn_alias_and_missing_role(self):
        result_dir = self.directory / "smoke"
        result = self.run_entry("run_double_benchmark.sh", DN_ATK_CASE_JSON=bash_path(OP_DIR / "atk_chunk_gated_delta_rule_fwd_mss.json"),
                                GDN_ATK_RESULT_DIR=bash_path(result_dir))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))["expected_cases"], 6)
        result = self.run_entry("run_double_benchmark.sh", GDN_ATK_CASE_JSON=bash_path(OP_DIR / "atk_chunk_gated_delta_rule_fwd_mss.json"),
                                GDN_ATK_RESULT_DIR=bash_path(self.directory / "missing"), GDN_FAKE_ATK_MODE="missing_role")
        self.assertNotEqual(result.returncode, 0)

    def test_default_matrix_runs_500_cases_and_resumes(self):
        matrix = self.directory / "matrix"
        result = self.run_entry("run_matrix.sh", GDN_ATK_MATRIX_ROOT=bash_path(matrix))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((matrix / "aggregate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["case_ids"], list(range(500)))
        self.assertEqual(summary["shard_count"], 20)
        self.assertTrue(summary["passed"])
        result = self.run_entry("run_matrix.sh", GDN_ATK_MATRIX_ROOT=bash_path(matrix))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("复用已有分片"), 20)

    def test_matrix_propagates_accuracy_failure(self):
        matrix = self.directory / "failed_matrix"
        result = self.run_entry("run_matrix.sh", GDN_ATK_CASE_JSON=bash_path(OP_DIR / "atk_chunk_gated_delta_rule_fwd_mss.json"),
                                GDN_ATK_MATRIX_ROOT=bash_path(matrix), GDN_ATK_SHARD_SIZE="3", GDN_FAKE_ATK_MODE="accuracy_fail")
        self.assertEqual(result.returncode, 99, result.stdout + result.stderr)
        self.assertFalse(json.loads((matrix / "aggregate_summary.json").read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fake-atk"]:
        fake_atk()
    else:
        unittest.main()

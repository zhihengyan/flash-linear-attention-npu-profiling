"""Offline behavioral checks; optional C++ host stub is not a CANN/NPU test."""
from contextlib import contextmanager, redirect_stdout
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
sys.path.insert(0, str(ASSETS))
import parse_timer_csv as parser
import trace_parser as trace


@contextmanager
def fixture():
    parent = Path.cwd().resolve()
    path = parent / (".op_timer_validation_" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != parent:
            raise RuntimeError("fixture cleanup escaped its workspace")
        shutil.rmtree(path)


def raw_fixture():
    raw = [0] * parser.TOTAL_BUFFER_SIZE
    raw[0:4] = [100, 300, 120, 180]
    return raw


def parse_raw(path, raw, **options):
    source, output = path / "raw.csv", path / "parsed.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(raw)
    with redirect_stdout(io.StringIO()):
        parser.parse_and_dump(source, output, **options)
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return output, reader.fieldnames, list(reader)


class ParserAndTrace(unittest.TestCase):
    def test_cycles_default_and_sparse_capacity(self):
        with fixture() as path:
            output, fields, rows = parse_raw(path, raw_fixture(), expected_core_ids=[0])
            self.assertEqual(int(rows[0]["KERNEL_TIMING_duration_cycles"]), 200)
            self.assertNotIn("KERNEL_TIMING_duration_us", fields)
            self.assertEqual(len(rows), parser.N_CORE_COUNT)
            self.assertFalse(any(field.endswith("_0_start_cycle") for field in fields))
            capture = io.StringIO()
            with redirect_stdout(capture):
                parser.print_summary(output)
            self.assertIn("200", capture.getvalue())
            self.assertIn("cycles", capture.getvalue())

    def test_verified_frequency_adds_us_without_losing_cycles(self):
        with fixture() as path:
            _, _, rows = parse_raw(path, raw_fixture(), cycles_per_us=250)
            self.assertEqual(int(rows[0]["KERNEL_TIMING_duration_cycles"]), 200)
            self.assertAlmostEqual(float(rows[0]["KERNEL_TIMING_duration_us"]), 0.8)

    def test_reject_bad_frequency(self):
        for frequency in (0, -1, math.inf, math.nan):
            with self.subTest(frequency=frequency):
                with self.assertRaises(ValueError):
                    parser.validate_frequency(frequency)
        parser.validate_frequency(None)

    def test_reject_length_and_keep_raw(self):
        with fixture() as path:
            with self.assertRaisesRegex(ValueError, "raw words"):
                parse_raw(path, raw_fixture()[:-1])
            self.assertTrue((path / "raw.csv").exists())
            self.assertFalse((path / "parsed.csv").exists())

    def test_reject_blank_capture(self):
        with fixture() as path:
            with self.assertRaisesRegex(ValueError, "no active"):
                parse_raw(path, [0] * parser.TOTAL_BUFFER_SIZE)

    def test_reject_missing_extra_or_invalid_expected_cores(self):
        for expected in ([0, 1], [1], [], [-1], [parser.N_CORE_COUNT]):
            with self.subTest(expected=expected), fixture() as path:
                with self.assertRaises(ValueError):
                    parse_raw(path, raw_fixture(), expected_core_ids=expected)

    def test_reject_incomplete_negative_or_outside_window(self):
        for values in ([0, 180], [190, 180], [90, 180], [120, 301]):
            with self.subTest(values=values), fixture() as path:
                raw = raw_fixture()
                raw[2:4] = values
                with self.assertRaises(ValueError):
                    parse_raw(path, raw)
        with fixture() as path:
            raw = raw_fixture()
            raw[0:2] = [0, 0]
            with self.assertRaisesRegex(ValueError, "KERNEL_TIMING"):
                parse_raw(path, raw)

    def test_dynamic_iterations_and_synthetic_accumulation(self):
        with fixture() as path:
            raw = raw_fixture()
            raw[parser.N_TIMING_COUNTER] = 1
            offset = parser.get_dynamic_timing_idx(0, 0) * 2
            raw[offset:offset+2] = [130, 1000]
            with patch.object(parser, "ACCUMULATED_EVENTS", {parser.DYNAMIC_TIMING_NAMES[0]}):
                output, _, rows = parse_raw(path, raw)
                name = parser.DYNAMIC_TIMING_NAMES[0] + "_0"
                self.assertEqual(int(rows[0][name + "_duration_cycles"]), 870)
                events = trace.generate_trace_json(output, cycles_per_us=250, shared_clock=True)
                self.assertNotIn(name, [event["name"] for event in events])
            with self.assertRaisesRegex(ValueError, "outside"):
                parse_raw(path, raw)

    def test_reject_corrupt_dynamic_counts(self):
        for count in (-1, 2**63):
            with self.subTest(count=count), fixture() as path:
                raw = raw_fixture()
                raw[parser.N_TIMING_COUNTER] = count
                with self.assertRaisesRegex(ValueError, "dynamic count"):
                    parse_raw(path, raw)

    def test_truncation_is_explicit_and_layout_stays_the_same(self):
        with fixture() as path:
            raw = raw_fixture()
            raw[parser.N_TIMING_COUNTER] = parser.MAX_DYNAMIC_ITER + 3
            output, fields, rows = parse_raw(path, raw)
            label = parser.DYNAMIC_TIMING_NAMES[0]
            self.assertEqual(int(rows[0][label + "_IterCount"]), parser.MAX_DYNAMIC_ITER + 3)
            self.assertEqual(int(rows[0][label + "_RecordedIterLimit"]), parser.MAX_DYNAMIC_ITER)
            self.assertEqual(int(rows[0][label + "_Truncated"]), 1)
            self.assertIn(f"{label}_{parser.MAX_DYNAMIC_ITER - 1}_start_cycle", fields)
            self.assertNotIn(f"{label}_{parser.MAX_DYNAMIC_ITER}_start_cycle", fields)
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["truncated_events"][0]["actual_iterations"], parser.MAX_DYNAMIC_ITER + 3)

    def test_reject_residual_dynamic_record_hidden_by_zero_count(self):
        with fixture() as path:
            raw = raw_fixture()
            offset = parser.get_dynamic_timing_idx(0, 0) * 2
            raw[offset:offset+2] = [130, 140]
            with self.assertRaisesRegex(ValueError, "declared iteration"):
                parse_raw(path, raw)

    def test_trace_requires_clock_and_supports_generic_names_and_points(self):
        with fixture() as path:
            raw = raw_fixture()
            raw[4:6] = [200, 200]
            output, _, _ = parse_raw(path, raw)
            for options in ({}, {"cycles_per_us": 250}, {"shared_clock": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    trace.generate_trace_json(output, **options)
            events = trace.generate_trace_json(output, cycles_per_us=250, shared_clock=True)
            stage = next(event for event in events if event["name"] == parser.FIXED_TIMING_NAMES[1])
            self.assertAlmostEqual(stage["ts"], 0.08)
            point = next(event for event in events if event["name"] == parser.FIXED_TIMING_NAMES[2])
            self.assertEqual(point["ph"], "i")
            self.assertNotIn("dur", point)
            self.assertTrue(json.dumps(events, allow_nan=False))

    def test_aiv_own_row_and_per_capture_origin(self):
        with fixture() as path:
            raw = raw_fixture()
            raw[:parser.N_TIMING_COUNTER_PER_CORE_ALIGN] = [0] * parser.N_TIMING_COUNTER_PER_CORE_ALIGN
            offset = parser.get_base_index(2)
            raw[offset:offset+4] = [10000, 10200, 10020, 10080]
            output, _, rows = parse_raw(path, raw, expected_core_ids=[2])
            self.assertEqual((rows[2]["core_type"], rows[2]["sub_id"]), ("AIV", "1"))
            events = trace.generate_trace_json(output, cycles_per_us=250, shared_clock=True)
            self.assertEqual(min(event["ts"] for event in events), 0)

    def test_cli_help_and_real_parse_and_trace(self):
        for script in ("parse_timer_csv.py", "trace_parser.py"):
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with fixture() as path:
            parse_raw(path, raw_fixture())
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / "parse_timer_csv.py"), "--raw", str(path / "raw.csv"),
                                     "--output", str(path / "cli.csv"), "--expected-core-ids", "0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / "trace_parser.py"), "--csv", str(path / "cli.csv"),
                                     "--output", str(path / "trace.json"), "--cycles-per-us", "250", "--shared-clock"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(len(json.loads((path / "trace.json").read_text())), 0)


CPP_STUB = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#define __aicore__
#define __gm__
#define __inline__ inline
#define PIPE_ALL 0
#define ASCEND_IS_AIV (AscendC::aiv)
using GM_ADDR = uint8_t*;
namespace AscendC {
bool aiv=false;
int block=1, sub=0, ratio=2, queries=0, reads=0, barriers=0;
int64_t cycle=100;
std::vector<uintptr_t> flushes;
int GetBlockIdx(){++queries;return block;}
int GetSubBlockIdx(){++queries;return sub;}
int GetTaskRation(){++queries;return ratio;}
int64_t GetSystemCycle(){++reads;return ++cycle;}
template<int> void PipeBarrier(){++barriers;}
template<class T> struct GlobalTensor {
 T* ptr=nullptr;
 void SetGlobalBuffer(T* value){ptr=value;}
 T& operator()(int index){return ptr[index];}
 GlobalTensor operator[](int index) const {GlobalTensor result;result.ptr=ptr+index;return result;}
};
using ::printf;
#if ASCEND_TIMER_EXPLICIT_GM_PUBLISH
enum class CacheLine {SINGLE_CACHE_LINE};
enum class DcciDst {CACHELINE_OUT};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(const GlobalTensor<T>& data){
 assert(reinterpret_cast<uintptr_t>(data.ptr)%64==0);
 flushes.push_back(reinterpret_cast<uintptr_t>(data.ptr));
}
#endif
}
#include "AscendTimerV2_device.hpp"
alignas(64) static int64_t raw[ASCEND_TIMER_TOTAL_BUFFER_SIZE] = {};
int main(){
 AscendTimerDevice timer;
 timer.Init(reinterpret_cast<GM_ADDR>(raw));
 TIMER_BLOCK(timer.TikNoBarrier());
 TIMER_BLOCK(timer.TokNoBarrier<Overwrite>(AscendTimer::KERNEL_TIMING_IDX));
#if !ENABLE_TIMER
 TIMER_BLOCK(unknown_function<unknown_type, other_unknown_type>());
 timer.Tik();timer.Tok();timer.Publish();
 assert(timer.GetTAddr()==nullptr);
 assert(AscendC::queries==0 && AscendC::reads==0 && AscendC::barriers==0 && AscendC::flushes.empty());
 for(auto word:raw){assert(word==0);}
#else
 assert(timer.GetCoreId()==3);
 int row=3*AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN;
 assert(raw[row]>0 && raw[row+1]>raw[row]);
 int before=AscendC::reads;
 timer.TokNoBarrier<Overwrite>(-1);
 timer.Tok<Overwrite>(static_cast<AscendTimer::DynamicTimingType>(-1),0);
 timer.Tik(static_cast<AscendTimer::DynamicTimingType>(0),AscendTimer::MAX_DYNAMIC_ITER);
 assert(AscendC::reads==before && AscendC::barriers==0);
 timer.TikNoBarrier(static_cast<AscendTimer::DynamicTimingType>(0),0);
 before=AscendC::reads;
 timer.Tok<Overwrite>(static_cast<AscendTimer::DynamicTimingType>(0),1);
 assert(AscendC::reads==before && AscendC::barriers==0);
 timer.TokNoBarrier<Overwrite>(static_cast<AscendTimer::DynamicTimingType>(0),0);
 timer.setDynamicActualIter(static_cast<AscendTimer::DynamicTimingType>(0),AscendTimer::MAX_DYNAMIC_ITER+1);
 assert(raw[AscendTimer::N_TIMING_COUNTER+3*AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN]==AscendTimer::MAX_DYNAMIC_ITER+1);
 timer.Publish();
#if ASCEND_TIMER_EXPLICIT_GM_PUBLISH
 int count=(AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN+AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN)/8;
 assert(static_cast<int>(AscendC::flushes.size())==count);
 uintptr_t base=reinterpret_cast<uintptr_t>(raw);
 for(auto addr:AscendC::flushes){
   int offset=(addr-base)/8;
   int dyn=AscendTimer::N_TIMING_COUNTER+3*AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
   assert((offset>=row && offset<row+AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN) ||
          (offset>=dyn && offset<dyn+AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN));
 }
#else
 assert(AscendC::flushes.empty());
#endif
 assert(AscendC::barriers==0);
 AscendC::aiv=true;AscendC::block=3;AscendC::sub=1;
 timer.Init(reinterpret_cast<GM_ADDR>(raw));assert(timer.GetCoreId()==5);
 AscendC::ratio=1;timer.Init(reinterpret_cast<GM_ADDR>(raw));assert(timer.GetTAddr()==nullptr);
 AscendC::aiv=false;AscendC::block=AscendTimer::N_CORE_COUNT;
 timer.Init(reinterpret_cast<GM_ADDR>(raw));assert(timer.GetTAddr()==nullptr);
 timer.Init(reinterpret_cast<GM_ADDR>(raw)+8);assert(timer.GetTAddr()==nullptr);
 timer.Init(nullptr);assert(timer.GetTAddr()==nullptr);
 before=AscendC::reads;timer.Tik();timer.Tok();timer.Publish();assert(AscendC::reads==before);
 assert(raw[0]==0 && raw[1]==0);
#endif
}
'''


class OptionalHostStub(unittest.TestCase):
    def test_cpp_switch_mapping_and_publication(self):
        compiler = os.getenv("ASCEND_TIMER_TEST_CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++17 compiler; CANN/NPU and host-stub verification are not performed")
        with fixture() as path:
            source = path / "stub.cpp"
            source.write_text(CPP_STUB, encoding="utf-8")
            for enabled, publish in ((0, 1), (1, 0), (1, 1)):
                with self.subTest(enabled=enabled, publish=publish):
                    binary = path / (f"stub_{enabled}_{publish}" + (".exe" if sys.platform == "win32" else ""))
                    command = [compiler] + (["c++"] if Path(compiler).stem.lower() == "zig" else [])
                    command += ["-std=c++17", "-I", str(ASSETS), f"-DENABLE_TIMER={enabled}",
                               f"-DASCEND_TIMER_EXPLICIT_GM_PUBLISH={publish}", str(source), "-o", str(binary)]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    result = subprocess.run([str(binary)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

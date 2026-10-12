"""Regression checks for the small patches to the original timer templates."""
from contextlib import contextmanager, redirect_stdout
import csv
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSETS))
import parse_timer_csv as timer
import trace_parser as trace


@contextmanager
def fixture():
    parent = Path.cwd().resolve()
    directory = parent / (".timer_patch_test_" + uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == parent
        shutil.rmtree(directory)


def capture(directory, *, frequency=None, bad=False, stage_pair=None):
    raw = [0] * timer.TOTAL_BUFFER_SIZE
    raw[:6] = [100, 300, 120, 180 if not bad else 110, 200, 200]
    if stage_pair is not None:
        raw[2:4] = stage_pair
    # One iteration per type keeps the original wide CSV small; unrelated core rows remain zero.
    for kind in range(timer.DYNAMIC_TYPE_COUNT):
        raw[timer.N_TIMING_COUNTER + kind] = 1
        offset = timer.get_dynamic_timing_idx(kind, 0) * 2
        raw[offset:offset+2] = [130, 140]
    source, output = directory / "raw.csv", directory / "parsed.csv"
    with source.open("w", newline="") as handle:
        csv.writer(handle).writerow(raw)
    with redirect_stdout(io.StringIO()):
        timer.parse_and_dump(source, output, frequency)
    return output


class ExportTests(unittest.TestCase):
    def test_default_cycles_and_original_row_layout(self):
        with fixture() as directory:
            output = capture(directory)
            with output.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(rows[0]["KERNEL_TIMING_duration_cycles"], "200")
            self.assertNotIn("KERNEL_TIMING_duration(us)", rows[0])
            self.assertEqual(len(rows), timer.N_CORE_COUNT)
            self.assertEqual((rows[2]["core_type"], rows[2]["sub_id"]), ("AIV", "1"))
            with redirect_stdout(io.StringIO()) as summary:
                timer.print_summary(output)
            self.assertIn("cycles", summary.getvalue())

    def test_confirmed_frequency_and_invalid_values(self):
        with fixture() as directory:
            with capture(directory, frequency=250).open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["KERNEL_TIMING_duration(us)"]), 0.8)
        for frequency in (0, -1, math.nan, math.inf):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                timer.validate_frequency(frequency)

    def test_negative_interval_and_empty_capture_keep_raw(self):
        with fixture() as directory:
            with self.assertRaises(ValueError):
                capture(directory, bad=True)
            self.assertTrue((directory / "raw.csv").exists())
            with (directory / "raw.csv").open("w", newline="") as handle:
                csv.writer(handle).writerow([0] * timer.TOTAL_BUFFER_SIZE)
            with self.assertRaises(ValueError):
                timer.parse_and_dump(directory / "raw.csv", directory / "empty.csv")

    def test_window_check_excludes_accumulated_synthetic_end(self):
        with fixture() as directory:
            with self.assertRaisesRegex(ValueError, "outside KERNEL_TIMING"):
                capture(directory, stage_pair=[120, 1000])
            with patch.object(timer, "ACCUMULATED_EVENTS", {"FIXED_STAGE_1"}):
                output = capture(directory, stage_pair=[120, 1000])
                events = trace.generate_trace_json(output, 0, 100, 250, True)
                self.assertNotIn("FIXED_STAGE_1", [event["name"] for event in events])
                for pair in ([90, 1000], [350, 1000]):
                    with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "outside KERNEL_TIMING"):
                        capture(directory, stage_pair=pair)

    def test_trace_clock_generic_names_points_and_accumulation(self):
        with fixture() as directory:
            output = capture(directory)
            with self.assertRaises(ValueError):
                trace.generate_trace_json(output, 0, 100)
            with self.assertRaises(ValueError):
                trace.generate_trace_json(output, 0, 100, 250)
            with patch.object(timer, "ACCUMULATED_EVENTS", {timer.DYNAMIC_TIMING_NAMES[0]}):
                events = trace.generate_trace_json(output, 0, 100, 250, True)
            names = [event["name"] for event in events]
            self.assertIn("FIXED_STAGE_1", names)
            self.assertNotIn(timer.DYNAMIC_TIMING_NAMES[0] + "_0", names)
            self.assertEqual(next(event for event in events if event["name"] == "FIXED_STAGE_1")["ts"], 0.08)
            self.assertEqual(next(event for event in events if event["name"] == "FIXED_STAGE_2")["ph"], "i")

    def test_original_cli_help_and_directory_arguments(self):
        for script in ("parse_timer_csv.py", "trace_parser.py"):
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--cycles-per-us", result.stdout)
        with fixture() as directory:
            capture(directory)
            shutil.copyfile(directory / "raw.csv", directory / "rank_0_npu_time.csv")
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / "parse_timer_csv.py"), "--dir", str(directory),
                                     "--rank-start", "0", "--rank-end", "0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, "-X", "utf8", str(ASSETS / "trace_parser.py"), "--timer-dir", str(directory),
                                     "--output-dir", str(directory / "trace"), "--rank-start", "0", "--rank-end", "0",
                                     "--cycles-per-us", "250", "--shared-clock"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((directory / "trace/rank_0_npu_time.json").exists())


CPP = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#define __aicore__
#define __gm__
#define __inline__ inline
#define PIPE_ALL 0
#define ASCEND_IS_AIV (AscendC::aiv)
using GM_ADDR=uint8_t*;
namespace AscendC {
bool aiv=false; int block=1, sub=0, queries=0, clocks=0, barriers=0; int64_t cycle=100;
std::vector<uintptr_t> flushes;
int GetBlockIdx(){++queries;return block;} int GetSubBlockIdx(){++queries;return sub;} int GetTaskRation(){++queries;return 2;}
int64_t GetSystemCycle(){++clocks;return ++cycle;} template<int> void PipeBarrier(){++barriers;}
template<class T> struct GlobalTensor {T* p=nullptr; void SetGlobalBuffer(T* value){p=value;}
 T& operator()(int i){return p[i];} GlobalTensor operator[](int i)const{GlobalTensor x;x.p=p+i;return x;}};
using ::printf;
#if ASCEND_TIMER_EXPLICIT_GM_PUBLISH
enum class CacheLine{SINGLE_CACHE_LINE}; enum class DcciDst{CACHELINE_OUT};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(const GlobalTensor<T>& x){
assert(reinterpret_cast<uintptr_t>(x.p)%64==0);flushes.push_back(reinterpret_cast<uintptr_t>(x.p));}
#endif
}
#include "AscendTimerV2_device.hpp"
alignas(64) static int64_t raw[ASCEND_TIMER_TOTAL_BUFFER_SIZE]={};
int main(){AscendTimerDevice t;TIMER_BLOCK(t.Init(reinterpret_cast<GM_ADDR>(raw)));
TIMER_BLOCK(t.TikNoBarrier());TIMER_BLOCK(t.TokNoBarrier<Overwrite>(AscendTimer::KERNEL_TIMING_IDX));
#if !ENABLE_TIMER
TIMER_BLOCK(undefined_function<undefined_type>());t.Tik();t.Tok();t.Publish();
assert(AscendC::queries==0 && AscendC::clocks==0 && AscendC::barriers==0 && AscendC::flushes.empty());
for(auto x:raw){assert(x==0);}
#else
int row=3*AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN;assert(t.GetCoreId()==3 && raw[row]>0 && raw[row+1]>raw[row]);
int before=AscendC::clocks;t.Tok<Overwrite>(static_cast<AscendTimer::DynamicTimingType>(-1),0);t.TokNoBarrier<Overwrite>(-1);
assert(AscendC::clocks==before && AscendC::barriers==0);t.Publish();
#if ASCEND_TIMER_EXPLICIT_GM_PUBLISH
assert(AscendC::flushes.size()==(AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN+AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN)/8);
for(auto addr:AscendC::flushes){int offset=(addr-reinterpret_cast<uintptr_t>(raw))/8;
int dyn=AscendTimer::N_TIMING_COUNTER+3*AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
assert((offset>=row && offset<row+AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN) || (offset>=dyn && offset<dyn+AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN));}
#else
assert(AscendC::flushes.empty());
#endif
AscendC::aiv=true;AscendC::block=3;AscendC::sub=1;t.Init(reinterpret_cast<GM_ADDR>(raw));assert(t.GetCoreId()==5);
AscendC::aiv=false;AscendC::block=AscendTimer::N_CORE_COUNT;t.Init(reinterpret_cast<GM_ADDR>(raw));t.zero(0);assert(raw[0]==0);
t.Init(nullptr);before=AscendC::clocks;t.Tik();t.Tok();t.Publish();assert(AscendC::clocks==before);
#endif
}
'''


class HostTests(unittest.TestCase):
    def test_original_cpp_api_with_switch_and_optional_publication(self):
        compiler = os.getenv("ASCEND_TIMER_TEST_CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; host stub not executed")
        with fixture() as directory:
            source = directory / "stub.cpp"
            source.write_text(CPP, encoding="utf-8")
            binaries = []
            for enabled, publish in ((0, 1), (1, 0), (1, 1)):
                with self.subTest(enabled=enabled, publish=publish):
                    binary = directory / (f"stub_{enabled}_{publish}" + (".exe" if sys.platform == "win32" else ""))
                    command = [compiler] + (["c++"] if Path(compiler).stem.lower() == "zig" else [])
                    command += ["-std=c++17", "-I", str(ASSETS), f"-DENABLE_TIMER={enabled}",
                                f"-DASCEND_TIMER_EXPLICIT_GM_PUBLISH={publish}", str(source), "-o", str(binary)]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    binaries.append(binary)
            for binary in binaries:
                with self.subTest(binary=binary.name):
                    try:
                        result = subprocess.run([str(binary)], capture_output=True, text=True)
                    except OSError as error:
                        if getattr(error, "winerror", None) == 4551:
                            self.skipTest("All three C++ stub variants compiled; Windows application control blocked execution")
                        raise
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

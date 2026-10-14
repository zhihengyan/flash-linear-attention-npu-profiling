"""Offline layout/decoder contracts; these do not replace CANN/NPU validation."""
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import sys
import types
from unittest import mock

import parse_timer_csv as p
from trace_parser import generate_trace_json

L = p.LAYOUT

def sample(kernels=(0,), actual=2):
    raw = [0] * p.TOTAL_BUFFER_SIZE
    for kernel in kernels:
        offset = kernel * L['BUFFER_WORDS_PER_KERNEL']
        header = [L['HEADER_MAGIC'], L['LAYOUT_VERSION'], kernel, 1,
                  L['CORE_GROUP_CAPACITY'], L['N_TIMING_COUNTER_PER_CORE_ALIGN'],
                  len(p.FIXED_TIMING_NAMES), 1, 310, len(p.DYNAMIC_TIMING_NAMES),
                  L['MAX_DYNAMIC_ITER'], L['DYNAMIC_ITER_PER_CORE_ALIGN'],
                  0, 8, 0, L['DETAIL_TAG_PER_CORE_ALIGN']]
        raw[offset:offset + len(header)] = header
        for core in range(3):
            row = offset + L['HEADER_WORDS'] + core * L['N_TIMING_COUNTER_PER_CORE_ALIGN']
            # Large integer ticks exercise conversion without float precision loss.
            base = 2**54 + core
            fixed = [base, base+1000, base+1, base+10, base+11, base+1000]
            if kernel == 0:
                fixed += [base+20, base+21, base+30, base+31]
            raw[row:row+len(fixed)] = fixed
            count = offset + L['HEADER_WORDS'] + L['N_TIMING_COUNTER'] + core * L['DYNAMIC_ITER_PER_CORE_ALIGN']
            raw[count] = actual
            raw[count + L['DYNAMIC_ITER_TOTAL_SIZE']] = min(actual, L['MAX_DYNAMIC_ITER'])
            for iteration in range(min(actual, L['MAX_DYNAMIC_ITER'])):
                index = row + 2 * (len(p.FIXED_TIMING_NAMES) + iteration)
                raw[index:index+2] = [base+40+iteration, base+41+iteration]
    return raw

class TimerContract(unittest.TestCase):
    def test_enum_names_and_alignment(self):
        self.assertEqual(len(p.FIXED_TIMING_NAMES), 6)
        self.assertEqual(L['FIRST_DETAIL_TYPE'], 28)
        self.assertEqual(len(p.DYNAMIC_TIMING_NAMES), 60)
        self.assertEqual(len(set(p.ENUMS['FixedTiming'] + p.ENUMS['DynamicTimingType'])), 66)
        for key in ('HEADER_WORDS', 'N_TIMING_COUNTER_PER_CORE_ALIGN', 'DYNAMIC_ITER_PER_CORE_ALIGN', 'DETAIL_TAG_PER_CORE_ALIGN', 'BUFFER_WORDS_PER_KERNEL'):
            self.assertEqual(L[key] % L['CACHE_LINE_WORDS'], 0)

    def test_phase6_and_repeated_slots(self):
        events, metadata = p.parse_timer(sample())
        repeated = [e for e in events if e['stage'] == 'CUMSUM' and e['core_id'] == 0]
        self.assertEqual([e['iteration'] for e in repeated], [0, 1])
        self.assertEqual(len({e['start_cycle'] for e in repeated}), 2)
        self.assertEqual(metadata[0]['iterations'][0]['actual'], 2)

    def test_three_launch_route(self):
        _, metadata = p.parse_timer(sample((1, 2, 3)))
        self.assertEqual({m['kernel'] for m in metadata}, {'PREPARE', 'FWD_H', 'FWD_O'})

    def test_truncation_preserves_actual_count(self):
        events, metadata = p.parse_timer(sample(actual=75))
        self.assertEqual(metadata[0]['iterations'][0], dict(core_id=0, stage='CUMSUM', actual=75, sampled=32, truncated=True))
        self.assertEqual(len([e for e in events if e['iteration'] >= 0]), 3*32)

    def test_wrong_length(self):
        with self.assertRaisesRegex(ValueError, 'length'):
            p.parse_timer(sample()[:-1])

    def test_wrong_layout(self):
        raw=sample();raw[1]=1
        with self.assertRaisesRegex(ValueError, 'metadata'):p.parse_timer(raw)

    def test_missing_slice(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):p.parse_timer(sample((1, 2)))

    def test_mixed_routes(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):p.parse_timer(sample((0, 1, 2, 3)))

    def test_missing_pair(self):
        raw=sample();raw[L['HEADER_WORDS']+2*len(p.FIXED_TIMING_NAMES)+1]=0
        with self.assertRaisesRegex(ValueError, 'unpaired'):p.parse_timer(raw)

    def test_outside_total_window(self):
        raw=sample();raw[L['HEADER_WORDS']+2*len(p.FIXED_TIMING_NAMES)+1]=2**54+1100
        with self.assertRaisesRegex(ValueError, 'outside KERNEL'):p.parse_timer(raw)

    def test_data_beyond_count(self):
        raw=sample();raw[L['HEADER_WORDS']+L['N_TIMING_COUNTER']]=1
        with self.assertRaisesRegex(ValueError, 'sampled count'):p.parse_timer(raw)

    def test_count_padding(self):
        raw=sample();raw[L['HEADER_WORDS']+L['N_TIMING_COUNTER']+len(p.DYNAMIC_TIMING_NAMES)]=1
        with self.assertRaisesRegex(ValueError, 'count padding'):p.parse_timer(raw)

    def test_trace_clock_guard_and_integer_origin(self):
        events,_=p.parse_timer(sample())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'trace.json'
            with self.assertRaisesRegex(ValueError, 'verified'):generate_trace_json(events,path,None)
            generate_trace_json(events,path,50,shared_clock=True)
            trace=json.loads(path.read_text())['traceEvents']
            repeated=[e for e in trace if e['name']=='CUMSUM']
            self.assertEqual(repeated[0]['ts'],40/50)
            self.assertEqual(repeated[0]['args']['iteration'],0)

class PerformanceEvidenceContract(unittest.TestCase):
    def test_successful_case_and_real_kernel_required(self):
        root=Path(__file__).resolve().parents[8]
        script=root/'tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py'
        spec=importlib.util.spec_from_file_location('gdn_performance_test',script)
        collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
        reader=types.ModuleType('summarize_atk_report')
        reader._statistic_result=lambda _:dict(case_ids=[0],execution_failed_case_ids=[])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, mock.patch.dict(sys.modules,{'summarize_atk_report':reader}):
            result=Path(directory);(result/'report.xlsx').touch()
            csv=result/'op_summary_test.csv'
            csv.write_text('Name,Task Duration(us)\nChunkGatedDeltaRuleFwd_0,10\n',encoding='utf-8')
            self.assertEqual(collector.verify_performance_evidence(result,[0])['executed_case_ids'],[0])
            with self.assertRaisesRegex(RuntimeError,'case evidence'):collector.verify_performance_evidence(result,[1])
            csv.write_text('Name,Task Duration(us)\nWrongKernel,10\n',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'actual GDN kernel'):collector.verify_performance_evidence(result,[0])

if __name__ == '__main__':
    unittest.main()

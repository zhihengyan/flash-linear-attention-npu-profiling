"""Small synthetic analysis fixtures; no device results are implied."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from summarize_timer import analyze, brief, main, read_events, union_cycles


def fixture(actual=2):
    events=[]
    counts=[]
    origin=2**54
    def add(core,stage,kind,iteration,start,end):
        events.append(dict(kernel='PHASE6',core_id=core,group_id=0,
            core_type='AIC' if core==0 else 'AIV',sub_id=core-1,
            stage=stage,kind=kind,iteration=iteration,start_cycle=origin+start,
            end_cycle=origin+end,duration_cycles=end-start))
    for core in range(3):
        add(core,'KERNEL_TIMING','window',-1,1,101)
        add(core,'MAIN_INIT','issue',-1,1,11)
        add(core,'MAIN_BODY','window',-1,11,101)
        add(core,'H_WH','issue',0,11,71)
        add(core,'H_WH','issue',1,71,81)
        add(core,'H_WAIT','wait',0,21,51)
        # Same existing hand-off covered by both fixed and dynamic windows.
        add(core,'PHASE6_HO_WAIT','window',-1,31,41)
        counts.extend([dict(core_id=core,stage='H_WH',actual=actual,sampled=2,truncated=actual>2),
                       dict(core_id=core,stage='H_WAIT',actual=1,sampled=1,truncated=False)])
    layout=dict(layout=dict(CORES_PER_GROUP=3,CORE_GROUP_CAPACITY=256,MAX_DYNAMIC_ITER=2),
                kernels=[dict(kernel='PHASE6',block_dim=1,iterations=counts)],cycles_per_us=None)
    capture=dict(decode_status='complete',output_comparison='ten outputs exactly match uninstrumented call',sample_count=1)
    return events,layout,capture


class TimerSummaryTest(unittest.TestCase):
    def test_nested_and_adjacent_union(self):
        self.assertEqual(union_cycles([(0,10),(2,8),(5,20),(20,22),(30,35)]),27)

    def test_waits_are_merged_and_not_added_to_compute(self):
        summary,cores=analyze(*fixture())
        self.assertEqual(cores[0]['observed_wait_union_cycles'],30)
        self.assertEqual(cores[0]['observed_wait_fraction'],.3)
        self.assertEqual(cores[0]['sampled_scope_union_cycles'],70)
        self.assertEqual(summary['warnings'],[])

    def test_stage_core_cycle_sum_and_actual_count_are_distinct(self):
        summary,_=analyze(*fixture(actual=100))
        stage=next(r for r in summary['stages'] if r['stage']=='H_WH' and r['core_type']=='AIV')
        self.assertEqual(stage['sample_sum_core_cycles'],140)
        self.assertEqual(stage['sampled_calls'],4)
        self.assertEqual(stage['actual_calls_per_core_min'],100)
        self.assertEqual(stage['truncated_cores'],2)
        self.assertEqual(summary['truncated_core_stage_counters'],3)
        self.assertIn('Do not extrapolate',summary['warnings'][0])

    def test_offsets_preserve_large_integer_precision(self):
        summary,_=analyze(*fixture())
        longest=summary['longest_dynamic_windows'][0]
        self.assertEqual(longest['start_offset_cycles'],10)

    def test_missing_sample_rejected(self):
        events,layout,capture=fixture();events=[e for e in events if not(e['core_id']==0 and e['stage']=='H_WH' and e['iteration']==1)]
        with self.assertRaisesRegex(ValueError,'missing samples'):analyze(events,layout,capture)

    def test_missing_core_rejected(self):
        events,layout,capture=fixture();events=[e for e in events if e['core_id']!=2]
        with self.assertRaisesRegex(ValueError,'cores/launches'):analyze(events,layout,capture)

    def test_outside_parent_rejected(self):
        events,layout,capture=fixture();events[3].update(end_cycle=2**54+102,duration_cycles=91)
        with self.assertRaisesRegex(ValueError,'outside'):analyze(events,layout,capture)

    def test_no_comparison_is_not_reported_as_passed(self):
        events,layout,_=fixture();summary,_=analyze(events,layout)
        self.assertEqual(summary['capture_decode_status'],'unknown')
        self.assertEqual(summary['output_comparison'],'unknown')
        self.assertEqual(len(summary['warnings']),2)

    def test_bad_count_metadata_rejected(self):
        events,layout,capture=fixture(actual=10);layout['kernels'][0]['iterations'][0]['sampled']=10
        with self.assertRaisesRegex(ValueError,'sampled/actual'):analyze(events,layout,capture)

    def test_duplicate_csv_event_rejected(self):
        events,_,_=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'events.csv'
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events+[events[0]])
            with self.assertRaisesRegex(ValueError,'duplicate'):read_events(path)

    def test_cli_small_summary_and_existing_results_preserved(self):
        events,layout,capture=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            directory=Path(directory)
            with (directory/'gdn_npu_time_parsed.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
            (directory/'timer_layout.json').write_text(json.dumps(layout),encoding='utf-8')
            (directory/'capture_metadata.json').write_text(json.dumps(capture),encoding='utf-8')
            with mock.patch('sys.stdout'),mock.patch('sys.stderr'):
                self.assertEqual(main([str(directory)]),0)
                self.assertEqual(main([str(directory)]),1)
            output=directory/'analysis'
            self.assertTrue((output/'core_summary.csv').is_file())
            self.assertTrue((output/'stage_summary.csv').is_file())
            self.assertLess((output/'brief.txt').stat().st_size,8000)
            result=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertIn('parsed_csv_sha256',result)
            self.assertIn('NOT latency shares',brief(result))


if __name__=='__main__':unittest.main()

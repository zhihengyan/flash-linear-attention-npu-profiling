"""Check the instrumentation inventory, existing synchronization order and ABI.

Compare to the pinned main baseline; print source evidence for independent review.
"""
from pathlib import Path
import ast
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[8]
BASELINE = '90176b44321ebdbf3ae2a9057243157d7085c6e1'
OP = ROOT / 'fla/ops/ascendc/gdn/chunk_gdn_fwd'
DOC = OP / 'chunk_gated_delta_rule_fwd/docs/timer_flow.md'

def inventory():
    items=[]
    for path in sorted(OP.rglob('*')):
        if path.suffix not in ('.h','.hpp','.cpp'):continue
        source=path.read_text(encoding='utf-8')
        for match in re.finditer(r'GDN_TIMER_SCOPE\(([^,]+),\s*(\w+)\)',source):
            line=source.count('\n',0,match.start())+1
            items.append(dict(file=path.relative_to(ROOT).as_posix(),line=line,stage=match[2],pointer=match[1]))
    return items

def strip_comments(source):
    return re.sub(r'//[^\n]*|/\*.*?\*/','',source,flags=re.S)

def sync_calls(source):
    # Comparing ordered calls catches removed/moved business waits or flags.
    source=strip_comments(source)
    return [re.sub(r'\s+','',m[0]) for m in re.finditer(
        r'(?:(?:AscendC|Catlass::Arch)::)?(?:CrossCoreWaitFlag|CrossCoreSetFlag|CrossCoreBarrier|SyncAll|WaitFlag|SetFlag|PipeBarrier)(?:<[^;]*?>)?\([^;]*?\);',source)]

def baseline(path):
    return subprocess.check_output(['git','show',BASELINE+':'+path],cwd=ROOT).decode('utf-8')

def main():
    import parse_timer_csv as parser
    items=inventory(); names=set(parser.DYNAMIC_TIMING_NAMES.values())
    used={i['stage'] for i in items}
    assert used==names,('stage coverage mismatch',used^names)
    doc=DOC.read_text(encoding='utf-8')
    for item in items:
        assert f"{item['file']}:{item['line']}" in doc,('missing document anchor',item)
    paths=subprocess.check_output(['git','diff','--name-only',BASELINE,'--',str(OP)],cwd=ROOT,stderr=subprocess.DEVNULL).decode().splitlines()
    checked=[]
    for relative in paths:
        path=ROOT/relative
        if path.suffix not in ('.h','.hpp','.cpp'):continue
        before=baseline(relative);after=path.read_text(encoding='utf-8')
        assert sync_calls(before)==sync_calls(after),('business synchronization changed',relative)
        checked.append(relative)
    # Existing public C declarations and the production Python signature persist.
    relative='fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h'
    before=strip_comments(baseline(relative));after=strip_comments((ROOT/relative).read_text(encoding='utf-8'))
    for declaration in re.findall(r'ACLNN_API\s+aclnnStatus\s+\w+\([^;]*;',before,re.S):
        assert re.sub(r'\s+','',declaration) in re.sub(r'\s+','',after),declaration
    relative='torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py'
    def signature(source):
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='npu_chunk_gated_delta_rule_fwd')
        return ast.dump(fn.args,include_attributes=False)
    assert signature(baseline(relative))==signature((ROOT/relative).read_text(encoding='utf-8'))
    print(json.dumps(dict(status='source_checks_passed',baseline=BASELINE,fixed_count=len(parser.FIXED_TIMING_NAMES),
        dynamic_count=len(names),scope_count=len(items),synchronization_files=checked,scopes=items),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

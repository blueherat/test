"""Continue validation/rollout only after the original driver was observed absent."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path('/home/zhoushunyu/eqvae')
OUT=Path(__file__).resolve().parent
RUNNER=ROOT/'experiments/run_raev2_paired_bridge_pilot.py'
PYTHON='/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def put(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    temp.replace(path)

for old_pid in (614895,615756):
    assert not Path(f'/proc/{old_pid}').exists(),f'old process still present {old_pid}'
train=json.loads((OUT/'train/summary.json').read_text())
assert train['complete'] and train['updates']==2048
assert sha(train['checkpoint']['path'])==train['checkpoint']['sha256']
assert sha(OUT/'train/request.json')==train['request']['sha256']
record=dict(pid=os.getpid(),created_utc=now(),complete=False,
    driver_sha256=sha(__file__),stages=[],
    recovery=dict(original_driver_pid=614895,original_train_pid=615756,
        missing_processes_confirmed_utc='2026-09-06T14:29:00+00:00',
        original_session_missing=True,train_restarted=False,
        original_outer_returncode='not captured',original_outer_wall='not captured',
        training_summary_sha256=sha(OUT/'train/summary.json'),
        training_final_checkpoint_sha256=train['checkpoint']['sha256'],
        reason='parent absent; fixed final training artifacts complete; continue unstarted stages only'))
with (OUT/'pipeline_resume_created.json').open('x') as f: json.dump(record,f,indent=2)
pilot=json.loads((OUT/'pilot/request.json').read_text())
supplement=json.loads((OUT/'supplemental_source_environment.json').read_text())
for mode in ('validate','rollout'):
    for name,item in pilot['sources'].items():
        assert sha(ROOT/name)==item['sha256'],name
    for item in supplement['sources']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    cmd=[PYTHON,'-u',str(RUNNER),'--mode',mode,'--output-dir',str(OUT/mode),
         '--prior-dir',str(OUT/'train')]
    assert not (OUT/mode).exists(),f'refuse repeated stage {mode}'
    entry=dict(mode=mode,argv=cmd,started_utc=now(),complete=False)
    record['stages'].append(entry)
    with (OUT/(mode+'.log')).open('x') as log:
        started=time.perf_counter()
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid
        put(OUT/'pipeline_resume_status.json',record)
        print(json.dumps(entry),flush=True)
        code=child.wait()
        entry.update(returncode=code,outer_child_wall_seconds=time.perf_counter()-started,
                     finished_utc=now(),complete=True)
    put(OUT/'pipeline_resume_status.json',record)
    print(json.dumps(entry),flush=True)
    if code:
        record['terminal_failure']=mode
        put(OUT/'pipeline_resume_status.json',record)
        sys.exit(code)
    summary=OUT/mode/'summary.json'
    assert json.loads(summary.read_text())['complete']
    entry['summary_sha256']=sha(summary)
record.update(complete=True,finished_utc=now(),
              resumed_outer_children_wall_seconds=sum(s['outer_child_wall_seconds'] for s in record['stages']))
put(OUT/'pipeline_resume_status.json',record)
print(json.dumps(record),flush=True)

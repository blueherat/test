"""Execute the three predetermined stages once; record whole child wall times."""
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

record=dict(pid=os.getpid(),created_utc=now(),complete=False,
            driver_sha256=sha(__file__),stages=[])
with (OUT/'pipeline_created.json').open('x') as f: json.dump(record,f,indent=2)
pilot=json.loads((OUT/'pilot/request.json').read_text())
supplement=json.loads((OUT/'supplemental_source_environment.json').read_text())
for mode in ('train','validate','rollout'):
    for name,item in pilot['sources'].items():
        assert sha(ROOT/name)==item['sha256'],name
    for item in supplement['sources']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    prior=OUT/('pilot' if mode=='train' else 'train')
    cmd=[PYTHON,'-u',str(RUNNER),'--mode',mode,'--output-dir',str(OUT/mode),'--prior-dir',str(prior)]
    assert not (OUT/mode).exists(),f'refuse repeated stage {mode}'
    entry=dict(mode=mode,argv=cmd,started_utc=now(),complete=False)
    record['stages'].append(entry)
    with (OUT/(mode+'.log')).open('x') as log:
        started=time.perf_counter()
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid
        put(OUT/'pipeline_status.json',record)
        print(json.dumps(entry),flush=True)
        code=child.wait()
        entry.update(returncode=code,outer_child_wall_seconds=time.perf_counter()-started,
                     finished_utc=now(),complete=True)
    put(OUT/'pipeline_status.json',record)
    print(json.dumps(entry),flush=True)
    if code:
        record['terminal_failure']=mode
        put(OUT/'pipeline_status.json',record)
        sys.exit(code)
    summary=OUT/mode/'summary.json'
    assert json.loads(summary.read_text())['complete']
    entry['summary_sha256']=sha(summary)
record.update(complete=True,finished_utc=now(),
              outer_children_wall_seconds=sum(s['outer_child_wall_seconds'] for s in record['stages']))
put(OUT/'pipeline_status.json',record)
print(json.dumps(record),flush=True)

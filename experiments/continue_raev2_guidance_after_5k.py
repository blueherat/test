"""Wait for the existing 5K process, then run the already frozen critic screen.

Never restarts the 5K study. A >=3% weak-guidance result is handed back for
cost validation instead of spending GPUs on another candidate unnecessarily.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')


def process_identity(pid):
    try:
        # Linux start time identifies this process despite future PID reuse.
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[19]
    except FileNotFoundError:
        return None


def main():
    out=DATA/'after_5k_continuation'
    out.mkdir(exist_ok=False)
    path=DATA/'weak_confirm5k/execution.json'
    initial=json.loads(path.read_text())
    pids=[initial['pid'],*[j['pid'] for j in initial['jobs']]]
    identities={pid:process_identity(pid) for pid in pids}
    if not any(identities.values()) and not initial.get('complete'):
        raise RuntimeError('prerequisite is stopped/incomplete; inspect it, do not restart')
    frozen_files=[ROOT/'experiments'/name for name in ['sample_raev2_ancestral_guidance.py',
        'run_raev2_ancestral_study.py','raev2_ancestral_guidance.py','raev2_transport_projection.py',
        'raev2_stochastic_weak.py','raev2_image_critic_guidance.py']]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    state={'pid':os.getpid(),'stage':'waiting_for_live_5k','complete':False,'source_hashes':sources,
        'prerequisite_process_identities':identities,'created_unix':time.time(),
        'next_screen':{'seed':202609071,'samples':1000,'modes':['critic_isotropic','critic_exchangeable']},
        'criterion':'stochastic_weak <= .97 * min(official,piecewise); then cost validation remains required'}
    def save():
        temp=out/'state.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(out/'state.json')
    save()
    while any(identity is not None and process_identity(pid)==identity for pid,identity in identities.items()):
        time.sleep(5)
    final=json.loads(path.read_text())
    if not final.get('complete'):
        state.update(stage='prerequisite_failed_requires_inspection',complete=True)
        save()
        raise RuntimeError('5K process stopped without complete evaluation; no automatic resampling')
    metrics={r['branch']:r for r in json.loads((path.parent/'metrics.json').read_text())}
    best=min(metrics[m]['fid'] for m in ['official','piecewise'])
    gain=1-metrics['stochastic_weak']['fid']/best
    state['weak_5k_improvement_vs_best_control']=gain
    state['weak_5k_metrics']=metrics
    if gain>=.03:
        state.update(stage='quality_gate_passed_cost_validation_required',complete=True)
        save()
        return
    for p,sha in sources.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('pending candidate source changed; explicit review required: '+p)
    command=[sys.executable,str(ROOT/'experiments/run_raev2_ancestral_study.py'),'--output',str(DATA/'critic_screen1k'),
        '--seed','202609071','--samples','1000','--modes','critic_isotropic','critic_exchangeable']
    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1'}
    with (out/'critic_screen.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(stage='running_frozen_critic_screen',child_pid=child.pid,command=command)
        save()
        code=child.wait()
    state.update(stage='critic_screen_complete' if code==0 else 'critic_screen_failed_requires_inspection',
        complete=True,exit_code=code)
    save()
    if code:raise RuntimeError('critic screen failed; inspect original outputs')


if __name__=='__main__':main()

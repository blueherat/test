"""After the current critic study, integrate and test the frozen two-mode arm.

The exact patch is prepared and reviewed before launch. Active sampling source
files stay unchanged until the current study has completed its source audit.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.continue_raev2_guidance_after_5k import process_identity
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=DATA/'after_critic_continuation'
    out.mkdir(exist_ok=False)
    prereq=DATA/'critic_screen1k/execution.json'
    initial=json.loads(prereq.read_text())
    ids={pid:process_identity(pid) for pid in [initial['pid'],*[j['pid'] for j in initial['jobs']]]}
    if not any(ids.values()) and not initial.get('complete'):
        raise RuntimeError('critic prerequisite is stopped/incomplete')
    lock=ROOT/'experiments/locks/raev2_two_mode_20260907'
    expected=json.loads((lock/'manifest.json').read_text())
    fixed=[lock/'integrate.patch',lock/'manifest.json',ROOT/'experiments/raev2_two_mode_ratio.py',
           DATA/'two_mode_ratio/finite_euler_calibration.json']
    frozen={str(p):digest(p) for p in fixed}
    state={'pid':os.getpid(),'stage':'waiting_for_live_critic','complete':False,'prerequisite_process_identities':ids,
        'frozen_inputs':frozen,'patch_manifest':expected,'next_arm':{'mode':'two_mode','samples':1000,'seed':202609071},
        'criterion':'a critic 1K gain >=3% versus both fixed 1K controls takes priority for independent confirmation'}
    def save():
        temp=out/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'state.json')
    save()
    while any(v is not None and process_identity(p)==v for p,v in ids.items()):time.sleep(5)
    if not json.loads(prereq.read_text()).get('complete'):
        state.update(stage='prerequisite_failed_requires_inspection',complete=True);save()
        raise RuntimeError('critic study failed; no restart or source modification')
    metrics=json.loads((prereq.parent/'metrics.json').read_text())
    controls=json.loads((DATA/'ancestral_screen1k/metrics.json').read_text())
    base=min(r['fid'] for r in controls if r['branch'] in ['official','piecewise'])
    state['critic_metrics']=metrics
    state['critic_best_improvement']=1-min(r['fid'] for r in metrics)/base
    if state['critic_best_improvement']>=.03:
        state.update(stage='critic_quality_signal_requires_independent_confirmation',complete=True);save();return
    for p,h in frozen.items():
        if digest(Path(p))!=h:raise RuntimeError('frozen two-mode input changed: '+p)
    for p,h in expected.items():
        if digest(ROOT/p)!=h['old_sha256']:raise RuntimeError('integration source mismatch: '+p)
    subprocess.run(['git','apply','--check',str(lock/'integrate.patch')],cwd=ROOT,check=True)
    subprocess.run(['git','apply',str(lock/'integrate.patch')],cwd=ROOT,check=True)
    for p,h in expected.items():
        if digest(ROOT/p)!=h['new_sha256']:raise RuntimeError('integrated source mismatch: '+p)
    state.update(stage='integrated_running_smoke');save()
    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1'}
    smoke=DATA/'two_mode_smoke'
    command=[sys.executable,str(ROOT/'experiments/sample_raev2_ancestral_guidance.py'),'--output',str(smoke),
        '--samples','8','--seed','202609071','--modes','official','two_mode','--parity']
    with (out/'smoke.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    with np.load(smoke/'shard0/official/samples.npz') as a,np.load(DATA/'ancestral_smoke_v2/shard0/official/samples.npz') as b:
        if not np.array_equal(a['arr_0'],b['arr_0']):raise RuntimeError('original official pixel parity failed')
    state.update(stage='running_two_mode_1k',official_smoke_pixel_parity=True);save()
    command=[sys.executable,str(ROOT/'experiments/run_raev2_ancestral_study.py'),'--output',str(DATA/'two_mode_screen1k'),
        '--samples','1000','--seed','202609071','--modes','two_mode']
    with (out/'screen.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(child_pid=child.pid,command=command);save();code=child.wait()
    state.update(stage='two_mode_1k_complete' if code==0 else 'two_mode_1k_failed_requires_inspection',complete=True,exit_code=code);save()
    if code:raise RuntimeError('two-mode screen failed; inspect original artifacts')


if __name__=='__main__':main()

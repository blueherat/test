"""Evaluate completed primary branches without occupying sampling GPUs."""
import argparse
import json
import os
import subprocess
import time
from . import common as c


def evaluate(model,stage,arm):
    root=c.ROOT/model/stage/arm
    summary=c.read(root/'summary.json');assert summary['complete'] and c.sha(root/'samples.npz')==summary['samples_sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1')
    output=root/'metrics.json'
    if output.exists():return c.read(output)
    if model=='sit_small':
        from experiments.small_sit_carrier_flow_20260909 import REFERENCE
        cmd=['/data/shared/envs/adm-fid/bin/python',str(c.WORK/'experiments/compute_adm_fid.py'),
            '--reference',str(REFERENCE),'--samples',str(root/'samples.npz'),'--output',str(root/'adm.json'),
            '--activations-output',str(root/'activations.npz'),'--batch-size','32']
    else:
        cmd=[c.PYTHON,str(c.WORK/'experiments/evaluate_raev2_official_samples.py'),
            '--branch',arm+'='+str(root/'samples.npz'),'--output',str(root/'official.csv'),
            '--batch-size','32','--device','cpu','--feature-cache-dir',str(root/'features')]
    with (root/'evaluation.log').open('a') as log:subprocess.run(cmd,env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
    row=c.read(root/'adm.json') if model=='sit_small' else c.read(root/'official.json')[0]
    result=dict(summary,fid=row['fid'],inception_score=row['inception_score'],metrics=row)
    c.atomic(output,result);print(model,stage,arm,result['fid'],flush=True)
    return result


def watch(model,stage,arms):
    done={}
    while len(done)<len(arms):
        for arm in arms:
            if arm not in done and (c.ROOT/model/stage/arm/'summary.json').exists():
                done[arm]=evaluate(model,stage,arm)
                c.atomic(c.ROOT/model/stage/'results.json',[done[a] for a in arms if a in done])
        if len(done)<len(arms):time.sleep(5)
    c.atomic(c.ROOT/model/stage/'evaluation_complete.json',dict(complete=True,arms=len(done)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--stage',default='cfg_screen_400');a=p.parse_args()
    if a.stage.startswith('cfg_'):
        from .cfg import configs
        arms=[s['arm'] for s in configs()]
    elif a.stage.startswith('ig_calibrated_'):
        from .calibrate import CONFIGS
        arms=[s['arm'] for s in CONFIGS]
    else:
        from .ig import CONFIGS
        arms=[s['arm'] for s in CONFIGS]
    watch(a.model,a.stage,arms)

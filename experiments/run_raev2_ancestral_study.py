#!/usr/bin/env python3
"""Four-GPU batch partition, strict merge, and official FID evaluation."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--seed',type=int,default=202609071)
    p.add_argument('--modes',nargs='+',default=['official','piecewise','ancestral','partial'])
    p.add_argument('--steps',type=int,default=100)
    a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
    jobs=[]
    began=time.perf_counter()
    frozen={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (
        ROOT/'experiments/sample_raev2_ancestral_guidance.py',ROOT/'experiments/raev2_ancestral_guidance.py',ROOT/'experiments/raev2_transport_projection.py',ROOT/'experiments/raev2_stochastic_weak.py',ROOT/'experiments/raev2_image_critic_guidance.py',ROOT/'experiments/raev2_two_mode_ratio.py',ROOT/'experiments/raev2_semantic_complement.py',ROOT/'experiments/raev2_semantic_quality_guidance.py',ROOT/'experiments/raev2_paired_ratio_model.py',Path(__file__).resolve())}
    state={'complete':False,'pid':os.getpid(),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'sources':frozen,'jobs':[]}
    def save():
        (out/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    (out/'frozen_source').mkdir()
    for source in frozen:
        shutil.copy2(source,out/'frozen_source'/Path(source).name)
    for rank in range(4):
        command=[sys.executable,str(ROOT/'experiments/sample_raev2_ancestral_guidance.py'),'--output',str(out),
            '--samples',str(a.samples),'--seed',str(a.seed),'--shard',str(rank),'--shards','4','--steps',str(a.steps),'--parity','--modes',*a.modes]
        log=open(out/f'shard{rank}.log','w')
        child=subprocess.Popen(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(rank)},stdout=log,stderr=subprocess.STDOUT)
        jobs.append((child,log))
        state['jobs'].append({'rank':rank,'pid':child.pid,'command':command})
        save()
    while any(child.poll() is None for child,_ in jobs):
        failed=[child.returncode for child,_ in jobs if child.poll() not in (None,0)]
        if failed:
            for child,_ in jobs:
                if child.poll() is None: child.terminate()
            raise RuntimeError(f'sampling worker failed: {failed}')
        time.sleep(2)
    for child,log in jobs:
        log.close()
        if child.returncode: raise RuntimeError(child.returncode)
    for path,sha in frozen.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('source changed during execution: '+path)
    common=None
    all_results={}
    for mode in a.modes:
        images=[]; ids=[]; records=[]; summaries=[]
        for rank in range(4):
            folder=out/f'shard{rank}'/mode
            summary=json.loads((folder/'summary.json').read_text())
            if not summary['complete']: raise RuntimeError('incomplete samples')
            data=np.load(folder/'samples.npz')
            images.append(data['arr_0']); ids.extend(data['ids'].tolist())
            records.extend(summary['initial_noise']); summaries.append(summary)
        ids=np.array(ids)
        order=np.argsort(ids)
        if not np.array_equal(ids[order],np.arange(a.samples)): raise RuntimeError('sample id coverage')
        records=sorted(records,key=lambda x:x['batch'])
        if common is not None and common!=records: raise RuntimeError('unpaired noise/labels')
        common=records
        folder=out/mode
        folder.mkdir()
        np.savez(folder/'samples.npz',np.concatenate(images)[order])
        result={'complete':True,'mode':mode,'count':a.samples,'seed':a.seed,'steps':a.steps,
            'trajectory_seconds_sum':sum(s['trajectory_seconds'] for s in summaries),
            'decode_seconds_sum':sum(s['decode_seconds'] for s in summaries),
            'sample_model_calls':sum(s['sample_model_calls'] for s in summaries),
            'extra_sample_weak_continuations':sum(s.get('extra_sample_weak_continuations',0) for s in summaries),
            'sample_critic_backward_calls':sum(s.get('sample_critic_backward_calls',0) for s in summaries),
            'sample_ratio_backward_calls':sum(s.get('sample_ratio_backward_calls',0) for s in summaries),
            'extra_sample_unconditional_calls':sum(s.get('extra_sample_unconditional_calls',0) for s in summaries),
            'paired_noise_labels_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'sample_sha256':hashlib.sha256((folder/'samples.npz').read_bytes()).hexdigest()}
        (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        all_results[mode]=result
    state['sampling_seconds']=time.perf_counter()-began
    state['sampling_complete']=True
    state['merged']=all_results
    save()
    command=[sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
        '--output',str(out/'metrics.csv'),'--batch-size','64']
    for mode in a.modes:
        command+=['--branch',f'{mode}={out/mode/"samples.npz"}']
    with open(out/'evaluation.log','w') as log:
        subprocess.run(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    state['complete']=True
    state['total_seconds']=time.perf_counter()-began
    save()
    print((out/'metrics.csv').read_text(),flush=True)


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Launch fixed paired quadrature conditions, verify coverage, and evaluate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=202609072)
    p.add_argument('--modes', nargs='+', default=['guidance_2m', 'full_2m', 'exponential_2m'])
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--no-evaluate', action='store_true')
    a = p.parse_args()
    if a.samples % 8 or a.samples // 8 < len(a.gpus):
        p.error('requires nonempty B8 shards')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
           'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1'}
    sources = [Path(__file__).resolve(), ROOT/'experiments/sample_raev2_guidance_quadrature.py',
        ROOT/'experiments/raev2_guidance_quadrature.py', ROOT/'experiments/sample_raev2_pfr_retiming.py',
        ROOT/'experiments/raev2_stage1_compat.py', ROOT/'experiments/evaluate_raev2_official_samples.py',
        ROOT/'external/RAEv2/src/stage2/models/DDT.py', ROOT/'external/RAEv2/src/stage2/models/model_utils.py',
        ROOT/'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml']
    frozen = {str(path):sha(path) for path in sources}
    for path in sources:
        dest = out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    state = {'complete':False, 'pid':os.getpid(), 'pid_starttime':Path('/proc/self/stat').read_text().split()[21],
             'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}, 'sources':frozen, 'jobs':[]}
    def save():
        temp = out/'execution.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(out/'execution.json')
    save()
    start = time.perf_counter()
    jobs = []
    try:
        for rank, gpu in enumerate(a.gpus):
            command = [sys.executable, str(ROOT/'experiments/sample_raev2_guidance_quadrature.py'),
                       '--output', str(out), '--samples', str(a.samples), '--seed',str(a.seed),
                       '--shard',str(rank),'--shards',str(len(a.gpus)), '--modes',*a.modes]
            log = (out/f'shard{rank}.log').open('w')
            child = subprocess.Popen(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT)
            jobs.append((child,log))
            state['jobs'].append({'rank':rank, 'gpu':gpu, 'pid':child.pid,
                'pid_starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21], 'command':command})
            save()
        while any(child.poll() is None for child,_ in jobs):
            if any(child.poll() not in (None,0) for child,_ in jobs):
                raise RuntimeError('sampling worker failed; inspect shard logs')
            time.sleep(2)
        if any(child.returncode for child,_ in jobs):
            raise RuntimeError('sampling worker failed')
        if any(sha(path) != value for path,value in frozen.items()):
            raise RuntimeError('source changed during sampling')
        common, results = None, {}
        for mode in a.modes:
            arrays, ids, records, summaries = [], [], [], []
            for rank in range(len(a.gpus)):
                folder = out/f'shard{rank}'/mode
                summary = json.loads((folder/'summary.json').read_text())
                if not summary['complete'] or sha(folder/'samples.npz') != summary['sample_sha256']:
                    raise RuntimeError('incomplete or changed shard')
                with np.load(folder/'samples.npz') as data:
                    arrays.append(data['arr_0']); ids.extend(data['ids'].tolist())
                records.extend(summary['initial_noise']); summaries.append(summary)
            ids = np.asarray(ids)
            order = np.argsort(ids)
            if not np.array_equal(ids[order], np.arange(a.samples)):
                raise RuntimeError('sample coverage or duplicates')
            records = sorted(records,key=lambda r:r['batch'])
            if common is not None and common != records:
                raise RuntimeError('noise or label pairing differs')
            common = records
            dest = out/mode
            dest.mkdir()
            np.savez(dest/'samples.npz', np.concatenate(arrays)[order])
            result = {'complete':True,'samples':a.samples,'mode':mode,'seed':a.seed,
                'sample_sha256':sha(dest/'samples.npz'),
                'trajectory_seconds_sum':sum(s['trajectory_seconds'] for s in summaries),
                'decode_seconds_sum':sum(s['decode_seconds'] for s in summaries),
                'sample_model_calls':sum(s['sample_model_calls'] for s in summaries),
                'paired_noise_labels_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
            (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            results[mode] = result
        state.update(sampling_complete=True, sampling_seconds=time.perf_counter()-start, merged=results)
        save()
        if not a.no_evaluate:
            command = [sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                       '--output',str(out/'metrics.csv'),'--batch-size','64']
            for mode in a.modes:
                command += ['--branch', f'{mode}={out/mode/"samples.npz"}']
            with (out/'evaluation.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(a.gpus[0])},stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(complete=True,total_seconds=time.perf_counter()-start)
        save()
        print(json.dumps(state['merged'],indent=2),flush=True)
        if not a.no_evaluate:
            print((out/'metrics.csv').read_text(),flush=True)
    except BaseException as error:
        state.update(error=repr(error), total_seconds=time.perf_counter()-start)
        save()
        for child,_ in jobs:
            if child.poll() is None:
                child.terminate()
        raise
    finally:
        for child,log in jobs:
            child.wait()
            log.close()


if __name__ == '__main__':
    main()

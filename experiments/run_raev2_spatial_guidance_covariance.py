"""Launch fixed spatial-covariance arms, verify pairing, and run official FID."""
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
sys.path.insert(0, str(ROOT))
from experiments.run_raev2_guidance_quadrature import sha
from experiments.sample_raev2_spatial_guidance_covariance import FIT, MODES, DATA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit', type=Path, default=FIT)
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=202609072)
    p.add_argument('--modes', nargs='+', choices=MODES, default=['spatial_full', 'spatial_diagonal'])
    p.add_argument('--gpus', nargs='+', type=int, default=[0,1,2,3])
    p.add_argument('--no-evaluate', action='store_true')
    a = p.parse_args()
    if a.samples % 8 or a.samples//8 < len(a.gpus):
        p.error('nonempty complete B8 shards required')
    if 'unit' in a.modes and (a.samples != 8 or len(a.gpus) != 1):
        p.error('unit covariance is an 8-image parity check only')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
           'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1'}
    sources = [Path(__file__).resolve(), ROOT/'experiments/sample_raev2_spatial_guidance_covariance.py',
        ROOT/'experiments/raev2_spatial_guidance_covariance.py', ROOT/'experiments/raev2_directional_variance.py',
        ROOT/'experiments/raev2_conditional_variance.py', ROOT/'experiments/sample_raev2_guidance_quadrature.py',
        ROOT/'experiments/raev2_guidance_quadrature.py', ROOT/'experiments/run_raev2_guidance_quadrature.py',
        ROOT/'experiments/sample_raev2_pfr_retiming.py', ROOT/'experiments/raev2_stage1_compat.py',
        ROOT/'experiments/evaluate_raev2_official_samples.py', ROOT/'external/RAEv2/src/stage2/models/DDT.py',
        ROOT/'external/RAEv2/src/stage2/models/model_utils.py',
        ROOT/'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml']
    frozen = {str(path):sha(path) for path in sources}
    calibration_files = [a.fit/'calibration.json', a.fit/'covariance.npz', DATA/'conditional_variance_fit/head.pt',
                         DATA/'directional_variance_moments/calibration.json', DATA/'guided_reverse_variance/calibration.json']
    calibration_hashes = {str(path.resolve()):sha(path) for path in calibration_files}
    for path in sources:
        dest = out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for index, path in enumerate(calibration_files):
        dest = out/'frozen_calibration'/f'{index}_{path.name}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    state = {'complete':False, 'pid':os.getpid(), 'pid_starttime':Path('/proc/self/stat').read_text().split()[21],
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'sources':frozen, 'calibration_sources':calibration_hashes, 'jobs':[]}
    def save():
        temporary = out/'execution.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(out/'execution.json')
    save()
    began = time.perf_counter()
    jobs = []
    try:
        for rank, gpu in enumerate(a.gpus):
            command = [sys.executable, str(ROOT/'experiments/sample_raev2_spatial_guidance_covariance.py'),
                '--output',str(out),'--fit',str(a.fit.resolve()),'--samples',str(a.samples),'--seed',str(a.seed),
                '--shard',str(rank),'--shards',str(len(a.gpus)),'--modes',*a.modes]
            log = (out/f'shard{rank}.log').open('w')
            child = subprocess.Popen(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},
                                     stdout=log,stderr=subprocess.STDOUT)
            jobs.append((child,log))
            state['jobs'].append({'rank':rank,'gpu':gpu,'pid':child.pid,
                'pid_starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':command})
            save()
        while any(child.poll() is None for child,_ in jobs):
            if any(child.poll() not in (None,0) for child,_ in jobs):
                raise RuntimeError('sampling worker failed; inspect shard logs')
            time.sleep(2)
        assert all(child.returncode == 0 for child,_ in jobs)
        assert all(sha(path) == value for path,value in frozen.items()), 'source changed during sampling'
        assert all(sha(path) == value for path,value in calibration_hashes.items()), 'calibration changed during sampling'
        common, results = None, {}
        for mode in a.modes:
            arrays, ids, records, summaries = [], [], [], []
            for rank in range(len(a.gpus)):
                folder = out/f'shard{rank}'/mode
                summary = json.loads((folder/'summary.json').read_text())
                assert summary['complete'] and sha(folder/'samples.npz') == summary['sample_sha256']
                with np.load(folder/'samples.npz') as data:
                    arrays.append(data['arr_0']); ids.extend(data['ids'].tolist())
                records.extend(summary['initial_noise']); summaries.append(summary)
            ids = np.asarray(ids)
            order = np.argsort(ids)
            assert np.array_equal(ids[order],np.arange(a.samples)), 'sample coverage'
            records.sort(key=lambda r:r['batch'])
            assert common is None or common == records, 'paired inputs differ'
            common = records
            dest = out/mode
            dest.mkdir()
            np.savez(dest/'samples.npz',np.concatenate(arrays)[order])
            result = {'complete':True,'samples':a.samples,'mode':mode,'seed':a.seed,
                'sample_sha256':sha(dest/'samples.npz'),
                'trajectory_seconds_sum':sum(s['trajectory_seconds'] for s in summaries),
                'decode_seconds_sum':sum(s['decode_seconds'] for s in summaries),
                'sample_model_calls':sum(s['sample_model_calls'] for s in summaries),
                'inactive_image_queries':sum(s['inactive_image_queries'] for s in summaries),
                'paired_noise_labels_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
            (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            results[mode] = result
        state.update(sampling_complete=True,sampling_seconds=time.perf_counter()-began,merged=results)
        save()
        if not a.no_evaluate:
            command = [sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                       '--output',str(out/'metrics.csv'),'--batch-size','64']
            for mode in a.modes:
                command += ['--branch',f'{mode}={out/mode/"samples.npz"}']
            with (out/'evaluation.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(a.gpus[0])},
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(complete=True,seconds=time.perf_counter()-began)
        save()
        print(json.dumps(results,indent=2),flush=True)
        if not a.no_evaluate:
            print((out/'metrics.csv').read_text(),flush=True)
    except BaseException as error:
        state.update(error=repr(error),seconds=time.perf_counter()-began)
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

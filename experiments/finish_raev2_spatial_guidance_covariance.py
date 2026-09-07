"""Finish existing orphaned sampling workers without repeating model queries."""
from __future__ import annotations
import argparse
import fcntl
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


def live(pid, starttime):
    path = Path(f'/proc/{pid}/stat')
    try:
        fields = path.read_text().split()
        return fields[21] == starttime and fields[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--audit-output', type=Path, required=True)
    args = p.parse_args()
    out = args.folder.resolve()
    lock = (out/'finishing.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((out/'execution.json').read_text())
    assert not state['complete'] and not live(state['pid'], state['pid_starttime']), 'original controller still live or already complete'
    backup = out/f"execution_before_recovery_{state['pid']}.json"
    assert not backup.exists(), 'this controller recovery was already attempted'
    shutil.copy2(out/'execution.json', backup)
    previous = {k:state[k] for k in ['pid','pid_starttime']}
    state.update(pid=os.getpid(), pid_starttime=Path('/proc/self/stat').read_text().split()[21],
        recovery={'previous_controller':previous, 'reason':'controller PID absent; existing workers retained',
                  'new_sampling_calls':0, 'source_sha256':sha(Path(__file__).resolve()),
                  'original_execution_sha256':sha(backup)})
    for source in [Path(__file__).resolve(), ROOT/'experiments/audit_raev2_spatial_guidance_covariance.py']:
        state['sources'][str(source)] = sha(source)
        dest = out/'frozen_source'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,dest)
    def save():
        temp = out/'execution.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(out/'execution.json')
    save()
    began = time.perf_counter()
    modes = state['args']['modes']
    count = state['args']['samples']
    env = {**os.environ, 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
           'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1'}
    try:
        while True:
            all_done = True
            for job in state['jobs']:
                done = all((out/f"shard{job['rank']}"/mode/'summary.json').exists() for mode in modes)
                if not done:
                    all_done = False
                    assert live(job['pid'],job['pid_starttime']), f"worker {job['rank']} exited before completing its samples"
            if all_done:
                break
            time.sleep(5)
        for source,expected in state['sources'].items():
            assert sha(Path(source)) == expected, 'source changed during sampling/recovery'
        for source,expected in state['calibration_sources'].items():
            assert sha(Path(source)) == expected, 'calibration changed during sampling'
        common, results = None, {}
        for mode in modes:
            arrays, ids, records, summaries = [], [], [], []
            for job in state['jobs']:
                folder = out/f"shard{job['rank']}"/mode
                summary = json.loads((folder/'summary.json').read_text())
                assert summary['complete'] and sha(folder/'samples.npz') == summary['sample_sha256']
                with np.load(folder/'samples.npz') as data:
                    arrays.append(data['arr_0']); ids.extend(data['ids'].tolist())
                records.extend(summary['initial_noise']); summaries.append(summary)
            ids = np.asarray(ids)
            order = np.argsort(ids)
            assert np.array_equal(ids[order],np.arange(count))
            records.sort(key=lambda r:r['batch'])
            assert common is None or common == records
            common = records
            dest = out/mode
            assert not dest.exists(), 'merged output already exists; inspect before replacing'
            dest.mkdir()
            np.savez(dest/'samples.npz',np.concatenate(arrays)[order])
            result = {'complete':True,'samples':count,'mode':mode,'seed':state['args']['seed'],
                'sample_sha256':sha(dest/'samples.npz'),
                'trajectory_seconds_sum':sum(s['trajectory_seconds'] for s in summaries),
                'decode_seconds_sum':sum(s['decode_seconds'] for s in summaries),
                'sample_model_calls':sum(s['sample_model_calls'] for s in summaries),
                'inactive_image_queries':sum(s['inactive_image_queries'] for s in summaries),
                'paired_noise_labels_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
            (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            results[mode] = result
        state.update(sampling_complete=True,merged=results)
        save()
        command = [sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                   '--output',str(out/'metrics.csv'),'--batch-size','64']
        for mode in modes:
            command += ['--branch',f'{mode}={out/mode/"samples.npz"}']
        with (out/'evaluation.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(state['args']['gpus'][0])},
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(complete=True,recovery_seconds=time.perf_counter()-began)
        save()
        subprocess.run([sys.executable,str(ROOT/'experiments/audit_raev2_spatial_guidance_covariance.py'),
                        '--folder',str(out),'--output',str(args.audit_output.resolve())],cwd=ROOT,env=env,check=True)
        print((out/'metrics.csv').read_text(),flush=True)
        (out/'recovery_completion.json').write_text(json.dumps({'complete':True,'audit_complete':True,
            'execution_sha256':sha(out/'execution.json'),'audit_sha256':sha(args.audit_output),
            'no_sampling_repeated':True,'seconds':time.perf_counter()-began},indent=2)+'\n')
    except BaseException as error:
        state['recovery']['error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()

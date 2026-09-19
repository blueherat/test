"""Four GPUs cooperate on each PFR arm, then evaluate before advancing."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.sample_official_sit_pfr_symmetric_20260909 import ARMS,DATA,array_sha,write_json
from experiments.raev2_training_core import file_sha256


def wait_files(paths, workers):
    while not all(p.is_file() for p in paths):
        for worker in workers:
            if worker.poll() is not None:
                raise RuntimeError(f'Worker exited before barrier: pid={worker.pid}, code={worker.returncode}')
        time.sleep(1)


def main():
    script = ROOT/'experiments/sample_official_sit_pfr_symmetric_20260909.py'
    env = dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    subprocess.run([sys.executable,str(script),'--prepare'],env=env,check=True)
    request = json.loads((DATA/'request.json').read_text())
    workers, logs, results = [], [], []
    began = time.perf_counter()
    try:
        for rank in range(4):
            log = (DATA/f'worker{rank}.log').open('x'); logs.append(log)
            workers.append(subprocess.Popen([sys.executable,str(script),'--rank',str(rank)],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
        write_json(DATA/'status.json',dict(phase='parity',pids=[w.pid for w in workers]))
        paths = [DATA/f'parity_rank{r}.json' for r in range(4)]
        wait_files(paths,workers)
        parity = [json.loads(p.read_text()) for p in paths]
        assert all(x['passed'] and x['prefix_parity'] for x in parity)
        assert sorted(i for x in parity for i in x['indices']) == list(range(16))
        write_json(DATA/'parity_passed.json',dict(passed=True,images=16,records=parity))
        print('All 16 historical time-only PFR pixels and Base-prefix outputs reproduced.',flush=True)
        with np.load(DATA/'inputs.npz') as bank: noise,labels = bank['noise'],bank['labels']
        for arm, (mode,scale) in ARMS.items():
            write_json(DATA/'status.json',dict(phase='sampling',arm=arm,pids=[w.pid for w in workers]))
            directory = DATA/arm
            wait_files([directory/f'rank{r}.json' for r in range(4)],workers)
            images = np.empty((1000,256,256,3),dtype=np.uint8)
            latents = np.empty((1000,4,32,32),dtype=np.float32)
            diagnostics = np.empty((1000,5,4),dtype=np.float32)
            covered = np.zeros(1000,dtype=np.int64)
            summaries = []
            for rank in range(4):
                summary = json.loads((directory/f'rank{rank}.json').read_text())
                path = directory/f'rank{rank}.npz'
                assert file_sha256(path) == summary['pixel_file_sha256']
                assert summary['request_sha256'] == file_sha256(DATA/'request.json')
                assert summary['mode'] == mode and summary['scale'] == scale
                with np.load(path) as shard:
                    idx = shard['indices']
                    assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                    assert array_sha(noise[idx]) == summary['noise_sha256']
                    assert array_sha(labels[idx]) == summary['label_sha256']
                    assert shard['arr_0'].dtype == np.uint8
                    assert np.isfinite(shard['latents']).all() and np.isfinite(shard['diagnostics']).all()
                    images[idx],latents[idx],diagnostics[idx] = shard['arr_0'],shard['latents'],shard['diagnostics']
                    covered[idx] += 1
                summaries.append(summary)
            assert np.all(covered == 1)
            assert sum(s['full_sample_calls'] for s in summaries) == 100000
            assert sum(s['prefix_sample_calls'] for s in summaries) == 50000
            assert sum(s['probe_batch_calls'] for s in summaries) == 0
            np.savez(directory/'samples.npz',arr_0=images)
            np.savez(directory/'diagnostics.npz',latents=latents,diagnostics=diagnostics,labels=labels)
            write_json(DATA/'status.json',dict(phase='evaluation',arm=arm,pids=[w.pid for w in workers]))
            cmd = [sys.executable,str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                '--branch',arm+'='+str(directory/'samples.npz'),'--output',str(directory/'fid.csv'),
                '--batch-size','32','--device','cuda','--feature-cache-dir',str(directory/'features')]
            with (directory/'evaluation.log').open('x') as log:
                subprocess.run(cmd,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics = json.loads((directory/'fid.json').read_text())[0]
            assert metrics['sample_sha256'] == file_sha256(directory/'samples.npz')
            row = dict(arm=arm,method=mode,scale=scale,metrics=metrics,full_per_image=100,prefix_per_image=50,
                gpu_inference_seconds_sum=sum(s['inference_seconds'] for s in summaries),
                inference_seconds_max=max(s['inference_seconds'] for s in summaries),
                diagnostic_sha256=file_sha256(directory/'diagnostics.npz'))
            write_json(directory/'result.json',row); results.append(row)
            write_json(DATA/'results.json',results)
            print(json.dumps(row),flush=True)
            write_json(directory/'advance.json',dict(complete=True))
        for worker in workers:
            if worker.wait() != 0: raise RuntimeError('Worker exit failure')
        for path,digest in request['sources'].items(): assert file_sha256(Path(path)) == digest,path
        write_json(DATA/'status.json',dict(phase='complete',seconds=time.perf_counter()-began,
            research_goal_achieved=False,results=results))
    except BaseException as error:
        write_json(DATA/'status.json',dict(phase='failed',error=repr(error)))
        raise
    finally:
        for worker in workers:
            if worker.poll() is None: worker.terminate()
        for worker in workers: worker.wait()
        for log in logs: log.close()


if __name__ == '__main__': main()

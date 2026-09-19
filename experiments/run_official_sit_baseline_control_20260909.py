"""Run four fixed baseline controls on four GPUs, checking historical parity first."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sample_official_sit_baseline_control_20260909 import ARMS, DATA, array_sha, write_json
from experiments.raev2_training_core import file_sha256


def wait_files(paths, workers):
    while not all(p.is_file() for p in paths):
        for worker in workers:
            if worker.poll() is not None:
                raise RuntimeError(f'Worker exited before barrier, pid={worker.pid}, returncode={worker.returncode}')
        time.sleep(1)


def main():
    worker_path = ROOT / 'experiments/sample_official_sit_baseline_control_20260909.py'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([sys.executable, str(worker_path), '--prepare'], env=env, check=True)
    request = json.loads((DATA / 'request.json').read_text())
    workers, logs = [], []
    started = time.perf_counter()
    try:
        for rank in range(4):
            log = (DATA / f'worker{rank}.log').open('x')
            logs.append(log)
            worker = subprocess.Popen([sys.executable, str(worker_path), '--rank', str(rank)],
                env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)), stdout=log, stderr=subprocess.STDOUT)
            workers.append(worker)
        write_json(DATA / 'status.json', dict(phase='parity', pids=[w.pid for w in workers]))
        paths = [DATA / f'parity_rank{r}.json' for r in range(4)]
        wait_files(paths, workers)
        parity = [json.loads(p.read_text()) for p in paths]
        assert all(x['passed'] for x in parity)
        assert sorted(i for x in parity for i in x['indices']) == list(range(16))
        write_json(DATA / 'parity_passed.json', dict(passed=True, images=16, records=parity))
        print('All 16 historical pixels reproduced exactly on four GPUs.', flush=True)
        results = []
        for arm in ARMS:
            write_json(DATA / 'status.json', dict(phase='sampling', arm=arm, pids=[w.pid for w in workers]))
            arm_dir = DATA / arm
            wait_files([arm_dir / f'rank{r}.json' for r in range(4)], workers)
            images = np.empty((1000, 256, 256, 3), dtype=np.uint8)
            covered = np.zeros(1000, dtype=np.int64)
            summaries = []
            with np.load(DATA / 'inputs.npz') as bank:
                noise, labels = bank['noise'], bank['labels']
            for rank in range(4):
                summary = json.loads((arm_dir / f'rank{rank}.json').read_text())
                path = arm_dir / f'rank{rank}.npz'
                assert file_sha256(path) == summary['pixel_file_sha256']
                assert summary['request_sha256'] == file_sha256(DATA / 'request.json')
                with np.load(path) as shard:
                    indices = shard['indices']
                    assert np.array_equal(indices, np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                    assert array_sha(noise[indices]) == summary['noise_sha256']
                    assert array_sha(labels[indices]) == summary['label_sha256']
                    images[indices] = shard['arr_0']
                    covered[indices] += 1
                summaries.append(summary)
            assert np.all(covered == 1)
            assert sum(s['full_sample_calls'] for s in summaries) == 115000
            np.savez(arm_dir / 'samples.npz', arr_0=images)
            write_json(DATA / 'status.json', dict(phase='evaluation', arm=arm))
            cmd = [sys.executable, str(ROOT / 'experiments/evaluate_raev2_official_samples.py'),
                '--branch', arm+'='+str(arm_dir / 'samples.npz'), '--output', str(arm_dir / 'fid.csv'),
                '--batch-size', '32', '--device', 'cuda', '--feature-cache-dir', str(arm_dir / 'features')]
            with (arm_dir / 'evaluation.log').open('x') as log:
                subprocess.run(cmd, env=dict(env, CUDA_VISIBLE_DEVICES='0'), stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((arm_dir / 'fid.json').read_text())[0]
            assert metrics['sample_sha256'] == file_sha256(arm_dir / 'samples.npz')
            row = dict(arm=arm, scale=ARMS[arm][0], high=ARMS[arm][1], metrics=metrics,
                       full_per_image=115, prefix_per_image=0,
                       gpu_inference_seconds_sum=sum(s['inference_seconds'] for s in summaries),
                       inference_seconds_max=max(s['inference_seconds'] for s in summaries))
            write_json(arm_dir / 'result.json', row)
            results.append(row)
            write_json(DATA / 'results.json', results)
            print(json.dumps(row), flush=True)
            write_json(arm_dir / 'advance.json', dict(complete=True))
        for worker in workers:
            if worker.wait() != 0:
                raise RuntimeError('Worker exit failure')
        for path, digest in request['sources'].items():
            assert file_sha256(Path(path)) == digest
        write_json(DATA / 'status.json', dict(phase='complete', seconds=time.perf_counter()-started,
                                             research_goal_achieved=False, results=results))
    except BaseException as error:
        write_json(DATA / 'status.json', dict(phase='failed', error=repr(error)))
        raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            worker.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

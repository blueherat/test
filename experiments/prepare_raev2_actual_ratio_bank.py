"""Freeze and build the one actual-trajectory training/validation data bank."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank-name', default='actual_ratio_bank')
    parser.add_argument('--train-count', type=int, default=5000)
    parser.add_argument('--validation-count', type=int, default=1000)
    parser.add_argument('--train-seed', type=int, default=202609090)
    parser.add_argument('--validation-seed', type=int, default=202609091)
    parser.add_argument('--train-time-seed', type=int, default=202609092)
    parser.add_argument('--validation-time-seed', type=int, default=202609093)
    args = parser.parse_args()
    assert all(n > 0 and n % 1000 == 0 and n % 8 == 0 for n in (args.train_count, args.validation_count))
    out = DATA / args.bank_name
    out.mkdir(exist_ok=False)
    plan = {'train': {'samples': args.train_count, 'seed': args.train_seed, 'time_seed': args.train_time_seed},
            'validation': {'samples': args.validation_count, 'seed': args.validation_seed, 'time_seed': args.validation_time_seed}}
    frozen = {str(path): sha(path) for path in (Path(__file__).resolve(),
              ROOT / 'experiments/cache_raev2_actual_ratio_states.py',
              ROOT / 'experiments/sample_raev2_ancestral_guidance.py')}
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(), 'plan': plan,
             'sources': frozen, 'splits': {}, 'purpose': 'actual native marginals, shared-noise ratio fit; no quality evaluation'}
    def save():
        temp = out / 'execution.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'execution.json')
    save()
    (out / 'frozen_source').mkdir()
    for path in frozen:
        p = Path(path)
        (out / 'frozen_source' / p.name).write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    for split, spec in plan.items():
        folder = out / split
        folder.mkdir()
        state['stage'] = split
        jobs = []
        state['jobs'] = []
        for rank in range(4):
            command = [sys.executable, str(ROOT / 'experiments/cache_raev2_actual_ratio_states.py'),
                       '--output', str(folder), '--shard', str(rank), '--samples', str(spec['samples']),
                       '--seed', str(spec['seed']), '--time-seed', str(spec['time_seed'])]
            log = (folder / f'shard{rank}.log').open('w')
            child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)},
                                     stdout=log, stderr=subprocess.STDOUT)
            jobs.append((child, log))
            state['jobs'].append({'pid': child.pid, 'rank': rank, 'command': command})
            save()
        while any(child.poll() is None for child, _ in jobs):
            failed = [child.returncode for child, _ in jobs if child.poll() not in (None, 0)]
            if failed:
                for child, _ in jobs:
                    if child.poll() is None:
                        child.terminate()
                state.update(stage='failed_requires_inspection_no_restart', failure_codes=failed)
                save()
                raise RuntimeError('cache worker failed; original files retained')
            time.sleep(2)
        for child, log in jobs:
            log.close()
            assert child.returncode == 0
        ids, summaries = [], []
        for rank in range(4):
            shard = folder / f'shard{rank}'
            summary = json.loads((shard / 'summary.json').read_text())
            assert summary['complete']
            metadata = np.load(shard / 'metadata.npz')['records']
            ids.extend(metadata[:, 0].astype(int).tolist())
            assert np.array_equal(metadata[:, 1], metadata[:, 0] % 1000)
            summaries.append({'shard': rank, 'count': summary['count'], 'files': summary['files'],
                              'sample_model_calls': summary['sample_model_calls'], 'seconds': summary['seconds']})
        assert sorted(ids) == list(range(spec['samples']))
        for path, digest in frozen.items():
            assert sha(Path(path)) == digest
        state['splits'][split] = summaries
        save()
    state.update(complete=True, stage='data_complete_no_fit_or_fid_yet', elapsed_seconds=time.time()-state['started_unix'])
    save()


if __name__ == '__main__':
    main()

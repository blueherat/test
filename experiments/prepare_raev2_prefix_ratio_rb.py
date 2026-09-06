"""One fixed conditional-enumeration feature preparation, with parity checks."""
import json
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
    out = DATA/'prefix_ratio_rb_features'
    out.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/'experiments/extract_raev2_prefix_ratio_rb.py',
               ROOT/'experiments/raev2_prefix_ratio_features.py', ROOT/'experiments/raev2_actual_ratio_data.py',
               ROOT/'experiments/sample_raev2_ancestral_guidance.py')}
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(), 'sources': sources,
             'independent_generated_states': 5000, 'real_alternatives_per_state': 5,
             'purpose': 'exact conditional average under the same empirical real distribution, not more independent states',
             'planned_head': 'same feature family, global normalization, unit-predictive-variance Gaussian prior and dim/5000 ridge; no temperature or time/layer search', 'jobs': []}
    def save():
        temp = out/'execution.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'execution.json')
    save()
    (out/'frozen_source').mkdir()
    for path in sources:
        p = Path(path)
        (out/'frozen_source'/p.name).write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    jobs = []
    for rank in range(4):
        command = [sys.executable, str(ROOT/'experiments/extract_raev2_prefix_ratio_rb.py'), '--shard', str(rank)]
        log = (out/f'shard{rank}.log').open('w')
        child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)}, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((child, log))
        state['jobs'].append({'rank': rank, 'pid': child.pid, 'command': command})
        save()
    while any(child.poll() is None for child, _ in jobs):
        failed = [child.returncode for child, _ in jobs if child.poll() not in (None, 0)]
        if failed:
            for child, _ in jobs:
                if child.poll() is None:
                    child.terminate()
            state.update(stage='failed_requires_inspection', exit_codes=failed)
            save()
            raise RuntimeError('conditional feature extraction failed; no automatic restart')
        time.sleep(2)
    ids, positives, negatives, times, seconds = [], [], [], [], 0.
    for rank, (child, log) in enumerate(jobs):
        log.close()
        assert child.returncode == 0
        folder = out/f'shard{rank}'
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['original_selected_pair_features_bitwise_equal']
        assert sha(folder/'features.npz') == summary['features_sha256']
        data = np.load(folder/'features.npz')
        ids.extend(data['ids'].tolist()); times.extend(data['times'].tolist())
        positives.append(data['positive']); negatives.append(data['negative'])
        seconds += summary['seconds']
    order = np.argsort(ids)
    assert np.array_equal(np.array(ids)[order], np.arange(5000))
    np.savez(out/'features.npz', ids=np.array(ids)[order], times=np.array(times, dtype=np.float32)[order],
             positive=np.concatenate(positives)[order], negative=np.concatenate(negatives)[order])
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    state.update(complete=True, stage='conditional_features_complete_no_head_fit_yet',
                 worker_seconds=seconds, features_sha256=sha(out/'features.npz'))
    save()


if __name__ == '__main__':
    main()

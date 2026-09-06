"""Fixed four-GPU prefix extraction on the completed actual trajectory bank."""
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
    out = DATA / 'prefix_ratio_features'
    out.mkdir(exist_ok=False)
    bank_path = DATA / 'actual_ratio_bank/execution.json'
    bank = json.loads(bank_path.read_text())
    assert bank['complete']
    sources = {str(path): sha(path) for path in (Path(__file__).resolve(),
        ROOT / 'experiments/extract_raev2_prefix_ratio_features.py', ROOT / 'experiments/raev2_prefix_ratio_features.py',
        ROOT / 'experiments/raev2_actual_ratio_data.py', ROOT / 'experiments/sample_raev2_ancestral_guidance.py')}
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(),
             'sources': sources, 'actual_bank_execution_sha256': sha(bank_path), 'splits': {},
             'feature_choice': 'pretrained native base depth 8; channel-wise first and second raw moments of RMS-normalized patch tokens; no layer search'}
    def save():
        temp = out / 'execution.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out / 'execution.json')
    save()
    (out / 'frozen_source').mkdir()
    for path in sources:
        p = Path(path)
        (out / 'frozen_source' / p.name).write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    for split, count in [('train', 5000), ('validation', 1000)]:
        folder = out / split
        folder.mkdir()
        state['stage'] = split
        state['jobs'] = []
        jobs = []
        for rank in range(4):
            command = [sys.executable, str(ROOT / 'experiments/extract_raev2_prefix_ratio_features.py'), '--shard', str(rank), '--split', split]
            log = (folder / f'shard{rank}.log').open('w')
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
                raise RuntimeError('feature worker failed, outputs retained')
            time.sleep(2)
        arrays, ids, times, summaries = [], [], [], []
        for rank, (child, log) in enumerate(jobs):
            log.close()
            assert child.returncode == 0
            shard = folder / f'shard{rank}'
            summary = json.loads((shard / 'summary.json').read_text())
            assert summary['complete'] and summary['native_fp32_prefix_parity']
            assert sha(shard / 'features.npz') == summary['features_sha256']
            data = np.load(shard / 'features.npz')
            ids.extend(data['ids'].tolist())
            times.extend(data['times'].tolist())
            arrays.append(data['features'])
            summaries.append(summary)
        order = np.argsort(ids)
        assert np.array_equal(np.array(ids)[order], np.arange(count))
        np.savez(folder / 'features.npz', ids=np.array(ids)[order], times=np.array(times, dtype=np.float32)[order], features=np.concatenate(arrays)[order])
        state['splits'][split] = {'count': count, 'features_sha256': sha(folder / 'features.npz'), 'worker_seconds': sum(s['seconds'] for s in summaries)}
        for path, digest in sources.items():
            assert sha(Path(path)) == digest
        save()
    state.update(complete=True, stage='features_complete_no_head_fit_yet')
    save()


if __name__ == '__main__':
    main()

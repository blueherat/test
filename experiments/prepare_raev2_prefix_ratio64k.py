"""Wait for the existing native bank, verify, extract once, then fit once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continue_raev2_guidance_after_5k import process_identity
from experiments.raev2_prefix_ratio64k_plan import PLAN, positive_indices
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA/'prefix_ratio64k_features'
    out.mkdir(exist_ok=False)
    native_path = DATA/'actual_ratio_bank64k/execution.json'
    real_path = DATA/'real_ratio_bank64k/execution.json'
    native = json.loads(native_path.read_text())
    real = json.loads(real_path.read_text())
    parent_pid = native['pid']
    identity = process_identity(parent_pid)
    assert native['complete'] or identity is not None, 'native prerequisite is not alive; no automatic restart'
    assert real['complete']
    paths = [Path(__file__).resolve(), ROOT/'experiments/extract_raev2_prefix_ratio64k.py',
        ROOT/'experiments/raev2_prefix_ratio64k_plan.py', ROOT/'experiments/raev2_prefix_ratio_features.py',
        ROOT/'experiments/raev2_actual_ratio_data.py', ROOT/'experiments/sample_raev2_ancestral_guidance.py',
        ROOT/'experiments/fit_raev2_prefix_ratio64k.py', ROOT/'experiments/raev2_conditional_ratio_objective.py']
    paths += sorted((ROOT/'external/RAEv2/src/stage2/models').glob('*.py'))
    sources = {str(p): sha(p) for p in paths}
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(), 'plan': PLAN,
             'stage': 'waiting_for_existing_native_bank', 'sources': sources, 'splits': {},
             'native_parent_pid': parent_pid, 'native_parent_process_identity': identity,
             'no_native_bank_restart': True, 'real_execution_sha256': sha(real_path),
             'all_input_hashes_verified': False}
    def save():
        temporary = out/'execution.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'execution.json')
    def check_sources():
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, 'source changed: '+path
    save()
    (out/'plan.json').write_text(json.dumps(PLAN, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_plan.json').write_text(json.dumps(PLAN, indent=2)+'\n')
    (out/'frozen_source').mkdir()
    for i, p in enumerate(paths):
        (out/'frozen_source'/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    try:
        while not native['complete']:
            assert process_identity(parent_pid) == identity, 'native parent stopped before completion; inspect, do not restart'
            time.sleep(5)
            native = json.loads(native_path.read_text())
        check_sources()
        assert sha(real_path) == state['real_execution_sha256']
        state.update(stage='verifying_all_bank_hashes', native_execution_sha256=sha(native_path))
        save()
        for split in ['train', 'validation']:
            spec = PLAN[split]
            assert native['plan'][split]['samples'] == real['splits'][split]['count'] == spec['count']
            all_ids = []
            # Verify complete, unique coverage and all files, not just sampled rows.
            for summary in native['splits'][split]:
                rank = summary['shard']
                folder = DATA/'actual_ratio_bank64k'/split/f'shard{rank}'
                for name, digest in summary['files'].items():
                    assert sha(folder/name) == digest
                request = json.loads((folder/'request.json').read_text())
                records = np.load(folder/'metadata.npz')['records']
                ids = records[:, 0].astype(np.int64)
                assert np.array_equal(records[:, 1], ids%1000)
                assert np.all(ids//8%4 == rank)
                query = np.tile(np.arange(1, 100), (spec['count']//8+98)//99)[:spec['count']//8]
                np.random.default_rng(native['plan'][split]['time_seed']).shuffle(query)
                assert np.array_equal(records[:, 2], query[ids//8])
                assert np.array_equal(records[:, 3], np.asarray(request['query_grid'])[query[ids//8]])
                all_ids.extend(ids.tolist())
            assert sorted(all_ids) == list(range(spec['count']))
            folder = DATA/'real_ratio_bank64k'/split
            for name, digest in real['splits'][split]['files'].items():
                assert sha(folder/name) == digest
            meta = np.load(folder/'metadata.npz')
            assert np.array_equal(meta['ids'], np.arange(spec['count']))
            assert np.array_equal(meta['labels'], meta['ids']%1000)
            lookup = np.load(folder/'real_lookup.npy')
            assert np.array_equal(lookup, meta['ids'].reshape(-1, 1000).T)
        state['all_input_hashes_verified'] = True
        state['native_data_model_calls'] = sum(s['sample_model_calls'] for v in native['splits'].values() for s in v)
        save()
        for split in ['train', 'validation']:
            spec = PLAN[split]
            folder = out/split
            folder.mkdir()
            jobs = []
            state.update(stage='extracting_'+split, jobs=[])
            save()
            for rank in range(4):
                command = [sys.executable, str(ROOT/'experiments/extract_raev2_prefix_ratio64k.py'),
                           '--split', split, '--shard', str(rank)]
                log = (folder/f'shard{rank}.log').open('w')
                child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)}, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((child, log))
                state['jobs'].append({'pid': child.pid, 'rank': rank, 'command': command})
                save()
            while any(child.poll() is None for child, _ in jobs):
                if any(child.poll() not in (None, 0) for child, _ in jobs):
                    for child, _ in jobs:
                        if child.poll() is None: child.terminate()
                    raise RuntimeError('feature worker failed; existing data retained, no automatic restart')
                time.sleep(2)
            positive = np.lib.format.open_memmap(folder/'positive.npy', mode='w+', dtype=np.float32,
                shape=(spec['count'], spec['positive_choices'], 2880))
            negative = np.lib.format.open_memmap(folder/'negative.npy', mode='w+', dtype=np.float32, shape=(spec['count'], 2880))
            times = np.empty(spec['count'], dtype=np.float32)
            seen = np.zeros(spec['count'], dtype=bool)
            seconds = 0.
            for rank, (child, log) in enumerate(jobs):
                log.close()
                assert child.returncode == 0
                shard = folder/f'shard{rank}'
                summary = json.loads((shard/'summary.json').read_text())
                assert summary['complete'] and summary['native_fp32_prefix_parity']
                assert summary['initial_noise_regeneration_and_saved_state_digest_batches_checked'] == 2
                for name, digest in summary['files'].items():
                    assert sha(shard/name) == digest
                meta = np.load(shard/'metadata.npz')
                ids = meta['ids']
                assert not seen[ids].any()
                assert np.array_equal(meta['positive_indices'], positive_indices(ids, split))
                seen[ids] = True
                positive[ids] = np.load(shard/'positive.npy', mmap_mode='r')
                negative[ids] = np.load(shard/'negative.npy', mmap_mode='r')
                times[ids] = meta['times']
                seconds += summary['seconds']
            assert seen.all()
            positive.flush(); negative.flush()
            np.savez(folder/'metadata.npz', ids=np.arange(spec['count']), times=times,
                     positive_indices=positive_indices(np.arange(spec['count']), split))
            check_sources()
            state['splits'][split] = {'count': spec['count'], 'positive_choices': spec['positive_choices'],
                'worker_seconds': seconds, 'files': {name: sha(folder/name) for name in ['positive.npy', 'negative.npy', 'metadata.npz']}}
            save()
        state.update(complete=True, stage='features_complete_before_one_fixed_fit', elapsed_seconds=time.time()-state['started_unix'])
        save()
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise
    # Keep the completed feature manifest immutable once the fitter reads it.
    command = [sys.executable, '-m', 'experiments.fit_raev2_prefix_ratio64k']
    with (out/'fit.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**env, 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'},
                                 stdout=log, stderr=subprocess.STDOUT)
        continuation = {'pid': os.getpid(), 'child_pid': child.pid, 'command': command,
                        'stage': 'fixed_convex_fit_running', 'complete': False}
        (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
        code = child.wait()
    continuation.update(exit_code=code, complete=code == 0,
                        stage='fit_complete_requires_entry_and_gradient_review' if code == 0 else 'fit_failed_no_automatic_restart')
    (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
    if code:
        raise RuntimeError('fixed head fit failed; no second solve is queued')


if __name__ == '__main__':
    main()

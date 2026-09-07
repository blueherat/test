"""One fixed directional-moment extraction after spherical5K, then exact fit."""
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
from experiments.raev2_directional_variance import PLAN
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    prerequisite_path = DATA/'conditional_variance_continuation/state.json'
    prerequisite = json.loads(prerequisite_path.read_text())
    assert prerequisite['complete'] and process_identity(prerequisite['pid']) is None
    assert max(prerequisite[s]['improvement_vs_best_control_percent'] for s in ['screen1k', 'confirm5k']) < 3
    previous_path = DATA/'conditional_variance_features/execution.json'
    previous = json.loads(previous_path.read_text())
    assert previous['complete'] and previous['all_used_input_hashes_verified']
    assert previous['train_validation_real_rows_disjoint']
    fit_path = DATA/'conditional_variance_fit/execution.json'
    fit = json.loads(fit_path.read_text())
    assert fit['complete'] and fit['validation']['entry_condition_passed']
    assert sha(previous_path) == fit['feature_execution_sha256']
    assert sha(DATA/'conditional_variance_fit/head.pt') == fit['checkpoint_sha256'] == PLAN['spherical_head_sha256']
    out = DATA/'directional_variance_moments'
    out.mkdir(exist_ok=False)
    paths = [Path(__file__).resolve(), ROOT/'experiments/raev2_directional_variance.py',
             ROOT/'experiments/extract_raev2_directional_variance.py', ROOT/'experiments/calibrate_raev2_directional_variance.py',
             ROOT/'experiments/raev2_conditional_variance.py', ROOT/'experiments/sample_raev2_ancestral_guidance.py',
             ROOT/'experiments/summarize_raev2_guidance_20260907.py']
    paths += sorted((ROOT/'external/RAEv2/src/stage2/models').glob('*.py'))
    sources = {str(p): sha(p) for p in paths}
    for path, expected in previous['sources'].items():
        if '/external/RAEv2/src/stage2/models/' in path or path.endswith('/raev2_conditional_variance.py'):
            assert sha(Path(path)) == expected
    state = {'complete': False, 'pid': os.getpid(), 'plan': PLAN, 'sources': sources, 'splits': {},
             'stage': 'validating_frozen_feature_and_head_inputs', 'previous_features_sha256': sha(previous_path),
             'previous_fit_sha256': sha(fit_path), 'completed_spherical_quality_sha256': sha(prerequisite_path),
             'prior_full_input_hash_verification_reused': previous['used_input_hashes'],
             'large_original_inputs_not_rehashed_again': True,
             'all_new_features_and_mse_will_be_checked_against_previous_full_arrays': True,
             'started_unix': time.time(), 'no_automatic_restart': True}
    def save():
        temp = out/'execution.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'execution.json')
    def check_sources():
        for path, expected in sources.items():
            assert sha(Path(path)) == expected, 'source changed: '+path
    save()
    (out/'frozen_source').mkdir()
    for i, p in enumerate(paths):
        (out/'frozen_source'/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    try:
        for split in ['train', 'validation']:
            count = PLAN[split]
            for name, expected in previous['splits'][split]['files'].items():
                assert sha(DATA/'conditional_variance_features'/split/name) == expected
            folder = out/split
            folder.mkdir()
            jobs = []
            state.update(stage='extracting_'+split, jobs=[])
            for rank in range(4):
                command = [sys.executable, str(ROOT/'experiments/extract_raev2_directional_variance.py'), '--split', split, '--shard', str(rank)]
                log = (folder/f'shard{rank}.log').open('w')
                child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)}, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((child, log))
                state['jobs'].append({'pid': child.pid, 'rank': rank, 'command': command})
                save()
            while any(child.poll() is None for child, _ in jobs):
                if any(child.poll() not in (None, 0) for child, _ in jobs):
                    for child, _ in jobs:
                        if child.poll() is None:
                            child.terminate()
                    raise RuntimeError('directional worker failed; retain outputs, no automatic restart')
                time.sleep(2)
            merged, summaries = {}, []
            seen = np.zeros(count, dtype=bool)
            for rank, (child, log) in enumerate(jobs):
                log.close()
                assert child.returncode == 0
                shard = folder/f'shard{rank}'
                summary = json.loads((shard/'summary.json').read_text())
                assert summary['complete'] and summary['all_cached_features_and_total_mse_bitwise_equal']
                assert summary['source_noise_regeneration_batches'] == 2
                assert sha(shard/'moments.npz') == summary['files']['moments.npz']
                data = np.load(shard/'moments.npz')
                ids = data['ids']
                assert np.array_equal(ids, np.arange(count).reshape(-1, 8)[rank::4].reshape(-1))
                assert not seen[ids].any()
                seen[ids] = True
                for key in data.files:
                    if key not in merged:
                        merged[key] = np.empty(count, dtype=data[key].dtype)
                    merged[key][ids] = data[key]
                summaries.append(summary)
            assert seen.all() and np.array_equal(merged['ids'], np.arange(count))
            meta = np.load(DATA/'conditional_variance_features'/split/'metadata.npz')
            for key in ['ids', 'labels', 'times', 'query_indices']:
                assert np.array_equal(merged[key], meta[key])
            np.savez(folder/'moments.npz', **merged)
            check_sources()
            state['splits'][split] = {'count': count, 'teacher_sample_main_calls': count,
                'worker_seconds_including_data_access': sum(s['seconds_including_data_access'] for s in summaries),
                'workers': summaries, 'moments_sha256': sha(folder/'moments.npz')}
            save()
        state.update(complete=True, stage='directional_moments_complete_before_single_scalar_fit', elapsed_seconds=time.time()-state['started_unix'])
        save()
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise
    command = [sys.executable, '-m', 'experiments.calibrate_raev2_directional_variance']
    with (out/'calibration.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        continuation = {'pid': os.getpid(), 'child_pid': child.pid, 'command': command, 'complete': False, 'stage': 'single_mean_fit_running'}
        (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
        code = child.wait()
    continuation.update(complete=code == 0, exit_code=code, stage='single_scalar_fit_complete_requires_review' if code == 0 else 'fit_failed_no_restart')
    (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
    assert code == 0


if __name__ == '__main__':
    main()

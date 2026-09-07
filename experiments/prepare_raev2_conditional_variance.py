"""Verify reused inputs, wait for legacy5K, then one native extraction and fit."""
import argparse
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
from experiments.raev2_conditional_variance import PLAN, query_indices
from experiments.summarize_raev2_guidance_20260907 import DATA, sha
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-verified-inputs-after-legacy-audit', action='store_true')
    args = parser.parse_args()
    out = DATA/'conditional_variance_features'
    previous = None
    if args.resume_verified_inputs_after_legacy_audit:
        previous = json.loads((out/'execution.json').read_text())
        assert previous['stage'] == 'failed_requires_inspection_no_restart'
        assert previous['error'] == 'legacy failed; inspect, no speculative restart'
        assert previous['all_used_input_hashes_verified'] and previous['train_validation_real_rows_disjoint']
        assert not previous['splits'] and not (out/'train').exists() and not (out/'validation').exists()
        assert process_identity(previous['pid']) is None
        (out/'execution.failed_before_gpu.json').write_bytes((out/'execution.json').read_bytes())
        (out/'frozen_source').rename(out/'frozen_source_before_metadata_audit_recovery')
    else:
        out.mkdir(exist_ok=False)
    native_path = DATA/'actual_ratio_bank64k/execution.json'
    real_path = DATA/'real_ratio_bank64k/execution.json'
    legacy_path = DATA/'final8_legacy5k_continuation/state.json'
    native, real, legacy = [json.loads(p.read_text()) for p in [native_path, real_path, legacy_path]]
    assert native['complete'] and real['complete']
    pid = legacy['pid']
    identity = process_identity(pid)
    assert legacy['complete'] or identity is not None, 'legacy prerequisite stopped; inspect first'
    paths = [Path(__file__).resolve(), ROOT/'experiments/extract_raev2_conditional_variance.py',
             ROOT/'experiments/raev2_conditional_variance.py', ROOT/'experiments/fit_raev2_conditional_variance.py',
             ROOT/'experiments/sample_raev2_ancestral_guidance.py', ROOT/'experiments/sample_raev2_pfr_retiming.py',
             ROOT/'experiments/summarize_raev2_guidance_20260907.py', ROOT/'experiments/continue_raev2_guidance_after_5k.py']
    paths += sorted((ROOT/'external/RAEv2/src/stage2/models').glob('*.py'))
    sources = {str(p): sha(p) for p in paths}
    calibration_path = DATA/'guided_reverse_variance/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['complete'] and not calibration['fid_used_for_fit']
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(), 'plan': PLAN,
             'stage': 'verifying_used_inputs_while_legacy_runs', 'sources': sources, 'splits': {},
             'legacy_pid': pid, 'legacy_process_identity': identity, 'no_automatic_restart': True,
             'real_execution_sha256': sha(real_path), 'native_execution_sha256': sha(native_path),
             'variance_calibration_sha256': sha(calibration_path), 'all_used_input_hashes_verified': False,
             'unused_native_states_not_read_or_rehashed': True, 'used_input_hashes': {}}
    if previous is not None:
        for path, digest in previous['sources'].items():
            if Path(path) != Path(__file__).resolve():
                assert sources[path] == digest, 'non-controller source changed during recovery: '+path
        assert state['plan'] == previous['plan']
        for key in ['real_execution_sha256', 'native_execution_sha256', 'variance_calibration_sha256']:
            assert state[key] == previous[key]
        state['used_input_hashes'] = previous['used_input_hashes']
        state['recovery'] = {'prior_execution_sha256': sha(out/'execution.failed_before_gpu.json'),
            'reason': 'legacy audit stopped on a descriptive metadata-only source_law field; independent audit repaired without resampling',
            'previous_completed_full_input_hash_verification_reused': True,
            'large_files_not_rehashed_a_second_time': True, 'no_previous_features_or_fit_to_select_from': True}
    def save():
        temporary = out/'execution.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'execution.json')
    def check_sources():
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, 'source changed: '+path
    def verify(path, expected=None):
        actual = sha(path)
        if expected is not None:
            assert actual == expected, 'input hash mismatch: '+str(path)
        state['used_input_hashes'][str(path)] = actual
    save()
    (out/'plan.json').write_text(json.dumps(PLAN, indent=2)+'\n')
    (out/'frozen_source').mkdir()
    for i, p in enumerate(paths):
        (out/'frozen_source'/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    try:
        reference_request = json.loads((DATA/'actual_ratio_bank64k/train/shard0/request.json').read_text())
        grid = np.asarray(reference_request['query_grid'][:-1], dtype=np.float32)
        if previous is None:
            verify(Path(DEFAULT_CHECKPOINT), reference_request['checkpoint_sha256'])
            verify(Path(DEFAULT_CONFIG), reference_request['config_sha256'])
            real_rows = []
            grid = None
            for split in ['train', 'validation']:
                count = PLAN[split]['count']
                assert native['plan'][split]['samples'] == real['splits'][split]['count'] == count
                for summary in native['splits'][split]:
                    rank = summary['shard']
                    folder = DATA/'actual_ratio_bank64k'/split/f'shard{rank}'
                    for name in ['noise.npy', 'metadata.npz']:
                        verify(folder/name, summary['files'][name])
                    for name in ['request.json', 'summary.json']:
                        verify(folder/name)
                    request = json.loads((folder/'request.json').read_text())
                    assert request['checkpoint_sha256'] == reference_request['checkpoint_sha256']
                    assert request['config_sha256'] == reference_request['config_sha256']
                    current_grid = np.asarray(request['query_grid'][:-1], dtype=np.float32)
                    assert len(current_grid) == 100 and current_grid[0] == 1 and current_grid[-1] > .05
                    if grid is None:
                        grid = current_grid
                    assert np.array_equal(current_grid, grid)
                    assert np.max(np.abs(grid-np.asarray([r['time'] for r in calibration['rows']]))) < 2e-7
                    records = np.load(folder/'metadata.npz')['records']
                    ids = np.arange(count).reshape(-1, 8)[rank::4].reshape(-1)
                    assert np.array_equal(records[:, 0], ids) and np.array_equal(records[:, 1], ids%1000)
                folder = DATA/'real_ratio_bank64k'/split
                for name, digest in real['splits'][split]['files'].items():
                    verify(folder/name, digest)
                meta = np.load(folder/'metadata.npz')
                assert np.array_equal(meta['ids'], np.arange(count)) and np.array_equal(meta['labels'], meta['ids']%1000)
                assert np.array_equal(np.load(folder/'real_lookup.npy'), meta['ids'].reshape(-1, 1000).T)
                assert len(np.unique(meta['rows'])) == count
                real_rows.append(meta['rows'])
                save()
            assert not np.intersect1d(*real_rows).size
        state.update(all_used_input_hashes_verified=True, train_validation_real_rows_disjoint=True,
                     stage='waiting_for_legacy5k_to_finish_before_gpu_use')
        save()
        while identity is not None and process_identity(pid) == identity:
            time.sleep(5)
        legacy = json.loads(legacy_path.read_text())
        assert legacy['complete'], 'legacy failed; inspect, no speculative restart'
        assert legacy['stage'] == 'legacy5k_complete_results_require_review'
        audit_path = ROOT/'experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['complete'] and audit['paired_inputs_and_all_merged_pixels_verified']
        state.update(legacy_audit_sha256=sha(audit_path), legacy_execution_sha256=sha(legacy_path))
        if max(row['improvement_vs_best_existing_control_percent'] for row in audit['rows']) >= 3:
            state.update(complete=True, stage='legacy_quality_positive_skip_new_fit_pending_final_review')
            save()
            return
        check_sources()
        assert sha(native_path) == state['native_execution_sha256'] and sha(real_path) == state['real_execution_sha256']
        assert sha(calibration_path) == state['variance_calibration_sha256']
        for split in ['train', 'validation']:
            count = PLAN[split]['count']
            folder = out/split
            folder.mkdir()
            jobs = []
            state.update(stage='extracting_'+split, jobs=[])
            save()
            for rank in range(4):
                command = [sys.executable, str(ROOT/'experiments/extract_raev2_conditional_variance.py'), '--split', split, '--shard', str(rank)]
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
                    raise RuntimeError('feature worker failed; retain evidence, no automatic restart')
                time.sleep(2)
            features = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float32, shape=(count, 2880))
            residual = np.empty(count, dtype=np.float64)
            seen = np.zeros(count, dtype=bool)
            seconds = 0.
            summaries = []
            for rank, (child, log) in enumerate(jobs):
                log.close()
                assert child.returncode == 0
                shard = folder/f'shard{rank}'
                summary = json.loads((shard/'summary.json').read_text())
                assert summary['complete'] and summary['native_outputs_bitwise_unchanged_with_hook']
                assert summary['checkpoint_sha256'] == reference_request['checkpoint_sha256']
                assert summary['config_sha256'] == reference_request['config_sha256']
                assert summary['source_noise_regeneration_batches'] == 2
                for name, digest in summary['files'].items():
                    assert sha(shard/name) == digest
                meta = np.load(shard/'metadata.npz')
                ids = meta['ids']
                assert np.array_equal(ids, np.arange(count).reshape(-1, 8)[rank::4].reshape(-1))
                assert not seen[ids].any() and np.array_equal(meta['labels'], ids%1000)
                query = query_indices(split)[ids//8]
                assert np.array_equal(meta['query_indices'], query) and np.array_equal(meta['times'], grid[query])
                seen[ids] = True
                features[ids] = np.load(shard/'features.npy', mmap_mode='r')
                residual[ids] = meta['residual_mse']
                seconds += summary['seconds']
                summaries.append(summary)
            assert seen.all()
            features.flush()
            ids = np.arange(count)
            query = query_indices(split)[ids//8]
            np.savez(folder/'metadata.npz', ids=ids, labels=ids%1000, times=grid[query], query_indices=query, residual_mse=residual)
            check_sources()
            state['splits'][split] = {'count': count, 'teacher_sample_main_calls': count, 'extra_parity_sample_main_calls': 32,
                'worker_seconds_including_data_access': seconds, 'workers': summaries,
                'files': {name: sha(folder/name) for name in ['features.npy', 'metadata.npz']}}
            save()
        state.update(complete=True, stage='features_complete_before_one_fixed_fit', elapsed_seconds=time.time()-state['started_unix'])
        save()
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise
    command = [sys.executable, '-m', 'experiments.fit_raev2_conditional_variance']
    with (out/'fit.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**env, 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}, stdout=log, stderr=subprocess.STDOUT)
        continuation = {'pid': os.getpid(), 'child_pid': child.pid, 'command': command,
                        'stage': 'fixed_convex_fit_running', 'complete': False}
        (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
        code = child.wait()
    continuation.update(exit_code=code, complete=code == 0,
                        stage='fit_complete_requires_entry_and_native_parity_review' if code == 0 else 'fit_failed_no_automatic_restart')
    (out/'continuation.json').write_text(json.dumps(continuation, indent=2)+'\n')
    if code:
        raise RuntimeError('fixed variance fit failed; no second solve queued')


if __name__ == '__main__':
    main()

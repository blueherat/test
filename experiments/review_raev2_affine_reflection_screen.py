#!/usr/bin/env python3
"""Read-only reconstruction of fixed screen identities, costs and FID."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

STUDY = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/affine_reflection_v1')
REFERENCE = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
REFERENCE_SHA256 = '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
EVALUATOR_COMMIT = '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
EVALUATOR_IDENTITY_SHA256 = 'e30383d1a479b15a9b62da85a3f1dde38f6c02795b2d36d7ae7996106c862fac'
GPU_UUID = 'GPU-7d3e4e7d-abfa-e06e-c264-796052797949'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def measured_costs(summary):
    assert summary['complete'] and summary['samples'] == 1000
    result = (summary['trajectory_wall_seconds'] + summary['decode_and_uint8_wall_seconds'],
              summary['total_wall_seconds_before_summary'])
    assert all(math.isfinite(value) and value > 0 for value in result)
    assert result[1] >= result[0]
    return result


def equal_numbers(actual, expected, *, atol=1e-9):
    assert len(actual) == len(expected)
    assert all(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol
               for a, b in zip(actual, expected)), (actual, expected)


def main():
    began, cpu = time.perf_counter(), time.process_time()
    output = STUDY/'independent_review.json'
    if output.exists():
        raise FileExistsError('preserve previous review')
    execution = read(STUDY/'screen_execution.json')
    assert execution['complete'] and execution['fid_started']
    plan = read(STUDY/'plan.json')
    assert sha(STUDY/'plan.json') == execution['plan_sha256']
    assert plan['source_freeze_complete']
    assert plan['gpu_uuid'] == execution['gpu_uuid'] == GPU_UUID
    checked = {}

    def verify(path, expected):
        p = Path(path).resolve()
        if str(p) not in checked:
            checked[str(p)] = sha(p)
        assert checked[str(p)] == expected, str(p)

    for p, h in plan['frozen_files'].items():
        verify(p, h)
    verify(REFERENCE, REFERENCE_SHA256)
    verify(STUDY/'evaluator_identity.json', EVALUATOR_IDENTITY_SHA256)
    evaluator_identity = read(STUDY/'evaluator_identity.json')
    assert evaluator_identity['frozen_before_fid']
    assert evaluator_identity['execution_fid_started_at_freeze'] is False
    pre_result = evaluator_identity['pre_result_review']
    verify(pre_result['path'], pre_result['sha256'])
    evaluator_root = Path(evaluator_identity['evaluator_root']).resolve()
    assert evaluator_identity['git_commit'] == EVALUATOR_COMMIT
    current_commit = subprocess.check_output(['git', '-C', str(evaluator_root), 'rev-parse', 'HEAD'], text=True).strip()
    worktree_changes = subprocess.check_output(['git', '-C', str(evaluator_root), 'status', '--porcelain'], text=True).strip()
    assert current_commit == EVALUATOR_COMMIT and not worktree_changes
    assert evaluator_identity['tracked_and_untracked_worktree_clean']
    for record in evaluator_identity['core_sources']:
        path = Path(record['path']).resolve()
        assert path.is_relative_to(evaluator_root)
        assert path.stat().st_size == record['bytes']
        verify(path, record['sha256'])
    match = read(STUDY/'cost_match.json')
    assert match['complete'] and match['frozen_before_fid']
    assert sha(STUDY/'cost_match.json') == execution['cost_match_sha256']
    assert match['same_gpu_uuid'] == GPU_UUID
    assert match['selection_records'] == execution['cost_selections']
    jobs = execution['jobs']
    assert jobs[0]['name'] == 'official100' and jobs[1]['name'] == 'reflection100'
    assert jobs[-1]['name'] == 'evaluation' and all(j['exit_code'] == 0 for j in jobs)
    assert evaluator_identity['frozen_utc'] < jobs[-1]['started_utc']
    assert all(a['finished_utc'] <= b['started_utc'] for a, b in zip(jobs[:-1], jobs[1:]))
    assert all(math.isfinite(j['outer_wall_seconds']) and j['outer_wall_seconds'] > 0 for j in jobs)
    assert len(jobs) == len(match['selection_records']) + 3
    assert match['selection_records']

    def summary_record(record, name):
        expected_path = (STUDY/name/'summary.json').resolve()
        assert Path(record['path']).resolve() == expected_path
        verify(expected_path, record['sha256'])
        value = read(expected_path)
        assert value['mode'] == ('reflection' if name == 'reflection100' else 'official')
        assert value['num_steps'] == int(name.removeprefix('reflection').removeprefix('official'))
        measured_costs(value)
        return value

    official100 = summary_record(match['official100'], 'official100')
    reflection100 = summary_record(match['reflection100'], 'reflection100')
    candidate_cost = measured_costs(reflection100)
    equal_numbers(match['T_W_candidate'], candidate_cost)
    previous_record, previous_summary = match['official100'], official100
    selection_audit = []
    for index, record in enumerate(match['selection_records']):
        verify(record['path'], record['sha256'])
        assert Path(record['path']).resolve() == (STUDY/f'cost_selection_{index:02d}.json').resolve()
        selection = read(record['path'])
        assert selection['fid_started'] is False
        assert selection['candidate'] == match['reflection100']
        assert selection['previous_official'] == previous_record
        equal_numbers(selection['candidate_T_W'], candidate_cost)
        previous_cost = measured_costs(previous_summary)
        if index == 0:
            expected_k = max(200, *(math.ceil(100*a/b) for a, b in zip(candidate_cost, previous_cost)))
        else:
            assert any(b < a for a, b in zip(candidate_cost, previous_cost)), 'unnecessary cost-only retry'
            previous_k = previous_summary['num_steps']
            expected_k = max(previous_k+1, *(math.ceil(previous_k*a/b) for a, b in zip(candidate_cost, previous_cost)))
        assert selection['selected_steps'] == expected_k
        job = jobs[index+2]
        assert job['name'] == f'official{expected_k}'
        assert jobs[index+1]['finished_utc'] <= selection['selected_utc'] <= job['started_utc']
        assert selection['selected_utc'] < jobs[-1]['started_utc']
        actual_path = STUDY/job['name']/'summary.json'
        # Intermediate higher-step runs also remain part of the recorded budget.
        current_record = {'path': str(actual_path.resolve()), 'sha256': sha(actual_path)}
        current_summary = summary_record(current_record, job['name'])
        selection_audit.append({'selected_steps': expected_k, 'previous_T_W': list(previous_cost),
                                'actual_T_W': list(measured_costs(current_summary)),
                                'summary_sha256': current_record['sha256']})
        if index+1 < len(match['selection_records']):
            next_selection = read(match['selection_records'][index+1]['path'])
            linked_record = next_selection['previous_official']
        else:
            linked_record = match['official_cost']
        assert Path(linked_record['path']).resolve() == actual_path.resolve()
        assert linked_record['sha256'] == current_record['sha256']
        previous_record, previous_summary = linked_record, current_summary
    assert match['final_official_steps'] == previous_summary['num_steps'] >= 200
    matched_cost = measured_costs(previous_summary)
    equal_numbers(match['T_W_official_cost'], matched_cost)
    assert all(b >= a for a, b in zip(candidate_cost, matched_cost))
    equal_numbers(match['relative_budget_excess'], [b/a-1 for a, b in zip(candidate_cost, matched_cost)], atol=1e-12)
    assert match['stage2_nfe_candidate'] == 200
    assert match['stage2_nfe_baseline'] == match['final_official_steps']
    verify(STUDY/'official_evaluation.json', execution['evaluation_sha256'])
    metrics = read(STUDY/'official_evaluation.json')
    assert [m['branch'] for m in metrics] == ['official100', 'reflection100', f'official{match["final_official_steps"]}']
    with np.load(REFERENCE, allow_pickle=False) as ref:
        mu_ref, cov_ref = ref['mu'].astype(np.float64), ref['sigma'].astype(np.float64)
    paired, batches_paired, results = None, None, []
    for metric in metrics:
        assert metric['fid_reference'] == 'imagenet_256_fid_stats'
        assert metric['evaluator_commit'] == EVALUATOR_COMMIT
        assert Path(metric['evaluator_root']).resolve() == evaluator_root
        name = metric['branch']
        directory = STUDY/name
        summary, request = read(directory/'summary.json'), read(directory/'request.json')
        verify(directory/'request.json', summary['request']['sha256'])
        assert summary['complete'] and summary['samples'] == 1000
        assert request['batch_size'] == 8 and request['sample_count'] == 1000
        assert summary['seed'] == 202609131
        for record in request['sources'].values():
            verify(record['path'], record['sha256'])
        for record in request['identities'].values():
            verify(record['path'], record['sha256'])
        keys = ('global_noise_sha256', 'noise_rng_state_sha256', 'global_labels_sha256')
        identity = tuple(summary[k] for k in keys)
        if paired is None:
            paired = identity
        assert paired == identity
        factor = 2 if name == 'reflection100' else 1
        steps = request['num_steps']
        assert summary['stage2_forward_calls'] == 125*steps*factor
        assert summary['stage2_sample_forwards'] == 1000*steps*factor
        assert summary['decoder_forward_calls'] == 125 and summary['decoder_sample_forwards'] == 1000
        assert summary['reflection_calls'] == (125*steps if factor == 2 else 0)
        assert summary['average_projection_calls'] == summary['reflection_calls']
        archive = summary['sample_archive']
        verify(archive['path'], archive['sha256'])
        assert archive['sha256'] == metric['sample_sha256']
        with np.load(archive['path'], allow_pickle=False) as z:
            images, ids, labels = z['arr_0'], z['ids'], z['labels']
        assert images.shape == (1000, 256, 256, 3) and images.dtype == np.uint8
        assert np.array_equal(ids, np.arange(1000)) and np.array_equal(labels, ids)
        verify(summary['batch_manifest']['path'], summary['batch_manifest']['sha256'])
        batches = read(summary['batch_manifest']['path'])['batches']
        assert len(batches) == 125
        batch_identities = []
        for i, batch in enumerate(batches):
            selected = list(range(8*i, 8*i+8))
            assert batch['global_ids'] == selected
            record = batch['archive']
            verify(record['path'], record['sha256'])
            with np.load(record['path'], allow_pickle=False) as z:
                assert np.array_equal(z['arr_0'], images[selected])
                assert np.array_equal(z['ids'], ids[selected]) and np.array_equal(z['labels'], labels[selected])
            assert batch['forward_counts']['observed'] == batch['forward_counts']['expected']
            batch_identities.append((batch['noise_sha256'], batch['labels_sha256']))
        if batches_paired is None:
            batches_paired = batch_identities
        assert batch_identities == batches_paired
        inference = sum(b['trajectory']['trajectory_wall_seconds']+b['trajectory']['decode_and_uint8_wall_seconds'] for b in batches)
        assert abs(inference-summary['inference_trajectory_plus_decode_wall_seconds']) < 1e-9
        equal_numbers([inference], [measured_costs(summary)[0]])
        feature = STUDY/'official_feature_cache'/f'{name}-{archive["sha256"][:16]}-inception.features.pt'
        x = torch.load(feature, map_location='cpu', weights_only=False).numpy().astype(np.float64)
        assert x.shape == (1000, 2048) and np.isfinite(x).all()
        mean = x.mean(axis=0)
        centered = x-mean
        # Independent low-rank Bures trace: nonzero eigenvalues equal those of
        # C_ref^(1/2) C_sample C_ref^(1/2). No producer metric implementation.
        gram = (centered @ cov_ref @ centered.T)/(len(x)-1)
        eigenvalues = np.linalg.eigvalsh((gram+gram.T)*.5)
        assert eigenvalues.min() > -1e-8
        fid = float(np.sum((mean-mu_ref)**2)+np.sum(centered**2)/(len(x)-1)+np.trace(cov_ref)
                    -2*np.sqrt(np.maximum(eigenvalues, 0)).sum())
        difference = fid-float(metric['fid'])
        assert abs(difference) < 2e-4, (name, fid, metric['fid'])
        results.append({'branch': name, 'reported_fid': metric['fid'], 'independent_low_rank_fid': fid,
                        'fid_difference': difference, 'gram_min_eigenvalue': float(eigenvalues.min()),
                        'features_sha256': sha(feature), 'trajectory_plus_decode_wall_seconds': inference,
                        'total_wall_seconds': summary['total_wall_seconds_before_summary'],
                        'outer_process_wall_seconds': next(job['outer_wall_seconds'] for job in jobs if job['name'] == name),
                        'stage2_nfe': steps*factor})
    candidate = results[1]['reported_fid']
    candidate_outer = results[1]['outer_process_wall_seconds']
    matched_outer = results[2]['outer_process_wall_seconds']
    report = {'complete': True, 'goal_complete': False, 'samples_per_arm': 1000,
              'method_frozen_before_quality_evaluation': True,
              'all_samples_preserved_and_batch_archives_match': True,
              'all_noise_and_label_hashes_paired': True, 'same_gpu_sequential_runs': True,
              'relative_fid_gain_vs_official100': 1-candidate/results[0]['reported_fid'],
              'relative_fid_gain_vs_cost_official': 1-candidate/results[2]['reported_fid'],
              'results': results, 'verified_artifact_count': len(checked), 'verified_artifacts': checked,
              'cost_selection_recomputed': selection_audit,
              'cost_summary_hashes_T_W_and_evaluation_hash_verified': True,
              'fixed_reference_sha256': REFERENCE_SHA256,
              'fixed_evaluator_commit': EVALUATOR_COMMIT,
              'supplemental_evaluator_identity_sha256': EVALUATOR_IDENTITY_SHA256,
              'evaluator_core_files_commit_and_clean_worktree_verified': True,
              'outer_process_cost_comparison': {
                  'candidate_seconds': candidate_outer, 'final_official_seconds': matched_outer,
                  'final_official_covers_candidate': matched_outer >= candidate_outer,
                  'relative_official_budget_excess': matched_outer/candidate_outer-1,
                  'all_sampling_and_evaluation_jobs': [{'name': job['name'], 'outer_wall_seconds': job['outer_wall_seconds']} for job in jobs],
                  'scope': 'Outer wall includes imports, main, summary/print/teardown and driver polling. This is reported separately; K was selected using the frozen main W and inference T rule, and is not changed after FID even if outer wall does not match.'},
              'review_script_sha256': sha(__file__), 'wall_seconds': time.perf_counter()-began,
              'cpu_seconds': time.process_time()-cpu, 'new_model_or_gpu_calls': 0,
              'scope': 'existing features reconstruct FID independently; this does not independently re-extract Inception features or recreate unsaved latent endpoints'}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'verified_artifacts'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

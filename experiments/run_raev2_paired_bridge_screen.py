#!/usr/bin/env python3
"""Run the frozen paired-bridge screen sequentially; select cost before FID.

This driver owns parity and refuses to resume or overwrite an earlier execution.
Interrupted children retain independent sessions, PIDs, logs and partial outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
STUDY = RESTART/'paired_bridge_v1/screen_v1'
GPU = 'GPU-7d3e4e7d-abfa-e06e-c264-796052797949'
PROTOCOL = 'raev2_paired_bridge_fixed_1k_v1'
SAMPLING_PROTOCOL = 'raev2_paired_bridge_sampling_v1'
SEED, COUNT, BATCH = 202609151, 1000, 8
SAMPLER = ROOT/'experiments/sample_raev2_paired_bridge.py'
EVALUATOR = ROOT/'experiments/evaluate_raev2_official_samples.py'
EVALUATOR_ROOT = Path('/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals')
EVALUATOR_COMMIT = '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
SCREEN_PROTOCOL = ROOT/'docs/RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md'
BRIDGE = RESTART/'paired_bridge_v1/train/final.pt'
BRIDGE_SHA = '5013cbf075ddffa5c2ae021fc916be0615ca41d9817d3af91e6b3e46c86e1983'
REFERENCE = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
REFERENCE_SHA = '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
INCEPTION = Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
INCEPTION_SHA = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
IDENTITY_KEYS = ('global_noise_sha256', 'noise_rng_state_sha256', 'global_labels_sha256')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def put(path, value, *, exclusive=False):
    """Immutable new records use O_EXCL; mutable execution state uses replace."""
    path = Path(path)
    temporary = path if exclusive else path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)


def read_record(record):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('expected an artifact {path,sha256,size_bytes}')
    if artifact(record['path']) != record:
        raise ValueError('artifact changed: '+str(record['path']))
    return json.loads(Path(record['path']).read_text())


def valid_pair(pair):
    if len(pair) != 2 or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                and math.isfinite(x) and x > 0 for x in pair):
        raise ValueError('costs must be two finite positive numbers')
    if pair[1] < pair[0]:
        raise ValueError('whole-run W cannot be less than trajectory-through-decode T')
    return tuple(pair)


def costs(summary):
    if summary.get('complete') is not True or summary.get('samples') != COUNT:
        raise ValueError('incomplete sampling cannot enter cost comparison')
    return valid_pair((summary['trajectory_through_decode_wall_seconds'],
                       summary['total_wall_seconds_before_summary']))


def scaled_ceiling(steps, candidate, baseline):
    # Exact arithmetic on the recorded binary floats avoids rounding a tiny
    # positive budget shortfall down to an apparent exact integer ratio.
    return math.ceil(steps*Fraction(candidate)/Fraction(baseline))


def initial_steps(candidate, baseline):
    candidate, baseline = valid_pair(candidate), valid_pair(baseline)
    return max(100, *(scaled_ceiling(100, a, b) for a, b in zip(candidate, baseline)))


def next_steps(steps, candidate, baseline):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 100:
        raise ValueError('previous official step count must be an integer >=100')
    candidate, baseline = valid_pair(candidate), valid_pair(baseline)
    return max(steps+1, *(scaled_ceiling(steps, a, b) for a, b in zip(candidate, baseline)))


def covered(candidate, baseline):
    return all(b >= a for a, b in zip(valid_pair(candidate), valid_pair(baseline)))


def gpu_status(*, strict):
    result = subprocess.run(
        ['nvidia-smi', '--id='+GPU,
         '--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw,temperature.gpu',
         '--format=csv,noheader'], capture_output=True, text=True)
    status = {'observed_utc': stamp(), 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    correct = (result.returncode == 0 and len(rows) == 1 and len(rows[0]) == 7
               and rows[0][0].strip() == '3' and rows[0][1].strip() == GPU)
    status['fixed_gpu_identity_verified'] = correct
    if strict and not correct:
        raise RuntimeError('fixed physical GPU3 identity unavailable: '+json.dumps(status))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY)
    args = parser.parse_args()
    study = args.study.expanduser().resolve()
    plan_path, execution_path = study/'plan.json', study/'screen_execution.json'
    if execution_path.exists():
        raise FileExistsError('refusing to restart an existing screen execution; inspect its PIDs and logs')
    plan = json.loads(plan_path.read_text())
    plan_sha = sha(plan_path)
    logs, parity_dir = study/'screen_logs', study/'parity'
    reserved = [logs, parity_dir, study/'candidate100', study/'control100', study/'cost_match.json',
                study/'official_evaluation.csv', study/'official_evaluation.json', study/'official_feature_cache']
    reserved += list(study.glob('official[0-9]*'))+list(study.glob('cost_selection_*.json'))
    for path in reserved:
        if path.exists():
            raise FileExistsError('refusing existing screen log/output: '+str(path))
    began = time.perf_counter()
    state = {'protocol': PROTOCOL, 'complete': False, 'driver_pid': os.getpid(),
             'started_utc': stamp(), 'plan_sha256': plan_sha, 'gpu_uuid': GPU,
             'physical_gpu_index': 3, 'jobs': [], 'cost_selections': [], 'sampling_results': {},
             'verification_wall_seconds': 0., 'fid_started': False,
             'no_automatic_restart': True, 'goal_complete': False, 'total_cost_complete': False}
    put(execution_path, state, exclusive=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_HOME='/home/zhoushunyu/.cache/torch')
    frozen = {}

    def verify():
        started = time.perf_counter()
        try:
            if sha(plan_path) != plan_sha:
                raise ValueError('plan changed during the screen')
            for path, expected in frozen.items():
                if sha(path) != expected:
                    raise ValueError('frozen file changed: '+str(path))
            commit = subprocess.run(['git', '-C', str(EVALUATOR_ROOT), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(['git', '-C', str(EVALUATOR_ROOT), 'status', '--porcelain',
                                    '--untracked-files=no'], capture_output=True, text=True, check=True).stdout
            if commit != EVALUATOR_COMMIT or dirty:
                raise ValueError('official evaluator commit or tracked sources changed')
        finally:
            state['verification_wall_seconds'] += time.perf_counter()-started
            put(execution_path, state)

    def run(name, argv):
        verify()
        path = logs/(name+'.log')
        if path.exists():
            raise FileExistsError(path)
        job = {'name': name, 'argv': argv, 'started_utc': stamp(), 'log': str(path),
               'gpu_before': gpu_status(strict=True), 'pid': None, 'exit_code': None}
        state['jobs'].append(job)
        put(execution_path, state)
        started, child = time.perf_counter(), None
        try:
            with path.open('xb') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = child.pid
                put(execution_path, state)
                while child.poll() is None:
                    time.sleep(1)
            job.update(exit_code=child.returncode, outer_wall_seconds=time.perf_counter()-started,
                       finished_utc=stamp(), gpu_after=gpu_status(strict=False))
            put(execution_path, state)
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; retain {path} and partial output')
            verify()
        except BaseException as error:
            code = child.poll() if child is not None else None
            job.update(exit_code=code, observed_wall_seconds=time.perf_counter()-started,
                       interruption_utc=stamp(), error=f'{type(error).__name__}: {error}',
                       child_observed_running=(child is not None and code is None),
                       child_not_terminated_by_driver=True)
            put(execution_path, state)
            raise

    def check_request(summary, output, mode, steps):
        request = read_record(summary['request'])
        if Path(summary['request']['path']) != output/'request.json':
            raise ValueError('unexpected request path')
        if (summary.get('protocol') != SAMPLING_PROTOCOL or summary.get('mode') != mode
                or summary.get('seed') != SEED or summary.get('num_steps') != steps
                or request.get('protocol') != SAMPLING_PROTOCOL or request.get('mode') != mode
                or request.get('num_steps') != steps or request.get('seed') != SEED
                or request.get('batch_size') != BATCH or request.get('fid_performed') is not False
                or request.get('cuda_visible_devices') != GPU):
            raise ValueError('sampler identity or protocol differs from frozen screen')
        for relative, record in request['sources'].items():
            if artifact(record['path']) != record or frozen.get((ROOT/relative).resolve()) != record['sha256']:
                raise ValueError('sample source is not frozen: '+relative)
        for record in (*request['identities'].values(), *(
                request['training'][key] for key in ('checkpoint', 'request', 'summary', 'supplemental_sources'))):
            if frozen.get(Path(record['path']).resolve()) != record['sha256']:
                raise ValueError('sample model/training identity is not frozen: '+record['path'])
        return request

    def paired(reference, other):
        for key in IDENTITY_KEYS:
            value = reference[key]
            if not isinstance(value, str) or len(value) != 64 or value != other[key]:
                raise ValueError('paired input identity mismatch: '+key)

    def sample(name, mode, steps):
        output = study/name
        if output.exists():
            raise FileExistsError(output)
        run(name, [sys.executable, str(SAMPLER), '--mode', mode, '--output-dir', str(output),
                   '--parity-dir', str(parity_dir), '--num-steps', str(steps)])
        summary = json.loads((output/'summary.json').read_text())
        request = check_request(summary, output, mode, steps)
        measured = costs(summary)
        if (summary.get('command') != 'sample' or request.get('sample_count') != COUNT
                or summary.get('global_ids') != list(range(COUNT))
                or summary.get('stage2_nfe_per_sample') != steps
                or summary.get('fid_performed') is not False):
            raise ValueError('incomplete or unexpected full sampling cohort')
        expected = {'stage2_forward_calls': steps*COUNT//BATCH, 'stage2_sample_forwards': steps*COUNT,
                    'decoder_forward_calls': COUNT//BATCH, 'decoder_sample_forwards': COUNT}
        if mode != 'official':
            expected.update({mode+'_forward_calls': 2*steps*COUNT//BATCH,
                             mode+'_sample_forwards': 2*steps*COUNT})
        if summary['forward_counts_observed'] != expected or summary['forward_counts_expected'] != expected:
            raise ValueError('actual forward hooks differ from full-arm budget')
        archive = summary['samples_npz']
        if (not isinstance(archive, dict) or Path(archive['path']).resolve() != output/'samples.npz'
                or artifact(output/'samples.npz') != archive):
            raise ValueError('invalid samples_npz artifact')
        record = artifact(output/'summary.json')
        state['sampling_results'][name] = {'summary': record, 'samples_npz': archive, 'T_W': measured}
        put(execution_path, state)
        return summary, record

    try:
        if (plan.get('protocol') != PROTOCOL or plan.get('source_freeze_complete') is not True
                or plan.get('gpu_uuid') != GPU
                or plan.get('cohort') != {'seed': SEED, 'n': COUNT, 'batch_size': BATCH}
                or not isinstance(plan.get('frozen_files'), dict) or not plan['frozen_files']):
            raise ValueError('complete fixed paired-bridge plan required')
        for raw_path, expected in plan['frozen_files'].items():
            path = Path(raw_path)
            path = (path if path.is_absolute() else ROOT/path).resolve()
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('frozen_files must map paths to SHA256 strings')
            if path in frozen and frozen[path] != expected:
                raise ValueError('conflicting frozen identity: '+str(path))
            frozen[path] = expected
        required = [Path(__file__), SAMPLER, EVALUATOR, SCREEN_PROTOCOL,
                    ROOT/'tests/test_sample_raev2_paired_bridge.py', ROOT/'experiments/raev2_paired_bridge.py',
                    ROOT/'experiments/run_raev2_paired_bridge_pilot.py', BRIDGE,
                    BRIDGE.parent/'summary.json', BRIDGE.parent/'request.json',
                    BRIDGE.parent.parent/'supplemental_source_environment.json', REFERENCE, INCEPTION]
        for path in required:
            if path.resolve() not in frozen:
                raise ValueError('required file absent from freeze: '+str(path))
        for path, expected in ((BRIDGE, BRIDGE_SHA), (REFERENCE, REFERENCE_SHA), (INCEPTION, INCEPTION_SHA)):
            if frozen[path.resolve()] != expected:
                raise ValueError('protocol identity differs: '+str(path))
        verify()
        logs.mkdir()
        run('parity', [sys.executable, str(SAMPLER), '--mode', 'parity', '--output-dir', str(parity_dir)])
        parity_path = parity_dir/'summary.json'
        parity = json.loads(parity_path.read_text())
        request = check_request(parity, parity_dir, 'parity', 100)
        if (parity.get('complete') is not True or parity.get('command') != 'parity'
                or parity.get('samples_per_loop') != 16 or request.get('sample_count') != 16
                or not all(parity.get(key) is True for key in ('stepwise_bitwise', 'endpoint_bitwise', 'pixel_bitwise'))):
            raise ValueError('full production/official/zero-candidate/zero-control parity must pass')
        parity_result = read_record(parity['parity_result'])
        finite_response = read_record(parity['finite_response'])
        if parity_result.get('complete') is not True or finite_response.get('complete') is not True:
            raise ValueError('parity detail or trained finite-response check incomplete')
        state['parity'] = artifact(parity_path)
        put(execution_path, state)
        official, official_record = sample('official100', 'official', 100)
        candidate, candidate_record = sample('candidate100', 'candidate', 100)
        control, control_record = sample('control100', 'control', 100)
        paired(official, candidate)
        paired(official, control)
        candidate_cost = costs(candidate)
        steps = initial_steps(candidate_cost, costs(official))
        previous, previous_record = official, official_record
        while True:
            reuse = steps == 100 and covered(candidate_cost, costs(official))
            selection = {'selected_utc': stamp(), 'selected_steps': steps, 'fid_started': False,
                         'candidate': candidate_record, 'previous_official': previous_record,
                         'candidate_T_W': candidate_cost, 'previous_official_T_W': costs(previous),
                         'reuse_official100': reuse,
                         'rule': 'K>=100; ceil100*initial ratios; then max(K+1,ceilK*ratios), both T and W; no control/FID input'}
            selection_path = study/f'cost_selection_{len(state["cost_selections"]):02d}.json'
            put(selection_path, selection, exclusive=True)
            state['cost_selections'].append(artifact(selection_path))
            put(execution_path, state)
            if reuse:
                matched, matched_record = official, official_record
            else:
                matched, matched_record = sample(f'official{steps}', 'official', steps)
            paired(candidate, matched)
            matched_cost = costs(matched)
            if covered(candidate_cost, matched_cost):
                break
            previous, previous_record = matched, matched_record
            steps = next_steps(steps, candidate_cost, matched_cost)
        match = {'complete': True, 'inference_cost_match_complete': True, 'frozen_before_fid': True,
                 'final_official_steps': steps, 'official100': official_record,
                 'candidate100': candidate_record, 'control100': control_record, 'official_cost': matched_record,
                 'T_W_candidate': candidate_cost, 'T_W_official_cost': matched_cost,
                 'relative_budget_excess': [b/a-1 for a, b in zip(candidate_cost, matched_cost)],
                 'stage2_nfe_candidate': 100, 'stage2_nfe_baseline': steps,
                 'auxiliary_forwards_per_candidate_sample': 200, 'control_did_not_select_cost': True,
                 'official100_reused_for_cost_baseline': steps == 100, 'same_gpu_uuid': GPU,
                 'timings_are_observed_point_estimates': True, 'selection_records': state['cost_selections'],
                 'all_intermediate_official_outputs_retained': True,
                 'additional_costs': {'historical_real_image_encoding_seconds': 113.860739042,
                     'historical_encoded_images': 6000, 'pilot_train_validate_rollout_runner_wall_seconds': 741.310023239,
                     'joint_candidate_control_training_seconds_included_above': 490.104689422,
                     'joint_training_not_divided_between_fields': True, 'prior_cpu_review_wall_seconds_approx': 13.897838,
                     'parity_summary': state['parity'], 'current_jobs_and_wall': str(execution_path),
                     'driver_source_verification_seconds_at_cost_freeze': state['verification_wall_seconds'],
                     'current_parity_sampling_evaluation_failed_and_intermediate_jobs_are_additional': True},
                 'total_cost_complete': False, 'goal_complete': False,
                 'cost_boundary': 'K matches inference T/W only. Data selection cost and exact historical training-driver outer wall are unclosed; training, diagnostics, parity, all arms, intermediate baselines, evaluation and driver verification remain additional disclosed costs.'}
        cost_path = study/'cost_match.json'
        put(cost_path, match, exclusive=True)
        state['cost_match_sha256'] = sha(cost_path)
        evaluation_arms = [('official100', official), ('candidate100', candidate), ('control100', control)]
        if steps != 100:
            evaluation_arms.append((f'official{steps}', matched))
        argv = [sys.executable, str(EVALUATOR)]
        for name, summary in evaluation_arms:
            archive = summary['samples_npz']
            if artifact(archive['path']) != archive:
                raise ValueError('sample archive changed before evaluation')
            argv += ['--branch', name+'='+archive['path']]
        argv += ['--output', str(study/'official_evaluation.csv'), '--batch-size', '64',
                 '--device', 'cuda', '--seed', '2020', '--fid-reference', 'imagenet_256_fid_stats',
                 '--evaluator-root', str(EVALUATOR_ROOT), '--feature-cache-dir', str(study/'official_feature_cache')]
        state.update(fid_started=True, fid_authorized_utc=stamp(),
                     evaluation_branches=[name for name, _ in evaluation_arms],
                     cost_baseline_evaluation_branch=f'official{steps}')
        put(execution_path, state)
        run('evaluation', argv)
        if sha(cost_path) != state['cost_match_sha256']:
            raise ValueError('pre-FID cost match changed')
        rows = json.loads((study/'official_evaluation.json').read_text())
        if not isinstance(rows, list) or len(rows) != len(evaluation_arms):
            raise ValueError('evaluation must cover exactly the frozen unique arms')
        for row, (name, summary) in zip(rows, evaluation_arms):
            if (row.get('branch') != name or row.get('sample_sha256') != summary['samples_npz']['sha256']
                    or row.get('evaluator_commit') != EVALUATOR_COMMIT
                    or row.get('fid_reference') != 'imagenet_256_fid_stats'
                    or not math.isfinite(row['fid'])):
                raise ValueError('evaluated identity or finite FID differs: '+name)
        state.update(complete=True, finished_utc=stamp(), driver_wall_seconds=time.perf_counter()-began,
                     evaluation=artifact(study/'official_evaluation.json'),
                     evaluation_csv=artifact(study/'official_evaluation.csv'))
        put(execution_path, state)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stopped_utc=stamp(),
                     driver_wall_seconds=time.perf_counter()-began)
        put(execution_path, state)
        raise


if __name__ == '__main__':
    main()

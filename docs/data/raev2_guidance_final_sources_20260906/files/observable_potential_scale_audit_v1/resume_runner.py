#!/usr/bin/env python3
"""Freeze and execute the single specified 5K scale audit, never select a method."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/zhoushunyu/eqvae')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906').resolve()
OUT = BASE / 'observable_potential_scale_audit_v1'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
CHECKPOINT = BASE / 'observable_potential_train_v1/potential_final.pt'
EXPECTED_CHECKPOINT = '495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e'
ARMS = [('official100', 'official', 100), ('potential100', 'potential', 100), ('official105', 'official', 105)]
SAMPLES, SEED, SHARDS = 5000, 202609101, 4
STATS_DIR = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419')
TORCH_HOME = Path('/home/zhoushunyu/.cache/torch')
EVALUATOR_ASSETS = {
    'reference': (STATS_DIR / 'imagenet_256_fid_stats.npz', '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'),
    'inception': (TORCH_HOME / 'hub/checkpoints/weights-inception-2015-12-05-6726825d.pth', '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'),
}
WORKERS = []
SOURCE_FILES = [
    'experiments/run_raev2_observable_potential_scale_audit.py',
    'experiments/sample_raev2_observable_potential.py',
    'experiments/merge_raev2_observable_potential.py',
    'experiments/evaluate_raev2_official_samples.py',
    'experiments/raev2_training_core.py',
    'experiments/train_raev2_observable_potential.py',
    'experiments/raev2_observable_potential.py',
    'experiments/sample_raev2_pfr_retiming.py',
    'experiments/audit_raev2_proximal_calibration.py',
    'experiments/raev2_stage1_compat.py',
    'external/RAEv2/src/utils/guidance_utils.py',
    'external/RAEv2/src/utils/model_utils.py',
    'external/RAEv2/src/stage2/models/DDT.py',
    'external/RAEv2/src/stage2/models/model_utils.py',
    'external/RAEv2/src/stage1/rae.py',
    'docs/RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md',
    'tests/test_sample_raev2_observable_potential.py',
    'tests/test_merge_raev2_observable_potential.py',
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze():
    if (OUT / 'request.json').exists():
        raise FileExistsError('already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = record(CHECKPOINT)
    if candidate['sha256'] != EXPECTED_CHECKPOINT:
        raise ValueError('frozen candidate changed')
    assets = {name: record(path) for name, (path, _) in EVALUATOR_ASSETS.items()}
    for name, (_, expected) in EVALUATOR_ASSETS.items():
        if assets[name]['sha256'] != expected:
            raise ValueError(f'evaluator asset changed: {name}')
    sources = {}
    for name in SOURCE_FILES:
        dest = OUT / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        sources[name] = record(dest)
    previous = json.loads((BASE / 'observable_potential_screen_v1/request.json').read_text())
    request = {
        'protocol': 'raev2_observable_potential_scale_audit_v1', 'created_utc': utc(),
        'samples': SAMPLES, 'seed': SEED, 'reserved_confirmation_seed': 202609102,
        'shards': SHARDS, 'batch_size': 8, 'arms_in_execution_order': ARMS,
        'candidate': candidate, 'sources': sources, 'previous_cost_records': previous['source_records'],
        'previous_1k_request': record(BASE / 'observable_potential_screen_v1/request.json'),
        'previous_1k_result': record(BASE / 'observable_potential_screen_v1/summary.json'),
        'known_previous_candidate_fid': 37.53166440688716,
        'new_5k_images_or_fid_before_freeze': False,
        'reason': 'Correct an overly strong operational 1K gate; not a new fitted method or an unseen-result initial screen.',
        'full_cost_comparison_complete': False, 'known_cost_step_lower_bound': 126,
        'unknown_preparation_cost_nonnegative': True,
        'fid_protocol': previous['evaluator_protocol'],
        'expected_evaluator_commit': '19dfb4c2705333eb8b97e454fb354d47d1fe135b',
        'evaluator_assets': assets,
        'evaluator_environment': {'NANOGEN_EVALS_STATS_DIR': str(STATS_DIR), 'TORCH_HOME': str(TORCH_HOME)},
        'success_claim_permitted': False,
        'cost_scope': 'All three arms run regardless of observed FID. Outer worker elapsed intervals and child CPU are research costs; sampling trajectories are separately measured. No full-cost match claimed.',
    }
    write(OUT / 'request.json', request)
    print(json.dumps(record(OUT / 'request.json')), flush=True)


def checked_run(command, log_path, env):
    started, cpu = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    start_utc = utc()
    with log_path.open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    summary = {'command': command, 'started_utc': start_utc, 'finished_utc': utc(),
               'returncode': result.returncode, 'wall_seconds': time.perf_counter() - started,
               'child_cpu_seconds': after.ru_utime + after.ru_stime - cpu.ru_utime - cpu.ru_stime,
               'includes_imports_and_exit': True, 'log': record(log_path)}
    write(log_path.with_suffix('.process.json'), summary)
    if result.returncode:
        raise RuntimeError(f'child failed; inspect {log_path}')
    return summary


def execute():
    request = json.loads((OUT / 'request.json').read_text())
    if (OUT / 'execution_resumed.json').exists():
        raise FileExistsError('resume already started')
    if not (OUT / 'failure.json').exists():
        raise ValueError('expected preserved interrupted-run record')
    for name, source in request['sources'].items():
        if record(ROOT / name)['sha256'] != source['sha256'] or record(source['path']) != source:
            raise ValueError(f'source changed after freeze: {name}')
    if record(CHECKPOINT) != request['candidate']:
        raise ValueError('candidate changed')
    for asset in request['evaluator_assets'].values():
        if record(asset['path']) != asset:
            raise ValueError('evaluator asset changed after freeze')
    evaluator_root = '/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals'
    commit = subprocess.check_output(['git', '-C', evaluator_root, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != request['expected_evaluator_commit']:
        raise ValueError('evaluator changed')
    started = time.perf_counter()
    write(OUT / 'execution_resumed.json', {'started_utc': utc(), 'pid': os.getpid(), 'request': record(OUT / 'request.json'), 'interrupted_run': record(OUT / 'failure.json'), 'resume_source': record(Path(__file__)), 'reason': 'All first two arms sampled; merge candidate and execute frozen official105. No image regeneration or changed method.'})
    phases = []
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               **request['evaluator_environment'])
    for name, mode, steps in ARMS:
        arm = OUT / name
        if name in {'official100', 'potential100'}:
            phase = json.loads((arm / 'outer_sampling.json').read_text())
            if len(phase['workers']) != SHARDS or any(w['returncode'] for w in phase['workers']):
                raise ValueError('previous sampling phase did not complete')
            for rank in range(SHARDS):
                worker = json.loads((arm / f'shard{rank}/summary.json').read_text())
                if worker.get('complete') is not True:
                    raise ValueError('previous worker incomplete')
            if name == 'official100':
                merged = json.loads((arm / 'merged/summary.json').read_text())
                if merged.get('complete') is not True or record(merged['sample_archive']['path']) != merged['sample_archive']:
                    raise ValueError('completed merge identity mismatch')
                phase['merge'] = json.loads((arm / 'merge.process.json').read_text())
            else:
                command = [PYTHON, '-u', str(ROOT / 'experiments/merge_raev2_observable_potential.py'),
                           '--num-samples', str(SAMPLES), '--output-dir', str(arm / 'merged'),
                           '--shards', *[str(arm / f'shard{r}') for r in range(SHARDS)]]
                phase['merge'] = checked_run(command, arm / 'merge_resume.log', dict(env, CUDA_VISIBLE_DEVICES=''))
            phases.append(phase)
            print(json.dumps({'phase': name, 'status': 'completed samples reused and merged'}), flush=True)
            continue
        arm.mkdir()
        active, completed = [], []
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        phase_started = time.perf_counter()
        for rank in range(SHARDS):
            command = [PYTHON, '-u', str(ROOT / 'experiments/sample_raev2_observable_potential.py'),
                       '--mode', mode, '--output-dir', str(arm / f'shard{rank}'),
                       '--potential-checkpoint', str(CHECKPOINT), '--num-steps', str(steps),
                       '--num-samples', str(SAMPLES), '--seed', str(SEED),
                       '--shard-index', str(rank), '--num-shards', str(SHARDS)]
            log_path = arm / f'shard{rank}.log'
            log = log_path.open('x')
            worker_started, worker_utc = time.perf_counter(), utc()
            p = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)), stdout=log, stderr=subprocess.STDOUT)
            WORKERS.append((p, log))
            active.append((rank, p, log, log_path, worker_started, worker_utc, command))
        print(json.dumps({'phase': name, 'status': 'sampling', 'pids': [x[1].pid for x in active]}), flush=True)
        while active:
            for worker in active[:]:
                rank, p, log, log_path, worker_started, worker_utc, command = worker
                if p.poll() is None:
                    continue
                log.close()
                completed.append({'rank': rank, 'returncode': p.returncode, 'pid': p.pid,
                                  'started_utc': worker_utc, 'exit_observed_utc': utc(),
                                  'outer_elapsed_seconds_including_poll_lag': time.perf_counter()-worker_started,
                                  'poll_interval_seconds': 1, 'command': command, 'log': record(log_path)})
                active.remove(worker)
                write(arm / 'outer_workers.json', completed)
            if active:
                time.sleep(1)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        phase = {'arm': name, 'sampling_phase_elapsed_seconds': time.perf_counter()-phase_started,
                 'sampling_child_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                 'workers': completed}
        write(arm / 'outer_sampling.json', phase)
        if any(w['returncode'] for w in completed):
            raise RuntimeError(f'sampling failed: {name}; no evaluation')
        command = [PYTHON, '-u', str(ROOT / 'experiments/merge_raev2_observable_potential.py'),
                   '--num-samples', str(SAMPLES), '--output-dir', str(arm / 'merged'),
                   '--shards', *[str(arm / f'shard{r}') for r in range(SHARDS)]]
        phase['merge'] = checked_run(command, arm / 'merge.log', dict(env, CUDA_VISIBLE_DEVICES=''))
        phases.append(phase)
        write(OUT / 'progress.json', {'completed_arms': [x['arm'] for x in phases], 'fid_performed': False})
        print(json.dumps({'phase': name, 'status': 'merged'}), flush=True)
    # Verify every initial batch before allowing the common evaluator to see images.
    signatures, batch_maps = [], []
    for name, _, _ in ARMS:
        merge_request = json.loads((OUT / name / 'merged/request.json').read_text())
        common = merge_request['common_request']
        signatures.append({k: v for k, v in common.items() if k not in {'mode', 'num_steps', 'time_grid'}})
        batches = {}
        for rank in range(SHARDS):
            manifest = json.loads((OUT / name / f'shard{rank}/batch_manifest.json').read_text())
            for batch in manifest['batches']:
                ids = tuple(batch['global_ids'])
                if ids in batches:
                    raise ValueError('duplicate batch')
                batches[ids] = batch['noise_sha256']
        batch_maps.append(batches)
    if not signatures[0] == signatures[1] == signatures[2] or not batch_maps[0] == batch_maps[1] == batch_maps[2] or len(batch_maps[0]) != SAMPLES // 8:
        raise ValueError('paired full protocol or batch noise mismatch')
    evaluation = OUT / 'evaluation'
    evaluation.mkdir()
    for asset in request['evaluator_assets'].values():
        if record(asset['path']) != asset:
            raise ValueError('evaluator asset changed during sampling')
    command = [PYTHON, '-u', str(ROOT / 'experiments/evaluate_raev2_official_samples.py'),
               '--batch-size', '64', '--seed', '2020', '--output', str(evaluation / 'metrics.csv')]
    for name, _, _ in ARMS:
        command += ['--branch', f'{name}_seed{SEED}={OUT / name / "merged/samples.npz"}']
    write(evaluation / 'request.json', {'created_utc': utc(), 'command': command,
          'all_625_batch_noises_match_across_three_arms': True, 'full_common_protocol_matches_except_mode_and_steps': True,
          'input_archives': {name: record(OUT / name / 'merged/samples.npz') for name, _, _ in ARMS},
          'evaluator_commit': commit, 'fid_protocol': request['fid_protocol'],
          'evaluator_assets': request['evaluator_assets'], 'evaluator_environment': request['evaluator_environment']})
    evaluation_cost = checked_run(command, evaluation / 'evaluation.log', dict(env, CUDA_VISIBLE_DEVICES='0'))
    write(OUT / 'execution_summary.json', {'complete': True, 'finished_utc': utc(), 'phases': phases,
          'evaluation_process': evaluation_cost, 'metrics': record(evaluation / 'metrics.json'),
          'outer_resume_wall_seconds_after_identity_checks': time.perf_counter()-started, 'full_cost_comparison_complete': False,
          'goal_achieved': False, 'interpretation': 'Fixed 5K scale audit only; no independent confirmation or full preparation-cost comparison.'})
    print((evaluation / 'metrics.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'execute'])
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        def stop_on_signal(signum, frame):
            raise KeyboardInterrupt(f'received signal {signum}')
        signal.signal(signal.SIGTERM, stop_on_signal)
        try:
            execute()
        except BaseException as error:
            for process, _ in WORKERS:
                if process.poll() is None:
                    process.terminate()
            for process, log in WORKERS:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            write(OUT / 'resume_failure.json', {'utc': utc(), 'error': f'{type(error).__name__}: {error}'})
            raise

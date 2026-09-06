#!/usr/bin/env python3
"""Sequential, same-GPU fixed reflection screen; cost selection precedes FID."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/affine_reflection_v1')
GPU = 'GPU-7d3e4e7d-abfa-e06e-c264-796052797949'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def costs(summary):
    if not summary['complete'] or summary['samples'] != 1000:
        raise ValueError('incomplete sampling cannot enter cost comparison')
    inference = summary['trajectory_wall_seconds']+summary['decode_and_uint8_wall_seconds']
    total = summary['total_wall_seconds_before_summary']
    if not all(math.isfinite(x) and x > 0 for x in (inference, total)) or total < inference:
        raise ValueError('invalid measured cost')
    return inference, total


def initial_steps(candidate, baseline):
    return max(200, *(math.ceil(100*a/b) for a, b in zip(candidate, baseline)))


def next_steps(k, candidate, baseline):
    return max(k+1, *(math.ceil(k*a/b) for a, b in zip(candidate, baseline)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--parity-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    execution_path = study/'screen_execution.json'
    if execution_path.exists():
        raise FileExistsError('refusing to restart an existing execution')
    plan_path = study/'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = sha(plan_path)
    parity = json.loads((args.parity_dir/'summary.json').read_text())
    if not parity['complete']:
        raise ValueError('completed native parity is required')
    logs = study/'screen_logs'
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TORCH_HOME='/home/zhoushunyu/.cache/torch')
    state = {'complete': False, 'driver_pid': os.getpid(), 'started_utc': stamp(),
             'plan_sha256': plan_sha, 'gpu_uuid': GPU, 'jobs': [], 'cost_selections': [],
             'fid_started': False, 'no_automatic_restart': True}
    put(execution_path, state)
    began = time.perf_counter()

    def verify():
        if sha(plan_path) != plan_sha:
            raise ValueError('plan changed during the experiment')
        for path, expected in plan['frozen_files'].items():
            p = Path(path)
            p = p if p.is_absolute() else ROOT/p
            if sha(p) != expected:
                raise ValueError('frozen file changed: '+str(p))

    def run(name, argv):
        verify()
        path = logs/(name+'.log')
        if path.exists():
            raise FileExistsError(path)
        telemetry = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                    '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
        job = {'name': name, 'argv': argv, 'started_utc': stamp(), 'log': str(path),
               'gpu_before': telemetry, 'exit_code': None}
        state['jobs'].append(job)
        with path.open('wb') as stream:
            start = time.perf_counter()
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            job['pid'] = child.pid
            put(execution_path, state)
            while child.poll() is None:
                time.sleep(1)
            job.update(exit_code=child.returncode, outer_wall_seconds=time.perf_counter()-start,
                       finished_utc=stamp())
        put(execution_path, state)
        if child.returncode:
            raise RuntimeError(f'{name} failed: {child.returncode}; inspect {path}, do not restart blindly')

    def sample(name, mode, steps):
        output = study/name
        run(name, [sys.executable, str(ROOT/'experiments/sample_raev2_affine_reflection.py'),
                   '--mode', mode, '--output-dir', str(output), '--parity-dir', str(args.parity_dir.resolve()),
                   '--num-steps', str(steps)])
        summary_path = output/'summary.json'
        result = json.loads(summary_path.read_text())
        costs(result)
        return result, {'path': str(summary_path), 'sha256': sha(summary_path)}

    try:
        official, official_record = sample('official100', 'official', 100)
        candidate, candidate_record = sample('reflection100', 'reflection', 100)
        tc, wc = costs(candidate)
        k = initial_steps((tc, wc), costs(official))
        identity_keys = ('global_noise_sha256', 'noise_rng_state_sha256', 'global_labels_sha256')
        for key in identity_keys:
            if official[key] != candidate[key]:
                raise ValueError('paired identity mismatch: '+key)
        previous_record = official_record
        while True:
            selection = {'selected_utc': stamp(), 'selected_steps': k,
                         'candidate': candidate_record, 'previous_official': previous_record,
                         'candidate_T_W': [tc, wc], 'fid_started': False,
                         'rule': 'K>=200; initial ceil100*cost ratios; subsequent monotone ceilK*ratios; both trajectory+decode and total wall'}
            selection_path = study/f'cost_selection_{len(state["cost_selections"]):02d}.json'
            if selection_path.exists():
                raise FileExistsError(selection_path)
            put(selection_path, selection)
            state['cost_selections'].append({'path': str(selection_path), 'sha256': sha(selection_path)})
            put(execution_path, state)
            matched, matched_record = sample(f'official{k}', 'official', k)
            for key in identity_keys:
                if matched[key] != candidate[key]:
                    raise ValueError('cost baseline paired identity mismatch: '+key)
            tk, wk = costs(matched)
            if tk >= tc and wk >= wc:
                break
            previous_record = matched_record
            k = next_steps(k, (tc, wc), (tk, wk))
        match = {'complete': True, 'frozen_before_fid': True, 'final_official_steps': k,
                 'official100': official_record, 'reflection100': candidate_record,
                 'official_cost': matched_record, 'T_W_candidate': [tc, wc],
                 'T_W_official_cost': [tk, wk], 'relative_budget_excess': [tk/tc-1, wk/wc-1],
                 'stage2_nfe_candidate': 200, 'stage2_nfe_baseline': k,
                 'extra_training_or_reference_encoding': False,
                 'same_gpu_uuid': GPU, 'timings_are_observed_point_estimates': True,
                 'selection_records': state['cost_selections']}
        put(study/'cost_match.json', match)
        state.update(cost_match_sha256=sha(study/'cost_match.json'), fid_started=True)
        put(execution_path, state)
        run('evaluation', [sys.executable, str(ROOT/'experiments/evaluate_raev2_official_samples.py'),
                          '--branch', f'official100={study}/official100/samples.npz',
                          '--branch', f'reflection100={study}/reflection100/samples.npz',
                          '--branch', f'official{k}={study}/official{k}/samples.npz',
                          '--output', str(study/'official_evaluation.csv'),
                          '--batch-size', '64', '--device', 'cuda', '--seed', '2020',
                          '--feature-cache-dir', str(study/'official_feature_cache')])
        state.update(complete=True, finished_utc=stamp(), driver_wall_seconds=time.perf_counter()-began,
                     evaluation_sha256=sha(study/'official_evaluation.json'), goal_complete=False)
        put(execution_path, state)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stopped_utc=stamp(),
                     driver_wall_seconds=time.perf_counter()-began)
        put(execution_path, state)
        raise


if __name__ == '__main__':
    main()

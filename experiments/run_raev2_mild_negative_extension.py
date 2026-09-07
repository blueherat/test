#!/usr/bin/env python3
"""Freeze and run four additional fixed 1K cohorts; never evaluate or restart.

prepare: CPU source/data identity audit and immutable plans.
launch: start four detached workers after reviewing plan.json.
worker: one GPU, nine sequential jobs, persistent process/exit/timing records.
Use --plan-sha256 with launch/worker to bind the reviewed root plan.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
OUTPUT = RESTART / 'scale_extension_5k_v1'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
DOCUMENT = ROOT / 'docs/RAEV2_MILD_NEGATIVE_5K_EXTENSION_PROTOCOL_20260907_ZH.md'
SEEDS = (202609171, 202609172, 202609173, 202609174)
SPATIAL = ROOT / 'experiments/sample_raev2_spatial_energy_balls_extension.py'
REFLECTION = ROOT / 'experiments/sample_raev2_affine_reflection_extension.py'
PROXIMAL = ROOT / 'experiments/sample_raev2_proximal_calibration.py'
ENERGY = ROOT / 'experiments/sample_raev2_energy_ball_guidance.py'
GLOBAL_CALIBRATION = RESTART / 'proximal_global_c_seed202609065/merged/calibration.pt'
ENERGY_CALIBRATION = RESTART / 'proximal_calibration_seed202609062/merged/calibration.pt'
PROTOCOL = 'raev2_mild_negative_fixed_1k_to_5k_extension_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path).absolute()
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'sha256': digest(path), 'size_bytes': path.stat().st_size}


def write_json(path, value, *, replace=False):
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name = [item.strip() for item in line.split(',', 2)]
        rows.append({'index': int(index), 'uuid': uuid, 'name': name})
    if [row['index'] for row in rows] != [0, 1, 2, 3] or any('4090' not in row['name'] for row in rows):
        raise ValueError(f'expected physical GPU0..3 RTX4090: {rows}')
    return rows


def read_verified(record):
    path = Path(record['path'])
    if digest(path) != record['sha256']:
        raise ValueError(f'frozen file changed: {path}')
    return json.loads(path.read_text())


def verify_sources(plan):
    for record in plan['frozen_sources'].values():
        if digest(record['path']) != record['sha256']:
            raise ValueError(f'frozen source changed: {record["path"]}')


def make_jobs(index, seed, spatial_manifest, reflection_manifest):
    cohort = OUTPUT / f'cohort_{index}'
    def wrapped(script, manifest, mode, name, parity=None, steps=100):
        argv = [PYTHON, str(script), '--extension-manifest', manifest['path'],
                '--extension-manifest-sha256', manifest['sha256'], '--cohort-index', str(index),
                '--mode', mode, '--output-dir', str(cohort / name), '--num-steps', str(steps)]
        if parity:
            argv += ['--parity-dir', str(cohort / parity)]
        return {'name': name, 'argv': argv, 'output_dir': str(cohort / name)}
    def legacy(script, mode, name, extra=()):
        argv = [PYTHON, str(script), '--mode', mode, '--output-dir', str(cohort / name),
                '--sample-count', '1000', '--batch-size', '8', '--seed', str(seed),
                '--precision', 'bf16', *extra]
        return {'name': name, 'argv': argv, 'output_dir': str(cohort / name)}
    return [
        wrapped(SPATIAL, spatial_manifest, 'parity', 'spatial_parity16'),
        wrapped(REFLECTION, reflection_manifest, 'parity', 'reflection_parity16'),
        legacy(PROXIMAL, 'official', 'legacy_official100'),
        legacy(PROXIMAL, 'proximal', 'global_proximal100', ['--calibration', str(GLOBAL_CALIBRATION)]),
        legacy(ENERGY, 'energy_ball', 'legacy_energy100',
               ['--calibration-parent', str(ENERGY_CALIBRATION),
                '--paired-official', str(cohort / 'legacy_official100')]),
        wrapped(REFLECTION, reflection_manifest, 'official', 'native_official100', 'reflection_parity16'),
        wrapped(SPATIAL, spatial_manifest, 'global_ball', 'native_global100', 'spatial_parity16'),
        wrapped(REFLECTION, reflection_manifest, 'reflection', 'reflection100', 'reflection_parity16'),
        wrapped(REFLECTION, reflection_manifest, 'official', 'native_official201', 'reflection_parity16', 201),
    ]


def prepare():
    started = time.perf_counter()
    if OUTPUT.exists():
        raise FileExistsError(f'refusing to overwrite preparation or existing outputs: {OUTPUT}')
    gpus = gpu_inventory()
    spatial_parent = artifact(RESTART / 'spatial_energy_balls_v1/plan.json')
    reflection_parent = artifact(RESTART / 'affine_reflection_v1/plan.json')
    spatial = read_verified(spatial_parent)
    reflection = read_verified(reflection_parent)
    document = artifact(DOCUMENT)
    identities = {}
    def freeze(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        key = str(path.resolve())
        if key not in identities:
            identities[key] = artifact(path)
        if expected is not None and identities[key]['sha256'] != expected:
            raise ValueError(f'historical frozen identity changed: {path}')
        return identities[key]
    for path, expected in reflection['frozen_files'].items():
        freeze(path, expected['sha256'] if isinstance(expected, dict) else expected)
    for path in (Path(__file__), DOCUMENT, SPATIAL, REFLECTION, PROXIMAL, ENERGY,
                 GLOBAL_CALIBRATION, ENERGY_CALIBRATION, Path(spatial['reference']['path'])):
        freeze(path)
    historical = {}
    arms = {
        'legacy_official100': RESTART / 'proximal_seed202609066/official',
        'global_proximal100': RESTART / 'proximal_seed202609066/global',
        'legacy_energy100': RESTART / 'energy_ball_seed202609066/energy_ball',
        'spatial_official100': RESTART / 'spatial_energy_balls_v1/official100',
        'native_global100': RESTART / 'spatial_energy_balls_v1/global100',
        'reflection_official100': RESTART / 'affine_reflection_v1/official100',
        'reflection100': RESTART / 'affine_reflection_v1/reflection100',
        'reflection_official200_cost_only': RESTART / 'affine_reflection_v1/official200',
        'reflection_official201': RESTART / 'affine_reflection_v1/official201',
    }
    for name, directory in arms.items():
        records = {filename: freeze(directory / filename) for filename in
                   ('request.json', 'summary.json', 'batch_manifest.json', 'samples.npz')}
        request = json.loads((directory / 'request.json').read_text())
        for source, expected in request.get('source_sha256', {}).items():
            freeze(source, expected)
        for source, record in request.get('sources', {}).items():
            freeze(ROOT / source, record['sha256'])
        for optional in ('noise_manifest.json', 'sampling_input.json'):
            if (directory / optional).exists():
                records[optional] = freeze(directory / optional)
        historical[name] = {'directory': str(directory), 'artifacts': records}
    # Every known sampler dependency and reference is checked once before GPU.
    for path in (Path(spatial['reference']['path']).parent / 'request.json',
                 Path(spatial['reference']['path']).parent / 'summary.json'):
        freeze(path)
    source_identities = {name: value for name, value in identities.items()
                         if Path(value['path']).suffix in ('.py', '.yaml', '.md')}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for family, parent, parent_record, script, indices in (
        ('spatial', spatial, spatial_parent, SPATIAL, range(1, 5)),
        ('reflection', reflection, reflection_parent, REFLECTION, range(0, 5)),
    ):
        cohorts = []
        for index in indices:
            seed = SEEDS[index - 1] if index else reflection['cohort']['seed']
            plan = copy.deepcopy(parent)
            plan.update({'created_at_utc': utc(), 'extension_cohort_index': index,
                         'historical_plan': parent_record, 'protocol_document': document})
            plan['cohort']['seed'] = seed
            if family == 'reflection':
                plan['gpu_uuid'] = gpus[index - 1]['uuid'] if index else gpus[3]['uuid']
                plan['frozen_files'][str(script)] = digest(script)
                plan['frozen_files'][str(DOCUMENT)] = document['sha256']
            else:
                plan['state'] = 'extension_pre_gpu_frozen'
            path = OUTPUT / 'plans' / f'{family}_cohort_{index}.json'
            write_json(path, plan)
            cohorts.append({'index': index, 'seed': seed, 'plan': artifact(path)})
        manifest = {'created_at_utc': utc(),
                    'protocol': ('raev2_fixed_1k_cohort_spatial_energy_balls_extension_v1' if family == 'spatial'
                                 else 'raev2_fixed_1k_cohort_affine_reflection_extension_v1'),
                    'historical_plan': parent_record,
                    'historical_sampler_sha256': digest(ROOT / 'experiments' / ('sample_raev2_spatial_energy_balls.py' if family == 'spatial' else 'sample_raev2_affine_reflection.py')),
                    'extension_wrapper_sha256': digest(script),
                    ('additional_cohorts' if family == 'spatial' else 'cohorts'): cohorts}
        path = OUTPUT / f'{family}_extension_manifest.json'
        write_json(path, manifest)
        manifests[family] = artifact(path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.sample_raev2_spatial_energy_balls_extension import validate_extension as spatial_validate
    from experiments.sample_raev2_affine_reflection_extension import validate_extension as reflection_validate
    for index in range(1, 5):
        spatial_validate(manifests['spatial'], index)
    for index in range(5):
        reflection_validate(manifests['reflection'], index)
    plan = {'protocol': PROTOCOL, 'created_at_utc': utc(), 'protocol_document': document,
            'python': PYTHON, 'root': str(ROOT), 'output': str(OUTPUT), 'seeds': list(SEEDS),
            'source_freeze_complete': True, 'frozen_sources': source_identities,
            'frozen_files': identities, 'historical_1k': historical, 'manifests': manifests,
            'cohorts': [{'index': index, 'seed': seed, 'gpu': gpus[index - 1],
                         'jobs': make_jobs(index, seed, manifests['spatial'], manifests['reflection'])}
                        for index, seed in enumerate(SEEDS, 1)],
            'evaluation': 'none in this driver; all cohorts and cost decisions finish before uniform FID',
            'no_automatic_restart': True, 'prepare_wall_seconds': time.perf_counter() - started}
    write_json(OUTPUT / 'plan.json', plan)
    print(json.dumps({'prepared': artifact(OUTPUT / 'plan.json'), 'launch_performed': False,
                      'frozen_file_count': len(identities), 'cohorts': 4, 'jobs_per_cohort': 9}))


def load_plan(expected):
    plan = read_verified({'path': str(OUTPUT / 'plan.json'), 'sha256': expected})
    if plan['protocol'] != PROTOCOL or not plan['source_freeze_complete']:
        raise ValueError('unexpected or incomplete root plan')
    verify_sources(plan)
    return plan


def launch(expected):
    plan = load_plan(expected)
    record_path = OUTPUT / 'launch.json'
    lock = OUTPUT / 'launch.lock'
    with lock.open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'created_at_utc': utc()}))
    record = {'plan_sha256': expected, 'launcher_pid': os.getpid(), 'started_at_utc': utc(),
              'workers': [], 'launch_complete': False}
    write_json(record_path, record)
    try:
        for cohort in plan['cohorts']:
            index = cohort['index']
            directory = OUTPUT / f'cohort_{index}'
            directory.mkdir(exist_ok=False)
            argv = [PYTHON, str(Path(__file__).absolute()), 'worker', '--cohort-index', str(index),
                    '--plan-sha256', expected]
            environment = dict(os.environ)
            environment.update({'CUDA_VISIBLE_DEVICES': cohort['gpu']['uuid'], 'PYTHONUNBUFFERED': '1',
                                'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                'TORCH_HOME': '/home/zhoushunyu/.cache/torch'})
            log_path = directory / 'worker.log'
            with log_path.open('xb') as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            record['workers'].append({'cohort_index': index, 'pid': process.pid, 'argv': argv,
                                      'gpu': cohort['gpu'], 'launched_at_utc': utc(), 'log': str(log_path)})
            write_json(record_path, record, replace=True)
        record['launch_complete'] = True
        record['finished_at_utc'] = utc()
        write_json(record_path, record, replace=True)
        print(json.dumps(record))
    except BaseException:
        record['error'] = traceback.format_exc()
        write_json(record_path, record, replace=True)
        raise


def worker(expected, index):
    plan = load_plan(expected)
    cohort = next(row for row in plan['cohorts'] if row['index'] == index)
    directory = OUTPUT / f'cohort_{index}'
    directory.mkdir(exist_ok=True)
    with (directory / 'worker.lock').open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'created_at_utc': utc()}))
    state_path = directory / 'execution.json'
    started = time.perf_counter()
    state = {'protocol': PROTOCOL, 'plan_sha256': expected, 'cohort_index': index,
             'seed': cohort['seed'], 'worker_pid': os.getpid(), 'worker_ppid': os.getppid(),
             'gpu': cohort['gpu'], 'started_at_utc': utc(), 'jobs': [], 'complete': False,
             'terminal': False, 'fid_started': False}
    write_json(state_path, state)
    try:
        actual_gpu = next(row for row in gpu_inventory() if row['index'] == cohort['gpu']['index'])
        if actual_gpu != cohort['gpu'] or os.environ.get('CUDA_VISIBLE_DEVICES') != actual_gpu['uuid']:
            raise ValueError('worker physical GPU identity or visibility differs from frozen plan')
        for job in cohort['jobs']:
            verify_sources(plan)
            if Path(job['output_dir']).exists():
                raise FileExistsError(f'refusing restart or overwrite: {job["output_dir"]}')
            log_path = directory / f'{job["name"]}.log'
            entry = {**job, 'started_at_utc': utc(), 'gpu': actual_gpu, 'log': str(log_path)}
            state['jobs'].append(entry)
            write_json(state_path, state, replace=True)
            job_started = time.perf_counter()
            with log_path.open('xb') as log:
                process = subprocess.Popen(job['argv'], cwd=ROOT, env=os.environ.copy(), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = process.pid
                write_json(state_path, state, replace=True)
                code = process.wait()
            entry.update({'exit_code': code, 'finished_at_utc': utc(),
                          'outer_wall_seconds': time.perf_counter() - job_started})
            write_json(state_path, state, replace=True)
            if code != 0:
                raise RuntimeError(f'{job["name"]} exited {code}; preserve partial outputs, no restart')
            summary = Path(job['output_dir']) / 'summary.json'
            if not summary.is_file():
                raise RuntimeError(f'{job["name"]} returned success without a summary')
            entry['summary'] = artifact(summary)
            write_json(state_path, state, replace=True)
        state['complete'] = True
    except BaseException:
        state['error'] = traceback.format_exc()
        raise
    finally:
        state.update({'terminal': True, 'finished_at_utc': utc(),
                      'worker_outer_wall_seconds': time.perf_counter() - started})
        write_json(state_path, state, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'launch', 'worker'))
    parser.add_argument('--plan-sha256')
    parser.add_argument('--cohort-index', type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    if args.command != 'prepare' and not args.plan_sha256:
        parser.error('launch/worker require --plan-sha256 from the reviewed prepared plan')
    if args.command == 'worker' and args.cohort_index is None:
        parser.error('worker requires --cohort-index')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'launch':
        launch(args.plan_sha256)
    else:
        worker(args.plan_sha256, args.cohort_index)


if __name__ == '__main__':
    main()

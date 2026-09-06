#!/usr/bin/env python3
"""Run the fixed complete three-arm CPU analysis with outer process accounting."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent/'observable_potential_scale_audit_v1'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'


def record(path):
    path = Path(path).resolve()
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes':path.stat().st_size}


def write(path, payload):
    with path.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def main():
    request_path = HERE/'results_v1.process.request.json'
    if request_path.exists() or (HERE/'results_v1').exists():
        raise FileExistsError('analysis already started; do not automatically rerun')
    execution = json.loads((AUDIT/'execution_summary.json').read_text())
    if execution.get('complete') is not True or execution['evaluation_process']['returncode'] != 0:
        raise ValueError('complete common evaluation required')
    evaluation = json.loads((AUDIT/'evaluation/request.json').read_text())
    rows = json.loads((AUDIT/'evaluation/metrics.json').read_text())
    command = [PYTHON, '-u', str(HERE/'run_scale_analysis.py')]
    for name in ('official100', 'potential100', 'official105'):
        matches = [r for r in rows if r['branch'] == f'{name}_seed202609101']
        if len(matches) != 1:
            raise ValueError('missing/duplicate fixed metric row')
        row = matches[0]
        cache = AUDIT/'evaluation/official_feature_cache'/f"{row['branch']}-{row['sample_sha256'][:16]}-inception.features.pt"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        command += ['--arm', name, str(cache), str(AUDIT/name/'merged/samples.npz'), str(AUDIT/name/'merged/summary.json')]
    command += ['--evaluation-request', str(AUDIT/'evaluation/request.json'),
                '--metrics-json', str(AUDIT/'evaluation/metrics.json'),
                '--execution-summary', str(AUDIT/'execution_summary.json'),
                '--reference', evaluation['evaluator_assets']['reference']['path'],
                '--output-dir', str(HERE/'results_v1')]
    overrides = {'CUDA_VISIBLE_DEVICES':'', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'OMP_NUM_THREADS':'4'}
    write(request_path, {'command':command, 'environment_overrides':overrides,
                        'source_records':[record(__file__), record(HERE/'run_scale_analysis.py')],
                        'created_utc':datetime.now(timezone.utc).isoformat()})
    log_path = HERE/'results_v1.process.log'
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    with log_path.open('x') as log:
        result = subprocess.run(command, cwd=HERE, env=dict(os.environ, **overrides), stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter()-start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cost = {'returncode':result.returncode, 'started_utc':started_utc,
            'finished_utc':datetime.now(timezone.utc).isoformat(), 'child_wall_seconds_including_imports_and_exit':wall,
            'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            'child_maximum_resident_set_kib':after.ru_maxrss, 'gpu_calls':0, 'model_calls':0,
            'request':record(request_path), 'log':record(log_path)}
    write(HERE/'results_v1.process.json', cost)
    print(json.dumps(cost), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

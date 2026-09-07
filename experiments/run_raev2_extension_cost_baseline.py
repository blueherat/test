#!/usr/bin/env python3
"""Conditional, immutable uniform-K official baselines after the initial cost audit.

Never evaluates images, launches candidates, restarts, or replaces frozen plans.
prepare requires a reviewed completed cost-summary SHA; launch additionally
requires the SHA of this helper's resulting plan. GPU3 runs cohort4 then cohort0.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_raev2_mild_negative_extension as base
from experiments import audit_raev2_mild_negative_extension_cost as audit
from experiments.sample_raev2_affine_reflection_extension import validate_extension

EXTENSION = base.OUTPUT
DEFAULT_OUTPUT = EXTENSION / 'cost_baseline_v1'
PROTOCOL = 'raev2_extension_conditional_uniform_official_cost_baseline_v1'
utc, artifact, read_verified, write_json = base.utc, base.artifact, base.read_verified, base.write_json


def require_decision(summary):
    if (summary.get('complete') is not True
            or summary.get('protocol') != 'raev2_mild_negative_extension_cost_review_v1'
            or summary.get('evaluation_released') is not False
            or summary.get('new_jobs_launched') is not False):
        raise ValueError('a completed initial cost-only audit is required')
    d = summary['reflection201_decisions']['pooled5k_protocol_decision']
    if d.get('official_steps') != 201 or d.get('paired_cohorts') != 5:
        raise ValueError('only the initial pooled-five-cohort official201 decision is accepted')
    k = d.get('suggested_uniform_steps')
    covered = d.get('both_recorded_metrics_covered')
    if type(covered) is not bool or type(k) is not int or k < 201:
        raise ValueError('malformed initial cost decision')
    if d.get('requires_five_new_official_cohorts') is not (not covered):
        raise ValueError('inconsistent conditional launch flag')
    if (covered and k != 201) or (not covered and k <= 201):
        raise ValueError('inconsistent uniform step suggestion')
    return None if covered else k


def verify_parity(directory, manifest, cohort_row):
    directory = Path(directory)
    request_record, summary_record = artifact(directory / 'request.json'), artifact(directory / 'summary.json')
    request, summary = read_verified(request_record), read_verified(summary_record)
    index, seed = cohort_row['index'], cohort_row['seed']
    if (summary.get('complete') is not True or summary.get('command') != 'parity'
            or summary.get('endpoint_bitwise') is not True or summary.get('pixel_bitwise') is not True
            or summary.get('samples_per_loop') != 16 or summary.get('num_steps') != 100
            or summary.get('seed') != seed or summary.get('request', {}).get('sha256') != request_record['sha256']
            or request.get('command') != 'parity' or request.get('sample_count') != 16
            or request.get('batch_size') != 8 or request.get('num_steps') != 100
            or request.get('seed') != seed or request.get('extension_cohort_index') != index):
        raise ValueError(f'mismatched completed native parity: {directory}')
    for actual, expected in [(request['extension_manifest'], manifest),
                             (request['identities']['root_plan'], cohort_row['plan'])]:
        if actual['sha256'] != expected['sha256'] or Path(actual['path']).resolve() != Path(expected['path']).resolve():
            raise ValueError('parity belongs to a different extension manifest/cohort plan')
    records = [request_record, summary_record]
    for name in ('candidate_formula_check', 'geometry_diagnostics', 'parity_result'):
        record = artifact(summary[name]['path'])
        if record['sha256'] != summary[name]['sha256']:
            raise ValueError('changed parity diagnostic artifact')
        records.append(record)
    return records


def make_lanes(extension, output, root_plan, manifest, reflection_rows, k):
    rows = {row['index']: row for row in reflection_rows}
    lanes = []
    for new_cohort in root_plan['cohorts']:
        indices = [new_cohort['index']] + ([0] if new_cohort['index'] == 4 else [])
        jobs = []
        for index in indices:
            row = rows[index]
            directory = output / f'cohort_{index}'
            parity = (extension / f'cohort_{index}/reflection_parity16' if index else
                      directory / 'reflection_parity16')
            for mode, steps, target in ([('parity', 100, parity)] if index == 0 else []) + [
                    ('official', k, directory / f'official{k}')]:
                argv = [base.PYTHON, str(base.REFLECTION), '--extension-manifest', manifest['path'],
                        '--extension-manifest-sha256', manifest['sha256'], '--cohort-index', str(index),
                        '--mode', mode, '--output-dir', str(target), '--num-steps', str(steps)]
                if mode == 'official':
                    argv += ['--parity-dir', str(parity)]
                jobs.append({'name': f'cohort_{index}_{mode}{steps}', 'cohort_index': index,
                             'seed': row['seed'], 'mode': mode, 'num_steps': steps,
                             'output_dir': str(target), 'parity_dir': str(parity), 'argv': argv})
        lanes.append({'gpu': new_cohort['gpu'], 'cohort_indices': indices, 'jobs': jobs})
    return lanes


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f'no overwrite or resume: {args.output}')
    cost_record = {'path': str(EXTENSION / 'cost_review_v1/summary.json'), 'sha256': args.cost_summary_sha256}
    cost = read_verified(cost_record)
    k = require_decision(cost)
    if cost.get('root_plan_sha256') != args.root_plan_sha256:
        raise ValueError('cost decision belongs to another root plan')
    root_record = {'path': str(EXTENSION / 'plan.json'), 'sha256': args.root_plan_sha256}
    root_plan = read_verified(root_record)
    base.verify_sources(root_plan)
    request_record = artifact(EXTENSION / 'cost_review_v1/request.json')
    request = read_verified(request_record)
    if (Path(request['source']['path']).resolve() != Path(audit.__file__).resolve()
            or request['source']['sha256'] != base.digest(audit.__file__)):
        raise ValueError('cost-audit implementation identity changed')
    # These are the cost auditor's metadata/source inputs, not image arrays or FID files.
    inputs = [artifact(cost_record['path']), artifact(root_record['path']), request_record]
    for record in request['inputs'].values():
        if base.digest(record['path']) != record['sha256']:
            raise ValueError(f'changed cost input: {record["path"]}')
        inputs.append(record)
    candidate_rows, official_rows = [], []
    for name, target in [('reflection100', candidate_rows), ('reflection_official201', official_rows)]:
        summary = read_verified(root_plan['historical_1k'][name]['artifacts']['summary.json'])
        target.append({'trajectory_through_decode_wall_seconds': summary['trajectory_through_decode_wall_seconds'],
                       'runner_wall_seconds': summary['total_wall_seconds_before_summary']})
    for cohort in root_plan['cohorts']:
        execution = json.loads((EXTENSION / f'cohort_{cohort["index"]}/execution.json').read_text())
        if (execution.get('complete') is not True or execution.get('terminal') is not True
                or execution.get('fid_started') is not False or execution.get('plan_sha256') != args.root_plan_sha256
                or any(job.get('exit_code') != 0 for job in execution['jobs'])):
            raise ValueError('all original workers must be terminal and successful before prepare')
        for name, target in [('reflection100', candidate_rows), ('native_official201', official_rows)]:
            job = next(job for job in execution['jobs'] if job['name'] == name)
            summary = read_verified(job['summary'])
            target.append({'trajectory_through_decode_wall_seconds': summary['trajectory_through_decode_wall_seconds'],
                           'runner_wall_seconds': summary['total_wall_seconds_before_summary']})
    actual = audit.cost_decision(candidate_rows, official_rows)
    recorded = cost['reflection201_decisions']['pooled5k_protocol_decision']
    if any(actual[key] != recorded[key] for key in ('metrics', 'suggested_uniform_steps', 'both_recorded_metrics_covered')):
        raise ValueError('stored suggestion differs from independent exact pooled T/W recomputation')
    if k is None:
        print(json.dumps({'new_plan_created': False, 'launch_performed': False,
                          'reason': 'official201 already covers both pooled cost metrics'}))
        return
    manifest = root_plan['manifests']['reflection']
    reflection = read_verified(manifest)
    inputs.append(manifest)
    rows = reflection['cohorts']
    for row in rows:
        _, _, (cohort_plan, _, _) = validate_extension(manifest, row['index'])
        gpu = root_plan['cohorts'][row['index']-1 if row['index'] else 3]['gpu']
        if cohort_plan['gpu_uuid'] != gpu['uuid']:
            raise ValueError('frozen cohort GPU is incompatible with the fixed queue')
        inputs.append(row['plan'])
        if row['index']:
            inputs.extend(verify_parity(EXTENSION / f'cohort_{row["index"]}/reflection_parity16', manifest, row))
    lanes = make_lanes(EXTENSION, args.output, root_plan, manifest, rows, k)
    sources = [artifact(path) for path in (Path(__file__), Path(base.__file__), Path(audit.__file__), base.REFLECTION)]
    plan = {'protocol': PROTOCOL, 'created_at_utc': utc(), 'output': str(args.output),
            'uniform_steps': k, 'root_plan': artifact(root_record['path']), 'initial_cost_summary': artifact(cost_record['path']),
            'reflection_manifest': manifest, 'reflection_cohorts': rows, 'sources': sources,
            'inputs': list({str(Path(r['path']).resolve()): r for r in inputs}.values()), 'lanes': lanes,
            'candidate_jobs': 0, 'fid_read_or_started': False, 'automatic_restart': False,
            'scope': 'One conditional uniform-K attempt only; preserve all prior 201/200 samples and costs. A later cost review must establish whether this K covers both pooled metrics; no evaluation release.'}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    print(json.dumps({'plan': artifact(args.output / 'plan.json'), 'launch_performed': False}))


def load_plan(args):
    plan = read_verified({'path': str(args.output / 'plan.json'), 'sha256': args.plan_sha256})
    if plan.get('protocol') != PROTOCOL or Path(plan['output']).resolve() != args.output.resolve():
        raise ValueError('unexpected cost-baseline plan/output')
    for record in [*plan['sources'], *plan['inputs']]:
        if base.digest(record['path']) != record['sha256']:
            raise ValueError(f'prepared input/source changed: {record["path"]}')
    root_plan = read_verified(plan['root_plan'])
    base.verify_sources(root_plan)
    return plan


def launch(args):
    plan = load_plan(args)
    for lane in plan['lanes']:
        for job in lane['jobs']:
            if Path(job['output_dir']).exists():
                raise FileExistsError(f'no overwrite or restart: {job["output_dir"]}')
    with (args.output / 'launch.lock').open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'created_at_utc': utc()}))
    record = {'plan_sha256': args.plan_sha256, 'launcher_pid': os.getpid(), 'started_at_utc': utc(),
              'workers': [], 'launch_complete': False, 'fid_started': False}
    record_path = args.output / 'launch.json'
    write_json(record_path, record)
    try:
        for lane in plan['lanes']:
            gpu = lane['gpu']
            directory = args.output / f'gpu_{gpu["index"]}'
            directory.mkdir(exist_ok=False)
            argv = [base.PYTHON, str(Path(__file__).resolve()), 'worker', '--output', str(args.output),
                    '--plan-sha256', args.plan_sha256, '--gpu-index', str(gpu['index'])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TORCH_HOME='/home/zhoushunyu/.cache/torch')
            log_path = directory / 'worker.log'
            with log_path.open('xb') as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['workers'].append({'gpu': gpu, 'cohort_indices': lane['cohort_indices'], 'argv': argv,
                                      'pid': process.pid, 'launched_at_utc': utc(), 'log': str(log_path)})
            write_json(record_path, record, replace=True)
        record['launch_complete'] = True
    except BaseException:
        record['error'] = traceback.format_exc()
        raise
    finally:
        record['finished_at_utc'] = utc()
        write_json(record_path, record, replace=True)
    print(json.dumps(record))


def worker(args):
    plan = load_plan(args)
    lane = next(lane for lane in plan['lanes'] if lane['gpu']['index'] == args.gpu_index)
    directory = args.output / f'gpu_{args.gpu_index}'
    with (directory / 'worker.lock').open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'created_at_utc': utc()}))
    state_path, started = directory / 'execution.json', time.perf_counter()
    state = {'protocol': PROTOCOL, 'plan_sha256': args.plan_sha256, 'worker_pid': os.getpid(),
             'worker_ppid': os.getppid(), 'gpu': lane['gpu'], 'cohort_indices': lane['cohort_indices'],
             'started_at_utc': utc(), 'jobs': [], 'complete': False, 'terminal': False, 'fid_started': False}
    write_json(state_path, state)
    try:
        actual_gpu = next(g for g in base.gpu_inventory() if g['index'] == args.gpu_index)
        if actual_gpu != lane['gpu'] or os.environ.get('CUDA_VISIBLE_DEVICES') != actual_gpu['uuid']:
            raise ValueError('physical GPU identity/visibility changed')
        for job in lane['jobs']:
            load_plan(args)
            row = next(row for row in plan['reflection_cohorts'] if row['index'] == job['cohort_index'])
            validate_extension(plan['reflection_manifest'], row['index'])
            if job['mode'] == 'official':
                verify_parity(job['parity_dir'], plan['reflection_manifest'], row)
            if Path(job['output_dir']).exists():
                raise FileExistsError(f'no restart: {job["output_dir"]}')
            entry = {**job, 'gpu': actual_gpu, 'started_at_utc': utc(), 'log': str(directory / f'{job["name"]}.log')}
            state['jobs'].append(entry)
            write_json(state_path, state, replace=True)
            job_started = time.perf_counter()
            with Path(entry['log']).open('xb') as log:
                process = subprocess.Popen(job['argv'], cwd=ROOT, env=os.environ.copy(), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = process.pid
                write_json(state_path, state, replace=True)
                code = process.wait()
            entry.update(exit_code=code, finished_at_utc=utc(), outer_wall_seconds=time.perf_counter()-job_started)
            write_json(state_path, state, replace=True)
            if code:
                raise RuntimeError(f'{job["name"]} exited {code}; partial outputs preserved')
            entry['summary'] = artifact(Path(job['output_dir']) / 'summary.json')
            summary = read_verified(entry['summary'])
            if (summary.get('complete') is not True or summary.get('seed') != row['seed']
                    or summary.get('mode') != job['mode'] or summary.get('num_steps') != job['num_steps']):
                raise ValueError('job succeeded without a matching complete summary')
            if job['mode'] == 'parity':
                verify_parity(job['output_dir'], plan['reflection_manifest'], row)
            else:
                counts = summary['forward_counts_observed']
                if summary.get('samples') != 1000 or counts['stage2_forward_calls'] != 125*plan['uniform_steps'] or counts['stage2_sample_forwards'] != 1000*plan['uniform_steps']:
                    raise ValueError('official cohort sample/NFE accounting mismatch')
            write_json(state_path, state, replace=True)
        state['complete'] = True
    except BaseException:
        state['error'] = traceback.format_exc()
        raise
    finally:
        state.update(terminal=True, finished_at_utc=utc(), worker_outer_wall_seconds=time.perf_counter()-started)
        write_json(state_path, state, replace=True)


def self_test():
    import copy
    def summary(t, w):
        rows = lambda t,w: [{'trajectory_through_decode_wall_seconds': t, 'runner_wall_seconds': w}]*5
        return {'complete': True, 'protocol': 'raev2_mild_negative_extension_cost_review_v1',
                'evaluation_released': False, 'new_jobs_launched': False,
                'reflection201_decisions': {'pooled5k_protocol_decision': audit.cost_decision(rows(t,w), rows(1.,1.))}}
    assert require_decision(summary(1.,1.)) is None
    import math
    assert require_decision(summary(1.,math.nextafter(1.,math.inf))) == 202
    for key, value in [('complete', False), ('evaluation_released', True), ('new_jobs_launched', True)]:
        bad = copy.deepcopy(summary(2.,2.)); bad[key] = value
        try: require_decision(bad)
        except ValueError: pass
        else: raise AssertionError(f'failed to reject {key}')
    fake = {'cohorts':[{'index':i,'gpu':{'index':i-1,'uuid':f'gpu{i-1}'}} for i in range(1,5)]}
    rows = [{'index':i,'seed':202609131 if not i else 202609170+i} for i in range(5)]
    lanes = make_lanes(Path('/extension'), Path('/output'), fake, {'path':'/manifest','sha256':'a'*64}, rows, 203)
    assert [lane['cohort_indices'] for lane in lanes] == [[1],[2],[3],[4,0]]
    jobs = [job for lane in lanes for job in lane['jobs']]
    assert len(jobs) == 6 and len({job['output_dir'] for job in jobs}) == 6
    assert [j['mode'] for j in lanes[-1]['jobs']] == ['official','parity','official']
    assert {j['num_steps'] for j in jobs if j['mode']=='official'} == {203}
    assert [j['cohort_index'] for j in jobs if j['mode']=='official'] == [1,2,3,4,0]
    assert lanes[-1]['jobs'][-1]['parity_dir'] == '/output/cohort_0/reflection_parity16'
    assert lanes[0]['jobs'][0]['parity_dir'] == '/extension/cohort_1/reflection_parity16'
    print(json.dumps({'self_test':'passed','actual_plan_created':False,'jobs_launched':False,
                      'scope':'cost completeness/release rejection, binary64 nextafter, uniform five cohorts, parity reuse and cohort0 GPU3 ordering'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','launch','worker','self-test'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cost-summary-sha256')
    parser.add_argument('--root-plan-sha256', default=audit.PLAN_SHA)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--gpu-index', type=int, choices=range(4))
    args = parser.parse_args()
    args.output = args.output.expanduser().absolute()
    if args.command=='prepare' and not args.cost_summary_sha256:
        parser.error('prepare requires --cost-summary-sha256 of the completed reviewed initial audit')
    if args.command in ('launch','worker') and not args.plan_sha256:
        parser.error('launch/worker require --plan-sha256 of this helper prepared plan')
    if args.command=='worker' and args.gpu_index is None:
        parser.error('worker requires --gpu-index')
    {'prepare':prepare,'launch':launch,'worker':worker,'self-test':lambda _:self_test()}[args.command](args)


if __name__ == '__main__':
    main()

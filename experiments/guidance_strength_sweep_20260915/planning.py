"""Mixed original-training DAG and reusable, adaptive strength-search points."""
import math
from . import config as k
from experiments.guidance_loss_50k_20260914 import planning as original


def initial_jobs():
    result = [dict(job, engine='original') for job in original.jobs()]
    result.append(dict(id='sweep_preflight', action='sweep_preflight', gpu=True,
                       depends=['gpu_preflight'], engine='sweep'))
    return result


def point_jobs(point):
    method, tick = k.parse(point)
    source = k.anchor(point)
    if source is not None:
        return [dict(id='reuse_' + point, action='reuse', arm=point, gpu=False,
            depends=['sweep_preflight', f'evaluate_{source}_screen1000'], engine='sweep')]
    deps = ['sweep_preflight']
    if method in k.old.ARMS:
        deps.append('train_' + method)
    first = dict(id='sweep_sample_' + point, action='sweep_sample', arm=point, gpu=True, depends=deps, engine='sweep')
    second = dict(id='sweep_evaluate_' + point, action='sweep_evaluate', arm=point, gpu=False,
                  depends=[first['id']], engine='sweep')
    return [first, second]


def point_job_id(point):
    return point_jobs(point)[-1]['id']


def point_metrics(point):
    return k.ROOT / 'points' / point / 'metrics.json'


def output(job):
    if job.get('engine') == 'original':
        return original.output(job)
    if job['action'] == 'sweep_preflight':
        return k.ROOT / 'gpu_preflight.json'
    return k.ROOT / 'points' / job['arm'] / ('summary.json' if job['action'] == 'sweep_sample' else 'metrics.json')


def verify_output(job):
    if job.get('engine') == 'original':
        return original.verify_output(job)
    path = output(job)
    if not path.exists():
        return False
    row = k.read(path)
    assert row['request_sha256'] == k.sha(k.ROOT / 'request.json')
    if job['action'] == 'sweep_preflight':
        return bool(row['passed'])
    assert row['complete'] and row['point'] == job['arm']
    for group in ('files', 'head_provenance', 'dependencies'):
        for p, digest in row.get(group, {}).items():
            assert k.sha(p) == digest, p
    if row.get('valid', True):
        assert row['primary_samples'] == 1000
        assert row['counts'] == k.counts(job['arm'])
        assert k.sha(row['samples_path']) == row['samples_sha256']
    return True


def best_intervals(points, width):
    """Rank regular adjacent intervals; refine() connects the selected regions."""
    points = sorted(points, key=lambda p: p['tick'])
    choices = []
    for left, right in zip(points, points[1:]):
        if right['tick'] - left['tick'] != width:
            continue
        if not all(p.get('valid', True) and p.get('fid') is not None and math.isfinite(p['fid']) for p in (left, right)):
            continue
        choices.append(dict(left=left['tick'], right=right['tick'],
            mean_fid=(left['fid'] + right['fid']) / 2,
            best_endpoint_fid=min(left['fid'], right['fid'])))
    return sorted(choices, key=lambda x: (x['mean_fid'], x['best_endpoint_fid'], x['left']))[:2]


def refine(points, width, next_width):
    assert width % next_width == 0 and next_width < width
    intervals = best_intervals(points, width)
    ticks = list(range(min(row['left'] for row in intervals),
                       max(row['right'] for row in intervals) + 1, next_width)) if intervals else []
    return intervals, ticks

from . import config as k
from experiments.guidance_strength_sweep_20260915 import planning as previous

original = previous.original
point_metrics = previous.point_metrics
point_job_id = previous.point_job_id
best_intervals = previous.best_intervals
refine = previous.refine


def check_id(rank):
    return f'four_gpu_check_{rank}'


def initial_jobs():
    return previous.initial_jobs() + [dict(id=check_id(rank), action='four_gpu_check',
        fold=rank, gpu=True, depends=['sweep_preflight'], engine='sweep') for rank in range(k.SHARDS)]


def point_jobs(point):
    jobs = previous.point_jobs(point)
    if k.anchor(point) is not None:
        return jobs
    sample, evaluate = jobs
    shards = [dict(id=f'{sample["id"]}__part{rank}', action='sweep_shard', arm=point,
        fold=rank, gpu=True, depends=sample['depends'] + [check_id(i) for i in range(k.SHARDS)],
        engine='sweep') for rank in range(k.SHARDS)]
    return shards + [dict(sample, action='sweep_assemble', gpu=False,
        depends=[job['id'] for job in shards]), evaluate]


def output(job):
    if job['action'] == 'four_gpu_check':
        return k.ROOT / 'four_gpu_checks' / f'{job["fold"]}.json'
    if job['action'] == 'sweep_shard':
        return k.ROOT / 'points' / job['arm'] / 'shards' / f'{job["fold"]}.json'
    if job['action'] == 'sweep_assemble':
        return previous.output(dict(job, action='sweep_sample'))
    return previous.output(job)


def verify_output(job):
    action = job['action']
    if action == 'sweep_assemble':
        return previous.verify_output(dict(job, action='sweep_sample'))
    if action not in ('four_gpu_check', 'sweep_shard'):
        return previous.verify_output(job)
    if action == 'sweep_shard' and previous.verify_output(dict(job, action='sweep_sample')):
        return True
    path = output(job)
    if not path.exists():
        return False
    row = k.read(path)
    assert row['dispatch_sha256'] == k.sha(k.AMENDMENT)
    assert row['request_sha256'] == k.sha(k.ROOT / 'request.json')
    assert row['rank'] == int(job['fold'])
    if action == 'four_gpu_check':
        return bool(row['passed'])
    assert row['complete'] and row['point'] == job['arm']
    if row.get('valid', True):
        assert row['batch_starts'] == list(k.batch_starts(int(job['fold'])))
        assert len(row['records']) == len(row['batch_starts'])
    for group in ('head_provenance', 'files'):
        for filename, digest in row.get(group, {}).items():
            assert k.sha(filename) == digest, filename
    return True

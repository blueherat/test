"""Bounded subprocess checks for parallel admission, dependencies and recovery."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest import mock

from experiments import guidance_loss_parallel_20260914 as p


def main():
    jobs = p.plan()
    assert len(jobs) == 79
    assert sum(j['action'] == 'sample' for j in jobs) == 23
    assert {'train_cfg_residual', 'train_cfg_contrast', 'endpoints_cfg'} <= {j['id'] for j in jobs}
    assert not any(j.get('stage') == 'confirm5000' for j in jobs)
    small = [dict(id='a', depends=[], gpu=True), dict(id='b', depends=['a'], gpu=True),
             dict(id='c', depends=[], gpu=False)]
    ready, blocked = p.classify(small, {'a': {'state': 'running'}})
    assert [j['id'] for j in ready] == ['c'] and not blocked
    ready, blocked = p.classify(small, {'a': {'state': 'failed'}})
    assert [j['id'] for j in ready] == ['c'] and [j['id'] for j in blocked] == ['b']
    result = dict(passed=False, cfg_jobs_retained=True, no_5k_jobs=True,
                  pending_dependencies_not_blocked=True, failure_isolated=True)
    real_root = p.k.ROOT
    real_popen = subprocess.Popen
    real_sleep = time.sleep
    with tempfile.TemporaryDirectory(prefix='eqvae_parallel_check_') as directory:
        root = Path(directory)
        (root / 'request.json').write_text('{}')
        amendment = root / 'dispatch_request.json'
        amendment.write_text('{}')
        graph = [dict(id=f'g{i}', action='train', arm=f'g{i}', depends=[], gpu=True) for i in range(5)]
        graph += [dict(id='cpu', action='weights', depends=[], gpu=False),
                  dict(id='dependent', action='train', arm='dependent', depends=['g0'], gpu=True)]
        # One already completed job is recovered without spawning it again.
        (root / 'done_saved').write_text('ok')
        graph += [dict(id='saved', action='train', arm='saved', depends=[], gpu=True)]
        launched, peaks = [], dict(gpu=0, cpu=0, overlap=False)
        snapshot = [dict(index=i, uuid=f'GPU-parallel-test-{i}', memory_mib=20,
                        utilization=0, compute_pids=[]) for i in range(4)]

        def spawn(command, **kwargs):
            action = command[command.index(p.k.MODULE + '.worker') + 1]
            name = command[command.index('--arm') + 1] if '--arm' in command else 'cpu'
            if name == 'dependent':
                assert (root / 'done_g0').exists()
            launched.append(name)
            duration = .35 if name == 'cpu' else .18
            script = 'import pathlib,sys,time;time.sleep(float(sys.argv[1]));pathlib.Path(sys.argv[2]).write_text("ok")'
            return real_popen([sys.executable, '-c', script, str(duration), str(root / ('done_' + name))], **kwargs)

        def refresh(supervisor):
            supervisor.gpus = snapshot
            supervisor.streak = {g['uuid']: 2 for g in snapshot}

        def publish(supervisor, phase=None, **extra):
            gpu = [v for v in supervisor.active.values() if v['job']['gpu']]
            cpu = [v for v in supervisor.active.values() if not v['job']['gpu']]
            assert len(gpu) <= 4 and len(cpu) <= 1
            assert len({v['gpu_uuid'] for v in gpu}) == len(gpu)
            peaks['gpu'] = max(peaks['gpu'], len(gpu))
            peaks['cpu'] = max(peaks['cpu'], len(cpu))
            peaks['overlap'] |= bool(gpu and cpu)
            p.k.atomic(root / 'jobs.json', supervisor.ledger)

        with mock.patch.object(p.k, 'ROOT', root), mock.patch.object(p, 'AMENDMENT', amendment), \
             mock.patch.object(p, 'verify_dispatch', return_value={}), \
             mock.patch.object(p, 'report'), mock.patch.object(p.idle, 'gpu_snapshot', return_value=snapshot), \
             mock.patch.object(p.planning, 'verify_output', side_effect=lambda j: (root / ('done_' + j['id'])).exists()), \
             mock.patch.object(p.Supervisor, 'refresh_gpus', refresh), \
             mock.patch.object(p.Supervisor, 'publish', publish), \
             mock.patch.object(p.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(p.time, 'sleep', side_effect=lambda _: real_sleep(.02)):
            supervisor = p.Supervisor(graph)
            supervisor.run()
            assert peaks == dict(gpu=4, cpu=1, overlap=True), peaks
            assert len(launched) == len(set(launched)) == 7 and 'saved' not in launched
            assert all(v['state'] == 'complete' for v in supervisor.ledger.values())
            before = list(launched)
            p.Supervisor(graph).run()
            assert launched == before, 'Restart duplicated completed work'
            (root / 'STOP_AFTER_CURRENT').write_text('test')
            stop_graph = [dict(id='stopped', action='train', arm='stopped', depends=[], gpu=True)]
            p.Supervisor(stop_graph).run()
            assert launched == before, 'Stop marker did not prevent admission'
        result.update(peaks=peaks, four_distinct_gpu_leases=True, cpu_gpu_overlap=True,
            dependency_receipt_precedes_child=True, completed_jobs_not_relaunched=True,
            restart_deduplicated=True, stop_prevents_admission=True, passed=True)
    result['scientific_request_sha256'] = p.k.sha(real_root / 'request.json')
    result['scheduler_sha256'] = p.k.sha(Path(p.__file__))
    result['test_sha256'] = p.k.sha(Path(__file__))
    p.k.atomic(real_root / 'parallel_preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

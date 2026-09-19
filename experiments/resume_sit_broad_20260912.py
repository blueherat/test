"""Resume the two original frozen sweeps after the current priority batch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

WORK = Path('/home/zhoushunyu/eqvae')
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS / 'sit_broad_resume_20260912'
PRIORITY = EXPS / 'sit_refined_priority_20260911'
PROTOCOL = WORK / 'docs/SIT_BROAD_RESUME_20260912_ZH.md'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
OLD_PAUSE = 'User reprioritization: broad queue remains paused.\n'
QUEUES = [
    dict(key='control', root=str(EXPS/'sit_control_output_50ideas_20260910'),
         stage='control_screen_1k', total=709,
         module='experiments.run_sit_control_50ideas_20260910'),
    dict(key='apg', root=str(EXPS/'sit_apg_mechanism_extension_20260911'),
         stage='apg_extension_screen_1k', total=201,
         module='experiments.sit_apg_mechanism_20260911.pipeline'),
]


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(phase, **details):
    value = dict(phase=phase, controller_pid=os.getpid(),
                 updated_unix=time.time(), **details)
    atomic(ROOT/'status.json', value)
    return value


def request_path(queue):
    return Path(queue['root'])/queue['stage']/'request.json'


def committed_snapshot(queue):
    """Check small commit records; the native controller rehashes all artifacts."""
    request = read(request_path(queue))
    digest = sha(request_path(queue))
    records = {}
    for config in request['configs']:
        base = request_path(queue).parent/config['arm']
        if not (base/'commit.json').exists():
            continue
        commit = read(base/'commit.json')
        expected = hashlib.sha256(json.dumps(
            config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert commit['request_sha256'] == digest
        assert commit['config_sha256'] == expected
        assert sha(base/'result.json') == commit['files']['result.json']
        result = read(base/'result.json')
        assert result['request_sha256'] == digest
        records[config['arm']] = sha(base/'commit.json')
    assert len(request['configs']) == queue['total']
    assert len({c['arm'] for c in request['configs']}) == queue['total']
    return records


def arm():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():
        verify_plan()
        print(json.dumps(read(ROOT/'status.json'), ensure_ascii=False), flush=True)
        return
    schedule = read(PRIORITY/'schedule.json')
    checked = {}
    queues = []
    for queue in QUEUES:
        request = read(request_path(queue))
        for path, digest in request['sources'].items():
            if path not in checked:
                assert sha(path) == digest, path
                checked[path] = digest
            else:
                assert checked[path] == digest, path
        completed = committed_snapshot(queue)
        queues.append(dict(queue, request_sha256=sha(request_path(queue)),
                           existing_commits=completed, completed_when_armed=len(completed)))
    priority_remaining = 100-len(read(PRIORITY/'results.json'))
    plan = dict(authorized_user_instruction='自动恢复大队列，把之前的跑完',
        created_unix=time.time(), queues=queues, samples_per_arm=1000,
        priority_total=100, priority_schedule_sha256=sha(PRIORITY/'schedule.json'),
        priority_block_arms={b['key']: b['arms'] for b in schedule['blocks']},
        automatic_broad_resume=True, no_automatic_5k=True,
        script_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
        frozen_sources_checked=checked,
        total_original_configurations=sum(q['total'] for q in queues),
        remaining_original_configurations=sum(q['total']-q['completed_when_armed'] for q in queues),
        priority_remaining_when_armed=priority_remaining)
    assert plan['remaining_original_configurations']-priority_remaining == 510
    atomic(ROOT/'plan.json', plan)
    atomic(PRIORITY/'automatic_resume.json', dict(
        enabled=True, supersedes_automatic_broad_resume_false=True,
        plan=str(ROOT/'plan.json'), status=str(ROOT/'status.json'),
        user_instruction=plan['authorized_user_instruction']))
    record('armed', priority=read(PRIORITY/'status.json'),
           remaining_after_priority=510, queues=[q['key'] for q in queues])
    print(json.dumps(dict(armed=True, remaining_after_priority=510,
                         source_files_checked=len(checked)), ensure_ascii=False), flush=True)


def verify_plan():
    plan = read(ROOT/'plan.json')
    assert sha(__file__) == plan['script_sha256']
    assert sha(PROTOCOL) == plan['protocol_sha256']
    assert sha(PRIORITY/'schedule.json') == plan['priority_schedule_sha256']
    for q in plan['queues']:
        assert sha(request_path(q)) == q['request_sha256']
    return plan


@contextmanager
def exclusive(path):
    stream = Path(path).open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream
    finally:
        stream.close()


def stop_requested():
    return (ROOT/'STOP_AFTER_CURRENT').exists() or (PRIORITY/'STOP_AFTER_CURRENT').exists()


def priority_ready(state):
    return (state.get('phase') == 'complete'
            and state.get('completed') == state.get('total') == 100
            and state.get('numerical_failures') == 0)


def queue_complete(queue):
    root = Path(queue['root'])
    state = read(root/'status.json')
    stage = read(root/queue['stage']/'status.json')
    audit = root/queue['stage']/'analysis_audit.json'
    return (state.get('phase') == 'complete' and 'stage' not in state
        and state.get('total_arms') == queue['total']
        and state.get('no_further_sampling_queued') is True
        and stage.get('phase') == 'complete'
        and stage.get('completed') == stage.get('total') == queue['total']
        and audit.exists() and read(audit).get('passed') is True)


def request_stop():
    for root in [ROOT, PRIORITY, *[Path(q['root']) for q in QUEUES]]:
        (root/'STOP_AFTER_CURRENT').touch()


def run_queue(queue, plan):
    root = Path(queue['root'])
    if queue_complete(queue):
        return True
    with exclusive(root/'pipeline.lock'):
        if stop_requested():
            record('stopped_after_current', next_queue=queue['key'])
            return False
        marker = root/'STOP_AFTER_CURRENT'
        if marker.exists():
            text = marker.read_text()
            if text != OLD_PAUSE:
                record('stopped_by_queue_marker', queue=queue['key'], marker_text=text)
                return False
            archived = ROOT/f'{queue["key"]}_prior_pause_{time.time_ns()}.txt'
            marker.rename(archived)
            atomic(ROOT/f'{queue["key"]}_pause_override.json', dict(
                archived=str(archived), sha256=sha(archived), changed_unix=time.time(),
                authorization=plan['authorized_user_instruction']))
    before = committed_snapshot(queue)
    for arm, digest in queue['existing_commits'].items():
        assert before[arm] == digest, arm
    command = [PYTHON, '-u', '-m', queue['module'], '--pipeline']
    with (ROOT/f'{queue["key"]}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        atomic(ROOT/'active_process.json', dict(
            queue=queue['key'], child_pid=process.pid, command=command,
            launched_unix=time.time(), completed_before=len(before)))
        print(json.dumps(dict(started=queue['key'], child_pid=process.pid,
                              completed_before=len(before), total=queue['total'])), flush=True)
        while process.poll() is None:
            if stop_requested():
                (root/'STOP_AFTER_CURRENT').touch()
            record('running_original_queue', queue=queue['key'], child_pid=process.pid,
                   original_status=read(root/'status.json'),
                   stop_requested=stop_requested())
            time.sleep(10)
        if process.returncode:
            raise RuntimeError(f'{queue["key"]} exited {process.returncode}; see {log.name}')
    if not queue_complete(queue):
        record('stopped_after_current', queue=queue['key'], original_status=read(root/'status.json'))
        return False
    after = committed_snapshot(queue)
    assert len(after) == queue['total']
    assert all(after[arm] == digest for arm, digest in before.items())
    atomic(ROOT/f'{queue["key"]}_completed.json', dict(
        passed=True, total=len(after), newly_completed=len(after)-len(before),
        all_existing_commits_preserved=True, request_sha256=sha(request_path(queue)),
        analysis_audit_sha256=sha(root/queue['stage']/'analysis_audit.json'),
        finished_unix=time.time()))
    return True


def run():
    plan = verify_plan()
    with exclusive(ROOT/'controller.lock'):
        signal.signal(signal.SIGTERM, lambda *_: request_stop())
        signal.signal(signal.SIGINT, lambda *_: request_stop())
        try:
            while True:
                state = read(PRIORITY/'status.json')
                if stop_requested():
                    record('stopped_while_waiting', priority=state)
                    return
                if state.get('phase') in ('failed', 'stopped_after_current'):
                    raise RuntimeError(f'Priority did not complete: {state}')
                if priority_ready(state):
                    break
                if state.get('phase') == 'complete':
                    raise RuntimeError(f'Priority completion is inconsistent: {state}')
                record('waiting_for_priority', priority=state, gpu_workers_started=False,
                       queue_order=[q['key'] for q in plan['queues']])
                time.sleep(10)
            while True:
                try:
                    priority_lock = (PRIORITY/'controller.lock').open('a')
                    fcntl.flock(priority_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    priority_lock.close()
                    time.sleep(1)
            try:
                assert priority_ready(read(PRIORITY/'status.json'))
                for key, arms in plan['priority_block_arms'].items():
                    receipt = read(PRIORITY/f'{key}_completed.json')
                    assert receipt.get('passed') and receipt.get('selected_complete')
                    assert set(receipt['commits']) == set(arms)
                atomic(ROOT/'priority_released.json', dict(
                    passed=True, released_unix=time.time(),
                    priority_status=read(PRIORITY/'status.json')))
                report = WORK/'docs/SIT_REFINED_PRIORITY_EXECUTION_20260911_ZH.md'
                report.write_text(report.read_text().replace(
                    '本批完成后自动停止；原大队列保持暂停。',
                    '优先批次已完成；按9月12日新指令，原大队列由独立接续器自动恢复。'
                    '详见[接续记录](SIT_BROAD_RESUME_20260912_ZH.md)。'))
                for queue in plan['queues']:
                    if not run_queue(queue, plan):
                        return
                record('complete', original_configurations_completed=910,
                       all_original_queues_complete=True, no_further_sampling_queued=True,
                       no_automatic_5k=True)
            finally:
                priority_lock.close()
        except BaseException as error:
            record('failed', error=repr(error))
            raise


def check():
    plan = verify_plan()
    states = [
        dict(phase='running_priority', completed=99, total=100, numerical_failures=0),
        dict(phase='complete', completed=99, total=100, numerical_failures=0),
        dict(phase='failed', completed=100, total=100, numerical_failures=0),
        dict(phase='complete', completed=100, total=100, numerical_failures=1),
    ]
    assert not any(priority_ready(s) for s in states)
    assert priority_ready(dict(phase='complete', completed=100, total=100, numerical_failures=0))
    try:
        with exclusive(PRIORITY/'controller.lock'):
            assert priority_ready(read(PRIORITY/'status.json'))
        locked = False
    except BlockingIOError:
        locked = True
    result = dict(passed=True, immutable_plan_verified=True,
        original_queue_totals=[q['total'] for q in plan['queues']],
        priority_completion_gate_checked=True, priority_lock_currently_held=locked,
        no_sampling_launched=True)
    atomic(ROOT/'readiness_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ['arm', 'run', 'check', 'status', 'stop-after-current']:
        group.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if args.arm: arm()
    elif args.run: run()
    elif args.check: check()
    elif args.stop_after_current: request_stop()
    else: print(json.dumps(read(ROOT/'status.json'), ensure_ascii=False, indent=2))

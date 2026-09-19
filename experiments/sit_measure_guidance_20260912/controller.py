"""New five-idea experiment, then the interrupted priority and original sweeps."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .data import ROOT, WORK
from . import catalog
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
PRIORITY = ROOT.parent/'sit_refined_priority_20260911'
OLD_ROOTS = [PRIORITY, catalog.OLD.parent,
    ROOT.parent/'sit_apg_mechanism_extension_20260911',
    ROOT.parent/'sit_broad_resume_20260912']
MODULE = 'experiments.sit_measure_guidance_20260912'


def status(phase, **details):
    atomic(ROOT/'controller_status.json', dict(phase=phase, controller_pid=os.getpid(),
        updated_unix=time.time(), **details))


def marker_identity(path):
    return dict(sha256=sha(path), mtime_ns=path.stat().st_mtime_ns)


def arm():
    if (ROOT/'execution_plan.json').exists():
        return verify()
    markers = {}
    for root in OLD_ROOTS:
        path = root/'STOP_AFTER_CURRENT'
        assert path.exists(), path
        markers[str(path)] = marker_identity(path)
    assert read(PRIORITY/'status.json')['phase'] == 'stopped_after_current'
    assert read(PRIORITY/'status.json')['completed'] == 93
    plan = dict(user_instruction='先停止一下；深度研究五个新的机制idea，先跑它们，再接续之前的idea',
        created_unix=time.time(), new_arms=59, new_candidates=24, new_controls=35,
        old_priority_remaining=7, old_original_arms_remaining=517,
        order=['new_five_ideas', 'old_priority_remaining', 'old_control_709', 'old_apg_201'],
        no_automatic_5k=True, stop_markers=markers,
        controller_sha256=sha(__file__),
        protocol_sha256=sha(WORK/'docs/SIT_MEASURE_GUIDANCE_PROTOCOL_20260912_ZH.md'),
        training_request_sha256=sha(ROOT/'training_request.json'),
        old_schedule_sha256=sha(PRIORITY/'schedule.json'),
        old_requests={str(path):sha(path) for path in [
            catalog.OLD/'request.json',
            ROOT.parent/'sit_apg_mechanism_extension_20260911/apg_extension_screen_1k/request.json']})
    atomic(ROOT/'execution_plan.json', plan)
    atomic(PRIORITY/'automatic_resume.json', dict(enabled=True, after_new_five_ideas=True,
        execution_plan=str(ROOT/'execution_plan.json'), controller_status=str(ROOT/'controller_status.json')))
    status('armed', order=plan['order'])
    return plan


def verify():
    plan = read(ROOT/'execution_plan.json')
    assert plan['controller_sha256'] == sha(__file__)
    assert plan['protocol_sha256'] == sha(WORK/'docs/SIT_MEASURE_GUIDANCE_PROTOCOL_20260912_ZH.md')
    assert plan['training_request_sha256'] == sha(ROOT/'training_request.json')
    assert plan['old_schedule_sha256'] == sha(PRIORITY/'schedule.json')
    for path, digest in plan['old_requests'].items():
        assert sha(path) == digest, path
    return plan


def stopped():
    return (ROOT/'STOP_AFTER_CURRENT').exists()


def command(module, args, phase):
    if stopped():
        return False
    with (ROOT/f'{phase}.log').open('a') as log:
        process = subprocess.Popen([PYTHON, '-u', '-m', module, *args], cwd=WORK,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        atomic(ROOT/'active_process.json', dict(phase=phase, child_pid=process.pid,
            command=[PYTHON, '-u', '-m', module, *args], launched_unix=time.time()))
        while process.poll() is None:
            if stopped() and phase.startswith('old_'):
                for root in OLD_ROOTS:
                    (root/'STOP_AFTER_CURRENT').touch()
            state_path = ROOT/'status.json'
            if phase == 'old_priority_remaining':
                state_path = PRIORITY/'status.json'
            elif phase.startswith('old_broad'):
                state_path = OLD_ROOTS[-1]/'status.json'
            status(phase, child_pid=process.pid, stop_requested=stopped(),
                current_stage=read(state_path) if state_path.exists() else {})
            time.sleep(10)
        if process.returncode:
            raise RuntimeError(f'{phase} failed with {process.returncode}: {log.name}')
    return not stopped()


def release_old_markers(plan):
    # Also check timestamps: a subsequent touch is a new stop even if bytes match.
    for path, identity in plan['stop_markers'].items():
        assert Path(path).exists() and marker_identity(Path(path)) == identity, \
            f'Stop marker changed since authorization: {path}'
    archived = ROOT/'archived_old_pause'
    archived.mkdir(exist_ok=True)
    for path, identity in plan['stop_markers'].items():
        path = Path(path)
        target = archived/(path.parent.name+'.txt')
        path.rename(target)
        assert sha(target) == identity['sha256']
    atomic(ROOT/'old_pause_released.json', dict(passed=True,
        execution_plan_sha256=sha(ROOT/'execution_plan.json'), released_unix=time.time(),
        authorization=plan['user_instruction']))


def run():
    plan = verify()
    with (ROOT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_:(ROOT/'STOP_AFTER_CURRENT').touch())
        signal.signal(signal.SIGINT, lambda *_:(ROOT/'STOP_AFTER_CURRENT').touch())
        try:
            if not (ROOT/'old_pause_released.json').exists():
                with (PRIORITY/'controller.lock').open('a') as priority_lock:
                    fcntl.flock(priority_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    assert read(ROOT/'training_status.json')['phase'] == 'complete'
                    if not (ROOT/'mechanism_check.json').exists():
                        if not command(MODULE+'.study', [], 'mechanism'):
                            return
                    if not (ROOT/'development_check.json').exists():
                        if not command(MODULE+'.pipeline', ['--check'], 'preflight'):
                            return
                    if not command(MODULE+'.pipeline', ['--pipeline'], 'new_five_ideas'):
                        return
                    final = read(ROOT/'status.json')
                    assert final['phase'] == 'complete' and final['total_arms'] == 59
                    assert read(ROOT/'measure_screen_1k/analysis_audit.json')['passed']
                    if stopped():
                        return
                    release_old_markers(plan)
            if not (PRIORITY/'status.json').exists():
                raise RuntimeError('Missing old priority status')
            state = read(PRIORITY/'status.json')
            if state.get('phase') != 'complete' or state.get('completed') != 100:
                if not command('experiments.run_sit_refined_priority_20260911',
                               ['--pipeline'], 'old_priority_remaining'):
                    return
            state = read(PRIORITY/'status.json')
            assert state['phase'] == 'complete' and state['completed'] == 100
            if not command('experiments.resume_sit_broad_20260912', ['--arm'], 'old_broad_arm'):
                return
            if not command('experiments.resume_sit_broad_20260912', ['--run'], 'old_broad_queues'):
                return
            assert read(OLD_ROOTS[-1]/'status.json')['phase'] == 'complete'
            status('complete', new_arms=59, original_arms_completed=910,
                all_authorized_queues_complete=True, no_automatic_5k=True)
        except BaseException as error:
            status('failed', error=repr(error))
            raise
        finally:
            if stopped():
                status('stopped_after_current')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--arm', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--stop-after-current', action='store_true')
    args = parser.parse_args()
    if args.arm:
        print(json.dumps(arm(), ensure_ascii=False), flush=True)
    elif args.run:
        run()
    else:
        (ROOT/'STOP_AFTER_CURRENT').touch()

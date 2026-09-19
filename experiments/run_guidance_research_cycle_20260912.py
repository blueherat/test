"""Run a reviewed research stage while the historical broad queues stay paused."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

WORK = Path('/home/zhoushunyu/eqvae')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = BASE/'guidance_research_cycles_20260912'
MEASURE = BASE/'sit_measure_guidance_20260912'
PRIORITY = BASE/'sit_refined_priority_20260911'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(plan_path):
    plan = read(plan_path)
    stage_root = Path(plan['stage_root'])
    assert digest(plan['protocol']) == plan['protocol_sha256']
    assert not (ROOT/'STOP_AFTER_CURRENT').exists()
    for prerequisite in plan.get('prerequisites', []):
        path = Path(prerequisite)
        while not path.exists():
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                return
            time.sleep(5)
        assert read(path)['passed'], path
    logs = ROOT/plan['id']
    logs.mkdir(parents=True, exist_ok=True)
    # Keep the previous automatic broad-queue supervisor from taking over.
    with (ROOT/'controller.lock').open('a') as own, \
            (MEASURE/'supervisor.lock').open('a') as previous, \
            (PRIORITY/'controller.lock').open('a') as priority:
        for lock in (own, previous, priority):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p, expected in plan['paused_old_roots'].items():
            marker = Path(p)/'STOP_AFTER_CURRENT'
            assert marker.exists() and digest(marker) == expected
        stop = stage_root/'STOP_AFTER_CURRENT'
        if stop.exists():
            assert plan.get('release_stage_stop_sha256') == digest(stop)
            stop.rename(logs/'released_stage_stop.txt')
        def request_stop(*_):
            (ROOT/'STOP_AFTER_CURRENT').touch()
            stop.touch()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        with (logs/'run.log').open('a') as log:
            command = [PYTHON, '-u', '-m', plan['module'], '--pipeline']
            proc = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            write(logs/'launch.json', dict(command=command, pid=proc.pid,
                plan_sha256=digest(plan_path), supervisor_pid=os.getpid(), started_unix=time.time()))
            while proc.poll() is None:
                if (ROOT/'STOP_AFTER_CURRENT').exists():
                    stop.touch()
                state = stage_root/'status.json'
                write(ROOT/'status.json', dict(phase='running_reviewed_stage', stage=plan['id'],
                    controller_pid=os.getpid(), child_pid=proc.pid, updated_unix=time.time(),
                    stage_status=read(state) if state.exists() else {},
                    old_broad_queue_paused=True))
                time.sleep(5)
            if proc.returncode:
                write(ROOT/'status.json', dict(phase='failed', stage=plan['id'],
                    returncode=proc.returncode, log=str(logs/'run.log')))
                raise RuntimeError(f'Stage failed: {logs / "run.log"}')
        state = read(stage_root/'status.json')
        if stop.exists() or (ROOT/'STOP_AFTER_CURRENT').exists():
            write(ROOT/'status.json', dict(phase='stopped_after_current', stage=plan['id'],
                stage_status=state, old_broad_queue_paused=True))
            return
        assert state['phase'] == 'complete', state
        assert read(plan['audit'])['passed']
        write(logs/'complete.json', dict(passed=True, finished_unix=time.time(),
            plan_sha256=digest(plan_path), audit_sha256=digest(plan['audit']), stage_status=state))
        write(ROOT/'status.json', dict(phase='ready_for_research_review', stage=plan['id'],
            stage_status=state, old_broad_queue_paused=True, no_automatic_parameter_expansion=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    run(args.plan)

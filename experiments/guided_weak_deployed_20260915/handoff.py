"""Activate the follow-up only after the current guided-weak idea finishes all evaluations."""
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from . import k,REQUEST,verify
from experiments.guidance_dynamic_50k_20260915.pipeline import process_ticks


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    verify()
    root=k.ROOT/'deployed_handoff'
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'handoff.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=k.read(k.ROOT/'launch.json')
    assert launch['command']==[k.PYTHON,'-u','-m','experiments.guidance_dynamic_recovery_20260915.pipeline']
    pid,ticks=launch['pid'],launch['process_start_ticks']
    k.atomic(root/'status.json',dict(phase='waiting_current_idea',pid=os.getpid(),controller_pid=pid,
        candidate_request_sha256=k.sha(REQUEST),recorded_utc=now()))
    while True:
        if process_ticks(pid)!=ticks:
            k.atomic(root/'status.json',dict(phase='cancelled_controller_changed_or_exited',recorded_utc=now()))
            return
        if (k.ROOT/'STOP_AFTER_CURRENT').exists() or (k.ROOT/'failure.json').exists():
            k.atomic(root/'status.json',dict(phase='cancelled_existing_stop_or_failure',recorded_utc=now()))
            return
        searches=k.read(k.ROOT/'searches.json')
        if searches.get('sit_small/guided_weak',{}).get('phase')=='complete':
            break
        time.sleep(5)
    verify()
    assert k.read(root/'cpu_checks.json')['passed']
    assert k.read(root/'planning_checks.json')['passed']
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+600
    while process_ticks(pid)==ticks:
        if time.monotonic()>deadline:
            k.atomic(root/'status.json',dict(phase='needs_attention_drain_timeout',recorded_utc=now()))
            return
        time.sleep(2)
    ledger=k.read(k.ROOT/'jobs.json')
    assert not any(row.get('state')=='running' and process_ticks(row['pid'])==row['process_start_ticks']
                   for row in ledger.values())
    marker=k.ROOT/'STOP_AFTER_CURRENT'
    assert marker.read_text()==f'Controller signal {signal.SIGTERM}; save and drain.\n'
    assert not (k.ROOT/'failure.json').exists()
    archive=root/'previous_controller'
    archive.mkdir()
    for name in ('launch.json','status.json','jobs.json','searches.json'):
        shutil.copy2(k.ROOT/name,archive/name)
    marker.rename(archive/'STOP_AFTER_CURRENT')
    command=[k.PYTHON,'-u','-m','experiments.guidance_dynamic_recovery_20260915.pipeline_theory']
    log=k.ROOT/'logs/controller_deployed.log'
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=k.WORK,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    updated=dict(launch,pid=child.pid,process_start_ticks=process_ticks(child.pid),command=command,
        log=str(log),started_utc=now(),candidate_request_sha256=k.sha(REQUEST))
    k.atomic(k.ROOT/'launch.json',updated)
    for old in ('guidance_strength_sweep_20260915','guidance_loss_50k_20260914'):
        path=k.EXPS/old/'ACTIVE_CONTROLLER.json'
        if path.exists():k.atomic(path,updated)
    k.atomic(root/'status.json',dict(phase='activated',controller=updated,recorded_utc=now()))


if __name__=='__main__':
    main()

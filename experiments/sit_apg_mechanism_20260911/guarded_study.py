"""Temporarily yield the existing queue at an arm boundary and always resume it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from experiments.sit_apg_mechanism_20260911 import study
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha


def alive(pid):
    try:
        os.kill(int(pid),0)
        return True
    except ProcessLookupError:
        return False


def main():
    study.prepare()
    root=study.DEPENDENCY
    marker=root/'STOP_AFTER_CURRENT'
    owner=f'apg_mechanism_20260911_{os.getpid()}'
    initial=read(root/'status.json')
    record=dict(owner=owner,started_unix=time.time(),before=initial,
                request_sha256=sha(root/'control_screen_1k/request.json'))
    metadata=study.ROOT/'mechanism_pause.json'
    atomic(metadata,record)
    already_complete=initial.get('phase')=='complete' and initial.get('no_further_sampling_queued') is True
    wrote=False
    try:
        if not already_complete:
            assert initial['phase'] in ('sampling','evaluating','preflight'),initial
            with marker.open('x') as stream:
                stream.write(owner+'\n')
            wrote=True
            deadline=time.time()+1800
            while True:
                state=read(root/'status.json')
                pids=[initial['controller_pid'],*initial['worker_pids']]
                if state.get('phase')=='stopped_after_current' and not any(alive(p) for p in pids):
                    assert state.get('worker_exit_codes')==[0]*4,state
                    break
                if state.get('phase')=='failed':raise RuntimeError(state)
                if time.time()>deadline:raise TimeoutError('Waiting for the existing arm to commit')
                print(json.dumps(dict(waiting_for_arm_boundary=True,phase=state.get('phase'),completed=state.get('completed'))),flush=True)
                time.sleep(10)
            record['stopped']=state;atomic(metadata,record)
        result=subprocess.run([study.engine.PYTHON,'-u','-m',study.MODULE,'--run'],cwd=study.WORK,
                              env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        record['mechanism_exit_code']=result.returncode;atomic(metadata,record)
        if result.returncode:raise RuntimeError(f'Mechanism study exited {result.returncode}')
    finally:
        if wrote:
            assert marker.read_text().strip()==owner,'Stop marker changed ownership'
            (study.ROOT/'mechanism_stop_marker.txt').write_text(marker.read_text())
            marker.unlink()
            state=read(root/'status.json')
            if state.get('phase')=='stopped_after_current':
                pane='sit_control53_0911:0.0'
                deadline=time.time()+30
                while subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{pane_dead}'],text=True).strip()!='1':
                    if time.time()>deadline:raise RuntimeError('The prior controller pane has not exited')
                    time.sleep(1)
                command='exec '+shlex.join(['env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
                    study.engine.PYTHON,'-u','-m','experiments.run_sit_control_50ideas_20260910','--pipeline'])
                command+=' >> '+shlex.quote(str(root/'controller.log'))+' 2>&1'
                subprocess.run(['tmux','respawn-pane','-t',pane,command],check=True)
                record['resume_command']=command
                record['resumed_unix']=time.time()
                record['resumed_pane_pid']=int(subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{pane_pid}'],text=True))
        record['finished_unix']=time.time();atomic(metadata,record)


if __name__=='__main__':
    main()

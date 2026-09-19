"""Run extension preflight between committed arms, and always restore the queue."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from experiments.sit_apg_mechanism_20260911 import pipeline as task
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def main():
    root = task.WAIT_ROOT
    marker = root/'STOP_AFTER_CURRENT'
    owner = f'apg_extension_preflight_{os.getpid()}'
    initial = read(root/'status.json')
    record = dict(owner=owner, before=initial, started_unix=time.time(),
        dependency_request_sha256=sha(root/'control_screen_1k/request.json'))
    metadata = task.ROOT/'preflight_pause.json'
    atomic(metadata, record)
    wrote = False
    try:
        if not task.dependency_complete(initial):
            assert initial['phase'] in ('sampling', 'evaluating', 'preflight'), initial
            with marker.open('x') as stream:stream.write(owner+'\n')
            wrote = True
            pids = [initial['controller_pid'], *initial['worker_pids']]
            deadline = time.time()+1800
            while True:
                status = read(root/'status.json')
                if status.get('phase') == 'stopped_after_current' and not any(alive(p) for p in pids):
                    assert status['worker_exit_codes'] == [0]*4
                    break
                if status.get('phase') == 'failed':raise RuntimeError(status)
                if time.time() > deadline:raise TimeoutError('Existing arm has not committed')
                print(dict(waiting_for_boundary=True, phase=status.get('phase'), completed=status.get('completed')), flush=True)
                time.sleep(10)
            record['stopped'] = status;atomic(metadata, record)
        result = subprocess.run([task.engine.PYTHON, '-u', '-m', task.MODULE, '--check'], cwd=task.WORK,
            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        record['preflight_exit_code'] = result.returncode;atomic(metadata, record)
        if result.returncode:raise RuntimeError(f'Extension preflight exited {result.returncode}')
    finally:
        if wrote:
            assert marker.read_text().strip() == owner, 'Stop marker ownership changed'
            (task.ROOT/'preflight_stop_marker.txt').write_text(marker.read_text())
            marker.unlink()
            state = read(root/'status.json')
            if state.get('phase') == 'stopped_after_current':
                pane = 'sit_control53_0911:0.0'
                deadline = time.time()+30
                while subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_dead}'], text=True).strip() != '1':
                    if time.time() > deadline:raise RuntimeError('Previous controller pane did not exit')
                    time.sleep(1)
                command = 'exec '+shlex.join(['env', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                    task.engine.PYTHON, '-u', '-m', 'experiments.run_sit_control_50ideas_20260910', '--pipeline'])
                command += ' >> '+shlex.quote(str(root/'controller.log'))+' 2>&1'
                subprocess.run(['tmux', 'respawn-pane', '-t', pane, command], check=True)
                record['resumed_unix'] = time.time()
                record['resumed_pane_pid'] = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_pid}'], text=True))
        record['finished_unix'] = time.time();atomic(metadata, record)


if __name__ == '__main__':
    main()

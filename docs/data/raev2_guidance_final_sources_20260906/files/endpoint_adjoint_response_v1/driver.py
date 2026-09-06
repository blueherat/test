"""Execute the frozen pilot once; retain external costs and every subprocess log."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/zhoushunyu/eqvae')
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
SCRIPT = ROOT / 'experiments/audit_raev2_endpoint_adjoint_response.py'
PARTS = ('0,142', '285,428', '570,713', '856,999')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def run(phase, rank=None):
    tag = phase if rank is None else f'{phase}_rank{rank}'
    cmd = [PYTHON, str(SCRIPT), '--phase', phase, '--output-root', str(OUT)]
    if rank is not None:
        cmd.extend(['--ids', PARTS[rank]])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if rank is None else str(rank),
               TORCH_HOME='/home/zhoushunyu/.cache/torch', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    start, utc = time.perf_counter(), now()
    with (OUT / (tag + '.log')).open('x') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    result = {'tag': tag, 'command': cmd, 'pid': proc.pid, 'started_at_utc': utc,
              'finished_at_utc': now(), 'exit_code': proc.returncode,
              'outer_wall_seconds': time.perf_counter()-start,
              'child_user_cpu_seconds': usage.ru_utime, 'child_system_cpu_seconds': usage.ru_stime,
              'child_max_rss_bytes': usage.ru_maxrss*1024,
              'timing_scope': 'Immediately before Popen through wait4; includes imports and interpreter exit.'}
    write(tag + '.process.json', result)
    print(json.dumps(result), flush=True)
    if proc.returncode:
        raise RuntimeError(f'{tag} failed with {proc.returncode}; inspect retained log')
    return result

def main():
    start = time.perf_counter()
    write('driver_identity.json', {'pid': os.getpid(), 'started_at_utc': now(),
          'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'request_sha256': hashlib.sha256((OUT/'request.json').read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
          'no_retry_or_gain_changes': True})
    records = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda rank: run('collect', rank), range(4)))
        records.append(run('finalize'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda rank: run('replay', rank), range(4)))
        records.append(run('summarize'))
        write('execution_summary.json', {'complete': True, 'finished_at_utc': now(),
              'driver_wall_seconds': time.perf_counter()-start, 'processes': records,
              'excludes': 'CPU prepare executed before driver; final summary write and driver exit.'})
    except BaseException as e:
        write('driver_failure.json', {'complete': False, 'error': repr(e),
              'finished_at_utc': now(), 'driver_wall_seconds': time.perf_counter()-start,
              'completed_phases': records})
        raise

if __name__ == '__main__':
    main()

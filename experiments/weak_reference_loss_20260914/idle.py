"""Read-only resource admission; no CUDA imports and no modification of other queues."""
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import time

from . import config as k

PREDECESSORS = (
    b'experiments.raev2_shallow_ig_20260914.tmux_sweep',
    b'experiments.raev2_shallow_ig_20260914.selection_only\x00pipeline',
    b'experiments.ig_sg_5k_20260914.selection_only\x00pipeline',
)


def predecessors():
    rows=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            command=(proc/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        for pattern in PREDECESSORS:
            if pattern in command:
                rows.append(dict(pid=int(proc.name),queue=pattern.decode().replace('\x00',' ')))
                break
    return rows


def gpu_snapshot():
    def query(argument):
        result=subprocess.run(['nvidia-smi',argument,'--format=csv,noheader,nounits'],
                              capture_output=True,text=True,check=True,timeout=15)
        return list(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    gpu_rows=query('--query-gpu=index,uuid,memory.used,utilization.gpu')
    process_rows=query('--query-compute-apps=gpu_uuid,pid')
    processes={}
    for row in process_rows:
        if len(row)==2 and row[0].startswith('GPU-'):
            processes.setdefault(row[0],[]).append(int(row[1]))
    rows=[]
    for index,uuid,memory,utilization in gpu_rows:
        # Unknown utilization/memory is not evidence that a GPU is idle.
        try:
            rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),
                             utilization=int(utilization),compute_pids=processes.get(uuid,[])))
        except ValueError:
            continue
    return rows


def eligible(row):
    return not row['compute_pids'] and row['memory_mib']<600 and row['utilization']<=5


def wait_gpu(update, interval=20, required=3):
    streak={}
    while True:
        k.check_stop()
        prior=predecessors()
        try:
            rows=gpu_snapshot()
        except (subprocess.SubprocessError,OSError) as error:
            streak.clear()
            update('waiting_gpu',query_error=repr(error),predecessors=prior)
            time.sleep(interval)
            continue
        available=[]
        for row in rows:
            uuid=row['uuid']
            streak[uuid]=streak.get(uuid,0)+1 if not prior and eligible(row) else 0
            row['idle_observations']=streak[uuid]
            if streak[uuid]>=required:
                available.append(row)
        update('waiting_predecessor' if prior else 'waiting_gpu',predecessors=prior,gpus=rows)
        for row in available:
            lock=Path('/tmp',f"eqvae_idle_{row['uuid']}.lock").open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            # Recheck after acquiring the advisory lease.
            try:
                current={x['uuid']:x for x in gpu_snapshot()}
                admitted=not predecessors() and row['uuid'] in current and eligible(current[row['uuid']])
            except (subprocess.SubprocessError,OSError):
                admitted=False
            if admitted:
                return row,lock
            lock.close()
        time.sleep(interval)

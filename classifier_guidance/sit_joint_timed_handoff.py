"""Audit, resume on two GPUs within a deadline, then continue on GPU1 alone."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_joint_gan_20260922')
HEAD=Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/block1/training_50k/checkpoint_050000.pt')
TARGET=30128


def terminate_owned_tree(process,grace=5.):
    """Only descendants of this launch; never GPU-wide or name-based killing."""
    try:parent=psutil.Process(process.pid)
    except psutil.NoSuchProcess:return
    owned=parent.children(recursive=True)+[parent]
    for child in owned:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=grace)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=grace)
    process.wait(timeout=max(1,grace))


def run_bounded(command,cutoff=None,grace=5.,cwd=None):
    # Convert once to monotonic time so wall-clock corrections cannot extend
    # the allocated duration. The normal trainer checkpoint deadline is earlier.
    remaining=None if cutoff is None else max(0.,cutoff-time.time())
    if remaining==0:return dict(returncode=None,forced=True,launched=False)
    end=None if remaining is None else time.monotonic()+remaining
    process=subprocess.Popen(command,cwd=cwd,start_new_session=True)
    while process.poll() is None:
        if end is not None and time.monotonic()>=end:
            terminate_owned_tree(process,grace)
            return dict(returncode=process.returncode,forced=True,launched=True)
        time.sleep(.2 if end is not None else 1.)
    return dict(returncode=process.returncode,forced=False,launched=True)


def main(args):
    from experiments.adversarial_weak_training_20260915 import common as c
    root=args.root
    lock=(root/'handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    # Reserve three minutes for orderly saving. A separate process watchdog
    # enforces release even if the worker hangs; it fires one minute early.
    soft=deadline-180;hard=deadline-60
    c.atomic(root/'handoff_plan.json',dict(requested_utc=c.now(),extra_gpu=3,retained_gpu=1,
        deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        orderly_save_utc=datetime.fromtimestamp(soft,timezone.utc).isoformat(),
        watchdog_utc=datetime.fromtimestamp(hard,timezone.utc).isoformat(),
        deadline_epoch=deadline,global_batch=32,microbatch=8,target_global_step=TARGET))
    old=root/'training_30k'
    (old/'STOP_AFTER_CURRENT').touch()
    begin=time.monotonic()
    while not (old/'exit.json').exists():
        if time.monotonic()-begin>180:raise TimeoutError('Original trainer did not safely pause')
        time.sleep(1)
    assert c.read(old/'exit.json')['exit_code']==0
    previous=c.read(old/'latest.json');assert previous['phase']=='paused'
    checkpoint=Path(previous['checkpoint'])
    c.atomic(root/'pipeline_complete.json',dict(complete=False,superseded_by='handoff_progress.json',
        paused_step=previous['step'],updated_utc=c.now()))

    def launch(name,gpus,task,extra):
        from experiments.weak_reference_loss_20260914.idle import gpu_snapshot,eligible
        run=root/name
        command=[sys.executable,'-u','-m','classifier_guidance.launch','--gpus',gpus,'--task',task,
                 '--output',str(run),'--','--head-checkpoint',str(HEAD),*extra]
        c.atomic(root/'handoff_progress.json',dict(phase=name,gpus=gpus,command=command,
            checkpoint=str(checkpoint),deadline_utc=args.deadline,updated_utc=c.now()))
        wait_start=time.monotonic();requested=set(gpus.split(','))
        while not all(eligible(r) for r in gpu_snapshot() if str(r['index']) in requested):
            if time.monotonic()-wait_start>120:raise TimeoutError(f'Waiting for released GPUs {gpus}')
            if ',' in gpus and time.time()>=hard:
                return dict(returncode=None,forced=True,launched=False)
            time.sleep(1)
        result=run_bounded(command,cutoff=hard if ',' in gpus else None,cwd=c.WORK)
        c.atomic(root/(name+'_launch_result.json'),dict(result,updated_utc=c.now()))
        return result

    def train(name,gpus,updates):
        nonlocal checkpoint
        extra=['--resume',str(checkpoint),'--updates',str(updates),'--global-batch','32','--microbatch','8',
               '--save-every','300']
        if ',' in gpus:extra+=['--stop-at-epoch',str(soft)]
        result=launch(name,gpus,'sit-joint',extra)
        latest=root/name/'latest.json'
        if latest.exists():
            item=c.read(latest);checkpoint=Path(item['checkpoint'])
            assert c.sha(checkpoint)==item['sha256']
        return result

    def checkpoint_step():
        # The name is assigned from a verified trainer latest.json; avoid
        # loading all optimizer tensors just to read the completed step.
        return int(checkpoint.stem.split('_')[-1])

    try:
        audit=launch('distributed_audit_20260923','1,3','sit-joint-distributed-audit',
                     ['--checkpoint',str(checkpoint)])
        if audit['returncode']!=0 or not c.read(root/'distributed_audit_20260923/result.json')['passed']:
            raise RuntimeError('Distributed audit failed; resuming on one GPU')
        # Exercise both world-size transitions on real checkpoints before
        # committing the eight-hour unattended stage. These are real updates.
        for name,gpus,count in [('dual_resume_check_20260923','1,3',2),
                                ('single_resume_check_20260923','1',1)]:
            result=train(name,gpus,count)
            if result['returncode']!=0:raise RuntimeError(f'{name} failed')
        if time.time()<soft:
            train('dual_until_deadline_20260923','1,3',TARGET-checkpoint_step())
    except Exception as exc:
        c.atomic(root/'handoff_warning.json',dict(error=str(exc),checkpoint=str(checkpoint),updated_utc=c.now()))
    finally:
        # Only GPU1 appears in this command. Every launch leases and releases
        # its own cards; the GPU3 lease never survives into the single stage.
        if checkpoint_step()<TARGET:
            result=train('single_after_deadline_20260923','1',TARGET-checkpoint_step())
            if result['returncode']!=0:raise RuntimeError('Single-GPU continuation failed; inspect worker.log')
    c.atomic(root/'handoff_complete.json',dict(complete=True,checkpoint=str(checkpoint),updated_utc=c.now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--deadline',required=True,help='Absolute timezone-aware ISO timestamp')
    main(p.parse_args())

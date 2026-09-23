"""Lease explicitly selected idle GPUs, freeze sources, then run the trainer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main(args):
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
    from experiments.adversarial_weak_training_20260915 import common as c
    requested = args.gpus.split(',')
    if len(set(requested)) != len(requested):
        raise ValueError('GPU selection contains duplicates')
    leases, devices = [], []
    def available(row):
        # A display service can appear in nvidia-smi's compute-apps table.
        # Allow only explicitly identified, verified system desktop processes;
        # retain the memory/utilization limits and reject all other workloads.
        ignored=[]
        for pid in row['compute_pids']:
            if pid not in args.allow_desktop_pid:continue
            try:executable=Path(f'/proc/{pid}/exe').resolve(strict=True)
            except (OSError,RuntimeError):
                # A system-owned desktop process can hide /proc/PID/exe while
                # exposing its command line. Match its complete executable path.
                try:executable=Path(Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[0].decode())
                except (OSError,UnicodeError):continue
            if executable==Path('/usr/libexec/gnome-initial-setup'):ignored.append(pid)
        return eligible(dict(row,compute_pids=[pid for pid in row['compute_pids'] if pid not in ignored]))
    for key in requested:
        row = next(r for r in gpu_snapshot() if str(r['index']) == key or r['uuid'] == key)
        if not available(row):
            raise RuntimeError(f"GPU {key} is not idle: {row}")
        lease = Path('/tmp', f"eqvae_idle_{row['uuid']}.lock").open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not available(next(r for r in gpu_snapshot() if r['uuid'] == row['uuid'])):
            raise RuntimeError(f'GPU {key} became busy')
        leases.append(lease);devices.append(row)
    if args.arguments[:1] == ['--']:
        args.arguments = args.arguments[1:]
    if '--output' in args.arguments:
        raise ValueError('Specify output once, before --')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a new output directory, and --resume to continue a checkpoint')
    sources = c.source_receipts()
    sources.update({str(p.resolve()):c.sha(p) for p in Path(__file__).parent.glob('*.py')})
    from experiments.lifting_scale_sweep_20260909 import source_paths
    for model in ('sit_small','jit','raev2'):
        for p in source_paths(model):
            sources[str(p.resolve())] = c.sha(p)
    for subdir in ('stage2/models','stage1/decoders'):
        for p in (c.WORK/'external/RAEv2/src'/subdir).glob('*.py'):
            sources[str(p.resolve())] = c.sha(p)
    if args.task.startswith(('jit-ssg', 'jit-schedule')) or args.task == 'learned-guidance-eval':
        from .jit_ssg import LITERATURE
        for p in LITERATURE.rglob('*.py'):
            sources[str(p.resolve())] = c.sha(p)
    entry = {'head':'classifier_guidance.train','schedule':'classifier_guidance.train_schedule',
             'learned-guidance-eval':'classifier_guidance.evaluate_learned_guidance',
             'sit-joint':'classifier_guidance.train_sit_joint',
             'sit-joint-audit':'classifier_guidance.audit_sit_joint',
             'sit-joint-distributed-audit':'classifier_guidance.audit_sit_joint_distributed',
             'schedule-eval':'classifier_guidance.evaluate_schedule',
             'diffusion-head':'classifier_guidance.train_capacity',
             'capacity-eval':'classifier_guidance.evaluate_capacity',
             'capacity-audit':'classifier_guidance.audit_head_capacity',
             'capacity-convergence':'classifier_guidance.audit_capacity_convergence',
             'sit-adapter-eval':'classifier_guidance.evaluate_sit_transformer',
             'sit-adapter-audit':'classifier_guidance.audit_sit_transformer',
             'jit-ssg':'classifier_guidance.jit_ssg',
             'jit-schedule-probe':'classifier_guidance.probe_jit_schedule',
             'jit-schedule-distributed-audit':'classifier_guidance.audit_jit_schedule_distributed',
             'jit-schedule':'classifier_guidance.train_jit_schedule',
             'jit-ssg-eval':'classifier_guidance.evaluate_jit_ssg'}[args.task]
    command = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--virtual-local-rank',
               f'--nproc-per-node={len(devices)}','-m','--',entry,
               '--output',str(args.output.resolve()),*args.arguments]
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model',default='sit_small')
    selector.add_argument('--resume',type=Path)
    selector.add_argument('--head-checkpoint',type=Path)
    selector.add_argument('--checkpoint',type=Path)
    selection,_=selector.parse_known_args(args.arguments)
    from experiments.lifting_scale_sweep_20260909 import asset_paths
    assets={str(p.resolve()):c.sha(p) for p in asset_paths(selection.model)}
    for p in (selection.resume,selection.head_checkpoint,selection.checkpoint):
        if p is not None: assets[str(p.resolve())]=c.sha(p)
    c.atomic(args.output/'request.json',dict(created_utc=c.now(),sources=sources,assets=assets,command=command,
        devices=devices,allowed_desktop_pids=args.allow_desktop_pid,task=args.task,objective=('paired 5K evaluation of saved raw or EMA learned guidance, without parameter updates' if args.task=='learned-guidance-eval'
            else 'SiT joint weak head and signed full-trajectory scale endpoint GAN; soft independent-probe residual energy anchor' if args.task.startswith('sit-joint')
            else 'endpoint binary GAN fits full-trajectory JiT 1-block guidance coefficients; both predictors frozen' if args.task.startswith('jit-schedule')
            else 'read-only raw/EMA checkpoint convergence comparison' if args.task=='capacity-convergence'
            else 'SiT native adapter capacity and guidance-window comparison' if args.task.startswith('sit-adapter')
            else 'published SSG adapter on JiT, real-data capacity comparison' if args.task.startswith('jit-ssg')
            else 'matched head prediction / standalone head generation audit' if args.task=='capacity-audit'
            else 'ordinary real-data velocity MSE, frozen SiT depth4 features' if args.task=='diffusion-head'
            else '5K scalar extrapolation search on fixed diffusion-trained head' if args.task=='capacity-eval'
            else 'paired signed-schedule endpoint sampling and ADM evaluation' if args.task=='schedule-eval'
            else 'binary logistic real-RGB versus actual guided endpoint GAN with feature-space R1'),
        precision=(('published JiT BF16 predictors' if selection.model=='jit' else 'original SiT FP32/TF32 sampling') if args.task=='learned-guidance-eval'
            else 'published JiT BF16 predictor compute; FP32 scales and RGB feedback' if args.task.startswith('jit-schedule')
            else 'BF16 autocast / FP32 MSE; original fixed validation protocol' if args.task=='capacity-convergence'
            else 'BF16 autocast / FP32 parameters as SSG' if args.task.startswith('jit-ssg')
            else 'BF16 autocast, FP32 trainable weights, same as original diffusion head training' if args.task=='diffusion-head'
            else 'same as deployed source; no BF16 conversion of FP32 SiT'),
        coefficient=('50 signed learnable extra a_i, initialized .5; paper total w_i=1+a_i' if args.task.startswith('jit-schedule')
            else 'none; read-only head prediction errors' if args.task=='capacity-convergence'
            else 'extra a in S+a*f(t)*(S-W); explicit legacy or full interval' if args.task.startswith('sit-adapter')
            else 'none in training; sampling uses paper total w=1+a' if args.task.startswith('jit-ssg')
            else 'none; evaluate weak head itself' if args.task=='capacity-audit'
            else 'none in training objective' if args.task=='diffusion-head'
            else 'extra a in S+a*f(t)*(S-W); f=6/7,1,0 on the original intervals' if args.task=='capacity-eval'
            else 'signed per-step a_i in S+a_i(S-W), every step active' if args.task!='head'
            else 'extra a in S + a f(t)(S-W); total w=1+a for RAE'),
        old_queues='stop markers and run directories untouched'))
    for path,digest in sources.items():
        source=Path(path)
        relative = source.relative_to(c.WORK) if source.is_relative_to(c.WORK) else Path('external_sources')/digest/source.name
        target=args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        assert c.sha(target)==digest
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(r['uuid'] for r in devices),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_COMPILE_DISABLE='1',
             PYTHONUNBUFFERED='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    with (args.output/'worker.log').open('w') as stream:
        result=subprocess.run(command,cwd=c.WORK,env=env,stdout=stream,stderr=subprocess.STDOUT)
    c.atomic(args.output/'exit.json',dict(exit_code=result.returncode,finished_utc=c.now()))
    for lease in leases:lease.close()
    raise SystemExit(result.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',required=True,help='Explicit comma separated idle GPU indices or UUIDs')
    p.add_argument('--allow-desktop-pid',type=int,action='append',default=[],help='Admit a verified gnome-initial-setup PID without stopping it; all other idle checks remain')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=('head','schedule','schedule-eval','learned-guidance-eval','diffusion-head','capacity-eval','capacity-audit','capacity-convergence','sit-adapter-eval','sit-adapter-audit','sit-joint','sit-joint-audit','sit-joint-distributed-audit','jit-ssg','jit-ssg-eval','jit-schedule-probe','jit-schedule-distributed-audit','jit-schedule'),default='head')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    main(p.parse_args())

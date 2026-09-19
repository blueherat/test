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
    for key in requested:
        row = next(r for r in gpu_snapshot() if str(r['index']) == key or r['uuid'] == key)
        if not eligible(row):
            raise RuntimeError(f"GPU {key} is not idle: {row}")
        lease = Path('/tmp', f"eqvae_idle_{row['uuid']}.lock").open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not eligible(next(r for r in gpu_snapshot() if r['uuid'] == row['uuid'])):
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
    command = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--virtual-local-rank',
               f'--nproc-per-node={len(devices)}','-m','--','classifier_guidance.train',
               '--output',str(args.output.resolve()),*args.arguments]
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model',default='sit_small')
    selector.add_argument('--resume',type=Path)
    selector.add_argument('--head-checkpoint',type=Path)
    selection,_=selector.parse_known_args(args.arguments)
    from experiments.lifting_scale_sweep_20260909 import asset_paths
    assets={str(p.resolve()):c.sha(p) for p in asset_paths(selection.model)}
    for p in (selection.resume,selection.head_checkpoint):
        if p is not None: assets[str(p.resolve())]=c.sha(p)
    c.atomic(args.output/'request.json',dict(created_utc=c.now(),sources=sources,assets=assets,command=command,
        devices=devices,objective='binary logistic real-RGB versus actual guided endpoint GAN with feature-space R1',
        precision='same as deployed source; no BF16 conversion of FP32 SiT',
        coefficient='extra a in S + a f(t)(S-W); total w=1+a for RAE',
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
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    main(p.parse_args())

"""Endpoint GAN fits signed coefficients; original SiT and depth4 IG stay frozen."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.lifting_scale_sweep_20260909 import SMALL_CKPT, SMALL_HEAD
from .adapters import image_decoder
from .data import ImageNetData
from .features import enable_feedback_checkpointing
from .schedules import GuidanceSchedule, for_native_sit
from .training import step as gan_step


def main(args):
    if args.updates < 1 or args.warmup < 0 or args.global_batch < 1 or args.save_every < 1 or args.diagnostic_every < 1:
        raise ValueError('Invalid training counts')
    if not math.isfinite(args.coefficient):raise ValueError('Initial coefficient must be finite')
    rank, world = c.setup()
    if args.global_batch % world or torch.cuda.device_count() != 1:
        raise ValueError('Use the leased launcher with a divisible global batch')
    run = args.output
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest: raise RuntimeError(f'Source changed: {path}')
    torch.manual_seed(args.seed)
    adapter = Adapter('sit_small')
    adapter.model.eval().requires_grad_(False)
    adapter.native.eval().requires_grad_(False)
    frozen = dict(strong=fingerprint(adapter.model), weak=fingerprint(adapter.native))
    provenance = dict(strong=str(SMALL_CKPT), strong_sha256=c.sha(SMALL_CKPT),
                      weak=str(SMALL_HEAD), weak_sha256=c.sha(SMALL_HEAD),
                      weak_depth=4, weak_training_steps=50000, weak_weights='ema')
    schedule = GuidanceSchedule(args.steps,args.coefficient).cuda().eval()
    ema = copy.deepcopy(schedule).requires_grad_(False)
    critic = BinaryCritic(classes=100).cuda()
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    enable_feedback_checkpointing(adapter, feature)
    decode = image_decoder(adapter)
    data = ImageNetData('sit_small',args.seed+1009*rank)
    optimizer = torch.optim.Adam(schedule.parameters(),lr=args.lr_a,betas=(.9,.99))
    optimizer_d = torch.optim.Adam(critic.parameters(),lr=args.lr_d,betas=(0.,.99))
    start = 0
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['objective'] != 'binary_gan_signed_schedule' or state['frozen'] != frozen:
            raise ValueError('Checkpoint objective or frozen model differs')
        if state['provenance'] != provenance or state['args']['steps'] != args.steps:
            raise ValueError('Checkpoint source or solver differs')
        if state['args']['warmup'] != args.warmup:
            raise ValueError('Resume must preserve the checkpoint warmup boundary')
        schedule.load_state_dict(state['schedule']);ema.load_state_dict(state['ema'])
        critic.load_state_dict(state['critic'])
        optimizer.load_state_dict(state['optimizer']);optimizer_d.load_state_dict(state['optimizer_d'])
        for group in optimizer.param_groups: group['lr']=args.lr_a
        for group in optimizer_d.param_groups: group['lr']=args.lr_d
        if rank < len(state['rngs']): data.load_state_dict(state['rngs'][rank])
        start = state['step']
        if rank == 0:
            c.atomic(run/'resume.json',dict(step=start,exact_global_stream=len(state['rngs'])==world,
                checkpoint=str(args.resume),checkpoint_sha256=c.sha(args.resume)))
        del state
    else:
        total=torch.zeros(2048,device='cuda',dtype=torch.float64);squares=torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                real,_,_=data.draw(args.global_batch//world)
                values=feature(real).double();total+=values.sum(0);squares+=values.square().sum(0)
            if world>1:dist.all_reduce(total);dist.all_reduce(squares)
            count=16*args.global_batch;mean=total/count
            critic.feature_mean.copy_(mean.float())
            critic.feature_std.copy_((squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())
    example=torch.zeros(args.global_batch//world,4,32,32,device='cuda')
    labels=torch.zeros(len(example),device='cuda',dtype=torch.long)
    begin=time.perf_counter()
    sample=for_native_sit(adapter,schedule,example,labels,graphs=not args.eager)
    torch.cuda.synchronize()
    if rank==0:
        c.atomic(run/'initial_state.json',dict(provenance=provenance,frozen=frozen,
            coefficients=schedule.coefficients.tolist(),data=data.provenance,steps=args.steps,
            solver='Heun, left-step coefficient shared by both stages',trainable_parameters=args.steps,
            all_steps_active=True,signed=True,world_size=world,global_batch=args.global_batch,
            setup_seconds=time.perf_counter()-begin))
    del example,labels
    adapter.values.clear();c.barrier()

    def save(current,phase):
        assert all(p.grad is None for m in (adapter.model,adapter.native) for p in m.parameters())
        assert dict(strong=fingerprint(adapter.model),weak=fingerprint(adapter.native))==frozen
        signature=dict(schedule=fingerprint(schedule),ema=fingerprint(ema),critic=fingerprint(critic))
        payload=dict(signature=signature,rng=data.state_dict())
        states=[None]*world
        if world>1:dist.all_gather_object(states,payload)
        else:states[0]=payload
        assert all(s['signature']==signature for s in states)
        if rank==0:
            cpu=lambda m:{k:v.detach().cpu() for k,v in m.state_dict().items()}
            path=run/f'checkpoint_{current:06d}.pt'
            state=dict(step=current,objective='binary_gan_signed_schedule',model='sit_small',
                schedule=cpu(schedule),ema=cpu(ema),critic=cpu(critic),optimizer=optimizer.state_dict(),
                optimizer_d=optimizer_d.state_dict(),rngs=[s['rng'] for s in states],frozen=frozen,
                provenance=provenance,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                request_sha256=c.sha(run/'request.json'))
            temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
            c.atomic(run/'latest.json',dict(step=current,phase=phase,checkpoint=str(path),
                sha256=c.sha(path),frozen_unchanged=True,replicas_identical=True,
                coefficients=schedule.coefficients.tolist()))

    # For a new run updates counts coefficient updates AFTER critic warmup.
    target=start+args.updates+max(0,args.warmup-start)
    names=('d_ce','r1','g_loss','schedule_gradient_norm','real_accuracy','fake_accuracy','endpoint_rms')
    final=start
    for current in range(start+1,target+1):
        stopped=torch.tensor(int((run/'STOP_AFTER_CURRENT').exists()),device='cuda')
        if world>1:dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        if stopped.item():save(final,'paused');break
        updating=current>args.warmup
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        real,noise,labels=data.draw(args.global_batch//world)
        diagnostics={} if current==start+1 or current%args.diagnostic_every==0 else None
        metrics=gan_step(head=schedule,critic=critic,optimizer_w=optimizer,optimizer_d=optimizer_d,
            sample=sample,decode=decode,feature=feature,real=real,noise=noise,labels=labels,
            r1=args.r1,feature_chunk=0,update_head=updating,diagnostics=diagnostics)
        if updating:
            with torch.no_grad():ema.coefficients.lerp_(schedule.coefficients,1-args.ema)
        if world>1:dist.all_reduce(metrics);metrics/=world
        values=metrics.cpu().tolist();torch.cuda.synchronize()
        coefficients=schedule.coefficients.detach().cpu()
        if not torch.isfinite(coefficients).all():raise FloatingPointError('Nonfinite schedule')
        row=dict(zip(names,values),step=current,phase='training' if updating else 'critic_warmup',
            coefficient_updates=max(0,current-args.warmup),target_step=target,
            seconds=time.perf_counter()-begin,coefficients=coefficients.tolist(),
            coefficient_min=float(coefficients.min()),coefficient_max=float(coefficients.max()),
            negative_count=int((coefficients<0).sum()),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,updated_utc=c.now())
        middle=args.steps//2
        differences=coefficients[1:]-coefficients[:-1]
        row.update(boundary_left=float(coefficients[middle-1]),boundary_right=float(coefficients[middle]),
            boundary_jump=float(differences[middle-1]),boundary_abs_jump=float(differences[middle-1].abs()),
            max_adjacent_abs_jump=float(differences.abs().max()),
            mean_adjacent_abs_jump=float(differences.abs().mean()))
        if updating:
            row['gradient_first_half_norm']=float(schedule.coefficients.grad[:args.steps//2].norm())
            row['gradient_second_half_norm']=float(schedule.coefficients.grad[args.steps//2:].norm())
        if diagnostics is not None:
            # Keep one collective even for vector-valued per-coefficient probes.
            flat=torch.cat([v.reshape(-1) for v in diagnostics.values()])
            if world>1:dist.all_reduce(flat);flat/=world
            flat=flat.cpu();offset=0;observed={}
            for key,value in diagnostics.items():
                size=value.numel();part=flat[offset:offset+size];offset+=size
                observed[key]=part.tolist() if value.ndim else part.item()
            observed.update(step=current,coefficient_updates=max(0,current-args.warmup),updated_utc=c.now())
            if rank==0:
                with (run/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(observed)+'\n')
                c.atomic(run/'diagnostics_latest.json',observed)
        if rank==0:
            with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            c.atomic(run/'progress.json',row)
            print({k:v for k,v in row.items() if k!='coefficients'},flush=True)
        final=current;adapter.values.clear()
        del real,noise,labels,metrics
        if current%args.save_every==0:save(current,row['phase'])
    else:
        save(final,'complete')
        if rank==0:c.atomic(run/'complete.json',dict(complete=True,step=final,updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    p.add_argument('--updates',type=int,required=True,help='Schedule updates, excluding initial D warmup')
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--coefficient',type=float,default=.6,help='Initial first-half coefficient; tail starts at zero')
    p.add_argument('--global-batch',type=int,default=24)
    p.add_argument('--lr-a',type=float,default=1e-3)
    p.add_argument('--lr-d',type=float,default=1e-4)
    p.add_argument('--r1',type=float,default=1.)
    p.add_argument('--ema',type=float,default=.99)
    p.add_argument('--seed',type=int,default=2026091921)
    p.add_argument('--save-every',type=int,default=64)
    p.add_argument('--diagnostic-every',type=int,default=25)
    p.add_argument('--resume',type=Path)
    p.add_argument('--eager',action='store_true')
    main(p.parse_args())

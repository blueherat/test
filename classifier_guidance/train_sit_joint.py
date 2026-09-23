"""Endpoint GAN joint SiT training with a world-size-independent global stream."""
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
from .adapters import image_decoder
from .data import ImageNetData
from .evaluate_sit_transformer import load_head
from .features import enable_feedback_checkpointing
from .sit_joint import JointGuidance, GapProbe, make_sampler, joint_step

OBJECTIVE = 'sit_joint_signed_schedule_gap_anchor_v1'


def main(args):
    rank, world = c.setup()
    if torch.cuda.device_count() != 1:
        raise ValueError('Use one explicitly leased GPU per worker')
    if min(args.updates, args.microbatch, args.global_batch, args.save_every, args.profile_every) < 1:
        raise ValueError('Counts must be positive')
    if args.global_batch % (args.microbatch*world) or args.warmup < 0:
        raise ValueError('Invalid accumulation or warmup')
    if not all(math.isfinite(v) for v in (args.coefficient,args.lr_w,args.lr_a,args.lr_d,args.norm_weight)):
        raise ValueError('Nonfinite configuration')
    if min(args.lr_w,args.lr_a,args.lr_d) <= 0 or args.norm_weight < 0 or not 0 <= args.ema < 1:
        raise ValueError('Invalid optimizer/EMA configuration')
    run = args.output
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest:
            raise RuntimeError(f'Source changed: {path}')
    torch.manual_seed(args.seed)
    adapter = Adapter('sit_small')
    weak, source = load_head(adapter, args.head_checkpoint)
    if source['config']['variant'] != 'block1':
        raise ValueError('This first controlled experiment uses the native 1-block adapter')
    del source
    reference = copy.deepcopy(weak).requires_grad_(False)
    joint = JointGuidance(weak.requires_grad_(True),64,args.coefficient).cuda().eval()
    ema = copy.deepcopy(joint).requires_grad_(False)
    frozen = dict(strong=fingerprint(adapter.model), vae=fingerprint(adapter.rt.vae),
                  initial_weak=fingerprint(reference))
    provenance = dict(path=str(args.head_checkpoint.resolve()),sha256=c.sha(args.head_checkpoint),
                      variant='block1',depth=4,steps=50000,weights='ema')
    critic = BinaryCritic(classes=100).cuda()
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    enable_feedback_checkpointing(adapter, feature)
    decode = image_decoder(adapter)
    data = ImageNetData('sit_small',args.seed+1009)
    probe = GapProbe(adapter,joint,reference,args.seed+2018)
    local_count = args.global_batch//world

    def record(name,value):
        if rank==0:c.atomic(run/name,value)

    def draw():
        if rank==0:batch=data.draw(args.global_batch)
        else:
            batch=(torch.empty(args.global_batch,3,256,256,device='cuda'),
                   torch.empty(args.global_batch,4,32,32,device='cuda'),
                   torch.empty(args.global_batch,device='cuda',dtype=torch.long))
        if world>1:
            for value in batch:dist.broadcast(value,src=0)
        sl=slice(rank*local_count,(rank+1)*local_count)
        return tuple(value[sl] for value in batch),batch
    optimizer = torch.optim.Adam([
        dict(params=list(joint.weak.parameters()),lr=args.lr_w,name='weak'),
        dict(params=list(joint.schedule.parameters()),lr=args.lr_a,name='schedule')],betas=(.9,.99))
    optimizer_d = torch.optim.Adam(critic.parameters(),lr=args.lr_d,betas=(0.,.99))
    start = 0
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['objective'] != OBJECTIVE or state['frozen'] != frozen or state['provenance'] != provenance:
            raise ValueError('Resume objective, frozen assets or initial head differs')
        for key in ('warmup','coefficient','global_batch','microbatch','norm_weight','r1','ema','seed','lr_w','lr_a','lr_d'):
            if state['args'][key] != getattr(args,key):
                raise ValueError(f'Resume would change {key}')
        joint.load_state_dict(state['joint']); ema.load_state_dict(state['ema'])
        critic.load_state_dict(state['critic'])
        optimizer.load_state_dict(state['optimizer']); optimizer_d.load_state_dict(state['optimizer_d'])
        data.load_state_dict(state['data_rng'])
        probe.generator.set_state(state['probe_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        start = state['step']
        for module,key in ((joint,'joint'),(ema,'ema'),(critic,'critic')):
            assert all(torch.equal(v.detach().cpu(),state[key][name]) for name,v in module.state_dict().items())
        record('resume.json',dict(step=start,path=str(args.resume),sha256=c.sha(args.resume),
            weights_restored_exactly=True,both_optimizers_restored=True,data_and_probe_rng_restored=True,
            exact_global_stream=True,bitwise_training_trajectory=False,
            previous_world_size=state.get('world_size',1),world_size=world))
        del state
    else:
        total = torch.zeros(2048,device='cuda',dtype=torch.float64)
        squares = torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                (real,_,_),global_values = draw()
                for chunk in real.split(args.microbatch):
                    values = feature(chunk).double()
                    total += values.sum(0); squares += values.square().sum(0)
            if world>1:dist.all_reduce(total);dist.all_reduce(squares)
            count = 16*args.global_batch
            mean = total/count
            critic.feature_mean.copy_(mean.float())
            critic.feature_std.copy_((squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())
        del real, values, chunk,global_values
    # Graph capture precedes any ordinary autograd on the live joint leaves.
    begin = time.perf_counter()
    example = torch.zeros(args.microbatch,4,32,32,device='cuda')
    labels = torch.zeros(args.microbatch,device='cuda',dtype=torch.long)
    sample = make_sampler(adapter,joint,example,labels,graphs=not args.eager)
    torch.cuda.synchronize()
    record('initial_state.json',dict(provenance=provenance,frozen=frozen,data=data.provenance,
        coefficients=joint.schedule.coefficients.tolist(),start_step=start,steps=64,nfe=128,
        solver='FP32 TF32, 64 Heun steps; left-interval scale shared by both stages',
        weak_parameters=sum(p.numel() for p in joint.weak.parameters()),coefficient_parameters=64,
        global_batch=args.global_batch,microbatch=args.microbatch,accumulation=args.global_batch//args.microbatch,
        world_size=world,local_batch=local_count,local_accumulation=local_count//args.microbatch,
        probe='fresh real VAE posterior + fresh noise; uniform shared t; minibatch log energy ratio squared',
        setup_seconds=time.perf_counter()-begin))
    del example, labels
    adapter.values.clear()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}

    def save(step, phase):
        for module in (adapter.model,adapter.native,adapter.rt.vae,reference,feature):
            assert all(p.grad is None for p in module.parameters())
        assert frozen == dict(strong=fingerprint(adapter.model),vae=fingerprint(adapter.rt.vae),
                              initial_weak=fingerprint(reference))
        signature=dict(joint=fingerprint(joint),ema=fingerprint(ema),critic=fingerprint(critic))
        replicas=[None]*world
        local=dict(signature=signature,probe_rng=probe.generator.get_state())
        if world>1:dist.all_gather_object(replicas,local)
        else:replicas[0]=local
        assert all(r['signature']==signature and torch.equal(r['probe_rng'],local['probe_rng']) for r in replicas)
        if rank!=0:return
        cpu = lambda m:{k:v.detach().cpu() for k,v in m.state_dict().items()}
        state = dict(objective=OBJECTIVE,model='sit_small',step=step,joint_updates=max(0,step-args.warmup),
            joint=cpu(joint),ema=cpu(ema),critic=cpu(critic),optimizer=optimizer.state_dict(),
            optimizer_d=optimizer_d.state_dict(),data_rng=data.state_dict(),probe_rng=probe.generator.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),frozen=frozen,
            provenance=provenance,args=config,world_size=world,request_sha256=c.sha(run/'request.json'))
        path = run/f'checkpoint_{step:06d}.pt'
        temp = path.with_suffix('.tmp'); torch.save(state,temp); temp.replace(path)
        c.atomic(run/'latest.json',dict(step=step,phase=phase,checkpoint=str(path),sha256=c.sha(path),
            frozen_unchanged=True,joint_updates=max(0,step-args.warmup),
            world_size=world,replicas_identical=True,probe_rngs_identical=True))

    target = start+args.updates+max(0,args.warmup-start)
    final = start
    for current in range(start+1,target+1):
        stop_reason=0
        if rank==0:
            if (run/'STOP_AFTER_CURRENT').exists():stop_reason=1
            if args.stop_at_epoch and time.time()>=args.stop_at_epoch:stop_reason=2
        stopped=torch.tensor(stop_reason,device='cuda')
        if world>1:dist.broadcast(stopped,src=0)
        if stopped.item():
            save(final,'time_limit' if stopped.item()==2 else 'paused');break
        updating = current > args.warmup
        begin = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        (real,noise,labels),global_values = draw()
        metrics = joint_step(joint=joint,critic=critic,optimizer=optimizer,optimizer_d=optimizer_d,
            sample=sample,decode=decode,feature=feature,real=real,noise=noise,labels=labels,
            microbatch=args.microbatch,probe=probe,norm_weight=args.norm_weight,r1=args.r1,update=updating,
            probe_real=global_values[0][:args.microbatch],probe_labels=global_values[2][:args.microbatch])
        if updating:
            with torch.no_grad():
                for p,q in zip(ema.parameters(),joint.parameters()):p.lerp_(q,1-args.ema)
        coefficients = joint.schedule.coefficients.detach().cpu()
        if not torch.isfinite(coefficients).all():raise FloatingPointError('Nonfinite coefficients')
        torch.cuda.synchronize()
        perf=torch.tensor([time.perf_counter()-begin,torch.cuda.max_memory_allocated()/1024**3,
                           torch.cuda.max_memory_reserved()/1024**3],device='cuda',dtype=torch.float64)
        if world>1:dist.all_reduce(perf,op=dist.ReduceOp.MAX)
        seconds,allocated,reserved=perf.cpu().tolist()
        row = dict(metrics,step=current,joint_updates=max(0,current-args.warmup),target_step=target,
            phase='training' if updating else 'critic_warmup',seconds=seconds,
            world_size=world,global_batch=args.global_batch,microbatch=args.microbatch,
            local_batch=local_count,local_accumulation=local_count//args.microbatch,
            coefficients=coefficients.tolist(),ema_coefficients=ema.schedule.coefficients.tolist(),
            coefficient_min=coefficients.min().item(),coefficient_max=coefficients.max().item(),
            negative_count=int((coefficients<0).sum()),
            peak_allocated_gib=allocated,peak_reserved_gib=reserved,updated_utc=c.now())
        if rank==0:
            with (run/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            record('progress.json',row)
            print({k:v for k,v in row.items() if not isinstance(v,list)},flush=True)
        if (start==0 and current==1) or current%args.profile_every==0:
            profile = dict(step=current,distribution='fresh real posterior interpolants; raw joint head',
                rows=probe.profile(global_values[0][:args.microbatch],global_values[2][:args.microbatch]),updated_utc=c.now())
            if rank==0:
                with (run/'profiles.jsonl').open('a') as stream:stream.write(json.dumps(profile)+'\n')
                record('profile_latest.json',profile)
        adapter.values.clear()
        del real,noise,labels,global_values
        final = current
        if current%args.save_every==0:save(current,row['phase'])
    else:
        save(final,'complete')
        record('complete.json',dict(complete=True,step=final,updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    p.add_argument('--head-checkpoint',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True,help='Joint updates, excluding initial D warmup')
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--coefficient',type=float,default=.75)
    p.add_argument('--global-batch',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--lr-w',type=float,default=1e-6)
    p.add_argument('--lr-a',type=float,default=2e-4)
    p.add_argument('--lr-d',type=float,default=1e-4)
    p.add_argument('--norm-weight',type=float,default=.1)
    p.add_argument('--r1',type=float,default=1.)
    p.add_argument('--ema',type=float,default=.99)
    p.add_argument('--seed',type=int,default=2026092207)
    p.add_argument('--save-every',type=int,default=300)
    p.add_argument('--profile-every',type=int,default=100)
    p.add_argument('--resume',type=Path)
    p.add_argument('--stop-at-epoch',type=float,default=0,help='Save and stop at this UTC epoch deadline, checked before each update')
    p.add_argument('--eager',action='store_true')
    main(p.parse_args())

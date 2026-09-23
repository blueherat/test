"""Endpoint GAN learns 50 JiT guidance coefficients with both predictors frozen."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from experiments.train_imagenet100_sit_flow import capture_rng_state, restore_rng_state
from .jit_schedule import HEAD, load_runtime, constant_schedule, optimize, sampler, signature, decoder
from .data import ImageNetData
from .features import enable_feedback_checkpointing
from .training_accumulation import step as gan_step


def main(args):
    rank, world = c.setup()
    assert torch.cuda.device_count()==1, 'One explicitly leased GPU per worker'
    if min(args.updates, args.global_batch, args.save_every, args.diagnostic_every)<1 or args.warmup<0:
        raise ValueError('Invalid counts')
    if not math.isfinite(args.coefficient):
        raise ValueError('Initial coefficient must be finite')
    microbatch=args.microbatch or args.global_batch
    if microbatch<1 or args.global_batch%(microbatch*world):
        raise ValueError('Global batch must be divisible by microbatch times world size')
    local_count=args.global_batch//world
    run = args.output
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        assert c.sha(path)==digest, path
    torch.set_float32_matmul_precision('high')
    runtime, provenance = load_runtime(args.head_checkpoint)
    original_frozen = signature(runtime.net)
    saved_bytes = optimize(runtime, precast=True, checkpoint_backbone=args.checkpoint_backbone)
    frozen = signature(runtime.net)
    torch.manual_seed(args.seed)
    schedule = constant_schedule(args.coefficient)
    ema = copy.deepcopy(schedule).requires_grad_(False)
    critic = BinaryCritic(classes=1000).cuda()
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    if args.checkpoint_feedback:
        enable_feedback_checkpointing(SimpleNamespace(name='jit'), feature)
    data = ImageNetData('jit', args.seed)

    def draw():
        # Draw the identical global stream on rank zero even when world size
        # changes. Other ranks receive disjoint slices, never duplicate samples.
        if rank==0:
            batch=data.draw(args.global_batch)
        else:
            batch=(torch.empty(args.global_batch,3,256,256,device='cuda'),
                   torch.empty(args.global_batch,3,256,256,device='cuda'),
                   torch.empty(args.global_batch,device='cuda',dtype=torch.long))
        if world>1:
            for value in batch:dist.broadcast(value,src=0)
        sl=slice(rank*local_count,(rank+1)*local_count)
        return tuple(value[sl] for value in batch), batch

    def record(name, value):
        if rank==0:c.atomic(run/name,value)

    def reduce_diagnostics(diagnostics):
        flat=torch.cat([value.reshape(-1) for value in diagnostics.values()])
        if world>1:dist.all_reduce(flat);flat/=world
        flat=flat.cpu();offset=0;observed={}
        for key,value in diagnostics.items():
            size=value.numel();part=flat[offset:offset+size];offset+=size
            observed[key]=part.tolist() if value.ndim else part.item()
        return observed
    optimizer = torch.optim.Adam(schedule.parameters(), lr=args.lr_a, betas=(.9, .99))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0., .99))
    config = dict(global_batch=args.global_batch, microbatch=microbatch, warmup=args.warmup, initial_extra_a=args.coefficient,
                  lr_a=args.lr_a, lr_d=args.lr_d, r1=args.r1, ema=args.ema, seed=args.seed,
                  checkpoint_feedback=args.checkpoint_feedback, checkpoint_backbone=args.checkpoint_backbone,
                  precast=True, steps=50, nfe=99, solver='49 Heun plus final Euler', cfg=1,
                  signed=True, all_steps_active=True, trainable_parameters=50,
                  objective='binary non-saturating real-RGB endpoint GAN; feature-space R1')
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['objective']=='jit_block1_gan_signed_schedule'
        old_config=dict(state['config'])
        old_config.setdefault('microbatch',old_config['global_batch'])
        changed = [key for key in config if old_config.get(key)!=config[key]]
        if changed and not (args.allow_batch_change and set(changed)<={'global_batch','microbatch'}):
            raise ValueError(f'Resume configuration differs: {changed}')
        assert state['provenance']==provenance and state['frozen']==frozen
        schedule.load_state_dict(state['schedule']); ema.load_state_dict(state['ema'])
        critic.load_state_dict(state['critic'])
        optimizer.load_state_dict(state['optimizer']); optimizer_d.load_state_dict(state['optimizer_d'])
        data.load_state_dict(state['data_rng'])
        runtime_rngs=state.get('runtime_rngs',[state['runtime_rng']])
        restore_rng_state(runtime_rngs[rank%len(runtime_rngs)], torch.device('cuda'))
        start = state['step']
        for module, key in ((schedule,'schedule'),(ema,'ema'),(critic,'critic')):
            assert all(torch.equal(v.detach().cpu(), state[key][n]) for n,v in module.state_dict().items())
        record('resume.json', dict(step=start, checkpoint=str(args.resume), sha256=c.sha(args.resume),
                  weights_restored_exactly=True, both_optimizers_restored=True, data_rng_restored=True,
                  previous_global_batch=state['config']['global_batch'], global_batch=args.global_batch,
                  microbatch=microbatch, accumulation_steps=args.global_batch//microbatch,
                  world_size=world, previous_world_size=state.get('world_size',1),
                  local_batch=local_count, local_accumulation_steps=local_count//microbatch,
                  exact_global_stream=not changed,
                  bitwise_training_trajectory=False,
                  batch_change_note='Larger fresh batches change subsequent stochastic trajectory' if changed else None))
        del state
    else:
        total = torch.zeros(2048, device='cuda', dtype=torch.float64); squares = torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                (real, _, _), global_batch = draw()
                values = torch.cat([feature(x) for x in real.split(microbatch)]).double()
                total += values.sum(0); squares += values.square().sum(0)
            if world>1:dist.all_reduce(total);dist.all_reduce(squares)
            count = 16*args.global_batch; mean = total/count
            critic.feature_mean.copy_(mean.float())
            critic.feature_std.copy_((squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())
        del global_batch,real,values
    noise = torch.zeros(microbatch, 3, 256, 256, device='cuda')
    labels = torch.zeros(microbatch, device='cuda', dtype=torch.long)
    begin = time.perf_counter()
    sample = sampler(runtime, schedule, noise, labels)
    torch.cuda.synchronize()
    record('initial_state.json', dict(config=config, provenance=provenance, data=data.provenance,
              world_size=world,local_batch=local_count,local_accumulation_steps=local_count//microbatch,
              original_frozen=original_frozen, frozen=frozen, precast_saved_weight_bytes=saved_bytes,
              coefficients=schedule.coefficients.tolist(), setup_seconds=time.perf_counter()-begin))
    del noise, labels
    c.barrier()

    def save(step, phase):
        assert signature(runtime.net)==frozen
        assert all(not p.requires_grad and p.grad is None for p in runtime.net.parameters())
        replica=dict(schedule=signature(schedule),ema=signature(ema),critic=signature(critic))
        local=dict(signature=replica,runtime_rng=capture_rng_state(torch.device('cuda')))
        replicas=[None]*world
        if world>1:dist.all_gather_object(replicas,local)
        else:replicas[0]=local
        assert all(item['signature']==replica for item in replicas), 'Worker replicas diverged'
        if rank!=0:return
        cpu = lambda m: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
        payload = dict(objective='jit_block1_gan_signed_schedule', step=step, config=config,
                       schedule=cpu(schedule), ema=cpu(ema), critic=cpu(critic),
                       optimizer=optimizer.state_dict(), optimizer_d=optimizer_d.state_dict(),
                       data_rng=data.state_dict(), runtime_rng=capture_rng_state(torch.device('cuda')),
                       runtime_rngs=[item['runtime_rng'] for item in replicas],world_size=world,
                       provenance=provenance, frozen=frozen, request_sha256=c.sha(run/'request.json'))
        path = run/f'checkpoint_{step:06d}.pt'
        torch.save(payload, path.with_suffix('.tmp')); path.with_suffix('.tmp').replace(path)
        c.atomic(run/'latest.json', dict(step=step, phase=phase, checkpoint=str(path), sha256=c.sha(path),
                  frozen_unchanged=True, coefficients=schedule.coefficients.tolist(),
                  replicas_identical=True,world_size=world,
                  ema_coefficients=ema.coefficients.tolist()))

    target = start+args.updates+max(0,args.warmup-start)
    final = start
    names = ('d_ce','r1','g_loss','schedule_gradient_norm','real_accuracy','fake_accuracy','endpoint_rms')
    for current in range(start+1, target+1):
        stopped=torch.tensor(int((run/'STOP_AFTER_CURRENT').exists()),device='cuda')
        if world>1:dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        if stopped.item():
            save(final, 'paused');break
        updating = current>args.warmup
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.perf_counter()
        (real, noise, labels), global_batch = draw()
        diagnostics = {} if current==start+1 or current==args.warmup+1 or current%args.diagnostic_every==0 else None
        metrics = gan_step(head=schedule, critic=critic, optimizer_w=optimizer, optimizer_d=optimizer_d,
                           sample=sample, decode=decoder, feature=feature, real=real, noise=noise, labels=labels,
                           microbatch=microbatch, r1=args.r1, update_head=updating, diagnostics=diagnostics)
        if updating:
            with torch.no_grad():
                ema.coefficients.lerp_(schedule.coefficients, 1-args.ema)
        if world>1:
            metrics[-1].square_()
            dist.all_reduce(metrics);metrics/=world
            metrics[-1].sqrt_()
        values = metrics.cpu().tolist(); coefficients = schedule.coefficients.detach().cpu()
        assert torch.isfinite(coefficients).all()
        torch.cuda.synchronize()
        perf=torch.tensor([time.perf_counter()-begin,torch.cuda.max_memory_allocated()/1024**3,
                           torch.cuda.max_memory_reserved()/1024**3],device='cuda',dtype=torch.float64)
        if world>1:dist.all_reduce(perf,op=dist.ReduceOp.MAX)
        seconds,allocated,reserved=perf.cpu().tolist()
        row = dict(zip(names, values), step=current, target_step=target,
                   phase='training' if updating else 'critic_warmup', coefficient_updates=max(0,current-args.warmup),
                   global_batch=args.global_batch, microbatch=microbatch, accumulation_steps=args.global_batch//microbatch,
                   world_size=world,local_batch=local_count,local_accumulation_steps=local_count//microbatch,
                   seconds=seconds, coefficients=coefficients.tolist(),ema_coefficients=ema.coefficients.tolist(),
                   coefficient_min=float(coefficients.min()), coefficient_max=float(coefficients.max()),
                   negative_count=int((coefficients<0).sum()),
                   max_adjacent_abs_jump=float(coefficients.diff().abs().max()),
                   peak_allocated_gib=allocated,peak_reserved_gib=reserved,updated_utc=c.now())
        if rank==0:
            row.update(labels=global_batch[2].cpu().tolist(),
                       noise_sha256=c.original.array_sha(global_batch[1].cpu().numpy()),
                       real_sha256=c.original.array_sha(global_batch[0].cpu().numpy()))
        if updating:
            row.update(gradient_first_half_norm=float(schedule.coefficients.grad[:25].norm()),
                       gradient_second_half_norm=float(schedule.coefficients.grad[25:].norm()))
        if diagnostics is not None:
            observed = reduce_diagnostics(diagnostics)
            observed.update(step=current,coefficient_updates=max(0,current-args.warmup),updated_utc=c.now())
            record('diagnostics_latest.json', observed)
            if rank==0:
                with (run/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(observed)+'\n')
        if rank==0:
            with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            record('progress.json', row)
            print({n:v for n,v in row.items() if n not in ('coefficients','ema_coefficients','labels')}, flush=True)
        final = current
        del real, noise, labels, metrics, global_batch
        if current%args.save_every==0:
            save(current, row['phase'])
    else:
        save(final, 'complete')
        record('complete.json', dict(complete=True, step=final, coefficient_updates=max(0,final-args.warmup),
                                    updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('jit',),default='jit')
    p.add_argument('--head-checkpoint',type=Path,default=HEAD)
    p.add_argument('--updates',type=int,default=128)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--global-batch',type=int,default=8)
    p.add_argument('--microbatch',type=int,help='Actual model batch; accumulate one global D update and one scale update')
    p.add_argument('--coefficient',type=float,default=.5)
    p.add_argument('--lr-a',type=float,default=1e-3)
    p.add_argument('--lr-d',type=float,default=1e-4)
    p.add_argument('--r1',type=float,default=1.)
    p.add_argument('--ema',type=float,default=.99)
    p.add_argument('--seed',type=int,default=2026092201)
    p.add_argument('--save-every',type=int,default=300)
    p.add_argument('--diagnostic-every',type=int,default=16)
    p.add_argument('--checkpoint-feedback',action='store_true')
    p.add_argument('--checkpoint-backbone',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--allow-batch-change',action='store_true',help='Explicitly permit only a global-batch change on resume')
    main(p.parse_args())

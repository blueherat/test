"""Reusable binary endpoint-GAN trainer; use classifier_guidance.launch for GPUs."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

# Original JiT functions are deliberately eager; acceleration is explicit replay.
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
import torch
import torch.distributed as dist
from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.guidance_dynamic_50k_20260915.models import fingerprint
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from .adapters import load, image_decoder, materialize_autocast_weights
from .data import ImageNetData
from .features import enable_feedback_checkpointing, enable_backbone_checkpointing
from .sampler import for_adapter
from .training import step as train_step


def main(args):
    if args.updates < 1 or args.global_batch < 1 or args.feature_chunk < 1 or args.save_every < 1:
        raise ValueError('updates, batch, feature chunk and save interval must be positive')
    import math
    if not math.isfinite(args.coefficient) or args.coefficient == 0:
        raise ValueError('Training coefficient must be finite and nonzero; a=0 has no weak-head gradient')
    rank, world = c.setup()
    if args.global_batch % world:
        raise ValueError('global batch must be divisible by world size')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Use the launcher with --virtual-local-rank: one visible GPU per worker')
    count = args.global_batch//world
    run = args.output
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest:
            raise RuntimeError(f'Source changed since launch: {path}')
    torch.manual_seed(args.seed)
    adapter, head, provenance = load(args.model, args.head_checkpoint, args.head_key)
    if args.precast:
        materialize_autocast_weights(adapter)
    if args.model == 'jit' and not args.resume and provenance['steps'] < 50000 and not args.engineering_fixture:
        raise ValueError('JiT only has a 3K MLP fixture here; supply a formal head before quality training, or explicitly select --engineering-fixture for system tests')
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    critic = BinaryCritic(classes=adapter.cfg['classes']).cuda()
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    decode = image_decoder(adapter)
    if args.feedback == 'checkpoint':
        enable_feedback_checkpointing(adapter, feature)
    if args.checkpoint_backbone:
        enable_backbone_checkpointing(adapter)
    data = ImageNetData(args.model, args.seed + 1009*rank)
    ow = torch.optim.Adam(head.parameters(),lr=args.lr_w,betas=(.9,.99))
    od = torch.optim.Adam(critic.parameters(),lr=args.lr_d,betas=(0.,.99))
    start = 0
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=False)
        assert state['objective'] == 'binary_gan'
        assert state.get('model','sit_small') == args.model
        assert state.get('data_protocol') == 'real_rgb_official_continuous_v2'
        head.load_state_dict(state['head']);ema.load_state_dict(state['ema']);critic.load_state_dict(state['critic'])
        ow.load_state_dict(state['optimizer_w']);od.load_state_dict(state['optimizer_d'])
        for group in ow.param_groups: group['lr'] = args.lr_w
        for group in od.param_groups: group['lr'] = args.lr_d
        start = int(state['step'])
        if rank < len(state['rngs']): data.load_state_dict(state['rngs'][rank])
        provenance = dict(kind='binary_resume',path=str(args.resume),sha256=c.sha(args.resume),
                          steps=start,initial_fixture=provenance)
        if rank == 0:
            c.atomic(run/'resume.json',dict(checkpoint=str(args.resume),sha256=c.sha(args.resume),
                saved_world_size=len(state['rngs']),world_size=world,exact_global_stream=len(state['rngs'])==world))
        del state
    else:
        total=torch.zeros(2048,device='cuda',dtype=torch.float64);squares=torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                real,_,_=data.draw(count);values=feature(real).double()
                total+=values.sum(0);squares+=values.square().sum(0)
            if world>1: dist.all_reduce(total);dist.all_reduce(squares)
            n=16*count*world;mean=total/n
            critic.feature_mean.copy_(mean.float())
            critic.feature_std.copy_((squares/n-mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())

    # Capture uses synthetic scratch inputs; it does not consume the data RNG.
    example=torch.zeros((count,*adapter.cfg['shape']),device='cuda')
    labels=torch.zeros(count,device='cuda',dtype=torch.long)
    setup=time.perf_counter()
    sampler=for_adapter(adapter,head,example,labels,args.coefficient,graphs=not args.eager)
    torch.cuda.synchronize()
    if rank == 0:
        c.atomic(run/'initial_state.json',dict(head=fingerprint(head),critic=fingerprint(critic),
            head_provenance=provenance,data=data.provenance,setup_seconds=time.perf_counter()-setup,
            model=args.model,steps=64 if args.model=='sit_small' else 100,world_size=world,
            precision='original',feedback=args.feedback))
    del example,labels
    adapter.values.clear()
    c.barrier()

    def save(step, phase):
        signature=dict(head=fingerprint(head),critic=fingerprint(critic),ema=fingerprint(ema))
        payload=dict(rng=data.state_dict(),signature=signature)
        replicas=[None]*world
        if world>1: dist.all_gather_object(replicas,payload)
        else: replicas[0]=payload
        assert all(r['signature']==signature for r in replicas), 'Worker parameters diverged'
        if rank==0:
            cpu=lambda m:{k:v.detach().cpu() for k,v in m.state_dict().items()}
            state=dict(step=step,objective='binary_gan',model=args.model,data_protocol='real_rgb_official_continuous_v2',
                head=cpu(head),ema=cpu(ema),critic=cpu(critic),optimizer_w=ow.state_dict(),optimizer_d=od.state_dict(),
                rngs=[r['rng'] for r in replicas],args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                request_sha256=c.sha(run/'request.json'))
            path=run/f'checkpoint_{step:06d}.pt';temporary=path.with_suffix('.tmp')
            torch.save(state,temporary);temporary.replace(path)
            c.atomic(run/'latest.json',dict(phase=phase,step=step,checkpoint=str(path),sha256=c.sha(path),replicas_identical=True))

    names=('d_ce','r1','g_loss','head_gradient_norm','real_accuracy','fake_accuracy','endpoint_rms')
    final=start
    for current in range(start+1,start+args.updates+1):
        stopped=torch.tensor(int((run/'STOP_AFTER_CURRENT').exists()),device='cuda')
        if world>1: dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        if stopped.item():
            save(final,'paused');break
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        real,noise,labels=data.draw(count)
        metrics=train_step(head=head,critic=critic,optimizer_w=ow,optimizer_d=od,
            sample=sampler,decode=decode,feature=feature,real=real,noise=noise,labels=labels,
            r1=args.r1,feature_chunk=args.feature_chunk if args.feedback=='chunk' else 0,
            update_head=current>args.warmup)
        if current>args.warmup:
            with torch.no_grad():
                for a,b in zip(ema.parameters(),head.parameters()): a.lerp_(b,1-args.ema)
        if world>1: dist.all_reduce(metrics);metrics/=world
        values=metrics.cpu().tolist();torch.cuda.synchronize()
        row=dict(zip(names,values),step=current,seconds=time.perf_counter()-begin,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,updated_utc=c.now())
        if rank==0:
            with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            c.atomic(run/'progress.json',dict(row,phase='training',target_step=start+args.updates))
            print(row,flush=True)
        final=current;adapter.values.clear()
        del real,noise,labels,metrics
        if current % args.save_every==0: save(current,'training')
    else:
        save(final,'complete')
        if rank==0:
            c.atomic(run/'complete.json',dict(complete=True,step=final,model=args.model,updated_utc=c.now()))
            c.atomic(run/'progress.json',dict(phase='complete',step=final,model=args.model,updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small','jit','raev2'),default='sit_small')
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--global-batch',type=int,default=24)
    p.add_argument('--resume',type=Path)
    p.add_argument('--head-checkpoint',type=Path,help='Explicit initial MLP head checkpoint')
    p.add_argument('--head-key',default='ema',help='Nested state key, e.g. ema.mlp or ema.context')
    p.add_argument('--coefficient',type=float,required=True,help='Extra coefficient a: S + a f(t) (S-W); for RAE total w=1+a')
    p.add_argument('--feedback',choices=('checkpoint','chunk','direct'),default='checkpoint')
    p.add_argument('--feature-chunk',type=int,default=1)
    p.add_argument('--eager',action='store_true')
    p.add_argument('--precast',action='store_true',help='Materialize frozen JiT/RAEv2 Linear/Conv weights in their existing BF16 compute dtype')
    p.add_argument('--checkpoint-backbone',action='store_true',help='Trade extra frozen-block recomputation for less graph memory')
    p.add_argument('--engineering-fixture',action='store_true')
    p.add_argument('--seed',type=int,default=2026091597)
    p.add_argument('--lr-w',type=float,default=1e-6)
    p.add_argument('--lr-d',type=float,default=1e-4)
    p.add_argument('--r1',type=float,default=1.)
    p.add_argument('--ema',type=float,default=.99)
    p.add_argument('--warmup',type=int,default=16)
    p.add_argument('--save-every',type=int,default=100)
    main(p.parse_args())

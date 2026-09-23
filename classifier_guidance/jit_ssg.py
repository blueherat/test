"""SSG's published JiT adapter, with a detached frozen prefix and real-data training."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from experiments import jit_internal_guidance as jig
from experiments import train_imagenet100_sit_flow as base
from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.data import RealDataset
from experiments.guidance_dynamic_50k_20260915.models import fingerprint

ROOT = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/jit_ssg_capacity_20260920')
LITERATURE = ROOT / 'literature'


class Readout(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.blocks = net.ssg_adapter_blocks
        self.final = net.intermediate_layer
        self.rope = net.feat_rope_incontext
        self.context_length = net.in_context_len

    def forward(self, tokens, condition):
        for block in self.blocks:
            tokens = block(tokens, condition, self.rope)
        return jig.unpatchify(self.final(tokens[:, self.context_length:], condition))


class Runtime:
    def __init__(self, blocks):
        receipt = c.read(LITERATURE / 'manifest.json')
        for row in receipt['files']:
            if c.sha(LITERATURE / row['file']) != row['sha256']:
                raise RuntimeError(f"SSG source changed: {row['file']}")
        assert (LITERATURE / 'model_jit.py').read_bytes() == (jig.REPO / 'model_jit.py').read_bytes()
        spec = importlib.util.spec_from_file_location('eqvae_pinned_ssg', LITERATURE / 'model_jit_ssg.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        torch.manual_seed(2026092093)
        net = module.JiTSSG_models['JiT-B/16'](input_size=256, in_channels=3, num_classes=1000,
            attn_drop=0., proj_drop=0., ssg_layer=6, ssg_adapter_depth=blocks,
            ssg_adapter_init_layer=13-blocks if blocks else 0)
        checkpoint = torch.load(jig.CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
        weights = {name.removeprefix('net.'): value for name, value in checkpoint['model_ema1'].items()}
        missing, unexpected = net.load_state_dict(weights, strict=False)
        assert not unexpected and all(n.startswith(('ssg_adapter_blocks.', 'intermediate_layer.')) for n in missing)
        net.copy_next_blocks_to_adapter(); net.copy_final_head_to_intermediate()
        self.net = net.cuda().eval().requires_grad_(False)
        for rope in (net.feat_rope, net.feat_rope_incontext):
            rope.freqs_cos = rope.freqs_cos.cuda(); rope.freqs_sin = rope.freqs_sin.cuda()
        self.head = Readout(net).cuda().eval()

    def frozen_hash(self):
        digest = hashlib.sha256()
        for name, value in self.net.state_dict().items():
            if name.startswith(('ssg_adapter_blocks.', 'intermediate_layer.')): continue
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def prefix(self, z, t, labels):
        tokens, condition, y = self.net._prepare_tokens(z, t, labels)
        for index in range(6): tokens = self.net._run_block(tokens, condition, y, index)
        return tokens, condition


def data_loader(split, batch, seed):
    data = RealDataset('jit', split)
    sampler = DistributedSampler(data, num_replicas=1, rank=0, shuffle=split=='train',
                                 seed=seed, drop_last=split=='train')
    loader = DataLoader(data, batch_size=batch, sampler=sampler, drop_last=split=='train',
        num_workers=4 if split=='train' else 2, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, generator=torch.Generator().manual_seed(seed+999))
    return loader, sampler


def draw(batch, generator, training=True):
    image, labels, ids = batch
    image = image.cuda(non_blocking=True).mul(2).sub(1); labels = labels.cuda(non_blocking=True)
    n = len(labels)
    if training:
        flips = torch.rand(n, generator=generator, device='cuda') < .5
        image = torch.where(flips[:,None,None,None], image.flip(-1), image)
        labels = torch.where(torch.rand(n, generator=generator, device='cuda') < .1, 1000, labels)
    t = torch.sigmoid(torch.randn(n, generator=generator, device='cuda')*.8-.8)
    noise = torch.randn(image.shape, generator=generator, device='cuda')
    z = t[:,None,None,None]*image + (1-t[:,None,None,None])*noise
    return image, z, t, labels, ids


def velocity_loss(pred, clean, z, t):
    denominator = (1-t[:,None,None,None]).clamp_min(.05)
    # Match DenoiserSSG.forward including the target's clamp and reduction order.
    target = (clean-z)/denominator
    predicted = (pred.float()-z)/denominator
    return (target-predicted).square().mean(dim=(1,2,3)).mean()


@torch.no_grad()
def validate(runtime, head, run, step):
    loader, _ = data_loader('validation', 32, 2026092093)
    generator = torch.Generator(device='cuda').manual_seed(2026092094)
    total = 0.; count = 0
    for raw in loader:
        clean, z, t, labels, _ = draw(raw, generator, training=False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, condition = runtime.prefix(z,t,labels)
            pred = head(tokens,condition)
        total += float(velocity_loss(pred,clean,z,t))*len(t); count += len(t)
    c.atomic(run/f'validation_{step:06d}.json',dict(step=step, samples=count, ema_velocity_mse=total/count,
        same_validation_inputs_for_all_heads=True))


def train(args):
    rank, world = c.setup()
    if world != 1: raise ValueError('One independent head per GPU')
    if args.global_batch % args.microbatch: raise ValueError('Microbatch must divide global batch')
    torch.set_float32_matmul_precision('high')
    run = args.output
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest: raise RuntimeError(f'Source changed: {path}')
    runtime = Runtime(args.blocks)
    frozen = runtime.frozen_hash()
    head = runtime.head.requires_grad_(True).train()
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(.9,.95), weight_decay=0., fused=True)
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    loader, sampler = data_loader('train', args.global_batch, args.seed)
    config = dict(blocks=args.blocks, attachment_layer=6, initialized_from_layers=list(range(13-args.blocks,13)),
        copied_pretrained_final_head=True, pretrained_checkpoint=str(jig.CHECKPOINT), checkpoint_sha256=c.sha(jig.CHECKPOINT),
        source_commit=c.read(LITERATURE/'manifest.json')['commit'], training_images=len(loader.dataset),
        global_batch=args.global_batch, microbatch=args.microbatch, seed=args.seed,
        lr=args.lr, betas=[.9,.95], weight_decay=0., ema=.9999, precision='BF16 autocast, FP32 parameters',
        loss='published SSG velocity MSE from clean prediction; t denominator clamp .05',
        source='all real training images; epoch shuffle, fresh flips, time, noise and label dropout',
        P_mean=-.8, P_std=.8, label_drop=.1, parameters=sum(p.numel() for p in head.parameters()))
    start = 0
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=False)
        assert state['config']==config and state['frozen']==frozen
        head.load_state_dict(state['head']); ema.load_state_dict(state['ema']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['data_rng']); base.restore_rng_state(state['runtime_rng'],torch.device('cuda'))
        start = state['step']
    assert start < args.steps
    c.atomic(run/'initial_state.json',dict(config=config,resumed_step=start,frozen=frozen))
    stream = base.infinite_train_batches(loader,sampler,start)

    def save(step,phase):
        assert runtime.frozen_hash()==frozen
        assert all(p.grad is None for n,p in runtime.net.named_parameters()
                   if not n.startswith(('ssg_adapter_blocks.','intermediate_layer.')))
        path = run/f'checkpoint_{step:06d}.pt'
        state = dict(step=step,config=config,frozen=frozen,
            head={n:v.detach().cpu() for n,v in head.state_dict().items()},
            ema={n:v.detach().cpu() for n,v in ema.state_dict().items()},optimizer=optimizer.state_dict(),
            data_rng=generator.get_state(),runtime_rng=base.capture_rng_state(torch.device('cuda')),
            request_sha256=c.sha(run/'request.json'))
        torch.save(state,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
        c.atomic(run/'latest.json',dict(step=step,phase=phase,checkpoint=str(path),sha256=c.sha(path),strong_unchanged=True))

    begin = time.perf_counter(); total = 0.; count = 0; first_batches=[]
    for step in range(start+1,args.steps+1):
        if (run/'STOP_AFTER_CURRENT').exists(): save(step-1,'paused'); return
        clean,z,t,labels,ids = draw(next(stream),generator)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.zeros((),device='cuda')
        for offset in range(0,args.global_batch,args.microbatch):
            sl=slice(offset,offset+args.microbatch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens,condition=runtime.prefix(z[sl],t[sl],labels[sl])
                pred=head(tokens,condition)
            loss=velocity_loss(pred,clean[sl],z[sl],t[sl])
            if step==start+1 and offset==0:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    reference=runtime.net.forward_intermediate(z[sl],t[sl],labels[sl])
                torch.testing.assert_close(pred,reference,rtol=0,atol=0)
                reference_loss=velocity_loss(reference,clean[sl],z[sl],t[sl])
                torch.testing.assert_close(loss,reference_loss,rtol=0,atol=0)
                assert not tokens.requires_grad and not condition.requires_grad
                c.atomic(run/'objective_audit.json',dict(passed=True,official_forward_exact=True,
                    official_loss_exact=True,detached_frozen_prefix=True,loss=float(loss.detach())))
            (loss*(args.microbatch/args.global_batch)).backward()
            batch_loss += loss.detach()*(args.microbatch/args.global_batch)
        if not torch.isfinite(batch_loss): raise FloatingPointError(f'Loss at {step}')
        if step==start+1:
            grad_norms={n:float(p.grad.norm()) for n,p in head.named_parameters()}
            assert all(np.isfinite(v) for v in grad_norms.values())
            c.atomic(run/'gradient_audit.json',dict(parameter_gradient_norms=grad_norms))
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()),.9999)
            torch._foreach_add_(list(ema.parameters()),list(head.parameters()),alpha=.0001)
        total += float(batch_loss);count += 1
        if step<=start+4:
            first_batches.append(dict(step=step,ids=ids.tolist(),
                input_sha256=c.original.array_sha(z.cpu().numpy()),time_sha256=c.original.array_sha(t.cpu().numpy()),
                labels=labels.cpu().tolist()))
            c.atomic(run/'first_batches.json',first_batches)
        if step==start+1 or step%100==0 or step==args.steps:
            elapsed=(time.perf_counter()-begin)/count
            row=dict(step=step,target_steps=args.steps,loss=total/count,seconds_per_step=elapsed,
                remaining_hours=(args.steps-step)*elapsed/3600,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,updated_utc=c.now())
            c.atomic(run/'progress.json',row)
            with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True);total=0.;count=0;begin=time.perf_counter()
        if step%5000==0: validate(runtime,ema,run,step);begin=time.perf_counter()
        if step%args.save_every==0 or step==args.steps:
            save(step,'complete' if step==args.steps else 'training');begin=time.perf_counter()
    c.atomic(run/'complete.json',dict(complete=True,step=args.steps,config=config))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('jit',),default='jit')
    p.add_argument('--blocks',type=int,choices=(0,1,2),required=True)
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--global-batch',type=int,default=256)
    p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--seed',type=int,default=2026092093)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--resume',type=Path)
    train(p.parse_args())

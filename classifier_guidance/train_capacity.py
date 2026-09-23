"""Full-data flow-matching MSE; only the intermediate readout head is trained."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.data import Stream
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.training import compute_loss
from experiments import train_imagenet100_sit_flow as base
from .capacity_heads import make, describe


def main(args):
    rank, world = c.setup()
    if args.global_batch % world or torch.cuda.device_count() != 1:
        raise ValueError('Use the GPU lease launcher and a divisible batch')
    if args.steps < 1 or args.save_every < 1:
        raise ValueError('Counts must be positive')
    run = args.output
    request = c.read(run / 'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest:
            raise RuntimeError(f'Source changed before launch: {path}')
    k.GLOBAL_BATCH = args.global_batch
    context = SimpleNamespace(rank=rank, world_size=world, device=torch.device('cuda', 0))
    adapter = Adapter('sit_small')
    adapter.model.eval().requires_grad_(False)
    frozen = fingerprint(adapter.model)
    torch.manual_seed(args.seed)
    head = make(args.variant, adapter.model).cuda().train()
    calibration = k.model_root('sit_small') / 'calibration/head.pt'
    initial = torch.load(calibration, map_location='cpu', weights_only=False)['initial']
    # MLPs share fixed normalization estimated from the real training split.
    # Their weights are random; native adapters copy the strong model's tail.
    normalized = hasattr(head, 'token_mean')
    if normalized:
        with torch.no_grad():
            for name in ('token_mean', 'token_std', 'condition_mean', 'condition_std'):
                getattr(head, name).copy_(initial[name])
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(.9, .999),
                                  weight_decay=0., fused=True)
    generator = torch.Generator(device='cuda').manual_seed(args.seed + rank)
    config = dict(variant=args.variant, architecture=describe(head), seed=args.seed,
                  global_batch=args.global_batch, world_size=world, lr=args.lr,
                  betas=[.9, .999], weight_decay=0., ema=.9999, precision='bf16',
                  calibration_sha256=c.sha(calibration), data_root=str(k.SIT_DATA),
                  loss='mean((W(tx+(1-t)eps,t,c) - (x-eps))^2)',
                  time='Uniform[0,1]', source='full real train, fresh VAE posterior/time/noise')
    if not normalized:
        config['initialization'] = 'copied pretrained SiT tail and final layer; no MLP normalization'
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['objective'] != 'capacity_diffusion_only' or state['config'] != config or state['frozen'] != frozen:
            raise ValueError('Resume configuration, objective or strong model differs')
        head.load_state_dict(state['head']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['rngs'][rank]['data'])
        base.restore_rng_state(state['rngs'][rank]['runtime'], context.device)
        start = state['step']
    if start >= args.steps:
        raise ValueError('Target must exceed the resumed step')
    initial_parameters = {name: p.detach().clone() for name, p in head.named_parameters()}
    wrapped = DDP(head, device_ids=[0], broadcast_buffers=False, static_graph=True,
                  gradient_as_bucket_view=True) if world > 1 else head
    stream = Stream('sit_small', 'real', context, start_step=start)
    manifest = c.read(k.SIT_DATA / 'manifest.json')
    if rank == 0:
        c.atomic(run / 'initial_state.json', dict(config=config, frozen=frozen,
            resumed_step=start, data_manifest=manifest, training_images=len(stream.loader.dataset),
            normalization_weights_only=normalized,
            head_start=('random' if normalized else 'copied pretrained SiT tail and final layer') if not args.resume else 'exact resume',
            source_losses_include_no_gan_or_guided_target=True))

    def save(step, phase):
        assert fingerprint(adapter.model) == frozen
        assert all(p.grad is None and not p.requires_grad for p in adapter.model.parameters())
        signature = dict(head=fingerprint(head), ema=fingerprint(ema))
        local = dict(signature=signature, rng=dict(data=generator.get_state(),
                     runtime=base.capture_rng_state(context.device)))
        states = [None] * world
        if world > 1: dist.all_gather_object(states, local)
        else: states[0] = local
        assert all(s['signature'] == signature for s in states)
        if rank == 0:
            cpu = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
            path = run / f'checkpoint_{step:06d}.pt'
            state = dict(objective='capacity_diffusion_only', step=step, config=config,
                         head=cpu(head), ema=cpu(ema), optimizer=optimizer.state_dict(),
                         rngs=[s['rng'] for s in states], frozen=frozen,
                         request_sha256=c.sha(run / 'request.json'))
            torch.save(state, path.with_suffix('.tmp')); path.with_suffix('.tmp').replace(path)
            c.atomic(run / 'latest.json', dict(step=step, phase=phase, checkpoint=str(path),
                sha256=c.sha(path), strong_unchanged=True, replicas_identical=True,
                parameter_deltas={n: float((p - initial_parameters[n]).norm()) for n, p in head.named_parameters()}))
        c.barrier()

    begin = time.perf_counter(); total = torch.zeros((), device='cuda'); count = 0
    audit_data = []
    for step in range(start + 1, args.steps + 1):
        if step % 100 == 1:
            stop = torch.tensor(int((run / 'STOP_AFTER_CURRENT').exists()), device='cuda')
            if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                save(step - 1, 'paused'); return
        batch = stream.draw(generator)
        t = batch['t'][:, None, None, None]
        z = t * batch['positive'] + (1 - t) * batch['noise']
        target = adapter.patch(batch['positive'] - batch['noise'])
        with torch.no_grad(), adapter.autocast(training=True):
            features = adapter.features(z, batch['t'], batch['labels'])
        optimizer.zero_grad(set_to_none=True)
        with adapter.autocast(training=True):
            prediction = wrapped(features['context'], features['condition'])
        # Match the historical ordinary-FM reduction order as well as its math.
        loss = (prediction.float() - target).square().flatten(1).mean(1).mean()
        if step == start + 1:
            with torch.no_grad():
                reference = compute_loss(adapter, head, batch, dict(loss='fm', source='real'))
            torch.testing.assert_close(loss.detach(), reference, rtol=0, atol=0)
            if rank == 0:
                c.atomic(run / 'objective_audit.json', dict(passed=True, ordinary_fm_exact=True,
                    fm_loss=float(reference), features_detached=not features['context'].requires_grad,
                    no_full_teacher_prediction_in_target=True))
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward(); optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), .9999)
            torch._foreach_add_(list(ema.parameters()), list(head.parameters()), alpha=.0001)
        total += loss.detach(); count += 1
        if step <= start + 4:
            audit_data.append(dict(step=step, ids=batch['pos_id'].cpu().tolist(),
                                   noise_sha256=c.original.array_sha(batch['noise'].cpu().numpy()),
                                   time_sha256=c.original.array_sha(batch['t'].cpu().numpy())))
            if rank == 0: c.atomic(run / 'first_batches.json', audit_data)
        if step == start + 1 or step % 100 == 0 or step == args.steps:
            if world > 1: dist.all_reduce(total)
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - begin) / count
            row = dict(step=step, target_steps=args.steps, loss=float(total / (count * world)),
                       seconds_per_step=seconds, estimated_remaining_hours=(args.steps-step)*seconds/3600,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, updated_utc=c.now())
            if rank == 0:
                c.atomic(run / 'progress.json', row)
                with (run / 'train.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
                print(row, flush=True)
            total.zero_(); count = 0; begin = time.perf_counter()
        if step % 5000 == 0:
            validation = Stream('sit_small', 'real', context, validation=True)
            vg = torch.Generator(device='cuda').manual_seed(args.seed + 7001 + rank)
            values = torch.zeros(2, device='cuda', dtype=torch.float64)
            with torch.no_grad():
                for _ in range(min(8, len(validation.loader))):
                    b = validation.draw(vg)
                    value = compute_loss(adapter, ema, b, dict(loss='fm', source='real'))
                    values[0] += value.double() * len(b['t']); values[1] += len(b['t'])
            if world > 1: dist.all_reduce(values)
            if rank == 0:
                c.atomic(run / f'validation_{step:06d}.json', dict(step=step,
                    ema_velocity_mse=float(values[0]/values[1]), samples=int(values[1])))
            del validation; begin = time.perf_counter()
        adapter.values.clear()
        if step % args.save_every == 0 or step == args.steps:
            save(step, 'complete' if step == args.steps else 'training'); begin = time.perf_counter()
    if rank == 0:
        c.atomic(run / 'complete.json', dict(complete=True, step=args.steps, config=config, updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('sit_small',), default='sit_small')
    p.add_argument('--variant', choices=('shallow', 'moderate', 'large', 'linear', 'block1', 'block2'), required=True)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--global-batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--resume', type=Path)
    main(p.parse_args())

"""Correctness and paired full-GAN benchmarks for the frozen JiT 1-block schedule."""
import argparse
import copy
import gc
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from .jit_schedule import HEAD, load_runtime, constant_schedule, optimize, sampler, reference, signature, decoder
from .evaluate_jit_ssg import integrate, official_field
from .data import ImageNetData
from .features import enable_feedback_checkpointing
from .training_accumulation import step


def difference(actual, expected):
    a, b = actual.double().flatten(), expected.double().flatten()
    return dict(max_abs=float((a-b).abs().max()), relative=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)))


def audit(runtime, schedule, run):
    generator = torch.Generator(device='cuda').manual_seed(2026092203)
    z = torch.randn((1, 3, 256, 256), device='cuda', generator=generator)
    labels = torch.tensor([23], device='cuda')
    before = signature(runtime.net)
    with torch.no_grad():
        expected = integrate(official_field(runtime, 1.5, 1.), z, labels)
        actual = reference(runtime, schedule, z, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    unoptimized = sampler(runtime, schedule, z, labels, graphs=False)
    noise = z.clone().requires_grad_(True)
    endpoint = reference(runtime, schedule, noise, labels)
    gradients = torch.autograd.grad(endpoint.square().mean(), (noise, schedule.coefficients))
    actual = unoptimized(noise, labels)
    eager_gradients = torch.autograd.grad(actual.square().mean(), (noise, schedule.coefficients))
    eager_checks = {name: difference(a, b) for name, a, b in zip(('input', 'schedule'), eager_gradients, gradients)}
    assert max(x['relative'] for x in eager_checks.values()) < .005, eager_checks
    assert signature(runtime.net) == before
    endpoint_reference = endpoint.detach().clone()
    del endpoint, actual, unoptimized
    gc.collect()
    # An audit must not retain a default-stream AccumulateGrad node while
    # capturing its VJP on a side stream. Use fresh identical leaf storage.
    schedule = copy.deepcopy(schedule)
    optimize(runtime, precast=True)
    frozen = signature(runtime.net)
    engine = sampler(runtime, schedule, z, labels)
    actual = engine(noise, labels)
    actual_gradients = torch.autograd.grad(actual.square().mean(), (noise, schedule.coefficients))
    torch.testing.assert_close(actual, endpoint_reference, rtol=0, atol=0)
    graph_checks = {name: difference(a, b) for name, a, b in zip(('input', 'schedule'), actual_gradients, gradients)}
    assert max(x['relative'] for x in graph_checks.values()) < .005, graph_checks
    assert int((actual_gradients[1].abs()>1e-10).sum()) == 50
    first = dict(official_constant_endpoint_exact=True, optimized_endpoint_exact=True,
                 eager_gradient=eager_checks, optimized_gradient=graph_checks,
                 nonzero_schedule_gradients=50, first25_norm=float(actual_gradients[1][:25].norm()),
                 last25_norm=float(actual_gradients[1][25:].norm()))
    del endpoint_reference, actual, gradients, actual_gradients, eager_gradients
    # New labels/noise and signed scales also refresh captured graph inputs.
    with torch.no_grad():
        schedule.coefficients.copy_(torch.linspace(-.1, .6, 50, device='cuda'))
        schedule.coefficients[25] = 0.
    noise = torch.randn(z.shape, generator=generator, device='cuda', requires_grad=True)
    labels = torch.tensor([917], device='cuda')
    expected = reference(runtime, schedule, noise, labels)
    eg = torch.autograd.grad(expected.square().mean(), (noise, schedule.coefficients))
    actual = engine(noise, labels)
    ag = torch.autograd.grad(actual.square().mean(), (noise, schedule.coefficients))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    second = {name: difference(a, b) for name, a, b in zip(('input', 'schedule'), ag, eg)}
    assert max(x['relative'] for x in second.values()) < .005, second
    assert abs(float(ag[1][25])) > 1e-10
    assert signature(runtime.net) == frozen and all(p.grad is None for p in runtime.net.parameters())
    c.atomic(run/'result.json', dict(passed=True, constant=first, changed_signed_schedule=second,
        zero_coefficient_gradient=float(ag[1][25]), frozen_unchanged=True,
        solver='49 Heun plus 1 Euler; full discrete input/scale gradients; no CFG'))


def benchmark(runtime, schedule, args):
    microbatch=args.microbatch or args.batch
    if args.batch%microbatch:raise ValueError('Microbatch must divide global batch')
    if args.mode != 'baseline':
        optimize(runtime, precast=args.precast, checkpoint_backbone=args.checkpoint_backbone)
    frozen = signature(runtime.net)
    torch.manual_seed(2026092201)
    critic = BinaryCritic(classes=1000).cuda()
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    if args.checkpoint_feedback:
        enable_feedback_checkpointing(SimpleNamespace(name='jit'), feature)
    real, noise, labels = ImageNetData('jit', 2026092201).draw(args.batch)
    optimizer = torch.optim.Adam(schedule.parameters(), lr=1e-3, betas=(.9, .99))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0., .99))
    cpu = lambda m: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
    initial = dict(schedule=cpu(schedule), critic=cpu(critic), optimizer=optimizer.state_dict(), optimizer_d=optimizer_d.state_dict())
    start = time.perf_counter()
    sample = ((lambda z,y: reference(runtime, schedule, z, y)) if args.mode=='baseline'
              else sampler(runtime, schedule, noise[:microbatch], labels[:microbatch]))
    torch.cuda.synchronize(); setup = time.perf_counter()-start
    rows = []
    for index in range(args.repeats+1):
        schedule.load_state_dict(initial['schedule']); critic.load_state_dict(initial['critic'])
        optimizer.load_state_dict(copy.deepcopy(initial['optimizer']))
        optimizer_d.load_state_dict(copy.deepcopy(initial['optimizer_d']))
        optimizer.zero_grad(set_to_none=True); optimizer_d.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = step(head=schedule, critic=critic, optimizer_w=optimizer, optimizer_d=optimizer_d,
                       sample=sample, decode=decoder, feature=feature, real=real, noise=noise, labels=labels,
                       microbatch=microbatch)
        torch.cuda.synchronize()
        row = dict(seconds=time.perf_counter()-start, metrics=metrics.tolist(),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        rows.append(row); print(index, row, flush=True)
        c.atomic(args.output/'progress.json', dict(iterations=rows))
        np.save(args.output/f'gradient_{index}.npy', schedule.coefficients.grad.cpu().numpy())
    np.save(args.output/'schedule_after.npy', schedule.coefficients.detach().cpu().numpy())
    np.save(args.output/'critic_after.npy', torch.cat([p.detach().flatten() for p in critic.parameters()]).cpu().numpy())
    assert signature(runtime.net)==frozen and all(p.grad is None for p in runtime.net.parameters())
    result = dict(complete=True, mode=args.mode, batch=args.batch, microbatch=microbatch,
                  accumulation_steps=args.batch//microbatch, repeats=args.repeats,
                  precast=args.precast, checkpoint_feedback=args.checkpoint_feedback,
                  checkpoint_backbone=args.checkpoint_backbone, initial_extra_a=.5, initial_total_w=1.5,
                  setup_seconds=setup, iterations=rows,
                  median_seconds=statistics.median(r['seconds'] for r in rows[1:]),
                  peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows[1:]),
                  peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows[1:]),
                  frozen_unchanged=True, full_steps=50, nfe=99, cfg=1,
                  excludes='one-time loading, capture, data I/O and checkpoint saves',
                  included='full endpoint sampling, RGB feedback, D update with R1, full schedule backward and Adam')
    c.atomic(args.output/'result.json', result)


def main(args):
    _, world = c.setup(); assert world==1
    request = c.read(args.output/'request.json')
    for path,digest in request['sources'].items():
        assert c.sha(path)==digest, path
    torch.set_float32_matmul_precision('high')
    runtime, provenance = load_runtime(args.head_checkpoint)
    schedule = constant_schedule()
    c.atomic(args.output/'provenance.json', provenance)
    if args.mode=='audit': audit(runtime, schedule, args.output)
    else: benchmark(runtime, schedule, args)
    c.atomic(args.output/'complete.json', dict(complete=True, updated_utc=c.now()))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('jit',),default='jit')
    p.add_argument('--head-checkpoint',type=Path,default=HEAD)
    p.add_argument('--mode',choices=('audit','baseline','optimized'),required=True)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--microbatch',type=int)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--precast',action='store_true')
    p.add_argument('--checkpoint-feedback',action='store_true')
    p.add_argument('--checkpoint-backbone',action='store_true')
    main(p.parse_args())

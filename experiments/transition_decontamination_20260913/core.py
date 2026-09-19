import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from experiments.context_reference_5k_20260912 import reference
from experiments.small_sit_guidance_tuning_20260910 import adg_field
import experiments.small_sit_guidance_tuning_20260910 as angular
from experiments.sit_guidance_portfolio_20260910 import operators as prior
from . import kernel

ROOT = c.EXPS / 'transition_decontamination_20260913'
PROTOCOL = c.WORK / 'docs/TRANSITION_DECONTAMINATION_PROTOCOL_20260913_ZH.md'
ARMS = ('ode', 'ode_competitor', 'sde', 'positive', 'mean_gaussian', 'moment_gaussian')
TRACKS = ('ig', 'cfg')
BATCH = 8


def configure():
    c.ROOT = ROOT


def stage(track, n=400):
    assert track in TRACKS and n in (400, 1000)
    return f'{track}_' + ('screen_400' if n == 400 else 'confirm_1000')


def seed(track, n=400):
    return (2026121431 if track == 'ig' else 2026121433) + (10 if n == 1000 else 0)


def verify(track, n=400):
    p = ROOT / 'sit_small' / stage(track, n) / 'request.json'
    return p, local.verify_request(p)


def prepare(track, n=400):
    configure()
    assert c.read(ROOT / 'numerical_checks.json')['passed']
    if n == 1000:
        assert c.read(ROOT / 'decision.json')[track]['passes_gate']
    bank = c.prepare_bank('sit_small', stage(track, n), n, seed(track, n))
    tr = reference.m.training_root('sit_small')
    local.verify_request(tr / 'request.json')
    from experiments.guidance_pasted_20260912.audit import REFS
    sources = c.source_manifest([Path(__file__), Path(__file__).with_name('run.py'),
        c.WORK/'experiments/guidance_pasted_20260912/evaluate.py',
        c.WORK/'experiments/guidance_pasted_20260912/audit.py', Path(kernel.__file__), Path(reference.__file__),
        Path(local.__file__), Path(angular.__file__), Path(prior.__file__),
        c.WORK/'experiments/compute_adm_fid.py', c.WORK/'train_gen/evaluator.py', PROTOCOL])
    request = dict(track=track, model='SiT-S/2 ImageNet100', samples=n, batch=BATCH,
        stage=stage(track, n), seed=seed(track, n), arms=list(ARMS), training_updates=0,
        single_path=True, full_calls=128 if track == 'ig' else 224, prefix_calls=0,
        sources=sources, assets={str(p): c.sha(p) for p in (*c.asset_paths('sit_small'), REFS['sit_small'])},
        heads={str(tr/f): c.sha(tr/f) for f in ('head.pt', 'summary.json', 'request.json')},
        inputs={str(p): c.sha(p) for p in (*bank.glob('*.npy'), ROOT/'kernel_lut.npz', ROOT/'numerical_checks.json')},
        stochastic_rule='128 Euler; diffusion rate 2*(1-t)^2; kappa=alpha/(1+alpha)',
        rule='positive FID <= every other arm - 2 and IS >= .9*ode; no parameter sweep')
    path = ROOT / 'sit_small' / stage(track, n) / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)


def runtime():
    rt = c.runtime('sit_small')
    rt.provenance = reference.install(rt)
    rt.block_counts = [0] * len(rt.model.blocks)
    rt.head_count = 0
    rt.counter_handles = []
    for i, block in enumerate(rt.model.blocks):
        def count(module, args, index=i):
            rt.block_counts[index] += 1
        rt.counter_handles.append(block.register_forward_pre_hook(count))
    def head_count(module, args):
        rt.head_count += 1
    rt.counter_handles.append(rt.head.module.register_forward_pre_hook(head_count))
    return rt


def query(rt, z, t, track, active):
    if not active:
        return rt.field(z, t, 'full'), None
    if track == 'ig':
        return rt.pair(z, t)
    strong = rt.field(z, t, 'full')
    labels = rt.labels
    try:
        rt.labels = torch.full_like(labels, 100)
        weak = rt.field(z, t, 'full')
    finally:
        rt.labels = labels
    return strong, weak


@torch.inference_mode()
def sample(rt, lut, noise, labels, track, arm, random_seed, zero=False):
    before, blocks_before, heads_before = rt.counts.copy(), rt.block_counts.copy(), rt.head_count
    rt.labels = labels
    z = noise.clone()
    if arm.startswith('ode'):
        rt.grid = torch.linspace(0, 1, 65, device='cuda')
        history, pending = {}, {}
        def field(x, t, left, step, substage):
            alpha = 0. if zero else c.amount(rt, left, track)
            strong, weak = query(rt, x, t, track, alpha != 0)
            if weak is None:
                return strong
            if arm == 'ode_competitor':
                if track == 'ig':
                    return adg_field(x, strong, weak, t, alpha)[0]
                gap = strong - weak
                momentum = gap - .5 * history.get('momentum', torch.zeros_like(gap))
                bounded = prior.cap(momentum, 2 * prior.norm(gap))
                direction = bounded - prior.projection(bounded, x + (1 - t) * strong)
                if substage == 0:
                    pending['momentum'] = momentum.detach()
                else:
                    history['momentum'] = pending['momentum']
                return strong + 2. * direction
            return strong + alpha * (strong - weak)
        z, _ = c.integrate(rt, z, labels, field)
    else:
        generator = torch.Generator(device='cuda').manual_seed(random_seed)
        increments = torch.randn((128, *z.shape), device='cuda', generator=generator)
        grid = torch.linspace(0, 1, 129, device='cuda')
        for index, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
            alpha = 0. if zero else c.amount(rt, index / 128, track)
            strong, weak = query(rt, z, t, track, alpha != 0)
            h = u - t
            multiplier = 1 + t * (1 - t)
            ms = z + h * (multiplier * strong - (1 - t) * z)
            mw = ms if weak is None else z + h * (multiplier * weak - (1 - t) * z)
            sigma = torch.sqrt(2 * h) * (1 - t)
            if arm == 'sde':
                guided = strong if weak is None else strong + alpha * (strong - weak)
                z = z + h * (multiplier * guided - (1 - t) * z) + sigma * increments[index]
            else:
                delta = None if weak is None else h * multiplier * (weak - strong)
                z = lut.update(ms, mw, sigma, increments[index], alpha, arm, delta=delta)
    assert torch.isfinite(z).all() and z.abs().max() < 1e6
    calls = {k: rt.counts[k] - before[k] for k in before}
    block_calls = np.array(rt.block_counts) - blocks_before
    head_calls = rt.head_count - heads_before
    expected = 128 if track == 'ig' or zero else 224
    assert calls == dict(full=expected, prefix=0), calls
    assert np.all(block_calls == expected), block_calls
    assert head_calls == (64 if track == 'ig' and not zero else 0), head_calls
    return z, dict(full_calls=expected, prefix_calls=0, head_calls=head_calls, block_calls=block_calls)


@torch.inference_mode()
def preflight(rt, lut, n=400):
    from experiments.cfg_null_readout_20260913 import core as old_cfg
    checks = {}
    for track in TRACKS:
        noise, _, labels = c.bank('sit_small', stage(track, n))
        x, y = c.cuda(noise[:2]), c.cuda(labels[:2])
        result, counts = sample(rt, lut, x, y, track, 'ode', 17)
        if track == 'ig':
            expected, old_counts = reference.sample(rt, x, y)
        else:
            expected, old_counts = old_cfg.sample(rt, {}, x, y, 'native_base')
        assert torch.equal(result, expected), float((result - expected).abs().max())
        if track == 'cfg':
            result_apg, _ = sample(rt, lut, x, y, track, 'ode_competitor', 17)
            old_apg, _ = old_cfg.sample(rt, {}, x, y, 'apg')
            assert torch.equal(result_apg, old_apg)
        strong_zero = sample(rt, lut, x, y, track, 'sde', 17, zero=True)[0]
        for arm in ('positive', 'mean_gaussian', 'moment_gaussian'):
            actual_zero, _ = sample(rt, lut, x, y, track, arm, 17, zero=True)
            assert torch.equal(actual_zero, strong_zero)
            actual, cost = sample(rt, lut, x, y, track, arm, 17)
            assert torch.isfinite(actual).all()
        # Independent explicit affine drift verifies the chosen Gaussian SDE baseline.
        actual, cost = sample(rt, lut, x, y, track, 'sde', 17)
        rt.labels = y
        g = torch.Generator(device='cuda').manual_seed(17)
        noises = torch.randn((128, *x.shape), device='cuda', generator=g)
        z = x.clone()
        for i in range(128):
            t = z.new_tensor(i / 128)
            a = c.amount(rt, i / 128, track)
            s, w = query(rt, z, t, track, a != 0)
            v = s if w is None else s + a * (s - w)
            z = z + ((1 + t * (1-t)) * v - (1-t) * z) / 128 + (2/128)**.5 * (1-t) * noises[i]
        error = float((actual - z).abs().max())
        # Equivalent FP32 arithmetic accumulates differently over a full nonlinear trajectory.
        assert error < 2e-4, error
        checks[track] = dict(ode_reference_exact=True, zero_guidance_exact=True,
                             sde_independent_drift_max_error=error, measured_full_calls=cost['full_calls'])
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in rt.head.module.parameters())
    return dict(passed=True, checks=checks, provenance=rt.provenance)


@torch.inference_mode()
def worker(rank, world, parent, n=400):
    configure()
    for track in TRACKS:
        verify(track, n)
    rt, lut = runtime(), kernel.Kernel(ROOT / 'kernel_lut.npz')
    for track in TRACKS:
        path, request = verify(track, n)
        rh = c.sha(path)
        noise, _, labels = c.bank('sit_small', stage(track, n))
        for arm in ARMS:
            for start in range(rank * BATCH, n, world * BATCH):
                c.check_parent(parent)
                out = ROOT / 'sit_small' / stage(track, n) / arm / f'rank{rank}/batch{start:04d}.npz'
                if out.exists():
                    assert c.read(out.with_suffix('.json'))['sha256'] == c.sha(out)
                    continue
                x, y = c.cuda(noise[start:start+BATCH]), c.cuda(labels[start:start+BATCH])
                step_seed = seed(track, n) + 100000 + start
                torch.cuda.synchronize()
                begin = time.perf_counter()
                z, counts = sample(rt, lut, x, y, track, arm, step_seed)
                pixels = rt.decode(z)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                c.save_batch(out, pixels, z.cpu().numpy(), y.cpu().numpy(),
                    dict(seconds=elapsed, step_seed=step_seed, **counts), rh, start, c.array_sha(noise[start:start+len(y)]))
                c.atomic(ROOT / f'progress{rank}.json', dict(pid=os.getpid(), phase='sampling', track=track,
                    arm=arm, start=start, samples=n, batch=len(y)))
            print(rank, track, arm, 'finished', flush=True)


def collect(track, arm, n=400):
    configure()
    rp, request = verify(track, n)
    root = ROOT / 'sit_small' / stage(track, n) / arm
    files = sorted(root.glob('rank*/batch*.npz'), key=lambda p: int(p.stem[5:]))
    noise, _, labels = c.bank('sit_small', stage(track, n))
    images, records, starts, seconds = [], [], [], 0.
    for p in files:
        meta = c.read(p.with_suffix('.json'))
        assert meta['sha256'] == c.sha(p)
        with np.load(p) as d:
            start, count = int(d['start']), len(d['arr_0'])
            np.testing.assert_array_equal(d['labels'], labels[start:start+count])
            assert str(d['noise_sha256']) == c.array_sha(noise[start:start+count])
            assert str(d['request_sha256']) == c.sha(rp)
            assert int(d['step_seed']) == seed(track, n) + 100000 + start
            assert int(d['full_calls']) == request['full_calls'] and int(d['prefix_calls']) == 0
            assert np.all(d['block_calls'] == request['full_calls'])
            assert int(d['head_calls']) == (64 if track == 'ig' else 0)
            assert d['arr_0'].dtype == np.uint8 and d['arr_0'].shape == (count, 256, 256, 3)
            assert np.isfinite(d['latents']).all()
            images.append(d['arr_0']); starts.extend(range(start, start+count)); seconds += float(d['seconds'])
        records.append(dict(file=str(p), sha256=c.sha(p)))
    assert starts == list(range(n))
    path = root / 'samples.npz'
    np.savez(path, arr_0=np.concatenate(images))
    result = dict(complete=True, model='sit_small', track=track, arm=arm, stage=stage(track, n),
        primary_samples=n, generated_paths=n, full_calls_per_output=request['full_calls'],
        prefix_calls_at_inference=0, head_calls=64 if track == 'ig' else 0, seconds=seconds,
        samples_sha256=c.sha(path), request_sha256=c.sha(rp), records=records)
    c.atomic(root / 'summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true'); p.add_argument('--preflight', action='store_true')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world', type=int, default=3)
    p.add_argument('--parent', type=int, default=0); p.add_argument('--n', type=int, default=400)
    a = p.parse_args(); configure()
    if a.preflight:
        for track in TRACKS:
            c.prepare_bank('sit_small', stage(track, a.n), a.n, seed(track, a.n))
        result = preflight(runtime(), kernel.Kernel(ROOT/'kernel_lut.npz'), a.n)
        c.atomic(ROOT/'implementation_checks.json', result)
        print(result, flush=True)
    elif a.worker:
        worker(a.rank, a.world, a.parent, a.n)

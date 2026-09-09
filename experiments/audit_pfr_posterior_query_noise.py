"""Repeated posterior-coupled weak queries on unchanged IG trajectories."""
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_training_core import file_sha256


def tensor_hash(x):
    return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sit', 'raev2'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    root = Path('/home/zhoushunyu/data/eqvae/experiments/pfr_posterior_query_noise_20260908') / args.model
    root.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / 'experiments/raev2_training_core.py']
    if args.model == 'sit':
        from experiments.run_internal_guidance_sit_audit import load_model
        from experiments.audit_official_sit_pfr_interface import prefix
        repo = ROOT / 'research_repos/internal_guidance_study/Internal-Guidance/SiT'
        ck = Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
        model, meta = load_model(repo=repo, checkpoint_path=ck, model_name='SiT-XL/2', encoder_depth=8, state_key='ema', device=device)
        source_paths += [ROOT / 'experiments/run_internal_guidance_sit_audit.py', ROOT / 'experiments/audit_official_sit_pfr_interface.py', repo / 'models/sit.py', repo / 'samplers.py']
        shape = (4, 32, 32)
        grid = torch.linspace(1, 0, 101, dtype=torch.float64, device=device)
        scale = 1.35

        def pair(z, t, y):
            f, b, _ = model(z.float(), t, y)
            return f, b, b + scale * (f - b), z.float() - t[:, None, None, None] * b

        def weak(z, t, y):
            return z.float() - t[:, None, None, None] * prefix(model, z.float(), t, y)
    else:
        from experiments import sample_raev2_pfr_retiming as n
        cfg = n.load_config(n.DEFAULT_CONFIG)
        ck = n.DEFAULT_CHECKPOINT
        model = n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
        state = torch.load(ck, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(state['ema'], strict=True)
        del state
        meta = dict(config=str(n.DEFAULT_CONFIG), diagnostic_precision='fp32')
        source_paths += [Path(n.__file__), ROOT / 'experiments/raev2_pfr_retiming.py', n.DEFAULT_CONFIG,
                         ROOT / 'external/RAEv2/src/stage2/models/dit.py']
        # Capture the actual class source, avoiding assumptions about its module path.
        import inspect
        source_paths[-1] = Path(inspect.getfile(type(model)))
        shape = tuple(cfg.misc.latent_size)
        grid = n.shifted_time_grid(100, 8., device)
        scale = 1.78

        def velocity(clean, z, t):
            return n.clean_to_velocity(clean, z, t, denominator_floor=float(cfg.transport.t_eps))

        def pair(z, t, y):
            f, b = model(z, t, context=y, attn_mask=None)
            s = scale if float(cfg.guidance.ig.t_min) <= float(t[0]) <= float(cfg.guidance.ig.t_max) else 1.
            return velocity(f, z, t), velocity(b, z, t), velocity(b + s * (f - b), z, t), b

        def weak(z, t, y):
            return n.evaluate_base_head_only(model, z, t, context=y, attn_mask=None)
    gc.collect()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_paths = list(dict.fromkeys(source_paths))
    sources = {str(p): file_sha256(p) for p in source_paths}
    for i, p in enumerate(source_paths):
        shutil.copy2(p, root / f'{i}_{p.name}')
    checkpoint_sha = file_sha256(ck)
    expected = ('a7f4eb9f417295a14e5063b70020ab3f55b60a4905ccda7b244343deaf9840cd' if args.model == 'sit'
                else '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a')
    assert checkpoint_sha == expected
    generator = torch.Generator(device=device).manual_seed(202609431)
    labels = torch.arange(32, device=device, dtype=torch.long) * 999 // 31
    events = sorted({int(torch.argmin((grid - target).abs())) for target in [1., .9, .75, .55]})
    assert len(events) == 4 and all(float(grid[i]) > .5 for i in events)
    query_generator = torch.Generator(device=device).manual_seed(202609433)
    query_rng_initial = tensor_hash(query_generator.get_state())
    rows, noise, endpoints = [], [], []
    full_calls = prefix_calls = parity_calls = 0
    started = time.perf_counter()
    for start in range(0, 32, 4):
        z = torch.randn(4, *shape, device=device, generator=generator)
        noise.append(z.cpu())
        if args.model == 'sit':
            z = z.double()
        y = labels[start:start + 4]
        for step, (t, next_t) in enumerate(zip(grid[:-1], grid[1:])):
            ts = (torch.ones(4, device=device, dtype=grid.dtype) * t).float()
            f, bvel, drift, current = pair(z, ts, y)
            full_calls += 1
            if step in events:
                assert torch.equal(weak(z, ts, y), current)
                prefix_calls += 1
                parity_calls += 1
                now = float(t)
                future = max(.5, now - 1 / 32)
                tf = torch.full((4,), future, device=device)
                aa = (1-now)/(1-future) * future**2/now**2
                bb = (1-future) - aa*(1-now)
                vv = future**2 * (1-(1-now)**2*future**2/((1-future)**2*now**2))
                assert vv >= 0
                center = aa*z.float() + bb*current
                raw_future = weak(z, tf, y)
                prefix_calls += 1
                raw = scale*(bvel-(z.float()-raw_future)/future)
                revisions = []
                for repetition in range(16):
                    eta = torch.randn(4, *shape, device=device, generator=query_generator)
                    plus = weak(center + np.sqrt(vv)*eta, tf, y)
                    minus = weak(center - np.sqrt(vv)*eta, tf, y)
                    prefix_calls += 2
                    revisions.append(-scale/now*(current-.5*(plus+minus)))
                values = torch.stack(revisions).double().flatten(2)
                gram = torch.einsum('kbd,lbd->bkl', values, values)/values.shape[-1]
                mean = values.mean(0)
                raw_flat = raw.double().flatten(1)
                raw_energy = raw_flat.square().mean(1)
                cross = (mean*raw_flat).mean(1)
                for j in range(4):
                    g = gram[j].cpu().numpy()
                    mean_energy = float(g.mean())
                    variance = float((np.trace(g)-16*mean_energy)/15)
                    signal = mean_energy-variance/16
                    split_cross = float(g[:8,8:].mean())
                    split_cos = split_cross/np.sqrt(g[:8,:8].mean()*g[8:,8:].mean())
                    assert variance >= -1e-12
                    rows.append(dict(sample=start+j, label=int(y[j]), step=step, noise_time=now,
                                     future_time=future, A=aa, B=bb, conditional_noise_variance=vv,
                                     mean_energy=mean_energy, pair_variance=variance,
                                     unbiased_signal_energy=signal,
                                     single_pair_signal_to_noise=signal/max(variance,1e-30),
                                     split_half_cosine=float(split_cos), raw_energy=float(raw_energy[j]),
                                     mean_raw_cosine=float(cross[j]/torch.sqrt(raw_energy[j]*mean_energy)),
                                     gram=g.tolist()))
            z = z + (next_t - t) * (drift.double() if args.model == 'sit' else drift)
        assert torch.isfinite(z).all()
        endpoints.append(z.cpu())
        print(json.dumps(dict(done=start+4, seconds=time.perf_counter()-started)), flush=True)
    torch.cuda.synchronize()
    assert full_calls == 800 and prefix_calls == 1088 and parity_calls == 32
    noise_sha = tensor_hash(torch.cat(noise))
    endpoint_sha = tensor_hash(torch.cat(endpoints))
    old = json.loads((ROOT/'experiments/results/terminal_defect_20260908'/f'pfr_ig_direction_{args.model}.json').read_text())
    assert old['noise_sha256'] == noise_sha and old['endpoint_sha256'] == endpoint_sha
    summary = []
    for step in events:
        selected = [r for r in rows if r['step']==step]
        signal = sum(r['unbiased_signal_energy'] for r in selected)
        variance = sum(r['pair_variance'] for r in selected)
        summary.append(dict(step=step, noise_time=selected[0]['noise_time'], pooled_single_pair_snr=signal/variance,
                            median_split_half_cosine=float(np.median([r['split_half_cosine'] for r in selected])),
                            mean_raw_cosine=float(np.mean([r['mean_raw_cosine'] for r in selected]))))
    result = dict(complete=True, model=args.model, samples=32, seed=202609431, batch_size=4,
                  repetitions=16, events=events, rows=rows, summary=summary,
                  full_calls=full_calls, prefix_calls=prefix_calls, parity_calls=parity_calls,
                  seconds=time.perf_counter()-started, sources=sources, checkpoint_sha256=checkpoint_sha,
                  metadata=meta, grid=grid.cpu().tolist(), labels=labels.cpu().tolist(),
                  noise_sha256=noise_sha, endpoint_sha256=endpoint_sha,
                  query_seed=202609433, query_rng_initial=query_rng_initial,
                  query_rng_final=tensor_hash(query_generator.get_state()),
                  scope='Unchanged IG endpoints matched to preceding diagnostic. Repeated weak queries only; no quality or efficacy claim.')
    out = ROOT/'experiments/results/terminal_defect_20260908'/f'pfr_posterior_query_noise_{args.model}.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:result[k] for k in ['complete','summary','seconds','full_calls','prefix_calls']}), flush=True)


if __name__ == '__main__':
    main()

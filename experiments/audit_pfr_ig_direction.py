"""Decompose fixed PFR revisions against IG on unmodified trajectories.

No decoder, quality score, controller fitting, or trajectory intervention.
"""
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


def summarize(rows, power):
    w = np.array([r['dt'] ** power for r in rows])
    q2 = np.array([r['ig_gap_energy'] for r in rows])
    d2 = np.array([r['revision_energy'] for r in rows])
    cross = np.array([r['cross'] for r in rows])
    denominator = np.sum(w * d2)
    assert denominator > 0 and np.all(q2 > 0)
    by_time = 0.0
    for step in sorted(set(r['step'] for r in rows)):
        mask = np.array([r['step'] == step for r in rows])
        by_time += np.sum(w[mask] * cross[mask]) ** 2 / np.sum(w[mask] * q2[mask])
    return dict(dt_power=power,
                best_constant_scale_increment=float(np.sum(w * cross) / np.sum(w * q2)),
                constant_explained_fraction=float(np.sum(w * cross) ** 2 / np.sum(w * q2) / denominator),
                time_only_scale_explained_fraction=float(by_time / denominator),
                sample_and_time_scale_explained_fraction=float(np.sum(w * cross ** 2 / q2) / denominator),
                positive_scale_increment_fraction=float(np.mean(cross > 0)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sit', 'raev2'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    root = Path('/home/zhoushunyu/data/eqvae/experiments/pfr_ig_direction_20260908') / args.model
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
            return f, b, b + scale * (f - b)

        def weak(z, t, y):
            return prefix(model, z.float(), t, y)
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
            return velocity(f, z, t), velocity(b, z, t), velocity(b + s * (f - b), z, t)

        def weak(z, t, y):
            return velocity(n.evaluate_base_head_only(model, z, t, context=y, attn_mask=None), z, t)
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
    rows, noise, endpoints = [], [], []
    full_calls = prefix_calls = parity_calls = 0
    started = time.perf_counter()
    for start in range(0, 32, 4):
        z = torch.randn(4, *shape, device=device, generator=generator)
        noise.append(z.cpu())
        if args.model == 'sit':
            z = z.double()
        y = labels[start:start + 4]
        for step, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
            ts = (torch.ones(4, device=device, dtype=grid.dtype) * t).float()
            f, b, drift = pair(z, ts, y)
            full_calls += 1
            if step in [0, 25, 50, 75]:
                assert torch.equal(weak(z, ts, y), b)
                prefix_calls += 1
                parity_calls += 1
            if float(t) > .5:
                tf = torch.full((4,), max(.5, float(t) - 1 / 32), device=device)
                future = weak(z, tf, y)
                prefix_calls += 1
                # Match sampler FP32 subtraction before FP64 diagnostic reductions.
                q = (f - b).double().flatten(1)
                d = (scale * (b - future)).double().flatten(1)
                q2, d2, cross = q.square().mean(1), d.square().mean(1), (q * d).mean(1)
                coefficient = cross / q2
                residual = d - coefficient[:, None] * q
                assert torch.allclose(residual.square().mean(1), d2 - cross.square() / q2, atol=1e-12, rtol=1e-10)
                assert torch.all((residual * q).mean(1).abs() < 1e-10)
                for j in range(4):
                    rows.append(dict(sample=start + j, label=int(y[j]), step=step, noise_time=float(t),
                                     dt=float(t - s), ig_gap_energy=float(q2[j]), revision_energy=float(d2[j]),
                                     cross=float(cross[j]), scale_increment=float(coefficient[j]),
                                     cosine=float(cross[j] / torch.sqrt(q2[j] * d2[j]))))
            z = z + (s - t) * (drift.double() if args.model == 'sit' else drift)
        assert torch.isfinite(z).all()
        endpoints.append(z.cpu())
        print(json.dumps(dict(done=start + 4, seconds=time.perf_counter() - started)), flush=True)
    torch.cuda.synchronize()
    assert full_calls == 800 and parity_calls == 32
    assert prefix_calls == 8 * int((grid[:-1] > .5).sum()) + 32
    result = dict(complete=True, model=args.model, samples=32, seed=202609431, batch_size=4,
                  rows=rows, summary=[summarize(rows, p) for p in [1, 2]],
                  full_calls=full_calls, prefix_calls=prefix_calls, parity_calls=parity_calls,
                  seconds=time.perf_counter() - started, sources=sources, checkpoint_sha256=checkpoint_sha,
                  metadata=meta, grid=grid.cpu().tolist(), labels=labels.cpu().tolist(),
                  noise_sha256=tensor_hash(torch.cat(noise)), endpoint_sha256=tensor_hash(torch.cat(endpoints)),
                  scope='Unmodified IG trajectories, FP32 model/no TF32; SiT FP64 state, RAE FP32 state. No quality or cross-model causal claim.')
    out = ROOT / 'experiments/results/terminal_defect_20260908' / f'pfr_ig_direction_{args.model}.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ['complete', 'summary', 'seconds', 'full_calls', 'prefix_calls']}), flush=True)


if __name__ == '__main__':
    main()

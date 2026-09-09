"""Finite time-response attribution on unchanged native trajectories.

Shapley averages all orders of switching feature, readout, and (RAE only)
velocity-conversion times. Crossed clocks are diagnostics, not trained models.
"""
import argparse
import gc
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_training_core import file_sha256


def thash(x):
    return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()


def attribute(corners, names):
    n = len(names)
    parts = []
    for j in range(n):
        part = torch.zeros_like(corners[0], dtype=torch.float64)
        for mask in range(1 << n):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            part += weight * (corners[mask].double() - corners[mask | (1 << j)].double())
        parts.append(part)
    raw = corners[0].double() - corners[(1 << n)-1].double()
    assert torch.allclose(sum(parts), raw, rtol=1e-10, atol=1e-12)
    return parts, raw


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['sit', 'raev2'], required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    out = Path('/home/zhoushunyu/data/eqvae/experiments/pfr_time_routes_20260908') / a.model
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'experiments/raev2_training_core.py']
    if a.model == 'sit':
        from experiments.run_internal_guidance_sit_audit import load_model
        repo = ROOT / 'research_repos/internal_guidance_study/Internal-Guidance/SiT'
        ck = Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
        model, _ = load_model(repo=repo, checkpoint_path=ck, model_name='SiT-XL/2', encoder_depth=8, state_key='ema', device=device)
        sources += [ROOT / 'experiments/run_internal_guidance_sit_audit.py', repo / 'models/sit.py', repo / 'samplers.py']
        names = ['features', 'readout']
        shape, beta, events = (4, 32, 32), 1.35, [0, 10, 25, 45]
        grid = torch.linspace(1, 0, 101, dtype=torch.float64, device=device)

        def pair(z, t, y):
            f, b, _ = model(z.float(), t, y)
            return b, b + beta * (f-b)

        def corners(z, t, r, y):
            hs, cs = [], []
            for time_ in [t, r]:
                h = model.x_embedder(z.float()) + model.pos_embed
                label, _ = model.y_embedder(y, model.training)
                c = model.t_embedder(time_) + label
                for block in model.blocks[:model.encoder_depth]:
                    h = block(h, c)
                hs.append(h)
                cs.append(c)
            return {i | (j << 1): model.unpatchify(model.final_layer_xr(hs[i], cs[j]))
                    for i, j in itertools.product(range(2), repeat=2)}
    else:
        from experiments import sample_raev2_pfr_retiming as native
        cfg = native.load_config(native.DEFAULT_CONFIG)
        ck = native.DEFAULT_CHECKPOINT
        model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
        state = torch.load(ck, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(state['ema'], strict=True)
        del state
        sources += [Path(native.__file__), ROOT / 'experiments/raev2_pfr_retiming.py', native.DEFAULT_CONFIG, Path(inspect.getfile(type(model)))]
        names = ['features', 'readout', 'conversion']
        shape, beta, events = tuple(cfg.misc.latent_size), 1.78, [0, 47, 73, 87]
        grid = native.shifted_time_grid(100, 8., device)

        def vel(clean, z, t):
            return native.clean_to_velocity(clean, z, t, denominator_floor=float(cfg.transport.t_eps))

        def pair(z, t, y):
            f, b = model(z, t, context=y, attn_mask=None)
            scale = beta if float(cfg.guidance.ig.t_min) <= float(t[0]) <= float(cfg.guidance.ig.t_max) else 1.
            return vel(b, z, t), vel(b + scale*(f-b), z, t)

        def corners(z, t, r, y):
            hs, cs = [], []
            for time_ in [t, r]:
                cond = dict(context=y, attn_mask=None)
                seq, c = model._build_sequence(z, time_, cond)
                mask = model._build_attn_mask(seq, cond)
                for block in model.blocks[:model.base_model_depth]:
                    seq = block(seq, model.enc_rope, mask)
                hs.append(seq[:, :model.s_embedder.num_patches, :])
                cs.append(c)
            result = {}
            for i, j in itertools.product(range(2), repeat=2):
                h = F.silu(cs[j] + hs[i])
                clean = model.unpatchify(model.base_final_layer(h, h), model.s_patch_size)
                for k, time_ in enumerate([t, r]):
                    result[i | (j << 1) | (k << 2)] = vel(clean, z, time_)
            return result
    gc.collect()
    source_hashes = {str(s): file_sha256(s) for s in dict.fromkeys(sources)}
    for i, s in enumerate(source_hashes):
        shutil.copy2(s, out / f'{i}_{Path(s).name}')
    ckhash = file_sha256(ck)
    old = json.loads((ROOT / f'experiments/results/terminal_defect_20260908/pfr_head_kinematic_response_{a.model}.json').read_text())
    assert ckhash == old['checkpoint_sha256']
    gen = torch.Generator(device='cuda').manual_seed(202609431)
    labels = torch.arange(32, device=device, dtype=torch.long)*999//31
    noise, endpoints, rows = [], [], []
    full_calls = prefix_calls = readout_calls = 0
    start_time = time.perf_counter()
    for start in range(0, 32, 4):
        z = torch.randn(4, *shape, device=device, generator=gen)
        noise.append(z.cpu())
        if a.model == 'sit':
            z = z.double()
        y = labels[start:start+4]
        for step, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
            ts = (torch.ones(4, device=device, dtype=grid.dtype)*t).float()
            b, drift = pair(z, ts, y)
            full_calls += 1
            if step in events:
                r = torch.full((4,), max(.5, float(t)-1/32), device=device)
                bf, _ = pair(z, r, y)
                full_calls += 1
                c = corners(z, ts, r, y)
                prefix_calls += 2
                readout_calls += 4
                assert torch.equal(c[0], b) and torch.equal(c[max(c)], bf)
                parts, raw = attribute(c, names)
                parts = [beta*v.flatten(1) for v in parts]
                raw = beta*raw.flatten(1)
                vectors = torch.stack([raw] + parts, dim=1)
                gram = vectors @ vectors.transpose(1, 2) / vectors.shape[-1]
                for j in range(4):
                    row = dict(sample=start+j, step=step, time=float(t), future_time=float(r[j]),
                               gram=gram[j].cpu().tolist(), names=['raw']+names)
                    previous = next(v for v in old['rows'] if v['sample']==start+j and v['step']==step)
                    assert math.isclose(row['gram'][0][0], previous['weak_energy'], rel_tol=1e-10, abs_tol=1e-12)
                    rows.append(row)
            z = z + (s-t)*(drift.double() if a.model=='sit' else drift)
        endpoints.append(z.cpu())
        print(json.dumps(dict(done=start+4, seconds=time.perf_counter()-start_time)), flush=True)
    torch.cuda.synchronize()
    assert full_calls == 832 and prefix_calls == 64 and readout_calls == 128
    assert thash(torch.cat(noise)) == old['noise_sha256']
    assert thash(torch.cat(endpoints)) == old['endpoint_sha256']
    summary = []
    for step in events:
        g = np.sum([v['gram'] for v in rows if v['step']==step], axis=0)
        summary.append(dict(step=step, energy_ratio={n: float(g[j,j]/g[0,0]) for j,n in enumerate(names, 1)},
                            signed_raw_dot_ratio={n: float(g[0,j]/g[0,0]) for j,n in enumerate(names, 1)}))
    result = dict(complete=True, model=a.model, rows=rows, summary=summary, sources=source_hashes,
                  checkpoint_sha256=ckhash, noise_sha256=old['noise_sha256'], endpoint_sha256=old['endpoint_sha256'],
                  full_calls=full_calls, prefix_calls=prefix_calls, explicit_readout_calls=readout_calls,
                  native_corner_parity=True, unchanged_endpoint_parity=True,
                  seconds=time.perf_counter()-start_time,
                  scope='Finite all-order clock attribution; signed shares sum to one, energies need not. Crossed times are off-training-path. No quality or error-identification claim.')
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    target = ROOT / f'experiments/results/terminal_defect_20260908/pfr_time_routes_{a.model}.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ['summary', 'seconds', 'full_calls', 'prefix_calls']}), flush=True)


if __name__ == '__main__':
    main()

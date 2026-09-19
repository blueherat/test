"""Small source-anchored differential-flow demonstration, not a quality benchmark.

SiT convention: t=0 noise, t=1 data. Fresh noise each outer round, fixed
within that round; reference and target share noise unless explicitly ablated.
This is a FlowEdit-inspired construction, not its official full algorithm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.guidance_pasted_20260912 import common

SOURCE = Path('/home/zhoushunyu/data/eqvae/experiments/recursive_guidance_focus_20260913/focused_screen_1k/cfg_gain_l2/round00/rank0/batch0000.npz')
DEFAULT_OUT = Path('/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913')
ARMS = ('shared_identity', 'independent_noise_control', 'cfg_delta_025',
        'ig_delta_025', 'codec_only')


def metrics(z, pixels, previous, previous_pixels, original, original_pixels):
    z = z.detach().double().cpu().numpy()
    def latent(ref):
        ref = ref.detach().double().cpu().numpy()
        return np.mean((z-ref)**2, axis=(1, 2, 3)).tolist()
    def pixel(ref):
        mse = np.mean(((pixels.astype(np.float64)-ref)/255.)**2, axis=(1, 2, 3))
        return mse.tolist(), [None if v == 0 else float(-10*np.log10(v)) for v in mse]
    local_mse, local_psnr = pixel(previous_pixels)
    anchor_mse, anchor_psnr = pixel(original_pixels)
    return dict(latent_mse_previous=latent(previous), latent_mse_original=latent(original),
                pixel_mse_previous=local_mse, pixel_mse_original=anchor_mse,
                psnr_previous=local_psnr, psnr_original=anchor_psnr,
                psnr_null_means_infinite=True,
                pixel_exact_previous=np.all(pixels == previous_pixels, axis=(1, 2, 3)).tolist(),
                pixel_exact_original=np.all(pixels == original_pixels, axis=(1, 2, 3)).tolist())


def field(rt, state, t, arm, left):
    if arm == 'ig_delta_025' and left < .5:
        strong, weak = rt.pair(state, t)
        return strong + .25*(strong-weak)
    strong = rt.field(state, t, 'full')
    if arm == 'cfg_delta_025' and left < .75:
        labels = rt.labels
        try:
            rt.labels = torch.full_like(labels, 100)
            null = rt.field(state, t, 'full')
        finally:
            rt.labels = labels
        return strong + .25*(strong-null)
    return strong


def differential_flow(rt, source, labels, noise, target_noise, arm, grid):
    rt.labels = labels
    delta = torch.zeros_like(source)
    before = dict(rt.counts)
    max_difference = 0.
    def evaluate(d, t, left):
        reference_state = t*source + (1-t)*noise
        target_state = t*source + (1-t)*target_noise + d
        # Deliberately separate calls, including the same-field identity arm.
        reference_velocity = rt.field(reference_state, t, 'full')
        target_velocity = field(rt, target_state, t, arm, left)
        return target_velocity-reference_velocity
    for t, u in zip(grid[:-1], grid[1:]):
        h = u-t
        # Guidance schedule is held over each accepted interval, including its
        # right-end Heun evaluation; .5 and .75 are exact grid boundaries.
        first = evaluate(delta, t, float(t))
        second = evaluate(delta+h*first, u, float(t))
        max_difference = max(max_difference, float(first.abs().max()), float(second.abs().max()))
        delta = delta + .5*h*(first+second)
        if not torch.isfinite(delta).all() or float(delta.abs().max()) > 1e5:
            raise FloatingPointError(f'Unstable arm {arm}, time {float(u)}')
    return source+delta, dict(full_calls=rt.counts['full']-before['full'],
                             prefix_calls=rt.counts['prefix']-before['prefix'],
                             maximum_velocity_difference=max_difference)


def codec(rt, pixels):
    image = torch.from_numpy(pixels.copy()).cuda().permute(0, 3, 1, 2).float()/127.5-1.
    # Deterministic posterior mean: isolate repeated lossy coding from sampling.
    return rt.vae.encode(image).latent_dist.mode()*rt.small.SD_VAE_SCALING_FACTOR


def gallery(out, arm, labels, rounds):
    size, top, row_header = 144, 32, 100
    canvas = Image.new('RGB', (row_header+(rounds+1)*size, top+len(labels)*(size+20)), 'white')
    draw = ImageDraw.Draw(canvas)
    for r in range(rounds+1):
        draw.text((row_header+r*size+5, 8), f'R{r}', fill='black')
        with np.load(out/arm/f'round{r:02d}.npz') as data:
            for i, pixels in enumerate(data['pixels']):
                y = top+i*(size+20)
                canvas.paste(Image.fromarray(pixels).resize((size, size)), (row_header+r*size, y))
                draw.text((5, y+8), f'Image {i}\nclass {labels[i]}', fill='black')
    canvas.save(out/arm/'rounds.png')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026091307)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Use a new output path; preserving {args.out}')
    args.out.mkdir(parents=True)
    started = time.time()
    common.atomic(args.out/'status.json', dict(status='loading', started=started))
    rt = common.runtime('sit_small')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    with np.load(SOURCE) as data:
        original = torch.from_numpy(data['latents'][:args.samples].copy()).cuda()
        labels = torch.from_numpy(data['labels'][:args.samples].copy()).cuda()
        old_pixels = data['arr_0'][:args.samples].copy()
    original_pixels = rt.decode(original)
    grid = torch.linspace(0, 1, 65, device='cuda')[16:]
    request = dict(operation='source-anchored differential flow', source=str(SOURCE),
                   source_sha256=common.sha(SOURCE), samples=args.samples, rounds=args.rounds,
                   seed=args.seed, arms=list(ARMS), t_start=.25, t_end=1., heun_steps=48,
                   labels=labels.cpu().tolist(), latent_scale=float(rt.small.SD_VAE_SCALING_FACTOR),
                   fresh_noise_each_round=True, noise_fixed_within_round=True,
                   noise_shared_between_arms=True, no_target_only_suffix=True,
                   handoff='latent except codec_only; pixels saved after every round',
                   source_pixels_equal_current_decode=bool(np.array_equal(old_pixels, original_pixels)),
                   guidance_schedule='held per accepted interval; cfg until .75, IG until .5',
                   quality_benchmark=False, source_paths=rt.sources,
                   script_sha256=common.sha(Path(__file__)))
    common.atomic(args.out/'request.json', request)
    rng = torch.Generator(device='cuda').manual_seed(args.seed)
    noises = [torch.randn(original.shape, generator=rng, device='cuda') for _ in range(args.rounds)]
    other_noises = [torch.randn(original.shape, generator=rng, device='cuda') for _ in range(args.rounds)]
    np.savez(args.out/'noise_bank.npz', shared=torch.stack(noises).cpu().numpy(),
             independent=torch.stack(other_noises).cpu().numpy())
    rows = []
    for arm in ARMS:
        folder = args.out/arm
        folder.mkdir()
        current, pixels = original.clone(), original_pixels.copy()
        np.savez(folder/'round00.npz', latents=current.cpu().numpy(), pixels=pixels, labels=labels.cpu().numpy())
        for r in range(1, args.rounds+1):
            begin = time.time()
            previous, previous_pixels = current, pixels
            if arm == 'codec_only':
                current = codec(rt, previous_pixels)
                counts = dict(full_calls=0, prefix_calls=0, maximum_velocity_difference=None)
            else:
                target_noise = other_noises[r-1] if arm == 'independent_noise_control' else noises[r-1]
                current, counts = differential_flow(rt, previous, labels, noises[r-1], target_noise, arm, grid)
            pixels = rt.decode(current)
            torch.cuda.synchronize()
            assert np.isfinite(current.cpu().numpy()).all()
            row = dict(arm=arm, round=r, seconds=time.time()-begin, **counts,
                       **metrics(current, pixels, previous, previous_pixels, original, original_pixels))
            if arm == 'shared_identity':
                assert torch.equal(current, previous), 'Same-field differential identity failed'
                assert np.array_equal(pixels, previous_pixels), 'Deterministic decode failed'
                assert counts['maximum_velocity_difference'] == 0
            np.savez(folder/f'round{r:02d}.npz', latents=current.cpu().numpy(), pixels=pixels, labels=labels.cpu().numpy())
            for i, p in enumerate(pixels):
                Image.fromarray(p).save(folder/f'image{i:02d}_round{r:02d}.png')
            rows.append(row)
            common.atomic(args.out/'results.json', rows)
            common.atomic(args.out/'status.json', dict(status='running', arm=arm, round=r, completed=len(rows), total=len(ARMS)*args.rounds))
            print(json.dumps(row), flush=True)
        gallery(args.out, arm, labels.cpu().tolist(), args.rounds)
    # Non-zero arms must actually exercise the update; no assertion on quality.
    for arm in ('cfg_delta_025', 'ig_delta_025', 'independent_noise_control'):
        assert any(max(row['latent_mse_previous']) > 0 for row in rows if row['arm'] == arm)
    common.atomic(args.out/'summary.json', dict(complete=True, seconds=time.time()-started,
        identity_all_rounds_latent_and_pixel_exact=True, no_short_circuit=True,
        nonzero_controls_changed=True, outputs_excluding_r0=args.samples*args.rounds*len(ARMS),
        handoff=request['handoff'], quality_or_cfg_ig_benefit_established=False))
    common.atomic(args.out/'status.json', dict(status='complete', completed=len(rows), total=len(rows)))


if __name__ == '__main__':
    main()

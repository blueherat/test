"""Fixed-reference image reconstruction, with genuine five-round latent refeeding.

This is a copy-task pilot, not a generation quality benchmark. Neither reference
is an oracle. No random variable is drawn after preparing the initial images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BASE = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow')
CACHE = BASE / 'imagenet100_cmc_sdvae'
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913')
CHECKPOINTS = {
    'strong': BASE / 'runs/sit-s-2_seed0/checkpoints/step_00800000.pt',
    'weak': BASE / 'runs/sit-s-2_seed0/checkpoints/step_00400000.pt',
    'ref700': BASE / 'runs/sit-s-2_seed0/checkpoints/step_00700000.pt',
    'ref_alt300': BASE / 'runs/sit-s-2_velocity-velocity-loss_t-logit-normal-m-0p8-s-0p8_seed0/checkpoints/step_00300000.pt',
}
ARMS = ('strong', 'weak', 'cfg', 'ag')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    labels = np.load(CACHE / 'validation_labels.npy', mmap_mode='r')
    source = np.load(CACHE / 'validation_source_indices.npy', mmap_mode='r')
    moments = np.load(CACHE / 'validation_moments.npy', mmap_mode='r')
    rng = np.random.default_rng(2026091391)
    classes = rng.permutation(100)[:16]
    indices = np.array([rng.choice(np.flatnonzero(labels == c)) for c in classes])
    chosen = np.array(moments[indices])
    posterior_noise = rng.standard_normal((16, 4, 32, 32), dtype=np.float32)
    clean = (chosen[:, :4] + chosen[:, 4:] * posterior_noise) * np.float32(.18215)
    np.savez(args.out / 'inputs.npz', clean=clean, labels=np.array(labels[indices]),
             indices=indices, source_ids=np.array(source[indices]), moments=chosen,
             posterior_noise=posterior_noise)
    request = dict(operation='T_G(x) = G(E_reference(x))', samples=16, rounds=5,
        solver='Heun', steps_per_direction=128, check_steps=256, check_images=4,
        arms=list(ARMS), references=['ref700', 'ref_alt300'],
        coefficients=dict(cfg_extra=1.25, ag_extra=.5, cutoff=.75),
        reference_guidance=0, reverse_time='1 to 0, negative dt, fresh field queries',
        reference_caveats={'ref700': 'same training run as tested checkpoints; NOT independent',
            'ref_alt300': 'separate training run, same seed and architecture, different time weighting; NOT oracle'},
        handoff='latent: each round inverts the previous round output; decode only for display',
        noise='VAE posterior sampled once in frozen R0; no noise during any round',
        source_split='validation', source_cache=str(CACHE), seed=2026091391,
        source_manifest_sha256=sha(CACHE/'manifest.json'),
        input_sha256=sha(args.out/'inputs.npz'), script_sha256=sha(__file__),
        checkpoints={k: str(v) for k, v in CHECKPOINTS.items()},
        scope='copy and reference sensitivity pilot; no quality ranking or FID claim')
    atomic(args.out/'request.json', request)
    print(json.dumps(request), flush=True)


class Fields:
    def __init__(self, reference, device):
        import torch
        from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
        from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO
        module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
        self.models, self.metadata, self.calls = {}, {}, 0
        for key in ('strong', 'weak', reference):
            model, sem, meta = load_sit_field_model(checkpoint_path=CHECKPOINTS[key],
                weights='ema', sit_module=module, source_metadata=source, device=torch.device(device))
            assert sem.prediction_target == 'velocity'
            self.models[key] = model, sem
            self.metadata[key] = meta

    def call(self, key, z, t, labels):
        from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field
        self.calls += 1
        model, sem = self.models[key]
        return evaluate_sit_field(model, sem, z, z.new_full((len(z),), t), labels)

    def velocity(self, arm, z, t, labels, active):
        import torch
        if arm in self.models:
            return self.call(arm, z, t, labels)
        strong = self.call('strong', z, t, labels)
        if not active:
            return strong
        if arm == 'cfg':
            return strong + 1.25 * (strong - self.call('strong', z, t, torch.full_like(labels, 100)))
        if arm == 'ag':
            return strong + .5 * (strong - self.call('weak', z, t, labels))
        raise ValueError(arm)


def integrate(fields, z, labels, arm, steps, reverse=False):
    import torch
    begin = fields.calls
    direction = -1 if reverse else 1
    h = direction / steps
    for k in range(steps):
        t = 1-k/steps if reverse else k/steps
        # Physical lower interval endpoint; same coefficient for both stages.
        active = min(t, t+h) < .75
        one = fields.velocity(arm, z, t, labels, active)
        two = fields.velocity(arm, z+h*one, t+h, labels, active)
        z = z + .5*h*(one+two)
        if not torch.isfinite(z).all() or z.abs().max() > 1e5:
            raise FloatingPointError(f'{arm}, reverse={reverse}, step={k}')
    count = fields.calls-begin
    expected = 2*steps + (2*int(.75*steps) if arm in ('cfg', 'ag') else 0)
    assert count == expected, (count, expected)
    return z


def mse(a, b):
    return (a.double()-b.double()).square().flatten(1).mean(1).cpu().numpy()


def gallery(folder, labels):
    from PIL import Image, ImageDraw
    size, top, left = 128, 24, 92
    canvas = Image.new('RGB', (left+6*size, top+len(labels)*size), 'white')
    draw = ImageDraw.Draw(canvas)
    for r in range(6):
        draw.text((left+r*size+4, 5), f'R{r}', fill='black')
        with np.load(folder/f'round{r:02d}.npz') as d:
            for i, pixels in enumerate(d['pixels']):
                canvas.paste(Image.fromarray(pixels).resize((size,size)), (left+r*size, top+i*size))
                draw.text((3, top+i*size+6), f'{i}: class {labels[i]}', fill='black')
    canvas.save(folder/'rounds.png')


def run(args):
    import torch
    from diffusers.models import AutoencoderKL
    from PIL import Image
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    request = json.loads((args.out/'request.json').read_text())
    assert request['script_sha256'] == sha(__file__)
    assert request['input_sha256'] == sha(args.out/'inputs.npz')
    root = args.out/args.reference
    root.mkdir(exist_ok=False)
    start = time.monotonic()
    atomic(root/'status.json', dict(status='loading'))
    fields = Fields(args.reference, args.device)
    atomic(root/'models.json', fields.metadata)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', local_files_only=True)
    vae.to(args.device).eval().requires_grad_(False)
    def decode(z):
        pieces = []
        for begin in range(0, len(z), 2):
            p = vae.decode(z[begin:begin+2]/.18215).sample
            pieces.append((127.5*p+128).clamp(0,255).permute(0,2,3,1).to('cpu', torch.uint8).numpy())
        return np.concatenate(pieces)
    data = np.load(args.out/'inputs.npz')
    original = torch.from_numpy(data['clean']).to(args.device)
    labels = torch.from_numpy(data['labels']).to(device=args.device, dtype=torch.long)
    rows = []
    with torch.inference_mode():
        original_pixels = decode(original)
        # Solver refinement and self-inverse checks, not a model-quality gate.
        check_x, check_labels = original[:4], labels[:4]
        checks, saved = {}, {}
        for steps in (128, 256):
            encoded = integrate(fields, check_x, check_labels, args.reference, steps, True)
            saved[f'encoded_{steps}'] = encoded.cpu().numpy()
            for arm in (*ARMS, args.reference):
                reconstructed = integrate(fields, encoded, check_labels, arm, steps)
                saved[f'{arm}_{steps}'] = reconstructed.cpu().numpy()
                checks[f'{arm}_{steps}_copy_mse'] = mse(reconstructed, check_x).tolist()
        for arm in (*ARMS, args.reference):
            error = np.mean((saved[f'{arm}_128'].astype(np.float64)-saved[f'{arm}_256'])**2, axis=(1,2,3))
            checks[f'{arm}_solver_change_mse'] = error.tolist()
        atomic(root/'solver_check.json', checks)
        np.savez(root/'solver_check.npz', original=check_x.cpu().numpy(), **saved)
        print(json.dumps(dict(reference=args.reference, solver_check=checks)), flush=True)
        for arm in ARMS:
            folder = root/arm
            folder.mkdir()
            current, pixels = original.clone(), original_pixels.copy()
            np.savez(folder/'round00.npz', latents=current.cpu().numpy(), pixels=pixels, labels=data['labels'])
            for r in range(1, 6):
                before, begin = fields.calls, time.monotonic()
                previous, previous_pixels = current, pixels
                encoded = integrate(fields, previous, labels, args.reference, 128, True)
                current = integrate(fields, encoded, labels, arm, 128)
                pixels = decode(current)
                pixel_mse = ((pixels.astype(np.float64)-original_pixels)/255)**2
                pixel_mse = pixel_mse.mean(axis=(1,2,3))
                row = dict(reference=args.reference, arm=arm, round=r,
                    latent_mse_original=mse(current, original).tolist(),
                    latent_mse_previous=mse(current, previous).tolist(),
                    pixel_mse_original=pixel_mse.tolist(),
                    pixel_mse_previous=(((pixels.astype(np.float64)-previous_pixels)/255)**2).mean(axis=(1,2,3)).tolist(),
                    psnr_original=[None if v == 0 else float(-10*np.log10(v)) for v in pixel_mse],
                    encoded_mean=encoded.double().flatten(1).mean(1).cpu().tolist(),
                    encoded_rms=encoded.double().square().flatten(1).mean(1).sqrt().cpu().tolist(),
                    prior_note='mean/rms are descriptions, NOT prior-distribution certification',
                    full_calls_per_image=fields.calls-before, seconds=time.monotonic()-begin)
                np.savez(folder/f'round{r:02d}.npz', latents=current.cpu().numpy(),
                         encoded=encoded.cpu().numpy(), pixels=pixels, labels=data['labels'])
                for i, p in enumerate(pixels):
                    Image.fromarray(p).save(folder/f'image{i:02d}_round{r:02d}.png')
                rows.append(row)
                atomic(root/'results.json', rows)
                atomic(root/'status.json', dict(status='running', arm=arm, round=r, completed=len(rows), total=20))
                print(json.dumps(dict(reference=args.reference, arm=arm, round=r,
                    mse=float(np.mean(row['latent_mse_original'])), psnr=float(np.mean(row['psnr_original'])),
                    seconds=row['seconds'])), flush=True)
            gallery(folder, data['labels'].tolist())
    atomic(root/'summary.json', dict(complete=True, rounds_per_arm=5, images_per_arm=16,
        outputs=320, all_calls=fields.calls, elapsed=time.monotonic()-start,
        quality_benefit_established=False, handoff=request['handoff']))
    atomic(root/'status.json', dict(status='complete', completed=20, total=20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'run'])
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--reference', choices=['ref700', 'ref_alt300'])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.phase == 'prepare':
        prepare(args)
    else:
        assert args.reference
        run(args)


if __name__ == '__main__':
    main()

"""Compare one frozen ratio critic on actual/native and re-noised marginals.

64 fixed, spread-out classes, one shared Gaussian noise per triplet. The real
and old generated endpoints are independent of this new Gaussian draw. Native
states use exactly the original B8 BF16 IG and FP32 Euler updates. No images,
FID, fitting, or guidance updates are produced by this diagnostic.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sample_raev2_ancestral_guidance import (
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, instantiate_from_config,
    shifted_time_grid, native_clean, seed_for, digest)
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic
from experiments.raev2_image_critic_guidance import exact_fp32
from experiments.train_raev2_paired_ratio import Banks
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    out = DATA / 'paired_ratio_actual_marginal_audit'
    out.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(DEFAULT_CONFIG)
    model = instantiate_from_config(config.stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    ratio_path = DATA / 'paired_ratio_fit/critic.pt'
    fit = torch.load(ratio_path, map_location='cpu', weights_only=False)
    critic = PairedRatioCritic(fit['state_dict']['class_features']).cuda().eval().requires_grad_(False)
    critic.load_state_dict(fit['state_dict'])
    manifest = json.loads((DATA / 'paired_ratio_data/manifest.json').read_text())
    banks = Banks(manifest['splits']['validation'])
    grid = shifted_time_grid(100, 8., torch.device('cuda')).cpu().tolist()
    rows, inputs = [], []
    began = time.perf_counter()
    for batch in range(8):
        ids = np.arange(batch*8, batch*8+8)
        labels_np = ids * 1000 // 64
        labels = torch.from_numpy(labels_np).cuda()
        real, fake = banks.batch(labels_np, np.zeros(8, dtype=int), np.zeros(8, dtype=int), 'cuda')
        generator = torch.Generator(device='cuda').manual_seed(seed_for(202609089, batch, 'initial'))
        noise = torch.randn(8, 1024, 16, 16, device='cuda', generator=generator)
        state = noise.clone()
        inputs.append({'batch': batch, 'labels': labels_np.tolist(), 'noise_sha256': digest(noise)})
        for step, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
            times = torch.full((8,), t, device='cuda')
            signal = 1 - times
            real_z = signal[:, None, None, None]*real + times[:, None, None, None]*noise
            fake_z = signal[:, None, None, None]*fake + times[:, None, None, None]*noise
            with exact_fp32():
                values = critic(torch.cat((real_z, fake_z, state)), times.repeat(3), labels.repeat(3))
                p, q, actual = values.chunk(3)
                if step:
                    old_loss = paired_scaled_logistic(p, q, signal)
                    actual_loss = paired_scaled_logistic(p, actual, signal)
                    for j in range(8):
                        rows.append([int(ids[j]), int(labels_np[j]), step, t,
                                     p[j].item(), q[j].item(), actual[j].item(),
                                     old_loss[j].item(), actual_loss[j].item()])
                else:
                    assert torch.equal(p, q) and torch.equal(q, actual)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                full, base = model(state, times, context=labels, attn_mask=None)
                clean = native_clean(full, base, t, 'official')
            state = state - (t-s) * ((state-clean) / max(t, .05))
        assert torch.isfinite(state).all()
        print('completed classes', (batch+1)*8, flush=True)
    rows = np.array(rows)
    np.savez(out / 'records.npz', records=rows)
    # Full-time class means: no selection of a convenient time segment.
    paired = rows[:, 7:].reshape(8, 99, 8, 2).transpose(0, 2, 1, 3).reshape(64, 99, 2).mean(1)
    difference = paired[:, 1] - paired[:, 0]
    result = {'complete': True, 'classes': 64, 'positive_signal_times_per_class': 99,
              'seed': 202609089, 'inputs': inputs, 'checkpoint_sha256': sha(Path(DEFAULT_CHECKPOINT)),
              'config_sha256': sha(Path(DEFAULT_CONFIG)), 'critic_sha256': sha(ratio_path),
              'source_sha256': sha(Path(__file__).resolve()), 'records_sha256': sha(out / 'records.npz'),
              'mean_loss_real_vs_renoised_endpoints': float(paired[:, 0].mean()),
              'mean_loss_real_vs_actual_native': float(paired[:, 1].mean()),
              'actual_minus_renoised_loss': float(difference.mean()),
              'difference_class_standard_error': float(difference.std(ddof=1)/8),
              'elapsed_seconds': time.perf_counter()-began,
              'no_images_no_fid_no_fitting': True,
              'interpretation_limit': 'a fixed-critic distribution-transfer diagnostic on 64 predetermined classes, not an optimal density-ratio or distance estimate'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_actual_marginal_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

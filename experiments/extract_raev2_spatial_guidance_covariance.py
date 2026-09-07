"""Extract one fixed real/noise/time query per row; verify archived teacher parity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sample_raev2_ancestral_guidance import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config,
    native_clean, seed_for, digest,
)
from experiments.raev2_conditional_variance import NativePrefixTap, ConditionalVarianceHead
from experiments.raev2_directional_variance import fixed_mse, unit_disagreement
from experiments.raev2_spatial_guidance_covariance import PLAN, standardized_residual
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    out = args.output.resolve()/args.split/f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha(p) for p in [Path(__file__).resolve(),
        ROOT/'experiments/raev2_spatial_guidance_covariance.py',
        ROOT/'experiments/raev2_directional_variance.py',
        ROOT/'experiments/raev2_conditional_variance.py',
        ROOT/'experiments/sample_raev2_ancestral_guidance.py']}
    previous = DATA/'conditional_variance_features'/args.split
    previous_meta = np.load(previous/'metadata.npz')
    previous_features = np.load(previous/'features.npy', mmap_mode='r')
    old_moments = np.load(DATA/'directional_variance_moments'/args.split/'moments.npz')
    count = PLAN[args.split]
    ids = np.arange(count).reshape(-1, 8)[args.shard::4].reshape(-1)
    native = DATA/'actual_ratio_bank64k'/args.split/f'shard{args.shard}'
    request = json.loads((native/'request.json').read_text())
    summary = json.loads((native/'summary.json').read_text())
    assert request['state_key'] == 'ema' and request['samples'] == count
    assert sha(DEFAULT_CONFIG) == request['config_sha256']
    assert np.array_equal(old_moments['ids'], np.arange(count))
    noise = np.load(native/'noise.npy', mmap_mode='r')
    real = np.load(DATA/'real_ratio_bank64k'/args.split/'latents.npy', mmap_mode='r')
    assert noise.shape == (len(ids), 1024, 16, 16) and real.shape == (count, 1024, 16, 16)
    assert np.array_equal(np.load(native/'metadata.npz')['records'][:, 0], ids)
    times_np = previous_meta['times'][ids]
    calibration_path = DATA/'guided_reverse_variance/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    head_path = DATA/'conditional_variance_fit/head.pt'
    assert sha(head_path) == PLAN['spherical_head_sha256']
    head = ConditionalVarianceHead(head_path).cuda().eval().requires_grad_(False)
    assert head.calibration_sha256 == sha(calibration_path)
    global_path = DATA/'directional_variance_moments/calibration.json'
    global_calibration = json.loads(global_path.read_text())
    assert global_calibration['complete'] and global_calibration['validation']['entry_condition_passed']
    assert global_calibration['spherical_head_sha256'] == sha(head_path)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    for begin in [0, len(ids)-8]:
        batch = int(ids[begin]//8)
        rng = torch.Generator(device='cuda').manual_seed(seed_for(request['seed'], batch, 'initial'))
        regenerated = torch.randn(8, 1024, 16, 16, device='cuda', generator=rng)
        assert np.array_equal(regenerated.cpu().numpy(), noise[begin:begin+8])
        assert digest(regenerated) == summary['batch_records'][begin//8]['noise_sha256']
    tap = NativePrefixTap(model)
    residuals = np.empty((len(ids), PLAN['fitted_dimension']), dtype=np.float64)
    fields = {key: np.empty(len(ids), dtype=np.float64) for key in
              ['global_beta', 'token_projection_ratio', 'predicted_mse']}
    active_np = np.empty(len(ids), dtype=bool)
    began = time.perf_counter()
    for begin in range(0, len(ids), 8):
        batch_ids = ids[begin:begin+8]
        x = torch.from_numpy(np.array(real[batch_ids], copy=True)).cuda().float()
        epsilon = torch.from_numpy(np.array(noise[begin:begin+8], copy=True)).cuda()
        times = torch.from_numpy(times_np[begin:begin+8].copy()).cuda()
        labels = torch.from_numpy(batch_ids % 1000).cuda()
        state = (1-times[:, None, None, None])*x + times[:, None, None, None]*epsilon
        with torch.autocast('cuda', dtype=torch.bfloat16):
            full, base = model(state, times, context=labels, attn_mask=None)
            clean = native_clean(full, base, float(times_np[begin]), 'official')
        features = tap.take()
        assert np.array_equal(features.cpu().numpy(), previous_features[batch_ids]), 'cached native feature parity'
        error = (x-clean).double()
        mse = error.square().flatten(1).mean(1)
        assert np.array_equal(mse.cpu().numpy(), previous_meta['residual_mse'][batch_ids]), 'cached total MSE parity'
        index = int(previous_meta['query_indices'][batch_ids[0]])
        assert np.all(previous_meta['query_indices'][batch_ids] == index)
        predicted = fixed_mse(head, features, calibration['rows'][index]['mse'])
        assert np.array_equal(predicted.cpu().numpy(), old_moments['predicted_mse'][batch_ids])
        u_global, _, _ = unit_disagreement(full, base)
        beta = (u_global*error).flatten(1).sum(1).square()/predicted
        assert np.array_equal(beta.cpu().numpy(), old_moments['beta'][batch_ids]), 'cached global projection parity'
        residual, active, token_ratio, reconstructed_beta = standardized_residual(error, full, base, predicted)
        assert torch.isfinite(residual).all()
        torch.testing.assert_close(reconstructed_beta[active], beta[active], atol=1e-7, rtol=1e-11)
        torch.testing.assert_close(residual.square().sum(1)[active],
                                   (token_ratio-reconstructed_beta)[active], atol=1e-7, rtol=1e-10)
        assert (token_ratio <= mse*262144/predicted*(1+1e-12)).all()
        residuals[begin:begin+8] = residual.cpu().numpy()
        active_np[begin:begin+8] = active.cpu().numpy()
        for name, value in [('global_beta', beta), ('token_projection_ratio', token_ratio), ('predicted_mse', predicted)]:
            fields[name][begin:begin+8] = value.cpu().numpy()
        if (begin//8+1) % 64 == 0 or begin+8 == len(ids):
            progress = {'samples': begin+8, 'target': len(ids), 'seconds': time.perf_counter()-began}
            temporary = out/'progress.tmp'
            temporary.write_text(json.dumps(progress)+'\n')
            temporary.replace(out/'progress.json')
            print(json.dumps(progress), flush=True)
    assert tap.calls == len(ids)//8
    tap.close()
    np.savez(out/'moments.npz', ids=ids, labels=ids % 1000, times=times_np,
             query_indices=previous_meta['query_indices'][ids], active=active_np, residual=residuals, **fields)
    for path, expected in sources.items():
        assert sha(Path(path)) == expected
    record = {'complete': True, 'count': len(ids), 'teacher_sample_main_calls': len(ids),
              'cached_native_features_total_mse_scalar_variance_global_projection_bitwise_equal': True,
              'source_noise_regeneration_batches': 2, 'active_rows': int(active_np.sum()),
              'seconds_including_data_access': time.perf_counter()-began, 'sources': sources,
              'config_sha256': request['config_sha256'], 'checkpoint_sha256': request['checkpoint_sha256'],
              'spherical_head_sha256': sha(head_path), 'global_calibration_sha256': sha(global_path),
              'files': {'moments.npz': sha(out/'moments.npz')}}
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

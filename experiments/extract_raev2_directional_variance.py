"""Repeat the fixed teacher inputs once, checking all cached features and MSE."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config, native_clean, seed_for, digest
from experiments.raev2_conditional_variance import NativePrefixTap, ConditionalVarianceHead
from experiments.raev2_directional_variance import PLAN, unit_disagreement, fixed_mse
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    out = DATA/'directional_variance_moments'/args.split/f'shard{args.shard}'
    out.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT/'experiments/raev2_directional_variance.py',
        ROOT/'experiments/raev2_conditional_variance.py', ROOT/'experiments/sample_raev2_ancestral_guidance.py']}
    previous = DATA/'conditional_variance_features'/args.split
    previous_meta = np.load(previous/'metadata.npz')
    previous_features = np.load(previous/'features.npy', mmap_mode='r')
    count = PLAN[args.split]
    ids = np.arange(count).reshape(-1, 8)[args.shard::4].reshape(-1)
    native = DATA/'actual_ratio_bank64k'/args.split/f'shard{args.shard}'
    request = json.loads((native/'request.json').read_text())
    summary = json.loads((native/'summary.json').read_text())
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
    fields = {key: np.empty(len(ids), dtype=np.float64) for key in ['beta', 'projection_energy', 'gap_energy', 'predicted_mse']}
    active_np = np.empty(len(ids), dtype=bool)
    began = time.perf_counter()
    for begin in range(0, len(ids), 8):
        batch_ids = ids[begin:begin+8]
        x = torch.from_numpy(np.array(real[batch_ids], copy=True)).cuda().float()
        epsilon = torch.from_numpy(np.array(noise[begin:begin+8], copy=True)).cuda()
        times = torch.from_numpy(times_np[begin:begin+8].copy()).cuda()
        labels = torch.from_numpy(batch_ids%1000).cuda()
        state = (1-times[:, None, None, None])*x+times[:, None, None, None]*epsilon
        with torch.autocast('cuda', dtype=torch.bfloat16):
            full, base = model(state, times, context=labels, attn_mask=None)
            clean = native_clean(full, base, float(times_np[begin]), 'official')
        features = tap.take()
        assert np.array_equal(features.cpu().numpy(), previous_features[batch_ids]), 'cached native feature parity failed'
        error = (x-clean).double()
        mse = error.square().flatten(1).mean(1)
        assert np.array_equal(mse.cpu().numpy(), previous_meta['residual_mse'][batch_ids]), 'cached total MSE parity failed'
        u, active, gap = unit_disagreement(full, base)
        projection = (u*error).flatten(1).sum(1).square()
        assert (projection <= mse*262144*(1+1e-12)).all()
        index = int(previous_meta['query_indices'][batch_ids[0]])
        predicted = fixed_mse(head, features, calibration['rows'][index]['mse'])
        beta = projection/predicted
        assert torch.isfinite(beta).all() and (beta >= 0).all() and torch.isfinite(predicted).all() and (predicted > 0).all()
        for name, value in [('beta', beta), ('projection_energy', projection), ('gap_energy', gap), ('predicted_mse', predicted)]:
            fields[name][begin:begin+8] = value.cpu().numpy()
        active_np[begin:begin+8] = active.cpu().numpy()
        if (begin//8+1)%64 == 0 or begin+8 == len(ids):
            print(json.dumps({'count': begin+8, 'target': len(ids), 'seconds': time.perf_counter()-began}), flush=True)
    tap.close()
    np.savez(out/'moments.npz', ids=ids, labels=ids%1000, times=times_np,
             query_indices=previous_meta['query_indices'][ids], active=active_np, **fields)
    for path, expected in sources.items():
        assert sha(Path(path)) == expected
    record = {'complete': True, 'count': len(ids), 'teacher_sample_main_calls': len(ids),
              'all_cached_features_and_total_mse_bitwise_equal': True, 'source_noise_regeneration_batches': 2,
              'seconds_including_data_access': time.perf_counter()-began, 'sources': sources,
              'spherical_head_sha256': sha(head_path), 'files': {'moments.npz': sha(out/'moments.npz')}}
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

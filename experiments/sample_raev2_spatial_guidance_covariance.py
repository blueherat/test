"""Fixed paired RAEv2 sampling for a fitted global-orthogonal covariance."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'external/RAEv2/src'):
    sys.path.insert(0, str(path))
from experiments.sample_raev2_guidance_quadrature import (
    put, digest, seed_for, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config,
    shifted_time_grid, file_sha256, instantiate_from_config,
)
from experiments.raev2_guidance_quadrature import native_clean
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.raev2_conditional_variance import NativePrefixTap, ConditionalVarianceHead
from experiments.raev2_directional_variance import colored_noise as global_colored_noise
from experiments.raev2_spatial_guidance_covariance import PLAN, colored_noise
from experiments.summarize_raev2_guidance_20260907 import DATA

FIT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_spatial_guidance_covariance_20260907/repaired/fit')
MODES = ['official', 'global', 'unit', 'spatial_full', 'spatial_diagonal']


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit', type=Path, default=FIT)
    p.add_argument('--modes', nargs='+', choices=MODES, default=['spatial_full', 'spatial_diagonal'])
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=202609072)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    if args.samples <= 0 or args.samples % 8 or not 0 <= args.shard < args.shards:
        p.error('positive complete B8 batches and valid shard required')
    if 'unit' in args.modes and (args.samples != 8 or args.shards != 1):
        p.error('unit covariance is an 8-image implementation check only')
    out = args.output.resolve()/f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    calibration = json.loads((args.fit/'calibration.json').read_text())
    assert calibration['complete'] and calibration['entry_condition_passed'] and calibration['plan'] == PLAN
    assert calibration['covariance_sha256'] == file_sha256(args.fit/'covariance.npz')
    assert calibration['global_calibration_sha256'] == file_sha256(DATA/'directional_variance_moments/calibration.json')
    assert calibration['spherical_head_sha256'] == file_sha256(DATA/'conditional_variance_fit/head.pt')
    roots = {}
    with np.load(args.fit/'covariance.npz') as values:
        for mode, name in [('spatial_full', 'root_full'), ('spatial_diagonal', 'root_diagonal')]:
            roots[mode] = torch.from_numpy(values[name].copy()).cuda()
    roots['unit'] = torch.eye(PLAN['fitted_dimension'], dtype=torch.float64, device='cuda')
    kappa = calibration['kappa']
    head = ConditionalVarianceHead(DATA/'conditional_variance_fit/head.pt').cuda().eval().requires_grad_(False)
    scalar_path = DATA/'guided_reverse_variance/calibration.json'
    assert head.calibration_sha256 == file_sha256(scalar_path)
    scalar = json.loads(scalar_path.read_text())
    os.environ.setdefault('DINOV3_CKPT_DIR', '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3')
    install_raev2_decoder_config_compat()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(DEFAULT_CONFIG)
    decoder = instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(ckpt['ema'], strict=True)
    del ckpt
    shift = math.sqrt((cfg.misc.time_dist_shift_dim or math.prod(cfg.misc.latent_size))/cfg.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, torch.device('cuda')).cpu().tolist()
    assert len(scalar['rows']) == 100 and max(abs(t-r['time']) for t,r in zip(grid[:-1], scalar['rows'])) < 2e-7
    request = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    request.update(protocol='raev2_spatial_guidance_covariance_v1', plan=PLAN, steps=100, batch=8,
        checkpoint_sha256=file_sha256(DEFAULT_CHECKPOINT), config_sha256=file_sha256(DEFAULT_CONFIG),
        state_key='ema', time_grid=grid, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        precision='native BF16 heads/mix, FP32 Euler and colored noise, FP64 covariance algebra, TF32 on',
        decoder_sha256=file_sha256(Path(cfg.stage_1.params['pretrained_decoder_path'])),
        stats_sha256=file_sha256(Path(cfg.stage_1.params['normalization_stat_path'])),
        covariance_calibration_sha256=file_sha256(args.fit/'calibration.json'),
        covariance_sha256=file_sha256(args.fit/'covariance.npz'),
        spherical_head_sha256=calibration['spherical_head_sha256'],
        global_calibration_sha256=calibration['global_calibration_sha256'],
        scalar_calibration_sha256=file_sha256(scalar_path), kappa=kappa,
        refresh_noise='one native FP32 Gaussian per original time, seed_for(seed,global_batch,refresh)')
    assert scalar['baseline_checkpoint']['sha256'] == request['checkpoint_sha256']
    assert scalar['config']['sha256'] == request['config_sha256']
    put(out/'request.json', request)
    batch_ids = list(range(args.shard, args.samples//8, args.shards))
    assert batch_ids
    from utils.guidance_utils import forward_with_internalguidance
    dummy = torch.zeros(8, *cfg.misc.latent_size, device='cuda')
    times, labels = torch.full((8,), .8, device='cuda'), torch.arange(8, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        f,b = model(dummy, times, context=labels, attn_mask=None)
        ref = forward_with_internalguidance(model, torch.cat([dummy,dummy]), torch.cat([times,times]),
              ig_scale=1.78, ig_interval=(.1,1.), context=torch.cat([labels,labels]), attn_mask=None)[:8]
        assert torch.equal(ref.float(), native_clean(f,b,.8))
        decoder.decode(f.float())
    torch.cuda.synchronize()
    put(out/'warmup.json', {'native_ig_parity': True})
    for mode in args.modes:
        target = out/mode
        target.mkdir()
        tap = None if mode == 'official' else NativePrefixTap(model)
        records, arrays, ids_all, diagnostics = [], [], [], []
        inactive_count = 0
        trajectory_seconds = decode_seconds = 0.
        began = time.perf_counter()
        for batch_id in batch_ids:
            ids = np.arange(batch_id*8, (batch_id+1)*8)
            rng = torch.Generator(device='cuda').manual_seed(seed_for(args.seed, batch_id, 'initial'))
            refresh = torch.Generator(device='cuda').manual_seed(seed_for(args.seed, batch_id, 'refresh'))
            state = torch.randn(8, *cfg.misc.latent_size, device='cuda', generator=rng)
            labels = torch.from_numpy(ids % 1000).cuda()
            records.append({'batch': batch_id, 'noise_sha256': digest(state), 'labels_sha256': digest(labels)})
            torch.cuda.synchronize()
            start = time.perf_counter()
            for index, (t,s) in enumerate(zip(grid[:-1], grid[1:])):
                times = torch.full((8,), t, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    full,base = model(state, times, context=labels, attn_mask=None)
                    clean = native_clean(full,base,t)
                if mode == 'official':
                    state = state-(t-s)*((state-clean)/max(t, float(cfg.transport.t_eps)))
                else:
                    features = tap.take()
                    noise = torch.randn(state.shape, device='cuda', generator=refresh)
                    if mode == 'global':
                        transformed, active = global_colored_noise(noise, full, base, kappa)
                    else:
                        transformed, active = colored_noise(noise, full, base, roots[mode], kappa)
                    inactive_count += int((~active).sum())
                    coefficient, log_ratio = head.noise_coefficient(features, scalar['rows'][index]['mse'], (t-s)/t)
                    euler = state-(t-s)*((state-clean)/t)
                    state = euler+coefficient[:,None,None,None]*transformed
                    if batch_id == batch_ids[0]:
                        diagnostics.append({'step': index, 'time': t,
                            'mean_log_variance_ratio': float(log_ratio.mean()),
                            'colored_noise_rms': float(transformed.square().mean().sqrt())})
            torch.cuda.synchronize()
            trajectory_seconds += time.perf_counter()-start
            assert torch.isfinite(state).all()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pixels = decoder.decode(state).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            torch.cuda.synchronize()
            decode_seconds += time.perf_counter()-start
            arrays.append(pixels)
            ids_all.extend(ids.tolist())
            if batch_id == batch_ids[0]:
                from PIL import Image
                Image.fromarray(np.concatenate(list(pixels[:4]), axis=1)).save(target/'preview.png')
                np.save(target/'first_endpoint.npy', state.cpu().numpy())
            put(target/'progress.json', {'samples': len(ids_all), 'target': len(batch_ids)*8,
                                        'elapsed_seconds': time.perf_counter()-began})
            if len(arrays) % 10 == 0 or batch_id == batch_ids[-1]:
                print(json.dumps({'mode':mode, 'shard':args.shard, 'samples':len(ids_all)}), flush=True)
        if tap is not None:
            assert tap.calls == len(batch_ids)*100
            tap.close()
        np.savez(target/'samples.npz', np.concatenate(arrays), ids=np.asarray(ids_all, dtype=np.int64))
        put(target/'summary.json', {'complete':True, 'mode':mode, 'samples':len(ids_all),
            'sample_sha256':file_sha256(target/'samples.npz'), 'trajectory_seconds':trajectory_seconds,
            'decode_seconds':decode_seconds, 'total_seconds':time.perf_counter()-began,
            'sample_model_calls':len(ids_all)*100, 'initial_noise':records, 'first_batch_diagnostics':diagnostics,
            'inactive_image_queries':inactive_count, 'max_memory_allocated':torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()

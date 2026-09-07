#!/usr/bin/env python3
"""Native paired samples for a finite causal weak reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'external/RAEv2/src'):
    sys.path.insert(0, str(path))

from experiments.raev2_guidance_quadrature import native_clean
from experiments.raev2_causal_reference import MODES, PLAN, euler, calibrated_clean
from experiments.raev2_pfr_retiming import evaluate_base_head_only
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, shifted_time_grid
from experiments.raev2_training_core import file_sha256
from utils.model_utils import instantiate_from_config


def put(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_for(seed, batch, namespace):
    # Exact historical paired-input namespace, independent of GPU sharding.
    return int.from_bytes(hashlib.sha256(f'raev2-ag-20260907:{namespace}:{seed}:{batch}'.encode()).digest()[:8], 'little') % 2**63


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=MODES, default=['causal_reference'])
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=202609072)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    if args.samples <= 0 or args.samples % args.batch or not 0 <= args.shard < args.shards:
        p.error('requires positive complete batches and a valid shard')
    if args.steps != 100 or args.batch != 8:
        p.error('this study is fixed to the original 100 steps and B8')
    if 'causal_null' in args.modes and (args.samples != 8 or args.shards != 1):
        p.error('causal_null is only an eight-image implementation check')
    out = args.output.resolve() / f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('DINOV3_CKPT_DIR', '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3')
    install_raev2_decoder_config_compat()
    torch.cuda.set_device(0)
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
    shift = math.sqrt((cfg.misc.time_dist_shift_dim or math.prod(cfg.misc.latent_size)) / cfg.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.steps, shift, torch.device('cuda')).cpu().tolist()
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    request.update(protocol='raev2_causal_reference_v1', plan=PLAN, checkpoint_sha256=file_sha256(DEFAULT_CHECKPOINT),
                   config_sha256=file_sha256(DEFAULT_CONFIG), state_key='ema', time_grid=grid,
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                   precision='native BF16 heads/mix, FP32 finite reference and Euler, TF32 on',
                   source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [Path(__file__).resolve(), ROOT/'experiments/raev2_guidance_quadrature.py', ROOT/'experiments/raev2_causal_reference.py', ROOT/'experiments/raev2_pfr_retiming.py', ROOT/'external/RAEv2/src/stage2/models/DDT.py']},
                   decoder_sha256=file_sha256(Path(cfg.stage_1.params['pretrained_decoder_path'])),
                   stats_sha256=file_sha256(Path(cfg.stage_1.params['normalization_stat_path'])))
    put(out/'request.json', request)
    batch_ids = list(range(args.shard, args.samples // args.batch, args.shards))
    if not batch_ids:
        raise ValueError('empty shard')
    from utils.guidance_utils import forward_with_internalguidance
    dummy = torch.zeros(args.batch, *cfg.misc.latent_size, device='cuda')
    times = torch.full((args.batch,), .8, device='cuda')
    labels = torch.arange(args.batch, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        f, b = model(dummy, times, context=labels, attn_mask=None)
        ref = forward_with_internalguidance(model, torch.cat([dummy, dummy]), torch.cat([times, times]),
              ig_scale=1.78, ig_interval=(.1, 1.), context=torch.cat([labels, labels]), attn_mask=None)[:args.batch]
        if not torch.equal(ref.float(), native_clean(f, b, .8)):
            raise AssertionError('native IG arithmetic parity failed')
        prefix = evaluate_base_head_only(model, dummy, times, context=labels, attn_mask=None)
        assert torch.equal(prefix, b), 'exact prefix/base parity'
        decoder.decode(f.float())
    torch.cuda.synchronize()
    put(out/'warmup.json', {'native_ig_parity': True, 'native_prefix_base_parity': True})
    for mode in args.modes:
        target = out/mode
        target.mkdir()
        records, arrays, ids_all, diagnostics = [], [], [], []
        trajectory_seconds = decode_seconds = 0.
        started = time.perf_counter()
        for batch_id in batch_ids:
            ids = np.arange(batch_id * args.batch, (batch_id + 1) * args.batch)
            rng = torch.Generator(device='cuda').manual_seed(seed_for(args.seed, batch_id, 'initial'))
            state = torch.randn(args.batch, *cfg.misc.latent_size, device='cuda', generator=rng)
            labels = torch.from_numpy(ids % 1000).cuda()
            records.append({'batch': batch_id, 'noise_sha256': digest(state), 'labels_sha256': digest(labels)})
            torch.cuda.synchronize()
            start = time.perf_counter()
            for index, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
                times = torch.full((args.batch,), t, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    full, base = model(state, times, context=labels, attn_mask=None)
                    clean = native_clean(full, base, t)
                response = torch.zeros_like(clean)
                if mode != 'official' and t >= .1:
                    query_full = euler(state, full.float(), t, s)
                    query_guided = query_full if mode == 'causal_null' else euler(state, clean, t, s)
                    future_time = torch.full((args.batch,), s, device='cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        base_full = evaluate_base_head_only(model, query_full, future_time, context=labels, attn_mask=None).float()
                        base_guided = evaluate_base_head_only(model, query_guided, future_time, context=labels, attn_mask=None).float()
                    response = base_guided-base_full
                    effective = calibrated_clean(clean, base_guided, base_full)
                else:
                    effective = clean
                if batch_id == batch_ids[0]:
                    message = clean-full.float()
                    diagnostics.append({'step': index, 'time': t, 'future_time': s,
                        'message_rms': float(message.square().mean().sqrt()),
                        'finite_response_rms': float(response.square().mean().sqrt()),
                        'clean_correction_rms': float((effective-clean).square().mean().sqrt()),
                        'response_message_dot': float((response.double()*message.double()).sum()),
                        'message_squared_norm': float(message.double().square().sum())})
                state = euler(state, effective, t, s)
            torch.cuda.synchronize()
            trajectory_seconds += time.perf_counter() - start
            if not torch.isfinite(state).all():
                raise FloatingPointError(f'nonfinite {mode} batch {batch_id}')
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pixels = decoder.decode(state).clamp(0, 1).mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            torch.cuda.synchronize()
            decode_seconds += time.perf_counter() - start
            arrays.append(pixels)
            ids_all.extend(ids.tolist())
            if batch_id == batch_ids[0]:
                from PIL import Image
                Image.fromarray(np.concatenate(list(pixels[:4]), axis=1)).save(target/'preview.png')
                np.save(target/'first_endpoint.npy', state.cpu().numpy())
            put(target/'progress.json', {'samples': len(ids_all), 'target': len(batch_ids)*args.batch,
                                        'elapsed_seconds': time.perf_counter()-started})
            if len(arrays) % 10 == 0 or batch_id == batch_ids[-1]:
                print(json.dumps({'mode':mode, 'shard':args.shard, 'samples':len(ids_all),
                                  'seconds':round(time.perf_counter()-started, 2)}), flush=True)
        np.savez(target/'samples.npz', np.concatenate(arrays), ids=np.array(ids_all, dtype=np.int64))
        put(target/'summary.json', {'complete':True, 'mode':mode, 'samples':len(ids_all),
            'sample_sha256':file_sha256(target/'samples.npz'), 'trajectory_seconds':trajectory_seconds,
            'decode_seconds':decode_seconds, 'total_seconds':time.perf_counter()-started,
            'sample_model_calls':len(ids_all)*args.steps,
            'prefix_sample_calls':len(ids_all)*2*sum(t>=.1 for t in grid[:-1]) if mode!='official' else 0,
            'inactive_current_prefix_calls':0, 'initial_noise':records, 'first_batch_diagnostics':diagnostics,
            'max_memory_allocated':torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()

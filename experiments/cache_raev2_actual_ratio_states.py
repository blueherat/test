"""One preselected native query state and its initial noise per generated id.

Writes FP32 states/noise so high-noise paired differences survive storage.
No decoder, images, FID or candidate guidance is used in this data preparation.
"""
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
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, instantiate_from_config,
    shifted_time_grid, native_clean, seed_for, digest)
from experiments.summarize_raev2_guidance_20260907 import sha


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--time-seed', type=int, required=True)
    p.add_argument('--shard', type=int, required=True)
    args = p.parse_args()
    assert args.samples % 8 == 0 and 0 <= args.shard < 4
    out = args.output / f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(path): sha(path) for path in (Path(__file__).resolve(),
               ROOT / 'experiments/sample_raev2_ancestral_guidance.py')}
    config = load_config(DEFAULT_CONFIG)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = instantiate_from_config(config.stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    grid = shifted_time_grid(100, 8., torch.device('cuda')).cpu().tolist()
    batches = np.arange(args.shard, args.samples // 8, 4)
    count = len(batches) * 8
    query = np.tile(np.arange(1, 100), (args.samples // 8 + 98) // 99)[:args.samples // 8]
    np.random.default_rng(args.time_seed).shuffle(query)
    states = np.lib.format.open_memmap(out / 'states.npy', mode='w+', dtype=np.float32, shape=(count, 1024, 16, 16))
    noises = np.lib.format.open_memmap(out / 'noise.npy', mode='w+', dtype=np.float32, shape=states.shape)
    request = {'samples': args.samples, 'seed': args.seed, 'time_seed': args.time_seed,
               'shard': args.shard, 'batch': 8, 'query_grid': grid, 'sources': sources,
               'checkpoint_sha256': sha(Path(DEFAULT_CHECKPOINT)), 'config_sha256': sha(Path(DEFAULT_CONFIG)),
               'state_key': 'ema', 'state_and_noise_precision': 'FP32', 'model_precision': 'native BF16',
               'source_law': 'original official IG100 trajectory before its selected query',
               'no_decoder_no_images_no_fid': True}
    (out / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    records, metadata = [], []
    began = time.perf_counter()
    calls = 0
    for local, batch in enumerate(batches):
        ids = np.arange(batch*8, batch*8+8)
        labels = torch.from_numpy(ids % 1000).cuda()
        rng = torch.Generator(device='cuda').manual_seed(seed_for(args.seed, int(batch), 'initial'))
        noise = torch.randn(8, 1024, 16, 16, device='cuda', generator=rng)
        state = noise.clone()
        stop = int(query[batch])
        for step in range(stop):
            t, s = grid[step:step+2]
            times = torch.full((8,), t, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                full, base = model(state, times, context=labels, attn_mask=None)
                clean = native_clean(full, base, t, 'official')
            state = state - (t-s) * ((state-clean) / max(t, .05))
            calls += 8
        assert torch.isfinite(state).all()
        states[local*8:local*8+8] = state.cpu().numpy()
        noises[local*8:local*8+8] = noise.cpu().numpy()
        metadata.extend([[int(i), int(i % 1000), stop, grid[stop]] for i in ids])
        records.append({'batch': int(batch), 'noise_sha256': digest(noise),
                        'state_sha256': digest(state), 'query_index': stop})
        if (local+1) % 8 == 0 or local+1 == len(batches):
            progress = {'count': (local+1)*8, 'target': count, 'sample_model_calls': calls,
                        'seconds': time.perf_counter()-began}
            (out / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps(progress), flush=True)
    states.flush()
    noises.flush()
    np.savez(out / 'metadata.npz', records=np.array(metadata))
    for path, digest_expected in sources.items():
        assert sha(Path(path)) == digest_expected, 'source changed during cache preparation'
    result = {'complete': True, 'count': count, 'sample_model_calls': calls,
              'seconds': time.perf_counter()-began, 'batch_records': records,
              'request_sha256': sha(out / 'request.json'),
              'files': {name: sha(out / name) for name in ('states.npy', 'noise.npy', 'metadata.npz')}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

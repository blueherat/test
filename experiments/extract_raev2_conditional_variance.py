"""One native teacher query per existing real/noise pair; no rollout or decode."""
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
from experiments.raev2_conditional_variance import PLAN, query_indices, NativePrefixTap
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    spec = PLAN[args.split]
    out = DATA/'conditional_variance_features'/args.split/f'shard{args.shard}'
    out.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT/'experiments/raev2_conditional_variance.py',
        ROOT/'experiments/sample_raev2_ancestral_guidance.py']}
    real_record = json.loads((DATA/'real_ratio_bank64k/execution.json').read_text())['splits'][args.split]
    real = np.load(real_record['real_path'], mmap_mode='r')
    folder = DATA/'actual_ratio_bank64k'/args.split/f'shard{args.shard}'
    native_request = json.loads((folder/'request.json').read_text())
    native_summary = json.loads((folder/'summary.json').read_text())
    metadata = np.load(folder/'metadata.npz')['records']
    ids = metadata[:, 0].astype(np.int64)
    expected = np.arange(spec['count']).reshape(-1, 8)[args.shard::4].reshape(-1)
    assert np.array_equal(ids, expected) and np.array_equal(metadata[:, 1], ids%1000)
    noise = np.load(folder/'noise.npy', mmap_mode='r')
    assert real.shape == (spec['count'], 1024, 16, 16) and real.dtype == np.float16
    assert noise.shape == (len(ids), 1024, 16, 16) and noise.dtype == np.float32
    grid = np.asarray(native_request['query_grid'][:-1], dtype=np.float32)
    assert len(grid) == 100 and grid[0] == 1 and grid[-1] > .05
    query = query_indices(args.split)[ids//8]
    times_np = grid[query]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    for begin in [0, len(ids)-8]:
        batch = int(ids[begin]//8)
        rng = torch.Generator(device='cuda').manual_seed(seed_for(native_request['seed'], batch, 'initial'))
        regenerated = torch.randn(8, 1024, 16, 16, device='cuda', generator=rng)
        assert np.array_equal(regenerated.cpu().numpy(), noise[begin:begin+8])
        assert digest(regenerated) == native_summary['batch_records'][begin//8]['noise_sha256']
    tap = NativePrefixTap(model)
    features = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float32, shape=(len(ids), 2880))
    residual = np.empty(len(ids), dtype=np.float64)
    began = time.perf_counter()
    parity = False
    for begin in range(0, len(ids), 8):
        x = torch.from_numpy(np.array(real[ids[begin:begin+8]], copy=True)).cuda().float()
        epsilon = torch.from_numpy(np.array(noise[begin:begin+8], copy=True)).cuda()
        times = torch.from_numpy(times_np[begin:begin+8].copy()).cuda()
        labels = torch.from_numpy(ids[begin:begin+8]%1000).cuda()
        state = (1-times[:, None, None, None])*x+times[:, None, None, None]*epsilon
        with torch.autocast('cuda', dtype=torch.bfloat16):
            full, base = model(state, times, context=labels, attn_mask=None)
            clean = native_clean(full, base, float(times_np[begin]), 'official')
        phi = tap.take()
        if not parity:
            tap.close()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                original_full, original_base = model(state, times, context=labels, attn_mask=None)
            assert torch.equal(full, original_full) and torch.equal(base, original_base), 'feature hook changed native outputs'
            tap = NativePrefixTap(model)
            parity = True
        r = (x-clean).double().square().flatten(1).mean(1)
        assert torch.isfinite(phi).all() and torch.isfinite(r).all() and (r > 0).all()
        features[begin:begin+8] = phi.cpu().numpy()
        residual[begin:begin+8] = r.cpu().numpy()
        if (begin//8+1)%32 == 0 or begin+8 == len(ids):
            print(json.dumps({'count': begin+8, 'target': len(ids), 'seconds': time.perf_counter()-began}), flush=True)
    token_dtype = tap.token_dtype
    tap.close()
    features.flush()
    np.savez(out/'metadata.npz', ids=ids, labels=ids%1000, times=times_np, query_indices=query, residual_mse=residual)
    for path, expected_digest in sources.items():
        assert sha(Path(path)) == expected_digest
    result = {'complete': True, 'count': len(ids), 'source_noise_regeneration_batches': 2,
              'native_outputs_bitwise_unchanged_with_hook': parity, 'observed_token_dtype': token_dtype,
              'teacher_sample_main_calls': len(ids), 'extra_parity_sample_main_calls': 8,
              'no_decoder_or_input_backward': True, 'seconds': time.perf_counter()-began,
              'checkpoint_sha256': sha(Path(DEFAULT_CHECKPOINT)), 'config_sha256': sha(Path(DEFAULT_CONFIG)),
              'sources': sources, 'files': {name: sha(out/name) for name in ['features.npy', 'metadata.npz']}}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Exact enumeration of five real endpoints per existing native/noise pair.

Rao-Blackwellizes the empirical real-endpoint draw; no new generated state,
time, feature layer or classifier hyperparameter is selected.
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
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config
from experiments.raev2_actual_ratio_data import ActualPairs
from experiments.raev2_prefix_ratio_features import prefix_features
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    out = DATA / 'prefix_ratio_rb_features' / f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    manifest = json.loads((DATA/'paired_ratio_data/manifest.json').read_text())
    bank = ActualPairs(DATA/'actual_ratio_bank/train', manifest['splits']['train'], 5000)
    previous_path = DATA / 'prefix_ratio_features/train/features.npz'
    previous = np.load(previous_path)
    old = previous['features']
    ids = np.arange(5000).reshape(-1, 8)[args.shard::4].reshape(-1)
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/'experiments/raev2_actual_ratio_data.py',
               ROOT/'experiments/raev2_prefix_ratio_features.py')}
    positives, negatives, times = [], [], []
    began = time.perf_counter()
    for start in range(0, len(ids), 8):
        batch_ids = ids[start:start+8]
        all_positive = []
        for k in range(5):
            p, q, t, labels = bank.batch(batch_ids, np.full(8, k), 'cuda')
            fp = prefix_features(model, p, t, labels).cpu().numpy()
            if k == 0:
                fq = prefix_features(model, q, t, labels).cpu().numpy()
                assert np.array_equal(fq, old[batch_ids, 1]), 'original negative prefix features changed'
                negatives.append(fq)
                times.extend(t.cpu().tolist())
            selected = batch_ids//1000 == k
            assert np.array_equal(fp[selected], old[batch_ids[selected], 0]), 'original selected positive features changed'
            all_positive.append(fp)
        positives.append(np.stack(all_positive, axis=1))
        if (start//8+1) % 16 == 0:
            print('completed', start+8, 'of', len(ids), flush=True)
    np.savez(out/'features.npz', ids=ids, times=np.array(times, dtype=np.float32),
             positive=np.concatenate(positives), negative=np.concatenate(negatives))
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    record = {'complete': True, 'count': len(ids), 'real_alternatives_per_noise': 5,
              'sources': sources, 'seconds': time.perf_counter()-began,
              'original_selected_pair_features_bitwise_equal': True,
              'original_features_sha256': sha(previous_path), 'features_sha256': sha(out/'features.npz'),
              'fid_used': False}
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

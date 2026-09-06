"""Extract fixed prefix features on paired real/native states, without FID."""
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
from experiments.raev2_prefix_ratio_features import prefix_features, token_statistics
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    args = parser.parse_args()
    count = 5000 if args.split == 'train' else 1000
    out = DATA / 'prefix_ratio_features' / args.split / f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    manifest = json.loads((DATA / 'paired_ratio_data/manifest.json').read_text())
    bank = ActualPairs(DATA / 'actual_ratio_bank' / args.split, manifest['splits'][args.split], count)
    ids = np.arange(count).reshape(-1, 8)[args.shard::4].reshape(-1)
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(),
        ROOT / 'experiments/raev2_actual_ratio_data.py', ROOT / 'experiments/raev2_prefix_ratio_features.py')}
    feature_rows, time_rows = [], []
    began = time.perf_counter()
    parity = False
    for start in range(0, len(ids), 8):
        batch_ids = ids[start:start+8]
        real_k = batch_ids//1000 if args.split == 'train' else np.zeros(8, dtype=int)
        p, q, t, labels = bank.batch(batch_ids, real_k, 'cuda')
        if not parity:
            captured = []
            hook = model.blocks[model.base_model_depth-1].register_forward_hook(lambda module, inputs, output: captured.append(output))
            model(q, t, context=labels, attn_mask=None)
            hook.remove()
            expected = token_statistics(captured[0][:, :model.s_embedder.num_patches])
            actual = prefix_features(model, q, t, labels)
            assert torch.equal(expected, actual), 'standalone/native prefix FP32 feature mismatch'
            del captured
            parity = True
        # Preserve the original B8 layout for the prefix on both sides.
        fp = prefix_features(model, p, t, labels)
        fq = prefix_features(model, q, t, labels)
        assert torch.isfinite(fp).all() and torch.isfinite(fq).all()
        feature_rows.append(torch.stack((fp, fq), dim=1).cpu().numpy())
        time_rows.extend(t.cpu().tolist())
        if (start//8+1) % 16 == 0:
            print('completed', start+8, 'of', len(ids), flush=True)
    features = np.concatenate(feature_rows)
    np.savez(out / 'features.npz', ids=ids, times=np.array(time_rows, dtype=np.float32), features=features)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    summary = {'complete': True, 'count': len(ids), 'feature_dim': features.shape[-1],
               'prefix_depth': model.base_model_depth, 'native_fp32_prefix_parity': parity,
               'seconds': time.perf_counter()-began, 'sources': sources,
               'checkpoint_sha256': sha(Path(DEFAULT_CHECKPOINT)), 'config_sha256': sha(Path(DEFAULT_CONFIG)),
               'features_sha256': sha(out / 'features.npz'), 'fid_used': False,
               'real_pairing': 'one fixed real endpoint per actual id; same class and initial noise',
               'precision': 'FP32 prefix and stored features, TF32 disabled'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Fix disjoint real sources and historical fake seeds for one critic fit."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA / 'paired_ratio_data'
    out.mkdir(exist_ok=False)
    real = DATA.parent / 'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
    train_meta = np.load(real / 'train/metadata.npz')
    valid_meta = np.load(real / 'validation/metadata.npz')
    print('metadata keys', train_meta.files, valid_meta.files)
    print('train shapes', {k: train_meta[k].shape for k in train_meta.files})
    row_key = next(k for k in ('source_rows', 'rows') if k in train_meta.files)
    assert not np.intersect1d(train_meta[row_key], valid_meta[row_key]).size
    splits = {}
    for split, seed, meta in [('train', 20260801, train_meta), ('validation', 20260802, valid_meta)]:
        label_key = 'labels'
        labels = meta[label_key]
        count = 5 if split == 'train' else 1
        assert len(labels) == count * 1000
        lookup = np.stack([np.flatnonzero(labels == c) for c in range(1000)])
        assert lookup.shape == (1000, count)
        np.save(out / f'{split}_real_lookup.npy', lookup)
        fake_root = DATA.parent / 'raev2_ig_scale_response' / f'n5000_seed{seed}_scales7_v1/latents'
        fake_files = [fake_root / f'scale_s1p780000_rank{rank:02d}.npy' for rank in range(4)]
        for p in fake_files:
            assert np.load(p, mmap_mode='r').shape == (1250, 1024, 16, 16)
        sources = [real / split / 'latents.npy', real / split / 'metadata.npz', *fake_files]
        splits[split] = {'real_path': str(sources[0]), 'real_lookup': str(out / f'{split}_real_lookup.npy'),
                         'fake_paths': [str(p) for p in fake_files],
                         'fake_global_id_rule': 'rank = id % 4; row = id // 4; label = id % 1000',
                         'real_per_class': count, 'fake_per_class': 5,
                         'sources': {str(p): sha(p) for p in sources}}
    checkpoint = Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt')
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)['ema']
    class_features = state['ctx_embedder.embedding_table.weight'][:1000].clone()
    assert class_features.shape == (1000, 1440)
    torch.save(class_features, out / 'class_features.pt')
    record = {'complete': True, 'cuda_used': False, 'splits': splits,
              'checkpoint_sha256': sha(checkpoint),
              'class_features_sha256': sha(out / 'class_features.pt'),
              'real_source_rows_disjoint': True,
              'validation_limit': 'held out from this fit; historical banks were used in earlier research',
              'no_fid_fitting': True,
              'fake_protocol_limit': 'historical native IG EMA, 100 steps, B4, FP16 endpoints; current rollout is B8'}
    (out / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    (ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_data.json').write_text(
        json.dumps(record, indent=2) + '\n')
    print(out, 'class shape', tuple(class_features.shape))


if __name__ == '__main__':
    main()

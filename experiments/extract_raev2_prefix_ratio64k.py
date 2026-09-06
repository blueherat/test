"""Fixed frozen-prefix features for the 64K/8K actual-law experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config, seed_for, digest
from experiments.raev2_actual_ratio_data import ActualPairs
from experiments.raev2_prefix_ratio_features import prefix_features, token_statistics
from experiments.raev2_prefix_ratio64k_plan import PLAN, positive_indices
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    args = parser.parse_args()
    spec = PLAN[args.split]
    out = DATA/'prefix_ratio64k_features'/args.split/f'shard{args.shard}'
    out.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(),
        ROOT/'experiments/raev2_actual_ratio_data.py', ROOT/'experiments/raev2_prefix_ratio_features.py',
        ROOT/'experiments/raev2_prefix_ratio64k_plan.py')}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    assert model.base_model_depth == 8
    manifest = json.loads((DATA/'real_ratio_bank64k/execution.json').read_text())
    assert manifest['complete']
    bank = ActualPairs(DATA/'actual_ratio_bank64k'/args.split, manifest['splits'][args.split], spec['count'])
    assert bank.real_lookup.shape == (1000, spec['real_per_class'])
    ids = np.arange(spec['count']).reshape(-1, 8)[args.shard::4].reshape(-1)
    native_folder = DATA/'actual_ratio_bank64k'/args.split/f'shard{args.shard}'
    native_request = json.loads((native_folder/'request.json').read_text())
    native_summary = json.loads((native_folder/'summary.json').read_text())
    assert native_summary['complete']
    for local in [0, len(ids)//8-1]:
        global_batch = int(ids[local*8]//8)
        rng = torch.Generator(device='cuda').manual_seed(seed_for(native_request['seed'], global_batch, 'initial'))
        epsilon = torch.randn(8, 1024, 16, 16, device='cuda', generator=rng)
        saved = bank.noise[args.shard][local*8:local*8+8]
        assert np.array_equal(epsilon.cpu().numpy(), saved), 'regenerated initial noise mismatch'
        assert digest(epsilon) == native_summary['batch_records'][local]['noise_sha256']
        state = torch.from_numpy(np.array(bank.states[args.shard][local*8:local*8+8], copy=True))
        assert digest(state) == native_summary['batch_records'][local]['state_sha256']
    # Whole global B8 batches retain native attention/numerical layout.
    positive = np.lib.format.open_memmap(out/'positive.npy', mode='w+', dtype=np.float32,
                                       shape=(len(ids), spec['positive_choices'], 2880))
    negative = np.lib.format.open_memmap(out/'negative.npy', mode='w+', dtype=np.float32, shape=(len(ids), 2880))
    all_k = positive_indices(ids, args.split)
    began, parity = time.perf_counter(), False
    for start in range(0, len(ids), 8):
        batch_ids = ids[start:start+8]
        choices = all_k[start:start+8]
        for k in range(spec['positive_choices']):
            p, q, times, labels = bank.batch(batch_ids, choices[:, k], 'cuda')
            if not parity:
                captured = []
                hook = model.blocks[7].register_forward_hook(lambda module, inputs, output: captured.append(output))
                model(q, times, context=labels, attn_mask=None)
                hook.remove()
                expected = token_statistics(captured[0][:, :model.s_embedder.num_patches])
                observed = prefix_features(model, q, times, labels)
                assert torch.equal(expected, observed), 'native FP32 prefix mismatch'
                del captured, expected, observed
                parity = True
            fp = prefix_features(model, p, times, labels)
            assert torch.isfinite(fp).all()
            positive[start:start+8, k] = fp.cpu().numpy()
            if k == 0:
                fq = prefix_features(model, q, times, labels)
                assert torch.isfinite(fq).all()
                negative[start:start+8] = fq.cpu().numpy()
        if (start//8+1) % 32 == 0 or start+8 == len(ids):
            print(json.dumps({'count': start+8, 'target': len(ids), 'seconds': time.perf_counter()-began}), flush=True)
    positive.flush(); negative.flush()
    np.savez(out/'metadata.npz', ids=ids, times=bank.times[ids], positive_indices=all_k)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    record = {'complete': True, 'count': len(ids), 'positive_choices': spec['positive_choices'],
              'native_fp32_prefix_parity': parity, 'seconds': time.perf_counter()-began, 'sources': sources,
              'checkpoint_sha256': sha(Path(DEFAULT_CHECKPOINT)), 'config_sha256': sha(Path(DEFAULT_CONFIG)),
              'files': {name: sha(out/name) for name in ['positive.npy', 'negative.npy', 'metadata.npz']},
              'fid_used': False, 'precision': 'FP32 features, TF32 disabled',
              'prefix_evaluations': len(ids)*(spec['positive_choices']+1),
              'extra_parity_full_evaluations': 8, 'extra_parity_prefix_evaluations': 8}
    record['initial_noise_regeneration_and_saved_state_digest_batches_checked'] = 2
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

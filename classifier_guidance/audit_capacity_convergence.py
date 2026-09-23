"""Read-only comparison of raw and EMA heads on the original fixed validation states."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.data import Stream
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from .capacity_heads import make
from .evaluate_sit_transformer import ROOT


@torch.no_grad()
def main(args):
    rank, world = c.setup()
    assert rank == 0 and world == 1
    request = c.read(args.output/'request.json')
    for path, digest in request['sources'].items():
        assert c.sha(path) == digest, path
    k.GLOBAL_BATCH = 256
    adapter = Adapter('sit_small')
    frozen = fingerprint(adapter.model)
    heads, receipts, signatures = {}, {}, {}
    for variant in ('shallow', 'linear', 'moderate', 'large', 'block1', 'block2'):
        root = ROOT if variant not in ('moderate', 'large') else ROOT.parent/'sit_mlp_capacity_diffusion_20260920'
        for step in (40000, 45000, 50000):
            path = root/variant/'training_50k'/f'checkpoint_{step:06d}.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['objective'] == 'capacity_diffusion_only' and state['frozen'] == frozen
            assert state['step'] == step and state['config']['ema'] == .9999
            stem = f'{variant}_{step}'
            receipts[stem] = dict(path=str(path), sha256=c.sha(path),
                                  recorded_ema_mse=c.read(path.parent/f'validation_{step:06d}.json')['ema_velocity_mse'])
            for weights in ('head', 'ema'):
                key = f'{stem}_{weights}'
                head = make(variant, adapter.model).cuda().eval().requires_grad_(False)
                head.load_state_dict(state[weights], strict=True)
                heads[key] = head; signatures[key] = fingerprint(head)
            del state
    context = SimpleNamespace(rank=0, world_size=1, device=torch.device('cuda', 0))
    stream = Stream('sit_small', 'real', context, validation=True)
    generator = torch.Generator(device='cuda').manual_seed(7001)
    errors = {key: [] for key in heads}
    totals = {key: torch.zeros((), device='cuda', dtype=torch.float64) for key in heads}
    batches = []
    for batch_index in range(8):
        batch = stream.draw(generator)
        t = batch['t'][:, None, None, None]
        z = t*batch['positive'] + (1-t)*batch['noise']
        target = adapter.patch(batch['positive']-batch['noise'])
        with adapter.autocast(training=True):
            features = adapter.features(z, batch['t'], batch['labels'])
        for key, head in heads.items():
            with adapter.autocast(training=True):
                prediction = head(features['context'], features['condition'])
            per_image = (prediction.float()-target).square().flatten(1).mean(1)
            totals[key] += per_image.mean().double()*len(per_image)
            errors[key].append(per_image.cpu().numpy())
        batches.append(dict(ids=batch['pos_id'].cpu().tolist(),
                            noise_sha256=c.original.array_sha(batch['noise'].cpu().numpy()),
                            time_sha256=c.original.array_sha(batch['t'].cpu().numpy())))
        c.atomic(args.output/'progress.json', dict(completed_batches=batch_index+1, total_batches=8))
    errors = {key: np.concatenate(value) for key, value in errors.items()}
    rows = []
    for stem, receipt in receipts.items():
        raw, ema = (float(totals[f'{stem}_{key}']/2048) for key in ('head', 'ema'))
        assert abs(ema-receipt['recorded_ema_mse']) < 1e-7, (stem, ema, receipt)
        variant, step = stem.rsplit('_', 1)
        difference = errors[f'{stem}_head'].astype(np.float64)-errors[f'{stem}_ema']
        rows.append(dict(variant=variant, step=int(step), raw_mse=raw, ema_mse=ema,
                         raw_minus_ema_paired_mean=float(difference.mean()),
                         raw_minus_ema_paired_sem=float(difference.std(ddof=1)/np.sqrt(len(difference)))))
    assert fingerprint(adapter.model) == frozen
    assert signatures == {key: fingerprint(head) for key, head in heads.items()}
    np.savez(args.output/'per_image_errors.npz', **errors)
    result = dict(passed=True, read_only=True, samples=2048, seed=7001,
                  precision='BF16 autocast / FP32 MSE; exactly the original training validation protocol',
                  ema_decay=.9999, all_recorded_validation_losses_reproduced=True,
                  strong_and_heads_unchanged=True, checkpoints=receipts, batches=batches, rows=rows)
    c.atomic(args.output/'result.json', result)
    c.atomic(args.output/'complete.json', dict(complete=True, passed=True, updated_utc=c.now()))
    for row in rows:
        print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('sit_small',), default='sit_small')
    main(parser.parse_args())

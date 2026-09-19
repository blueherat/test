"""Check accumulated five-round drift at twice the integration resolution.

Four predeclared images only; does not select guidance parameters or overwrite
the main pilot. Reuses the already computed 256-step first copy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fm_common_inverse_copy_20260913.run import Fields, integrate, mse, atomic, sha, ARMS


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True, choices=['ref700', 'ref_alt300'])
    p.add_argument('--out', type=Path, default=Path('/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2'))
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = args.out/args.reference
    assert json.loads((folder/'summary.json').read_text())['complete']
    if (folder/'multiround_refine.json').exists():
        raise FileExistsError('Refinement already completed')
    start = time.monotonic()
    fields = Fields(args.reference, 'cuda:0')
    data = np.load(args.out/'inputs.npz')
    original = torch.from_numpy(data['clean'][:4]).cuda()
    labels = torch.from_numpy(data['labels'][:4]).cuda().long()
    first = np.load(folder/'solver_check.npz')
    rows, arrays = [], {}
    for arm in ARMS:
        current = torch.from_numpy(first[f'{arm}_256']).cuda()
        for r in range(1,6):
            if r > 1:
                code = integrate(fields, current, labels, args.reference, 256, True)
                current = integrate(fields, code, labels, arm, 256)
            with np.load(folder/arm/f'round{r:02d}.npz') as d:
                coarse = torch.from_numpy(d['latents'][:4]).cuda()
            change = mse(current, coarse)
            drift = mse(current, original)
            row = dict(arm=arm, round=r, solver_change_mse=change.tolist(),
                refined_copy_mse=drift.tolist(), mean_change_mse=float(change.mean()),
                mean_refined_copy_mse=float(drift.mean()))
            rows.append(row)
            arrays[f'{arm}_round{r:02d}'] = current.cpu().numpy()
            print(json.dumps(dict(reference=args.reference, **row)), flush=True)
    np.savez(folder/'multiround_refine.npz', **arrays)
    atomic(folder/'multiround_refine.json', dict(rows=rows, seconds=time.monotonic()-start,
        source_sha256=sha(__file__), evaluated_images=4, steps=256,
        calls=fields.calls, first_copy_reused=True))


if __name__ == '__main__':
    main()

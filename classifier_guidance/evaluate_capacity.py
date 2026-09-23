"""One scalar coefficient, 5000 paired inputs, original Heun/ADM protocol."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.sampling import integrate
from .capacity_heads import make
from .sampler import for_adapter


def main(args):
    rank, world = c.setup()
    run = args.output; target = run/'samples'
    request = c.read(run/'request.json')
    for path, digest in request['sources'].items():
        if c.sha(path) != digest: raise RuntimeError(f'Source changed: {path}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['objective'] != 'capacity_diffusion_only': raise ValueError('Wrong head objective')
    if state['step'] != 50000 and not args.audit_only: raise ValueError('Require exactly 50K updates')
    digest = c.sha(args.checkpoint)
    adapter = Adapter('sit_small')
    assert fingerprint(adapter.model) == state['frozen']
    head = make(state['config']['variant']).cuda().eval()
    head.load_state_dict(state['ema'], strict=True)
    signature = fingerprint(head)
    bank = c.original.model_root('sit_small')/'quality_inputs'
    noise = np.load(bank/'noise.npy', mmap_mode='r'); labels = np.load(bank/'labels.npy')
    assert len(noise) == len(labels) == 5000
    assert np.all(np.bincount(labels, minlength=100) == 50)
    example = torch.from_numpy(np.array(noise[:8])).cuda()
    example_labels = torch.from_numpy(labels[:8].copy()).long().cuda()
    sample = for_adapter(adapter, head, example, example_labels, args.coefficient)
    with torch.no_grad():
        actual = sample(example, example_labels)
        reference, counts = integrate(adapter, head, 'real', args.coefficient, example, example_labels)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if rank == 0:
        c.atomic(run/'parity.json', dict(passed=True, original_sampler_endpoint_exact=True,
            checkpoint_sha256=digest, coefficient=args.coefficient, reference_counts=counts,
            noise_sha256=c.sha(bank/'noise.npy'), labels_sha256=c.sha(bank/'labels.npy')))
    if args.audit_only:
        c.barrier()
        if dist.is_initialized(): dist.destroy_process_group()
        return
    for start in tuple(range(0, 5000, 8))[rank::world]:
        if (run/'STOP_AFTER_CURRENT').exists(): raise RuntimeError('Stopped by marker')
        z = torch.from_numpy(np.array(noise[start:start+8])).cuda()
        y = torch.from_numpy(labels[start:start+8].copy()).long().cuda()
        begin = time.perf_counter()
        with torch.no_grad():
            result = sample(z, y)
            if not torch.isfinite(result).all(): raise FloatingPointError(f'Invalid endpoint {start}')
            pixels = adapter.pixels(result)
        torch.cuda.synchronize()
        path = target/'batches'/f'{start:05d}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez(f, arr_0=pixels, labels=y.cpu().numpy(), start=start)
        path.with_suffix('.tmp').replace(path)
        c.atomic(path.with_suffix('.json'), dict(start=start, samples=8, sha256=c.sha(path),
            noise_sha256=c.original.array_sha(noise[start:start+8]), checkpoint_sha256=digest,
            coefficient=args.coefficient, seconds=time.perf_counter()-begin))
        if start//8 % 10 == rank % 10:
            c.atomic(run/f'progress_rank{rank}.json', dict(start=start, samples=5000, updated_utc=c.now()))
        adapter.values.clear()
    c.barrier()
    assert fingerprint(adapter.model) == state['frozen'] and fingerprint(head) == signature
    if rank == 0:
        images = []; records = []
        for start in range(0, 5000, 8):
            path = target/'batches'/f'{start:05d}.npz'; record = c.read(path.with_suffix('.json'))
            assert c.sha(path) == record['sha256'] and record['checkpoint_sha256'] == digest
            assert record['coefficient'] == args.coefficient and record['samples'] == 8
            assert record['noise_sha256'] == c.original.array_sha(noise[start:start+8])
            with np.load(path) as value:
                assert int(value['start']) == start and value['arr_0'].shape == (8,256,256,3)
                np.testing.assert_array_equal(value['labels'], labels[start:start+8])
                images.append(value['arr_0'])
            records.append(record)
        path = target/'samples.npz'
        with path.with_suffix('.tmp').open('wb') as f: np.savez(f, arr_0=np.concatenate(images))
        path.with_suffix('.tmp').replace(path)
        c.atomic(target/'summary.json', dict(complete=True, valid=True, n=5000,
            samples_path=str(path), samples_sha256=c.sha(path), records=records,
            checkpoint=str(args.checkpoint), checkpoint_sha256=digest, weights='ema',
            coefficient=args.coefficient, architecture=state['config']['architecture'],
            noise_sha256=c.sha(bank/'noise.npy'), labels_sha256=c.sha(bank/'labels.npy'),
            strong_unchanged=True, head_unchanged=True, solver='64-step Heun, shared left-step amount',
            formula='S+a*f(t)*(S-W); f=6/7 before .25, 1 before .5, 0 afterwards', batch=8))
        del images
        torch.cuda.empty_cache()
        command = [c.PYTHON, '-u', '-m', 'experiments.adversarial_weak_training_20260915.score',
                   '--stage', str(target), '--shared-gpu', os.environ['CUDA_VISIBLE_DEVICES']]
        with (target/'score_holder.log').open('w') as log:
            subprocess.run(command, cwd=c.WORK, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = c.read(target/'metrics.json')
        c.atomic(run/'complete.json', dict(complete=True, samples=5000, fid=metrics['fid'],
            coefficient=args.coefficient, updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', choices=('sit_small',), default='sit_small')
    p.add_argument('--coefficient', type=float, required=True)
    p.add_argument('--audit-only', action='store_true')
    main(p.parse_args())

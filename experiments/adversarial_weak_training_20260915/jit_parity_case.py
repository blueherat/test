"""Isolate the reproducible JiT forward-mode discrepancy, on a leased idle GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path


def main(args):
    from experiments.weak_reference_loss_20260914 import idle
    lease = Path('/tmp', f'eqvae_idle_{args.gpu}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert idle.eligible(next(r for r in idle.gpu_snapshot() if r['uuid'] == args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
    from experiments.guidance_dynamic_50k_20260915.sampling import integrate
    from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
    from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import ordinary_rollout
    if args.case in ('feature', 'all'):
        from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    if args.case == 'all':
        from . import common as c
        c.source_receipts()
    if args.case in ('seed', 'all'):
        torch.manual_seed(2026091507)
    adapter = Adapter('jit')
    head = adapter.make_head()
    fixture = '/home/zhoushunyu/data/eqvae/experiments/jit_readout_transfer_20260913/training/head.pt'
    head.load_state_dict(torch.load(fixture, map_location='cpu', weights_only=False)['ema']['mlp'])
    head.eval().requires_grad_(True)
    if args.case == 'all':
        fingerprint(head)
    parameters = tuple(head.parameters())
    generator = torch.Generator(device='cuda').manual_seed(2026091507)
    noise = torch.randn((4, 3, 256, 256), generator=generator, device='cuda')
    labels = torch.arange(4, device='cuda', dtype=torch.long)
    steps = GuidanceSteps(adapter, head, labels, .3, method='real')
    torch.cuda.synchronize()
    deployed, counts = integrate(adapter, head, 'real', .3, noise, labels)
    torch.cuda.synchronize()
    with torch.no_grad():
        plain = ordinary_rollout(steps, len(steps), noise)
    result = dict(case=args.case, max_error=float((deployed - plain).abs().max()),
        relative_error=float((deployed - plain).norm() / deployed.norm()),
        source_file=adapter.jig.model_jit.__file__,
        strong_fingerprint=fingerprint(adapter.model), head_fingerprint=fingerprint(head),
        precision=torch.get_float32_matmul_precision(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        counts=counts, no_quality_claim=True, training_updates=0)
    # Paired local checks follow the reproduction, so cannot warm it up first.
    rows = []
    state = noise
    for index in range(100):
        with torch.inference_mode():
            reference = steps(index, state)
        with torch.no_grad():
            actual = steps(index, state)
        error = float((reference - actual).abs().max())
        if error:
            rows.append(dict(index=index, max_error=error))
        state = reference.clone()
    result['same_input_step_differences'] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('none', 'seed', 'feature', 'all'), required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

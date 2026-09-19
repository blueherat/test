"""JiT endpoint-gradient readiness check; the historical 3K head is only a fixture.

This does not train, select a model, evaluate FID, or substitute a 3K head for
the full-data 50K baseline required for a formal cross-model experiment.
"""

import argparse
import fcntl
import gc
import os
from pathlib import Path
import time


def main(args):
    # Must precede torch-fidelity/torchvision importing torch._dynamo.config.
    # jit_internal_guidance's later setdefault is otherwise too late.
    os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
    from experiments.weak_reference_loss_20260914 import idle
    lease = Path('/tmp', f'eqvae_idle_{args.gpu}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = {row['uuid']: row for row in idle.gpu_snapshot()}
    assert args.gpu in snapshot and idle.eligible(snapshot[args.gpu])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
    from experiments.guidance_dynamic_50k_20260915.sampling import integrate
    from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
    from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import ordinary_rollout, recomputed_rollout
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    from . import common as c

    args.output.mkdir(parents=True, exist_ok=True)
    fixture = c.original.EXPS / 'jit_readout_transfer_20260913/training/head.pt'
    result = dict(model='jit', kind='engineering_preflight_only', updates=0,
        fixture=str(fixture), fixture_sha256=c.sha(fixture), fixture_steps=3000,
        formal_50k_baseline=False, seed=args.seed, batch=args.batch,
        coefficient=.3, gpu=snapshot[args.gpu], sources=c.source_receipts())

    def progress(stage, **values):
        result.update(values)
        c.atomic(args.output / 'progress.json', dict(result, stage=stage, updated_utc=c.now()))
        print(stage, values, flush=True)

    torch.manual_seed(args.seed)
    adapter = Adapter('jit')
    import torch._dynamo.config as dynamo_config
    assert dynamo_config.disable, 'The original JiT baseline uses eager execution'
    result['torch_compile_disabled_before_dependency_import'] = True
    head = adapter.make_head()
    state = torch.load(fixture, map_location='cpu', weights_only=False)
    assert state['steps'] == 3000
    head.load_state_dict(state['ema']['mlp'])
    head.eval().requires_grad_(True)
    assert not any(p.requires_grad for p in adapter.model.parameters())
    initial = fingerprint(head)
    parameters = tuple(head.parameters())
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    noise = torch.randn((args.batch, 3, 256, 256), generator=generator, device='cuda')
    labels = torch.arange(args.batch, device='cuda', dtype=torch.long)
    steps = GuidanceSteps(adapter, head, labels, .3, method='real')
    torch.cuda.synchronize()
    started = time.perf_counter()
    deployed, counts = integrate(adapter, head, 'real', .3, noise, labels)
    torch.cuda.synchronize()
    deployed_seconds = time.perf_counter() - started
    with torch.no_grad():
        actual = ordinary_rollout(steps, len(steps), noise)
    torch.testing.assert_close(deployed, actual, atol=0, rtol=0)
    progress('forward_parity_passed', forward_bitwise_equal=True,
        deployed_counts=counts, deployed_seconds=deployed_seconds)

    with torch.no_grad():
        boundary = ordinary_rollout(steps, 48, noise)
    crossing = GuidanceSteps(adapter, head, labels, .3, method='real', start=48, stop=52)
    expected = ordinary_rollout(crossing, len(crossing), boundary)
    reference_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    recomputed = recomputed_rollout(crossing, len(crossing), boundary, parameters)
    gradients = torch.autograd.grad(recomputed.square().mean(), parameters)
    torch.testing.assert_close(recomputed, expected, atol=0, rtol=0)
    reference_vector = torch.cat([p.flatten() for p in reference_gradients])
    actual_vector = torch.cat([p.flatten() for p in gradients])
    relative = float((reference_vector - actual_vector).norm() / reference_vector.norm())
    cosine = float(torch.nn.functional.cosine_similarity(reference_vector, actual_vector, dim=0))
    # Report BF16 numerical agreement, not an exact arithmetic identity claim.
    assert relative < .02 and cosine > .999, (relative, cosine)
    progress('cross_switch_gradient_checked', relative_error=relative, gradient_cosine=cosine)
    del expected, recomputed, reference_gradients, gradients, reference_vector, actual_vector
    adapter.values.clear()
    gc.collect()
    torch.cuda.empty_cache()

    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    critic = torch.nn.Linear(2048, 1).cuda().eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    endpoint = recomputed_rollout(steps, len(steps), noise, parameters)
    loss = torch.nn.functional.softplus(critic(feature((endpoint.float() + 1.) / 2.))).mean()
    gradients = torch.autograd.grad(loss, parameters)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    norm = float(torch.cat([p.flatten() for p in gradients]).norm())
    assert 0 < norm < float('inf')
    assert fingerprint(head) == initial
    assert not any(p.grad is not None for p in adapter.model.parameters())
    progress('complete', complete=True, image_feedback_gradient_norm=norm,
        full_sampler_image_backward_seconds=duration,
        backward_training_over_inference=duration / deployed_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
        head_unchanged=True, critic_fitted=False, no_quality_claim=True)
    c.atomic(args.output / 'result.json', result)
    lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026091507)
    main(parser.parse_args())

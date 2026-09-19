"""Bounded, leased-GPU checks on the actual SiT model, without parameter updates.

Checks deployed forward parity, full versus recomputed gradients across the
guidance switch, terminal image-feedback gradients, cost, and suffix inversion.
No discriminator fitting, weak-head training, or FID improvement is claimed.
"""

import argparse
import fcntl
import gc
import os
from pathlib import Path
import time


def main(args):
    # Acquire the same advisory lease as the active queue, before CUDA allocation.
    from experiments.weak_reference_loss_20260914 import idle
    lease = Path("/tmp", f"eqvae_idle_{args.gpu}.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = {row["uuid"]: row for row in idle.gpu_snapshot()}
    if args.gpu not in rows or not idle.eligible(rows[args.gpu]):
        raise RuntimeError("Requested GPU is no longer idle; preflight did not start")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import torch
    from experiments.guidance_dynamic_50k_20260915 import config as k, sampling
    from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
    from experiments.guidance_dynamic_50k_20260915.data import RealDataset
    from experiments import train_imagenet100_sit_flow as data_base
    from .discrete_adjoint import ordinary_rollout, recomputed_rollout
    from .sampler import GuidanceSteps, ReverseFrozenSuffix
    from .objectives import guided_generator_loss

    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(kind="real_model_engineering_preflight_not_quality_training",
                  gpu=rows[args.gpu], pid=os.getpid(), model="sit_small", batch=args.batch,
                  head=args.head, coefficient=args.coefficient, seed=2026091504,
                  sources={str(path.resolve()): k.sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                  checkpoint=str(k.model_root("sit_small") / "training" / args.head / "head.pt"))
    result["checkpoint_sha256"] = k.sha(result["checkpoint"])

    def progress(stage, **values):
        result.update(values)
        k.atomic(args.output / "progress.json", dict(result, stage=stage))
        print(stage, values, flush=True)

    torch.manual_seed(result["seed"])
    adapter = Adapter("sit_small")
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    head = adapter.loaded_head(args.head).requires_grad_(True)
    parameters = tuple(head.parameters())
    initial_fingerprint = fingerprint(head)
    generator = torch.Generator(device="cuda").manual_seed(result["seed"])
    noise = torch.randn((args.batch, 4, 32, 32), generator=generator, device="cuda")
    labels = torch.arange(args.batch, device="cuda", dtype=torch.long)
    steps = GuidanceSteps(adapter, head, labels, args.coefficient, args.head)

    torch.cuda.synchronize()
    before = time.perf_counter()
    deployed, counts = sampling.integrate(adapter, head, args.head, args.coefficient, noise, labels)
    torch.cuda.synchronize()
    deployed_seconds = time.perf_counter() - before
    with torch.no_grad():
        differentiable_forward = ordinary_rollout(steps, len(steps), noise)
    torch.testing.assert_close(deployed, differentiable_forward, rtol=0., atol=0.)
    progress("deployed_forward_parity_passed", forward_bitwise_equal=True,
             deployed_counts=counts, deployed_seconds=deployed_seconds)

    with torch.no_grad():
        boundary_input = ordinary_rollout(steps, 30, noise)
    across_switch = GuidanceSteps(adapter, head, labels, args.coefficient, args.head, start=30, stop=34)
    reference = ordinary_rollout(across_switch, len(across_switch), boundary_input)
    expected = torch.autograd.grad(reference.square().mean(), parameters)
    recomputed = recomputed_rollout(across_switch, len(across_switch), boundary_input, parameters)
    actual = torch.autograd.grad(recomputed.square().mean(), parameters)
    expected_vector = torch.cat([value.flatten() for value in expected])
    actual_vector = torch.cat([value.flatten() for value in actual])
    relative_error = float((actual_vector - expected_vector).norm() / expected_vector.norm())
    torch.testing.assert_close(recomputed, reference, rtol=0., atol=0.)
    if relative_error > 2e-4:
        raise AssertionError(f"Recomputed gradient relative error {relative_error}")
    progress("actual_sit_gradient_parity_passed", cross_switch_head_gradient_relative_error=relative_error,
             cross_switch_head_gradient_norm=float(actual_vector.norm()))
    del reference, recomputed, expected, actual, expected_vector, actual_vector
    adapter.values.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # A fixed random four-way readout of actual pretrained image features is
    # deliberately only an image-to-head gradient diagnostic, not a trained D.
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    critic = torch.nn.Linear(2048, 4).cuda().eval().requires_grad_(False)
    adapter.rt.vae.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = time.perf_counter()
    generated = recomputed_rollout(steps, len(steps), noise, parameters)
    decoded = adapter.rt.vae.decode(generated.float() / data_base.SD_VAE_SCALING_FACTOR).sample
    images = (decoded + 1.) / 2.  # Continuous image path: no uint8 or hard clamp.
    loss = guided_generator_loss(critic(feature(images)))
    gradients = torch.autograd.grad(loss, parameters)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - before
    gradient_norm = float(torch.cat([value.flatten() for value in gradients]).norm())
    if not 0 < gradient_norm < float("inf"):
        raise AssertionError(f"Invalid endpoint image-feedback gradient {gradient_norm}")
    progress("full_sampler_image_feedback_passed",
             image_feedback_gradient_norm=gradient_norm,
             full_rollout_image_loss_backward_seconds=elapsed,
             image_loss_with_backward_over_deployed_forward=elapsed / deployed_seconds,
             peak_allocated_gib=torch.cuda.max_memory_allocated() / (1024 ** 3),
             training_updates=0, critic_fitted=False)
    del generated, decoded, images, loss, gradients, feature, critic
    adapter.values.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # Common frozen-tail pullback feasibility on real training examples and
    # current guided outputs, with endpoint-level cycle error explicitly shown.
    dataset = RealDataset("sit_small", "train")
    items = [dataset[index] for index in range(args.batch)]
    moments = torch.stack([item[0] for item in items]).cuda()
    real_labels = torch.tensor([item[1] for item in items], device="cuda", dtype=torch.long)
    real = data_base.sample_sdvae_posterior(moments, torch.randn(noise.shape, device="cuda", generator=generator))
    cycles = {}
    for name, endpoint, targets in (("real_training_latents", real, real_labels),
                                    ("guided_endpoints", differentiable_forward, labels)):
        reverse = ReverseFrozenSuffix(adapter, targets)
        suffix = GuidanceSteps(adapter, head, targets, args.coefficient, args.head, start=32)
        with torch.no_grad():
            pulled_back = ordinary_rollout(reverse, len(reverse), endpoint)
            recovered = ordinary_rollout(suffix, len(suffix), pulled_back)
        difference = recovered.double() - endpoint.double()
        cycles[name] = dict(
            absolute_rms=float(difference.square().mean().sqrt()),
            relative_l2=float(difference.norm() / endpoint.double().norm()),
            endpoint_max_abs_error=float(difference.abs().max()),
            finite=bool(torch.isfinite(recovered).all() and torch.isfinite(pulled_back).all()),
        )
    assert fingerprint(head) == initial_fingerprint
    assert not any(parameter.grad is not None for parameter in adapter.model.parameters())
    progress("complete", complete=True, suffix_backward_heun_cycle=cycles,
             head_parameters_unchanged=True, strong_parameters_received_no_gradient=True,
             inversion_is_exact=False)
    k.atomic(args.output / "result.json", result)
    lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--head", default="guided_weak")
    parser.add_argument("--coefficient", type=float, default=1.05)
    main(parser.parse_args())

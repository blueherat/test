#!/usr/bin/env python3
"""End-to-end pMF/FD/AdvFD gradient smoke test.

This deliberately projects Inception features to keep the first hardware test
small. It validates the computational graph only and is not a numerical paper
reproduction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.advfd_cleanroom.core import (
    calibrate_features,
    frechet_from_features,
    normalized_frechet_loss,
)
from experiments.advfd_cleanroom.feature_extractors import (
    DifferentiableInception2048,
    generator_output_to_unit_interval,
)
from experiments.advfd_cleanroom.generators import (
    load_pmf_b16,
    pmf_one_step,
    seeded_pmf_noise,
)
from experiments.raev2_training_core import DeterministicImageNetPacked


def fixed_projection(
    input_dim: int, output_dim: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    matrix, _ = torch.linalg.qr(matrix, mode="reduced")
    return matrix.to(device)


def gradient_norm(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        ).sqrt()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pmf-repo",
        type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/pMF"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth"
        ),
    )
    parser.add_argument(
        "--packed-data",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_smoke.json"
        ),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    model = load_pmf_b16(
        repo=args.pmf_repo, checkpoint=args.checkpoint, device=device
    ).train()
    static_encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    adaptive_encoder = DifferentiableInception2048(trainable=True).to(device).eval()
    adaptive_encoder.load_state_dict(static_encoder.state_dict())
    projection = fixed_projection(
        2048, args.projection_dim, seed=9917, device=device
    )

    dataset = DeterministicImageNetPacked(
        args.packed_data,
        split="train",
        image_size=256,
        horizontal_flip=True,
        augmentation_seed=81001,
    )
    real_images = torch.stack(
        [dataset[index][0] for index in range(args.batch_size)]
    ).to(device)
    labels = torch.arange(args.batch_size, device=device, dtype=torch.long) % 1000
    noise = seeded_pmf_noise(
        model,
        batch_size=args.batch_size,
        seed=81233,
        device=device,
    )

    generator_optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0
    )
    critic_optimizer = torch.optim.AdamW(
        adaptive_encoder.parameters(),
        lr=2e-6,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    generator_optimizer.zero_grad(set_to_none=True)
    adaptive_encoder.requires_grad_(False)
    generated = pmf_one_step(model, noise, labels)
    generated_unit = generator_output_to_unit_interval(generated)
    with torch.no_grad():
        real_static = static_encoder(real_images) @ projection
        real_adaptive = adaptive_encoder(real_images) @ projection
    generated_static = static_encoder(generated_unit) @ projection
    generated_adaptive = adaptive_encoder(generated_unit) @ projection
    static = frechet_from_features(real_static, generated_static)
    adv_real, adv_fake = calibrate_features(
        real_adaptive,
        generated_adaptive,
        mode="real",
        epsilon=1e-3,
    )
    adaptive = frechet_from_features(adv_real, adv_fake)
    generator_loss = normalized_frechet_loss(static) + 0.05 * normalized_frechet_loss(
        adaptive
    )
    generator_loss.backward()
    generator_grad_norm = gradient_norm(model)
    generator_optimizer.step()

    critic_optimizer.zero_grad(set_to_none=True)
    adaptive_encoder.requires_grad_(True)
    with torch.no_grad():
        generated_detached = generator_output_to_unit_interval(
            pmf_one_step(model, noise, labels)
        )
    real_critic = (adaptive_encoder(real_images) @ projection).detach()
    fake_critic = adaptive_encoder(generated_detached) @ projection
    critic_real, critic_fake = calibrate_features(
        real_critic,
        fake_critic,
        mode="real",
        epsilon=1e-3,
    )
    critic_fd = frechet_from_features(critic_real, critic_fake).total
    (-critic_fd).backward()
    critic_grad_norm = gradient_norm(adaptive_encoder)
    clipped_norm = torch.nn.utils.clip_grad_norm_(adaptive_encoder.parameters(), 1.0)
    critic_optimizer.step()

    result = {
        "protocol": "paper_only_pmf_projected_inception_gradient_smoke_v1",
        "batch_size": args.batch_size,
        "projection_dim": args.projection_dim,
        "static_fd": float(static.total.detach()),
        "adaptive_fd_g_step": float(adaptive.total.detach()),
        "generator_loss": float(generator_loss.detach()),
        "generator_grad_norm": generator_grad_norm,
        "adaptive_fd_d_step": float(critic_fd.detach()),
        "critic_grad_norm": critic_grad_norm,
        "critic_preclip_norm": float(clipped_norm),
        "generated_min": float(generated.detach().min()),
        "generated_max": float(generated.detach().max()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "paper_reproduction_metric": False,
    }
    if not all(
        torch.isfinite(torch.tensor(value))
        for key, value in result.items()
        if isinstance(value, float)
    ):
        raise FloatingPointError(result)
    if generator_grad_norm <= 0.0 or critic_grad_norm <= 0.0:
        raise RuntimeError(f"Broken gradient path: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

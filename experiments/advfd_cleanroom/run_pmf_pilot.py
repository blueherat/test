#!/usr/bin/env python3
"""Paper-only pMF-B pilot for FD-Loss and AdvFD.

This is intentionally a scaled experiment: Inception features are projected to a
fixed low-dimensional subspace and the paper schedule can be compressed. It is
used to validate the population-statistics and alternating-optimization
mechanism before a full numerical reproduction. No AdvFD repository code is
imported or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from experiments.advfd_cleanroom.core import (
    EMAMomentTracker,
    Moments,
    StreamingMomentAccumulator,
    batch_moments,
    calibrate_moments,
    fit_calibration_from_moments,
    frechet_from_moments,
    moments_from_mean_and_covariance,
    normalized_frechet_loss,
    project_moments,
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


Variant = Literal["base", "static", "raw", "real"]


@dataclass(frozen=True)
class PilotConfig:
    stage: str
    variant: str
    steps: int
    batch_size: int
    feature_dim: int
    adaptive_feature_dim: int
    warmstart_samples: int
    eval_samples: int
    generator_lr: float
    critic_lr: float
    static_ema: float
    adaptive_ema: float
    adaptive_weight: float
    adaptive_start: int
    adaptive_warmup: int
    critic_frequency: int
    lr_warmup: int
    schedule_total_steps: int
    seed: int
    amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("warmstart", "train", "evaluate"), required=True)
    parser.add_argument("--variant", choices=("base", "static", "raw", "real"), default="base")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--warmstart-file", type=Path, default=None,
        help="Defaults to OUTPUT_ROOT/warmstart.pt",
    )
    parser.add_argument(
        "--pmf-repo", type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/pMF"),
    )
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--evaluation-tag",
        default=None,
        help="Optional filename tag so repeated evaluations do not overwrite prior results.",
    )
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/JiT/fid_stats/jit_in256_stats.npz"),
    )
    parser.add_argument(
        "--packed-data", type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument(
        "--adaptive-feature-dim",
        type=int,
        default=None,
        help="Optional adaptive-critic projection dimension; static FD keeps feature-dim.",
    )
    parser.add_argument("--warmstart-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument(
        "--balanced-eval-labels",
        action="store_true",
        help="Match pMF FID evaluation by emitting an equal count per class.",
    )
    parser.add_argument(
        "--per-sample-eval-noise",
        action="store_true",
        help="Match pMF evaluation by deriving each sample from its own CPU RNG.",
    )
    parser.add_argument("--eval-noise-seed", type=int, default=42)
    parser.add_argument(
        "--quantize-eval-images",
        action="store_true",
        help="Match pMF FID by rounding generated [0,1] images to uint8 levels.",
    )
    parser.add_argument(
        "--adaptive-eval-samples",
        type=int,
        default=None,
        help="Strict held-out sample count for adaptive-critic diagnostics; defaults to eval-samples.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--generator-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=2e-6)
    parser.add_argument("--static-ema", type=float, default=0.999)
    parser.add_argument("--adaptive-ema", type=float, default=0.99)
    parser.add_argument("--adaptive-weight", type=float, default=0.05)
    parser.add_argument("--adaptive-start", type=int, default=10)
    parser.add_argument("--adaptive-warmup", type=int, default=40)
    parser.add_argument("--critic-frequency", type=int, default=2)
    parser.add_argument("--lr-warmup", type=int, default=10)
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        default=None,
        help="Cosine horizon; lets a short run follow a longer paper schedule prefix.",
    )
    parser.add_argument("--seed", type=int, default=260823)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_artifact_names(tag: str | None) -> tuple[str, str]:
    if tag is None:
        return "evaluation.json", "samples_grid.png"
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in tag):
        raise ValueError("evaluation tag may contain only letters, digits, '-' and '_'")
    return f"evaluation_{tag}.json", f"samples_grid_{tag}.png"


def resolve_adaptive_eval_samples(eval_samples: int, requested: int | None) -> int:
    count = eval_samples if requested is None else requested
    if count <= 0 or count > eval_samples:
        raise ValueError("adaptive-eval-samples must be in [1, eval-samples]")
    return count


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fixed_projection(
    input_dim: int, output_dim: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("feature_dim cannot exceed Inception dimension")
    if output_dim == input_dim:
        return torch.eye(input_dim, device=device, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    matrix, _ = torch.linalg.qr(matrix, mode="reduced")
    return matrix.to(device=device, dtype=torch.float32)


def resolve_adaptive_feature_dim(feature_dim: int, requested: int | None) -> int:
    dimension = feature_dim if requested is None else requested
    if dimension <= 0 or dimension > 2048:
        raise ValueError("adaptive-feature-dim must be in [1, 2048]")
    if dimension != feature_dim and feature_dim != 2048:
        raise ValueError(
            "separate adaptive-feature-dim requires a full 2048-dimensional warm-start"
        )
    return dimension


def balanced_evaluation_labels(
    *, start: int, count: int, total: int, num_classes: int, device: torch.device
) -> torch.Tensor:
    if total <= 0 or total % num_classes != 0:
        raise ValueError("balanced evaluation requires total divisible by num_classes")
    if start < 0 or count < 0 or start + count > total:
        raise ValueError("balanced evaluation label slice is out of range")
    samples_per_class = total // num_classes
    indices = torch.arange(start, start + count, device=device)
    return torch.div(indices, samples_per_class, rounding_mode="floor").long()


def per_sample_evaluation_noise(
    *,
    start: int,
    count: int,
    sample_shape: tuple[int, ...],
    initial_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if start < 0 or count < 0:
        raise ValueError("evaluation noise slice must be nonnegative")
    samples = []
    for sample_index in range(start, start + count):
        generator = torch.Generator("cpu").manual_seed(
            int(sample_index ^ initial_seed) % (1 << 32)
        )
        samples.append(torch.randn(sample_shape, generator=generator, dtype=dtype))
    if not samples:
        return torch.empty((0, *sample_shape), device=device, dtype=dtype)
    return torch.stack(samples).to(device=device)


def quantize_unit_images(images: torch.Tensor) -> torch.Tensor:
    return images.clamp(0.0, 1.0).mul(255.0).round().div(255.0)


def moments_payload(moments: Moments) -> dict[str, torch.Tensor]:
    return {
        "mean": moments.mean.detach().cpu(),
        "second": moments.second.detach().cpu(),
        "covariance": moments.covariance.detach().cpu(),
    }


def moments_from_payload(payload: dict[str, torch.Tensor], device: torch.device) -> Moments:
    mean = payload["mean"].to(device=device, dtype=torch.float32)
    covariance = payload["covariance"].to(device=device, dtype=torch.float32)
    return moments_from_mean_and_covariance(mean, covariance)


def load_real_inception_moments(path: Path, device: torch.device) -> Moments:
    arrays = np.load(path)
    if set(arrays.files) != {"mu", "sigma"}:
        raise ValueError(f"Unexpected Inception stats keys: {arrays.files}")
    mean = torch.from_numpy(np.asarray(arrays["mu"])).to(device=device, dtype=torch.float32)
    covariance = torch.from_numpy(np.asarray(arrays["sigma"])).to(
        device=device, dtype=torch.float32
    )
    return moments_from_mean_and_covariance(mean, covariance)


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def generated_images(
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    amp: bool,
) -> tuple[torch.Tensor, float]:
    with autocast_context(noise.device, amp):
        raw = pmf_one_step(model, noise, labels)
    outside = ((raw < -1.0) | (raw > 1.0)).float().mean().item()
    return generator_output_to_unit_interval(raw.float()), float(outside)


def feature_forward(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
    *,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(images.device, amp):
        full = encoder(images)
    full = full.float()
    return full, full @ projection


def warmstart(args: argparse.Namespace, device: torch.device) -> None:
    output = args.warmstart_file or args.output_root / "warmstart.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_pmf_b16(repo=args.pmf_repo, checkpoint=args.base_checkpoint, device=device)
    model.eval().requires_grad_(False)
    encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    projection = fixed_projection(2048, args.feature_dim, seed=9917, device=device)
    real_full = load_real_inception_moments(args.real_stats, device)
    real_projected = project_moments(real_full, projection)

    accumulator = StreamingMomentAccumulator(dtype=torch.float32)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    label_generator = torch.Generator(device=device).manual_seed(args.seed + 103)
    remaining = args.warmstart_samples
    clamp_weighted = 0.0
    started = time.perf_counter()
    while remaining:
        batch = min(args.batch_size, remaining)
        noise = torch.randn(
            batch, 3, 256, 256,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        ) * float(model.noise_scale)
        labels = torch.randint(
            0, 1000, (batch,), generator=label_generator, device=device
        )
        with torch.inference_mode():
            images, clamp_fraction = generated_images(model, noise, labels, amp=not args.no_amp)
            _, projected = feature_forward(
                encoder, images, projection, amp=not args.no_amp
            )
        accumulator.update(projected)
        clamp_weighted += clamp_fraction * batch
        remaining -= batch
        if accumulator.count % max(256, args.batch_size) == 0 or remaining == 0:
            print(
                f"warmstart {accumulator.count}/{args.warmstart_samples} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    payload = {
        "protocol": "advfd_paper_only_pmf_projected_pilot_v1",
        "paper_reproduction_metric": False,
        "official_advfd_code_consulted_for_this_implementation": False,
        "projection": projection.cpu(),
        "real_full": moments_payload(real_full),
        "real_projected": moments_payload(real_projected),
        "base_fake_projected": moments_payload(accumulator.moments()),
        "warmstart_samples": accumulator.count,
        "base_clamp_fraction": clamp_weighted / accumulator.count,
        "feature_dim": args.feature_dim,
        "projection_seed": 9917,
        "seed": args.seed,
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "real_stats_sha256": sha256(args.real_stats),
        "git_head": git_head(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(payload, output)
    print(json.dumps({k: v for k, v in payload.items() if not isinstance(v, dict) and not torch.is_tensor(v)}, indent=2))
    print(f"saved {output}", flush=True)


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


def gradient_norm(module: torch.nn.Module) -> float:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def scheduled_lr(base_lr: float, step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def adaptive_scale(args: argparse.Namespace, step: int) -> float:
    if step < args.adaptive_start:
        return 0.0
    if args.adaptive_warmup <= 0:
        return args.adaptive_weight
    fraction = min((step - args.adaptive_start + 1) / args.adaptive_warmup, 1.0)
    return args.adaptive_weight * fraction


def adaptive_components(
    real: Moments,
    fake: Moments,
    *,
    variant: Variant,
) -> tuple[Any, Moments, Moments]:
    mode = "none" if variant == "raw" else "real"
    calibration = fit_calibration_from_moments(
        real, fake, mode=mode, epsilon=1e-3, detach_statistics=True
    )
    calibrated_real = calibrate_moments(real, calibration)
    calibrated_fake = calibrate_moments(fake, calibration)
    return (
        frechet_from_moments(calibrated_real, calibrated_fake),
        calibrated_real,
        calibrated_fake,
    )


def effective_rank(covariance: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.mT)).clamp_min(0)
    total = eigenvalues.sum()
    denominator = eigenvalues.square().sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return float((total.square() / denominator).detach())


def covariance_diagnostics(covariance: torch.Tensor) -> dict[str, float | int]:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.mT))
    return {
        "minimum_eigenvalue": float(eigenvalues.min().detach()),
        "maximum_eigenvalue": float(eigenvalues.max().detach()),
        "negative_eigenvalues": int((eigenvalues < 0).sum().detach()),
        "trace": float(eigenvalues.sum().detach()),
    }


def component_rms(moments: Moments) -> float:
    energy = moments.mean.square().sum() + torch.trace(moments.covariance)
    return float((energy / moments.mean.numel()).clamp_min(0).sqrt().detach())


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_dataset(args: argparse.Namespace, *, flip: bool) -> DeterministicImageNetPacked:
    return DeterministicImageNetPacked(
        args.packed_data,
        split="train",
        image_size=256,
        horizontal_flip=flip,
        augmentation_seed=args.seed + 211,
    )


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    adaptive_encoder: torch.nn.Module | None,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer | None,
    trackers: dict[str, EMAMomentTracker],
    step: int,
    config: PilotConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "adaptive_encoder": None if adaptive_encoder is None else adaptive_encoder.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "critic_optimizer": None if critic_optimizer is None else critic_optimizer.state_dict(),
            "trackers": {name: tracker.state_dict() for name, tracker in trackers.items()},
            "step": step,
            "config": asdict(config),
            "paper_reproduction_metric": False,
        },
        path,
    )


def train(args: argparse.Namespace, device: torch.device) -> Path:
    if args.variant == "base":
        raise ValueError("base is evaluation-only")
    warm_path = args.warmstart_file or args.output_root / "warmstart.pt"
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    if int(warm["feature_dim"]) != args.feature_dim:
        raise ValueError("Warm-start feature dimension does not match")
    projection = warm["projection"].to(device=device, dtype=torch.float32)
    real_static = moments_from_payload(warm["real_projected"], device)
    base_fake = moments_from_payload(warm["base_fake_projected"], device)
    adaptive_feature_dim = resolve_adaptive_feature_dim(
        args.feature_dim, args.adaptive_feature_dim
    )
    adaptive_projection = projection
    real_adaptive_init = real_static
    base_fake_adaptive = base_fake
    if adaptive_feature_dim != args.feature_dim:
        adaptive_projection = fixed_projection(
            2048, adaptive_feature_dim, seed=19231, device=device
        )
        real_adaptive_init = project_moments(
            moments_from_payload(warm["real_full"], device), adaptive_projection
        )
        base_fake_adaptive = project_moments(base_fake, adaptive_projection)

    model = load_pmf_b16(repo=args.pmf_repo, checkpoint=args.base_checkpoint, device=device)
    model.train()
    static_encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    adaptive_encoder: torch.nn.Module | None = None
    if args.variant in ("raw", "real"):
        adaptive_encoder = DifferentiableInception2048(trainable=True).to(device).eval()
        adaptive_encoder.load_state_dict(static_encoder.state_dict())
        set_trainable(adaptive_encoder, False)

    generator_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.generator_lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    critic_optimizer = None
    if adaptive_encoder is not None:
        critic_optimizer = torch.optim.AdamW(
            adaptive_encoder.parameters(),
            lr=args.critic_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

    trackers = {"static_fake": EMAMomentTracker(args.static_ema)}
    trackers["static_fake"].initialize_from_moments(base_fake)
    if adaptive_encoder is not None:
        trackers["adaptive_real"] = EMAMomentTracker(args.adaptive_ema)
        trackers["adaptive_fake"] = EMAMomentTracker(args.adaptive_ema)
        trackers["adaptive_real"].initialize_from_moments(real_adaptive_init)
        trackers["adaptive_fake"].initialize_from_moments(base_fake_adaptive)

    dataset = make_dataset(args, flip=True)
    training_indices = range(0, len(dataset) - args.eval_samples)
    loader_generator = torch.Generator().manual_seed(args.seed + 307)
    loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    batches = infinite_batches(loader)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 401)
    label_generator = torch.Generator(device=device).manual_seed(args.seed + 409)

    variant_dir = args.output_root / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = variant_dir / "train_metrics.csv"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to append to existing run: {metrics_path}")
    config = PilotConfig(
        stage="train",
        variant=args.variant,
        steps=args.steps,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        adaptive_feature_dim=adaptive_feature_dim,
        warmstart_samples=args.warmstart_samples,
        eval_samples=args.eval_samples,
        generator_lr=args.generator_lr,
        critic_lr=args.critic_lr,
        static_ema=args.static_ema,
        adaptive_ema=args.adaptive_ema,
        adaptive_weight=args.adaptive_weight,
        adaptive_start=args.adaptive_start,
        adaptive_warmup=args.adaptive_warmup,
        critic_frequency=args.critic_frequency,
        lr_warmup=args.lr_warmup,
        schedule_total_steps=args.schedule_total_steps or args.steps,
        seed=args.seed,
        amp=not args.no_amp,
    )
    (variant_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "protocol": "advfd_paper_only_pmf_projected_pilot_v1",
                "paper_reproduction_metric": False,
                "official_advfd_code_consulted_for_this_implementation": False,
                "base_checkpoint_sha256": sha256(args.base_checkpoint),
                "warmstart_file": str(warm_path),
                "warmstart_sha256": sha256(warm_path),
                "git_head": git_head(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    last_log = started
    for step in range(args.steps):
        real_images, _, _ = next(batches)
        real_images = real_images.to(device=device, non_blocking=True)
        labels = torch.randint(
            0, 1000, (args.batch_size,), generator=label_generator, device=device
        )
        noise = torch.randn(
            args.batch_size, 3, 256, 256,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        ) * float(model.noise_scale)

        lr = scheduled_lr(
            args.generator_lr,
            step,
            args.schedule_total_steps or args.steps,
            args.lr_warmup,
        )
        for group in generator_optimizer.param_groups:
            group["lr"] = lr
        generator_optimizer.zero_grad(set_to_none=True)
        generated, clamp_fraction = generated_images(
            model, noise, labels, amp=not args.no_amp
        )
        _, static_features = feature_forward(
            static_encoder, generated, projection, amp=not args.no_amp
        )
        static_effective = trackers["static_fake"].preview(static_features)
        static_components = frechet_from_moments(real_static, static_effective)
        loss = normalized_frechet_loss(static_components)

        adaptive_components_g = None
        adaptive_real_g = adaptive_fake_g = None
        adv_lambda = (
            0.0 if adaptive_encoder is None else adaptive_scale(args, step)
        )
        if adaptive_encoder is not None:
            set_trainable(adaptive_encoder, False)
            with torch.no_grad():
                _, adaptive_real_features = feature_forward(
                    adaptive_encoder,
                    real_images,
                    adaptive_projection,
                    amp=not args.no_amp,
                )
            _, adaptive_fake_features = feature_forward(
                adaptive_encoder,
                generated,
                adaptive_projection,
                amp=not args.no_amp,
            )
            adaptive_real_g = trackers["adaptive_real"].preview(adaptive_real_features)
            adaptive_fake_g = trackers["adaptive_fake"].preview(adaptive_fake_features)
            adaptive_components_g, calibrated_real, calibrated_fake = adaptive_components(
                adaptive_real_g,
                adaptive_fake_g,
                variant=args.variant,
            )
            if adv_lambda:
                loss = loss + adv_lambda * normalized_frechet_loss(adaptive_components_g)

        loss.backward()
        generator_grad = gradient_norm(model)
        generator_optimizer.step()
        trackers["static_fake"].commit(static_effective)

        critic_fd = critic_norm = critic_preclip = float("nan")
        do_critic = (
            adaptive_encoder is not None
            and step >= args.adaptive_start
            and (step + 1) % args.critic_frequency == 0
        )
        if do_critic:
            assert critic_optimizer is not None
            critic_optimizer.zero_grad(set_to_none=True)
            set_trainable(adaptive_encoder, True)
            with torch.no_grad():
                post_images, _ = generated_images(
                    model, noise, labels, amp=not args.no_amp
                )
            _, d_real_features = feature_forward(
                adaptive_encoder,
                real_images,
                adaptive_projection,
                amp=not args.no_amp,
            )
            _, d_fake_features = feature_forward(
                adaptive_encoder,
                post_images.detach(),
                adaptive_projection,
                amp=not args.no_amp,
            )
            d_real = trackers["adaptive_real"].preview(d_real_features.detach())
            d_fake = trackers["adaptive_fake"].preview(d_fake_features)
            d_components, _, _ = adaptive_components(
                d_real, d_fake, variant=args.variant
            )
            critic_objective = normalized_frechet_loss(d_components)
            (-critic_objective).backward()
            critic_norm = gradient_norm(adaptive_encoder)
            clipped = torch.nn.utils.clip_grad_norm_(adaptive_encoder.parameters(), 1.0)
            critic_preclip = float(clipped)
            critic_optimizer.step()
            critic_fd = float(d_components.total.detach())
            trackers["adaptive_real"].commit(d_real)
            trackers["adaptive_fake"].commit(d_fake)
            set_trainable(adaptive_encoder, False)
        elif adaptive_encoder is not None:
            assert adaptive_real_g is not None and adaptive_fake_g is not None
            trackers["adaptive_real"].commit(adaptive_real_g)
            trackers["adaptive_fake"].commit(adaptive_fake_g)

        now = time.perf_counter()
        row: dict[str, Any] = {
            "step": step + 1,
            "generator_lr": lr,
            "loss": float(loss.detach()),
            "static_fd": float(static_components.total.detach()),
            "static_fd_mean": float(static_components.mean.detach()),
            "static_fd_covariance": float(static_components.covariance.detach()),
            "adaptive_weight": adv_lambda,
            "adaptive_fd": float("nan") if adaptive_components_g is None else float(adaptive_components_g.total.detach()),
            "adaptive_fd_mean": float("nan") if adaptive_components_g is None else float(adaptive_components_g.mean.detach()),
            "adaptive_fd_covariance": float("nan") if adaptive_components_g is None else float(adaptive_components_g.covariance.detach()),
            "generator_grad_norm": generator_grad,
            "critic_updated": int(do_critic),
            "critic_fd": critic_fd,
            "critic_grad_norm": critic_norm,
            "critic_preclip_norm": critic_preclip,
            "clamp_fraction": clamp_fraction,
            "elapsed_seconds": now - started,
            "seconds_since_last_log": now - last_log,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        append_csv(metrics_path, row)
        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)
            last_log = now
        if args.save_every and (step + 1) % args.save_every == 0:
            save_training_checkpoint(
                variant_dir / f"checkpoint_step{step + 1:06d}.pt",
                model=model,
                adaptive_encoder=adaptive_encoder,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                trackers=trackers,
                step=step + 1,
                config=config,
            )

    final_path = variant_dir / "checkpoint_final.pt"
    save_training_checkpoint(
        final_path,
        model=model,
        adaptive_encoder=adaptive_encoder,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        trackers=trackers,
        step=args.steps,
        config=config,
    )
    print(f"saved {final_path}", flush=True)
    return final_path


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    warm_path = args.warmstart_file or args.output_root / "warmstart.pt"
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    projection = warm["projection"].to(device=device, dtype=torch.float32)
    adaptive_feature_dim = resolve_adaptive_feature_dim(
        args.feature_dim, args.adaptive_feature_dim
    )
    adaptive_projection = (
        projection
        if adaptive_feature_dim == args.feature_dim
        else fixed_projection(2048, adaptive_feature_dim, seed=19231, device=device)
    )
    real_full = moments_from_payload(warm["real_full"], device)
    real_projected = moments_from_payload(warm["real_projected"], device)
    checkpoint = args.checkpoint
    adaptive_eval_samples = resolve_adaptive_eval_samples(
        args.eval_samples, args.adaptive_eval_samples
    )
    if args.variant != "base" and checkpoint is None:
        checkpoint = args.output_root / args.variant / "checkpoint_final.pt"

    model = load_pmf_b16(repo=args.pmf_repo, checkpoint=args.base_checkpoint, device=device)
    adaptive_encoder = None
    tracker_payload = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(incompatible)
        tracker_payload = payload.get("trackers")
        if payload.get("adaptive_encoder") is not None:
            adaptive_encoder = DifferentiableInception2048(trainable=False).to(device).eval()
            adaptive_encoder.load_state_dict(payload["adaptive_encoder"], strict=True)
    model.eval().requires_grad_(False)
    static_encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    if adaptive_encoder is not None:
        adaptive_encoder.requires_grad_(False)

    fake_full: list[torch.Tensor] = []
    fake_adaptive: list[torch.Tensor] = []
    first_images: list[torch.Tensor] = []
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 701)
    label_generator = torch.Generator(device=device).manual_seed(args.seed + 709)
    clamp_weighted = 0.0
    produced = 0
    started = time.perf_counter()
    while produced < args.eval_samples:
        batch = min(args.batch_size, args.eval_samples - produced)
        if args.per_sample_eval_noise:
            noise = per_sample_evaluation_noise(
                start=produced,
                count=batch,
                sample_shape=(3, 256, 256),
                initial_seed=args.eval_noise_seed,
                device=device,
                dtype=torch.float32,
            )
        else:
            noise = torch.randn(
                batch, 3, 256, 256,
                generator=noise_generator,
                device=device,
                dtype=torch.float32,
            )
        noise = noise * float(model.noise_scale)
        labels = (
            balanced_evaluation_labels(
                start=produced,
                count=batch,
                total=args.eval_samples,
                num_classes=1000,
                device=device,
            )
            if args.balanced_eval_labels
            else torch.randint(
                0, 1000, (batch,), generator=label_generator, device=device
            )
        )
        with torch.inference_mode():
            images, clamp_fraction = generated_images(model, noise, labels, amp=not args.no_amp)
            if args.quantize_eval_images:
                images = quantize_unit_images(images)
            full, _ = feature_forward(static_encoder, images, projection, amp=not args.no_amp)
            fake_full.append(full.cpu())
            if adaptive_encoder is not None and produced < adaptive_eval_samples:
                _, adaptive = feature_forward(
                    adaptive_encoder,
                    images,
                    adaptive_projection,
                    amp=not args.no_amp,
                )
                take = min(batch, adaptive_eval_samples - produced)
                fake_adaptive.append(adaptive[:take].cpu())
        if sum(batch_.shape[0] for batch_ in first_images) < 64:
            first_images.append(images[: min(batch, 64)].cpu())
        clamp_weighted += clamp_fraction * batch
        produced += batch
        if produced % max(128, args.batch_size) == 0 or produced == args.eval_samples:
            print(f"eval fake {produced}/{args.eval_samples}", flush=True)

    fake_full_cpu = torch.cat(fake_full)
    fake_full_tensor = fake_full_cpu.to(device)
    fake_full_moments = batch_moments(fake_full_tensor)
    fake_projected_moments = project_moments(fake_full_moments, projection)
    full_fd = frechet_from_moments(real_full, fake_full_moments)
    projected_fd = frechet_from_moments(real_projected, fake_projected_moments)

    # The 2048-dimensional, 5K-sample covariance is rank deficient. Its GPU
    # float32 eigendecomposition can move FID by a few hundredths across
    # otherwise pixel-identical evaluations. Keep that diagnostic for history,
    # but also report a deterministic CPU float64 value for paired comparisons.
    fake_full_moments_float64 = batch_moments(fake_full_cpu.to(torch.float64))
    real_full_float64 = moments_from_mean_and_covariance(
        real_full.mean.detach().cpu().to(torch.float64),
        real_full.covariance.detach().cpu().to(torch.float64),
    )
    full_fd_float64 = frechet_from_moments(
        real_full_float64,
        fake_full_moments_float64,
    )
    result: dict[str, Any] = {
        "protocol": "advfd_paper_only_pmf_projected_pilot_v1",
        "paper_reproduction_metric": False,
        "variant": args.variant,
        "eval_samples": args.eval_samples,
        "balanced_eval_labels": bool(args.balanced_eval_labels),
        "per_sample_eval_noise": bool(args.per_sample_eval_noise),
        "eval_noise_seed": args.eval_noise_seed,
        "quantize_eval_images": bool(args.quantize_eval_images),
        "fid_inception_2048_small_sample": float(full_fd.total),
        "fid_inception_2048_mean": float(full_fd.mean),
        "fid_inception_2048_covariance": float(full_fd.covariance),
        "fid_inception_2048_float64": float(full_fd_float64.total),
        "fid_inception_2048_float64_mean": float(full_fd_float64.mean),
        "fid_inception_2048_float64_covariance": float(
            full_fd_float64.covariance
        ),
        "fid_inception_2048_float64_device": "cpu",
        "fd_projected": float(projected_fd.total),
        "fd_projected_mean": float(projected_fd.mean),
        "fd_projected_covariance": float(projected_fd.covariance),
        "clamp_fraction": clamp_weighted / produced,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "evaluation_tag": args.evaluation_tag,
    }

    if adaptive_encoder is not None:
        real_dataset = make_dataset(args, flip=False)
        start_index = len(real_dataset) - adaptive_eval_samples
        real_loader = DataLoader(
            Subset(real_dataset, range(start_index, len(real_dataset))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        real_adaptive: list[torch.Tensor] = []
        with torch.inference_mode():
            for real_images, _, _ in real_loader:
                real_images = real_images.to(device=device, non_blocking=True)
                _, features = feature_forward(
                    adaptive_encoder,
                    real_images,
                    adaptive_projection,
                    amp=not args.no_amp,
                )
                real_adaptive.append(features.cpu())
        # A learned critic can acquire a large common feature offset. Computing
        # E[ff^T] - E[f]E[f]^T in float32 then suffers catastrophic cancellation
        # and can report an indefinite covariance. This is evaluation-only
        # diagnostics, so use float64 without changing the paper-style training
        # arithmetic.
        real_moments = batch_moments(
            torch.cat(real_adaptive).to(device=device, dtype=torch.float64)
        )
        fake_moments = batch_moments(
            torch.cat(fake_adaptive).to(device=device, dtype=torch.float64)
        )
        adaptive_fd, calibrated_real, calibrated_fake = adaptive_components(
            real_moments, fake_moments, variant=args.variant
        )
        result.update(
            {
                "adaptive_heldout_fd": float(adaptive_fd.total),
                "adaptive_eval_samples": adaptive_eval_samples,
                "adaptive_feature_dim": adaptive_feature_dim,
                "adaptive_heldout_fd_mean": float(adaptive_fd.mean),
                "adaptive_heldout_fd_covariance": float(adaptive_fd.covariance),
                "adaptive_real_raw_rms": component_rms(real_moments),
                "adaptive_fake_raw_rms": component_rms(fake_moments),
                "adaptive_real_calibrated_rms": component_rms(calibrated_real),
                "adaptive_fake_calibrated_rms": component_rms(calibrated_fake),
                "adaptive_real_effective_rank": effective_rank(calibrated_real.covariance),
                "adaptive_fake_effective_rank": effective_rank(calibrated_fake.covariance),
                "adaptive_real_raw_covariance": covariance_diagnostics(
                    real_moments.covariance
                ),
                "adaptive_fake_raw_covariance": covariance_diagnostics(
                    fake_moments.covariance
                ),
                "adaptive_real_calibrated_covariance": covariance_diagnostics(
                    calibrated_real.covariance
                ),
                "adaptive_fake_calibrated_covariance": covariance_diagnostics(
                    calibrated_fake.covariance
                ),
                "adaptive_diagnostic_moment_dtype": "float64",
            }
        )
        if tracker_payload is not None:
            train_real_tracker = EMAMomentTracker(args.adaptive_ema)
            train_fake_tracker = EMAMomentTracker(args.adaptive_ema)
            train_real_tracker.load_state_dict(tracker_payload["adaptive_real"])
            train_fake_tracker.load_state_dict(tracker_payload["adaptive_fake"])
            train_real_state = train_real_tracker.state_dict()
            train_fake_state = train_fake_tracker.state_dict()
            train_real_mean = train_real_state["mean"].to(
                device=device, dtype=torch.float64
            )
            train_fake_mean = train_fake_state["mean"].to(
                device=device, dtype=torch.float64
            )
            train_real = moments_from_mean_and_covariance(
                train_real_mean,
                train_real_state["second"].to(device=device, dtype=torch.float64)
                - torch.outer(
                    train_real_mean,
                    train_real_mean,
                ),
            )
            train_fake = moments_from_mean_and_covariance(
                train_fake_mean,
                train_fake_state["second"].to(device=device, dtype=torch.float64)
                - torch.outer(
                    train_fake_mean,
                    train_fake_mean,
                ),
            )
            train_fd, _, _ = adaptive_components(
                train_real, train_fake, variant=args.variant
            )
            result["adaptive_train_ema_fd"] = float(train_fd.total)
            result["adaptive_train_to_heldout_ratio"] = float(
                train_fd.total / adaptive_fd.total.clamp_min(1e-12)
            )
            result["adaptive_train_real_raw_covariance"] = covariance_diagnostics(
                train_real.covariance
            )
            result["adaptive_train_fake_raw_covariance"] = covariance_diagnostics(
                train_fake.covariance
            )

    variant_dir = args.output_root / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    evaluation_name, grid_name = evaluation_artifact_names(args.evaluation_tag)
    output = variant_dir / evaluation_name
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    images = torch.cat(first_images)[:64]
    save_image(images, variant_dir / grid_name, nrow=8)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.feature_dim <= 0:
        raise ValueError("steps must be nonnegative and sizes must be positive")
    resolve_adaptive_feature_dim(args.feature_dim, args.adaptive_feature_dim)
    if args.schedule_total_steps is not None and args.schedule_total_steps < args.steps:
        raise ValueError("schedule-total-steps must be at least steps")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.stage == "warmstart":
        warmstart(args, device)
    elif args.stage == "train":
        checkpoint = train(args, device)
        args.checkpoint = checkpoint
        evaluate(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()

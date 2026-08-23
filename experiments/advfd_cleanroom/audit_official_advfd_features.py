#!/usr/bin/env python3
"""Audit a saved AdvFD critic on fresh real and paired generated images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generated-folder", type=Path, required=True)
    parser.add_argument(
        "--imagenet-root", type=Path, default=Path("/data/shared/imagenet-1k")
    )
    parser.add_argument("--real-split", default="validation")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--repr-model", default="inception")
    parser.add_argument("--adv-state-name", default=None)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    return parser.parse_args()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_adv_state(
    checkpoint: dict[str, Any], state_name: str | None
) -> dict[str, Any]:
    states = checkpoint.get("fd_adv_states")
    if not isinstance(states, list) or not states:
        raise KeyError("checkpoint does not contain fd_adv_states")
    if state_name is None:
        if len(states) != 1:
            names = [str(state.get("name")) for state in states]
            raise ValueError(f"multiple adaptive states found; choose one: {names}")
        return states[0]
    matches = [state for state in states if state.get("name") == state_name]
    if len(matches) != 1:
        raise KeyError(f"expected one adaptive state named {state_name!r}")
    return matches[0]


def local_subset(dataset, limit: int, rank: int, world_size: int) -> Subset:
    limit = min(int(limit), len(dataset))
    return Subset(dataset, list(range(rank, limit, world_size)))


def feature_summary(mu: np.ndarray, covariance: np.ndarray) -> dict[str, float | int]:
    dimension = int(mu.size)
    covariance_trace = float(np.trace(covariance))
    covariance_frobenius_sq = float(np.square(covariance).sum())
    second_moment = covariance_trace + float(np.dot(mu, mu))
    return {
        "feature_dim": dimension,
        "mean_norm": float(np.linalg.norm(mu)),
        "feature_rms": math.sqrt(max(second_moment / dimension, 0.0)),
        "vector_rms": math.sqrt(max(second_moment, 0.0)),
        "covariance_trace": covariance_trace,
        "covariance_frobenius": math.sqrt(max(covariance_frobenius_sq, 0.0)),
        "covariance_participation_rank": (
            covariance_trace**2 / covariance_frobenius_sq
            if covariance_frobenius_sq > 0.0
            else 0.0
        ),
        "covariance_diag_min": float(np.diag(covariance).min()),
        "covariance_diag_max": float(np.diag(covariance).max()),
    }


def scalar_distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty scalar distribution")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def covariance_from_ema_state(
    stats: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    mu = stats["mu_ema"].detach().double().cpu().numpy()
    second = stats["m2_ema"].detach().double().cpu().numpy()
    covariance = second - np.outer(mu, mu)
    return mu, 0.5 * (covariance + covariance.T)


def equally_weighted_mixture_moments(
    first_mu: np.ndarray,
    first_covariance: np.ndarray,
    second_mu: np.ndarray,
    second_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mixture_mu = 0.5 * (first_mu + second_mu)
    first_offset = first_mu - mixture_mu
    second_offset = second_mu - mixture_mu
    mixture_covariance = 0.5 * (
        first_covariance
        + np.outer(first_offset, first_offset)
        + second_covariance
        + np.outer(second_offset, second_offset)
    )
    return mixture_mu, 0.5 * (mixture_covariance + mixture_covariance.T)


def whiten_moment_pair(
    real_mu: np.ndarray,
    real_covariance: np.ndarray,
    fake_mu: np.ndarray,
    fake_covariance: np.ndarray,
    *,
    anchor_mu: np.ndarray,
    anchor_covariance: np.ndarray,
    epsilon: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Apply the official regularized Gaussian whitening to two distributions."""

    if epsilon <= 0.0:
        raise ValueError("whiten epsilon must be positive")
    real_mu_t = torch.from_numpy(real_mu).double()
    fake_mu_t = torch.from_numpy(fake_mu).double()
    anchor_mu_t = torch.from_numpy(anchor_mu).double()
    anchor_covariance_t = torch.from_numpy(anchor_covariance).double()
    dimension = int(anchor_mu_t.numel())
    identity = torch.eye(dimension, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(
        0.5 * (anchor_covariance_t + anchor_covariance_t.T) + epsilon * identity
    )
    inverse_roots = eigenvalues.clamp_min(epsilon).rsqrt()
    transform = eigenvectors * inverse_roots.unsqueeze(0)

    def apply(mu: torch.Tensor, covariance: np.ndarray):
        covariance_t = torch.from_numpy(covariance).double() + epsilon * identity
        transformed_mu = (mu - anchor_mu_t) @ transform
        transformed_covariance = transform.T @ covariance_t @ transform
        transformed_covariance = 0.5 * (
            transformed_covariance + transformed_covariance.T
        )
        return transformed_mu, transformed_covariance

    return apply(real_mu_t, real_covariance), apply(fake_mu_t, fake_covariance)


def calibrated_fd_summary(
    real_mu: np.ndarray,
    real_covariance: np.ndarray,
    fake_mu: np.ndarray,
    fake_covariance: np.ndarray,
    *,
    anchor_mu: np.ndarray,
    anchor_covariance: np.ndarray,
    epsilon: float,
    frechet_from_moments,
    moments_from_mean_and_covariance,
) -> dict[str, float]:
    real, fake = whiten_moment_pair(
        real_mu,
        real_covariance,
        fake_mu,
        fake_covariance,
        anchor_mu=anchor_mu,
        anchor_covariance=anchor_covariance,
        epsilon=epsilon,
    )
    distance = frechet_from_moments(
        moments_from_mean_and_covariance(*real),
        moments_from_mean_and_covariance(*fake),
    )
    return {
        "total": float(distance.total),
        "mean": float(distance.mean),
        "covariance": float(distance.covariance),
    }


@torch.inference_mode()
def extract_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    use_amp = not getattr(model, "is_inception", False)
    with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        primary, _ = model(images)
    return primary.float()


@torch.inference_mode()
def accumulate_dataset(
    dataset,
    models: dict[str, torch.nn.Module],
    *,
    limit: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, int]],
    dict[str, dict[str, float | int]],
]:
    subset = local_subset(dataset, limit, rank, world_size)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    accumulators: dict[str, dict[str, torch.Tensor | int]] = {}
    sample_metrics: dict[str, list[torch.Tensor]] = {
        "reference_feature_rms": [],
        "adaptive_feature_rms": [],
        "adaptive_to_reference_feature_rms_ratio": [],
        "adaptive_reference_residual_rms": [],
        "adaptive_reference_cosine": [],
    }

    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.cuda(non_blocking=True)
        batch_features: dict[str, torch.Tensor] = {}
        for name, model in models.items():
            features = extract_features(model, images).double()
            batch_features[name] = features
            if name not in accumulators:
                dim = int(features.shape[-1])
                accumulators[name] = {
                    "sum": torch.zeros(dim, dtype=torch.float64, device="cuda"),
                    "outer": torch.zeros(
                        dim, dim, dtype=torch.float64, device="cuda"
                    ),
                    "count": 0,
                }
            accumulator = accumulators[name]
            accumulator["sum"].add_(features.sum(0))
            accumulator["outer"].addmm_(features.T, features)
            accumulator["count"] = int(accumulator["count"]) + int(
                features.shape[0]
            )

        reference_features = batch_features["reference"]
        adaptive_features = batch_features["adaptive"]
        dimension_scale = math.sqrt(float(reference_features.shape[-1]))
        reference_rms = torch.linalg.vector_norm(
            reference_features, dim=-1
        ) / dimension_scale
        adaptive_rms = torch.linalg.vector_norm(
            adaptive_features, dim=-1
        ) / dimension_scale
        residual_rms = torch.linalg.vector_norm(
            adaptive_features - reference_features, dim=-1
        ) / dimension_scale
        cosine = torch.nn.functional.cosine_similarity(
            adaptive_features, reference_features, dim=-1
        )
        sample_metrics["reference_feature_rms"].append(reference_rms.float())
        sample_metrics["adaptive_feature_rms"].append(adaptive_rms.float())
        sample_metrics["adaptive_to_reference_feature_rms_ratio"].append(
            (adaptive_rms / reference_rms.clamp_min(1e-12)).float()
        )
        sample_metrics["adaptive_reference_residual_rms"].append(
            residual_rms.float()
        )
        sample_metrics["adaptive_reference_cosine"].append(cosine.float())

    results: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name, accumulator in accumulators.items():
        feature_sum = accumulator["sum"]
        feature_outer = accumulator["outer"]
        count = torch.tensor(
            [int(accumulator["count"])], dtype=torch.long, device="cuda"
        )
        if world_size > 1:
            torch.distributed.reduce(feature_sum, dst=0)
            torch.distributed.reduce(feature_outer, dst=0)
            torch.distributed.reduce(count, dst=0)
        if rank == 0:
            total = int(count.item())
            if total < 2:
                raise ValueError("at least two images are required")
            sum_cpu = feature_sum.cpu().numpy()
            outer_cpu = feature_outer.cpu().numpy()
            mu = sum_cpu / total
            covariance = (
                outer_cpu - np.outer(sum_cpu, sum_cpu) / total
            ) / (total - 1)
            covariance = 0.5 * (covariance + covariance.T)
            results[name] = (mu, covariance, total)

    local_sample_metrics = {
        name: torch.cat(chunks).cpu().numpy()
        for name, chunks in sample_metrics.items()
    }
    if world_size > 1:
        gathered: list[dict[str, np.ndarray] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_sample_metrics)
        merged_sample_metrics = {
            name: np.concatenate(
                [rank_metrics[name] for rank_metrics in gathered if rank_metrics]
            )
            for name in local_sample_metrics
        }
    else:
        merged_sample_metrics = local_sample_metrics
    sample_summaries = (
        {
            name: scalar_distribution_summary(values)
            for name, values in merged_sample_metrics.items()
        }
        if rank == 0
        else {}
    )
    return results, sample_summaries


def main() -> None:
    args = parse_args()
    official_root = args.official_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    generated_folder = args.generated_folder.expanduser().resolve()
    imagenet_root = args.imagenet_root.expanduser().resolve()
    for required in (official_root / "main_fd.py", checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not generated_folder.is_dir():
        raise FileNotFoundError(generated_folder)

    sys.path.insert(0, str(EQVAE_ROOT))
    sys.path.insert(0, str(official_root))
    from experiments.advfd_cleanroom.core import (
        frechet_from_moments,
        moments_from_mean_and_covariance,
    )
    from experiments.raev2_training_core import DeterministicImageNetParquet
    from frechet_distance.datasets import ImageFolderDataset
    from frechet_distance.repr_models import load_repr_model
    from utils.distributed_util import enable_distributed, get_global_rank, get_world_size

    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    checkpoint_metadata = {
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
    }
    adv_state = select_adv_state(checkpoint, args.adv_state_name)
    checkpoint_real_mu, checkpoint_real_covariance = covariance_from_ema_state(
        adv_state["real_stats"]
    )
    ref_model, feature_dim, _, _ = load_repr_model(
        args.repr_model, device="cuda"
    )
    adv_model, adv_feature_dim, _, _ = load_repr_model(
        args.repr_model, device="cuda"
    )
    if feature_dim != adv_feature_dim:
        raise ValueError("reference and adaptive feature dimensions differ")
    adv_model.load_state_dict(adv_state["model"], strict=True)
    ref_model.eval()
    adv_model.eval()
    del checkpoint

    models = {"reference": ref_model, "adaptive": adv_model}
    real_dataset = DeterministicImageNetParquet(
        imagenet_root,
        split=args.real_split,
        image_size=args.img_size,
        horizontal_flip=False,
    )
    fake_dataset = ImageFolderDataset(str(generated_folder), img_size=args.img_size)
    real_stats, real_sample_metrics = accumulate_dataset(
        real_dataset,
        models,
        limit=args.num_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )
    fake_stats, fake_sample_metrics = accumulate_dataset(
        fake_dataset,
        models,
        limit=args.num_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        model_results: dict[str, dict[str, Any]] = {}
        for name in models:
            real_mu, real_covariance, real_count = real_stats[name]
            fake_mu, fake_covariance, fake_count = fake_stats[name]
            mean_fd = float(np.square(real_mu - fake_mu).sum())
            distance = frechet_from_moments(
                moments_from_mean_and_covariance(
                    torch.from_numpy(real_mu), torch.from_numpy(real_covariance)
                ),
                moments_from_mean_and_covariance(
                    torch.from_numpy(fake_mu), torch.from_numpy(fake_covariance)
                ),
            )
            total_fd = float(distance.total)
            model_results[name] = {
                "real_count": real_count,
                "fake_count": fake_count,
                "real": feature_summary(real_mu, real_covariance),
                "fake": feature_summary(fake_mu, fake_covariance),
                "heldout_real_fake_fd": total_fd,
                "heldout_real_fake_fd_mean": float(distance.mean),
                "heldout_real_fake_fd_covariance": float(distance.covariance),
                "mean_gap_norm": math.sqrt(max(mean_fd, 0.0)),
                "covariance_gap_frobenius": float(
                    np.linalg.norm(real_covariance - fake_covariance, ord="fro")
                ),
            }

            if name == "adaptive":
                fresh_anchor = (real_mu, real_covariance)
                pooled_anchor = equally_weighted_mixture_moments(
                    real_mu, real_covariance, fake_mu, fake_covariance
                )
                model_results[name]["calibrated_heldout_fd"] = {
                    "checkpoint_real_ema": calibrated_fd_summary(
                        real_mu,
                        real_covariance,
                        fake_mu,
                        fake_covariance,
                        anchor_mu=checkpoint_real_mu,
                        anchor_covariance=checkpoint_real_covariance,
                        epsilon=args.whiten_eps,
                        frechet_from_moments=frechet_from_moments,
                        moments_from_mean_and_covariance=moments_from_mean_and_covariance,
                    ),
                    "fresh_validation_real": calibrated_fd_summary(
                        real_mu,
                        real_covariance,
                        fake_mu,
                        fake_covariance,
                        anchor_mu=fresh_anchor[0],
                        anchor_covariance=fresh_anchor[1],
                        epsilon=args.whiten_eps,
                        frechet_from_moments=frechet_from_moments,
                        moments_from_mean_and_covariance=moments_from_mean_and_covariance,
                    ),
                    "fresh_validation_pooled": calibrated_fd_summary(
                        real_mu,
                        real_covariance,
                        fake_mu,
                        fake_covariance,
                        anchor_mu=pooled_anchor[0],
                        anchor_covariance=pooled_anchor[1],
                        epsilon=args.whiten_eps,
                        frechet_from_moments=frechet_from_moments,
                        moments_from_mean_and_covariance=moments_from_mean_and_covariance,
                    ),
                }

        reference = model_results["reference"]
        adaptive = model_results["adaptive"]
        result = {
            "protocol": "official_advfd_fresh_validation_and_paired_fake_v1",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "adv_state_name": adv_state.get("name"),
            "feature_transform": adv_state.get("feature_transform", "unknown"),
            "whiten_epsilon": args.whiten_eps,
            "generated_folder": str(generated_folder),
            "imagenet_root": str(imagenet_root),
            "real_split": args.real_split,
            "requested_images": args.num_images,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "official_root": str(official_root),
            "official_commit": git_head(official_root),
            "models": model_results,
            "samplewise_feature_diagnostics": {
                "real": real_sample_metrics,
                "fake": fake_sample_metrics,
            },
            "ratios": {
                "adaptive_to_reference_real_feature_rms": (
                    adaptive["real"]["feature_rms"]
                    / reference["real"]["feature_rms"]
                ),
                "adaptive_to_reference_fake_feature_rms": (
                    adaptive["fake"]["feature_rms"]
                    / reference["fake"]["feature_rms"]
                ),
                "adaptive_fake_to_real_feature_rms": (
                    adaptive["fake"]["feature_rms"]
                    / adaptive["real"]["feature_rms"]
                ),
                "reference_fake_to_real_feature_rms": (
                    reference["fake"]["feature_rms"]
                    / reference["real"]["feature_rms"]
                ),
                "adaptive_to_reference_heldout_fd": (
                    adaptive["heldout_real_fake_fd"]
                    / reference["heldout_real_fake_fd"]
                ),
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if world_size > 1:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

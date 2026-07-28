"""Post-hoc audit of prior errors in decoder-relevant class directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_imagenette_latent_prior_tradeoff import (  # noqa: E402
    condition_embeddings,
    load_run_config,
)
from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    INTERFACE_DIM,
    OrthogonalLatentInterface,
    build_prior,
    deterministic_datasets,
    fixed_eval_subset,
    fixed_orthogonal_basis,
    load_frozen_models,
    sample_prior_coordinates,
    sliced_wasserstein_distance,
)
from experiments.mnist_spectral_rollout_toy import configure_fp32  # noqa: E402


def class_direction_basis(
    train_embedding: torch.Tensor,
    labels: torch.Tensor,
    class_count: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_mean = train_embedding.mean(dim=0)
    centroids = torch.stack(
        [train_embedding[labels == label].mean(dim=0) for label in range(class_count)]
    )
    centered_centroids = centroids - global_mean
    _u, singular_values, vh = torch.linalg.svd(centered_centroids.double(), full_matrices=False)
    tolerance = float(singular_values.max()) * 1e-8
    rank = min(class_count - 1, int((singular_values > tolerance).sum()))
    basis = vh[:rank].T.float().contiguous()
    return global_mean, centroids, basis


def nearest_centroid_metrics(
    embedding: torch.Tensor,
    centroids: torch.Tensor,
    target_distribution: torch.Tensor,
) -> dict[str, float | list[float]]:
    normalized = F.normalize(embedding, dim=1)
    normalized_centroids = F.normalize(centroids, dim=1)
    prediction = (normalized @ normalized_centroids.T).argmax(dim=1)
    histogram = torch.bincount(prediction, minlength=len(centroids)).float()
    histogram = histogram / histogram.sum()
    entropy = -(histogram * histogram.clamp_min(1e-12).log()).sum()
    return {
        "class_entropy": float(entropy),
        "effective_classes": float(entropy.exp()),
        "class_tv": float(0.5 * (histogram - target_distribution).abs().sum()),
        "class_histogram": histogram.tolist(),
    }


def subspace_distances(
    real: torch.Tensor,
    generated: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    real_centered = real - mean
    generated_centered = generated - mean
    real_semantic = real_centered @ basis
    generated_semantic = generated_centered @ basis
    real_residual = real_centered - real_semantic @ basis.T
    generated_residual = generated_centered - generated_semantic @ basis.T
    semantic_swd = sliced_wasserstein_distance(
        real_semantic, generated_semantic, directions=256, seed=seed
    )
    residual_swd = sliced_wasserstein_distance(
        real_residual, generated_residual, directions=256, seed=seed + 1
    )
    semantic_mean_error = float(
        (generated_semantic.mean(dim=0) - real_semantic.mean(dim=0)).norm()
    )
    residual_mean_error = float(
        (generated_residual.mean(dim=0) - real_residual.mean(dim=0)).norm()
    )
    return {
        "semantic_swd": semantic_swd,
        "residual_swd": residual_swd,
        "semantic_mean_error": semantic_mean_error,
        "residual_mean_error": residual_mean_error,
        "semantic_to_residual_swd_ratio": semantic_swd / max(residual_swd, 1e-12),
    }


@torch.no_grad()
def analyze_run(run: Path, device_name: str, overwrite: bool = False) -> Path:
    output = run / "semantic_gap_audit.json"
    if output.is_file() and not overwrite:
        return output
    config = load_run_config(run, device_name)
    configure_fp32(config.prior_seed)
    device = torch.device(device_name)
    _train_dataset, val_dataset = deterministic_datasets(config.data_root, config.image_size)
    _encoder, decoder, _frozen = load_frozen_models(config, device)
    cache = torch.load(run / "latent_cache.pt", map_location="cpu", weights_only=True)
    prior_state = torch.load(run / "prior_state.pt", map_location="cpu", weights_only=True)
    prior = build_prior(config, device)
    prior.load_state_dict(prior_state["prior_ema"])
    prior.eval()
    interface = OrthogonalLatentInterface(
        config.latent_dim,
        fixed_orthogonal_basis(INTERFACE_DIM, config.basis_seed),
    ).to(device)
    count = min(config.quality_count, len(val_dataset))
    eval_subset = fixed_eval_subset(val_dataset, count, seed=2_027)
    eval_indices = torch.as_tensor(eval_subset.indices, dtype=torch.long)
    val_latent = cache["val_latent"][eval_indices]
    val_labels = cache["val_labels"][eval_indices]
    prior_latent = sample_prior_coordinates(
        prior,
        interface,
        count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )
    empirical_generator = torch.Generator(device="cpu").manual_seed(config.prior_seed + 1_101)
    empirical_indices = torch.randint(
        len(cache["train_latent"]), (count,), generator=empirical_generator
    )
    empirical_latent = cache["train_latent"][empirical_indices]
    train_condition = condition_embeddings(
        decoder, cache["train_latent"], batch_size=config.eval_batch_size
    )
    val_condition = condition_embeddings(
        decoder, val_latent, batch_size=config.eval_batch_size
    )
    empirical_condition = condition_embeddings(
        decoder, empirical_latent, batch_size=config.eval_batch_size
    )
    prior_condition = condition_embeddings(
        decoder, prior_latent, batch_size=config.eval_batch_size
    )
    mean, centroids, basis = class_direction_basis(
        train_condition, cache["train_labels"]
    )
    target = torch.bincount(val_labels, minlength=10).float()
    target = target / target.sum()
    val_prediction = (
        F.normalize(val_condition, dim=1) @ F.normalize(centroids, dim=1).T
    ).argmax(dim=1)
    payload = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "semantic_subspace_rank": int(basis.shape[1]),
        "val_nearest_centroid_accuracy": float((val_prediction == val_labels).float().mean()),
        **{
            f"empirical_{key}": value
            for key, value in nearest_centroid_metrics(
                empirical_condition, centroids, target
            ).items()
        },
        **{
            f"prior_{key}": value
            for key, value in nearest_centroid_metrics(prior_condition, centroids, target).items()
        },
        **{
            f"empirical_{key}": value
            for key, value in subspace_distances(
                val_condition,
                empirical_condition,
                mean,
                basis,
                seed=config.prior_seed + 2_101,
            ).items()
        },
        **{
            f"prior_{key}": value
            for key, value in subspace_distances(
                val_condition,
                prior_condition,
                mean,
                basis,
                seed=config.prior_seed + 2_201,
            ).items()
        },
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return analyze_run(args.run, args.device, args.overwrite)


if __name__ == "__main__":
    main()

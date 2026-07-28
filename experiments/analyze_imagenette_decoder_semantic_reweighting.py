"""Post-hoc decomposition of the decoded gap by semantic mass reweighting.

The audit never changes a latent prior or decoder. It estimates class-mass
ratios on four folds, applies them to the held-out fold, and recomputes a
weighted feature FID. Cyclically shifted class assignments are negative
controls with the same ratio magnitudes but the wrong semantic correspondence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_latent_prior_tradeoff import (
    IMAGENETTE_SYNSET_TO_IMAGENET_INDEX,
    ResNet18Evaluator,
)


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def normalized_weighted_moments(
    features: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = features.detach().double().cpu().numpy()
    weight = weights.detach().double().cpu().numpy()
    if values.ndim != 2 or weight.shape != (len(values),):
        raise ValueError("features and weights have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(weight).all():
        raise FloatingPointError("non-finite weighted moment input")
    if (weight < 0.0).any() or float(weight.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with positive mass")
    weight = weight / weight.sum()
    mean = weight @ values
    centered = values - mean
    correction = 1.0 - float(np.square(weight).sum())
    if correction <= 0.0:
        raise ValueError("weighted covariance has fewer than two effective samples")
    covariance = (centered * weight[:, None]).T @ centered / correction
    covariance = 0.5 * (covariance + covariance.T)
    effective_count = 1.0 / float(np.square(weight).sum())
    return mean, covariance, effective_count


def frechet_components(
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
) -> tuple[float, float]:
    difference = reference_mean - candidate_mean
    eigenvalues, eigenvectors = np.linalg.eigh(reference_covariance)
    reference_root = (
        eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    ) @ eigenvectors.T
    middle = reference_root @ candidate_covariance @ reference_root
    middle = 0.5 * (middle + middle.T)
    covariance_root_trace = float(
        np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum()
    )
    mean_component = float(difference @ difference)
    covariance_component = float(
        + np.trace(reference_covariance)
        + np.trace(candidate_covariance)
        - 2.0 * covariance_root_trace
    )
    return max(mean_component, 0.0), max(covariance_component, 0.0)


def frechet_from_moments(
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
) -> float:
    mean_component, covariance_component = frechet_components(
        reference_mean,
        reference_covariance,
        candidate_mean,
        candidate_covariance,
    )
    return mean_component + covariance_component


def crossfit_class_weights(
    empirical_labels: torch.Tensor,
    prior_labels: torch.Tensor,
    *,
    seed: int,
    folds: int = 5,
    smoothing: float = 1.0,
    application_shift: int = 0,
) -> torch.Tensor:
    empirical = empirical_labels.detach().long().cpu().numpy()
    prior = prior_labels.detach().long().cpu().numpy()
    if empirical.shape != prior.shape or empirical.ndim != 1:
        raise ValueError("empirical and prior labels must have equal vector shapes")
    class_count = int(max(empirical.max(), prior.max())) + 1
    if class_count < 2 or folds < 2 or len(empirical) < folds:
        raise ValueError("cross-fitting requires multiple classes, folds, and samples")
    generator = np.random.default_rng(int(seed))
    fold_ids = np.empty(len(empirical), dtype=np.int64)
    fold_ids[generator.permutation(len(empirical))] = np.arange(len(empirical)) % folds
    weights = np.empty(len(prior), dtype=np.float64)
    for fold in range(folds):
        heldout = fold_ids == fold
        train = ~heldout
        empirical_count = np.bincount(empirical[train], minlength=class_count).astype(np.float64)
        prior_count = np.bincount(prior[train], minlength=class_count).astype(np.float64)
        empirical_probability = (empirical_count + smoothing) / (
            empirical_count.sum() + smoothing * class_count
        )
        prior_probability = (prior_count + smoothing) / (
            prior_count.sum() + smoothing * class_count
        )
        ratio = empirical_probability / prior_probability
        applied_label = (prior[heldout] + int(application_shift)) % class_count
        weights[heldout] = ratio[applied_label]
    return torch.from_numpy(weights)


def restricted_predictions(
    features: torch.Tensor,
    evaluator: ResNet18Evaluator,
) -> torch.Tensor:
    synsets = sorted(IMAGENETTE_SYNSET_TO_IMAGENET_INDEX)
    indices = torch.tensor(
        [IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset] for synset in synsets],
        dtype=torch.long,
    )
    weight = evaluator.classifier.weight.detach().cpu()[indices].to(features.dtype)
    bias = evaluator.classifier.bias.detach().cpu()[indices].to(features.dtype)
    return (features @ weight.T + bias).argmax(dim=1)


def weighted_histogram(labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    class_count = int(labels.max()) + 1
    histogram = torch.zeros(class_count, dtype=torch.float64)
    histogram.scatter_add_(0, labels.long(), weights.double())
    return histogram / histogram.sum()


def analyze_run(
    run: Path,
    evaluator: ResNet18Evaluator,
    *,
    overwrite: bool = False,
) -> dict[str, float | int]:
    output = run / "decoder_semantic_reweight_posthoc.json"
    if output.is_file() and not overwrite:
        return json.loads(output.read_text())
    empirical_payload = torch.load(
        run / "features_empirical.pt", map_location="cpu", weights_only=True
    )
    prior_payload = torch.load(
        run / "features_prior.pt", map_location="cpu", weights_only=True
    )
    real = empirical_payload["real_features"].double()
    empirical = empirical_payload["generated_features"].double()
    prior = prior_payload["generated_features"].double()
    if not (real.shape == empirical.shape == prior.shape):
        raise RuntimeError("saved feature shapes differ")
    empirical_label = restricted_predictions(empirical, evaluator)
    prior_label = restricted_predictions(prior, evaluator)
    summary = json.loads((run / "summary.json").read_text())
    seed = int(summary["prior_seed"]) + 7_001

    reference_mean, reference_covariance, _ = normalized_weighted_moments(
        real, torch.ones(len(real))
    )
    prior_mean, prior_covariance, _ = normalized_weighted_moments(
        prior, torch.ones(len(prior))
    )
    empirical_mean, empirical_covariance, _ = normalized_weighted_moments(
        empirical, torch.ones(len(empirical))
    )
    original_prior_fid = frechet_from_moments(
        reference_mean, reference_covariance, prior_mean, prior_covariance
    )
    recomputed_empirical_fid = frechet_from_moments(
        reference_mean, reference_covariance, empirical_mean, empirical_covariance
    )
    semantic_weights = crossfit_class_weights(
        empirical_label, prior_label, seed=seed
    )
    semantic_mean, semantic_covariance, semantic_ess = normalized_weighted_moments(
        prior, semantic_weights
    )
    semantic_fid = frechet_from_moments(
        reference_mean, reference_covariance, semantic_mean, semantic_covariance
    )
    prior_mean_component, prior_covariance_component = frechet_components(
        reference_mean, reference_covariance, prior_mean, prior_covariance
    )
    empirical_mean_component, empirical_covariance_component = frechet_components(
        reference_mean, reference_covariance, empirical_mean, empirical_covariance
    )
    semantic_mean_component, semantic_covariance_component = frechet_components(
        reference_mean, reference_covariance, semantic_mean, semantic_covariance
    )

    control_fids = []
    for shift in range(1, 10):
        control_weights = crossfit_class_weights(
            empirical_label,
            prior_label,
            seed=seed,
            application_shift=shift,
        )
        control_mean, control_covariance, _ = normalized_weighted_moments(
            prior, control_weights
        )
        control_fids.append(
            frechet_from_moments(
                reference_mean,
                reference_covariance,
                control_mean,
                control_covariance,
            )
        )

    empirical_histogram = weighted_histogram(
        empirical_label, torch.ones(len(empirical_label))
    )
    prior_histogram = weighted_histogram(prior_label, torch.ones(len(prior_label)))
    weighted_prior_histogram = weighted_histogram(prior_label, semantic_weights)
    empirical_fid = float(summary["empirical_feature_fid"])
    formal_prior_fid = float(summary["end_to_end_feature_fid"])
    modeling_gap = formal_prior_fid - empirical_fid
    payload: dict[str, float | int] = {
        "latent_dim": int(summary["latent_dim"]),
        "frozen_seed": int(summary["frozen_seed"]),
        "count": len(real),
        "empirical_fid": empirical_fid,
        "formal_prior_fid": formal_prior_fid,
        "recomputed_empirical_fid": recomputed_empirical_fid,
        "recomputed_empirical_fid_abs_error": abs(
            recomputed_empirical_fid - empirical_fid
        ),
        "recomputed_prior_fid": original_prior_fid,
        "recomputed_prior_fid_abs_error": abs(original_prior_fid - formal_prior_fid),
        "modeling_gap": modeling_gap,
        "semantic_reweighted_fid": semantic_fid,
        "empirical_fid_mean_component": empirical_mean_component,
        "empirical_fid_covariance_component": empirical_covariance_component,
        "prior_fid_mean_component": prior_mean_component,
        "prior_fid_covariance_component": prior_covariance_component,
        "semantic_fid_mean_component": semantic_mean_component,
        "semantic_fid_covariance_component": semantic_covariance_component,
        "modeling_mean_component_gap": (
            prior_mean_component - empirical_mean_component
        ),
        "modeling_covariance_component_gap": (
            prior_covariance_component - empirical_covariance_component
        ),
        "semantic_fid_improvement": formal_prior_fid - semantic_fid,
        "semantic_fraction_of_modeling_gap": (
            (formal_prior_fid - semantic_fid) / modeling_gap
            if abs(modeling_gap) > 1e-12
            else float("nan")
        ),
        "semantic_effective_sample_count": semantic_ess,
        "class_tv_before": float(
            0.5 * (prior_histogram - empirical_histogram).abs().sum()
        ),
        "class_tv_after": float(
            0.5 * (weighted_prior_histogram - empirical_histogram).abs().sum()
        ),
        "shift_control_fid_mean": float(np.mean(control_fids)),
        "shift_control_fid_best": float(np.min(control_fids)),
        "semantic_improvement_over_control_mean": float(
            np.mean(control_fids) - semantic_fid
        ),
        "semantic_improvement_over_control_best": float(
            np.min(control_fids) - semantic_fid
        ),
    }
    if not all(math.isfinite(float(value)) for value in payload.values()):
        raise FloatingPointError("non-finite semantic reweighting metric")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def plot_summary(table: pd.DataFrame, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {16: "#4C78A8", 64: "#F2A541", 256: "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(CAPACITIES))
    for axis, metric, title, ylabel in (
        (axes[0], "modeling_gap", "Original decoded modeling gap", "FID gap"),
        (
            axes[1],
            "semantic_fid_improvement",
            "FID recovered by class-mass reweighting",
            "FID reduction",
        ),
        (
            axes[2],
            "semantic_fraction_of_modeling_gap",
            "Fraction of gap recovered",
            "recovered fraction",
        ),
    ):
        means = [table[table.latent_dim == dim][metric].mean() for dim in CAPACITIES]
        sems = [table[table.latent_dim == dim][metric].sem() for dim in CAPACITIES]
        axis.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color=[colors[dim] for dim in CAPACITIES],
            edgecolor="#333333",
            linewidth=0.8,
        )
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(dim) for dim in CAPACITIES])
        axis.set_xlabel("latent capacity")

    mean_gap = [
        table[table.latent_dim == dim].modeling_mean_component_gap.mean()
        for dim in CAPACITIES
    ]
    covariance_gap = [
        table[table.latent_dim == dim].modeling_covariance_component_gap.mean()
        for dim in CAPACITIES
    ]
    axes[3].bar(
        x,
        mean_gap,
        color="#4C78A8",
        edgecolor="#333333",
        linewidth=0.8,
        label="feature mean",
    )
    axes[3].bar(
        x,
        covariance_gap,
        bottom=mean_gap,
        color="#F2A541",
        edgecolor="#333333",
        linewidth=0.8,
        label="feature covariance",
    )
    axes[3].set_title("Exact FID-gap decomposition")
    axes[3].set_ylabel("FID gap component")
    axes[3].set_xticks(x, [str(dim) for dim in CAPACITIES])
    axes[3].set_xlabel("latent capacity")
    axes[3].legend(frameon=False)
    fig.suptitle("Imagenette-64 semantic mass reweighting audit", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(root: Path, *, overwrite: bool = False) -> dict:
    evaluator = ResNet18Evaluator().eval()
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            run = root / f"d{capacity}_seed{seed}_p0"
            rows.append(analyze_run(run, evaluator, overwrite=overwrite))
            print(f"complete d{capacity} seed{seed}", flush=True)
    table = pd.DataFrame(rows).sort_values(["frozen_seed", "latent_dim"])
    output = root / "comparison_p0"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "decoder_semantic_reweight_runs.csv", index=False)
    summary = table.groupby("latent_dim").agg(
        seed_count=("frozen_seed", "count"),
        modeling_gap_mean=("modeling_gap", "mean"),
        semantic_fid_improvement_mean=("semantic_fid_improvement", "mean"),
        semantic_fid_improvement_std=("semantic_fid_improvement", "std"),
        semantic_fraction_mean=("semantic_fraction_of_modeling_gap", "mean"),
        semantic_fraction_std=("semantic_fraction_of_modeling_gap", "std"),
        class_tv_before_mean=("class_tv_before", "mean"),
        class_tv_after_mean=("class_tv_after", "mean"),
        semantic_ess_mean=("semantic_effective_sample_count", "mean"),
        improvement_over_control_mean=("semantic_improvement_over_control_mean", "mean"),
        modeling_mean_component_gap_mean=("modeling_mean_component_gap", "mean"),
        modeling_covariance_component_gap_mean=(
            "modeling_covariance_component_gap", "mean"
        ),
    ).reset_index()
    summary.to_csv(output / "decoder_semantic_reweight_capacity_summary.csv", index=False)
    checks = {
        "complete_grid": len(table) == len(CAPACITIES) * len(SEEDS),
        "recomputed_fid_max_abs_error": float(
            table.recomputed_prior_fid_abs_error.max()
        ),
        "recomputed_empirical_fid_max_abs_error": float(
            table.recomputed_empirical_fid_abs_error.max()
        ),
        "positive_improvement_seed_count_by_capacity": {
            str(capacity): int(
                (table[table.latent_dim == capacity].semantic_fid_improvement > 0).sum()
            )
            for capacity in CAPACITIES
        },
        "better_than_control_mean_seed_count_by_capacity": {
            str(capacity): int(
                (
                    table[table.latent_dim == capacity]
                    .semantic_improvement_over_control_mean
                    > 0
                ).sum()
            )
            for capacity in CAPACITIES
        },
    }
    (output / "decoder_semantic_reweight_summary.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n"
    )
    plot_summary(table, output / "decoder_semantic_reweight.png")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(checks, indent=2, ensure_ascii=False), flush=True)
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return summarize(args.root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

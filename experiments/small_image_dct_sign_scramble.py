"""Preserve DCT power exactly while destroying spatial/semantic organization.

The DCT is real-valued, so coefficient signs are the 0/pi analogue of Fourier
phase.  Multiplying every non-DC coefficient by an independent random sign
preserves each squared coefficient and every radial-band energy exactly.  The
experiment then measures how much image semantics and feature distributions can
change despite a perfect spectral-energy proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (
    configure_fp32,
    frechet_distance,
    resolve_device,
    sliced_wasserstein,
    train_feature_classifier,
)
from experiments.small_image_basis_transport import (
    DATASETS,
    dct_pixel_basis,
    load_small_image_tensors,
)
from experiments.rae_spectral_direction_loss import radial_band_index


DEFAULT_OUTPUT_ROOT = (
    Path.home() / "data/eqvae/experiments/small_image_dct_sign_scramble"
)


@dataclass(frozen=True)
class SignScrambleConfig:
    dataset: str = "fashion_mnist"
    data_root: Path = Path("/data/shared/fashion_mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    train_size: int = 8_192
    test_size: int = 4_096
    classifier_epochs: int = 3
    batch_size: int = 256
    band_count: int = 8
    projection_count: int = 64
    seed: int = 0
    device: str = "cuda:0"
    save: bool = True


def dct_sign_scramble(
    images: torch.Tensor,
    basis: torch.Tensor,
    *,
    seed: int,
    preserve_dc: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scrambled images and before/after DCT coefficients."""

    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("expected grayscale images with shape [B,1,H,W]")
    dimension = int(images.shape[-2] * images.shape[-1])
    if images.shape[-2] != images.shape[-1] or basis.shape != (dimension, dimension):
        raise ValueError("basis must match square input images")
    basis = basis.to(device=images.device, dtype=images.dtype)
    coefficients = images.flatten(1) @ basis
    generator = torch.Generator(device=images.device).manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        coefficients.shape,
        generator=generator,
        device=images.device,
        dtype=torch.int64,
    )
    signs = signs.to(coefficients.dtype).mul_(2.0).sub_(1.0)
    if preserve_dc:
        signs[:, 0] = 1.0
    scrambled_coefficients = coefficients * signs
    scrambled = (scrambled_coefficients @ basis.T).reshape_as(images)
    return scrambled, coefficients, scrambled_coefficients


def _random_directions(
    dimension: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    directions = torch.randn(
        (int(dimension), int(count)),
        generator=generator,
        device=device,
    )
    return directions / directions.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _class_statistics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=1)
    mean_probability = probabilities.mean(dim=0)
    return {
        "classifier_accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
        "classifier_confidence": float(probabilities.max(dim=1).values.mean()),
        "predicted_class_entropy": float(
            -(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
        ),
    }


def grouped_coefficient_power(
    coefficients: torch.Tensor,
    group_index: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample mean squared coefficient in each group."""

    if coefficients.ndim != 2 or group_index.shape != (coefficients.shape[1],):
        raise ValueError("group_index must match coefficient dimension")
    group_index = group_index.to(coefficients.device)
    group_count = int(group_index.max()) + 1
    sums = torch.zeros(
        (len(coefficients), group_count),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    sums.scatter_add_(
        1,
        group_index[None].expand(len(coefficients), -1),
        coefficients.square(),
    )
    counts = torch.bincount(group_index, minlength=group_count).to(coefficients.dtype)
    return sums / counts[None]


@torch.no_grad()
def evaluate_scramble(
    reference: torch.Tensor,
    labels: torch.Tensor,
    scrambled: torch.Tensor,
    coefficients: torch.Tensor,
    scrambled_coefficients: torch.Tensor,
    classifier: torch.nn.Module,
    normalization: Mapping[str, float],
    *,
    band_count: int,
    projection_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, torch.Tensor]]:
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    reference_pixels = (reference * std + mean).clamp(0.0, 1.0)
    scrambled_unclipped_pixels = scrambled * std + mean
    scrambled_pixels = scrambled_unclipped_pixels.clamp(0.0, 1.0)
    reference_input = (reference_pixels - mean) / std
    scrambled_input = (scrambled_pixels - mean) / std
    reference_logits, reference_features = classifier(
        reference_input, return_features=True
    )
    unclipped_logits, unclipped_features = classifier(
        scrambled, return_features=True
    )
    scrambled_logits, scrambled_features = classifier(
        scrambled_input, return_features=True
    )
    pixel_directions = _random_directions(
        reference[0].numel(), projection_count, seed + 31, reference.device
    )
    feature_directions = _random_directions(
        reference_features.shape[1], projection_count, seed + 37, reference.device
    )
    rows = []
    for variant, logits in (
        ("reference", reference_logits),
        ("dct_sign_scrambled_unclipped", unclipped_logits),
        ("dct_sign_scrambled_clipped", scrambled_logits),
    ):
        row = {"variant": variant}
        row.update(_class_statistics(logits, labels))
        rows.append(row)

    coefficient_power_error = (
        coefficients.square() - scrambled_coefficients.square()
    ).abs()
    power_scale = coefficients.square().abs().max().clamp_min(1e-15)
    groups = radial_band_index(
        reference.shape[-1], int(band_count)
    ).flatten()
    reference_bands = grouped_coefficient_power(coefficients, groups)
    scrambled_bands = grouped_coefficient_power(scrambled_coefficients, groups)
    band_error = (reference_bands - scrambled_bands).abs()
    band_scale = reference_bands.abs().max().clamp_min(1e-15)
    audit = {
        "component_power_max_abs_error": float(coefficient_power_error.max()),
        "component_power_max_relative_error": float(
            coefficient_power_error.max() / power_scale
        ),
        "total_power_relative_error": float(
            abs(
                coefficients.square().sum()
                - scrambled_coefficients.square().sum()
            )
            / coefficients.square().sum().clamp_min(1e-15)
        ),
        "radial_band_power_max_abs_error": float(band_error.max()),
        "radial_band_power_max_relative_error": float(
            band_error.max() / band_scale
        ),
        "per_image_mean_max_abs_error": float(
            (reference.mean(dim=(1, 2, 3)) - scrambled.mean(dim=(1, 2, 3)))
            .abs()
            .max()
        ),
        "unclipped_pixel_fraction": float(
            (
                (scrambled_unclipped_pixels < 0.0)
                | (scrambled_unclipped_pixels > 1.0)
            )
            .float()
            .mean()
        ),
        "pixel_swd_after_decode": sliced_wasserstein(
            reference_pixels.flatten(1),
            scrambled_pixels.flatten(1),
            pixel_directions,
        ),
        "feature_swd_after_decode": sliced_wasserstein(
            reference_features,
            scrambled_features,
            feature_directions,
        ),
        "feature_fid_after_decode": frechet_distance(
            reference_features, scrambled_features
        ),
        "feature_swd_unclipped": sliced_wasserstein(
            reference_features,
            unclipped_features,
            feature_directions,
        ),
        "feature_fid_unclipped": frechet_distance(
            reference_features, unclipped_features
        ),
    }
    images = {
        "reference": reference_pixels[:16].cpu(),
        "dct_sign_scrambled": scrambled_pixels[:16].cpu(),
    }
    return pd.DataFrame(rows), audit, images


def _save_panel(images: Mapping[str, torch.Tensor], path: Path) -> None:
    count = min(16, len(images["reference"]))
    figure, axes = plt.subplots(2, count, figsize=(1.25 * count, 3.0), squeeze=False)
    for row, (name, values) in enumerate(images.items()):
        for column in range(count):
            axes[row, column].imshow(values[column, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].axis("off")
        axes[row, 0].set_title(name.replace("_", " "), loc="left", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_study(
    config: SignScrambleConfig = SignScrambleConfig(),
) -> tuple[pd.DataFrame, dict[str, float], Path | None]:
    dataset = config.dataset.strip().lower()
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {config.dataset}")
    configure_fp32(config.seed)
    device = resolve_device(config.device)
    loaded = load_small_image_tensors(
        dataset,
        config.data_root,
        config.train_size,
        config.test_size,
        config.seed,
        download=True,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    train_labels = loaded["train_labels"].to(device)
    test_labels = loaded["test_labels"].to(device)
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=config.classifier_epochs,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    basis = dct_pixel_basis(test.shape[-1]).to(device)
    scrambled, coefficients, scrambled_coefficients = dct_sign_scramble(
        test,
        basis,
        seed=config.seed + 101,
    )
    metrics, audit, images = evaluate_scramble(
        test,
        test_labels,
        scrambled,
        coefficients,
        scrambled_coefficients,
        classifier,
        loaded["normalization"],
        band_count=config.band_count,
        projection_count=config.projection_count,
        seed=config.seed,
    )
    audit["reference_classifier_accuracy_from_training"] = float(classifier_accuracy)

    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = (
            config.output_root.expanduser()
            / f"{dataset}_seed{config.seed}_{timestamp}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["data_root"] = str(config.data_root)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "semantic_metrics.csv", index=False)
        (result_dir / "spectral_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        _save_panel(images, result_dir / "comparison.png")
    return metrics, audit, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fashion_mnist")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    root = args.data_root or DATASETS[args.dataset][1]
    config = SignScrambleConfig(
        dataset=args.dataset,
        data_root=root,
        device=args.device,
        seed=args.seed,
        save=not args.no_save,
    )
    if args.quick:
        config = SignScrambleConfig(
            dataset=args.dataset,
            data_root=root,
            device=args.device,
            seed=args.seed,
            train_size=512,
            test_size=256,
            classifier_epochs=1,
            batch_size=128,
            save=not args.no_save,
        )
    metrics, audit, result_dir = run_study(config)
    print(f"result_dir={result_dir}")
    print(metrics.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

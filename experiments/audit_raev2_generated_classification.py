#!/usr/bin/env python3
"""Audit requested-class fidelity of paired RAEv2 sample archives.

The RAEv2 samplers in this repository assign label ``sample_id % 1000``.
This script evaluates those labels with an external ImageNet classifier.  The
result is a semantic diagnostic; it is not a replacement for FID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def parse_branch(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    path = Path(raw_path).expanduser().resolve()
    if not separator or not name or not path.is_file():
        raise argparse.ArgumentTypeError("branch must be NAME=/path/to/samples.npz")
    return name, path


def balanced_labels(sample_count: int, num_classes: int = 1000) -> np.ndarray:
    if sample_count <= 0 or num_classes <= 0:
        raise ValueError("sample count and number of classes must be positive")
    return np.arange(sample_count, dtype=np.int64) % num_classes


def summarize_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels have incompatible shapes")
    log_probabilities = logits.float().log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    target_log_probability = log_probabilities.gather(1, labels[:, None]).squeeze(1)
    target_probability = target_log_probability.exp()
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    predicted = logits.argmax(dim=1)
    histogram = torch.bincount(predicted, minlength=logits.shape[1]).float()
    predicted_marginal = histogram / histogram.sum()
    nonzero = predicted_marginal > 0
    return {
        "top1_accuracy": float(predicted.eq(labels).float().mean()),
        "top5_accuracy": float(top5.eq(labels[:, None]).any(dim=1).float().mean()),
        "target_probability_mean": float(target_probability.mean()),
        "target_probability_median": float(target_probability.median()),
        "target_log_probability_mean": float(target_log_probability.mean()),
        "maximum_probability_mean": float(probabilities.max(dim=1).values.mean()),
        "predictive_entropy_mean": float(
            -(probabilities * log_probabilities).sum(dim=1).mean()
        ),
        "occupied_top1_classes": float(nonzero.sum()),
        "top1_class_entropy": float(
            -(predicted_marginal[nonzero] * predicted_marginal[nonzero].log()).sum()
        ),
    }


def classifier_forward(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    images = F.interpolate(
        images,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return model((images - mean) / std)


def load_images(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["arr_0"]:
            raise ValueError(f"{path} must contain only arr_0")
        images = archive["arr_0"]
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"unexpected image archive shape or dtype: {path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    branches = dict(args.branch)
    if len(branches) != len(args.branch) or args.baseline not in branches:
        raise ValueError("branch names must be unique and include the baseline")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    images_by_branch = {name: load_images(path) for name, path in branches.items()}
    sample_counts = {len(images) for images in images_by_branch.values()}
    if len(sample_counts) != 1:
        raise ValueError("paired branches must contain the same number of samples")
    sample_count = sample_counts.pop()
    labels = torch.from_numpy(balanced_labels(sample_count))

    device = torch.device(args.device)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights).to(device).eval().requires_grad_(False)
    rows: list[dict[str, float | str]] = []
    per_image: list[pd.DataFrame] = []
    with torch.inference_mode():
        for name, images in images_by_branch.items():
            parts = []
            for start in range(0, sample_count, args.batch_size):
                batch = (
                    torch.from_numpy(images[start : start + args.batch_size])
                    .to(device=device, dtype=torch.float32)
                    .permute(0, 3, 1, 2)
                    .div_(255.0)
                )
                parts.append(classifier_forward(model, batch).cpu())
            logits = torch.cat(parts)
            rows.append({"branch": name, **summarize_logits(logits, labels)})
            log_probabilities = logits.float().log_softmax(dim=1)
            probabilities = log_probabilities.exp()
            per_image.append(
                pd.DataFrame(
                    {
                        "branch": name,
                        "sample_id": np.arange(sample_count),
                        "requested_class": labels.numpy(),
                        "predicted_class": logits.argmax(dim=1).numpy(),
                        "target_probability": probabilities.gather(
                            1, labels[:, None]
                        ).squeeze(1).numpy(),
                        "target_log_probability": log_probabilities.gather(
                            1, labels[:, None]
                        ).squeeze(1).numpy(),
                    }
                )
            )

    frame = pd.DataFrame(rows)
    baseline = frame.set_index("branch").loc[args.baseline]
    for column in (
        "top1_accuracy",
        "top5_accuracy",
        "target_probability_mean",
        "target_log_probability_mean",
        "predictive_entropy_mean",
    ):
        frame[f"delta_{column}"] = frame[column] - float(baseline[column])

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "classification_summary.csv", index=False)
    pd.concat(per_image, ignore_index=True).to_csv(
        output_dir / "classification_per_image.csv", index=False
    )
    (output_dir / "classification_summary.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

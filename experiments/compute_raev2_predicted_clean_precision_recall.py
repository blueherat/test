"""Compute post-hoc improved precision/recall for predicted-clean images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_decoded_distribution_audit import _load_feature_shards, time_suffix
from experiments.run_raev2_predicted_clean_audit import (
    HEADS,
    STATE_BRANCHES,
    _load_prediction_shards,
    condition_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--real-features",
        type=Path,
        default=Path(
            "/home/zhoushunyu/.cache/torch/fidelity_cache/"
            "raev2_imagenet256_virtual_reference-inception-v3-compat-features-2048.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-samples", type=int, default=5000)
    parser.add_argument("--neighborhood", type=int, default=3)
    parser.add_argument("--distance-batch", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9187)
    return parser.parse_args()


def manifold_radii(
    features: torch.Tensor,
    *,
    neighborhood: int,
    batch_size: int,
) -> torch.Tensor:
    """Distance to each point's k-th non-self neighbor."""

    if features.ndim != 2 or features.shape[0] <= neighborhood:
        raise ValueError("not enough feature vectors for the requested neighborhood")
    if neighborhood <= 0 or batch_size <= 0:
        raise ValueError("neighborhood and batch size must be positive")
    values = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            distances = torch.cdist(features[start : start + batch_size], features)
            values.append(distances.kthvalue(neighborhood + 1, dim=1).values)
    return torch.cat(values)


def manifold_coverage(
    queries: torch.Tensor,
    centers: torch.Tensor,
    center_radii: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    if queries.ndim != 2 or centers.ndim != 2 or queries.shape[1] != centers.shape[1]:
        raise ValueError("query and center feature dimensions differ")
    if center_radii.shape != (centers.shape[0],):
        raise ValueError("center radius shape differs from center count")
    covered = 0
    with torch.inference_mode():
        for start in range(0, queries.shape[0], batch_size):
            distances = torch.cdist(queries[start : start + batch_size], centers)
            covered += int((distances <= center_radii.unsqueeze(0)).any(dim=1).sum().item())
    return covered / queries.shape[0]


def precision_recall(
    generated: torch.Tensor,
    reference: torch.Tensor,
    reference_radii: torch.Tensor,
    *,
    neighborhood: int,
    batch_size: int,
) -> tuple[float, float, torch.Tensor]:
    generated_radii = manifold_radii(
        generated, neighborhood=neighborhood, batch_size=batch_size
    )
    precision = manifold_coverage(
        generated, reference, reference_radii, batch_size=batch_size
    )
    recall = manifold_coverage(
        reference, generated, generated_radii, batch_size=batch_size
    )
    return precision, recall, generated_radii


def pr_effect_rows(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("head_on_full_state", "ig_on_full", "full_on_full"),
        ("head_on_ig_state", "ig_on_ig", "full_on_ig"),
        ("history_under_full_head", "full_on_ig", "full_on_full"),
        ("history_under_ig_head", "ig_on_ig", "ig_on_full"),
        ("on_policy_total", "ig_on_ig", "full_on_full"),
    )
    rows: list[dict[str, Any]] = []
    metrics = (
        "precision_real",
        "recall_real",
        "precision_reconstruction",
        "recall_reconstruction",
    )
    for requested_time, group in frame.groupby("requested_time"):
        indexed = group.set_index("condition")
        for effect, positive, negative in comparisons:
            row: dict[str, Any] = {
                "requested_time": requested_time,
                "actual_time": indexed.loc[positive, "actual_time"],
                "effect": effect,
                "positive_condition": positive,
                "negative_condition": negative,
            }
            for metric in metrics:
                row[f"{metric}_delta"] = (
                    indexed.loc[positive, metric] - indexed.loc[negative, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["actual_time", "effect"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    if args.reference_samples <= args.neighborhood:
        raise ValueError("reference sample count must exceed neighborhood")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "precision_recall"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "raev2_predicted_clean_2x2_v1":
        raise ValueError("input is not a predicted-clean 2x2 audit")
    if manifest.get("inception_feature") != "2048":
        raise ValueError("precision/recall requires 2048-dimensional Inception features")
    world_size = int(manifest["world_size"])
    ids, labels, test_mask, predictions, _ = _load_prediction_shards(run_dir, world_size)
    reference_dir = Path(manifest["decoded_reference_run"])
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    ref_ids, ref_labels, ref_test, reference_features, _ = _load_feature_shards(
        reference_dir, int(reference_manifest["world_size"])
    )
    if not np.array_equal(ids, ref_ids):
        raise RuntimeError("prediction and reconstruction IDs differ")
    if not np.array_equal(labels, ref_labels) or not np.array_equal(test_mask, ref_test):
        raise RuntimeError("prediction and reconstruction protocol differs")
    reconstruction_np = reference_features[f"feat_p_{time_suffix(0.0)}"]
    real = torch.load(
        args.real_features.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(real, torch.Tensor) or real.ndim != 2:
        raise ValueError("real feature cache has an unexpected format")
    rng = np.random.default_rng(args.seed)
    real_indices = rng.choice(real.shape[0], args.reference_samples, replace=False)
    reconstruction_indices = rng.choice(
        reconstruction_np.shape[0], args.reference_samples, replace=False
    )
    device = torch.device(args.device)
    real_features = real[real_indices].float().to(device=device)
    reconstruction = torch.from_numpy(
        reconstruction_np[reconstruction_indices].astype(np.float32, copy=False)
    ).to(device=device)
    real_radii = manifold_radii(
        real_features,
        neighborhood=args.neighborhood,
        batch_size=args.distance_batch,
    )
    reconstruction_radii = manifold_radii(
        reconstruction,
        neighborhood=args.neighborhood,
        batch_size=args.distance_batch,
    )

    conditions = [
        condition_name(head, state) for state in STATE_BRANCHES for head in HEADS
    ]
    rows = []
    for requested_time in manifest["requested_times"]:
        suffix = time_suffix(float(requested_time))
        actual_time = next(
            float(row["actual_time"])
            for row in manifest["matched_times"]
            if float(row["requested_time"]) == float(requested_time)
        )
        for condition in conditions:
            generated = torch.from_numpy(
                predictions[f"feat_{condition}_{suffix}"].astype(np.float32, copy=False)
            ).to(device=device)
            generated_radii = manifold_radii(
                generated,
                neighborhood=args.neighborhood,
                batch_size=args.distance_batch,
            )
            precision_real = manifold_coverage(
                generated, real_features, real_radii, batch_size=args.distance_batch
            )
            recall_real = manifold_coverage(
                real_features, generated, generated_radii, batch_size=args.distance_batch
            )
            precision_reconstruction = manifold_coverage(
                generated,
                reconstruction,
                reconstruction_radii,
                batch_size=args.distance_batch,
            )
            recall_reconstruction = manifold_coverage(
                reconstruction,
                generated,
                generated_radii,
                batch_size=args.distance_batch,
            )
            rows.append(
                {
                    "requested_time": float(requested_time),
                    "actual_time": actual_time,
                    "condition": condition,
                    "precision_real": precision_real,
                    "recall_real": recall_real,
                    "precision_reconstruction": precision_reconstruction,
                    "recall_reconstruction": recall_reconstruction,
                }
            )
            print(f"[precision/recall] t={requested_time:g} {condition}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["actual_time", "condition"], ascending=[False, True])
    effects = pr_effect_rows(frame)
    frame.to_csv(output_dir / "predicted_clean_precision_recall.csv", index=False)
    effects.to_csv(output_dir / "predicted_clean_precision_recall_effects.csv", index=False)
    output_manifest = {
        "protocol": "raev2_predicted_clean_precision_recall_v1",
        "source_run": str(run_dir),
        "real_features": str(args.real_features.expanduser().resolve()),
        "reference_samples": args.reference_samples,
        "neighborhood": args.neighborhood,
        "seed": args.seed,
        "definition": "improved precision/recall using k-nearest-neighbor manifolds",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()

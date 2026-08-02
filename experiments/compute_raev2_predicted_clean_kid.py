"""Compute post-hoc KID for a completed predicted-clean 2x2 audit."""

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
    parser.add_argument("--subsets", type=int, default=50)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7341)
    return parser.parse_args()


def polynomial_mmd_unbiased(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased degree-3 polynomial-kernel MMD used by KID."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("KID inputs must be feature matrices with matching dimensions")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("KID requires at least two samples per set")
    dimension = float(x.shape[1])
    k_xx = (x @ x.T / dimension + 1.0).pow(3)
    k_yy = (y @ y.T / dimension + 1.0).pow(3)
    k_xy = (x @ y.T / dimension + 1.0).pow(3)
    m = x.shape[0]
    n = y.shape[0]
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
    return xx + yy - 2.0 * k_xy.mean()


def kid_subsets(
    first: np.ndarray,
    second: np.ndarray,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    if subsets <= 0 or subset_size < 2:
        raise ValueError("KID subset count and size must be positive")
    if subset_size > min(first.shape[0], second.shape[0]):
        raise ValueError("KID subset is larger than an input sample set")
    rng = np.random.default_rng(seed)
    values = np.empty(subsets, dtype=np.float64)
    first_tensor = torch.from_numpy(first.astype(np.float32, copy=False))
    second_tensor = torch.from_numpy(second.astype(np.float32, copy=False))
    with torch.inference_mode():
        for repeat in range(subsets):
            first_indices = rng.choice(first.shape[0], subset_size, replace=False)
            second_indices = rng.choice(second.shape[0], subset_size, replace=False)
            x = first_tensor[first_indices].to(device=device)
            y = second_tensor[second_indices].to(device=device)
            values[repeat] = float(polynomial_mmd_unbiased(x, y).cpu().item())
    standard_error = float(values.std(ddof=1) / np.sqrt(subsets)) if subsets > 1 else float("nan")
    return float(values.mean()), standard_error, values


def kid_effect_rows(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("head_on_full_state", "ig_on_full", "full_on_full"),
        ("head_on_ig_state", "ig_on_ig", "full_on_ig"),
        ("history_under_full_head", "full_on_ig", "full_on_full"),
        ("history_under_ig_head", "ig_on_ig", "ig_on_full"),
        ("on_policy_total", "ig_on_ig", "full_on_full"),
    )
    rows: list[dict[str, Any]] = []
    for requested_time, group in frame.groupby("requested_time"):
        indexed = group.set_index("condition")
        for effect, positive, negative in comparisons:
            rows.append(
                {
                    "requested_time": requested_time,
                    "actual_time": indexed.loc[positive, "actual_time"],
                    "effect": effect,
                    "positive_condition": positive,
                    "negative_condition": negative,
                    "kid_real_delta": (
                        indexed.loc[positive, "kid_real"]
                        - indexed.loc[negative, "kid_real"]
                    ),
                    "kid_reconstruction_delta": (
                        indexed.loc[positive, "kid_reconstruction"]
                        - indexed.loc[negative, "kid_reconstruction"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["actual_time", "effect"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "kid"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "raev2_predicted_clean_2x2_v1":
        raise ValueError("input is not a predicted-clean 2x2 audit")
    if manifest.get("inception_feature") != "2048":
        raise ValueError("standard KID requires 2048-dimensional Inception features")
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
    reconstruction = reference_features[f"feat_p_{time_suffix(0.0)}"]
    real = torch.load(
        args.real_features.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(real, torch.Tensor) or real.ndim != 2:
        raise ValueError("real Inception feature cache has an unexpected format")
    real_features = real.float().cpu().numpy()
    device = torch.device(args.device)
    conditions = [
        condition_name(head, state) for state in STATE_BRANCHES for head in HEADS
    ]
    rows = []
    replicate_archive: dict[str, np.ndarray] = {}
    for time_index, requested_time in enumerate(manifest["requested_times"]):
        suffix = time_suffix(float(requested_time))
        actual_time = next(
            float(row["actual_time"])
            for row in manifest["matched_times"]
            if float(row["requested_time"]) == float(requested_time)
        )
        for condition in conditions:
            generated = predictions[f"feat_{condition}_{suffix}"]
            # Reuse subset IDs across all four 2x2 conditions at a time point.
            # This makes condition deltas paired and substantially less noisy.
            base_seed = args.seed + 10_000 * time_index
            kid_real, kid_real_se, real_values = kid_subsets(
                generated,
                real_features,
                subsets=args.subsets,
                subset_size=args.subset_size,
                seed=base_seed,
                device=device,
            )
            kid_recon, kid_recon_se, recon_values = kid_subsets(
                generated,
                reconstruction,
                subsets=args.subsets,
                subset_size=args.subset_size,
                seed=base_seed + 1,
                device=device,
            )
            replicate_archive[f"real_{condition}_{suffix}"] = real_values
            replicate_archive[f"reconstruction_{condition}_{suffix}"] = recon_values
            rows.append(
                {
                    "requested_time": float(requested_time),
                    "actual_time": actual_time,
                    "condition": condition,
                    "kid_real": kid_real,
                    "kid_real_x1000": 1000.0 * kid_real,
                    "kid_real_subset_std": float(real_values.std(ddof=1)),
                    "kid_real_standard_error": kid_real_se,
                    "kid_reconstruction": kid_recon,
                    "kid_reconstruction_x1000": 1000.0 * kid_recon,
                    "kid_reconstruction_subset_std": float(recon_values.std(ddof=1)),
                    "kid_reconstruction_standard_error": kid_recon_se,
                }
            )
            print(f"[KID] t={requested_time:g} {condition}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["actual_time", "condition"], ascending=[False, True])
    effects = kid_effect_rows(frame)
    frame.to_csv(output_dir / "predicted_clean_kid.csv", index=False)
    effects.to_csv(output_dir / "predicted_clean_kid_effects.csv", index=False)
    np.savez_compressed(output_dir / "kid_subset_values.npz", **replicate_archive)
    kid_manifest = {
        "protocol": "raev2_predicted_clean_kid_v1",
        "source_run": str(run_dir),
        "real_features": str(args.real_features.expanduser().resolve()),
        "kernel": "((x dot y) / 2048 + 1)^3",
        "estimator": "unbiased subset MMD",
        "subsets": args.subsets,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "note": "Unbiased KID estimates may be slightly negative.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(kid_manifest, indent=2), encoding="utf-8"
    )
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze latent and ADM-feature distributions from the SiT control audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from experiments.imagenet100_sit_terminal_distribution import (
        gaussian_frechet_distance,
        linear_rbf_mmd2,
        sliced_wasserstein_distance,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from imagenet100_sit_terminal_distribution import (
        gaussian_frechet_distance,
        linear_rbf_mmd2,
        sliced_wasserstein_distance,
    )
    from train_imagenet100_sit_flow import atomic_json_dump


DEFAULT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "terminal_distribution_audit_800k_v1/seed0"
)
DEFAULT_C2ST_TIMES = (0.2, 0.4, 0.6, 0.8, 0.95, 1.0)


def _parse_times(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item) for item in value.split(",") if item.strip())
    if not parsed or not all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("expected finite comma-separated times")
    return parsed


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _rbf_bandwidth(features: np.ndarray, *, max_samples: int = 512) -> float:
    features = np.asarray(features, dtype=np.float64)
    selected = features[: min(max_samples, len(features))]
    differences = selected[:, None, :] - selected[None, :, :]
    squared = np.square(differences).sum(axis=2)
    upper = squared[np.triu_indices(len(selected), k=1)]
    positive = upper[upper > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(positive)))


def _bootstrap_distribution_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    bandwidth: float,
    reps: int,
    rng: np.random.Generator,
    null_split: bool,
    include_frechet: bool,
) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("bootstrap feature arrays must have equal shapes")
    sample_count = len(first)
    half = sample_count // 2
    if half < 4:
        raise ValueError("distribution bootstrap requires at least eight samples")
    swd_values: list[float] = []
    mmd_values: list[float] = []
    frechet_values: list[float] = []
    for _ in range(reps):
        if null_split:
            permutation = rng.permutation(sample_count)
            first_indices = permutation[:half]
            second_indices = permutation[half : 2 * half]
            first_sample = first[first_indices]
            second_sample = first[second_indices]
        else:
            first_indices = rng.choice(sample_count, size=half, replace=False)
            second_indices = rng.choice(sample_count, size=half, replace=False)
            first_sample = first[first_indices]
            second_sample = second[second_indices]
        swd_values.append(sliced_wasserstein_distance(first_sample, second_sample))
        mmd_values.append(
            linear_rbf_mmd2(
                first_sample,
                second_sample,
                bandwidth=bandwidth,
            )
        )
        if include_frechet:
            frechet_values.append(
                gaussian_frechet_distance(first_sample, second_sample)
            )
    output: dict[str, float] = {}
    for name, values in (("swd", swd_values), ("linear_mmd2", mmd_values)):
        for statistic, value in _summary(values).items():
            output[f"{name}_{statistic}"] = value
    if include_frechet:
        for statistic, value in _summary(frechet_values).items():
            output[f"feature_frechet_{statistic}"] = value
    return output


def _c2st_auc(
    first: np.ndarray,
    second: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
    max_groups: int | None = None,
) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("C2ST arrays must have equal shapes")
    sample_count = len(first)
    group_count = sample_count if max_groups is None else min(max_groups, sample_count)
    if group_count < 4:
        raise ValueError("C2ST requires at least four paired groups")
    auc_values: list[float] = []
    for _ in range(reps):
        group_order = rng.permutation(sample_count)[:group_count]
        split = max(1, int(0.7 * group_count))
        train_groups = group_order[:split]
        test_groups = group_order[split:]
        train_x = np.concatenate((first[train_groups], second[train_groups]))
        train_y = np.concatenate(
            (np.zeros(len(train_groups)), np.ones(len(train_groups)))
        )
        test_x = np.concatenate((first[test_groups], second[test_groups]))
        test_y = np.concatenate((np.zeros(len(test_groups)), np.ones(len(test_groups))))
        scaler = StandardScaler().fit(train_x)
        classifier = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
        ).fit(scaler.transform(train_x), train_y)
        probabilities = classifier.predict_proba(scaler.transform(test_x))[:, 1]
        auc = float(roc_auc_score(test_y, probabilities))
        auc_values.append(max(auc, 1.0 - auc))
    return {
        f"c2st_auc_{statistic}": value
        for statistic, value in _summary(auc_values).items()
    }


def _c2st_split_null(
    features: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_count = len(features)
    half = sample_count // 2
    if half < 4:
        raise ValueError("C2ST split null requires at least eight samples")
    auc_values = []
    for _ in range(reps):
        permutation = rng.permutation(sample_count)[: 2 * half]
        result = _c2st_auc(
            features[permutation[:half]],
            features[permutation[half:]],
            reps=1,
            rng=rng,
        )
        auc_values.append(result["c2st_auc_mean"])
    return {
        f"c2st_auc_{statistic}": value
        for statistic, value in _summary(auc_values).items()
    }


def _random_projection(
    dimension: int,
    output_dimension: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if output_dimension > dimension:
        raise ValueError("projection dimension cannot exceed latent dimension")
    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(
        dimension,
        output_dimension,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    orthonormal, _ = torch.linalg.qr(matrix, mode="reduced")
    return orthonormal


def _trajectory_path_rows(
    trajectories: np.ndarray,
    branch_names: tuple[str, ...],
) -> list[dict[str, object]]:
    baseline_endpoint = np.asarray(trajectories[0, -1], dtype=np.float32)
    rows: list[dict[str, object]] = []
    for branch_index, name in enumerate(branch_names):
        branch = np.asarray(trajectories[branch_index], dtype=np.float32)
        increments = branch[1:] - branch[:-1]
        chord_lengths = np.sqrt(np.square(increments).mean(axis=(2, 3, 4))).sum(axis=0)
        endpoint_shift = np.sqrt(
            np.square(branch[-1] - baseline_endpoint).mean(axis=(1, 2, 3))
        )
        row: dict[str, object] = {"condition": name}
        for metric, values in (
            ("snapshot_path_length_rms", chord_lengths),
            ("endpoint_paired_rms", endpoint_shift),
        ):
            for statistic, value in _summary(values).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def _latent_distribution_rows(
    trajectories: np.ndarray,
    branch_names: tuple[str, ...],
    times: tuple[float, ...],
    *,
    projection_dim: int,
    bootstrap_reps: int,
    c2st_reps: int,
    c2st_times: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    branch_count, _, sample_count = trajectories.shape[:3]
    latent_dimension = int(np.prod(trajectories.shape[3:]))
    projection = _random_projection(
        latent_dimension,
        projection_dim,
        seed=seed,
        device=device,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    pairs = list(itertools.combinations(range(branch_count), 2))
    for time_index, time_value in enumerate(times):
        states = torch.from_numpy(
            np.array(trajectories[:, time_index], dtype=np.float32, copy=True)
        ).to(device)
        flat = states.flatten(2)
        projected = (flat.reshape(branch_count * sample_count, -1) @ projection).reshape(
            branch_count,
            sample_count,
            projection_dim,
        ).cpu().numpy()
        baseline_scale = float(np.sqrt(np.var(projected[0], axis=0).mean()))
        bandwidth = _rbf_bandwidth(projected[0])
        for branch_index, name in enumerate(branch_names):
            row: dict[str, object] = {
                "time": float(time_value),
                "condition_a": name,
                "condition_b": name,
                "comparison": "within_condition_split",
                "paired_rms": 0.0,
                "baseline_projection_scale": baseline_scale,
                "rbf_bandwidth": bandwidth,
            }
            row.update(
                _bootstrap_distribution_metrics(
                    projected[branch_index],
                    projected[branch_index],
                    bandwidth=bandwidth,
                    reps=bootstrap_reps,
                    rng=rng,
                    null_split=True,
                    include_frechet=False,
                )
            )
            if any(math.isclose(time_value, target, abs_tol=1e-8) for target in c2st_times):
                row.update(
                    _c2st_split_null(
                        projected[branch_index],
                        reps=c2st_reps,
                        rng=rng,
                    )
                )
            else:
                row.update({"c2st_auc_mean": float("nan"), "c2st_auc_std": float("nan")})
            rows.append(row)
        for first_index, second_index in pairs:
            paired_rms = float(
                torch.sqrt(
                    torch.square(flat[first_index] - flat[second_index]).mean()
                ).item()
            )
            row = {
                "time": float(time_value),
                "condition_a": branch_names[first_index],
                "condition_b": branch_names[second_index],
                "comparison": "cross_condition",
                "paired_rms": paired_rms,
                "baseline_projection_scale": baseline_scale,
                "rbf_bandwidth": bandwidth,
            }
            row.update(
                _bootstrap_distribution_metrics(
                    projected[first_index],
                    projected[second_index],
                    bandwidth=bandwidth,
                    reps=bootstrap_reps,
                    rng=rng,
                    null_split=False,
                    include_frechet=False,
                )
            )
            if any(math.isclose(time_value, target, abs_tol=1e-8) for target in c2st_times):
                row.update(
                    _c2st_auc(
                        projected[first_index],
                        projected[second_index],
                        reps=c2st_reps,
                        rng=rng,
                        max_groups=sample_count // 2,
                    )
                )
            else:
                row.update({"c2st_auc_mean": float("nan"), "c2st_auc_std": float("nan")})
            rows.append(row)
        del states, flat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[latent metrics] t={time_value:g}", flush=True)
    return rows


def _feature_distribution_rows(
    root: Path,
    branch_names: tuple[str, ...],
    *,
    feature_dim: int,
    bootstrap_reps: int,
    c2st_reps: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    activations: dict[str, np.ndarray] = {}
    quality_rows: list[dict[str, object]] = []
    for name in branch_names:
        activation_path = root / "adm_activations" / f"{name}.npz"
        result_path = root / "adm_results" / f"{name}.json"
        if not activation_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"missing ADM artifacts for {name}")
        with np.load(activation_path) as payload:
            activations[name] = np.asarray(payload["pool_3"], dtype=np.float32)
        result = _load_json(result_path)
        quality_rows.append(
            {
                "condition": name,
                "fid": float(result["fid"]),
                "sfid": float(result["sfid"]),
                "inception_score": float(result["inception_score"]),
            }
        )
    sample_counts = {len(value) for value in activations.values()}
    if len(sample_counts) != 1:
        raise ValueError("ADM activation counts differ between conditions")
    pooled = np.concatenate([activations[name] for name in branch_names])
    components = min(feature_dim, pooled.shape[0] - 1, pooled.shape[1])
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=seed,
    ).fit(pooled)
    reduced = {name: pca.transform(activations[name]) for name in branch_names}
    bandwidth = _rbf_bandwidth(reduced["baseline"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for name in branch_names:
        row: dict[str, object] = {
            "condition_a": name,
            "condition_b": name,
            "comparison": "within_condition_split",
            "pca_components": components,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "rbf_bandwidth": bandwidth,
        }
        row.update(
            _bootstrap_distribution_metrics(
                reduced[name],
                reduced[name],
                bandwidth=bandwidth,
                reps=bootstrap_reps,
                rng=rng,
                null_split=True,
                include_frechet=True,
            )
        )
        row.update(
            _c2st_split_null(
                reduced[name],
                reps=c2st_reps,
                rng=rng,
            )
        )
        rows.append(row)
    for first_name, second_name in itertools.combinations(branch_names, 2):
        row = {
            "condition_a": first_name,
            "condition_b": second_name,
            "comparison": "cross_condition",
            "pca_components": components,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "rbf_bandwidth": bandwidth,
        }
        row.update(
            _bootstrap_distribution_metrics(
                reduced[first_name],
                reduced[second_name],
                bandwidth=bandwidth,
                reps=bootstrap_reps,
                rng=rng,
                null_split=False,
                include_frechet=True,
            )
        )
        row.update(
            _c2st_auc(
                reduced[first_name],
                reduced[second_name],
                reps=c2st_reps,
                rng=rng,
                max_groups=len(reduced[first_name]) // 2,
            )
        )
        rows.append(row)
    return rows, quality_rows


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("format") != "eqvae_imagenet100_sit_terminal_distribution_audit_v1":
        raise ValueError("unsupported terminal-distribution audit manifest")
    branch_names = tuple(manifest["branches"])
    times = tuple(float(value) for value in manifest["times"])
    trajectories = np.load(root / "trajectory_snapshots_fp16.npy", mmap_mode="r")
    expected_shape = tuple(manifest["trajectory_shape"])
    if trajectories.shape != expected_shape:
        raise ValueError("trajectory shape does not match manifest")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA analysis was requested but unavailable")
        torch.cuda.set_device(device)
    path_rows = _trajectory_path_rows(trajectories, branch_names)
    _write_csv(path_rows, root / "trajectory_path_summary.csv")
    latent_rows = _latent_distribution_rows(
        trajectories,
        branch_names,
        times,
        projection_dim=args.latent_projection_dim,
        bootstrap_reps=args.bootstrap_reps,
        c2st_reps=args.c2st_reps,
        c2st_times=args.c2st_times,
        seed=args.seed,
        device=device,
    )
    _write_csv(latent_rows, root / "latent_distribution_pairwise.csv")
    feature_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    if not args.skip_features:
        feature_rows, quality_rows = _feature_distribution_rows(
            root,
            branch_names,
            feature_dim=args.feature_projection_dim,
            bootstrap_reps=args.bootstrap_reps,
            c2st_reps=args.c2st_reps,
            seed=args.seed,
        )
        _write_csv(feature_rows, root / "endpoint_feature_pairwise.csv")
        _write_csv(quality_rows, root / "endpoint_quality.csv")
    summary = {
        "format": "eqvae_imagenet100_sit_terminal_distribution_analysis_v1",
        "root": str(root),
        "branches": list(branch_names),
        "times": list(times),
        "num_samples": int(manifest["num_samples"]),
        "latent_projection_dim": int(args.latent_projection_dim),
        "feature_projection_dim": int(args.feature_projection_dim),
        "bootstrap_reps": int(args.bootstrap_reps),
        "c2st_reps": int(args.c2st_reps),
        "c2st_groups_per_class": int(trajectories.shape[2] // 2),
        "c2st_times": list(args.c2st_times),
        "seed": int(args.seed),
        "rows": {
            "trajectory_path": len(path_rows),
            "latent_distribution": len(latent_rows),
            "endpoint_feature": len(feature_rows),
            "endpoint_quality": len(quality_rows),
        },
    }
    atomic_json_dump(summary, root / "analysis_manifest.json")
    (root / "ANALYSIS_COMPLETE").touch()
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--latent-projection-dim", type=int, default=64)
    parser.add_argument("--feature-projection-dim", type=int, default=128)
    parser.add_argument("--bootstrap-reps", type=int, default=8)
    parser.add_argument("--c2st-reps", type=int, default=3)
    parser.add_argument(
        "--c2st-times",
        type=_parse_times,
        default=DEFAULT_C2ST_TIMES,
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

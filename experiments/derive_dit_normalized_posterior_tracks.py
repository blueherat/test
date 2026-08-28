#!/usr/bin/env python3
"""Derive dimension-stable posterior-direction tracks without visual labels.

The source posterior product stores the exact same-covariance path quantities
for a weak conditional CFG=1 branch.  Its historical ``full_D_nats`` name is
slightly misleading: the stored value is

    K_k = 0.5 * ||Sigma_k^{-1/2}(mu_Q - mu_P)||^2 = D_k / 2.

This analysis keeps the predictable separation strength ``D/d`` apart from
the realized standardized alignment

    Z_k = (Delta log LR_k + K_k) / sqrt(2 K_k).

Under the actually implemented P transition, Z_k is conditionally N(0, 1).
The program is deliberately label-free: it accepts no review, consensus,
candidate-score, threshold, alert, or endpoint-image input.  It publishes a
small q2 (sampling k=100..149) feature set for later exploratory evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
STATUS = "COMPLETE_LABEL_FREE_NORMALIZED_POSTERIOR_DIRECTION_ANALYSIS"
EXPERIMENT = "dit_normalized_posterior_direction_label_free"
SOURCE_STATUS = "COMPLETE_LABEL_FREE_SUPPLEMENTARY_ANALYSIS"
Q_NAME = "weak_conditional_cfg1"
PHASE_NAME = "q2"
PHASE_START = 100
PHASE_STOP = 150
PHASE_LENGTH = PHASE_STOP - PHASE_START
STOCHASTIC_STEPS = 249
GUIDED_DIMENSION = 3 * 32 * 32
TILE_COUNT = 16
TILE_DIMENSION = 3 * 8 * 8
MAD_FACTOR = 1.4826
MAD_FLOOR = 1e-6
POSITIVE_FLOOR = 1e-300

IDENTIFIER_ARRAYS = ("sample_index", "global_seed", "class_slot", "class_id")
FULL_K = f"{Q_NAME}_full_D_nats"
FULL_INCREMENT = f"{Q_NAME}_full_log_lr_increment"
TILE_K = f"{Q_NAME}_tile4x4_tile_component_D"
TILE_INCREMENT = f"{Q_NAME}_tile4x4_tile_component_increment"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def require_identity(value: dict[str, Any], description: str) -> str:
    payload = dict(value)
    observed = payload.pop("identity_sha256", None)
    expected = canonical_sha256(payload)
    if not isinstance(observed, str) or observed != expected:
        raise RuntimeError(f"{description} canonical identity failed")
    return observed


def validate_source(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"posterior source must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    summary_path = root / "summary.json"
    inventory_path = root / "source_inventory.json"
    for path in (
        manifest_path,
        completion_path,
        summary_path,
        inventory_path,
        root / "time_series.npz",
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source member is missing or indirect: {path}")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    summary = load_json(summary_path)
    identity = require_identity(manifest, "posterior source manifest")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or completion.get("summary_file_sha256") != sha256_file(summary_path)
        or summary.get("status") != SOURCE_STATUS
        or summary.get("supervision_audit", {}).get("labels_read_or_emitted") is not False
        or summary.get("supervision_audit", {}).get("auc_computed") is not False
    ):
        raise RuntimeError("posterior source is not a complete label-free product")
    members = {str(item.get("name")): item for item in manifest.get("files", [])}
    required_members = {"time_series.npz", "summary.json", "feature_catalog.csv"}
    if not required_members.issubset(members):
        raise RuntimeError("posterior source manifest lacks required members")
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"posterior source member changed: {path}")
    trace_identities = manifest.get("trace_identity_sha256_ordered")
    if (
        not isinstance(trace_identities, list)
        or not trace_identities
        or not all(isinstance(value, str) and len(value) == 64 for value in trace_identities)
        or manifest.get("source_inventory_sha256") != sha256_file(inventory_path)
    ):
        raise RuntimeError("posterior source lacks immutable trace/source-inventory lineage")
    with np.load(root / "time_series.npz", allow_pickle=False) as archive:
        required = {*IDENTIFIER_ARRAYS, FULL_K, FULL_INCREMENT, TILE_K, TILE_INCREMENT}
        if not required.issubset(archive.files):
            raise RuntimeError(f"posterior time series lacks arrays: {sorted(required-set(archive.files))}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    n = int(summary.get("sample_count", -1))
    if n <= 0 or any(arrays[name].shape != (n,) for name in IDENTIFIER_ARRAYS):
        raise RuntimeError("posterior identifier arrays have wrong shape")
    if arrays[FULL_K].shape != (n, STOCHASTIC_STEPS):
        raise RuntimeError("full K has wrong shape")
    if arrays[FULL_INCREMENT].shape != (n, STOCHASTIC_STEPS):
        raise RuntimeError("full LR increment has wrong shape")
    if arrays[TILE_K].shape != (n, STOCHASTIC_STEPS, TILE_COUNT):
        raise RuntimeError("tile K has wrong shape")
    if arrays[TILE_INCREMENT].shape != (n, STOCHASTIC_STEPS, TILE_COUNT):
        raise RuntimeError("tile LR increment has wrong shape")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("posterior source contains non-finite values")
    keys = np.stack((arrays["global_seed"], arrays["class_id"]), axis=1)
    if len(np.unique(keys, axis=0)) != n:
        raise RuntimeError("posterior source sample keys are not unique")
    lineage = {
        "root": str(root),
        "manifest_identity_sha256": identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "time_series_file_sha256": sha256_file(root / "time_series.npz"),
        "source_inventory_file_sha256": sha256_file(inventory_path),
        "source_inventory_path": str(inventory_path),
        "trace_identity_sha256_ordered": trace_identities,
        "imported_validation_helper_sha256": manifest.get(
            "imported_validation_helper_sha256"
        ),
        "ordered_classes": summary.get("ordered_classes"),
        "ordered_seeds": summary.get("ordered_seeds"),
        "sample_count": n,
    }
    return lineage, arrays


def derive_tracks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    k = arrays[FULL_K].astype(np.float64, copy=False)
    increment = arrays[FULL_INCREMENT].astype(np.float64, copy=False)
    tile_k = arrays[TILE_K].astype(np.float64, copy=False)
    tile_increment = arrays[TILE_INCREMENT].astype(np.float64, copy=False)
    if np.any(k <= 0.0) or np.any(tile_k <= 0.0):
        raise RuntimeError("normalized direction is undefined for non-positive K")
    if not np.allclose(tile_k.sum(axis=2), k, rtol=5e-12, atol=5e-10):
        raise RuntimeError("tile K values do not reconstruct full K")
    if not np.allclose(
        tile_increment.sum(axis=2), increment, rtol=5e-12, atol=5e-10
    ):
        raise RuntimeError("tile LR increments do not reconstruct full increment")

    # Existing K is D/2.  Adding K to the LR increment recovers w^T xi.
    linear = increment + k
    z_full = linear / np.sqrt(2.0 * k)
    tile_linear = tile_increment + tile_k
    tile_z = tile_linear / np.sqrt(2.0 * tile_k)

    hot_index = np.argmax(tile_k, axis=2)
    hot_k = np.take_along_axis(tile_k, hot_index[..., None], axis=2)[..., 0]
    hot_linear = np.take_along_axis(
        tile_linear, hot_index[..., None], axis=2
    )[..., 0]
    z_hot = hot_linear / np.sqrt(2.0 * hot_k)
    rest_k = k - hot_k
    rest_linear = linear - hot_linear
    if np.any(rest_k <= 0.0):
        raise RuntimeError("hot tile exhausted the full Mahalanobis separation")
    z_rest = rest_linear / np.sqrt(2.0 * rest_k)
    z_contrast = (z_hot - z_rest) / math.sqrt(2.0)

    probabilities = tile_k / k[..., None]
    concentration = np.max(probabilities, axis=2)
    effective_count = 1.0 / np.sum(probabilities * probabilities, axis=2)
    log_d_per_dimension = np.log(
        np.maximum(2.0 * k / GUIDED_DIMENSION, POSITIVE_FLOOR)
    )

    # Independent reconstruction of full Z from the sixteen tile directions.
    reconstructed = tile_linear.sum(axis=2) / np.sqrt(2.0 * tile_k.sum(axis=2))
    if not np.allclose(z_full, reconstructed, rtol=2e-13, atol=2e-13):
        raise RuntimeError("tile standardized directions do not reconstruct full Z")
    tracks = {
        "weak_conditional_cfg1_log_D_per_guided_dimension": log_d_per_dimension,
        "weak_conditional_cfg1_D_tile4x4_concentration": concentration,
        "weak_conditional_cfg1_D_tile4x4_effective_count": effective_count,
        "weak_conditional_cfg1_standardized_alignment_full": z_full,
        "weak_conditional_cfg1_standardized_alignment_hot_tile": z_hot,
        "weak_conditional_cfg1_standardized_alignment_rest": z_rest,
        "weak_conditional_cfg1_standardized_alignment_hot_minus_rest": z_contrast,
    }
    if not all(
        value.shape == k.shape and np.isfinite(value).all() for value in tracks.values()
    ):
        raise RuntimeError("derived normalized posterior track is invalid")
    if np.any((concentration < 1.0 / TILE_COUNT) | (concentration > 1.0)):
        raise RuntimeError("tile concentration lies outside its mathematical range")
    if np.any((effective_count < 1.0) | (effective_count > TILE_COUNT + 1e-10)):
        raise RuntimeError("effective tile count lies outside its mathematical range")
    # Keep tile_z out of the exported family: selecting max realized tile-Z
    # would create an avoidable 16-way multiple-comparison problem.
    del tile_z
    return {name: np.ascontiguousarray(value) for name, value in tracks.items()}


def centered_cusum_range(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    standardized = (values - median) / (MAD_FACTOR * mad + MAD_FLOOR)
    cumulative = np.concatenate(([0.0], np.cumsum(standardized)))
    return float((np.max(cumulative) - np.min(cumulative)) / math.sqrt(len(values)))


def reduce_q2(
    tracks: dict[str, np.ndarray], identifiers: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(next(iter(identifiers.values())))
    rows: list[dict[str, Any]] = [
        {
            "sample_index": int(identifiers["sample_index"][i]),
            "global_seed": int(identifiers["global_seed"][i]),
            "class_slot": int(identifiers["class_slot"][i]),
            "class_id": int(identifiers["class_id"][i]),
        }
        for i in range(n)
    ]
    catalog: list[dict[str, Any]] = []
    for track_name, track in sorted(tracks.items()):
        window = track[:, PHASE_START:PHASE_STOP]
        realized = "standardized_alignment" in track_name
        availability = "online_causal" if realized else "predictable"
        latest_step = PHASE_STOP if realized else PHASE_STOP - 1
        formulas: dict[str, str] = {
            "mean": "mean",
            "standard_deviation": "population standard deviation",
            "max_positive_jump": "maximum positive adjacent difference, floored at zero",
            "max_negative_jump": "maximum negative adjacent difference magnitude, floored at zero",
            "centered_cusum_range": (
                "range of cumulative within-window median/MAD standardized values divided by sqrt(50)"
            ),
        }
        if realized:
            formulas.update(
                {
                    "normalized_sum": "sum of the 50 conditionally standardized alignments divided by sqrt(50)",
                    "maximum": "maximum",
                    "minimum": "minimum",
                    "max_absolute": "maximum absolute value",
                }
            )
        for i in range(n):
            values = window[i]
            delta = np.diff(values)
            reductions: dict[str, float] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "max_positive_jump": float(max(0.0, np.max(delta))),
                "max_negative_jump": float(max(0.0, np.max(-delta))),
                "centered_cusum_range": centered_cusum_range(values),
            }
            if realized:
                reductions.update(
                    {
                        "normalized_sum": float(np.sum(values) / math.sqrt(PHASE_LENGTH)),
                        "maximum": float(np.max(values)),
                        "minimum": float(np.min(values)),
                        "max_absolute": float(np.max(np.abs(values))),
                    }
                )
            for reduction, scalar in reductions.items():
                rows[i][f"{track_name}__{PHASE_NAME}_{reduction}"] = scalar
        for reduction, formula in formulas.items():
            feature = f"{track_name}__{PHASE_NAME}_{reduction}"
            catalog.append(
                {
                    "feature": feature,
                    "track": track_name,
                    "family": (
                        "normalized_weak_process_realized_alignment"
                        if realized
                        else "dimension_stable_weak_process_separation"
                    ),
                    "reduction": f"{PHASE_NAME}_{reduction}",
                    "feature_formula": f"{formula} over sampling k=100..149",
                    "track_length": STOCHASTIC_STEPS,
                    "availability": availability,
                    "latest_required_sampling_step": latest_step,
                    "latest_required_internal_timestep": 249 - latest_step,
                    "observation_timing": (
                        "after_transition_at_latest_step"
                        if realized
                        else "before_transition_at_latest_step"
                    ),
                    "preterminal_actionable": True,
                    "uses_realized_innovation": realized,
                    "selection_warning": (
                        "hot tile is selected predictably by largest D before observing current innovation"
                        if "hot_tile" in track_name
                        else "none"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("reduced normalized posterior features are non-finite")
    return frame, pd.DataFrame(catalog)


def publish(input_root: Path, output: Path) -> Path:
    lineage, arrays = validate_source(input_root)
    tracks = derive_tracks(arrays)
    identifiers = {name: arrays[name] for name in IDENTIFIER_ARRAYS}
    features, catalog = reduce_q2(tracks, identifiers)
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        features.to_csv(staging / "sample_features_label_free.csv", index=False)
        catalog.to_csv(staging / "feature_catalog.csv", index=False)
        np.savez_compressed(
            staging / "time_series_label_free.npz",
            **identifiers,
            sampling_step_249=np.arange(STOCHASTIC_STEPS, dtype=np.int16),
            internal_timestep_from_249=np.arange(249, 0, -1, dtype=np.int16),
            **tracks,
        )
        shutil.copy2(Path(__file__).resolve(), staging / "analysis_source.py")
        source_inventory = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "label_free": True,
            "upstream_posterior_product": {
                "root": lineage["root"],
                "manifest_identity_sha256": lineage["manifest_identity_sha256"],
                "manifest_file_sha256": lineage["manifest_file_sha256"],
                "time_series_file_sha256": lineage["time_series_file_sha256"],
                "source_inventory_path": lineage["source_inventory_path"],
                "source_inventory_file_sha256": lineage[
                    "source_inventory_file_sha256"
                ],
            },
            "ordered_classes": lineage["ordered_classes"],
            "ordered_seeds": lineage["ordered_seeds"],
            "trace_identity_sha256_ordered": lineage[
                "trace_identity_sha256_ordered"
            ],
            "labels_reviews_scores_thresholds_alerts_or_images_read": False,
        }
        write_json(staging / "source_inventory.json", source_inventory)
        z_names = [name for name in tracks if "standardized_alignment" in name]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "sample_count": len(features),
            "track_count": len(tracks),
            "scalar_feature_count": len(catalog),
            "phase": {
                "name": PHASE_NAME,
                "sampling_start_inclusive": PHASE_START,
                "sampling_stop_exclusive": PHASE_STOP,
                "latest_predictable_sampling_step": PHASE_STOP - 1,
                "latest_realized_sampling_step": PHASE_STOP,
            },
            "notation_audit": {
                "source_full_D_nats_is_K_equal_D_over_2": True,
                "D_definition": "sum(((mu_Q-mu_P)/sigma_P)^2)",
                "Z_definition": "(one_step_log_LR_increment + K)/sqrt(2*K)",
                "guided_dimension": GUIDED_DIMENSION,
                "tile_guided_dimension": TILE_DIMENSION,
            },
            "standard_normal_sanity": {
                name: {
                    "all_steps_mean": float(np.mean(tracks[name])),
                    "all_steps_standard_deviation": float(np.std(tracks[name])),
                    "q2_mean": float(np.mean(tracks[name][:, PHASE_START:PHASE_STOP])),
                    "q2_standard_deviation": float(
                        np.std(tracks[name][:, PHASE_START:PHASE_STOP])
                    ),
                }
                for name in z_names
            },
            "supervision_audit": {
                "labels_read_or_emitted": False,
                "labels_reviews_consensus_scores_thresholds_alerts_or_images_read": False,
                "auc_or_quality_threshold_computed": False,
            },
            "claim_limits": {
                "Z_is_not_bad_case_posterior_probability": True,
                "D_per_dimension_is_branch_separation_not_visual_quality": True,
                "later_label_join_is_exploratory_until_new_confirmation": True,
            },
            "input_lineage": lineage,
        }
        write_json(staging / "summary.json", summary)
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "analysis_status": STATUS,
            "analysis_source_sha256": sha256_file(staging / "analysis_source.py"),
            "source_inventory_sha256": sha256_file(
                staging / "source_inventory.json"
            ),
            "source_manifest_identity_sha256": lineage["manifest_identity_sha256"],
            "trace_identity_sha256_ordered": lineage[
                "trace_identity_sha256_ordered"
            ],
            "imported_validation_helper_sha256": lineage[
                "imported_validation_helper_sha256"
            ],
            "label_free": True,
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "summary_file_sha256": sha256_file(staging / "summary.json"),
                "sample_count": len(features),
                "label_free": True,
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    rng = np.random.Generator(np.random.PCG64(17))
    n = 5
    tile_k = rng.uniform(0.01, 0.2, size=(n, STOCHASTIC_STEPS, TILE_COUNT))
    tile_linear = rng.normal(size=tile_k.shape) * np.sqrt(2.0 * tile_k)
    tile_increment = tile_linear - tile_k
    arrays = {
        FULL_K: tile_k.sum(axis=2),
        FULL_INCREMENT: tile_increment.sum(axis=2),
        TILE_K: tile_k,
        TILE_INCREMENT: tile_increment,
    }
    tracks = derive_tracks(arrays)
    assert len(tracks) == 7
    assert tracks["weak_conditional_cfg1_D_tile4x4_concentration"].min() >= 1 / 16
    ids = {
        "sample_index": np.arange(n),
        "global_seed": np.arange(100, 100 + n),
        "class_slot": np.zeros(n, dtype=np.int16),
        "class_id": np.full(n, 207, dtype=np.int16),
    }
    features, catalog = reduce_q2(tracks, ids)
    assert len(features) == n and len(catalog) == 51
    assert not any("label" in name.lower() for name in features.columns)
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input_root is None or args.output is None:
        parser.error("--input-root and --output are required")
    result = publish(args.input_root, args.output)
    print(json.dumps({"output": str(result), "status": STATUS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

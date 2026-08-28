#!/usr/bin/env python3
"""Apply the frozen bad/good candidates to a fresh label-free feature cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v3"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_lock(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"candidate lock must be a real directory: {root}")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    protocol = load_json(root / "candidate_protocol.json")
    if manifest.get("status") != "complete" or completion.get("complete") is not True:
        raise RuntimeError("candidate lock is incomplete")
    if completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json"):
        raise RuntimeError("candidate lock manifest changed")
    if completion.get("protocol_file_sha256") != sha256_file(root / "candidate_protocol.json"):
        raise RuntimeError("candidate protocol changed")
    if completion.get("protocol_identity_sha256") != protocol.get("identity_sha256"):
        raise RuntimeError("candidate protocol identity mismatch")
    if (
        protocol.get("schema_version") != 3
        or protocol.get("status")
        != "FROZEN_BEFORE_FRESH_CONFIRMATION_GENERATION_AND_LABEL_JOIN"
    ):
        raise RuntimeError("scorer accepts only the final v3 candidate protocol")
    for item in manifest.get("files", []):
        path = root / item["name"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"candidate lock member changed: {path}")
    return protocol


def validate_feature_product(
    root: Path, expected_source_hash: str
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"feature product must be a real directory: {root}")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    if manifest.get("status") != "complete" or completion.get("complete") is not True:
        raise RuntimeError(f"feature product incomplete: {root}")
    if completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json"):
        raise RuntimeError(f"feature product manifest changed: {root}")
    if manifest.get("analysis_source_sha256") != expected_source_hash:
        raise RuntimeError(
            f"feature extractor source differs from frozen discovery source: {root}"
        )
    files = {item["name"]: item for item in manifest.get("files", [])}
    for name in ("sample_features.csv", "summary.json", "source_inventory.json"):
        item = files.get(name)
        path = root / name
        if (
            item is None
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"feature product member is not manifest-bound: {path}")
    summary = load_json(root / "summary.json")
    source_inventory = load_json(root / "source_inventory.json")
    if summary.get("labels_joined", False) is not False:
        raise RuntimeError("fresh feature product joined labels")
    if summary.get("supervision_audit", {}).get("labels_read_or_emitted", False) is not False:
        raise RuntimeError("fresh posterior feature product read or emitted labels")
    if "locked_consensus" in source_inventory and source_inventory["locked_consensus"] is not None:
        raise RuntimeError("fresh feature product records a locked consensus input")
    sample_path = root / "sample_features.csv"
    frame = pd.read_csv(sample_path)
    # The frozen primary extractor intentionally emits these two schema columns
    # even in label-free mode.  Their values, rather than their mere presence,
    # prove that no visual supervision was joined.
    if "label" in frame.columns and not frame["label"].eq("unlabeled").all():
        raise RuntimeError("fresh primary feature table contains non-unlabeled labels")
    if "raw_consensus_label" in frame.columns and not frame["raw_consensus_label"].isna().all():
        raise RuntimeError("fresh primary feature table contains raw consensus labels")
    return manifest, frame, source_inventory


def validate_shared_trace_lineage(
    protocol: dict[str, Any],
    primary_manifest: dict[str, Any],
    posterior_manifest: dict[str, Any],
    primary_inventory: dict[str, Any],
    posterior_inventory: dict[str, Any],
) -> None:
    frozen_primary_analysis_hash = protocol["source_products"]["primary_label_free"][
        "analysis_source_sha256"
    ]
    if (
        primary_manifest.get("analysis_source_sha256") != frozen_primary_analysis_hash
        or posterior_manifest.get("imported_validation_helper_sha256")
        != frozen_primary_analysis_hash
    ):
        raise RuntimeError("fresh posterior reductions do not import the frozen primary helper")
    primary_identities = primary_manifest.get("trace_identity_sha256_ordered")
    posterior_identities = posterior_manifest.get("trace_identity_sha256_ordered")
    if (
        not isinstance(primary_identities, list)
        or primary_identities != posterior_identities
        or len(primary_identities) != protocol["fresh_confirmation"]["seeds"]["count"]
    ):
        raise RuntimeError("primary and posterior products do not bind the same trace cohort")
    primary_runs = primary_inventory.get("trace_runs")
    posterior_runs = posterior_inventory.get("trace_runs")
    if not isinstance(primary_runs, list) or not isinstance(posterior_runs, list):
        raise RuntimeError("fresh source inventory lacks trace-run records")
    core_fields = (
        "cfg_epsilon_channels",
        "cfg_scale",
        "classes",
        "completion_sha256",
        "global_seed",
        "identity_sha256",
        "manifest_sha256",
        "source_snapshot_sha256",
        "trace_sha256",
    )
    primary_core = [{key: run.get(key) for key in core_fields} for run in primary_runs]
    posterior_core = [{key: run.get(key) for key in core_fields} for run in posterior_runs]
    if primary_core != posterior_core:
        raise RuntimeError("primary and posterior products were extracted from different traces")
    if [run.get("identity_sha256") for run in primary_runs] != primary_identities:
        raise RuntimeError("fresh manifest trace identities differ from source inventory order")

    frozen_root = Path(protocol["source_products"]["primary_label_free"]["path"])
    frozen_manifest_path = frozen_root / "manifest.json"
    frozen_manifest = load_json(frozen_manifest_path)
    frozen_inventory_path = frozen_root / "source_inventory.json"
    frozen_inventory = load_json(frozen_inventory_path)
    if (
        frozen_manifest.get("identity_sha256")
        != protocol["source_products"]["primary_label_free"]["manifest_identity_sha256"]
        or sha256_file(frozen_manifest_path)
        != protocol["source_products"]["primary_label_free"]["manifest_file_sha256"]
    ):
        raise RuntimeError("frozen discovery feature product is unavailable or changed")
    frozen_files = {item["name"]: item for item in frozen_manifest.get("files", [])}
    inventory_item = frozen_files.get(frozen_inventory_path.name)
    if (
        inventory_item is None
        or frozen_inventory_path.stat().st_size != inventory_item.get("bytes")
        or sha256_file(frozen_inventory_path) != inventory_item.get("sha256")
    ):
        raise RuntimeError("frozen discovery source inventory changed")
    frozen_runs = frozen_inventory.get("trace_runs")
    if not isinstance(frozen_runs, list) or not frozen_runs:
        raise RuntimeError("frozen discovery product lacks sampler lineage")
    expected_snapshots = protocol["sampler_lineage_contract"]["source_snapshot_sha256"]
    expected_classes = protocol["fresh_confirmation"]["classes"]
    for run in primary_runs:
        if (
            run.get("cfg_scale") != 4.0
            or run.get("cfg_epsilon_channels") != 3
            or run.get("classes") != expected_classes
            or run.get("source_snapshot_sha256") != expected_snapshots
        ):
            raise RuntimeError("fresh trace sampler/source contract differs from discovery")


def frozen_old_score(
    protocol: dict[str, Any], primary_root: Path, primary_frame: pd.DataFrame
) -> np.ndarray:
    """Recompute the old control with discovery medians/scales, never fresh ones."""

    frozen_root = Path(protocol["source_products"]["primary_label_free"]["path"])
    frozen_manifest_path = frozen_root / "manifest.json"
    frozen_manifest = load_json(frozen_manifest_path)
    if (
        frozen_manifest.get("identity_sha256")
        != protocol["source_products"]["primary_label_free"]["manifest_identity_sha256"]
        or sha256_file(frozen_manifest_path)
        != protocol["source_products"]["primary_label_free"]["manifest_file_sha256"]
    ):
        raise RuntimeError("old-score discovery reference manifest identity changed")
    frozen_files = {item["name"]: item for item in frozen_manifest.get("files", [])}
    fresh_manifest = load_json(primary_root / "manifest.json")
    fresh_files = {item["name"]: item for item in fresh_manifest.get("files", [])}
    reference_path = frozen_root / "label_free_reference_stats.npz"
    series_path = primary_root / "time_series.npz"
    for path, item in (
        (reference_path, frozen_files.get(reference_path.name)),
        (series_path, fresh_files.get(series_path.name)),
    ):
        if (
            item is None
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"old-score input is not manifest-bound: {path}")
    if sha256_file(reference_path) != protocol["source_products"]["primary_label_free"][
        "label_free_reference_stats_sha256"
    ]:
        raise RuntimeError("old-score discovery reference differs from the frozen hash")
    with np.load(reference_path, allow_pickle=False) as archive:
        rough_median = archive["pred_roughness_median"].astype(np.float64)
        rough_scale = archive["pred_roughness_scale"].astype(np.float64)
        amplitude_median = archive["pred_amplitude_median"].astype(np.float64)
        amplitude_scale = archive["pred_amplitude_scale"].astype(np.float64)
    with np.load(series_path, allow_pickle=False) as archive:
        keys = ["sample_index", "global_seed", "class_slot", "class_id"]
        for key in keys:
            if not np.array_equal(archive[key], primary_frame[key].to_numpy()):
                raise RuntimeError(f"fresh time-series sample axis differs at {key}")
        rough = archive["pred_xstart_log_normalized_dirichlet_mean_channels"].astype(
            np.float64
        )
        amplitude = archive["pred_xstart_log_spatial_variance_mean_channels"].astype(
            np.float64
        )
    if (
        rough.shape != amplitude.shape
        or rough.shape != (len(primary_frame), 250)
        or rough_median.shape != (250,)
        or rough_scale.shape != (250,)
        or amplitude_median.shape != (250,)
        or amplitude_scale.shape != (250,)
        or np.any(rough_scale <= 0.0)
        or np.any(amplitude_scale <= 0.0)
    ):
        raise RuntimeError("old-score frozen reference shapes/scales are invalid")
    q1_rough = np.mean((rough[:, 50:100] - rough_median[50:100]) / rough_scale[50:100], axis=1)
    q2_amplitude = np.mean(
        (amplitude[:, 100:150] - amplitude_median[100:150])
        / amplitude_scale[100:150],
        axis=1,
    )
    return (q1_rough + q2_amplitude) / np.sqrt(2.0)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def score(
    lock_root: Path, primary_root: Path, posterior_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = validate_lock(lock_root)
    source = protocol["source_products"]
    primary_manifest, primary, primary_inventory = validate_feature_product(
        primary_root, source["primary_label_free"]["analysis_source_sha256"]
    )
    posterior_manifest, posterior, posterior_inventory = validate_feature_product(
        posterior_root, source["posterior_label_free"]["analysis_source_sha256"]
    )
    validate_shared_trace_lineage(
        protocol,
        primary_manifest,
        posterior_manifest,
        primary_inventory,
        posterior_inventory,
    )
    keys = ["sample_index", "run_index", "global_seed", "class_slot", "class_id"]
    if len(primary) != len(posterior) or not np.array_equal(
        primary[keys].to_numpy(), posterior[keys].to_numpy()
    ):
        raise RuntimeError("fresh primary and posterior feature cohorts do not align")
    expected_classes = tuple(protocol["fresh_confirmation"]["classes"])
    expected_seed_spec = protocol["fresh_confirmation"]["seeds"]
    expected_seeds = tuple(
        range(expected_seed_spec["start_inclusive"], expected_seed_spec["stop_inclusive"] + 1)
    )
    if tuple(sorted(int(value) for value in primary["class_id"].unique())) != expected_classes:
        raise RuntimeError("fresh cohort classes differ from the frozen protocol")
    if tuple(sorted(int(value) for value in primary["global_seed"].unique())) != expected_seeds:
        raise RuntimeError("fresh cohort seeds differ from the frozen protocol")
    if len(primary) != protocol["fresh_confirmation"]["trajectory_count"]:
        raise RuntimeError("fresh cohort size differs from the frozen protocol")
    class_slots = {class_id: index for index, class_id in enumerate(expected_classes)}
    expected_rows = {
        (seed, class_slots[class_id], class_id)
        for seed in expected_seeds
        for class_id in expected_classes
    }
    observed_rows = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in primary[["global_seed", "class_slot", "class_id"]].itertuples(index=False)
    }
    if (
        observed_rows != expected_rows
        or primary[["global_seed", "class_id"]].duplicated().any()
        or posterior[["global_seed", "class_id"]].duplicated().any()
        or tuple(primary["sample_index"].astype(int)) != tuple(range(len(primary)))
    ):
        raise RuntimeError("fresh cohort is not the exact frozen seed x class Cartesian product")

    feature_a = protocol["single_feature_backups"]["A"]["feature"]
    feature_b = protocol["single_feature_backups"]["B"]["feature"]
    e_controls = protocol["negative_controls"]["exact_path_evidence_running_maxima"]
    if feature_b not in primary:
        raise RuntimeError("fresh primary feature product lacks a frozen column")
    if feature_a not in posterior or any(name not in posterior for name in e_controls):
        raise RuntimeError("fresh posterior feature product lacks a frozen column")

    result = primary[keys + ["trace_dir", "endpoint_png_path"]].copy()
    calibration_spec = protocol["fresh_confirmation"]["label_free_conformal_calibration"][
        "seeds"
    ]
    calibration_mask = result["global_seed"].between(
        calibration_spec["start_inclusive"], calibration_spec["stop_inclusive"]
    )
    result["cohort_role"] = np.where(
        calibration_mask, "label_free_calibration", "inferential_evaluation"
    )
    result["A_posterior_logstd_concentration_jump"] = posterior[feature_a].to_numpy(float)
    result["B_withheld_channel_predx0_cusum"] = primary[feature_b].to_numpy(float)
    result["old_fixed_predicted_clean_score_control"] = frozen_old_score(
        protocol, primary_root, primary
    )
    result["z_A_low_is_bad"] = np.nan
    result["z_B_high_is_bad"] = np.nan
    result["S_AND"] = np.nan
    references = protocol["normalization"]["class_reference"]
    for class_id in expected_classes:
        mask = result["class_id"] == class_id
        ref = references[str(class_id)]
        a_stats = ref["statistics"]["A_low_is_bad"]
        b_stats = ref["statistics"]["B_high_is_bad"]
        result.loc[mask, "z_A_low_is_bad"] = (
            -result.loc[mask, "A_posterior_logstd_concentration_jump"] - a_stats["median"]
        ) / a_stats["scale"]
        result.loc[mask, "z_B_high_is_bad"] = (
            result.loc[mask, "B_withheld_channel_predx0_cusum"] - b_stats["median"]
        ) / b_stats["scale"]
        result.loc[mask, "S_AND"] = np.minimum(
            result.loc[mask, "z_A_low_is_bad"], result.loc[mask, "z_B_high_is_bad"]
        )
    for name in e_controls:
        short = name.replace("__full_maximum", "")
        values = posterior[name].to_numpy(float)
        result[f"control_{short}"] = values
        result[f"control_{short}_trigger_alpha0p10"] = values >= np.log(10.0)
        result[f"control_{short}_trigger_alpha0p05"] = values >= np.log(20.0)
    numeric = result.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise RuntimeError("frozen score output contains non-finite numbers")
    if result[["z_A_low_is_bad", "z_B_high_is_bad", "S_AND"]].isna().any().any():
        raise RuntimeError("some samples were not covered by the frozen class reference")
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE_LABEL_FREE_FROZEN_CANDIDATE_SCORES_DO_NOT_JOIN_DRAFT_LABELS",
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "sample_count": int(len(result)),
        "classes": list(expected_classes),
        "seeds": list(expected_seeds),
        "labels_read_or_emitted": False,
        "thresholds_reestimated": False,
        "alerts_emitted": False,
        "calibration_scores_are_disjoint_from_inferential_evaluation": True,
        "old_control_reference_reestimated": False,
        "old_control_reference_source": "manifest-bound discovery label_free_reference_stats.npz",
        "primary_scores": ["z_A_low_is_bad", "z_B_high_is_bad", "S_AND"],
        "primary_manifest_identity_sha256": primary_manifest["identity_sha256"],
        "posterior_manifest_identity_sha256": posterior_manifest["identity_sha256"],
    }
    return result, metadata


def publish(lock_root: Path, primary_root: Path, posterior_root: Path, output: Path) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite frozen score output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        scores, summary = score(lock_root.resolve(), primary_root.resolve(), posterior_root.resolve())
        scores.to_csv(staging / "frozen_candidate_scores_label_free.csv", index=False)
        write_json(staging / "summary.json", summary)
        shutil.copy2(Path(__file__).resolve(), staging / "scorer_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": summary[
                "candidate_protocol_identity_sha256"
            ],
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
                "scores_file_sha256": sha256_file(
                    staging / "frozen_candidate_scores_label_free.csv"
                ),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-root", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--posterior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = publish(args.lock_root, args.primary_root, args.posterior_root, args.output)
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

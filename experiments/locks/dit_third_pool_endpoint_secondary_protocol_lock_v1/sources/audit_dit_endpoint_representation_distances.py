#!/usr/bin/env python3
"""Aggregate-only endpoint representation-distance audit for frozen DiT pools.

The six distance hypotheses are fixed before labels are opened: for each of
two sealed endpoint representations, use L2-normalized cosine distance to the
class clean centroid, 5-nearest-clean cosine distance, and a class-centered
Mahalanobis distance with one pooled within-class Ledoit-Wolf covariance.

Discovery seeds 50..129 are scored out of fold (global_seed mod 5; all three
classes stay in one fold).  Expansion seeds 130..249 are the primary audit and
are scored from a reference fitted to all discovery clean-good endpoints.
Every direction is fixed as distance-high-is-bad.  Expansion p-values use
100,000 permutations of intact three-class label vectors at global-seed grain,
including mild labels, followed by Holm correction across exactly six tests.

Outputs are aggregate only.  Endpoint distances are terminal retrospective
diagnostics and cannot authorize online intervention.  Additional Spearman
tables describe whether the two already-frozen preterminal risks B (decoded
pred_xstart blur) and C (negative c3 structure jump) track endpoint distance;
these correlations are explicitly exploratory and emit no sample scores/ranks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import ledoit_wolf_shrinkage


SCHEMA_VERSION = 1
EXPERIMENT = "dit_endpoint_representation_distance_audit_v1"
CLASSES = (207, 602, 795)
DISCOVERY_SEEDS = tuple(range(50, 130))
EXPANSION_SEEDS = tuple(range(130, 250))
GOOD = "clean_good"
BAD = "clear_bad"
MILD = "mild_or_disputed"
LABELS = (GOOD, BAD, MILD)
FOLDS = 5
KNN_K = 5
MINIMUM_GROUP_SIZE = 5
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 20260827
REPRESENTATIONS = (
    ("inception_fid_pool2048", 2048),
    ("dinov2_registers_large_cls1024", 1024),
)
DISTANCES = (
    "cosine_to_class_clean_centroid",
    "shared_ledoitwolf_mahalanobis",
    "knn5_mean_cosine_to_class_clean",
)
METRICS = tuple(f"{representation}__{distance}" for representation, _ in REPRESENTATIONS for distance in DISTANCES)
B_FEATURE = "decoded_local_blur_severity__mean"
C_SOURCE_FEATURE = (
    "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
)
PRETERMINAL_RISKS = ("B_decoded_predxstart_blur_high", "C_c3_structure_jump_low")

DEFAULT_BASE = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
DEFAULT_EMBEDDINGS = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/endpoint_embeddings_label_free_v1"
)
DEFAULT_ORIGINAL_LABELS = Path(
    "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
DEFAULT_EXPANSION_LABELS = Path(
    "experiments/annotations/dit_expansion_eval360_adjudicated_consensus_lock_v1"
)
DEFAULT_ORIGINAL_VISUAL = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/predxstart_visual_label_free_v1"
)
DEFAULT_EXPANSION_VISUAL = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/predxstart_visual_label_free_v1"
)
DEFAULT_ORIGINAL_PRIMARY = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/custom_label_free_v1"
)
DEFAULT_EXPANSION_PRIMARY = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/primary_label_free_v1"
)
DEFAULT_PROTOCOL = Path(
    "experiments/locks/dit_endpoint_representation_distance_protocol_v1/protocol.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/endpoint_representation_distance_audit_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_identity(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied.pop("identity_sha256", None)
    return canonical_sha256(copied)


def canonical_self_hash(value: Mapping[str, Any], key: str) -> str:
    copied = dict(value)
    copied.pop(key, None)
    return canonical_sha256(copied)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_manifest_members(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"manifest has no files: {root}")
    members: dict[str, Any] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError(f"invalid manifest member: {root}")
        name = item["name"]
        if name in members or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe/duplicate manifest member: {name}")
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"manifest member changed: {path}")
        members[name] = item
    return members


def validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"protocol lock is unavailable: {path}")
    protocol = read_json(path)
    identity = canonical_identity(protocol)
    expected_metric_specs = [
        {
            "name": metric,
            "direction": "distance_high_is_bad",
            "role": "terminal_retrospective_only",
        }
        for metric in METRICS
    ]
    if (
        protocol.get("identity_sha256") != identity
        or protocol.get("status")
        != "FROZEN_BEFORE_ENDPOINT_EMBEDDING_LABEL_JOIN"
        or protocol.get("analysis_source_sha256")
        != sha256_file(Path(__file__).resolve())
        or protocol.get("cohorts", {}).get("discovery_seeds")
        != [DISCOVERY_SEEDS[0], DISCOVERY_SEEDS[-1]]
        or protocol.get("cohorts", {}).get("expansion_seeds")
        != [EXPANSION_SEEDS[0], EXPANSION_SEEDS[-1]]
        or protocol.get("cohorts", {}).get("classes") != list(CLASSES)
        or protocol.get("reference", {}).get("crossfit_folds") != FOLDS
        or protocol.get("reference", {}).get("fold_assignment")
        != "global_seed mod 5; intact three-class seed block"
        or protocol.get("reference", {}).get("knn_k") != KNN_K
        or protocol.get("metric_family") != expected_metric_specs
        or protocol.get("test", {}).get("permutation_draws") != PERMUTATION_DRAWS
        or protocol.get("test", {}).get("permutation_seed") != PERMUTATION_SEED
        or protocol.get("test", {}).get("permutation_unit")
        != "intact global-seed three-class label vector including mild"
        or protocol.get("test", {}).get("multiple_testing")
        != "Holm across exactly six frozen expansion bad-vs-good distance tests"
        or protocol.get("publication", {}).get("output") != str(DEFAULT_OUTPUT)
        or protocol.get("publication", {}).get("aggregate_only") is not True
        or protocol.get("publication", {}).get("no_overwrite") is not True
        or protocol.get("publication", {}).get("minimum_group_size")
        != MINIMUM_GROUP_SIZE
    ):
        raise RuntimeError("endpoint-distance protocol lock differs from the code contract")
    return protocol, identity


def validate_embedding_product(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"embedding product is invalid: {root}")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    completion = read_json(root / "completion.json")
    summary = read_json(root / "summary.json")
    inventory = read_json(root / "source_inventory.json")
    manifest_identity = canonical_identity(manifest)
    members = validate_manifest_members(root, manifest)
    expected_members = {
        "analysis_source.py",
        "embeddings.npz",
        "protocol_snapshot.json",
        "provenance.json",
        "representation_catalog.csv",
        "sample_index.csv",
        "source_inventory.json",
        "summary.json",
    }
    entries = list(root.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RuntimeError("embedding product contains a directory/symlink")
    actual = {path.name for path in entries}
    if (
        set(members) != expected_members
        or actual != expected_members | {"manifest.json", "completion.json"}
        or manifest.get("identity_sha256") != manifest_identity
        or protocol.get("input_identities", {}).get(
            "endpoint_embedding_manifest_identity_sha256"
        )
        != manifest_identity
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("payload_sha256")
        != canonical_self_hash(completion, "payload_sha256")
        or summary.get("status") != "COMPLETE_LABEL_FREE_ENDPOINT_EMBEDDINGS"
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("distances_or_scores_computed") is not False
        or summary.get("images_saved") is not False
        or summary.get("preterminal_actionable") is not False
        or summary.get("sample_count") != 600
        or summary.get("seed_count") != 200
        or summary.get("ordered_classes") != list(CLASSES)
        or summary.get("representations") != dict(REPRESENTATIONS)
    ):
        raise RuntimeError("embedding product or frozen identity changed")
    index = pd.read_csv(root / "sample_index.csv")
    required_index = {
        "sample_index",
        "global_seed",
        "class_id",
        "endpoint_sha256",
        "trace_identity_sha256",
    }
    if (
        not required_index.issubset(index.columns)
        or any(name in index for name in ("label", "primary_label", "raw_consensus_label"))
        or index.duplicated(["global_seed", "class_id"]).any()
        or index["sample_index"].to_list() != list(range(600))
    ):
        raise RuntimeError("embedding index is malformed/supervised")
    expected_axis = {
        (seed, class_id)
        for seed in (*DISCOVERY_SEEDS, *EXPANSION_SEEDS)
        for class_id in CLASSES
    }
    if set(map(tuple, index[["global_seed", "class_id"]].to_numpy())) != expected_axis:
        raise RuntimeError("embedding product does not have the exact 600-sample axis")
    arrays: dict[str, np.ndarray] = {}
    records = inventory.get("embedding_arrays")
    with np.load(root / "embeddings.npz", allow_pickle=False) as archive:
        if set(archive.files) != {name for name, _ in REPRESENTATIONS}:
            raise RuntimeError("embedding array family changed")
        for name, dimension in REPRESENTATIONS:
            value = np.asarray(archive[name], dtype=np.float64)
            recorded = records.get(name) if isinstance(records, dict) else None
            raw = np.asarray(archive[name])
            if (
                value.shape != (600, dimension)
                or not np.isfinite(value).all()
                or not isinstance(recorded, dict)
                or recorded.get("shape") != [600, dimension]
                or recorded.get("dtype") != raw.dtype.str
                or recorded.get("raw_sha256") != sha256_array(raw)
            ):
                raise RuntimeError(f"embedding array changed: {name}")
            arrays[name] = value
    return index, arrays, {
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "analysis_source_sha256": manifest.get("analysis_source_sha256"),
        "embedding_array_sha256": {
            name: records[name]["raw_sha256"] for name, _ in REPRESENTATIONS
        },
    }


def validate_label_lock(
    root: Path, expected_seeds: tuple[int, ...], expected_status: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"label lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    manifest = read_json(manifest_path)
    consensus = read_json(consensus_path)
    completion = read_json(root / "completion.json")
    manifest_id = canonical_identity(manifest)
    consensus_id = canonical_identity(consensus)
    members = validate_manifest_members(root, manifest)
    expected_n = len(expected_seeds) * len(CLASSES)
    if (
        manifest.get("identity_sha256") != manifest_id
        or consensus.get("identity_sha256") != consensus_id
        or manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_id
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("consensus_identity_sha256") != consensus_id
        or completion.get("locked_row_count") != expected_n
        or manifest.get("consensus_identity_sha256") != consensus_id
        or consensus.get("status") != expected_status
        or "adjudication_locked.json" not in members
    ):
        raise RuntimeError(f"invalid final label lock: {root}")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_n:
        raise RuntimeError("label consensus row count changed")
    records = []
    for row in rows:
        seed = row.get("global_seed")
        class_id = row.get("class_id")
        label = row.get("primary_label")
        if (
            seed not in expected_seeds
            or class_id not in CLASSES
            or label not in LABELS
            or row.get("binary_primary_included") != (label != MILD)
        ):
            raise RuntimeError("label consensus row is invalid")
        records.append({"global_seed": seed, "class_id": class_id, "label": label})
    frame = pd.DataFrame(records)
    expected_axis = {
        (seed, class_id) for seed in expected_seeds for class_id in CLASSES
    }
    if (
        frame.duplicated(["global_seed", "class_id"]).any()
        or set(map(tuple, frame[["global_seed", "class_id"]].to_numpy())) != expected_axis
        or frame.label.value_counts().to_dict() != consensus.get("counts")
    ):
        raise RuntimeError("label lock axis/counts changed")
    return frame, {
        "manifest_identity_sha256": manifest_id,
        "consensus_identity_sha256": consensus_id,
        "counts": {label: int((frame.label == label).sum()) for label in LABELS},
    }


def validate_label_free_scalar_product(
    root: Path,
    expected_seeds: tuple[int, ...],
    feature: str,
    expected_manifest_identity: str,
    product_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    completion = read_json(root / "completion.json")
    summary = read_json(root / "summary.json")
    identity = canonical_identity(manifest)
    members = validate_manifest_members(root, manifest)
    if (
        identity != expected_manifest_identity
        or manifest.get("identity_sha256") != identity
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or "sample_features.csv" not in members
        or "feature_catalog.csv" not in members
        or summary.get("labels_joined", False) is not False
        or summary.get("labels_read_or_emitted", False) is not False
    ):
        raise RuntimeError(f"invalid/supervised scalar product: {product_name}")
    catalog = pd.read_csv(root / "feature_catalog.csv")
    selected = catalog[catalog["feature"].astype(str) == feature]
    if len(selected) != 1:
        raise RuntimeError(f"{product_name} lacks exactly one frozen feature")
    row = selected.iloc[0]
    actionable_raw = row.get("preterminal_actionable")
    actionable = (
        bool(actionable_raw)
        if isinstance(actionable_raw, (bool, np.bool_))
        else str(actionable_raw).lower() == "true"
    )
    if int(row["latest_required_sampling_step"]) != 149 or not actionable:
        raise RuntimeError(f"{product_name} feature is no longer preterminal at k149")
    frame = pd.read_csv(root / "sample_features.csv")
    if feature not in frame or frame.duplicated(["global_seed", "class_id"]).any():
        raise RuntimeError(f"{product_name} feature table changed")
    cohort = frame[
        frame.global_seed.isin(expected_seeds) & frame.class_id.isin(CLASSES)
    ][["global_seed", "class_id", feature]].copy()
    expected_axis = {
        (seed, class_id) for seed in expected_seeds for class_id in CLASSES
    }
    if (
        set(map(tuple, cohort[["global_seed", "class_id"]].to_numpy())) != expected_axis
        or not np.isfinite(cohort[feature].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"{product_name} exact cohort/values changed")
    return cohort, {
        "manifest_identity_sha256": identity,
        "feature": feature,
        "feature_catalog_sha256": sha256_file(root / "feature_catalog.csv"),
        "sample_features_sha256": sha256_file(root / "sample_features.csv"),
    }


@dataclass
class ReferenceModel:
    centers: dict[int, np.ndarray]
    unit_centers: dict[int, np.ndarray]
    clean_points: dict[int, np.ndarray]
    shrinkage: float
    isotropic: float
    covariance_eigenvalues: np.ndarray
    covariance_basis: np.ndarray


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise RuntimeError("representation has invalid row norms")
    return value / norms


def fit_reference(
    embeddings: np.ndarray, classes: np.ndarray, labels: np.ndarray
) -> tuple[ReferenceModel, dict[str, Any]]:
    centers: dict[int, np.ndarray] = {}
    unit_centers: dict[int, np.ndarray] = {}
    clean_points: dict[int, np.ndarray] = {}
    residual_parts = []
    counts = {}
    for class_id in CLASSES:
        mask = (classes == class_id) & (labels == GOOD)
        points = embeddings[mask]
        if len(points) < KNN_K + 1:
            raise RuntimeError(f"too few clean references for class {class_id}")
        center = points.mean(axis=0)
        center_norm = np.linalg.norm(center)
        if not np.isfinite(center_norm) or center_norm <= 0:
            raise RuntimeError("class clean centroid has zero/invalid norm")
        centers[class_id] = center
        unit_centers[class_id] = center / center_norm
        clean_points[class_id] = points
        residual_parts.append(points - center)
        counts[class_id] = len(points)
    residuals = np.concatenate(residual_parts, axis=0)
    shrinkage = float(
        ledoit_wolf_shrinkage(
            residuals, assume_centered=True, block_size=min(1000, embeddings.shape[1])
        )
    )
    _, singular, basis = np.linalg.svd(residuals, full_matrices=False)
    eigenvalues = singular**2 / len(residuals)
    tau = float(np.sum(eigenvalues) / embeddings.shape[1])
    isotropic = max(shrinkage * tau, 1e-12)
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1 or tau <= 0:
        raise RuntimeError("Ledoit-Wolf reference fit is invalid")
    model = ReferenceModel(
        centers=centers,
        unit_centers=unit_centers,
        clean_points=clean_points,
        shrinkage=shrinkage,
        isotropic=isotropic,
        covariance_eigenvalues=eigenvalues,
        covariance_basis=basis,
    )
    return model, {
        "clean_counts": counts,
        "pooled_residual_count": len(residuals),
        "dimension": embeddings.shape[1],
        "ledoit_wolf_shrinkage": shrinkage,
        "isotropic_variance_component": isotropic,
        "effective_rank_bound": int(np.sum(eigenvalues > 1e-15)),
    }


def score_reference(
    model: ReferenceModel, embeddings: np.ndarray, classes: np.ndarray
) -> dict[str, np.ndarray]:
    cosine = np.empty(len(embeddings), dtype=np.float64)
    mahalanobis = np.empty(len(embeddings), dtype=np.float64)
    knn = np.empty(len(embeddings), dtype=np.float64)
    basis = model.covariance_basis
    eig = model.covariance_eigenvalues
    parallel_denominator = model.isotropic + (1.0 - model.shrinkage) * eig
    for class_id in CLASSES:
        mask = classes == class_id
        points = embeddings[mask]
        cosine[mask] = 1.0 - points @ model.unit_centers[class_id]
        residual = points - model.centers[class_id]
        projections = residual @ basis.T
        residual_norm2 = np.sum(residual**2, axis=1)
        parallel_norm2 = np.sum(projections**2, axis=1)
        orthogonal_norm2 = np.maximum(residual_norm2 - parallel_norm2, 0.0)
        distance2 = orthogonal_norm2 / model.isotropic + np.sum(
            projections**2 / parallel_denominator[None, :], axis=1
        )
        mahalanobis[mask] = np.sqrt(np.maximum(distance2, 0.0))
        neighbor_distance = np.clip(
            1.0 - points @ model.clean_points[class_id].T, 0.0, 2.0
        )
        nearest = np.partition(neighbor_distance, KNN_K - 1, axis=1)[:, :KNN_K]
        knn[mask] = nearest.mean(axis=1)
    scores = {
        "cosine_to_class_clean_centroid": cosine,
        "shared_ledoitwolf_mahalanobis": mahalanobis,
        "knn5_mean_cosine_to_class_clean": knn,
    }
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("reference scoring produced non-finite distances")
    return scores


def compute_distances(
    index: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = index[["sample_index", "global_seed", "class_id"]].merge(
        labels, on=["global_seed", "class_id"], how="inner", validate="one_to_one"
    )
    if len(frame) != len(index):
        raise RuntimeError("embedding/label join lost rows")
    source_positions = frame["sample_index"].to_numpy(dtype=int)
    if len(set(source_positions)) != len(source_positions):
        raise RuntimeError("embedding sample indices are not unique")
    classes = frame.class_id.to_numpy(dtype=int)
    seeds = frame.global_seed.to_numpy(dtype=int)
    y = frame.label.to_numpy(dtype=str)
    fit_summaries = []
    for representation, _dimension in REPRESENTATIONS:
        values = normalize_rows(arrays[representation][source_positions])
        for distance in DISTANCES:
            frame[f"{representation}__{distance}"] = np.nan
        discovery_mask = np.isin(seeds, DISCOVERY_SEEDS)
        expansion_mask = np.isin(seeds, EXPANSION_SEEDS)
        for fold in range(FOLDS):
            train = discovery_mask & ((seeds % FOLDS) != fold)
            test = discovery_mask & ((seeds % FOLDS) == fold)
            model, summary = fit_reference(values[train], classes[train], y[train])
            scores = score_reference(model, values[test], classes[test])
            for distance, score in scores.items():
                frame.loc[test, f"{representation}__{distance}"] = score
            fit_summaries.append(
                {"representation": representation, "reference": "discovery_crossfit", "fold": fold, **summary}
            )
        final_model, summary = fit_reference(
            values[discovery_mask], classes[discovery_mask], y[discovery_mask]
        )
        scores = score_reference(
            final_model, values[expansion_mask], classes[expansion_mask]
        )
        for distance, score in scores.items():
            frame.loc[expansion_mask, f"{representation}__{distance}"] = score
        fit_summaries.append(
            {"representation": representation, "reference": "all_discovery_clean_for_expansion", "fold": None, **summary}
        )
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("some endpoint distances were not assigned")
    return frame, fit_summaries


def auc_bad_high(scores: np.ndarray, labels: np.ndarray, positive: str) -> tuple[float, int]:
    positive_scores = np.asarray(scores[labels == positive], dtype=np.float64)
    good = np.sort(np.asarray(scores[labels == GOOD], dtype=np.float64))
    pairs = len(positive_scores) * len(good)
    if pairs == 0:
        return float("nan"), 0
    left = np.searchsorted(good, positive_scores, side="left")
    right = np.searchsorted(good, positive_scores, side="right")
    wins = float(np.sum(left + 0.5 * (right - left)))
    return wins / pairs, pairs


def auc_summary(
    scores: np.ndarray, labels: np.ndarray, classes: np.ndarray, positive: str
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class = {}
    counts = {}
    for class_id in CLASSES:
        mask = classes == class_id
        auc, pairs = auc_bad_high(scores[mask], labels[mask], positive)
        per_class[class_id] = auc
        counts[class_id] = {
            positive: int(np.sum(labels[mask] == positive)),
            GOOD: int(np.sum(labels[mask] == GOOD)),
        }
        if pairs:
            numerator += auc * pairs
            denominator += pairs
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return {
        "pair_weighted_auc": numerator / denominator if denominator else float("nan"),
        "macro_auc": float(np.mean(finite)) if finite else float("nan"),
        "per_class_auc": per_class,
        "counts": counts,
        "pair_count": denominator,
    }


def _tie_groups(sorted_scores: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(sorted_scores)]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops) if stop - start > 1]


def block_permutation_pvalues(
    expansion: pd.DataFrame,
    draws: int,
    seed: int,
    batch_size: int = 1000,
) -> np.ndarray:
    seeds = np.asarray(EXPANSION_SEEDS, dtype=int)
    label_matrix = np.empty((len(seeds), len(CLASSES)), dtype=object)
    score_tensor = np.empty((len(METRICS), len(CLASSES), len(seeds)), dtype=np.float64)
    for class_index, class_id in enumerate(CLASSES):
        part = expansion[expansion.class_id == class_id].set_index("global_seed").loc[seeds]
        label_matrix[:, class_index] = part.label.to_numpy(dtype=str)
        for metric_index, metric in enumerate(METRICS):
            score_tensor[metric_index, class_index] = part[metric].to_numpy(dtype=float)
    observed = np.asarray(
        [
            auc_summary(
                expansion[metric].to_numpy(float),
                expansion.label.to_numpy(str),
                expansion.class_id.to_numpy(int),
                BAD,
            )["pair_weighted_auc"]
            for metric in METRICS
        ],
        dtype=np.float64,
    )
    class_pairs = []
    for class_index in range(len(CLASSES)):
        labels = label_matrix[:, class_index]
        class_pairs.append(int(np.sum(labels == BAD) * np.sum(labels == GOOD)))
    denominator = int(sum(class_pairs))
    threshold = observed * denominator
    exceed = np.zeros(len(METRICS), dtype=np.int64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        permutations = np.argsort(
            rng.random((size, len(seeds))), axis=1, kind="quicksort"
        )
        permuted_labels = label_matrix[permutations]
        numerator = np.zeros((size, len(METRICS)), dtype=np.float64)
        for class_index in range(len(CLASSES)):
            permuted_class = permuted_labels[:, :, class_index]
            for metric_index in range(len(METRICS)):
                scores = score_tensor[metric_index, class_index]
                order = np.argsort(scores, kind="mergesort")
                ordered_labels = permuted_class[:, order]
                good = ordered_labels == GOOD
                bad = ordered_labels == BAD
                cumulative_good = np.cumsum(good, axis=1)
                credit = cumulative_good - good + 0.5 * good
                for start, stop in _tie_groups(scores[order]):
                    before = (
                        cumulative_good[:, start - 1]
                        if start
                        else np.zeros(size, dtype=np.float64)
                    )
                    within = np.sum(good[:, start:stop], axis=1)
                    credit[:, start:stop] = before[:, None] + 0.5 * within[:, None]
                numerator[:, metric_index] += np.sum(credit * bad, axis=1)
        exceed += np.sum(numerator >= threshold[None, :] - 1e-12, axis=0)
        completed += size
    return (exceed + 1.0) / (draws + 1.0)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def build_metric_results(
    distances: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    discovery = distances[distances.global_seed.isin(DISCOVERY_SEEDS)].copy()
    expansion = distances[distances.global_seed.isin(EXPANSION_SEEDS)].copy()
    pvalues = block_permutation_pvalues(expansion, draws, seed)
    holm = holm_adjust(pvalues)
    rows = []
    for metric_index, metric in enumerate(METRICS):
        representation, distance = metric.split("__", 1)
        row: dict[str, Any] = {
            "metric": metric,
            "representation": representation,
            "distance": distance,
            "direction": "distance_high_is_bad",
            "availability": "terminal_endpoint_only",
            "preterminal_actionable": False,
            "exploratory_not_confirmatory": True,
            "expansion_block_permutation_p_one_sided": pvalues[metric_index],
            "expansion_holm_p_across_six": holm[metric_index],
        }
        for cohort_name, cohort in (("discovery_crossfit", discovery), ("expansion_primary", expansion)):
            binary = auc_summary(
                cohort[metric].to_numpy(float),
                cohort.label.to_numpy(str),
                cohort.class_id.to_numpy(int),
                BAD,
            )
            mild = auc_summary(
                cohort[metric].to_numpy(float),
                cohort.label.to_numpy(str),
                cohort.class_id.to_numpy(int),
                MILD,
            )
            row[f"{cohort_name}_bad_vs_good_pair_weighted_auc"] = binary["pair_weighted_auc"]
            row[f"{cohort_name}_bad_vs_good_macro_auc"] = binary["macro_auc"]
            row[f"{cohort_name}_mild_vs_good_pair_weighted_auc_descriptive"] = mild[
                "pair_weighted_auc"
            ]
            for class_id in CLASSES:
                row[f"{cohort_name}_bad_vs_good_auc_class_{class_id}"] = binary[
                    "per_class_auc"
                ][class_id]
                row[f"{cohort_name}_bad_count_class_{class_id}"] = binary["counts"][class_id][BAD]
                row[f"{cohort_name}_good_count_class_{class_id}"] = binary["counts"][class_id][GOOD]
                row[f"{cohort_name}_mild_count_class_{class_id}"] = mild["counts"][class_id][MILD]
        rows.append(row)
    return pd.DataFrame(rows)


def build_group_summaries(distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_name, seeds in (("discovery_crossfit", DISCOVERY_SEEDS), ("expansion_primary", EXPANSION_SEEDS)):
        cohort = distances[distances.global_seed.isin(seeds)]
        for metric in METRICS:
            representation, distance = metric.split("__", 1)
            for class_id in CLASSES:
                for label in LABELS:
                    values = cohort.loc[(cohort.class_id == class_id) & (cohort.label == label), metric].to_numpy(float)
                    if len(values) == 0:
                        continue
                    publish_statistics = len(values) >= MINIMUM_GROUP_SIZE
                    rows.append(
                        {
                            "cohort": cohort_name,
                            "representation": representation,
                            "distance": distance,
                            "class_id": class_id,
                            "label": label,
                            "count": len(values),
                            "statistics_suppressed": not publish_statistics,
                            "mean": float(np.mean(values)) if publish_statistics else float("nan"),
                            "standard_deviation": float(np.std(values, ddof=1)) if publish_statistics else float("nan"),
                            "minimum": float(np.min(values)) if publish_statistics else float("nan"),
                            "q25": float(np.quantile(values, 0.25)) if publish_statistics else float("nan"),
                            "median": float(np.median(values)) if publish_statistics else float("nan"),
                            "q75": float(np.quantile(values, 0.75)) if publish_statistics else float("nan"),
                            "maximum": float(np.max(values)) if publish_statistics else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def centered_rank_correlation(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    xr = np.empty(len(x), dtype=np.float64)
    yr = np.empty(len(y), dtype=np.float64)
    for class_id in CLASSES:
        mask = classes == class_id
        xrank = rankdata(x[mask], method="average")
        yrank = rankdata(y[mask], method="average")
        xr[mask] = xrank - xrank.mean()
        yr[mask] = yrank - yrank.mean()
    if np.std(xr) <= 0 or np.std(yr) <= 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def build_correlations(distances: pd.DataFrame, risks: pd.DataFrame) -> pd.DataFrame:
    joined = distances.merge(
        risks, on=["global_seed", "class_id"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(distances):
        raise RuntimeError("preterminal-risk/distance axis differs")
    rows = []
    for cohort_name, seeds in (("discovery_crossfit", DISCOVERY_SEEDS), ("expansion_primary", EXPANSION_SEEDS), ("combined_descriptive", (*DISCOVERY_SEEDS, *EXPANSION_SEEDS))):
        cohort = joined[joined.global_seed.isin(seeds)]
        classes = cohort.class_id.to_numpy(int)
        for risk in PRETERMINAL_RISKS:
            x = cohort[risk].to_numpy(float)
            for metric in METRICS:
                y = cohort[metric].to_numpy(float)
                pooled = spearmanr(x, y).statistic
                rows.append(
                    {
                        "cohort": cohort_name,
                        "preterminal_risk": risk,
                        "endpoint_metric": metric,
                        "scope": "pooled_raw_descriptive",
                        "class_id": "pooled",
                        "count": len(cohort),
                        "spearman_rho": float(pooled),
                        "exploratory_no_multiplicity_claim": True,
                    }
                )
                rows.append(
                    {
                        "cohort": cohort_name,
                        "preterminal_risk": risk,
                        "endpoint_metric": metric,
                        "scope": "pooled_within_class_centered_midranks",
                        "class_id": "pooled",
                        "count": len(cohort),
                        "spearman_rho": centered_rank_correlation(x, y, classes),
                        "exploratory_no_multiplicity_claim": True,
                    }
                )
                for class_id in CLASSES:
                    mask = classes == class_id
                    rho = spearmanr(x[mask], y[mask]).statistic
                    rows.append(
                        {
                            "cohort": cohort_name,
                            "preterminal_risk": risk,
                            "endpoint_metric": metric,
                            "scope": "within_class",
                            "class_id": class_id,
                            "count": int(np.sum(mask)),
                            "spearman_rho": float(rho),
                            "exploratory_no_multiplicity_claim": True,
                        }
                    )
    return pd.DataFrame(rows)


def _payload_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish(
    output: Path,
    metric_results: pd.DataFrame,
    group_summaries: pd.DataFrame,
    correlations: pd.DataFrame,
    fit_summaries: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_identity: str,
) -> None:
    output = output.expanduser().resolve()
    if output != DEFAULT_OUTPUT.expanduser().resolve():
        raise RuntimeError("output differs from frozen protocol")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        metric_results.to_csv(staging / "metric_results.csv", index=False)
        group_summaries.to_csv(staging / "group_summaries.csv", index=False)
        correlations.to_csv(staging / "preterminal_endpoint_correlations.csv", index=False)
        write_json(staging / "reference_fit_summaries.json", {"fits": list(fit_summaries)})
        write_json(staging / "protocol_snapshot.json", dict(protocol))
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        methodology = {
            "schema_version": SCHEMA_VERSION,
            "status": "EXPLORATORY_TERMINAL_ENDPOINT_DISTANCE_AUDIT",
            "warning": (
                "Endpoint distances use the terminal image and are retrospective only. "
                "They cannot be used as online guidance, rollback, or rejection triggers."
            ),
            "reference": (
                "L2-normalized embeddings; per-class discovery clean centroids and clean "
                "neighbors; one pooled within-class Ledoit-Wolf covariance. Discovery is "
                "global_seed-mod-5 cross-fit; expansion uses all discovery clean-good."
            ),
            "test": (
                "class-matched tie-aware bad-vs-good pair-weighted AUC; 100000 one-sided "
                "permutations of intact three-class seed label vectors including mild; "
                "Holm across the six fixed distance-high-is-bad hypotheses"
            ),
            "correlations": (
                "descriptive Spearman only: raw pooled, within-class centered midranks, "
                "and separate classes; no significance or causal claim"
            ),
            "output_privacy": (
                "aggregate tables only; no sample key, label, score, distance, rank, "
                "endpoint path, image, or trace path is emitted; group statistics are "
                "suppressed below the frozen minimum group size of 5"
            ),
        }
        write_json(staging / "methodology.json", methodology)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_EXPLORATORY_ENDPOINT_REPRESENTATION_DISTANCE_AUDIT",
            "metric_count": len(METRICS),
            "permutation_draws": PERMUTATION_DRAWS,
            "permutation_seed": PERMUTATION_SEED,
            "protocol_identity_sha256": protocol_identity,
            "row_level_payload_emitted": False,
            "image_payload_emitted": False,
            "preterminal_actionable": False,
            "exploratory_not_confirmatory": True,
            "minimum_group_size": MINIMUM_GROUP_SIZE,
        }
        write_json(staging / "summary.json", summary)
        payloads = [_payload_record(path) for path in sorted(staging.iterdir())]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "experiment": EXPERIMENT,
            "protocol_identity_sha256": protocol_identity,
            "input_lineage": dict(lineage),
            "files": payloads,
        }
        manifest["identity_sha256"] = canonical_identity(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "source_file_sha256": sha256_file(staging / "analysis_source.py"),
            "row_level_payload_emitted": False,
            "image_payload_emitted": False,
        }
        write_json(staging / "completion.json", completion)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def synthetic_self_test() -> None:
    rng = np.random.default_rng(3)
    records = []
    arrays = {name: [] for name, _ in REPRESENTATIONS}
    for seed in range(20):
        for class_id in CLASSES:
            label = BAD if seed % 10 == 0 else (MILD if seed % 10 == 1 else GOOD)
            records.append({"sample_index": len(records), "global_seed": seed, "class_id": class_id, "label": label})
            for name, dimension in REPRESENTATIONS:
                value = rng.normal(size=dimension)
                if label == BAD:
                    value[0] += 4.0
                arrays[name].append(value)
    frame = pd.DataFrame(records)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    # Exercise reference fitting/scoring directly on a sufficiently large clean set.
    for name, _ in REPRESENTATIONS:
        values = normalize_rows(arrays[name])
        model, summary = fit_reference(values, frame.class_id.to_numpy(int), frame.label.to_numpy(str))
        scores = score_reference(model, values, frame.class_id.to_numpy(int))
        assert summary["pooled_residual_count"] >= 3 * (KNN_K + 1)
        assert set(scores) == set(DISTANCES)
    p = np.array([0.01, 0.04, 0.03, 0.5, 0.2, 0.1])
    adjusted = holm_adjust(p)
    assert np.all(adjusted >= p)
    assert np.isclose(
        centered_rank_correlation(
            np.arange(60.0), np.arange(60.0), frame.class_id.to_numpy(int)
        ),
        1.0,
    )
    privacy_rows = []
    for seed in DISCOVERY_SEEDS:
        for class_id in CLASSES:
            label = BAD if seed == DISCOVERY_SEEDS[0] else GOOD
            row = {"global_seed": seed, "class_id": class_id, "label": label}
            for metric in METRICS:
                row[metric] = float(seed + class_id)
            privacy_rows.append(row)
    privacy_summary = build_group_summaries(pd.DataFrame(privacy_rows))
    suppressed = privacy_summary[privacy_summary["count"] < MINIMUM_GROUP_SIZE]
    statistic_columns = [
        "mean", "standard_deviation", "minimum", "q25", "median", "q75", "maximum"
    ]
    assert len(suppressed) > 0
    assert suppressed["statistics_suppressed"].all()
    assert suppressed[statistic_columns].isna().all().all()
    permutation_rows = []
    for seed in EXPANSION_SEEDS:
        for class_id in CLASSES:
            label = BAD if seed % 40 == 0 else (MILD if seed % 17 == 0 else GOOD)
            row = {"global_seed": seed, "class_id": class_id, "label": label}
            for metric_index, metric in enumerate(METRICS):
                row[metric] = (
                    (2.0 if label == BAD else 0.0)
                    + 0.01 * metric_index
                    + rng.normal(0.0, 0.1)
                )
            permutation_rows.append(row)
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_p = block_permutation_pvalues(
        permutation_frame, draws=99, seed=11, batch_size=20
    )
    assert permutation_p.shape == (len(METRICS),)
    assert np.isfinite(permutation_p).all()
    print("synthetic endpoint-distance audit self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-product", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--original-label-lock", type=Path, default=DEFAULT_ORIGINAL_LABELS)
    parser.add_argument("--expansion-label-lock", type=Path, default=DEFAULT_EXPANSION_LABELS)
    parser.add_argument("--original-visual-product", type=Path, default=DEFAULT_ORIGINAL_VISUAL)
    parser.add_argument("--expansion-visual-product", type=Path, default=DEFAULT_EXPANSION_VISUAL)
    parser.add_argument("--original-primary-product", type=Path, default=DEFAULT_ORIGINAL_PRIMARY)
    parser.add_argument("--expansion-primary-product", type=Path, default=DEFAULT_EXPANSION_PRIMARY)
    parser.add_argument("--permutation-draws", type=int, default=PERMUTATION_DRAWS)
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if (
        args.permutation_draws != PERMUTATION_DRAWS
        or args.permutation_seed != PERMUTATION_SEED
        or args.output.expanduser().resolve() != DEFAULT_OUTPUT.expanduser().resolve()
    ):
        raise ValueError("draws, seed, and output must match the frozen protocol")
    protocol, protocol_identity = validate_protocol(args.protocol_lock)

    # Validate every label-free input and compute no distance before the protocol is fixed.
    index, arrays, embedding_lineage = validate_embedding_product(
        args.embedding_product, protocol
    )
    product_ids = protocol.get("input_identities", {}).get(
        "preterminal_product_manifest_identity_sha256", {}
    )
    ob, ob_lineage = validate_label_free_scalar_product(
        args.original_visual_product,
        DISCOVERY_SEEDS,
        B_FEATURE,
        product_ids.get("original_visual"),
        "original visual B",
    )
    eb, eb_lineage = validate_label_free_scalar_product(
        args.expansion_visual_product,
        EXPANSION_SEEDS,
        B_FEATURE,
        product_ids.get("expansion_visual"),
        "expansion visual B",
    )
    oc, oc_lineage = validate_label_free_scalar_product(
        args.original_primary_product,
        DISCOVERY_SEEDS,
        C_SOURCE_FEATURE,
        product_ids.get("original_primary"),
        "original primary C",
    )
    ec, ec_lineage = validate_label_free_scalar_product(
        args.expansion_primary_product,
        EXPANSION_SEEDS,
        C_SOURCE_FEATURE,
        product_ids.get("expansion_primary"),
        "expansion primary C",
    )
    risks = pd.concat(
        [
            ob.merge(oc, on=["global_seed", "class_id"], validate="one_to_one"),
            eb.merge(ec, on=["global_seed", "class_id"], validate="one_to_one"),
        ],
        ignore_index=True,
    ).rename(columns={B_FEATURE: PRETERMINAL_RISKS[0]})
    risks[PRETERMINAL_RISKS[1]] = -risks.pop(C_SOURCE_FEATURE)
    if len(risks) != 600 or not np.isfinite(risks[list(PRETERMINAL_RISKS)].to_numpy(float)).all():
        raise RuntimeError("frozen B/C preterminal risk table is invalid")

    # Labels are opened only after the embedding family, protocol, and B/C formulas are fixed.
    original_labels, original_label_lineage = validate_label_lock(
        args.original_label_lock,
        DISCOVERY_SEEDS,
        "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
    )
    expansion_labels, expansion_label_lineage = validate_label_lock(
        args.expansion_label_lock,
        EXPANSION_SEEDS,
        "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN",
    )
    frozen_label_ids = protocol.get("input_identities", {}).get(
        "label_consensus_identity_sha256", {}
    )
    if {
        "original": original_label_lineage["consensus_identity_sha256"],
        "expansion": expansion_label_lineage["consensus_identity_sha256"],
    } != frozen_label_ids:
        raise RuntimeError("label consensus differs from the frozen protocol")
    labels = pd.concat([original_labels, expansion_labels], ignore_index=True)
    distances, fit_summaries = compute_distances(index, arrays, labels)
    metric_results = build_metric_results(
        distances, args.permutation_draws, args.permutation_seed
    )
    group_summaries = build_group_summaries(distances)
    correlations = build_correlations(distances, risks)
    lineage = {
        "protocol_identity_sha256": protocol_identity,
        "embedding_product": embedding_lineage,
        "original_label_lock": original_label_lineage,
        "expansion_label_lock": expansion_label_lineage,
        "original_visual_B": ob_lineage,
        "expansion_visual_B": eb_lineage,
        "original_primary_C": oc_lineage,
        "expansion_primary_C": ec_lineage,
    }
    publish(
        args.output,
        metric_results,
        group_summaries,
        correlations,
        fit_summaries,
        lineage,
        protocol,
        protocol_identity,
    )
    print(f"published aggregate endpoint-distance audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()

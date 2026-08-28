#!/usr/bin/env python3
"""Exploratory full-inventory screen with an independent replication cohort.

The original evaluation cohort (seeds 50..129) is used *only* to choose one
scalar reduction and one direction per trajectory track.  Those choices are
then frozen in memory and evaluated on the disjoint expansion cohort
(seeds 130..249).  Mild/disputed endpoints are excluded throughout.

This is deliberately an exploratory, next-candidate-generating analysis.  A
replicating track winner is not a confirmatory result because thousands of
features and many tracks were screened before it was named.  The script emits
aggregate track tables only; joined sample labels and feature values never
leave memory.
"""

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
from scipy.stats import rankdata


KEYS = ("global_seed", "class_id")
GOOD = "clean_good"
BAD = "clear_bad"
MILD = "mild_or_disputed"
LABELS = (GOOD, BAD, MILD)
CLASSES = (207, 602, 795)
DISCOVERY_SEEDS = tuple(range(50, 130))
REPLICATION_SEEDS = tuple(range(130, 250))
DEFAULT_PERMUTATION_DRAWS = 100_000
DEFAULT_PERMUTATION_SEED = 20260827
EXPECTED_PRIMARY_FEATURES = 2_871
EXPECTED_POSTERIOR_FEATURES = 3_922
EXPECTED_NORMALIZED_POSTERIOR_FEATURES = 51
EXPECTED_TOTAL_FEATURES = 6_844
EXPECTED_TOTAL_TRACKS = 190

DEFAULT_BASE = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence"
)
DEFAULT_ORIGINAL_LABELS = Path(
    "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
DEFAULT_EXPANSION_LABELS = Path(
    "experiments/annotations/dit_expansion_eval360_adjudicated_consensus_lock_v1"
)
DEFAULT_ORIGINAL_PRIMARY = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/custom_label_free_v1"
)
DEFAULT_ORIGINAL_POSTERIOR = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/posterior_label_free_v1"
)
DEFAULT_EXPANSION_PRIMARY = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/primary_label_free_v1"
)
DEFAULT_EXPANSION_POSTERIOR = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/posterior_label_free_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/inventory_replication_exploratory_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("identity_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_identity(value: dict[str, Any], description: str) -> str:
    observed = value.get("identity_sha256")
    expected = canonical_sha256(value)
    if observed != expected:
        raise RuntimeError(f"{description} canonical identity mismatch")
    return expected


def validate_manifest_members(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
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


def _as_bool(series: pd.Series, description: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False}
    lowered = series.astype(str).str.lower()
    if not lowered.isin(mapping).all():
        raise RuntimeError(f"{description} is not Boolean")
    return lowered.map(mapping).astype(bool)


def validate_label_lock(
    root: Path,
    expected_seeds: tuple[int, ...],
    expected_status: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"label lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    completion_path = root / "completion.json"
    manifest = read_json(manifest_path)
    consensus = read_json(consensus_path)
    completion = read_json(completion_path)
    manifest_id = require_identity(manifest, "label-lock manifest")
    consensus_id = require_identity(consensus, "label consensus")
    members = validate_manifest_members(root, manifest)
    expected_n = len(expected_seeds) * len(CLASSES)
    if (
        manifest.get("status") != "complete"
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
        raise RuntimeError(f"invalid final adjudicated label lock: {root}")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_n:
        raise RuntimeError("label consensus has the wrong row count")
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("label row is not an object")
        seed = row.get("global_seed")
        class_id = row.get("class_id")
        label = row.get("primary_label")
        if (
            seed not in expected_seeds
            or class_id not in CLASSES
            or label not in LABELS
            or row.get("binary_primary_included") != (label != MILD)
        ):
            raise RuntimeError("invalid label row")
        records.append(
            {"global_seed": int(seed), "class_id": int(class_id), "label": label}
        )
    frame = pd.DataFrame(records)
    expected_axis = {(seed, class_id) for seed in expected_seeds for class_id in CLASSES}
    if (
        frame.duplicated(list(KEYS)).any()
        or set(map(tuple, frame[list(KEYS)].to_numpy())) != expected_axis
        or frame["label"].value_counts().to_dict() != consensus.get("counts")
    ):
        raise RuntimeError("label-lock cohort/counts are inconsistent")
    return frame, {
        "manifest_identity_sha256": manifest_id,
        "consensus_identity_sha256": consensus_id,
        "manifest_file_sha256": sha256_file(manifest_path),
        "counts": {label: int((frame["label"] == label).sum()) for label in LABELS},
    }


def validate_feature_product(
    root: Path,
    expected_seeds: tuple[int, ...],
    product_name: str,
    *,
    sample_filename: str = "sample_features.csv",
    require_supervision_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"feature product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    completion = read_json(root / "completion.json")
    summary = read_json(root / "summary.json")
    manifest_id = require_identity(manifest, f"{product_name} manifest")
    members = validate_manifest_members(root, manifest)
    trace_identities = manifest.get("trace_identity_sha256_ordered")
    if (
        not isinstance(trace_identities, list)
        or not trace_identities
        or not all(
            isinstance(value, str) and len(value) == 64 for value in trace_identities
        )
    ):
        raise RuntimeError(f"{product_name} has invalid ordered trace identities")
    required = {sample_filename, "feature_catalog.csv", "summary.json", "source_inventory.json"}
    supervision_audit = summary.get("supervision_audit")
    if not required.issubset(members):
        raise RuntimeError(f"{product_name} manifest lacks required members")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_id
        or summary.get("labels_joined", False) is not False
        or summary.get("supervision_audit", {}).get("labels_read_or_emitted", False)
        is not False
        or (
            require_supervision_audit
            and (
                not isinstance(supervision_audit, dict)
                or supervision_audit.get("labels_read_or_emitted") is not False
            )
        )
    ):
        raise RuntimeError(f"invalid or supervised feature product: {root}")
    frame = pd.read_csv(root / sample_filename)
    catalog = pd.read_csv(root / "feature_catalog.csv")
    required_catalog = {
        "feature",
        "track",
        "family",
        "reduction",
        "latest_required_sampling_step",
        "preterminal_actionable",
    }
    if not required_catalog.issubset(catalog.columns):
        raise RuntimeError(f"{product_name} catalog schema is incomplete")
    if catalog["feature"].duplicated().any() or catalog["feature"].isna().any():
        raise RuntimeError(f"{product_name} catalog feature names are not unique")
    features = catalog["feature"].astype(str).tolist()
    if not set(features).issubset(frame.columns):
        raise RuntimeError(f"{product_name} sample table lacks catalog features")
    if "label" in frame and not frame["label"].eq("unlabeled").all():
        raise RuntimeError(f"{product_name} contains labels")
    if "raw_consensus_label" in frame and not frame["raw_consensus_label"].isna().all():
        raise RuntimeError(f"{product_name} contains raw consensus labels")
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError(f"{product_name} sample axis has duplicates")
    cohort = frame[
        frame["global_seed"].isin(expected_seeds) & frame["class_id"].isin(CLASSES)
    ].copy()
    expected_axis = {(seed, class_id) for seed in expected_seeds for class_id in CLASSES}
    if set(map(tuple, cohort[list(KEYS)].to_numpy())) != expected_axis:
        raise RuntimeError(f"{product_name} does not contain the exact requested cohort")
    values = cohort[features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{product_name} contains non-finite candidate features")
    catalog = catalog.copy()
    catalog["feature"] = catalog["feature"].astype(str)
    catalog["track"] = catalog["track"].astype(str)
    catalog["preterminal_actionable"] = _as_bool(
        catalog["preterminal_actionable"], f"{product_name} preterminal_actionable"
    )
    catalog["latest_required_sampling_step"] = pd.to_numeric(
        catalog["latest_required_sampling_step"], errors="raise"
    ).astype(int)
    return cohort[list(KEYS) + features], catalog, {
        "manifest_identity_sha256": manifest_id,
        "manifest_file_sha256": sha256_file(manifest_path),
        "sample_features_filename": sample_filename,
        "sample_features_sha256": sha256_file(root / sample_filename),
        "feature_catalog_sha256": sha256_file(root / "feature_catalog.csv"),
        "analysis_source_sha256": manifest.get("analysis_source_sha256"),
        "imported_validation_helper_sha256": manifest.get(
            "imported_validation_helper_sha256"
        ),
        "full_product_sample_count": int(summary.get("sample_count", len(frame))),
        "selected_cohort_sample_count": len(cohort),
        "scalar_feature_count": len(features),
        "track_count": int(catalog["track"].nunique()),
        "trace_identity_ordered_sha256": hashlib.sha256(
            json.dumps(
                trace_identities,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "trace_identity_count": len(trace_identities),
    }


def validate_catalog_replication(
    discovery: pd.DataFrame, replication: pd.DataFrame, description: str
) -> None:
    columns = [
        "feature",
        "track",
        "family",
        "reduction",
        "latest_required_sampling_step",
        "preterminal_actionable",
    ]
    left = discovery[columns].sort_values("feature").reset_index(drop=True)
    right = replication[columns].sort_values("feature").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError(f"{description} discovery/replication catalogs differ")


def merge_products(
    primary: pd.DataFrame,
    posterior: pd.DataFrame,
    primary_catalog: pd.DataFrame,
    posterior_catalog: pd.DataFrame,
    normalized_posterior: pd.DataFrame | None = None,
    normalized_posterior_catalog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_features = primary_catalog["feature"].astype(str).tolist()
    posterior_features = posterior_catalog["feature"].astype(str).tolist()
    if set(primary_features).intersection(posterior_features):
        raise RuntimeError("primary and posterior feature names overlap")
    merged = primary.merge(
        posterior[list(KEYS) + posterior_features],
        on=list(KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(primary):
        raise RuntimeError("primary/posterior sample axes differ")
    pc = primary_catalog.copy()
    qc = posterior_catalog.copy()
    pc.insert(0, "source_product", "primary")
    qc.insert(0, "source_product", "posterior")
    catalogs = [pc, qc]
    if (normalized_posterior is None) != (normalized_posterior_catalog is None):
        raise RuntimeError("normalized posterior frame/catalog must be supplied together")
    if normalized_posterior is not None and normalized_posterior_catalog is not None:
        normalized_features = normalized_posterior_catalog["feature"].astype(str).tolist()
        if set(primary_features + posterior_features).intersection(normalized_features):
            raise RuntimeError("normalized posterior feature names overlap another product")
        merged = merged.merge(
            normalized_posterior[list(KEYS) + normalized_features],
            on=list(KEYS),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(primary):
            raise RuntimeError("normalized posterior sample axis differs")
        nc = normalized_posterior_catalog.copy()
        nc.insert(0, "source_product", "normalized_posterior")
        catalogs.append(nc)
    return merged, pd.concat(catalogs, ignore_index=True, sort=False)


def join_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined = features.merge(labels, on=list(KEYS), how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise RuntimeError("feature and label axes differ")
    return joined[joined["label"].isin((GOOD, BAD))].reset_index(drop=True)


def auc_bad_high(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    bad = np.asarray(scores[labels == BAD], dtype=np.float64)
    good = np.sort(np.asarray(scores[labels == GOOD], dtype=np.float64))
    pairs = len(bad) * len(good)
    if pairs == 0:
        return float("nan"), 0
    left = np.searchsorted(good, bad, side="left")
    right = np.searchsorted(good, bad, side="right")
    wins = float(np.sum(left + 0.5 * (right - left)))
    return wins / pairs, pairs


def auc_summary(
    scores: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    numer = 0.0
    denominator = 0
    per_class: dict[int, float] = {}
    counts: dict[int, dict[str, int]] = {}
    for class_id in CLASSES:
        mask = classes == class_id
        y = labels[mask]
        auc, pairs = auc_bad_high(scores[mask], y)
        per_class[class_id] = auc
        counts[class_id] = {
            "bad": int(np.sum(y == BAD)), "good": int(np.sum(y == GOOD))
        }
        if pairs:
            numer += auc * pairs
            denominator += pairs
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return {
        "pair_weighted_auc": numer / denominator if denominator else float("nan"),
        "macro_auc": float(np.mean(finite)) if finite else float("nan"),
        "per_class_auc": per_class,
        "counts": counts,
        "pair_count": denominator,
    }


def _tier_catalog(catalog: pd.DataFrame, tier: str) -> pd.DataFrame:
    actionable = catalog["preterminal_actionable"]
    step = catalog["latest_required_sampling_step"]
    if tier == "preterminal_step149":
        return catalog[actionable & (step <= 149)].copy()
    if tier == "preterminal_step199":
        # This intentionally contains the <=149 features too.  It answers the
        # separate question: what is the best track reduction if we can wait
        # fifty more steps?
        return catalog[actionable & (step <= 199)].copy()
    if tier == "retrospective":
        return catalog[(~actionable) | (step > 199)].copy()
    raise ValueError(tier)


def select_track_winners(
    discovery: pd.DataFrame, catalog: pd.DataFrame, tier: str
) -> pd.DataFrame:
    eligible = _tier_catalog(catalog, tier)
    labels = discovery["label"].to_numpy()
    classes = discovery["class_id"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for track, group in eligible.groupby("track", sort=True):
        candidates: list[dict[str, Any]] = []
        for item in group.sort_values("feature").to_dict("records"):
            raw = discovery[item["feature"]].to_numpy(dtype=np.float64)
            high = auc_summary(raw, labels, classes)
            low_auc = 1.0 - high["pair_weighted_auc"]
            if high["pair_weighted_auc"] >= low_auc:
                direction = "bad_high"
                signed = raw
            else:
                direction = "bad_low"
                signed = -raw
            stats = auc_summary(signed, labels, classes)
            candidates.append({**item, "direction": direction, "stats": stats})
        candidates.sort(
            key=lambda item: (
                -item["stats"]["pair_weighted_auc"],
                -item["stats"]["macro_auc"],
                item["feature"],
                item["direction"],
            )
        )
        winner = candidates[0]
        stats = winner.pop("stats")
        row = {
            "tier": tier,
            "source_product": winner["source_product"],
            "track": track,
            "family": winner["family"],
            "feature": winner["feature"],
            "reduction": winner["reduction"],
            "direction": winner["direction"],
            "latest_required_sampling_step": int(
                winner["latest_required_sampling_step"]
            ),
            "preterminal_actionable": bool(winner["preterminal_actionable"]),
            "discovery_pair_weighted_auc": stats["pair_weighted_auc"],
            "discovery_macro_auc": stats["macro_auc"],
            "discovery_pair_count": stats["pair_count"],
        }
        for class_id in CLASSES:
            row[f"discovery_auc_class_{class_id}"] = stats["per_class_auc"][class_id]
            row[f"discovery_bad_count_class_{class_id}"] = stats["counts"][class_id]["bad"]
            row[f"discovery_good_count_class_{class_id}"] = stats["counts"][class_id]["good"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_replication(
    replication: pd.DataFrame, winners: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = replication["label"].to_numpy()
    classes = replication["class_id"].to_numpy(dtype=int)
    result = winners.copy()
    score_columns: list[np.ndarray] = []
    for index, row in result.iterrows():
        raw = replication[row["feature"]].to_numpy(dtype=np.float64)
        signed = raw if row["direction"] == "bad_high" else -raw
        score_columns.append(signed)
        stats = auc_summary(signed, labels, classes)
        result.loc[index, "replication_pair_weighted_auc"] = stats[
            "pair_weighted_auc"
        ]
        result.loc[index, "replication_macro_auc"] = stats["macro_auc"]
        result.loc[index, "replication_pair_count"] = stats["pair_count"]
        for class_id in CLASSES:
            result.loc[index, f"replication_auc_class_{class_id}"] = stats[
                "per_class_auc"
            ][class_id]
            result.loc[index, f"replication_bad_count_class_{class_id}"] = stats[
                "counts"
            ][class_id]["bad"]
            result.loc[index, f"replication_good_count_class_{class_id}"] = stats[
                "counts"
            ][class_id]["good"]
    matrix = (
        np.column_stack(score_columns)
        if score_columns
        else np.empty((len(replication), 0), dtype=np.float64)
    )
    return result, matrix


def permutation_pvalues(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    observed_auc: np.ndarray,
    draws: int,
    seed: int,
    batch_size: int = 2_000,
) -> np.ndarray:
    """One-sided within-class fixed-count permutation p-values for all columns."""

    if draws < 1 or scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("invalid permutation inputs")
    feature_count = scores.shape[1]
    if feature_count == 0:
        return np.empty((0,), dtype=np.float64)
    class_blocks: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []
    denominator = 0
    for class_id in CLASSES:
        indices = np.flatnonzero(classes == class_id)
        y = labels[indices]
        n_bad = int(np.sum(y == BAD))
        n_good = int(np.sum(y == GOOD))
        if n_bad == 0 or n_good == 0:
            continue
        ranks = np.column_stack(
            [rankdata(scores[indices, column], method="average") for column in range(feature_count)]
        ).astype(np.float64)
        pair_count = n_bad * n_good
        denominator += pair_count
        class_blocks.append(
            (indices, ranks, n_bad, pair_count, n_bad * (n_bad + 1.0) / 2.0)
        )
    if not class_blocks or denominator == 0:
        return np.full(feature_count, np.nan)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(feature_count, dtype=np.int64)
    threshold = np.asarray(observed_auc, dtype=np.float64) * denominator
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        numerator = np.zeros((size, feature_count), dtype=np.float64)
        for indices, ranks, n_bad, _pair_count, offset in class_blocks:
            # Independent continuous random keys induce a uniform subset of
            # fixed size.  The same label permutations are shared by all track
            # winners, which is valid and makes the screen tractable.
            keys = rng.random((size, len(indices)))
            chosen = np.argpartition(keys, n_bad - 1, axis=1)[:, :n_bad]
            mask = np.zeros((size, len(indices)), dtype=np.float64)
            mask[np.arange(size)[:, None], chosen] = 1.0
            numerator += mask @ ranks - offset
        exceed += np.sum(numerator >= threshold[None, :] - 1e-12, axis=0)
        completed += size
    return (exceed + 1.0) / (draws + 1.0)


def adjust_pvalues(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p.copy(), p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    bh_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    holm_ranked = np.maximum.accumulate(ranked * (m - np.arange(m)))
    bh = np.empty(m, dtype=np.float64)
    holm = np.empty(m, dtype=np.float64)
    bh[order] = np.minimum(bh_ranked, 1.0)
    holm[order] = np.minimum(holm_ranked, 1.0)
    return bh, holm


def analyze(
    discovery: pd.DataFrame,
    replication: pd.DataFrame,
    catalog: pd.DataFrame,
    draws: int,
    permutation_seed: int,
) -> dict[str, pd.DataFrame]:
    tiers = ("preterminal_step149", "preterminal_step199", "retrospective")
    outputs: dict[str, pd.DataFrame] = {}
    evaluated: list[pd.DataFrame] = []
    score_matrices: list[np.ndarray] = []
    for tier in tiers:
        winners = select_track_winners(discovery, catalog, tier)
        table, scores = evaluate_replication(replication, winners)
        evaluated.append(table)
        score_matrices.append(scores)
    combined = pd.concat(evaluated, ignore_index=True)
    all_scores = np.column_stack(score_matrices)
    pvalues = permutation_pvalues(
        all_scores,
        replication["label"].to_numpy(),
        replication["class_id"].to_numpy(dtype=int),
        combined["replication_pair_weighted_auc"].to_numpy(dtype=np.float64),
        draws,
        permutation_seed,
    )
    bh, holm = adjust_pvalues(pvalues)
    combined["replication_permutation_p_one_sided"] = pvalues
    combined["replication_bh_q_across_all_track_tier_winners"] = bh
    combined["replication_holm_p_across_all_track_tier_winners"] = holm
    combined["exploratory_not_confirmatory"] = True
    for tier in tiers:
        outputs[tier] = (
            combined[combined["tier"] == tier]
            .sort_values(
                ["replication_pair_weighted_auc", "discovery_pair_weighted_auc", "track"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
    return outputs


def _payload_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish(
    output: Path,
    tables: dict[str, pd.DataFrame],
    lineage: dict[str, Any],
    draws: int,
    permutation_seed: int,
    cohort_summary: dict[str, Any],
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        filenames = {
            "preterminal_step149": "primary_preterminal_step149_track_winners.csv",
            "preterminal_step199": "secondary_preterminal_step199_track_winners.csv",
            "retrospective": "retrospective_track_winners.csv",
        }
        for tier, table in tables.items():
            table.to_csv(staging / filenames[tier], index=False)
        source = Path(__file__).resolve()
        shutil.copyfile(source, staging / "analysis_source.py")
        methodology = {
            "schema_version": 1,
            "status": "EXPLORATORY_NEXT_CANDIDATE_GENERATION_ONLY",
            "warning": (
                "Feature and direction were selected after screening the original cohort. "
                "The disjoint expansion estimates replication, but this full-inventory "
                "exercise is exploratory and cannot confirm a newly named detector."
            ),
            "labels": {
                "positive": BAD,
                "negative": GOOD,
                "excluded": MILD,
            },
            "selection": (
                "Within each tier and trajectory track, maximize original-cohort "
                "within-class pair-weighted directional AUC; deterministic ties use "
                "macro AUC then lexicographic feature name."
            ),
            "tiers": {
                "primary_preterminal_step149": (
                    "preterminal_actionable=true and latest_required_sampling_step<=149"
                ),
                "secondary_preterminal_step199": (
                    "preterminal_actionable=true and latest_required_sampling_step<=199; "
                    "intentionally includes the <=149 candidate set"
                ),
                "retrospective": (
                    "preterminal_actionable=false or latest_required_sampling_step>199"
                ),
            },
            "replication_test": {
                "statistic": "within-class pair-count-weighted AUC, frozen direction",
                "null": "fixed bad count, labels permuted separately within each class",
                "alternative": "AUC greater than chance in the frozen direction",
                "draws": draws,
                "rng": "numpy.default_rng(PCG64)",
                "rng_seed": permutation_seed,
                "monte_carlo_p": "(1 + exceedances)/(1 + draws)",
                "multiple_testing": (
                    "BH and Holm jointly across every emitted track-tier winner; tiers "
                    "overlap, so identical or related hypotheses may appear more than once"
                ),
            },
            "output_privacy": (
                "aggregate track tables only; no sample key, seed, row label, row score, "
                "endpoint, or trace path is emitted"
            ),
        }
        write_json(staging / "methodology.json", methodology)
        summary = {
            "schema_version": 1,
            "status": "COMPLETE_EXPLORATORY_INVENTORY_REPLICATION_SCREEN",
            "exploratory_not_confirmatory": True,
            "cohorts": cohort_summary,
            "inventory": {
                "scalar_feature_count": int(
                    lineage["original_primary"]["scalar_feature_count"]
                    + lineage["original_posterior"]["scalar_feature_count"]
                    + lineage["original_normalized_posterior"]["scalar_feature_count"]
                ),
                "catalog_track_count": int(
                    len(
                        set().union(
                            *(set(table["track"]) for table in tables.values())
                        )
                    )
                ),
                "winner_counts": {tier: len(table) for tier, table in tables.items()},
            },
            "permutation_draws": draws,
            "permutation_seed": permutation_seed,
            "row_level_payload_emitted": False,
        }
        write_json(staging / "summary.json", summary)
        payload_names = [*filenames.values(), "analysis_source.py", "methodology.json", "summary.json"]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "dit_bad_good_inventory_replication_exploratory_v1",
            "exploratory_not_confirmatory": True,
            "input_lineage": lineage,
            "files": [_payload_record(staging / name) for name in payload_names],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "source_file_sha256": sha256_file(staging / "analysis_source.py"),
            "row_level_payload_emitted": False,
        }
        write_json(staging / "completion.json", completion)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def synthetic_self_test() -> None:
    rng = np.random.default_rng(7)
    records: list[dict[str, Any]] = []
    for cohort_shift in (0, 100):
        for class_id in CLASSES:
            for index in range(10):
                bad = index < 2
                records.append(
                    {
                        "global_seed": cohort_shift + index,
                        "class_id": class_id,
                        "label": BAD if bad else GOOD,
                        "signal__early": (4.0 if bad else 0.0) + rng.normal(0, 0.05),
                        "signal__late": (-3.0 if bad else 0.0) + rng.normal(0, 0.05),
                        "noise__early": rng.normal(),
                        "normalized__early": (2.0 if bad else 0.0)
                        + rng.normal(0, 0.05),
                    }
                )
    frame = pd.DataFrame(records)
    discovery = frame[frame["global_seed"] < 100].reset_index(drop=True)
    replication = frame[frame["global_seed"] >= 100].reset_index(drop=True)
    catalog = pd.DataFrame(
        [
            {
                "source_product": "synthetic",
                "feature": "signal__early",
                "track": "signal",
                "family": "test",
                "reduction": "early",
                "latest_required_sampling_step": 100,
                "preterminal_actionable": True,
            },
            {
                "source_product": "normalized_posterior",
                "feature": "normalized__early",
                "track": "normalized_signal",
                "family": "normalized_posterior_test",
                "reduction": "early",
                "latest_required_sampling_step": 120,
                "preterminal_actionable": True,
            },
            {
                "source_product": "synthetic",
                "feature": "signal__late",
                "track": "signal",
                "family": "test",
                "reduction": "late",
                "latest_required_sampling_step": 180,
                "preterminal_actionable": True,
            },
            {
                "source_product": "synthetic",
                "feature": "noise__early",
                "track": "noise",
                "family": "test",
                "reduction": "early",
                "latest_required_sampling_step": 100,
                "preterminal_actionable": True,
            },
        ]
    )
    tables = analyze(discovery, replication, catalog, draws=999, permutation_seed=11)
    primary = tables["preterminal_step149"]
    signal = primary[primary["track"] == "signal"].iloc[0]
    assert signal["feature"] == "signal__early"
    assert signal["direction"] == "bad_high"
    assert signal["replication_pair_weighted_auc"] == 1.0
    normalized = primary[primary["track"] == "normalized_signal"].iloc[0]
    assert normalized["source_product"] == "normalized_posterior"
    assert normalized["replication_pair_weighted_auc"] == 1.0
    p = np.array([0.01, 0.04, 0.03])
    bh, holm = adjust_pvalues(p)
    assert np.allclose(bh, [0.03, 0.04, 0.04])
    assert np.allclose(holm, [0.03, 0.06, 0.06])
    temporary_parent = Path(tempfile.mkdtemp(prefix="inventory-replication-selftest-"))
    try:
        output = temporary_parent / "published"
        fake_lineage = {
            "original_primary": {"scalar_feature_count": 2},
            "original_posterior": {"scalar_feature_count": 1},
            "original_normalized_posterior": {"scalar_feature_count": 0},
        }
        publish(
            output,
            tables,
            fake_lineage,
            draws=999,
            permutation_seed=11,
            cohort_summary={"synthetic": {"binary_included": len(frame)}},
        )
        manifest = read_json(output / "manifest.json")
        completion = read_json(output / "completion.json")
        assert require_identity(manifest, "synthetic manifest") == completion[
            "manifest_identity_sha256"
        ]
        validate_manifest_members(output, manifest)
        try:
            publish(
                output,
                tables,
                fake_lineage,
                draws=999,
                permutation_seed=11,
                cohort_summary={},
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("publisher overwrote an existing result")
    finally:
        shutil.rmtree(temporary_parent)
    print("synthetic self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-label-lock", type=Path, default=DEFAULT_ORIGINAL_LABELS)
    parser.add_argument("--expansion-label-lock", type=Path, default=DEFAULT_EXPANSION_LABELS)
    parser.add_argument("--original-primary", type=Path, default=DEFAULT_ORIGINAL_PRIMARY)
    parser.add_argument("--original-posterior", type=Path, default=DEFAULT_ORIGINAL_POSTERIOR)
    parser.add_argument("--expansion-primary", type=Path, default=DEFAULT_EXPANSION_PRIMARY)
    parser.add_argument("--expansion-posterior", type=Path, default=DEFAULT_EXPANSION_POSTERIOR)
    parser.add_argument(
        "--original-normalized-posterior-root", type=Path
    )
    parser.add_argument(
        "--expansion-normalized-posterior-root", type=Path
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-draws", type=int, default=DEFAULT_PERMUTATION_DRAWS)
    parser.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if (
        args.original_normalized_posterior_root is None
        or args.expansion_normalized_posterior_root is None
    ):
        raise ValueError(
            "both normalized-posterior roots are required outside --self-test"
        )
    if args.permutation_draws < 1:
        raise ValueError("--permutation-draws must be positive")
    original_labels, original_label_lineage = validate_label_lock(
        args.original_label_lock,
        DISCOVERY_SEEDS,
        "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
    )
    expansion_labels, expansion_label_lineage = validate_label_lock(
        args.expansion_label_lock,
        REPLICATION_SEEDS,
        "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN",
    )
    op, opc, op_lineage = validate_feature_product(
        args.original_primary, DISCOVERY_SEEDS, "original primary"
    )
    oq, oqc, oq_lineage = validate_feature_product(
        args.original_posterior, DISCOVERY_SEEDS, "original posterior"
    )
    ep, epc, ep_lineage = validate_feature_product(
        args.expansion_primary, REPLICATION_SEEDS, "expansion primary"
    )
    eq, eqc, eq_lineage = validate_feature_product(
        args.expansion_posterior, REPLICATION_SEEDS, "expansion posterior"
    )
    on, onc, on_lineage = validate_feature_product(
        args.original_normalized_posterior_root,
        DISCOVERY_SEEDS,
        "original normalized posterior",
        sample_filename="sample_features_label_free.csv",
        require_supervision_audit=True,
    )
    en, enc, en_lineage = validate_feature_product(
        args.expansion_normalized_posterior_root,
        REPLICATION_SEEDS,
        "expansion normalized posterior",
        sample_filename="sample_features_label_free.csv",
        require_supervision_audit=True,
    )
    validate_catalog_replication(opc, epc, "primary")
    validate_catalog_replication(oqc, eqc, "posterior")
    validate_catalog_replication(onc, enc, "normalized posterior")
    if (
        op_lineage["scalar_feature_count"] != EXPECTED_PRIMARY_FEATURES
        or ep_lineage["scalar_feature_count"] != EXPECTED_PRIMARY_FEATURES
        or oq_lineage["scalar_feature_count"] != EXPECTED_POSTERIOR_FEATURES
        or eq_lineage["scalar_feature_count"] != EXPECTED_POSTERIOR_FEATURES
        or on_lineage["scalar_feature_count"]
        != EXPECTED_NORMALIZED_POSTERIOR_FEATURES
        or en_lineage["scalar_feature_count"]
        != EXPECTED_NORMALIZED_POSTERIOR_FEATURES
        or op_lineage["feature_catalog_sha256"]
        != ep_lineage["feature_catalog_sha256"]
        or oq_lineage["feature_catalog_sha256"]
        != eq_lineage["feature_catalog_sha256"]
        or on_lineage["feature_catalog_sha256"]
        != en_lineage["feature_catalog_sha256"]
        or op_lineage["trace_identity_ordered_sha256"]
        != oq_lineage["trace_identity_ordered_sha256"]
        or ep_lineage["trace_identity_ordered_sha256"]
        != eq_lineage["trace_identity_ordered_sha256"]
        or ep_lineage["trace_identity_ordered_sha256"]
        != en_lineage["trace_identity_ordered_sha256"]
        or op_lineage["trace_identity_count"] != 100
        or oq_lineage["trace_identity_count"] != 100
        or ep_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or eq_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or on_lineage["trace_identity_count"]
        not in (len(DISCOVERY_SEEDS), 100)
        or en_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or op_lineage["analysis_source_sha256"]
        != ep_lineage["analysis_source_sha256"]
        or oq_lineage["analysis_source_sha256"] != eq_lineage["analysis_source_sha256"]
        or oq_lineage["imported_validation_helper_sha256"]
        != eq_lineage["imported_validation_helper_sha256"]
        or oq_lineage["imported_validation_helper_sha256"]
        != op_lineage["analysis_source_sha256"]
        or on_lineage["analysis_source_sha256"]
        != en_lineage["analysis_source_sha256"]
    ):
        raise RuntimeError("discovery and replication extractor lineages differ")
    if (
        on_lineage["trace_identity_count"] == op_lineage["trace_identity_count"]
        and on_lineage["trace_identity_ordered_sha256"]
        != op_lineage["trace_identity_ordered_sha256"]
    ):
        raise RuntimeError("original normalized posterior came from different traces")
    original_features, catalog = merge_products(op, oq, opc, oqc, on, onc)
    expansion_features, expansion_catalog = merge_products(ep, eq, epc, eqc, en, enc)
    validate_catalog_replication(catalog, expansion_catalog, "combined")
    if (
        catalog["feature"].nunique() != EXPECTED_TOTAL_FEATURES
        or len(catalog) != EXPECTED_TOTAL_FEATURES
        or catalog["track"].nunique() != EXPECTED_TOTAL_TRACKS
    ):
        raise RuntimeError("combined inventory is not the frozen 6844-feature/190-track set")
    discovery = join_labels(original_features, original_labels)
    replication = join_labels(expansion_features, expansion_labels)
    tables = analyze(
        discovery,
        replication,
        catalog,
        draws=args.permutation_draws,
        permutation_seed=args.permutation_seed,
    )
    lineage = {
        "original_label_lock": original_label_lineage,
        "expansion_label_lock": expansion_label_lineage,
        "original_primary": op_lineage,
        "original_posterior": oq_lineage,
        "original_normalized_posterior": on_lineage,
        "expansion_primary": ep_lineage,
        "expansion_posterior": eq_lineage,
        "expansion_normalized_posterior": en_lineage,
    }
    cohort_summary = {
        "discovery": {
            "total_locked": len(original_labels),
            "binary_included": len(discovery),
            "counts": original_label_lineage["counts"],
            "class_count": len(CLASSES),
        },
        "independent_replication": {
            "total_locked": len(expansion_labels),
            "binary_included": len(replication),
            "counts": expansion_label_lineage["counts"],
            "class_count": len(CLASSES),
        },
    }
    publish(
        args.output,
        tables,
        lineage,
        args.permutation_draws,
        args.permutation_seed,
        cohort_summary,
    )
    print(f"published aggregate exploratory screen: {args.output.resolve()}")


if __name__ == "__main__":
    main()

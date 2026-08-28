#!/usr/bin/env python3
"""Evaluate the frozen DiT Doob-consistency discovery score.

The script first materializes and hashes a complete label-free score table.
Only after that table is immutable in the staging directory does it open the
already locked targeted-100 visual consensus.  This is discovery-only: the
same labels were available before this candidate existed and cannot validate
or authorize an intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/dit_doob_consistency_discovery_v1.json"
DEFAULT_PROBES = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_probe_v1/shard_00_of_01"
)
DEFAULT_LABELS = (
    ROOT
    / "experiments/annotations/dit_targeted100_adjudicated_consensus_lock_v2/consensus_locked.json"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_analysis_v1"
)
EXPECTED_CONFIG_SHA256 = "a6ae08eb0846eb1e9cd25ed1696147e965255b54dc9ae3521618e7312869f0ec"
EXPECTED_RECEIPT_IDENTITY = "1c9d03669c497cd586adc30352f30c44a5306f99e63f5d8285b4e4fa7c64b55f"
EXPECTED_LABEL_IDENTITY = "693b8f7ce291ed6006c1161baf75c2769d035633479fd4903d1e7f67ece32bbe"
CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
SEEDS = tuple(range(10, 30))
CHECKPOINTS = (99, 149, 199)
HORIZONS = (1, 2, 4, 8, 16)
PERMUTATIONS = 100000
PERMUTATION_SEED = 2026082811
BOOTSTRAPS = 100000
BOOTSTRAP_SEED = 2026082812


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_self_hash(value: Mapping[str, Any], key: str, *, context: str) -> None:
    payload = dict(value)
    observed = payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"self hash failed: {context}")


def load_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError("CSV rows do not have one ordered schema")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def auc_higher(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    count_positive = int(positive.sum())
    count_negative = len(positive) - count_positive
    if count_positive == 0 or count_negative == 0:
        return None
    ranks = midranks(values)
    u = float(ranks[positive].sum() - count_positive * (count_positive + 1) / 2.0)
    return u / (count_positive * count_negative)


def stratified_permutation_p(
    scores: Sequence[float],
    labels: Sequence[bool],
    class_ids: Sequence[int],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    classes = np.asarray(class_ids, dtype=np.int64)
    observed = auc_higher(values, positive)
    if observed is None:
        raise ValueError("permutation test requires both labels")
    ranks = midranks(values)
    n_positive = int(positive.sum())
    n_negative = len(values) - n_positive
    groups = [np.flatnonzero(classes == value) for value in sorted(set(classes.tolist()))]
    positive_by_group = [int(positive[index].sum()) for index in groups]
    rng = np.random.default_rng(PERMUTATION_SEED)
    greater = 0
    chunk = 2000
    for start in range(0, PERMUTATIONS, chunk):
        size = min(chunk, PERMUTATIONS - start)
        selected_rank_sums = np.zeros(size, dtype=np.float64)
        for indices, count in zip(groups, positive_by_group):
            if count == 0:
                continue
            keys = rng.random((size, len(indices)))
            chosen = np.argpartition(keys, count - 1, axis=1)[:, :count]
            selected_rank_sums += ranks[indices][chosen].sum(axis=1)
        aucs = (
            selected_rank_sums - n_positive * (n_positive + 1) / 2.0
        ) / (n_positive * n_negative)
        greater += int(np.count_nonzero(aucs >= observed - 1e-15))
    return {
        "observed_auc": observed,
        "replicates": PERMUTATIONS,
        "seed": PERMUTATION_SEED,
        "greater_or_equal": greater,
        "plus_one_one_sided_p": (greater + 1) / (PERMUTATIONS + 1),
        "strata": {
            str(int(classes[index[0]])): {
                "rows": len(index),
                "positives": count,
            }
            for index, count in zip(groups, positive_by_group)
        },
    }


def stratified_bootstrap_auc(
    scores: Sequence[float],
    labels: Sequence[bool],
    class_ids: Sequence[int],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    classes = np.asarray(class_ids, dtype=np.int64)
    groups = [np.flatnonzero(classes == value) for value in sorted(set(classes.tolist()))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[np.ndarray] = []
    invalid = 0
    chunk = 1000
    comparison = (
        (values[:, None] > values[None, :]).astype(np.float64)
        + 0.5 * (values[:, None] == values[None, :]).astype(np.float64)
    )
    for start in range(0, BOOTSTRAPS, chunk):
        size = min(chunk, BOOTSTRAPS - start)
        counts = np.zeros((size, len(values)), dtype=np.int16)
        for indices in groups:
            draws = rng.integers(0, len(indices), size=(size, len(indices)))
            for row in range(size):
                counts[row, indices] = np.bincount(draws[row], minlength=len(indices))
        positive_counts = counts * positive[None, :]
        negative_counts = counts * (~positive)[None, :]
        n_positive = positive_counts.sum(axis=1, dtype=np.int64)
        n_negative = negative_counts.sum(axis=1, dtype=np.int64)
        valid = (n_positive > 0) & (n_negative > 0)
        invalid += int((~valid).sum())
        if np.any(valid):
            numerator = np.einsum(
                "bi,ij,bj->b",
                positive_counts[valid].astype(np.float64),
                comparison,
                negative_counts[valid].astype(np.float64),
                optimize=True,
            )
            estimates.append(numerator / (n_positive[valid] * n_negative[valid]))
    valid_values = np.concatenate(estimates) if estimates else np.asarray([], dtype=np.float64)
    if not len(valid_values):
        raise RuntimeError("every bootstrap replicate lacked a binary endpoint")
    return {
        "replicates": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "valid_replicates": int(len(valid_values)),
        "invalid_no_binary_endpoint": invalid,
        "q025": float(np.quantile(valid_values, 0.025)),
        "median": float(np.quantile(valid_values, 0.5)),
        "q975": float(np.quantile(valid_values, 0.975)),
    }


def validate_inputs(config: Path, probes: Path) -> tuple[dict[str, Any], list[Path]]:
    if sha256_file(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen discovery configuration changed")
    specification = load_json(config)
    receipt_path = probes / "receipt.json"
    receipt = load_json(receipt_path)
    validate_self_hash(receipt, "identity_sha256", context=str(receipt_path))
    if (
        receipt.get("identity_sha256") != EXPECTED_RECEIPT_IDENTITY
        or receipt.get("status") != "complete"
        or receipt.get("seeds") != list(SEEDS)
        or receipt.get("method", {}).get("checkpoints") != list(CHECKPOINTS)
        or receipt.get("method", {}).get("horizons") != list(HORIZONS)
        or receipt.get("method", {}).get("quality_direction_selected") is not False
        or receipt.get("firewall", {}).get("labels_reviews_pngs_decoded_images_opened") is not False
    ):
        raise RuntimeError("probe receipt identity or firewall changed")
    seed_dirs = [probes / f"seed{seed:02d}" for seed in SEEDS]
    if any(path.is_symlink() or not path.is_dir() for path in seed_dirs):
        raise RuntimeError("one or more seed products are absent")
    return specification, seed_dirs


def materialize_label_free_scores(seed_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows: list[dict[str, Any]] = []
    temporal_by_key: dict[tuple[int, int, int], float] = {}
    for seed_dir in seed_dirs:
        record_path = seed_dir / "record.json"
        record = load_json(record_path)
        validate_self_hash(record, "identity_sha256", context=str(record_path))
        for name, expected in record.get("files", {}).items():
            path = seed_dir / name
            if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
                raise RuntimeError(f"seed product changed: {path}")
        score_rows = load_csv(seed_dir / "sample_scores.csv")
        if len(score_rows) != len(CLASSES) * len(CHECKPOINTS):
            raise RuntimeError(f"sample score axis changed: {seed_dir}")
        checkpoint_rows.extend(
            {
                "global_seed": int(row["global_seed"]),
                "class_slot": int(row["class_slot"]),
                "class_id": int(row["class_id"]),
                "checkpoint": int(row["checkpoint"]),
                "consistency_coherence": float(row["dyadic_coherence"]),
                "probe_update_energy": float(row["dyadic_update_energy_sum"]),
            }
            for row in score_rows
        )
        source_root = Path(record["source_trace"]["root"])
        trace_path = source_root / "trace.npz"
        if sha256_file(trace_path) != record["source_trace"]["trace_file_sha256"]:
            raise RuntimeError(f"source trace changed after probing: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as archive:
            prediction = np.ascontiguousarray(archive["pred_xstart"], dtype=np.float64)
        if prediction.shape != (8, 250, 4, 32, 32) or not np.isfinite(prediction).all():
            raise RuntimeError(f"source prediction array changed: {trace_path}")
        for checkpoint in CHECKPOINTS:
            current = prediction[:, checkpoint]
            following = prediction[:, checkpoint + 1]
            numerator = np.sqrt(np.mean((following - current) ** 2, axis=(1, 2, 3)))
            denominator = np.sqrt(np.mean(current**2, axis=(1, 2, 3))) + 1e-12
            for slot, class_id in enumerate(CLASSES):
                temporal_by_key[(int(record["global_seed"]), class_id, checkpoint)] = float(
                    numerator[slot] / denominator[slot]
                )

    expected_axis = {
        (seed, class_id, checkpoint)
        for seed in SEEDS
        for class_id in CLASSES
        for checkpoint in CHECKPOINTS
    }
    observed_axis = {
        (row["global_seed"], row["class_id"], row["checkpoint"])
        for row in checkpoint_rows
    }
    if observed_axis != expected_axis or set(temporal_by_key) != expected_axis:
        raise RuntimeError("label-free score axis is incomplete or duplicated")
    for row in checkpoint_rows:
        row["single_path_temporal_change"] = temporal_by_key[
            (row["global_seed"], row["class_id"], row["checkpoint"])
        ]

    fields = (
        "consistency_coherence",
        "probe_update_energy",
        "single_path_temporal_change",
    )
    reference_rows: list[dict[str, Any]] = []
    references: dict[tuple[int, int, str], tuple[float, float]] = {}
    for class_id in CLASSES:
        for checkpoint in CHECKPOINTS:
            block = [
                row
                for row in checkpoint_rows
                if row["class_id"] == class_id and row["checkpoint"] == checkpoint
            ]
            if len(block) != len(SEEDS):
                raise RuntimeError("reference block does not contain all 20 label-free paths")
            for field in fields:
                values = np.asarray([float(row[field]) for row in block], dtype=np.float64)
                mean = float(values.mean())
                standard_deviation = float(values.std(ddof=1))
                if standard_deviation <= 0 or not math.isfinite(standard_deviation):
                    raise RuntimeError(f"degenerate reference scale: {class_id}/{checkpoint}/{field}")
                references[(class_id, checkpoint, field)] = (mean, standard_deviation)
                reference_rows.append(
                    {
                        "class_id": class_id,
                        "checkpoint": checkpoint,
                        "field": field,
                        "count": len(values),
                        "mean": mean,
                        "sample_standard_deviation": standard_deviation,
                        "median": float(np.median(values)),
                    }
                )
    for row in checkpoint_rows:
        for field in fields:
            mean, standard_deviation = references[(row["class_id"], row["checkpoint"], field)]
            row[f"{field}_z"] = (float(row[field]) - mean) / (standard_deviation + 1e-12)

    path_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for class_id in CLASSES:
            block = sorted(
                (
                    row
                    for row in checkpoint_rows
                    if row["global_seed"] == seed and row["class_id"] == class_id
                ),
                key=lambda row: row["checkpoint"],
            )
            if [row["checkpoint"] for row in block] != list(CHECKPOINTS):
                raise RuntimeError("path score lacks the three fixed checkpoints")
            path_rows.append(
                {
                    "global_seed": seed,
                    "class_id": class_id,
                    "consistency_score": float(
                        np.mean([row["consistency_coherence_z"] for row in block])
                    ),
                    "temporal_change_control": float(
                        np.mean([row["single_path_temporal_change_z"] for row in block])
                    ),
                    "probe_energy_control": float(
                        np.mean([row["probe_update_energy_z"] for row in block])
                    ),
                    **{
                        f"consistency_checkpoint_{checkpoint}": next(
                            row["consistency_coherence_z"]
                            for row in block
                            if row["checkpoint"] == checkpoint
                        )
                        for checkpoint in CHECKPOINTS
                    },
                }
            )
    return path_rows, reference_rows


def load_labels(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    validate_self_hash(payload, "identity_sha256", context=str(path))
    if (
        payload.get("identity_sha256") != EXPECTED_LABEL_IDENTITY
        or payload.get("status") != "LOCKED_COMPLETE_BEFORE_ANY_TARGETED100_TRAJECTORY_METRIC_JOIN"
        or payload.get("counts")
        != {"clean_good": 69, "clear_bad": 5, "mild_or_disputed": 26}
        or len(payload.get("rows", [])) != 100
        or payload.get("rule", {}).get("metric_trajectory_or_signal_used") is not False
    ):
        raise RuntimeError("locked visual consensus identity or scope changed")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in payload["rows"]:
        key = (int(row["seed"]), int(row["class_id"]))
        if key in rows:
            raise RuntimeError(f"duplicate visual label key: {key}")
        rows[key] = row
    return rows, payload


def score_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    scores = [float(row[field]) for row in rows]
    labels = [row["primary_label"] == "clear_bad" for row in rows]
    classes = [int(row["class_id"]) for row in rows]
    return {
        "field": field,
        "direction": "higher_is_worse",
        "auc": auc_higher(scores, labels),
        "mean_clear_bad": float(np.mean([score for score, label in zip(scores, labels) if label])),
        "mean_clean_good": float(np.mean([score for score, label in zip(scores, labels) if not label])),
        "permutation": stratified_permutation_p(scores, labels, classes),
        "bootstrap": stratified_bootstrap_auc(scores, labels, classes),
    }


def run(args: argparse.Namespace) -> None:
    config = args.config.expanduser().resolve()
    probes = args.probes.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    specification, seed_dirs = validate_inputs(config, probes)
    path_rows, reference_rows = materialize_label_free_scores(seed_dirs)
    if len(path_rows) != 160 or len(reference_rows) != 8 * 3 * 3:
        raise RuntimeError("label-free feature product has the wrong axis")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        label_free_path = staging / "label_free_path_scores.csv"
        reference_path = staging / "label_free_reference_stats.csv"
        write_csv(label_free_path, path_rows)
        write_csv(reference_path, reference_rows)
        label_free_seal = {
            "path_scores_sha256": sha256_file(label_free_path),
            "reference_stats_sha256": sha256_file(reference_path),
            "path_count": len(path_rows),
            "labels_opened_before_these_files_were_written_and_hashed": False,
        }
        label_free_seal["identity_sha256"] = canonical_sha256(label_free_seal)
        write_json(staging / "label_free_seal.json", label_free_seal)

        # External labels are first opened below this line.
        label_rows, label_payload = load_labels(labels_path)
        joined: list[dict[str, Any]] = []
        for score in path_rows:
            label = label_rows.get((score["global_seed"], score["class_id"]))
            if label is None:
                continue
            joined.append(
                {
                    **score,
                    "primary_label": label["primary_label"],
                    "sample_key": label["sample_key"],
                    "majority_flags": "|".join(label.get("majority_flags", [])),
                }
            )
        if Counter(row["primary_label"] for row in joined) != Counter(
            {"clean_good": 69, "clear_bad": 5, "mild_or_disputed": 26}
        ):
            raise RuntimeError("score-label join does not reproduce locked label counts")
        primary_rows = [
            row for row in joined if row["primary_label"] in {"clear_bad", "clean_good"}
        ]
        if len(primary_rows) != 74:
            raise RuntimeError("primary binary comparison must contain exactly 74 rows")

        primary = score_summary(primary_rows, "consistency_score")
        temporal = score_summary(primary_rows, "temporal_change_control")
        energy = score_summary(primary_rows, "probe_energy_control")
        checkpoint_results = {
            str(checkpoint): score_summary(
                primary_rows, f"consistency_checkpoint_{checkpoint}"
            )
            for checkpoint in CHECKPOINTS
        }

        reference_by_class = defaultdict(list)
        for row in path_rows:
            reference_by_class[int(row["class_id"])].append(float(row["consistency_score"]))
        bad_details = []
        for row in primary_rows:
            if row["primary_label"] != "clear_bad":
                continue
            reference = np.asarray(reference_by_class[int(row["class_id"])], dtype=np.float64)
            score = float(row["consistency_score"])
            bad_details.append(
                {
                    "sample_key": row["sample_key"],
                    "class_id": int(row["class_id"]),
                    "score": score,
                    "within_class_reference_percentile_leq": float(np.mean(reference <= score)),
                    "above_class_reference_median": bool(score > float(np.median(reference))),
                    "majority_flags": row["majority_flags"],
                }
            )
        per_positive_class = {}
        for class_id in sorted({int(row["class_id"]) for row in bad_details}):
            block = [row for row in primary_rows if int(row["class_id"]) == class_id]
            per_positive_class[str(class_id)] = {
                "clear_bad": sum(row["primary_label"] == "clear_bad" for row in block),
                "clean_good": sum(row["primary_label"] == "clean_good" for row in block),
                "auc": auc_higher(
                    [float(row["consistency_score"]) for row in block],
                    [row["primary_label"] == "clear_bad" for row in block],
                ),
            }

        bad_above = sum(row["above_class_reference_median"] for row in bad_details)
        checkpoint_auc_values = [
            float(checkpoint_results[str(checkpoint)]["auc"]) for checkpoint in CHECKPOINTS
        ]
        positive_class_aucs = [
            float(value["auc"])
            for value in per_positive_class.values()
            if value["auc"] is not None
        ]
        gates = {
            "primary_auc_at_least_0p65": float(primary["auc"]) >= 0.65,
            "at_least_3_of_5_bad_above_class_median": bad_above >= 3,
            "permutation_p_at_most_0p10": float(
                primary["permutation"]["plus_one_one_sided_p"]
            )
            <= 0.10,
            "not_below_temporal_change_control": float(primary["auc"])
            >= float(temporal["auc"]),
            "not_below_probe_energy_control": float(primary["auc"])
            >= float(energy["auc"]),
            "at_least_two_checkpoints_noninverted": sum(
                value >= 0.5 for value in checkpoint_auc_values
            )
            >= 2,
            "at_least_two_positive_classes_noninverted": sum(
                value >= 0.5 for value in positive_class_aucs
            )
            >= 2,
        }
        advance = all(gates.values())
        strong = (
            float(primary["auc"]) >= 0.75
            and bad_above >= 4
            and float(primary["auc"]) >= max(float(temporal["auc"]), float(energy["auc"])) + 0.03
        )
        decision = (
            "ADVANCE_TO_NEW_POOL_STRONG_DISCOVERY"
            if advance and strong
            else "ADVANCE_TO_NEW_POOL_WEAK_DISCOVERY"
            if advance
            else "STOP_AS_BAD_CASE_DETECTOR_NO_SIGN_OR_AXIS_RESCUE"
        )
        results: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_DOOB_CONSISTENCY_DISCOVERY_ANALYSIS_V1",
            "status": "complete",
            "decision": decision,
            "gates": gates,
            "strong_discovery": strong,
            "primary": primary,
            "controls": {
                "single_path_temporal_change": temporal,
                "probe_total_update_energy": energy,
            },
            "descriptive_fixed_checkpoints": checkpoint_results,
            "bad_details": bad_details,
            "bad_above_class_reference_median": bad_above,
            "per_positive_class": per_positive_class,
            "counts": dict(Counter(row["primary_label"] for row in joined)),
            "integrity": {
                "config_sha256": sha256_file(config),
                "probe_receipt_identity_sha256": EXPECTED_RECEIPT_IDENTITY,
                "label_free_seal_identity_sha256": label_free_seal["identity_sha256"],
                "locked_consensus_identity_sha256": label_payload["identity_sha256"],
                "label_free_scores_materialized_and_hashed_before_labels_opened": True,
                "labels_or_external_metrics_used_to_construct_score": False,
                "direction_flipped_after_join": False,
                "single_checkpoint_or_horizon_selected_after_join": False,
                "same_pool_result_is_discovery_not_confirmation": True,
            },
            "interpretation": {
                "mechanics": "The score estimates a real frozen denoiser-kernel consistency defect.",
                "quality_boundary": "Only the preregistered higher-score association can support bad-case semantics.",
                "prior_art": "The consistency identity and cross-product estimator originate in Consistent Diffusion Models; only the frozen-model inference diagnostic use is tested here."
            },
        }
        write_csv(staging / "joined_discovery_rows.csv", joined)
        results["joined_discovery_rows_sha256"] = sha256_file(
            staging / "joined_discovery_rows.csv"
        )
        results["identity_sha256"] = canonical_sha256(results)
        write_json(staging / "results.json", results)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "output": str(output), "decision": decision, "identity_sha256": results["identity_sha256"]}, indent=2))


def self_test() -> None:
    assert math.isclose(auc_higher([0, 1, 2, 3], [False, True, False, True]), 0.75)
    assert math.isclose(auc_higher([0, 0, 1, 1], [False, True, False, True]), 0.5)
    test = stratified_permutation_p(
        [0.0, 1.0, 2.0, 3.0], [False, True, False, True], [0, 0, 1, 1]
    )
    assert 0 < test["plus_one_one_sided_p"] <= 1
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    run(parse_args())

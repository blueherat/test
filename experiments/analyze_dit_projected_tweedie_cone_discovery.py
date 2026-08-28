#!/usr/bin/env python3
"""Evaluate the frozen PTCV discovery score after label-free numerical QA.

The script validates and independently recomputes all projected matrices and
scores, applies every numerical gate, then materializes and hashes the full
160-path score product.  Only after that seal exists does it open the prior
locked visual consensus.  This pool is discovery-only, never confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .dit_projected_tweedie_cone import cone_metrics, finite_difference_stability
except ImportError:  # pragma: no cover - direct CLI invocation.
    from dit_projected_tweedie_cone import cone_metrics, finite_difference_stability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/dit_projected_tweedie_cone_discovery_v1.json"
DEFAULT_PROBES = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_projected_tweedie_cone_probe_v1"
)
DEFAULT_LABELS = (
    ROOT
    / "experiments/annotations/dit_targeted100_adjudicated_consensus_lock_v2/consensus_locked.json"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_projected_tweedie_cone_discovery_analysis_v1"
)
EXPECTED_CONFIG_SHA256 = "60b359111b734bcbed12b9f9f2e6e1ef748b2328c1a9a150c3b0c41f3c8c7394"
EXPECTED_CORE_SHA256 = "986f0fc8bbf22b84731ffb9b8b73bc9d73db263ae7f32d05e4ec812acf6900fe"
EXPECTED_RUNNER_SHA256 = "25a4c07e779fc5117225b2b0787a093ac6ddb2b81377a87fabe6459d28f27997"
EXPECTED_LABEL_IDENTITY = "693b8f7ce291ed6006c1161baf75c2769d035633479fd4903d1e7f67ece32bbe"
EXPECTED_RECEIPTS = {
    0: "8ce958fe145be8e540ad46497dd280b49b80dcff76dd115b42140bf8f6a7323b",
    1: "c46c4e404c311dc93b05d82a2ea24e36540fcf29c5c163dd15c346099205e3e4",
    2: "f961676f3b04d0b9e06df91f76b1b942e188df5dcb8146162af7d58eab4e46ce",
    3: "732a6a249646e59315914d7a758a46ee37dcebe828c9aa4700c48a8f1fd761bf",
}
CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
SEEDS = tuple(range(10, 30))
CHECKPOINTS = (99, 149, 199)
INTERNAL_TIMESTEPS = (150, 100, 50)
RELATIVE_RADII = (0.001953125, 0.00390625)
BASIS_SHA256 = "698fa3fcf6a67265ccdb618f3d1c6642affd03aa41dbcb5ffce8d6f36529d179"
BOOTSTRAPS = 100000
BOOTSTRAP_SEED = 2026082821


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


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
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
        raise ValueError("refusing to write an empty CSV")
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


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    first = midranks(np.asarray(values_a, dtype=np.float64))
    second = midranks(np.asarray(values_b, dtype=np.float64))
    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        raise RuntimeError("Spearman correlation is undefined for a constant axis")
    return float(np.corrcoef(first, second)[0, 1])


def auc_higher(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    count_positive = int(positive.sum())
    count_negative = len(positive) - count_positive
    if count_positive == 0 or count_negative == 0:
        return None
    ranks = midranks(values)
    rank_sum = float(ranks[positive].sum())
    u = rank_sum - count_positive * (count_positive + 1) / 2.0
    return u / (count_positive * count_negative)


def exact_stratified_randomization(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    ranks = midranks(values)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("randomization test needs both endpoints")
    observed_sum = float(ranks[positive].sum())
    observed_auc = (
        observed_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    blocks: list[np.ndarray] = []
    strata = {}
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id)
        count = int(positive[indices].sum())
        strata[str(class_id)] = {"rows": len(indices), "positives": count}
        if count:
            blocks.append(
                np.asarray(
                    [
                        float(ranks[list(choice)].sum())
                        for choice in itertools.combinations(indices.tolist(), count)
                    ],
                    dtype=np.float64,
                )
            )
    combined = blocks[0]
    for block in blocks[1:]:
        combined = (combined[:, None] + block[None, :]).reshape(-1)
    greater_equal = int(np.count_nonzero(combined >= observed_sum - 1e-12))
    return {
        "observed_auc": float(observed_auc),
        "assignments": int(len(combined)),
        "greater_or_equal": greater_equal,
        "exact_one_sided_p": float(greater_equal / len(combined)),
        "strata": strata,
    }


def stratified_bootstrap_auc(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    groups = [
        np.flatnonzero(class_ids == value) for value in sorted(set(class_ids.tolist()))
    ]
    comparison = (
        (values[:, None] > values[None, :]).astype(np.float64)
        + 0.5 * (values[:, None] == values[None, :]).astype(np.float64)
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[np.ndarray] = []
    invalid = 0
    chunk = 1000
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
    valid_values = np.concatenate(estimates)
    return {
        "replicates": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "valid_replicates": int(len(valid_values)),
        "invalid_no_binary_endpoint": invalid,
        "q025": float(np.quantile(valid_values, 0.025)),
        "median": float(np.quantile(valid_values, 0.5)),
        "q975": float(np.quantile(valid_values, 0.975)),
    }


def class_standardize(
    rows: Sequence[Mapping[str, Any]], raw_field: str, output_field: str
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        by_class[int(row["class_id"])].append(row)
    if set(by_class) != set(CLASSES) or any(len(block) != len(SEEDS) for block in by_class.values()):
        raise RuntimeError("class standardization does not have 20 paths per class")
    for class_id, block in by_class.items():
        values = np.asarray([float(row[raw_field]) for row in block], dtype=np.float64)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError(f"degenerate reference for {raw_field}/class {class_id}")
        for row in block:
            row[output_field] = (
                float(row[raw_field]) - mean
            ) / (standard_deviation + 1e-12)
    return copied


def load_probe_products(probes: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_by_seed: dict[int, dict[str, Any]] = {}
    receipt_identities = {}
    for shard_index, expected_identity in EXPECTED_RECEIPTS.items():
        shard = probes / f"shard_{shard_index:02d}_of_04"
        receipt_path = shard / "receipt.json"
        receipt = load_json(receipt_path)
        validate_self_hash(receipt, "identity_sha256", context=str(receipt_path))
        expected_seeds = [
            seed for index, seed in enumerate(SEEDS) if index % 4 == shard_index
        ]
        if (
            receipt.get("identity_sha256") != expected_identity
            or receipt.get("status") != "complete"
            or receipt.get("runner_source_sha256") != EXPECTED_RUNNER_SHA256
            or receipt.get("core_source_sha256") != EXPECTED_CORE_SHA256
            or receipt.get("seeds") != expected_seeds
            or receipt.get("method", {}).get("basis_raw_sha256") != BASIS_SHA256
            or receipt.get("method", {}).get("relative_radii") != list(RELATIVE_RADII)
            or receipt.get("method", {}).get("quality_direction_selected") is not False
            or receipt.get("firewall", {}).get(
                "labels_reviews_pngs_decoded_images_opened"
            )
            is not False
        ):
            raise RuntimeError(f"probe receipt changed: {receipt_path}")
        receipt_identities[str(shard_index)] = receipt["identity_sha256"]
        for item in receipt.get("records", []):
            seed = int(item["global_seed"])
            if seed in receipt_by_seed:
                raise RuntimeError("duplicate seed in probe receipts")
            receipt_by_seed[seed] = item
    if set(receipt_by_seed) != set(SEEDS):
        raise RuntimeError("probe receipts do not cover the frozen seed axis")

    checkpoint_products: list[dict[str, Any]] = []
    maximum_csv_error = 0.0
    maximum_path_csv_error = 0.0
    replay_count = 0
    for seed in SEEDS:
        shard_index = (seed - SEEDS[0]) % 4
        seed_dir = probes / f"shard_{shard_index:02d}_of_04" / f"seed{seed:02d}"
        record_path = seed_dir / "record.json"
        record = load_json(record_path)
        validate_self_hash(record, "identity_sha256", context=str(record_path))
        receipt_record = receipt_by_seed[seed]
        if (
            int(record.get("global_seed", -1)) != seed
            or record.get("identity_sha256") != receipt_record.get("identity_sha256")
            or Path(receipt_record.get("output", "")).resolve() != seed_dir
            or record.get("runner_source_sha256") != EXPECTED_RUNNER_SHA256
            or record.get("core_source_sha256") != EXPECTED_CORE_SHA256
            or record.get("class_ids") != list(CLASSES)
            or record.get("checkpoints") != list(CHECKPOINTS)
            or record.get("internal_timesteps") != list(INTERNAL_TIMESTEPS)
            or record.get("basis", {}).get("raw_sha256") != BASIS_SHA256
            or record.get("finite_difference", {}).get("relative_l2_radii")
            != list(RELATIVE_RADII)
            or record.get("firewall", {}).get("cfg_prediction_used_as_metric") is not False
        ):
            raise RuntimeError(f"receipt-to-record chain failed for seed {seed}")
        replay = record.get("raw_replay", [])
        if (
            len(replay) != len(CHECKPOINTS)
            or any(row.get("bitwise_exact") is not True for row in replay)
            or [int(row["checkpoint"]) for row in replay] != list(CHECKPOINTS)
        ):
            raise RuntimeError(f"raw conditional replay gate failed for seed {seed}")
        replay_count += len(replay)
        for name, expected in record.get("files", {}).items():
            path = seed_dir / name
            if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get(
                "sha256"
            ):
                raise RuntimeError(f"probe seed file changed: {path}")
        checkpoint_csv = load_csv(seed_dir / "checkpoint_scores.csv")
        path_csv = load_csv(seed_dir / "path_scores.csv")
        checkpoint_by_axis = {
            (int(row["class_id"]), int(row["checkpoint"])): row
            for row in checkpoint_csv
        }
        path_by_class = {int(row["class_id"]): row for row in path_csv}
        if len(checkpoint_by_axis) != len(CLASSES) * len(CHECKPOINTS) or len(path_by_class) != len(CLASSES):
            raise RuntimeError("probe CSV axis changed or duplicated")

        with np.load(seed_dir / "ptcv.npz", allow_pickle=False) as archive:
            expected_files = {
                "global_seed",
                "class_ids",
                "checkpoints",
                "internal_timesteps",
                "relative_radii",
                "basis",
                "raw_conditional_pred_xstart",
                "absolute_radii",
                "projected_matrices_by_radius",
                "richardson_projected_matrices",
            }
            if set(archive.files) != expected_files:
                raise RuntimeError("PTCV archive schema changed")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if (
            int(arrays["global_seed"]) != seed
            or not np.array_equal(arrays["class_ids"], np.asarray(CLASSES))
            or not np.array_equal(arrays["checkpoints"], np.asarray(CHECKPOINTS))
            or not np.array_equal(arrays["internal_timesteps"], np.asarray(INTERNAL_TIMESTEPS))
            or not np.array_equal(arrays["relative_radii"], np.asarray(RELATIVE_RADII))
            or raw_sha256(arrays["basis"]) != BASIS_SHA256
            or arrays["basis"].shape != (16, 4, 32, 32)
            or arrays["raw_conditional_pred_xstart"].shape != (3, 8, 4, 32, 32)
            or arrays["absolute_radii"].shape != (3, 8, 2)
            or arrays["projected_matrices_by_radius"].shape != (3, 8, 2, 16, 16)
            or arrays["richardson_projected_matrices"].shape != (3, 8, 16, 16)
            or any(not np.isfinite(array).all() for array in arrays.values())
        ):
            raise RuntimeError(f"PTCV archive tensor contract changed for seed {seed}")

        source_trace = Path(record["source_trace"]["root"]) / "trace.npz"
        if sha256_file(source_trace) != record["source_trace"]["trace_file_sha256"]:
            raise RuntimeError(f"source trace changed after PTCV probing: {source_trace}")
        with np.load(source_trace, allow_pickle=False) as archive:
            cfg_prediction = np.ascontiguousarray(archive["pred_xstart"], dtype=np.float64)
        if cfg_prediction.shape != (8, 250, 4, 32, 32) or not np.isfinite(cfg_prediction).all():
            raise RuntimeError("source CFG pred_xstart contract changed")

        for class_slot, class_id in enumerate(CLASSES):
            class_checkpoint_rows = []
            for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                small = arrays["projected_matrices_by_radius"][
                    checkpoint_index, class_slot, 0
                ]
                large = arrays["projected_matrices_by_radius"][
                    checkpoint_index, class_slot, 1
                ]
                richardson = arrays["richardson_projected_matrices"][
                    checkpoint_index, class_slot
                ]
                if not np.allclose(
                    richardson, (4.0 * small - large) / 3.0, rtol=0.0, atol=1e-14
                ):
                    raise RuntimeError("saved Richardson matrix identity failed")
                small_metrics = cone_metrics(small)
                large_metrics = cone_metrics(large)
                final_metrics = cone_metrics(richardson)
                stability = finite_difference_stability(small, large, richardson)
                expected = checkpoint_by_axis[(class_id, checkpoint)]
                for computed, field in (
                    (final_metrics["cone_distance_squared"], "cone_distance_squared"),
                    (final_metrics["matrix_energy"], "matrix_energy"),
                    (final_metrics["normalized_cone_violation"], "normalized_cone_violation"),
                    (final_metrics["skew_energy"], "skew_energy"),
                    (final_metrics["negative_eigen_energy"], "negative_eigen_energy"),
                    (stability["difference_over_richardson_norm"], "difference_over_richardson_norm"),
                ):
                    maximum_csv_error = max(
                        maximum_csv_error, abs(float(computed) - float(expected[field]))
                    )
                current_cfg = cfg_prediction[class_slot, checkpoint]
                following_cfg = cfg_prediction[class_slot, checkpoint + 1]
                temporal_change = float(
                    np.sqrt(np.mean((following_cfg - current_cfg) ** 2))
                    / (np.sqrt(np.mean(current_cfg**2)) + 1e-12)
                )
                raw_conditional = arrays["raw_conditional_pred_xstart"][
                    checkpoint_index, class_slot
                ].astype(np.float64)
                cfg_gap = float(
                    np.sqrt(np.mean((raw_conditional - current_cfg) ** 2))
                    / (np.sqrt(np.mean(raw_conditional**2)) + 1e-12)
                )
                row = {
                    "global_seed": seed,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "small_distance": small_metrics["cone_distance_squared"],
                    "small_energy": small_metrics["matrix_energy"],
                    "large_distance": large_metrics["cone_distance_squared"],
                    "large_energy": large_metrics["matrix_energy"],
                    "richardson_distance": final_metrics["cone_distance_squared"],
                    "richardson_energy": final_metrics["matrix_energy"],
                    "richardson_skew_energy": final_metrics["skew_energy"],
                    "richardson_negative_energy": final_metrics[
                        "negative_eigen_energy"
                    ],
                    "richardson_cone_violation": final_metrics[
                        "normalized_cone_violation"
                    ],
                    "minimum_symmetric_eigenvalue": final_metrics[
                        "minimum_symmetric_eigenvalue"
                    ],
                    "negative_eigenvalue_count": final_metrics[
                        "negative_eigenvalue_count"
                    ],
                    "finite_difference_gap": stability[
                        "difference_over_richardson_norm"
                    ],
                    "temporal_change": temporal_change,
                    "raw_conditional_cfg_gap": cfg_gap,
                }
                class_checkpoint_rows.append(row)
                checkpoint_products.append(row)

            rich_distance = sum(row["richardson_distance"] for row in class_checkpoint_rows)
            rich_energy = sum(row["richardson_energy"] for row in class_checkpoint_rows)
            raw_path = rich_distance / max(rich_energy, 1e-30)
            maximum_path_csv_error = max(
                maximum_path_csv_error,
                abs(raw_path - float(path_by_class[class_id]["path_cone_violation"])),
            )
    if maximum_csv_error > 1e-12 or maximum_path_csv_error > 1e-12:
        raise RuntimeError("independent PTCV recomputation did not match runner CSV")
    return checkpoint_products, {
        "receipt_identities": receipt_identities,
        "raw_replay_count": replay_count,
        "maximum_checkpoint_csv_error": maximum_csv_error,
        "maximum_path_csv_error": maximum_path_csv_error,
    }


def build_label_free_paths(
    checkpoint_products: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_rows = []
    for seed in SEEDS:
        for class_id in CLASSES:
            block = sorted(
                (
                    row
                    for row in checkpoint_products
                    if int(row["global_seed"]) == seed and int(row["class_id"]) == class_id
                ),
                key=lambda row: int(row["checkpoint"]),
            )
            if [int(row["checkpoint"]) for row in block] != list(CHECKPOINTS):
                raise RuntimeError("checkpoint product lacks one frozen path axis")
            def ratio(prefix: str, selected: Sequence[Mapping[str, Any]] = block) -> float:
                return sum(float(row[f"{prefix}_distance"]) for row in selected) / max(
                    sum(float(row[f"{prefix}_energy"]) for row in selected), 1e-30
                )

            rich_distance = sum(float(row["richardson_distance"]) for row in block)
            rich_energy = sum(float(row["richardson_energy"]) for row in block)
            raw_rows.append(
                {
                    "global_seed": seed,
                    "class_id": class_id,
                    "raw_path_cone_violation": rich_distance / max(rich_energy, 1e-30),
                    "small_path_cone_violation": ratio("small"),
                    "large_path_cone_violation": ratio("large"),
                    "path_matrix_energy": rich_energy,
                    "finite_difference_gap": max(
                        float(row["finite_difference_gap"]) for row in block
                    ),
                    "single_path_temporal_change": float(
                        np.mean([float(row["temporal_change"]) for row in block])
                    ),
                    "raw_conditional_cfg_gap": float(
                        np.mean([float(row["raw_conditional_cfg_gap"]) for row in block])
                    ),
                    "path_skew_fraction": sum(
                        float(row["richardson_skew_energy"]) for row in block
                    )
                    / max(rich_energy, 1e-30),
                    "path_negative_eigen_fraction": sum(
                        float(row["richardson_negative_energy"]) for row in block
                    )
                    / max(rich_energy, 1e-30),
                    "minimum_symmetric_eigenvalue": min(
                        float(row["minimum_symmetric_eigenvalue"]) for row in block
                    ),
                    "maximum_checkpoint_cone_violation": max(
                        float(row["richardson_cone_violation"]) for row in block
                    ),
                    **{
                        f"energy_share_checkpoint_{int(row['checkpoint'])}": float(
                            row["richardson_energy"]
                        )
                        / max(rich_energy, 1e-30)
                        for row in block
                    },
                    **{
                        f"raw_without_checkpoint_{omitted}": ratio(
                            "richardson",
                            [row for row in block if int(row["checkpoint"]) != omitted],
                        )
                        for omitted in CHECKPOINTS
                    },
                }
            )
    if len(raw_rows) != len(SEEDS) * len(CLASSES):
        raise RuntimeError("label-free path axis is incomplete")

    standardized = raw_rows
    fields = (
        ("raw_path_cone_violation", "consistency_score"),
        ("path_matrix_energy", "matrix_energy_control"),
        ("finite_difference_gap", "finite_difference_gap_control"),
        ("single_path_temporal_change", "temporal_change_control"),
        ("raw_conditional_cfg_gap", "raw_conditional_cfg_gap_control"),
        ("raw_without_checkpoint_99", "score_without_checkpoint_99"),
        ("raw_without_checkpoint_149", "score_without_checkpoint_149"),
        ("raw_without_checkpoint_199", "score_without_checkpoint_199"),
    )
    for raw_field, output_field in fields:
        standardized = class_standardize(standardized, raw_field, output_field)

    reference_rows = []
    for class_id in CLASSES:
        block = [row for row in standardized if int(row["class_id"]) == class_id]
        for raw_field, output_field in fields:
            values = np.asarray([float(row[raw_field]) for row in block], dtype=np.float64)
            reference_rows.append(
                {
                    "class_id": class_id,
                    "raw_field": raw_field,
                    "standardized_field": output_field,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "sample_standard_deviation": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                }
            )

    small = [float(row["small_path_cone_violation"]) for row in standardized]
    large = [float(row["large_path_cone_violation"]) for row in standardized]
    rich = [float(row["raw_path_cone_violation"]) for row in standardized]
    gaps = np.asarray(
        [float(row["finite_difference_gap"]) for row in standardized], dtype=np.float64
    )
    numerical = {
        "spearman_small_large": spearman(small, large),
        "spearman_small_richardson": spearman(small, rich),
        "spearman_large_richardson": spearman(large, rich),
        "median_maximum_matrix_gap": float(np.median(gaps)),
        "q95_maximum_matrix_gap": float(np.quantile(gaps, 0.95)),
        "maximum_matrix_gap": float(gaps.max()),
        "negative_eigen_checkpoint_rows": sum(
            float(row["richardson_negative_energy"]) > 0.0
            for row in checkpoint_products
        ),
        "negative_eigen_paths": sum(
            float(row["path_negative_eigen_fraction"]) > 0.0 for row in standardized
        ),
        "energy_share_checkpoint_summary": {
            str(checkpoint): {
                "median": float(
                    np.median(
                        [
                            float(row[f"energy_share_checkpoint_{checkpoint}"])
                            for row in standardized
                        ]
                    )
                ),
                "q05": float(
                    np.quantile(
                        [
                            float(row[f"energy_share_checkpoint_{checkpoint}"])
                            for row in standardized
                        ],
                        0.05,
                    )
                ),
                "q95": float(
                    np.quantile(
                        [
                            float(row[f"energy_share_checkpoint_{checkpoint}"])
                            for row in standardized
                        ],
                        0.95,
                    )
                ),
            }
            for checkpoint in CHECKPOINTS
        },
    }
    numerical["gates"] = {
        "spearman_small_large_at_least_0p8": numerical["spearman_small_large"] >= 0.8,
        "spearman_small_richardson_at_least_0p8": numerical[
            "spearman_small_richardson"
        ]
        >= 0.8,
        "spearman_large_richardson_at_least_0p8": numerical[
            "spearman_large_richardson"
        ]
        >= 0.8,
        "median_gap_at_most_0p05": numerical["median_maximum_matrix_gap"] <= 0.05,
        "q95_gap_at_most_0p10": numerical["q95_maximum_matrix_gap"] <= 0.10,
        "all_path_scores_in_unit_interval": all(0.0 <= value <= 1.0 for value in rich),
    }
    numerical["all_gates_pass"] = all(numerical["gates"].values())
    return standardized, reference_rows, numerical


def load_labels(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    validate_self_hash(payload, "identity_sha256", context=str(path))
    if (
        payload.get("identity_sha256") != EXPECTED_LABEL_IDENTITY
        or payload.get("status")
        != "LOCKED_COMPLETE_BEFORE_ANY_TARGETED100_TRAJECTORY_METRIC_JOIN"
        or payload.get("counts")
        != {"clean_good": 69, "clear_bad": 5, "mild_or_disputed": 26}
        or len(payload.get("rows", [])) != 100
        or payload.get("rule", {}).get("metric_trajectory_or_signal_used") is not False
    ):
        raise RuntimeError("locked visual consensus identity or scope changed")
    rows = {}
    for row in payload["rows"]:
        key = (int(row["seed"]), int(row["class_id"]))
        if key in rows:
            raise RuntimeError("duplicate visual label key")
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
        "mean_clear_bad": float(
            np.mean([score for score, label in zip(scores, labels) if label])
        ),
        "mean_clean_good": float(
            np.mean([score for score, label in zip(scores, labels) if not label])
        ),
        "exact_randomization": exact_stratified_randomization(scores, labels, classes),
        "bootstrap": stratified_bootstrap_auc(scores, labels, classes),
    }


def run(args: argparse.Namespace) -> None:
    config = args.config.expanduser().resolve()
    probes = args.probes.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if sha256_file(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen PTCV discovery configuration changed")
    if sha256_file(ROOT / "experiments/dit_projected_tweedie_cone.py") != EXPECTED_CORE_SHA256:
        raise RuntimeError("frozen PTCV core changed")
    if sha256_file(ROOT / "experiments/run_dit_projected_tweedie_cone_probe.py") != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("frozen PTCV runner changed")
    specification = load_json(config)
    checkpoint_products, provenance = load_probe_products(probes)
    path_rows, reference_rows, numerical = build_label_free_paths(checkpoint_products)
    if len(path_rows) != 160 or len(reference_rows) != len(CLASSES) * 8:
        raise RuntimeError("label-free product has the wrong axis")
    if provenance["raw_replay_count"] != len(SEEDS) * len(CHECKPOINTS):
        raise RuntimeError("not all raw conditional replays were verified")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        path_file = staging / "label_free_path_scores.csv"
        reference_file = staging / "label_free_reference_stats.csv"
        checkpoint_file = staging / "label_free_checkpoint_products.csv"
        write_csv(path_file, path_rows)
        write_csv(reference_file, reference_rows)
        write_csv(checkpoint_file, checkpoint_products)
        label_free_seal = {
            "path_scores_sha256": sha256_file(path_file),
            "reference_stats_sha256": sha256_file(reference_file),
            "checkpoint_products_sha256": sha256_file(checkpoint_file),
            "path_count": len(path_rows),
            "checkpoint_product_count": len(checkpoint_products),
            "config_sha256": sha256_file(config),
            "labels_opened_before_these_files_were_written_and_hashed": False,
            "numerical_gates": numerical["gates"],
            "all_numerical_gates_pass": numerical["all_gates_pass"],
        }
        label_free_seal["identity_sha256"] = canonical_sha256(label_free_seal)
        write_json(staging / "label_free_seal.json", label_free_seal)
        write_json(staging / "label_free_numerical_audit.json", numerical)

        if not numerical["all_gates_pass"]:
            results: dict[str, Any] = {
                "schema_version": 1,
                "artifact_kind": "DIT_PROJECTED_TWEEDIE_CONE_DISCOVERY_ANALYSIS_V1",
                "status": "stopped_before_labels",
                "decision": "STOP_NUMERICALLY_UNRESOLVED_LABELS_NOT_OPENED",
                "numerical": numerical,
                "provenance": provenance,
                "integrity": {
                    "config_sha256": sha256_file(config),
                    "label_free_seal_identity_sha256": label_free_seal[
                        "identity_sha256"
                    ],
                    "labels_opened": False,
                },
            }
            results["identity_sha256"] = canonical_sha256(results)
            write_json(staging / "results.json", results)
            os.replace(staging, output)
            print(json.dumps({"status": "stopped_before_labels", "output": str(output)}, indent=2))
            return

        # The locked visual labels are first opened below this line.
        label_rows, label_payload = load_labels(labels_path)
        joined = []
        for score in path_rows:
            label = label_rows.get((int(score["global_seed"]), int(score["class_id"])))
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
            raise RuntimeError("PTCV score-label join changed the locked label counts")
        primary_rows = [
            row for row in joined if row["primary_label"] in {"clear_bad", "clean_good"}
        ]
        if len(primary_rows) != 74:
            raise RuntimeError("primary comparison must contain 74 rows")

        primary = score_summary(primary_rows, "consistency_score")
        control_fields = (
            "matrix_energy_control",
            "finite_difference_gap_control",
            "temporal_change_control",
            "raw_conditional_cfg_gap_control",
        )
        controls = {field: score_summary(primary_rows, field) for field in control_fields}
        checkpoint_leave_out = {
            str(checkpoint): score_summary(
                primary_rows, f"score_without_checkpoint_{checkpoint}"
            )
            for checkpoint in CHECKPOINTS
        }

        per_positive_class = {}
        positive_classes = sorted(
            {
                int(row["class_id"])
                for row in primary_rows
                if row["primary_label"] == "clear_bad"
            }
        )
        leave_one_positive_class_out = {}
        for class_id in positive_classes:
            block = [row for row in primary_rows if int(row["class_id"]) == class_id]
            per_positive_class[str(class_id)] = {
                "clear_bad": sum(row["primary_label"] == "clear_bad" for row in block),
                "clean_good": sum(row["primary_label"] == "clean_good" for row in block),
                "auc": auc_higher(
                    [float(row["consistency_score"]) for row in block],
                    [row["primary_label"] == "clear_bad" for row in block],
                ),
            }
            remaining = [row for row in primary_rows if int(row["class_id"]) != class_id]
            leave_one_positive_class_out[str(class_id)] = {
                "remaining_clear_bad": sum(
                    row["primary_label"] == "clear_bad" for row in remaining
                ),
                "auc": auc_higher(
                    [float(row["consistency_score"]) for row in remaining],
                    [row["primary_label"] == "clear_bad" for row in remaining],
                ),
            }

        reference_by_class = defaultdict(list)
        for row in path_rows:
            reference_by_class[int(row["class_id"])].append(
                float(row["raw_path_cone_violation"])
            )
        bad_details = []
        for row in primary_rows:
            if row["primary_label"] != "clear_bad":
                continue
            reference = np.asarray(
                reference_by_class[int(row["class_id"])], dtype=np.float64
            )
            raw_score = float(row["raw_path_cone_violation"])
            bad_details.append(
                {
                    "sample_key": row["sample_key"],
                    "class_id": int(row["class_id"]),
                    "majority_flags": row["majority_flags"],
                    "raw_path_cone_violation": raw_score,
                    "standardized_score": float(row["consistency_score"]),
                    "rank_leq_of_20": int(np.count_nonzero(reference <= raw_score)),
                    "within_class_percentile_leq": float(np.mean(reference <= raw_score)),
                    "above_class_median": bool(raw_score > float(np.median(reference))),
                    "path_skew_fraction": float(row["path_skew_fraction"]),
                    "path_negative_eigen_fraction": float(
                        row["path_negative_eigen_fraction"]
                    ),
                }
            )
        bad_above = sum(row["above_class_median"] for row in bad_details)

        blur_rows = [
            row
            for row in primary_rows
            if row["primary_label"] == "clean_good"
            or (
                row["primary_label"] == "clear_bad"
                and "global_blur" in row["majority_flags"].split("|")
            )
        ]
        structural_rows = [
            row
            for row in primary_rows
            if row["primary_label"] == "clean_good"
            or (
                row["primary_label"] == "clear_bad"
                and "global_blur" not in row["majority_flags"].split("|")
            )
        ]
        subtype = {
            "global_blur_clear_bad_vs_all_clean": score_summary(
                blur_rows, "consistency_score"
            ),
            "non_global_blur_structural_clear_bad_vs_all_clean": score_summary(
                structural_rows, "consistency_score"
            ),
        }

        minimum_control_auc = min(float(value["auc"]) for value in controls.values())
        maximum_control_auc = max(float(value["auc"]) for value in controls.values())
        class_loo_values = [
            float(value["auc"])
            for value in leave_one_positive_class_out.values()
            if int(value["remaining_clear_bad"]) >= 2 and value["auc"] is not None
        ]
        checkpoint_loo_values = [
            float(value["auc"]) for value in checkpoint_leave_out.values()
        ]
        class_auc_values = [
            float(value["auc"])
            for value in per_positive_class.values()
            if value["auc"] is not None
        ]
        gates = {
            "all_label_free_numerical_gates_pass": numerical["all_gates_pass"],
            "primary_auc_at_least_0p65": float(primary["auc"]) >= 0.65,
            "at_least_3_of_5_bad_above_class_median": bad_above >= 3,
            "exact_randomization_p_at_most_0p10": float(
                primary["exact_randomization"]["exact_one_sided_p"]
            )
            <= 0.10,
            "not_below_any_fixed_control": float(primary["auc"])
            >= maximum_control_auc,
            "every_positive_class_leave_out_auc_at_least_0p55": bool(class_loo_values)
            and all(value >= 0.55 for value in class_loo_values),
            "every_checkpoint_leave_out_auc_at_least_0p55": all(
                value >= 0.55 for value in checkpoint_loo_values
            ),
            "at_least_two_positive_classes_strictly_noninverted": sum(
                value > 0.5 for value in class_auc_values
            )
            >= 2,
        }
        advance = all(gates.values())
        strong = (
            advance
            and float(primary["auc"]) >= 0.75
            and bad_above >= 4
            and float(primary["exact_randomization"]["exact_one_sided_p"]) <= 0.05
            and all(value >= 0.60 for value in class_loo_values)
            and all(value >= 0.60 for value in checkpoint_loo_values)
            and float(primary["auc"]) >= maximum_control_auc + 0.03
        )
        decision = (
            "ADVANCE_TO_NEW_POOL_STRONG_DISCOVERY"
            if strong
            else "ADVANCE_TO_NEW_POOL_WEAK_DISCOVERY"
            if advance
            else "STOP_AS_ARTIFACT_DETECTOR_NO_SIGN_AXIS_COMPONENT_RESCUE"
        )

        joined_path = staging / "joined_discovery_rows.csv"
        write_csv(joined_path, joined)
        results: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_PROJECTED_TWEEDIE_CONE_DISCOVERY_ANALYSIS_V1",
            "status": "complete",
            "decision": decision,
            "strong_discovery": strong,
            "gates": gates,
            "primary": primary,
            "controls": controls,
            "leave_one_checkpoint_out": checkpoint_leave_out,
            "leave_one_positive_class_out": leave_one_positive_class_out,
            "per_positive_class": per_positive_class,
            "bad_details": bad_details,
            "bad_above_class_median": bad_above,
            "subtype_descriptive_only": subtype,
            "counts": dict(Counter(row["primary_label"] for row in joined)),
            "label_free_numerical": numerical,
            "provenance": provenance,
            "integrity": {
                "config_sha256": sha256_file(config),
                "label_free_seal_identity_sha256": label_free_seal["identity_sha256"],
                "locked_consensus_identity_sha256": label_payload["identity_sha256"],
                "label_free_scores_written_and_hashed_before_labels_opened": True,
                "labels_or_external_metrics_used_to_construct_score": False,
                "quality_direction_flipped_after_join": False,
                "checkpoint_basis_radius_or_component_selected_after_join": False,
                "same_pool_result_is_discovery_not_confirmation": True,
                "joined_rows_sha256": sha256_file(joined_path),
            },
            "interpretation": {
                "mechanics": (
                    "The score is a numerically resolved projected distance of the raw "
                    "conditional denoiser Jacobian from the symmetric PSD cone."
                ),
                "quality_boundary": (
                    "Only the frozen high-score association and full robustness gates may "
                    "support artifact semantics."
                ),
                "prior_art": (
                    "Tweedie Jacobian identities, symmetry/PSD violations, Jacobian OOD "
                    "features, and temporal artifact correction are prior art; only the "
                    "per-trajectory cone-distance detector is under test."
                ),
                "energy_weighting": (
                    "The path score is the block-diagonal cone ratio and therefore weights "
                    "checkpoints by projected Jacobian energy, not equally."
                ),
            },
            "unused_minimum_control_auc_for_audit": minimum_control_auc,
        }
        results["identity_sha256"] = canonical_sha256(results)
        write_json(staging / "results.json", results)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "decision": decision,
                "identity_sha256": results["identity_sha256"],
            },
            indent=2,
        )
    )


def self_test() -> None:
    assert math.isclose(auc_higher([0, 1, 2, 3], [False, True, False, True]), 0.75)
    assert math.isclose(auc_higher([0, 0, 1, 1], [False, True, False, True]), 0.5)
    assert math.isclose(spearman([1, 2, 3], [10, 20, 30]), 1.0)
    exact = exact_stratified_randomization(
        [0.0, 1.0, 2.0, 3.0], [False, True, False, True], [0, 0, 1, 1]
    )
    assert exact["assignments"] == 4
    rows = [
        {"class_id": class_id, "value": float(seed)}
        for class_id in CLASSES
        for seed in SEEDS
    ]
    scaled = class_standardize(rows, "value", "z")
    for class_id in CLASSES:
        values = [float(row["z"]) for row in scaled if int(row["class_id"]) == class_id]
        assert abs(float(np.mean(values))) < 1e-14
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

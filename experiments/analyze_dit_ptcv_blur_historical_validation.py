#!/usr/bin/env python3
"""Evaluate the frozen blur-specific PTCV hypothesis on expansion eval360.

The order of operations is part of the protocol.  The analyzer first validates
the frozen configuration, numerical sources, four extraction receipts, every
seed record, every emitted file, and every source trace.  It then independently
recomputes the 360 path scores and fixed controls from NPZ tensors, writes and
fsyncs the complete label-free product, and seals its hashes.  Only if every
predeclared numerical gate passes is the historical visual lock opened.

This is a disjoint-seed historical validation, not prospective confirmation.
Success can only authorize a newly sampled blur confirmation; it never
authorizes rejection, guidance, rollback, or selective resampling.
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
DEFAULT_CONFIG = ROOT / "experiments/configs/dit_ptcv_blur_historical_validation_v1.json"
DEFAULT_PROBES = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_projected_tweedie_cone_expansion_eval360_probe_v1"
)
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_confirmation_expansion_v1_custom_traces_cfg_locked"
)
DEFAULT_LABELS = (
    ROOT
    / "experiments/annotations/"
    "dit_expansion_eval360_adjudicated_consensus_lock_v1/consensus_locked.json"
)
DEFAULT_PRIOR_BLUR = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "bad_good_metric_confirmation_expansion_v1/"
    "predxstart_visual_label_free_v1/sample_features.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_ptcv_blur_historical_validation_v1"
)

EXPECTED_CONFIG_SHA256 = "f90be2d24adb234bd9dcbba09ef6b92ea568d085628a15fe8721d25db26eb274"
EXPECTED_CORE_SHA256 = "986f0fc8bbf22b84731ffb9b8b73bc9d73db263ae7f32d05e4ec812acf6900fe"
EXPECTED_EXPANSION_RUNNER_SHA256 = (
    "585d9738067761f7d4a57cf2d019efecc1b28f6939025aba855aa8598a7e67b9"
)
EXPECTED_FROZEN_RUNNER_SHA256 = (
    "25a4c07e779fc5117225b2b0787a093ac6ddb2b81377a87fabe6459d28f27997"
)
EXPECTED_STRICT_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
EXPECTED_TRACE_RUNNER_SHA256 = (
    "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"
)
EXPECTED_MODELS_SHA256 = "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca"
EXPECTED_LABEL_FILE_SHA256 = (
    "6aa24778d62780ec0eb373bda45fabee5a504a3de4fd01fa86d56bfa93e30347"
)
EXPECTED_LABEL_IDENTITY = "fc478b7ae04b67869d0dfca3b63f169a5266bacc1c8701e1ae20368516e793fd"
EXPECTED_SOURCE_POOL_CONTRACT = (
    "16f1a89ab5c432d163a74dbbcaa52f4f41447b5bfd63fcd85581f2be14170e25"
)
EXPECTED_BASIS_SHA256 = "698fa3fcf6a67265ccdb618f3d1c6642affd03aa41dbcb5ffce8d6f36529d179"

CLASSES = (207, 602, 795)
SEEDS = tuple(range(130, 250))
CHECKPOINTS = (99, 149, 199)
INTERNAL_TIMESTEPS = (150, 100, 50)
RELATIVE_RADII = (0.001953125, 0.00390625)
SHARD_COUNT = 4
PERMITTED_RUNNER_ARRAYS = (
    "state_before",
    "conditional_epsilon_raw",
    "internal_timestep",
    "alpha_bar",
)
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 2026082827
PRIOR_BLUR_FIELD = "decoded_local_blur_severity__mean"

STRICT_STANDARDIZED_FIELDS = (
    ("raw_path_cone_violation", "consistency_score"),
    ("path_matrix_energy", "matrix_energy_control"),
    ("finite_difference_gap", "finite_difference_gap_control"),
    ("single_path_temporal_change", "temporal_change_control"),
    ("raw_conditional_cfg_gap", "raw_conditional_cfg_gap_control"),
    ("raw_without_checkpoint_99", "score_without_checkpoint_99"),
    ("raw_without_checkpoint_149", "score_without_checkpoint_149"),
    ("raw_without_checkpoint_199", "score_without_checkpoint_199"),
)
DESCRIPTIVE_STANDARDIZED_FIELDS = (
    ("path_skew_fraction", "skew_component_score"),
    ("path_negative_eigen_fraction", "negative_eigen_component_score"),
    ("raw_checkpoint_99", "checkpoint_99_score"),
    ("raw_checkpoint_149", "checkpoint_149_score"),
    ("raw_checkpoint_199", "checkpoint_199_score"),
)


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
        raise RuntimeError(f"expected a real CSV file: {path}")
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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    ranks_a = midranks(np.asarray(first, dtype=np.float64))
    ranks_b = midranks(np.asarray(second, dtype=np.float64))
    if float(ranks_a.std()) == 0.0 or float(ranks_b.std()) == 0.0:
        raise RuntimeError("Spearman correlation is undefined for a constant axis")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def auc_higher(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None
    rank_sum = float(midranks(values)[positive].sum())
    u = rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(u / (n_positive * n_negative))


def _subset_sum_distribution(weights: Sequence[int], selected: int) -> dict[int, int]:
    """Count fixed-cardinality subset sums by dynamic programming."""

    if not 0 <= selected <= len(weights):
        raise ValueError("invalid selected cardinality")
    levels: list[dict[int, int]] = [defaultdict(int) for _ in range(selected + 1)]
    levels[0][0] = 1
    seen = 0
    for weight in weights:
        seen += 1
        for count in range(min(selected, seen), 0, -1):
            for prior_sum, ways in tuple(levels[count - 1].items()):
                levels[count][prior_sum + int(weight)] += ways
    result = dict(levels[selected])
    if sum(result.values()) != math.comb(len(weights), selected):
        raise RuntimeError("subset-sum DP count does not match the binomial coefficient")
    return result


def exact_stratified_rank_sum_dp(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> dict[str, Any]:
    """Exact one-sided stratified rank-sum test without assignment enumeration."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    if len(values) != len(positive) or len(values) != len(class_ids):
        raise ValueError("rank-sum axes differ")
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("rank-sum test needs both binary endpoints")

    ranks = midranks(values)
    scaled_ranks = np.rint(2.0 * ranks).astype(np.int64)
    if not np.array_equal(scaled_ranks.astype(np.float64), 2.0 * ranks):
        raise RuntimeError("midranks were not exactly representable on the half-rank lattice")
    observed = int(scaled_ranks[positive].sum())
    combined: dict[int, int] = {0: 1}
    strata: dict[str, Any] = {}
    assignment_count = 1
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id)
        selected = int(positive[indices].sum())
        distribution = _subset_sum_distribution(
            scaled_ranks[indices].tolist(), selected
        )
        assignment_count *= math.comb(len(indices), selected)
        next_combined: dict[int, int] = defaultdict(int)
        for first_sum, first_ways in combined.items():
            for second_sum, second_ways in distribution.items():
                next_combined[first_sum + second_sum] += first_ways * second_ways
        combined = dict(next_combined)
        strata[str(class_id)] = {
            "rows": int(len(indices)),
            "positives": selected,
            "assignments": math.comb(len(indices), selected),
            "subset_sum_support": len(distribution),
        }
    if sum(combined.values()) != assignment_count:
        raise RuntimeError("stratified DP assignment count failed")
    greater_equal = sum(ways for rank_sum, ways in combined.items() if rank_sum >= observed)
    observed_auc = auc_higher(values, positive)
    return {
        "algorithm": "exact_fixed_class_count_subset_sum_dynamic_programming",
        "rank_lattice_scale": 2,
        "direction": "higher_is_worse_one_sided",
        "observed_auc": observed_auc,
        "observed_scaled_rank_sum": observed,
        "assignments": assignment_count,
        "greater_or_equal": greater_equal,
        "exact_one_sided_p": float(greater_equal / assignment_count),
        "combined_sum_support": len(combined),
        "strata": strata,
        "assignment_vectors_enumerated": False,
    }


def _exact_stratified_enumeration_for_self_test(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> tuple[int, int]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    scaled = np.rint(2.0 * midranks(values)).astype(np.int64)
    observed = int(scaled[positive].sum())
    blocks: list[list[int]] = []
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id).tolist()
        selected = int(positive[indices].sum())
        blocks.append(
            [
                int(sum(scaled[index] for index in choice))
                for choice in itertools.combinations(indices, selected)
            ]
        )
    total = 0
    greater_equal = 0
    for assignment in itertools.product(*blocks):
        total += 1
        greater_equal += sum(assignment) >= observed
    return greater_equal, total


def stratified_bootstrap_auc(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> dict[str, Any]:
    """Resample whole primary rows within each class with a fixed RNG seed."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    positive_values = values[positive]
    negative_values = values[~positive]
    if len(positive_values) == 0 or len(negative_values) == 0:
        raise ValueError("bootstrap needs both endpoints")
    comparison = (
        (positive_values[:, None] > negative_values[None, :]).astype(np.float64)
        + 0.5
        * (positive_values[:, None] == negative_values[None, :]).astype(np.float64)
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    chunks: list[np.ndarray] = []
    invalid = 0
    chunk_size = 2000
    class_axis = sorted(set(class_ids.tolist()))
    class_groups = [np.flatnonzero(class_ids == class_id) for class_id in class_axis]
    positive_indices = np.flatnonzero(positive)
    negative_indices = np.flatnonzero(~positive)
    for start in range(0, BOOTSTRAPS, chunk_size):
        size = min(chunk_size, BOOTSTRAPS - start)
        counts = np.zeros((size, len(values)), dtype=np.int16)
        for indices in class_groups:
            draws = rng.integers(0, len(indices), size=(size, len(indices)))
            for row_index in range(size):
                counts[row_index, indices] = np.bincount(
                    draws[row_index], minlength=len(indices)
                )
        positive_counts = counts[:, positive_indices]
        negative_counts = counts[:, negative_indices]
        n_positive = positive_counts.sum(axis=1, dtype=np.int64)
        n_negative = negative_counts.sum(axis=1, dtype=np.int64)
        valid = (n_positive > 0) & (n_negative > 0)
        invalid += int((~valid).sum())
        if not np.any(valid):
            continue
        numerator = np.einsum(
            "bi,ij,bj->b",
            positive_counts[valid].astype(np.float64),
            comparison,
            negative_counts[valid].astype(np.float64),
            optimize=True,
        )
        chunks.append(numerator / (n_positive[valid] * n_negative[valid]))
    estimates = np.concatenate(chunks)
    if len(estimates) + invalid != BOOTSTRAPS or not np.isfinite(estimates).all():
        raise RuntimeError("bootstrap output contract failed")
    return {
        "design": "resample_primary_rows_with_replacement_within_each_class",
        "replicates": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "valid_replicates": int(len(estimates)),
        "invalid_no_binary_endpoint": invalid,
        "q025": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "q975": float(np.quantile(estimates, 0.975)),
    }


def class_standardize(
    rows: Sequence[Mapping[str, Any]],
    raw_field: str,
    output_field: str,
    *,
    allow_degenerate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied = [dict(row) for row in rows]
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        by_class[int(row["class_id"])].append(row)
    if set(by_class) != set(CLASSES) or any(
        len(block) != len(SEEDS) for block in by_class.values()
    ):
        raise RuntimeError("class standardization lacks exactly 120 paths per class")
    references = []
    for class_id in CLASSES:
        block = by_class[class_id]
        values = np.asarray([float(row[raw_field]) for row in block], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite class reference: {raw_field}/{class_id}")
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=1))
        degenerate = not math.isfinite(standard_deviation) or standard_deviation <= 0.0
        if degenerate and not allow_degenerate:
            raise RuntimeError(f"degenerate class reference: {raw_field}/{class_id}")
        for row in block:
            row[output_field] = (
                0.0
                if degenerate
                else (float(row[raw_field]) - mean) / (standard_deviation + 1e-12)
            )
        references.append(
            {
                "class_id": class_id,
                "raw_field": raw_field,
                "standardized_field": output_field,
                "count": len(values),
                "mean": mean,
                "sample_standard_deviation": standard_deviation,
                "median": float(np.median(values)),
                "degenerate_mapped_to_zero_descriptive_only": degenerate,
            }
        )
    return copied, references


def _trace_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    records = [
        row
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and row.get("relative_path") == "trace.npz"
    ]
    if len(records) != 1:
        raise RuntimeError("source manifest lacks exactly one trace.npz record")
    return records[0]


def validate_and_load_source_cfg(
    trace_root: Path, seed: int, record_source: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = trace_root / f"expansion_v1_seed{seed}"
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"source trace directory changed: {source}")
    manifest_path = source / "manifest.json"
    completion_path = source / "completion.json"
    trace_path = source / "trace.npz"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity", {})
    protocol = identity.get("protocol", {})
    trace_record = _trace_record(manifest)
    manifest_sha = sha256_file(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("identity_sha256") != canonical_sha256(identity)
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_sha256") != manifest_sha
        or protocol.get("global_torch_seed") != seed
        or tuple(protocol.get("class_ids_ordered", [])) != CLASSES
        or protocol.get("sampling_steps") != 250
        or protocol.get("cfg_scale") != 4.0
        or identity.get("runner_source", {}).get("sha256")
        != EXPECTED_TRACE_RUNNER_SHA256
        or identity.get("strict_reproduction_helper", {}).get("sha256")
        != EXPECTED_STRICT_SHA256
        or identity.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or identity.get("source", {}).get("pinned_source_sha256", {}).get("models.py")
        != EXPECTED_MODELS_SHA256
        or sha256_file(source / "runner_source.py") != EXPECTED_TRACE_RUNNER_SHA256
        or sha256_file(source / "strict_reproduction_helper.py") != EXPECTED_STRICT_SHA256
    ):
        raise RuntimeError(f"source trace manifest chain failed: {source}")
    if trace_path.is_symlink() or not trace_path.is_file():
        raise RuntimeError(f"source trace file changed: {trace_path}")
    actual_trace_sha = sha256_file(trace_path)
    if (
        trace_record.get("bytes") != trace_path.stat().st_size
        or trace_record.get("sha256") != actual_trace_sha
    ):
        raise RuntimeError(f"source trace file hash failed: {trace_path}")

    manifest_arrays = manifest.get("trace_array_records", {})
    loaded_hashes = record_source.get("loaded_array_raw_sha256", {})
    permitted_hashes = record_source.get("permitted_array_raw_sha256", {})
    summary_payload = {
        "global_seed": seed,
        "source_directory": str(source),
        "identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": manifest_sha,
        "trace_file_bytes": int(trace_record["bytes"]),
        "trace_file_sha256_recorded": trace_record["sha256"],
        "permitted_array_raw_sha256": {
            name: manifest_arrays.get(name, {}).get("raw_sha256")
            for name in PERMITTED_RUNNER_ARRAYS
        },
    }
    if (
        Path(str(record_source.get("source_directory", ""))).resolve() != source
        or int(record_source.get("global_seed", -1)) != seed
        or record_source.get("identity_sha256") != manifest["identity_sha256"]
        or record_source.get("manifest_file_sha256") != manifest_sha
        or record_source.get("trace_file_bytes") != trace_record["bytes"]
        or record_source.get("trace_file_sha256_recorded") != trace_record["sha256"]
        or record_source.get("contract_sha256") != canonical_sha256(summary_payload)
        or permitted_hashes != summary_payload["permitted_array_raw_sha256"]
        or loaded_hashes != permitted_hashes
        or record_source.get("arrays_loaded") != list(PERMITTED_RUNNER_ARRAYS)
        or record_source.get("whole_trace_file_sha256_recomputed") is not False
    ):
        raise RuntimeError(f"probe-to-source trace chain failed: seed {seed}")

    with np.load(trace_path, allow_pickle=False) as archive:
        required = ("pred_xstart", "internal_timestep", "alpha_bar")
        if any(name not in archive.files for name in required):
            raise RuntimeError(f"source trace lacks controls: {trace_path}")
        cfg_prediction = np.ascontiguousarray(archive["pred_xstart"])
        internal = np.ascontiguousarray(archive["internal_timestep"])
        alpha_bar = np.ascontiguousarray(archive["alpha_bar"])
    if (
        cfg_prediction.shape != (3, 250, 4, 32, 32)
        or cfg_prediction.dtype != np.float32
        or internal.shape != (250,)
        or alpha_bar.shape != (250,)
        or not np.isfinite(cfg_prediction).all()
        or not np.array_equal(
            internal[np.asarray(CHECKPOINTS)], np.asarray(INTERNAL_TIMESTEPS)
        )
        or raw_sha256(cfg_prediction) != manifest_arrays.get("pred_xstart", {}).get("raw_sha256")
        or raw_sha256(internal) != manifest_arrays.get("internal_timestep", {}).get("raw_sha256")
        or raw_sha256(alpha_bar) != manifest_arrays.get("alpha_bar", {}).get("raw_sha256")
    ):
        raise RuntimeError(f"source control tensor contract failed: {trace_path}")
    return (
        cfg_prediction.astype(np.float64),
        alpha_bar.astype(np.float64),
        {
            "source_identity_sha256": manifest["identity_sha256"],
            "source_manifest_file_sha256": manifest_sha,
            "source_trace_file_sha256": actual_trace_sha,
            "source_pred_xstart_raw_sha256": raw_sha256(cfg_prediction),
        },
    )


def load_probe_products(
    probes: Path, trace_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if probes.is_symlink() or not probes.is_dir():
        raise RuntimeError(f"expected a real probe root: {probes}")
    if trace_root.is_symlink() or not trace_root.is_dir():
        raise RuntimeError(f"expected a real trace root: {trace_root}")
    receipt_by_seed: dict[int, Mapping[str, Any]] = {}
    receipt_identities: dict[str, str] = {}
    for shard_index in range(SHARD_COUNT):
        shard = probes / f"shard_{shard_index:02d}_of_{SHARD_COUNT:02d}"
        receipt_path = shard / "receipt.json"
        receipt = load_json(receipt_path)
        validate_self_hash(receipt, "identity_sha256", context=str(receipt_path))
        expected_seeds = [
            seed for index, seed in enumerate(SEEDS) if index % SHARD_COUNT == shard_index
        ]
        source_contract = receipt.get("source_pool_contract", {})
        if isinstance(source_contract, dict):
            validate_self_hash(
                source_contract,
                "identity_sha256",
                context=f"{receipt_path}:source_pool_contract",
            )
        if (
            receipt.get("status") != "complete"
            or receipt.get("runner_source_sha256") != EXPECTED_EXPANSION_RUNNER_SHA256
            or receipt.get("shard_index") != shard_index
            or receipt.get("shard_count") != SHARD_COUNT
            or receipt.get("seeds") != expected_seeds
            or receipt.get("path_count") != len(expected_seeds) * len(CLASSES)
            or receipt.get("dependency_sha256")
            != {
                "strict": EXPECTED_STRICT_SHA256,
                "frozen_runner": EXPECTED_FROZEN_RUNNER_SHA256,
                "core": EXPECTED_CORE_SHA256,
                "checkpoint": EXPECTED_CHECKPOINT_SHA256,
            }
            or source_contract.get("identity_sha256") != EXPECTED_SOURCE_POOL_CONTRACT
            or source_contract.get("seeds") != list(SEEDS)
            or source_contract.get("classes") != list(CLASSES)
            or receipt.get("method", {}).get("basis_raw_sha256") != EXPECTED_BASIS_SHA256
            or receipt.get("method", {}).get("checkpoints") != list(CHECKPOINTS)
            or receipt.get("method", {}).get("relative_radii") != list(RELATIVE_RADII)
            or receipt.get("method", {}).get("quality_direction_selected") is not False
            or receipt.get("firewall", {}).get("quality_labels_or_reviews_opened") is not False
            or receipt.get("firewall", {}).get("png_files_opened") is not False
            or receipt.get("firewall", {}).get("decoded_image_array_loaded") is not False
            or receipt.get("firewall", {}).get("endpoint_array_loaded") is not False
            or receipt.get("firewall", {}).get("external_metric_or_embedding_opened") is not False
        ):
            raise RuntimeError(f"probe receipt contract failed: {receipt_path}")
        records = receipt.get("records", [])
        if len(records) != len(expected_seeds):
            raise RuntimeError(f"receipt record count changed: {receipt_path}")
        receipt_identities[str(shard_index)] = receipt["identity_sha256"]
        for item in records:
            seed = int(item.get("global_seed", -1))
            if seed in receipt_by_seed or seed not in expected_seeds:
                raise RuntimeError("receipt seed coverage is duplicated or mis-sharded")
            receipt_by_seed[seed] = item
    if set(receipt_by_seed) != set(SEEDS):
        raise RuntimeError("receipts do not cover exactly seeds 130..249")

    checkpoint_products: list[dict[str, Any]] = []
    source_hash_rows = []
    record_chain_rows = []
    maximum_checkpoint_csv_error = 0.0
    maximum_path_csv_error = 0.0
    replay_batches = 0
    for seed in SEEDS:
        shard_index = (seed - SEEDS[0]) % SHARD_COUNT
        seed_dir = probes / f"shard_{shard_index:02d}_of_{SHARD_COUNT:02d}" / f"seed{seed:03d}"
        record_path = seed_dir / "record.json"
        record = load_json(record_path)
        validate_self_hash(record, "identity_sha256", context=str(record_path))
        receipt_record = receipt_by_seed[seed]
        firewall = record.get("firewall", {})
        if (
            record.get("status") != "complete"
            or int(record.get("global_seed", -1)) != seed
            or record.get("identity_sha256") != receipt_record.get("identity_sha256")
            or Path(str(receipt_record.get("output", ""))).resolve() != seed_dir
            or record.get("runner_source_sha256") != EXPECTED_EXPANSION_RUNNER_SHA256
            or record.get("frozen_runner_source_sha256") != EXPECTED_FROZEN_RUNNER_SHA256
            or record.get("core_source_sha256") != EXPECTED_CORE_SHA256
            or record.get("class_ids") != list(CLASSES)
            or record.get("checkpoints") != list(CHECKPOINTS)
            or record.get("internal_timesteps") != list(INTERNAL_TIMESTEPS)
            or record.get("basis", {}).get("raw_sha256") != EXPECTED_BASIS_SHA256
            or record.get("finite_difference", {}).get("relative_l2_radii")
            != list(RELATIVE_RADII)
            or record.get("path_score", {}).get("uses_all_projected_matrix_components")
            is not True
            or firewall.get("quality_labels_or_reviews_opened") is not False
            or firewall.get("png_files_opened") is not False
            or firewall.get("decoded_image_array_loaded") is not False
            or firewall.get("endpoint_array_loaded") is not False
            or firewall.get("external_metric_or_embedding_opened") is not False
            or firewall.get("cfg_prediction_used_as_metric") is not False
        ):
            raise RuntimeError(f"receipt-to-record contract failed: seed {seed}")
        replay = record.get("raw_replay", [])
        if (
            len(replay) != len(CHECKPOINTS)
            or [int(row.get("checkpoint", -1)) for row in replay] != list(CHECKPOINTS)
            or any(row.get("bitwise_exact") is not True for row in replay)
        ):
            raise RuntimeError(f"raw conditional replay failed: seed {seed}")
        replay_batches += len(replay)

        file_records = record.get("files", {})
        if set(file_records) != {"ptcv.npz", "checkpoint_scores.csv", "path_scores.csv"}:
            raise RuntimeError(f"seed file schema changed: seed {seed}")
        for name, expected in file_records.items():
            path = seed_dir / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != expected.get("bytes")
                or sha256_file(path) != expected.get("sha256")
            ):
                raise RuntimeError(f"seed file integrity failed: {path}")
        record_chain_rows.append(
            {
                "global_seed": seed,
                "record_identity_sha256": record["identity_sha256"],
                "files": file_records,
            }
        )

        cfg_prediction, source_alpha_bar, source_hashes = validate_and_load_source_cfg(
            trace_root, seed, record.get("source_trace", {})
        )
        source_hash_rows.append({"global_seed": seed, **source_hashes})
        checkpoint_csv = load_csv(seed_dir / "checkpoint_scores.csv")
        path_csv = load_csv(seed_dir / "path_scores.csv")
        checkpoint_by_axis = {
            (int(row["class_id"]), int(row["checkpoint"])): row for row in checkpoint_csv
        }
        path_by_class = {int(row["class_id"]): row for row in path_csv}
        if (
            len(checkpoint_by_axis) != len(CLASSES) * len(CHECKPOINTS)
            or len(path_by_class) != len(CLASSES)
        ):
            raise RuntimeError(f"runner CSV axis changed: seed {seed}")

        with np.load(seed_dir / "ptcv.npz", allow_pickle=False) as archive:
            expected_members = {
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
            if set(archive.files) != expected_members:
                raise RuntimeError(f"PTCV archive schema changed: seed {seed}")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if (
            int(arrays["global_seed"]) != seed
            or not np.array_equal(arrays["class_ids"], np.asarray(CLASSES))
            or not np.array_equal(arrays["checkpoints"], np.asarray(CHECKPOINTS))
            or not np.array_equal(arrays["internal_timesteps"], np.asarray(INTERNAL_TIMESTEPS))
            or not np.array_equal(arrays["relative_radii"], np.asarray(RELATIVE_RADII))
            or raw_sha256(arrays["basis"]) != EXPECTED_BASIS_SHA256
            or arrays["basis"].shape != (16, 4, 32, 32)
            or arrays["raw_conditional_pred_xstart"].shape != (3, 3, 4, 32, 32)
            or arrays["absolute_radii"].shape != (3, 3, 2)
            or arrays["projected_matrices_by_radius"].shape != (3, 3, 2, 16, 16)
            or arrays["richardson_projected_matrices"].shape != (3, 3, 16, 16)
            or any(not np.isfinite(array).all() for array in arrays.values())
        ):
            raise RuntimeError(f"PTCV tensor contract changed: seed {seed}")

        recomputed_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            for class_slot, class_id in enumerate(CLASSES):
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
                    raise RuntimeError(f"Richardson identity failed: seed {seed}")
                small_metrics = cone_metrics(small)
                large_metrics = cone_metrics(large)
                final_metrics = cone_metrics(richardson)
                stability = finite_difference_stability(small, large, richardson)
                current_cfg = cfg_prediction[class_slot, checkpoint]
                next_cfg = cfg_prediction[class_slot, checkpoint + 1]
                temporal_change = float(
                    np.sqrt(np.mean((next_cfg - current_cfg) ** 2))
                    / (np.sqrt(np.mean(current_cfg**2)) + 1e-12)
                )
                raw_conditional = arrays["raw_conditional_pred_xstart"][
                    checkpoint_index, class_slot
                ].astype(np.float64)
                conditional_cfg_gap = float(
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
                    "richardson_cone_violation": final_metrics[
                        "normalized_cone_violation"
                    ],
                    "richardson_skew_energy": final_metrics["skew_energy"],
                    "richardson_negative_energy": final_metrics[
                        "negative_eigen_energy"
                    ],
                    "minimum_symmetric_eigenvalue": final_metrics[
                        "minimum_symmetric_eigenvalue"
                    ],
                    "maximum_symmetric_eigenvalue": final_metrics[
                        "maximum_symmetric_eigenvalue"
                    ],
                    "negative_eigenvalue_count": final_metrics[
                        "negative_eigenvalue_count"
                    ],
                    "finite_difference_gap": stability[
                        "difference_over_richardson_norm"
                    ],
                    "small_large_frobenius_difference": stability[
                        "small_large_frobenius_difference"
                    ],
                    "difference_over_mean_input_norm": stability[
                        "difference_over_mean_input_norm"
                    ],
                    "small_radius_minimum_secant": float(np.min(np.diag(small))),
                    "large_radius_minimum_secant": float(np.min(np.diag(large))),
                    "small_radius_negative_secant_count": int(
                        np.count_nonzero(np.diag(small) < 0.0)
                    ),
                    "large_radius_negative_secant_count": int(
                        np.count_nonzero(np.diag(large) < 0.0)
                    ),
                    "temporal_change": temporal_change,
                    "raw_conditional_cfg_gap": conditional_cfg_gap,
                }
                expected = checkpoint_by_axis[(class_id, checkpoint)]
                exact_integer_fields = {
                    "global_seed": seed,
                    "class_slot": class_slot,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "internal_timestep": INTERNAL_TIMESTEPS[checkpoint_index],
                    "negative_eigenvalue_count": final_metrics[
                        "negative_eigenvalue_count"
                    ],
                    "small_radius_negative_secant_count": row[
                        "small_radius_negative_secant_count"
                    ],
                    "large_radius_negative_secant_count": row[
                        "large_radius_negative_secant_count"
                    ],
                }
                if any(int(expected[field]) != value for field, value in exact_integer_fields.items()):
                    raise RuntimeError(f"checkpoint CSV integer mismatch: seed {seed}")
                float_fields = {
                    "alpha_bar": float(source_alpha_bar[checkpoint]),
                    "small_absolute_radius": float(
                        arrays["absolute_radii"][checkpoint_index, class_slot, 0]
                    ),
                    "large_absolute_radius": float(
                        arrays["absolute_radii"][checkpoint_index, class_slot, 1]
                    ),
                    "cone_distance_squared": final_metrics["cone_distance_squared"],
                    "matrix_energy": final_metrics["matrix_energy"],
                    "normalized_cone_violation": final_metrics[
                        "normalized_cone_violation"
                    ],
                    "skew_energy": final_metrics["skew_energy"],
                    "negative_eigen_energy": final_metrics[
                        "negative_eigen_energy"
                    ],
                    "skew_fraction": final_metrics["skew_fraction"],
                    "negative_eigen_fraction": final_metrics[
                        "negative_eigen_fraction"
                    ],
                    "minimum_symmetric_eigenvalue": final_metrics[
                        "minimum_symmetric_eigenvalue"
                    ],
                    "maximum_symmetric_eigenvalue": final_metrics[
                        "maximum_symmetric_eigenvalue"
                    ],
                    "small_radius_cone_violation": small_metrics[
                        "normalized_cone_violation"
                    ],
                    "large_radius_cone_violation": large_metrics[
                        "normalized_cone_violation"
                    ],
                    "small_radius_minimum_eigenvalue": small_metrics[
                        "minimum_symmetric_eigenvalue"
                    ],
                    "large_radius_minimum_eigenvalue": large_metrics[
                        "minimum_symmetric_eigenvalue"
                    ],
                    "small_radius_minimum_secant": row[
                        "small_radius_minimum_secant"
                    ],
                    "large_radius_minimum_secant": row[
                        "large_radius_minimum_secant"
                    ],
                    "small_large_frobenius_difference": stability[
                        "small_large_frobenius_difference"
                    ],
                    "difference_over_richardson_norm": stability[
                        "difference_over_richardson_norm"
                    ],
                    "difference_over_mean_input_norm": stability[
                        "difference_over_mean_input_norm"
                    ],
                }
                for field, computed in float_fields.items():
                    maximum_checkpoint_csv_error = max(
                        maximum_checkpoint_csv_error,
                        abs(float(expected[field]) - float(computed)),
                    )
                recomputed_by_class[class_id].append(row)
                checkpoint_products.append(row)

        for class_id in CLASSES:
            block = sorted(recomputed_by_class[class_id], key=lambda row: row["checkpoint"])
            distance = sum(float(row["richardson_distance"]) for row in block)
            energy = sum(float(row["richardson_energy"]) for row in block)
            skew = sum(float(row["richardson_skew_energy"]) for row in block)
            negative = sum(float(row["richardson_negative_energy"]) for row in block)
            computed_path = {
                "path_cone_violation": distance / max(energy, 1e-30),
                "path_skew_fraction": skew / max(energy, 1e-30),
                "path_negative_eigen_fraction": negative / max(energy, 1e-30),
                "path_matrix_energy": energy,
                "maximum_checkpoint_cone_violation": max(
                    float(row["richardson_cone_violation"]) for row in block
                ),
                "maximum_small_large_relative_gap": max(
                    float(row["finite_difference_gap"]) for row in block
                ),
                "minimum_finite_secant": min(
                    min(
                        float(row["small_radius_minimum_secant"]),
                        float(row["large_radius_minimum_secant"]),
                    )
                    for row in block
                ),
                **{
                    f"checkpoint_{int(row['checkpoint'])}_cone_violation": float(
                        row["richardson_cone_violation"]
                    )
                    for row in block
                },
            }
            expected = path_by_class[class_id]
            if int(expected["global_seed"]) != seed or int(expected["class_id"]) != class_id:
                raise RuntimeError(f"path CSV axis mismatch: seed {seed}")
            for field, computed in computed_path.items():
                maximum_path_csv_error = max(
                    maximum_path_csv_error,
                    abs(float(expected[field]) - float(computed)),
                )
    if maximum_checkpoint_csv_error > 1e-12 or maximum_path_csv_error > 1e-12:
        raise RuntimeError("independent NPZ recomputation differs from runner CSV")
    return checkpoint_products, {
        "receipt_identities": receipt_identities,
        "receipt_count": SHARD_COUNT,
        "seed_record_count": len(SEEDS),
        "source_file_hash_count": len(source_hash_rows),
        "source_hash_rows": source_hash_rows,
        "source_hash_rows_identity_sha256": canonical_sha256(source_hash_rows),
        "record_chain_rows": record_chain_rows,
        "record_chain_rows_identity_sha256": canonical_sha256(record_chain_rows),
        "raw_replay_checkpoint_batches": replay_batches,
        "raw_replay_path_checkpoints": replay_batches * len(CLASSES),
        "maximum_checkpoint_csv_error": maximum_checkpoint_csv_error,
        "maximum_path_csv_error": maximum_path_csv_error,
    }


def build_label_free_paths(
    checkpoint_products: Sequence[Mapping[str, Any]], specification: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_axis = {
        (int(row["global_seed"]), int(row["class_id"]), int(row["checkpoint"])): row
        for row in checkpoint_products
    }
    if len(by_axis) != len(SEEDS) * len(CLASSES) * len(CHECKPOINTS):
        raise RuntimeError("checkpoint product axis is incomplete or duplicated")
    raw_rows = []
    for seed in SEEDS:
        for class_id in CLASSES:
            block = [by_axis[(seed, class_id, checkpoint)] for checkpoint in CHECKPOINTS]

            def ratio(prefix: str, selected: Sequence[Mapping[str, Any]] = block) -> float:
                return sum(float(row[f"{prefix}_distance"]) for row in selected) / max(
                    sum(float(row[f"{prefix}_energy"]) for row in selected), 1e-30
                )

            rich_energy = sum(float(row["richardson_energy"]) for row in block)
            raw_rows.append(
                {
                    "global_seed": seed,
                    "class_id": class_id,
                    "raw_path_cone_violation": ratio("richardson"),
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
                        f"raw_checkpoint_{checkpoint}": float(
                            by_axis[(seed, class_id, checkpoint)][
                                "richardson_cone_violation"
                            ]
                        )
                        for checkpoint in CHECKPOINTS
                    },
                    **{
                        f"energy_share_checkpoint_{checkpoint}": float(
                            by_axis[(seed, class_id, checkpoint)]["richardson_energy"]
                        )
                        / max(rich_energy, 1e-30)
                        for checkpoint in CHECKPOINTS
                    },
                    **{
                        f"raw_without_checkpoint_{omitted}": ratio(
                            "richardson",
                            [
                                row
                                for row in block
                                if int(row["checkpoint"]) != omitted
                            ],
                        )
                        for omitted in CHECKPOINTS
                    },
                }
            )
    if len(raw_rows) != 360:
        raise RuntimeError("label-free path axis is not 360")

    standardized = raw_rows
    reference_rows: list[dict[str, Any]] = []
    for raw_field, output_field in STRICT_STANDARDIZED_FIELDS:
        standardized, references = class_standardize(
            standardized, raw_field, output_field, allow_degenerate=False
        )
        reference_rows.extend(references)
    for raw_field, output_field in DESCRIPTIVE_STANDARDIZED_FIELDS:
        standardized, references = class_standardize(
            standardized, raw_field, output_field, allow_degenerate=True
        )
        reference_rows.extend(references)

    small = [float(row["small_path_cone_violation"]) for row in standardized]
    large = [float(row["large_path_cone_violation"]) for row in standardized]
    rich = [float(row["raw_path_cone_violation"]) for row in standardized]
    gaps = np.asarray(
        [float(row["finite_difference_gap"]) for row in standardized], dtype=np.float64
    )
    thresholds = specification["label_free_numerical_gates_before_any_join"]
    all_checkpoint_scores = []
    for row in checkpoint_products:
        all_checkpoint_scores.extend(
            [
                float(row["richardson_cone_violation"]),
                float(row["small_distance"]) / max(float(row["small_energy"]), 1e-30),
                float(row["large_distance"]) / max(float(row["large_energy"]), 1e-30),
            ]
        )
    all_path_scores = rich + small + large
    numerical: dict[str, Any] = {
        "spearman_small_large": spearman(small, large),
        "spearman_small_richardson": spearman(small, rich),
        "spearman_large_richardson": spearman(large, rich),
        "median_per_path_maximum_matrix_gap": float(np.median(gaps)),
        "q95_per_path_maximum_matrix_gap": float(np.quantile(gaps, 0.95)),
        "maximum_per_path_matrix_gap": float(gaps.max()),
        "path_count": len(standardized),
        "checkpoint_product_count": len(checkpoint_products),
        "all_checkpoint_scores_in_unit_interval": all(
            0.0 <= value <= 1.0 for value in all_checkpoint_scores
        ),
        "all_path_scores_in_unit_interval": all(
            0.0 <= value <= 1.0 for value in all_path_scores
        ),
        "negative_eigen_checkpoint_rows": sum(
            float(row["richardson_negative_energy"]) > 0.0
            for row in checkpoint_products
        ),
        "negative_eigen_paths": sum(
            float(row["path_negative_eigen_fraction"]) > 0.0
            for row in standardized
        ),
        "energy_share_checkpoint_summary": {
            str(checkpoint): {
                "q05": float(
                    np.quantile(
                        [
                            float(row[f"energy_share_checkpoint_{checkpoint}"])
                            for row in standardized
                        ],
                        0.05,
                    )
                ),
                "median": float(
                    np.median(
                        [
                            float(row[f"energy_share_checkpoint_{checkpoint}"])
                            for row in standardized
                        ]
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
        "all_360_raw_conditional_checkpoint_replays_bitwise_exact": True,
        "all_matrices_and_scores_finite": all(
            math.isfinite(float(value))
            for row in standardized
            for value in row.values()
            if isinstance(value, (int, float, np.integer, np.floating))
        ),
        "all_cone_scores_in_closed_unit_interval": numerical[
            "all_checkpoint_scores_in_unit_interval"
        ]
        and numerical["all_path_scores_in_unit_interval"],
        "spearman_small_vs_large_path_score_at_least": numerical[
            "spearman_small_large"
        ]
        >= float(thresholds["spearman_small_vs_large_path_score_at_least"]),
        "spearman_small_vs_richardson_path_score_at_least": numerical[
            "spearman_small_richardson"
        ]
        >= float(thresholds["spearman_small_vs_richardson_path_score_at_least"]),
        "spearman_large_vs_richardson_path_score_at_least": numerical[
            "spearman_large_richardson"
        ]
        >= float(thresholds["spearman_large_vs_richardson_path_score_at_least"]),
        "median_per_path_maximum_small_large_relative_matrix_gap_at_most": numerical[
            "median_per_path_maximum_matrix_gap"
        ]
        <= float(
            thresholds[
                "median_per_path_maximum_small_large_relative_matrix_gap_at_most"
            ]
        ),
        "q95_per_path_maximum_small_large_relative_matrix_gap_at_most": numerical[
            "q95_per_path_maximum_matrix_gap"
        ]
        <= float(
            thresholds[
                "q95_per_path_maximum_small_large_relative_matrix_gap_at_most"
            ]
        ),
    }
    numerical["all_gates_pass"] = all(numerical["gates"].values())
    return standardized, reference_rows, numerical


def load_labels_after_seal(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_LABEL_FILE_SHA256:
        raise RuntimeError("locked expansion visual file changed")
    payload = load_json(path)
    validate_self_hash(payload, "identity_sha256", context=str(path))
    expected_counts = {"clear_bad": 9, "clean_good": 304, "mild_or_disputed": 47}
    blinding = payload.get("blinding_audit", {})
    if (
        payload.get("identity_sha256") != EXPECTED_LABEL_IDENTITY
        or payload.get("status")
        != "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN"
        or payload.get("cohort") != "expansion_seed130_249"
        or payload.get("counts") != expected_counts
        or payload.get("retained_clear_bad_count") != 9
        or len(payload.get("rows", [])) != 360
        or blinding.get("labels_locked_before_score_join") is not True
        or blinding.get("candidate_scores_visible_to_reviewers") is not False
        or blinding.get("metric_values_visible_to_reviewers") is not False
        or blinding.get("trajectories_visible_to_reviewers") is not False
    ):
        raise RuntimeError("locked expansion visual identity or blinding scope changed")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in payload["rows"]:
        seed = int(row.get("global_seed", row.get("seed", -1)))
        class_id = int(row.get("class_id", -1))
        key = (seed, class_id)
        label = row.get("primary_label")
        flags = row.get("majority_flags", [])
        if (
            key in rows
            or seed not in SEEDS
            or class_id not in CLASSES
            or label not in expected_counts
            or not isinstance(flags, list)
            or row.get("binary_primary_included") is not (
                label in {"clear_bad", "clean_good"}
            )
        ):
            raise RuntimeError("locked visual row contract changed")
        if label == "clear_bad" and "global_blur" not in flags:
            raise RuntimeError("a retained clear-bad lacks the frozen global_blur phenotype")
        rows[key] = row
    expected_axis = {(seed, class_id) for seed in SEEDS for class_id in CLASSES}
    if set(rows) != expected_axis or Counter(
        row["primary_label"] for row in rows.values()
    ) != Counter(expected_counts):
        raise RuntimeError("locked visual row axis or counts changed")
    return rows, payload


def load_prior_blur_metric(path: Path) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    rows = load_csv(path)
    forbidden = {"label", "raw_consensus_label", "primary_label"}
    if forbidden & set(rows[0]):
        raise RuntimeError("prior blur product unexpectedly contains visual labels")
    values: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["global_seed"]), int(row["class_id"]))
        value = float(row[PRIOR_BLUR_FIELD])
        if key in values or key[0] not in SEEDS or key[1] not in CLASSES or not math.isfinite(value):
            raise RuntimeError("prior blur product axis or value changed")
        values[key] = value
    expected_axis = {(seed, class_id) for seed in SEEDS for class_id in CLASSES}
    if set(values) != expected_axis:
        raise RuntimeError("prior blur product does not cover expansion eval360 exactly")
    standardized: dict[tuple[int, int], float] = {}
    reference = {}
    for class_id in CLASSES:
        keys = [(seed, class_id) for seed in SEEDS]
        current = np.asarray([values[key] for key in keys], dtype=np.float64)
        mean = float(current.mean())
        sd = float(current.std(ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            raise RuntimeError("prior blur metric has a degenerate class reference")
        for key, value in zip(keys, current):
            standardized[key] = float((value - mean) / (sd + 1e-12))
        reference[str(class_id)] = {"count": len(keys), "mean": mean, "sample_sd": sd}
    return standardized, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "field": PRIOR_BLUR_FIELD,
        "direction": "higher_is_more_blurry",
        "class_reference": reference,
    }


def descriptive_score_summary(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    scores = [float(row[field]) for row in rows]
    labels = [row["primary_label"] == "clear_bad" for row in rows]
    bad = [score for score, label in zip(scores, labels) if label]
    good = [score for score, label in zip(scores, labels) if not label]
    return {
        "field": field,
        "direction": "higher_is_worse",
        "auc": auc_higher(scores, labels),
        "mean_clear_bad": float(np.mean(bad)),
        "mean_clean_good": float(np.mean(good)),
        "median_clear_bad": float(np.median(bad)),
        "median_clean_good": float(np.median(good)),
    }


def primary_score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(row["consistency_score"]) for row in rows]
    labels = [row["primary_label"] == "clear_bad" for row in rows]
    classes = [int(row["class_id"]) for row in rows]
    result = descriptive_score_summary(rows, "consistency_score")
    result["exact_class_stratified_rank_sum"] = exact_stratified_rank_sum_dp(
        scores, labels, classes
    )
    result["class_stratified_bootstrap"] = stratified_bootstrap_auc(
        scores, labels, classes
    )
    return result


def run(args: argparse.Namespace) -> None:
    config = args.config.expanduser().resolve()
    probes = args.probes.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    prior_blur_path = args.prior_blur.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output}")
    if sha256_file(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen blur historical-validation config changed")
    if sha256_file(ROOT / "experiments/dit_projected_tweedie_cone.py") != EXPECTED_CORE_SHA256:
        raise RuntimeError("frozen PTCV numerical core changed")
    if (
        sha256_file(ROOT / "experiments/run_dit_projected_tweedie_cone_expansion_probe.py")
        != EXPECTED_EXPANSION_RUNNER_SHA256
    ):
        raise RuntimeError("frozen expansion PTCV runner changed")
    if (
        sha256_file(ROOT / "experiments/run_dit_projected_tweedie_cone_probe.py")
        != EXPECTED_FROZEN_RUNNER_SHA256
    ):
        raise RuntimeError("frozen discovery PTCV runner changed")
    specification = load_json(config)
    analyzer_source_sha256 = sha256_file(Path(__file__).resolve())
    dependencies = specification.get("frozen_implementation_dependencies", {})
    if (
        specification.get("status")
        != "FROZEN_AFTER_TARGETED100_PTCV_DISCOVERY_BEFORE_ANY_EXPANSION_PTCV_EXTRACTION_OR_SCORE_LABEL_JOIN"
        or specification.get("validation_population", {}).get("trajectory_count") != 360
        or specification.get("validation_population", {}).get("ordered_class_ids")
        != list(CLASSES)
        or dependencies.get("ptcv_numerical_core_sha256") != EXPECTED_CORE_SHA256
        or dependencies.get("strict_dit_helper_sha256") != EXPECTED_STRICT_SHA256
        or dependencies.get("trace_runner_sha256") != EXPECTED_TRACE_RUNNER_SHA256
        or dependencies.get("dit_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("frozen validation specification semantics changed")

    checkpoint_products, provenance = load_probe_products(probes, trace_root)
    path_rows, reference_rows, numerical = build_label_free_paths(
        checkpoint_products, specification
    )
    if (
        len(path_rows) != 360
        or len(checkpoint_products) != 1080
        or provenance["raw_replay_checkpoint_batches"] != 360
        or provenance["raw_replay_path_checkpoints"] != 1080
        or len(reference_rows)
        != len(CLASSES)
        * (len(STRICT_STANDARDIZED_FIELDS) + len(DESCRIPTIVE_STANDARDIZED_FIELDS))
    ):
        raise RuntimeError("label-free recomputation has the wrong frozen axis")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        path_file = staging / "label_free_path_scores.csv"
        reference_file = staging / "label_free_reference_stats.csv"
        checkpoint_file = staging / "label_free_checkpoint_products.csv"
        numerical_file = staging / "label_free_numerical_audit.json"
        provenance_file = staging / "label_free_provenance_audit.json"
        write_csv(path_file, path_rows)
        write_csv(reference_file, reference_rows)
        write_csv(checkpoint_file, checkpoint_products)
        write_json(numerical_file, numerical)
        write_json(provenance_file, provenance)
        fsync_directory(staging)
        label_free_seal: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_PTCV_BLUR_HISTORICAL_VALIDATION_LABEL_FREE_SEAL_V1",
            "path_scores_sha256": sha256_file(path_file),
            "reference_stats_sha256": sha256_file(reference_file),
            "checkpoint_products_sha256": sha256_file(checkpoint_file),
            "numerical_audit_sha256": sha256_file(numerical_file),
            "provenance_audit_sha256": sha256_file(provenance_file),
            "path_count": len(path_rows),
            "checkpoint_product_count": len(checkpoint_products),
            "config_sha256": sha256_file(config),
            "core_sha256": EXPECTED_CORE_SHA256,
            "runner_sha256": EXPECTED_EXPANSION_RUNNER_SHA256,
            "receipt_identities": provenance["receipt_identities"],
            "source_hash_rows_identity_sha256": provenance[
                "source_hash_rows_identity_sha256"
            ],
            "record_chain_rows_identity_sha256": provenance[
                "record_chain_rows_identity_sha256"
            ],
            "analyzer_source_sha256": analyzer_source_sha256,
            "labels_opened_before_score_reference_numerical_files_were_fsynced_and_hashed": False,
            "labels_file_sha256_computed_before_seal": False,
            "all_numerical_gates_pass": numerical["all_gates_pass"],
            "numerical_gates": numerical["gates"],
        }
        label_free_seal["identity_sha256"] = canonical_sha256(label_free_seal)
        write_json(staging / "label_free_seal.json", label_free_seal)
        fsync_directory(staging)

        if not numerical["all_gates_pass"]:
            results: dict[str, Any] = {
                "schema_version": 1,
                "artifact_kind": "DIT_PTCV_BLUR_HISTORICAL_VALIDATION_V1",
                "status": "stopped_before_labels",
                "decision": "STOP_NUMERICALLY_UNRESOLVED_LABELS_NOT_OPENED",
                "label_free_numerical": numerical,
                "provenance": provenance,
                "integrity": {
                    "config_sha256": EXPECTED_CONFIG_SHA256,
                    "analyzer_source_sha256": analyzer_source_sha256,
                    "label_free_seal_identity_sha256": label_free_seal[
                        "identity_sha256"
                    ],
                    "labels_opened": False,
                },
            }
            results["identity_sha256"] = canonical_sha256(results)
            write_json(staging / "results.json", results)
            fsync_directory(staging)
            os.replace(staging, output)
            fsync_directory(output.parent)
            print(
                json.dumps(
                    {"status": "stopped_before_labels", "output": str(output)}, indent=2
                )
            )
            return

        # The visual lock is first opened below this line, after the fsynced seal.
        label_rows, label_payload = load_labels_after_seal(labels_path)
        prior_blur, prior_blur_provenance = load_prior_blur_metric(prior_blur_path)
        joined = []
        for score in path_rows:
            key = (int(score["global_seed"]), int(score["class_id"]))
            label = label_rows[key]
            joined.append(
                {
                    **score,
                    "sample_key": label["sample_key"],
                    "primary_label": label["primary_label"],
                    "majority_flags": "|".join(label.get("majority_flags", [])),
                    "handcrafted_blur_B_score": prior_blur[key],
                }
            )
        counts = Counter(row["primary_label"] for row in joined)
        if counts != Counter({"clear_bad": 9, "clean_good": 304, "mild_or_disputed": 47}):
            raise RuntimeError("score-label join changed the locked endpoint counts")
        if any(
            "global_blur" not in row["majority_flags"].split("|")
            for row in joined
            if row["primary_label"] == "clear_bad"
        ):
            raise RuntimeError("primary blur phenotype assertion failed after join")
        primary_rows = [
            row for row in joined if row["primary_label"] in {"clear_bad", "clean_good"}
        ]
        if len(primary_rows) != 313:
            raise RuntimeError("primary endpoint must contain 9 bad and 304 good paths")

        primary = primary_score_summary(primary_rows)
        control_fields = (
            "matrix_energy_control",
            "finite_difference_gap_control",
            "temporal_change_control",
            "raw_conditional_cfg_gap_control",
        )
        controls = {
            field: descriptive_score_summary(primary_rows, field)
            for field in control_fields
        }
        checkpoint_leave_out = {
            str(checkpoint): descriptive_score_summary(
                primary_rows, f"score_without_checkpoint_{checkpoint}"
            )
            for checkpoint in CHECKPOINTS
        }
        checkpoint_specific = {
            str(checkpoint): descriptive_score_summary(
                primary_rows, f"checkpoint_{checkpoint}_score"
            )
            for checkpoint in CHECKPOINTS
        }
        component_specific = {
            "skew_component": descriptive_score_summary(
                primary_rows, "skew_component_score"
            ),
            "negative_eigen_component": descriptive_score_summary(
                primary_rows, "negative_eigen_component_score"
            ),
        }

        positive_classes = sorted(
            {
                int(row["class_id"])
                for row in primary_rows
                if row["primary_label"] == "clear_bad"
            }
        )
        per_class = {}
        leave_one_positive_class_out = {}
        for class_id in CLASSES:
            block = [row for row in primary_rows if int(row["class_id"]) == class_id]
            per_class[str(class_id)] = {
                "clear_bad": sum(row["primary_label"] == "clear_bad" for row in block),
                "clean_good": sum(row["primary_label"] == "clean_good" for row in block),
                "auc": auc_higher(
                    [float(row["consistency_score"]) for row in block],
                    [row["primary_label"] == "clear_bad" for row in block],
                ),
            }
        for class_id in positive_classes:
            remaining = [
                row for row in primary_rows if int(row["class_id"]) != class_id
            ]
            leave_one_positive_class_out[str(class_id)] = {
                "omitted_clear_bad": per_class[str(class_id)]["clear_bad"],
                "remaining_clear_bad": sum(
                    row["primary_label"] == "clear_bad" for row in remaining
                ),
                "remaining_clean_good": sum(
                    row["primary_label"] == "clean_good" for row in remaining
                ),
                "auc": auc_higher(
                    [float(row["consistency_score"]) for row in remaining],
                    [row["primary_label"] == "clear_bad" for row in remaining],
                ),
            }

        reference_by_class: dict[int, list[float]] = defaultdict(list)
        for row in path_rows:
            reference_by_class[int(row["class_id"])].append(
                float(row["raw_path_cone_violation"])
            )
        bad_details = []
        for row in primary_rows:
            if row["primary_label"] != "clear_bad":
                continue
            reference = np.asarray(reference_by_class[int(row["class_id"])])
            raw_score = float(row["raw_path_cone_violation"])
            bad_details.append(
                {
                    "sample_key": row["sample_key"],
                    "global_seed": int(row["global_seed"]),
                    "class_id": int(row["class_id"]),
                    "majority_flags": row["majority_flags"],
                    "raw_path_cone_violation": raw_score,
                    "standardized_score": float(row["consistency_score"]),
                    "rank_leq_of_120": int(np.count_nonzero(reference <= raw_score)),
                    "within_class_percentile_leq": float(np.mean(reference <= raw_score)),
                    "above_class_median": bool(raw_score > float(np.median(reference))),
                    "path_skew_fraction": float(row["path_skew_fraction"]),
                    "path_negative_eigen_fraction": float(
                        row["path_negative_eigen_fraction"]
                    ),
                    **{
                        f"checkpoint_{checkpoint}_score": float(
                            row[f"checkpoint_{checkpoint}_score"]
                        )
                        for checkpoint in CHECKPOINTS
                    },
                }
            )
        bad_details.sort(key=lambda row: row["standardized_score"], reverse=True)
        bad_above_median = sum(row["above_class_median"] for row in bad_details)

        prior_blur_summary = descriptive_score_summary(
            primary_rows, "handcrafted_blur_B_score"
        )
        prior_blur_summary.update(
            {
                "spearman_with_ptcv_all_360": spearman(
                    [float(row["handcrafted_blur_B_score"]) for row in joined],
                    [float(row["consistency_score"]) for row in joined],
                ),
                "provenance": prior_blur_provenance,
                "role": "descriptive_only_not_a_primary_score_or_gate",
            }
        )

        primary_auc = float(primary["auc"])
        exact_p = float(
            primary["exact_class_stratified_rank_sum"]["exact_one_sided_p"]
        )
        bootstrap_lower = float(
            primary["class_stratified_bootstrap"]["q025"]
        )
        maximum_control_auc = max(float(value["auc"]) for value in controls.values())
        class_loo_auc = [
            value["auc"] for value in leave_one_positive_class_out.values()
        ]
        positive_class_auc = [
            per_class[str(class_id)]["auc"] for class_id in positive_classes
        ]
        gates = {
            "every_label_free_numerical_gate_passed_before_labels_opened": numerical[
                "all_gates_pass"
            ],
            "primary_higher_direction_auc_at_least_0p70": primary_auc >= 0.70,
            "exact_higher_direction_class_stratified_p_at_most_0p05": exact_p <= 0.05,
            "at_least_6_of_9_clear_bad_above_class_median": bad_above_median >= 6,
            "bootstrap_95pct_lower_bound_strictly_above_0p50": bootstrap_lower > 0.50,
            "primary_auc_not_below_any_fixed_simple_control": primary_auc
            >= maximum_control_auc,
            "every_positive_class_leave_out_auc_at_least_0p55": bool(class_loo_auc)
            and all(value is not None and float(value) >= 0.55 for value in class_loo_auc),
            "at_least_two_positive_classes_have_auc_strictly_above_0p50": sum(
                value is not None and float(value) > 0.50 for value in positive_class_auc
            )
            >= 2,
        }
        advance = all(gates.values())
        strong_gates = {
            "all_advance_gates_pass": advance,
            "primary_auc_at_least_0p80": primary_auc >= 0.80,
            "exact_p_at_most_0p01": exact_p <= 0.01,
            "at_least_8_of_9_clear_bad_above_class_median": bad_above_median >= 8,
            "bootstrap_lower_bound_strictly_above_0p55": bootstrap_lower > 0.55,
            "every_positive_class_leave_out_auc_at_least_0p60": bool(class_loo_auc)
            and all(value is not None and float(value) >= 0.60 for value in class_loo_auc),
            "primary_exceeds_every_control_by_at_least_0p03": primary_auc
            >= maximum_control_auc + 0.03,
        }
        strong = all(strong_gates.values())
        decision = (
            "STRONG_HISTORICAL_VALIDATION_AUTHORIZE_PROSPECTIVE_BLUR_CONFIRMATION_ONLY"
            if strong
            else "ADVANCE_NARROW_BLUR_SIGNAL_TO_PROSPECTIVE_CONFIRMATION_ONLY"
            if advance
            else "STOP_PTCV_BLUR_SPECIFIC_HYPOTHESIS_NO_RESCUE"
        )

        joined_path = staging / "joined_historical_validation_rows.csv"
        write_csv(joined_path, joined)
        results = {
            "schema_version": 1,
            "artifact_kind": "DIT_PTCV_BLUR_HISTORICAL_VALIDATION_V1",
            "status": "complete",
            "decision": decision,
            "advance_narrow_blur_signal": advance,
            "strong_historical_validation": strong,
            "advance_gates": gates,
            "strong_gates": strong_gates,
            "primary": primary,
            "controls": controls,
            "leave_one_positive_class_out": leave_one_positive_class_out,
            "per_class": per_class,
            "leave_one_checkpoint_out_descriptive": checkpoint_leave_out,
            "checkpoint_specific_descriptive": checkpoint_specific,
            "component_specific_descriptive": component_specific,
            "prior_handcrafted_blur_B_descriptive": prior_blur_summary,
            "bad_details": bad_details,
            "bad_above_class_median": bad_above_median,
            "counts": dict(counts),
            "label_free_numerical": numerical,
            "provenance": provenance,
            "integrity": {
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "core_sha256": EXPECTED_CORE_SHA256,
                "expansion_runner_sha256": EXPECTED_EXPANSION_RUNNER_SHA256,
                "analyzer_source_sha256": analyzer_source_sha256,
                "label_free_seal_identity_sha256": label_free_seal[
                    "identity_sha256"
                ],
                "locked_consensus_file_sha256": EXPECTED_LABEL_FILE_SHA256,
                "locked_consensus_identity_sha256": label_payload["identity_sha256"],
                "joined_rows_sha256": sha256_file(joined_path),
                "score_reference_and_numerical_files_fsynced_and_hashed_before_labels_opened": True,
                "quality_direction_flipped_after_join": False,
                "sign_component_checkpoint_class_basis_radius_endpoint_or_combination_rescue_used": False,
                "historical_pool_previously_inspected_not_claimed_as_final_prospective_confirmation": True,
            },
            "claim_boundary": {
                "positive_result": (
                    "At most supports the unchanged full PTCV score as a narrow historical "
                    "global-blur association and authorizes a new prospective blur test."
                ),
                "negative_result": (
                    "Stops the PTCV blur-specific hypothesis without sign, component, "
                    "checkpoint, class, basis, radius, endpoint, or combination rescue."
                ),
                "never_authorized_here": [
                    "generic artifact detector claim",
                    "structural or topology detector claim",
                    "rejection",
                    "guidance",
                    "rollback",
                    "selective resampling",
                ],
                "path_weighting": (
                    "The unchanged full score is the block-diagonal cone ratio and weights "
                    "checkpoints by projected Jacobian energy, not equally."
                ),
            },
        }
        results["identity_sha256"] = canonical_sha256(results)
        write_json(staging / "results.json", results)
        fsync_directory(staging)
        os.replace(staging, output)
        fsync_directory(output.parent)
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
    assert math.isclose(spearman([1, 2, 3], [4, 5, 6]), 1.0)
    scores = [0.0, 1.0, 1.0, 3.0, 0.5, 2.0, 4.0]
    labels = [False, True, False, True, True, False, False]
    classes = [0, 0, 0, 0, 1, 1, 1]
    dynamic = exact_stratified_rank_sum_dp(scores, labels, classes)
    enumerated_ge, enumerated_total = _exact_stratified_enumeration_for_self_test(
        scores, labels, classes
    )
    assert dynamic["assignments"] == enumerated_total
    assert dynamic["greater_or_equal"] == enumerated_ge
    assert math.isclose(
        dynamic["exact_one_sided_p"], enumerated_ge / enumerated_total
    )
    distribution = _subset_sum_distribution([1, 2, 3, 4], 2)
    assert sum(distribution.values()) == math.comb(4, 2)
    assert distribution[5] == 2
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--prior-blur", type=Path, default=DEFAULT_PRIOR_BLUR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    run(parse_args())

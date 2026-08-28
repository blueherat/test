#!/usr/bin/env python3
"""Post-unseal integrity and robustness audit for Doob-consistency discovery V1.

This script does *not* repair or reinterpret the frozen decision rule.  It
independently recomputes every C_h, V_h, and rho from ``probes.npz``; checks the
receipt -> seed record -> file chain; verifies the reported primary AUC and an
exact class-stratified randomization p-value; and reports leave-one-axis-out
sensitivities.  Those sensitivities are descriptive because the V1 config did
not operationally define its "not solely one class/checkpoint/horizon" clause.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBES = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_probe_v1/shard_00_of_01"
)
DEFAULT_ANALYSIS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_analysis_v1"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_analysis_v1_audit"
)
EXPECTED_RECEIPT_IDENTITY = "1c9d03669c497cd586adc30352f30c44a5306f99e63f5d8285b4e4fa7c64b55f"
EXPECTED_ANALYSIS_IDENTITY = "6cb1228ef4d3657b2c6f7d33247438d0f0e5ff04b367cdb406856e955c5c8ce9"
FORMAL_ANALYZER_SHA256 = "c4eec1ea61a458f4c04051fa297595d9f9e3d7970f59ef7cf139d803319961b8"
CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
SEEDS = tuple(range(10, 30))
CHECKPOINTS = (99, 149, 199)
HORIZONS = (1, 2, 4, 8, 16)
PROBE_COUNT = 4


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


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def pairwise_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    pos = values[positive]
    neg = values[~positive]
    if not len(pos) or not len(neg):
        raise ValueError("AUC requires both endpoints")
    comparison = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(comparison.mean())


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


def exact_stratified_p(
    scores: Sequence[float], labels: Sequence[bool], classes: Sequence[int]
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    class_ids = np.asarray(classes, dtype=np.int64)
    ranks = midranks(values)
    n_positive = int(positive.sum())
    n_negative = len(values) - n_positive
    observed_sum = float(ranks[positive].sum())
    observed_auc = (
        observed_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    stratum_sums: list[np.ndarray] = []
    strata: dict[str, Any] = {}
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id)
        count = int(positive[indices].sum())
        strata[str(class_id)] = {"rows": len(indices), "positives": count}
        if count:
            stratum_sums.append(
                np.asarray(
                    [
                        float(ranks[list(choice)].sum())
                        for choice in itertools.combinations(indices.tolist(), count)
                    ],
                    dtype=np.float64,
                )
            )
    if not stratum_sums:
        raise RuntimeError("no positive stratum")
    combined = stratum_sums[0]
    for values_for_stratum in stratum_sums[1:]:
        combined = (combined[:, None] + values_for_stratum[None, :]).reshape(-1)
    greater_equal = int(np.count_nonzero(combined >= observed_sum - 1e-12))
    return {
        "observed_auc": float(observed_auc),
        "assignments": int(len(combined)),
        "greater_or_equal": greater_equal,
        "exact_one_sided_p": float(greater_equal / len(combined)),
        "strata": strata,
    }


def cross_and_energy(deltas: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(deltas, dtype=np.float64).reshape(len(deltas), -1)
    dimension = flat.shape[1]
    gram = flat @ flat.T / dimension
    cross = float(gram.sum(dtype=np.float64) - np.trace(gram))
    cross /= len(flat) * (len(flat) - 1)
    energy = float(np.einsum("md,md->", flat, flat, dtype=np.float64))
    energy /= len(flat) * dimension
    ratio = cross / max(energy, 1e-12)
    if not -1.0 / (len(flat) - 1) - 1e-10 <= ratio <= 1.0 + 1e-10:
        raise RuntimeError("coherence left its algebraic bounds")
    return cross, energy


def standardized_path_scores(
    rho: Mapping[tuple[int, int, int], float], checkpoints: Sequence[int]
) -> dict[tuple[int, int], float]:
    z: dict[tuple[int, int, int], float] = {}
    for class_id in CLASSES:
        for checkpoint in checkpoints:
            values = np.asarray(
                [rho[(seed, class_id, checkpoint)] for seed in SEEDS], dtype=np.float64
            )
            scale = float(values.std(ddof=1))
            if not math.isfinite(scale) or scale <= 0:
                raise RuntimeError("degenerate standardization block")
            mean = float(values.mean())
            for seed, value in zip(SEEDS, values):
                z[(seed, class_id, checkpoint)] = float((value - mean) / (scale + 1e-12))
    return {
        (seed, class_id): float(
            np.mean([z[(seed, class_id, checkpoint)] for checkpoint in checkpoints])
        )
        for seed in SEEDS
        for class_id in CLASSES
    }


def rank_details(
    all_rows: Sequence[Mapping[str, str]], primary_rows: Sequence[Mapping[str, str]], field: str
) -> list[dict[str, Any]]:
    by_class: dict[int, list[float]] = defaultdict(list)
    for row in all_rows:
        by_class[int(row["class_id"])].append(float(row[field]))
    details = []
    for row in primary_rows:
        if row["primary_label"] != "clear_bad":
            continue
        values = np.asarray(by_class[int(row["class_id"])], dtype=np.float64)
        score = float(row[field])
        details.append(
            {
                "sample_key": row["sample_key"],
                "class_id": int(row["class_id"]),
                "score": score,
                "rank_leq_of_20": int(np.count_nonzero(values <= score)),
                "within_class_percentile_leq": float(np.mean(values <= score)),
            }
        )
    return details


def run(args: argparse.Namespace) -> None:
    probes = args.probes.expanduser().resolve()
    analysis = args.analysis.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    original = load_json(analysis / "results.json")
    validate_self_hash(original, "identity_sha256", context=str(analysis / "results.json"))
    if original.get("identity_sha256") != EXPECTED_ANALYSIS_IDENTITY:
        raise RuntimeError("formal analysis identity changed")
    joined = load_csv(analysis / "joined_discovery_rows.csv")
    label_free = load_csv(analysis / "label_free_path_scores.csv")
    if sha256_file(analysis / "joined_discovery_rows.csv") != original.get(
        "joined_discovery_rows_sha256"
    ):
        raise RuntimeError("formal joined rows changed")

    receipt = load_json(probes / "receipt.json")
    validate_self_hash(receipt, "identity_sha256", context=str(probes / "receipt.json"))
    if receipt.get("identity_sha256") != EXPECTED_RECEIPT_IDENTITY:
        raise RuntimeError("probe receipt identity changed")
    receipt_records = receipt.get("records", [])
    receipt_by_seed = {int(row["global_seed"]): row for row in receipt_records}
    if set(receipt_by_seed) != set(SEEDS) or len(receipt_by_seed) != len(receipt_records):
        raise RuntimeError("receipt seed axis changed or duplicated")

    recomputed: dict[tuple[int, int, int, int], tuple[float, float]] = {}
    full_rho: dict[tuple[int, int, int], float] = {}
    max_horizon_error = 0.0
    max_rho_error = 0.0
    for seed in SEEDS:
        seed_dir = probes / f"seed{seed:02d}"
        record = load_json(seed_dir / "record.json")
        validate_self_hash(record, "identity_sha256", context=str(seed_dir / "record.json"))
        receipt_record = receipt_by_seed[seed]
        if (
            int(record.get("global_seed", -1)) != seed
            or record.get("identity_sha256") != receipt_record.get("identity_sha256")
            or Path(receipt_record.get("output", "")).resolve() != seed_dir
            or record.get("class_ids") != list(CLASSES)
            or record.get("checkpoints") != list(CHECKPOINTS)
            or record.get("horizons") != list(HORIZONS)
            or record.get("probe_count") != PROBE_COUNT
            or record.get("rng_namespace")
            != "eqvae-dit-doob-consistency-discovery-probe-v1"
        ):
            raise RuntimeError(f"receipt-to-record chain failed for seed {seed}")
        for name, expected in record.get("files", {}).items():
            path = seed_dir / name
            if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get(
                "sha256"
            ):
                raise RuntimeError(f"seed file integrity failed: {path}")

        horizon_rows = load_csv(seed_dir / "horizon_scores.csv")
        sample_rows = load_csv(seed_dir / "sample_scores.csv")
        horizon_by_axis = {
            (int(row["class_id"]), int(row["checkpoint"]), int(row["horizon"])): row
            for row in horizon_rows
        }
        sample_by_axis = {
            (int(row["class_id"]), int(row["checkpoint"])): row for row in sample_rows
        }
        if len(horizon_by_axis) != len(CLASSES) * len(CHECKPOINTS) * len(HORIZONS):
            raise RuntimeError("horizon CSV axis changed or duplicated")
        if len(sample_by_axis) != len(CLASSES) * len(CHECKPOINTS):
            raise RuntimeError("sample CSV axis changed or duplicated")

        with np.load(seed_dir / "probes.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "global_seed",
                "class_ids",
                "checkpoints",
                "horizons",
                "current_pred_xstart",
                "probe_pred_xstart",
            }:
                raise RuntimeError("probe archive schema changed")
            if (
                int(archive["global_seed"]) != seed
                or not np.array_equal(archive["class_ids"], np.asarray(CLASSES))
                or not np.array_equal(archive["checkpoints"], np.asarray(CHECKPOINTS))
                or not np.array_equal(archive["horizons"], np.asarray(HORIZONS))
            ):
                raise RuntimeError("probe archive axes changed")
            current = np.ascontiguousarray(archive["current_pred_xstart"])
            future = np.ascontiguousarray(archive["probe_pred_xstart"])
        if current.shape != (3, 8, 4, 32, 32) or future.shape != (
            3,
            4,
            5,
            8,
            4,
            32,
            32,
        ):
            raise RuntimeError("probe tensor shape changed")
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError("probe tensor contains non-finite values")

        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            for class_slot, class_id in enumerate(CLASSES):
                cross_sum = 0.0
                energy_sum = 0.0
                for horizon_index, horizon in enumerate(HORIZONS):
                    deltas = (
                        future[checkpoint_index, :, horizon_index, class_slot].astype(np.float64)
                        - current[checkpoint_index, class_slot]
                    )
                    cross, energy = cross_and_energy(deltas)
                    recomputed[(seed, class_id, checkpoint, horizon)] = (cross, energy)
                    expected = horizon_by_axis[(class_id, checkpoint, horizon)]
                    max_horizon_error = max(
                        max_horizon_error,
                        abs(cross - float(expected["pair_cross_u_stat"])),
                        abs(energy - float(expected["update_energy"])),
                    )
                    cross_sum += cross
                    energy_sum += energy
                rho = cross_sum / max(energy_sum, 1e-12)
                full_rho[(seed, class_id, checkpoint)] = rho
                max_rho_error = max(
                    max_rho_error,
                    abs(rho - float(sample_by_axis[(class_id, checkpoint)]["dyadic_coherence"])),
                )
    if max_horizon_error > 1e-12 or max_rho_error > 1e-12:
        raise RuntimeError("independent score recomputation did not match recorded scores")

    full_scores = standardized_path_scores(full_rho, CHECKPOINTS)
    label_free_by_key = {
        (int(row["global_seed"]), int(row["class_id"])): row for row in label_free
    }
    if set(label_free_by_key) != set(full_scores):
        raise RuntimeError("formal label-free axis changed")
    max_path_error = max(
        abs(full_scores[key] - float(label_free_by_key[key]["consistency_score"]))
        for key in full_scores
    )
    if max_path_error > 1e-12:
        raise RuntimeError("independent path score did not match formal label-free score")

    primary = [
        row for row in joined if row["primary_label"] in {"clear_bad", "clean_good"}
    ]
    scores = [float(row["consistency_score"]) for row in primary]
    labels = [row["primary_label"] == "clear_bad" for row in primary]
    classes = [int(row["class_id"]) for row in primary]
    independent_auc = pairwise_auc(scores, labels)
    if not math.isclose(independent_auc, float(original["primary"]["auc"]), abs_tol=1e-15):
        raise RuntimeError("independent AUC did not reproduce the formal result")
    exact = exact_stratified_p(scores, labels, classes)

    checkpoint_loo = {}
    for omitted in CHECKPOINTS:
        fields = [f"consistency_checkpoint_{value}" for value in CHECKPOINTS if value != omitted]
        values = [float(np.mean([float(row[field]) for field in fields])) for row in primary]
        checkpoint_loo[str(omitted)] = pairwise_auc(values, labels)

    horizon_loo = {}
    for omitted in HORIZONS:
        reduced_rho = {}
        for seed in SEEDS:
            for class_id in CLASSES:
                for checkpoint in CHECKPOINTS:
                    pairs = [
                        recomputed[(seed, class_id, checkpoint, horizon)]
                        for horizon in HORIZONS
                        if horizon != omitted
                    ]
                    reduced_rho[(seed, class_id, checkpoint)] = sum(v[0] for v in pairs) / max(
                        sum(v[1] for v in pairs), 1e-12
                    )
        reduced_scores = standardized_path_scores(reduced_rho, CHECKPOINTS)
        values = [
            reduced_scores[(int(row["global_seed"]), int(row["class_id"]))]
            for row in primary
        ]
        horizon_loo[str(omitted)] = pairwise_auc(values, labels)

    class_loo = {}
    for omitted in sorted(set(classes)):
        keep = [class_id != omitted for class_id in classes]
        kept_labels = [label for label, flag in zip(labels, keep) if flag]
        if not any(kept_labels):
            class_loo[str(omitted)] = None
            continue
        class_loo[str(omitted)] = pairwise_auc(
            [score for score, flag in zip(scores, keep) if flag], kept_labels
        )

    bad_loo = {}
    for index, row in enumerate(primary):
        if row["primary_label"] != "clear_bad":
            continue
        keep = [position != index for position in range(len(primary))]
        bad_loo[row["sample_key"]] = pairwise_auc(
            [score for score, flag in zip(scores, keep) if flag],
            [label for label, flag in zip(labels, keep) if flag],
        )

    audit: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "DIT_DOOB_CONSISTENCY_DISCOVERY_V1_POST_UNSEAL_AUDIT",
        "status": "complete",
        "formal_result": {
            "identity_sha256": original["identity_sha256"],
            "reported_decision": original["decision"],
            "formal_analyzer_sha256_at_run": FORMAL_ANALYZER_SHA256,
            "decision_reassessment": "DOWNGRADE_TO_BLUR_SPECIFIC_EXPLORATORY_SIGNAL",
            "reason": (
                "The frozen no-single-class/checkpoint/horizon clause was not fully "
                "operationalized by the V1 analyzer; class leave-out shows material "
                "dependence on class 795. The original artifact is preserved, not rewritten."
            ),
        },
        "data_integrity": {
            "receipt_to_record_to_file_chain_verified": True,
            "all_probe_arrays_finite_and_axes_exact": True,
            "raw_probe_moments_recomputed": len(SEEDS)
            * len(CLASSES)
            * len(CHECKPOINTS)
            * len(HORIZONS),
            "max_abs_horizon_moment_error": max_horizon_error,
            "max_abs_checkpoint_rho_error": max_rho_error,
            "max_abs_path_score_error": max_path_error,
        },
        "independent_primary_check": {
            "auc": independent_auc,
            "exact_class_stratified_randomization": exact,
            "reported_monte_carlo_p": original["primary"]["permutation"][
                "plus_one_one_sided_p"
            ],
            "bootstrap_interval_reported": original["primary"]["bootstrap"],
        },
        "descriptive_post_unseal_sensitivity": {
            "warning": "Not a replacement preregistered gate and not confirmatory.",
            "leave_one_class_out_auc": class_loo,
            "leave_one_checkpoint_out_auc": checkpoint_loo,
            "leave_one_horizon_out_auc": horizon_loo,
            "leave_one_clear_bad_out_auc": bad_loo,
        },
        "bad_rank_details": {
            field: rank_details(label_free, primary, field)
            for field in (
                "consistency_score",
                "temporal_change_control",
                "probe_energy_control",
            )
        },
        "interpretation": {
            "supported": (
                "A weak, directionally preregistered association concentrated on the three "
                "global-blur bad cases; it beats the two simple motion/energy controls."
            ),
            "not_supported": (
                "A general detector for blur, fusion, and structural misalignment, or a "
                "validated intervention trigger."
            ),
            "next_test": (
                "Freeze the same score and high-is-worse direction on a new blur-enriched "
                "pool with fresh blinded labels; keep structural errors as a separate endpoint."
            ),
        },
    }
    audit["identity_sha256"] = canonical_sha256(audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "audit.json", audit)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "output": str(output), "identity_sha256": audit["identity_sha256"]}, indent=2))


def self_test() -> None:
    assert math.isclose(pairwise_auc([0, 1, 2, 3], [False, True, False, True]), 0.75)
    deltas = np.asarray([[[1.0]], [[1.0]], [[1.0]], [[1.0]]])
    cross, energy = cross_and_energy(deltas)
    assert math.isclose(cross, 1.0) and math.isclose(energy, 1.0)
    cancel = np.asarray([[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]])
    cross, energy = cross_and_energy(cancel)
    assert math.isclose(cross / energy, -1.0 / 3.0)
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    run(parse_args())

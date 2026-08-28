#!/usr/bin/env python3
"""Extract a label-free denoiser-consistency diagnostic from frozen DiT scouts.

For a shared sampler state at index 0, let ``U_0`` be the current predicted
clean latent and let ``U_m(h)`` be the prediction after ``h`` transitions of
fresh branch ``m``.  With ``Delta_m(h) = U_m(h) - U_0``, the cross-branch
U-statistic

    mean_{m != n} <Delta_m(h), Delta_n(h)>

is conditionally unbiased for

    || E[U(h) | F_0] - U_0 ||^2.

The latter vanishes when the denoiser is harmonic under the *implemented*
reverse kernel.  This is the discrete operational version of the denoiser
consistency / reverse-martingale property.  The script never reads PNGs,
labels, external embeddings, or the earlier max-nonconformity selection.

This extractor is a mechanics probe, not a validated bad-image detector.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_INPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_prospective_v1_outputs"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_doob_consistency_probe_v1"
)
LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_2"
ARTIFACT_KIND = "DIT_V22_DOOB_CONSISTENCY_PROBE_V1"
FRESH_ATTEMPTS = (1, 2, 3, 4)
HORIZONS = (1, 2, 4, 8, 16)
EPSILON = 1e-12


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


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"self hash failed: {path}")
    return value


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid prospective lock tree: {root}")
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    protocol = load_self_hashed(root / "protocol.json", "identity_sha256")
    if (
        manifest.get("artifact_kind") != LOCK_KIND
        or manifest.get("status") != "complete"
        or protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256")
        or len(protocol.get("jobs", [])) != 128
    ):
        raise RuntimeError("prospective lock identity or scope changed")
    return manifest, protocol


def validate_job_manifest(path: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    completion_path = path / "completion.json"
    manifest = load_self_hashed(manifest_path, "identity_sha256")
    completion = load_self_hashed(completion_path, "payload_sha256")
    target = manifest.get("target", {})
    rollback = manifest.get("rollback", {})
    binding = manifest.get("prospective_binding", {})
    if (
        manifest.get("runner") != "intervene_dit_v22_transient_escape_suffix"
        or manifest.get("quality_scores_or_labels_used_by_runner") is not False
        or manifest.get("attempt_ranking_or_selection") is not False
        or target.get("global_seed") != job.get("global_seed")
        or target.get("class_id") != job.get("class_id")
        or target.get("slot") != job.get("class_slot")
        or rollback.get("sampling_step_index_zero_based") != job.get("rollback_sampling_step")
        or binding.get("job_index") != job.get("job_index")
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError(f"prospective job binding changed: {path}")
    return manifest


def load_prediction(path: Path, *, expected_attempt: int) -> tuple[np.ndarray, dict[str, Any]]:
    branch_path = path / "branch.json"
    branch = load_self_hashed(branch_path, "payload_sha256")
    if branch.get("attempt_index") != expected_attempt or branch.get("role") != "fresh_target_suffix":
        raise RuntimeError(f"wrong fresh branch identity: {path}")
    trace_record = branch.get("trace_npz", {})
    trace_path = path / "trace.npz"
    if (
        trace_record.get("relative_path") != f"branches/attempt_{expected_attempt:03d}/trace.npz"
        or trace_record.get("bytes") != trace_path.stat().st_size
        or trace_record.get("sha256") != sha256_file(trace_path)
    ):
        raise RuntimeError(f"fresh trace file changed: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as archive:
        if "target_pred_xstart" not in archive.files or "internal_timestep" not in archive.files:
            raise RuntimeError(f"required arrays missing: {trace_path}")
        prediction = np.ascontiguousarray(archive["target_pred_xstart"], dtype=np.float64)
        timesteps = np.ascontiguousarray(archive["internal_timestep"])
    expected = trace_record.get("arrays", {}).get("target_pred_xstart", {})
    if (
        prediction.ndim != 4
        or prediction.shape[1:] != (4, 32, 32)
        or len(prediction) <= max(HORIZONS)
        or timesteps.shape != (len(prediction),)
        or not np.array_equal(np.diff(timesteps.astype(np.int64)), -np.ones(len(timesteps) - 1, dtype=np.int64))
        or expected.get("shape") != list(prediction.shape)
        or expected.get("dtype") != "float32"
        or expected.get("raw_sha256") != raw_sha256(prediction.astype(np.float32))
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError(f"fresh prediction array failed validation: {trace_path}")
    return prediction, {
        "branch_payload_sha256": branch["payload_sha256"],
        "branch_file_sha256": sha256_file(branch_path),
        "trace_file_sha256": trace_record["sha256"],
        "prediction_raw_sha256": expected["raw_sha256"],
    }


def consistency_moments(deltas: np.ndarray) -> dict[str, float]:
    """Return full-vector U-statistic moments for ``[M, ...]`` updates."""

    if deltas.ndim < 2 or len(deltas) < 2 or not np.isfinite(deltas).all():
        raise ValueError("deltas must be finite with shape [M, ...] and M>=2")
    flat = deltas.reshape(len(deltas), -1).astype(np.float64, copy=False)
    dimension = flat.shape[1]
    individual = np.einsum("md,md->m", flat, flat, dtype=np.float64) / dimension
    gram = flat @ flat.T / dimension
    off_diagonal_sum = float(gram.sum(dtype=np.float64) - np.trace(gram))
    pair_cross = off_diagonal_sum / (len(flat) * (len(flat) - 1))
    update_energy = float(individual.mean(dtype=np.float64))
    mean_update = flat.mean(axis=0, dtype=np.float64)
    mean_update_energy = float(np.dot(mean_update, mean_update) / dimension)
    within_energy = float(
        np.einsum("md,md->", flat - mean_update, flat - mean_update, dtype=np.float64)
        / (len(flat) * dimension)
    )
    reconstructed = update_energy / len(flat) + (len(flat) - 1) * pair_cross / len(flat)
    if not math.isclose(mean_update_energy, reconstructed, rel_tol=2e-10, abs_tol=2e-12):
        raise RuntimeError("cross-branch moment identity failed")
    coherence = pair_cross / max(update_energy, EPSILON)
    lower = -1.0 / (len(flat) - 1)
    if not lower - 1e-10 <= coherence <= 1.0 + 1e-10:
        raise RuntimeError("normalized consistency coherence left its algebraic bounds")
    return {
        "pair_cross_u_stat": float(pair_cross),
        "update_energy": update_energy,
        "mean_update_energy": mean_update_energy,
        "within_energy": within_energy,
        "coherence": float(coherence),
    }


def analyze_job(
    input_root: Path, job: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    job_root = Path(str(job["outdir"])).expanduser().resolve()
    if job_root.parent != input_root or not job_root.is_dir() or job_root.is_symlink():
        raise RuntimeError(f"job output is outside the expected input root: {job_root}")
    manifest = validate_job_manifest(job_root, job)
    predictions = []
    provenance = []
    for attempt in FRESH_ATTEMPTS:
        prediction, record = load_prediction(
            job_root / "branches" / f"attempt_{attempt:03d}", expected_attempt=attempt
        )
        predictions.append(prediction)
        provenance.append({"attempt": attempt, **record})
    stacked = np.stack(predictions, axis=0)
    references = stacked[:, 0]
    if not np.array_equal(references, np.repeat(references[:1], len(references), axis=0)):
        raise RuntimeError(f"fresh branches do not share the exact h=0 prediction: {job_root}")
    reference = references[0]

    horizon_rows: list[dict[str, Any]] = []
    aggregate_cross = 0.0
    aggregate_energy = 0.0
    for horizon in HORIZONS:
        moments = consistency_moments(stacked[:, horizon] - reference)
        aggregate_cross += moments["pair_cross_u_stat"]
        aggregate_energy += moments["update_energy"]
        horizon_rows.append(
            {
                "job_index": int(job["job_index"]),
                "global_seed": int(job["global_seed"]),
                "class_id": int(job["class_id"]),
                "rollback_sampling_step": int(job["rollback_sampling_step"]),
                "horizon": horizon,
                **moments,
            }
        )
    aggregate_coherence = aggregate_cross / max(aggregate_energy, EPSILON)
    if not -1.0 / (len(FRESH_ATTEMPTS) - 1) - 1e-10 <= aggregate_coherence <= 1.0 + 1e-10:
        raise RuntimeError("aggregate consistency coherence left its algebraic bounds")
    job_row = {
        "job_index": int(job["job_index"]),
        "global_seed": int(job["global_seed"]),
        "class_id": int(job["class_id"]),
        "rollback_sampling_step": int(job["rollback_sampling_step"]),
        "dyadic_horizons": "|".join(str(value) for value in HORIZONS),
        "dyadic_pair_cross_sum": aggregate_cross,
        "dyadic_update_energy_sum": aggregate_energy,
        "dyadic_coherence": aggregate_coherence,
        "positive_dyadic_coherence_descriptive": max(aggregate_coherence, 0.0),
        "job_manifest_identity_sha256": manifest["identity_sha256"],
    }
    return job_row, horizon_rows, provenance


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("quantiles require a finite nonempty vector")
    return {
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("correlation vectors have incompatible shapes")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError("CSV rows do not share an exact ordered schema")
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


def run(args: argparse.Namespace) -> None:
    lock = args.lock.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    lock_manifest, protocol = validate_lock(lock)
    if input_root.is_symlink() or not input_root.is_dir():
        raise RuntimeError(f"invalid input root: {input_root}")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    jobs = sorted(protocol["jobs"], key=lambda row: int(row["job_index"]))
    if [int(row["job_index"]) for row in jobs] != list(range(128)):
        raise RuntimeError("job axis is not exactly 0..127")
    job_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for ordinal, job in enumerate(jobs, start=1):
        row, by_horizon, provenance = analyze_job(input_root, job)
        job_rows.append(row)
        horizon_rows.extend(by_horizon)
        provenance_rows.extend(
            {"job_index": int(job["job_index"]), **record} for record in provenance
        )
        print(f"validated {ordinal}/{len(jobs)} jobs", flush=True)

    by_horizon_values = {
        str(horizon): [
            float(row["coherence"])
            for row in horizon_rows
            if int(row["horizon"]) == horizon
        ]
        for horizon in HORIZONS
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "status": "LABEL_FREE_MECHANICS_PROBE_NOT_A_QUALITY_DETECTOR",
        "scientific_object": {
            "denoiser": "implemented DiT pred_xstart latent",
            "implemented_kernel": "frozen ancestral DDPM-250 P with CFG=4",
            "fresh_branch_count": len(FRESH_ATTEMPTS),
            "horizons": list(HORIZONS),
            "primary_label_free_diagnostic": "dyadic_coherence",
            "formula": (
                "sum_h mean_{m!=n}<Delta_m(h),Delta_n(h)> / "
                "sum_h mean_m||Delta_m(h)||^2"
            ),
            "conditional_expectation_identity": (
                "E[mean_{m!=n}<Delta_m,Delta_n>|F0]="
                "||E[U(h)|F0]-U0||^2"
            ),
            "interpretation_boundary": (
                "A positive consistency defect is a necessary-law violation relative to the "
                "implemented sampler/denoiser pair; it is not by itself a probability of a bad image."
            ),
            "prior_art_boundary": (
                "The two-branch cross-product estimator is the Consistent Diffusion Models CP "
                "training loss (Daras et al., NeurIPS 2023). This artifact tests a frozen-model "
                "inference diagnostic and makes no first-invention claim for the identity or estimator."
            ),
        },
        "firewall": {
            "pngs_opened": False,
            "visual_labels_reviews_or_private_mapping_opened": False,
            "external_embeddings_opened": False,
            "previous_max_nonconformity_selection_opened_or_used": False,
            "allowed_arrays": ["internal_timestep", "target_pred_xstart"],
        },
        "input": {
            "lock_identity_sha256": lock_manifest["identity_sha256"],
            "protocol_identity_sha256": protocol["identity_sha256"],
            "input_root": str(input_root),
            "job_count": len(job_rows),
            "fresh_trace_count": len(provenance_rows),
        },
        "mechanics": {
            "dyadic_coherence": quantiles(
                [float(row["dyadic_coherence"]) for row in job_rows]
            ),
            "per_horizon_coherence": {
                key: quantiles(values) for key, values in by_horizon_values.items()
            },
            "cross_horizon_pearson": {
                f"h{left}_h{right}": pearson(
                    by_horizon_values[str(left)], by_horizon_values[str(right)]
                )
                for index, left in enumerate(HORIZONS)
                for right in HORIZONS[index + 1 :]
            },
            "positive_dyadic_count": sum(
                float(row["dyadic_coherence"]) > 0.0 for row in job_rows
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_csv(staging / "job_scores.csv", job_rows)
        write_csv(staging / "horizon_scores.csv", horizon_rows)
        write_json(staging / "source_provenance.json", provenance_rows)
        summary["files"] = {
            name: {
                "bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in ("job_scores.csv", "horizon_scores.csv", "source_provenance.json")
        }
        summary["identity_sha256"] = canonical_sha256(summary)
        write_json(staging / "summary.json", summary)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "output": str(output), "identity_sha256": summary["identity_sha256"]}, indent=2))


def self_test() -> None:
    common = np.ones((4, 2, 2), dtype=np.float64)
    perfect = consistency_moments(common)
    assert math.isclose(perfect["pair_cross_u_stat"], 1.0)
    assert math.isclose(perfect["coherence"], 1.0)

    cancelling = np.asarray(
        [
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, -1.0]],
        ]
    )
    cancelled = consistency_moments(cancelling)
    assert math.isclose(cancelled["mean_update_energy"], 0.0, abs_tol=1e-14)
    assert math.isclose(cancelled["coherence"], -1.0 / 3.0, abs_tol=1e-14)

    rng = np.random.default_rng(20260828)
    estimates = []
    drift = np.linspace(-0.2, 0.3, 32)
    expected = float(np.dot(drift, drift) / len(drift))
    for _ in range(4000):
        updates = drift + rng.normal(size=(4, len(drift)))
        estimates.append(consistency_moments(updates)["pair_cross_u_stat"])
    assert abs(float(np.mean(estimates)) - expected) < 0.015
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    run(parse_args())

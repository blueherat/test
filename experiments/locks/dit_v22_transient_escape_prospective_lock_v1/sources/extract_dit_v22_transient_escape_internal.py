#!/usr/bin/env python3
"""Seal the prospective DiT-v2.2 h10 selector before opening any images.

Only mechanical manifests and the ``internal_timestep`` /
``target_pred_xstart`` arrays are read.  The selector uses the truly shared
h=0 prediction for multiscale normalization and ranks the four fresh scouts
symmetrically.  PNG pixels, reviews, B/E/O, endpoint quality scores, FID, and
external representations are outside this program's input surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1"
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_internal_v1"
)
LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1"
PRODUCT_KIND = "DIT_V22_TRANSIENT_ESCAPE_INTERNAL_PRODUCT_V1"
ALLOWED_TRACE_KEYS = {"internal_timestep", "target_pred_xstart"}
SCALES = (1, 2, 4)
HORIZONS = (5, 10, 20)
PRIMARY_HORIZON = 10
EPSILON = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"self hash failed: {path}")
    return value


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid prospective lock tree: {root}")
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != LOCK_KIND or manifest.get("status") != "complete":
        raise RuntimeError("wrong prospective lock kind/status")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("prospective lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"prospective lock member changed: {name}")
    protocol = load_self_hashed(root / "protocol.json", "identity_sha256")
    config = load_json(root / "frozen_config.json")
    if (
        protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256")
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
        or len(protocol.get("jobs", [])) != 128
        or config.get("internal_selector", {}).get("primary_horizon_index") != PRIMARY_HORIZON
    ):
        raise RuntimeError("prospective protocol/config scope changed")
    return manifest, protocol, config


def center_channels(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {value.shape}")
    return value - np.mean(value, axis=(1, 2), keepdims=True, dtype=np.float64)


def average_pool(array: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return np.asarray(array, dtype=np.float64)
    channels, height, width = array.shape
    if height % scale or width % scale:
        raise ValueError("pool scale does not divide latent dimensions")
    return np.mean(
        array.reshape(channels, height // scale, scale, width // scale, scale),
        axis=(2, 4),
        dtype=np.float64,
    )


def rms(array: np.ndarray) -> float:
    value = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value, dtype=np.float64)))


def shared_reference_norms(prefix: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    centered = center_channels(prefix)
    raw = {scale: rms(average_pool(centered, scale)) for scale in SCALES}
    effective = {scale: raw[scale] + EPSILON for scale in SCALES}
    if any(not math.isfinite(value) or value <= 0 for value in effective.values()):
        raise RuntimeError("invalid shared-prefix normalizer")
    return raw, effective


def pair_distance(left: np.ndarray, right: np.ndarray, norms: Mapping[int, float]) -> float:
    delta = center_channels(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    values = [rms(average_pool(delta, scale)) / norms[scale] for scale in SCALES]
    result = float(np.mean(values, dtype=np.float64))
    if not math.isfinite(result) or result < 0:
        raise RuntimeError("invalid pair distance")
    return result


def distance_matrix(branches: np.ndarray, norms: Mapping[int, float]) -> np.ndarray:
    if branches.ndim != 4:
        raise ValueError("branches must have shape [K,C,H,W]")
    count = branches.shape[0]
    matrix = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = pair_distance(branches[left], branches[right], norms)
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def nonconformity(matrix: np.ndarray) -> dict[int, float]:
    indices = range(matrix.shape[0])
    return {
        index + 1: float(np.mean([matrix[index, other] for other in indices if other != index]))
        for index in indices
    }


def tied_extreme(
    values: Mapping[int, float],
    slot_map: Mapping[int, str],
    *,
    maximum: bool,
) -> tuple[int, bool]:
    extreme = max(values.values()) if maximum else min(values.values())
    tied = [
        attempt
        for attempt, value in values.items()
        if math.isclose(value, extreme, rel_tol=0.0, abs_tol=1e-15)
    ]
    return min(tied, key=lambda attempt: slot_map[attempt]), len(tied) > 1


def load_branch_arrays(
    job: Mapping[str, Any], inventory: list[dict[str, Any]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    root = Path(job["outdir"])
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = load_self_hashed(manifest_path, "identity_sha256")
    completion = load_self_hashed(completion_path, "payload_sha256")
    streams = [row.get("seed") for row in manifest.get("branches", {}).get("fresh_stream_seeds", [])]
    if (
        manifest.get("target", {}).get("global_seed") != job["global_seed"]
        or manifest.get("target", {}).get("class_id") != job["class_id"]
        or manifest.get("target", {}).get("slot") != job["class_slot"]
        or manifest.get("rollback", {}).get("sampling_step_index_zero_based") != job["rollback_sampling_step"]
        or streams != job["fresh_stream_seeds"]
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
    ):
        raise RuntimeError(f"completed branch bundle differs from frozen job: {root}")
    inventory.extend(
        [
            {"path": str(manifest_path), "basename": "manifest.json", "sha256": sha256_file(manifest_path)},
            {"path": str(completion_path), "basename": "completion.json", "sha256": sha256_file(completion_path)},
        ]
    )
    timesteps: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    for attempt in range(5):
        branch_root = root / "branches" / f"attempt_{attempt:03d}"
        metadata_path = branch_root / "branch.json"
        metadata = load_self_hashed(metadata_path, "payload_sha256")
        trace_record = metadata.get("trace_npz", {})
        trace_path = branch_root / "trace.npz"
        if (
            metadata.get("attempt_index") != attempt
            or trace_record.get("sha256") != sha256_file(trace_path)
        ):
            raise RuntimeError(f"branch metadata/trace changed: {branch_root}")
        with np.load(trace_path, allow_pickle=False) as archive:
            if not ALLOWED_TRACE_KEYS.issubset(set(archive.files)):
                raise RuntimeError("required internal arrays are absent")
            current_t = np.ascontiguousarray(archive["internal_timestep"])
            current_pred = np.ascontiguousarray(archive["target_pred_xstart"])
        if (
            current_t.ndim != 1
            or current_pred.shape != (len(current_t), 4, 32, 32)
            or current_pred.dtype != np.float32
            or not np.isfinite(current_pred).all()
        ):
            raise RuntimeError(f"invalid internal branch trace: {trace_path}")
        if timesteps is None:
            timesteps = current_t
        elif not np.array_equal(timesteps, current_t):
            raise RuntimeError("branch timestep axes differ")
        predictions.append(current_pred)
        inventory.extend(
            [
                {"path": str(metadata_path), "basename": "branch.json", "sha256": sha256_file(metadata_path)},
                {
                    "path": str(trace_path),
                    "basename": "trace.npz",
                    "sha256": sha256_file(trace_path),
                    "arrays_read": sorted(ALLOWED_TRACE_KEYS),
                },
            ]
        )
    assert timesteps is not None
    return timesteps, predictions


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def build(args: argparse.Namespace) -> None:
    lock = args.lock.expanduser().resolve()
    output = args.output.expanduser().absolute()
    if output.exists():
        manifest = load_self_hashed(output / "manifest.json", "identity_sha256")
        completion = load_self_hashed(output / "completion.json", "identity_sha256")
        if (
            manifest.get("artifact_kind") != PRODUCT_KIND
            or completion.get("product_identity_sha256") != manifest.get("identity_sha256")
        ):
            raise RuntimeError(f"existing internal product differs: {output}")
        print(json.dumps({"status": "validated", "output": str(output), "identity_sha256": manifest["identity_sha256"]}, indent=2))
        return
    lock_manifest, protocol, config = validate_lock(lock)
    inventory: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for job in protocol["jobs"]:
        timesteps, predictions = load_branch_arrays(job, inventory)
        if any(horizon >= len(timesteps) for horizon in HORIZONS):
            raise RuntimeError("frozen horizon exceeds suffix length")
        shared = predictions[0][0]
        if not all(np.array_equal(shared, branch[0]) for branch in predictions):
            raise RuntimeError("five branches do not share the h0 prediction exactly")
        raw_norms, effective_norms = shared_reference_norms(shared)
        slot_map = {int(key): value for key, value in job["physical_attempt_to_anonymous_slot"].items()}
        primary_values: dict[int, float] | None = None
        primary_max: int | None = None
        primary_min: int | None = None
        primary_max_tie = False
        primary_min_tie = False
        for horizon in HORIZONS:
            fresh = np.stack([predictions[attempt][horizon] for attempt in range(1, 5)], axis=0)
            matrix = distance_matrix(fresh, effective_norms)
            values = nonconformity(matrix)
            maximum, maximum_tie = tied_extreme(values, slot_map, maximum=True)
            minimum, minimum_tie = tied_extreme(values, slot_map, maximum=False)
            ranks = {
                attempt: sum(value >= values[attempt] for value in values.values()) / 4.0
                for attempt in values
            }
            row: dict[str, Any] = {
                "job_index": job["job_index"],
                "global_seed": job["global_seed"],
                "class_id": job["class_id"],
                "class_slot": job["class_slot"],
                "rollback_step": job["rollback_sampling_step"],
                "horizon": horizon,
                "horizon_internal_timestep": int(timesteps[horizon]),
                "primary_horizon": horizon == PRIMARY_HORIZON,
                "shared_h0_pred_xstart_raw_sha256": raw_sha256(shared),
                "normalizer_source": "exact_shared_h0_pred_xstart",
                "normalizer_scale1_raw": raw_norms[1],
                "normalizer_scale2_raw": raw_norms[2],
                "normalizer_scale4_raw": raw_norms[4],
                "max_nonconformity_attempt": maximum,
                "max_nonconformity_slot": slot_map[maximum],
                "max_tie": maximum_tie,
                "min_nonconformity_attempt": minimum,
                "min_nonconformity_slot": slot_map[minimum],
                "min_tie": minimum_tie,
            }
            for attempt in range(1, 5):
                row[f"attempt{attempt}_slot"] = slot_map[attempt]
                row[f"attempt{attempt}_nonconformity"] = values[attempt]
                row[f"attempt{attempt}_rank_p_descriptive"] = ranks[attempt]
            features.append(row)
            matrices.append(
                {
                    "job_index": job["job_index"],
                    "global_seed": job["global_seed"],
                    "class_id": job["class_id"],
                    "horizon": horizon,
                    "physical_attempt_order": [1, 2, 3, 4],
                    "distance_matrix": matrix.tolist(),
                    "nonconformity": {str(key): value for key, value in values.items()},
                }
            )
            if horizon == PRIMARY_HORIZON:
                primary_values = values
                primary_max = maximum
                primary_min = minimum
                primary_max_tie = maximum_tie
                primary_min_tie = minimum_tie
        assert primary_values is not None and primary_max is not None and primary_min is not None
        random_attempt = int(job["hash_random_control_attempt"])
        selections.append(
            {
                "job_index": job["job_index"],
                "global_seed": job["global_seed"],
                "class_id": job["class_id"],
                "rollback_step": job["rollback_sampling_step"],
                "primary_horizon": PRIMARY_HORIZON,
                "selector": "argmax_h10_fresh_mean_nonconformity",
                "max_attempt": primary_max,
                "max_slot": slot_map[primary_max],
                "max_tie": primary_max_tie,
                "medoid_attempt": primary_min,
                "medoid_slot": slot_map[primary_min],
                "medoid_tie": primary_min_tie,
                "hash_random_attempt": random_attempt,
                "hash_random_slot": job["hash_random_control_slot"],
                "max_nonconformity": primary_values[primary_max],
                "medoid_nonconformity": primary_values[primary_min],
                "all_four_fresh_endpoints_must_remain_available_to_external_judge": True,
            }
        )
    if len(features) != 128 * 3 or len(matrices) != 128 * 3 or len(selections) != 128:
        raise RuntimeError("prospective internal product row count changed")
    if len({row["job_index"] for row in selections}) != 128:
        raise RuntimeError("prospective internal product job identity changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_bytes(staging / "features.csv", csv_bytes(features))
        write_json(staging / "features.json", features)
        write_json(staging / "distance_matrices.json", matrices)
        write_json(staging / "sealed_selections.json", selections)
        inventory_payload = {
            "schema_version": 1,
            "opened_file_count": len(inventory),
            "opened_files": inventory,
            "opened_basenames": sorted({row["basename"] for row in inventory}),
            "trace_arrays_read": sorted(ALLOWED_TRACE_KEYS),
            "png_pixels_review_labels_B_E_O_quality_FID_or_embeddings_opened": False,
        }
        inventory_payload["identity_sha256"] = canonical_sha256(inventory_payload)
        write_json(staging / "input_inventory.json", inventory_payload)
        shutil.copy2(lock / "frozen_config.json", staging / "frozen_config.json")
        shutil.copy2(Path(__file__).resolve(), staging / "extractor_source.py")
        payload_files = sorted(path for path in staging.iterdir() if path.is_file())
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": PRODUCT_KIND,
            "status": "complete",
            "scientific_role": "prospective_sampler_internal_selection_sealed_before_external_judging",
            "lock_identity_sha256": lock_manifest["identity_sha256"],
            "protocol_identity_sha256": protocol["identity_sha256"],
            "counts": {"jobs": 128, "feature_rows": len(features), "selection_rows": len(selections)},
            "selector": "step149_h10_argmax_fresh_mean_nonconformity",
            "normalizer_source": "exact shared h0 predicted-clean latent",
            "attempt0_O_lowO_B_E_or_external_metric_computed": False,
            "png_pixels_opened": False,
            "all_outputs_retained": True,
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payload_files
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "product_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "sealed_selections_file_sha256": sha256_file(staging / "sealed_selections.json"),
            "external_judging_may_begin_after_this_product": True,
        }
        completion["identity_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "sealed",
                "output": str(output),
                "identity_sha256": manifest["identity_sha256"],
                "jobs": 128,
                "png_or_external_metrics_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def self_test() -> None:
    prefix = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    raw, norms = shared_reference_norms(prefix)
    branches = np.stack([prefix + index for index in range(4)], axis=0)
    matrix = distance_matrix(branches, norms)
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0):
        raise AssertionError("distance symmetry self-test failed")
    values = nonconformity(matrix)
    slot_map = {1: "D", 2: "B", 3: "A", 4: "C"}
    maximum, _ = tied_extreme(values, slot_map, maximum=True)
    minimum, _ = tied_extreme(values, slot_map, maximum=False)
    if maximum not in values or minimum not in values or any(value <= 0 for value in norms.values()):
        raise AssertionError("selector self-test failed")
    print("self-test passed: shared normalization, symmetric distances, anonymous tie breaks")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        build(parsed)

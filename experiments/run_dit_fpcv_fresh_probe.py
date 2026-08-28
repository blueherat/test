#!/usr/bin/env python3
"""Run label-free FPCV probes on the frozen 240-path fresh DiT cohort.

At sampling-array checkpoints 99, 149, and 199, this runner evaluates the raw
class-conditional, unclipped posterior-mean predictor on a 33-point projected
cross-polytope.  A finite cyclic-monotonicity score is computed independently
at two fixed radii.  The large radius is the only primary score; the small
radius is a numerical control.  No endpoint, decoded image, embedding, review,
or quality label is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import scipy
import torch

try:
    from . import reproduce_dit_imagenet256 as strict
    from .dit_finite_posterior_cyclic_violation import (
        CROSS_POLYTOPE_DIMENSION,
        CROSS_POLYTOPE_POINT_COUNT,
        LARGE_RELATIVE_RADIUS,
        SMALL_RELATIVE_RADIUS,
        cross_polytope_coordinates,
        project_outputs,
        score_projected_cross_polytope,
    )
    from .dit_projected_tweedie_cone import (
        DEFAULT_FREQUENCIES,
        build_hadamard_dct_basis,
    )
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    from dit_finite_posterior_cyclic_violation import (
        CROSS_POLYTOPE_DIMENSION,
        CROSS_POLYTOPE_POINT_COUNT,
        LARGE_RELATIVE_RADIUS,
        SMALL_RELATIVE_RADIUS,
        cross_polytope_coordinates,
        project_outputs,
        score_projected_cross_polytope,
    )
    from dit_projected_tweedie_cone import (
        DEFAULT_FREQUENCIES,
        build_hadamard_dct_basis,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae")
)
DEFAULT_DIT_ROOT = DEFAULT_DATA_ROOT / "baselines/DiT"
DEFAULT_TRACE_ROOT = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/dit_bad_good_confirmation_v1_custom_traces_cfg_locked"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "cross_scale_evidence/dit_fpcv_fresh_probe_v1"
)
RUNNER_NAME = "run_dit_fpcv_fresh_probe"
ARTIFACT_KIND = "DIT_FPCV_FRESH_PROBE_SHARD_V1"
SEED_ARTIFACT_KIND = "DIT_FPCV_FRESH_PROBE_SEED_V1"
CLASSES = (207, 602, 795)
SEEDS = tuple(range(50, 130))
CHECKPOINTS = (99, 149, 199)
EXPECTED_INTERNAL_TIMESTEPS = (150, 100, 50)
RELATIVE_RADII = (SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS)
RADIUS_ROLES = ("small_control", "large_primary")
SHARD_COUNT = 4
# Three points per chunk gives 11 identically shaped 3-class batches for all
# 33 points.  Unequal final batches can select different TF32/GEMM kernels and
# inject a direction-dependent numerical artifact into the assignment score.
QUERY_POINT_CHUNK = 3
EXPECTED_TRACE_RUNNER_SHA256 = "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
EXPECTED_STRICT_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
EXPECTED_PTCV_CORE_SHA256 = "986f0fc8bbf22b84731ffb9b8b73bc9d73db263ae7f32d05e4ec812acf6900fe"
EXPECTED_FPCV_CORE_SHA256 = "7c2f636eadb44a802b4cec8410f6462da2759e5b1fca6bbdd6238a232879c29a"
EXPECTED_BASIS_RAW_SHA256 = "698fa3fcf6a67265ccdb618f3d1c6642affd03aa41dbcb5ffce8d6f36529d179"
EXPECTED_CHECKPOINT_SHA256 = "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"
EXPECTED_MODELS_SHA256 = "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca"
SOURCE_ARRAYS = (
    "state_before",
    "conditional_epsilon_raw",
    "internal_timestep",
    "alpha_bar",
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError("CSV row schema changed")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_source_trace(
    source: Path, expected_seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate lineage and read exactly the four approved trace arrays."""

    manifest_path = source / "manifest.json"
    completion_path = source / "completion.json"
    trace_path = source / "trace.npz"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity", {})
    protocol = identity.get("protocol", {})
    source_identity = identity.get("source", {})
    trace_record = next(
        (
            row
            for row in manifest.get("outputs", [])
            if isinstance(row, dict) and row.get("relative_path") == "trace.npz"
        ),
        None,
    )
    if (
        manifest.get("status") != "complete"
        or identity.get("runner") != "trace_dit_imagenet256_custom_batch"
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or protocol.get("global_torch_seed") != expected_seed
        or tuple(protocol.get("class_ids_ordered", [])) != CLASSES
        or protocol.get("batch_size_before_duplication") != len(CLASSES)
        or protocol.get("sampling_steps") != 250
        or protocol.get("cfg_scale") != 4.0
        or protocol.get("clip_denoised") is not False
        or identity.get("runner_source", {}).get("sha256")
        != EXPECTED_TRACE_RUNNER_SHA256
        or identity.get("strict_reproduction_helper", {}).get("sha256")
        != EXPECTED_STRICT_SHA256
        or identity.get("checkpoint", {}).get("sha256")
        != EXPECTED_CHECKPOINT_SHA256
        or source_identity.get("pinned_source_sha256", {}).get("models.py")
        != EXPECTED_MODELS_SHA256
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("output_count") != 8
        or not isinstance(trace_record, dict)
        or trace_record.get("sha256") != sha256_file(trace_path)
    ):
        raise RuntimeError(f"fresh source trace validation failed: {source}")

    with np.load(trace_path, allow_pickle=False) as archive:
        if any(name not in archive.files for name in SOURCE_ARRAYS):
            raise RuntimeError(f"approved source trace arrays are missing: {source}")
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in SOURCE_ARRAYS
        }
    if set(arrays) != set(SOURCE_ARRAYS):
        raise RuntimeError("source-array firewall changed")
    if (
        arrays["state_before"].shape != (len(CLASSES), 250, 4, 32, 32)
        or arrays["conditional_epsilon_raw"].shape
        != (len(CLASSES), 250, 4, 32, 32)
        or arrays["internal_timestep"].shape != (250,)
        or arrays["alpha_bar"].shape != (250,)
        or arrays["state_before"].dtype != np.float32
        or arrays["conditional_epsilon_raw"].dtype != np.float32
        or arrays["internal_timestep"].dtype != np.int16
        or arrays["alpha_bar"].dtype != np.float64
        or not np.array_equal(
            arrays["internal_timestep"], np.arange(249, -1, -1, dtype=np.int16)
        )
        or not all(np.isfinite(arrays[name]).all() for name in SOURCE_ARRAYS)
        or np.any(arrays["alpha_bar"] <= 0.0)
        or np.any(arrays["alpha_bar"] > 1.0)
    ):
        raise RuntimeError(f"fresh source trace tensor contract changed: {source}")
    records = manifest.get("trace_array_records", {})
    for name in SOURCE_ARRAYS:
        if records.get(name, {}).get("raw_sha256") != raw_sha256(arrays[name]):
            raise RuntimeError(f"source raw-array hash failed for {name}: {source}")
    return manifest, arrays


def raw_conditional_prediction(
    diffusion: Any,
    model: Any,
    states: torch.Tensor,
    *,
    internal_t: int,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    """Query raw class-conditional, unclipped ``pred_xstart`` (never CFG)."""

    if states.ndim != 4 or class_ids.shape != (len(states),):
        raise ValueError("raw conditional query batch contract changed")
    timesteps = torch.full(
        (len(states),), internal_t, dtype=torch.long, device=states.device
    )
    with torch.no_grad():
        output = diffusion.p_mean_variance(
            model.forward,
            states,
            timesteps,
            clip_denoised=False,
            model_kwargs={"y": class_ids},
        )
    prediction = output["pred_xstart"].contiguous()
    if prediction.shape != states.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("raw conditional posterior-mean prediction is invalid")
    return prediction


def replay_raw_conditional(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    recorded_epsilon: np.ndarray,
    internal_t: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the recorded conditional half before making FPCV queries."""

    reference = torch.from_numpy(state).to(device)
    class_ids = torch.tensor(CLASSES, dtype=torch.long, device=device)
    null_ids = torch.full_like(class_ids, strict.NULL_CLASS_ID)
    full_states = torch.cat([reference, reference], dim=0)
    full_ids = torch.cat([class_ids, null_ids], dim=0)
    prediction = raw_conditional_prediction(
        diffusion,
        model,
        full_states,
        internal_t=internal_t,
        class_ids=full_ids,
    )[: len(CLASSES)]
    timesteps = torch.full(
        (len(CLASSES),), internal_t, dtype=torch.long, device=device
    )
    expected = diffusion._predict_xstart_from_eps(
        x_t=reference,
        t=timesteps,
        eps=torch.from_numpy(recorded_epsilon).to(device),
    )
    maximum = float((prediction - expected).abs().max())
    bitwise = bool(torch.equal(prediction, expected))
    if not bitwise:
        raise RuntimeError(
            f"raw conditional replay is not bitwise exact at internal_t={internal_t}; "
            f"max_abs={maximum}"
        )
    return (
        np.ascontiguousarray(prediction.cpu().numpy(), dtype=np.float32),
        {"bitwise_exact": bitwise, "maximum_absolute_error": maximum},
    )


def query_projected_cross_polytopes(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    raw_replay: np.ndarray,
    internal_t: int,
    alpha_bar: float,
    basis: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query one shared center and 32 vertices per class and per radius."""

    if state.shape != (len(CLASSES), 4, 32, 32):
        raise ValueError("FPCV state batch contract changed")
    if CROSS_POLYTOPE_POINT_COUNT % QUERY_POINT_CHUNK != 0:
        raise RuntimeError("FPCV requires equal-shape cross-polytope query batches")
    reference = torch.from_numpy(state).to(device)
    class_ids = torch.tensor(CLASSES, dtype=torch.long, device=device)
    unit_coordinates = cross_polytope_coordinates()
    basis_device = torch.from_numpy(np.asarray(basis, dtype=np.float32)).to(device)
    coordinate_device = torch.from_numpy(unit_coordinates.astype(np.float32)).to(device)
    latent_directions = torch.einsum(
        "pr,rchw->pchw", coordinate_device, basis_device
    ).contiguous()
    dimension = int(np.prod(state.shape[1:]))
    rms = np.sqrt(
        np.mean(np.asarray(state, dtype=np.float64) ** 2, axis=(1, 2, 3), dtype=np.float64)
    )
    sigma = math.sqrt(max(0.0, 1.0 - alpha_bar))
    base_scale = math.sqrt(dimension) * np.maximum(rms, sigma)
    intended_radii = np.ascontiguousarray(
        base_scale[:, None] * np.asarray(RELATIVE_RADII, dtype=np.float64)[None, :],
        dtype=np.float64,
    )
    # The network state is float32, so bind the score to the exact scalar that
    # is actually multiplied into Q on device rather than an unqueried float64
    # idealization of that scalar.
    absolute_radii = np.ascontiguousarray(
        intended_radii.astype(np.float32).astype(np.float64), dtype=np.float64
    )
    projected_by_radius: list[np.ndarray] = []
    center_errors: list[np.ndarray] = []
    for radius_index in range(len(RELATIVE_RADII)):
        absolute = torch.from_numpy(
            absolute_radii[:, radius_index].astype(np.float32)
        ).to(device)
        output_chunks: list[np.ndarray] = []
        for start in range(0, CROSS_POLYTOPE_POINT_COUNT, QUERY_POINT_CHUNK):
            stop = min(start + QUERY_POINT_CHUNK, CROSS_POLYTOPE_POINT_COUNT)
            directions = latent_directions[start:stop]
            point_count = stop - start
            queries = (
                reference[:, None, :, :, :]
                + absolute[:, None, None, None, None]
                * directions[None, :, :, :, :]
            ).reshape(len(CLASSES) * point_count, 4, 32, 32)
            query_ids = (
                class_ids[:, None]
                .expand(len(CLASSES), point_count)
                .reshape(len(CLASSES) * point_count)
            )
            predictions = raw_conditional_prediction(
                diffusion,
                model,
                queries,
                internal_t=internal_t,
                class_ids=query_ids,
            ).reshape(len(CLASSES), point_count, 4, 32, 32)
            output_chunks.append(
                np.ascontiguousarray(predictions.cpu().numpy(), dtype=np.float32)
            )
        raw_outputs = np.ascontiguousarray(
            np.concatenate(output_chunks, axis=1), dtype=np.float32
        )
        if raw_outputs.shape != (
            len(CLASSES),
            CROSS_POLYTOPE_POINT_COUNT,
            4,
            32,
            32,
        ):
            raise RuntimeError("cross-polytope query output shape changed")
        projected_by_radius.append(
            np.stack(
                [project_outputs(basis, raw_outputs[slot]) for slot in range(len(CLASSES))],
                axis=0,
            )
        )
        center_errors.append(
            np.max(
                np.abs(
                    raw_outputs[:, 0].astype(np.float64)
                    - np.asarray(raw_replay, dtype=np.float64)
                ),
                axis=(1, 2, 3),
            )
        )
    return (
        absolute_radii,
        np.ascontiguousarray(np.stack(projected_by_radius, axis=1), dtype=np.float64),
        np.ascontiguousarray(np.stack(center_errors, axis=1), dtype=np.float64),
    )


def score_checkpoint(
    *,
    global_seed: int,
    checkpoint: int,
    internal_t: int,
    alpha_bar: float,
    absolute_radii: np.ndarray,
    projected_outputs: np.ndarray,
    center_errors: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if projected_outputs.shape != (
        len(CLASSES),
        len(RELATIVE_RADII),
        CROSS_POLYTOPE_POINT_COUNT,
        CROSS_POLYTOPE_DIMENSION,
    ):
        raise RuntimeError("projected cross-polytope tensor contract changed")
    rows: list[dict[str, Any]] = []
    assignments = np.empty(
        (len(CLASSES), len(RELATIVE_RADII), CROSS_POLYTOPE_POINT_COUNT),
        dtype=np.int16,
    )
    for class_slot, class_id in enumerate(CLASSES):
        for radius_index, (relative_radius, radius_role) in enumerate(
            zip(RELATIVE_RADII, RADIUS_ROLES)
        ):
            metrics = score_projected_cross_polytope(
                projected_outputs[class_slot, radius_index],
                absolute_radius=float(absolute_radii[class_slot, radius_index]),
            )
            permutation = np.asarray(metrics["optimal_permutation"], dtype=np.int16)
            assignments[class_slot, radius_index] = permutation
            rows.append(
                {
                    "global_seed": global_seed,
                    "class_slot": class_slot,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "internal_timestep": internal_t,
                    "alpha_bar": alpha_bar,
                    "radius_role": radius_role,
                    "relative_radius": relative_radius,
                    "absolute_radius": float(absolute_radii[class_slot, radius_index]),
                    "cyclic_violation": metrics["cyclic_violation"],
                    "normalization_denominator": metrics["normalization_denominator"],
                    "normalized_cyclic_violation": metrics[
                        "normalized_cyclic_violation"
                    ],
                    "identity_affinity": metrics["identity_affinity"],
                    "optimal_affinity": metrics["optimal_affinity"],
                    "centered_input_frobenius_norm": metrics[
                        "centered_input_frobenius_norm"
                    ],
                    "centered_output_frobenius_norm": metrics[
                        "centered_output_frobenius_norm"
                    ],
                    "identity_is_optimal_within_tolerance": metrics[
                        "identity_is_optimal_within_tolerance"
                    ],
                    "assignment_fixed_point_count": int(
                        np.count_nonzero(
                            permutation
                            == np.arange(CROSS_POLYTOPE_POINT_COUNT, dtype=np.int16)
                        )
                    ),
                    "center_max_abs_from_exact_replay": float(
                        center_errors[class_slot, radius_index]
                    ),
                }
            )
    return rows, assignments


def aggregate_paths(checkpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate with the frozen ``sum(V)/sum(denominator)`` rule."""

    paths: list[dict[str, Any]] = []
    for class_id in CLASSES:
        by_role: dict[str, list[Mapping[str, Any]]] = {}
        for role in RADIUS_ROLES:
            block = sorted(
                (
                    row
                    for row in checkpoint_rows
                    if int(row["class_id"]) == class_id and row["radius_role"] == role
                ),
                key=lambda row: int(row["checkpoint"]),
            )
            if [int(row["checkpoint"]) for row in block] != list(CHECKPOINTS):
                raise RuntimeError("FPCV path lacks its frozen checkpoint axis")
            by_role[role] = block
        primary = by_role["large_primary"]
        control = by_role["small_control"]

        def path_ratio(rows: Sequence[Mapping[str, Any]]) -> float:
            numerator = sum(float(row["cyclic_violation"]) for row in rows)
            denominator = sum(float(row["normalization_denominator"]) for row in rows)
            return numerator / max(denominator, 1e-30)

        paths.append(
            {
                "global_seed": int(primary[0]["global_seed"]),
                "class_id": class_id,
                "path_fpcv_large_primary": path_ratio(primary),
                "path_fpcv_small_control": path_ratio(control),
                "large_sum_cyclic_violation": sum(
                    float(row["cyclic_violation"]) for row in primary
                ),
                "large_sum_normalization_denominator": sum(
                    float(row["normalization_denominator"]) for row in primary
                ),
                "small_sum_cyclic_violation": sum(
                    float(row["cyclic_violation"]) for row in control
                ),
                "small_sum_normalization_denominator": sum(
                    float(row["normalization_denominator"]) for row in control
                ),
                "maximum_large_checkpoint_fpcv": max(
                    float(row["normalized_cyclic_violation"]) for row in primary
                ),
                "maximum_small_checkpoint_fpcv": max(
                    float(row["normalized_cyclic_violation"]) for row in control
                ),
                "maximum_radius_center_replay_gap": max(
                    float(row["center_max_abs_from_exact_replay"])
                    for row in primary + control
                ),
                **{
                    f"checkpoint_{int(row['checkpoint'])}_large_fpcv": float(
                        row["normalized_cyclic_violation"]
                    )
                    for row in primary
                },
                **{
                    f"checkpoint_{int(row['checkpoint'])}_small_fpcv": float(
                        row["normalized_cyclic_violation"]
                    )
                    for row in control
                },
            }
        )
    return paths


def run_source(
    diffusion: Any,
    model: Any,
    *,
    trace_root: Path,
    output_root: Path,
    global_seed: int,
    basis: np.ndarray,
    basis_metadata: Sequence[Mapping[str, int]],
    device: torch.device,
) -> dict[str, Any]:
    source = trace_root / f"confirmation_v1_seed{global_seed:03d}"
    manifest, arrays = validate_source_trace(source, global_seed)
    destination = output_root / f"seed{global_seed:03d}"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite seed product: {destination}")
    started = time.time()
    raw_replays: list[np.ndarray] = []
    absolute_radii_all: list[np.ndarray] = []
    projected_all: list[np.ndarray] = []
    center_errors_all: list[np.ndarray] = []
    assignments_all: list[np.ndarray] = []
    replay_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint, expected_t in zip(CHECKPOINTS, EXPECTED_INTERNAL_TIMESTEPS):
        internal_t = int(arrays["internal_timestep"][checkpoint])
        if internal_t != expected_t:
            raise RuntimeError("sampling checkpoint to internal timestep mapping changed")
        alpha_bar = float(arrays["alpha_bar"][checkpoint])
        diffusion_alpha = float(diffusion.alphas_cumprod[internal_t])
        if not math.isclose(alpha_bar, diffusion_alpha, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("saved alpha_bar does not match the implemented diffusion")
        state = np.ascontiguousarray(arrays["state_before"][:, checkpoint])
        raw_replay, replay = replay_raw_conditional(
            diffusion,
            model,
            state=state,
            recorded_epsilon=np.ascontiguousarray(
                arrays["conditional_epsilon_raw"][:, checkpoint]
            ),
            internal_t=internal_t,
            device=device,
        )
        absolute_radii, projected, center_errors = query_projected_cross_polytopes(
            diffusion,
            model,
            state=state,
            raw_replay=raw_replay,
            internal_t=internal_t,
            alpha_bar=alpha_bar,
            basis=basis,
            device=device,
        )
        rows, assignments = score_checkpoint(
            global_seed=global_seed,
            checkpoint=checkpoint,
            internal_t=internal_t,
            alpha_bar=alpha_bar,
            absolute_radii=absolute_radii,
            projected_outputs=projected,
            center_errors=center_errors,
        )
        raw_replays.append(raw_replay)
        absolute_radii_all.append(absolute_radii)
        projected_all.append(projected)
        center_errors_all.append(center_errors)
        assignments_all.append(assignments)
        replay_rows.append(
            {"checkpoint": checkpoint, "internal_timestep": internal_t, **replay}
        )
        checkpoint_rows.extend(rows)
    path_rows = aggregate_paths(checkpoint_rows)
    arrays_out = {
        "global_seed": np.asarray(global_seed, dtype=np.int64),
        "class_ids": np.asarray(CLASSES, dtype=np.int16),
        "checkpoints": np.asarray(CHECKPOINTS, dtype=np.int16),
        "internal_timesteps": np.asarray(EXPECTED_INTERNAL_TIMESTEPS, dtype=np.int16),
        "relative_radii_small_then_large": np.asarray(RELATIVE_RADII, dtype=np.float64),
        "primary_radius_index": np.asarray(1, dtype=np.int8),
        "basis": np.ascontiguousarray(basis, dtype=np.float64),
        "unit_cross_polytope_coordinates": cross_polytope_coordinates(),
        "raw_conditional_replay_pred_xstart": np.ascontiguousarray(
            np.stack(raw_replays, axis=0), dtype=np.float32
        ),
        "absolute_radii": np.ascontiguousarray(
            np.stack(absolute_radii_all, axis=0), dtype=np.float64
        ),
        "projected_outputs": np.ascontiguousarray(
            np.stack(projected_all, axis=0), dtype=np.float64
        ),
        "optimal_assignments": np.ascontiguousarray(
            np.stack(assignments_all, axis=0), dtype=np.int16
        ),
        "radius_center_max_abs_from_exact_replay": np.ascontiguousarray(
            np.stack(center_errors_all, axis=0), dtype=np.float64
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=output_root))
    try:
        atomic_npz(staging / "fpcv.npz", arrays_out)
        write_csv(staging / "checkpoint_scores.csv", checkpoint_rows)
        write_csv(staging / "path_scores.csv", path_rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": SEED_ARTIFACT_KIND,
            "status": "complete",
            "runner": RUNNER_NAME,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "fpcv_core_source_sha256": sha256_file(
                ROOT / "experiments/dit_finite_posterior_cyclic_violation.py"
            ),
            "ptcv_basis_source_sha256": sha256_file(
                ROOT / "experiments/dit_projected_tweedie_cone.py"
            ),
            "global_seed": global_seed,
            "class_ids": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
            "basis": {
                "kind": "exact_PTCV_normalized_hadamard4_tensor_dct2",
                "frequencies": [list(value) for value in DEFAULT_FREQUENCIES],
                "dimension": len(basis),
                "metadata": list(basis_metadata),
                "raw_sha256": raw_sha256(basis),
                "maximum_orthonormality_error": float(
                    np.max(
                        np.abs(
                            basis.reshape(len(basis), -1)
                            @ basis.reshape(len(basis), -1).T
                            - np.eye(len(basis))
                        )
                    )
                ),
            },
            "method": {
                "name": "finite_posterior_cyclic_violation",
                "queried_mapping": "raw class-conditional unclipped pred_xstart",
                "point_set": "center plus/minus each of 16 fixed PTCV directions",
                "point_count_per_radius": CROSS_POLYTOPE_POINT_COUNT,
                "one_shared_center_per_radius": True,
                "relative_radii_small_then_large": list(RELATIVE_RADII),
                "large_primary_absolute_scale": (
                    "sqrt(d)*max(RMS(state),sqrt(1-alpha_bar))/32"
                ),
                "small_control_absolute_scale": (
                    "sqrt(d)*max(RMS(state),sqrt(1-alpha_bar))/64"
                ),
                "primary_radius": "large_only",
                "assignment": "SciPy linear_sum_assignment maximize=True",
                "numerics": "float32 network; float64 projection, affinity, Hungarian cost, and score",
                "checkpoint_score": "V/(2*||HY||_F*||HZ||_F+1e-30)",
                "path_score": "sum_checkpoint(V)/sum_checkpoint(denominator)",
                "quality_direction": "higher means more finite cyclic violation; fixed before label join",
            },
            "npz_axis_contract": {
                "raw_conditional_replay_pred_xstart": "[checkpoint,class,C,H,W]",
                "absolute_radii": "[checkpoint,class,radius_small_then_large]",
                "projected_outputs": "[checkpoint,class,radius_small_then_large,point,basis_coordinate]",
                "optimal_assignments": "[checkpoint,class,radius_small_then_large,input_point]",
                "radius_center_max_abs_from_exact_replay": "[checkpoint,class,radius_small_then_large]",
            },
            "raw_replay": replay_rows,
            "source_trace": {
                "root": str(source),
                "identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(source / "manifest.json"),
                "trace_file_sha256": sha256_file(source / "trace.npz"),
                "arrays_read": list(SOURCE_ARRAYS),
                "raw_array_sha256": {
                    name: raw_sha256(arrays[name]) for name in SOURCE_ARRAYS
                },
            },
            "firewall": {
                "quality_labels_or_reviews_opened": False,
                "png_or_decoded_image_array_opened": False,
                "endpoint_or_final_latent_opened": False,
                "external_metric_or_embedding_opened": False,
                "source_trace_arrays_read_exactly": list(SOURCE_ARRAYS),
                "random_suffix_or_endpoint_generated": False,
                "baseline_state_changed": False,
                "cfg_prediction_used_as_metric": False,
            },
            "cost_accounting": {
                "raw_conditional_cross_polytope_sample_evaluations": (
                    len(CHECKPOINTS)
                    * len(CLASSES)
                    * len(RELATIVE_RADII)
                    * CROSS_POLYTOPE_POINT_COUNT
                ),
                "exact_replay_conditional_plus_null_sample_evaluations": (
                    len(CHECKPOINTS) * 2 * len(CLASSES)
                ),
                "model_forward_calls": (
                    len(CHECKPOINTS)
                    * (
                        len(RELATIVE_RADII)
                        * math.ceil(CROSS_POLYTOPE_POINT_COUNT / QUERY_POINT_CHUNK)
                        + 1
                    )
                ),
                "query_point_chunk": QUERY_POINT_CHUNK,
                "cross_polytope_query_batch_samples": (
                    len(CLASSES) * QUERY_POINT_CHUNK
                ),
                "all_cross_polytope_query_batches_equal_shape": True,
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("fpcv.npz", "checkpoint_scores.csv", "path_scores.csv")
            },
            "wall_seconds": time.time() - started,
        }
        record["identity_sha256"] = canonical_sha256(record)
        write_json(staging / "record.json", record)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "global_seed": global_seed,
        "output": str(destination),
        "identity_sha256": record["identity_sha256"],
        "wall_seconds": record["wall_seconds"],
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.shard_count != SHARD_COUNT:
        raise ValueError(f"fresh FPCV protocol requires exactly {SHARD_COUNT} shards")
    if not 0 <= args.shard_index < SHARD_COUNT:
        raise ValueError("shard index must lie in [0,4)")
    dit_root = args.dit_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    fpcv_core = ROOT / "experiments/dit_finite_posterior_cyclic_violation.py"
    ptcv_core = ROOT / "experiments/dit_projected_tweedie_cone.py"
    if sha256_file(Path(strict.__file__).resolve()) != EXPECTED_STRICT_SHA256:
        raise RuntimeError("strict DiT helper changed")
    if sha256_file(ptcv_core) != EXPECTED_PTCV_CORE_SHA256:
        raise RuntimeError("PTCV basis source changed")
    if sha256_file(fpcv_core) != EXPECTED_FPCV_CORE_SHA256:
        raise RuntimeError("FPCV numerical core changed")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("DiT checkpoint changed")
    repository = strict.validate_repository(dit_root, checkpoint)
    if repository.get("pinned_source_sha256", {}).get("models.py") != EXPECTED_MODELS_SHA256:
        raise RuntimeError("pinned DiT source changed")
    selected = [
        seed
        for index, seed in enumerate(SEEDS)
        if index % SHARD_COUNT == args.shard_index
    ]
    if len(selected) != 20:
        raise RuntimeError("fresh FPCV shard must contain exactly 20 seeds")
    shard = output_root / f"shard_{args.shard_index:02d}_of_{SHARD_COUNT:02d}"
    if shard.exists():
        raise RuntimeError(f"refusing to overwrite shard: {shard}")
    output_root.mkdir(parents=True, exist_ok=True)
    basis, basis_metadata = build_hadamard_dct_basis()
    if (
        basis.shape != (CROSS_POLYTOPE_DIMENSION, 4, 32, 32)
        or raw_sha256(basis) != EXPECTED_BASIS_RAW_SHA256
    ):
        raise RuntimeError("FPCV did not receive the exact frozen PTCV Q/r16 basis")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    prior_cwd = Path.cwd()
    prior_path = list(sys.path)
    prior_grad = torch.is_grad_enabled()
    preexisting = {
        name
        for name in sys.modules
        if name == "models"
        or name == "download"
        or name == "diffusion"
        or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError(f"ambiguous pre-imported DiT modules: {sorted(preexisting)}")
    started = time.time()
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from download import find_model
        from models import DiT_models

        device = torch.device("cuda")
        torch.manual_seed(20260828 + args.shard_index)
        torch.set_grad_enabled(False)
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("FPCV requires a float32 model")
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        records = []
        for ordinal, seed in enumerate(selected, start=1):
            print(
                f"FPCV shard {args.shard_index}: seed {seed} ({ordinal}/{len(selected)})",
                flush=True,
            )
            records.append(
                run_source(
                    diffusion,
                    model,
                    trace_root=trace_root,
                    output_root=shard,
                    global_seed=seed,
                    basis=basis,
                    basis_metadata=basis_metadata,
                    device=device,
                )
            )
    finally:
        torch.set_grad_enabled(prior_grad)
        os.chdir(prior_cwd)
        sys.path[:] = prior_path
        for name in list(sys.modules):
            if (
                name in {"models", "download", "diffusion"}
                or name.startswith("diffusion.")
            ) and name not in preexisting:
                sys.modules.pop(name, None)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "status": "complete",
        "runner": RUNNER_NAME,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "fpcv_core_source_sha256": sha256_file(fpcv_core),
        "ptcv_basis_source_sha256": sha256_file(ptcv_core),
        "shard_index": args.shard_index,
        "shard_count": SHARD_COUNT,
        "seeds": selected,
        "records": records,
        "method": {
            "classes": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
            "basis_dimension": len(basis),
            "basis_raw_sha256": raw_sha256(basis),
            "cross_polytope_point_count": CROSS_POLYTOPE_POINT_COUNT,
            "relative_radii_small_then_large": list(RELATIVE_RADII),
            "primary_radius": "large_only",
            "path_score": "sum(V)/sum(2*||HY||_F*||HZ||_F+epsilon)",
            "queried_mapping": "raw class-conditional unclipped pred_xstart",
            "quality_direction_selected": False,
            "status": "label_free_numerical_probe_only",
        },
        "firewall": {
            "quality_labels_or_reviews_opened": False,
            "png_or_decoded_image_array_opened": False,
            "endpoint_or_final_latent_opened": False,
            "external_metric_or_embedding_opened": False,
            "source_trace_arrays_read_exactly": list(SOURCE_ARRAYS),
            "random_suffix_or_endpoint_generated": False,
            "baseline_state_changed": False,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "cost_accounting": {
            "seeds": len(selected),
            "query_point_chunk": QUERY_POINT_CHUNK,
            "cross_polytope_query_batch_samples": len(CLASSES) * QUERY_POINT_CHUNK,
            "all_cross_polytope_query_batches_equal_shape": True,
            "model_forward_calls_per_seed": (
                len(CHECKPOINTS)
                * (
                    len(RELATIVE_RADII)
                    * math.ceil(CROSS_POLYTOPE_POINT_COUNT / QUERY_POINT_CHUNK)
                    + 1
                )
            ),
            "network_sample_evaluations_per_seed": (
                len(CHECKPOINTS)
                * len(CLASSES)
                * (
                    len(RELATIVE_RADII) * CROSS_POLYTOPE_POINT_COUNT + 2
                )
            ),
        },
        "wall_seconds": time.time() - started,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    write_json(shard / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "shard": str(shard),
                "identity_sha256": receipt["identity_sha256"],
            },
            indent=2,
        )
    )


def self_test() -> None:
    basis, metadata = build_hadamard_dct_basis()
    assert basis.shape == (16, 4, 32, 32) and len(metadata) == 16
    assert raw_sha256(basis) == EXPECTED_BASIS_RAW_SHA256
    coordinates = cross_polytope_coordinates()
    projected = np.empty(
        (
            len(CLASSES),
            len(RELATIVE_RADII),
            CROSS_POLYTOPE_POINT_COUNT,
            CROSS_POLYTOPE_DIMENSION,
        ),
        dtype=np.float64,
    )
    radii = np.empty((len(CLASSES), len(RELATIVE_RADII)), dtype=np.float64)
    for class_slot in range(len(CLASSES)):
        for radius_index, relative in enumerate(RELATIVE_RADII):
            absolute = (1.0 + class_slot) * relative
            radii[class_slot, radius_index] = absolute
            projected[class_slot, radius_index] = absolute * coordinates
    rows, assignments = score_checkpoint(
        global_seed=50,
        checkpoint=99,
        internal_t=150,
        alpha_bar=0.5,
        absolute_radii=radii,
        projected_outputs=projected,
        center_errors=np.zeros_like(radii),
    )
    assert len(rows) == len(CLASSES) * len(RELATIVE_RADII)
    assert assignments.shape == (len(CLASSES), 2, CROSS_POLYTOPE_POINT_COUNT)
    assert all(float(row["normalized_cyclic_violation"]) < 1e-14 for row in rows)
    fake: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for class_slot, class_id in enumerate(CLASSES):
            for radius_role in RADIUS_ROLES:
                fake.append(
                    {
                        "global_seed": 50,
                        "class_slot": class_slot,
                        "class_id": class_id,
                        "checkpoint": checkpoint,
                        "radius_role": radius_role,
                        "cyclic_violation": 1.0,
                        "normalization_denominator": 4.0,
                        "normalized_cyclic_violation": 0.25,
                        "center_max_abs_from_exact_replay": 0.0,
                    }
                )
    paths = aggregate_paths(fake)
    assert len(paths) == len(CLASSES)
    assert all(math.isclose(row["path_fpcv_large_primary"], 0.25) for row in paths)
    assert all(math.isclose(row["path_fpcv_small_control"], 0.25) for row in paths)

    # Exercise the class-major/chunk-major query layout on CPU.  The fake raw
    # posterior mean m(x)=x/2 is affine PSD, so every finite score is zero and
    # the one center included in each radius exactly matches the replay value.
    observed_query_batch_sizes: list[int] = []

    class FakeDiffusion:
        @staticmethod
        def p_mean_variance(
            model_forward: Any,
            states: torch.Tensor,
            timesteps: torch.Tensor,
            *,
            clip_denoised: bool,
            model_kwargs: Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            del model_forward, timesteps, clip_denoised, model_kwargs
            observed_query_batch_sizes.append(len(states))
            return {"pred_xstart": 0.5 * states}

    class FakeModel:
        @staticmethod
        def forward() -> None:
            return None

    state = np.random.default_rng(9).normal(
        size=(len(CLASSES), 4, 32, 32)
    ).astype(np.float32)
    query_radii, query_outputs, query_center_errors = query_projected_cross_polytopes(
        FakeDiffusion(),
        FakeModel(),
        state=state,
        raw_replay=0.5 * state,
        internal_t=100,
        alpha_bar=0.5,
        basis=basis,
        device=torch.device("cpu"),
    )
    query_rows, query_assignments = score_checkpoint(
        global_seed=50,
        checkpoint=149,
        internal_t=100,
        alpha_bar=0.5,
        absolute_radii=query_radii,
        projected_outputs=query_outputs,
        center_errors=query_center_errors,
    )
    assert query_outputs.shape == (3, 2, 33, 16)
    assert query_assignments.shape == (3, 2, 33)
    assert observed_query_batch_sizes == [
        len(CLASSES) * QUERY_POINT_CHUNK
    ] * (len(RELATIVE_RADII) * CROSS_POLYTOPE_POINT_COUNT // QUERY_POINT_CHUNK)
    assert np.max(query_center_errors) == 0.0
    assert all(float(row["normalized_cyclic_violation"]) == 0.0 for row in query_rows)
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=False)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = (
        args.dit_root / "pretrained_models" / strict.CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint
    )
    if args.self_test:
        self_test()
        raise SystemExit(0)
    if args.shard_index is None:
        parser.error("--shard-index is required unless --self-test is used")
    return args


if __name__ == "__main__":
    run(parse_args())

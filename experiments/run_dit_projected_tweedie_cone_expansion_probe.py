#!/usr/bin/env python3
"""Extract label-free PTCV scores for the frozen 360-path expansion pool.

The input pool contains 120 CFG trajectories with the three ordered ImageNet
classes ``[207, 602, 795]``.  This runner applies exactly the frozen PTCV
numerical construction used by ``run_dit_projected_tweedie_cone_probe.py``:

* saved CFG states at sampling checkpoints 99, 149, and 199;
* raw class-conditional, unclipped ``pred_xstart`` queries;
* the fixed 16-dimensional Hadamard-DCT basis;
* centered finite differences at relative radii 2^-9 and 2^-8;
* Richardson extrapolation; and
* the full-component block ratio, sum(dist^2)/sum(||B||_F^2), over checkpoints.

This is an observation-only extractor.  It deliberately does not open quality
labels, reviews, PNGs, decoded images, endpoint arrays, or external metrics.
Only four provenance-checked trace members are loaded: ``state_before``,
``conditional_epsilon_raw``, ``internal_timestep``, and ``alpha_bar``.
"""

from __future__ import annotations

import argparse
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
import torch

try:
    from . import reproduce_dit_imagenet256 as strict
    from . import run_dit_projected_tweedie_cone_probe as frozen_runner
    from .dit_projected_tweedie_cone import (
        DEFAULT_FREQUENCIES,
        LARGE_RELATIVE_RADIUS,
        SMALL_RELATIVE_RADIUS,
        build_hadamard_dct_basis,
        cone_metrics,
        finite_difference_stability,
        projected_matrix,
        richardson_matrix,
    )
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    import run_dit_projected_tweedie_cone_probe as frozen_runner
    from dit_projected_tweedie_cone import (
        DEFAULT_FREQUENCIES,
        LARGE_RELATIVE_RADIUS,
        SMALL_RELATIVE_RADIUS,
        build_hadamard_dct_basis,
        cone_metrics,
        finite_difference_stability,
        projected_matrix,
        richardson_matrix,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")
)
DEFAULT_DIT_ROOT = DEFAULT_DATA_ROOT / "baselines/DiT"
if not DEFAULT_DIT_ROOT.exists():
    DEFAULT_DIT_ROOT = Path("/data/users/zhoushunyu/eqvae/baselines/DiT")
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_confirmation_expansion_v1_custom_traces_cfg_locked"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_projected_tweedie_cone_expansion_eval360_probe_v1"
)

RUNNER_NAME = "run_dit_projected_tweedie_cone_expansion_probe"
ARTIFACT_KIND = "DIT_PROJECTED_TWEEDIE_CONE_EXPANSION_EVAL360_PROBE_SHARD_V1"
SEED_ARTIFACT_KIND = (
    "DIT_PROJECTED_TWEEDIE_CONE_EXPANSION_EVAL360_PROBE_SEED_V1"
)
CLASSES = (207, 602, 795)
SEEDS = tuple(range(130, 250))
CHECKPOINTS = (99, 149, 199)
EXPECTED_INTERNAL_TIMESTEPS = (150, 100, 50)
FROZEN_SHARD_COUNT = 4

PERMITTED_TRACE_ARRAYS = (
    "state_before",
    "conditional_epsilon_raw",
    "internal_timestep",
    "alpha_bar",
)
ENDPOINT_OR_IMAGE_ARRAYS = frozenset(("final_latents", "decoded_images"))
EXPECTED_ARRAY_CONTRACTS: dict[str, tuple[tuple[int, ...], np.dtype[Any], str]] = {
    "state_before": ((3, 250, 4, 32, 32), np.dtype(np.float32), "<f4"),
    "conditional_epsilon_raw": (
        (3, 250, 4, 32, 32),
        np.dtype(np.float32),
        "<f4",
    ),
    "internal_timestep": ((250,), np.dtype(np.int16), "<i2"),
    "alpha_bar": ((250,), np.dtype(np.float64), "<f8"),
}

EXPECTED_TRACE_RUNNER_SHA256 = (
    "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
)
EXPECTED_STRICT_SHA256 = (
    "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
)
EXPECTED_FROZEN_RUNNER_SHA256 = (
    "25a4c07e779fc5117225b2b0787a093ac6ddb2b81377a87fabe6459d28f27997"
)
EXPECTED_CORE_SHA256 = (
    "986f0fc8bbf22b84731ffb9b8b73bc9d73db263ae7f32d05e4ec812acf6900fe"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"
)
EXPECTED_MODELS_SHA256 = (
    "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca"
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
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_directory(trace_root: Path, global_seed: int) -> Path:
    if global_seed not in SEEDS:
        raise ValueError(f"seed is outside the frozen expansion pool: {global_seed}")
    return trace_root / f"expansion_v1_seed{global_seed}"


def destination_directory(output_root: Path, global_seed: int) -> Path:
    if global_seed not in SEEDS:
        raise ValueError(f"seed is outside the frozen expansion pool: {global_seed}")
    return output_root / f"seed{global_seed:03d}"


def _one_trace_record(manifest: Mapping[str, Any], source: Path) -> dict[str, Any]:
    records = [
        row
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and row.get("relative_path") == "trace.npz"
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one trace.npz record: {source}")
    return records[0]


def validate_source_manifest(
    source: Path, expected_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one source without opening any member of ``trace.npz``."""

    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"expected a real source directory: {source}")
    manifest_path = source / "manifest.json"
    completion_path = source / "completion.json"
    trace_path = source / "trace.npz"
    if trace_path.is_symlink() or not trace_path.is_file():
        raise RuntimeError(f"expected a real trace archive: {trace_path}")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity", {})
    protocol = identity.get("protocol", {})
    trace_record = _one_trace_record(manifest, source)

    if (
        manifest.get("status") != "complete"
        or identity.get("runner") != "trace_dit_imagenet256_custom_batch"
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or identity.get("intervention") is not None
        or identity.get("reference_baseline_dir") is not None
        or protocol.get("global_torch_seed") != expected_seed
        or tuple(protocol.get("class_ids_ordered", [])) != CLASSES
        or protocol.get("batch_size_before_duplication") != len(CLASSES)
        or protocol.get("sampler_batch_size") != 2 * len(CLASSES)
        or protocol.get("sampling_steps") != 250
        or protocol.get("internal_timestep_order") != "249..0"
        or protocol.get("cfg_scale") != 4.0
        or protocol.get("clip_denoised") is not False
        or protocol.get("trace_axis_order") != "[B, sampling_step, C, H, W]"
        or identity.get("runner_source", {}).get("sha256")
        != EXPECTED_TRACE_RUNNER_SHA256
        or identity.get("strict_reproduction_helper", {}).get("sha256")
        != EXPECTED_STRICT_SHA256
        or identity.get("checkpoint", {}).get("sha256")
        != EXPECTED_CHECKPOINT_SHA256
        or identity.get("source", {})
        .get("pinned_source_sha256", {})
        .get("models.py")
        != EXPECTED_MODELS_SHA256
        or manifest.get("identity_sha256") != canonical_sha256(identity)
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or trace_record.get("bytes") != trace_path.stat().st_size
        or not isinstance(trace_record.get("sha256"), str)
        or len(trace_record["sha256"]) != 64
    ):
        raise RuntimeError(f"source manifest validation failed: {source}")

    manifest_records = manifest.get("trace_array_records", {})
    output_records = trace_record.get("arrays", {})
    identity_records = identity.get("trace_arrays", {})
    permitted_hashes: dict[str, str] = {}
    for name in PERMITTED_TRACE_ARRAYS:
        shape, _, dtype_string = EXPECTED_ARRAY_CONTRACTS[name]
        expected_metadata = {
            "dtype": dtype_string,
            "shape": list(shape),
        }
        manifest_record = manifest_records.get(name, {})
        output_record = output_records.get(name, {})
        identity_record = identity_records.get(name, {})
        if (
            {key: manifest_record.get(key) for key in ("dtype", "shape")}
            != expected_metadata
            or {key: output_record.get(key) for key in ("dtype", "shape")}
            != expected_metadata
            or identity_record != expected_metadata
            or output_record != manifest_record
            or not isinstance(manifest_record.get("raw_sha256"), str)
            or len(manifest_record["raw_sha256"]) != 64
        ):
            raise RuntimeError(
                f"source manifest array contract failed for {name}: {source}"
            )
        permitted_hashes[name] = manifest_record["raw_sha256"]

    summary = {
        "global_seed": expected_seed,
        "source_directory": str(source),
        "identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": completion["manifest_sha256"],
        "trace_file_bytes": int(trace_record["bytes"]),
        "trace_file_sha256_recorded": trace_record["sha256"],
        "permitted_array_raw_sha256": permitted_hashes,
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    return manifest, summary


def preflight_source_pool(trace_root: Path) -> dict[str, Any]:
    """Seal all 120 source manifests without opening trace members or images."""

    if trace_root.is_symlink() or not trace_root.is_dir():
        raise RuntimeError(f"expected a real expansion trace root: {trace_root}")
    expected_names = {f"expansion_v1_seed{seed}" for seed in SEEDS}
    observed_names = {
        path.name
        for path in trace_root.iterdir()
        if path.name.startswith("expansion_v1_seed") and path.is_dir()
    }
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise RuntimeError(
            f"expansion source directory set changed; missing={missing}, extra={extra}"
        )
    records = []
    for seed in SEEDS:
        _, summary = validate_source_manifest(source_directory(trace_root, seed), seed)
        records.append(summary)
    contract = {
        "schema_version": 1,
        "kind": "DIT_EXPANSION_EVAL360_SOURCE_MANIFEST_CONTRACT_V1",
        "trace_root": str(trace_root),
        "seed_count": len(SEEDS),
        "path_count": len(SEEDS) * len(CLASSES),
        "seeds": list(SEEDS),
        "classes": list(CLASSES),
        "records_sha256": canonical_sha256(records),
        "trace_members_opened": [],
        "quality_labels_reviews_pngs_endpoints_external_metrics_opened": False,
    }
    contract["identity_sha256"] = canonical_sha256(contract)
    return contract


def load_permitted_trace_arrays(
    source: Path, manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Load and hash only the four allow-listed trace arrays."""

    trace_path = source / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as archive:
        missing = [name for name in PERMITTED_TRACE_ARRAYS if name not in archive.files]
        if missing:
            raise RuntimeError(f"source trace arrays missing {missing}: {source}")
        arrays = {
            name: np.ascontiguousarray(archive[name]) for name in PERMITTED_TRACE_ARRAYS
        }

    records = manifest.get("trace_array_records", {})
    for name, array in arrays.items():
        shape, dtype, _ = EXPECTED_ARRAY_CONTRACTS[name]
        if (
            array.shape != shape
            or array.dtype != dtype
            or not np.isfinite(array).all()
            or records.get(name, {}).get("raw_sha256") != raw_sha256(array)
        ):
            raise RuntimeError(f"source trace tensor contract failed for {name}: {source}")
    if (
        not np.array_equal(
            arrays["internal_timestep"], np.arange(249, -1, -1, dtype=np.int16)
        )
        or np.any(arrays["alpha_bar"] <= 0.0)
        or np.any(arrays["alpha_bar"] > 1.0)
    ):
        raise RuntimeError(f"source schedule contract changed: {source}")
    return arrays


def raw_conditional_prediction(
    diffusion: Any,
    model: Any,
    states: torch.Tensor,
    *,
    internal_t: int,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Delegate the unchanged raw, unclipped query to the frozen runner."""

    return frozen_runner.raw_conditional_prediction(
        diffusion,
        model,
        states,
        internal_t=internal_t,
        labels=labels,
    )


def replay_raw_conditional(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    recorded_epsilon: np.ndarray,
    internal_t: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference = torch.from_numpy(state).to(device)
    labels = torch.tensor(CLASSES, dtype=torch.long, device=device)
    null = torch.full_like(labels, strict.NULL_CLASS_ID)
    full = torch.cat([reference, reference], dim=0)
    full_labels = torch.cat([labels, null], dim=0)
    prediction = raw_conditional_prediction(
        diffusion, model, full, internal_t=internal_t, labels=full_labels
    )[: len(CLASSES)]
    timestep = torch.full(
        (len(CLASSES),), internal_t, dtype=torch.long, device=device
    )
    expected = diffusion._predict_xstart_from_eps(
        x_t=reference,
        t=timestep,
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


def directional_matrices(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    internal_t: int,
    alpha_bar: float,
    basis: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = torch.from_numpy(state).to(device)
    labels = torch.tensor(CLASSES, dtype=torch.long, device=device)
    basis_device = torch.from_numpy(basis.astype(np.float32)).to(device)
    dimension = int(np.prod(state.shape[1:]))
    rms = torch.sqrt(torch.mean(reference.float() ** 2, dim=(1, 2, 3)))
    noise_sigma = math.sqrt(max(0.0, 1.0 - alpha_bar))
    scale = math.sqrt(dimension) * torch.maximum(
        rms, torch.full_like(rms, noise_sigma)
    )
    matrices = []
    radii = []
    for relative in (SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS):
        absolute = relative * scale
        radii.append(
            np.ascontiguousarray(absolute.cpu().numpy(), dtype=np.float64)
        )
        derivatives: list[np.ndarray] = []
        for direction in basis_device:
            perturbation = absolute[:, None, None, None] * direction[None, :, :, :]
            full = torch.cat([reference + perturbation, reference - perturbation], dim=0)
            full_labels = torch.cat([labels, labels], dim=0)
            prediction = raw_conditional_prediction(
                diffusion, model, full, internal_t=internal_t, labels=full_labels
            )
            plus, minus = prediction.chunk(2, dim=0)
            derivative = (plus - minus) / (
                2.0 * absolute[:, None, None, None]
            )
            derivatives.append(
                np.ascontiguousarray(derivative.cpu().numpy(), dtype=np.float64)
            )
        stacked = np.stack(derivatives, axis=1)
        matrices.append(
            np.stack(
                [
                    projected_matrix(basis, stacked[slot])
                    for slot in range(len(CLASSES))
                ],
                axis=0,
            )
        )
    small, large = matrices
    richardson = np.stack(
        [
            richardson_matrix(small[slot], large[slot])
            for slot in range(len(CLASSES))
        ],
        axis=0,
    )
    return (
        np.ascontiguousarray(np.stack(radii, axis=-1), dtype=np.float64),
        np.ascontiguousarray(np.stack([small, large], axis=1), dtype=np.float64),
        np.ascontiguousarray(richardson, dtype=np.float64),
    )


def score_checkpoint(
    *,
    global_seed: int,
    checkpoint: int,
    internal_t: int,
    alpha_bar: float,
    matrices_by_radius: np.ndarray,
    richardson: np.ndarray,
    radii: np.ndarray,
) -> list[dict[str, Any]]:
    if matrices_by_radius.shape[0:2] != (len(CLASSES), 2):
        raise RuntimeError("finite-difference matrix axis changed")
    rows: list[dict[str, Any]] = []
    for slot, class_id in enumerate(CLASSES):
        small = matrices_by_radius[slot, 0]
        large = matrices_by_radius[slot, 1]
        final = richardson[slot]
        small_metrics = cone_metrics(small)
        large_metrics = cone_metrics(large)
        final_metrics = cone_metrics(final)
        stability = finite_difference_stability(small, large, final)
        rows.append(
            {
                "global_seed": global_seed,
                "class_slot": slot,
                "class_id": class_id,
                "checkpoint": checkpoint,
                "internal_timestep": internal_t,
                "alpha_bar": alpha_bar,
                "small_absolute_radius": float(radii[slot, 0]),
                "large_absolute_radius": float(radii[slot, 1]),
                "cone_distance_squared": final_metrics["cone_distance_squared"],
                "matrix_energy": final_metrics["matrix_energy"],
                "normalized_cone_violation": final_metrics[
                    "normalized_cone_violation"
                ],
                "skew_energy": final_metrics["skew_energy"],
                "negative_eigen_energy": final_metrics["negative_eigen_energy"],
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
                "negative_eigenvalue_count": final_metrics[
                    "negative_eigenvalue_count"
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
                "small_radius_minimum_secant": float(np.min(np.diag(small))),
                "large_radius_minimum_secant": float(np.min(np.diag(large))),
                "small_radius_negative_secant_count": int(
                    np.count_nonzero(np.diag(small) < 0.0)
                ),
                "large_radius_negative_secant_count": int(
                    np.count_nonzero(np.diag(large) < 0.0)
                ),
                **stability,
            }
        )
    return rows


def aggregate_paths(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for class_id in CLASSES:
        block = sorted(
            (row for row in checkpoint_rows if int(row["class_id"]) == class_id),
            key=lambda row: int(row["checkpoint"]),
        )
        if [int(row["checkpoint"]) for row in block] != list(CHECKPOINTS):
            raise RuntimeError("path lacks the frozen checkpoint axis")
        distance = sum(float(row["cone_distance_squared"]) for row in block)
        energy = sum(float(row["matrix_energy"]) for row in block)
        skew = sum(float(row["skew_energy"]) for row in block)
        negative = sum(float(row["negative_eigen_energy"]) for row in block)
        paths.append(
            {
                "global_seed": int(block[0]["global_seed"]),
                "class_id": class_id,
                "path_cone_violation": distance / max(energy, 1e-30),
                "path_skew_fraction": skew / max(energy, 1e-30),
                "path_negative_eigen_fraction": negative / max(energy, 1e-30),
                "path_matrix_energy": energy,
                "maximum_checkpoint_cone_violation": max(
                    float(row["normalized_cone_violation"]) for row in block
                ),
                "maximum_small_large_relative_gap": max(
                    float(row["difference_over_richardson_norm"]) for row in block
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
                        row["normalized_cone_violation"]
                    )
                    for row in block
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
    source = source_directory(trace_root, global_seed)
    manifest, source_summary = validate_source_manifest(source, global_seed)
    arrays = load_permitted_trace_arrays(source, manifest)
    destination = destination_directory(output_root, global_seed)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite seed product: {destination}")

    started = time.time()
    raw_predictions = []
    absolute_radii = []
    radius_matrices = []
    richardson_matrices = []
    replay_rows = []
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint, expected_t in zip(CHECKPOINTS, EXPECTED_INTERNAL_TIMESTEPS):
        internal_t = int(arrays["internal_timestep"][checkpoint])
        if internal_t != expected_t:
            raise RuntimeError("sampling checkpoint to internal timestep mapping changed")
        alpha_bar = float(arrays["alpha_bar"][checkpoint])
        diffusion_alpha = float(diffusion.alphas_cumprod[internal_t])
        if not math.isclose(
            alpha_bar, diffusion_alpha, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError("saved alpha_bar does not match implemented diffusion")
        current_state = np.ascontiguousarray(arrays["state_before"][:, checkpoint])
        raw_prediction, replay = replay_raw_conditional(
            diffusion,
            model,
            state=current_state,
            recorded_epsilon=np.ascontiguousarray(
                arrays["conditional_epsilon_raw"][:, checkpoint]
            ),
            internal_t=internal_t,
            device=device,
        )
        radii, matrices, richardson = directional_matrices(
            diffusion,
            model,
            state=current_state,
            internal_t=internal_t,
            alpha_bar=alpha_bar,
            basis=basis,
            device=device,
        )
        raw_predictions.append(raw_prediction)
        absolute_radii.append(radii)
        radius_matrices.append(matrices)
        richardson_matrices.append(richardson)
        replay_rows.append(
            {"checkpoint": checkpoint, "internal_timestep": internal_t, **replay}
        )
        checkpoint_rows.extend(
            score_checkpoint(
                global_seed=global_seed,
                checkpoint=checkpoint,
                internal_t=internal_t,
                alpha_bar=alpha_bar,
                matrices_by_radius=matrices,
                richardson=richardson,
                radii=radii,
            )
        )
    path_rows = aggregate_paths(checkpoint_rows)
    arrays_out = {
        "global_seed": np.asarray(global_seed, dtype=np.int64),
        "class_ids": np.asarray(CLASSES, dtype=np.int16),
        "checkpoints": np.asarray(CHECKPOINTS, dtype=np.int16),
        "internal_timesteps": np.asarray(
            EXPECTED_INTERNAL_TIMESTEPS, dtype=np.int16
        ),
        "relative_radii": np.asarray(
            (SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS), dtype=np.float64
        ),
        "basis": np.ascontiguousarray(basis, dtype=np.float64),
        "raw_conditional_pred_xstart": np.ascontiguousarray(
            np.stack(raw_predictions, axis=0), dtype=np.float32
        ),
        "absolute_radii": np.ascontiguousarray(
            np.stack(absolute_radii, axis=0), dtype=np.float64
        ),
        "projected_matrices_by_radius": np.ascontiguousarray(
            np.stack(radius_matrices, axis=0), dtype=np.float64
        ),
        "richardson_projected_matrices": np.ascontiguousarray(
            np.stack(richardson_matrices, axis=0), dtype=np.float64
        ),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=output_root)
    )
    try:
        frozen_runner.atomic_npz(staging / "ptcv.npz", arrays_out)
        frozen_runner.write_csv(staging / "checkpoint_scores.csv", checkpoint_rows)
        frozen_runner.write_csv(staging / "path_scores.csv", path_rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": SEED_ARTIFACT_KIND,
            "status": "complete",
            "runner": RUNNER_NAME,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_runner_source_sha256": sha256_file(
                Path(frozen_runner.__file__).resolve()
            ),
            "core_source_sha256": sha256_file(
                ROOT / "experiments/dit_projected_tweedie_cone.py"
            ),
            "global_seed": global_seed,
            "class_ids": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
            "path_score": {
                "name": "full_component_block_cone_ratio",
                "formula": "sum_k dist_F_squared(B_k,S_+)/sum_k frobenius_squared(B_k)",
                "checkpoint_weighting": "matrix-energy weighted",
                "uses_all_projected_matrix_components": True,
            },
            "basis": {
                "kind": "normalized_hadamard4_tensor_dct2",
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
            "finite_difference": {
                "kind": "centered_two_radius_richardson",
                "relative_l2_radii": [
                    SMALL_RELATIVE_RADIUS,
                    LARGE_RELATIVE_RADIUS,
                ],
                "absolute_scale": "sqrt(d)*max(RMS(state),sqrt(1-alpha_bar))",
                "richardson": "(4*B_small-B_large)/3",
                "dtype": (
                    "float32 network query; float64 projection and eigendecomposition"
                ),
            },
            "raw_replay": replay_rows,
            "source_trace": {
                **source_summary,
                "arrays_loaded": list(PERMITTED_TRACE_ARRAYS),
                "array_members_explicitly_not_loaded": sorted(
                    set(manifest.get("trace_array_records", {}))
                    - set(PERMITTED_TRACE_ARRAYS)
                ),
                "loaded_array_raw_sha256": {
                    name: raw_sha256(arrays[name]) for name in PERMITTED_TRACE_ARRAYS
                },
                "whole_trace_file_sha256_recomputed": False,
            },
            "firewall": {
                "quality_labels_or_reviews_opened": False,
                "png_files_opened": False,
                "decoded_image_array_loaded": False,
                "endpoint_array_loaded": False,
                "external_metric_or_embedding_opened": False,
                "random_suffix_or_endpoint_generated": False,
                "baseline_state_changed": False,
                "quality_direction_selected_by_runner": False,
                "trace_array_allowlist_enforced": list(PERMITTED_TRACE_ARRAYS),
                "queried_mapping": "raw class-conditional unclipped pred_xstart",
                "cfg_prediction_used_as_metric": False,
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in (
                    "ptcv.npz",
                    "checkpoint_scores.csv",
                    "path_scores.csv",
                )
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


def validate_frozen_dependencies(checkpoint: Path) -> dict[str, str]:
    paths = {
        "strict": Path(strict.__file__).resolve(),
        "frozen_runner": Path(frozen_runner.__file__).resolve(),
        "core": ROOT / "experiments/dit_projected_tweedie_cone.py",
        "checkpoint": checkpoint,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "strict": EXPECTED_STRICT_SHA256,
        "frozen_runner": EXPECTED_FROZEN_RUNNER_SHA256,
        "core": EXPECTED_CORE_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"frozen PTCV dependency changed; expected={expected}, observed={observed}"
        )
    return observed


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < FROZEN_SHARD_COUNT:
        raise ValueError("shard index must lie in [0, 4)")
    if args.shard_count != FROZEN_SHARD_COUNT:
        raise ValueError("expansion_eval360 extraction is frozen to exactly four shards")

    dit_root = args.dit_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dependency_hashes = validate_frozen_dependencies(checkpoint)
    source_contract = preflight_source_pool(trace_root)
    source = strict.validate_repository(dit_root, checkpoint)
    if (
        source.get("pinned_source_sha256", {}).get("models.py")
        != EXPECTED_MODELS_SHA256
    ):
        raise RuntimeError("pinned DiT source changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    selected = [
        seed
        for index, seed in enumerate(SEEDS)
        if index % FROZEN_SHARD_COUNT == args.shard_index
    ]
    if len(selected) != len(SEEDS) // FROZEN_SHARD_COUNT:
        raise RuntimeError("four-shard partition no longer has 30 seeds per shard")
    shard = output_root / (
        f"shard_{args.shard_index:02d}_of_{FROZEN_SHARD_COUNT:02d}"
    )
    if shard.exists():
        raise RuntimeError(f"refusing to overwrite shard: {shard}")
    output_root.mkdir(parents=True, exist_ok=True)
    basis, basis_metadata = build_hadamard_dct_basis()

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
    device = torch.device("cuda")
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from download import find_model
        from models import DiT_models

        torch.manual_seed(20260828 + args.shard_index)
        torch.set_grad_enabled(False)
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("PTCV requires a float32 model")
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        records = []
        for ordinal, seed in enumerate(selected, start=1):
            print(
                f"PTCV expansion shard {args.shard_index}: "
                f"seed {seed} ({ordinal}/{len(selected)})",
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
        "dependency_sha256": dependency_hashes,
        "shard_index": args.shard_index,
        "shard_count": FROZEN_SHARD_COUNT,
        "seeds": selected,
        "path_count": len(selected) * len(CLASSES),
        "records": records,
        "source_pool_contract": source_contract,
        "method": {
            "classes": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
            "basis_dimension": len(basis),
            "basis_raw_sha256": raw_sha256(basis),
            "frequencies": [list(value) for value in DEFAULT_FREQUENCIES],
            "relative_radii": [SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS],
            "primary_matrix": "Richardson (4*B_small-B_large)/3",
            "path_score": (
                "sum checkpoint full cone-distance-squared / "
                "sum checkpoint matrix-energy"
            ),
            "queried_mapping": "raw class-conditional unclipped pred_xstart",
            "quality_direction_selected": False,
            "status": "label_free_numerical_probe_only",
        },
        "firewall": {
            "quality_labels_or_reviews_opened": False,
            "png_files_opened": False,
            "decoded_image_array_loaded": False,
            "endpoint_array_loaded": False,
            "external_metric_or_embedding_opened": False,
            "random_suffix_or_endpoint_generated": False,
            "baseline_state_changed": False,
            "trace_array_allowlist_enforced": list(PERMITTED_TRACE_ARRAYS),
            "whole_trace_file_hash_not_recomputed_to_avoid_endpoint_bytes": True,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
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
    assert not (ENDPOINT_OR_IMAGE_ARRAYS & set(PERMITTED_TRACE_ARRAYS))
    assert source_directory(Path("/trace"), 130) == Path(
        "/trace/expansion_v1_seed130"
    )
    assert destination_directory(Path("/out"), 249) == Path("/out/seed249")

    partitions = [
        [
            seed
            for index, seed in enumerate(SEEDS)
            if index % FROZEN_SHARD_COUNT == shard
        ]
        for shard in range(FROZEN_SHARD_COUNT)
    ]
    assert all(len(partition) == 30 for partition in partitions)
    assert sorted(seed for partition in partitions for seed in partition) == list(SEEDS)

    fake = []
    for checkpoint in CHECKPOINTS:
        for slot, class_id in enumerate(CLASSES):
            fake.append(
                {
                    "global_seed": 130,
                    "class_slot": slot,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "cone_distance_squared": 1.0,
                    "matrix_energy": 4.0,
                    "skew_energy": 0.25,
                    "negative_eigen_energy": 0.75,
                    "normalized_cone_violation": 0.25,
                    "difference_over_richardson_norm": 0.1,
                    "small_radius_minimum_secant": 0.2,
                    "large_radius_minimum_secant": 0.1,
                }
            )
    paths = aggregate_paths(fake)
    assert len(paths) == len(CLASSES)
    assert all(math.isclose(row["path_cone_violation"], 0.25) for row in paths)
    assert all(math.isclose(row["path_skew_fraction"], 0.0625) for row in paths)
    assert all(
        math.isclose(row["path_negative_eigen_fraction"], 0.1875)
        for row in paths
    )
    print("self-test passed")


def print_preflight(trace_root: Path) -> None:
    contract = preflight_source_pool(trace_root.expanduser().resolve())
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=False)
    parser.add_argument("--shard-count", type=int, default=FROZEN_SHARD_COUNT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = (
        args.dit_root / "pretrained_models" / strict.CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint
    )
    if args.self_test and args.preflight_only:
        parser.error("--self-test and --preflight-only are mutually exclusive")
    if args.self_test:
        self_test()
        raise SystemExit(0)
    if args.preflight_only:
        print_preflight(args.trace_root)
        raise SystemExit(0)
    if args.shard_index is None:
        parser.error(
            "--shard-index is required unless --self-test or --preflight-only is used"
        )
    return args


if __name__ == "__main__":
    run(parse_args())

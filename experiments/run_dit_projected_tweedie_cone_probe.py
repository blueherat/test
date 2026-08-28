#!/usr/bin/env python3
"""Run label-free single-path Projected Tweedie-cone probes on DiT traces.

At three frozen trajectory checkpoints this runner queries the *raw class-
conditional*, unclipped clean-latent predictor around the saved CFG trajectory
state.  Centered finite differences in a fixed Hadamard-DCT subspace estimate
``B = Q^T J Q``.  The output records its exact distance to the symmetric PSD
cone, two-radius numerical stability, and no endpoint image or quality label.
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
import torch

try:
    from . import reproduce_dit_imagenet256 as strict
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
DEFAULT_DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
DEFAULT_DIT_ROOT = DEFAULT_DATA_ROOT / "baselines/DiT"
if not DEFAULT_DIT_ROOT.exists():
    DEFAULT_DIT_ROOT = Path("/data/users/zhoushunyu/eqvae/baselines/DiT")
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_custom_traces_cfg_locked"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_projected_tweedie_cone_probe_v1"
)
RUNNER_NAME = "run_dit_projected_tweedie_cone_probe"
ARTIFACT_KIND = "DIT_PROJECTED_TWEEDIE_CONE_PROBE_SHARD_V1"
CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
SEEDS = tuple(range(10, 30))
CHECKPOINTS = (99, 149, 199)
EXPECTED_INTERNAL_TIMESTEPS = (150, 100, 50)
EXPECTED_TRACE_RUNNER_SHA256 = "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
EXPECTED_STRICT_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
EXPECTED_CORE_SHA256 = "986f0fc8bbf22b84731ffb9b8b73bc9d73db263ae7f32d05e4ec812acf6900fe"
EXPECTED_CHECKPOINT_SHA256 = "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"


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
        raise RuntimeError(f"expected a JSON object: {path}")
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
    manifest_path = source / "manifest.json"
    completion_path = source / "completion.json"
    trace_path = source / "trace.npz"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity", {})
    protocol = identity.get("protocol", {})
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
        or protocol.get("sampling_steps") != 250
        or protocol.get("cfg_scale") != 4.0
        or protocol.get("clip_denoised") is not False
        or identity.get("runner_source", {}).get("sha256")
        != EXPECTED_TRACE_RUNNER_SHA256
        or identity.get("strict_reproduction_helper", {}).get("sha256")
        != EXPECTED_STRICT_SHA256
        or identity.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or not isinstance(trace_record, dict)
        or trace_record.get("sha256") != sha256_file(trace_path)
    ):
        raise RuntimeError(f"source trace validation failed: {source}")

    required = (
        "state_before",
        "conditional_epsilon_raw",
        "internal_timestep",
        "alpha_bar",
    )
    with np.load(trace_path, allow_pickle=False) as archive:
        if any(name not in archive.files for name in required):
            raise RuntimeError(f"source trace arrays missing: {source}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    if (
        arrays["state_before"].shape != (8, 250, 4, 32, 32)
        or arrays["conditional_epsilon_raw"].shape != (8, 250, 4, 32, 32)
        or arrays["internal_timestep"].shape != (250,)
        or arrays["alpha_bar"].shape != (250,)
        or arrays["state_before"].dtype != np.float32
        or arrays["conditional_epsilon_raw"].dtype != np.float32
        or arrays["internal_timestep"].dtype != np.int16
        or arrays["alpha_bar"].dtype != np.float64
        or not np.array_equal(
            arrays["internal_timestep"], np.arange(249, -1, -1, dtype=np.int16)
        )
        or not np.isfinite(arrays["state_before"]).all()
        or not np.isfinite(arrays["conditional_epsilon_raw"]).all()
        or not np.isfinite(arrays["alpha_bar"]).all()
        or np.any(arrays["alpha_bar"] <= 0.0)
        or np.any(arrays["alpha_bar"] > 1.0)
    ):
        raise RuntimeError(f"source trace tensor contract changed: {source}")
    records = manifest.get("trace_array_records", {})
    for name in required:
        if records.get(name, {}).get("raw_sha256") != raw_sha256(arrays[name]):
            raise RuntimeError(f"source raw-array hash failed for {name}: {source}")
    return manifest, arrays


def raw_conditional_prediction(
    diffusion: Any,
    model: Any,
    states: torch.Tensor,
    *,
    internal_t: int,
    labels: torch.Tensor,
) -> torch.Tensor:
    if states.ndim != 4 or labels.shape != (len(states),):
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
            model_kwargs={"y": labels},
        )
    prediction = output["pred_xstart"].contiguous()
    if prediction.shape != states.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("raw conditional prediction is invalid")
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
    relative_radii = (SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS)
    matrices = []
    radii = []
    for relative in relative_radii:
        absolute = relative * scale
        radii.append(np.ascontiguousarray(absolute.cpu().numpy(), dtype=np.float64))
        derivatives: list[np.ndarray] = []
        for direction in basis_device:
            perturbation = absolute[:, None, None, None] * direction[None, :, :, :]
            full = torch.cat([reference + perturbation, reference - perturbation], dim=0)
            full_labels = torch.cat([labels, labels], dim=0)
            prediction = raw_conditional_prediction(
                diffusion, model, full, internal_t=internal_t, labels=full_labels
            )
            plus, minus = prediction.chunk(2, dim=0)
            derivative = (plus - minus) / (2.0 * absolute[:, None, None, None])
            derivatives.append(
                np.ascontiguousarray(derivative.cpu().numpy(), dtype=np.float64)
            )
        stacked = np.stack(derivatives, axis=1)  # [class, direction, C, H, W]
        matrices.append(
            np.stack(
                [projected_matrix(basis, stacked[slot]) for slot in range(len(CLASSES))],
                axis=0,
            )
        )
    small, large = matrices
    richardson = np.stack(
        [richardson_matrix(small[slot], large[slot]) for slot in range(len(CLASSES))],
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


def aggregate_paths(checkpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    source = trace_root / f"targeted_scan_v1_seed{global_seed}"
    manifest, arrays = validate_source_trace(source, global_seed)
    destination = output_root / f"seed{global_seed:02d}"
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
        if not math.isclose(alpha_bar, diffusion_alpha, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("saved alpha_bar does not match the implemented diffusion")
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
        replay_rows.append({"checkpoint": checkpoint, "internal_timestep": internal_t, **replay})
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
        "internal_timesteps": np.asarray(EXPECTED_INTERNAL_TIMESTEPS, dtype=np.int16),
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
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=output_root))
    try:
        atomic_npz(staging / "ptcv.npz", arrays_out)
        write_csv(staging / "checkpoint_scores.csv", checkpoint_rows)
        write_csv(staging / "path_scores.csv", path_rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_PROJECTED_TWEEDIE_CONE_PROBE_SEED_V1",
            "status": "complete",
            "runner": RUNNER_NAME,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "core_source_sha256": sha256_file(
                ROOT / "experiments/dit_projected_tweedie_cone.py"
            ),
            "global_seed": global_seed,
            "class_ids": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
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
                "relative_l2_radii": [SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS],
                "absolute_scale": "sqrt(d)*max(RMS(state),sqrt(1-alpha_bar))",
                "richardson": "(4*B_small-B_large)/3",
                "dtype": "float32 network query; float64 projection and eigendecomposition",
            },
            "raw_replay": replay_rows,
            "source_trace": {
                "root": str(source),
                "identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(source / "manifest.json"),
                "trace_file_sha256": sha256_file(source / "trace.npz"),
                "arrays_read": [
                    "state_before",
                    "conditional_epsilon_raw",
                    "internal_timestep",
                    "alpha_bar",
                ],
                "raw_array_sha256": {
                    name: raw_sha256(arrays[name])
                    for name in (
                        "state_before",
                        "conditional_epsilon_raw",
                        "internal_timestep",
                        "alpha_bar",
                    )
                },
            },
            "firewall": {
                "labels_reviews_pngs_decoded_images_opened": False,
                "external_metric_or_embedding_opened": False,
                "random_suffix_or_endpoint_generated": False,
                "baseline_state_changed": False,
                "quality_direction_selected_by_runner": False,
                "queried_mapping": "raw class-conditional unclipped pred_xstart",
                "cfg_prediction_used_as_metric": False,
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("ptcv.npz", "checkpoint_scores.csv", "path_scores.csv")
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
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    dit_root = args.dit_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    core_path = ROOT / "experiments/dit_projected_tweedie_cone.py"
    if sha256_file(Path(strict.__file__).resolve()) != EXPECTED_STRICT_SHA256:
        raise RuntimeError("strict DiT helper changed")
    if sha256_file(core_path) != EXPECTED_CORE_SHA256:
        raise RuntimeError("PTCV numerical core changed")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("DiT checkpoint changed")
    source = strict.validate_repository(dit_root, checkpoint)
    if source.get("pinned_source_sha256", {}).get("models.py") != (
        "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca"
    ):
        raise RuntimeError("pinned DiT source changed")
    selected = [
        seed
        for index, seed in enumerate(SEEDS)
        if index % args.shard_count == args.shard_index
    ]
    if not selected:
        raise RuntimeError("selected shard has no seeds")
    shard = output_root / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
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
        if name == "models" or name == "download" or name == "diffusion" or name.startswith("diffusion.")
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
            raise RuntimeError("PTCV requires a float32 model")
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        records = []
        for ordinal, seed in enumerate(selected, start=1):
            print(
                f"PTCV shard {args.shard_index}: seed {seed} ({ordinal}/{len(selected)})",
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
        "core_source_sha256": sha256_file(core_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "seeds": selected,
        "records": records,
        "method": {
            "classes": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "internal_timesteps": list(EXPECTED_INTERNAL_TIMESTEPS),
            "basis_dimension": len(basis),
            "basis_raw_sha256": raw_sha256(basis),
            "frequencies": [list(value) for value in DEFAULT_FREQUENCIES],
            "relative_radii": [SMALL_RELATIVE_RADIUS, LARGE_RELATIVE_RADIUS],
            "primary_matrix": "Richardson (4*B_small-B_large)/3",
            "queried_mapping": "raw class-conditional unclipped pred_xstart",
            "quality_direction_selected": False,
            "status": "label_free_numerical_probe_only",
        },
        "firewall": {
            "labels_reviews_pngs_decoded_images_opened": False,
            "external_metric_or_embedding_opened": False,
            "random_suffix_or_endpoint_generated": False,
            "baseline_state_changed": False,
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
    fake = []
    for checkpoint in CHECKPOINTS:
        for slot, class_id in enumerate(CLASSES):
            fake.append(
                {
                    "global_seed": 10,
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
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=False)
    parser.add_argument("--shard-count", type=int, default=4)
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

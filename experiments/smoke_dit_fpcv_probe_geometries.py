#!/usr/bin/env python3
"""Label-free geometry screen for finite posterior cyclic monotonicity.

This is deliberately separate from the frozen FPCV core and runner.  It reads
only the four source-trace arrays already approved by the FPCV firewall and
never opens an endpoint, PNG, decoded image, embedding, review, or quality
label.

The original 33-point cross-polytope is a coarse point set.  A strongly
monotone radial component can give its identity assignment a positive margin
even when the projected map has a small rotational (skew-Jacobian) component.
This smoke test compares it with predeclared dense closed polygons, translated
polygons, and fixed protocol-seeded point clouds.  Every point set retains the
same exact null: for a true Gaussian posterior mean, identity matching must be
optimal and the cyclic violation is zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import torch
from scipy.linalg import hadamard

try:
    from . import reproduce_dit_imagenet256 as strict
    from .dit_finite_posterior_cyclic_violation import finite_cyclic_metrics
    from .dit_projected_tweedie_cone import build_hadamard_dct_basis
    from .run_dit_fpcv_fresh_probe import (
        CHECKPOINTS,
        CLASSES,
        DEFAULT_DATA_ROOT,
        DEFAULT_DIT_ROOT,
        DEFAULT_TRACE_ROOT,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_INTERNAL_TIMESTEPS,
        EXPECTED_MODELS_SHA256,
        EXPECTED_STRICT_SHA256,
        SOURCE_ARRAYS,
        canonical_sha256,
        project_outputs,
        raw_conditional_prediction,
        raw_sha256,
        replay_raw_conditional,
        sha256_file,
        validate_source_trace,
    )
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    from dit_finite_posterior_cyclic_violation import finite_cyclic_metrics
    from dit_projected_tweedie_cone import build_hadamard_dct_basis
    from run_dit_fpcv_fresh_probe import (
        CHECKPOINTS,
        CLASSES,
        DEFAULT_DATA_ROOT,
        DEFAULT_DIT_ROOT,
        DEFAULT_TRACE_ROOT,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_INTERNAL_TIMESTEPS,
        EXPECTED_MODELS_SHA256,
        EXPECTED_STRICT_SHA256,
        SOURCE_ARRAYS,
        canonical_sha256,
        project_outputs,
        raw_conditional_prediction,
        raw_sha256,
        replay_raw_conditional,
        sha256_file,
        validate_source_trace,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/dit_fpcv_geometry_redesign_smoke_seed050_v1"
)
PROJECTED_DIMENSION = 16
LATENT_DIMENSION = 4 * 32 * 32
PROTOCOL_POINT_SEED = 202608280731
DEFAULT_RING_POINTS = 128
DEFAULT_RELATIVE_RADII = (1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0)
DEFAULT_CHECKPOINTS = (99,)
QUERY_POINT_CHUNK = 8
POLYGON_SUBSET_SIZES = (4, 8, 16, 32, 64, 128)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclasses.dataclass(frozen=True)
class GeometryBlock:
    """One finite point set scored independently by cyclic assignment."""

    name: str
    family: str
    coordinates: np.ndarray
    ordered_cycle: bool
    metadata: Mapping[str, Any]


def _ring(
    first: np.ndarray,
    second: np.ndarray,
    *,
    point_count: int,
    center: np.ndarray | None = None,
    radius: float = 1.0,
) -> np.ndarray:
    if point_count < 3 or radius <= 0.0:
        raise ValueError("a polygon needs at least three points and positive radius")
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (PROJECTED_DIMENSION,) or second.shape != first.shape:
        raise ValueError("polygon directions have the wrong dimension")
    if (
        not math.isclose(float(first @ first), 1.0, abs_tol=2e-13)
        or not math.isclose(float(second @ second), 1.0, abs_tol=2e-13)
        or abs(float(first @ second)) > 2e-13
    ):
        raise ValueError("polygon directions must be orthonormal")
    origin = (
        np.zeros(PROJECTED_DIMENSION, dtype=np.float64)
        if center is None
        else np.asarray(center, dtype=np.float64)
    )
    angles = 2.0 * math.pi * np.arange(point_count, dtype=np.float64) / point_count
    values = (
        origin[None, :]
        + radius * np.cos(angles)[:, None] * first[None, :]
        + radius * np.sin(angles)[:, None] * second[None, :]
    )
    return np.ascontiguousarray(values, dtype=np.float64)


def _fixed_cloud(*, point_count: int = DEFAULT_RING_POINTS) -> np.ndarray:
    """Return one sample-independent, versioned spherical Q-space cloud."""

    generator = np.random.Generator(np.random.PCG64(PROTOCOL_POINT_SEED))
    values = generator.standard_normal((point_count, PROJECTED_DIMENSION))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    values -= np.mean(values, axis=0, keepdims=True, dtype=np.float64)
    values /= np.max(np.linalg.norm(values, axis=1))
    return np.ascontiguousarray(values, dtype=np.float64)


def build_geometry_blocks(
    *, ring_points: int = DEFAULT_RING_POINTS
) -> list[GeometryBlock]:
    """Build fixed central/translated polygons and point clouds.

    No state, model output, endpoint, or label enters this construction.
    """

    if ring_points != DEFAULT_RING_POINTS:
        raise ValueError(
            f"this smoke protocol fixes ring_points={DEFAULT_RING_POINTS}"
        )
    axis = np.eye(PROJECTED_DIMENSION, dtype=np.float64)
    rotated = hadamard(PROJECTED_DIMENSION).astype(np.float64) / math.sqrt(
        PROJECTED_DIMENSION
    )
    blocks: list[GeometryBlock] = []

    for frame_name, frame in (("axis", axis), ("hadamard", rotated)):
        for pair in range(PROJECTED_DIMENSION // 2):
            first = frame[:, 2 * pair]
            second = frame[:, 2 * pair + 1]
            blocks.append(
                GeometryBlock(
                    name=f"{frame_name}_central_ring_pair{pair:02d}",
                    family=f"{frame_name}_central_ring",
                    coordinates=_ring(
                        first, second, point_count=ring_points, radius=1.0
                    ),
                    ordered_cycle=True,
                    metadata={
                        "frame": frame_name,
                        "pair": pair,
                        "center_norm": 0.0,
                        "ring_radius": 1.0,
                    },
                )
            )

    # Probe local geometry away from x_star while staying inside the same
    # outer radius: ||0.75*w +/- 0.25*unit_circle|| < 0.80.
    for pair in range(PROJECTED_DIMENSION // 2):
        first = axis[:, 2 * pair]
        second = axis[:, 2 * pair + 1]
        offset_direction = axis[:, (2 * pair + 2) % PROJECTED_DIMENSION]
        for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
            center = sign * 0.75 * offset_direction
            blocks.append(
                GeometryBlock(
                    name=f"axis_offset_{sign_name}_ring_pair{pair:02d}",
                    family="axis_offset_ring_pm",
                    coordinates=_ring(
                        first,
                        second,
                        point_count=ring_points,
                        center=center,
                        radius=0.25,
                    ),
                    ordered_cycle=True,
                    metadata={
                        "frame": "axis",
                        "pair": pair,
                        "offset_sign": int(sign),
                        "offset_axis": int((2 * pair + 2) % PROJECTED_DIMENSION),
                        "center_norm": 0.75,
                        "ring_radius": 0.25,
                    },
                )
            )
            # A deliberately coarse noncentral triangle is included as a
            # direct control for the polygon-density argument.
            blocks.append(
                GeometryBlock(
                    name=f"axis_offset_{sign_name}_triangle_pair{pair:02d}",
                    family="axis_offset_triangle_pm",
                    coordinates=_ring(
                        first,
                        second,
                        point_count=3,
                        center=center,
                        radius=0.25,
                    ),
                    ordered_cycle=True,
                    metadata={
                        "frame": "axis",
                        "pair": pair,
                        "offset_sign": int(sign),
                        "offset_axis": int((2 * pair + 2) % PROJECTED_DIMENSION),
                        "center_norm": 0.75,
                        "ring_radius": 0.25,
                    },
                )
            )

    cloud = _fixed_cloud(point_count=ring_points)
    cloud_offset = 0.65 * rotated[:, 0]
    for name, center in (
        ("central", np.zeros(PROJECTED_DIMENSION, dtype=np.float64)),
        ("offset_plus", cloud_offset),
        ("offset_minus", -cloud_offset),
    ):
        scale = 1.0 if name == "central" else 0.30
        blocks.append(
            GeometryBlock(
                name=f"fixed_cloud_{name}",
                family="fixed_protocol_seeded_cloud",
                coordinates=np.ascontiguousarray(
                    center[None, :] + scale * cloud, dtype=np.float64
                ),
                ordered_cycle=False,
                metadata={
                    "protocol_point_seed": PROTOCOL_POINT_SEED,
                    "center_norm": float(np.linalg.norm(center)),
                    "cloud_scale": scale,
                },
            )
        )

    names = [block.name for block in blocks]
    if len(names) != len(set(names)):
        raise RuntimeError("geometry block names are not unique")
    for block in blocks:
        if (
            block.coordinates.ndim != 2
            or block.coordinates.shape[1] != PROJECTED_DIMENSION
            or len(block.coordinates) < 3
            or not np.isfinite(block.coordinates).all()
        ):
            raise RuntimeError(f"invalid coordinates for {block.name}")
    return blocks


def concatenate_blocks(
    blocks: Sequence[GeometryBlock],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    slices: list[tuple[int, int]] = []
    cursor = 0
    for block in blocks:
        stop = cursor + len(block.coordinates)
        slices.append((cursor, stop))
        cursor = stop
    return (
        np.ascontiguousarray(
            np.concatenate([block.coordinates for block in blocks], axis=0),
            dtype=np.float64,
        ),
        slices,
    )


def query_projected_points(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    coordinates: np.ndarray,
    absolute_radii: np.ndarray,
    internal_t: int,
    basis: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Query all classes with fixed-size batches and return float64 projections."""

    if state.shape != (len(CLASSES), 4, 32, 32):
        raise ValueError("state shape changed")
    if absolute_radii.shape != (len(CLASSES),):
        raise ValueError("absolute radii shape changed")
    reference = torch.from_numpy(np.ascontiguousarray(state)).to(device)
    class_ids = torch.tensor(CLASSES, dtype=torch.long, device=device)
    basis_device = torch.from_numpy(np.asarray(basis, dtype=np.float32)).to(device)
    coordinate_device = torch.from_numpy(
        np.asarray(coordinates, dtype=np.float32)
    ).to(device)
    radii_device = torch.from_numpy(
        np.asarray(absolute_radii, dtype=np.float32)
    ).to(device)
    output_chunks: list[np.ndarray] = []
    for start in range(0, len(coordinates), QUERY_POINT_CHUNK):
        stop = min(start + QUERY_POINT_CHUNK, len(coordinates))
        actual_count = stop - start
        chunk = coordinate_device[start:stop]
        if actual_count < QUERY_POINT_CHUNK:
            chunk = torch.cat(
                [
                    chunk,
                    chunk[-1:].expand(QUERY_POINT_CHUNK - actual_count, -1),
                ],
                dim=0,
            )
        latent_offsets = torch.einsum(
            "pr,rchw->pchw", chunk, basis_device
        ).contiguous()
        queries = (
            reference[:, None]
            + radii_device[:, None, None, None, None]
            * latent_offsets[None]
        ).reshape(len(CLASSES) * QUERY_POINT_CHUNK, 4, 32, 32)
        query_ids = (
            class_ids[:, None]
            .expand(len(CLASSES), QUERY_POINT_CHUNK)
            .reshape(len(CLASSES) * QUERY_POINT_CHUNK)
        )
        predictions = raw_conditional_prediction(
            diffusion,
            model,
            queries,
            internal_t=internal_t,
            class_ids=query_ids,
        ).reshape(len(CLASSES), QUERY_POINT_CHUNK, 4, 32, 32)
        raw = np.ascontiguousarray(
            predictions[:, :actual_count].cpu().numpy(), dtype=np.float32
        )
        output_chunks.append(
            np.stack(
                [project_outputs(basis, raw[slot]) for slot in range(len(CLASSES))],
                axis=0,
            )
        )
    outputs = np.ascontiguousarray(
        np.concatenate(output_chunks, axis=1), dtype=np.float64
    )
    expected = (len(CLASSES), len(coordinates), PROJECTED_DIMENSION)
    if outputs.shape != expected or not np.isfinite(outputs).all():
        raise RuntimeError("projected query output contract changed")
    return outputs


def cyclic_shift_gains(inputs: np.ndarray, outputs: np.ndarray) -> dict[str, float]:
    """Return the two explicit one-step polygon cycle gains."""

    identity = float(np.sum(outputs * inputs, dtype=np.float64))
    forward = float(np.sum(outputs * np.roll(inputs, -1, axis=0), dtype=np.float64))
    backward = float(np.sum(outputs * np.roll(inputs, 1, axis=0), dtype=np.float64))
    return {
        "forward_cycle_gain": forward - identity,
        "backward_cycle_gain": backward - identity,
        "best_oriented_cycle_gain": max(forward, backward) - identity,
    }


def score_blocks(
    blocks: Sequence[GeometryBlock],
    slices: Sequence[tuple[int, int]],
    outputs: np.ndarray,
    *,
    global_seed: int,
    class_id: int,
    checkpoint: int,
    relative_radius: float,
    absolute_radius: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block, (start, stop) in zip(blocks, slices):
        base_inputs = block.coordinates
        block_outputs = outputs[start:stop]
        subset_sizes = (
            [size for size in POLYGON_SUBSET_SIZES if size <= len(base_inputs)]
            if block.ordered_cycle and len(base_inputs) == DEFAULT_RING_POINTS
            else [len(base_inputs)]
        )
        for subset_size in subset_sizes:
            if subset_size == len(base_inputs):
                indices = np.arange(len(base_inputs), dtype=np.int64)
            else:
                if len(base_inputs) % subset_size != 0:
                    raise RuntimeError("polygon subset does not divide its parent ring")
                indices = np.arange(
                    0, len(base_inputs), len(base_inputs) // subset_size, dtype=np.int64
                )
            inputs = absolute_radius * base_inputs[indices]
            selected_outputs = block_outputs[indices]
            metrics = finite_cyclic_metrics(inputs, selected_outputs)
            permutation = np.asarray(metrics["optimal_permutation"], dtype=np.int64)
            cycle = (
                cyclic_shift_gains(inputs, selected_outputs)
                if block.ordered_cycle
                else {
                    "forward_cycle_gain": None,
                    "backward_cycle_gain": None,
                    "best_oriented_cycle_gain": None,
                }
            )
            rows.append(
                {
                    "global_seed": global_seed,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "relative_radius": relative_radius,
                    "absolute_radius": absolute_radius,
                    "geometry_name": block.name,
                    "geometry_family": block.family,
                    "parent_point_count": len(base_inputs),
                    "scored_point_count": subset_size,
                    "cyclic_violation": metrics["cyclic_violation"],
                    "normalization_denominator": metrics[
                        "normalization_denominator"
                    ],
                    "normalized_cyclic_violation": metrics[
                        "normalized_cyclic_violation"
                    ],
                    "identity_is_optimal_within_tolerance": metrics[
                        "identity_is_optimal_within_tolerance"
                    ],
                    "assignment_moved_point_count": int(
                        np.count_nonzero(permutation != np.arange(subset_size))
                    ),
                    **cycle,
                    **dict(block.metadata),
                }
            )
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Block-weighted strict-null summaries; no max is used as a primary score."""

    keys = sorted(
        {
            (
                int(row["class_id"]),
                int(row["checkpoint"]),
                float(row["relative_radius"]),
                str(row["geometry_family"]),
                int(row["scored_point_count"]),
            )
            for row in rows
        }
    )
    summaries: list[dict[str, Any]] = []
    for class_id, checkpoint, relative_radius, family, point_count in keys:
        block = [
            row
            for row in rows
            if int(row["class_id"]) == class_id
            and int(row["checkpoint"]) == checkpoint
            and float(row["relative_radius"]) == relative_radius
            and str(row["geometry_family"]) == family
            and int(row["scored_point_count"]) == point_count
        ]
        numerator = sum(float(row["cyclic_violation"]) for row in block)
        denominator = sum(float(row["normalization_denominator"]) for row in block)
        summaries.append(
            {
                "class_id": class_id,
                "checkpoint": checkpoint,
                "relative_radius": relative_radius,
                "geometry_family": family,
                "scored_point_count": point_count,
                "block_count": len(block),
                "resolved_block_count": sum(
                    float(row["cyclic_violation"]) > float(row["normalization_denominator"])
                    * 1e-12
                    for row in block
                ),
                "positive_raw_gain_block_count": sum(
                    float(row["cyclic_violation"]) > 0.0 for row in block
                ),
                "aggregate_normalized_cyclic_violation": numerator
                / max(denominator, 1e-30),
                "maximum_block_normalized_cyclic_violation": max(
                    float(row["normalized_cyclic_violation"]) for row in block
                ),
                "maximum_best_oriented_cycle_gain": max(
                    (
                        float(row["best_oriented_cycle_gain"])
                        for row in block
                        if row["best_oriented_cycle_gain"] is not None
                    ),
                    default=None,
                ),
            }
        )
    return summaries


def self_test() -> None:
    blocks = build_geometry_blocks()
    assert len(blocks) == 51
    coordinates, slices = concatenate_blocks(blocks)
    assert len(slices) == len(blocks)
    assert coordinates.shape == (4528, PROJECTED_DIMENSION)
    assert raw_sha256(coordinates) == raw_sha256(coordinates.copy())

    # The five-point cross-polytope misses a modest rotational defect because
    # the identity map supplies a coarse positive matching margin.
    cross = np.asarray([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
    rotation = np.asarray([[0.0, -0.08], [0.08, 0.0]], dtype=np.float64)
    defective = np.eye(2, dtype=np.float64) + rotation
    assert finite_cyclic_metrics(cross, cross @ defective.T)[
        "cyclic_violation"
    ] == 0.0
    ring = np.stack(
        [
            np.cos(2 * np.pi * np.arange(128) / 128),
            np.sin(2 * np.pi * np.arange(128) / 128),
        ],
        axis=1,
    )
    assert finite_cyclic_metrics(ring, ring @ defective.T)[
        "normalized_cyclic_violation"
    ] > 0.0

    # Every geometry, including translated clouds, remains exactly valid for
    # a symmetric PSD affine map.
    generator = np.random.default_rng(7)
    factor = generator.normal(size=(7, PROJECTED_DIMENSION))
    positive_semidefinite = factor.T @ factor
    for block in blocks:
        valid_outputs = block.coordinates @ positive_semidefinite.T + 3.0
        assert finite_cyclic_metrics(block.coordinates, valid_outputs)[
            "normalized_cyclic_violation"
        ] < 2e-13
    print("self-test passed")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.global_seed != 50:
        raise ValueError("this label-free geometry smoke is fixed to seed 50")
    checkpoints = tuple(args.checkpoints)
    if any(value not in CHECKPOINTS for value in checkpoints) or len(set(checkpoints)) != len(
        checkpoints
    ):
        raise ValueError(f"checkpoints must be a unique subset of {CHECKPOINTS}")
    relative_radii = tuple(float(value) for value in args.relative_radii)
    if (
        not relative_radii
        or len(set(relative_radii)) != len(relative_radii)
        or any(not math.isfinite(value) or value <= 0.0 for value in relative_radii)
    ):
        raise ValueError("relative radii must be unique, finite, and positive")

    dit_root = args.dit_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite smoke output: {output_root}")
    if sha256_file(Path(strict.__file__).resolve()) != EXPECTED_STRICT_SHA256:
        raise RuntimeError("strict DiT helper changed")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("DiT checkpoint changed")
    repository = strict.validate_repository(dit_root, checkpoint_path)
    if repository.get("pinned_source_sha256", {}).get("models.py") != EXPECTED_MODELS_SHA256:
        raise RuntimeError("pinned DiT source changed")

    source = trace_root / f"confirmation_v1_seed{args.global_seed:03d}"
    manifest, arrays = validate_source_trace(source, args.global_seed)
    basis, basis_metadata = build_hadamard_dct_basis()
    if basis.shape != (PROJECTED_DIMENSION, 4, 32, 32):
        raise RuntimeError("basis shape changed")
    blocks = build_geometry_blocks(ring_points=args.ring_points)
    coordinates, slices = concatenate_blocks(blocks)

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
    all_outputs: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    replay_records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from download import find_model
        from models import DiT_models

        device = torch.device("cuda")
        torch.manual_seed(20260828)
        torch.set_grad_enabled(False)
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(checkpoint_path)))
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))

        for checkpoint in checkpoints:
            checkpoint_slot = CHECKPOINTS.index(checkpoint)
            expected_t = EXPECTED_INTERNAL_TIMESTEPS[checkpoint_slot]
            internal_t = int(arrays["internal_timestep"][checkpoint])
            if internal_t != expected_t:
                raise RuntimeError("checkpoint/internal timestep map changed")
            alpha_bar = float(arrays["alpha_bar"][checkpoint])
            state = np.ascontiguousarray(arrays["state_before"][:, checkpoint])
            _, replay = replay_raw_conditional(
                diffusion,
                model,
                state=state,
                recorded_epsilon=np.ascontiguousarray(
                    arrays["conditional_epsilon_raw"][:, checkpoint]
                ),
                internal_t=internal_t,
                device=device,
            )
            replay_records.append(
                {"checkpoint": checkpoint, "internal_timestep": internal_t, **replay}
            )
            rms = np.sqrt(
                np.mean(state.astype(np.float64) ** 2, axis=(1, 2, 3), dtype=np.float64)
            )
            sigma = math.sqrt(max(0.0, 1.0 - alpha_bar))
            base_scale = math.sqrt(LATENT_DIMENSION) * np.maximum(rms, sigma)
            for relative_radius in relative_radii:
                absolute_radii = np.ascontiguousarray(
                    (base_scale * relative_radius).astype(np.float32).astype(np.float64)
                )
                print(
                    f"geometry smoke seed={args.global_seed} checkpoint={checkpoint} "
                    f"radius={relative_radius:.8f} points={len(coordinates)}",
                    flush=True,
                )
                projected = query_projected_points(
                    diffusion,
                    model,
                    state=state,
                    coordinates=coordinates,
                    absolute_radii=absolute_radii,
                    internal_t=internal_t,
                    basis=basis,
                    device=device,
                )
                all_outputs.append(projected)
                all_radii.append(absolute_radii)
                for class_slot, class_id in enumerate(CLASSES):
                    score_rows.extend(
                        score_blocks(
                            blocks,
                            slices,
                            projected[class_slot],
                            global_seed=args.global_seed,
                            class_id=class_id,
                            checkpoint=checkpoint,
                            relative_radius=relative_radius,
                            absolute_radius=float(absolute_radii[class_slot]),
                        )
                    )

                # Repeat the first 128-point central ring after a non-chunk
                # cyclic reorder.  Mapping it back checks batch-position and
                # repeat stability without repeating the full suite.
                repeat_coordinates = blocks[0].coordinates
                order = np.roll(np.arange(len(repeat_coordinates)), 3)
                repeated = query_projected_points(
                    diffusion,
                    model,
                    state=state,
                    coordinates=repeat_coordinates[order],
                    absolute_radii=absolute_radii,
                    internal_t=internal_t,
                    basis=basis,
                    device=device,
                )
                restored = np.empty_like(repeated)
                restored[:, order] = repeated
                original = projected[:, slices[0][0] : slices[0][1]]
                for class_slot, class_id in enumerate(CLASSES):
                    first_score = finite_cyclic_metrics(
                        absolute_radii[class_slot] * repeat_coordinates,
                        original[class_slot],
                    )
                    repeat_score = finite_cyclic_metrics(
                        absolute_radii[class_slot] * repeat_coordinates,
                        restored[class_slot],
                    )
                    repeat_records.append(
                        {
                            "class_id": class_id,
                            "checkpoint": checkpoint,
                            "relative_radius": relative_radius,
                            "maximum_projected_output_absolute_difference": float(
                                np.max(np.abs(original[class_slot] - restored[class_slot]))
                            ),
                            "first_normalized_cyclic_violation": first_score[
                                "normalized_cyclic_violation"
                            ],
                            "repeat_normalized_cyclic_violation": repeat_score[
                                "normalized_cyclic_violation"
                            ],
                            "absolute_score_difference": abs(
                                float(first_score["normalized_cyclic_violation"])
                                - float(repeat_score["normalized_cyclic_violation"])
                            ),
                        }
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

    summaries = aggregate_rows(score_rows)
    output_root.mkdir(parents=True)
    np.savez_compressed(
        output_root / "label_free_geometry_smoke.npz",
        global_seed=np.asarray(args.global_seed, dtype=np.int64),
        classes=np.asarray(CLASSES, dtype=np.int16),
        checkpoints=np.asarray(checkpoints, dtype=np.int16),
        relative_radii=np.asarray(relative_radii, dtype=np.float64),
        basis=np.ascontiguousarray(basis, dtype=np.float64),
        concatenated_coordinates=coordinates,
        block_starts=np.asarray([value[0] for value in slices], dtype=np.int32),
        block_stops=np.asarray([value[1] for value in slices], dtype=np.int32),
        absolute_radii=np.ascontiguousarray(np.stack(all_radii), dtype=np.float64),
        projected_outputs=np.ascontiguousarray(np.stack(all_outputs), dtype=np.float64),
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "DIT_FPCV_GEOMETRY_REDESIGN_LABEL_FREE_SMOKE_V1",
        "status": "complete",
        "global_seed": args.global_seed,
        "checkpoints": list(checkpoints),
        "relative_radii": list(relative_radii),
        "classes": list(CLASSES),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "strict_helper_sha256": sha256_file(Path(strict.__file__).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "basis_raw_sha256": raw_sha256(basis),
        "coordinate_raw_sha256": raw_sha256(coordinates),
        "protocol_point_seed": PROTOCOL_POINT_SEED,
        "basis_metadata": basis_metadata,
        "geometry_blocks": [
            {
                "name": block.name,
                "family": block.family,
                "point_count": len(block.coordinates),
                "ordered_cycle": block.ordered_cycle,
                "coordinate_raw_sha256": raw_sha256(block.coordinates),
                **dict(block.metadata),
            }
            for block in blocks
        ],
        "replay_records": replay_records,
        "repeat_order_stability": repeat_records,
        "score_rows": score_rows,
        "aggregate_rows": summaries,
        "source_trace": {
            "root": str(source),
            "identity_sha256": manifest["identity_sha256"],
            "arrays_read_exactly": list(SOURCE_ARRAYS),
            "array_raw_sha256": {
                name: raw_sha256(arrays[name]) for name in SOURCE_ARRAYS
            },
        },
        "firewall": {
            "quality_labels_or_reviews_opened": False,
            "png_or_decoded_image_array_opened": False,
            "endpoint_or_final_latent_opened": False,
            "external_metric_or_embedding_opened": False,
            "random_suffix_or_endpoint_generated": False,
            "baseline_state_changed": False,
        },
        "cost": {
            "unique_geometry_points_per_checkpoint_radius": len(coordinates),
            "repeat_points_per_checkpoint_radius": len(blocks[0].coordinates),
            "query_point_chunk": QUERY_POINT_CHUNK,
            "all_query_batches_same_padded_shape": True,
            "sample_equivalent_for_geometry_queries": len(checkpoints)
            * len(relative_radii)
            * len(CLASSES)
            * len(coordinates),
            "sample_equivalent_for_repeat_queries": len(checkpoints)
            * len(relative_radii)
            * len(CLASSES)
            * len(blocks[0].coordinates),
        },
        "wall_seconds": time.time() - started,
    }
    record["identity_sha256"] = canonical_sha256(record)
    with (output_root / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_root),
                "identity_sha256": record["identity_sha256"],
                "aggregate_rows": len(summaries),
                "wall_seconds": record["wall_seconds"],
            },
            indent=2,
        )
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--global-seed", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument(
        "--relative-radii", type=float, nargs="+", default=list(DEFAULT_RELATIVE_RADII)
    )
    parser.add_argument("--ring-points", type=int, default=DEFAULT_RING_POINTS)
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
    return args


if __name__ == "__main__":
    run(parse_args())

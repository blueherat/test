#!/usr/bin/env python3
"""Measure local density action of SiT guidance components.

For a perturbation ``u`` and an approximate marginal score ``s``, the local
density source is

    div(p u) / p = div(u) + u dot s.

This diagnostic deliberately does not claim to predict the finite-gamma
endpoint distribution.  It only checks which vector-field components are
locally density-active on matched teacher and unguided-rollout states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from experiments.imagenet100_sit_static_pair import (
        common_unique_orthogonal_directions,
        decompose_relative_to_anchor,
        output_to_field_velocity,
    )
    from experiments.run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_V270,
        DEFAULT_X400,
        _parse_floats,
    )
    from experiments.run_imagenet100_sit_guidance_conservativity import (
        _collect_rollout_states,
        _rademacher_like,
        _teacher_states,
        _validation_bank,
    )
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_static_pair import (
        common_unique_orthogonal_directions,
        decompose_relative_to_anchor,
        output_to_field_velocity,
    )
    from run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_V270,
        DEFAULT_X400,
        _parse_floats,
    )
    from run_imagenet100_sit_guidance_conservativity import (
        _collect_rollout_states,
        _rademacher_like,
        _teacher_states,
        _validation_bank,
    )
    from sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )


DEFAULT_CACHE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
)
COMPONENT_NAMES = (
    "anchor_v400",
    "other_x400",
    "other_v270",
    "gap_x_full",
    "gap_v_full",
    "gap_x_orthogonal",
    "gap_v_orthogonal",
    "x_common_on_v",
    "x_unique_to_v",
    "v_common_on_x",
    "v_unique_to_x",
)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stack_guidance_components(
    anchor: torch.Tensor,
    x_other: torch.Tensor,
    v_other: torch.Tensor,
) -> torch.Tensor:
    """Return all fields and guidance decompositions in a fixed order."""

    if anchor.shape != x_other.shape or anchor.shape != v_other.shape:
        raise ValueError("anchor, x_other, and v_other must have identical shapes")
    x_full = anchor - x_other
    v_full = anchor - v_other
    _, x_orthogonal = decompose_relative_to_anchor(anchor, x_full)
    _, v_orthogonal = decompose_relative_to_anchor(anchor, v_full)
    reciprocal = common_unique_orthogonal_directions(anchor, x_other, v_other)
    values = (
        anchor,
        x_other,
        v_other,
        x_full,
        v_full,
        x_orthogonal,
        v_orthogonal,
        reciprocal["x_common_on_v"],
        reciprocal["x_unique_to_v"],
        reciprocal["v_common_on_x"],
        reciprocal["v_unique_to_x"],
    )
    return torch.stack(values, dim=0)


def approximate_linear_flow_score(
    state: torch.Tensor,
    anchor_velocity: torch.Tensor,
    time_value: float | torch.Tensor,
) -> torch.Tensor:
    """Approximate the linear-flow marginal score using the anchor velocity.

    For ``z_t = (1-t) eps + t x`` and the population conditional velocity,
    ``score_t(z) = (t v*(z,t) - z) / (1-t)``.
    """

    if state.shape != anchor_velocity.shape:
        raise ValueError("state and anchor velocity must have identical shapes")
    time_tensor = torch.as_tensor(time_value, device=state.device, dtype=state.dtype)
    if time_tensor.numel() != 1 or not 0.0 < float(time_tensor) < 1.0:
        raise ValueError("time_value must be a scalar strictly inside (0, 1)")
    return (time_tensor * anchor_velocity - state) / (1.0 - time_tensor)


def hutchinson_component_divergence(
    component_field: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    probes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-sample divergence for a stack of independent fields.

    ``component_field`` returns ``[K, B, ...]`` while ``state`` and every probe
    have shape ``[B, ...]``.  The returned divergence has shape ``[P, K, B]``.
    """

    if probes.ndim != state.ndim + 1 or tuple(probes.shape[1:]) != tuple(state.shape):
        raise ValueError("probes must have shape [P, *state.shape]")
    primal: torch.Tensor | None = None
    estimates: list[torch.Tensor] = []
    for probe in probes:
        current_primal, tangent = jvp(component_field, (state,), (probe,))
        if current_primal.ndim != state.ndim + 1:
            raise ValueError("component field must return [K, B, ...]")
        if tuple(current_primal.shape[1:]) != tuple(state.shape):
            raise ValueError("component field output does not match state shape")
        if primal is None:
            primal = current_primal
        reduce_dims = tuple(range(2, tangent.ndim))
        estimates.append((tangent * probe.unsqueeze(0)).sum(dim=reduce_dims))
    assert primal is not None
    return primal, torch.stack(estimates, dim=0)


def component_jacobian_symmetry_probe(
    component_field: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    probe: torch.Tensor,
    component_indices: list[int],
) -> dict[str, torch.Tensor]:
    """Compare ``J q`` and ``J^T q`` for selected stacked components."""

    if state.shape != probe.shape:
        raise ValueError("state and probe must have identical shapes")
    primal, jacobian_vector = jvp(component_field, (state,), (probe,))
    transpose_primal, pullback = torch.func.vjp(component_field, state)
    if not torch.allclose(primal, transpose_primal, rtol=1e-5, atol=1e-6):
        raise RuntimeError("JVP and VJP primal evaluations disagree")
    selected_jvp = jacobian_vector[component_indices]
    transpose_vectors = []
    for component_index in component_indices:
        cotangent = torch.zeros_like(primal)
        cotangent[component_index] = probe
        transpose_vectors.append(pullback(cotangent)[0])
    selected_vjp = torch.stack(transpose_vectors)
    antisymmetric = selected_jvp - selected_vjp
    reduce_dims = tuple(range(2, selected_jvp.ndim))
    jvp_rms = selected_jvp.square().mean(dim=reduce_dims).sqrt()
    vjp_rms = selected_vjp.square().mean(dim=reduce_dims).sqrt()
    antisymmetric_rms = antisymmetric.square().mean(dim=reduce_dims).sqrt()
    dot = (selected_jvp * selected_vjp).sum(dim=reduce_dims)
    cosine = dot / (
        selected_jvp.square().sum(dim=reduce_dims).sqrt()
        * selected_vjp.square().sum(dim=reduce_dims).sqrt()
    ).clamp_min(torch.finfo(state.dtype).tiny)
    energy_fraction = antisymmetric.square().mean(dim=reduce_dims) / (
        2.0
        * (
            selected_jvp.square().mean(dim=reduce_dims)
            + selected_vjp.square().mean(dim=reduce_dims)
        )
    ).clamp_min(torch.finfo(state.dtype).tiny)
    field_rms = primal[component_indices].square().mean(dim=reduce_dims).sqrt()
    return {
        "field_rms": field_rms,
        "jvp_rms": jvp_rms,
        "vjp_rms": vjp_rms,
        "antisymmetric_rms": antisymmetric_rms,
        "antisymmetric_over_jvp_rms": antisymmetric_rms
        / jvp_rms.clamp_min(torch.finfo(state.dtype).tiny),
        "antisymmetric_energy_fraction": energy_fraction,
        "jvp_vjp_cosine": cosine,
    }


def density_action_terms(
    components: torch.Tensor,
    score: torch.Tensor,
    divergence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-dimension score work and local density action."""

    if components.ndim != score.ndim + 1:
        raise ValueError("components must have shape [K, *score.shape]")
    if tuple(components.shape[1:]) != tuple(score.shape):
        raise ValueError("component and score shapes do not match")
    if divergence.ndim != 3 or tuple(divergence.shape[1:]) != tuple(components.shape[:2]):
        raise ValueError("divergence must have shape [P, K, B]")
    feature_dims = tuple(range(2, components.ndim))
    dimension = math.prod(components.shape[2:])
    score_work = (components * score.unsqueeze(0)).sum(dim=feature_dims) / dimension
    divergence_per_dim = divergence / dimension
    action = divergence_per_dim + score_work.unsqueeze(0)
    field_rms = components.square().mean(dim=feature_dims).sqrt()
    score_rms = score.square().flatten(1).mean(1).sqrt()
    return {
        "field_rms": field_rms,
        "score_rms": score_rms,
        "score_work_per_dim": score_work,
        "divergence_per_dim": divergence_per_dim,
        "density_action_per_dim": action,
    }


class TripletFields:
    """Evaluate v400, x400, and v270 on exactly the same state and labels."""

    def __init__(self, models, semantics, labels: torch.Tensor) -> None:
        self.models = models
        self.semantics = semantics
        self.labels = labels

    def _labels_for(self, state: torch.Tensor) -> torch.Tensor:
        if len(state) % len(self.labels) != 0:
            raise ValueError("state batch is not divisible by labels")
        return self.labels.repeat(len(state) // len(self.labels))

    def evaluate(
        self,
        index: int,
        time_value: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        times = time_value.expand(len(state))
        with sdpa_kernel(SDPBackend.MATH):
            output = self.models[index](state, times, self._labels_for(state))
        return output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=self.semantics[index],
        )

    def anchor(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.evaluate(0, time_value, state)

    def components(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return stack_guidance_components(
            self.evaluate(0, time_value, state),
            self.evaluate(1, time_value, state),
            self.evaluate(2, time_value, state),
        )


def _load_triplet(args: argparse.Namespace, labels: torch.Tensor, device: torch.device):
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    models = []
    semantics = []
    metadata = []
    checkpoints = []
    for path in (args.anchor_checkpoint, args.x400_checkpoint, args.v270_checkpoint):
        model, field_semantics, field_metadata, checkpoint = _load_field_model(
            checkpoint_path=path.expanduser().resolve(),
            requested_field="auto",
            weights=args.weights,
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
        assert model is not None
        models.append(model)
        semantics.append(field_semantics)
        metadata.append(field_metadata)
        checkpoints.append(checkpoint)
    validate_pair_compatibility(
        checkpoints[0], checkpoints[1], metadata[0], metadata[1]
    )
    validate_pair_compatibility(
        checkpoints[0], checkpoints[2], metadata[0], metadata[2], allow_step_mismatch=True
    )
    expected_targets = ("velocity", "x", "velocity")
    actual_targets = tuple(item.prediction_target for item in semantics)
    if actual_targets != expected_targets:
        raise ValueError(f"expected targets {expected_targets}, got {actual_targets}")
    return TripletFields(models, semantics, labels), {
        "official_sit": source_metadata,
        "anchor": metadata[0],
        "x400": metadata[1],
        "v270": metadata[2],
    }


def _probe_rows_to_sample_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    keys = ("sample_id", "validation_index", "label", "source", "time", "component")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    sample_rows: list[dict[str, object]] = []
    for key, selected in grouped.items():
        base = dict(zip(keys, key, strict=True))
        divergence = float(np.mean([float(item["divergence_per_dim"]) for item in selected]))
        score_work = float(selected[0]["score_work_per_dim"])
        action = divergence + score_work
        base.update(
            {
                "probes": len(selected),
                "field_rms": float(selected[0]["field_rms"]),
                "score_rms": float(selected[0]["score_rms"]),
                "score_work_per_dim": score_work,
                "divergence_per_dim": divergence,
                "density_action_per_dim": action,
            }
        )
        sample_rows.append(base)
    return sample_rows


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _aggregate(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in sample_rows:
        key = (str(row["source"]), float(row["time"]), str(row["component"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (source, time_value, component), selected in sorted(grouped.items()):
        arrays = {
            name: np.asarray([float(row[name]) for row in selected], dtype=np.float64)
            for name in (
                "field_rms",
                "score_rms",
                "score_work_per_dim",
                "divergence_per_dim",
                "density_action_per_dim",
            )
        }
        row: dict[str, object] = {
            "source": source,
            "time": time_value,
            "component": component,
            "samples": len(selected),
        }
        for name, values in arrays.items():
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_abs_mean"] = float(np.abs(values).mean())
            row[f"{name}_rms"] = _rms(values)
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_q10"] = float(np.quantile(values, 0.10))
            row[f"{name}_q90"] = float(np.quantile(values, 0.90))
        tiny = np.finfo(np.float64).tiny
        row["action_per_unit_field"] = row["density_action_per_dim_rms"] / max(
            row["field_rms_rms"], tiny
        )
        row["action_cancellation_ratio"] = row["density_action_per_dim_rms"] / max(
            row["divergence_per_dim_rms"] + row["score_work_per_dim_rms"], tiny
        )
        output.append(row)
    return output


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.probes <= 0:
        raise ValueError("sample, batch, and probe counts must be positive")
    if args.num_samples % args.batch_size != 0:
        raise ValueError("num-samples must be divisible by batch-size")
    if any(value <= 0 or value >= 1 for value in args.times):
        raise ValueError("all times must lie strictly inside (0, 1)")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cache_dir = args.cache_dir.expanduser().resolve()
    bank = _validation_bank(cache_dir, num_samples=args.num_samples, seed=args.seed)
    fields, model_metadata = _load_triplet(
        args,
        bank["labels"][: args.batch_size].to(device),
        device,
    )
    output_dir = (
        args.output_root.expanduser().resolve()
        / "density_action"
        / f"n{args.num_samples}_p{args.probes}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    probe_generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)
    probe_rows: list[dict[str, object]] = []

    for start in range(0, args.num_samples, args.batch_size):
        stop = start + args.batch_size
        fields.labels = bank["labels"][start:stop].to(device)
        clean = bank["clean"][start:stop].to(device)
        noise = bank["bridge_noise"][start:stop].to(device)
        teacher = _teacher_states(clean, noise, args.times)
        rollout = _collect_rollout_states(
            fields.anchor,
            noise,
            steps=args.heun_steps,
            requested_times=args.times,
        )
        for source, source_states in (("teacher", teacher), ("v400_rollout", rollout)):
            for time_value in args.times:
                state = source_states[time_value]
                time_tensor = torch.tensor(time_value, device=device)
                probes = torch.stack(
                    [_rademacher_like(state, probe_generator) for _ in range(args.probes)]
                )

                def component_field(current_state: torch.Tensor) -> torch.Tensor:
                    return fields.components(time_tensor, current_state)

                components, divergence = hutchinson_component_divergence(
                    component_field,
                    state,
                    probes,
                )
                score = approximate_linear_flow_score(
                    state,
                    components[0],
                    time_tensor,
                )
                terms = density_action_terms(components, score, divergence)
                for component_index, component in enumerate(COMPONENT_NAMES):
                    for probe_index in range(args.probes):
                        for local_index in range(args.batch_size):
                            probe_rows.append(
                                {
                                    "sample_id": start + local_index,
                                    "validation_index": int(bank["indices"][start + local_index]),
                                    "label": int(bank["labels"][start + local_index]),
                                    "source": source,
                                    "time": float(time_value),
                                    "component": component,
                                    "probe": probe_index,
                                    "field_rms": float(
                                        terms["field_rms"][component_index, local_index]
                                        .detach()
                                        .cpu()
                                    ),
                                    "score_rms": float(
                                        terms["score_rms"][local_index].detach().cpu()
                                    ),
                                    "score_work_per_dim": float(
                                        terms["score_work_per_dim"][
                                            component_index, local_index
                                        ]
                                        .detach()
                                        .cpu()
                                    ),
                                    "divergence_per_dim": float(
                                        terms["divergence_per_dim"][
                                            probe_index, component_index, local_index
                                        ]
                                        .detach()
                                        .cpu()
                                    ),
                                    "density_action_per_dim": float(
                                        terms["density_action_per_dim"][
                                            probe_index, component_index, local_index
                                        ]
                                        .detach()
                                        .cpu()
                                    ),
                                }
                            )
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "samples": [start, stop],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    sample_rows = _probe_rows_to_sample_rows(probe_rows)
    aggregate_rows = _aggregate(sample_rows)
    _write_csv(probe_rows, output_dir / "per_sample_probe_metrics.csv")
    _write_csv(sample_rows, output_dir / "per_sample_metrics.csv")
    _write_csv(aggregate_rows, output_dir / "metrics_by_source_time_component.csv")
    summary = {
        "format": "eqvae_sit400_guidance_density_action_v1",
        "interpretation_scope": (
            "Local source div(pu)/p on matched states; not a finite-gamma endpoint predictor"
        ),
        "score_formula": "score_hat(z,t) = (t * v400(z,t) - z) / (1 - t)",
        "component_names": list(COMPONENT_NAMES),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "probes": args.probes,
        "times": args.times,
        "heun_steps": args.heun_steps,
        "seed": args.seed,
        "probe_seed": args.probe_seed,
        "precision": "fp32",
        "allow_tf32": False,
        "math_attention": True,
        "state_sources": ["teacher", "v400_rollout"],
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "validation_indices_sha256": _tensor_sha256(bank["indices"]),
        "labels_sha256": _tensor_sha256(bank["labels"]),
        "clean_latents_sha256": _tensor_sha256(bank["clean"]),
        "bridge_noise_sha256": _tensor_sha256(bank["bridge_noise"]),
        "models": model_metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate_rows": aggregate_rows,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(output_dir),
                "elapsed_seconds": summary["elapsed_seconds"],
                "aggregate_rows": len(aggregate_rows),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--x400-checkpoint", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument(
        "--times",
        type=_parse_floats,
        default=_parse_floats("0.1,0.3,0.5,0.7,0.9"),
    )
    parser.add_argument("--heun-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--probe-seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

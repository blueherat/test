#!/usr/bin/env python3
"""Sample ImageNet-100 SiT with CFG/AG fixed-point foresight calibration.

This is a controlled flow-matching analogue of NeurIPS 2025 Foresight
Guidance.  The base grid runs from noise at ``t=0`` to data at ``t=1``.  At a
scheduled base step, the sampler advances over a future interval with the
guided field, returns with a designated reference field, and repeats that
round trip before taking the ordinary local guided step.

For AutoGuidance the same local field has two exact Euler decompositions:

* weak reference: ``W + (1 + gamma) * (S - W)``;
* strong reference: ``S - gamma * (W - S)``.

They produce the same one-step AG update but different long-horizon operators.
The explicit ``--ag-reference`` control is therefore part of the experiment,
not an implementation detail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

try:
    from experiments.foresight_fixed_point_flow import (
        ForesightEvent,
        anchored_foresight_step,
        conjugated_future_gap_step,
        cross_time_norm_matched_gap_step,
        euler_flow_map,
        foresight_round_trip,
        future_raw_gap_step,
        guided_field,
        implicit_autoguidance_euler_step,
        integrate_flow_map,
        iterate_anchored_foresight_operator,
        iterate_anchored_gap_operator,
        iterate_conjugated_future_gap_operator,
        local_calibrated_autoguidance_euler_step,
        parse_foresight_schedule,
        sample_cosine,
        sample_rms,
        schedule_by_step,
        scheduled_autoguidance_euler_step,
        split_guided_euler_step,
    )
    from experiments.imagenet100_sit_static_pair import output_to_field_velocity
    from experiments.imagenet100_sit_multiscale_guidance import (
        schedule_depth,
        select_per_sample,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        InternalHeadSpec,
        evaluate_source_with_heads,
        load_internal_head_for_source,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from foresight_fixed_point_flow import (
        ForesightEvent,
        anchored_foresight_step,
        conjugated_future_gap_step,
        cross_time_norm_matched_gap_step,
        euler_flow_map,
        foresight_round_trip,
        future_raw_gap_step,
        guided_field,
        implicit_autoguidance_euler_step,
        integrate_flow_map,
        iterate_anchored_foresight_operator,
        iterate_anchored_gap_operator,
        iterate_conjugated_future_gap_operator,
        local_calibrated_autoguidance_euler_step,
        parse_foresight_schedule,
        sample_cosine,
        sample_rms,
        schedule_by_step,
        scheduled_autoguidance_euler_step,
        split_guided_euler_step,
    )
    from imagenet100_sit_static_pair import output_to_field_velocity
    from imagenet100_sit_multiscale_guidance import schedule_depth, select_per_sample
    from imagenet100_sit_multiscale_models import (
        InternalHeadSpec,
        evaluate_source_with_heads,
        load_internal_head_for_source,
    )
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_STRONG_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_WEAK_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00500000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "foresight_fixed_point_v1/smoke"
)
DEFAULT_FORESIGHT_SCHEDULE = "0:5:2,5:5:2,15:5:1"


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@dataclass
class FieldFamily:
    reference: object
    target: object
    guided: object
    foresight_forward: object
    local_scale: float
    metadata: dict[str, object]
    counters: dict[str, int]


def _model_velocity(
    model: torch.nn.Module,
    semantics,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    times = time_value.expand(len(state))
    if autocast_dtype is None:
        output = model(state, times, labels)
    else:
        with torch.autocast("cuda", dtype=autocast_dtype):
            output = model(state, times, labels)
    return output_to_field_velocity(
        output,
        state=state,
        time_value=times,
        semantics=semantics,
    )


def build_ag_fields(
    strong_model: torch.nn.Module,
    weak_model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    strong_semantics,
    weak_semantics,
    gamma: float,
    foresight_gamma: float | None,
    reference_choice: str,
    autocast_dtype: torch.dtype | None,
) -> FieldFamily:
    counters = {"strong_forwards": 0, "weak_forwards": 0}

    def strong(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counters["strong_forwards"] += 1
        return _model_velocity(
            strong_model,
            strong_semantics,
            state,
            time_value,
            labels[: len(state)],
            autocast_dtype=autocast_dtype,
        )

    def weak(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counters["weak_forwards"] += 1
        return _model_velocity(
            weak_model,
            weak_semantics,
            state,
            time_value,
            labels[: len(state)],
            autocast_dtype=autocast_dtype,
        )

    def ag_field(
        time_value: torch.Tensor, state: torch.Tensor, *, field_gamma: float
    ) -> torch.Tensor:
        strong_value = strong(time_value, state)
        if field_gamma == 0.0:
            return strong_value
        weak_value = weak(time_value, state)
        return strong_value + float(field_gamma) * (strong_value - weak_value)

    def ag_guided(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return ag_field(time_value, state, field_gamma=gamma)

    resolved_foresight_gamma = (
        float(gamma) if foresight_gamma is None else float(foresight_gamma)
    )

    def ag_foresight(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return ag_field(
            time_value,
            state,
            field_gamma=resolved_foresight_gamma,
        )

    if reference_choice == "weak":
        reference, target, local_scale = weak, strong, 1.0 + float(gamma)
    elif reference_choice == "strong":
        reference, target, local_scale = strong, weak, -float(gamma)
    else:
        raise ValueError("AG reference must be 'strong' or 'weak'")
    return FieldFamily(
        reference=reference,
        target=target,
        guided=ag_guided,
        foresight_forward=ag_foresight,
        local_scale=local_scale,
        metadata={
            "family": "ag",
            "gamma": float(gamma),
            "foresight_gamma": resolved_foresight_gamma,
            "reference_choice": reference_choice,
            "decomposition": (
                "W + (1+gamma)*(S-W)"
                if reference_choice == "weak"
                else "S + (-gamma)*(W-S)"
            ),
            "common_guided_field": "S + gamma*(S-W)",
        },
        counters=counters,
    )


def build_ig_fields(
    strong_model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    strong_semantics,
    heads: dict[str, InternalHeadSpec],
    depths: tuple[int, ...],
    gamma: float,
    autocast_dtype: torch.dtype | None,
    gamma_segments: tuple[tuple[float, float], ...] | None = None,
) -> FieldFamily:
    """Build scheduled Internal Guidance as an AG-compatible field family.

    The final SiT output is the strong field and the time-selected internal
    readout is the weak field. ``guided`` evaluates both in one shared
    backbone pass. Standalone strong/weak queries are intentionally counted
    separately because the foresight operator may request them independently.
    """

    if not heads:
        raise ValueError("Internal Guidance requires at least one internal head")
    depth_to_name: dict[int, str] = {}
    for name, spec in heads.items():
        if spec.prediction_target != "velocity":
            raise ValueError("scheduled IG foresight currently requires velocity heads")
        if spec.depth in depth_to_name:
            raise ValueError(f"multiple internal heads at depth {spec.depth}")
        depth_to_name[spec.depth] = name
    if tuple(sorted(depth_to_name)) != tuple(sorted(depths)):
        raise ValueError("--ig-depths must exactly match the loaded internal heads")
    if len(set(depths)) != len(depths):
        raise ValueError("IG schedule depths must be unique")

    counters = {
        "strong_backbone_forwards": 0,
        "weak_probe_backbone_forwards": 0,
        "guided_shared_backbone_forwards": 0,
    }

    def gamma_at(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if gamma_segments is None:
            values = torch.full(
                (len(state),),
                float(gamma),
                device=state.device,
                dtype=state.dtype,
            )
        else:
            times = time_value.expand(len(state)).to(device=state.device)
            boundaries = torch.tensor(
                [end for end, _ in gamma_segments[:-1]],
                device=state.device,
                dtype=times.dtype,
            )
            segment_values = torch.tensor(
                [value for _, value in gamma_segments],
                device=state.device,
                dtype=state.dtype,
            )
            indices = torch.bucketize(times.contiguous(), boundaries, right=True)
            values = segment_values[indices]
        return values.view(len(state), *([1] * (state.ndim - 1)))

    def evaluate_pair(
        time_value: torch.Tensor,
        state: torch.Tensor,
        *,
        counter: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counters[counter] += 1
        times = time_value.expand(len(state)).contiguous()
        context = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with context:
            full, trained, _ = evaluate_source_with_heads(
                strong_model,
                state,
                times,
                labels[: len(state)],
                heads=heads,
                source_semantics=strong_semantics,
            )
        selected_depth = (
            torch.full_like(times, int(depths[0]), dtype=torch.long)
            if len(depths) == 1
            else schedule_depth(
                times,
                order="coarse_to_fine",
                depths=depths,
            )
        )
        weak_by_depth = {
            depth: trained[depth_to_name[depth]] for depth in depths
        }
        weak = select_per_sample(weak_by_depth, selected_depth)
        return full, weak, selected_depth

    def strong(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counters["strong_backbone_forwards"] += 1
        return _model_velocity(
            strong_model,
            strong_semantics,
            state,
            time_value,
            labels[: len(state)],
            autocast_dtype=autocast_dtype,
        )

    def weak(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        full, weak_value, _ = evaluate_pair(
            time_value,
            state,
            counter="weak_probe_backbone_forwards",
        )
        if gamma_segments is not None:
            # Make target-reference equal the actual scheduled IG correction.
            # This lets the generic foresight operator compare the correction
            # that the production controller would apply at each time.
            return full - gamma_at(time_value, state) * (full - weak_value)
        return weak_value

    def guided(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        full, weak_value, _ = evaluate_pair(
            time_value,
            state,
            counter="guided_shared_backbone_forwards",
        )
        return full + gamma_at(time_value, state) * (full - weak_value)

    schedule_payload = (
        [
            {"end": float(end), "gamma": float(value)}
            for end, value in gamma_segments
        ]
        if gamma_segments is not None
        else None
    )
    effective_gamma = 1.0 if gamma_segments is not None else float(gamma)

    return FieldFamily(
        reference=weak,
        target=strong,
        guided=guided,
        foresight_forward=guided,
        local_scale=1.0 + effective_gamma,
        metadata={
            "family": "ag",
            "provider": "scheduled_internal_guidance",
            "reference_choice": "weak",
            "gamma": float(gamma),
            "gamma_segments": schedule_payload,
            "depths": list(depths),
            "head_checkpoints": {
                name: spec.checkpoint for name, spec in heads.items()
            },
            "decomposition": (
                "R_eff(t) + 2*(S-R_eff(t)); "
                "S-R_eff(t)=gamma(t)*(S-W_depth(t))"
                if gamma_segments is not None
                else "W_depth(t) + (1+gamma)*(S-W_depth(t))"
            ),
            "common_guided_field": (
                "S + gamma(t)*(S-W_depth(t))"
                if gamma_segments is not None
                else "S + gamma*(S-W_depth(t))"
            ),
        },
        counters=counters,
    )


def build_cfg_fields(
    model: torch.nn.Module,
    semantics,
    labels: torch.Tensor,
    *,
    cfg_scale: float,
    guided_channels: int,
    autocast_dtype: torch.dtype | None,
) -> FieldFamily:
    if guided_channels <= 0 or guided_channels > LATENT_SHAPE[0]:
        raise ValueError("guided_channels must lie within the latent channel count")
    null_labels = torch.full_like(labels, NUM_CLASSES)
    counters = {"conditional_forwards": 0, "unconditional_forwards": 0}

    def conditional(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counters["conditional_forwards"] += 1
        return _model_velocity(
            model,
            semantics,
            state,
            time_value,
            labels[: len(state)],
            autocast_dtype=autocast_dtype,
        )

    def unconditional(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counters["unconditional_forwards"] += 1
        return _model_velocity(
            model,
            semantics,
            state,
            time_value,
            null_labels[: len(state)],
            autocast_dtype=autocast_dtype,
        )

    def reference(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        uncond = unconditional(time_value, state)
        if guided_channels == state.shape[1]:
            return uncond
        cond = conditional(time_value, state)
        return torch.cat(
            (uncond[:, :guided_channels], cond[:, guided_channels:]), dim=1
        )

    def cfg_guided(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cond = conditional(time_value, state)
        uncond = unconditional(time_value, state)
        result = cond.clone()
        result[:, :guided_channels] = guided_field(
            uncond[:, :guided_channels],
            cond[:, :guided_channels],
            scale=cfg_scale,
        )
        return result

    return FieldFamily(
        reference=reference,
        target=conditional,
        guided=cfg_guided,
        foresight_forward=cfg_guided,
        local_scale=float(cfg_scale),
        metadata={
            "family": "cfg",
            "cfg_scale": float(cfg_scale),
            "guided_channels": int(guided_channels),
            "reference": (
                "unconditional"
                if guided_channels == LATENT_SHAPE[0]
                else "unconditional guided channels + conditional remaining channels"
            ),
        },
        counters=counters,
    )


def _diagnose_round_trip(
    state: torch.Tensor,
    *,
    time_value: torch.Tensor,
    future_time: torch.Tensor,
    fields: FieldFamily,
    event: ForesightEvent,
    foresight_relaxation: float,
    neighbor_relative_rms: float,
    diagnostic_generator: torch.Generator,
) -> dict[str, object]:
    counters_before = dict(fields.counters)
    batch = state.detach()
    random_direction = torch.randn(
        batch.shape,
        device=batch.device,
        dtype=batch.dtype,
        generator=diagnostic_generator,
    )
    direction_rms = sample_rms(random_direction).reshape(
        -1, *([1] * (batch.ndim - 1))
    )
    state_rms = sample_rms(batch).reshape(-1, *([1] * (batch.ndim - 1)))
    perturbation = (
        float(neighbor_relative_rms)
        * state_rms.clamp_min(1e-3)
        * random_direction
        / direction_rms.clamp_min(torch.finfo(batch.dtype).tiny)
    )
    neighbor = batch + perturbation

    base_mapped = foresight_round_trip(
        batch,
        time_value=time_value,
        future_time=future_time,
        forward_field=fields.foresight_forward,
        inverse_field=fields.reference,
        relaxation=foresight_relaxation,
    )
    neighbor_mapped = foresight_round_trip(
        neighbor,
        time_value=time_value,
        future_time=future_time,
        forward_field=fields.foresight_forward,
        inverse_field=fields.reference,
        relaxation=foresight_relaxation,
    )
    input_distance = sample_rms(neighbor - batch)
    output_distance = sample_rms(neighbor_mapped - base_mapped)
    contraction = output_distance / input_distance.clamp_min(
        torch.finfo(output_distance.dtype).tiny
    )

    current = batch
    iterations: list[dict[str, object]] = []
    previous_move: torch.Tensor | None = None
    horizon = future_time - time_value
    for iteration in range(event.iterations + 1):
        reference_now = fields.reference(time_value, current)
        target_now = fields.target(time_value, current)
        gap = target_now - reference_now
        record: dict[str, object] = {
            "iteration": iteration,
            "current_gap_rms": _summary(sample_rms(gap)),
            "one_step_future_path_gap_rms": _summary(sample_rms(horizon * gap)),
        }
        if iteration == event.iterations:
            iterations.append(record)
            break
        updated = foresight_round_trip(
            current,
            time_value=time_value,
            future_time=future_time,
            forward_field=fields.foresight_forward,
            inverse_field=fields.reference,
            relaxation=foresight_relaxation,
        )
        move = updated - current
        record["operator_move_rms"] = _summary(sample_rms(move))
        if previous_move is not None:
            record["move_cosine_to_previous"] = _summary(
                sample_cosine(move, previous_move)
            )
            record["move_rms_ratio_to_previous"] = _summary(
                sample_rms(move)
                / sample_rms(previous_move).clamp_min(torch.finfo(move.dtype).tiny)
            )
        iterations.append(record)
        previous_move = move
        current = updated

    diagnostic_forwards = {
        key: int(value) - int(counters_before.get(key, 0))
        for key, value in fields.counters.items()
    }
    fields.counters.update(counters_before)
    return {
        "step_index": event.step_index,
        "time": float(time_value),
        "future_time": float(future_time),
        "lookahead_steps": event.lookahead_steps,
        "iterations_requested": event.iterations,
        "foresight_relaxation": float(foresight_relaxation),
        "neighbor_relative_rms": float(neighbor_relative_rms),
        "local_operator_l2_ratio": _summary(contraction),
        "local_operator_squared_l2_ratio": _summary(contraction.square()),
        "diagnostic_model_forwards": diagnostic_forwards,
        "iteration_trace": iterations,
    }


def _diagnose_anchored_operator(
    state: torch.Tensor,
    *,
    time_value: torch.Tensor,
    future_time: torch.Tensor,
    fields: FieldFamily,
    event: ForesightEvent,
    anchored_strength: float,
    neighbor_relative_rms: float,
    diagnostic_generator: torch.Generator,
) -> dict[str, object]:
    counters_before = dict(fields.counters)
    anchor = state.detach()
    random_direction = torch.randn(
        anchor.shape,
        device=anchor.device,
        dtype=anchor.dtype,
        generator=diagnostic_generator,
    )
    direction_rms = sample_rms(random_direction).reshape(
        -1, *([1] * (anchor.ndim - 1))
    )
    anchor_rms = sample_rms(anchor).reshape(-1, *([1] * (anchor.ndim - 1)))
    perturbation = (
        float(neighbor_relative_rms)
        * anchor_rms.clamp_min(1e-3)
        * random_direction
        / direction_rms.clamp_min(torch.finfo(anchor.dtype).tiny)
    )
    neighbor = anchor + perturbation
    base_mapped, _ = anchored_foresight_step(
        anchor,
        anchor,
        time_value=time_value,
        future_time=future_time,
        forward_field=fields.foresight_forward,
        inverse_field=fields.reference,
        strength=anchored_strength,
    )
    neighbor_mapped, _ = anchored_foresight_step(
        anchor,
        neighbor,
        time_value=time_value,
        future_time=future_time,
        forward_field=fields.foresight_forward,
        inverse_field=fields.reference,
        strength=anchored_strength,
    )
    contraction = sample_rms(neighbor_mapped - base_mapped) / sample_rms(
        neighbor - anchor
    ).clamp_min(torch.finfo(anchor.dtype).tiny)

    endpoint, moves, discrepancies = iterate_anchored_foresight_operator(
        anchor,
        time_value=time_value,
        future_time=future_time,
        iterations=event.iterations,
        forward_field=fields.foresight_forward,
        inverse_field=fields.reference,
        strength=anchored_strength,
    )
    iteration_trace = []
    for index, (move, discrepancy) in enumerate(
        zip(moves, discrepancies, strict=True), start=1
    ):
        record: dict[str, object] = {
            "iteration": index,
            "move_rms": _summary(sample_rms(move)),
            "future_discrepancy_rms": _summary(sample_rms(discrepancy)),
            "anchor_displacement_rms": _summary(sample_rms(endpoint - anchor))
            if index == len(moves)
            else None,
        }
        if index > 1:
            previous = moves[index - 2]
            record["move_cosine_to_previous"] = _summary(
                sample_cosine(move, previous)
            )
            record["move_rms_ratio_to_previous"] = _summary(
                sample_rms(move)
                / sample_rms(previous).clamp_min(torch.finfo(move.dtype).tiny)
            )
        iteration_trace.append(record)

    diagnostic_forwards = {
        key: int(value) - int(counters_before.get(key, 0))
        for key, value in fields.counters.items()
    }
    fields.counters.update(counters_before)
    return {
        "operator": "input-anchored nonzero-discrepancy Picard",
        "step_index": event.step_index,
        "time": float(time_value),
        "future_time": float(future_time),
        "lookahead_steps": event.lookahead_steps,
        "iterations_requested": event.iterations,
        "anchored_strength": float(anchored_strength),
        "neighbor_relative_rms": float(neighbor_relative_rms),
        "local_operator_l2_ratio": _summary(contraction),
        "local_operator_squared_l2_ratio": _summary(contraction.square()),
        "diagnostic_model_forwards": diagnostic_forwards,
        "iteration_trace": iteration_trace,
    }


def _diagnose_conjugated_future_gap(
    state: torch.Tensor,
    *,
    time_values: tuple[torch.Tensor, ...],
    fields: FieldFamily,
    calibration_strength: float,
    flow_integrator: str,
) -> dict[str, object]:
    """Audit numerical inversion and the defining conjugacy on a small batch."""

    counters_before = dict(fields.counters)
    future = integrate_flow_map(
        state,
        time_values=time_values,
        field=fields.target,
        method=flow_integrator,
    )
    future_gap = fields.target(time_values[-1], future) - fields.reference(
        time_values[-1], future
    )
    desired_future = future + float(calibration_strength) * future_gap
    calibrated, _, _ = conjugated_future_gap_step(
        state,
        time_values=time_values,
        strong_field=fields.target,
        weak_field=fields.reference,
        calibration_strength=calibration_strength,
        flow_integrator=flow_integrator,
    )
    achieved_future = integrate_flow_map(
        calibrated,
        time_values=time_values,
        field=fields.target,
        method=flow_integrator,
    )
    round_trip = integrate_flow_map(
        future,
        time_values=tuple(reversed(time_values)),
        field=fields.target,
        method=flow_integrator,
    )
    intended_move = desired_future - future
    diagnostic_forwards = {
        key: int(value) - int(counters_before.get(key, 0))
        for key, value in fields.counters.items()
    }
    fields.counters.update(counters_before)
    return {
        "strong_round_trip_rms": _summary(sample_rms(round_trip - state)),
        "strong_round_trip_relative_to_state": _summary(
            sample_rms(round_trip - state)
            / sample_rms(state).clamp_min(torch.finfo(state.dtype).tiny)
        ),
        "intended_future_calibration_rms": _summary(sample_rms(intended_move)),
        "conjugacy_residual_rms": _summary(
            sample_rms(achieved_future - desired_future)
        ),
        "conjugacy_residual_relative_to_calibration": _summary(
            sample_rms(achieved_future - desired_future)
            / sample_rms(intended_move).clamp_min(torch.finfo(state.dtype).tiny)
        ),
        "diagnostic_model_forwards": diagnostic_forwards,
        "flow_integrator": flow_integrator,
    }


def integrate_condition(
    initial_noise: torch.Tensor,
    *,
    fields: FieldFamily,
    method: str,
    num_steps: int,
    foresight_events: tuple[ForesightEvent, ...],
    local_iterations: int,
    foresight_relaxation: float,
    foresight_event_local_mode: str,
    anchored_gamma: float,
    anchored_strength_multiplier: float,
    conjugate_flow_integrator: str,
    diagnostics: bool,
    diagnostic_samples: int,
    neighbor_relative_rms: float,
    diagnostic_generator: torch.Generator,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    schedule = schedule_by_step(foresight_events, num_steps=num_steps)
    if method not in {
        "foresight",
        "anchored",
        "implicit_ag",
        "scheduled_ag",
        "local_calibration_ag",
        "future_dir_current_norm_ag",
        "current_dir_future_norm_ag",
        "future_raw_ag",
        "conjugate_ag",
    } and schedule:
        raise ValueError("a foresight schedule requires a fixed-point method")
    if method in {
        "anchored",
        "implicit_ag",
        "scheduled_ag",
        "local_calibration_ag",
        "future_dir_current_norm_ag",
        "current_dir_future_norm_ag",
        "future_raw_ag",
        "conjugate_ag",
    } and anchored_gamma <= 0.0:
        raise ValueError("anchored AG methods require a positive gamma")
    state = initial_noise.float()
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=state.device)
    diagnostic_records: list[dict[str, object]] = []

    for step_index, (time_value, next_time) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        event = schedule.get(step_index)
        if event is not None and method == "scheduled_ag":
            if fields.metadata.get("family") != "ag":
                raise ValueError("scheduled AG requires an AutoGuidance field family")
            if fields.metadata.get("reference_choice") != "weak":
                raise ValueError("scheduled AG requires weak reference / strong target")
            state = scheduled_autoguidance_euler_step(
                state,
                time_value=time_value,
                next_time=next_time,
                strong_field=fields.target,
                weak_field=fields.reference,
                gamma=float(anchored_gamma),
                multiplier=float(anchored_strength_multiplier),
            )
            if diagnostics:
                diagnostic_records.append(
                    {
                        "operator": "current-state scheduled AutoGuidance",
                        "step_index": step_index,
                        "time": float(time_value),
                        "effective_gamma": float(anchored_gamma)
                        * float(anchored_strength_multiplier),
                    }
                )
            continue
        if event is not None and method == "local_calibration_ag":
            if fields.metadata.get("family") != "ag":
                raise ValueError(
                    "local calibrated AG requires an AutoGuidance field family"
                )
            if fields.metadata.get("reference_choice") != "weak":
                raise ValueError(
                    "local calibrated AG requires weak reference / strong target"
                )
            state, calibration = local_calibrated_autoguidance_euler_step(
                state,
                time_value=time_value,
                next_time=next_time,
                strong_field=fields.target,
                weak_field=fields.reference,
                gamma=float(anchored_gamma),
                multiplier=float(anchored_strength_multiplier),
            )
            if diagnostics:
                count = min(int(diagnostic_samples), len(calibration))
                diagnostic_records.append(
                    {
                        "operator": "zero-lookahead AG calibration plus strong response",
                        "step_index": step_index,
                        "time": float(time_value),
                        "effective_gamma": float(anchored_gamma)
                        * float(anchored_strength_multiplier),
                        "calibration_rms": _summary(sample_rms(calibration[:count])),
                    }
                )
            continue
        if event is not None and method in {
            "future_dir_current_norm_ag",
            "current_dir_future_norm_ag",
        }:
            if fields.metadata.get("family") != "ag":
                raise ValueError("cross-time gap matching requires AutoGuidance")
            if fields.metadata.get("reference_choice") != "weak":
                raise ValueError(
                    "cross-time gap matching requires weak reference / strong target"
                )
            interval_grid = tuple(
                grid[step_index : step_index + event.lookahead_steps + 1]
            )
            calibration_strength = (
                float(next_time - time_value)
                * float(anchored_gamma)
                * float(anchored_strength_multiplier)
            )
            direction = (
                "future_match_current"
                if method == "future_dir_current_norm_ag"
                else "current_match_future"
            )
            traces: list[dict[str, object]] = []
            for iteration in range(event.iterations):
                state, current_gap, future_gap, move = (
                    cross_time_norm_matched_gap_step(
                        state,
                        time_values=interval_grid,
                        strong_field=fields.target,
                        weak_field=fields.reference,
                        calibration_strength=calibration_strength,
                        direction=direction,
                        flow_integrator=conjugate_flow_integrator,
                    )
                )
                if diagnostics:
                    count = min(int(diagnostic_samples), len(state))
                    current_subset = current_gap[:count]
                    future_subset = future_gap[:count]
                    traces.append(
                        {
                            "iteration": iteration + 1,
                            "current_gap_rms": _summary(
                                sample_rms(current_subset)
                            ),
                            "future_gap_rms": _summary(sample_rms(future_subset)),
                            "future_to_current_rms_ratio": _summary(
                                sample_rms(future_subset)
                                / sample_rms(current_subset).clamp_min(
                                    torch.finfo(current_subset.dtype).tiny
                                )
                            ),
                            "current_future_cosine": _summary(
                                sample_cosine(current_subset, future_subset)
                            ),
                            "selected_move_rms": _summary(
                                sample_rms(move[:count])
                            ),
                        }
                    )
            if diagnostics:
                diagnostic_records.append(
                    {
                        "operator": "cross-time AG direction/norm matched control",
                        "direction": direction,
                        "step_index": step_index,
                        "time": float(time_value),
                        "future_time": float(interval_grid[-1]),
                        "lookahead_steps": event.lookahead_steps,
                        "iterations_requested": event.iterations,
                        "future_calibration_strength": calibration_strength,
                        "iteration_trace": traces,
                    }
                )
            state = state + (next_time - time_value) * fields.target(
                time_value, state
            )
            continue
        if event is not None and method in {"future_raw_ag", "conjugate_ag"}:
            if fields.metadata.get("family") != "ag":
                raise ValueError("future AG requires an AutoGuidance field family")
            if fields.metadata.get("reference_choice") != "weak":
                raise ValueError("future AG requires weak reference / strong target")
            interval_grid = tuple(
                grid[step_index : step_index + event.lookahead_steps + 1]
            )
            calibration_strength = (
                float(next_time - time_value)
                * float(anchored_gamma)
                * float(anchored_strength_multiplier)
            )
            conjugacy_audit = None
            if diagnostics and method == "conjugate_ag":
                count = min(int(diagnostic_samples), len(state))
                conjugacy_audit = _diagnose_conjugated_future_gap(
                    state[:count],
                    time_values=interval_grid,
                    fields=fields,
                    calibration_strength=calibration_strength,
                    flow_integrator=conjugate_flow_integrator,
                )
            if method == "conjugate_ag":
                state, moves, future_gaps = iterate_conjugated_future_gap_operator(
                    state,
                    time_values=interval_grid,
                    iterations=event.iterations,
                    strong_field=fields.target,
                    weak_field=fields.reference,
                    calibration_strength=calibration_strength,
                    flow_integrator=conjugate_flow_integrator,
                )
                operator_name = "strong-flow-conjugated future AutoGuidance"
            else:
                moves = []
                future_gaps = []
                for _ in range(event.iterations):
                    updated, future_gap, move = future_raw_gap_step(
                        state,
                        time_values=interval_grid,
                        strong_field=fields.target,
                        weak_field=fields.reference,
                        calibration_strength=calibration_strength,
                        flow_integrator=conjugate_flow_integrator,
                    )
                    state = updated
                    moves.append(move)
                    future_gaps.append(future_gap)
                operator_name = "raw future AutoGuidance gap in current coordinates"
            if diagnostics:
                count = min(int(diagnostic_samples), len(state))
                trace: list[dict[str, object]] = []
                previous_move: torch.Tensor | None = None
                for iteration, (move, gap) in enumerate(
                    zip(moves, future_gaps, strict=True)
                ):
                    move_subset = move[:count]
                    gap_subset = gap[:count]
                    record: dict[str, object] = {
                        "iteration": iteration + 1,
                        "current_move_rms": _summary(sample_rms(move_subset)),
                        "future_gap_rms": _summary(sample_rms(gap_subset)),
                    }
                    if previous_move is not None:
                        previous_subset = previous_move[:count]
                        record["move_rms_ratio_to_previous"] = _summary(
                            sample_rms(move_subset)
                            / sample_rms(previous_subset).clamp_min(
                                torch.finfo(move_subset.dtype).tiny
                            )
                        )
                        record["move_cosine_to_previous"] = _summary(
                            sample_cosine(move_subset, previous_subset)
                        )
                    trace.append(record)
                    previous_move = move
                diagnostic_records.append(
                    {
                        "operator": operator_name,
                        "step_index": step_index,
                        "time": float(time_value),
                        "future_time": float(interval_grid[-1]),
                        "lookahead_steps": event.lookahead_steps,
                        "iterations_requested": event.iterations,
                        "future_calibration_strength": calibration_strength,
                        **(
                            {"conjugacy_audit": conjugacy_audit}
                            if method == "conjugate_ag"
                            else {}
                        ),
                        "iteration_trace": trace,
                    }
                )
            # The future calibration replaces this step's ordinary AG term;
            # the strong model then transports the calibrated state onward.
            state = state + (next_time - time_value) * fields.target(
                time_value, state
            )
            continue
        if event is not None and method == "implicit_ag":
            if fields.metadata.get("family") != "ag":
                raise ValueError("implicit AG requires an AutoGuidance field family")
            if fields.metadata.get("reference_choice") != "weak":
                raise ValueError("implicit AG requires weak reference / strong target")
            state, moves, gaps = implicit_autoguidance_euler_step(
                state,
                time_value=time_value,
                next_time=next_time,
                iterations=event.iterations,
                strong_field=fields.target,
                weak_field=fields.reference,
                gamma=float(anchored_gamma),
            )
            if diagnostics:
                count = min(int(diagnostic_samples), len(state))
                trace: list[dict[str, object]] = []
                previous_move: torch.Tensor | None = None
                for iteration, (move, gap) in enumerate(zip(moves, gaps, strict=True)):
                    move_subset = move[:count]
                    gap_subset = gap[:count]
                    record: dict[str, object] = {
                        "iteration": iteration + 1,
                        "move_rms": _summary(sample_rms(move_subset)),
                        "gap_rms": _summary(sample_rms(gap_subset)),
                    }
                    if previous_move is not None:
                        previous_subset = previous_move[:count]
                        record["move_rms_ratio_to_previous"] = _summary(
                            sample_rms(move_subset)
                            / sample_rms(previous_subset).clamp_min(
                                torch.finfo(move_subset.dtype).tiny
                            )
                        )
                        record["move_cosine_to_previous"] = _summary(
                            sample_cosine(move_subset, previous_subset)
                        )
                    trace.append(record)
                    previous_move = move
                diagnostic_records.append(
                    {
                        "operator": "local implicit AutoGuidance discrepancy",
                        "step_index": step_index,
                        "time": float(time_value),
                        "iterations_requested": event.iterations,
                        "step_strength": float(next_time - time_value)
                        * float(anchored_gamma),
                        "iteration_trace": trace,
                    }
                )
            continue
        if event is not None:
            future_time = grid[step_index + event.lookahead_steps]
            anchored_strength = (
                float(anchored_strength_multiplier)
                * float(anchored_gamma)
                / float(event.lookahead_steps)
            )
            if diagnostics:
                count = min(int(diagnostic_samples), len(state))
                diagnostic_records.append(
                    _diagnose_anchored_operator(
                        state[:count],
                        time_value=time_value,
                        future_time=future_time,
                        fields=fields,
                        event=event,
                        anchored_strength=anchored_strength,
                        neighbor_relative_rms=neighbor_relative_rms,
                        diagnostic_generator=diagnostic_generator,
                    )
                    if method == "anchored"
                    else _diagnose_round_trip(
                        state[:count],
                        time_value=time_value,
                        future_time=future_time,
                        fields=fields,
                        event=event,
                        foresight_relaxation=foresight_relaxation,
                        neighbor_relative_rms=neighbor_relative_rms,
                        diagnostic_generator=diagnostic_generator,
                    )
                )
            if method == "anchored":
                state, _, _ = iterate_anchored_foresight_operator(
                    state,
                    time_value=time_value,
                    future_time=future_time,
                    iterations=event.iterations,
                    forward_field=fields.foresight_forward,
                    inverse_field=fields.reference,
                    strength=anchored_strength,
                )
            else:
                for _ in range(event.iterations):
                    state = foresight_round_trip(
                        state,
                        time_value=time_value,
                        future_time=future_time,
                        forward_field=fields.foresight_forward,
                        inverse_field=fields.reference,
                        relaxation=foresight_relaxation,
                    )

        use_reference_local_step = (
            event is not None and foresight_event_local_mode == "reference"
        )
        use_target_local_step = (
            event is not None and foresight_event_local_mode == "target"
        )
        if use_reference_local_step:
            state = state + (next_time - time_value) * fields.reference(
                time_value, state
            )
        elif use_target_local_step:
            state = state + (next_time - time_value) * fields.target(
                time_value, state
            )
        elif method == "closed":
            state = state + (next_time - time_value) * fields.guided(
                time_value, state
            )
        elif local_iterations == 1:
            # One calibration plus the reference update is algebraically the
            # guided Euler step.  Calling the common guided field avoids a
            # redundant conditional evaluation for SiT's three-channel CFG.
            state = state + (next_time - time_value) * fields.guided(
                time_value, state
            )
        else:
            state, _ = split_guided_euler_step(
                state,
                time_value=time_value,
                next_time=next_time,
                reference_field=fields.reference,
                target_field=fields.target,
                scale=fields.local_scale,
                calibration_iterations=local_iterations,
            )
    return state, diagnostic_records


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    if args.ig_gamma_segments is not None:
        if args.family != "ig":
            raise ValueError("--ig-gamma-segments is only valid for --family ig")
        if not math.isclose(float(args.ag_gamma), 1.0, abs_tol=1e-12):
            raise ValueError(
                "segmented IG embeds gamma(t) in the effective gap; use --ag-gamma 1"
            )
    if args.method not in {
        "foresight",
        "anchored",
        "implicit_ag",
        "scheduled_ag",
        "local_calibration_ag",
        "future_dir_current_norm_ag",
        "current_dir_future_norm_ag",
        "future_raw_ag",
        "conjugate_ag",
    } and args.foresight_schedule:
        raise ValueError("clear --foresight-schedule for closed/split methods")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(
        device, limit_gib=args.cuda_allocator_limit_gib
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    strong_model, strong_semantics, strong_metadata, strong_payload = _load_field_model(
        checkpoint_path=args.strong_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    assert strong_model is not None
    weak_model = None
    weak_semantics = None
    weak_metadata = None
    weak_payload = None
    internal_heads: dict[str, InternalHeadSpec] = {}
    if args.family == "ag":
        weak_model, weak_semantics, weak_metadata, weak_payload = _load_field_model(
            checkpoint_path=args.weak_checkpoint.expanduser().resolve(),
            requested_field="auto",
            weights=args.weights,
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
        assert weak_model is not None and weak_semantics is not None
        validate_pair_compatibility(
            strong_payload,
            weak_payload,
            strong_metadata,
            weak_metadata,
            allow_step_mismatch=True,
        )
    elif args.family == "ig":
        if args.weights != "ema":
            raise ValueError("the validated internal heads require EMA source weights")
        if not args.internal_head:
            raise ValueError("--family ig requires at least one --internal-head")
        for name, checkpoint_path in args.internal_head:
            if name in internal_heads:
                raise ValueError(f"duplicate internal head name: {name}")
            internal_heads[name] = load_internal_head_for_source(
                checkpoint_path=checkpoint_path,
                name=name,
                head_weights=args.internal_head_weights,
                model=strong_model,
                sit_module=sit_module,
                source_checkpoint_path=args.strong_checkpoint,
                source_metadata=source_metadata,
                device=device,
            )
        weak_metadata = {
            "kind": "scheduled_internal_heads",
            "depths": list(args.ig_depths),
            "head_weights": args.internal_head_weights,
            "heads": {
                name: {
                    "checkpoint": spec.checkpoint,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "depth": spec.depth,
                    "prediction_target": spec.prediction_target,
                }
                for name, spec in internal_heads.items()
            },
        }
    del strong_payload, weak_payload

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)

    sample_generator = torch.Generator(device=device).manual_seed(args.global_seed)
    diagnostic_generator = torch.Generator(device=device).manual_seed(
        args.global_seed + 91_273
    )
    events = parse_foresight_schedule(args.foresight_schedule)
    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    family_totals: dict[str, int] = {}
    diagnostics: list[dict[str, object]] = []
    preview = None
    cursor = 0
    started = time.perf_counter()

    while cursor < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - cursor)
        if args.sample_rng_mode == "per_batch":
            batch_index = cursor // args.batch_size
            batch_generator = torch.Generator(device=device).manual_seed(
                args.global_seed + batch_index
            )
        else:
            batch_generator = sample_generator
        noise = torch.randn(
            (batch_size, *LATENT_SHAPE),
            device=device,
            generator=batch_generator,
        )
        labels = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,),
            device=device,
            generator=batch_generator,
        )
        if args.family == "cfg":
            fields = build_cfg_fields(
                strong_model,
                strong_semantics,
                labels,
                cfg_scale=args.cfg_scale,
                guided_channels=args.cfg_guided_channels,
                autocast_dtype=autocast_dtype,
            )
        elif args.family == "ag":
            assert weak_model is not None and weak_semantics is not None
            fields = build_ag_fields(
                strong_model,
                weak_model,
                labels,
                strong_semantics=strong_semantics,
                weak_semantics=weak_semantics,
                gamma=args.ag_gamma,
                foresight_gamma=args.foresight_ag_gamma,
                reference_choice=args.ag_reference,
                autocast_dtype=autocast_dtype,
            )
        else:
            fields = build_ig_fields(
                strong_model,
                labels,
                strong_semantics=strong_semantics,
                heads=internal_heads,
                depths=args.ig_depths,
                gamma=args.ag_gamma,
                autocast_dtype=autocast_dtype,
                gamma_segments=args.ig_gamma_segments,
            )
        run_diagnostics = cursor == 0 and args.diagnostic_samples > 0
        latents, batch_diagnostics = integrate_condition(
            noise,
            fields=fields,
            method=args.method,
            num_steps=args.num_steps,
            foresight_events=events,
            local_iterations=args.local_iterations,
            foresight_relaxation=args.foresight_relaxation,
            foresight_event_local_mode=args.foresight_event_local_mode,
            anchored_gamma=(
                1.0
                if args.family == "ig" and args.ig_gamma_segments is not None
                else args.ag_gamma
            ),
            anchored_strength_multiplier=args.anchored_strength_multiplier,
            conjugate_flow_integrator=args.conjugate_flow_integrator,
            diagnostics=run_diagnostics,
            diagnostic_samples=args.diagnostic_samples,
            neighbor_relative_rms=args.neighbor_relative_rms,
            diagnostic_generator=diagnostic_generator,
        )
        if not torch.isfinite(latents).all():
            raise FloatingPointError("non-finite latent endpoint")
        decoded = decode_latents_in_chunks(
            vae,
            latents,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.cpu().contiguous().numpy().tobytes())
        for key, value in fields.counters.items():
            family_totals[key] = family_totals.get(key, 0) + int(value)
        if run_diagnostics:
            diagnostics = batch_diagnostics
        if preview is None:
            preview = decoded[: min(16, len(decoded))].cpu()
        cursor = stop
        if cursor == batch_size or cursor == args.num_samples or cursor % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "generated": cursor,
                        "total": args.num_samples,
                        "elapsed_seconds": elapsed,
                        "images_per_second": cursor / elapsed,
                    }
                ),
                flush=True,
            )

    sample_path = output_dir / f"samples_n{args.num_samples}.npz"
    label_path = output_dir / f"sample_labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    assert preview is not None
    save_image(
        preview,
        output_dir / "preview.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    histogram = np.bincount(labels_array.astype(np.int64), minlength=NUM_CLASSES)
    manifest = {
        "format": "eqvae_imagenet100_sit_foresight_fixed_point_samples_v1",
        "scope": "FSG mechanism replication and AutoGuidance extension on SiT",
        "method": args.method,
        "family": args.family,
        "strong": strong_metadata,
        "weak": weak_metadata,
        "weights": args.weights,
        "field_definition": fields.metadata,
        "num_steps": int(args.num_steps),
        "base_integrator": "explicit_euler",
        "local_iterations": int(args.local_iterations),
        "local_update_implementation": (
            "direct guided Euler (one-iteration algebraic equivalent)"
            if args.local_iterations == 1
            else "explicit repeated calibration then reference Euler"
        ),
        "foresight_relaxation": float(args.foresight_relaxation),
        "foresight_event_local_mode": args.foresight_event_local_mode,
        "anchored_strength_multiplier": float(args.anchored_strength_multiplier),
        "conjugate_flow_integrator": args.conjugate_flow_integrator,
        "foresight_schedule": [event.__dict__ for event in events],
        "foresight_operator": {
            "closed": "none",
            "split": "none",
            "foresight": (
                "Phi_reference(t <- t+delta) o Phi_guided(t+delta <- t), "
                "one Euler evaluation per leg"
            ),
            "anchored": "input-anchored long-horizon discrepancy calibration",
            "implicit_ag": "current-state implicit AG gap calibration",
            "scheduled_ag": "current-state AG with scheduled gamma multiplier",
            "local_calibration_ag": (
                "zero-lookahead AG calibration followed by strong-field response"
            ),
            "future_dir_current_norm_ag": (
                "future AG direction rescaled to current-gap sample RMS"
            ),
            "current_dir_future_norm_ag": (
                "current AG direction rescaled to future-gap sample RMS"
            ),
            "future_raw_ag": (
                "future AG gap injected directly in current latent coordinates"
            ),
            "conjugate_ag": (
                "strong-flow inverse o future AG calibration o strong-flow forward"
            ),
        }[args.method],
        "requested_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "vae_decode_batch_size": int(args.vae_decode_batch_size),
        "global_seed": int(args.global_seed),
        "sample_rng_mode": args.sample_rng_mode,
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": histogram.tolist(),
        "precision": args.precision,
        "allow_tf32": bool(args.allow_tf32),
        "samples": str(sample_path),
        "labels": str(label_path),
        "operator_diagnostics": diagnostics,
        "model_forward_totals": family_totals,
        "elapsed_seconds": time.perf_counter() - started,
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps({"event": "complete", **manifest}, indent=2), flush=True)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("internal head must use NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("internal head must use non-empty NAME=PATH")
    return name, Path(path)


def parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("IG depths must be comma-separated integers") from error
    if len(depths) < 1 or any(depth < 1 for depth in depths):
        raise argparse.ArgumentTypeError("IG depths must be positive")
    if len(set(depths)) != len(depths):
        raise argparse.ArgumentTypeError("IG depths must be unique")
    return depths


def parse_gamma_segments(value: str) -> tuple[tuple[float, float], ...]:
    """Parse end:gamma pieces covering [0, 1], e.g. .25:.6,.5:.7,1:0."""

    segments: list[tuple[float, float]] = []
    try:
        for item in value.split(","):
            end_text, gamma_text = item.split(":", maxsplit=1)
            segments.append((float(end_text), float(gamma_text)))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "IG gamma segments must use END:GAMMA comma-separated syntax"
        ) from error
    if not segments:
        raise argparse.ArgumentTypeError("IG gamma segments cannot be empty")
    previous = 0.0
    for end, gamma in segments:
        if not math.isfinite(end) or not math.isfinite(gamma) or gamma < 0.0:
            raise argparse.ArgumentTypeError("IG segment values must be finite and gamma >= 0")
        if end <= previous or end > 1.0:
            raise argparse.ArgumentTypeError("IG segment ends must strictly increase to 1")
        previous = end
    if not math.isclose(segments[-1][0], 1.0, abs_tol=1e-8):
        raise argparse.ArgumentTypeError("IG gamma segments must end at 1")
    return tuple(segments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("cfg", "ag", "ig"), required=True)
    parser.add_argument(
        "--method",
        choices=(
            "closed",
            "split",
            "foresight",
            "anchored",
            "implicit_ag",
            "scheduled_ag",
            "local_calibration_ag",
            "future_dir_current_norm_ag",
            "current_dir_future_norm_ag",
            "future_raw_ag",
            "conjugate_ag",
        ),
        required=True,
    )
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG_CHECKPOINT)
    parser.add_argument("--weak-checkpoint", type=Path, default=DEFAULT_WEAK_CHECKPOINT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--internal-head",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--internal-head-weights",
        choices=("ema", "model"),
        default="ema",
    )
    parser.add_argument("--ig-depths", type=parse_depths, default=(4, 10))
    parser.add_argument(
        "--ig-gamma-segments",
        type=parse_gamma_segments,
        default=None,
        metavar="END:GAMMA,...",
        help=(
            "Piecewise-constant IG strength covering [0,1]. For the best "
            "depth4 controller use .25:.6,.5:.7,1:0."
        ),
    )
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-guided-channels", type=int, default=3)
    parser.add_argument("--ag-gamma", type=float, default=1.0)
    parser.add_argument(
        "--foresight-ag-gamma",
        type=float,
        default=None,
        help=(
            "AG strength used only on the forward leg of the long-horizon "
            "operator. Defaults to --ag-gamma; zero means pure-strong foresight."
        ),
    )
    parser.add_argument("--ag-reference", choices=("strong", "weak"), default="weak")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--local-iterations", type=int, default=1)
    parser.add_argument(
        "--foresight-relaxation",
        type=float,
        default=1.0,
        help="Relaxation rho in x <- x + rho * (T(x) - x).",
    )
    parser.add_argument(
        "--foresight-event-local-mode",
        choices=("guided", "reference", "target"),
        default="guided",
        help=(
            "Local step taken after a foresight event. 'reference' lets the "
            "round trip replace, rather than stack on, the local guidance."
        ),
    )
    parser.add_argument(
        "--anchored-strength-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the first-order-matched anchored strength "
            "gamma/lookahead_steps."
        ),
    )
    parser.add_argument(
        "--conjugate-flow-integrator",
        choices=("euler", "heun", "rk4"),
        default="rk4",
        help="Fixed-step solver used only inside the conjugated AG subflow.",
    )
    parser.add_argument(
        "--foresight-schedule",
        default=DEFAULT_FORESIGHT_SCHEDULE,
        help="Comma-separated step:lookahead:iterations entries; use '' to disable.",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--diagnostic-samples", type=int, default=8)
    parser.add_argument("--neighbor-relative-rms", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--sample-rng-mode",
        choices=("continuous", "per_batch"),
        default="continuous",
        help=(
            "Noise/label RNG protocol. 'per_batch' reseeds each batch with "
            "global_seed + batch_index, matching the validated IG sweeps."
        ),
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

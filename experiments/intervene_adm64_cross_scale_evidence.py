#!/usr/bin/env python3
"""Bounded ADM64 rollback experiment driven by cross-scale path evidence.

This is an *exploratory intervention* runner, not a production sampler.  It
keeps the official classifier-guided ADM transition P exactly unchanged until
the fixed equal-weight mixture E-process first reaches ``1 / alpha``.  At most
once, it then returns to a predeclared earlier evidence checkpoint and draws
one independent P suffix.  There is no retry loop and no endpoint selection.

For every triggered path the runner also draws a second, independent suffix
from the exact same rollback state.  This ``same_checkpoint_random_control``
has identical neural-evaluation and Gaussian-draw counts to the intervention;
it measures retry-to-retry variability, but does *not* by itself isolate the
benefit of evidence-based trigger selection.  That design limitation is
recorded in every manifest.

The three decoded outputs are saved separately:

* the complete original P path (required to match the frozen baseline PNG),
* the one-shot evidence-triggered rollback suffix, and
* the same-checkpoint independent-resampling control.

If the mixture never crosses, all three decoded images are bitwise identical
to the frozen baseline and neither suffix is evaluated.  ``--force-intervention``
is available only as an unmistakably labelled oracle diagnostic and is
excluded from method claims.  It requires an explicit fixed internal timestep.

Every executed P transition and every cross-scale evidence event is recorded.
By default, exact state trajectories and full checkpoint diagnostic tensors
are also saved in self-hashed NPZ files; the exact rollback state is saved as
an independently hashed float32 NPY file.  Outputs are fail-closed and are
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Deterministic cuBLAS must be configured before a CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:  # Support CLI and package imports.
    from .adm64_path_evidence import (
        kl_tempered_score_mean_shift_from_standard_deviation,
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
    )
    from .observe_adm64_cross_scale_evidence import (
        DEFAULT_ALPHA,
        DEFAULT_HEAT_SHIFTS,
        DEFAULT_MAX_CONDITIONAL_KL,
        EVIDENCE_INTERNAL_TIMESTEPS,
        BaselineReference,
        ComponentSpec,
        _canonical_payload_sha,
        _close,
        _log_equal_weight_mixture,
        _mapping_manifest_record,
        build_component_specs,
        decoded_pixels,
        original_schedule_and_timestep_map,
        parse_float_spec,
        pixel_sha256,
    )
    from .reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        CLASSIFIER_SCALE,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        Protocol,
        atomic_json_dump,
        chunks,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        parse_int_spec,
        pixels_from_sample,
        protocol_from_args,
        sample_batch_invariant,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
        validate_output_set as validate_baseline_output_set,
    )
except ImportError:  # pragma: no cover - used by ``python experiments/...``.
    from adm64_path_evidence import (
        kl_tempered_score_mean_shift_from_standard_deviation,
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
    )
    from observe_adm64_cross_scale_evidence import (
        DEFAULT_ALPHA,
        DEFAULT_HEAT_SHIFTS,
        DEFAULT_MAX_CONDITIONAL_KL,
        EVIDENCE_INTERNAL_TIMESTEPS,
        BaselineReference,
        ComponentSpec,
        _canonical_payload_sha,
        _close,
        _log_equal_weight_mixture,
        _mapping_manifest_record,
        build_component_specs,
        decoded_pixels,
        original_schedule_and_timestep_map,
        parse_float_spec,
        pixel_sha256,
    )
    from reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        CLASSIFIER_SCALE,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        Protocol,
        atomic_json_dump,
        chunks,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        parse_int_spec,
        pixels_from_sample,
        protocol_from_args,
        sample_batch_invariant,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
        validate_output_set as validate_baseline_output_set,
    )


EXPERIMENT = "adm64_cross_scale_evidence_one_shot_rollback"
SCHEMA_VERSION = 1
# The action restores the exact pre-transition x_t whose innovation caused the
# crossing and redraws from t onward.  A nonzero lag is supported only as an
# explicitly predeclared exploratory sensitivity analysis.
DEFAULT_ROLLBACK_LAG_CHECKPOINTS = 0
BRANCH_RNG_NAMESPACE = "eqvae-adm64-cross-scale-rollback-branch-v1"
OUTPUT_ROLES = ("original_baseline", "intervention", "same_checkpoint_random_control")


@dataclass
class SegmentResult:
    final_state: torch.Tensor
    transitions: list[dict[str, Any]]
    evidence: dict[str, Any]
    checkpoint_states: dict[int, torch.Tensor]
    trace_arrays: dict[str, np.ndarray]
    gaussian_draws: int
    current_unet_evaluations: int
    shifted_unet_evaluations: int
    action_trigger: dict[str, Any] | None


@dataclass
class PairRunResult:
    original: SegmentResult
    intervention: SegmentResult | None
    random_control: SegmentResult | None
    action: dict[str, Any] | None
    rollback_state: torch.Tensor | None
    baseline_stream_seed: int
    intervention_stream_seed: int | None
    control_stream_seed: int | None


def branch_stream_seed(
    pair: Pair,
    branch_role: str,
    trigger_internal_timestep: int,
    rollback_internal_timestep: int,
) -> int:
    """Domain-separated deterministic seed for exactly one suffix draw stream."""

    if branch_role not in ("intervention", "same_checkpoint_random_control"):
        raise ValueError(f"invalid suffix branch role: {branch_role}")
    class_id, public_seed = pair
    payload = (
        f"{BRANCH_RNG_NAMESPACE}\0{branch_role}\0{class_id}\0{public_seed}\0"
        f"{trigger_internal_timestep}\0{rollback_internal_timestep}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _tensor_numpy(tensor: torch.Tensor, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    if dtype is not None:
        array = np.ascontiguousarray(array.astype(dtype, copy=False))
    return array


def _array_raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    array = _tensor_numpy(tensor)
    values = array.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("a recorded sampler tensor contains NaN or infinity")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_bytes_sha256": _array_raw_sha256(array),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "root_mean_square": float(np.sqrt(np.mean(np.square(values)))),
        "l2_norm": float(np.linalg.norm(values.reshape(-1))),
    }


def _new_evidence_state(
    components: Sequence[ComponentSpec],
    *,
    component_cumulative: Sequence[float] | None = None,
    component_running_max: Sequence[float] | None = None,
    component_first_crossing: Sequence[int | None] | None = None,
    mixture_running_max: float = 0.0,
    mixture_first_crossing: int | None = None,
) -> dict[str, Any]:
    count = len(components)
    cumulative = list(component_cumulative or [0.0] * count)
    running = list(component_running_max or [0.0] * count)
    first = list(component_first_crossing or [None] * count)
    if not (len(cumulative) == len(running) == len(first) == count):
        raise ValueError("initial evidence state has the wrong component count")
    return {
        "components": [
            {
                "component_index": component.index,
                "component_id": component.component_id,
                "additive_heat_shift": component.additive_heat_shift,
                "mixture_weight": component.mixture_weight,
                "prefix_cumulative_log_e": float(cumulative[index]),
                "prefix_running_max_log_e": float(running[index]),
                "prefix_first_crossing_checkpoint_index": first[index],
                "events": [],
                "cumulative": float(cumulative[index]),
                "running_max": float(running[index]),
                "first_crossing": first[index],
            }
            for index, component in enumerate(components)
        ],
        "mixture_events": [],
        "mixture_prefix_running_max_log_e": float(mixture_running_max),
        "mixture_prefix_first_crossing_checkpoint_index": mixture_first_crossing,
        "mixture_running_max": float(mixture_running_max),
        "mixture_first_crossing": mixture_first_crossing,
    }


def _finalize_evidence(
    state: dict[str, Any], components: Sequence[ComponentSpec]
) -> dict[str, Any]:
    component_records = []
    for record in state["components"]:
        component_records.append(
            {
                "component_index": record["component_index"],
                "component_id": record["component_id"],
                "additive_heat_shift": record["additive_heat_shift"],
                "mixture_weight": record["mixture_weight"],
                "prefix_cumulative_log_e": record["prefix_cumulative_log_e"],
                "prefix_running_max_log_e": record["prefix_running_max_log_e"],
                "prefix_first_crossing_checkpoint_index": record[
                    "prefix_first_crossing_checkpoint_index"
                ],
                "events": record["events"],
                "final_cumulative_log_e": record["cumulative"],
                "running_max_log_e": record["running_max"],
                "first_crossing_checkpoint_index": record["first_crossing"],
            }
        )
    final_logs = [float(record["final_cumulative_log_e"]) for record in component_records]
    return {
        "components": component_records,
        "mixture": {
            "weights": [component.mixture_weight for component in components],
            "prefix_running_max_log_e": state["mixture_prefix_running_max_log_e"],
            "prefix_first_crossing_checkpoint_index": state[
                "mixture_prefix_first_crossing_checkpoint_index"
            ],
            "events": state["mixture_events"],
            "final_log_e": _log_equal_weight_mixture(final_logs),
            "running_max_log_e": state["mixture_running_max"],
            "first_crossing_checkpoint_index": state["mixture_first_crossing"],
        },
    }


def _prefix_evidence_state(
    baseline_evidence: dict[str, Any],
    rollback_checkpoint_index: int,
    components: Sequence[ComponentSpec],
) -> dict[str, Any]:
    """Return evidence accumulated strictly before the rollback transition."""

    component_cumulative: list[float] = []
    component_running: list[float] = []
    component_first: list[int | None] = []
    for record in baseline_evidence["components"]:
        prior_events = [
            event
            for event in record["events"]
            if int(event["checkpoint_index"]) < rollback_checkpoint_index
        ]
        if prior_events:
            component_cumulative.append(float(prior_events[-1]["cumulative_log_e"]))
            component_running.append(float(prior_events[-1]["running_max_log_e"]))
        else:
            component_cumulative.append(0.0)
            component_running.append(0.0)
        crossing = record["first_crossing_checkpoint_index"]
        component_first.append(
            int(crossing) if crossing is not None and int(crossing) < rollback_checkpoint_index else None
        )
    mixture_prior = [
        event
        for event in baseline_evidence["mixture"]["events"]
        if int(event["checkpoint_index"]) < rollback_checkpoint_index
    ]
    mixture_running = (
        float(mixture_prior[-1]["running_max_log_e_mixture"]) if mixture_prior else 0.0
    )
    mixture_crossing = baseline_evidence["mixture"]["first_crossing_checkpoint_index"]
    mixture_first = (
        int(mixture_crossing)
        if mixture_crossing is not None and int(mixture_crossing) < rollback_checkpoint_index
        else None
    )
    return _new_evidence_state(
        components,
        component_cumulative=component_cumulative,
        component_running_max=component_running,
        component_first_crossing=component_first,
        mixture_running_max=mixture_running,
        mixture_first_crossing=mixture_first,
    )


def _run_segment(
    diffusion: Any,
    model: torch.nn.Module,
    cond_fn: Callable[..., torch.Tensor],
    pair: Pair,
    x_start: torch.Tensor,
    *,
    start_internal_timestep: int,
    generator: torch.Generator,
    generator_seed: int,
    gaussian_draws_before_segment: int,
    device: torch.device,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    evidence_state: dict[str, Any],
    action_policy: str,
    forced_checkpoint_index: int | None,
    save_tensor_trace: bool,
    checkpoints: Sequence[int] = EVIDENCE_INTERNAL_TIMESTEPS,
) -> SegmentResult:
    """Run one exact P segment and observe Q/P evidence without changing P."""

    if action_policy not in ("none", "natural_first_crossing", "oracle_forced_diagnostic"):
        raise ValueError(f"unknown action policy: {action_policy}")
    if action_policy == "oracle_forced_diagnostic" and forced_checkpoint_index is None:
        raise ValueError("forced diagnostic needs an explicit checkpoint index")
    if start_internal_timestep < 0 or start_internal_timestep >= diffusion.num_timesteps:
        raise ValueError("segment start timestep lies outside the diffusion")
    if x_start.shape[0] != 1:
        raise ValueError("all neural evaluations must remain singleton")

    class_id, _ = pair
    threshold = -math.log(alpha)
    checkpoint_to_index = {
        int(internal_t): index
        for index, internal_t in enumerate(checkpoints)
    }
    x = x_start.detach().clone()
    transitions: list[dict[str, Any]] = []
    checkpoint_states: dict[int, torch.Tensor] = {}
    action_trigger: dict[str, Any] | None = None
    gaussian_draws = int(gaussian_draws_before_segment)
    shifted_evaluations = 0
    current_evaluations = 0

    trace_states: list[np.ndarray] = []
    trace_checkpoint_indices: list[int] = []
    trace_epsilon_current: list[np.ndarray] = []
    trace_pred_xstart: list[np.ndarray] = []
    trace_p_standard_deviation: list[np.ndarray] = []
    trace_theta: list[np.ndarray] = []
    trace_checkpoint_innovation: list[np.ndarray] = []
    if save_tensor_trace:
        trace_states.append(_tensor_numpy(x[0]))

    for step_index, internal_t in enumerate(range(start_internal_timestep, -1, -1)):
        t = torch.tensor([internal_t], dtype=torch.long, device=device)
        y = torch.tensor([class_id], dtype=torch.long, device=device)
        model_kwargs = {"y": y}
        raw_box: list[torch.Tensor] = []
        raw_timestep_box: list[int] = []

        def capturing_model_fn(
            x_in: torch.Tensor,
            original_t: torch.Tensor,
            y: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if y is None:
                raise ValueError("class label y is required")
            raw = model(x_in, original_t, y)
            raw_box.append(raw)
            raw_timestep_box.append(int(original_t.item()))
            return raw

        with torch.no_grad():
            out = diffusion.p_mean_variance(
                capturing_model_fn,
                x,
                t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            current_evaluations += 1
            if len(raw_box) != 1:
                raise AssertionError("P p_mean_variance must make exactly one U-Net call")
            expected_original_t = int(timestep_map[internal_t])
            if raw_timestep_box != [expected_original_t]:
                raise AssertionError(
                    f"wrapped current timestep mismatch: {raw_timestep_box} != {expected_original_t}"
                )
            raw_current = raw_box[0]
            channels = x.shape[1]
            if raw_current.shape[1] != channels * 2:
                raise AssertionError("learned-variance ADM U-Net must return exactly 2*C channels")
            epsilon_current = raw_current[:, :channels]
            unguided_mean = out["mean"]
            guided_mean = diffusion.condition_mean(
                cond_fn,
                out,
                x,
                t,
                model_kwargs=model_kwargs,
            )
            classifier_mean_shift = guided_mean - unguided_mean
            p_standard_deviation = torch.exp(0.5 * out["log_variance"])

            checkpoint_index = checkpoint_to_index.get(internal_t)
            pending: list[tuple[Any, dict[str, Any], torch.Tensor]] = []
            if checkpoint_index is not None and internal_t > 0:
                # This is the precise pre-transition rollback state x_t.  It is
                # captured before the innovation that may cause a crossing.
                checkpoint_states[checkpoint_index] = x.detach().clone()
                for component in components:
                    mapping = component.mapping
                    current_original_t = int(mapping.current_timestep[checkpoint_index])
                    shifted_original_t = int(mapping.shifted_timestep[checkpoint_index])
                    if current_original_t != expected_original_t:
                        raise AssertionError("cross-scale map disagrees with the spaced timestep")
                    alpha_current = float(original_alpha_bar[current_original_t])
                    alpha_shifted = float(original_alpha_bar[shifted_original_t])
                    rho = math.sqrt(alpha_shifted / alpha_current)
                    if shifted_original_t == current_original_t:
                        epsilon_shifted = epsilon_current
                    else:
                        shifted_input = x * rho
                        shifted_t = torch.tensor(
                            [shifted_original_t], dtype=torch.long, device=device
                        )
                        shifted_raw = model(shifted_input, shifted_t, y)
                        shifted_evaluations += 1
                        if shifted_raw.shape[1] != channels * 2:
                            raise AssertionError("shifted U-Net must return exactly 2*C channels")
                        epsilon_shifted = shifted_raw[:, :channels]
                    theta = normalized_heat_score_pullback_difference(
                        epsilon_current,
                        epsilon_shifted,
                        torch.tensor([alpha_current], dtype=torch.float64, device=device),
                        torch.tensor([alpha_shifted], dtype=torch.float64, device=device),
                    )
                    tempered = kl_tempered_score_mean_shift_from_standard_deviation(
                        theta, p_standard_deviation, max_conditional_kl
                    )
                    event = {
                        "checkpoint_index": checkpoint_index,
                        "internal_timestep": internal_t,
                        "current_original_timestep": current_original_t,
                        "shifted_original_timestep": shifted_original_t,
                        "current_alpha_bar": alpha_current,
                        "shifted_alpha_bar": alpha_shifted,
                        "rho": rho,
                        "current_heat_variance": float(
                            mapping.current_heat_variance[checkpoint_index]
                        ),
                        "target_heat_variance": float(
                            mapping.target_heat_variance[checkpoint_index]
                        ),
                        "shifted_heat_variance": float(
                            mapping.shifted_heat_variance[checkpoint_index]
                        ),
                        "actual_heat_shift": float(mapping.actual_heat_shift[checkpoint_index]),
                        "absolute_mapping_error": float(
                            mapping.absolute_mapping_error[checkpoint_index]
                        ),
                        "shifted_model_evaluated": shifted_original_t != current_original_t,
                        "pre_transition_state_raw_bytes_sha256": _tensor_record(x)[
                            "raw_bytes_sha256"
                        ],
                        "epsilon_current": _tensor_record(epsilon_current),
                        "epsilon_shifted": _tensor_record(epsilon_shifted),
                        "theta": _tensor_record(theta),
                    }
                    pending.append((tempered, event, theta))

                if save_tensor_trace:
                    trace_checkpoint_indices.append(checkpoint_index)
                    trace_epsilon_current.append(_tensor_numpy(epsilon_current[0]))
                    trace_pred_xstart.append(_tensor_numpy(out["pred_xstart"][0]))
                    trace_p_standard_deviation.append(
                        _tensor_numpy(p_standard_deviation[0])
                    )
                    trace_theta.append(
                        np.stack(
                            [_tensor_numpy(theta[0]) for _, _, theta in pending], axis=0
                        )
                    )

            if internal_t > 0:
                noise = torch.randn(
                    tuple(x.shape),
                    generator=generator,
                    device=device,
                    dtype=x.dtype,
                )
                gaussian_draws += 1
                x_next = guided_mean + p_standard_deviation * noise
            else:
                noise = None
                x_next = guided_mean

            mixture_event: dict[str, Any] | None = None
            if pending:
                if noise is None or checkpoint_index is None:
                    raise AssertionError("an evidence checkpoint must be stochastic")
                if save_tensor_trace:
                    trace_checkpoint_innovation.append(_tensor_numpy(noise[0]))
                for component_index, (tempered, event, _) in enumerate(pending):
                    raw_increment = same_covariance_log_lr_from_noise(
                        tempered.raw_whitened_shift, noise
                    )
                    increment = same_covariance_log_lr_from_noise(
                        tempered.whitened_shift, noise
                    )
                    raw_projection = float(raw_increment.innovation_projection.item())
                    raw_kl = float(tempered.raw_kl.item())
                    if not _close(float(raw_increment.conditional_kl.item()), raw_kl):
                        raise AssertionError("raw KL does not match raw whitened shift")
                    value = float(increment.value.item())
                    projection = float(increment.innovation_projection.item())
                    applied_kl = float(increment.conditional_kl.item())
                    record = evidence_state["components"][component_index]
                    record["cumulative"] += value
                    record["running_max"] = max(record["running_max"], record["cumulative"])
                    crossed_now = record["cumulative"] >= threshold
                    if crossed_now and record["first_crossing"] is None:
                        record["first_crossing"] = checkpoint_index
                    event.update(
                        {
                            "raw_conditional_kl": raw_kl,
                            "raw_innovation_projection": raw_projection,
                            "tempering_scale": float(tempered.scale.item()),
                            "applied_conditional_kl": applied_kl,
                            "max_conditional_kl": max_conditional_kl,
                            "innovation_projection": projection,
                            "log_lr_increment": value,
                            "cumulative_log_e": record["cumulative"],
                            "running_max_log_e": record["running_max"],
                            "crossed_threshold_at_checkpoint": crossed_now,
                            "crossed_threshold_ever": record["first_crossing"] is not None,
                            "trigger_causing_innovation": _tensor_record(noise),
                            "post_transition_state_raw_bytes_sha256": _tensor_record(x_next)[
                                "raw_bytes_sha256"
                            ],
                        }
                    )
                    record["events"].append(event)

                component_logs = [
                    float(record["cumulative"])
                    for record in evidence_state["components"]
                ]
                mixture_log_e = _log_equal_weight_mixture(component_logs)
                evidence_state["mixture_running_max"] = max(
                    evidence_state["mixture_running_max"], mixture_log_e
                )
                mixture_crossed = mixture_log_e >= threshold
                if mixture_crossed and evidence_state["mixture_first_crossing"] is None:
                    evidence_state["mixture_first_crossing"] = checkpoint_index
                mixture_event = {
                    "checkpoint_index": checkpoint_index,
                    "internal_timestep": internal_t,
                    "log_e_mixture": mixture_log_e,
                    "running_max_log_e_mixture": evidence_state["mixture_running_max"],
                    "crossed_threshold_at_checkpoint": mixture_crossed,
                    "crossed_threshold_ever": evidence_state["mixture_first_crossing"] is not None,
                }
                evidence_state["mixture_events"].append(mixture_event)

            state_before_record = _tensor_record(x)
            state_after_record = _tensor_record(x_next)
            innovation_record = _tensor_record(noise) if noise is not None else None
            transition = {
                "segment_step_index": step_index,
                "internal_timestep": internal_t,
                "original_timestep": expected_original_t,
                "stochastic": internal_t > 0,
                "generator_seed": generator_seed,
                "gaussian_draw_ordinal_in_stream": gaussian_draws if noise is not None else None,
                "state_before": state_before_record,
                "epsilon_current": _tensor_record(epsilon_current),
                "predicted_xstart": _tensor_record(out["pred_xstart"]),
                "unguided_p_mean": _tensor_record(unguided_mean),
                "classifier_mean_shift": _tensor_record(classifier_mean_shift),
                "guided_p_mean": _tensor_record(guided_mean),
                "p_standard_deviation": _tensor_record(p_standard_deviation),
                "innovation": innovation_record,
                "state_after": state_after_record,
                "evidence_checkpoint_index": checkpoint_index,
                "evidence_evaluated": mixture_event is not None,
                "mixture_log_e_after_transition": (
                    float(mixture_event["log_e_mixture"])
                    if mixture_event is not None
                    else None
                ),
            }
            transitions.append(transition)

            should_act = False
            trigger_kind: str | None = None
            if action_trigger is None and checkpoint_index is not None:
                if (
                    action_policy == "natural_first_crossing"
                    and mixture_event is not None
                    and evidence_state["mixture_first_crossing"] == checkpoint_index
                    and bool(mixture_event["crossed_threshold_at_checkpoint"])
                ):
                    should_act = True
                    trigger_kind = "anytime_valid_mixture_first_crossing"
                elif (
                    action_policy == "oracle_forced_diagnostic"
                    and checkpoint_index == forced_checkpoint_index
                ):
                    should_act = True
                    trigger_kind = "oracle_forced_diagnostic_excluded_from_method_claims"
            if should_act:
                if mixture_event is None or trigger_kind is None:
                    raise AssertionError("action must be attached to an evidence transition")
                action_trigger = {
                    "trigger_kind": trigger_kind,
                    "checkpoint_index": checkpoint_index,
                    "internal_timestep": internal_t,
                    "original_timestep": expected_original_t,
                    "mixture_log_e_after_transition": float(
                        mixture_event["log_e_mixture"]
                    ),
                    "log_e_crossing_threshold": threshold,
                    "threshold_crossed_at_trigger": bool(
                        mixture_event["crossed_threshold_at_checkpoint"]
                    ),
                    "natural_first_crossing_checkpoint_index": evidence_state[
                        "mixture_first_crossing"
                    ],
                    "pre_transition_state_raw_bytes_sha256": state_before_record[
                        "raw_bytes_sha256"
                    ],
                    "trigger_causing_innovation": innovation_record,
                    "post_transition_state_raw_bytes_sha256": state_after_record[
                        "raw_bytes_sha256"
                    ],
                    "transition_segment_step_index": step_index,
                }

        x = x_next.detach()
        if save_tensor_trace:
            trace_states.append(_tensor_numpy(x[0]))

    trace_arrays: dict[str, np.ndarray] = {}
    if save_tensor_trace:
        trace_arrays = {
            "states": np.stack(trace_states, axis=0),
            "internal_timesteps": np.arange(
                start_internal_timestep, -1, -1, dtype=np.int16
            ),
            "evidence_checkpoint_indices": np.asarray(
                trace_checkpoint_indices, dtype=np.int16
            ),
            "checkpoint_epsilon_current": np.stack(trace_epsilon_current, axis=0),
            "checkpoint_predicted_xstart": np.stack(trace_pred_xstart, axis=0),
            "checkpoint_p_standard_deviation": np.stack(
                trace_p_standard_deviation, axis=0
            ),
            "checkpoint_theta": np.stack(trace_theta, axis=0),
            "checkpoint_innovation": np.stack(
                trace_checkpoint_innovation, axis=0
            ),
        }
        if trace_arrays["states"].shape[0] != len(transitions) + 1:
            raise AssertionError("state trace must include segment start and every endpoint")

    return SegmentResult(
        final_state=x,
        transitions=transitions,
        evidence=_finalize_evidence(evidence_state, components),
        checkpoint_states=checkpoint_states,
        trace_arrays=trace_arrays,
        gaussian_draws=gaussian_draws,
        current_unet_evaluations=current_evaluations,
        shifted_unet_evaluations=shifted_evaluations,
        action_trigger=action_trigger,
    )


def sample_intervention_pair(
    diffusion: Any,
    model: torch.nn.Module,
    cond_fn: Callable[..., torch.Tensor],
    pair: Pair,
    *,
    device: torch.device,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    rollback_lag_checkpoints: int,
    force_intervention: bool,
    force_internal_timestep: int | None,
    save_tensor_trace: bool,
    checkpoints: Sequence[int] = EVIDENCE_INTERNAL_TIMESTEPS,
    channels: int = 3,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> PairRunResult:
    """Generate the original path and at most one pair of matched P suffixes."""

    if diffusion.num_timesteps != len(timestep_map):
        raise ValueError("diffusion and supplied timestep map disagree")
    if tuple(checkpoints) != tuple(sorted(checkpoints, reverse=True)):
        raise ValueError("evidence checkpoints must be in reverse order")
    if not checkpoints or any(t <= 0 or t >= diffusion.num_timesteps for t in checkpoints):
        raise ValueError("all evidence checkpoints must be stochastic internal timesteps")
    if rollback_lag_checkpoints < 0:
        raise ValueError("rollback lag cannot be negative")
    if force_intervention:
        if force_internal_timestep is None:
            raise ValueError(
                "--force-intervention requires an explicit --force-internal-timestep"
            )
        if force_internal_timestep not in checkpoints:
            raise ValueError("forced timestep must be one of the predeclared evidence checkpoints")
        forced_checkpoint_index = tuple(checkpoints).index(force_internal_timestep)
        action_policy = "oracle_forced_diagnostic"
    else:
        if force_internal_timestep is not None:
            raise ValueError(
                "--force-internal-timestep is forbidden unless --force-intervention is set"
            )
        forced_checkpoint_index = None
        action_policy = "natural_first_crossing"

    baseline_seed = sample_stream_seed(pair[1])
    baseline_generator = torch.Generator(device=device).manual_seed(baseline_seed)
    initial_state = torch.randn(
        (1, channels, image_size, image_size),
        generator=baseline_generator,
        device=device,
        dtype=dtype,
    )
    baseline = _run_segment(
        diffusion,
        model,
        cond_fn,
        pair,
        initial_state,
        start_internal_timestep=diffusion.num_timesteps - 1,
        generator=baseline_generator,
        generator_seed=baseline_seed,
        gaussian_draws_before_segment=1,
        device=device,
        original_alpha_bar=original_alpha_bar,
        timestep_map=timestep_map,
        components=components,
        max_conditional_kl=max_conditional_kl,
        alpha=alpha,
        evidence_state=_new_evidence_state(components),
        action_policy=action_policy,
        forced_checkpoint_index=forced_checkpoint_index,
        save_tensor_trace=save_tensor_trace,
        checkpoints=checkpoints,
    )
    if baseline.gaussian_draws != diffusion.num_timesteps:
        raise AssertionError(
            "baseline must use one initial draw and one draw for every stochastic transition"
        )

    trigger = baseline.action_trigger
    if trigger is None:
        # This is a literal no-op: no suffix RNG is constructed or consumed.
        return PairRunResult(
            original=baseline,
            intervention=None,
            random_control=None,
            action=None,
            rollback_state=None,
            baseline_stream_seed=baseline_seed,
            intervention_stream_seed=None,
            control_stream_seed=None,
        )

    trigger_index = int(trigger["checkpoint_index"])
    rollback_index = max(0, trigger_index - rollback_lag_checkpoints)
    rollback_internal_t = int(checkpoints[rollback_index])
    if rollback_index not in baseline.checkpoint_states:
        raise AssertionError("the predeclared rollback state was not retained")
    rollback_state = baseline.checkpoint_states[rollback_index].detach().clone()
    rollback_record = _tensor_record(rollback_state)
    if rollback_lag_checkpoints == 0 and rollback_record["raw_bytes_sha256"] != trigger[
        "pre_transition_state_raw_bytes_sha256"
    ]:
        raise AssertionError("lag-zero rollback did not restore the trigger transition's exact x_t")

    prefix_state = _prefix_evidence_state(
        baseline.evidence, rollback_index, components
    )
    intervention_seed = branch_stream_seed(
        pair,
        "intervention",
        int(trigger["internal_timestep"]),
        rollback_internal_t,
    )
    control_seed = branch_stream_seed(
        pair,
        "same_checkpoint_random_control",
        int(trigger["internal_timestep"]),
        rollback_internal_t,
    )
    if len({baseline_seed, intervention_seed, control_seed}) != 3:
        raise AssertionError("baseline/intervention/control RNG domains collided")

    intervention_generator = torch.Generator(device=device).manual_seed(intervention_seed)
    intervention = _run_segment(
        diffusion,
        model,
        cond_fn,
        pair,
        rollback_state,
        start_internal_timestep=rollback_internal_t,
        generator=intervention_generator,
        generator_seed=intervention_seed,
        gaussian_draws_before_segment=0,
        device=device,
        original_alpha_bar=original_alpha_bar,
        timestep_map=timestep_map,
        components=components,
        max_conditional_kl=max_conditional_kl,
        alpha=alpha,
        evidence_state=prefix_state,
        action_policy="none",
        forced_checkpoint_index=None,
        save_tensor_trace=save_tensor_trace,
        checkpoints=checkpoints,
    )

    # Construct a fresh prefix state: _run_segment mutates its evidence state.
    control_prefix_state = _prefix_evidence_state(
        baseline.evidence, rollback_index, components
    )
    control_generator = torch.Generator(device=device).manual_seed(control_seed)
    random_control = _run_segment(
        diffusion,
        model,
        cond_fn,
        pair,
        rollback_state,
        start_internal_timestep=rollback_internal_t,
        generator=control_generator,
        generator_seed=control_seed,
        gaussian_draws_before_segment=0,
        device=device,
        original_alpha_bar=original_alpha_bar,
        timestep_map=timestep_map,
        components=components,
        max_conditional_kl=max_conditional_kl,
        alpha=alpha,
        evidence_state=control_prefix_state,
        action_policy="none",
        forced_checkpoint_index=None,
        save_tensor_trace=save_tensor_trace,
        checkpoints=checkpoints,
    )

    same_compute_fields = (
        "gaussian_draws",
        "current_unet_evaluations",
        "shifted_unet_evaluations",
    )
    for field in same_compute_fields:
        if getattr(intervention, field) != getattr(random_control, field):
            raise AssertionError(f"same-compute control mismatch in {field}")
    if len(intervention.transitions) != len(random_control.transitions):
        raise AssertionError("same-compute suffix transition counts differ")
    expected_suffix_draws = rollback_internal_t
    if intervention.gaussian_draws != expected_suffix_draws:
        raise AssertionError(
            f"suffix from t={rollback_internal_t} must draw {expected_suffix_draws} innovations"
        )

    action = {
        **trigger,
        "rollback_policy": (
            "restore the pre-transition state at trigger checkpoint and redraw from t onward"
            if rollback_lag_checkpoints == 0
            else "predeclared earlier-checkpoint sensitivity analysis"
        ),
        "rollback_lag_checkpoints": rollback_lag_checkpoints,
        "rollback_checkpoint_index": rollback_index,
        "rollback_internal_timestep": rollback_internal_t,
        "rollback_state": rollback_record,
        "intervention_stream_seed": intervention_seed,
        "control_stream_seed": control_seed,
        "one_shot_no_loop": True,
        "same_compute_control_scope": (
            "independent P suffix from the identical evidence-selected rollback state; "
            "controls retry randomness but not trigger selection"
        ),
        "method_claim_eligible": not force_intervention,
    }
    return PairRunResult(
        original=baseline,
        intervention=intervention,
        random_control=random_control,
        action=action,
        rollback_state=rollback_state,
        baseline_stream_seed=baseline_seed,
        intervention_stream_seed=intervention_seed,
        control_stream_seed=control_seed,
    )


def output_image_path(output_dir: Path, role: str, pair: Pair) -> Path:
    if role not in OUTPUT_ROLES:
        raise ValueError(f"unknown output role: {role}")
    class_id, seed = pair
    return output_dir / "images" / role / f"class_{class_id:04d}" / f"{seed:019d}.png"


def signal_path(output_dir: Path, pair: Pair) -> Path:
    class_id, seed = pair
    return output_dir / "signals" / f"class_{class_id:04d}" / f"{seed:019d}.json"


def trace_path(output_dir: Path, role: str, pair: Pair) -> Path:
    if role not in OUTPUT_ROLES:
        raise ValueError(f"unknown trace role: {role}")
    class_id, seed = pair
    return output_dir / "traces" / role / f"class_{class_id:04d}" / f"{seed:019d}.npz"


def rollback_state_path(output_dir: Path, pair: Pair) -> Path:
    class_id, seed = pair
    return output_dir / "rollback_states" / f"class_{class_id:04d}" / f"{seed:019d}.npy"


def _assert_absent(paths: Sequence[Path], pair: Pair) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing/partial output for {pair}: {existing[0]}")


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite tensor trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npy_dump(array: np.ndarray, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite rollback state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trace_file_record(
    path: Path, arrays: dict[str, np.ndarray], output_dir: Path
) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(output_dir).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "keys": sorted(arrays),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "raw_bytes_sha256": _array_raw_sha256(value),
            }
            for key, value in sorted(arrays.items())
        },
    }


def _save_output_png(
    pixels: np.ndarray,
    path: Path,
    pair: Pair,
    role: str,
    *,
    manifest_identity_sha256: str,
    runner_sha256: str,
    signal_payload_sha256: str,
    original_baseline_pixel_sha256: str,
) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite output image: {path}")
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"unexpected output pixels: {pixels.shape}/{pixels.dtype}")
    digest = pixel_sha256(pixels)
    metadata = PngInfo()
    fields = {
        "experiment": EXPERIMENT,
        "output_role": role,
        "class_id": str(pair[0]),
        "seed": str(pair[1]),
        "baseline_sample_stream_seed": str(sample_stream_seed(pair[1])),
        "pixel_sha256": digest,
        "original_baseline_pixel_sha256": original_baseline_pixel_sha256,
        "signal_payload_sha256": signal_payload_sha256,
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
    }
    for key, value in fields.items():
        metadata.add_text(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def _segment_signal_record(
    result: SegmentResult,
    *,
    execution: str,
    stream_seed: int,
    includes_initial_latent_draw: bool,
    trace_record: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "execution": execution,
        "stream_seed": stream_seed,
        "includes_initial_latent_draw": includes_initial_latent_draw,
        "gaussian_draws_in_stream": result.gaussian_draws,
        "transition_count": len(result.transitions),
        "current_unet_evaluations": result.current_unet_evaluations,
        "shifted_unet_evaluations": result.shifted_unet_evaluations,
        "initial_state": result.transitions[0]["state_before"],
        "final_state": result.transitions[-1]["state_after"],
        "transitions": result.transitions,
        "evidence": result.evidence,
        "tensor_trace": trace_record,
    }


def save_pair_bundle(
    output_dir: Path,
    pair: Pair,
    result: PairRunResult,
    original_pixels: np.ndarray,
    intervention_pixels: np.ndarray,
    control_pixels: np.ndarray,
    baseline_pixels: np.ndarray,
    *,
    manifest_identity_sha256: str,
    runner_sha256: str,
    baseline_manifest_identity_sha256: str,
    alpha: float,
    alpha_role: str,
    max_conditional_kl: float,
    rollback_lag_checkpoints: int,
    force_intervention: bool,
    force_internal_timestep: int | None,
    save_tensor_trace: bool,
) -> None:
    """Atomically write one fail-closed bundle; signal JSON is the final commit marker."""

    image_paths = {
        role: output_image_path(output_dir, role, pair) for role in OUTPUT_ROLES
    }
    pair_signal_path = signal_path(output_dir, pair)
    original_trace_path = trace_path(output_dir, "original_baseline", pair)
    intervention_trace_path = trace_path(output_dir, "intervention", pair)
    control_trace_path = trace_path(output_dir, "same_checkpoint_random_control", pair)
    state_path = rollback_state_path(output_dir, pair)
    all_possible = [
        *image_paths.values(),
        pair_signal_path,
        original_trace_path,
        intervention_trace_path,
        control_trace_path,
        state_path,
    ]
    _assert_absent(all_possible, pair)

    baseline_sha = pixel_sha256(baseline_pixels)
    original_sha = pixel_sha256(original_pixels)
    intervention_sha = pixel_sha256(intervention_pixels)
    control_sha = pixel_sha256(control_pixels)
    if original_sha != baseline_sha or not np.array_equal(original_pixels, baseline_pixels):
        raise RuntimeError("instrumented original P endpoint differs from frozen baseline")

    trace_records: dict[str, dict[str, Any] | None] = {
        role: None for role in OUTPUT_ROLES
    }
    if save_tensor_trace:
        _atomic_npz_dump(result.original.trace_arrays, original_trace_path)
        trace_records["original_baseline"] = _trace_file_record(
            original_trace_path, result.original.trace_arrays, output_dir
        )
        if result.action is not None:
            if result.intervention is None or result.random_control is None:
                raise AssertionError("triggered output lacks a suffix branch")
            _atomic_npz_dump(result.intervention.trace_arrays, intervention_trace_path)
            _atomic_npz_dump(result.random_control.trace_arrays, control_trace_path)
            trace_records["intervention"] = _trace_file_record(
                intervention_trace_path, result.intervention.trace_arrays, output_dir
            )
            trace_records["same_checkpoint_random_control"] = _trace_file_record(
                control_trace_path, result.random_control.trace_arrays, output_dir
            )

    rollback_file_record: dict[str, Any] | None = None
    if result.action is not None:
        if result.rollback_state is None:
            raise AssertionError("triggered output lacks its rollback tensor")
        rollback_array = _tensor_numpy(result.rollback_state[0], dtype=np.float32)
        _atomic_npy_dump(rollback_array, state_path)
        rollback_file_record = {
            "relative_path": state_path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(state_path),
            "bytes": state_path.stat().st_size,
            "shape": list(rollback_array.shape),
            "dtype": str(rollback_array.dtype),
            "raw_bytes_sha256": _array_raw_sha256(rollback_array),
        }
        if rollback_file_record["raw_bytes_sha256"] != result.action["rollback_state"][
            "raw_bytes_sha256"
        ]:
            raise AssertionError("saved rollback tensor differs from the action state")

    original_record = _segment_signal_record(
        result.original,
        execution="complete_original_p_path_with_read_only_evidence",
        stream_seed=result.baseline_stream_seed,
        includes_initial_latent_draw=True,
        trace_record=trace_records["original_baseline"],
    )
    if result.action is None:
        if result.intervention is not None or result.random_control is not None:
            raise AssertionError("no-crossing path unexpectedly evaluated a suffix")
        if not (
            np.array_equal(original_pixels, intervention_pixels)
            and np.array_equal(original_pixels, control_pixels)
        ):
            raise AssertionError("no-crossing outputs must be exact aliases of baseline pixels")
        intervention_record: dict[str, Any] = {
            "execution": "no_crossing_bitwise_alias_no_compute",
            "stream_seed": None,
            "gaussian_draws_in_stream": 0,
            "transition_count": 0,
            "current_unet_evaluations": 0,
            "shifted_unet_evaluations": 0,
            "transitions": [],
            "evidence": None,
            "tensor_trace": {
                "alias_of": "original_baseline",
                "no_separate_file": True,
            },
        }
        control_record = dict(intervention_record)
    else:
        if result.intervention is None or result.random_control is None:
            raise AssertionError("triggered pair is missing a suffix")
        intervention_record = _segment_signal_record(
            result.intervention,
            execution="one_independent_p_suffix_from_rollback_state",
            stream_seed=int(result.intervention_stream_seed),
            includes_initial_latent_draw=False,
            trace_record=trace_records["intervention"],
        )
        control_record = _segment_signal_record(
            result.random_control,
            execution="same_checkpoint_independent_p_suffix_control",
            stream_seed=int(result.control_stream_seed),
            includes_initial_latent_draw=False,
            trace_record=trace_records["same_checkpoint_random_control"],
        )

    saved_action = dict(result.action) if result.action is not None else None
    claim_eligible = (
        not force_intervention and alpha_role == "primary_anytime_valid_alpha_0.05"
    )
    if saved_action is not None:
        saved_action["method_claim_eligible"] = claim_eligible
        saved_action["mechanics_only_posthoc_exploration"] = not claim_eligible
    signal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "class_id": pair[0],
        "seed": pair[1],
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
        "baseline_manifest_identity_sha256": baseline_manifest_identity_sha256,
        "alpha": alpha,
        "alpha_role": alpha_role,
        "log_e_crossing_threshold": -math.log(alpha),
        "max_conditional_kl": max_conditional_kl,
        "rollback_lag_checkpoints": rollback_lag_checkpoints,
        "force_intervention": force_intervention,
        "force_internal_timestep": force_internal_timestep,
        "method_claim_eligible": claim_eligible,
        "mechanics_only_posthoc_exploration": not claim_eligible,
        "intervention_count": int(result.action is not None),
        "retry_limit": 1,
        "looping_or_endpoint_selection": False,
        "internal_timesteps": list(EVIDENCE_INTERNAL_TIMESTEPS),
        "action": saved_action,
        "rollback_state_file": rollback_file_record,
        "rng": {
            "baseline_stream_seed": result.baseline_stream_seed,
            "intervention_stream_seed": result.intervention_stream_seed,
            "same_checkpoint_random_control_stream_seed": result.control_stream_seed,
            "branch_seed_namespace": BRANCH_RNG_NAMESPACE,
        },
        "paths": {
            "original_baseline": original_record,
            "intervention": intervention_record,
            "same_checkpoint_random_control": control_record,
        },
        "outputs": {
            "original_baseline": {
                "relative_path": image_paths["original_baseline"].relative_to(output_dir).as_posix(),
                "pixel_sha256": original_sha,
            },
            "intervention": {
                "relative_path": image_paths["intervention"].relative_to(output_dir).as_posix(),
                "pixel_sha256": intervention_sha,
            },
            "same_checkpoint_random_control": {
                "relative_path": image_paths["same_checkpoint_random_control"].relative_to(output_dir).as_posix(),
                "pixel_sha256": control_sha,
            },
            "frozen_baseline_pixel_sha256": baseline_sha,
        },
    }
    signal["payload_sha256"] = _canonical_payload_sha(signal, "payload_sha256")

    for role, pixels in (
        ("original_baseline", original_pixels),
        ("intervention", intervention_pixels),
        ("same_checkpoint_random_control", control_pixels),
    ):
        _save_output_png(
            pixels,
            image_paths[role],
            pair,
            role,
            manifest_identity_sha256=manifest_identity_sha256,
            runner_sha256=runner_sha256,
            signal_payload_sha256=signal["payload_sha256"],
            original_baseline_pixel_sha256=baseline_sha,
        )
    atomic_json_dump(signal, pair_signal_path)


def _require_tensor_record(record: Any, context: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"missing tensor record: {context}")
    required = {
        "shape", "dtype", "raw_bytes_sha256", "minimum", "maximum", "mean",
        "standard_deviation", "root_mean_square", "l2_norm",
    }
    if set(record) != required:
        raise RuntimeError(f"tensor-record fields changed at {context}")
    if not isinstance(record["shape"], list) or not isinstance(record["dtype"], str):
        raise RuntimeError(f"invalid tensor identity at {context}")
    digest = record["raw_bytes_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"invalid tensor hash at {context}")
    for key in required - {"shape", "dtype", "raw_bytes_sha256"}:
        if not isinstance(record[key], (int, float)) or not math.isfinite(float(record[key])):
            raise RuntimeError(f"non-finite tensor diagnostic {context}/{key}")


def _validate_transition_record(
    path_record: dict[str, Any],
    *,
    start_internal_timestep: int,
    stream_seed: int,
    includes_initial_latent_draw: bool,
    checkpoints: Sequence[int],
) -> None:
    transitions = path_record.get("transitions")
    expected_count = start_internal_timestep + 1
    if not isinstance(transitions, list) or len(transitions) != expected_count:
        raise RuntimeError("wrong number of recorded P transitions")
    if path_record.get("transition_count") != expected_count:
        raise RuntimeError("transition accounting disagrees with transition list")
    expected_draws = start_internal_timestep + int(includes_initial_latent_draw)
    if path_record.get("gaussian_draws_in_stream") != expected_draws:
        raise RuntimeError("Gaussian draw accounting is invalid")
    if path_record.get("stream_seed") != stream_seed:
        raise RuntimeError("path RNG seed is invalid")
    previous_after: str | None = None
    checkpoint_set = set(checkpoints)
    draw_ordinal = int(includes_initial_latent_draw)
    for segment_step_index, (expected_t, transition) in enumerate(
        zip(range(start_internal_timestep, -1, -1), transitions)
    ):
        fixed = {
            "segment_step_index": segment_step_index,
            "internal_timestep": expected_t,
            "stochastic": expected_t > 0,
            "generator_seed": stream_seed,
            "evidence_evaluated": expected_t in checkpoint_set,
            "evidence_checkpoint_index": (
                tuple(checkpoints).index(expected_t) if expected_t in checkpoint_set else None
            ),
        }
        if any(transition.get(key) != value for key, value in fixed.items()):
            raise RuntimeError(f"transition identity mismatch at internal t={expected_t}")
        for tensor_key in (
            "state_before", "epsilon_current", "predicted_xstart", "unguided_p_mean",
            "classifier_mean_shift", "guided_p_mean", "p_standard_deviation", "state_after",
        ):
            _require_tensor_record(transition.get(tensor_key), f"t={expected_t}/{tensor_key}")
        before_hash = transition["state_before"]["raw_bytes_sha256"]
        after_hash = transition["state_after"]["raw_bytes_sha256"]
        if previous_after is not None and before_hash != previous_after:
            raise RuntimeError(f"P state-chain hash broke before internal t={expected_t}")
        previous_after = after_hash
        if expected_t > 0:
            draw_ordinal += 1
            if transition.get("gaussian_draw_ordinal_in_stream") != draw_ordinal:
                raise RuntimeError(f"Gaussian draw ordinal mismatch at internal t={expected_t}")
            _require_tensor_record(transition.get("innovation"), f"t={expected_t}/innovation")
        elif transition.get("gaussian_draw_ordinal_in_stream") is not None or transition.get(
            "innovation"
        ) is not None:
            raise RuntimeError("deterministic final transition must not draw noise")
        if expected_t not in checkpoint_set and transition.get(
            "mixture_log_e_after_transition"
        ) is not None:
            raise RuntimeError("noncheckpoint transition unexpectedly changed evidence")


def _validate_evidence_record(
    evidence: dict[str, Any],
    components: Sequence[ComponentSpec],
    *,
    expected_checkpoint_indices: Sequence[int],
    max_conditional_kl: float,
    alpha: float,
    transition_by_t: dict[int, dict[str, Any]],
    checkpoints: Sequence[int],
) -> None:
    threshold = -math.log(alpha)
    records = evidence.get("components")
    if not isinstance(records, list) or len(records) != len(components):
        raise RuntimeError("wrong evidence component count")
    cumulative_by_event: list[list[float]] = [
        [0.0] * len(components) for _ in expected_checkpoint_indices
    ]
    for component, record in zip(components, records):
        identity = {
            "component_index": component.index,
            "component_id": component.component_id,
            "additive_heat_shift": component.additive_heat_shift,
            "mixture_weight": component.mixture_weight,
        }
        if any(record.get(key) != value for key, value in identity.items()):
            raise RuntimeError("evidence component identity changed")
        prefix_cumulative = float(record.get("prefix_cumulative_log_e", math.nan))
        prefix_running = float(record.get("prefix_running_max_log_e", math.nan))
        prefix_first = record.get("prefix_first_crossing_checkpoint_index")
        if not math.isfinite(prefix_cumulative) or not math.isfinite(prefix_running):
            raise RuntimeError("non-finite prefix evidence")
        events = record.get("events")
        if not isinstance(events, list) or len(events) != len(expected_checkpoint_indices):
            raise RuntimeError("wrong evidence event count")
        cumulative = prefix_cumulative
        running = prefix_running
        first_crossing = prefix_first
        for local_index, (checkpoint_index, event) in enumerate(
            zip(expected_checkpoint_indices, events)
        ):
            internal_t = int(checkpoints[checkpoint_index])
            mapping = component.mapping
            fixed = {
                "checkpoint_index": checkpoint_index,
                "internal_timestep": internal_t,
                "current_original_timestep": int(mapping.current_timestep[checkpoint_index]),
                "shifted_original_timestep": int(mapping.shifted_timestep[checkpoint_index]),
            }
            if any(event.get(key) != value for key, value in fixed.items()):
                raise RuntimeError("evidence checkpoint mapping identity changed")
            expected_mapping_values = {
                "current_heat_variance": float(
                    mapping.current_heat_variance[checkpoint_index]
                ),
                "target_heat_variance": float(
                    mapping.target_heat_variance[checkpoint_index]
                ),
                "shifted_heat_variance": float(
                    mapping.shifted_heat_variance[checkpoint_index]
                ),
                "actual_heat_shift": float(mapping.actual_heat_shift[checkpoint_index]),
                "absolute_mapping_error": float(
                    mapping.absolute_mapping_error[checkpoint_index]
                ),
            }
            if any(
                not _close(float(event.get(key, math.nan)), expected)
                for key, expected in expected_mapping_values.items()
            ):
                raise RuntimeError("evidence heat-coordinate mapping values changed")
            alpha_current = 1.0 / (
                1.0 + float(mapping.current_heat_variance[checkpoint_index])
            )
            alpha_shifted = 1.0 / (
                1.0 + float(mapping.shifted_heat_variance[checkpoint_index])
            )
            rho = math.sqrt(alpha_shifted / alpha_current)
            if not _close(
                float(event.get("current_alpha_bar", math.nan)), alpha_current
            ) or not _close(
                float(event.get("shifted_alpha_bar", math.nan)), alpha_shifted
            ) or not _close(float(event.get("rho", math.nan)), rho):
                raise RuntimeError("evidence alpha/rho values changed")
            transition = transition_by_t[internal_t]
            if event.get("pre_transition_state_raw_bytes_sha256") != transition[
                "state_before"
            ]["raw_bytes_sha256"]:
                raise RuntimeError("evidence event is attached to the wrong pre-transition state")
            if event.get("post_transition_state_raw_bytes_sha256") != transition[
                "state_after"
            ]["raw_bytes_sha256"]:
                raise RuntimeError("evidence event is attached to the wrong post-transition state")
            event_noise = event.get("trigger_causing_innovation")
            _require_tensor_record(event_noise, "evidence/innovation")
            if event_noise["raw_bytes_sha256"] != transition["innovation"][
                "raw_bytes_sha256"
            ]:
                raise RuntimeError("evidence likelihood used the wrong P innovation")
            for tensor_key in ("epsilon_current", "epsilon_shifted", "theta"):
                _require_tensor_record(event.get(tensor_key), f"evidence/{tensor_key}")
            numeric_keys = (
                "current_alpha_bar", "shifted_alpha_bar", "rho",
                "current_heat_variance", "target_heat_variance", "shifted_heat_variance",
                "actual_heat_shift", "absolute_mapping_error", "raw_conditional_kl",
                "raw_innovation_projection", "tempering_scale", "applied_conditional_kl",
                "max_conditional_kl", "innovation_projection", "log_lr_increment",
                "cumulative_log_e", "running_max_log_e",
            )
            if any(
                not isinstance(event.get(key), (int, float))
                or not math.isfinite(float(event[key]))
                for key in numeric_keys
            ):
                raise RuntimeError("evidence event has missing/non-finite arithmetic")
            raw_kl = float(event["raw_conditional_kl"])
            scale = float(event["tempering_scale"])
            applied_kl = float(event["applied_conditional_kl"])
            raw_projection = float(event["raw_innovation_projection"])
            projection = float(event["innovation_projection"])
            increment = float(event["log_lr_increment"])
            if raw_kl < -2e-14 or applied_kl < -2e-14 or applied_kl > max_conditional_kl + 2e-11:
                raise RuntimeError("evidence KL cap is invalid")
            if not 0 < scale <= 1.0 + 1e-14:
                raise RuntimeError("evidence tempering scale is invalid")
            identities = (
                _close(applied_kl, raw_kl * scale * scale),
                _close(projection, raw_projection * scale),
                _close(increment, projection - applied_kl),
                _close(increment, scale * raw_projection - scale * scale * raw_kl),
                _close(float(event["max_conditional_kl"]), max_conditional_kl),
            )
            if not all(identities):
                raise RuntimeError("evidence LR/KL arithmetic identity failed")
            current_t = int(mapping.current_timestep[checkpoint_index])
            shifted_t = int(mapping.shifted_timestep[checkpoint_index])
            if event.get("shifted_model_evaluated") != (shifted_t != current_t):
                raise RuntimeError("shifted model-evaluation flag is invalid")
            if shifted_t == current_t and (abs(raw_kl) > 2e-14 or abs(raw_projection) > 2e-14):
                raise RuntimeError("identity heat mapping must define Q=P")
            cumulative += increment
            running = max(running, cumulative)
            crossed = cumulative >= threshold
            if first_crossing is None and crossed:
                first_crossing = checkpoint_index
            if not _close(float(event["cumulative_log_e"]), cumulative) or not _close(
                float(event["running_max_log_e"]), running
            ):
                raise RuntimeError("cumulative evidence arithmetic failed")
            if event.get("crossed_threshold_at_checkpoint") != crossed or event.get(
                "crossed_threshold_ever"
            ) != (first_crossing is not None):
                raise RuntimeError("component threshold flag is invalid")
            cumulative_by_event[local_index][component.index] = cumulative
        if not _close(float(record.get("final_cumulative_log_e", math.nan)), cumulative):
            raise RuntimeError("component final evidence is invalid")
        if not _close(float(record.get("running_max_log_e", math.nan)), running):
            raise RuntimeError("component running maximum is invalid")
        if record.get("first_crossing_checkpoint_index") != first_crossing:
            raise RuntimeError("component first crossing is invalid")

    mixture = evidence.get("mixture")
    expected_weights = [component.mixture_weight for component in components]
    if not isinstance(mixture, dict) or mixture.get("weights") != expected_weights:
        raise RuntimeError("mixture weights changed")
    mixture_events = mixture.get("events")
    if not isinstance(mixture_events, list) or len(mixture_events) != len(
        expected_checkpoint_indices
    ):
        raise RuntimeError("wrong mixture-event count")
    mixture_running = float(mixture.get("prefix_running_max_log_e", math.nan))
    mixture_first = mixture.get("prefix_first_crossing_checkpoint_index")
    if not math.isfinite(mixture_running):
        raise RuntimeError("non-finite mixture prefix running maximum")
    for local_index, (checkpoint_index, event) in enumerate(
        zip(expected_checkpoint_indices, mixture_events)
    ):
        expected_log = _log_equal_weight_mixture(cumulative_by_event[local_index])
        mixture_running = max(mixture_running, expected_log)
        crossed = expected_log >= threshold
        if mixture_first is None and crossed:
            mixture_first = checkpoint_index
        if event.get("checkpoint_index") != checkpoint_index or event.get(
            "internal_timestep"
        ) != checkpoints[checkpoint_index]:
            raise RuntimeError("mixture event identity changed")
        if not _close(float(event.get("log_e_mixture", math.nan)), expected_log) or not _close(
            float(event.get("running_max_log_e_mixture", math.nan)), mixture_running
        ):
            raise RuntimeError("mixture arithmetic failed")
        if event.get("crossed_threshold_at_checkpoint") != crossed or event.get(
            "crossed_threshold_ever"
        ) != (mixture_first is not None):
            raise RuntimeError("mixture crossing flag failed")
    final_expected = _log_equal_weight_mixture(
        [float(record["final_cumulative_log_e"]) for record in records]
    )
    if not _close(float(mixture.get("final_log_e", math.nan)), final_expected):
        raise RuntimeError("mixture final evidence is invalid")
    if not _close(float(mixture.get("running_max_log_e", math.nan)), mixture_running):
        raise RuntimeError("mixture running maximum is invalid")
    if mixture.get("first_crossing_checkpoint_index") != mixture_first:
        raise RuntimeError("mixture first crossing is invalid")


def _validate_trace_file(
    output_dir: Path,
    expected_path: Path,
    trace_record: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
) -> None:
    if trace_record.get("relative_path") != expected_path.relative_to(output_dir).as_posix():
        raise RuntimeError("tensor-trace path identity changed")
    if not expected_path.is_file() or sha256_file(expected_path) != trace_record.get("sha256"):
        raise RuntimeError("tensor-trace file hash is invalid")
    if expected_path.stat().st_size != trace_record.get("bytes"):
        raise RuntimeError("tensor-trace byte count is invalid")
    try:
        with np.load(expected_path, allow_pickle=False) as archive:
            if sorted(archive.files) != trace_record.get("keys"):
                raise RuntimeError("tensor-trace key set changed")
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise RuntimeError(f"cannot validate tensor trace {expected_path}") from exc
    for key, record in trace_record.get("arrays", {}).items():
        value = arrays.get(key)
        if value is None:
            raise RuntimeError("trace-array record names a missing array")
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": _array_raw_sha256(value),
        }
        if record != expected:
            raise RuntimeError(f"trace-array identity failed for {key}")
    states = arrays.get("states")
    if states is None or states.shape[0] != len(transitions) + 1:
        raise RuntimeError("trace state count is invalid")
    for index, transition in enumerate(transitions):
        if _array_raw_sha256(states[index]) != transition["state_before"][
            "raw_bytes_sha256"
        ]:
            raise RuntimeError("trace/pre-transition state hash mismatch")
    if _array_raw_sha256(states[-1]) != transitions[-1]["state_after"][
        "raw_bytes_sha256"
    ]:
        raise RuntimeError("trace/final state hash mismatch")


def validate_signal_payload(
    signal: dict[str, Any],
    pair: Pair,
    *,
    output_dir: Path,
    baseline: BaselineReference,
    manifest_identity_sha256: str,
    runner_sha256: str,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    alpha_role: str,
    rollback_lag_checkpoints: int,
    force_intervention: bool,
    force_internal_timestep: int | None,
    save_tensor_trace: bool,
) -> None:
    if signal.get("payload_sha256") != _canonical_payload_sha(signal, "payload_sha256"):
        raise RuntimeError(f"self-hashed signal payload is invalid for {pair}")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "class_id": pair[0],
        "seed": pair[1],
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
        "baseline_manifest_identity_sha256": baseline.manifest_identity_sha256,
        "alpha": alpha,
        "alpha_role": alpha_role,
        "log_e_crossing_threshold": -math.log(alpha),
        "max_conditional_kl": max_conditional_kl,
        "rollback_lag_checkpoints": rollback_lag_checkpoints,
        "force_intervention": force_intervention,
        "force_internal_timestep": force_internal_timestep,
        "retry_limit": 1,
        "looping_or_endpoint_selection": False,
        "internal_timesteps": list(EVIDENCE_INTERNAL_TIMESTEPS),
        "method_claim_eligible": (
            not force_intervention and alpha_role == "primary_anytime_valid_alpha_0.05"
        ),
        "mechanics_only_posthoc_exploration": (
            force_intervention or alpha_role != "primary_anytime_valid_alpha_0.05"
        ),
    }
    mismatches = {
        key: (signal.get(key), value)
        for key, value in fixed.items()
        if signal.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"signal configuration mismatch for {pair}: {mismatches}")
    paths = signal.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(OUTPUT_ROLES):
        raise RuntimeError("signal path records are incomplete")

    original = paths["original_baseline"]
    _validate_transition_record(
        original,
        start_internal_timestep=NUM_SPACED_STEPS - 1,
        stream_seed=sample_stream_seed(pair[1]),
        includes_initial_latent_draw=True,
        checkpoints=EVIDENCE_INTERNAL_TIMESTEPS,
    )
    original_transition_by_t = {
        int(event["internal_timestep"]): event for event in original["transitions"]
    }
    _validate_evidence_record(
        original["evidence"],
        components,
        expected_checkpoint_indices=range(len(EVIDENCE_INTERNAL_TIMESTEPS)),
        max_conditional_kl=max_conditional_kl,
        alpha=alpha,
        transition_by_t=original_transition_by_t,
        checkpoints=EVIDENCE_INTERNAL_TIMESTEPS,
    )
    rng = signal.get("rng")
    if not isinstance(rng, dict) or rng.get("baseline_stream_seed") != sample_stream_seed(
        pair[1]
    ) or rng.get("branch_seed_namespace") != BRANCH_RNG_NAMESPACE:
        raise RuntimeError("signal baseline RNG identity is invalid")

    action = signal.get("action")
    intervention = paths["intervention"]
    control = paths["same_checkpoint_random_control"]
    rollback_record = signal.get("rollback_state_file")
    if action is None:
        if force_intervention:
            raise RuntimeError("forced diagnostic cannot finish without an action")
        if signal.get("intervention_count") != 0:
            raise RuntimeError("no-action signal has nonzero intervention count")
        if original["evidence"]["mixture"]["first_crossing_checkpoint_index"] is not None:
            raise RuntimeError("natural threshold crossing was not acted upon")
        if rollback_record is not None:
            raise RuntimeError("no-action signal unexpectedly saved a rollback state")
        if rng.get("intervention_stream_seed") is not None or rng.get(
            "same_checkpoint_random_control_stream_seed"
        ) is not None:
            raise RuntimeError("no-action signal unexpectedly constructed suffix RNG streams")
        for role_record in (intervention, control):
            if role_record.get("execution") != "no_crossing_bitwise_alias_no_compute":
                raise RuntimeError("no-action output is not a no-compute alias")
            if role_record.get("transition_count") != 0 or role_record.get(
                "gaussian_draws_in_stream"
            ) != 0 or role_record.get("transitions") != []:
                raise RuntimeError("no-action alias contains hidden computation")
    else:
        if not isinstance(action, dict) or signal.get("intervention_count") != 1:
            raise RuntimeError("triggered signal has invalid action count")
        if action.get("method_claim_eligible") != signal.get(
            "method_claim_eligible"
        ) or action.get("mechanics_only_posthoc_exploration") != signal.get(
            "mechanics_only_posthoc_exploration"
        ):
            raise RuntimeError("action-level claim label differs from the run-level label")
        trigger_index = int(action.get("checkpoint_index", -1))
        if trigger_index < 0 or trigger_index >= len(EVIDENCE_INTERNAL_TIMESTEPS):
            raise RuntimeError("action trigger checkpoint is invalid")
        trigger_t = EVIDENCE_INTERNAL_TIMESTEPS[trigger_index]
        if action.get("internal_timestep") != trigger_t:
            raise RuntimeError("action trigger timestep is invalid")
        original_trigger_transition = original_transition_by_t[trigger_t]
        if action.get("pre_transition_state_raw_bytes_sha256") != original_trigger_transition[
            "state_before"
        ]["raw_bytes_sha256"] or action.get(
            "post_transition_state_raw_bytes_sha256"
        ) != original_trigger_transition[
            "state_after"
        ]["raw_bytes_sha256"]:
            raise RuntimeError("action does not identify its exact causal P transition")
        trigger_noise = action.get("trigger_causing_innovation")
        _require_tensor_record(trigger_noise, "action/trigger-causing innovation")
        if trigger_noise["raw_bytes_sha256"] != original_trigger_transition["innovation"][
            "raw_bytes_sha256"
        ]:
            raise RuntimeError("action records the wrong trigger-causing innovation")
        if force_intervention:
            if action.get("trigger_kind") != (
                "oracle_forced_diagnostic_excluded_from_method_claims"
            ) or trigger_t != force_internal_timestep or action.get("method_claim_eligible"):
                raise RuntimeError("forced diagnostic is not unmistakably excluded")
        else:
            if action.get("trigger_kind") != "anytime_valid_mixture_first_crossing":
                raise RuntimeError("natural action has the wrong trigger kind")
            if not action.get("threshold_crossed_at_trigger"):
                raise RuntimeError("natural action occurred without threshold crossing")
            if original["evidence"]["mixture"][
                "first_crossing_checkpoint_index"
            ] != trigger_index:
                raise RuntimeError("action was not taken at the first mixture crossing")

        rollback_index = max(0, trigger_index - rollback_lag_checkpoints)
        rollback_t = EVIDENCE_INTERNAL_TIMESTEPS[rollback_index]
        if action.get("rollback_checkpoint_index") != rollback_index or action.get(
            "rollback_internal_timestep"
        ) != rollback_t:
            raise RuntimeError("rollback checkpoint violates the predeclared policy")
        _require_tensor_record(action.get("rollback_state"), "action/rollback state")
        if rollback_lag_checkpoints == 0 and action["rollback_state"][
            "raw_bytes_sha256"
        ] != original_trigger_transition["state_before"]["raw_bytes_sha256"]:
            raise RuntimeError("lag-zero action did not restore pre-transition x_t")

        if not isinstance(rollback_record, dict):
            raise RuntimeError("triggered action lacks an exact rollback-state file")
        expected_state_path = rollback_state_path(output_dir, pair)
        if rollback_record.get("relative_path") != expected_state_path.relative_to(
            output_dir
        ).as_posix() or not expected_state_path.is_file():
            raise RuntimeError("rollback-state path is invalid")
        if sha256_file(expected_state_path) != rollback_record.get("sha256") or expected_state_path.stat().st_size != rollback_record.get("bytes"):
            raise RuntimeError("rollback-state file identity is invalid")
        try:
            rollback_array = np.ascontiguousarray(
                np.load(expected_state_path, allow_pickle=False)
            )
        except Exception as exc:
            raise RuntimeError("cannot load rollback state") from exc
        expected_rollback_file = {
            "shape": list(rollback_array.shape),
            "dtype": str(rollback_array.dtype),
            "raw_bytes_sha256": _array_raw_sha256(rollback_array),
        }
        if any(rollback_record.get(key) != value for key, value in expected_rollback_file.items()):
            raise RuntimeError("rollback-state tensor identity is invalid")
        if rollback_record["raw_bytes_sha256"] != action["rollback_state"][
            "raw_bytes_sha256"
        ]:
            raise RuntimeError("rollback-state file differs from the recorded x_t")

        intervention_seed = branch_stream_seed(
            pair, "intervention", trigger_t, rollback_t
        )
        control_seed = branch_stream_seed(
            pair, "same_checkpoint_random_control", trigger_t, rollback_t
        )
        if rng.get("intervention_stream_seed") != intervention_seed or rng.get(
            "same_checkpoint_random_control_stream_seed"
        ) != control_seed:
            raise RuntimeError("suffix RNG seed derivation is invalid")
        for role, record, seed in (
            ("intervention", intervention, intervention_seed),
            ("same_checkpoint_random_control", control, control_seed),
        ):
            _validate_transition_record(
                record,
                start_internal_timestep=rollback_t,
                stream_seed=seed,
                includes_initial_latent_draw=False,
                checkpoints=EVIDENCE_INTERNAL_TIMESTEPS,
            )
            if record["initial_state"]["raw_bytes_sha256"] != rollback_record[
                "raw_bytes_sha256"
            ]:
                raise RuntimeError(f"{role} did not start from the exact rollback x_t")
            branch_transition_by_t = {
                int(event["internal_timestep"]): event for event in record["transitions"]
            }
            _validate_evidence_record(
                record["evidence"],
                components,
                expected_checkpoint_indices=range(
                    rollback_index, len(EVIDENCE_INTERNAL_TIMESTEPS)
                ),
                max_conditional_kl=max_conditional_kl,
                alpha=alpha,
                transition_by_t=branch_transition_by_t,
                checkpoints=EVIDENCE_INTERNAL_TIMESTEPS,
            )
        same_compute_keys = (
            "gaussian_draws_in_stream", "transition_count", "current_unet_evaluations",
            "shifted_unet_evaluations",
        )
        if any(intervention.get(key) != control.get(key) for key in same_compute_keys):
            raise RuntimeError("same-checkpoint control is not same-compute")

    # Trace files are optional only by manifest-level precommitment.
    if save_tensor_trace:
        _validate_trace_file(
            output_dir,
            trace_path(output_dir, "original_baseline", pair),
            original.get("tensor_trace"),
            original["transitions"],
        )
        if action is not None:
            _validate_trace_file(
                output_dir,
                trace_path(output_dir, "intervention", pair),
                intervention.get("tensor_trace"),
                intervention["transitions"],
            )
            _validate_trace_file(
                output_dir,
                trace_path(output_dir, "same_checkpoint_random_control", pair),
                control.get("tensor_trace"),
                control["transitions"],
            )
    elif original.get("tensor_trace") is not None:
        raise RuntimeError("manifest disabled traces but signal contains one")


def validate_output_set(
    output_dir: Path,
    baseline: BaselineReference,
    pairs: Sequence[Pair],
    *,
    manifest_identity_sha256: str,
    runner_sha256: str,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    alpha_role: str,
    rollback_lag_checkpoints: int,
    force_intervention: bool,
    force_internal_timestep: int | None,
    save_tensor_trace: bool,
    require_all: bool,
) -> set[Pair]:
    possible_files: dict[Path, Pair] = {}
    for pair in pairs:
        possible_files[signal_path(output_dir, pair).resolve()] = pair
        possible_files[rollback_state_path(output_dir, pair).resolve()] = pair
        for role in OUTPUT_ROLES:
            possible_files[output_image_path(output_dir, role, pair).resolve()] = pair
            possible_files[trace_path(output_dir, role, pair).resolve()] = pair
    scanned: set[Path] = set()
    for relative_root, pattern in (
        ("signals", "*.json"),
        ("images", "*.png"),
        ("traces", "*.npz"),
        ("rollback_states", "*.npy"),
    ):
        root = output_dir / relative_root
        if root.exists():
            scanned.update(path.resolve() for path in root.rglob(pattern))
    unexpected = sorted(scanned - set(possible_files))
    if unexpected:
        raise RuntimeError(f"output contains an unexpected file: {unexpected[0]}")

    complete: set[Pair] = set()
    for pair in pairs:
        pair_signal_path = signal_path(output_dir, pair)
        pair_files = [
            output_image_path(output_dir, role, pair) for role in OUTPUT_ROLES
        ] + [
            trace_path(output_dir, role, pair) for role in OUTPUT_ROLES
        ] + [rollback_state_path(output_dir, pair)]
        if not pair_signal_path.exists():
            partial = [path for path in pair_files if path.exists()]
            if partial:
                raise RuntimeError(f"strict resume found partial output for {pair}: {partial[0]}")
            continue
        try:
            signal = json.loads(pair_signal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read signal {pair_signal_path}") from exc
        validate_signal_payload(
            signal,
            pair,
            output_dir=output_dir,
            baseline=baseline,
            manifest_identity_sha256=manifest_identity_sha256,
            runner_sha256=runner_sha256,
            components=components,
            max_conditional_kl=max_conditional_kl,
            alpha=alpha,
            alpha_role=alpha_role,
            rollback_lag_checkpoints=rollback_lag_checkpoints,
            force_intervention=force_intervention,
            force_internal_timestep=force_internal_timestep,
            save_tensor_trace=save_tensor_trace,
        )

        baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
        baseline_sha = pixel_sha256(baseline_pixels)
        output_pixels: dict[str, np.ndarray] = {}
        for role in OUTPUT_ROLES:
            path = output_image_path(output_dir, role, pair)
            if not path.is_file():
                raise RuntimeError(f"missing output image for {pair}/{role}")
            try:
                with Image.open(path) as image:
                    metadata = dict(image.info)
                    if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                        raise RuntimeError("wrong output PNG mode/size")
                    image.verify()
            except Exception as exc:
                raise RuntimeError(f"cannot validate output PNG {path}") from exc
            pixels = decoded_pixels(path)
            digest = pixel_sha256(pixels)
            output_pixels[role] = pixels
            expected_metadata = {
                "experiment": EXPERIMENT,
                "output_role": role,
                "class_id": str(pair[0]),
                "seed": str(pair[1]),
                "baseline_sample_stream_seed": str(sample_stream_seed(pair[1])),
                "pixel_sha256": digest,
                "original_baseline_pixel_sha256": baseline_sha,
                "signal_payload_sha256": signal["payload_sha256"],
                "manifest_identity_sha256": manifest_identity_sha256,
                "runner_sha256": runner_sha256,
            }
            metadata_mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if metadata_mismatches:
                raise RuntimeError(f"PNG provenance mismatch for {pair}/{role}")
            output_record = signal["outputs"][role]
            if output_record.get("relative_path") != path.relative_to(
                output_dir
            ).as_posix() or output_record.get("pixel_sha256") != digest:
                raise RuntimeError(f"signal/image identity mismatch for {pair}/{role}")
        if not np.array_equal(output_pixels["original_baseline"], baseline_pixels):
            raise RuntimeError(f"instrumented original path differs from frozen P for {pair}")
        if signal["outputs"].get("frozen_baseline_pixel_sha256") != baseline_sha:
            raise RuntimeError("frozen baseline pixel hash is invalid")
        if signal.get("action") is None and not (
            np.array_equal(output_pixels["original_baseline"], output_pixels["intervention"])
            and np.array_equal(
                output_pixels["original_baseline"],
                output_pixels["same_checkpoint_random_control"],
            )
        ):
            raise RuntimeError("no-crossing output is not bitwise baseline")

        expected_trace_paths = set()
        if save_tensor_trace:
            expected_trace_paths.add(trace_path(output_dir, "original_baseline", pair))
            if signal.get("action") is not None:
                expected_trace_paths.update(
                    {
                        trace_path(output_dir, "intervention", pair),
                        trace_path(output_dir, "same_checkpoint_random_control", pair),
                    }
                )
        for role in OUTPUT_ROLES:
            path = trace_path(output_dir, role, pair)
            if path.exists() != (path in expected_trace_paths):
                raise RuntimeError(f"unexpected/missing trace file for {pair}/{role}")
        if rollback_state_path(output_dir, pair).exists() != (
            signal.get("action") is not None
        ):
            raise RuntimeError("rollback-state file presence disagrees with action")
        complete.add(pair)
    if require_all and len(complete) != len(pairs):
        raise RuntimeError(f"only {len(complete)}/{len(pairs)} intervention bundles are complete")
    return complete


def build_manifest(
    args: argparse.Namespace,
    protocol: Protocol,
    device: torch.device,
    model_checkpoint_record: dict[str, Any],
    classifier_checkpoint_record: dict[str, Any],
    baseline: BaselineReference,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    components: Sequence[ComponentSpec],
    alpha_role: str,
) -> dict[str, Any]:
    source_root = args.guided_diffusion_root.resolve()
    source_revision = git_revision(source_root)
    source_dirty = git_tracked_dirty(source_root)
    if source_revision != GUIDED_DIFFUSION_REVISION:
        raise RuntimeError(
            f"guided-diffusion revision mismatch: {source_revision} != {GUIDED_DIFFUSION_REVISION}"
        )
    if source_dirty:
        raise RuntimeError("guided-diffusion has tracked source edits")
    runner_path = Path(__file__).resolve()
    baseline_runner_path = Path(__file__).with_name("reproduce_adm64_guided.py").resolve()
    observe_runner_path = Path(__file__).with_name(
        "observe_adm64_cross_scale_evidence.py"
    ).resolve()
    evidence_path = Path(__file__).with_name("adm64_path_evidence.py").resolve()
    pair_set_sha = sha256_json([[class_id, seed] for class_id, seed in protocol.pairs])
    alpha_schedule_sha = hashlib.sha256(
        np.ascontiguousarray(original_alpha_bar, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "bounded_exploratory_one_shot_rollback_intervention",
        "protocol": args.protocol,
        "class_ids": list(protocol.class_ids),
        "seeds": list(protocol.seeds),
        "pair_order": "class_major_then_seed_major",
        "pair_set_sha256": pair_set_sha,
        "sample_count": len(protocol.pairs),
        "checkpoints": {
            "diffusion": model_checkpoint_record,
            "classifier": classifier_checkpoint_record,
        },
        "pure_p_baseline": {
            "root": str(baseline.root),
            "manifest_identity_sha256": baseline.manifest_identity_sha256,
            "runner_sha256": baseline.runner_sha256,
            "pair_set_sha256": baseline.pair_set_sha256,
            "requirement": "decoded original endpoint equals frozen baseline byte-for-byte",
        },
        "official_model_config": OFFICIAL_MODEL_CONFIG,
        "official_classifier_config": OFFICIAL_CLASSIFIER_CONFIG,
        "p_sampler": {
            "name": "OpenAI classifier-guided ancestral DDPM",
            "classifier_scale": CLASSIFIER_SCALE,
            "timestep_respacing": "250",
            "clip_denoised": True,
            "learned_diagonal_variance": True,
            "transition": "x_{t-1}=guided_mean_t+stored_sigma_t*epsilon_t for t>0",
            "final_timestep": "t=0 deterministic guided mean; no Gaussian draw",
            "state_dtype": "torch.float32",
            "neural_eval_batch_size": 1,
        },
        "evidence": {
            "policy_id": "global_cross_scale_score_difference_v1",
            "policy_extension_boundary": (
                "the predictable Q construction is confined to the pre-noise checkpoint block; "
                "future local-Q policies must preserve the same pre-noise measurability and LR interface"
            ),
            "original_schedule": "official cosine 1000-step alpha_bar",
            "alpha_bar_float64_bytes_sha256": alpha_schedule_sha,
            "normalized_heat_time": "nu=(1-alpha_bar)/alpha_bar",
            "spaced_timestep_map": timestep_map.tolist(),
            "internal_checkpoints_reverse_order": list(EVIDENCE_INTERNAL_TIMESTEPS),
            "components": [
                _mapping_manifest_record(EVIDENCE_INTERNAL_TIMESTEPS, component)
                for component in components
            ],
            "mixture": "fixed equal-weight arithmetic mixture of component E-processes",
            "max_conditional_kl_per_component_checkpoint": args.max_conditional_kl,
            "alpha": args.alpha,
            "alpha_role": alpha_role,
            "log_e_crossing_threshold": -math.log(args.alpha),
            "q_mean": "mu_Q=mu_P+variance_P*theta after predictable KL tempering",
            "q_covariance": "exactly P covariance",
            "classifier": "included in P mean; excluded from cross-scale theta",
            "increment": "dot(stored_sigma*tempered_theta, actual P innovation)-conditional_KL",
            "noncheckpoint_steps": "Q=P and LR increment is zero",
        },
        "action": {
            "trigger": (
                "fixed-mixture first crossing"
                if not args.force_intervention
                else "ORACLE FORCED DIAGNOSTIC; EXCLUDED FROM ALL METHOD CLAIMS"
            ),
            "force_intervention": args.force_intervention,
            "force_internal_timestep": args.force_internal_timestep,
            "method_claim_eligible": (
                not args.force_intervention
                and alpha_role == "primary_anytime_valid_alpha_0.05"
            ),
            "mechanics_only_posthoc_exploration": (
                args.force_intervention
                or alpha_role != "primary_anytime_valid_alpha_0.05"
            ),
            "rollback_lag_checkpoints": args.rollback_lag_checkpoints,
            "primary_rollback_semantics": (
                "restore exact pre-transition x_t that preceded the trigger-causing innovation, "
                "then redraw t,t-1,...,1 once and execute deterministic t=0"
            ),
            "suffix_kernel": "unchanged P sampler; no additional guidance force",
            "maximum_interventions_per_path": 1,
            "maximum_retry_suffixes_per_path": 1,
            "looping": False,
            "endpoint_selection": False,
        },
        "random_control": {
            "name": "same_checkpoint_random_control",
            "definition": "second independent P suffix from the identical rollback x_t",
            "same_compute_as_intervention": True,
            "interpretation_limit": (
                "controls retry-to-retry randomness; it does not randomize or isolate trigger selection"
            ),
        },
        "rng": {
            "baseline_seed_function": "sample_stream_seed from frozen ADM baseline runner",
            "branch_seed_namespace": BRANCH_RNG_NAMESPACE,
            "branch_seed_inputs": (
                "role,class_id,public_seed,trigger_internal_timestep,rollback_internal_timestep"
            ),
            "independent_domain_separated_suffix_streams": True,
            "model_and_evidence_calls_consume_no_random_numbers": True,
        },
        "tensor_audit": {
            "save_tensor_traces": args.save_tensor_traces,
            "transition_json": (
                "all x-before/x-after, epsilon, predicted-x0, unguided/guided means, "
                "classifier shift, learned sigma, and actual innovations have shape/dtype/hash/statistics"
            ),
            "trace_npz": (
                "exact float state at segment start and after every transition, plus full checkpoint "
                "epsilon, predicted-x0, sigma, theta, and actual innovation tensors; sigma*theta "
                "and the innovation exactly reconstruct spatial K/R/log-LR maps"
            ),
            "rollback_npy": "exact float32 pre-transition rollback x_t",
        },
        "outputs": {
            "images": "images/{original_baseline,intervention,same_checkpoint_random_control}/...",
            "signals": "signals/class_{class_id:04d}/{seed:019d}.json",
            "traces": "traces/{role}/class_{class_id:04d}/{seed:019d}.npz",
            "rollback_states": "rollback_states/class_{class_id:04d}/{seed:019d}.npy",
            "no_crossing": "three decoded images equal baseline; no suffix RNG/evaluation",
            "strict_resume": "self-hashed manifest/signal plus exact file, tensor, pixel, and provenance validation",
        },
        "sources": {
            "guided_diffusion_root": str(source_root),
            "guided_diffusion_revision": source_revision,
            "guided_diffusion_tracked_dirty": source_dirty,
            "guided_diffusion_python_tree_sha256": sha256_python_tree(
                source_root / "guided_diffusion"
            ),
            "baseline_runner": {
                "path": str(baseline_runner_path),
                "sha256": sha256_file(baseline_runner_path),
            },
            "observation_runner_reused_definitions": {
                "path": str(observe_runner_path),
                "sha256": sha256_file(observe_runner_path),
            },
            "evidence_primitives": {
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
            },
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", None),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
    }
    manifest["identity_sha256"] = sha256_json(manifest)
    return manifest


def create_or_validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read existing manifest: {path}") from exc
        if existing.get("identity_sha256") != _canonical_payload_sha(
            existing, "identity_sha256"
        ):
            raise RuntimeError("existing intervention manifest self-hash is invalid")
        if existing != manifest:
            differing = sorted(
                key
                for key in set(existing) | set(manifest)
                if existing.get(key) != manifest.get(key)
            )
            raise RuntimeError(
                "output directory has an incompatible manifest; differing keys: "
                + ", ".join(differing)
            )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory without manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, path)


def _validate_cli_configuration(args: argparse.Namespace) -> str:
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if not math.isfinite(args.alpha) or not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between zero and one")
    if args.alpha != DEFAULT_ALPHA and not args.exploratory_alpha:
        raise ValueError(
            "non-primary --alpha requires --exploratory-alpha so the run is labelled explicitly"
        )
    if not math.isfinite(args.max_conditional_kl) or args.max_conditional_kl <= 0:
        raise ValueError("--max-conditional-kl must be finite and strictly positive")
    if args.rollback_lag_checkpoints < 0:
        raise ValueError("--rollback-lag-checkpoints cannot be negative")
    if args.force_intervention:
        if args.force_internal_timestep is None:
            raise ValueError(
                "--force-intervention is an oracle diagnostic and requires the explicit "
                "--force-internal-timestep"
            )
        if args.force_internal_timestep not in EVIDENCE_INTERNAL_TIMESTEPS:
            raise ValueError("--force-internal-timestep must be a predeclared evidence checkpoint")
    elif args.force_internal_timestep is not None:
        raise ValueError(
            "--force-internal-timestep is forbidden without --force-intervention"
        )
    return (
        "exploratory_alpha_explicitly_opted_in"
        if args.exploratory_alpha
        else "primary_anytime_valid_alpha_0.05"
    )


def load_baseline_reference_allowing_subset(
    root: Path,
    protocol: Protocol,
    *,
    expected_model_sha256: str,
    expected_classifier_sha256: str,
) -> BaselineReference:
    """Validate a complete frozen baseline and permit a strict pair subset."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen ADM baseline manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read baseline manifest: {manifest_path}") from exc
    identity = manifest.get("identity_sha256")
    if not isinstance(identity, str) or identity != _canonical_payload_sha(
        manifest, "identity_sha256"
    ):
        raise RuntimeError("frozen baseline manifest self-hash is invalid")
    if manifest.get("experiment") != "adm64_classifier_guided_reproduction":
        raise RuntimeError("baseline directory is not an ADM64 reproduction")
    for name, expected in (
        ("diffusion", expected_model_sha256),
        ("classifier", expected_classifier_sha256),
    ):
        if manifest.get("checkpoints", {}).get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"baseline {name} checkpoint identity is incompatible")
    class_ids = manifest.get("class_ids")
    seeds = manifest.get("seeds")
    if not isinstance(class_ids, list) or not isinstance(seeds, list):
        raise RuntimeError("baseline manifest lacks its class/seed axes")
    full_pairs: tuple[Pair, ...] = tuple(
        (int(class_id), int(seed)) for class_id in class_ids for seed in seeds
    )
    if manifest.get("sample_count") != len(full_pairs):
        raise RuntimeError("baseline sample count is inconsistent")
    requested = set(protocol.pairs)
    if not requested.issubset(set(full_pairs)):
        missing = sorted(requested - set(full_pairs))
        raise RuntimeError(f"requested pair is absent from the frozen baseline: {missing[0]}")
    pair_set_sha = sha256_json([[class_id, seed] for class_id, seed in full_pairs])
    if manifest.get("pair_set_sha256") != pair_set_sha:
        raise RuntimeError("baseline pair-set self-identity is invalid")
    baseline_runner = Path(__file__).with_name("reproduce_adm64_guided.py").resolve()
    runner_sha = manifest.get("runner", {}).get("sha256")
    if runner_sha != sha256_file(baseline_runner):
        raise RuntimeError("baseline was not produced by the current frozen runner")
    validated = validate_baseline_output_set(
        root,
        full_pairs,
        identity,
        runner_sha,
        require_all=True,
    )
    if len(validated) != len(full_pairs):
        raise RuntimeError("frozen baseline output validation was incomplete")
    completion = validate_existing_completion(
        root / "completion.json",
        manifest_identity_sha256=identity,
        pair_set_sha256=pair_set_sha,
        total_expected=len(full_pairs),
    )
    if completion is None:
        raise RuntimeError("frozen baseline has no strict completion marker")
    return BaselineReference(root.resolve(), identity, runner_sha, pair_set_sha)


def run_intervention(args: argparse.Namespace, protocol: Protocol, alpha_role: str) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official FP16 ADM64 sampler requires CUDA")
    if args.output_dir.resolve() == args.baseline_dir.resolve():
        raise ValueError("intervention output must differ from the frozen baseline directory")
    configure_determinism()
    torch.cuda.set_device(device)
    model_checkpoint_record = validate_checkpoint(args.model_path, DIFFUSION_CHECKPOINT)
    classifier_checkpoint_record = validate_checkpoint(
        args.classifier_path, CLASSIFIER_CHECKPOINT
    )
    original_alpha_bar, timestep_map = original_schedule_and_timestep_map(
        args.guided_diffusion_root
    )
    components = build_component_specs(
        original_alpha_bar, timestep_map, args.heat_shifts
    )
    baseline = load_baseline_reference_allowing_subset(
        args.baseline_dir,
        protocol,
        expected_model_sha256=model_checkpoint_record["sha256"],
        expected_classifier_sha256=classifier_checkpoint_record["sha256"],
    )
    manifest = build_manifest(
        args,
        protocol,
        device,
        model_checkpoint_record,
        classifier_checkpoint_record,
        baseline,
        original_alpha_bar,
        timestep_map,
        components,
        alpha_role,
    )
    create_or_validate_manifest(args.output_dir, manifest)
    manifest_identity = manifest["identity_sha256"]
    runner_sha = manifest["runner"]["sha256"]
    complete_pairs = validate_output_set(
        args.output_dir,
        baseline,
        protocol.pairs,
        manifest_identity_sha256=manifest_identity,
        runner_sha256=runner_sha,
        components=components,
        max_conditional_kl=args.max_conditional_kl,
        alpha=args.alpha,
        alpha_role=alpha_role,
        rollback_lag_checkpoints=args.rollback_lag_checkpoints,
        force_intervention=args.force_intervention,
        force_internal_timestep=args.force_internal_timestep,
        save_tensor_trace=args.save_tensor_traces,
        require_all=False,
    )
    completion_path = args.output_dir / "completion.json"
    completion = validate_existing_completion(
        completion_path,
        manifest_identity_sha256=manifest_identity,
        pair_set_sha256=manifest["pair_set_sha256"],
        total_expected=len(protocol.pairs),
    )
    if completion is not None:
        if len(complete_pairs) != len(protocol.pairs):
            raise RuntimeError("completion marker exists but strict pair bundles are missing")
        print(json.dumps(completion, ensure_ascii=False, indent=2))
        return

    pending = [pair for pair in protocol.pairs if pair not in complete_pairs]
    start = time.monotonic()
    generated = 0
    if pending:
        model, diffusion, classifier = load_official_models(
            args.guided_diffusion_root,
            args.model_path,
            args.classifier_path,
            device,
        )
        if list(diffusion.timestep_map) != timestep_map.tolist():
            raise RuntimeError("loaded SpacedDiffusion timestep map differs from manifest")
        if not np.allclose(
            np.asarray(diffusion.alphas_cumprod, dtype=np.float64),
            original_alpha_bar[timestep_map],
            rtol=2e-13,
            atol=2e-15,
        ):
            raise RuntimeError("loaded spaced alpha_bar differs from frozen schedule")
        _, cond_fn = make_guided_functions(model, classifier)
        for logical_batch in chunks(pending, args.batch):
            # Logical batching affects only scheduling.  Every path and neural
            # evaluation is singleton, including both retry branches.
            for pair in logical_batch:
                result = sample_intervention_pair(
                    diffusion,
                    model,
                    cond_fn,
                    pair,
                    device=device,
                    original_alpha_bar=original_alpha_bar,
                    timestep_map=timestep_map,
                    components=components,
                    max_conditional_kl=args.max_conditional_kl,
                    alpha=args.alpha,
                    rollback_lag_checkpoints=args.rollback_lag_checkpoints,
                    force_intervention=args.force_intervention,
                    force_internal_timestep=args.force_internal_timestep,
                    save_tensor_trace=args.save_tensor_traces,
                )
                original_pixels = pixels_from_sample(result.original.final_state[0])
                frozen_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
                if not np.array_equal(original_pixels, frozen_pixels):
                    raise RuntimeError(
                        f"instrumented original P path differs from frozen baseline: {pair}"
                    )
                if result.action is None:
                    intervention_pixels = original_pixels.copy()
                    control_pixels = original_pixels.copy()
                else:
                    if result.intervention is None or result.random_control is None:
                        raise AssertionError("triggered result lacks suffix outputs")
                    intervention_pixels = pixels_from_sample(
                        result.intervention.final_state[0]
                    )
                    control_pixels = pixels_from_sample(
                        result.random_control.final_state[0]
                    )
                save_pair_bundle(
                    args.output_dir,
                    pair,
                    result,
                    original_pixels,
                    intervention_pixels,
                    control_pixels,
                    frozen_pixels,
                    manifest_identity_sha256=manifest_identity,
                    runner_sha256=runner_sha,
                    baseline_manifest_identity_sha256=baseline.manifest_identity_sha256,
                    alpha=args.alpha,
                    alpha_role=alpha_role,
                    max_conditional_kl=args.max_conditional_kl,
                    rollback_lag_checkpoints=args.rollback_lag_checkpoints,
                    force_intervention=args.force_intervention,
                    force_internal_timestep=args.force_internal_timestep,
                    save_tensor_trace=args.save_tensor_traces,
                )
                generated += 1
                if result.action is None:
                    action_text = "no crossing; exact no-op"
                else:
                    action_text = (
                        f"trigger t={result.action['internal_timestep']}, "
                        f"rollback x_{result.action['rollback_internal_timestep']}"
                    )
                print(
                    f"saved {generated}/{len(pending)} new pair {pair}: {action_text}",
                    flush=True,
                )

    final_pairs = validate_output_set(
        args.output_dir,
        baseline,
        protocol.pairs,
        manifest_identity_sha256=manifest_identity,
        runner_sha256=runner_sha,
        components=components,
        max_conditional_kl=args.max_conditional_kl,
        alpha=args.alpha,
        alpha_role=alpha_role,
        rollback_lag_checkpoints=args.rollback_lag_checkpoints,
        force_intervention=args.force_intervention,
        force_internal_timestep=args.force_internal_timestep,
        save_tensor_trace=args.save_tensor_traces,
        require_all=True,
    )
    trigger_count = 0
    for pair in protocol.pairs:
        signal = json.loads(signal_path(args.output_dir, pair).read_text(encoding="utf-8"))
        trigger_count += int(signal["intervention_count"])
    final_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest_identity,
        "pair_set_sha256": manifest["pair_set_sha256"],
        "generated_this_run": generated,
        "already_complete": len(complete_pairs),
        "total_expected": len(protocol.pairs),
        "total_complete": len(final_pairs),
        "intervention_count": trigger_count,
        "no_action_count": len(protocol.pairs) - trigger_count,
        "force_intervention": args.force_intervention,
        "alpha": args.alpha,
        "alpha_role": alpha_role,
        "logical_batch_requested": args.batch,
        "neural_eval_batch_size": 1,
        "wall_seconds": time.monotonic() - start,
        "finished_at_unix": time.time(),
    }
    atomic_json_dump(final_completion, completion_path)
    print(json.dumps(final_completion, ensure_ascii=False, indent=2))


class _ToyDiffusion:
    num_timesteps = 5
    timestep_map = [0, 2, 4, 6, 8]

    def p_mean_variance(
        self,
        model_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        original = torch.tensor(self.timestep_map, dtype=t.dtype, device=t.device)[t]
        raw = model_fn(x, original, **kwargs["model_kwargs"])
        channels = x.shape[1]
        epsilon, variance_logits = raw[:, :channels], raw[:, channels:]
        variance = torch.exp(-3.0 + 0.02 * variance_logits)
        return {
            "mean": 0.91 * x + 0.015 * epsilon,
            "variance": variance,
            "log_variance": variance.log(),
            "pred_xstart": epsilon,
        }

    def condition_mean(
        self,
        cond_fn: Callable[..., torch.Tensor],
        out: dict[str, torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        original = torch.tensor(self.timestep_map, dtype=t.dtype, device=t.device)[t]
        return out["mean"] + out["variance"] * cond_fn(
            x, original, **kwargs["model_kwargs"]
        )


class _ToyModel(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None or x.shape[0] != 1:
            raise AssertionError("toy model requires singleton class conditioning")
        epsilon = 0.07 * x + t.view(-1, 1, 1, 1).to(x.dtype) * 0.002
        epsilon = epsilon + y.view(-1, 1, 1, 1).to(x.dtype) * 0.0001
        variance_logits = torch.tanh(0.03 * x)
        return torch.cat([epsilon, variance_logits], dim=1)


def run_self_test() -> None:
    device = torch.device("cpu")
    diffusion = _ToyDiffusion()
    alpha_bar = np.array(
        [0.99, 0.96, 0.90, 0.80, 0.68, 0.54, 0.39, 0.24, 0.10],
        dtype=np.float64,
    )
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    checkpoints = (4, 2)
    components = build_component_specs(
        alpha_bar, timestep_map, (0.1, 1.0), checkpoints
    )
    pair: Pair = (3, 7)

    def toy_cond(
        x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            raise AssertionError("toy classifier requires y")
        return torch.full_like(x, 0.003) + y.view(-1, 1, 1, 1) * 0.00001

    forced = sample_intervention_pair(
        diffusion,
        _ToyModel(),
        toy_cond,
        pair,
        device=device,
        original_alpha_bar=alpha_bar,
        timestep_map=timestep_map,
        components=components,
        max_conditional_kl=0.02,
        alpha=0.3,
        rollback_lag_checkpoints=0,
        force_intervention=True,
        force_internal_timestep=2,
        save_tensor_trace=True,
        checkpoints=checkpoints,
        channels=1,
        image_size=2,
    )
    if forced.action is None or forced.intervention is None or forced.random_control is None:
        raise AssertionError("forced toy diagnostic did not produce exactly two suffix branches")
    if forced.action["trigger_kind"] != (
        "oracle_forced_diagnostic_excluded_from_method_claims"
    ) or forced.action["method_claim_eligible"]:
        raise AssertionError("forced diagnostic was not excluded from method claims")
    if forced.action["rollback_internal_timestep"] != 2:
        raise AssertionError("lag-zero rollback did not select trigger timestep")
    if forced.action["rollback_state"]["raw_bytes_sha256"] != forced.action[
        "pre_transition_state_raw_bytes_sha256"
    ]:
        raise AssertionError("lag-zero rollback did not restore exact pre-transition x_t")
    trigger_transition = forced.original.transitions[
        int(forced.action["transition_segment_step_index"])
    ]
    if forced.action["trigger_causing_innovation"][
        "raw_bytes_sha256"
    ] != trigger_transition["innovation"]["raw_bytes_sha256"]:
        raise AssertionError("forced action lost its trigger-causing innovation")
    if forced.intervention.gaussian_draws != 2 or forced.random_control.gaussian_draws != 2:
        raise AssertionError("toy suffix draw accounting failed")
    if forced.intervention_stream_seed == forced.control_stream_seed:
        raise AssertionError("toy intervention/control streams are not independent")
    if forced.original.trace_arrays["states"].shape[0] != 6:
        raise AssertionError("toy full state trace is incomplete")
    _validate_transition_record(
        {
            **_segment_signal_record(
                forced.original,
                execution="toy",
                stream_seed=forced.baseline_stream_seed,
                includes_initial_latent_draw=True,
                trace_record=None,
            )
        },
        start_internal_timestep=4,
        stream_seed=forced.baseline_stream_seed,
        includes_initial_latent_draw=True,
        checkpoints=checkpoints,
    )
    original_by_t = {
        int(event["internal_timestep"]): event
        for event in forced.original.transitions
    }
    _validate_evidence_record(
        forced.original.evidence,
        components,
        expected_checkpoint_indices=range(len(checkpoints)),
        max_conditional_kl=0.02,
        alpha=0.3,
        transition_by_t=original_by_t,
        checkpoints=checkpoints,
    )
    for branch, seed in (
        (forced.intervention, int(forced.intervention_stream_seed)),
        (forced.random_control, int(forced.control_stream_seed)),
    ):
        _validate_transition_record(
            _segment_signal_record(
                branch,
                execution="toy_suffix",
                stream_seed=seed,
                includes_initial_latent_draw=False,
                trace_record=None,
            ),
            start_internal_timestep=2,
            stream_seed=seed,
            includes_initial_latent_draw=False,
            checkpoints=checkpoints,
        )
        branch_by_t = {
            int(event["internal_timestep"]): event for event in branch.transitions
        }
        _validate_evidence_record(
            branch.evidence,
            components,
            expected_checkpoint_indices=range(1, len(checkpoints)),
            max_conditional_kl=0.02,
            alpha=0.3,
            transition_by_t=branch_by_t,
            checkpoints=checkpoints,
        )

    # Extra evidence evaluations and complete transition logging must leave the
    # original P path bitwise equal to the frozen baseline implementation.
    baseline_samples, accounting = sample_batch_invariant(
        diffusion,
        _ToyModel(),
        toy_cond,
        [pair],
        device=device,
        channels=1,
        image_size=2,
    )
    if not torch.equal(baseline_samples[0], forced.original.final_state[0]):
        raise AssertionError("instrumented toy original path differs from pure P")
    if accounting["gaussian_draws_per_path_including_initial"] != forced.original.gaussian_draws:
        raise AssertionError("toy original-path draw counts disagree")

    no_action = sample_intervention_pair(
        diffusion,
        _ToyModel(),
        toy_cond,
        pair,
        device=device,
        original_alpha_bar=alpha_bar,
        timestep_map=timestep_map,
        components=components,
        max_conditional_kl=0.02,
        alpha=1e-12,
        rollback_lag_checkpoints=0,
        force_intervention=False,
        force_internal_timestep=None,
        save_tensor_trace=False,
        checkpoints=checkpoints,
        channels=1,
        image_size=2,
    )
    if no_action.action is not None or no_action.intervention is not None or no_action.random_control is not None:
        raise AssertionError("no-crossing toy path was not a literal no-op")
    if not torch.equal(no_action.original.final_state, forced.original.final_state):
        raise AssertionError("action policy changed the original P path")

    base_namespace = argparse.Namespace(
        batch=1,
        alpha=0.3,
        exploratory_alpha=False,
        max_conditional_kl=0.2,
        rollback_lag_checkpoints=0,
        force_intervention=False,
        force_internal_timestep=None,
    )
    try:
        _validate_cli_configuration(base_namespace)
    except ValueError:
        pass
    else:
        raise AssertionError("unlabelled exploratory alpha was accepted")
    base_namespace.exploratory_alpha = True
    if _validate_cli_configuration(base_namespace) != "exploratory_alpha_explicitly_opted_in":
        raise AssertionError("explicit exploratory-alpha label failed")

    with tempfile.TemporaryDirectory(prefix="adm64-rollback-self-test-") as temporary:
        root = Path(temporary)
        trace = root / "trace.npz"
        _atomic_npz_dump(forced.original.trace_arrays, trace)
        trace_record = _trace_file_record(trace, forced.original.trace_arrays, root)
        _validate_trace_file(
            root, trace, trace_record, forced.original.transitions
        )
        try:
            _atomic_npz_dump(forced.original.trace_arrays, trace)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tensor trace overwrite was not rejected")

    print(
        "self-test passed: exact original P identity, explicit exploratory alpha, "
        "pre-transition x_t rollback, trigger innovation provenance, one-shot independent "
        "same-compute suffixes, no-crossing no-op, full traces, and no-overwrite guards"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    guided_root = data_root / "baselines" / "guided-diffusion"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("smoke", "custom"), default="smoke")
    parser.add_argument("--classes", type=parse_int_spec, default=None)
    parser.add_argument("--seeds", type=parse_int_spec, default=None)
    parser.add_argument("--heat-shifts", type=parse_float_spec, default=DEFAULT_HEAT_SHIFTS)
    parser.add_argument("--max-conditional-kl", type=float, default=DEFAULT_MAX_CONDITIONAL_KL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--exploratory-alpha",
        action="store_true",
        help="required opt-in/label whenever alpha differs from the primary 0.05",
    )
    parser.add_argument(
        "--rollback-lag-checkpoints",
        type=int,
        default=DEFAULT_ROLLBACK_LAG_CHECKPOINTS,
        help="0 restores the trigger transition's exact pre-noise x_t (primary policy)",
    )
    parser.add_argument(
        "--force-intervention",
        action="store_true",
        help="ORACLE DIAGNOSTIC ONLY; excluded from every method claim",
    )
    parser.add_argument(
        "--force-internal-timestep",
        type=int,
        default=None,
        help="required explicit evidence checkpoint for --force-intervention",
    )
    parser.add_argument(
        "--save-tensor-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save exact state trajectories and full checkpoint tensors (default: enabled)",
    )
    parser.add_argument("--guided-diffusion-root", type=Path, default=guided_root)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=guided_root / "checkpoints" / DIFFUSION_CHECKPOINT.filename,
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=guided_root / "checkpoints" / CLASSIFIER_CHECKPOINT.filename,
    )
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="logical scheduling only; every neural evaluation remains singleton",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    alpha_role = _validate_cli_configuration(args)
    protocol = protocol_from_args(args)
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    if args.baseline_dir is None:
        args.baseline_dir = (
            data_root / "cross_scale_evidence" / "adm64_guided" / args.protocol
        )
    if args.output_dir is None:
        alpha_tag = format(args.alpha, ".8g").replace(".", "p")
        force_tag = (
            f"_oracle_force_t{args.force_internal_timestep}"
            if args.force_intervention
            else ""
        )
        args.output_dir = (
            data_root
            / "cross_scale_evidence"
            / "adm64_cross_scale_intervention"
            / f"{args.protocol}_alpha_{alpha_tag}{force_tag}"
        )
    args.guided_diffusion_root = args.guided_diffusion_root.resolve()
    args.model_path = args.model_path.resolve()
    args.classifier_path = args.classifier_path.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "protocol": args.protocol,
                    "class_ids": list(protocol.class_ids),
                    "seeds": list(protocol.seeds),
                    "sample_count": len(protocol.pairs),
                    "baseline_dir": str(args.baseline_dir),
                    "output_dir": str(args.output_dir),
                    "heat_shifts": list(args.heat_shifts),
                    "max_conditional_kl": args.max_conditional_kl,
                    "alpha": args.alpha,
                    "alpha_role": alpha_role,
                    "log_e_crossing_threshold": -math.log(args.alpha),
                    "rollback_lag_checkpoints": args.rollback_lag_checkpoints,
                    "rollback_semantics": "restore pre-transition x_t and redraw t onward",
                    "force_intervention": args.force_intervention,
                    "force_internal_timestep": args.force_internal_timestep,
                    "method_claim_eligible": (
                        not args.force_intervention
                        and alpha_role == "primary_anytime_valid_alpha_0.05"
                    ),
                    "mechanics_only_posthoc_exploration": (
                        args.force_intervention
                        or alpha_role != "primary_anytime_valid_alpha_0.05"
                    ),
                    "retry_limit": 1,
                    "same_checkpoint_random_control": True,
                    "save_tensor_traces": args.save_tensor_traces,
                    "internal_checkpoints": list(EVIDENCE_INTERNAL_TIMESTEPS),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_intervention(args, protocol, alpha_role)


if __name__ == "__main__":
    main()

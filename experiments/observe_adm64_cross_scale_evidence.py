#!/usr/bin/env python3
"""Observe cross-scale path evidence on the frozen official ADM ImageNet-64 sampler.

This runner never changes the sampled path.  ``P`` is exactly the official
classifier-guided, 250-step stochastic ADM sampler implemented by
``reproduce_adm64_guided.py``.  At 32 predeclared reverse checkpoints it forms
four (by default) predictable same-covariance alternatives ``Q_j`` from fixed
additive shifts in normalized heat time and records the exact finite-step
Gaussian log likelihood ratios.  There is no threshold action, rejection,
rollback, resampling, or guidance modification in this file.

The primary cross-scale construction is deliberately unique:

* original 1000-step cosine ``alpha_bar`` and ``nu=(1-alpha_bar)/alpha_bar``;
* nearest original timestep to ``nu + Delta_j``;
* preserve the heat-coordinate state ``z=x/sqrt(alpha_bar)`` by evaluating the
  shifted U-Net at ``rho*x``, ``rho=sqrt(alpha_bar_plus/alpha_bar)``;
* use only the first three (epsilon) U-Net channels in
  ``theta=rho*s_plus(rho*x)-s_current(x)``;
* retain classifier guidance only in the actually sampled P mean;
* define ``delta=variance_P*theta``, predictably scaled to a per-checkpoint KL
  cap, and retain P's learned diagonal variance under Q;
* update log(Q/P) from the actual standard-normal innovation in float64.

Every output image is required to match the separately generated pure-baseline
PNG for the same ``(class_id, seed)`` byte-for-byte after decoding.  Neural
evaluations are singleton and each path owns a seed-only RNG stream, so this is
also invariant to the logical ``--batch`` grouping on the recorded platform.
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

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:  # Support both ``python experiments/file.py`` and package-style imports.
    from .adm64_path_evidence import (
        AdditiveHeatShiftMapping,
        kl_tempered_score_mean_shift,
        kl_tempered_score_mean_shift_from_standard_deviation,
        log_e_mixture,
        nearest_additive_heat_shift,
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
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
        SeedRandomStreams,
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
except ImportError:  # pragma: no cover - exercised by the CLI entry point.
    from adm64_path_evidence import (
        AdditiveHeatShiftMapping,
        kl_tempered_score_mean_shift,
        kl_tempered_score_mean_shift_from_standard_deviation,
        log_e_mixture,
        nearest_additive_heat_shift,
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
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
        SeedRandomStreams,
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


EXPERIMENT = "adm64_cross_scale_evidence_observe_only"
SCHEMA_VERSION = 2
DEFAULT_HEAT_SHIFTS = (0.01, 0.1, 1.0, 10.0)
DEFAULT_MAX_CONDITIONAL_KL = 0.2
DEFAULT_ALPHA = 0.05
EVIDENCE_INTERNAL_TIMESTEPS = tuple(range(249, 0, -8))

if EVIDENCE_INTERNAL_TIMESTEPS != tuple(
    [249, 241, 233, 225, 217, 209, 201, 193, 185, 177, 169, 161, 153, 145, 137, 129,
     121, 113, 105, 97, 89, 81, 73, 65, 57, 49, 41, 33, 25, 17, 9, 1]
):  # pragma: no cover - a source-edit guard.
    raise AssertionError("the predeclared 32-checkpoint schedule changed")


@dataclass(frozen=True)
class ComponentSpec:
    index: int
    component_id: str
    additive_heat_shift: float
    mixture_weight: float
    mapping: AdditiveHeatShiftMapping


@dataclass(frozen=True)
class BaselineReference:
    root: Path
    manifest_identity_sha256: str
    runner_sha256: str
    pair_set_sha256: str


def parse_float_spec(value: str) -> tuple[float, ...]:
    result: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid floating-point value: {token}") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise argparse.ArgumentTypeError("heat shifts must be finite and strictly positive")
        result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("heat-shift specification is empty")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("heat-shift specification contains duplicates")
    return tuple(result)


def _add_guided_source_to_path(root: Path) -> None:
    required = root / "guided_diffusion" / "gaussian_diffusion.py"
    if not required.is_file():
        raise FileNotFoundError(f"not an OpenAI guided-diffusion checkout: {root}")
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)


def original_schedule_and_timestep_map(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact official cosine schedule without constructing the U-Net."""

    _add_guided_source_to_path(root)
    from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
    from guided_diffusion.respace import space_timesteps

    betas = np.asarray(get_named_beta_schedule("cosine", 1_000), dtype=np.float64)
    if betas.shape != (1_000,):
        raise RuntimeError(f"unexpected original beta schedule shape: {betas.shape}")
    alpha_bar = np.cumprod(1.0 - betas, dtype=np.float64)
    timestep_map = np.asarray(sorted(space_timesteps(1_000, "250")), dtype=np.int64)
    if timestep_map.shape != (NUM_SPACED_STEPS,):
        raise RuntimeError(f"unexpected spaced timestep map shape: {timestep_map.shape}")
    if timestep_map[0] != 0 or timestep_map[-1] != 999:
        raise RuntimeError("official spaced schedule does not span original timesteps 0..999")
    return alpha_bar, timestep_map


def build_component_specs(
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    heat_shifts: Sequence[float],
    checkpoints: Sequence[int] = EVIDENCE_INTERNAL_TIMESTEPS,
) -> tuple[ComponentSpec, ...]:
    if not heat_shifts:
        raise ValueError("at least one heat shift is required")
    internal = np.asarray(checkpoints, dtype=np.int64)
    if internal.ndim != 1 or internal.size == 0:
        raise ValueError("evidence checkpoints must be a non-empty one-dimensional sequence")
    if np.any(internal <= 0) or np.any(internal >= len(timestep_map)):
        raise ValueError("evidence checkpoints must be stochastic internal timesteps")
    if not np.all(np.diff(internal) < 0):
        raise ValueError("evidence checkpoints must be in reverse sampling order")
    original = timestep_map[internal]
    weight = 1.0 / len(heat_shifts)
    return tuple(
        ComponentSpec(
            index=index,
            component_id=f"heat_shift_{index:02d}",
            additive_heat_shift=float(shift),
            mixture_weight=weight,
            mapping=nearest_additive_heat_shift(original_alpha_bar, original, float(shift)),
        )
        for index, shift in enumerate(heat_shifts)
    )


def _mapping_manifest_record(
    checkpoints: Sequence[int], component: ComponentSpec
) -> dict[str, Any]:
    mapping = component.mapping
    rows = []
    for index, internal_t in enumerate(checkpoints):
        rows.append(
            {
                "checkpoint_index": index,
                "internal_timestep": int(internal_t),
                "current_original_timestep": int(mapping.current_timestep[index]),
                "shifted_original_timestep": int(mapping.shifted_timestep[index]),
                "current_heat_variance": float(mapping.current_heat_variance[index]),
                "target_heat_variance": float(mapping.target_heat_variance[index]),
                "shifted_heat_variance": float(mapping.shifted_heat_variance[index]),
                "actual_heat_shift": float(mapping.actual_heat_shift[index]),
                "absolute_mapping_error": float(mapping.absolute_mapping_error[index]),
            }
        )
    return {
        "component_index": component.index,
        "component_id": component.component_id,
        "additive_heat_shift": component.additive_heat_shift,
        "mixture_weight": component.mixture_weight,
        "active_checkpoint_count": int(np.count_nonzero(
            mapping.shifted_timestep != mapping.current_timestep
        )),
        "identity_checkpoint_count": int(np.count_nonzero(
            mapping.shifted_timestep == mapping.current_timestep
        )),
        "checkpoint_mapping": rows,
    }


def _canonical_payload_sha(payload: dict[str, Any], digest_key: str) -> str:
    without_digest = dict(payload)
    without_digest.pop(digest_key, None)
    return sha256_json(without_digest)


def load_baseline_reference(
    root: Path,
    protocol: Protocol,
    *,
    expected_model_sha256: str,
    expected_classifier_sha256: str,
) -> BaselineReference:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing pure ADM64 baseline manifest: {manifest_path}\n"
            "Generate the exact same protocol with reproduce_adm64_guided.py first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read baseline manifest: {manifest_path}") from exc
    identity = manifest.get("identity_sha256")
    if not isinstance(identity, str) or _canonical_payload_sha(
        manifest, "identity_sha256"
    ) != identity:
        raise RuntimeError("pure-baseline manifest identity hash is invalid")

    expected_values = {
        "experiment": "adm64_classifier_guided_reproduction",
        "class_ids": list(protocol.class_ids),
        "seeds": list(protocol.seeds),
        "sample_count": len(protocol.pairs),
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_values.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"pure-baseline protocol is incompatible: {mismatches}")
    checkpoint_mismatches = {}
    for name, expected_sha in (
        ("diffusion", expected_model_sha256),
        ("classifier", expected_classifier_sha256),
    ):
        observed = manifest.get("checkpoints", {}).get(name, {}).get("sha256")
        if observed != expected_sha:
            checkpoint_mismatches[name] = (observed, expected_sha)
    if checkpoint_mismatches:
        raise RuntimeError(f"pure-baseline checkpoints are incompatible: {checkpoint_mismatches}")

    runner_sha = manifest.get("runner", {}).get("sha256")
    pair_set_sha = manifest.get("pair_set_sha256")
    if not isinstance(runner_sha, str) or not isinstance(pair_set_sha, str):
        raise RuntimeError("pure-baseline manifest lacks runner/pair-set identity")
    current_baseline_runner = Path(__file__).with_name("reproduce_adm64_guided.py").resolve()
    current_baseline_runner_sha = sha256_file(current_baseline_runner)
    if runner_sha != current_baseline_runner_sha:
        raise RuntimeError(
            "pure-baseline runner hash differs from the current frozen baseline runner: "
            f"{runner_sha} != {current_baseline_runner_sha}"
        )
    expected_pair_set_sha = sha256_json(
        [[class_id, seed] for class_id, seed in protocol.pairs]
    )
    if pair_set_sha != expected_pair_set_sha:
        raise RuntimeError(
            f"pure-baseline pair-set hash is invalid: {pair_set_sha} != {expected_pair_set_sha}"
        )
    validated = validate_baseline_output_set(
        root,
        protocol.pairs,
        identity,
        runner_sha,
        require_all=True,
    )
    if len(validated) != len(protocol.pairs):
        raise AssertionError("pure-baseline validation returned an incomplete pair set")
    completion = validate_existing_completion(
        root / "completion.json",
        manifest_identity_sha256=identity,
        pair_set_sha256=pair_set_sha,
        total_expected=len(protocol.pairs),
    )
    if completion is None:
        raise RuntimeError("pure baseline has no strict completion.json")
    return BaselineReference(root.resolve(), identity, runner_sha, pair_set_sha)


def observed_pair_path(output_dir: Path, pair: Pair) -> Path:
    class_id, seed = pair
    return output_dir / "images" / f"class_{class_id:04d}" / f"{seed:019d}.png"


def signal_pair_path(output_dir: Path, pair: Pair) -> Path:
    class_id, seed = pair
    return output_dir / "signals" / f"class_{class_id:04d}" / f"{seed:019d}.json"


def decoded_pixels(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"mode/size is {image.mode}/{image.size}")
            pixels = np.ascontiguousarray(np.asarray(image))
    except Exception as exc:
        raise RuntimeError(f"cannot decode RGB ADM64 PNG {path}: {exc}") from exc
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
        raise RuntimeError(f"unexpected decoded pixels in {path}: {pixels.shape}/{pixels.dtype}")
    return pixels


def pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def save_observed_pair(
    pixels: np.ndarray,
    signal_payload: dict[str, Any],
    output_dir: Path,
    pair: Pair,
    manifest_identity_sha256: str,
    runner_sha256: str,
) -> None:
    class_id, seed = pair
    decoded_sha = pixel_sha256(pixels)
    if signal_payload.get("pixel_sha256") != decoded_sha:
        raise ValueError("signal payload pixel hash does not match the sample")
    signal = dict(signal_payload)
    signal["payload_sha256"] = _canonical_payload_sha(signal, "payload_sha256")

    signal_path = signal_pair_path(output_dir, pair)
    image_path = observed_pair_path(output_dir, pair)
    if signal_path.exists() or image_path.exists():
        raise RuntimeError(f"refusing to overwrite a partial/existing pair: {pair}")

    metadata = PngInfo()
    fields = {
        "experiment": EXPERIMENT,
        "class_id": str(class_id),
        "seed": str(seed),
        "sample_stream_seed": str(sample_stream_seed(seed)),
        "pixel_sha256": decoded_sha,
        "baseline_pixel_sha256": str(signal["baseline_pixel_sha256"]),
        "signal_payload_sha256": str(signal["payload_sha256"]),
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
    }
    for key, value in fields.items():
        metadata.add_text(key, value)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = image_path.with_name(image_path.name + ".tmp")
    Image.fromarray(pixels, mode="RGB").save(
        temporary_image, format="PNG", pnginfo=metadata
    )
    os.replace(temporary_image, image_path)
    atomic_json_dump(signal, signal_path)


def _log_equal_weight_mixture(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).mean()))


def _close(left: float, right: float, tolerance: float = 2e-11) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance * (
        1.0 + abs(left) + abs(right)
    )


def validate_signal_payload(
    signal: dict[str, Any],
    pair: Pair,
    manifest_identity_sha256: str,
    runner_sha256: str,
    baseline_manifest_identity_sha256: str,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
) -> None:
    if signal.get("payload_sha256") != _canonical_payload_sha(signal, "payload_sha256"):
        raise RuntimeError(f"signal payload hash is invalid for {pair}")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "class_id": pair[0],
        "seed": pair[1],
        "sample_stream_seed": sample_stream_seed(pair[1]),
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
        "baseline_manifest_identity_sha256": baseline_manifest_identity_sha256,
        "max_conditional_kl": max_conditional_kl,
        "alpha": alpha,
        "log_e_crossing_threshold": -math.log(alpha),
        "checkpoint_count": len(EVIDENCE_INTERNAL_TIMESTEPS),
        "intervention_count": 0,
    }
    mismatches = {
        key: (signal.get(key), expected)
        for key, expected in expected_scalars.items()
        if signal.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"signal identity/config mismatch for {pair}: {mismatches}")
    if signal.get("internal_timesteps") != list(EVIDENCE_INTERNAL_TIMESTEPS):
        raise RuntimeError(f"signal checkpoint schedule mismatch for {pair}")

    component_records = signal.get("components")
    if not isinstance(component_records, list) or len(component_records) != len(components):
        raise RuntimeError(f"wrong component count in signal for {pair}")
    component_cumulative_by_check = [
        [0.0 for _ in components] for _ in EVIDENCE_INTERNAL_TIMESTEPS
    ]
    threshold = -math.log(alpha)
    for component, record in zip(components, component_records):
        expected_component = {
            "component_index": component.index,
            "component_id": component.component_id,
            "additive_heat_shift": component.additive_heat_shift,
            "mixture_weight": component.mixture_weight,
        }
        if any(record.get(key) != value for key, value in expected_component.items()):
            raise RuntimeError(f"component identity mismatch for {pair}/{component.component_id}")
        events = record.get("events")
        if not isinstance(events, list) or len(events) != len(EVIDENCE_INTERNAL_TIMESTEPS):
            raise RuntimeError(f"wrong event count for {pair}/{component.component_id}")
        cumulative = 0.0
        running_max = 0.0
        first_crossing: int | None = None
        mapping = component.mapping
        for checkpoint_index, event in enumerate(events):
            fixed = {
                "checkpoint_index": checkpoint_index,
                "internal_timestep": EVIDENCE_INTERNAL_TIMESTEPS[checkpoint_index],
                "current_original_timestep": int(mapping.current_timestep[checkpoint_index]),
                "shifted_original_timestep": int(mapping.shifted_timestep[checkpoint_index]),
            }
            if any(event.get(key) != value for key, value in fixed.items()):
                raise RuntimeError(f"event mapping identity mismatch for {pair}/{component.component_id}")
            current_original_t = int(mapping.current_timestep[checkpoint_index])
            shifted_original_t = int(mapping.shifted_timestep[checkpoint_index])
            expected_mapping_values = {
                "current_heat_variance": float(mapping.current_heat_variance[checkpoint_index]),
                "target_heat_variance": float(mapping.target_heat_variance[checkpoint_index]),
                "shifted_heat_variance": float(mapping.shifted_heat_variance[checkpoint_index]),
                "actual_heat_shift": float(mapping.actual_heat_shift[checkpoint_index]),
                "absolute_mapping_error": float(mapping.absolute_mapping_error[checkpoint_index]),
            }
            if any(
                not _close(float(event.get(key, math.nan)), expected)
                for key, expected in expected_mapping_values.items()
            ):
                raise RuntimeError(f"event heat mapping values mismatch for {pair}/{component.component_id}")
            alpha_current = 1.0 / (1.0 + float(mapping.current_heat_variance[checkpoint_index]))
            alpha_shifted = 1.0 / (1.0 + float(mapping.shifted_heat_variance[checkpoint_index]))
            expected_rho = math.sqrt(alpha_shifted / alpha_current)
            if not _close(float(event.get("current_alpha_bar", math.nan)), alpha_current) or not _close(
                float(event.get("shifted_alpha_bar", math.nan)), alpha_shifted
            ) or not _close(float(event.get("rho", math.nan)), expected_rho):
                raise RuntimeError(f"event alpha/rho mapping mismatch for {pair}/{component.component_id}")
            if event.get("shifted_model_evaluated") != (
                shifted_original_t != current_original_t
            ):
                raise RuntimeError(f"shifted-evaluation flag mismatch for {pair}/{component.component_id}")
            numeric_keys = (
                "current_alpha_bar", "shifted_alpha_bar", "rho",
                "current_heat_variance", "target_heat_variance", "shifted_heat_variance",
                "actual_heat_shift", "absolute_mapping_error", "raw_conditional_kl",
                "tempering_scale", "applied_conditional_kl", "max_conditional_kl",
                "raw_innovation_projection", "innovation_projection",
                "log_lr_increment", "cumulative_log_e",
                "running_max_log_e",
            )
            if any(not isinstance(event.get(key), (int, float)) or not math.isfinite(float(event[key])) for key in numeric_keys):
                raise RuntimeError(f"non-finite/missing event number for {pair}/{component.component_id}")
            raw_kl = float(event["raw_conditional_kl"])
            applied_kl = float(event["applied_conditional_kl"])
            scale = float(event["tempering_scale"])
            if raw_kl < -1e-14 or applied_kl < -1e-14 or applied_kl > max_conditional_kl + 2e-11:
                raise RuntimeError(f"invalid KL diagnostics for {pair}/{component.component_id}")
            if not 0 < scale <= 1.0 + 1e-14:
                raise RuntimeError(f"invalid tempering scale for {pair}/{component.component_id}")
            if not _close(applied_kl, raw_kl * scale * scale):
                raise RuntimeError(f"KL tempering identity failed for {pair}/{component.component_id}")
            if shifted_original_t == current_original_t and (
                abs(raw_kl) > 2e-14
                or abs(float(event["raw_innovation_projection"])) > 2e-14
            ):
                raise RuntimeError(f"identity heat mapping must define Q=P for {pair}/{component.component_id}")
            if not _close(float(event["max_conditional_kl"]), max_conditional_kl):
                raise RuntimeError(f"event KL cap mismatch for {pair}/{component.component_id}")
            increment = float(event["log_lr_increment"])
            raw_projection = float(event["raw_innovation_projection"])
            projection = float(event["innovation_projection"])
            if not _close(projection, scale * raw_projection):
                raise RuntimeError(f"innovation tempering identity failed for {pair}/{component.component_id}")
            if not _close(increment, projection - applied_kl):
                raise RuntimeError(f"likelihood increment identity failed for {pair}/{component.component_id}")
            if not _close(increment, scale * raw_projection - scale * scale * raw_kl):
                raise RuntimeError(f"raw likelihood reconstruction failed for {pair}/{component.component_id}")
            cumulative += increment
            running_max = max(running_max, cumulative)
            if not _close(float(event["cumulative_log_e"]), cumulative) or not _close(
                float(event["running_max_log_e"]), running_max
            ):
                raise RuntimeError(f"cumulative/running-max identity failed for {pair}/{component.component_id}")
            crossed_now = cumulative >= threshold
            if event.get("crossed_threshold_at_checkpoint") != crossed_now:
                raise RuntimeError(f"crossing flag mismatch for {pair}/{component.component_id}")
            if first_crossing is None and crossed_now:
                first_crossing = checkpoint_index
            if event.get("crossed_threshold_ever") != (first_crossing is not None):
                raise RuntimeError(f"running crossing flag mismatch for {pair}/{component.component_id}")
            component_cumulative_by_check[checkpoint_index][component.index] = cumulative
        if not _close(float(record.get("final_cumulative_log_e", math.nan)), cumulative):
            raise RuntimeError(f"component final log E mismatch for {pair}/{component.component_id}")
        if not _close(float(record.get("running_max_log_e", math.nan)), running_max):
            raise RuntimeError(f"component final running maximum mismatch for {pair}/{component.component_id}")
        if record.get("first_crossing_checkpoint_index") != first_crossing:
            raise RuntimeError(f"component first crossing mismatch for {pair}/{component.component_id}")

    mixture = signal.get("mixture")
    if not isinstance(mixture, dict) or mixture.get("weights") != [
        component.mixture_weight for component in components
    ]:
        raise RuntimeError(f"mixture weights are invalid for {pair}")
    mixture_events = mixture.get("events")
    if not isinstance(mixture_events, list) or len(mixture_events) != len(EVIDENCE_INTERNAL_TIMESTEPS):
        raise RuntimeError(f"mixture event count is invalid for {pair}")
    mixture_running_max = 0.0
    mixture_first: int | None = None
    for checkpoint_index, event in enumerate(mixture_events):
        expected = _log_equal_weight_mixture(component_cumulative_by_check[checkpoint_index])
        mixture_running_max = max(mixture_running_max, expected)
        crossed_now = expected >= threshold
        if mixture_first is None and crossed_now:
            mixture_first = checkpoint_index
        if event.get("checkpoint_index") != checkpoint_index or event.get(
            "internal_timestep"
        ) != EVIDENCE_INTERNAL_TIMESTEPS[checkpoint_index]:
            raise RuntimeError(f"mixture checkpoint identity mismatch for {pair}")
        if not _close(float(event.get("log_e_mixture", math.nan)), expected) or not _close(
            float(event.get("running_max_log_e_mixture", math.nan)), mixture_running_max
        ):
            raise RuntimeError(f"mixture arithmetic mismatch for {pair}")
        if event.get("crossed_threshold_at_checkpoint") != crossed_now or event.get(
            "crossed_threshold_ever"
        ) != (mixture_first is not None):
            raise RuntimeError(f"mixture crossing flag mismatch for {pair}")
    if not _close(float(mixture.get("final_log_e", math.nan)), _log_equal_weight_mixture(
        [float(record["final_cumulative_log_e"]) for record in component_records]
    )):
        raise RuntimeError(f"mixture final log E mismatch for {pair}")
    if not _close(float(mixture.get("running_max_log_e", math.nan)), mixture_running_max):
        raise RuntimeError(f"mixture final running maximum mismatch for {pair}")
    if mixture.get("first_crossing_checkpoint_index") != mixture_first:
        raise RuntimeError(f"mixture first crossing mismatch for {pair}")


def validate_observed_output_set(
    output_dir: Path,
    baseline: BaselineReference,
    pairs: Sequence[Pair],
    manifest_identity_sha256: str,
    runner_sha256: str,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    *,
    require_all: bool,
) -> set[Pair]:
    expected_images = {observed_pair_path(output_dir, pair).resolve(): pair for pair in pairs}
    expected_signals = {signal_pair_path(output_dir, pair).resolve(): pair for pair in pairs}
    image_root, signal_root = output_dir / "images", output_dir / "signals"
    actual_images = set(path.resolve() for path in image_root.rglob("*.png")) if image_root.exists() else set()
    actual_signals = set(path.resolve() for path in signal_root.rglob("*.json")) if signal_root.exists() else set()
    unexpected = sorted((actual_images - set(expected_images)) | (actual_signals - set(expected_signals)))
    if unexpected:
        raise RuntimeError(f"output contains unexpected image/signal: {unexpected[0]}")

    complete: set[Pair] = set()
    for pair in pairs:
        image_path = observed_pair_path(output_dir, pair)
        signal_path = signal_pair_path(output_dir, pair)
        if image_path.exists() != signal_path.exists():
            raise RuntimeError(f"strict resume found a partial image/signal pair: {pair}")
        if not image_path.exists():
            continue
        try:
            signal = json.loads(signal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read signal file {signal_path}") from exc
        validate_signal_payload(
            signal, pair, manifest_identity_sha256, runner_sha256,
            baseline.manifest_identity_sha256,
            components, max_conditional_kl, alpha,
        )
        with Image.open(image_path) as image:
            metadata = dict(image.info)
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"invalid observed image mode/size: {image_path}")
            image.verify()
        pixels = decoded_pixels(image_path)
        observed_sha = pixel_sha256(pixels)
        baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
        baseline_sha = pixel_sha256(baseline_pixels)
        if not np.array_equal(pixels, baseline_pixels):
            raise RuntimeError(f"observe-only P path differs from pure baseline pixels: {pair}")
        expected_metadata = {
            "experiment": EXPERIMENT,
            "class_id": str(pair[0]),
            "seed": str(pair[1]),
            "sample_stream_seed": str(sample_stream_seed(pair[1])),
            "pixel_sha256": observed_sha,
            "baseline_pixel_sha256": baseline_sha,
            "signal_payload_sha256": str(signal["payload_sha256"]),
            "manifest_identity_sha256": manifest_identity_sha256,
            "runner_sha256": runner_sha256,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"observed PNG metadata mismatch for {pair}: {mismatches}")
        if signal.get("pixel_sha256") != observed_sha or signal.get(
            "baseline_pixel_sha256"
        ) != baseline_sha:
            raise RuntimeError(f"signal pixel identity mismatch for {pair}")
        complete.add(pair)
    if require_all and len(complete) != len(pairs):
        raise RuntimeError(f"only {len(complete)}/{len(pairs)} observed pairs are complete")
    return complete


def _new_path_record(
    pair: Pair,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
) -> dict[str, Any]:
    return {
        "pair": pair,
        "components": [
            {
                "component_index": component.index,
                "component_id": component.component_id,
                "additive_heat_shift": component.additive_heat_shift,
                "mixture_weight": component.mixture_weight,
                "events": [],
                "cumulative": 0.0,
                "running_max": 0.0,
                "first_crossing": None,
            }
            for component in components
        ],
        "mixture_events": [],
        "mixture_running_max": 0.0,
        "mixture_first_crossing": None,
        "max_conditional_kl": max_conditional_kl,
        "alpha": alpha,
    }


def sample_observe_batch(
    diffusion: Any,
    model: torch.nn.Module,
    cond_fn: Callable[..., torch.Tensor],
    pairs: Sequence[Pair],
    *,
    device: torch.device,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    components: Sequence[ComponentSpec],
    max_conditional_kl: float,
    alpha: float,
    checkpoints: Sequence[int] = EVIDENCE_INTERNAL_TIMESTEPS,
    channels: int = 3,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    """Sample P unchanged while evaluating predictable Q/P evidence."""

    if not pairs:
        raise ValueError("cannot sample an empty pair batch")
    if diffusion.num_timesteps != len(timestep_map):
        raise ValueError("diffusion and supplied spaced-timestep map disagree")
    if tuple(checkpoints) != tuple(sorted(checkpoints, reverse=True)):
        raise ValueError("checkpoints must be in reverse sampling order")
    checkpoint_to_index = {int(t): index for index, t in enumerate(checkpoints)}
    if len(checkpoint_to_index) != len(checkpoints):
        raise ValueError("duplicate evidence checkpoint")
    if len(components) < 1:
        raise ValueError("at least one evidence component is required")

    streams = SeedRandomStreams(device, pairs)
    states = torch.cat(
        [
            streams.randn(index, (1, channels, image_size, image_size), dtype)
            for index in range(len(pairs))
        ],
        dim=0,
    )
    records = [
        _new_path_record(pair, components, max_conditional_kl, alpha) for pair in pairs
    ]
    shifted_evaluations = 0
    current_evaluations = 0
    threshold = -math.log(alpha)

    for internal_t in range(diffusion.num_timesteps - 1, -1, -1):
        next_states: list[torch.Tensor] = []
        for path_index, (class_id, _) in enumerate(pairs):
            x = states[path_index : path_index + 1]
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
                    raise AssertionError("P p_mean_variance did not make exactly one U-Net call")
                expected_original_t = int(timestep_map[internal_t])
                if raw_timestep_box != [expected_original_t]:
                    raise AssertionError(
                        f"wrapped current timestep mismatch: {raw_timestep_box} != {expected_original_t}"
                    )
                raw_current = raw_box[0]
                if raw_current.shape[1] != channels * 2:
                    raise AssertionError("official learned-variance U-Net must return 2*C channels")
                epsilon_current = raw_current[:, :channels]
                guided_mean = diffusion.condition_mean(
                    cond_fn,
                    out,
                    x,
                    t,
                    model_kwargs=model_kwargs,
                )
                # This stored FP32 standard deviation is the exact multiplier
                # used by the pure P transition below.  The LR code promotes it
                # directly to FP64, avoiding a square-then-square-root round
                # trip before constructing sigma*theta.
                p_standard_deviation = (
                    torch.exp(0.5 * out["log_variance"])
                    if internal_t > 0
                    else None
                )
                pending: list[tuple[Any, dict[str, Any]]] = []
                checkpoint_index = checkpoint_to_index.get(internal_t)
                if checkpoint_index is not None and internal_t > 0:
                    for component in components:
                        mapping = component.mapping
                        current_original_t = int(mapping.current_timestep[checkpoint_index])
                        shifted_original_t = int(mapping.shifted_timestep[checkpoint_index])
                        if current_original_t != expected_original_t:
                            raise AssertionError("precomputed current-timestep mapping is inconsistent")
                        alpha_current = float(original_alpha_bar[current_original_t])
                        alpha_shifted = float(original_alpha_bar[shifted_original_t])
                        rho = math.sqrt(alpha_shifted / alpha_current)
                        if shifted_original_t == current_original_t:
                            epsilon_shifted = epsilon_current
                        else:
                            # This is a direct call to the underlying U-Net.  The
                            # timestep is already an original 0..999 training
                            # index and must not pass through SpacedDiffusion's
                            # wrapper a second time.
                            shifted_input = x * rho
                            shifted_t = torch.tensor(
                                [shifted_original_t], dtype=torch.long, device=device
                            )
                            raw_shifted = model(shifted_input, shifted_t, y)
                            shifted_evaluations += 1
                            if raw_shifted.shape[1] != channels * 2:
                                raise AssertionError("shifted U-Net output must have 2*C channels")
                            epsilon_shifted = raw_shifted[:, :channels]
                        theta = normalized_heat_score_pullback_difference(
                            epsilon_current,
                            epsilon_shifted,
                            torch.tensor([alpha_current], dtype=torch.float64, device=device),
                            torch.tensor([alpha_shifted], dtype=torch.float64, device=device),
                        )
                        if p_standard_deviation is None:
                            raise AssertionError("stochastic checkpoint has no standard deviation")
                        tempered = kl_tempered_score_mean_shift_from_standard_deviation(
                            theta, p_standard_deviation, max_conditional_kl
                        )
                        pending.append(
                            (
                                tempered,
                                {
                                    "checkpoint_index": checkpoint_index,
                                    "internal_timestep": internal_t,
                                    "current_original_timestep": current_original_t,
                                    "shifted_original_timestep": shifted_original_t,
                                    "current_alpha_bar": alpha_current,
                                    "shifted_alpha_bar": alpha_shifted,
                                    "rho": rho,
                                    "current_heat_variance": float(mapping.current_heat_variance[checkpoint_index]),
                                    "target_heat_variance": float(mapping.target_heat_variance[checkpoint_index]),
                                    "shifted_heat_variance": float(mapping.shifted_heat_variance[checkpoint_index]),
                                    "actual_heat_shift": float(mapping.actual_heat_shift[checkpoint_index]),
                                    "absolute_mapping_error": float(mapping.absolute_mapping_error[checkpoint_index]),
                                    "shifted_model_evaluated": shifted_original_t != current_original_t,
                                },
                            )
                        )

                if internal_t > 0:
                    noise = streams.randn(path_index, x.shape, dtype)
                    if p_standard_deviation is None:
                        raise AssertionError("stochastic transition has no standard deviation")
                    x_next = guided_mean + p_standard_deviation * noise
                else:
                    noise = None
                    x_next = guided_mean

                if pending:
                    if noise is None:
                        raise AssertionError("deterministic final transition cannot carry evidence")
                    path_record = records[path_index]
                    for component_index, (tempered, event) in enumerate(pending):
                        raw_increment = same_covariance_log_lr_from_noise(
                            tempered.raw_whitened_shift, noise
                        )
                        increment = same_covariance_log_lr_from_noise(
                            tempered.whitened_shift, noise
                        )
                        raw_projection = float(raw_increment.innovation_projection.item())
                        raw_kl_from_increment = float(raw_increment.conditional_kl.item())
                        value = float(increment.value.item())
                        projection = float(increment.innovation_projection.item())
                        applied_kl = float(increment.conditional_kl.item())
                        if not _close(raw_kl_from_increment, float(tempered.raw_kl.item())):
                            raise AssertionError("raw KL and raw whitened-shift norm disagree")
                        component_record = path_record["components"][component_index]
                        component_record["cumulative"] += value
                        component_record["running_max"] = max(
                            component_record["running_max"], component_record["cumulative"]
                        )
                        crossed_now = component_record["cumulative"] >= threshold
                        if crossed_now and component_record["first_crossing"] is None:
                            component_record["first_crossing"] = int(event["checkpoint_index"])
                        event.update(
                            {
                                "raw_conditional_kl": float(tempered.raw_kl.item()),
                                "raw_innovation_projection": raw_projection,
                                "tempering_scale": float(tempered.scale.item()),
                                "applied_conditional_kl": applied_kl,
                                "max_conditional_kl": max_conditional_kl,
                                "innovation_projection": projection,
                                "log_lr_increment": value,
                                "cumulative_log_e": component_record["cumulative"],
                                "running_max_log_e": component_record["running_max"],
                                "crossed_threshold_at_checkpoint": crossed_now,
                                "crossed_threshold_ever": component_record["first_crossing"] is not None,
                            }
                        )
                        component_record["events"].append(event)

                    component_log_e = torch.tensor(
                        [record["cumulative"] for record in path_record["components"]],
                        dtype=torch.float64,
                        device=device,
                    ).reshape(1, -1)
                    weights = [component.mixture_weight for component in components]
                    mixture_log_e = float(log_e_mixture(component_log_e, weights).item())
                    path_record["mixture_running_max"] = max(
                        path_record["mixture_running_max"], mixture_log_e
                    )
                    mixture_crossed = mixture_log_e >= threshold
                    if mixture_crossed and path_record["mixture_first_crossing"] is None:
                        path_record["mixture_first_crossing"] = checkpoint_index
                    path_record["mixture_events"].append(
                        {
                            "checkpoint_index": checkpoint_index,
                            "internal_timestep": internal_t,
                            "log_e_mixture": mixture_log_e,
                            "running_max_log_e_mixture": path_record["mixture_running_max"],
                            "crossed_threshold_at_checkpoint": mixture_crossed,
                            "crossed_threshold_ever": path_record["mixture_first_crossing"] is not None,
                        }
                    )
            next_states.append(x_next.detach())
        states = torch.cat(next_states, dim=0)

    expected_draws = diffusion.num_timesteps
    if streams.draw_counts != [expected_draws] * len(pairs):
        raise AssertionError(
            f"unexpected RNG consumption: {streams.draw_counts} != {expected_draws} per path"
        )
    finalized: list[dict[str, Any]] = []
    for path_record in records:
        component_records = []
        for record in path_record["components"]:
            if len(record["events"]) != len(checkpoints):
                raise AssertionError("a path is missing one or more evidence checkpoints")
            component_records.append(
                {
                    "component_index": record["component_index"],
                    "component_id": record["component_id"],
                    "additive_heat_shift": record["additive_heat_shift"],
                    "mixture_weight": record["mixture_weight"],
                    "events": record["events"],
                    "final_cumulative_log_e": record["cumulative"],
                    "running_max_log_e": record["running_max"],
                    "first_crossing_checkpoint_index": record["first_crossing"],
                }
            )
        final_component_logs = [record["final_cumulative_log_e"] for record in component_records]
        finalized.append(
            {
                "components": component_records,
                "mixture": {
                    "weights": [component.mixture_weight for component in components],
                    "events": path_record["mixture_events"],
                    "final_log_e": _log_equal_weight_mixture(final_component_logs),
                    "running_max_log_e": path_record["mixture_running_max"],
                    "first_crossing_checkpoint_index": path_record["mixture_first_crossing"],
                },
            }
        )
    return states, finalized, {
        "reverse_steps": int(diffusion.num_timesteps),
        "stochastic_reverse_steps": int(diffusion.num_timesteps - 1),
        "deterministic_final_steps": 1,
        "gaussian_draws_per_path_including_initial": expected_draws,
        "neural_eval_batch_size": 1,
        "current_unet_evaluations_per_path": current_evaluations // len(pairs),
        "shifted_unet_evaluations_per_path": shifted_evaluations // len(pairs),
        "evidence_checkpoints_per_path": len(checkpoints),
        "interventions": 0,
    }


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
    evidence_path = Path(__file__).with_name("adm64_path_evidence.py").resolve()
    baseline_runner_path = Path(__file__).with_name("reproduce_adm64_guided.py").resolve()
    pair_set_sha = sha256_json([[class_id, seed] for class_id, seed in protocol.pairs])
    alpha_sha = hashlib.sha256(
        np.ascontiguousarray(original_alpha_bar, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "observe_only_no_intervention_no_rejection_no_rollback_no_resampling",
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
            "acceptance_rule": "every decoded observed PNG must be byte-identical to its baseline PNG",
        },
        "official_model_config": OFFICIAL_MODEL_CONFIG,
        "official_classifier_config": OFFICIAL_CLASSIFIER_CONFIG,
        "p_sampler": {
            "name": "OpenAI classifier-guided ancestral DDPM",
            "classifier_scale": CLASSIFIER_SCALE,
            "timestep_respacing": "250",
            "clip_denoised": True,
            "learned_diagonal_variance": True,
            "final_internal_timestep_zero": "P deterministic mean; Q=P; no LR and no Gaussian draw",
            "state_dtype": "torch.float32",
            "innovation_dtype": "torch.float32 drawn from the path-owned CUDA generator",
        },
        "primary_evidence_definition": {
            "original_schedule": {
                "name": "official cosine 1000-step alpha_bar",
                "length": int(original_alpha_bar.size),
                "alpha_bar_float64_bytes_sha256": alpha_sha,
                "normalized_heat_time": "nu=(1-alpha_bar)/alpha_bar",
            },
            "spaced_timestep_map": timestep_map.tolist(),
            "internal_checkpoints_reverse_order": list(EVIDENCE_INTERNAL_TIMESTEPS),
            "checkpoint_count": len(EVIDENCE_INTERNAL_TIMESTEPS),
            "components": [
                _mapping_manifest_record(EVIDENCE_INTERNAL_TIMESTEPS, component)
                for component in components
            ],
            "mixture": "fixed equal-weight arithmetic mixture of component E-processes",
            "max_conditional_kl_per_component_checkpoint": args.max_conditional_kl,
            "cap_policy": (
                "one fixed cap shared by all heat-shift components; raw sufficient statistics "
                "are retained for predeclared diagnostics, but the primary cap may not be "
                "selected after viewing endpoint labels"
            ),
            "alpha": args.alpha,
            "log_e_crossing_threshold": -math.log(args.alpha),
            "q_mean": "mu_Q=mu_P+v_P*theta after predictable per-checkpoint KL tempering",
            "q_covariance": (
                "the same diagonal Gaussian covariance as P; likelihood arithmetic uses "
                "P's stored FP32 sigma=exp(0.5*learned_log_variance) promoted directly "
                "to FP64, without a square-then-square-root round trip"
            ),
            "theta": "rho*(-epsilon_shift/sqrt(1-alpha_bar_shift))-(-epsilon_current/sqrt(1-alpha_bar_current))",
            "same_heat_state": "shifted U-Net input is rho*x, rho=sqrt(alpha_bar_shift/alpha_bar_current)",
            "unet_channels": "first three epsilon channels only; shifted variance channels ignored",
            "classifier": "classifier gradient remains in P mean only; excluded from theta",
            "noncheckpoint_steps": "Q=P and log likelihood increment is exactly zero",
            "increment": "float64 dot(sigma*tempered_theta, actual_drawn_noise)-applied_KL",
            "crossings_are_observed_only": True,
        },
        "rng": {
            "owner": "one torch.Generator per path initialized solely from public seed",
            "seed_function": "imported unchanged from reproduce_adm64_guided.py",
            "paired_across_classes": True,
            "evidence_model_calls_consume_no_random_numbers": True,
        },
        "batch_invariance": {
            "neural_eval_batch_size": 1,
            "logical_batch_affects_only_scheduling": True,
        },
        "outputs": {
            "png": "images/class_{class_id:04d}/{seed:019d}.png",
            "signal": "signals/class_{class_id:04d}/{seed:019d}.json",
            "signal_contents": "mapping, raw/applied KL, raw/applied innovation projection, cap/scale, increments, cumulative/running-max/crossing for components and mixture",
            "strict_resume": "manifest, self-hashed signals, PNG metadata/pixel hash, expected file set, and pure-P pixel identity",
        },
        "sources": {
            "guided_diffusion_root": str(source_root),
            "guided_diffusion_revision": source_revision,
            "guided_diffusion_tracked_dirty": source_dirty,
            "guided_diffusion_python_tree_sha256": sha256_python_tree(source_root / "guided_diffusion"),
            "baseline_runner": {"path": str(baseline_runner_path), "sha256": sha256_file(baseline_runner_path)},
            "evidence_primitives": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
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
        if existing != manifest:
            differing = sorted(
                key for key in set(existing) | set(manifest)
                if existing.get(key) != manifest.get(key)
            )
            raise RuntimeError(
                f"output directory has an incompatible manifest; differing keys: {', '.join(differing)}"
            )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory without a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, path)


def run_observation(args: argparse.Namespace, protocol: Protocol) -> None:
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if not math.isfinite(args.max_conditional_kl) or args.max_conditional_kl <= 0:
        raise ValueError("--max-conditional-kl must be finite and strictly positive")
    if not math.isfinite(args.alpha) or not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between zero and one")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official FP16 ADM64 sampler requires CUDA")
    if args.output_dir.resolve() == args.baseline_dir.resolve():
        raise ValueError("observed output directory must differ from the pure-baseline directory")

    configure_determinism()
    torch.cuda.set_device(device)
    model_checkpoint_record = validate_checkpoint(args.model_path, DIFFUSION_CHECKPOINT)
    classifier_checkpoint_record = validate_checkpoint(args.classifier_path, CLASSIFIER_CHECKPOINT)
    original_alpha_bar, timestep_map = original_schedule_and_timestep_map(
        args.guided_diffusion_root
    )
    components = build_component_specs(
        original_alpha_bar, timestep_map, args.heat_shifts
    )
    baseline = load_baseline_reference(
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
    )
    create_or_validate_manifest(args.output_dir, manifest)
    manifest_identity = manifest["identity_sha256"]
    runner_sha = manifest["runner"]["sha256"]
    complete_pairs = validate_observed_output_set(
        args.output_dir, baseline, protocol.pairs, manifest_identity, runner_sha,
        components, args.max_conditional_kl, args.alpha, require_all=False,
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
            raise RuntimeError("completion.json exists but one or more strict outputs are missing")
        print(json.dumps(completion, ensure_ascii=False, indent=2))
        return

    pending_pairs = [pair for pair in protocol.pairs if pair not in complete_pairs]
    start = time.monotonic()
    generated = 0
    sampling_record: dict[str, int] | None = None
    if pending_pairs:
        model, diffusion, classifier = load_official_models(
            args.guided_diffusion_root, args.model_path, args.classifier_path, device
        )
        if list(diffusion.timestep_map) != timestep_map.tolist():
            raise RuntimeError("loaded SpacedDiffusion timestep map differs from frozen manifest map")
        if not np.allclose(
            np.asarray(diffusion.alphas_cumprod, dtype=np.float64),
            original_alpha_bar[timestep_map],
            rtol=2e-13,
            atol=2e-15,
        ):
            raise RuntimeError("loaded spaced alpha_bar differs from original-schedule restriction")
        _, cond_fn = make_guided_functions(model, classifier)
        for logical_batch in chunks(pending_pairs, args.batch):
            samples, evidence_records, sampling_record = sample_observe_batch(
                diffusion,
                model,
                cond_fn,
                logical_batch,
                device=device,
                original_alpha_bar=original_alpha_bar,
                timestep_map=timestep_map,
                components=components,
                max_conditional_kl=args.max_conditional_kl,
                alpha=args.alpha,
            )
            for index, pair in enumerate(logical_batch):
                pixels = pixels_from_sample(samples[index])
                baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
                if not np.array_equal(pixels, baseline_pixels):
                    raise RuntimeError(
                        f"observe-only sampler changed P pixels relative to pure baseline: {pair}"
                    )
                signal = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": EXPERIMENT,
                    "class_id": pair[0],
                    "seed": pair[1],
                    "sample_stream_seed": sample_stream_seed(pair[1]),
                    "manifest_identity_sha256": manifest_identity,
                    "runner_sha256": runner_sha,
                    "pixel_sha256": pixel_sha256(pixels),
                    "baseline_pixel_sha256": pixel_sha256(baseline_pixels),
                    "baseline_manifest_identity_sha256": baseline.manifest_identity_sha256,
                    "max_conditional_kl": args.max_conditional_kl,
                    "alpha": args.alpha,
                    "log_e_crossing_threshold": -math.log(args.alpha),
                    "internal_timesteps": list(EVIDENCE_INTERNAL_TIMESTEPS),
                    "checkpoint_count": len(EVIDENCE_INTERNAL_TIMESTEPS),
                    "intervention_count": 0,
                    **evidence_records[index],
                }
                validate_signal_payload(
                    {**signal, "payload_sha256": _canonical_payload_sha(signal, "payload_sha256")},
                    pair, manifest_identity, runner_sha,
                    baseline.manifest_identity_sha256, components,
                    args.max_conditional_kl, args.alpha,
                )
                save_observed_pair(
                    pixels, signal, args.output_dir, pair, manifest_identity, runner_sha
                )
                generated += 1
            elapsed = time.monotonic() - start
            print(
                f"observed {generated}/{len(pending_pairs)} new P paths "
                f"({len(complete_pairs)} already complete, {elapsed:.1f}s)",
                flush=True,
            )

    final_pairs = validate_observed_output_set(
        args.output_dir, baseline, protocol.pairs, manifest_identity, runner_sha,
        components, args.max_conditional_kl, args.alpha, require_all=True,
    )
    final_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest_identity,
        "pair_set_sha256": manifest["pair_set_sha256"],
        "generated_this_run": generated,
        "already_complete": len(complete_pairs),
        "total_expected": len(protocol.pairs),
        "total_complete": len(final_pairs),
        "logical_batch_requested": args.batch,
        "neural_eval_batch_size": 1,
        "interventions": 0,
        "sampling_record": sampling_record,
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
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None or x.shape[0] != 1:
            raise AssertionError("toy model requires singleton class conditioning")
        self.calls.append(int(t.item()))
        epsilon = 0.07 * x + t.view(-1, 1, 1, 1).to(x.dtype) * 0.002
        epsilon = epsilon + y.view(-1, 1, 1, 1).to(x.dtype) * 0.0001
        variance_logits = torch.tanh(0.03 * x)
        return torch.cat([epsilon, variance_logits], dim=1)


def run_self_test() -> None:
    if len(EVIDENCE_INTERNAL_TIMESTEPS) != 32 or EVIDENCE_INTERNAL_TIMESTEPS[-1] != 1:
        raise AssertionError("frozen checkpoint schedule is invalid")
    if parse_float_spec("0.01,0.1,1,10") != DEFAULT_HEAT_SHIFTS:
        raise AssertionError("heat-shift parser failed")

    device = torch.device("cpu")
    diffusion = _ToyDiffusion()
    alpha_bar = np.array(
        [0.99, 0.96, 0.90, 0.80, 0.68, 0.54, 0.39, 0.24, 0.10], dtype=np.float64
    )
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    checkpoints = (4, 2)
    components = build_component_specs(alpha_bar, timestep_map, (0.1, 1.0), checkpoints)
    pairs: tuple[Pair, ...] = ((3, 7), (9, 7), (3, 8))

    def toy_cond(
        x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            raise AssertionError("toy classifier requires y")
        return torch.full_like(x, 0.003) + y.view(-1, 1, 1, 1) * 0.00001

    def grouped(size: int) -> tuple[dict[Pair, torch.Tensor], dict[Pair, dict[str, Any]]]:
        samples_by_pair: dict[Pair, torch.Tensor] = {}
        signals_by_pair: dict[Pair, dict[str, Any]] = {}
        for logical_batch in chunks(pairs, size):
            model = _ToyModel()
            samples, records, accounting = sample_observe_batch(
                diffusion,
                model,
                toy_cond,
                logical_batch,
                device=device,
                original_alpha_bar=alpha_bar,
                timestep_map=timestep_map,
                components=components,
                max_conditional_kl=0.02,
                alpha=0.05,
                checkpoints=checkpoints,
                channels=1,
                image_size=2,
            )
            if accounting["interventions"] != 0 or accounting[
                "gaussian_draws_per_path_including_initial"
            ] != diffusion.num_timesteps:
                raise AssertionError("toy accounting failed")
            for index, pair in enumerate(logical_batch):
                samples_by_pair[pair] = samples[index].clone()
                signals_by_pair[pair] = records[index]
        return samples_by_pair, signals_by_pair

    singleton_samples, singleton_signals = grouped(1)
    for logical_batch_size in (2, 3):
        samples, signals = grouped(logical_batch_size)
        for pair in pairs:
            if not torch.equal(samples[pair], singleton_samples[pair]):
                raise AssertionError("observe path changed with logical grouping")
            if signals[pair] != singleton_signals[pair]:
                raise AssertionError("evidence changed with logical grouping")

    # Compare to the uninstrumented P sampler: extra shifted U-Net evaluations
    # must not consume RNG or alter a single endpoint bit.
    baseline_model = _ToyModel()
    baseline_samples, _ = sample_batch_invariant(
        diffusion,
        baseline_model,
        toy_cond,
        pairs,
        device=device,
        channels=1,
        image_size=2,
    )
    for index, pair in enumerate(pairs):
        if not torch.equal(baseline_samples[index], singleton_samples[pair]):
            raise AssertionError("instrumented P path differs from pure toy baseline")

    for pair, record in singleton_signals.items():
        if len(record["components"]) != 2 or any(
            len(component["events"]) != len(checkpoints)
            for component in record["components"]
        ):
            raise AssertionError(f"toy evidence event count failed for {pair}")
        for component in record["components"]:
            if any(
                event["applied_conditional_kl"] > 0.02 + 2e-12
                for event in component["events"]
            ):
                raise AssertionError("toy KL cap failed")

    with tempfile.TemporaryDirectory(prefix="adm64-evidence-observe-self-test-") as temporary:
        path = Path(temporary) / "payload.json"
        payload = {"finite": True, "value": 1.25}
        payload["payload_sha256"] = _canonical_payload_sha(payload, "payload_sha256")
        atomic_json_dump(payload, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded["payload_sha256"] != _canonical_payload_sha(loaded, "payload_sha256"):
            raise AssertionError("self-hashed atomic signal failed")

    print(
        "self-test passed: fixed checkpoints, heat mapping, exact toy P-path identity, "
        "seed-only grouping invariance, KL-capped FP64 LR records, and atomic self-hashed signals"
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
    parser.add_argument("--guided-diffusion-root", type=Path, default=guided_root)
    parser.add_argument(
        "--model-path", type=Path,
        default=guided_root / "checkpoints" / DIFFUSION_CHECKPOINT.filename,
    )
    parser.add_argument(
        "--classifier-path", type=Path,
        default=guided_root / "checkpoints" / CLASSIFIER_CHECKPOINT.filename,
    )
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch", type=int, default=4,
        help="logical scheduling batch; all neural evaluations remain singleton",
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
    protocol = protocol_from_args(args)
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    if args.baseline_dir is None:
        args.baseline_dir = data_root / "cross_scale_evidence" / "adm64_guided" / args.protocol
    if args.output_dir is None:
        args.output_dir = (
            data_root / "cross_scale_evidence" / "adm64_cross_scale_evidence" / args.protocol
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
                    "protocol": args.protocol,
                    "class_ids": list(protocol.class_ids),
                    "seeds": list(protocol.seeds),
                    "sample_count": len(protocol.pairs),
                    "baseline_dir": str(args.baseline_dir),
                    "output_dir": str(args.output_dir),
                    "heat_shifts": list(args.heat_shifts),
                    "max_conditional_kl": args.max_conditional_kl,
                    "alpha": args.alpha,
                    "internal_checkpoints": list(EVIDENCE_INTERNAL_TIMESTEPS),
                    "observe_only": True,
                    "interventions": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_observation(args, protocol)


if __name__ == "__main__":
    main()

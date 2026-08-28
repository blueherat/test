#!/usr/bin/env python3
"""Oracle/mechanics-only localized suffix resampling for frozen ADM64 traces.

This diagnostic starts from an explicitly supplied pre-transition ADM state
``x_t`` in a previously validated all-step local-evidence trace.  It produces:

* an original replay using every stored baseline innovation (which must decode
  pixel-identically to the frozen baseline),
* N independently seeded localized attempts, where each later Gaussian
  innovation is fresh only inside a fixed, user-supplied image-space rectangle
  and is copied from the observed baseline outside that rectangle, and
* one same-checkpoint full-fresh P-suffix control.

There is no detector, evidence-selected rollback, retry loop, scoring, ranking,
or post-hoc choice of a preferred attempt.  The rectangle and rollback time are
oracle inputs, and the outside-mask future innovations are read from the saved
reference trajectory.  In particular, a localized attempt is a hybrid, trace-
conditioned transition: outside-mask innovations are fixed to the already
observed baseline path.  It is *not* a fresh P suffix and may not cite the
conditional Ville/retry or output-distribution bounds of the proposed method.
The denoiser couples spatial positions, so replacing innovation only inside the
rectangle does not guarantee that final pixels outside it remain unchanged.

The complete output directory is staged and atomically installed.  Existing
targets are never overwritten.  Every executed transition records state,
prediction, mean, sigma, original/fresh/used innovation, and endpoint hashes;
exact state and used-noise trajectories are stored in self-hashed NPZ files.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:  # Package and direct CLI imports.
    from .observe_adm64_cross_scale_evidence import (
        BaselineReference,
        _canonical_payload_sha,
        decoded_pixels,
        original_schedule_and_timestep_map,
        pixel_sha256,
    )
    from .observe_adm64_local_path_evidence import (
        EXPERIMENT as LOCAL_TRACE_EXPERIMENT,
        SCHEMA_VERSION as LOCAL_TRACE_SCHEMA_VERSION,
        STOCHASTIC_INTERNAL_TIMESTEPS,
        _validate_trace_file as validate_local_trace_file,
        build_local_spec,
        image_path as local_trace_image_path,
        signal_path as local_trace_signal_path,
        trace_path as local_trace_path,
    )
    from .reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        atomic_json_dump,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        pixels_from_sample,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
        validate_sample_png,
    )
except ImportError:  # pragma: no cover.
    from observe_adm64_cross_scale_evidence import (
        BaselineReference,
        _canonical_payload_sha,
        decoded_pixels,
        original_schedule_and_timestep_map,
        pixel_sha256,
    )
    from observe_adm64_local_path_evidence import (
        EXPERIMENT as LOCAL_TRACE_EXPERIMENT,
        SCHEMA_VERSION as LOCAL_TRACE_SCHEMA_VERSION,
        STOCHASTIC_INTERNAL_TIMESTEPS,
        _validate_trace_file as validate_local_trace_file,
        build_local_spec,
        image_path as local_trace_image_path,
        signal_path as local_trace_signal_path,
        trace_path as local_trace_path,
    )
    from reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        atomic_json_dump,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        pixels_from_sample,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
        validate_sample_png,
    )


EXPERIMENT = "adm64_oracle_localized_hybrid_suffix"
SCHEMA_VERSION = 1
RNG_NAMESPACE = "eqvae-adm64-oracle-localized-suffix-v1"
MAX_ATTEMPTS = 32
Rectangle = tuple[int, int, int, int]


@dataclass(frozen=True)
class InputTrace:
    root: Path
    manifest_identity_sha256: str
    signal_payload_sha256: str
    trace_sha256: str
    trace_record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    total_K_budget: float
    grid_size: int


@dataclass
class BranchResult:
    role: str
    stream_seed: int | None
    final_state: torch.Tensor
    transitions: list[dict[str, Any]]
    trace_arrays: dict[str, np.ndarray]
    gaussian_draws: int


def parse_rectangle(value: str) -> Rectangle:
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 4:
        raise argparse.ArgumentTypeError("rectangle must be x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rectangle coordinates must be integers") from exc
    if not (0 <= x0 < x1 <= IMAGE_SIZE and 0 <= y0 < y1 <= IMAGE_SIZE):
        raise argparse.ArgumentTypeError(
            "rectangle uses half-open image coordinates and must lie inside [0,64)^2"
        )
    return x0, y0, x1, y1


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    array = _tensor_numpy(tensor)
    values = array.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("sampler produced a non-finite tensor")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_bytes_sha256": _array_sha(array),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "root_mean_square": float(np.sqrt(np.mean(np.square(values)))),
    }


def suffix_stream_seed(
    pair: Pair,
    role: str,
    rollback_internal_timestep: int,
    rectangle: Rectangle,
    attempt_index: int,
) -> int:
    if role not in ("localized_attempt", "full_fresh_control"):
        raise ValueError(f"invalid fresh suffix role: {role}")
    x0, y0, x1, y1 = rectangle
    payload = (
        f"{RNG_NAMESPACE}\0{role}\0{pair[0]}\0{pair[1]}\0"
        f"{rollback_internal_timestep}\0{x0},{y0},{x1},{y1}\0{attempt_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _read_self_hashed_json(path: Path, digest_key: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read JSON: {path}") from exc
    digest = payload.get(digest_key)
    if not isinstance(digest, str) or digest != _canonical_payload_sha(payload, digest_key):
        raise RuntimeError(f"invalid self-hash in {path}")
    return payload


def load_baseline_pair(
    root: Path,
    pair: Pair,
    *,
    model_sha256: str,
    classifier_sha256: str,
) -> BaselineReference:
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    if manifest.get("experiment") != "adm64_classifier_guided_reproduction":
        raise RuntimeError("baseline input is not the frozen ADM64 reproduction")
    for name, expected in (("diffusion", model_sha256), ("classifier", classifier_sha256)):
        if manifest.get("checkpoints", {}).get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"baseline {name} checkpoint differs from the requested model")
    classes, seeds = manifest.get("class_ids"), manifest.get("seeds")
    if not isinstance(classes, list) or not isinstance(seeds, list):
        raise RuntimeError("baseline manifest lacks class/seed axes")
    full_pairs = tuple((int(c), int(s)) for c in classes for s in seeds)
    if pair not in full_pairs:
        raise RuntimeError(f"requested pair {pair} is absent from baseline")
    pair_set_sha = sha256_json([[c, s] for c, s in full_pairs])
    if manifest.get("pair_set_sha256") != pair_set_sha:
        raise RuntimeError("baseline pair-set identity is invalid")
    runner = Path(__file__).with_name("reproduce_adm64_guided.py").resolve()
    runner_sha = sha256_file(runner)
    if manifest.get("runner", {}).get("sha256") != runner_sha:
        raise RuntimeError("baseline runner hash differs from current frozen source")
    completion = validate_existing_completion(
        root / "completion.json",
        manifest_identity_sha256=manifest["identity_sha256"],
        pair_set_sha256=pair_set_sha,
        total_expected=len(full_pairs),
    )
    if completion is None:
        raise RuntimeError("baseline input has no strict completion marker")
    validate_sample_png(
        baseline_pair_path(root, pair), pair, manifest["identity_sha256"], runner_sha
    )
    return BaselineReference(
        root.resolve(), manifest["identity_sha256"], runner_sha, pair_set_sha
    )


def load_input_trace(
    root: Path,
    pair: Pair,
    baseline: BaselineReference,
    *,
    model_sha256: str,
    classifier_sha256: str,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
) -> InputTrace:
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    if manifest.get("schema_version") != LOCAL_TRACE_SCHEMA_VERSION or manifest.get(
        "experiment"
    ) != LOCAL_TRACE_EXPERIMENT:
        raise RuntimeError("input is not an all-step ADM64 local-evidence trace")
    for name, expected in (("diffusion", model_sha256), ("classifier", classifier_sha256)):
        if manifest.get("checkpoints", {}).get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"input trace {name} checkpoint identity changed")
    pure_p = manifest.get("pure_p_baseline", {})
    if pure_p.get("manifest_identity_sha256") != baseline.manifest_identity_sha256:
        raise RuntimeError("input trace and supplied frozen baseline have different identities")
    classes, seeds = manifest.get("class_ids"), manifest.get("seeds")
    if not isinstance(classes, list) or not isinstance(seeds, list):
        raise RuntimeError("input trace manifest lacks class/seed axes")
    full_pairs = tuple((int(c), int(s)) for c in classes for s in seeds)
    if pair not in full_pairs:
        raise RuntimeError(f"requested pair {pair} is absent from input trace")
    pair_set_sha = sha256_json([[c, s] for c, s in full_pairs])
    if manifest.get("pair_set_sha256") != pair_set_sha:
        raise RuntimeError("input trace pair-set identity is invalid")
    completion = validate_existing_completion(
        root / "completion.json",
        manifest_identity_sha256=manifest["identity_sha256"],
        pair_set_sha256=pair_set_sha,
        total_expected=len(full_pairs),
    )
    if completion is None:
        raise RuntimeError("input trace has no strict completion marker")
    observer = Path(__file__).with_name("observe_adm64_local_path_evidence.py").resolve()
    if manifest.get("runner", {}).get("sha256") != sha256_file(observer):
        raise RuntimeError("input trace was not produced by the current local observer")

    local_config = manifest.get("local_operational_Q", {})
    try:
        total_K_budget = float(local_config["total_K_budget"])
        grid_size = int(local_config["grid_size_per_axis"])
    except Exception as exc:
        raise RuntimeError("input trace manifest lacks local-Q configuration") from exc
    spec = build_local_spec(original_alpha_bar, timestep_map, total_K_budget)
    signal_file = local_trace_signal_path(root, pair)
    signal = _read_self_hashed_json(signal_file, "payload_sha256")
    fixed_signal = {
        "schema_version": LOCAL_TRACE_SCHEMA_VERSION,
        "experiment": LOCAL_TRACE_EXPERIMENT,
        "class_id": pair[0],
        "seed": pair[1],
        "sample_stream_seed": sample_stream_seed(pair[1]),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "baseline_manifest_identity_sha256": baseline.manifest_identity_sha256,
        "observer_changed_P": False,
        "operational_LR_only": True,
        "image_quality_claimed": False,
    }
    mismatches = {
        key: (signal.get(key), value)
        for key, value in fixed_signal.items()
        if signal.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"input trace signal identity mismatch: {mismatches}")
    trace_file = local_trace_path(root, pair)
    trace_record = signal.get("trace")
    if not isinstance(trace_record, dict):
        raise RuntimeError("input signal lacks its trace record")
    reconstructed_summary = validate_local_trace_file(
        trace_file,
        trace_record,
        root,
        spec=spec,
        total_K_budget=total_K_budget,
        grid_size=grid_size,
    )
    if signal.get("summary") != reconstructed_summary:
        raise RuntimeError("input trace summary does not reconstruct")
    try:
        with np.load(trace_file, allow_pickle=False) as archive:
            arrays = {
                key: np.ascontiguousarray(archive[key]) for key in archive.files
            }
    except Exception as exc:
        raise RuntimeError(f"cannot load validated trace {trace_file}") from exc

    observed_pixels = decoded_pixels(local_trace_image_path(root, pair))
    baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
    if not np.array_equal(observed_pixels, baseline_pixels):
        raise RuntimeError("input observer endpoint is not pixel-identical to frozen baseline")
    return InputTrace(
        root=root.resolve(),
        manifest_identity_sha256=manifest["identity_sha256"],
        signal_payload_sha256=signal["payload_sha256"],
        trace_sha256=trace_record["sha256"],
        trace_record=trace_record,
        arrays=arrays,
        total_K_budget=total_K_budget,
        grid_size=grid_size,
    )


def _trace_row_by_t(trace: dict[str, np.ndarray]) -> dict[int, int]:
    timesteps = trace.get("internal_timestep")
    if timesteps is None or timesteps.ndim != 1:
        raise ValueError("input trace lacks a one-dimensional internal timestep array")
    mapping = {int(t): index for index, t in enumerate(timesteps.tolist())}
    if len(mapping) != len(timesteps):
        raise ValueError("input trace contains duplicate timesteps")
    return mapping


def run_suffix_branch(
    diffusion: Any,
    model_fn: Callable[..., torch.Tensor],
    cond_fn: Callable[..., torch.Tensor],
    pair: Pair,
    trace: dict[str, np.ndarray],
    *,
    rollback_internal_timestep: int,
    rectangle: Rectangle,
    role: str,
    stream_seed: int | None,
    device: torch.device,
    timestep_map: np.ndarray,
) -> BranchResult:
    """Replay or resample one suffix from a validated pre-transition x_t."""

    if role not in ("original_replay", "localized_attempt", "full_fresh_control"):
        raise ValueError(f"invalid branch role: {role}")
    if role == "original_replay" and stream_seed is not None:
        raise ValueError("original replay must not construct a fresh RNG")
    if role != "original_replay" and stream_seed is None:
        raise ValueError("fresh suffix branch requires its domain-separated seed")
    rows = _trace_row_by_t(trace)
    if rollback_internal_timestep not in rows:
        raise ValueError("rollback timestep is absent from the stochastic trace")
    if rollback_internal_timestep <= 0 or rollback_internal_timestep >= diffusion.num_timesteps:
        raise ValueError("rollback timestep must be a stochastic ADM internal timestep")
    start_row = rows[rollback_internal_timestep]
    x = torch.from_numpy(trace["x_t"][start_row]).unsqueeze(0).to(device=device)
    if x.dtype != torch.float32:
        raise RuntimeError("validated ADM rollback state must be float32")
    generator = (
        torch.Generator(device=device).manual_seed(int(stream_seed))
        if stream_seed is not None
        else None
    )
    x0, y0, x1, y1 = rectangle
    mask = torch.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=torch.bool, device=device)
    mask[:, :, y0:y1, x0:x1] = True
    transitions: list[dict[str, Any]] = []
    states_before: list[np.ndarray] = []
    used_innovations: list[np.ndarray] = []
    fresh_innovations: list[np.ndarray] = []
    predicted_xstarts: list[np.ndarray] = []
    draws = 0

    for step_index, internal_t in enumerate(range(rollback_internal_timestep, -1, -1)):
        if role == "original_replay" and internal_t > 0:
            stored_x = torch.from_numpy(trace["x_t"][rows[internal_t]]).unsqueeze(0).to(
                device=device
            )
            if not torch.equal(x, stored_x):
                raise RuntimeError(
                    f"original replay state differs from trace before internal t={internal_t}"
                )
        t = torch.tensor([internal_t], dtype=torch.long, device=device)
        y = torch.tensor([pair[0]], dtype=torch.long, device=device)
        kwargs = {"y": y}
        with torch.no_grad():
            out = diffusion.p_mean_variance(
                model_fn,
                x,
                t,
                clip_denoised=True,
                model_kwargs=kwargs,
            )
            guided_mean = diffusion.condition_mean(
                cond_fn, out, x, t, model_kwargs=kwargs
            )
            p_sigma = torch.exp(0.5 * out["log_variance"])
            if role == "original_replay" and internal_t > 0:
                stored_pred = torch.from_numpy(
                    trace["pred_xstart"][rows[internal_t]]
                ).unsqueeze(0).to(device=device)
                stored_sigma = torch.from_numpy(
                    trace["p_standard_deviation"][rows[internal_t]]
                ).unsqueeze(0).to(device=device)
                if not torch.equal(out["pred_xstart"], stored_pred):
                    raise RuntimeError(
                        f"original replay pred_xstart differs from trace at t={internal_t}"
                    )
                if not torch.equal(p_sigma, stored_sigma):
                    raise RuntimeError(
                        f"original replay learned sigma differs from trace at t={internal_t}"
                    )

            original_noise: torch.Tensor | None = None
            fresh_noise: torch.Tensor | None = None
            used_noise: torch.Tensor | None = None
            if internal_t > 0:
                original_noise = torch.from_numpy(
                    trace["innovation"][rows[internal_t]]
                ).unsqueeze(0).to(device=device)
                if role == "original_replay":
                    used_noise = original_noise
                else:
                    if generator is None:
                        raise AssertionError("fresh suffix has no RNG")
                    fresh_noise = torch.randn(
                        tuple(x.shape),
                        generator=generator,
                        device=device,
                        dtype=x.dtype,
                    )
                    draws += 1
                    if role == "localized_attempt":
                        used_noise = torch.where(mask, fresh_noise, original_noise)
                        expanded_mask = mask.expand_as(used_noise)
                        if not torch.equal(
                            used_noise.masked_select(~expanded_mask),
                            original_noise.masked_select(~expanded_mask),
                        ) or not torch.equal(
                            used_noise.masked_select(expanded_mask),
                            fresh_noise.masked_select(expanded_mask),
                        ):
                            raise AssertionError("hard-mask hybrid innovation construction failed")
                    else:
                        used_noise = fresh_noise
                x_next = guided_mean + p_sigma * used_noise
            else:
                x_next = guided_mean

        state_before_record = _tensor_record(x)
        state_after_record = _tensor_record(x_next)
        transitions.append(
            {
                "step_index": step_index,
                "internal_timestep": internal_t,
                "original_timestep": int(timestep_map[internal_t]),
                "stochastic": internal_t > 0,
                "state_before": state_before_record,
                "predicted_xstart": _tensor_record(out["pred_xstart"]),
                "guided_p_mean": _tensor_record(guided_mean),
                "p_standard_deviation": _tensor_record(p_sigma),
                "original_observed_innovation": (
                    _tensor_record(original_noise) if original_noise is not None else None
                ),
                "fresh_innovation": (
                    _tensor_record(fresh_noise) if fresh_noise is not None else None
                ),
                "used_innovation": (
                    _tensor_record(used_noise) if used_noise is not None else None
                ),
                "fresh_draw_ordinal": draws if fresh_noise is not None else None,
                "state_after": state_after_record,
            }
        )
        states_before.append(_tensor_numpy(x[0]))
        predicted_xstarts.append(_tensor_numpy(out["pred_xstart"][0]))
        if used_noise is not None:
            used_innovations.append(_tensor_numpy(used_noise[0]))
        if fresh_noise is not None:
            fresh_innovations.append(_tensor_numpy(fresh_noise[0]))

        if role == "original_replay" and internal_t > 1:
            expected_next = torch.from_numpy(trace["x_t"][rows[internal_t - 1]]).unsqueeze(
                0
            ).to(device=device)
            if not torch.equal(x_next, expected_next):
                raise RuntimeError(
                    f"original innovation did not reproduce stored x_{internal_t - 1}"
                )
        x = x_next.detach()

    expected_draws = rollback_internal_timestep if role != "original_replay" else 0
    if draws != expected_draws:
        raise AssertionError(f"fresh suffix drew {draws}, expected {expected_draws}")
    trace_arrays = {
        "transition_internal_timestep": np.arange(
            rollback_internal_timestep, -1, -1, dtype=np.int16
        ),
        "stochastic_internal_timestep": np.arange(
            rollback_internal_timestep, 0, -1, dtype=np.int16
        ),
        "states_before": np.ascontiguousarray(np.stack(states_before), dtype=np.float32),
        "predicted_xstart": np.ascontiguousarray(
            np.stack(predicted_xstarts), dtype=np.float32
        ),
        "used_innovation": np.ascontiguousarray(
            np.stack(used_innovations), dtype=np.float32
        ),
        "fresh_innovation": (
            np.ascontiguousarray(np.stack(fresh_innovations), dtype=np.float32)
            if fresh_innovations
            else np.empty((0, *x.shape[1:]), dtype=np.float32)
        ),
        "final_state": np.ascontiguousarray(_tensor_numpy(x[0]), dtype=np.float32),
    }
    return BranchResult(
        role=role,
        stream_seed=stream_seed,
        final_state=x,
        transitions=transitions,
        trace_arrays=trace_arrays,
        gaussian_draws=draws,
    )


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trace_record(path: Path, arrays: dict[str, np.ndarray], root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "keys": sorted(arrays),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "raw_bytes_sha256": _array_sha(value),
            }
            for key, value in sorted(arrays.items())
        },
    }


def branch_ids(attempt_count: int) -> tuple[str, ...]:
    return (
        "original_replay",
        *(f"localized_attempt_{index:03d}" for index in range(attempt_count)),
        "full_fresh_control",
    )


def branch_image_path(root: Path, branch_id: str) -> Path:
    return root / "images" / f"{branch_id}.png"


def branch_trace_path(root: Path, branch_id: str) -> Path:
    return root / "traces" / f"{branch_id}.npz"


def _save_png(
    pixels: np.ndarray,
    path: Path,
    *,
    pair: Pair,
    branch_id: str,
    manifest_identity: str,
    runner_sha: str,
    baseline_pixel_sha: str,
) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite image: {path}")
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"invalid ADM64 output pixels: {pixels.shape}/{pixels.dtype}")
    metadata = PngInfo()
    fields = {
        "experiment": EXPERIMENT,
        "oracle_diagnostic": "true",
        "method_claim_eligible": "false",
        "class_id": str(pair[0]),
        "seed": str(pair[1]),
        "branch_id": branch_id,
        "pixel_sha256": pixel_sha256(pixels),
        "frozen_baseline_pixel_sha256": baseline_pixel_sha,
        "manifest_identity_sha256": manifest_identity,
        "runner_sha256": runner_sha,
    }
    for key, value in fields.items():
        metadata.add_text(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def build_manifest(
    args: argparse.Namespace,
    pair: Pair,
    rectangle: Rectangle,
    device: torch.device,
    diffusion_checkpoint: dict[str, Any],
    classifier_checkpoint: dict[str, Any],
    baseline: BaselineReference,
    input_trace: InputTrace,
    timestep_map: np.ndarray,
) -> dict[str, Any]:
    source_root = args.guided_diffusion_root.resolve()
    revision = git_revision(source_root)
    dirty = git_tracked_dirty(source_root)
    if revision != GUIDED_DIFFUSION_REVISION or dirty is not False:
        raise RuntimeError("guided-diffusion source is not the clean pinned revision")
    runner = Path(__file__).resolve()
    baseline_runner = runner.with_name("reproduce_adm64_guided.py")
    local_observer = runner.with_name("observe_adm64_local_path_evidence.py")
    attempt_seeds = [
        suffix_stream_seed(
            pair,
            "localized_attempt",
            args.rollback_internal_timestep,
            rectangle,
            index,
        )
        for index in range(args.attempt_count)
    ]
    control_seed = suffix_stream_seed(
        pair,
        "full_fresh_control",
        args.rollback_internal_timestep,
        rectangle,
        0,
    )
    if len(set([*attempt_seeds, control_seed])) != args.attempt_count + 1:
        raise AssertionError("domain-separated suffix RNG seeds collided")
    x0, y0, x1, y1 = rectangle
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "ORACLE_MECHANICS_ONLY_LOCALIZED_SUFFIX_DIAGNOSTIC",
        "method_claim_eligible": False,
        "paper_evidence_eligible": False,
        "pair": {"class_id": pair[0], "seed": pair[1]},
        "oracle_inputs": {
            "rollback_internal_timestep": args.rollback_internal_timestep,
            "rollback_source": (
                "explicit CLI choice; NOT selected by rho, evidence, an anytime-valid test, "
                "or an automated defect detector"
            ),
            "rectangle_xyxy_half_open": [x0, y0, x1, y1],
            "rectangle_array_indexing": "all channels, y0:y1, x0:x1 on the 64x64 ADM state",
            "rectangle_source": "explicit CLI oracle region",
            "attempt_count": args.attempt_count,
            "attempt_selection": "none; every attempt is retained and no best output is chosen",
        },
        "frozen_baseline": {
            "root": str(baseline.root),
            "manifest_identity_sha256": baseline.manifest_identity_sha256,
            "runner_sha256": baseline.runner_sha256,
            "pixel_path": str(baseline_pair_path(baseline.root, pair)),
            "pixel_sha256": pixel_sha256(
                decoded_pixels(baseline_pair_path(baseline.root, pair))
            ),
        },
        "input_all_step_local_trace": {
            "root": str(input_trace.root),
            "manifest_identity_sha256": input_trace.manifest_identity_sha256,
            "signal_payload_sha256": input_trace.signal_payload_sha256,
            "trace_sha256": input_trace.trace_sha256,
            "trace_record": input_trace.trace_record,
            "total_K_budget": input_trace.total_K_budget,
            "grid_size": input_trace.grid_size,
        },
        "checkpoints": {
            "diffusion": diffusion_checkpoint,
            "classifier": classifier_checkpoint,
        },
        "official_model_config": OFFICIAL_MODEL_CONFIG,
        "official_classifier_config": OFFICIAL_CLASSIFIER_CONFIG,
        "sampler": {
            "name": "OpenAI classifier-guided ancestral DDPM",
            "timestep_respacing": "250",
            "spaced_timestep_map": timestep_map.tolist(),
            "rollback_state": "exact trace x_t before the chosen transition",
            "original_replay": (
                "reuse every stored trace innovation from rollback t through t=1; "
                "execute deterministic t=0; require exact trace state chain and baseline pixels"
            ),
            "localized_attempt": (
                "at every t>0 use fresh standard Gaussian inside hard rectangle and the "
                "already observed baseline innovation outside"
            ),
            "full_fresh_control": (
                "independent full-image standard Gaussian at every t>0 from the same x_t"
            ),
            "maximum_attempts": MAX_ATTEMPTS,
            "no_retry_loop": True,
            "no_endpoint_scoring_ranking_or_selection": True,
        },
        "statistical_scope": {
            "paired_trace_conditioned_oracle_counterfactual": True,
            "online_sampling_method": False,
            "uses_saved_reference_future_innovations_outside_mask": True,
            "localized_hybrid_is_fresh_P_suffix": False,
            "reason": (
                "conditional on the observed trace, outside-mask innovations are fixed rather "
                "than freshly sampled from P"
            ),
            "conditional_Ville_retry_bound_applicable": False,
            "output_distribution_perturbation_bound_applicable": False,
            "full_fresh_control_is_P_suffix_conditional_on_supplied_x_t": True,
            "spatial_locality_guarantee": False,
            "spatial_warning": (
                "the hard mask localizes injected innovation only; convolution/attention and "
                "later denoising may propagate changes outside the rectangle"
            ),
        },
        "rng": {
            "namespace": RNG_NAMESPACE,
            "baseline_public_seed": pair[1],
            "baseline_sample_stream_seed": sample_stream_seed(pair[1]),
            "original_replay_fresh_rng": None,
            "localized_attempt_stream_seeds": attempt_seeds,
            "full_fresh_control_stream_seed": control_seed,
            "draws_per_fresh_branch": args.rollback_internal_timestep,
            "one full-shape float32 draw per stochastic suffix transition": True,
        },
        "outputs": {
            "branch_ids": list(branch_ids(args.attempt_count)),
            "images": "images/{branch_id}.png",
            "traces": "traces/{branch_id}.npz",
            "results": "results.json",
            "atomic_directory_install": True,
            "overwrite": False,
        },
        "sources": {
            "guided_diffusion_root": str(source_root),
            "guided_diffusion_revision": revision,
            "guided_diffusion_tracked_dirty": dirty,
            "guided_diffusion_python_tree_sha256": sha256_python_tree(
                source_root / "guided_diffusion"
            ),
            "baseline_runner": {
                "path": str(baseline_runner),
                "sha256": sha256_file(baseline_runner),
            },
            "local_trace_observer": {
                "path": str(local_observer),
                "sha256": sha256_file(local_observer),
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
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
    }
    manifest["identity_sha256"] = sha256_json(manifest)
    return manifest


def _require_tensor_record(record: Any, context: str) -> None:
    required = {
        "shape", "dtype", "raw_bytes_sha256", "minimum", "maximum", "mean",
        "root_mean_square",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError(f"invalid tensor record at {context}")
    if not isinstance(record["shape"], list) or not isinstance(record["dtype"], str):
        raise RuntimeError(f"invalid tensor identity at {context}")
    if not isinstance(record["raw_bytes_sha256"], str) or len(
        record["raw_bytes_sha256"]
    ) != 64:
        raise RuntimeError(f"invalid tensor SHA at {context}")
    for key in ("minimum", "maximum", "mean", "root_mean_square"):
        if not isinstance(record[key], (int, float)) or not math.isfinite(float(record[key])):
            raise RuntimeError(f"non-finite tensor statistic at {context}/{key}")


def save_branch(
    staging_root: Path,
    branch_id: str,
    result: BranchResult,
    pixels: np.ndarray,
    *,
    pair: Pair,
    manifest_identity: str,
    runner_sha: str,
    baseline_pixel_sha: str,
    attempt_index: int | None,
) -> dict[str, Any]:
    trace_file = branch_trace_path(staging_root, branch_id)
    image_file = branch_image_path(staging_root, branch_id)
    _atomic_npz_dump(result.trace_arrays, trace_file)
    trace_record = _trace_record(trace_file, result.trace_arrays, staging_root)
    _save_png(
        pixels,
        image_file,
        pair=pair,
        branch_id=branch_id,
        manifest_identity=manifest_identity,
        runner_sha=runner_sha,
        baseline_pixel_sha=baseline_pixel_sha,
    )
    return {
        "branch_id": branch_id,
        "role": result.role,
        "attempt_index": attempt_index,
        "stream_seed": result.stream_seed,
        "gaussian_draws": result.gaussian_draws,
        "transition_count": len(result.transitions),
        "transitions": result.transitions,
        "trace": trace_record,
        "image": {
            "relative_path": image_file.relative_to(staging_root).as_posix(),
            "pixel_sha256": pixel_sha256(pixels),
            "bytes": image_file.stat().st_size,
            "sha256": sha256_file(image_file),
        },
    }


def _load_npz_exact(path: Path, record: dict[str, Any], root: Path) -> dict[str, np.ndarray]:
    if record.get("relative_path") != path.relative_to(root).as_posix():
        raise RuntimeError("branch trace relative path is invalid")
    if not path.is_file() or path.stat().st_size != record.get("bytes"):
        raise RuntimeError("branch trace is missing or has wrong size")
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError("branch trace file SHA is invalid")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise RuntimeError(f"cannot load branch trace {path}") from exc
    if sorted(arrays) != record.get("keys"):
        raise RuntimeError("branch trace key set changed")
    if set(record.get("arrays", {})) != set(arrays):
        raise RuntimeError("branch trace array records are incomplete")
    for key, value in arrays.items():
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": _array_sha(value),
        }
        if record["arrays"].get(key) != expected:
            raise RuntimeError(f"branch trace array identity failed: {key}")
    return arrays


def validate_bundle(
    root: Path,
    manifest: dict[str, Any],
    input_trace: InputTrace,
    baseline: BaselineReference,
    pair: Pair,
    rectangle: Rectangle,
    *,
    attempt_count: int,
    rollback_internal_timestep: int,
    require_completion: bool,
) -> dict[str, Any]:
    stored_manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    if stored_manifest != manifest:
        raise RuntimeError("stored output manifest differs from in-memory manifest")
    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "class_id": pair[0],
        "seed": pair[1],
        "rollback_internal_timestep": rollback_internal_timestep,
        "rectangle_xyxy_half_open": list(rectangle),
        "attempt_count": attempt_count,
        "oracle_diagnostic": True,
        "method_claim_eligible": False,
        "conditional_Ville_retry_bound_applicable": False,
        "localized_hybrid_is_fresh_P_suffix": False,
        "online_sampling_method": False,
        "uses_saved_reference_future_innovations_outside_mask": True,
        "selection_performed": False,
        "selected_attempt": None,
    }
    mismatches = {
        key: (results.get(key), value)
        for key, value in fixed.items()
        if results.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"results identity/scope mismatch: {mismatches}")
    expected_ids = branch_ids(attempt_count)
    records = results.get("branches")
    if not isinstance(records, list) or [record.get("branch_id") for record in records] != list(
        expected_ids
    ):
        raise RuntimeError("results branch order/set changed")
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
    }
    if require_completion:
        expected_files.add((root / "completion.json").resolve())
    baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
    baseline_sha = pixel_sha256(baseline_pixels)
    rows = _trace_row_by_t(input_trace.arrays)
    x0, y0, x1, y1 = rectangle
    outside_mask = np.ones((3, IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    outside_mask[:, y0:y1, x0:x1] = False
    inside_mask = ~outside_mask

    for record in records:
        branch_id = record["branch_id"]
        role = record.get("role")
        if branch_id == "original_replay":
            expected_role, expected_seed, expected_index = "original_replay", None, None
        elif branch_id == "full_fresh_control":
            expected_role, expected_index = "full_fresh_control", None
            expected_seed = suffix_stream_seed(
                pair, expected_role, rollback_internal_timestep, rectangle, 0
            )
        else:
            expected_index = int(branch_id.rsplit("_", 1)[1])
            expected_role = "localized_attempt"
            expected_seed = suffix_stream_seed(
                pair, expected_role, rollback_internal_timestep, rectangle, expected_index
            )
        if role != expected_role or record.get("stream_seed") != expected_seed or record.get(
            "attempt_index"
        ) != expected_index:
            raise RuntimeError(f"branch identity/RNG mismatch: {branch_id}")
        expected_draws = 0 if role == "original_replay" else rollback_internal_timestep
        if record.get("gaussian_draws") != expected_draws or record.get(
            "transition_count"
        ) != rollback_internal_timestep + 1:
            raise RuntimeError(f"branch accounting mismatch: {branch_id}")

        trace_file = branch_trace_path(root, branch_id)
        image_file = branch_image_path(root, branch_id)
        expected_files.update({trace_file.resolve(), image_file.resolve()})
        arrays = _load_npz_exact(trace_file, record.get("trace", {}), root)
        expected_keys = {
            "transition_internal_timestep", "stochastic_internal_timestep", "states_before",
            "predicted_xstart", "used_innovation", "fresh_innovation", "final_state",
        }
        if set(arrays) != expected_keys:
            raise RuntimeError(f"branch trace schema changed: {branch_id}")
        n = rollback_internal_timestep
        tensor_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        expected_shapes = {
            "transition_internal_timestep": (n + 1,),
            "stochastic_internal_timestep": (n,),
            "states_before": (n + 1, *tensor_shape),
            "predicted_xstart": (n + 1, *tensor_shape),
            "used_innovation": (n, *tensor_shape),
            "fresh_innovation": ((0, *tensor_shape) if role == "original_replay" else (n, *tensor_shape)),
            "final_state": tensor_shape,
        }
        if any(arrays[key].shape != shape for key, shape in expected_shapes.items()):
            raise RuntimeError(f"branch trace shape mismatch: {branch_id}")
        if any(
            arrays[key].dtype != (np.dtype(np.int16) if "timestep" in key else np.dtype(np.float32))
            for key in arrays
        ):
            raise RuntimeError(f"branch trace dtype mismatch: {branch_id}")
        if not np.array_equal(
            arrays["transition_internal_timestep"],
            np.arange(rollback_internal_timestep, -1, -1, dtype=np.int16),
        ) or not np.array_equal(
            arrays["stochastic_internal_timestep"],
            np.arange(rollback_internal_timestep, 0, -1, dtype=np.int16),
        ):
            raise RuntimeError(f"branch timestep ordering changed: {branch_id}")

        transitions = record.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != n + 1:
            raise RuntimeError(f"branch transition list is incomplete: {branch_id}")
        previous_after: str | None = None
        for index, transition in enumerate(transitions):
            internal_t = rollback_internal_timestep - index
            if transition.get("step_index") != index or transition.get(
                "internal_timestep"
            ) != internal_t or transition.get("stochastic") != (internal_t > 0):
                raise RuntimeError(f"transition identity failed: {branch_id}/t={internal_t}")
            for key in (
                "state_before", "predicted_xstart", "guided_p_mean",
                "p_standard_deviation", "state_after",
            ):
                _require_tensor_record(transition.get(key), f"{branch_id}/t={internal_t}/{key}")
            if _array_sha(arrays["states_before"][index]) != transition["state_before"][
                "raw_bytes_sha256"
            ] or _array_sha(arrays["predicted_xstart"][index]) != transition[
                "predicted_xstart"
            ]["raw_bytes_sha256"]:
                raise RuntimeError(f"transition trace hashes failed: {branch_id}/t={internal_t}")
            if index == 0 and not np.array_equal(
                arrays["states_before"][index],
                input_trace.arrays["x_t"][rows[rollback_internal_timestep]],
            ):
                raise RuntimeError(f"branch did not start from supplied rollback x_t: {branch_id}")
            before_hash = transition["state_before"]["raw_bytes_sha256"]
            if previous_after is not None and before_hash != previous_after:
                raise RuntimeError(f"state chain broke: {branch_id}/t={internal_t}")
            previous_after = transition["state_after"]["raw_bytes_sha256"]
            if internal_t > 0:
                for key in ("original_observed_innovation", "used_innovation"):
                    _require_tensor_record(transition.get(key), f"{branch_id}/t={internal_t}/{key}")
                if _array_sha(arrays["used_innovation"][index]) != transition[
                    "used_innovation"
                ]["raw_bytes_sha256"]:
                    raise RuntimeError(f"used-noise trace hash failed: {branch_id}/t={internal_t}")
                original_noise = input_trace.arrays["innovation"][rows[internal_t]]
                if _array_sha(original_noise) != transition["original_observed_innovation"][
                    "raw_bytes_sha256"
                ]:
                    raise RuntimeError(f"original-noise provenance failed: {branch_id}/t={internal_t}")
                used = arrays["used_innovation"][index]
                if role == "original_replay":
                    if not np.array_equal(used, original_noise) or transition.get(
                        "fresh_innovation"
                    ) is not None:
                        raise RuntimeError("original replay did not use only stored innovations")
                    if not np.array_equal(
                        arrays["states_before"][index], input_trace.arrays["x_t"][rows[internal_t]]
                    ) or not np.array_equal(
                        arrays["predicted_xstart"][index],
                        input_trace.arrays["pred_xstart"][rows[internal_t]],
                    ):
                        raise RuntimeError("original replay state/prediction differs from input trace")
                    if transition.get("fresh_draw_ordinal") is not None:
                        raise RuntimeError("original replay consumed a fresh draw")
                else:
                    fresh = arrays["fresh_innovation"][index]
                    _require_tensor_record(
                        transition.get("fresh_innovation"), f"{branch_id}/t={internal_t}/fresh"
                    )
                    if _array_sha(fresh) != transition["fresh_innovation"][
                        "raw_bytes_sha256"
                    ]:
                        raise RuntimeError("fresh-noise trace hash failed")
                    if transition.get("fresh_draw_ordinal") != index + 1:
                        raise RuntimeError("fresh suffix draw ordinal changed")
                    if role == "localized_attempt":
                        if not np.array_equal(used[outside_mask], original_noise[outside_mask]):
                            raise RuntimeError("localized attempt changed innovation outside rectangle")
                        if not np.array_equal(used[inside_mask], fresh[inside_mask]):
                            raise RuntimeError("localized attempt failed to use fresh rectangle noise")
                    elif not np.array_equal(used, fresh):
                        raise RuntimeError("full-fresh control did not use its full fresh innovation")
            else:
                if any(
                    transition.get(key) is not None
                    for key in (
                        "original_observed_innovation", "fresh_innovation", "used_innovation",
                        "fresh_draw_ordinal",
                    )
                ):
                    raise RuntimeError("deterministic t=0 transition contains noise")
        if _array_sha(arrays["final_state"]) != transitions[-1]["state_after"][
            "raw_bytes_sha256"
        ]:
            raise RuntimeError(f"final-state trace hash failed: {branch_id}")

        if not image_file.is_file() or image_file.stat().st_size != record["image"].get(
            "bytes"
        ) or sha256_file(image_file) != record["image"].get("sha256"):
            raise RuntimeError(f"branch image file identity failed: {branch_id}")
        with Image.open(image_file) as image:
            metadata = dict(image.info)
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"wrong image mode/size: {branch_id}")
            image.verify()
        pixels = decoded_pixels(image_file)
        digest = pixel_sha256(pixels)
        if digest != record["image"].get("pixel_sha256") or record["image"].get(
            "relative_path"
        ) != image_file.relative_to(root).as_posix():
            raise RuntimeError(f"branch image record failed: {branch_id}")
        expected_metadata = {
            "experiment": EXPERIMENT,
            "oracle_diagnostic": "true",
            "method_claim_eligible": "false",
            "class_id": str(pair[0]),
            "seed": str(pair[1]),
            "branch_id": branch_id,
            "pixel_sha256": digest,
            "frozen_baseline_pixel_sha256": baseline_sha,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "runner_sha256": manifest["runner"]["sha256"],
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"branch PNG provenance failed: {branch_id}")
        if branch_id == "original_replay" and not np.array_equal(pixels, baseline_pixels):
            raise RuntimeError("original replay is not pixel-identical to frozen baseline")

    actual_files = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise RuntimeError(f"output file-set mismatch; extra={extra[:1]}, missing={missing[:1]}")
    if require_completion:
        completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
        expected_completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "results_payload_sha256": results["payload_sha256"],
            "branch_count": len(expected_ids),
            "attempt_count": attempt_count,
        }
        if any(completion.get(key) != value for key, value in expected_completion.items()):
            raise RuntimeError("completion marker is invalid")
    return results


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_args(args: argparse.Namespace) -> tuple[Pair, Rectangle]:
    if args.class_id is None or args.seed is None:
        raise ValueError("--class-id and --seed are required")
    if not 0 <= args.class_id < 1_000:
        raise ValueError("--class-id must lie in [0,999]")
    if not 0 <= args.seed <= (1 << 63) - 1:
        raise ValueError("--seed must lie in [0,2^63-1]")
    if args.rollback_internal_timestep is None or not (
        1 <= args.rollback_internal_timestep < NUM_SPACED_STEPS
    ):
        raise ValueError("--rollback-internal-timestep must lie in [1,249]")
    if args.rectangle_xyxy is None:
        raise ValueError("--rectangle-xyxy is required")
    if not 1 <= args.attempt_count <= MAX_ATTEMPTS:
        raise ValueError(f"--attempt-count must lie in [1,{MAX_ATTEMPTS}]")
    return (args.class_id, args.seed), args.rectangle_xyxy


def _assert_output_isolated(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output target: {args.output_dir}")
    protected = {
        "frozen baseline": args.baseline_dir,
        "input all-step trace": args.local_trace_dir,
        "guided-diffusion source": args.guided_diffusion_root,
        "diffusion checkpoint": args.model_path,
        "classifier checkpoint": args.classifier_path,
        "research source tree": Path(__file__).resolve().parent.parent,
    }
    overlap = [
        label for label, path in protected.items() if _paths_overlap(args.output_dir, path)
    ]
    if overlap:
        raise ValueError("output directory overlaps protected input/source: " + ", ".join(overlap))


def run_experiment(args: argparse.Namespace, pair: Pair, rectangle: Rectangle) -> None:
    _assert_output_isolated(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official FP16 ADM64 sampler requires CUDA")
    configure_determinism()
    torch.cuda.set_device(device)
    diffusion_checkpoint = validate_checkpoint(args.model_path, DIFFUSION_CHECKPOINT)
    classifier_checkpoint = validate_checkpoint(args.classifier_path, CLASSIFIER_CHECKPOINT)
    original_alpha_bar, timestep_map = original_schedule_and_timestep_map(
        args.guided_diffusion_root
    )
    baseline = load_baseline_pair(
        args.baseline_dir,
        pair,
        model_sha256=diffusion_checkpoint["sha256"],
        classifier_sha256=classifier_checkpoint["sha256"],
    )
    input_trace = load_input_trace(
        args.local_trace_dir,
        pair,
        baseline,
        model_sha256=diffusion_checkpoint["sha256"],
        classifier_sha256=classifier_checkpoint["sha256"],
        original_alpha_bar=original_alpha_bar,
        timestep_map=timestep_map,
    )
    if args.rollback_internal_timestep not in _trace_row_by_t(input_trace.arrays):
        raise RuntimeError("chosen rollback timestep is absent from validated input trace")
    manifest = build_manifest(
        args,
        pair,
        rectangle,
        device,
        diffusion_checkpoint,
        classifier_checkpoint,
        baseline,
        input_trace,
        timestep_map,
    )
    identity = manifest["identity_sha256"]
    runner_sha = manifest["runner"]["sha256"]
    baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
    baseline_pixel_sha = pixel_sha256(baseline_pixels)

    model, diffusion, classifier = load_official_models(
        args.guided_diffusion_root, args.model_path, args.classifier_path, device
    )
    if list(diffusion.timestep_map) != timestep_map.tolist():
        raise RuntimeError("loaded diffusion timestep map differs from validated trace schedule")
    model_fn, cond_fn = make_guided_functions(model, classifier)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.staging-", dir=args.output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        atomic_json_dump(manifest, staging / "manifest.json")
        records: list[dict[str, Any]] = []

        original = run_suffix_branch(
            diffusion,
            model_fn,
            cond_fn,
            pair,
            input_trace.arrays,
            rollback_internal_timestep=args.rollback_internal_timestep,
            rectangle=rectangle,
            role="original_replay",
            stream_seed=None,
            device=device,
            timestep_map=timestep_map,
        )
        original_pixels = pixels_from_sample(original.final_state[0])
        if not np.array_equal(original_pixels, baseline_pixels):
            raise RuntimeError("original suffix replay is not pixel-identical to frozen baseline")
        records.append(
            save_branch(
                staging,
                "original_replay",
                original,
                original_pixels,
                pair=pair,
                manifest_identity=identity,
                runner_sha=runner_sha,
                baseline_pixel_sha=baseline_pixel_sha,
                attempt_index=None,
            )
        )
        del original
        print("saved strict original replay", flush=True)

        for attempt_index in range(args.attempt_count):
            seed = suffix_stream_seed(
                pair,
                "localized_attempt",
                args.rollback_internal_timestep,
                rectangle,
                attempt_index,
            )
            branch = run_suffix_branch(
                diffusion,
                model_fn,
                cond_fn,
                pair,
                input_trace.arrays,
                rollback_internal_timestep=args.rollback_internal_timestep,
                rectangle=rectangle,
                role="localized_attempt",
                stream_seed=seed,
                device=device,
                timestep_map=timestep_map,
            )
            branch_id = f"localized_attempt_{attempt_index:03d}"
            records.append(
                save_branch(
                    staging,
                    branch_id,
                    branch,
                    pixels_from_sample(branch.final_state[0]),
                    pair=pair,
                    manifest_identity=identity,
                    runner_sha=runner_sha,
                    baseline_pixel_sha=baseline_pixel_sha,
                    attempt_index=attempt_index,
                )
            )
            del branch
            print(f"saved localized attempt {attempt_index + 1}/{args.attempt_count}", flush=True)

        control_seed = suffix_stream_seed(
            pair,
            "full_fresh_control",
            args.rollback_internal_timestep,
            rectangle,
            0,
        )
        control = run_suffix_branch(
            diffusion,
            model_fn,
            cond_fn,
            pair,
            input_trace.arrays,
            rollback_internal_timestep=args.rollback_internal_timestep,
            rectangle=rectangle,
            role="full_fresh_control",
            stream_seed=control_seed,
            device=device,
            timestep_map=timestep_map,
        )
        records.append(
            save_branch(
                staging,
                "full_fresh_control",
                control,
                pixels_from_sample(control.final_state[0]),
                pair=pair,
                manifest_identity=identity,
                runner_sha=runner_sha,
                baseline_pixel_sha=baseline_pixel_sha,
                attempt_index=None,
            )
        )
        del control
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": identity,
            "class_id": pair[0],
            "seed": pair[1],
            "rollback_internal_timestep": args.rollback_internal_timestep,
            "rectangle_xyxy_half_open": list(rectangle),
            "attempt_count": args.attempt_count,
            "oracle_diagnostic": True,
            "method_claim_eligible": False,
            "conditional_Ville_retry_bound_applicable": False,
            "localized_hybrid_is_fresh_P_suffix": False,
            "online_sampling_method": False,
            "uses_saved_reference_future_innovations_outside_mask": True,
            "selection_performed": False,
            "selected_attempt": None,
            "quality_judgment_performed_by_runner": False,
            "branches": records,
            "wall_seconds_before_validation": time.monotonic() - start,
        }
        results["payload_sha256"] = _canonical_payload_sha(results, "payload_sha256")
        atomic_json_dump(results, staging / "results.json")
        validate_bundle(
            staging,
            manifest,
            input_trace,
            baseline,
            pair,
            rectangle,
            attempt_count=args.attempt_count,
            rollback_internal_timestep=args.rollback_internal_timestep,
            require_completion=False,
        )
        completion = {
            "complete": True,
            "manifest_identity_sha256": identity,
            "results_payload_sha256": results["payload_sha256"],
            "branch_count": len(records),
            "attempt_count": args.attempt_count,
            "wall_seconds": time.monotonic() - start,
            "finished_at_unix": time.time(),
        }
        atomic_json_dump(completion, staging / "completion.json")
        validate_bundle(
            staging,
            manifest,
            input_trace,
            baseline,
            pair,
            rectangle,
            attempt_count=args.attempt_count,
            rollback_internal_timestep=args.rollback_internal_timestep,
            require_completion=True,
        )
        if args.output_dir.exists():
            raise RuntimeError("output target appeared during staging; refusing overwrite")
        os.replace(staging, args.output_dir)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


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
        if y is None:
            raise ValueError("toy model requires y")
        epsilon = 0.07 * x + t.view(-1, 1, 1, 1).to(x.dtype) * 0.002
        epsilon = epsilon + y.view(-1, 1, 1, 1).to(x.dtype) * 0.0001
        return torch.cat([epsilon, torch.tanh(0.03 * x)], dim=1)


def _toy_trace_and_endpoint(
    pair: Pair,
    diffusion: _ToyDiffusion,
    model: _ToyModel,
    cond_fn: Callable[..., torch.Tensor],
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(sample_stream_seed(pair[1]))
    x = torch.randn((1, 1, 4, 4), generator=generator, device=device)
    internal: list[int] = []
    states: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    for internal_t in range(diffusion.num_timesteps - 1, -1, -1):
        t = torch.tensor([internal_t], dtype=torch.long)
        y = torch.tensor([pair[0]], dtype=torch.long)
        kwargs = {"y": y}
        with torch.no_grad():
            out = diffusion.p_mean_variance(
                model, x, t, clip_denoised=True, model_kwargs=kwargs
            )
            mean = diffusion.condition_mean(cond_fn, out, x, t, model_kwargs=kwargs)
            sigma = torch.exp(0.5 * out["log_variance"])
            if internal_t > 0:
                noise = torch.randn(x.shape, generator=generator)
                internal.append(internal_t)
                states.append(_tensor_numpy(x[0]))
                innovations.append(_tensor_numpy(noise[0]))
                predictions.append(_tensor_numpy(out["pred_xstart"][0]))
                sigmas.append(_tensor_numpy(sigma[0]))
                x = mean + sigma * noise
            else:
                x = mean
    return {
        "internal_timestep": np.asarray(internal, dtype=np.int16),
        "x_t": np.ascontiguousarray(np.stack(states), dtype=np.float32),
        "innovation": np.ascontiguousarray(np.stack(innovations), dtype=np.float32),
        "pred_xstart": np.ascontiguousarray(np.stack(predictions), dtype=np.float32),
        "p_standard_deviation": np.ascontiguousarray(np.stack(sigmas), dtype=np.float32),
    }, x


def run_self_test() -> None:
    pair: Pair = (3, 7)
    rectangle = (0, 0, 2, 3)
    diffusion = _ToyDiffusion()
    model = _ToyModel()

    def cond_fn(
        x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("toy classifier requires y")
        return torch.full_like(x, 0.003) + y.view(-1, 1, 1, 1) * 0.00001

    trace, endpoint = _toy_trace_and_endpoint(pair, diffusion, model, cond_fn)
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    original = run_suffix_branch(
        diffusion,
        model,
        cond_fn,
        pair,
        trace,
        rollback_internal_timestep=3,
        rectangle=rectangle,
        role="original_replay",
        stream_seed=None,
        device=torch.device("cpu"),
        timestep_map=timestep_map,
    )
    if not torch.equal(original.final_state, endpoint) or original.gaussian_draws != 0:
        raise AssertionError("toy original-innovation replay is not exact")

    attempts: list[BranchResult] = []
    seeds: list[int] = []
    for index in range(2):
        seed = suffix_stream_seed(pair, "localized_attempt", 3, rectangle, index)
        seeds.append(seed)
        attempts.append(
            run_suffix_branch(
                diffusion,
                model,
                cond_fn,
                pair,
                trace,
                rollback_internal_timestep=3,
                rectangle=rectangle,
                role="localized_attempt",
                stream_seed=seed,
                device=torch.device("cpu"),
                timestep_map=timestep_map,
            )
        )
    control_seed = suffix_stream_seed(pair, "full_fresh_control", 3, rectangle, 0)
    control = run_suffix_branch(
        diffusion,
        model,
        cond_fn,
        pair,
        trace,
        rollback_internal_timestep=3,
        rectangle=rectangle,
        role="full_fresh_control",
        stream_seed=control_seed,
        device=torch.device("cpu"),
        timestep_map=timestep_map,
    )
    if len(set([*seeds, control_seed])) != 3:
        raise AssertionError("toy suffix RNG domains collided")
    if any(branch.gaussian_draws != 3 for branch in [*attempts, control]):
        raise AssertionError("toy fresh suffix draw count failed")
    rows = _trace_row_by_t(trace)
    outside = np.ones((1, 4, 4), dtype=bool)
    outside[:, rectangle[1] : rectangle[3], rectangle[0] : rectangle[2]] = False
    inside = ~outside
    for branch in attempts:
        for index, internal_t in enumerate((3, 2, 1)):
            used = branch.trace_arrays["used_innovation"][index]
            fresh = branch.trace_arrays["fresh_innovation"][index]
            observed = trace["innovation"][rows[internal_t]]
            if not np.array_equal(used[outside], observed[outside]):
                raise AssertionError("toy localized branch changed outside-mask innovation")
            if not np.array_equal(used[inside], fresh[inside]):
                raise AssertionError("toy localized branch did not use fresh inside-mask innovation")
    if not np.array_equal(
        control.trace_arrays["used_innovation"], control.trace_arrays["fresh_innovation"]
    ):
        raise AssertionError("toy full-fresh control is not fully fresh")
    if original.trace_arrays["fresh_innovation"].shape[0] != 0:
        raise AssertionError("toy original replay unexpectedly saved fresh noise")
    if parse_rectangle("0,0,32,45") != (0, 0, 32, 45):
        raise AssertionError("rectangle parser failed")
    with tempfile.TemporaryDirectory(prefix="adm64-localized-suffix-self-test-") as temporary:
        root = Path(temporary)
        path = root / "branch.npz"
        _atomic_npz_dump(attempts[0].trace_arrays, path)
        record = _trace_record(path, attempts[0].trace_arrays, root)
        loaded = _load_npz_exact(path, record, root)
        if any(
            not np.array_equal(loaded[key], attempts[0].trace_arrays[key]) for key in loaded
        ):
            raise AssertionError("toy branch NPZ round trip failed")
    print(
        "self-test passed: exact original replay, pre-transition rollback, hard-mask hybrid "
        "innovation, domain-separated attempts, full-fresh control, transition traces, and NPZ hashes"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    guided_root = data_root / "baselines" / "guided-diffusion"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rollback-internal-timestep", type=int, default=None)
    parser.add_argument(
        "--rectangle-xyxy",
        type=parse_rectangle,
        default=None,
        help="half-open x0,y0,x1,y1 in the 64x64 image/state grid",
    )
    parser.add_argument("--attempt-count", type=int, default=4)
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
    parser.add_argument("--local-trace-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    pair, rectangle = validate_args(args)
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    if args.baseline_dir is None:
        args.baseline_dir = (
            data_root / "cross_scale_evidence" / "adm64_guided" / "local_matched_seed115"
        )
    if args.local_trace_dir is None:
        args.local_trace_dir = (
            data_root
            / "cross_scale_evidence"
            / "adm64_local_path_evidence"
            / "local_matched_seed115_K0p5_grid4"
        )
    if args.output_dir is None:
        x0, y0, x1, y1 = rectangle
        args.output_dir = (
            data_root
            / "cross_scale_evidence"
            / "adm64_localized_suffix"
            / (
                f"class{pair[0]}_seed{pair[1]}_t{args.rollback_internal_timestep}_"
                f"rect{x0}_{y0}_{x1}_{y1}_n{args.attempt_count}"
            )
        )
    args.guided_diffusion_root = args.guided_diffusion_root.resolve()
    args.model_path = args.model_path.resolve()
    args.classifier_path = args.classifier_path.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.local_trace_dir = args.local_trace_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.dry_run:
        attempt_seeds = [
            suffix_stream_seed(
                pair,
                "localized_attempt",
                args.rollback_internal_timestep,
                rectangle,
                index,
            )
            for index in range(args.attempt_count)
        ]
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "role": "ORACLE_MECHANICS_ONLY_LOCALIZED_SUFFIX_DIAGNOSTIC",
                    "method_claim_eligible": False,
                    "conditional_Ville_retry_bound_applicable": False,
                    "localized_hybrid_is_fresh_P_suffix": False,
                    "online_sampling_method": False,
                    "uses_saved_reference_future_innovations_outside_mask": True,
                    "class_id": pair[0],
                    "seed": pair[1],
                    "rollback_internal_timestep": args.rollback_internal_timestep,
                    "rollback_source": "explicit CLI; not rho/evidence determined",
                    "rectangle_xyxy_half_open": list(rectangle),
                    "attempt_count": args.attempt_count,
                    "attempt_selection": "none",
                    "attempt_stream_seeds": attempt_seeds,
                    "full_fresh_control_stream_seed": suffix_stream_seed(
                        pair,
                        "full_fresh_control",
                        args.rollback_internal_timestep,
                        rectangle,
                        0,
                    ),
                    "baseline_dir": str(args.baseline_dir),
                    "local_trace_dir": str(args.local_trace_dir),
                    "output_dir": str(args.output_dir),
                    "gpu_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_experiment(args, pair, rectangle)


if __name__ == "__main__":
    main()

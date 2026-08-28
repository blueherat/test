#!/usr/bin/env python3
"""Oracle/mechanics-only DiT suffix repairability screen.

This diagnostic restores the *first-half* eight-image state ``x_t`` from a
strictly completed ``observe_dit_imagenet256_path_evidence.py`` bundle and
continues the frozen official DiT-XL/2 ancestral sampler.  It is deliberately
not an online detector or the proposed evidence-triggered method.

The screen is frozen in this source at internal timesteps 225, 180, 120, and 60.  A
single invocation executes one of those checkpoints and always retains five
branches in fixed order:

* attempt 0 is an exact replay control: all eight first-half innovations come
  from the observed baseline trace;
* attempts 1..4 use one independently seeded fresh innovation for the target
  batch item at every suffix transition, while the other seven first-half
  innovations remain the corresponding observed baseline innovations.

Every transition still evaluates the official B=8 -> 2B=16
``forward_with_cfg`` contract.  A complete fresh 2B Gaussian proposal is drawn
from the branch-local generator, including at t=0.  The assembled transition
uses its target first-half item only in attempts 1..4, reuses the saved trace
for all other first-half items, and uses its entire second half.  The t=0 draw
is consumed but multiplied by zero.  The pinned DiT ``forward_with_cfg``
reconstructs the network input from the first half before every call, so the
evolving second-half state cannot influence any later first-half prediction;
this source-level fact is validated as part of the frozen DiT repository
contract.

This is a trace-conditioned, target-only suffix counterfactual.  The rollback
time is supplied by an oracle protocol, not selected by evidence.  There is no
retry loop, image scoring, ranking, automatic checkpoint selection, or
best-of-N output.  It is ineligible for Ville, retry-cost, TV, or fresh-path
claims.  Its sole purpose is to ask whether the frozen model can repair the
obvious class-207/seed-2 fused-dog defect while preserving dog count, subject
identity, main pose, and composition.  Any such semantic/compositional change
is a failed repair, even if the resulting image looks attractive.

Each branch saves its target PNG, full eight-image grid, and a transition
trace.  Attempt 0 must be pixel-identical to the frozen baseline target and
grid.  Every non-target grid tile in every branch must remain pixel-identical
to the corresponding baseline tile.  Outputs are staged atomically, strictly
validated, self-hashed, and never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

# Match the frozen upstream sample.py and the strict observer.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:  # Package and direct CLI imports. ``BATCH_SIZE`` lives in the observer.
    from .observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        EXPERIMENT as OBSERVER_EXPERIMENT,
        FULL_BATCH_SIZE,
        TRACE_NAME as OBSERVER_TRACE_NAME,
        _load_trace_exact as load_observer_trace_exact,
        build_evidence_spec,
        load_schedule,
        validate_baseline_run,
        validate_output_bundle as validate_observer_bundle,
    )
    from .reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
        tensor_sha256,
        cuda_rng_state_sha256,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )
except ImportError:  # pragma: no cover - direct script execution.
    from observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        EXPERIMENT as OBSERVER_EXPERIMENT,
        FULL_BATCH_SIZE,
        TRACE_NAME as OBSERVER_TRACE_NAME,
        _load_trace_exact as load_observer_trace_exact,
        build_evidence_spec,
        load_schedule,
        validate_baseline_run,
        validate_output_bundle as validate_observer_bundle,
    )
    from reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
        tensor_sha256,
        cuda_rng_state_sha256,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )


EXPERIMENT = "dit_imagenet256_oracle_target_suffix_repairability"
SCHEMA_VERSION = 1
FROZEN_ROLLBACK_INTERNAL_TIMESTEPS = (225, 180, 120, 60)
FROZEN_FRESH_ATTEMPT_COUNT = 4
TOTAL_BRANCH_COUNT = 1 + FROZEN_FRESH_ATTEMPT_COUNT
DEFAULT_TARGET_BATCH_INDEX = 0
RNG_NAMESPACE = "eqvae-dit256-oracle-target-suffix-v1"

CHECKPOINT_RATIONALE = {
    225: "early-control: test semantic drift/object-count instability from an early rollback",
    180: "primary repairability window: early-to-middle structure formation",
    120: "primary repairability window: middle-to-late structure refinement",
    60: "late-control: test whether the defect is already too committed to repair",
}

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "final_first_half": np.dtype(np.float32),
    "fresh_full_proposal": np.dtype(np.float32),
    "target_p_mean": np.dtype(np.float32),
    "target_p_standard_deviation": np.dtype(np.float32),
    "target_pred_xstart": np.dtype(np.float32),
    "target_state_before": np.dtype(np.float32),
    "transition_internal_timestep": np.dtype(np.int16),
}


@dataclass(frozen=True)
class ObservedInput:
    root: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    baseline: Any
    arrays: dict[str, np.ndarray]
    timestep_map: np.ndarray

    @property
    def identity_sha256(self) -> str:
        return str(self.manifest["identity_sha256"])


@dataclass
class BranchResult:
    branch_id: str
    attempt_index: int
    role: str
    stream_seed: int
    final_first_half: torch.Tensor
    decoded: torch.Tensor
    transitions: list[dict[str, Any]]
    trace_arrays: dict[str, np.ndarray]
    full_2b_proposal_draws: int


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _array_raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    if not np.isfinite(array).all():
        raise RuntimeError("non-finite tensor encountered")
    return array


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    array = _tensor_numpy(tensor)
    values = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_bytes_sha256": _array_raw_sha256(array),
        "tensor_sha256": tensor_sha256(tensor),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "root_mean_square": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
    }


def _numpy_tensor_record(array: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(array)
    return _tensor_record(torch.from_numpy(canonical))


def _global_rng_state_sha256(device: torch.device) -> str:
    if device.type == "cuda":
        return cuda_rng_state_sha256()
    state = torch.get_rng_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _require_tensor_record(record: Any, shape: tuple[int, ...], context: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"missing tensor record: {context}")
    fixed = {"shape": list(shape), "dtype": "float32"}
    if any(record.get(key) != value for key, value in fixed.items()):
        raise RuntimeError(f"tensor shape/dtype changed: {context}")
    for key in ("raw_bytes_sha256", "tensor_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"invalid tensor digest: {context}/{key}")
    for key in ("minimum", "maximum", "mean", "root_mean_square"):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"invalid tensor statistic: {context}/{key}")


def _trace_rows(arrays: dict[str, np.ndarray]) -> dict[int, int]:
    timesteps = arrays.get("internal_timestep")
    if timesteps is None or timesteps.ndim != 1 or timesteps.size == 0:
        raise RuntimeError("observer trace lacks a one-dimensional timestep axis")
    count = int(timesteps.size)
    rows = {int(value): index for index, value in enumerate(timesteps.tolist())}
    if len(rows) != count or set(rows) != set(range(count)):
        raise RuntimeError("observer trace timestep axis is not a complete internal-time permutation")
    return rows


def branch_id(attempt_index: int) -> str:
    if not 0 <= attempt_index <= FROZEN_FRESH_ATTEMPT_COUNT:
        raise ValueError("attempt index lies outside the frozen branch set")
    return f"attempt_{attempt_index:03d}"


def branch_stream_seed(
    observer_identity_sha256: str,
    *,
    public_seed: int,
    rollback_internal_timestep: int,
    target_batch_index: int,
    attempt_index: int,
) -> int:
    if len(observer_identity_sha256) != 64:
        raise ValueError("observer identity must be SHA-256")
    payload = (
        f"{RNG_NAMESPACE}\0{observer_identity_sha256}\0{public_seed}\0"
        f"{rollback_internal_timestep}\0{target_batch_index}\0{attempt_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _with_upstream_imports(root: Path, callback: Any) -> Any:
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    preexisting = {
        name
        for name in sys.modules
        if name in {"models", "download", "diffusion"} or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError("ambiguous pre-imported upstream modules: " + repr(sorted(preexisting)))
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        return callback()
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for name in list(sys.modules):
            if name in {"models", "download", "diffusion"} or name.startswith("diffusion."):
                if name not in preexisting:
                    sys.modules.pop(name, None)


def load_observed_input(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    vae: dict[str, Any],
) -> ObservedInput:
    if any(path.is_symlink() for path in args.observe_dir.rglob("*")):
        raise RuntimeError("completed observer bundle contains a symlink")
    baseline = validate_baseline_run(
        args.baseline_dir,
        seed=args.seed,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    alpha, timestep_map = load_schedule(args.dit_root)
    spec = build_evidence_spec(alpha, timestep_map)
    results = validate_observer_bundle(args.observe_dir, baseline=baseline, spec=spec)
    manifest = _read_self_hashed_json(args.observe_dir / "manifest.json", "identity_sha256")
    completion = _read_self_hashed_json(args.observe_dir / "completion.json", "payload_sha256")
    fixed_manifest = {
        "experiment": OBSERVER_EXPERIMENT,
        "observe_only": True,
        "sampling_distribution_P_changed": False,
        "seed": args.seed,
        "class_ids_in_official_batch_order": list(CLASS_IDS),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in fixed_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"observer manifest identity/scope mismatch: {mismatches}")
    if completion.get("manifest_identity_sha256") != manifest["identity_sha256"]:
        raise RuntimeError("observer completion is not bound to its manifest")
    baseline_identity = baseline.manifest.get("identity", {})
    expected_observer_sources = {
        "dit": source,
        "checkpoint": baseline_identity.get("checkpoint"),
        "vae": vae,
    }
    source_mismatches = {
        key: (manifest.get("sources", {}).get(key), value)
        for key, value in expected_observer_sources.items()
        if manifest.get("sources", {}).get(key) != value
    }
    if source_mismatches:
        raise RuntimeError(f"observer source/checkpoint/VAE identity mismatch: {source_mismatches}")
    if checkpoint is not None and baseline_identity.get("checkpoint") != checkpoint:
        raise RuntimeError("observer/baseline checkpoint differs from the fully hashed requested checkpoint")
    trace_path = args.observe_dir / OBSERVER_TRACE_NAME
    arrays = load_observer_trace_exact(trace_path, results.get("trace", {}), args.observe_dir)
    rows = _trace_rows(arrays)
    if len(rows) != NUM_SAMPLING_STEPS:
        raise RuntimeError("completed DiT observer trace does not contain exactly 250 steps")
    return ObservedInput(
        root=args.observe_dir.resolve(),
        manifest=manifest,
        results=results,
        baseline=baseline,
        arrays=arrays,
        timestep_map=np.ascontiguousarray(timestep_map, dtype=np.int64),
    )


def canonical_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seed",
        str(args.seed),
        "--rollback-internal-timestep",
        str(args.rollback_internal_timestep),
        "--target-batch-index",
        str(args.target_batch_index),
        "--observe-dir",
        str(args.observe_dir),
        "--baseline-dir",
        str(args.baseline_dir),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]


def build_manifest(
    args: argparse.Namespace,
    *,
    observed: ObservedInput,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    observer_runner = runner.with_name("observe_dit_imagenet256_path_evidence.py")
    baseline_runner = runner.with_name("reproduce_dit_imagenet256.py")
    seeds = [
        branch_stream_seed(
            observed.identity_sha256,
            public_seed=args.seed,
            rollback_internal_timestep=args.rollback_internal_timestep,
            target_batch_index=args.target_batch_index,
            attempt_index=index,
        )
        for index in range(TOTAL_BRANCH_COUNT)
    ]
    if len(set(seeds)) != TOTAL_BRANCH_COUNT:
        raise AssertionError("domain-separated branch seeds collided")
    target_class = int(CLASS_IDS[args.target_batch_index])
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "POSTHOC_ORACLE_MECHANICS_ONLY_REPAIRABILITY_SCREEN",
        "online_sampling_method": False,
        "method_claim_eligible": False,
        "paper_success_claim_eligible_without_blind_review": False,
        "automatic_quality_scoring": False,
        "automatic_checkpoint_or_attempt_selection": False,
        "best_of_n_selection": False,
        "seed": args.seed,
        "official_class_ids_in_batch_order": list(CLASS_IDS),
        "target": {
            "batch_index": args.target_batch_index,
            "class_id": target_class,
            "frozen_primary_case": args.seed == 2 and args.target_batch_index == 0,
            "frozen_primary_case_description": (
                "class 207 golden retriever / global seed 2: conspicuous fused or duplicated "
                "torso-tail structure and incoherent hindquarters/limbs"
            ),
        },
        "frozen_screen_protocol": {
            "rollback_internal_timesteps": list(FROZEN_ROLLBACK_INTERNAL_TIMESTEPS),
            "checkpoint_rationale": {str(key): value for key, value in CHECKPOINT_RATIONALE.items()},
            "all_checkpoints_required_for_the_full_screen": True,
            "this_invocation_rollback_internal_timestep": args.rollback_internal_timestep,
            "fresh_attempt_count_per_checkpoint": FROZEN_FRESH_ATTEMPT_COUNT,
            "replay_control_count_per_checkpoint": 1,
            "branch_order": [branch_id(index) for index in range(TOTAL_BRANCH_COUNT)],
            "attempt_selection": "none; retain and report every frozen attempt",
            "checkpoint_selection": "none; the full screen runs all four frozen checkpoints",
            "success_requires": (
                "repair the conspicuous blur/fusion/limb-or-object misalignment while preserving "
                "dog count, subject identity, main pose, and composition"
            ),
            "automatic_failure_if_any_change": [
                "dog/object count",
                "subject identity",
                "main pose",
                "main composition",
            ],
            "attractive_but_semantically_changed_output_is_failure": True,
        },
        "frozen_observe_bundle": {
            "root": str(observed.root),
            "manifest_identity_sha256": observed.identity_sha256,
            "manifest_file_sha256": sha256_file(observed.root / "manifest.json"),
            "results_payload_sha256": observed.results["payload_sha256"],
            "results_file_sha256": sha256_file(observed.root / "results.json"),
            "completion_file_sha256": sha256_file(observed.root / "completion.json"),
            "trace_relative_path": OBSERVER_TRACE_NAME,
            "trace_sha256": observed.results["trace"]["sha256"],
            "strict_completion_and_trace_math_validated": True,
        },
        "frozen_baseline": {
            "root": str(observed.baseline.root),
            "manifest_identity_sha256": observed.baseline.identity_sha256,
            "manifest_file_sha256": sha256_file(observed.baseline.root / "manifest.json"),
            "completion_file_sha256": sha256_file(observed.baseline.root / "completion.json"),
            "outputs_sha256": observed.baseline.manifest["outputs_sha256"],
        },
        "sampler_contract": {
            "model": MODEL_NAME,
            "ancestral_ddpm_internal_steps": NUM_SAMPLING_STEPS,
            "cfg_scale": CFG_SCALE,
            "first_half_batch_size": BATCH_SIZE,
            "model_batch_size": FULL_BATCH_SIZE,
            "cfg_input_contract": "B=8 first half duplicated to conditional/null 2B=16 on every forward_with_cfg call",
            "cfg_guided_epsilon_channels": [0, 1, 2],
            "rollback_state": "exact observed first-half x_t before the chosen transition",
            "initial_second_half_state": "copy of restored first half; operationally irrelevant to later first-half predictions",
            "attempt_0": (
                "reuse all eight observed first-half innovations at every suffix step, "
                "including the zero-multiplied t=0 innovation"
            ),
            "attempts_1_to_4": (
                "replace only the target first-half innovation by the corresponding item "
                "of an independent branch-local fresh full-2B proposal; reuse observed "
                "innovations for the other seven first-half items"
            ),
            "fresh_full_2b_proposal_each_suffix_step": True,
            "fresh_full_2b_proposal_shape": [FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "second_half_of_each_proposal_used_in_transition": True,
            "unused_fresh_non_target_first_half_items_retained_in_trace": True,
            "terminal_t0_full_2b_proposal_consumed_then_zero_multiplied": True,
            "second_half_noninterference_reason": (
                "pinned forward_with_cfg discards the incoming second half for the next network "
                "evaluation by rebuilding its combined input solely from x[:B]"
            ),
            "non_target_first_half_state_equal_observer_trace_required_before_every_transition": True,
            "attempt_0_all_first_half_state_equal_observer_trace_required": True,
            "clip_denoised": False,
        },
        "statistical_scope": {
            "posthoc_oracle": True,
            "rollback_selected_by_anytime_evidence": False,
            "uses_saved_future_innovations_for_seven_non_target_items": True,
            "whole_batch_suffix_is_fresh_P": False,
            "fresh_full_path_from_initial_noise": False,
            "conditional_Ville_bound_applicable": False,
            "suffix_retry_cost_bound_applicable": False,
            "TV_distribution_perturbation_bound_applicable": False,
            "no_retry_loop": True,
            "purpose": "mechanical model repairability screen only",
        },
        "rng": {
            "namespace": RNG_NAMESPACE,
            "one_explicit_independent_branch_seed_per_attempt_including_replay_auxiliary_draws": True,
            "branch_stream_seeds": [
                {"branch_id": branch_id(index), "attempt_index": index, "seed": seeds[index]}
                for index in range(TOTAL_BRANCH_COUNT)
            ],
            "proposal_draws_per_branch_including_t0": args.rollback_internal_timestep + 1,
            "draw_shape": [FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "global_cuda_rng_used_for_suffix_proposals": False,
        },
        "sources": {
            "dit": source,
            "checkpoint": checkpoint,
            "vae": vae,
            "observer_runner": {"path": str(observer_runner), "sha256": sha256_file(observer_runner)},
            "baseline_runner": {"path": str(baseline_runner), "sha256": sha256_file(baseline_runner)},
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "dependencies": dependency_identity(),
        "canonical_command": canonical_command(args),
        "outputs": {
            "branches": [branch_id(index) for index in range(TOTAL_BRANCH_COUNT)],
            "target_png": "branches/{branch_id}/target.png",
            "full_eight_grid_png": "branches/{branch_id}/grid.png",
            "transition_trace": "branches/{branch_id}/trace.npz",
            "all_attempts_retained": True,
            "selected_attempt": None,
            "atomic_directory_install": True,
            "no_overwrite": True,
        },
    }
    manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
    return manifest


def run_suffix_branch(
    diffusion: Any,
    model: Any,
    vae: Any,
    observed: ObservedInput,
    *,
    rollback_internal_timestep: int,
    target_batch_index: int,
    attempt_index: int,
    stream_seed: int,
    device: torch.device,
) -> BranchResult:
    """Run one fixed branch from an already validated first-half trace state."""

    if rollback_internal_timestep not in FROZEN_ROLLBACK_INTERNAL_TIMESTEPS:
        raise ValueError("rollback timestep is outside the frozen screen")
    if not 0 <= target_batch_index < BATCH_SIZE:
        raise ValueError("target batch index is invalid")
    if not 0 <= attempt_index <= FROZEN_FRESH_ATTEMPT_COUNT:
        raise ValueError("attempt index is outside the frozen branch set")
    rows = _trace_rows(observed.arrays)
    start_row = rows[rollback_internal_timestep]
    first = torch.from_numpy(observed.arrays["x_t"][start_row]).to(device=device)
    if first.dtype != torch.float32 or tuple(first.shape) != (
        BATCH_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):
        raise RuntimeError("validated rollback first-half state has the wrong contract")
    x = torch.cat([first, first], dim=0)
    y_conditional = torch.tensor(CLASS_IDS, dtype=torch.long, device=device)
    y_null = torch.full((BATCH_SIZE,), NULL_CLASS_ID, dtype=torch.long, device=device)
    y = torch.cat([y_conditional, y_null], dim=0)
    model_kwargs = {"y": y, "cfg_scale": CFG_SCALE}
    generator = torch.Generator(device=device).manual_seed(int(stream_seed))
    role = "exact_replay_control" if attempt_index == 0 else "fresh_target_suffix_attempt"
    non_target = np.asarray(
        [index for index in range(BATCH_SIZE) if index != target_batch_index], dtype=np.int64
    )
    transitions: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    means: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    fresh_proposals: list[np.ndarray] = []
    draw_count = 0
    global_rng_before = _global_rng_state_sha256(device)

    for step_index, internal_t in enumerate(range(rollback_internal_timestep, -1, -1)):
        row = rows[internal_t]
        baseline_state = torch.from_numpy(observed.arrays["x_t"][row]).to(device=device)
        if attempt_index == 0:
            if not torch.equal(x[:BATCH_SIZE], baseline_state):
                raise RuntimeError(f"attempt 0 left the observed state chain before t={internal_t}")
        elif not torch.equal(x[:BATCH_SIZE][non_target.tolist()], baseline_state[non_target.tolist()]):
            raise RuntimeError(f"a non-target first-half state changed before t={internal_t}")

        t = torch.full((FULL_BATCH_SIZE,), internal_t, dtype=torch.long, device=device)
        rng_before_model = _global_rng_state_sha256(device)
        with torch.no_grad():
            out = diffusion.p_mean_variance(
                model.forward_with_cfg,
                x,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
        rng_after_model = _global_rng_state_sha256(device)
        if rng_after_model != rng_before_model:
            raise RuntimeError("DiT forward/p_mean_variance unexpectedly consumed global CUDA RNG")
        p_mean = out["mean"]
        pred_xstart = out["pred_xstart"]
        p_sigma = torch.exp(0.5 * out["log_variance"])
        expected_full_shape = (FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
        if tuple(p_mean.shape) != expected_full_shape or tuple(pred_xstart.shape) != expected_full_shape:
            raise RuntimeError("DiT transition output shape changed")
        if tuple(p_sigma.shape) != expected_full_shape:
            raise RuntimeError("DiT transition sigma is not expanded to the full state shape")

        baseline_pred = torch.from_numpy(observed.arrays["pred_xstart"][row]).to(device=device)
        baseline_mean = torch.from_numpy(observed.arrays["p_mean_first_half"][row]).to(device=device)
        baseline_sigma = torch.from_numpy(observed.arrays["p_standard_deviation"][row]).to(device=device)
        checked_indices = list(range(BATCH_SIZE)) if attempt_index == 0 else non_target.tolist()
        for name, actual, expected in (
            ("pred_xstart", pred_xstart[:BATCH_SIZE], baseline_pred),
            ("p_mean", p_mean[:BATCH_SIZE], baseline_mean),
            ("p_sigma", p_sigma[:BATCH_SIZE], baseline_sigma),
        ):
            if not torch.equal(actual[checked_indices], expected[checked_indices]):
                raise RuntimeError(f"{role} violated the frozen {name} control at t={internal_t}")

        # One explicit full-2B proposal per transition, including t=0.  For
        # target attempts only its target first-half item and entire second
        # half are used; non-target first-half items stay on the trace.
        fresh_full = torch.randn(
            expected_full_shape,
            dtype=x.dtype,
            device=device,
            generator=generator,
        )
        draw_count += 1
        baseline_first_noise = torch.from_numpy(
            observed.arrays["innovation_first_half"][row]
        ).to(device=device)
        used_first_noise = baseline_first_noise.clone()
        if attempt_index > 0:
            used_first_noise[target_batch_index] = fresh_full[target_batch_index]
        used_full_noise = torch.cat([used_first_noise, fresh_full[BATCH_SIZE:]], dim=0)
        nonzero = float(internal_t != 0)
        x_next = p_mean + nonzero * p_sigma * used_full_noise

        expected_first_next = p_mean[:BATCH_SIZE] + nonzero * p_sigma[:BATCH_SIZE] * used_first_noise
        if not torch.equal(x_next[:BATCH_SIZE], expected_first_next):
            raise AssertionError("assembled full transition changed first-half arithmetic")
        if internal_t > 0:
            expected_next_baseline = torch.from_numpy(
                observed.arrays["x_t"][rows[internal_t - 1]]
            ).to(device=device)
        else:
            expected_next_baseline = torch.from_numpy(
                observed.arrays["final_latents_first_half"]
            ).to(device=device)
        equality_indices = list(range(BATCH_SIZE)) if attempt_index == 0 else non_target.tolist()
        if not torch.equal(x_next[:BATCH_SIZE][equality_indices], expected_next_baseline[equality_indices]):
            raise RuntimeError(f"baseline-controlled first-half state failed after t={internal_t}")

        target_state = x[target_batch_index]
        target_pred = pred_xstart[target_batch_index]
        target_mean = p_mean[target_batch_index]
        target_sigma = p_sigma[target_batch_index]
        fresh_record = _tensor_record(fresh_full)
        transitions.append(
            {
                "step_index": step_index,
                "internal_timestep": internal_t,
                "original_timestep": int(observed.timestep_map[internal_t]),
                "stochastic_effect": internal_t > 0,
                "full_2b_proposal_draw_ordinal": draw_count,
                "full_2b_proposal": fresh_record,
                "t0_proposal_consumed_then_zero_multiplied": internal_t == 0,
                "target_uses_fresh_proposal": attempt_index > 0,
                "target_state_before": _tensor_record(target_state),
                "target_pred_xstart": _tensor_record(target_pred),
                "target_p_mean": _tensor_record(target_mean),
                "target_p_standard_deviation": _tensor_record(target_sigma),
                "target_baseline_innovation_raw_sha256": _array_raw_sha256(
                    observed.arrays["innovation_first_half"][row, target_batch_index]
                ),
                "target_used_innovation": _tensor_record(used_first_noise[target_batch_index]),
                "non_target_state_before_raw_sha256": _array_raw_sha256(
                    _tensor_numpy(x[:BATCH_SIZE][non_target.tolist()])
                ),
                "non_target_pred_xstart_raw_sha256": _array_raw_sha256(
                    _tensor_numpy(pred_xstart[:BATCH_SIZE][non_target.tolist()])
                ),
                "non_target_p_mean_raw_sha256": _array_raw_sha256(
                    _tensor_numpy(p_mean[:BATCH_SIZE][non_target.tolist()])
                ),
                "non_target_p_sigma_raw_sha256": _array_raw_sha256(
                    _tensor_numpy(p_sigma[:BATCH_SIZE][non_target.tolist()])
                ),
                "non_target_state_after_raw_sha256": _array_raw_sha256(
                    _tensor_numpy(x_next[:BATCH_SIZE][non_target.tolist()])
                ),
                "full_first_half_state_before": _tensor_record(x[:BATCH_SIZE]),
                "full_first_half_state_after": _tensor_record(x_next[:BATCH_SIZE]),
                "global_cuda_rng_before_model": rng_before_model,
                "global_cuda_rng_after_model": rng_after_model,
            }
        )
        states.append(_tensor_numpy(target_state))
        predicted.append(_tensor_numpy(target_pred))
        means.append(_tensor_numpy(target_mean))
        sigmas.append(_tensor_numpy(target_sigma))
        fresh_proposals.append(_tensor_numpy(fresh_full))
        x = x_next.detach()

    global_rng_after = _global_rng_state_sha256(device)
    if global_rng_after != global_rng_before:
        raise RuntimeError("branch-local generators leaked into the global CUDA RNG")
    expected_draws = rollback_internal_timestep + 1
    if draw_count != expected_draws:
        raise AssertionError(f"full-2B draw count changed: {draw_count} != {expected_draws}")
    final_first = x[:BATCH_SIZE].contiguous()
    if attempt_index == 0 and not torch.equal(
        final_first,
        torch.from_numpy(observed.arrays["final_latents_first_half"]).to(device=device),
    ):
        raise RuntimeError("attempt 0 final first-half latents differ from the observed baseline")
    with torch.no_grad():
        decoded = vae.decode(final_first / VAE_SCALING_FACTOR).sample
    trace_arrays = {
        "transition_internal_timestep": np.arange(
            rollback_internal_timestep, -1, -1, dtype=np.int16
        ),
        "target_state_before": np.ascontiguousarray(np.stack(states), dtype=np.float32),
        "target_pred_xstart": np.ascontiguousarray(np.stack(predicted), dtype=np.float32),
        "target_p_mean": np.ascontiguousarray(np.stack(means), dtype=np.float32),
        "target_p_standard_deviation": np.ascontiguousarray(np.stack(sigmas), dtype=np.float32),
        "fresh_full_proposal": np.ascontiguousarray(np.stack(fresh_proposals), dtype=np.float32),
        "final_first_half": np.ascontiguousarray(_tensor_numpy(final_first), dtype=np.float32),
    }
    if set(trace_arrays) != set(TRACE_DTYPES):
        raise AssertionError("branch trace key set changed")
    for key, dtype in TRACE_DTYPES.items():
        if trace_arrays[key].dtype != dtype:
            raise AssertionError(f"branch trace dtype changed: {key}")
    return BranchResult(
        branch_id=branch_id(attempt_index),
        attempt_index=attempt_index,
        role=role,
        stream_seed=stream_seed,
        final_first_half=final_first,
        decoded=decoded,
        transitions=transitions,
        trace_arrays=trace_arrays,
        full_2b_proposal_draws=draw_count,
    )


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise RuntimeError(f"trace parent is not a directory: {path.parent}")
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
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
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


def _load_npz_exact(path: Path, record: dict[str, Any], root: Path) -> dict[str, np.ndarray]:
    if record.get("relative_path") != path.relative_to(root).as_posix():
        raise RuntimeError("branch trace relative path changed")
    if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"branch trace file identity failed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if set(arrays) != set(TRACE_DTYPES) or sorted(arrays) != record.get("keys"):
        raise RuntimeError("branch trace key set changed")
    for key, array in arrays.items():
        expected = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "raw_bytes_sha256": _array_raw_sha256(array),
        }
        if array.dtype != TRACE_DTYPES[key] or record.get("arrays", {}).get(key) != expected:
            raise RuntimeError(f"branch trace array identity failed: {key}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"branch trace contains non-finite values: {key}")
    return arrays


def _branch_dir(root: Path, branch: str) -> Path:
    return root / "branches" / branch


def _grid_tile_pixels(grid_path: Path, batch_index: int) -> np.ndarray:
    if not 0 <= batch_index < BATCH_SIZE:
        raise ValueError("grid tile index is invalid")
    row, column = divmod(batch_index, 4)
    left = 2 + column * (IMAGE_SIZE + 2)
    top = 2 + row * (IMAGE_SIZE + 2)
    with Image.open(grid_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1_034, 518):
            raise RuntimeError(f"unexpected official grid geometry: {grid_path}")
        tile = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        return np.ascontiguousarray(np.asarray(tile, dtype=np.uint8))


def _pixel_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _png_record(path: Path, root: Path, *, grid: bool) -> dict[str, Any]:
    mode, size = ("RGB", (1_034, 518)) if grid else ("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    record = {"relative_path": path.relative_to(root).as_posix()}
    record.update(inspect_png(path, mode, size))
    return record


def save_branch_outputs(
    result: BranchResult,
    staging: Path,
    *,
    target_batch_index: int,
    save_image: Any,
) -> dict[str, Any]:
    directory = _branch_dir(staging, result.branch_id)
    directory.mkdir(parents=True, exist_ok=False)
    target_path = directory / "target.png"
    grid_path = directory / "grid.png"
    save_image(
        result.decoded[target_batch_index],
        target_path,
        nrow=1,
        padding=0,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(
        result.decoded,
        grid_path,
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    trace_path = directory / "trace.npz"
    _atomic_npz_dump(result.trace_arrays, trace_path)
    target_record = _png_record(target_path, staging, grid=False)
    grid_record = _png_record(grid_path, staging, grid=True)
    target_tile_sha = _pixel_sha(_grid_tile_pixels(grid_path, target_batch_index))
    if target_tile_sha != target_record["pixel_sha256"]:
        raise RuntimeError("target PNG differs from its corresponding full-grid tile")
    return {
        "branch_id": result.branch_id,
        "attempt_index": result.attempt_index,
        "role": result.role,
        "stream_seed": result.stream_seed,
        "target_uses_fresh_suffix": result.attempt_index > 0,
        "full_2b_proposal_draws": result.full_2b_proposal_draws,
        "transition_count": len(result.transitions),
        "transitions": result.transitions,
        "trace": _trace_record(trace_path, result.trace_arrays, staging),
        "target_image": target_record,
        "full_grid": grid_record,
        "target_grid_tile_pixel_sha256": target_tile_sha,
        "final_first_half": _tensor_record(result.final_first_half),
    }


def _expected_trace_shapes(rollback_internal_timestep: int) -> dict[str, tuple[int, ...]]:
    steps = rollback_internal_timestep + 1
    target_state = (steps, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return {
        "transition_internal_timestep": (steps,),
        "target_state_before": target_state,
        "target_pred_xstart": target_state,
        "target_p_mean": target_state,
        "target_p_standard_deviation": target_state,
        "fresh_full_proposal": (
            steps,
            FULL_BATCH_SIZE,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ),
        "final_first_half": (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
    }


def validate_bundle(
    root: Path,
    *,
    manifest: dict[str, Any],
    observed: ObservedInput,
    rollback_internal_timestep: int,
    target_batch_index: int,
    require_completion: bool,
) -> dict[str, Any]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("suffix bundle must not contain symlinks")
    stored_manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    if stored_manifest != manifest:
        raise RuntimeError("stored manifest differs from the frozen in-memory manifest")
    runner = Path(__file__).resolve()
    if manifest.get("runner", {}).get("sha256") != sha256_file(runner):
        raise RuntimeError("bundle was produced by a different suffix-runner source")
    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "oracle_mechanics_only": True,
        "online_sampling_method": False,
        "method_claim_eligible": False,
        "conditional_Ville_bound_applicable": False,
        "TV_bound_applicable": False,
        "rollback_internal_timestep": rollback_internal_timestep,
        "target_batch_index": target_batch_index,
        "target_class_id": int(CLASS_IDS[target_batch_index]),
        "fresh_attempt_count": FROZEN_FRESH_ATTEMPT_COUNT,
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
    expected_result_keys = {
        "schema_version",
        "experiment",
        "manifest_identity_sha256",
        "oracle_mechanics_only",
        "online_sampling_method",
        "method_claim_eligible",
        "conditional_Ville_bound_applicable",
        "TV_bound_applicable",
        "rollback_internal_timestep",
        "target_batch_index",
        "target_class_id",
        "fresh_attempt_count",
        "selection_performed",
        "selected_attempt",
        "branches",
        "branches_payload_sha256",
        "wall_seconds_before_validation",
        "platform",
        "payload_sha256",
    }
    if set(results) != expected_result_keys:
        raise RuntimeError("results JSON schema changed")
    branches = results.get("branches")
    expected_ids = [branch_id(index) for index in range(TOTAL_BRANCH_COUNT)]
    if not isinstance(branches, list) or [item.get("branch_id") for item in branches] != expected_ids:
        raise RuntimeError("branch set/order differs from the frozen protocol")

    baseline_grid = observed.baseline.root / "sample.png"
    baseline_target = observed.baseline.root / f"images/{target_batch_index:02d}_class{CLASS_IDS[target_batch_index]:04d}.png"
    baseline_grid_record = next(
        item for item in observed.baseline.output_records if item["relative_path"] == "sample.png"
    )
    baseline_target_record = next(
        item
        for item in observed.baseline.output_records
        if item["relative_path"] == f"images/{target_batch_index:02d}_class{CLASS_IDS[target_batch_index]:04d}.png"
    )
    if sha256_file(baseline_grid) != baseline_grid_record["sha256"] or sha256_file(baseline_target) != baseline_target_record["sha256"]:
        raise RuntimeError("frozen baseline PNG changed after input validation")
    baseline_tiles = [_grid_tile_pixels(baseline_grid, index) for index in range(BATCH_SIZE)]
    rows = _trace_rows(observed.arrays)
    non_target = [index for index in range(BATCH_SIZE) if index != target_batch_index]
    expected_files = {(root / "manifest.json").resolve(), (root / "results.json").resolve()}
    if require_completion:
        expected_files.add((root / "completion.json").resolve())
    shapes = _expected_trace_shapes(rollback_internal_timestep)
    expected_step_axis = np.arange(rollback_internal_timestep, -1, -1, dtype=np.int16)

    for attempt_index, record in enumerate(branches):
        expected_id = branch_id(attempt_index)
        expected_role = "exact_replay_control" if attempt_index == 0 else "fresh_target_suffix_attempt"
        expected_seed = branch_stream_seed(
            observed.identity_sha256,
            public_seed=int(manifest["seed"]),
            rollback_internal_timestep=rollback_internal_timestep,
            target_batch_index=target_batch_index,
            attempt_index=attempt_index,
        )
        fixed_branch = {
            "branch_id": expected_id,
            "attempt_index": attempt_index,
            "role": expected_role,
            "stream_seed": expected_seed,
            "target_uses_fresh_suffix": attempt_index > 0,
            "full_2b_proposal_draws": rollback_internal_timestep + 1,
            "transition_count": rollback_internal_timestep + 1,
        }
        branch_mismatches = {
            key: (record.get(key), value)
            for key, value in fixed_branch.items()
            if record.get(key) != value
        }
        if branch_mismatches:
            raise RuntimeError(f"branch identity/accounting mismatch: {expected_id}: {branch_mismatches}")
        expected_branch_keys = {
            "branch_id",
            "attempt_index",
            "role",
            "stream_seed",
            "target_uses_fresh_suffix",
            "full_2b_proposal_draws",
            "transition_count",
            "transitions",
            "trace",
            "target_image",
            "full_grid",
            "target_grid_tile_pixel_sha256",
            "final_first_half",
        }
        if set(record) != expected_branch_keys:
            raise RuntimeError(f"branch JSON schema changed: {expected_id}")
        directory = _branch_dir(root, expected_id)
        trace_path = directory / "trace.npz"
        target_path = directory / "target.png"
        grid_path = directory / "grid.png"
        expected_files.update({trace_path.resolve(), target_path.resolve(), grid_path.resolve()})
        arrays = _load_npz_exact(trace_path, record.get("trace", {}), root)
        if any(arrays[key].shape != shape for key, shape in shapes.items()):
            raise RuntimeError(f"branch trace shape mismatch: {expected_id}")
        if not np.array_equal(arrays["transition_internal_timestep"], expected_step_axis):
            raise RuntimeError(f"branch timestep axis changed: {expected_id}")
        if not np.array_equal(
            arrays["target_state_before"][0],
            observed.arrays["x_t"][rows[rollback_internal_timestep], target_batch_index],
        ):
            raise RuntimeError(f"branch did not restore the supplied target x_t: {expected_id}")

        transitions = record.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != rollback_internal_timestep + 1:
            raise RuntimeError(f"transition trail is incomplete: {expected_id}")
        for step_index, transition in enumerate(transitions):
            internal_t = rollback_internal_timestep - step_index
            row = rows[internal_t]
            fixed_transition = {
                "step_index": step_index,
                "internal_timestep": internal_t,
                "original_timestep": int(observed.timestep_map[internal_t]),
                "stochastic_effect": internal_t > 0,
                "full_2b_proposal_draw_ordinal": step_index + 1,
                "t0_proposal_consumed_then_zero_multiplied": internal_t == 0,
                "target_uses_fresh_proposal": attempt_index > 0,
            }
            if any(transition.get(key) != value for key, value in fixed_transition.items()):
                raise RuntimeError(f"transition identity changed: {expected_id}/t={internal_t}")
            expected_transition_keys = {
                "step_index",
                "internal_timestep",
                "original_timestep",
                "stochastic_effect",
                "full_2b_proposal_draw_ordinal",
                "full_2b_proposal",
                "t0_proposal_consumed_then_zero_multiplied",
                "target_uses_fresh_proposal",
                "target_state_before",
                "target_pred_xstart",
                "target_p_mean",
                "target_p_standard_deviation",
                "target_baseline_innovation_raw_sha256",
                "target_used_innovation",
                "non_target_state_before_raw_sha256",
                "non_target_pred_xstart_raw_sha256",
                "non_target_p_mean_raw_sha256",
                "non_target_p_sigma_raw_sha256",
                "non_target_state_after_raw_sha256",
                "full_first_half_state_before",
                "full_first_half_state_after",
                "global_cuda_rng_before_model",
                "global_cuda_rng_after_model",
            }
            if set(transition) != expected_transition_keys:
                raise RuntimeError(f"transition JSON schema changed: {expected_id}/t={internal_t}")
            for key, shape in (
                ("full_2b_proposal", (FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("target_state_before", (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("target_pred_xstart", (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("target_p_mean", (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("target_p_standard_deviation", (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("target_used_innovation", (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("full_first_half_state_before", (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
                ("full_first_half_state_after", (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
            ):
                _require_tensor_record(transition.get(key), shape, f"{expected_id}/t={internal_t}/{key}")
            array_bindings = {
                "full_2b_proposal": arrays["fresh_full_proposal"][step_index],
                "target_state_before": arrays["target_state_before"][step_index],
                "target_pred_xstart": arrays["target_pred_xstart"][step_index],
                "target_p_mean": arrays["target_p_mean"][step_index],
                "target_p_standard_deviation": arrays["target_p_standard_deviation"][step_index],
            }
            for key, array in array_bindings.items():
                tensor_record = transition[key]
                if tensor_record != _numpy_tensor_record(array):
                    raise RuntimeError(
                        f"transition tensor record does not reconstruct: {expected_id}/t={internal_t}/{key}"
                    )
            baseline_target_noise = observed.arrays["innovation_first_half"][row, target_batch_index]
            used_target_noise = (
                baseline_target_noise
                if attempt_index == 0
                else arrays["fresh_full_proposal"][step_index, target_batch_index]
            )
            if transition.get("target_baseline_innovation_raw_sha256") != _array_raw_sha256(baseline_target_noise):
                raise RuntimeError(f"baseline target-noise provenance failed: {expected_id}/t={internal_t}")
            if transition["target_used_innovation"] != _numpy_tensor_record(used_target_noise):
                raise RuntimeError(f"target used-noise construction failed: {expected_id}/t={internal_t}")
            expected_non_target_hashes = {
                "non_target_state_before_raw_sha256": _array_raw_sha256(
                    observed.arrays["x_t"][row, non_target]
                ),
                "non_target_pred_xstart_raw_sha256": _array_raw_sha256(
                    observed.arrays["pred_xstart"][row, non_target]
                ),
                "non_target_p_mean_raw_sha256": _array_raw_sha256(
                    observed.arrays["p_mean_first_half"][row, non_target]
                ),
                "non_target_p_sigma_raw_sha256": _array_raw_sha256(
                    observed.arrays["p_standard_deviation"][row, non_target]
                ),
            }
            if internal_t > 0:
                expected_non_target_after = observed.arrays["x_t"][rows[internal_t - 1], non_target]
            else:
                expected_non_target_after = observed.arrays["final_latents_first_half"][non_target]
            expected_non_target_hashes["non_target_state_after_raw_sha256"] = _array_raw_sha256(
                expected_non_target_after
            )
            if any(transition.get(key) != value for key, value in expected_non_target_hashes.items()):
                raise RuntimeError(f"non-target control provenance failed: {expected_id}/t={internal_t}")
            if transition.get("global_cuda_rng_before_model") != transition.get("global_cuda_rng_after_model"):
                raise RuntimeError(f"model call consumed global RNG: {expected_id}/t={internal_t}")

            multiplier = np.float32(1.0 if internal_t > 0 else 0.0)
            expected_target_after = (
                arrays["target_p_mean"][step_index]
                + multiplier
                * arrays["target_p_standard_deviation"][step_index]
                * used_target_noise
            )
            actual_target_after = (
                arrays["target_state_before"][step_index + 1]
                if step_index + 1 < len(transitions)
                else arrays["final_first_half"][target_batch_index]
            )
            if not np.array_equal(expected_target_after, actual_target_after):
                maximum = float(
                    np.max(
                        np.abs(expected_target_after.astype(np.float64) - actual_target_after.astype(np.float64)),
                        initial=0.0,
                    )
                )
                if maximum > 2e-6:
                    raise RuntimeError(
                        f"target transition does not reconstruct: {expected_id}/t={internal_t}/max_abs={maximum}"
                    )
            expected_full_before = np.ascontiguousarray(
                observed.arrays["x_t"][row].copy(), dtype=np.float32
            )
            expected_full_before[target_batch_index] = arrays["target_state_before"][step_index]
            baseline_full_after = (
                observed.arrays["x_t"][rows[internal_t - 1]].copy()
                if internal_t > 0
                else observed.arrays["final_latents_first_half"].copy()
            )
            expected_full_after = np.ascontiguousarray(
                baseline_full_after, dtype=np.float32
            )
            expected_full_after[target_batch_index] = actual_target_after
            if transition["full_first_half_state_before"] != _numpy_tensor_record(
                expected_full_before
            ) or transition["full_first_half_state_after"] != _numpy_tensor_record(
                expected_full_after
            ):
                raise RuntimeError(f"full first-half state record failed: {expected_id}/t={internal_t}")

            if attempt_index == 0:
                replay_expected = {
                    "target_state_before": observed.arrays["x_t"][row, target_batch_index],
                    "target_pred_xstart": observed.arrays["pred_xstart"][row, target_batch_index],
                    "target_p_mean": observed.arrays["p_mean_first_half"][row, target_batch_index],
                    "target_p_standard_deviation": observed.arrays["p_standard_deviation"][row, target_batch_index],
                }
                if any(not np.array_equal(arrays[key][step_index], value) for key, value in replay_expected.items()):
                    raise RuntimeError(f"attempt 0 differs from observer trace: t={internal_t}")

        expected_final = observed.arrays["final_latents_first_half"]
        if not np.array_equal(arrays["final_first_half"][non_target], expected_final[non_target]):
            raise RuntimeError(f"non-target final latents changed: {expected_id}")
        if attempt_index == 0 and not np.array_equal(arrays["final_first_half"], expected_final):
            raise RuntimeError("attempt 0 final latents differ from observer")
        final_record = record.get("final_first_half")
        _require_tensor_record(
            final_record,
            (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
            f"{expected_id}/final_first_half",
        )
        if final_record != _numpy_tensor_record(arrays["final_first_half"]):
            raise RuntimeError(f"final latent record is not bound to trace: {expected_id}")

        target_record = _png_record(target_path, root, grid=False)
        grid_record = _png_record(grid_path, root, grid=True)
        if record.get("target_image") != target_record or record.get("full_grid") != grid_record:
            raise RuntimeError(f"branch PNG record changed: {expected_id}")
        target_tile = _grid_tile_pixels(grid_path, target_batch_index)
        if _pixel_sha(target_tile) != target_record["pixel_sha256"] or record.get(
            "target_grid_tile_pixel_sha256"
        ) != target_record["pixel_sha256"]:
            raise RuntimeError(f"target PNG/grid tile mismatch: {expected_id}")
        for index in non_target:
            if not np.array_equal(_grid_tile_pixels(grid_path, index), baseline_tiles[index]):
                raise RuntimeError(f"non-target output tile changed: {expected_id}/batch={index}")
        if attempt_index == 0:
            if target_record["pixel_sha256"] != baseline_target_record["pixel_sha256"]:
                raise RuntimeError("attempt 0 target is not pixel-identical to baseline")
            if grid_record["pixel_sha256"] != baseline_grid_record["pixel_sha256"]:
                raise RuntimeError("attempt 0 grid is not pixel-identical to baseline")

    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"bundle file set changed; missing={sorted(expected_files-actual_files)[:2]}, "
            f"extra={sorted(actual_files-expected_files)[:2]}"
        )
    expected_directories = {
        (root / "branches").resolve(),
        *(_branch_dir(root, item).resolve() for item in expected_ids),
    }
    actual_directories = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_directories != expected_directories:
        raise RuntimeError(
            f"bundle directory set changed; missing={sorted(expected_directories-actual_directories)[:2]}, "
            f"extra={sorted(actual_directories-expected_directories)[:2]}"
        )
    branch_payload_sha256 = sha256_json(branches)
    if results.get("branches_payload_sha256") != branch_payload_sha256:
        raise RuntimeError("branch aggregate hash changed")
    if require_completion:
        completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
        fixed_completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(root / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(root / "results.json"),
            "branches_payload_sha256": branch_payload_sha256,
            "branch_count": TOTAL_BRANCH_COUNT,
            "fresh_attempt_count": FROZEN_FRESH_ATTEMPT_COUNT,
        }
        if any(completion.get(key) != value for key, value in fixed_completion.items()):
            raise RuntimeError("completion links/hashes changed")
    return results


def run_real(
    args: argparse.Namespace,
    *,
    observed: ObservedInput,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae_identity: dict[str, Any],
) -> None:
    if args.outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output path: {args.outdir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real DiT-XL/2 suffix screen")
    ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    manifest = build_manifest(
        args,
        observed=observed,
        source=source,
        checkpoint=checkpoint,
        vae=vae_identity,
    )
    started = time.time()
    args.outdir.parent.mkdir(parents=True, exist_ok=True)

    def _execute(staging: Path) -> list[dict[str, Any]]:
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models
        from torchvision.utils import save_image

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected = {
            "diffusion": (args.dit_root / "diffusion/__init__.py").resolve(),
            "download": (args.dit_root / "download.py").resolve(),
            "models": (args.dit_root / "models.py").resolve(),
        }
        if imported != expected:
            raise RuntimeError(f"upstream import shadowing detected: {imported} != {expected}")
        torch.manual_seed(args.seed)
        prior_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            device = torch.device("cuda")
            model = DiT_models[MODEL_NAME](input_size=LATENT_SIZE, num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(find_model(str(args.checkpoint)))
            model.eval()
            diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
            vae = AutoencoderKL.from_pretrained(
                str(args.vae_snapshot), local_files_only=True, use_safetensors=True
            ).to(device)
            if not np.array_equal(np.asarray(diffusion.timestep_map), observed.timestep_map):
                raise RuntimeError("runtime DiT timestep map differs from observer")
            baseline_final = torch.from_numpy(observed.arrays["final_latents_first_half"]).to(device)
            baseline_decoded = vae.decode(baseline_final / VAE_SCALING_FACTOR).sample
            branch_records: list[dict[str, Any]] = []
            for attempt_index in range(TOTAL_BRANCH_COUNT):
                stream_seed = branch_stream_seed(
                    observed.identity_sha256,
                    public_seed=args.seed,
                    rollback_internal_timestep=args.rollback_internal_timestep,
                    target_batch_index=args.target_batch_index,
                    attempt_index=attempt_index,
                )
                result = run_suffix_branch(
                    diffusion,
                    model,
                    vae,
                    observed,
                    rollback_internal_timestep=args.rollback_internal_timestep,
                    target_batch_index=args.target_batch_index,
                    attempt_index=attempt_index,
                    stream_seed=stream_seed,
                    device=device,
                )
                non_target = [index for index in range(BATCH_SIZE) if index != args.target_batch_index]
                if not torch.equal(result.decoded[non_target], baseline_decoded[non_target]):
                    raise RuntimeError(f"decoded non-target tensors changed in {result.branch_id}")
                if attempt_index == 0 and not torch.equal(result.decoded, baseline_decoded):
                    raise RuntimeError("attempt 0 decoded tensor differs from observed baseline latent decode")
                branch_records.append(
                    save_branch_outputs(
                        result,
                        staging,
                        target_batch_index=args.target_batch_index,
                        save_image=save_image,
                    )
                )
                print(
                    f"saved {result.branch_id} ({attempt_index + 1}/{TOTAL_BRANCH_COUNT})",
                    flush=True,
                )
            torch.cuda.synchronize()
            return branch_records
        finally:
            torch.set_grad_enabled(prior_grad)

    with tempfile.TemporaryDirectory(prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        branch_records = _with_upstream_imports(args.dit_root, lambda: _execute(staging))
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "oracle_mechanics_only": True,
            "online_sampling_method": False,
            "method_claim_eligible": False,
            "conditional_Ville_bound_applicable": False,
            "TV_bound_applicable": False,
            "rollback_internal_timestep": args.rollback_internal_timestep,
            "target_batch_index": args.target_batch_index,
            "target_class_id": int(CLASS_IDS[args.target_batch_index]),
            "fresh_attempt_count": FROZEN_FRESH_ATTEMPT_COUNT,
            "selection_performed": False,
            "selected_attempt": None,
            "branches": branch_records,
            "branches_payload_sha256": sha256_json(branch_records),
            "wall_seconds_before_validation": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": dependency_identity(),
                "cuda_device_count_visible": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
        }
        results["payload_sha256"] = _canonical_self_hash(results, "payload_sha256")
        atomic_json_dump(results, staging / "results.json")
        validate_bundle(
            staging,
            manifest=manifest,
            observed=observed,
            rollback_internal_timestep=args.rollback_internal_timestep,
            target_batch_index=args.target_batch_index,
            require_completion=False,
        )
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / "results.json"),
            "branches_payload_sha256": results["branches_payload_sha256"],
            "branch_count": TOTAL_BRANCH_COUNT,
            "fresh_attempt_count": FROZEN_FRESH_ATTEMPT_COUNT,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        validate_bundle(
            staging,
            manifest=manifest,
            observed=observed,
            rollback_internal_timestep=args.rollback_internal_timestep,
            target_batch_index=args.target_batch_index,
            require_completion=True,
        )
        _atomic_install_directory_noreplace(staging, args.outdir)
    final = validate_bundle(
        args.outdir,
        manifest=manifest,
        observed=observed,
        rollback_internal_timestep=args.rollback_internal_timestep,
        target_batch_index=args.target_batch_index,
        require_completion=True,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "outdir": str(args.outdir),
                "rollback_internal_timestep": args.rollback_internal_timestep,
                "branches": [item["branch_id"] for item in final["branches"]],
                "selection_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


class _ToyModel(torch.nn.Module):
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        del y, cfg_scale
        half = x[: len(x) // 2]
        value = 0.1 * half + t[: len(half)].float().view(-1, 1, 1, 1) * 0.01
        return torch.cat([value, value], dim=0)


class _ToyDiffusion:
    num_timesteps = 4
    timestep_map = np.arange(4, dtype=np.int64)

    def p_mean_variance(
        self,
        model_fn: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool,
        model_kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if clip_denoised:
            raise AssertionError("toy received clip_denoised=True")
        prediction = model_fn(x, t, **model_kwargs)
        mean = 0.8 * x + 0.1 * prediction
        sigma = torch.full_like(x, 0.2)
        return {
            "mean": mean,
            "pred_xstart": prediction,
            "log_variance": torch.log(torch.square(sigma)),
        }


class _ToyVAE:
    class _Decoded:
        def __init__(self, sample: torch.Tensor):
            self.sample = sample

    def decode(self, value: torch.Tensor) -> "_ToyVAE._Decoded":
        return self._Decoded(value)


def _toy_observed_input() -> ObservedInput:
    generator = torch.Generator(device="cpu").manual_seed(31)
    diffusion = _ToyDiffusion()
    model = _ToyModel()
    first = torch.randn(
        (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), generator=generator
    )
    x = torch.cat([first, first], dim=0)
    y = torch.cat(
        [torch.tensor(CLASS_IDS), torch.full((BATCH_SIZE,), NULL_CLASS_ID)], dim=0
    )
    kwargs = {"y": y, "cfg_scale": CFG_SCALE}
    states = []
    predictions = []
    means = []
    sigmas = []
    noises = []
    for internal_t in range(3, -1, -1):
        t = torch.full((FULL_BATCH_SIZE,), internal_t, dtype=torch.long)
        out = diffusion.p_mean_variance(
            model.forward_with_cfg, x, t, clip_denoised=False, model_kwargs=kwargs
        )
        sigma = torch.exp(0.5 * out["log_variance"])
        noise = torch.randn(x.shape, generator=generator)
        states.append(_tensor_numpy(x[:BATCH_SIZE]))
        predictions.append(_tensor_numpy(out["pred_xstart"][:BATCH_SIZE]))
        means.append(_tensor_numpy(out["mean"][:BATCH_SIZE]))
        sigmas.append(_tensor_numpy(sigma[:BATCH_SIZE]))
        noises.append(_tensor_numpy(noise[:BATCH_SIZE]))
        x = out["mean"] + float(internal_t > 0) * sigma * noise
    arrays = {
        "internal_timestep": np.asarray([3, 2, 1, 0], dtype=np.int16),
        "x_t": np.ascontiguousarray(np.stack(states), dtype=np.float32),
        "pred_xstart": np.ascontiguousarray(np.stack(predictions), dtype=np.float32),
        "p_mean_first_half": np.ascontiguousarray(np.stack(means), dtype=np.float32),
        "p_standard_deviation": np.ascontiguousarray(np.stack(sigmas), dtype=np.float32),
        "innovation_first_half": np.ascontiguousarray(np.stack(noises), dtype=np.float32),
        "final_latents_first_half": np.ascontiguousarray(_tensor_numpy(x[:BATCH_SIZE]), dtype=np.float32),
    }
    identity = "1" * 64
    return ObservedInput(
        root=Path("/toy"),
        manifest={"identity_sha256": identity},
        results={},
        baseline=None,
        arrays=arrays,
        timestep_map=np.arange(4, dtype=np.int64),
    )


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must begin without CUDA initialization")
    observed = _toy_observed_input()
    # Temporarily exercise t=3 with the production branch mechanism.  The
    # frozen real checkpoints are guarded by the public runner; this direct
    # unit test substitutes one member only around the call boundary.
    original_checkpoints = globals()["FROZEN_ROLLBACK_INTERNAL_TIMESTEPS"]
    globals()["FROZEN_ROLLBACK_INTERNAL_TIMESTEPS"] = (3,)
    try:
        replay_seed = branch_stream_seed(
            observed.identity_sha256,
            public_seed=2,
            rollback_internal_timestep=3,
            target_batch_index=0,
            attempt_index=0,
        )
        replay = run_suffix_branch(
            _ToyDiffusion(),
            _ToyModel(),
            _ToyVAE(),
            observed,
            rollback_internal_timestep=3,
            target_batch_index=0,
            attempt_index=0,
            stream_seed=replay_seed,
            device=torch.device("cpu"),
        )
        attempt_seed = branch_stream_seed(
            observed.identity_sha256,
            public_seed=2,
            rollback_internal_timestep=3,
            target_batch_index=0,
            attempt_index=1,
        )
        attempt = run_suffix_branch(
            _ToyDiffusion(),
            _ToyModel(),
            _ToyVAE(),
            observed,
            rollback_internal_timestep=3,
            target_batch_index=0,
            attempt_index=1,
            stream_seed=attempt_seed,
            device=torch.device("cpu"),
        )
    finally:
        globals()["FROZEN_ROLLBACK_INTERNAL_TIMESTEPS"] = original_checkpoints
    baseline_final = torch.from_numpy(observed.arrays["final_latents_first_half"])
    if not torch.equal(replay.final_first_half, baseline_final):
        raise AssertionError("toy attempt 0 did not exactly replay")
    if not torch.equal(attempt.final_first_half[1:], baseline_final[1:]):
        raise AssertionError("toy target suffix changed a non-target item")
    if torch.equal(attempt.final_first_half[0], baseline_final[0]):
        raise AssertionError("toy fresh target suffix did not change the target")
    if replay.full_2b_proposal_draws != 4 or attempt.full_2b_proposal_draws != 4:
        raise AssertionError("toy did not consume one full-2B proposal including t=0")
    if replay.trace_arrays["fresh_full_proposal"].shape != (
        4,
        FULL_BATCH_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):
        raise AssertionError("toy full-2B proposal trace shape changed")

    all_seeds = {
        branch_stream_seed(
            observed.identity_sha256,
            public_seed=2,
            rollback_internal_timestep=timestep,
            target_batch_index=0,
            attempt_index=attempt_index,
        )
        for timestep in FROZEN_ROLLBACK_INTERNAL_TIMESTEPS
        for attempt_index in range(TOTAL_BRANCH_COUNT)
    }
    if len(all_seeds) != len(FROZEN_ROLLBACK_INTERNAL_TIMESTEPS) * TOTAL_BRANCH_COUNT:
        raise AssertionError("frozen checkpoint/attempt stream seeds collided")

    payload = {"experiment": EXPERIMENT, "toy": True}
    payload["payload_sha256"] = _canonical_self_hash(payload, "payload_sha256")
    with tempfile.TemporaryDirectory(prefix="dit-suffix-self-test-") as temporary:
        root = Path(temporary)
        json_path = root / "payload.json"
        atomic_json_dump(payload, json_path)
        if _read_self_hashed_json(json_path, "payload_sha256") != payload:
            raise AssertionError("self-hashed JSON roundtrip failed")
        trace_path = root / "trace.npz"
        _atomic_npz_dump(attempt.trace_arrays, trace_path)
        record = _trace_record(trace_path, attempt.trace_arrays, root)
        loaded = _load_npz_exact(trace_path, record, root)
        if any(not np.array_equal(loaded[key], attempt.trace_arrays[key]) for key in loaded):
            raise AssertionError("strict NPZ roundtrip failed")
        publish_source = root / "publish_source"
        publish_source.mkdir()
        (publish_source / "marker").write_bytes(b"source")
        publish_target = root / "publish_target"
        _atomic_install_directory_noreplace(publish_source, publish_target)
        if publish_source.exists() or (publish_target / "marker").read_bytes() != b"source":
            raise AssertionError("atomic no-replace publication failed")
        racing_source = root / "racing_source"
        racing_source.mkdir()
        (racing_source / "marker").write_bytes(b"new")
        racing_target = root / "racing_target"
        racing_target.mkdir()
        (racing_target / "marker").write_bytes(b"existing")
        try:
            _atomic_install_directory_noreplace(racing_source, racing_target)
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic publication replaced an existing directory")
        if (racing_target / "marker").read_bytes() != b"existing" or not racing_source.is_dir():
            raise AssertionError("failed no-replace publication changed source or target")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: exact attempt-0 replay, target-only fresh suffix, "
        "non-target invariance, full-2B/t0 proposal consumption, frozen seed "
        "domain separation, self-hashes, strict NPZ validation, and atomic "
        "RENAME_NOREPLACE publication (CPU-only)"
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _atomic_install_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a staged directory without replacing any target.

    Plain ``os.replace`` can replace an empty directory created by a racing
    process.  Linux ``renameat2(RENAME_NOREPLACE)`` closes that gap while
    retaining same-filesystem atomic directory publication.  There is no
    overwrite-capable fallback: unsupported platforms fail closed.
    """

    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"staged source is not a plain directory: {source}")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory install requires Linux renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, "refusing to replace concurrently created output", target)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("filesystem/kernel does not support atomic RENAME_NOREPLACE")
    raise OSError(error_number, os.strerror(error_number), target)


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_dit = data_root / "baselines/DiT"
    default_vae = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--rollback-internal-timestep",
        type=int,
        choices=FROZEN_ROLLBACK_INTERNAL_TIMESTEPS,
        default=180,
        help="One member of the frozen (225,180,120,60) repairability screen.",
    )
    parser.add_argument(
        "--target-batch-index",
        type=int,
        choices=range(BATCH_SIZE),
        default=DEFAULT_TARGET_BATCH_INDEX,
    )
    parser.add_argument("--observe-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--dit-root", type=Path, default=default_dit)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=default_vae)
    parser.add_argument("--outdir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def normalize_and_validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 <= args.seed < 1 << 63:
        parser.error("--seed must lie in [0,2^63-1]")
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")).expanduser().absolute().resolve()
    args.baseline_dir = (
        data_root / "cross_scale_evidence/dit_imagenet256" / f"official_demo_seed{args.seed}"
        if args.baseline_dir is None
        else args.baseline_dir.expanduser().absolute().resolve()
    )
    if args.observe_dir is None:
        observer_sha_prefix = sha256_file(
            Path(__file__).resolve().with_name("observe_dit_imagenet256_path_evidence.py")
        )[:7]
        args.observe_dir = (
            data_root
            / "cross_scale_evidence/dit_imagenet256_path_evidence"
            / f"official_seed{args.seed}_dnu1_K0p5_grid4_{observer_sha_prefix}"
        )
    else:
        args.observe_dir = args.observe_dir.expanduser().absolute().resolve()
    requested_outdir = (
        data_root
        / "cross_scale_evidence/dit_imagenet256_suffix_repairability"
        / (
            f"official_demo_seed{args.seed}_batch{args.target_batch_index}_"
            f"class{CLASS_IDS[args.target_batch_index]:04d}_t{args.rollback_internal_timestep}_n4"
        )
        if args.outdir is None
        else args.outdir.expanduser().absolute()
    )
    if os.path.lexists(requested_outdir):
        parser.error(f"no-overwrite target already exists: {requested_outdir}")
    args.outdir = requested_outdir.resolve()
    protected = {
        "observe bundle": args.observe_dir,
        "frozen baseline": args.baseline_dir,
        "DiT source/checkpoint": args.dit_root,
        "VAE snapshot": args.vae_snapshot,
        "research repository": Path(__file__).resolve().parent.parent,
    }
    overlaps = [label for label, path in protected.items() if _paths_overlap(args.outdir, path)]
    if overlaps:
        parser.error("--outdir overlaps protected input/source path(s): " + ", ".join(overlaps))


def dry_run(
    args: argparse.Namespace,
    *,
    observed: ObservedInput,
    source: dict[str, Any],
    checkpoint_probe: dict[str, Any],
    vae: dict[str, Any],
) -> None:
    blockers = []
    if not checkpoint_probe["exists"]:
        blockers.append("checkpoint file missing")
    elif not checkpoint_probe["size_matches"]:
        blockers.append("checkpoint size mismatch/incomplete download")
    if not checkpoint_probe["sha256_pinned"]:
        blockers.append("checkpoint SHA is not pinned")
    seeds = [
        branch_stream_seed(
            observed.identity_sha256,
            public_seed=args.seed,
            rollback_internal_timestep=args.rollback_internal_timestep,
            target_batch_index=args.target_batch_index,
            attempt_index=index,
        )
        for index in range(TOTAL_BRANCH_COUNT)
    ]
    payload = {
        "status": "dry-run",
        "experiment": EXPERIMENT,
        "oracle_mechanics_only": True,
        "gpu_model_loaded": False,
        "frozen_rollback_internal_timesteps": list(FROZEN_ROLLBACK_INTERNAL_TIMESTEPS),
        "this_invocation_rollback_internal_timestep": args.rollback_internal_timestep,
        "fresh_attempt_count": FROZEN_FRESH_ATTEMPT_COUNT,
        "total_branch_count": TOTAL_BRANCH_COUNT,
        "target": {
            "batch_index": args.target_batch_index,
            "class_id": int(CLASS_IDS[args.target_batch_index]),
        },
        "branch_stream_seeds": seeds,
        "observe_bundle": {
            "root": str(observed.root),
            "identity_sha256": observed.identity_sha256,
            "strict_completion_and_trace_math_validated": True,
        },
        "source": source,
        "checkpoint_probe": checkpoint_probe,
        "vae": vae,
        "full_2b_proposal_draws_per_branch_including_t0": args.rollback_internal_timestep + 1,
        "automatic_scoring_ranking_or_selection": False,
        "method_claim_eligible": False,
        "outdir": str(args.outdir),
        "real_run_blockers": blockers,
        "static_inputs_ready": not blockers,
        "cuda_available": torch.cuda.is_available(),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    normalize_and_validate_args(args, parser)
    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    if args.dry_run:
        observed = load_observed_input(args, source=source, checkpoint=None, vae=vae)
        dry_run(
            args,
            observed=observed,
            source=source,
            checkpoint_probe=checkpoint_dry_probe(args.checkpoint),
            vae=vae,
        )
        return 0
    checkpoint = validate_checkpoint(args.checkpoint)
    observed = load_observed_input(
        args,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    run_real(
        args,
        observed=observed,
        source=source,
        checkpoint=checkpoint,
        vae_identity=vae,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

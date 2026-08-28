#!/usr/bin/env python3
"""Exploratory DiT-v2.2 custom-trace suffix repairability runner.

This runner receives one already completed ``trace_dit_imagenet256_custom_batch``
bundle and one fixed target slot.  A supplied *sampling-step index* selects the
saved ``state_before`` after that step's prediction is available and before its
transition is made.  Five branches are retained without scoring or selection:

* attempt 0 exactly reuses every saved first-half innovation;
* attempts 1..4 replace only the target slot's suffix innovations with four
  independently seeded streams.

All other first-half slots remain exact trace controls.  Every fresh branch
draws the official full 2B proposal at every suffix transition, including t=0;
the t=0 proposal is consumed and then multiplied by zero.  The frozen official
DiT-XL/2 DDPM-250, CFG=4, checkpoint, and MSE-VAE lineages are validated against
the source trace.  Outputs contain native target PNGs and mechanical trace
metadata only: no labels, endpoint scores, FID/features, ranking, or best-of-N.

This is a post-hoc repairability experiment, not an online v2.2 detector and
not evidence for a Ville/TV/retry guarantee.  Existing outputs are immutable;
a complete identical bundle is validated, and every other existing path is
refused.  New bundles are staged, validated, and installed atomically.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch

try:
    from . import reproduce_dit_imagenet256 as strict
    from . import sample_dit_imagenet256_custom as custom
    from . import trace_dit_imagenet256_custom_batch as custom_trace
except ImportError:  # pragma: no cover - direct CLI invocation
    import reproduce_dit_imagenet256 as strict
    import sample_dit_imagenet256_custom as custom
    import trace_dit_imagenet256_custom_batch as custom_trace


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


RUNNER_NAME = "intervene_dit_v22_custom_trace_suffix"
EXPERIMENT = "dit_v22_custom_trace_suffix_repairability"
SCHEMA_VERSION = 1
FRESH_ATTEMPTS = 4
BRANCH_COUNT = 1 + FRESH_ATTEMPTS
RNG_NAMESPACE = "eqvae-dit-v22-custom-trace-suffix-v1"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
COMPLETION_NAME = "completion.json"
SNAPSHOT_NAME = "runner_source.py"
PILOT_LOCK_KIND = "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2"

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "internal_timestep": np.dtype(np.int16),
    "target_state_before": np.dtype(np.float32),
    "target_pred_xstart": np.dtype(np.float32),
    "target_p_mean": np.dtype(np.float32),
    "target_p_standard_deviation": np.dtype(np.float32),
    "target_used_innovation": np.dtype(np.float32),
    "target_state_after": np.dtype(np.float32),
    "final_first_half": np.dtype(np.float32),
}


@dataclass(frozen=True)
class SavedTrace:
    root: Path
    manifest: dict[str, Any]
    identity: dict[str, Any]
    classes: tuple[int, ...]
    seed: int
    arrays: dict[str, np.ndarray]

    @property
    def identity_sha256(self) -> str:
        return str(self.manifest["identity_sha256"])


@dataclass
class BranchResult:
    attempt_index: int
    role: str
    stream_seed: int | None
    arrays: dict[str, np.ndarray]
    decoded_target: torch.Tensor
    transition_records: list[dict[str, Any]]
    fresh_full_2b_draw_count: int


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _self_hash(payload: Mapping[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return _sha256_json(stripped)


def _load_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    value = strict.load_json(path)
    observed = value.get(key)
    if not isinstance(observed, str) or observed != _self_hash(value, key):
        raise RuntimeError(f"invalid {key}: {path}")
    return value


def _raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError("expected a finite float32 tensor")
    return array


def _array_record(array: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(array)
    values = canonical.astype(np.float64, copy=False)
    return {
        "shape": list(canonical.shape),
        "dtype": str(canonical.dtype),
        "raw_sha256": _raw_sha(canonical),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "root_mean_square": float(np.sqrt(np.mean(values * values))),
    }


def _global_rng_sha(device: torch.device) -> str:
    state = torch.cuda.get_rng_state() if device.type == "cuda" else torch.get_rng_state()
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _branch_name(attempt_index: int) -> str:
    if type(attempt_index) is not int or not 0 <= attempt_index < BRANCH_COUNT:
        raise ValueError("attempt index is outside 0..4")
    return f"attempt_{attempt_index:03d}"


def _branch_seed(
    trace_identity_sha256: str,
    *,
    global_seed: int,
    rollback_sampling_step: int,
    target_slot: int,
    attempt_index: int,
) -> int:
    if attempt_index <= 0 or attempt_index >= BRANCH_COUNT:
        raise ValueError("only fresh attempts 1..4 have branch seeds")
    payload = (
        f"{RNG_NAMESPACE}\0{trace_identity_sha256}\0{global_seed}\0"
        f"{rollback_sampling_step}\0{target_slot}\0{attempt_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _trace_record_from_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("saved trace manifest has no output records")
    matches = [row for row in outputs if row.get("relative_path") == custom_trace.TRACE_NAME]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("saved trace manifest does not contain one trace.npz record")
    return matches[0]


def load_saved_trace(
    trace_dir: Path,
    *,
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    vae_snapshot: Mapping[str, Any],
) -> SavedTrace:
    if trace_dir.is_symlink() or not trace_dir.is_dir():
        raise RuntimeError(f"trace directory must be a real directory: {trace_dir}")
    if any(path.is_symlink() for path in trace_dir.rglob("*")):
        raise RuntimeError("saved custom trace bundle contains a symlink")
    manifest = strict.load_json(trace_dir / custom_trace.MANIFEST_NAME)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("saved trace manifest lacks identity")
    if identity.get("runner") != custom_trace.RUNNER_NAME:
        raise RuntimeError("input was not made by trace_dit_imagenet256_custom_batch")
    protocol = identity.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("saved trace identity lacks protocol")
    classes_raw = protocol.get("class_ids_ordered")
    classes = custom.parse_classes(",".join(map(str, classes_raw or [])))
    batch_size = len(classes)
    fixed_protocol = {
        "model": strict.MODEL_NAME,
        "batch_size_before_duplication": batch_size,
        "sampler_batch_size": 2 * batch_size,
        "sampling_steps": strict.NUM_SAMPLING_STEPS,
        "clip_denoised": False,
        "cfg_scale": strict.CFG_SCALE,
        "full_2B_randn_like_each_transition_including_t0": True,
        "trace_axis_order": "[B, sampling_step, C, H, W]",
        "internal_timestep_order": "249..0",
    }
    mismatches = {
        key: (protocol.get(key), expected)
        for key, expected in fixed_protocol.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"saved trace sampler contract changed: {mismatches}")
    for key, expected in (
        ("source", source),
        ("checkpoint", checkpoint),
        ("vae_snapshot", vae_snapshot),
    ):
        if identity.get(key) != expected:
            raise RuntimeError(f"saved trace {key} lineage differs from requested runtime")
    seed = protocol.get("global_torch_seed")
    if type(seed) is not int or seed < 0:
        raise RuntimeError("saved trace global seed is invalid")

    # This checks the manifest/completion/output aggregate, every PNG, source
    # snapshot, NPZ member/dtype/hash, and exact transition replay.
    custom_trace.validate_completed_output(trace_dir, identity, classes)
    trace_record = _trace_record_from_manifest(manifest)
    arrays = custom_trace._load_trace(
        trace_dir / custom_trace.TRACE_NAME,
        trace_record.get("arrays", {}),
        batch_size,
    )
    if manifest.get("identity_sha256") != strict.sha256_json(identity):
        raise RuntimeError("saved trace identity hash changed")
    return SavedTrace(
        root=trace_dir.resolve(),
        manifest=manifest,
        identity=identity,
        classes=classes,
        seed=seed,
        arrays=arrays,
    )


def _validate_target(trace: SavedTrace, target_slot: int, expected_class_id: int | None) -> int:
    if type(target_slot) is not int or not 0 <= target_slot < len(trace.classes):
        raise ValueError(f"target slot must lie in [0,{len(trace.classes) - 1}]")
    class_id = trace.classes[target_slot]
    if expected_class_id is not None and expected_class_id != class_id:
        raise RuntimeError(
            f"target slot {target_slot} is class {class_id}, not expected class {expected_class_id}"
        )
    return class_id


def _validate_rollback(trace: SavedTrace, sampling_step: int) -> tuple[int, int, int]:
    if type(sampling_step) is not int or not 0 <= sampling_step < strict.NUM_SAMPLING_STEPS:
        raise ValueError("rollback sampling step must lie in [0,249]")
    internal_t = int(trace.arrays["internal_timestep"][sampling_step])
    expected = strict.NUM_SAMPLING_STEPS - 1 - sampling_step
    if internal_t != expected:
        raise RuntimeError("saved sampling-step/internal-time mapping changed")
    transition_count = strict.NUM_SAMPLING_STEPS - sampling_step
    stochastic_transition_count = internal_t
    if transition_count != internal_t + 1:
        raise AssertionError("suffix transition accounting is inconsistent")
    return internal_t, transition_count, stochastic_transition_count


def _pilot_binding(
    pilot_lock: Path | None,
    *,
    trace: SavedTrace,
    target_slot: int,
    target_class: int,
    rollback_sampling_step: int,
) -> dict[str, Any] | None:
    """Validate and bind one output to the frozen v1.2 pilot selection."""

    if pilot_lock is None:
        return None
    root = pilot_lock.expanduser().resolve()
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"pilot lock must be a real symlink-free directory: {root}")
    manifest = _load_self_hashed_json(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != PILOT_LOCK_KIND or manifest.get("status") != "complete":
        raise RuntimeError("pilot lock kind/status changed")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("pilot lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != strict.sha256_file(path)
        ):
            raise RuntimeError(f"pilot lock member changed: {name}")
    protocol = _load_self_hashed_json(root / "protocol.json", "identity_sha256")
    if protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256"):
        raise RuntimeError("pilot protocol identity changed")
    if protocol.get("scientific_revision") != "v1.2":
        raise RuntimeError("pilot scientific revision changed")
    allowed_steps = protocol.get("intervention", {}).get("rollback_sampling_steps")
    if allowed_steps != [109, 149] or rollback_sampling_step not in allowed_steps:
        raise RuntimeError("rollback step is outside the frozen v1.2 pilot")
    selected = protocol.get("selected_paths")
    if not isinstance(selected, list):
        raise RuntimeError("pilot selection axis is missing")
    matches = [
        row
        for row in selected
        if isinstance(row, dict)
        and row.get("global_seed") == trace.seed
        and row.get("class_id") == target_class
        and row.get("class_slot") == target_slot
    ]
    if len(matches) != 1:
        raise RuntimeError("target seed/class/slot is not uniquely selected by the pilot")
    selected_row = matches[0]
    role = selected_row.get("role")
    if role not in {"joint_E_and_B", "B_only_exact_schedule_B_matched_control"}:
        raise RuntimeError("pilot target role changed")
    return {
        "pilot_lock_path": str(root),
        "pilot_lock_identity_sha256": manifest["identity_sha256"],
        "pilot_protocol_identity_sha256": protocol["identity_sha256"],
        "scientific_revision": protocol["scientific_revision"],
        "pair_index": int(selected_row["pair_index"]),
        "selected_role": role,
        "selected_row_identity_sha256": _sha256_json(selected_row),
        "rollback_step_allowed": True,
        "selection_uses_internal_B_E_only": True,
    }


def _with_upstream_imports(root: Path, callback: Any) -> Any:
    prior_cwd = Path.cwd()
    prior_path = list(sys.path)
    names = {"models", "download", "diffusion"}
    preexisting = {
        name for name in sys.modules if name in names or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError(f"ambiguous pre-imported upstream modules: {sorted(preexisting)}")
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        return callback()
    finally:
        os.chdir(prior_cwd)
        sys.path[:] = prior_path
        for name in list(sys.modules):
            if (name in names or name.startswith("diffusion.")) and name not in preexisting:
                sys.modules.pop(name, None)


def run_branch(
    diffusion: Any,
    model: Any,
    vae: Any,
    trace: SavedTrace,
    *,
    rollback_sampling_step: int,
    target_slot: int,
    attempt_index: int,
    stream_seed: int | None,
    timestep_map: np.ndarray,
    device: torch.device,
) -> BranchResult:
    """Execute one suffix while retaining exact non-target controls."""

    internal_start, transition_count, _ = _validate_rollback(trace, rollback_sampling_step)
    batch_size = len(trace.classes)
    full_batch_size = 2 * batch_size
    if attempt_index == 0 and stream_seed is not None:
        raise ValueError("attempt 0 must not have a fresh stream")
    if attempt_index > 0 and stream_seed is None:
        raise ValueError("fresh attempts require an independent stream seed")
    first = torch.from_numpy(trace.arrays["state_before"][:, rollback_sampling_step]).to(device)
    x = torch.cat([first, first], dim=0)
    y_cond = torch.tensor(trace.classes, dtype=torch.long, device=device)
    y_null = torch.full((batch_size,), strict.NULL_CLASS_ID, dtype=torch.long, device=device)
    model_kwargs = {
        "y": torch.cat([y_cond, y_null]),
        "cfg_scale": strict.CFG_SCALE,
    }
    generator = (
        None
        if stream_seed is None
        else torch.Generator(device=device).manual_seed(int(stream_seed))
    )
    non_target = [slot for slot in range(batch_size) if slot != target_slot]
    rows = range(rollback_sampling_step, strict.NUM_SAMPLING_STEPS)
    states: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    means: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    afters: list[np.ndarray] = []
    transition_records: list[dict[str, Any]] = []
    fresh_draw_count = 0
    global_before = _global_rng_sha(device)

    for suffix_ordinal, row in enumerate(rows):
        internal_t = int(trace.arrays["internal_timestep"][row])
        if internal_t != internal_start - suffix_ordinal:
            raise RuntimeError("trace suffix internal-time order changed")
        baseline_state = torch.from_numpy(trace.arrays["state_before"][:, row]).to(device)
        checked = list(range(batch_size)) if attempt_index == 0 else non_target
        if not torch.equal(x[:batch_size][checked], baseline_state[checked]):
            raise RuntimeError(f"controlled state differs before sampling step {row}")

        t = torch.full((full_batch_size,), internal_t, dtype=torch.long, device=device)
        rng_before_model = _global_rng_sha(device)
        with torch.no_grad():
            out = diffusion.p_mean_variance(
                model.forward_with_cfg,
                x,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
        rng_after_model = _global_rng_sha(device)
        if rng_after_model != rng_before_model:
            raise RuntimeError("model evaluation unexpectedly consumed global RNG")
        mean = out["mean"]
        pred = out["pred_xstart"]
        sigma = torch.exp(0.5 * out["log_variance"])
        expected_shape = (
            full_batch_size,
            strict.LATENT_CHANNELS,
            strict.LATENT_SIZE,
            strict.LATENT_SIZE,
        )
        if any(tuple(value.shape) != expected_shape for value in (mean, pred, sigma)):
            raise RuntimeError("runtime transition tensor shape changed")

        baseline_pred = torch.from_numpy(trace.arrays["pred_xstart"][:, row]).to(device)
        baseline_mean = torch.from_numpy(trace.arrays["p_mean"][:, row]).to(device)
        baseline_sigma = torch.from_numpy(trace.arrays["p_standard_deviation"][:, row]).to(device)
        for name, actual, expected_value in (
            ("pred_xstart", pred[:batch_size], baseline_pred),
            ("p_mean", mean[:batch_size], baseline_mean),
            ("p_standard_deviation", sigma[:batch_size], baseline_sigma),
        ):
            if not torch.equal(actual[checked], expected_value[checked]):
                raise RuntimeError(
                    f"controlled {name} differs at sampling step {row} / t={internal_t}"
                )

        saved_noise = torch.from_numpy(trace.arrays["transition_innovation"][:, row]).to(device)
        if attempt_index == 0:
            used_first = saved_noise
            fresh_full = None
        else:
            fresh_full = torch.randn(
                expected_shape,
                dtype=x.dtype,
                device=device,
                generator=generator,
            )
            fresh_draw_count += 1
            used_first = saved_noise.clone()
            used_first[target_slot] = fresh_full[target_slot]
        # The second half is immaterial to later first-half predictions because
        # forward_with_cfg reconstructs 2B from x[:B].  Keep it deterministic:
        # replay branches mirror the first half, fresh branches use their drawn
        # second half.  The target first-half contract is the scientific object.
        used_second = used_first if fresh_full is None else fresh_full[batch_size:]
        used_full = torch.cat([used_first, used_second], dim=0)
        nonzero = float(internal_t > 0)
        x_next = mean + nonzero * sigma * used_full

        if row + 1 < strict.NUM_SAMPLING_STEPS:
            expected_next = torch.from_numpy(trace.arrays["state_before"][:, row + 1]).to(device)
        else:
            expected_next = torch.from_numpy(trace.arrays["final_latents"]).to(device)
        if not torch.equal(x_next[:batch_size][checked], expected_next[checked]):
            raise RuntimeError(f"controlled state differs after sampling step {row}")

        state_array = _tensor_numpy(x[target_slot])
        pred_array = _tensor_numpy(pred[target_slot])
        mean_array = _tensor_numpy(mean[target_slot])
        sigma_array = _tensor_numpy(sigma[target_slot])
        used_array = _tensor_numpy(used_first[target_slot])
        after_array = _tensor_numpy(x_next[target_slot])
        states.append(state_array)
        predictions.append(pred_array)
        means.append(mean_array)
        sigmas.append(sigma_array)
        innovations.append(used_array)
        afters.append(after_array)
        transition_records.append(
            {
                "suffix_ordinal": suffix_ordinal,
                "sampling_step": row,
                "internal_timestep": internal_t,
                "original_timestep": int(timestep_map[internal_t]),
                "stochastic_effect": internal_t > 0,
                "innovation_source": "saved_baseline" if attempt_index == 0 else "fresh_branch",
                "full_2b_fresh_draw_ordinal": None if attempt_index == 0 else fresh_draw_count,
                "full_2b_fresh_proposal_raw_sha256": (
                    None if fresh_full is None else _raw_sha(_tensor_numpy(fresh_full))
                ),
                "t0_innovation_consumed_then_zero_multiplied": internal_t == 0,
                "target_state_before_raw_sha256": _raw_sha(state_array),
                "target_pred_xstart_raw_sha256": _raw_sha(pred_array),
                "target_p_mean_raw_sha256": _raw_sha(mean_array),
                "target_p_standard_deviation_raw_sha256": _raw_sha(sigma_array),
                "target_used_innovation_raw_sha256": _raw_sha(used_array),
                "target_state_after_raw_sha256": _raw_sha(after_array),
                "saved_target_innovation_raw_sha256": _raw_sha(
                    trace.arrays["transition_innovation"][target_slot, row]
                ),
                "global_rng_before_model_sha256": rng_before_model,
                "global_rng_after_model_sha256": rng_after_model,
            }
        )
        x = x_next

    if attempt_index == 0 and fresh_draw_count != 0:
        raise AssertionError("replay branch made a fresh proposal")
    if attempt_index > 0 and fresh_draw_count != transition_count:
        raise AssertionError("fresh branch did not draw once per transition including t=0")
    final_first = _tensor_numpy(x[:batch_size])
    if not np.array_equal(final_first[non_target], trace.arrays["final_latents"][non_target]):
        raise RuntimeError("a non-target final latent changed")
    if attempt_index == 0 and not np.array_equal(final_first, trace.arrays["final_latents"]):
        raise RuntimeError("attempt 0 final latent is not exact baseline replay")
    # Preserve the source trace's first-half VAE batch shape.  Decoding only
    # the target item can select a different CUDA convolution algorithm and
    # needlessly break the attempt-0 pixel-identical replay check even when the
    # final latent is exact.
    decoded_first = vae.decode(x[:batch_size] / strict.VAE_SCALING_FACTOR).sample
    expected_decoded_shape = (
        batch_size,
        3,
        strict.IMAGE_SIZE,
        strict.IMAGE_SIZE,
    )
    if (
        tuple(decoded_first.shape) != expected_decoded_shape
        or decoded_first.dtype != torch.float32
        or not bool(torch.isfinite(decoded_first).all())
    ):
        raise RuntimeError("VAE decoded tensor shape/dtype/finiteness changed")
    if _global_rng_sha(device) != global_before:
        raise RuntimeError("branch-local suffix or VAE decode unexpectedly mutated global RNG")
    decoded_target = decoded_first[target_slot]
    arrays = {
        "internal_timestep": np.arange(internal_start, -1, -1, dtype=np.int16),
        "target_state_before": np.ascontiguousarray(np.stack(states), dtype=np.float32),
        "target_pred_xstart": np.ascontiguousarray(np.stack(predictions), dtype=np.float32),
        "target_p_mean": np.ascontiguousarray(np.stack(means), dtype=np.float32),
        "target_p_standard_deviation": np.ascontiguousarray(np.stack(sigmas), dtype=np.float32),
        "target_used_innovation": np.ascontiguousarray(np.stack(innovations), dtype=np.float32),
        "target_state_after": np.ascontiguousarray(np.stack(afters), dtype=np.float32),
        "final_first_half": final_first,
    }
    return BranchResult(
        attempt_index=attempt_index,
        role="exact_baseline_replay" if attempt_index == 0 else "fresh_target_suffix",
        stream_seed=stream_seed,
        arrays=arrays,
        decoded_target=decoded_target,
        transition_records=transition_records,
        fresh_full_2b_draw_count=fresh_draw_count,
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def _atomic_install_directory_noreplace(source: Path, destination: Path) -> None:
    # Real runs stage as ``destination.parent/.name.staging-*/bundle``.  This
    # keeps source and destination on one filesystem while isolating all
    # incomplete files from the final namespace.
    if source.parent.parent.resolve() != destination.parent.resolve():
        raise ValueError("atomic publication source is outside the destination staging parent")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-overwrite publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        raise OSError(code, os.strerror(code), str(destination))


def _png_record(path: Path, root: Path) -> dict[str, Any]:
    record = {"relative_path": path.relative_to(root).as_posix()}
    record.update(custom.inspect_png(path, "RGB", (strict.IMAGE_SIZE, strict.IMAGE_SIZE)))
    return record


def _npz_record(path: Path, arrays: Mapping[str, np.ndarray], root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": strict.sha256_file(path),
        "arrays": {key: _array_record(arrays[key]) for key in sorted(arrays)},
    }


def _canonical_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trace-dir",
        str(args.trace_dir),
        "--target-slot",
        str(args.target_slot),
        "--rollback-sampling-step",
        str(args.rollback_sampling_step),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]
    if args.expect_class_id is not None:
        command.extend(["--expect-class-id", str(args.expect_class_id)])
    if getattr(args, "pilot_lock", None) is not None:
        command.extend(["--pilot-lock", str(args.pilot_lock)])
    return command


def build_manifest(
    args: argparse.Namespace,
    trace: SavedTrace,
    *,
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    vae_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    internal_t, transition_count, stochastic_count = _validate_rollback(
        trace, args.rollback_sampling_step
    )
    target_class = _validate_target(trace, args.target_slot, args.expect_class_id)
    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    seeds = [
        _branch_seed(
            trace.identity_sha256,
            global_seed=trace.seed,
            rollback_sampling_step=args.rollback_sampling_step,
            target_slot=args.target_slot,
            attempt_index=index,
        )
        for index in range(1, BRANCH_COUNT)
    ]
    if len(set(seeds)) != FRESH_ATTEMPTS:
        raise AssertionError("fresh branch seed collision")
    runner = Path(__file__).resolve()
    input_manifest = trace.root / custom_trace.MANIFEST_NAME
    input_completion = trace.root / custom_trace.COMPLETION_NAME
    input_archive = trace.root / custom_trace.TRACE_NAME
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "runner": RUNNER_NAME,
        "role": "EXPLORATORY_INTERNAL_TRIGGER_SELECTED_SUFFIX_REPAIRABILITY",
        "posthoc_exploratory": True,
        "online_detector_execution": False,
        "method_claim_eligible": False,
        "quality_scores_or_labels_used_by_runner": False,
        "FID_Inception_DINO_CLIP_or_embeddings_used": False,
        "attempt_ranking_or_selection": False,
        "best_of_n": False,
        "target": {
            "slot": args.target_slot,
            "class_id": target_class,
            "global_seed": trace.seed,
            "selection_provenance": "supplied before this runner; runner does not inspect scores or labels",
        },
        "pilot_binding": pilot_binding,
        "rollback": {
            "sampling_step_index_zero_based": args.rollback_sampling_step,
            "restored_state_time": "before this sampling-step transition",
            "trigger_decision_time": (
                "external to this generic runner and may be later than the restored state; "
                "the v1.2 pilot observes the complete step149 innovation before deciding, "
                "then restores this saved pre-transition state"
            ),
            "restored_tensor": "saved state_before[:, sampling_step]",
            "internal_timestep": internal_t,
            "suffix_transition_count_including_t0": transition_count,
            "stochastic_transition_count": stochastic_count,
            "terminal_t0_innovation_consumed_then_zero_multiplied": True,
        },
        "branches": {
            "order": [_branch_name(index) for index in range(BRANCH_COUNT)],
            "attempt_0": "exact replay using all saved first-half innovations including saved t=0 innovation",
            "attempts_1_to_4": "fresh independent full-2B branch draw per suffix transition; replace target first-half innovation only",
            "fresh_stream_seeds": [
                {"attempt_index": index, "seed": seeds[index - 1]}
                for index in range(1, BRANCH_COUNT)
            ],
            "all_outputs_retained": True,
            "selected_attempt": None,
        },
        "sampler": {
            "model": strict.MODEL_NAME,
            "sampler": "official ancestral DDPM-250 suffix",
            "cfg_scale": strict.CFG_SCALE,
            "clip_denoised": False,
            "classes_ordered": list(trace.classes),
            "first_half_batch_size": len(trace.classes),
            "model_batch_size": 2 * len(trace.classes),
            "non_target_first_half_uses_saved_innovations_and_must_replay_exactly": True,
            "forward_with_cfg_reconstructs_second_half_from_first_half_each_call": True,
        },
        "input_trace": {
            "root": str(trace.root),
            "identity_sha256": trace.identity_sha256,
            "manifest_sha256": strict.sha256_file(input_manifest),
            "completion_sha256": strict.sha256_file(input_completion),
            "trace_npz_sha256": strict.sha256_file(input_archive),
            "strict_completed_bundle_and_exact_replay_validated": True,
            "baseline_target_png": custom.individual_relative_paths(trace.classes)[args.target_slot],
        },
        "lineage": {
            "dit_source": source,
            "checkpoint": checkpoint,
            "vae_snapshot": vae_snapshot,
            "custom_trace_identity_source": {
                "dit_source": trace.identity["source"],
                "checkpoint": trace.identity["checkpoint"],
                "vae_snapshot": trace.identity["vae_snapshot"],
            },
        },
        "rng": {
            "namespace": RNG_NAMESPACE,
            "attempt_0_fresh_draw_count": 0,
            "fresh_attempt_full_2b_draws_each_including_t0": transition_count,
            "branch_local_generators_only": True,
            "global_rng_must_not_change_during_branch": True,
        },
        "statistical_scope": {
            "conditional_Ville_bound_applicable": False,
            "TV_bound_applicable": False,
            "retry_cost_bound_applicable": False,
            "repairability_only": True,
        },
        "runner_source": {"path": str(runner), "sha256": strict.sha256_file(runner)},
        "dependencies": strict.dependency_identity(),
        "canonical_command": _canonical_command(args),
        "outputs": {
            "native_target_png_per_branch": True,
            "mechanical_npz_and_json_per_branch": True,
            "scores_labels_features": None,
            "selected_attempt": None,
            "atomic_install": True,
            "no_overwrite": True,
        },
    }
    manifest["identity_sha256"] = _self_hash(manifest, "identity_sha256")
    return manifest


def save_branch(
    result: BranchResult,
    root: Path,
    *,
    target_slot: int,
    target_class: int,
    save_image: Any,
) -> dict[str, Any]:
    branch = _branch_name(result.attempt_index)
    directory = root / "branches" / branch
    directory.mkdir(parents=True, exist_ok=False)
    png_path = directory / f"slot{target_slot:02d}_class{target_class:04d}.png"
    save_image(
        result.decoded_target,
        png_path,
        nrow=1,
        padding=0,
        normalize=True,
        value_range=(-1, 1),
    )
    trace_path = directory / "trace.npz"
    _atomic_npz(trace_path, result.arrays)
    branch_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "branch": branch,
        "attempt_index": result.attempt_index,
        "role": result.role,
        "stream_seed": result.stream_seed,
        "fresh_full_2b_draw_count": result.fresh_full_2b_draw_count,
        "transition_count": len(result.transition_records),
        "transitions": result.transition_records,
        "target_png": _png_record(png_path, root),
        "trace_npz": _npz_record(trace_path, result.arrays, root),
    }
    branch_json["payload_sha256"] = _self_hash(branch_json, "payload_sha256")
    strict.atomic_json_dump(branch_json, directory / "branch.json")
    return {
        "branch": branch,
        "attempt_index": result.attempt_index,
        "role": result.role,
        "stream_seed": result.stream_seed,
        "branch_json_relative_path": (directory / "branch.json").relative_to(root).as_posix(),
        "branch_json_sha256": strict.sha256_file(directory / "branch.json"),
        "branch_payload_sha256": branch_json["payload_sha256"],
        "target_png": branch_json["target_png"],
        "trace_npz_sha256": branch_json["trace_npz"]["sha256"],
    }


def _load_npz_strict(path: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink() or strict.sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"branch trace file identity failed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if set(arrays) != set(TRACE_DTYPES) or set(record.get("arrays", {})) != set(TRACE_DTYPES):
        raise RuntimeError("branch trace member set changed")
    for key, array in arrays.items():
        if array.dtype != TRACE_DTYPES[key] or _array_record(array) != record["arrays"][key]:
            raise RuntimeError(f"branch trace array changed: {key}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"branch trace array is non-finite: {key}")
    return arrays


def validate_bundle(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    trace: SavedTrace,
    require_completion: bool,
) -> dict[str, Any]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("output bundle contains a symlink")
    stored_manifest = _load_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    if stored_manifest != manifest:
        raise RuntimeError("stored output manifest differs from requested identity")
    if strict.sha256_file(Path(__file__).resolve()) != manifest["runner_source"]["sha256"]:
        raise RuntimeError("current runner source differs from bundle identity")
    snapshot = root / SNAPSHOT_NAME
    if strict.sha256_file(snapshot) != manifest["runner_source"]["sha256"]:
        raise RuntimeError("runner source snapshot differs")
    results = _load_self_hashed_json(root / RESULTS_NAME, "payload_sha256")
    fixed_results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "branch_count": BRANCH_COUNT,
        "fresh_attempt_count": FRESH_ATTEMPTS,
        "selection_performed": False,
        "selected_attempt": None,
        "quality_scores_or_features_computed": False,
    }
    if any(results.get(key) != value for key, value in fixed_results.items()):
        raise RuntimeError("results scope or identity changed")
    branches = results.get("branches")
    if not isinstance(branches, list) or [row.get("branch") for row in branches] != [
        _branch_name(index) for index in range(BRANCH_COUNT)
    ]:
        raise RuntimeError("results branch order changed")

    step = int(manifest["rollback"]["sampling_step_index_zero_based"])
    internal_t, transition_count, _ = _validate_rollback(trace, step)
    target_slot = int(manifest["target"]["slot"])
    target_class = int(manifest["target"]["class_id"])
    non_target = [slot for slot in range(len(trace.classes)) if slot != target_slot]
    expected_suffix_axis = np.arange(internal_t, -1, -1, dtype=np.int16)
    expected_files = {
        (root / MANIFEST_NAME).resolve(),
        (root / RESULTS_NAME).resolve(),
        snapshot.resolve(),
    }
    if require_completion:
        expected_files.add((root / COMPLETION_NAME).resolve())
    baseline_relative = custom.individual_relative_paths(trace.classes)[target_slot]
    baseline_png = trace.root / baseline_relative
    baseline_png_record = custom.inspect_png(
        baseline_png, "RGB", (strict.IMAGE_SIZE, strict.IMAGE_SIZE)
    )
    for attempt_index, summary in enumerate(branches):
        branch = _branch_name(attempt_index)
        directory = root / "branches" / branch
        branch_path = directory / "branch.json"
        png_path = directory / f"slot{target_slot:02d}_class{target_class:04d}.png"
        npz_path = directory / "trace.npz"
        expected_files.update({branch_path.resolve(), png_path.resolve(), npz_path.resolve()})
        if strict.sha256_file(branch_path) != summary.get("branch_json_sha256"):
            raise RuntimeError(f"branch JSON changed: {branch}")
        branch_json = _load_self_hashed_json(branch_path, "payload_sha256")
        fixed_branch = {
            "branch": branch,
            "attempt_index": attempt_index,
            "role": "exact_baseline_replay" if attempt_index == 0 else "fresh_target_suffix",
            "stream_seed": (
                None
                if attempt_index == 0
                else _branch_seed(
                    trace.identity_sha256,
                    global_seed=trace.seed,
                    rollback_sampling_step=step,
                    target_slot=target_slot,
                    attempt_index=attempt_index,
                )
            ),
            "fresh_full_2b_draw_count": 0 if attempt_index == 0 else transition_count,
            "transition_count": transition_count,
        }
        if any(branch_json.get(key) != value for key, value in fixed_branch.items()):
            raise RuntimeError(f"branch accounting changed: {branch}")
        if branch_json["payload_sha256"] != summary.get("branch_payload_sha256"):
            raise RuntimeError(f"branch payload link changed: {branch}")
        transitions = branch_json.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != transition_count:
            raise RuntimeError(f"branch transition metadata is incomplete: {branch}")
        arrays = _load_npz_strict(npz_path, branch_json.get("trace_npz", {}))
        if not np.array_equal(arrays["internal_timestep"], expected_suffix_axis):
            raise RuntimeError(f"suffix timestep axis changed: {branch}")
        expected_shape = (transition_count, strict.LATENT_CHANNELS, strict.LATENT_SIZE, strict.LATENT_SIZE)
        if any(arrays[key].shape != expected_shape for key in TRACE_DTYPES if key not in {"internal_timestep", "final_first_half"}):
            raise RuntimeError(f"suffix tensor shape changed: {branch}")
        if arrays["final_first_half"].shape != (
            len(trace.classes), strict.LATENT_CHANNELS, strict.LATENT_SIZE, strict.LATENT_SIZE
        ):
            raise RuntimeError(f"final first-half shape changed: {branch}")
        if not np.array_equal(
            arrays["target_state_before"][0], trace.arrays["state_before"][target_slot, step]
        ):
            raise RuntimeError(f"rollback state changed: {branch}")
        for ordinal in range(transition_count):
            row = step + ordinal
            internal = internal_t - ordinal
            used = arrays["target_used_innovation"][ordinal]
            saved = trace.arrays["transition_innovation"][target_slot, row]
            transition = transitions[ordinal]
            fixed_transition = {
                "suffix_ordinal": ordinal,
                "sampling_step": row,
                "internal_timestep": internal,
                "stochastic_effect": internal > 0,
                "innovation_source": "saved_baseline" if attempt_index == 0 else "fresh_branch",
                "full_2b_fresh_draw_ordinal": None if attempt_index == 0 else ordinal + 1,
                "t0_innovation_consumed_then_zero_multiplied": internal == 0,
            }
            if any(transition.get(key) != value for key, value in fixed_transition.items()):
                raise RuntimeError(f"transition metadata changed: {branch}/{row}")
            full_proposal_hash = transition.get("full_2b_fresh_proposal_raw_sha256")
            if attempt_index == 0:
                if full_proposal_hash is not None:
                    raise RuntimeError(f"replay branch claims a fresh proposal: {branch}/{row}")
            elif not isinstance(full_proposal_hash, str) or len(full_proposal_hash) != 64:
                raise RuntimeError(f"fresh proposal hash is invalid: {branch}/{row}")
            expected_after = (
                arrays["target_p_mean"][ordinal]
                + np.float32(internal > 0)
                * arrays["target_p_standard_deviation"][ordinal]
                * used
            )
            if not np.array_equal(expected_after, arrays["target_state_after"][ordinal]):
                maximum = float(np.max(np.abs(expected_after.astype(np.float64) - arrays["target_state_after"][ordinal])))
                if maximum > 2e-6:
                    raise RuntimeError(f"transition arithmetic changed: {branch}/{row}/{maximum}")
            next_state = (
                arrays["target_state_before"][ordinal + 1]
                if ordinal + 1 < transition_count
                else arrays["final_first_half"][target_slot]
            )
            if not np.array_equal(next_state, arrays["target_state_after"][ordinal]):
                raise RuntimeError(f"target chain is discontinuous: {branch}/{row}")
            expected_hashes = {
                "target_state_before_raw_sha256": _raw_sha(arrays["target_state_before"][ordinal]),
                "target_pred_xstart_raw_sha256": _raw_sha(arrays["target_pred_xstart"][ordinal]),
                "target_p_mean_raw_sha256": _raw_sha(arrays["target_p_mean"][ordinal]),
                "target_p_standard_deviation_raw_sha256": _raw_sha(
                    arrays["target_p_standard_deviation"][ordinal]
                ),
                "target_used_innovation_raw_sha256": _raw_sha(used),
                "target_state_after_raw_sha256": _raw_sha(arrays["target_state_after"][ordinal]),
                "saved_target_innovation_raw_sha256": _raw_sha(saved),
            }
            if any(transition.get(key) != value for key, value in expected_hashes.items()):
                raise RuntimeError(f"transition tensor binding changed: {branch}/{row}")
            before_rng = transition.get("global_rng_before_model_sha256")
            after_rng = transition.get("global_rng_after_model_sha256")
            if (
                not isinstance(before_rng, str)
                or len(before_rng) != 64
                or after_rng != before_rng
            ):
                raise RuntimeError(f"model RNG record changed: {branch}/{row}")
            if attempt_index == 0:
                for name, baseline in (
                    ("target_state_before", trace.arrays["state_before"][target_slot, row]),
                    ("target_pred_xstart", trace.arrays["pred_xstart"][target_slot, row]),
                    ("target_p_mean", trace.arrays["p_mean"][target_slot, row]),
                    ("target_p_standard_deviation", trace.arrays["p_standard_deviation"][target_slot, row]),
                    ("target_used_innovation", saved),
                ):
                    if not np.array_equal(arrays[name][ordinal], baseline):
                        raise RuntimeError(f"attempt 0 differs from saved {name}: step {row}")
        if not np.array_equal(arrays["final_first_half"][non_target], trace.arrays["final_latents"][non_target]):
            raise RuntimeError(f"non-target final latent changed: {branch}")
        if attempt_index == 0 and not np.array_equal(arrays["final_first_half"], trace.arrays["final_latents"]):
            raise RuntimeError("attempt 0 final latent differs from baseline")
        observed_png = _png_record(png_path, root)
        if observed_png != branch_json.get("target_png") or observed_png != summary.get("target_png"):
            raise RuntimeError(f"target PNG record changed: {branch}")
        if attempt_index == 0 and observed_png["pixel_sha256"] != baseline_png_record["pixel_sha256"]:
            raise RuntimeError("attempt 0 target PNG is not pixel-identical to saved baseline")

    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("output file set changed")
    expected_dirs = {
        (root / "branches").resolve(),
        *((root / "branches" / _branch_name(index)).resolve() for index in range(BRANCH_COUNT)),
    }
    actual_dirs = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise RuntimeError("output directory set changed")
    if results.get("branches_sha256") != _sha256_json(branches):
        raise RuntimeError("branch summary aggregate changed")
    if require_completion:
        completion = _load_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
        fixed_completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": strict.sha256_file(root / MANIFEST_NAME),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": strict.sha256_file(root / RESULTS_NAME),
            "branches_sha256": results["branches_sha256"],
            "branch_count": BRANCH_COUNT,
        }
        if any(completion.get(key) != value for key, value in fixed_completion.items()):
            raise RuntimeError("completion binding changed")
    return results


def run_real(
    args: argparse.Namespace,
    trace: SavedTrace,
    *,
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    vae_snapshot: Mapping[str, Any],
) -> None:
    if args.outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.outdir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DiT suffix experiment")
    strict.ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    manifest = build_manifest(
        args, trace, source=source, checkpoint=checkpoint, vae_snapshot=vae_snapshot
    )
    target_class = trace.classes[args.target_slot]
    started = time.time()
    args.outdir.parent.mkdir(parents=True, exist_ok=True)

    def execute(staging: Path) -> list[dict[str, Any]]:
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
        torch.manual_seed(trace.seed)
        prior_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            device = torch.device("cuda")
            model = DiT_models[strict.MODEL_NAME](
                input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
            ).to(device)
            model.load_state_dict(find_model(str(args.checkpoint)))
            model.eval()
            diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
            timestep_map = np.ascontiguousarray(diffusion.timestep_map, dtype=np.int64)
            if timestep_map.shape != (strict.NUM_SAMPLING_STEPS,):
                raise RuntimeError("runtime timestep map shape changed")
            internal_axis = np.arange(
                strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int64
            )
            runtime_alpha = np.ascontiguousarray(
                np.asarray(diffusion.alphas_cumprod, dtype=np.float64)[internal_axis]
            )
            if not np.array_equal(runtime_alpha, trace.arrays["alpha_bar"]):
                raise RuntimeError("runtime DDPM alpha schedule differs from saved trace")
            vae = AutoencoderKL.from_pretrained(
                str(args.vae_snapshot), local_files_only=True, use_safetensors=True
            ).to(device)
            records = []
            for attempt_index in range(BRANCH_COUNT):
                stream_seed = (
                    None
                    if attempt_index == 0
                    else _branch_seed(
                        trace.identity_sha256,
                        global_seed=trace.seed,
                        rollback_sampling_step=args.rollback_sampling_step,
                        target_slot=args.target_slot,
                        attempt_index=attempt_index,
                    )
                )
                result = run_branch(
                    diffusion,
                    model,
                    vae,
                    trace,
                    rollback_sampling_step=args.rollback_sampling_step,
                    target_slot=args.target_slot,
                    attempt_index=attempt_index,
                    stream_seed=stream_seed,
                    timestep_map=timestep_map,
                    device=device,
                )
                records.append(
                    save_branch(
                        result,
                        staging,
                        target_slot=args.target_slot,
                        target_class=target_class,
                        save_image=save_image,
                    )
                )
                print(f"saved {_branch_name(attempt_index)} ({attempt_index + 1}/{BRANCH_COUNT})", flush=True)
            torch.cuda.synchronize()
            return records
        finally:
            torch.set_grad_enabled(prior_grad)

    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        strict.atomic_json_dump(manifest, staging / MANIFEST_NAME)
        _atomic_copy(Path(__file__).resolve(), staging / SNAPSHOT_NAME)
        branches = _with_upstream_imports(args.dit_root, lambda: execute(staging))
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "branch_count": BRANCH_COUNT,
            "fresh_attempt_count": FRESH_ATTEMPTS,
            "selection_performed": False,
            "selected_attempt": None,
            "quality_scores_or_features_computed": False,
            "branches": branches,
            "branches_sha256": _sha256_json(branches),
            "wall_seconds_before_validation": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": strict.dependency_identity(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
            },
        }
        results["payload_sha256"] = _self_hash(results, "payload_sha256")
        strict.atomic_json_dump(results, staging / RESULTS_NAME)
        validate_bundle(staging, manifest=manifest, trace=trace, require_completion=False)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": strict.sha256_file(staging / RESULTS_NAME),
            "branches_sha256": results["branches_sha256"],
            "branch_count": BRANCH_COUNT,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _self_hash(completion, "payload_sha256")
        strict.atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_bundle(staging, manifest=manifest, trace=trace, require_completion=True)
        _atomic_install_directory_noreplace(staging, args.outdir)
    validate_bundle(args.outdir, manifest=manifest, trace=trace, require_completion=True)
    print(json.dumps({"status": "complete", "outdir": str(args.outdir), "branches": BRANCH_COUNT}, indent=2))


class _ToyModel(torch.nn.Module):
    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, *, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        del y, cfg_scale
        half = x[: len(x) // 2]
        value = 0.1 * half + 0.01 * t[: len(half)].float().view(-1, 1, 1, 1)
        return torch.cat([value, value], dim=0)


class _ToyDiffusion:
    timestep_map = np.arange(4, dtype=np.int64)

    def p_mean_variance(
        self,
        model_fn: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool,
        model_kwargs: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        if clip_denoised:
            raise AssertionError("toy clip contract changed")
        pred = model_fn(x, t, **model_kwargs)
        mean = 0.8 * x + 0.1 * pred
        sigma = torch.full_like(x, 0.2)
        return {"mean": mean, "pred_xstart": pred, "log_variance": torch.log(sigma * sigma)}


class _ToyVAE:
    class Decoded:
        def __init__(self, sample: torch.Tensor):
            self.sample = sample

    def decode(self, value: torch.Tensor) -> "_ToyVAE.Decoded":
        return self.Decoded(value.repeat(1, 3, 1, 1))


def _toy_trace() -> SavedTrace:
    classes = (7, 11, 19)
    batch = len(classes)
    steps = 4
    generator = torch.Generator(device="cpu").manual_seed(41)
    diffusion = _ToyDiffusion()
    model = _ToyModel()
    first = torch.randn((batch, 1, 2, 2), generator=generator)
    x = torch.cat([first, first], dim=0)
    y = torch.cat([torch.tensor(classes), torch.full((batch,), strict.NULL_CLASS_ID)])
    states, preds, means, sigmas, noises = [], [], [], [], []
    for internal_t in range(steps - 1, -1, -1):
        out = diffusion.p_mean_variance(
            model.forward_with_cfg,
            x,
            torch.full((2 * batch,), internal_t, dtype=torch.long),
            clip_denoised=False,
            model_kwargs={"y": y, "cfg_scale": strict.CFG_SCALE},
        )
        sigma = torch.exp(0.5 * out["log_variance"])
        noise = torch.randn((2 * batch, 1, 2, 2), generator=generator)
        states.append(_tensor_numpy(x[:batch]))
        preds.append(_tensor_numpy(out["pred_xstart"][:batch]))
        means.append(_tensor_numpy(out["mean"][:batch]))
        sigmas.append(_tensor_numpy(sigma[:batch]))
        noises.append(_tensor_numpy(noise[:batch]))
        x = out["mean"] + float(internal_t > 0) * sigma * noise
    arrays = {
        "state_before": np.stack(states, axis=1),
        "pred_xstart": np.stack(preds, axis=1),
        "p_mean": np.stack(means, axis=1),
        "p_standard_deviation": np.stack(sigmas, axis=1),
        "transition_innovation": np.stack(noises, axis=1),
        "final_latents": _tensor_numpy(x[:batch]),
        "internal_timestep": np.arange(steps - 1, -1, -1, dtype=np.int16),
    }
    return SavedTrace(
        root=Path("/toy"),
        manifest={"identity_sha256": "1" * 64},
        identity={},
        classes=classes,
        seed=41,
        arrays=arrays,
    )


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must not initialize CUDA")
    trace = _toy_trace()
    # Production has 250 steps.  Temporarily substitute the toy horizon only
    # at the direct branch unit-test boundary.
    original_steps = strict.NUM_SAMPLING_STEPS
    original_channels = strict.LATENT_CHANNELS
    original_size = strict.LATENT_SIZE
    original_image_size = strict.IMAGE_SIZE
    strict.NUM_SAMPLING_STEPS = 4
    strict.LATENT_CHANNELS = 1
    strict.LATENT_SIZE = 2
    strict.IMAGE_SIZE = 2
    try:
        replay = run_branch(
            _ToyDiffusion(),
            _ToyModel(),
            _ToyVAE(),
            trace,
            rollback_sampling_step=1,
            target_slot=1,
            attempt_index=0,
            stream_seed=None,
            timestep_map=np.arange(4, dtype=np.int64),
            device=torch.device("cpu"),
        )
        seed = _branch_seed(
            trace.identity_sha256,
            global_seed=trace.seed,
            rollback_sampling_step=1,
            target_slot=1,
            attempt_index=1,
        )
        fresh = run_branch(
            _ToyDiffusion(),
            _ToyModel(),
            _ToyVAE(),
            trace,
            rollback_sampling_step=1,
            target_slot=1,
            attempt_index=1,
            stream_seed=seed,
            timestep_map=np.arange(4, dtype=np.int64),
            device=torch.device("cpu"),
        )
    finally:
        strict.NUM_SAMPLING_STEPS = original_steps
        strict.LATENT_CHANNELS = original_channels
        strict.LATENT_SIZE = original_size
        strict.IMAGE_SIZE = original_image_size
    if not np.array_equal(replay.arrays["final_first_half"], trace.arrays["final_latents"]):
        raise AssertionError("toy replay was not exact")
    if replay.fresh_full_2b_draw_count != 0 or fresh.fresh_full_2b_draw_count != 3:
        raise AssertionError("toy draw accounting including t=0 failed")
    if not np.array_equal(
        fresh.arrays["final_first_half"][[0, 2]], trace.arrays["final_latents"][[0, 2]]
    ):
        raise AssertionError("toy fresh branch changed non-target slots")
    if np.array_equal(fresh.arrays["final_first_half"][1], trace.arrays["final_latents"][1]):
        raise AssertionError("toy fresh suffix did not change target")
    if not np.array_equal(
        fresh.arrays["target_state_after"][-1], fresh.arrays["target_p_mean"][-1]
    ):
        raise AssertionError("toy t=0 innovation was not zero-multiplied")
    seeds = {
        _branch_seed(
            trace.identity_sha256,
            global_seed=trace.seed,
            rollback_sampling_step=149,
            target_slot=1,
            attempt_index=index,
        )
        for index in range(1, BRANCH_COUNT)
    }
    if len(seeds) != FRESH_ATTEMPTS:
        raise AssertionError("fresh branch seed collision")
    with tempfile.TemporaryDirectory(prefix="dit-v22-suffix-selftest-") as temporary:
        root = Path(temporary)
        npz = root / "trace.npz"
        _atomic_npz(npz, fresh.arrays)
        record = _npz_record(npz, fresh.arrays, root)
        loaded = _load_npz_strict(npz, record)
        if any(not np.array_equal(loaded[key], fresh.arrays[key]) for key in loaded):
            raise AssertionError("strict branch NPZ roundtrip failed")
        staging_parent = root / ".destination.staging-test"
        staging_parent.mkdir()
        source = staging_parent / "bundle"
        source.mkdir()
        (source / "marker").write_bytes(b"source")
        destination = root / "destination"
        _atomic_install_directory_noreplace(source, destination)
        if source.exists() or (destination / "marker").read_bytes() != b"source":
            raise AssertionError("atomic directory publication failed")
    print("self-test passed: exact replay, independent suffix, t=0 draw, NPZ, atomic install")


def dry_run(args: argparse.Namespace) -> None:
    custom.validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    # A dry run skips CUDA/model execution, not lineage verification.  The
    # full checkpoint digest must equal the digest authenticated by the trace.
    checkpoint = strict.validate_checkpoint(args.checkpoint)
    vae_snapshot = strict.validate_vae_snapshot(args.vae_snapshot)
    trace = load_saved_trace(
        args.trace_dir,
        source=source,
        checkpoint=checkpoint,
        vae_snapshot=vae_snapshot,
    )
    target_class = _validate_target(trace, args.target_slot, args.expect_class_id)
    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    internal_t, transition_count, stochastic_count = _validate_rollback(
        trace, args.rollback_sampling_step
    )
    if args.outdir.exists():
        raise RuntimeError(f"dry-run refuses an existing output path: {args.outdir}")
    print(
        json.dumps(
            {
                "status": "dry-run",
                "runner": RUNNER_NAME,
                "trace_dir": str(trace.root),
                "trace_identity_sha256": trace.identity_sha256,
                "classes_ordered": list(trace.classes),
                "global_seed": trace.seed,
                "target_slot": args.target_slot,
                "target_class_id": target_class,
                "pilot_binding": pilot_binding,
                "rollback_sampling_step": args.rollback_sampling_step,
                "rollback_internal_timestep": internal_t,
                "suffix_transition_count_including_t0": transition_count,
                "stochastic_transition_count": stochastic_count,
                "fresh_stream_seeds": [
                    _branch_seed(
                        trace.identity_sha256,
                        global_seed=trace.seed,
                        rollback_sampling_step=args.rollback_sampling_step,
                        target_slot=args.target_slot,
                        attempt_index=index,
                    )
                    for index in range(1, BRANCH_COUNT)
                ],
                "cuda_available": torch.cuda.is_available(),
                "outdir": str(args.outdir),
                "canonical_command": _canonical_command(args),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--target-slot", type=int)
    parser.add_argument(
        "--expect-class-id",
        "--target-class-id",
        dest="expect_class_id",
        type=int,
        help="optional fail-closed class-ID assertion for --target-slot",
    )
    parser.add_argument("--rollback-sampling-step", type=int)
    parser.add_argument(
        "--pilot-lock",
        type=Path,
        help="optional frozen v1.2 pilot lock; when supplied, target and rollback must be selected",
    )
    parser.add_argument("--dit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    required = (
        "trace_dir",
        "target_slot",
        "rollback_sampling_step",
        "dit_root",
        "checkpoint",
        "vae_snapshot",
        "outdir",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error("required arguments missing: " + ", ".join(missing))
    args.trace_dir = args.trace_dir.expanduser().resolve()
    args.dit_root = args.dit_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.vae_snapshot = args.vae_snapshot.expanduser().resolve()
    args.outdir = args.outdir.expanduser().absolute()
    if args.pilot_lock is not None:
        args.pilot_lock = args.pilot_lock.expanduser().resolve()
    if args.outdir.is_symlink():
        parser.error("--outdir must not be a symlink")
    input_paths = [args.trace_dir, args.dit_root, args.checkpoint, args.vae_snapshot]
    if args.pilot_lock is not None:
        input_paths.append(args.pilot_lock)
    for input_path in input_paths:
        try:
            args.outdir.relative_to(input_path)
        except ValueError:
            pass
        else:
            parser.error("--outdir must not lie inside an input path")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.dry_run:
        dry_run(args)
        return 0
    custom.validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    checkpoint = strict.validate_checkpoint(args.checkpoint)
    vae_snapshot = strict.validate_vae_snapshot(args.vae_snapshot)
    trace = load_saved_trace(
        args.trace_dir,
        source=source,
        checkpoint=checkpoint,
        vae_snapshot=vae_snapshot,
    )
    _validate_target(trace, args.target_slot, args.expect_class_id)
    _validate_rollback(trace, args.rollback_sampling_step)
    manifest = build_manifest(
        args, trace, source=source, checkpoint=checkpoint, vae_snapshot=vae_snapshot
    )
    if args.outdir.exists():
        validate_bundle(args.outdir, manifest=manifest, trace=trace, require_completion=True)
        print(f"validated completed immutable output: {args.outdir}; no GPU run")
        return 0
    run_real(
        args,
        trace,
        source=source,
        checkpoint=checkpoint,
        vae_snapshot=vae_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

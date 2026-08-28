#!/usr/bin/env python3
"""Record a full, baseline-exact DiT custom-batch trajectory.

This is an observation-only companion to ``sample_dit_imagenet256_custom.py``.
It preserves that runner's ordered 1--8 class, one-global-seed, 2B-CFG RNG
contract while recording the first B paths at every ancestral DDPM step.

The saved ``trace.npz`` contains ``state_before``, ``pred_xstart``, ``p_mean``,
``p_standard_deviation``, the raw conditional/unconditional model epsilon and
learned-range outputs, and the *raw* first-B slice of the full 2B
``torch.randn_like`` transition draw.  The t=0 draw is therefore retained even
though the upstream nonzero mask makes it irrelevant to the final state.
``final_latents``, decoded float images, reverse internal timesteps, and the
corresponding alpha-bars are also saved.  A transparent wrapper observes the
single raw model forward already made inside upstream ``forward_with_cfg``; it
does not add a neural evaluation or change the returned tensor.  No score,
selection, rejection, guidance modification, or intervention is performed.

If ``--reference-baseline-dir`` is supplied, it must be a completed output of
the custom baseline runner with matching scientific inputs.  Every resulting
native PNG is then required to be pixel-identical to that reference.  Existing
outputs are never overwritten: an identical completed directory is validated,
while every other pre-existing path is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import torch

try:  # Package and direct CLI imports.
    from . import reproduce_dit_imagenet256 as strict
    from . import sample_dit_imagenet256_custom as custom
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    import sample_dit_imagenet256_custom as custom


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


RUNNER_NAME = "trace_dit_imagenet256_custom_batch"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
TRACE_NAME = "trace.npz"
SOURCE_SNAPSHOTS = {
    "runner_source.py": lambda: Path(__file__).resolve(),
    "custom_baseline_helper.py": lambda: Path(custom.__file__).resolve(),
    "strict_reproduction_helper.py": lambda: Path(strict.__file__).resolve(),
}
STEP_ARRAY_NAMES = (
    "state_before",
    "pred_xstart",
    "p_mean",
    "p_standard_deviation",
    "transition_innovation",
    "conditional_epsilon_raw",
    "unconditional_epsilon_raw",
    "conditional_variance_values_raw",
    "unconditional_variance_values_raw",
)
TRACE_ARRAY_NAMES = (
    *STEP_ARRAY_NAMES,
    "final_latents",
    "decoded_images",
    "internal_timestep",
    "alpha_bar",
)


def _array_raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _trace_shapes(batch_size: int) -> dict[str, tuple[int, ...]]:
    if type(batch_size) is not int or not 1 <= batch_size <= custom.MAX_CLASSES:
        raise ValueError(f"batch_size must lie in [1,{custom.MAX_CLASSES}]")
    latent = (batch_size, strict.LATENT_CHANNELS, strict.LATENT_SIZE, strict.LATENT_SIZE)
    step_latent = (batch_size, strict.NUM_SAMPLING_STEPS, *latent[1:])
    return {
        **{name: step_latent for name in STEP_ARRAY_NAMES},
        "final_latents": latent,
        "decoded_images": (batch_size, 3, strict.IMAGE_SIZE, strict.IMAGE_SIZE),
        "internal_timestep": (strict.NUM_SAMPLING_STEPS,),
        "alpha_bar": (strict.NUM_SAMPLING_STEPS,),
    }


def _trace_dtypes() -> dict[str, np.dtype[Any]]:
    return {
        **{
            name: np.dtype(np.float32)
            for name in (*STEP_ARRAY_NAMES, "final_latents", "decoded_images")
        },
        "internal_timestep": np.dtype(np.int16),
        "alpha_bar": np.dtype(np.float64),
    }


def amplitude_normalized_dirichlet_roughness(
    values: np.ndarray, *, epsilon: float = 1e-12
) -> np.ndarray:
    """Return per-leading-item, channel-averaged spatial Dirichlet roughness.

    ``values`` must end in ``[C,H,W]``.  For each channel, the mean squared
    horizontal and vertical first differences are added and divided by that
    channel's spatial variance.  Constant channels contribute zero.  The
    result is invariant to nonzero per-channel affine amplitude scaling and is
    useful as a precisely defined *metric helper*, not as a validated detector.
    """

    array = np.asarray(values)
    if array.ndim < 3 or array.shape[-3] < 1 or array.shape[-2] < 2 or array.shape[-1] < 2:
        raise ValueError("values must end in a nonempty [C,H,W] with H,W >= 2")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("values must be a finite floating array")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    work = array.astype(np.float64, copy=False)
    centered = work - np.mean(work, axis=(-2, -1), keepdims=True)
    amplitude = np.mean(centered * centered, axis=(-2, -1))
    horizontal = np.mean(np.diff(work, axis=-1) ** 2, axis=(-2, -1))
    vertical = np.mean(np.diff(work, axis=-2) ** 2, axis=(-2, -1))
    energy = horizontal + vertical
    ratio = np.divide(
        energy,
        amplitude,
        out=np.zeros_like(energy),
        where=amplitude > epsilon,
    )
    return np.mean(ratio, axis=-1)


def numpy_replay(
    initial_latent: np.ndarray,
    p_mean: np.ndarray,
    p_standard_deviation: np.ndarray,
    transition_innovation: np.ndarray,
    internal_timestep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the retained first-B path using only saved transition tensors."""

    arrays = (initial_latent, p_mean, p_standard_deviation, transition_innovation)
    if any(value.dtype != np.dtype(np.float32) for value in arrays):
        raise ValueError("replay latent arrays must all be float32")
    if p_mean.shape != p_standard_deviation.shape or p_mean.shape != transition_innovation.shape:
        raise ValueError("mean, standard deviation, and innovation shapes must match")
    if p_mean.ndim != 5 or initial_latent.shape != (p_mean.shape[0], *p_mean.shape[2:]):
        raise ValueError("replay expects initial [B,C,H,W] and transitions [B,S,C,H,W]")
    if internal_timestep.dtype != np.dtype(np.int16) or internal_timestep.shape != (p_mean.shape[1],):
        raise ValueError("internal_timestep must be int16 [S]")
    if np.any(p_standard_deviation <= 0) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("replay arrays must be finite and standard deviations positive")
    states = np.empty_like(p_mean)
    current = np.ascontiguousarray(initial_latent)
    for step, internal_t in enumerate(internal_timestep.tolist()):
        states[:, step] = current
        current = np.ascontiguousarray(
            p_mean[:, step]
            + np.float32(internal_t > 0)
            * p_standard_deviation[:, step]
            * transition_innovation[:, step],
            dtype=np.float32,
        )
    return states, current


def _trace_array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    if set(arrays) != set(TRACE_ARRAY_NAMES):
        raise RuntimeError("trace array name set changed")
    records: dict[str, dict[str, Any]] = {}
    for name in TRACE_ARRAY_NAMES:
        array = arrays[name]
        records[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "raw_sha256": _array_raw_sha256(array),
        }
    return records


def validate_trace_arrays(arrays: Mapping[str, np.ndarray], batch_size: int) -> None:
    shapes = _trace_shapes(batch_size)
    dtypes = _trace_dtypes()
    if set(arrays) != set(TRACE_ARRAY_NAMES):
        raise RuntimeError(f"trace arrays differ from locked set: {sorted(arrays)}")
    for name in TRACE_ARRAY_NAMES:
        value = arrays[name]
        if value.shape != shapes[name] or value.dtype != dtypes[name]:
            raise RuntimeError(
                f"invalid {name}: shape={value.shape}, dtype={value.dtype}; "
                f"expected shape={shapes[name]}, dtype={dtypes[name]}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"trace array contains non-finite values: {name}")
    expected_t = np.arange(
        strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16
    )
    if not np.array_equal(arrays["internal_timestep"], expected_t):
        raise RuntimeError("internal timestep axis is not 249..0")
    if np.any(arrays["alpha_bar"] <= 0) or np.any(arrays["alpha_bar"] > 1):
        raise RuntimeError("alpha_bar values must lie in (0,1]")
    if np.any(arrays["p_standard_deviation"] <= 0):
        raise RuntimeError("p_standard_deviation must be strictly positive")
    replay_states, replay_final = numpy_replay(
        arrays["state_before"][:, 0],
        arrays["p_mean"],
        arrays["p_standard_deviation"],
        arrays["transition_innovation"],
        arrays["internal_timestep"],
    )
    if not np.array_equal(replay_states, arrays["state_before"]):
        error = float(np.max(np.abs(replay_states.astype(np.float64) - arrays["state_before"])))
        raise RuntimeError(f"NumPy path replay differs from state_before; max_abs={error}")
    if not np.array_equal(replay_final, arrays["final_latents"]):
        error = float(np.max(np.abs(replay_final.astype(np.float64) - arrays["final_latents"])))
        raise RuntimeError(f"NumPy path replay differs from final_latents; max_abs={error}")
    if not np.array_equal(arrays["final_latents"], arrays["p_mean"][:, -1]):
        raise RuntimeError("t=0 final latent is not exactly p_mean")
    if not np.array_equal(arrays["final_latents"], arrays["pred_xstart"][:, -1]):
        raise RuntimeError("t=0 final latent is not exactly pred_xstart")


def _load_trace(path: Path, records: Mapping[str, Any], batch_size: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(TRACE_ARRAY_NAMES):
                raise RuntimeError("trace.npz member set changed")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in TRACE_ARRAY_NAMES}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load trace archive: {path}") from exc
    validate_trace_arrays(arrays, batch_size)
    if _trace_array_records(arrays) != records:
        raise RuntimeError("trace array metadata or raw hashes changed")
    return arrays


def _canonical_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--classes",
        ",".join(map(str, args.classes)),
        "--seed",
        str(args.seed),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]
    if args.reference_baseline_dir is not None:
        command += ["--reference-baseline-dir", str(args.reference_baseline_dir)]
    return command


def build_identity(args: argparse.Namespace) -> dict[str, Any]:
    custom.validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    checkpoint = strict.validate_checkpoint(args.checkpoint)
    vae = strict.validate_vae_snapshot(args.vae_snapshot)
    command = _canonical_command(args)
    return {
        "schema": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "runner_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": strict.sha256_file(Path(__file__).resolve()),
        },
        "custom_baseline_helper": {
            "path": str(Path(custom.__file__).resolve()),
            "sha256": strict.sha256_file(Path(custom.__file__).resolve()),
        },
        "strict_reproduction_helper": {
            "path": str(Path(strict.__file__).resolve()),
            "sha256": strict.sha256_file(Path(strict.__file__).resolve()),
        },
        "observation_only": True,
        "quality_score": None,
        "selection": None,
        "intervention": None,
        "protocol": {
            "model": strict.MODEL_NAME,
            "image_size": strict.IMAGE_SIZE,
            "class_ids_ordered": list(args.classes),
            "batch_size_before_duplication": len(args.classes),
            "sampler_batch_size": 2 * len(args.classes),
            "sampling_steps": strict.NUM_SAMPLING_STEPS,
            "sampler": "ancestral DDPM, manual statement-equivalent p_sample loop",
            "clip_denoised": False,
            "cfg_scale": strict.CFG_SCALE,
            "cfg_epsilon_channels": 3,
            "global_torch_seed": args.seed,
            "full_2B_randn_like_each_transition_including_t0": True,
            "recorded_slice": "first B",
            "transition_innovation": "raw first-B slice of each full 2B draw",
            "pred_xstart_recorded_before_transition_draw": True,
            "raw_cfg_components_observed_from_same_model_forward": True,
            "raw_cfg_component_order": "first B conditional, second B unconditional",
            "raw_epsilon_channels": 4,
            "raw_learned_range_channels": 4,
            "trace_axis_order": "[B, sampling_step, C, H, W]",
            "internal_timestep_order": "249..0",
            "vae": strict.VAE_KIND,
            "vae_scaling_factor": strict.VAE_SCALING_FACTOR,
        },
        "trace_arrays": {
            name: {"shape": list(shape), "dtype": _trace_dtypes()[name].str}
            for name, shape in _trace_shapes(len(args.classes)).items()
        },
        "reference_baseline_dir": (
            str(args.reference_baseline_dir) if args.reference_baseline_dir else None
        ),
        "source": source,
        "checkpoint": checkpoint,
        "vae_snapshot": vae,
        "dependencies": strict.dependency_identity(),
        "canonical_command": command,
        "canonical_command_sha256": strict.sha256_json(command),
    }


def _inspect_reference_baseline(
    root: Path, identity: Mapping[str, Any], classes: Sequence[int], seed: int
) -> dict[str, Any]:
    manifest = strict.load_json(root / custom.MANIFEST_NAME)
    baseline_identity = manifest.get("identity")
    if not isinstance(baseline_identity, dict):
        raise RuntimeError("reference custom baseline lacks an identity object")
    custom.validate_completed_output(root, baseline_identity, classes)
    if baseline_identity.get("runner") != custom.RUNNER_NAME:
        raise RuntimeError("reference is not a completed custom baseline runner output")
    protocol = baseline_identity.get("protocol", {})
    if protocol.get("class_ids_ordered") != list(classes) or protocol.get("global_torch_seed") != seed:
        raise RuntimeError("reference custom baseline class order or global seed differs")
    for key in ("source", "checkpoint", "vae_snapshot"):
        if baseline_identity.get(key) != identity.get(key):
            raise RuntimeError(f"reference custom baseline {key} differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("reference custom baseline output records are missing")
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("reference custom baseline execution record is missing")
    return {
        "png_outputs": {
            record["relative_path"]: record
            for record in outputs
            if isinstance(record, dict) and isinstance(record.get("relative_path"), str)
        },
        "execution": execution,
        "manifest_sha256": strict.sha256_file(root / custom.MANIFEST_NAME),
    }


def _png_records(outdir: Path, classes: Sequence[int]) -> list[dict[str, Any]]:
    specs = custom.expected_output_specs(classes)
    records = []
    for relative in sorted(specs):
        mode, size = specs[relative]
        path = outdir / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or invalid output PNG: {path}")
        record = {"relative_path": relative}
        record.update(custom.inspect_png(path, mode, size))
        records.append(record)
    return records


def _require_reference_pixels(
    png_records: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]
) -> None:
    for record in png_records:
        relative = str(record["relative_path"])
        expected = reference.get(relative)
        if expected is None:
            raise RuntimeError(f"reference baseline lacks PNG: {relative}")
        if record.get("pixel_sha256") != expected.get("pixel_sha256"):
            raise RuntimeError(f"PNG pixels differ from reference custom baseline: {relative}")


def _require_reference_execution(
    execution: Mapping[str, Any], reference_execution: Mapping[str, Any]
) -> None:
    observed_rng = execution.get("rng_state_sha256")
    expected_rng = reference_execution.get("rng_state_sha256")
    if not isinstance(observed_rng, dict) or not isinstance(expected_rng, dict):
        raise RuntimeError("trace or reference lacks RNG-state hashes")
    rng_pairs = {
        "after_manual_seed": "after_manual_seed",
        "after_initial_noise": "after_initial_noise",
        "after_250_full_2B_transition_draws": "after_250_transition_noise_draws",
    }
    for observed_name, reference_name in rng_pairs.items():
        if observed_rng.get(observed_name) != expected_rng.get(reference_name):
            raise RuntimeError(
                f"trace RNG hash differs from reference: {observed_name}/{reference_name}"
            )
    observed_tensors = execution.get("tensor_sha256")
    expected_tensors = reference_execution.get("tensor_sha256")
    required_tensors = {
        "initial_noise_b",
        "final_latents_first_half_b",
        "final_latents_discarded_second_half_b",
        "decoded_samples_b",
    }
    if (
        not isinstance(observed_tensors, dict)
        or not isinstance(expected_tensors, dict)
        or any(observed_tensors.get(name) != expected_tensors.get(name) for name in required_tensors)
    ):
        raise RuntimeError("trace tensor hashes differ from reference custom baseline")


def _collect_payload_records(
    outdir: Path, classes: Sequence[int], arrays: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    allowed_files = {
        TRACE_NAME,
        *SOURCE_SNAPSHOTS,
        MANIFEST_NAME,
        COMPLETION_NAME,
        *custom.expected_output_specs(classes),
    }
    expected_dirs = {"images"}
    unexpected: list[str] = []
    observed_dirs: set[str] = set()
    for path in outdir.rglob("*"):
        relative = path.relative_to(outdir).as_posix()
        if path.is_symlink():
            unexpected.append(relative + " (symlink)")
        elif path.is_dir():
            observed_dirs.add(relative)
        elif not path.is_file() or relative not in allowed_files:
            unexpected.append(relative)
    if unexpected or observed_dirs != expected_dirs:
        raise RuntimeError(
            f"unexpected output layout: files={sorted(unexpected)}, dirs={sorted(observed_dirs)}"
        )
    png = _png_records(outdir, classes)
    array_records = _trace_array_records(arrays)
    trace_path = outdir / TRACE_NAME
    trace_record = {
        "relative_path": TRACE_NAME,
        "bytes": trace_path.stat().st_size,
        "sha256": strict.sha256_file(trace_path),
        "arrays": array_records,
    }
    snapshot_records = []
    for snapshot_name, source_resolver in SOURCE_SNAPSHOTS.items():
        snapshot_path = outdir / snapshot_name
        source_path = source_resolver()
        if (
            not snapshot_path.is_file()
            or snapshot_path.is_symlink()
            or strict.sha256_file(snapshot_path) != strict.sha256_file(source_path)
        ):
            raise RuntimeError(f"source snapshot is missing or differs: {snapshot_name}")
        snapshot_records.append(
            {
                "relative_path": snapshot_name,
                "bytes": snapshot_path.stat().st_size,
                "sha256": strict.sha256_file(snapshot_path),
            }
        )
    return [*png, trace_record, *snapshot_records], array_records


def _snapshot_sources(outdir: Path) -> None:
    for snapshot_name, source_resolver in SOURCE_SNAPSHOTS.items():
        source = source_resolver()
        destination = outdir / snapshot_name
        temporary = destination.with_name(destination.name + ".tmp")
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, destination)


def validate_completed_output(
    outdir: Path, identity: Mapping[str, Any], classes: Sequence[int]
) -> None:
    manifest_path = outdir / MANIFEST_NAME
    completion_path = outdir / COMPLETION_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"existing output is incomplete; refusing overwrite: {outdir}")
    manifest = strict.load_json(manifest_path)
    completion = strict.load_json(completion_path)
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("existing trace manifest is not complete")
    if manifest.get("identity") != identity:
        raise RuntimeError("existing trace identity differs; refusing overwrite")
    identity_hash = strict.sha256_json(identity)
    if manifest.get("identity_sha256") != identity_hash:
        raise RuntimeError("trace manifest identity hash is invalid")
    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, list) or not all(
        isinstance(item, dict) for item in recorded_outputs
    ):
        raise RuntimeError("trace manifest outputs are malformed")
    trace_record = next(
        (item for item in recorded_outputs if item.get("relative_path") == TRACE_NAME),
        None,
    )
    if not isinstance(trace_record, dict):
        raise RuntimeError("trace manifest lacks trace.npz record")
    trace_path = outdir / TRACE_NAME
    if trace_record.get("sha256") != strict.sha256_file(trace_path):
        raise RuntimeError("trace.npz file hash changed")
    arrays = _load_trace(trace_path, trace_record.get("arrays", {}), len(classes))
    records, _ = _collect_payload_records(outdir, classes, arrays)
    if records != manifest.get("outputs"):
        raise RuntimeError("trace output records changed")
    outputs_hash = strict.sha256_json(records)
    if manifest.get("outputs_sha256") != outputs_hash:
        raise RuntimeError("trace manifest output aggregate hash is invalid")
    expected_completion = {
        "schema": SCHEMA_VERSION,
        "identity_sha256": identity_hash,
        "manifest_sha256": strict.sha256_file(manifest_path),
        "outputs_sha256": outputs_hash,
        "output_count": len(records),
    }
    if completion != expected_completion:
        raise RuntimeError("trace completion record is invalid")
    print(f"validated completed custom DiT trace: {outdir}; no sampling run")


def run_trace(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DiT-XL/2 trajectory recorder")
    strict.ensure_single_process()
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    prior_grad = torch.is_grad_enabled()
    preexisting = {
        name
        for name in sys.modules
        if name == "models" or name == "download" or name == "diffusion" or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError(f"ambiguous pre-imported upstream modules: {sorted(preexisting)}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    try:
        os.chdir(args.dit_root)
        sys.path.insert(0, str(args.dit_root))
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

        # Match official/custom statement order from manual_seed through decode.
        torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        rng_after_seed = strict.cuda_rng_state_sha256()
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(args.checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device)

        classes = tuple(args.classes)
        b = len(classes)
        initial = torch.randn(
            b,
            strict.LATENT_CHANNELS,
            strict.LATENT_SIZE,
            strict.LATENT_SIZE,
            device=device,
        )
        rng_after_initial = strict.cuda_rng_state_sha256()
        state = torch.cat([initial, initial], dim=0)
        y = torch.cat(
            [
                torch.tensor(classes, device=device),
                torch.full((b,), strict.NULL_CLASS_ID, device=device),
            ]
        )
        model_kwargs = {"y": y, "cfg_scale": strict.CFG_SCALE}
        lists: dict[str, list[np.ndarray]] = {name: [] for name in STEP_ARRAY_NAMES}
        internal_axis = np.arange(
            strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16
        )
        alpha_bar = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)[
            internal_axis.astype(np.int64)
        ]

        original_model_forward = model.forward
        captured_raw: torch.Tensor | None = None
        raw_forward_calls = 0

        def observed_model_forward(*forward_args: Any, **forward_kwargs: Any) -> torch.Tensor:
            nonlocal captured_raw, raw_forward_calls
            if captured_raw is not None:
                raise RuntimeError("more than one raw model forward occurred in a sampler step")
            value = original_model_forward(*forward_args, **forward_kwargs)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("DiT raw forward no longer returns one tensor")
            captured_raw = value
            raw_forward_calls += 1
            return value

        # ``forward_with_cfg`` calls ``self.forward`` exactly once.  Replacing
        # that attribute with this transparent observer preserves its returned
        # tensor while exposing the pre-CFG conditional/unconditional branches.
        model.forward = observed_model_forward  # type: ignore[method-assign]
        try:
            for step, internal_t in enumerate(internal_axis.tolist()):
                t = torch.full((2 * b,), internal_t, device=device, dtype=torch.long)
                lists["state_before"].append(
                    np.ascontiguousarray(state[:b].cpu().numpy(), dtype=np.float32)
                )
                captured_raw = None
                out = diffusion.p_mean_variance(
                    model.forward_with_cfg,
                    state,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )
                raw = captured_raw
                if raw is None or raw.shape != (
                    2 * b,
                    2 * strict.LATENT_CHANNELS,
                    strict.LATENT_SIZE,
                    strict.LATENT_SIZE,
                ):
                    raise RuntimeError(
                        f"unexpected captured raw DiT output: "
                        f"{None if raw is None else tuple(raw.shape)}"
                    )
                conditional = raw[:b]
                unconditional = raw[b:]
                lists["conditional_epsilon_raw"].append(
                    np.ascontiguousarray(
                        conditional[:, : strict.LATENT_CHANNELS].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                )
                lists["unconditional_epsilon_raw"].append(
                    np.ascontiguousarray(
                        unconditional[:, : strict.LATENT_CHANNELS].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                )
                lists["conditional_variance_values_raw"].append(
                    np.ascontiguousarray(
                        conditional[:, strict.LATENT_CHANNELS :].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                )
                lists["unconditional_variance_values_raw"].append(
                    np.ascontiguousarray(
                        unconditional[:, strict.LATENT_CHANNELS :].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                )
                captured_raw = None

                pred = out["pred_xstart"][:b].contiguous()
                mean = out["mean"][:b].contiguous()
                sigma = torch.exp(0.5 * out["log_variance"][:b]).contiguous()
                lists["pred_xstart"].append(
                    np.ascontiguousarray(pred.cpu().numpy(), dtype=np.float32)
                )
                lists["p_mean"].append(
                    np.ascontiguousarray(mean.cpu().numpy(), dtype=np.float32)
                )
                lists["p_standard_deviation"].append(
                    np.ascontiguousarray(sigma.cpu().numpy(), dtype=np.float32)
                )

                # Upstream p_sample draws the complete 2B tensor even for t=0.
                noise_2b = torch.randn_like(state)
                lists["transition_innovation"].append(
                    np.ascontiguousarray(noise_2b[:b].cpu().numpy(), dtype=np.float32)
                )
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(state.shape) - 1)))
                state = (
                    out["mean"]
                    + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise_2b
                ).detach()
                if step % 25 == 0 or step + 1 == strict.NUM_SAMPLING_STEPS:
                    print(
                        f"recorded {step + 1}/{strict.NUM_SAMPLING_STEPS} transitions",
                        flush=True,
                    )
        finally:
            model.forward = original_model_forward  # type: ignore[method-assign]

        rng_after_diffusion = strict.cuda_rng_state_sha256()
        final_latents, discarded = state.chunk(2, dim=0)
        decoded = vae.decode(final_latents / strict.VAE_SCALING_FACTOR).sample
        custom.save_outputs(decoded, args.outdir, classes, save_image)
        torch.cuda.synchronize()
        arrays: dict[str, np.ndarray] = {
            name: np.ascontiguousarray(np.stack(rows, axis=1), dtype=np.float32)
            for name, rows in lists.items()
        }
        arrays.update(
            {
                "final_latents": np.ascontiguousarray(final_latents.cpu().numpy(), dtype=np.float32),
                "decoded_images": np.ascontiguousarray(decoded.cpu().numpy(), dtype=np.float32),
                "internal_timestep": internal_axis,
                "alpha_bar": np.ascontiguousarray(alpha_bar, dtype=np.float64),
            }
        )
        validate_trace_arrays(arrays, b)
        execution = {
            "rng_state_sha256": {
                "after_manual_seed": rng_after_seed,
                "after_initial_noise": rng_after_initial,
                "after_250_full_2B_transition_draws": rng_after_diffusion,
            },
            "tensor_sha256": {
                "initial_noise_b": strict.tensor_sha256(initial),
                "final_latents_first_half_b": strict.tensor_sha256(final_latents),
                "final_latents_discarded_second_half_b": strict.tensor_sha256(discarded),
                "decoded_samples_b": strict.tensor_sha256(decoded),
            },
            "observed_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "transition_randn_like_calls": strict.NUM_SAMPLING_STEPS,
            "raw_model_forward_calls_observed": raw_forward_calls,
            "raw_model_forward_calls_expected": strict.NUM_SAMPLING_STEPS,
            "transition_randn_like_shape": list(state.shape),
            "terminal_t0_full_draw_recorded_before_mask": True,
        }
        return arrays, execution
    finally:
        torch.set_grad_enabled(prior_grad)
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for name in list(sys.modules):
            if (
                name == "models"
                or name == "download"
                or name == "diffusion"
                or name.startswith("diffusion.")
            ) and name not in preexisting:
                sys.modules.pop(name, None)


def _save_trace(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{name: arrays[name] for name in TRACE_ARRAY_NAMES})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dry_run(args: argparse.Namespace) -> None:
    custom.validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    vae = strict.validate_vae_snapshot(args.vae_snapshot)
    checkpoint = strict.checkpoint_dry_probe(args.checkpoint)
    print(
        json.dumps(
            {
                "status": "dry-run",
                "runner": RUNNER_NAME,
                "classes": list(args.classes),
                "seed": args.seed,
                "trace_shapes": {key: list(value) for key, value in _trace_shapes(len(args.classes)).items()},
                "source": source,
                "checkpoint_probe": checkpoint,
                "vae_snapshot": vae,
                "cuda_available": torch.cuda.is_available(),
                "outdir": str(args.outdir),
                "reference_baseline_dir": str(args.reference_baseline_dir) if args.reference_baseline_dir else None,
                "canonical_command": _canonical_command(args),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_self_test() -> None:
    assert custom.parse_classes("207") == (207,)
    assert custom.parse_classes("207,360,387") == (207, 360, 387)
    assert _trace_shapes(1)["state_before"] == (1, 250, 4, 32, 32)
    assert _trace_shapes(8)["decoded_images"] == (8, 3, 256, 256)
    for invalid in (0, 9, True):
        try:
            _trace_shapes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid batch size accepted: {invalid!r}")

    # Full-2B draws, including t=0, feed a retained first-B NumPy replay.
    generator = torch.Generator(device="cpu").manual_seed(123)
    initial_t = torch.randn((2, 1, 8, 8), generator=generator)
    internal = np.array([2, 1, 0], dtype=np.int16)
    means: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    current = initial_t.clone()
    for index, timestep in enumerate(internal.tolist()):
        states.append(current.numpy().copy())
        mean_t = current * np.float32(0.5) + np.float32(index + 1)
        sigma_t = torch.full_like(current, np.float32(0.125 * (index + 1)))
        full_draw = torch.randn((4, 1, 8, 8), generator=generator)
        means.append(mean_t.numpy().copy())
        sigmas.append(sigma_t.numpy().copy())
        innovations.append(full_draw[:2].numpy().copy())
        current = mean_t + float(timestep != 0) * sigma_t * full_draw[:2]
    replay_states, replay_final = numpy_replay(
        initial_t.numpy().copy(),
        np.stack(means, axis=1),
        np.stack(sigmas, axis=1),
        np.stack(innovations, axis=1),
        internal,
    )
    assert np.array_equal(replay_states, np.stack(states, axis=1))
    assert np.array_equal(replay_final, current.numpy())
    state_after_full_draws = generator.get_state()
    generator_short = torch.Generator(device="cpu").manual_seed(123)
    torch.randn((2, 1, 8, 8), generator=generator_short)
    for _ in internal:
        torch.randn((2, 1, 8, 8), generator=generator_short)
    assert not torch.equal(state_after_full_draws, generator_short.get_state())

    constant = np.ones((2, 3, 4, 4), dtype=np.float32)
    assert np.array_equal(
        amplitude_normalized_dirichlet_roughness(constant), np.zeros(2)
    )
    ramp = np.broadcast_to(
        np.arange(3, dtype=np.float32)[None, None, None, :], (1, 1, 2, 3)
    ).copy()
    metric = amplitude_normalized_dirichlet_roughness(ramp)
    assert np.allclose(metric, np.array([1.5]), rtol=0, atol=1e-12)
    assert np.allclose(
        metric,
        amplitude_normalized_dirichlet_roughness(7.0 * ramp + 19.0),
        rtol=0,
        atol=1e-12,
    )

    # NPZ round-trip verifies member names, dtype, hashes, and shape rules on a
    # tiny contract fixture without allocating the real 160+ MiB trace.
    with tempfile.TemporaryDirectory(prefix="dit-custom-trace-self-test-") as temporary:
        path = Path(temporary) / TRACE_NAME
        fixture = {
            "state_before": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "pred_xstart": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "p_mean": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "p_standard_deviation": np.ones((1, 1, 1, 2, 2), dtype=np.float32),
            "transition_innovation": np.ones((1, 1, 1, 2, 2), dtype=np.float32),
            "conditional_epsilon_raw": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "unconditional_epsilon_raw": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "conditional_variance_values_raw": np.zeros(
                (1, 1, 1, 2, 2), dtype=np.float32
            ),
            "unconditional_variance_values_raw": np.zeros(
                (1, 1, 1, 2, 2), dtype=np.float32
            ),
            "final_latents": np.zeros((1, 1, 2, 2), dtype=np.float32),
            "decoded_images": np.zeros((1, 3, 2, 2), dtype=np.float32),
            "internal_timestep": np.zeros((1,), dtype=np.int16),
            "alpha_bar": np.ones((1,), dtype=np.float64),
        }
        _save_trace(path, fixture)
        with np.load(path, allow_pickle=False) as archive:
            loaded = {name: archive[name] for name in archive.files}
        assert set(loaded) == set(TRACE_ARRAY_NAMES)
        assert _trace_array_records(loaded) == _trace_array_records(fixture)

    print(
        "self-test passed: custom parameter/shape rules, complete-2B/t0 RNG "
        "semantics, exact NumPy replay, Dirichlet metric helper, and NPZ hashes"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    dit_root = data_root / "baselines/DiT"
    vae_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / strict.VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=custom.parse_classes, default=custom.DEFAULT_CHALLENGE_CLASS_IDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dit-root", type=Path, default=dit_root)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=vae_snapshot)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--reference-baseline-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def normalize_paths(args: argparse.Namespace) -> None:
    raw_root = args.dit_root.expanduser().absolute()
    if os.path.lexists(raw_root) and raw_root.is_symlink():
        raise RuntimeError(f"DiT root must not be a symlink: {raw_root}")
    args.dit_root = raw_root.resolve()
    args.checkpoint = (
        (args.dit_root / "pretrained_models" / strict.CHECKPOINT_FILENAME)
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute()
    )
    if os.path.lexists(args.checkpoint) and args.checkpoint.is_symlink():
        raise RuntimeError(f"checkpoint must not be a symlink: {args.checkpoint}")
    args.checkpoint = args.checkpoint.resolve()
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    if args.outdir is not None:
        requested = args.outdir.expanduser().absolute()
        if os.path.lexists(requested) and requested.is_symlink():
            raise RuntimeError(f"output directory must not be a symlink: {requested}")
        args.outdir = requested.resolve()
    if args.reference_baseline_dir is not None:
        requested = args.reference_baseline_dir.expanduser().absolute()
        if os.path.lexists(requested) and requested.is_symlink():
            raise RuntimeError(f"reference baseline must not be a symlink: {requested}")
        args.reference_baseline_dir = requested.resolve()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if type(args.seed) is not int or not 0 <= args.seed < 1 << 63:
        parser.error("--seed must be in [0, 2^63 - 1]")
    if args.outdir is None:
        parser.error("--outdir is required unless --self-test is used")
    protected = (args.dit_root, args.checkpoint, args.vae_snapshot)
    if any(custom._paths_overlap(args.outdir, path) for path in protected):
        parser.error("--outdir must not overlap DiT source/checkpoint/VAE inputs")
    if args.reference_baseline_dir is not None:
        if custom._paths_overlap(args.outdir, args.reference_baseline_dir):
            parser.error("--outdir must not overlap --reference-baseline-dir")
        if not args.reference_baseline_dir.is_dir() or args.reference_baseline_dir.is_symlink():
            parser.error("--reference-baseline-dir must be a non-symlink directory")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    validate_args(args, parser)
    if args.dry_run:
        dry_run(args)
        return 0

    identity = build_identity(args)
    identity_hash = strict.sha256_json(identity)
    reference: dict[str, Any] | None = None
    if args.reference_baseline_dir is not None:
        reference = _inspect_reference_baseline(
            args.reference_baseline_dir, identity, args.classes, args.seed
        )
    if args.outdir.exists():
        if not args.outdir.is_dir() or args.outdir.is_symlink():
            raise RuntimeError(f"output path is not a non-symlink directory: {args.outdir}")
        if any(args.outdir.iterdir()):
            validate_completed_output(args.outdir, identity, args.classes)
            if reference is not None:
                manifest = strict.load_json(args.outdir / MANIFEST_NAME)
                _require_reference_pixels(
                    [item for item in manifest["outputs"] if str(item["relative_path"]).endswith(".png")],
                    reference["png_outputs"],
                )
                _require_reference_execution(manifest["execution"], reference["execution"])
            return 0
        raise RuntimeError(f"refusing pre-existing empty output directory: {args.outdir}")

    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    running = {
        "schema": SCHEMA_VERSION,
        "status": "running",
        "identity": identity,
        "identity_sha256": identity_hash,
        "started_unix": started,
    }
    strict.atomic_json_dump(running, args.outdir / MANIFEST_NAME)
    try:
        _snapshot_sources(args.outdir)
        arrays, execution = run_trace(args)
        if reference is not None:
            _require_reference_execution(execution, reference["execution"])
        _save_trace(args.outdir / TRACE_NAME, arrays)
        outputs, array_records = _collect_payload_records(args.outdir, args.classes, arrays)
        if reference is not None:
            _require_reference_pixels(
                [item for item in outputs if str(item["relative_path"]).endswith(".png")],
                reference["png_outputs"],
            )
        outputs_hash = strict.sha256_json(outputs)
        finished = time.time()
        manifest = {
            **running,
            "status": "complete",
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "execution": execution,
            "trace_array_records": array_records,
            "reference_pixel_match": reference is not None,
            "reference_execution_hash_match": reference is not None,
            "reference_manifest_sha256": (
                reference["manifest_sha256"] if reference is not None else None
            ),
            "outputs": outputs,
            "outputs_sha256": outputs_hash,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": strict.dependency_identity(),
                "cuda_device_count_visible": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
        }
        manifest_path = args.outdir / MANIFEST_NAME
        strict.atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": SCHEMA_VERSION,
            "identity_sha256": identity_hash,
            "manifest_sha256": strict.sha256_file(manifest_path),
            "outputs_sha256": outputs_hash,
            "output_count": len(outputs),
        }
        strict.atomic_json_dump(completion, args.outdir / COMPLETION_NAME)
        validate_completed_output(args.outdir, identity, args.classes)
    except BaseException as exc:
        failed = {
            **running,
            "status": "failed",
            "failed_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        strict.atomic_json_dump(failed, args.outdir / MANIFEST_NAME)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

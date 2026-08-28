#!/usr/bin/env python3
"""Build the label-free B-gated e-process input from one completed DiT trace.

Only the six trace members needed for replay are opened; trajectory members
are validated and immediately reduced to the nine frozen preterminal
checkpoints before they are returned to any metric/model routine.  In
particular this replay adapter does not open final_latents, decoded_images,
endpoint PNGs, labels, reviews, candidate results, or external embeddings.
It re-evaluates the frozen DiT at the two shifted heat scales, verifies that
model/VAE observation consumes no CUDA randomness, temporarily decodes the
saved pred_xstart drafts, and emits the exact input schema consumed by
observe_dit_blur_focused_eprocess.py.

This is an observation-only adapter.  It never advances or modifies a sampler.
Its output is not execution-authorized until a compatible scientific protocol
(not the existing event-rich B/C v3) is explicitly frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np

try:
    from . import calibrate_dit_blur_focused_eprocess as calibrate
    from . import observe_dit_blur_focused_eprocess as core
    from . import reproduce_dit_imagenet256 as strict
    from . import trace_dit_imagenet256_custom_batch as trace_runner
except ImportError:  # pragma: no cover - direct CLI execution
    import calibrate_dit_blur_focused_eprocess as calibrate
    import observe_dit_blur_focused_eprocess as core
    import reproduce_dit_imagenet256 as strict
    import trace_dit_imagenet256_custom_batch as trace_runner


EXPERIMENT = "dit_blur_focused_eprocess_replay_input_label_free"
SCHEMA_VERSION = 1
OUTPUT_NAME = "observer_input.npz"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
PRETERMINAL_TRACE_ARRAYS = (
    "state_before",
    "pred_xstart",
    "p_standard_deviation",
    "transition_innovation",
    "internal_timestep",
    "alpha_bar",
)
CALIBRATION_KEYS = calibrate.CALIBRATION_KEYS
CALIBRATION_CLASS_KEYS = calibrate.CALIBRATION_CLASS_KEYS


def _read_trace_preterminal(trace_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = trace_dir / trace_runner.MANIFEST_NAME
    completion_path = trace_dir / trace_runner.COMPLETION_NAME
    trace_path = trace_dir / trace_runner.TRACE_NAME
    for path in (manifest_path, completion_path, trace_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular completed trace input: {path}")
    manifest = strict.load_json(manifest_path)
    completion = strict.load_json(completion_path)
    if manifest.get("schema") != trace_runner.SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("trace manifest is not a completed custom DiT trace")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("identity_sha256") != strict.sha256_json(identity):
        raise RuntimeError("trace identity hash is invalid")
    if (
        identity.get("runner") != trace_runner.RUNNER_NAME
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or identity.get("intervention") is not None
    ):
        raise RuntimeError("trace identity violates the label-free observation boundary")
    protocol = identity.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("trace protocol is missing")
    classes = protocol.get("class_ids_ordered")
    if (
        not isinstance(classes, list)
        or not classes
        or not all(type(value) is int and 0 <= value < strict.NUM_CLASSES for value in classes)
    ):
        raise RuntimeError("trace class list is malformed")
    if protocol.get("pred_xstart_recorded_before_transition_draw") is not True:
        raise RuntimeError("trace does not attest pre-innovation pred_xstart")
    if protocol.get("transition_innovation") != "raw first-B slice of each full 2B draw":
        raise RuntimeError("trace innovation contract changed")
    if completion.get("manifest_sha256") != strict.sha256_file(manifest_path):
        raise RuntimeError("trace completion does not bind the manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("trace outputs are malformed")
    record = next(
        (
            item
            for item in outputs
            if isinstance(item, dict) and item.get("relative_path") == trace_runner.TRACE_NAME
        ),
        None,
    )
    if not isinstance(record, dict) or record.get("sha256") != strict.sha256_file(trace_path):
        raise RuntimeError("trace archive hash differs from its manifest")
    array_records = record.get("arrays")
    if not isinstance(array_records, dict):
        raise RuntimeError("trace archive lacks per-array records")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if set(archive.files) != set(trace_runner.TRACE_ARRAY_NAMES):
                raise RuntimeError("trace archive member set changed")
            full_arrays = {
                name: np.ascontiguousarray(archive[name]) for name in PRETERMINAL_TRACE_ARRAYS
            }
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read the preterminal trace arrays") from exc
    for name, value in full_arrays.items():
        expected = array_records.get(name)
        if not isinstance(expected, dict) or core._array_record(value) != expected:
            raise RuntimeError(f"preterminal trace array hash/shape changed: {name}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"preterminal trace array is non-finite: {name}")
    batch = len(classes)
    expected_state = (batch, strict.NUM_SAMPLING_STEPS, 4, 32, 32)
    for name in ("state_before", "pred_xstart", "p_standard_deviation", "transition_innovation"):
        if full_arrays[name].dtype != np.float32 or full_arrays[name].shape != expected_state:
            raise RuntimeError(f"unexpected preterminal trace shape/dtype: {name}")
    if np.any(full_arrays["p_standard_deviation"] <= 0.0):
        raise RuntimeError("trace P standard deviation must be positive")
    if full_arrays["internal_timestep"].dtype != np.int16 or not np.array_equal(
        full_arrays["internal_timestep"],
        np.arange(strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16),
    ):
        raise RuntimeError("trace internal timestep axis changed")
    if full_arrays["alpha_bar"].dtype != np.float64 or full_arrays["alpha_bar"].shape != (
        strict.NUM_SAMPLING_STEPS,
    ):
        raise RuntimeError("trace alpha_bar axis changed")
    selected = np.asarray(core.CHECKPOINTS, dtype=np.int64)
    arrays = {
        name: np.ascontiguousarray(full_arrays[name][:, selected])
        for name in (
            "state_before",
            "pred_xstart",
            "p_standard_deviation",
            "transition_innovation",
        )
    }
    arrays["internal_timestep"] = np.ascontiguousarray(
        full_arrays["internal_timestep"][selected]
    )
    arrays["alpha_bar"] = np.ascontiguousarray(full_arrays["alpha_bar"][selected])
    metadata = {
        "classes": tuple(classes),
        "global_seed": int(protocol["global_torch_seed"]),
        "trace_manifest_path": str(manifest_path.resolve()),
        "trace_manifest_sha256": strict.sha256_file(manifest_path),
        "trace_archive_sha256": strict.sha256_file(trace_path),
        "trace_identity_sha256": manifest["identity_sha256"],
        "source": identity.get("source"),
        "checkpoint": identity.get("checkpoint"),
        "vae_snapshot": identity.get("vae_snapshot"),
    }
    return arrays, metadata


def _read_calibration(path: Path, classes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("calibration must be one regular non-symlink JSON")
    payload = strict.load_json(path)
    calibrate.validate_calibration(payload)
    rows = payload.get("classes")
    if not isinstance(rows, list) or len(rows) != len(classes):
        raise RuntimeError("calibration class rows do not match trace batch")
    mapping: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(CALIBRATION_CLASS_KEYS):
            raise RuntimeError("malformed calibration class row")
        class_id = row.get("class_id")
        if type(class_id) is not int or class_id in mapping:
            raise RuntimeError("calibration class ids are invalid or duplicated")
        gate = np.asarray(row.get("blur_gate_threshold_by_checkpoint"), dtype=np.float64)
        score = row.get("blur_score_threshold")
        if gate.shape != (len(core.CHECKPOINTS),) or not np.isfinite(gate).all():
            raise RuntimeError("calibration gate thresholds must be nine finite values")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError("calibration pure B threshold must be finite")
        mapping[class_id] = {
            "gate": gate,
            "score": float(score),
        }
    if set(mapping) != set(classes):
        raise RuntimeError("calibration classes differ from the ordered trace classes")
    gate = np.stack([mapping[value]["gate"] for value in classes], axis=0)
    score = np.asarray([mapping[value]["score"] for value in classes], dtype=np.float64)
    return np.ascontiguousarray(gate), np.ascontiguousarray(score), payload


def _validate_runtime_lineage(
    metadata: Mapping[str, Any], dit_root: Path, checkpoint: Path, vae_snapshot: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = strict.validate_repository(dit_root, checkpoint)
    checkpoint_identity = strict.validate_checkpoint(checkpoint)
    vae_identity = strict.validate_vae_snapshot(vae_snapshot)
    if metadata.get("source") != source:
        raise RuntimeError("runtime DiT source differs from the trace source")
    if metadata.get("checkpoint") != checkpoint_identity:
        raise RuntimeError("runtime checkpoint differs from the trace checkpoint")
    if metadata.get("vae_snapshot") != vae_identity:
        raise RuntimeError("runtime VAE differs from the trace VAE")
    return source, checkpoint_identity, vae_identity


def _decode_drafts(vae: Any, pred: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    import torch

    flat = np.ascontiguousarray(pred.reshape(-1, 4, 32, 32), dtype=np.float32)
    decoded_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            latent = torch.from_numpy(flat[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            decoded = vae.decode(latent / strict.VAE_SCALING_FACTOR).sample
            rgb = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
            decoded_rows.append(rgb.cpu().numpy().astype(np.float32))
    result = np.concatenate(decoded_rows, axis=0)
    return np.ascontiguousarray(
        result.reshape(pred.shape[0], pred.shape[1], 3, strict.IMAGE_SIZE, strict.IMAGE_SIZE)
    )


def build_observer_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    calibration_gate: np.ndarray,
    calibration_score: np.ndarray,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
    decode_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for shifted DiT replay")
    strict.ensure_single_process()
    _validate_runtime_lineage(metadata, dit_root, checkpoint, vae_snapshot)
    prior_cwd = Path.cwd()
    prior_path = list(sys.path)
    prior_grad = torch.is_grad_enabled()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected = {
            "diffusion": (dit_root / "diffusion/__init__.py").resolve(),
            "download": (dit_root / "download.py").resolve(),
            "models": (dit_root / "models.py").resolve(),
        }
        if imported != expected:
            raise RuntimeError("upstream DiT import shadowing detected")
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        vae = AutoencoderKL.from_pretrained(
            str(vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device=device, dtype=torch.float32).eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
        timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
        # trace alpha_bar is in sampling-step order (internal 249..0).
        expected_trace_alpha = alpha[arrays["internal_timestep"].astype(np.int64)]
        if not np.array_equal(expected_trace_alpha, arrays["alpha_bar"]):
            raise RuntimeError("runtime diffusion alpha schedule differs from trace")
        selected_pred = np.ascontiguousarray(arrays["pred_xstart"])
        selected_sigma = np.ascontiguousarray(arrays["p_standard_deviation"])
        selected_innovation = np.ascontiguousarray(arrays["transition_innovation"])
        classes = tuple(int(value) for value in metadata["classes"])
        batch = len(classes)
        y = torch.cat(
            [
                torch.tensor(classes, dtype=torch.long, device=device),
                torch.full((batch,), strict.NULL_CLASS_ID, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        theta = np.zeros(
            (batch, len(core.HEAT_SHIFTS), len(core.CHECKPOINTS), 4, 32, 32),
            dtype=np.float64,
        )
        maximum_current_reconstruction_error = 0.0
        rng_before_observation = strict.cuda_rng_state_sha256()
        with torch.inference_mode():
            for checkpoint_index, sampling_step in enumerate(core.CHECKPOINTS):
                internal_t = core.INTERNAL_TIMESTEPS[checkpoint_index]
                state_first = torch.from_numpy(
                    np.ascontiguousarray(arrays["state_before"][:, checkpoint_index])
                ).to(device=device, dtype=torch.float32)
                state = torch.cat([state_first, state_first], dim=0)
                original_t = int(timestep_map[internal_t])
                current_t = torch.full(
                    (2 * batch,), original_t, dtype=torch.long, device=device
                )
                current_output = model.forward_with_cfg(
                    state, current_t, y=y, cfg_scale=strict.CFG_SCALE
                )
                current_epsilon = current_output[:batch, :4]
                current_np = current_epsilon.cpu().numpy().astype(np.float64)
                alpha_current = float(alpha[internal_t])
                reconstructed = (
                    arrays["state_before"][:, checkpoint_index].astype(np.float64)
                    - math.sqrt(1.0 - alpha_current) * current_np
                ) / math.sqrt(alpha_current)
                error = float(
                    np.max(
                        np.abs(
                            reconstructed
                            - arrays["pred_xstart"][:, checkpoint_index].astype(np.float64)
                        )
                    )
                )
                maximum_current_reconstruction_error = max(
                    maximum_current_reconstruction_error, error
                )
                if error > 5e-4:
                    raise RuntimeError(
                        f"replayed current epsilon does not reconstruct pred_xstart: {error}"
                    )
                for scale_index, shifted_internal in enumerate(
                    row[checkpoint_index] for row in core.SHIFTED_INTERNAL_TIMESTEPS
                ):
                    if shifted_internal == internal_t:
                        continue
                    alpha_shifted = float(alpha[shifted_internal])
                    rho = math.sqrt(alpha_shifted / alpha_current)
                    shifted_t = torch.full(
                        (2 * batch,),
                        int(timestep_map[shifted_internal]),
                        dtype=torch.long,
                        device=device,
                    )
                    shifted_output = model.forward_with_cfg(
                        state * rho, shifted_t, y=y, cfg_scale=strict.CFG_SCALE
                    )
                    shifted_epsilon = (
                        shifted_output[:batch, :4].cpu().numpy().astype(np.float64)
                    )
                    theta[:, scale_index, checkpoint_index] = (
                        -rho * shifted_epsilon / math.sqrt(1.0 - alpha_shifted)
                        + current_np / math.sqrt(1.0 - alpha_current)
                    )
            decoded = _decode_drafts(
                vae, selected_pred, device=device, batch_size=decode_batch_size
            )
        rng_after_observation = strict.cuda_rng_state_sha256()
        if rng_after_observation != rng_before_observation:
            raise RuntimeError("shifted DiT/VAE observation consumed CUDA randomness")
        output = {
            "decoded_pred_xstart_rgb": decoded,
            "theta": np.ascontiguousarray(theta, dtype=np.float64),
            "p_standard_deviation": selected_sigma,
            "transition_innovation": selected_innovation,
            "sampling_step": np.asarray(core.CHECKPOINTS, dtype=np.int16),
            "shifted_internal_timestep": np.asarray(
                core.SHIFTED_INTERNAL_TIMESTEPS, dtype=np.int16
            ),
            "heat_shift": np.asarray(core.HEAT_SHIFTS, dtype=np.float64),
            "effective_nonidentity": np.asarray(
                core.EFFECTIVE_NONIDENTITY, dtype=np.uint8
            ),
            "blur_gate_threshold": np.ascontiguousarray(
                calibration_gate, dtype=np.float64
            ),
            "blur_score_threshold": np.ascontiguousarray(
                calibration_score, dtype=np.float64
            ),
            "class_id": np.asarray(classes, dtype=np.int16),
        }
        core.validate_observer_input(output)
        execution = {
            "cuda_rng_unchanged_across_all_current_shifted_and_vae_observations": True,
            "cuda_rng_sha256": rng_before_observation,
            "maximum_current_epsilon_pred_xstart_reconstruction_error": (
                maximum_current_reconstruction_error
            ),
            "current_model_evaluations": len(core.CHECKPOINTS),
            "shifted_model_evaluations": int(
                np.sum(np.asarray(core.EFFECTIVE_NONIDENTITY, dtype=np.uint8))
            ),
            "endpoint_arrays_or_images_loaded": False,
            "labels_reviews_external_representations_loaded": False,
        }
        return output, execution
    finally:
        torch.set_grad_enabled(prior_grad)
        os.chdir(prior_cwd)
        sys.path[:] = prior_path


def publish(args: argparse.Namespace) -> Path:
    if args.outdir.exists() or args.outdir.is_symlink():
        raise RuntimeError(f"refusing pre-existing output path: {args.outdir}")
    arrays, metadata = _read_trace_preterminal(args.trace_dir)
    calibration_gate, calibration_score, calibration = _read_calibration(
        args.calibration, metadata["classes"]
    )
    observer_arrays, execution = build_observer_arrays(
        arrays,
        metadata,
        calibration_gate=calibration_gate,
        calibration_score=calibration_score,
        dit_root=args.dit_root,
        checkpoint=args.checkpoint,
        vae_snapshot=args.vae_snapshot,
        decode_batch_size=args.decode_batch_size,
    )
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent)
    )
    try:
        output_path = staging / OUTPUT_NAME
        with output_path.open("wb") as handle:
            np.savez(handle, **observer_arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "execution_ready": False,
            "scientific_protocol_requirement": (
                "explicit compatible B/E scientific v4 before any real screen, "
                "or a later independent pool; existing B/C event-rich v3 is incompatible"
            ),
            "trace": metadata,
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": strict.sha256_file(args.calibration),
                "identity_sha256": calibration["identity_sha256"],
            },
            "implementation": {
                "adapter_path": str(Path(__file__).resolve()),
                "adapter_sha256": strict.sha256_file(Path(__file__).resolve()),
                "core_path": str(Path(core.__file__).resolve()),
                "core_sha256": strict.sha256_file(Path(core.__file__).resolve()),
            },
            "execution": execution,
            "output": {
                "relative_path": OUTPUT_NAME,
                "bytes": output_path.stat().st_size,
                "sha256": strict.sha256_file(output_path),
                "arrays": {
                    name: core._array_record(observer_arrays[name])
                    for name in core.INPUT_ARRAY_NAMES
                },
            },
        }
        manifest["identity_sha256"] = core._sha256_json(manifest)
        core._atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": manifest["identity_sha256"],
            "manifest_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "output_sha256": strict.sha256_file(output_path),
        }
        core._atomic_json_dump(completion, staging / COMPLETION_NAME)
        os.replace(staging, args.outdir)
        return args.outdir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    if core.CHECKPOINTS != tuple(range(69, 150, 10)):
        raise AssertionError("checkpoint window changed")
    if core.INTERNAL_TIMESTEPS != tuple(249 - value for value in core.CHECKPOINTS):
        raise AssertionError("checkpoint/internal-time mapping changed")
    if np.sum(np.asarray(core.EFFECTIVE_NONIDENTITY), axis=1).tolist() != [5, 8]:
        raise AssertionError("two-scale nonidentity counts changed")
    source = {
        "experiment": calibrate.SOURCE_EXPERIMENT,
        "manifest_identity_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "time_series_file_sha256": "3" * 64,
        "unused_archive_members_not_loaded": [],
    }
    payload = calibrate.derive_calibration(calibrate._synthetic_arrays(), source)
    calibrate.validate_calibration(payload)
    print(
        "self-test passed: checkpoint/shift schema, 5/8 nonidentity counts, "
        "lineage-bound label-free calibration, and no endpoint array in adapter inputs"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--dit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--decode-batch-size", type=int, default=18)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        real_values = (
            args.trace_dir,
            args.calibration,
            args.dit_root,
            args.checkpoint,
            args.vae_snapshot,
            args.outdir,
        )
        if any(value is not None for value in real_values):
            parser.error("--self-test cannot be combined with real inputs")
        self_test()
        return 0
    required = (
        "trace_dir",
        "calibration",
        "dit_root",
        "checkpoint",
        "vae_snapshot",
        "outdir",
    )
    if any(getattr(args, name) is None for name in required):
        parser.error("real replay requires trace, calibration, model, VAE, and output paths")
    if args.decode_batch_size <= 0:
        parser.error("--decode-batch-size must be positive")
    for name in required:
        value = getattr(args, name)
        setattr(args, name, value.expanduser().absolute())
    args.dit_root = args.dit_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.vae_snapshot = args.vae_snapshot.resolve()
    args.trace_dir = args.trace_dir.resolve()
    args.calibration = args.calibration.resolve()
    output = publish(args)
    print(f"published label-free B-gated e-process replay input: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sample pair-keyed DiT trajectories and retain only frozen B/C observables.

The post-anchor plan determines the active class union.  Every singleton
``(global_seed,class_id)`` stream uses the exact endpoint-screen SHA-256 seed
derivation and official 250-step, full-2B ancestral CFG transition semantics.
The only retained internal tensors are the nine B pred-xstart latents and/or
the fifty C channel-3 pred-xstart planes required by the candidate(s) for that
class.  No candidate score, label, review, embedding, external metric,
selection, or intervention is computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

try:
    from .dit_event_rich_dynamic_contract import (
        B_CANDIDATE,
        B_CHECKPOINTS,
        C_CANDIDATE,
        C_CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        load_json,
        pair_relative_directory,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        without_identity,
        write_json,
        write_json_exclusive,
    )
except ImportError:
    from dit_event_rich_dynamic_contract import (  # type: ignore
        B_CANDIDATE,
        B_CHECKPOINTS,
        C_CANDIDATE,
        C_CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        load_json,
        pair_relative_directory,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        without_identity,
        write_json,
        write_json_exclusive,
    )


RUNNER = "sample_dit_event_rich_dynamic_traces"
PAIR_SCHEMA = 1
TRACE_NAME = "internal_trace.npz"
ENDPOINT_NAME = "endpoint.png"
PAIR_MANIFEST = "manifest.json"
PAIR_COMPLETION = "completion.json"
POOL_MANIFEST = "pool_manifest.json"
POOL_COMPLETION = "pool_completion.json"

B_ARRAY = "b_pred_xstart"
C_ARRAY = "c_pred_xstart_c3"
C_ALPHA = "c_alpha_bar"
FINAL_LATENT = "final_latent"
B_STEP = "b_sampling_step"
C_STEP = "c_sampling_step"
B_INTERNAL_T = "b_internal_timestep"
C_INTERNAL_T = "c_internal_timestep"


def _load_module(path: Path, name: str) -> ModuleType:
    path = require_regular(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_lock(source_lock: Path) -> tuple[dict[str, Any], dict[str, Any], ModuleType]:
    source_lock = require_directory(source_lock, "dynamic confirmation source lock")
    manifest = load_json(require_regular(source_lock / "manifest.json", "source manifest"))
    completion = load_json(require_regular(source_lock / "completion.json", "source completion"))
    contract = load_json(require_regular(source_lock / "dynamic_contract.json", "dynamic contract"))
    manifest_identity = manifest.get("identity_sha256")
    contract_identity = contract.get("identity_sha256")
    if (
        canonical_sha256(without_identity(manifest)) != manifest_identity
        or canonical_sha256(without_identity(contract)) != contract_identity
        or manifest.get("status") != "complete"
        or manifest.get("dynamic_contract_identity_sha256") != contract_identity
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256")
        != sha256_file(source_lock / "manifest.json")
    ):
        raise RuntimeError("dynamic source lock identity mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("dynamic source-lock member list is malformed")
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    expected_source = "sources/sample_dit_event_rich_dynamic_traces.py"
    current_hash = sha256_file(Path(__file__).resolve())
    if by_name.get(expected_source, {}).get("sha256") != current_hash:
        raise RuntimeError("running trace source differs from frozen source snapshot")
    for name, row in by_name.items():
        path = require_regular(source_lock / str(name), f"source-lock member {name}")
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"dynamic source-lock member changed: {name}")
    strict = _load_module(
        source_lock / "sources/reproduce_dit_imagenet256.py",
        "_event_dynamic_frozen_strict",
    )
    return contract, manifest, strict


def required_arrays(plan: Mapping[str, Any], class_id: int) -> tuple[str, ...]:
    names = [FINAL_LATENT]
    if class_id in candidate_classes(plan, B_CANDIDATE):
        names.extend((B_ARRAY, B_STEP, B_INTERNAL_T))
    if class_id in candidate_classes(plan, C_CANDIDATE):
        names.extend((C_ARRAY, C_ALPHA, C_STEP, C_INTERNAL_T))
    return tuple(names)


def _array_contract(plan: Mapping[str, Any], class_id: int) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    result: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        FINAL_LATENT: ((4, 32, 32), np.dtype(np.float32)),
    }
    if class_id in candidate_classes(plan, B_CANDIDATE):
        result.update(
            {
                B_ARRAY: ((len(B_CHECKPOINTS), 4, 32, 32), np.dtype(np.float32)),
                B_STEP: ((len(B_CHECKPOINTS),), np.dtype(np.int16)),
                B_INTERNAL_T: ((len(B_CHECKPOINTS),), np.dtype(np.int16)),
            }
        )
    if class_id in candidate_classes(plan, C_CANDIDATE):
        result.update(
            {
                C_ARRAY: ((len(C_CHECKPOINTS), 32, 32), np.dtype(np.float32)),
                C_ALPHA: ((len(C_CHECKPOINTS),), np.dtype(np.float64)),
                C_STEP: ((len(C_CHECKPOINTS),), np.dtype(np.int16)),
                C_INTERNAL_T: ((len(C_CHECKPOINTS),), np.dtype(np.int16)),
            }
        )
    return result


def validate_arrays(arrays: Mapping[str, np.ndarray], plan: Mapping[str, Any], class_id: int) -> None:
    contract = _array_contract(plan, class_id)
    if set(arrays) != set(contract):
        raise RuntimeError(f"minimum trace member set changed: {sorted(arrays)}")
    for name, (shape, dtype) in contract.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name}: shape={value.shape}, dtype={value.dtype}")
    if B_STEP in arrays:
        if not np.array_equal(arrays[B_STEP], np.asarray(B_CHECKPOINTS, dtype=np.int16)):
            raise RuntimeError("B sampling-step axis changed")
        if not np.array_equal(
            arrays[B_INTERNAL_T], 249 - np.asarray(B_CHECKPOINTS, dtype=np.int16)
        ):
            raise RuntimeError("B internal-timestep axis changed")
    if C_STEP in arrays:
        if not np.array_equal(arrays[C_STEP], np.asarray(C_CHECKPOINTS, dtype=np.int16)):
            raise RuntimeError("C sampling-step axis changed")
        if not np.array_equal(
            arrays[C_INTERNAL_T], 249 - np.asarray(C_CHECKPOINTS, dtype=np.int16)
        ):
            raise RuntimeError("C internal-timestep axis changed")
        if np.any(arrays[C_ALPHA] <= 0) or np.any(arrays[C_ALPHA] > 1):
            raise RuntimeError("C alpha-bar values must lie in (0,1]")


def array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "raw_sha256": sha256_array(value),
        }
        for name, value in sorted(arrays.items())
    }


def inspect_png(path: Path) -> dict[str, Any]:
    path = require_regular(path, "dynamic endpoint PNG")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or tuple(image.size) != (256, 256):
            raise RuntimeError("dynamic endpoint must be RGB 256x256")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return {
        "name": ENDPOINT_NAME,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_hash,
        "mode": "RGB",
        "size": [256, 256],
    }


def pair_identity(
    *,
    source_lock: Path,
    source_contract: Mapping[str, Any],
    anchor_plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": PAIR_SCHEMA,
        "runner": RUNNER,
        "dynamic_source_lock": str(source_lock),
        "dynamic_contract_identity_sha256": source_contract["identity_sha256"],
        "event_protocol_identity_sha256": anchor_plan["protocol_identity_sha256"],
        "anchor_plan_identity_sha256": anchor_plan["identity_sha256"],
        "pair_key": {
            "phase": phase,
            "global_seed": global_seed,
            "class_id": class_id,
        },
        "derived_torch_seed": derive_pair_seed(global_seed, class_id),
        "observation_only": True,
        "candidate_scores_computed": False,
        "labels_reviews_or_external_representations_opened": False,
        "required_arrays": list(required_arrays(anchor_plan, class_id)),
        "sampler_contract": source_contract["sampler_contract"],
    }


def load_pair_arrays(path: Path, records: Mapping[str, Any], plan: Mapping[str, Any], class_id: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load minimum trace: {path}") from exc
    validate_arrays(arrays, plan, class_id)
    if array_records(arrays) != records:
        raise RuntimeError("minimum trace raw hashes or metadata changed")
    return arrays


def validate_pair(
    root: Path,
    *,
    source_lock: Path,
    source_contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    relative = pair_relative_directory(phase, global_seed, class_id)
    outdir = root / relative
    if not outdir.is_dir() or outdir.is_symlink():
        raise RuntimeError(f"dynamic trace pair missing: {outdir}")
    names = {path.name for path in outdir.iterdir()}
    if names != {TRACE_NAME, ENDPOINT_NAME, PAIR_MANIFEST, PAIR_COMPLETION}:
        raise RuntimeError(f"dynamic trace pair member set changed: {outdir}")
    manifest_path = require_regular(outdir / PAIR_MANIFEST, "pair manifest")
    completion_path = require_regular(outdir / PAIR_COMPLETION, "pair completion")
    trace_path = require_regular(outdir / TRACE_NAME, "minimum trace")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = pair_identity(
        source_lock=source_lock,
        source_contract=source_contract,
        anchor_plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
    )
    identity_hash = canonical_sha256(identity)
    if (
        manifest.get("schema_version") != PAIR_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_hash
    ):
        raise RuntimeError("dynamic pair identity changed")
    trace = manifest.get("trace")
    if not isinstance(trace, dict) or trace.get("sha256") != sha256_file(trace_path):
        raise RuntimeError("dynamic trace file hash changed")
    load_pair_arrays(trace_path, trace.get("arrays", {}), plan, class_id)
    endpoint = inspect_png(outdir / ENDPOINT_NAME)
    if manifest.get("endpoint") != endpoint:
        raise RuntimeError("dynamic endpoint record changed")
    expected_completion = {
        "complete": True,
        "identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "trace_sha256": trace["sha256"],
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
    }
    if completion != expected_completion:
        raise RuntimeError("dynamic pair completion receipt changed")
    return {
        "phase": phase,
        "global_seed": global_seed,
        "class_id": class_id,
        "relative_directory": relative,
        "pair_identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "completion_sha256": sha256_file(completion_path),
        "trace_sha256": trace["sha256"],
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
    }


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_pair(
    root: Path,
    *,
    decoded: torch.Tensor,
    arrays: Mapping[str, np.ndarray],
    save_image: Any,
    source_lock: Path,
    source_contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    relative = pair_relative_directory(phase, global_seed, class_id)
    destination = root / relative
    if os.path.lexists(destination):
        return validate_pair(
            root,
            source_lock=source_lock,
            source_contract=source_contract,
            plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        validate_arrays(arrays, plan, class_id)
        _save_npz(staging / TRACE_NAME, arrays)
        save_image(
            decoded,
            staging / ENDPOINT_NAME,
            nrow=1,
            padding=0,
            normalize=True,
            value_range=(-1, 1),
        )
        identity = pair_identity(
            source_lock=source_lock,
            source_contract=source_contract,
            anchor_plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
        trace_record = {
            "name": TRACE_NAME,
            "bytes": (staging / TRACE_NAME).stat().st_size,
            "sha256": sha256_file(staging / TRACE_NAME),
            "arrays": array_records(arrays),
        }
        endpoint = inspect_png(staging / ENDPOINT_NAME)
        manifest = {
            "schema_version": PAIR_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "trace": trace_record,
            "endpoint": endpoint,
        }
        write_json_exclusive(staging / PAIR_MANIFEST, manifest)
        write_json_exclusive(
            staging / PAIR_COMPLETION,
            {
                "complete": True,
                "identity_sha256": manifest["identity_sha256"],
                "manifest_sha256": sha256_file(staging / PAIR_MANIFEST),
                "trace_sha256": trace_record["sha256"],
                "endpoint_sha256": endpoint["sha256"],
                "endpoint_pixel_sha256": endpoint["pixel_sha256"],
            },
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_pair(
        root,
        source_lock=source_lock,
        source_contract=source_contract,
        plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
    )


def load_models(strict: ModuleType, contract: Mapping[str, Any], args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dynamic DiT trajectory sampling")
    dit_root = require_directory(args.dit_root, "DiT repository")
    checkpoint = require_regular(args.checkpoint, "DiT checkpoint")
    vae_snapshot = require_directory(args.vae_snapshot, "VAE snapshot")
    assets = contract["assets"]
    if (
        strict.validate_repository(dit_root, checkpoint) != assets["dit_repository"]
        or strict.validate_checkpoint(checkpoint) != assets["checkpoint"]
        or strict.validate_vae_snapshot(vae_snapshot) != assets["vae_snapshot"]
    ):
        raise RuntimeError("runtime DiT/VAE assets differ from frozen source lock")
    strict.ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.chdir(dit_root)
    sys.path.insert(0, str(dit_root))
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
        "diffusion": (dit_root / "diffusion/__init__.py").resolve(),
        "download": (dit_root / "download.py").resolve(),
        "models": (dit_root / "models.py").resolve(),
    }
    if imported != expected:
        raise RuntimeError(f"upstream import shadowing detected: {imported} != {expected}")
    device = torch.device("cuda")
    model = DiT_models[strict.MODEL_NAME](
        input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
    ).to(device)
    model.load_state_dict(find_model(str(checkpoint)))
    model.eval()
    diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
    vae = AutoencoderKL.from_pretrained(
        str(vae_snapshot), local_files_only=True, use_safetensors=True
    ).to(device)
    return model, diffusion, vae, save_image


def sample_one(
    *,
    model: Any,
    diffusion: Any,
    vae: Any,
    strict: ModuleType,
    plan: Mapping[str, Any],
    global_seed: int,
    class_id: int,
) -> tuple[torch.Tensor, dict[str, np.ndarray], dict[str, Any]]:
    pair_seed = derive_pair_seed(global_seed, class_id)
    torch.manual_seed(pair_seed)
    device = torch.device("cuda")
    rng_after_seed = strict.cuda_rng_state_sha256()
    initial = torch.randn(1, 4, 32, 32, device=device)
    rng_after_initial = strict.cuda_rng_state_sha256()
    state = torch.cat((initial, initial), dim=0)
    y = torch.tensor([class_id, strict.NULL_CLASS_ID], device=device)
    kwargs = {"y": y, "cfg_scale": strict.CFG_SCALE}
    need_b = class_id in candidate_classes(plan, B_CANDIDATE)
    need_c = class_id in candidate_classes(plan, C_CANDIDATE)
    b_values: list[np.ndarray] = []
    c_values: list[np.ndarray] = []
    internal_axis = np.arange(249, -1, -1, dtype=np.int16)
    alpha_full = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)[
        internal_axis.astype(np.int64)
    ]
    for sampling_step, internal_t in enumerate(internal_axis.tolist()):
        t = torch.full((2,), internal_t, device=device, dtype=torch.long)
        out = diffusion.p_mean_variance(
            model.forward_with_cfg,
            state,
            t,
            clip_denoised=False,
            model_kwargs=kwargs,
        )
        pred = out["pred_xstart"][0]
        if need_b and sampling_step in B_CHECKPOINTS:
            b_values.append(
                np.ascontiguousarray(pred.detach().cpu().numpy(), dtype=np.float32)
            )
        if need_c and sampling_step in C_CHECKPOINTS:
            c_values.append(
                np.ascontiguousarray(pred[3].detach().cpu().numpy(), dtype=np.float32)
            )
        noise_2b = torch.randn_like(state)
        nonzero = (t != 0).float().view(-1, 1, 1, 1)
        state = (
            out["mean"]
            + nonzero * torch.exp(0.5 * out["log_variance"]) * noise_2b
        ).detach()
    rng_after_diffusion = strict.cuda_rng_state_sha256()
    kept, discarded = state.chunk(2, dim=0)
    decoded = vae.decode(kept / strict.VAE_SCALING_FACTOR).sample
    torch.cuda.synchronize()
    arrays: dict[str, np.ndarray] = {
        FINAL_LATENT: np.ascontiguousarray(kept[0].cpu().numpy(), dtype=np.float32),
    }
    if need_b:
        arrays.update(
            {
                B_ARRAY: np.ascontiguousarray(np.stack(b_values), dtype=np.float32),
                B_STEP: np.asarray(B_CHECKPOINTS, dtype=np.int16),
                B_INTERNAL_T: 249 - np.asarray(B_CHECKPOINTS, dtype=np.int16),
            }
        )
    if need_c:
        arrays.update(
            {
                C_ARRAY: np.ascontiguousarray(np.stack(c_values), dtype=np.float32),
                C_ALPHA: np.ascontiguousarray(
                    alpha_full[np.asarray(C_CHECKPOINTS, dtype=np.int64)],
                    dtype=np.float64,
                ),
                C_STEP: np.asarray(C_CHECKPOINTS, dtype=np.int16),
                C_INTERNAL_T: 249 - np.asarray(C_CHECKPOINTS, dtype=np.int16),
            }
        )
    validate_arrays(arrays, plan, class_id)
    execution = {
        "derived_torch_seed": pair_seed,
        "rng_state_sha256": {
            "after_pair_seed_reset": rng_after_seed,
            "after_initial_noise": rng_after_initial,
            "after_250_full_2B_transition_draws": rng_after_diffusion,
        },
        "tensor_sha256": {
            "initial_noise_b1": strict.tensor_sha256(initial),
            "final_latent_kept_b1": strict.tensor_sha256(kept),
            "final_latent_discarded_b1": strict.tensor_sha256(discarded),
            "decoded_sample_b1": strict.tensor_sha256(decoded),
        },
        "transition_randn_like_calls": 250,
        "transition_draw_shape": [2, 4, 32, 32],
        "terminal_t0_draw_consumed_then_masked": True,
    }
    return decoded, arrays, execution


def run_sample(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, _, strict = load_source_lock(source_lock)
    event_lock = Path(source_contract["event_protocol"]["path"])
    protocol = validate_event_protocol(event_lock)
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    key = (args.phase, args.global_seed, args.class_id)
    if key not in set(exact_pairs(plan)):
        raise RuntimeError(f"requested pair is outside immutable dynamic axis: {key}")
    output_root = args.output_root.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pairs" / "calibration").mkdir(parents=True, exist_ok=True)
    (output_root / "pairs" / "confirmation").mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_root / POOL_MANIFEST) or os.path.lexists(
        output_root / POOL_COMPLETION
    ):
        raise RuntimeError("dynamic pool is already finalized; no further sampling is allowed")
    destination = output_root / pair_relative_directory(*key)
    if os.path.lexists(destination):
        record = validate_pair(
            output_root,
            source_lock=source_lock,
            source_contract=source_contract,
            plan=plan,
            phase=args.phase,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        print(json.dumps({"status": "reused", "record": record}, sort_keys=True))
        return
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_grad = torch.is_grad_enabled()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        model, diffusion, vae, save_image = load_models(strict, source_contract, args)
        started = time.time()
        decoded, arrays, execution = sample_one(
            model=model,
            diffusion=diffusion,
            vae=vae,
            strict=strict,
            plan=plan,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        record = publish_pair(
            output_root,
            decoded=decoded,
            arrays=arrays,
            save_image=save_image,
            source_lock=source_lock,
            source_contract=source_contract,
            plan=plan,
            phase=args.phase,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        print(
            json.dumps(
                {
                    "status": "sampled",
                    "elapsed_seconds": round(time.time() - started, 3),
                    "record": record,
                    "execution": execution,
                },
                sort_keys=True,
            )
        )
    finally:
        torch.set_grad_enabled(previous_grad)
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def parse_task_file(path: Path, plan: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    payload = load_json(require_regular(path, "dynamic worker task file"))
    if set(payload) != {
        "schema_version",
        "anchor_plan_identity_sha256",
        "tasks",
        "tasks_sha256",
    }:
        raise RuntimeError("dynamic task-file schema changed")
    rows = payload.get("tasks")
    if (
        payload.get("schema_version") != 1
        or payload.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or not isinstance(rows, list)
        or not rows
        or payload.get("tasks_sha256") != canonical_sha256(rows)
    ):
        raise RuntimeError("dynamic task-file identity changed")
    tasks: list[tuple[str, int, int]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"phase", "global_seed", "class_id"}:
            raise RuntimeError("dynamic task row schema changed")
        if (
            not isinstance(row["phase"], str)
            or type(row["global_seed"]) is not int
            or type(row["class_id"]) is not int
        ):
            raise RuntimeError("dynamic task row types changed")
        tasks.append((row["phase"], row["global_seed"], row["class_id"]))
    if len(set(tasks)) != len(tasks) or not set(tasks) <= set(exact_pairs(plan)):
        raise RuntimeError("dynamic task file is duplicated or outside the immutable axis")
    return tuple(tasks)


def run_tasks(args: argparse.Namespace) -> None:
    """Run an arbitrary immutable-axis shard with one model/VAE load."""

    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, _, strict = load_source_lock(source_lock)
    protocol = validate_event_protocol(Path(source_contract["event_protocol"]["path"]))
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    tasks = parse_task_file(args.tasks, plan)
    output_root = args.output_root.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pairs" / "calibration").mkdir(parents=True, exist_ok=True)
    (output_root / "pairs" / "confirmation").mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_root / POOL_MANIFEST) or os.path.lexists(
        output_root / POOL_COMPLETION
    ):
        raise RuntimeError("dynamic pool is already finalized; no further worker is allowed")
    pending: list[tuple[str, int, int]] = []
    reused = 0
    for phase, seed, class_id in tasks:
        destination = output_root / pair_relative_directory(phase, seed, class_id)
        if os.path.lexists(destination):
            validate_pair(
                output_root,
                source_lock=source_lock,
                source_contract=source_contract,
                plan=plan,
                phase=phase,
                global_seed=seed,
                class_id=class_id,
            )
            reused += 1
        else:
            pending.append((phase, seed, class_id))
    print(
        json.dumps(
            {"task_count": len(tasks), "reused": reused, "pending": len(pending)},
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        return
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_grad = torch.is_grad_enabled()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        model, diffusion, vae, save_image = load_models(strict, source_contract, args)
        for index, (phase, seed, class_id) in enumerate(pending, start=1):
            started = time.time()
            decoded, arrays, execution = sample_one(
                model=model,
                diffusion=diffusion,
                vae=vae,
                strict=strict,
                plan=plan,
                global_seed=seed,
                class_id=class_id,
            )
            record = publish_pair(
                output_root,
                decoded=decoded,
                arrays=arrays,
                save_image=save_image,
                source_lock=source_lock,
                source_contract=source_contract,
                plan=plan,
                phase=phase,
                global_seed=seed,
                class_id=class_id,
            )
            print(
                json.dumps(
                    {
                        "completed": index,
                        "pending_total": len(pending),
                        "elapsed_seconds": round(time.time() - started, 3),
                        "record": record,
                        "execution": execution,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        torch.set_grad_enabled(previous_grad)
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def pool_identity(source_contract: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    pairs = exact_pairs(plan)
    return {
        "schema_version": 1,
        "status": "EVENT_RICH_DYNAMIC_MINIMUM_TRACES_COMPLETE",
        "dynamic_contract_identity_sha256": source_contract["identity_sha256"],
        "event_protocol_identity_sha256": plan["protocol_identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "active_union_classes": plan["active_union_classes"],
        "calibration_seeds": list(range(1100, 1120)),
        "confirmation_seeds": list(range(1200, 1328)),
        "pair_axis_order": "phase calibration then confirmation; seed-major; active-union-class-minor",
        "pair_count": len(pairs),
        "confirmation_pair_count": len(plan["active_union_classes"]) * 128,
        "calibration_pair_count": len(plan["active_union_classes"]) * 20,
        "minimum_internal_observables_only": True,
        "candidate_scores_computed": False,
        "labels_reviews_external_representations_opened": False,
    }


def validate_pool(
    root: Path,
    *,
    source_lock: Path,
    source_contract: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    root = require_directory(root, "dynamic trace pool")
    manifest_path = require_regular(root / POOL_MANIFEST, "dynamic pool manifest")
    completion_path = require_regular(root / POOL_COMPLETION, "dynamic pool completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = pool_identity(source_contract, plan)
    identity_hash = canonical_sha256(identity)
    if (
        manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_hash
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("dynamic trace pool identity changed")
    expected_pairs = exact_pairs(plan)
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or len(rows) != len(expected_pairs):
        raise RuntimeError("dynamic trace pool pair receipt count changed")
    observed: list[dict[str, Any]] = []
    for pair in expected_pairs:
        observed.append(
            validate_pair(
                root,
                source_lock=source_lock,
                source_contract=source_contract,
                plan=plan,
                phase=pair[0],
                global_seed=pair[1],
                class_id=pair[2],
            )
        )
    if rows != observed or manifest.get("pairs_sha256") != canonical_sha256(observed):
        raise RuntimeError("dynamic trace pool pair receipts changed")
    expected_completion = {
        "complete": True,
        "identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "pairs_sha256": canonical_sha256(observed),
        "pair_count": len(observed),
    }
    if completion != expected_completion:
        raise RuntimeError("dynamic trace pool completion changed")
    return manifest


def run_finalize(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, _, _ = load_source_lock(source_lock)
    protocol = validate_event_protocol(Path(source_contract["event_protocol"]["path"]))
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    if not plan["active_union_classes"]:
        raise RuntimeError("both anchor rules are STOP; protocol forbids creating full traces")
    root = require_directory(args.output_root, "dynamic trace pool")
    if os.path.lexists(root / POOL_MANIFEST) or os.path.lexists(root / POOL_COMPLETION):
        validate_pool(
            root,
            source_lock=source_lock,
            source_contract=source_contract,
            plan=plan,
        )
        print("validated existing immutable dynamic trace pool; no overwrite")
        return
    rows = [
        validate_pair(
            root,
            source_lock=source_lock,
            source_contract=source_contract,
            plan=plan,
            phase=phase,
            global_seed=seed,
            class_id=class_id,
        )
        for phase, seed, class_id in exact_pairs(plan)
    ]
    identity = pool_identity(source_contract, plan)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "pairs": rows,
        "pairs_sha256": canonical_sha256(rows),
    }
    write_json_exclusive(root / POOL_MANIFEST, manifest)
    write_json_exclusive(
        root / POOL_COMPLETION,
        {
            "complete": True,
            "identity_sha256": manifest["identity_sha256"],
            "manifest_sha256": sha256_file(root / POOL_MANIFEST),
            "pairs_sha256": manifest["pairs_sha256"],
            "pair_count": len(rows),
        },
    )
    validate_pool(
        root,
        source_lock=source_lock,
        source_contract=source_contract,
        plan=plan,
    )
    print(json.dumps({"status": "complete", "pair_count": len(rows)}, sort_keys=True))


def run_self_test() -> None:
    synthetic_plan = {
        "B_decision": {"go": True, "selected_classes": [1, 2, 3, 4, 5, 6]},
        "C_decision": {"go": True, "selected_classes": [4, 5, 6, 7, 8, 9]},
    }
    b_only = {
        FINAL_LATENT: np.zeros((4, 32, 32), dtype=np.float32),
        B_ARRAY: np.zeros((9, 4, 32, 32), dtype=np.float32),
        B_STEP: np.asarray(B_CHECKPOINTS, dtype=np.int16),
        B_INTERNAL_T: 249 - np.asarray(B_CHECKPOINTS, dtype=np.int16),
    }
    validate_arrays(b_only, synthetic_plan, 1)
    both = dict(b_only)
    both.update(
        {
            C_ARRAY: np.zeros((50, 32, 32), dtype=np.float32),
            C_ALPHA: np.linspace(0.1, 0.9, 50, dtype=np.float64),
            C_STEP: np.asarray(C_CHECKPOINTS, dtype=np.int16),
            C_INTERNAL_T: 249 - np.asarray(C_CHECKPOINTS, dtype=np.int16),
        }
    )
    validate_arrays(both, synthetic_plan, 4)
    poisoned = dict(both)
    poisoned["endpoint_embedding"] = np.zeros(1, dtype=np.float32)
    try:
        validate_arrays(poisoned, synthetic_plan, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("extra/external trace array escaped the minimum schema")
    if derive_pair_seed(1000, 0) != 3026363209052735318:
        raise AssertionError("pair-keyed RNG derivation changed")
    with tempfile.TemporaryDirectory(prefix="event-dynamic-task-selftest-") as raw:
        path = Path(raw) / "tasks.json"
        task_plan = {
            **synthetic_plan,
            "identity_sha256": "a" * 64,
            "active_union_classes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        rows = [
            {"phase": "calibration", "global_seed": 1100, "class_id": 1},
            {"phase": "confirmation", "global_seed": 1200, "class_id": 9},
        ]
        write_json(
            path,
            {
                "schema_version": 1,
                "anchor_plan_identity_sha256": "a" * 64,
                "tasks": rows,
                "tasks_sha256": canonical_sha256(rows),
            },
        )
        if parse_task_file(path, task_plan) != (
            ("calibration", 1100, 1),
            ("confirmation", 1200, 9),
        ):
            raise AssertionError("dynamic task-file parser changed")
    print("self-test passed: dynamic minimum-trace axes and poison rejection; no GPU")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample-one")
    sample.add_argument("--anchor-plan", type=Path, required=True)
    sample.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    sample.add_argument("--global-seed", type=int, required=True)
    sample.add_argument("--class-id", type=int, required=True)
    sample.add_argument("--output-root", type=Path, required=True)
    sample.add_argument("--dit-root", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--vae-snapshot", type=Path, required=True)
    sample.set_defaults(func=run_sample)
    tasks = sub.add_parser("sample-tasks")
    tasks.add_argument("--anchor-plan", type=Path, required=True)
    tasks.add_argument("--tasks", type=Path, required=True)
    tasks.add_argument("--output-root", type=Path, required=True)
    tasks.add_argument("--dit-root", type=Path, required=True)
    tasks.add_argument("--checkpoint", type=Path, required=True)
    tasks.add_argument("--vae-snapshot", type=Path, required=True)
    tasks.set_defaults(func=run_tasks)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--anchor-plan", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.set_defaults(func=run_finalize)
    validate = sub.add_parser("validate")
    validate.add_argument("--anchor-plan", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)
    validate.set_defaults(func=run_finalize)
    test = sub.add_parser("self-test")
    test.set_defaults(func=lambda _args: run_self_test())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

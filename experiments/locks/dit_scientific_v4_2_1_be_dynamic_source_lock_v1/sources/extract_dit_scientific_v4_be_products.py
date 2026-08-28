#!/usr/bin/env python3
"""Create physically isolated label-free B/E and exact-ablation products.

The endpoint side of the trace pool is never opened.  ``B`` emits only the
frozen B persistence/alarm and its internal formation track.  ``E`` emits only
the frozen blur-latched e-process running-max log score, anytime alert, and a
physically isolated confirmation-only mechanics aggregate locked before
labels. ``E_no_gate`` and
``E_first_hit_full_budget`` are separate exact diagnostic artifacts and can
never substitute for a failed co-primary. Labels, endpoints, and external
representations are rejected.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch

try:
    from .calibrate_dit_scientific_v4_be import validate_calibration
    from .dit_scientific_v4_be_contract import (
        B_SCORE,
        CHECKPOINTS,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        E_ALERT,
        E_SCORE,
        METHOD_LOCK_ID,
        canonical_sha256,
        exact_pairs,
        load_json,
        publish_artifact,
        reject_forbidden_method_name,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
    )
    from .sample_dit_scientific_v4_be_traces import (
        load_source_lock,
        load_trace_array_whitelist,
        validate_trace_pool,
    )
except ImportError:
    from calibrate_dit_scientific_v4_be import validate_calibration  # type: ignore
    from dit_scientific_v4_be_contract import (  # type: ignore
        B_SCORE,
        CHECKPOINTS,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        E_ALERT,
        E_SCORE,
        METHOD_LOCK_ID,
        canonical_sha256,
        exact_pairs,
        load_json,
        publish_artifact,
        reject_forbidden_method_name,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
    )
    from sample_dit_scientific_v4_be_traces import (  # type: ignore
        load_source_lock,
        load_trace_array_whitelist,
        validate_trace_pool,
    )


EXTRACTOR = "extract_dit_scientific_v4_be_products"
PRODUCTS = (
    "B",
    "E_mechanics",
    "E",
    "E_no_gate",
    "E_first_hit_full_budget",
    "G_start",
)
REPLAY_BATCH_SIZE = 4
DECODE_BATCH_SIZE = 8
B_TRACE_ARRAYS = ("pred_xstart", "sampling_step", "internal_timestep")
E_TRACE_ARRAYS = (
    "state_before",
    "pred_xstart",
    "p_standard_deviation",
    "transition_innovation",
    "sampling_step",
    "internal_timestep",
    "alpha_bar",
)
E_MECHANICS_ARRAYS = (
    "applied_K",
    "start_time_index",
    "start_remaining_effective_count",
    "frozen_K_per_step_after_start",
    "direction_reused",
    "class_id",
    "effective_nonidentity",
)


def verify_source(manifest: Mapping[str, Any]) -> None:
    by_name = {row["name"]: row for row in manifest["files"]}
    expected = "sources/extract_dit_scientific_v4_be_products.py"
    if by_name.get(expected, {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running extractor differs from frozen source snapshot")


def load_calibration(
    root: Path, *, contract: Mapping[str, Any], protocol: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    reject_forbidden_method_name(root.name, "calibration artifact path")
    manifest, _ = validate_manifest_tree(root)
    if (
        manifest.get("artifact_kind") != "SCIENTIFIC_V4_LABEL_FREE_B_CALIBRATION"
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("labels_reviews_endpoint_or_external_representations_opened") is not False
    ):
        raise RuntimeError("calibration artifact lineage changed")
    payload = load_json(require_regular(root / "calibration.json", "calibration payload"))
    validate_calibration(payload)
    if manifest.get("calibration_identity_sha256") != payload["identity_sha256"]:
        raise RuntimeError("calibration manifest does not bind payload")
    if tuple(row["class_id"] for row in payload["classes"]) != tuple(plan["selected_classes"]):
        raise RuntimeError("calibration classes/order differ from trace plan")
    return payload, manifest


def calibration_arrays(
    payload: Mapping[str, Any], class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = {int(row["class_id"]): row for row in payload["classes"]}
    gates = np.stack(
        [np.asarray(rows[int(value)]["state_gate_threshold_by_checkpoint"], dtype=np.float64) for value in class_ids]
    )
    alarms = np.asarray(
        [float(rows[int(value)]["B_alarm_threshold"]) for value in class_ids], dtype=np.float64
    )
    if gates.shape != (len(class_ids), len(CHECKPOINTS)) or not np.isfinite(gates).all():
        raise RuntimeError("calibration gate matrix is invalid")
    return np.ascontiguousarray(gates), np.ascontiguousarray(alarms)


def _upstream_dit_module_names() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"models", "download", "diffusion"}
        or name.startswith("diffusion.")
    }


def reject_preexisting_upstream_dit_modules() -> None:
    preexisting = _upstream_dit_module_names()
    if preexisting:
        raise RuntimeError(
            "ambiguous pre-imported upstream DiT modules: "
            + repr(sorted(preexisting))
        )


def load_runtime(
    strict: Any, contract: Mapping[str, Any], args: argparse.Namespace, *, need_model: bool
) -> tuple[Any | None, Any, Any | None]:
    if need_model:
        reject_preexisting_upstream_dit_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B/E product extraction")
    vae_snapshot = require_directory(args.vae_snapshot, "VAE snapshot")
    if strict.validate_vae_snapshot(vae_snapshot) != contract["assets"]["vae_snapshot"]:
        raise RuntimeError("runtime VAE differs from frozen asset")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        str(vae_snapshot), local_files_only=True, use_safetensors=True
    ).to(device="cuda", dtype=torch.float32).eval()
    if not need_model:
        return None, vae, None
    dit_root = require_directory(args.dit_root, "DiT repository")
    checkpoint = require_regular(args.checkpoint, "DiT checkpoint")
    if (
        strict.validate_repository(dit_root, checkpoint) != contract["assets"]["dit_repository"]
        or strict.validate_checkpoint(checkpoint) != contract["assets"]["checkpoint"]
    ):
        raise RuntimeError("runtime DiT repository/checkpoint differs from frozen assets")
    strict.ensure_single_process()
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
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
            raise RuntimeError(
                f"upstream DiT import shadowing detected: {imported} != {expected}"
            )
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to("cuda")
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for name in sorted(_upstream_dit_module_names(), reverse=True):
            sys.modules.pop(name, None)
    return model, vae, diffusion


def decode_drafts(vae: Any, pred: np.ndarray, strict: Any, batch_size: int) -> np.ndarray:
    flat = pred.reshape(-1, 4, 32, 32)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            latent = torch.from_numpy(np.ascontiguousarray(flat[start : start + batch_size])).to(
                device="cuda", dtype=torch.float32
            )
            decoded = vae.decode(latent / strict.VAE_SCALING_FACTOR).sample
            rows.append(((decoded + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(
        np.concatenate(rows).reshape(pred.shape[0], len(CHECKPOINTS), 3, 256, 256)
    )


def build_observation(
    *,
    arrays: Mapping[str, np.ndarray],
    class_ids: np.ndarray,
    gate: np.ndarray,
    alarm: np.ndarray,
    model: Any,
    vae: Any,
    diffusion: Any,
    strict: Any,
    core: Any,
    decode_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    batch = len(class_ids)
    state_np = arrays["state_before"]
    pred_np = arrays["pred_xstart"]
    sigma = arrays["p_standard_deviation"]
    innovation = arrays["transition_innovation"]
    if state_np.shape != (batch, len(CHECKPOINTS), 4, 32, 32):
        raise RuntimeError("replay batch state axis changed")
    alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    expected_alpha = alpha[arrays["internal_timestep"].astype(np.int64)]
    if not np.array_equal(arrays["alpha_bar"], expected_alpha):
        raise RuntimeError("runtime diffusion schedule differs from trace")
    theta = np.zeros(
        (batch, len(core.HEAT_SHIFTS), len(CHECKPOINTS), 4, 32, 32), dtype=np.float64
    )
    y = torch.cat(
        [
            torch.from_numpy(class_ids.astype(np.int64)).to("cuda"),
            torch.full((batch,), strict.NULL_CLASS_ID, device="cuda", dtype=torch.long),
        ]
    )
    maximum_reconstruction_error = 0.0
    rng_before = strict.cuda_rng_state_sha256()
    with torch.inference_mode():
        for checkpoint_index in range(len(CHECKPOINTS)):
            internal_t = int(arrays["internal_timestep"][0, checkpoint_index])
            if not np.all(arrays["internal_timestep"][:, checkpoint_index] == internal_t):
                raise RuntimeError("replay batch internal timestep mismatch")
            state_first = torch.from_numpy(
                np.ascontiguousarray(state_np[:, checkpoint_index])
            ).to(device="cuda", dtype=torch.float32)
            state = torch.cat((state_first, state_first), dim=0)
            current_t = torch.full(
                (2 * batch,), int(timestep_map[internal_t]), device="cuda", dtype=torch.long
            )
            current = model.forward_with_cfg(
                state, current_t, y=y, cfg_scale=strict.CFG_SCALE
            )[:batch, :4]
            current_np = current.cpu().numpy().astype(np.float64)
            alpha_current = float(alpha[internal_t])
            reconstructed = (
                state_np[:, checkpoint_index].astype(np.float64)
                - math.sqrt(1.0 - alpha_current) * current_np
            ) / math.sqrt(alpha_current)
            error = float(np.max(np.abs(reconstructed - pred_np[:, checkpoint_index].astype(np.float64))))
            maximum_reconstruction_error = max(maximum_reconstruction_error, error)
            if error > 5e-4:
                raise RuntimeError("replayed current epsilon does not reconstruct saved pred_xstart")
            for scale_index, shifted_row in enumerate(core.SHIFTED_INTERNAL_TIMESTEPS):
                shifted_internal = int(shifted_row[checkpoint_index])
                if shifted_internal == internal_t:
                    continue
                alpha_shifted = float(alpha[shifted_internal])
                rho = math.sqrt(alpha_shifted / alpha_current)
                shifted_t = torch.full(
                    (2 * batch,),
                    int(timestep_map[shifted_internal]),
                    device="cuda",
                    dtype=torch.long,
                )
                shifted = model.forward_with_cfg(
                    state * rho, shifted_t, y=y, cfg_scale=strict.CFG_SCALE
                )[:batch, :4]
                shifted_np = shifted.cpu().numpy().astype(np.float64)
                theta[:, scale_index, checkpoint_index] = (
                    -rho * shifted_np / math.sqrt(1.0 - alpha_shifted)
                    + current_np / math.sqrt(1.0 - alpha_current)
                )
        decoded = decode_drafts(vae, pred_np, strict, decode_batch_size)
    rng_after = strict.cuda_rng_state_sha256()
    if rng_before != rng_after:
        raise RuntimeError("shifted DiT/VAE replay consumed CUDA RNG")
    observed = {
        "decoded_pred_xstart_rgb": decoded,
        "theta": np.ascontiguousarray(theta, dtype=np.float64),
        "p_standard_deviation": np.ascontiguousarray(sigma, dtype=np.float32),
        "transition_innovation": np.ascontiguousarray(innovation, dtype=np.float32),
        "sampling_step": np.asarray(CHECKPOINTS, dtype=np.int16),
        "shifted_internal_timestep": np.asarray(core.SHIFTED_INTERNAL_TIMESTEPS, dtype=np.int16),
        "heat_shift": np.asarray(core.HEAT_SHIFTS, dtype=np.float64),
        "effective_nonidentity": np.asarray(core.EFFECTIVE_NONIDENTITY, dtype=np.uint8),
        "blur_gate_threshold": np.ascontiguousarray(gate, dtype=np.float64),
        "blur_score_threshold": np.ascontiguousarray(alarm, dtype=np.float64),
        "class_id": np.ascontiguousarray(class_ids, dtype=np.int16),
    }
    core.validate_observer_input(observed)
    return observed, {
        "cuda_rng_unchanged": True,
        "maximum_current_epsilon_pred_xstart_reconstruction_error": maximum_reconstruction_error,
    }


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run_G_start(
    args: argparse.Namespace,
    *,
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    core: Any,
) -> None:
    """Publish an innovation-free score from the frozen start metadata only."""

    mechanics_root = require_directory(args.mechanics_product, "E mechanics product")
    expected_root = {
        "manifest.json",
        "completion.json",
        "internal_tracks.npz",
        "label_free_mechanics_audit.json",
    }
    members = list(mechanics_root.iterdir())
    if (
        {path.name for path in members} != expected_root
        or len(members) != len(expected_root)
        or any(path.is_symlink() or not path.is_file() for path in members)
    ):
        raise RuntimeError("G_start source is not the exact flat E mechanics tree")
    manifest, _ = validate_manifest_tree(mechanics_root)
    expected_axis = exact_pairs(plan, phases=("confirmation",))
    expected_axis_sha256 = canonical_sha256(
        [
            {"phase": phase, "global_seed": seed, "class_id": class_id}
            for phase, seed, class_id in expected_axis
        ]
    )
    if (
        manifest.get("artifact_kind") != "SCIENTIFIC_V4_E_MECHANICS_LABEL_FREE_PRODUCT"
        or manifest.get("product") != "E_mechanics"
        or manifest.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("row_count") != len(expected_axis)
        or manifest.get("ordered_pair_axis_sha256") != expected_axis_sha256
        or manifest.get("endpoint_images_or_envelopes_opened") is not False
        or manifest.get("labels_reviews_consensus_opened") is not False
        or manifest.get("FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened")
        is not False
    ):
        raise RuntimeError("G_start source mechanics product lineage changed")
    track_path = require_regular(
        mechanics_root / "internal_tracks.npz", "E mechanics internal tracks"
    )
    with np.load(track_path, allow_pickle=False) as archive:
        needed = (
            "start_time_index",
            "start_remaining_effective_count",
            "class_id",
        )
        if tuple(archive.files) != E_MECHANICS_ARRAYS:
            raise RuntimeError("G_start refuses non-predictable/extra mechanics arrays")
        start = np.ascontiguousarray(archive[needed[0]])
        remaining = np.ascontiguousarray(archive[needed[1]])
        class_id = np.ascontiguousarray(archive[needed[2]])
    if (
        start.dtype != np.int16
        or remaining.dtype != np.int16
        or start.shape != (len(expected_axis), 2)
        or remaining.shape != start.shape
        or class_id.shape != (len(expected_axis),)
        or class_id.dtype != np.int16
        or not np.array_equal(
            class_id, np.asarray([row[2] for row in expected_axis], dtype=np.int16)
        )
        or tuple(core.EFFECTIVE_STEP_COUNT_PER_SCALE) != (5, 8)
        or tuple(core.MIXTURE_WEIGHTS) != (0.5, 0.5)
    ):
        raise RuntimeError("G_start frozen start-score inputs changed")
    active = start >= 0
    score = np.sum(
        active
        * remaining.astype(np.float64)
        / np.asarray((5.0, 8.0), dtype=np.float64)[None, :]
        * np.asarray((0.5, 0.5), dtype=np.float64)[None, :],
        axis=1,
    )
    if not np.array_equal(score, core.gate_only_start_schedule_score_from_metadata(start, remaining)):
        raise RuntimeError("G_start formula differs from frozen method-v2 core")
    rows = [
        {
            "phase": phase,
            "global_seed": seed,
            "class_id": class_value,
            "G_start_schedule_diagnostic": float(score[index]),
        }
        for index, (phase, seed, class_value) in enumerate(expected_axis)
    ]
    publish_artifact(
        args.output,
        artifact_kind="SCIENTIFIC_V4_G_START_SCHEDULE_LABEL_FREE_PRODUCT",
        payloads={"scores.csv": csv_text(rows)},
        manifest_fields={
            "product": "G_start",
            "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "trace_pool_identity_sha256": manifest["trace_pool_identity_sha256"],
            "calibration_artifact_identity_sha256": manifest[
                "calibration_artifact_identity_sha256"
            ],
            "ordered_pair_axis_sha256": expected_axis_sha256,
            "mechanics_product_manifest_identity_sha256": manifest[
                "identity_sha256"
            ],
            "mechanics_product_manifest_file_sha256": sha256_file(
                mechanics_root / "manifest.json"
            ),
            "row_count": len(rows),
            "loaded_mechanics_array_names_exactly": list(needed),
            "transition_innovation_or_eprocess_increment_opened": False,
            "formula": "0.5*I(T_Delta1>=0)*h_Delta1/5 + 0.5*I(T_Delta4>=0)*h_Delta4/8",
            "endpoint_images_or_envelopes_opened": False,
            "labels_reviews_consensus_opened": False,
            "FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened": False,
            "intervention_or_rollback_performed": False,
        },
    )
    print(json.dumps({"product": "G_start", "path_count": len(rows)}, sort_keys=True))


def run(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, strict, core = load_source_lock(source_lock)
    verify_source(source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready; freeze a later activation lock")
    if (
        getattr(core, "SCHEMA_VERSION", None) != 2
        or not hasattr(core, "label_free_path_mechanics_audit")
    ):
        raise RuntimeError("dynamic source lock does not contain method-v2 observer core")
    validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    if args.product == "G_start":
        run_G_start(
            args,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
            core=core,
        )
        return
    trace_manifest, _ = validate_trace_pool(
        args.trace_pool, contract=contract, protocol=protocol, plan=plan
    )
    calibration, calibration_manifest = load_calibration(
        args.calibration, contract=contract, protocol=protocol, plan=plan
    )
    need_model = args.product != "B"
    model, vae, diffusion = load_runtime(strict, contract, args, need_model=need_model)
    all_rows: list[dict[str, Any]] = []
    track_chunks: dict[str, list[np.ndarray]] = {}
    # Calibration paths fit only the predictable B thresholds. Every method
    # product and the v2 mechanics audit use the fresh 768 confirmation paths.
    ordered_pairs = exact_pairs(plan, phases=("confirmation",))
    loaded_records: list[dict[str, Any]] = []
    observation_receipts: list[dict[str, Any]] = []
    for start in range(0, len(ordered_pairs), REPLAY_BATCH_SIZE):
        pairs = ordered_pairs[start : start + REPLAY_BATCH_SIZE]
        names = B_TRACE_ARRAYS if args.product == "B" else E_TRACE_ARRAYS
        per_pair = []
        for phase, seed, class_id in pairs:
            arrays, record = load_trace_array_whitelist(
                args.trace_pool,
                contract=contract,
                plan=plan,
                phase=phase,
                global_seed=seed,
                class_id=class_id,
                names=names,
            )
            per_pair.append(arrays)
            loaded_records.append(record)
        combined = {
            name: np.stack([row[name] for row in per_pair])
            for name in names
        }
        class_ids = np.asarray([pair[2] for pair in pairs], dtype=np.int16)
        gate, alarm = calibration_arrays(calibration, class_ids)
        if args.product == "B":
            rng_before = strict.cuda_rng_state_sha256()
            decoded = decode_drafts(vae, combined["pred_xstart"], strict, DECODE_BATCH_SIZE)
            if strict.cuda_rng_state_sha256() != rng_before:
                raise RuntimeError("B temporary VAE decode consumed CUDA RNG")
            blur = core.compute_blur_tiles(decoded)
            persistence, slope = core.blur_summary_tracks(blur.severity)
            alerts = persistence > alarm
            for index, (phase, seed, class_id) in enumerate(pairs):
                all_rows.append(
                    {
                        "phase": phase,
                        "global_seed": seed,
                        "class_id": class_id,
                        B_SCORE: float(persistence[index]),
                        "B_alarm": int(alerts[index]),
                    }
                )
            chunk_tracks = {
                "blur_severity": blur.severity,
                "B_formation_slope_diagnostic": slope,
            }
            observation_receipts.append({"cuda_rng_unchanged": True})
        else:
            assert model is not None and diffusion is not None
            observed, receipt = build_observation(
                arrays=combined,
                class_ids=class_ids,
                gate=gate,
                alarm=alarm,
                model=model,
                vae=vae,
                diffusion=diffusion,
                strict=strict,
                core=core,
                decode_batch_size=DECODE_BATCH_SIZE,
            )
            # Compute exactly one E branch.  In particular the no-gate ablation
            # is neither computed nor present in the co-primary E artifact.
            blur = core.compute_blur_tiles(observed["decoded_pred_xstart_rgb"])
            tracks = core.compute_eprocess_tracks(
                theta=observed["theta"],
                p_standard_deviation=observed["p_standard_deviation"],
                transition_innovation=observed["transition_innovation"],
                local_mask=blur.latent_mask,
                blur_severity=blur.severity,
                blur_gate_threshold=observed["blur_gate_threshold"],
                effective_nonidentity=observed["effective_nonidentity"],
                use_state_gate=args.product != "E_no_gate",
                one_shot_full_budget=args.product == "E_first_hit_full_budget",
            )
            score_names = {
                "E": (E_SCORE, E_ALERT),
                "E_no_gate": (
                    "E_no_state_gate_running_max_log",
                    "E_no_state_gate_alarm",
                ),
                "E_first_hit_full_budget": (
                    "E_first_hit_full_budget_running_max_log",
                    "E_first_hit_full_budget_alarm",
                ),
            }
            if args.product != "E_mechanics":
                score_name, alert_name = score_names[args.product]
                for index, (phase, seed, class_id) in enumerate(pairs):
                    all_rows.append(
                        {
                            "phase": phase,
                            "global_seed": seed,
                            "class_id": class_id,
                            score_name: float(
                                np.max(tracks.running_max_log_e[index], initial=0.0)
                            ),
                            alert_name: int(tracks.alarm[index]),
                        }
                    )
            effective = np.asarray(core.EFFECTIVE_NONIDENTITY, dtype=bool)
            state_open = blur.severity > gate
            gate_by_scale = np.stack(
                [np.any(state_open[:, effective[scale]], axis=1) for scale in range(len(core.HEAT_SHIFTS))],
                axis=1,
            )
            chunk_tracks = {
                "applied_K": tracks.applied_K,
                "total_K_by_scale": np.sum(tracks.applied_K, axis=2),
                "gate_open_by_scale": gate_by_scale.astype(np.uint8),
                "positive_K_step_count_by_scale": np.sum(
                    tracks.applied_K > 0.0, axis=2
                ).astype(np.int16),
                "eligible_nonidentity_step_count_by_scale": np.broadcast_to(
                    np.sum(effective, axis=1, dtype=np.int16)[None, :],
                    (len(pairs), len(core.HEAT_SHIFTS)),
                ).copy(),
                "start_time_index": tracks.start_time_index,
                "start_remaining_effective_count": tracks.start_remaining_effective_count,
                "frozen_K_per_step_after_start": tracks.frozen_K_per_step_after_start,
                "direction_reused": tracks.direction_reused,
                "class_id": np.ascontiguousarray(class_ids, dtype=np.int16),
            }
            total_K_chunk = chunk_tracks["total_K_by_scale"]
            maximum_K_chunk = np.max(tracks.applied_K, axis=2)
            chunk_tracks["maximum_single_step_K_fraction_by_scale"] = np.divide(
                maximum_K_chunk,
                total_K_chunk,
                out=np.zeros_like(maximum_K_chunk),
                where=total_K_chunk > 0.0,
            )
            if args.product != "E_mechanics":
                chunk_tracks.update({
                    "component_running_max_log_e": np.maximum.accumulate(
                        np.concatenate(
                            [
                                np.zeros((len(pairs), len(core.HEAT_SHIFTS), 1)),
                                tracks.component_log_e,
                            ],
                            axis=2,
                        ),
                        axis=2,
                    )[:, :, 1:],
                    "mixture_running_max_log_e": tracks.running_max_log_e,
                })
            receipt["batch_pair_keys_sha256"] = canonical_sha256(
                [{"phase": p[0], "global_seed": p[1], "class_id": p[2]} for p in pairs]
            )
            observation_receipts.append(receipt)
        for name, value in chunk_tracks.items():
            track_chunks.setdefault(name, []).append(np.ascontiguousarray(value))
    tracks = {name: np.ascontiguousarray(np.concatenate(values, axis=0)) for name, values in track_chunks.items()}
    if args.product == "E_mechanics":
        mechanics_names = E_MECHANICS_ARRAYS[:-1]
        tracks = {name: tracks[name] for name in mechanics_names}
        tracks["effective_nonidentity"] = np.asarray(
            core.EFFECTIVE_NONIDENTITY, dtype=np.uint8
        )
    payloads: dict[str, bytes | str] = {"internal_tracks.npz": npz_bytes(tracks)}
    if args.product != "E_mechanics":
        payloads["scores.csv"] = csv_text(all_rows)
    manifest_fields: dict[str, Any] = {
        "product": args.product,
        "dynamic_source_lock_manifest_identity_sha256": source_manifest[
            "identity_sha256"
        ],
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": protocol["identity_sha256"],
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "trace_pool_identity_sha256": trace_manifest["identity_sha256"],
        "calibration_artifact_identity_sha256": calibration_manifest["identity_sha256"],
        "row_count": len(ordered_pairs),
        "ordered_pair_axis_sha256": canonical_sha256(
            [{"phase": p[0], "global_seed": p[1], "class_id": p[2]} for p in ordered_pairs]
        ),
        "loaded_trace_array_names_exactly": list(
            B_TRACE_ARRAYS if args.product == "B" else E_TRACE_ARRAYS
        ),
        "loaded_trace_records_identity_sha256": canonical_sha256(loaded_records),
        "observation_receipts_identity_sha256": canonical_sha256(observation_receipts),
        "frozen_replay_batch_size": REPLAY_BATCH_SIZE,
        "frozen_VAE_decode_batch_size": DECODE_BATCH_SIZE,
        "endpoint_images_or_envelopes_opened": False,
        "labels_reviews_consensus_opened": False,
        "FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened": False,
        "intervention_or_rollback_performed": False,
    }
    if args.product == "E_mechanics":
        manifest_fields["mechanics_track_array_records"] = {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "raw_sha256": sha256_array(value),
            }
            for name, value in tracks.items()
        }
        mechanics = core.label_free_path_mechanics_audit(
            applied_K=tracks["applied_K"],
            direction_reused=tracks["direction_reused"],
            start_time_index=tracks["start_time_index"],
            start_remaining_effective_count=tracks[
                "start_remaining_effective_count"
            ],
            class_id=tracks["class_id"],
            effective_nonidentity=np.asarray(
                core.EFFECTIVE_NONIDENTITY, dtype=np.uint8
            ),
        )
        if (
            mechanics.get("sample_count") != len(ordered_pairs)
            or len(ordered_pairs) != 6 * len(CONFIRMATION_SEEDS)
            or mechanics.get("labels_endpoint_images_external_representations_used")
            is not False
            or mechanics.get("quality_or_power_interpretation") is not False
        ):
            raise RuntimeError("method-v2 confirmation mechanics audit schema changed")
        mechanics = {
            **mechanics,
            "phase": "confirmation",
            "ordered_pair_axis_sha256": manifest_fields[
                "ordered_pair_axis_sha256"
            ],
            "calibration_thresholds_fitted_on_these_paths": False,
            "confirmation_labels_scores_endpoints_or_external_representations_opened": False,
            "decision_not_made_by_product_extractor": True,
        }
        mechanics["identity_sha256"] = canonical_sha256(mechanics)
        payloads["label_free_mechanics_audit.json"] = json.dumps(
            mechanics, indent=2, sort_keys=True
        ) + "\n"
        manifest_fields["label_free_mechanics_audit_identity_sha256"] = mechanics[
            "identity_sha256"
        ]
    publish_artifact(
        args.output,
        artifact_kind=(
            "SCIENTIFIC_V4_E_MECHANICS_LABEL_FREE_PRODUCT"
            if args.product == "E_mechanics"
            else f"SCIENTIFIC_V4_{args.product}_LABEL_FREE_PRODUCT"
        ),
        payloads=payloads,
        manifest_fields=manifest_fields,
    )
    print(json.dumps({"product": args.product, "path_count": len(ordered_pairs)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--product", choices=PRODUCTS, required=True)
    result.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    result.add_argument("--trace-plan", type=Path, required=True)
    result.add_argument("--trace-pool", type=Path)
    result.add_argument("--calibration", type=Path)
    result.add_argument("--vae-snapshot", type=Path)
    result.add_argument("--mechanics-product", type=Path)
    result.add_argument("--dit-root", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.product == "G_start":
        if args.mechanics_product is None:
            raise RuntimeError("G_start extraction requires --mechanics-product")
        if any(
            value is not None
            for value in (
                args.trace_pool,
                args.calibration,
                args.vae_snapshot,
                args.dit_root,
                args.checkpoint,
            )
        ):
            raise RuntimeError(
                "G_start accepts only frozen start metadata, not traces/models/VAE"
            )
    elif any(
        value is None for value in (args.trace_pool, args.calibration, args.vae_snapshot)
    ):
        raise RuntimeError("B/E extraction requires trace pool, calibration, and VAE")
    elif args.product != "B" and (args.dit_root is None or args.checkpoint is None):
        raise RuntimeError("E/diagnostic extraction requires --dit-root and --checkpoint")
    elif args.mechanics_product is not None:
        raise RuntimeError("only G_start may receive --mechanics-product")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

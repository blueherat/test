#!/usr/bin/env python3
"""Retrospective, label-firewalled method-v2.2 pilot on the old DiT pool.

This script has two deliberately separate modes:

``score``
    Reads only completed preterminal DiT traces, uses seeds 250..269 to freeze
    the method-v2.2 B thresholds, and computes B/E plus the three prespecified
    diagnostics on a disjoint contiguous seed shard.  It never opens endpoint
    images, labels, reviews, FID, Inception, DINO, CLIP, or embeddings.

``evaluate``
    Validates one or more immutable score shards and only then joins the old
    frozen consensus rows.  The result is explicitly retrospective and cannot
    authorize rollback or substitute for the prospective v4.2.1 experiment.

The point of this pilot is speed: the old pool already contains full baseline
transitions, so only the 13 nonidentity shifted-DiT evaluations and temporary
pred-xstart VAE decodes are needed for each three-class seed batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import observe_dit_blur_focused_eprocess_v2 as core
    from . import replay_dit_blur_focused_eprocess_inputs as replay_v1
    from . import reproduce_dit_imagenet256 as strict
except ImportError:  # pragma: no cover - direct CLI execution
    import observe_dit_blur_focused_eprocess_v2 as core
    import replay_dit_blur_focused_eprocess_inputs as replay_v1
    import reproduce_dit_imagenet256 as strict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
DEFAULT_LABELS = (
    ROOT
    / "experiments/annotations/dit_bad_good_third_pool_consensus_lock_v1/consensus_rows.csv"
)
DEFAULT_DIT_ROOT = Path("/data/users/zhoushunyu/eqvae/baselines/DiT")
DEFAULT_CHECKPOINT = DEFAULT_DIT_ROOT / "pretrained_models/DiT-XL-2-256x256.pt"
DEFAULT_VAE = Path(
    "/home/zhoushunyu/.cache/huggingface/hub/"
    "models--stabilityai--sd-vae-ft-mse/snapshots/"
    "31f26fdeee1355a5c34592e401dd41e45d25a493"
)
METHOD_LOCK_ID = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
EXPECTED_CLASSES = (207, 602, 795)
CALIBRATION_SEEDS = tuple(range(250, 270))
EVALUATION_SEED_MIN = 270
EVALUATION_SEED_MAX_EXCLUSIVE = 850
SCORE_COLUMNS = (
    "global_seed",
    "class_id",
    "B_persistence",
    "B_alarm",
    "E_blur_gated_running_max_log",
    "E_blur_gated_alarm",
    "E_no_state_gate_running_max_log",
    "E_first_hit_full_budget_running_max_log",
    "G_start_schedule_diagnostic",
    "T_delta1",
    "h_delta1",
    "T_delta4",
    "h_delta4",
    "fallback_steps_delta1",
    "fallback_steps_delta4",
)


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def trace_dir(trace_root: Path, seed: int) -> Path:
    return trace_root / f"third_pool_v1_seed{seed}"


def load_seed(
    trace_root: Path, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, metadata = replay_v1._read_trace_preterminal(trace_dir(trace_root, seed))
    if metadata["global_seed"] != seed or tuple(metadata["classes"]) != EXPECTED_CLASSES:
        raise RuntimeError(f"old-pool trace axis changed at seed {seed}")
    return arrays, metadata


def combine_seed_arrays(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot combine an empty trace batch")
    batch_members = (
        "state_before",
        "pred_xstart",
        "p_standard_deviation",
        "transition_innovation",
    )
    result = {
        name: np.ascontiguousarray(np.concatenate([row[name] for row in rows], axis=0))
        for name in batch_members
    }
    sample_count = result["state_before"].shape[0]
    result["internal_timestep"] = np.broadcast_to(
        rows[0]["internal_timestep"][None, :],
        (sample_count, len(core.CHECKPOINTS)),
    ).copy()
    result["alpha_bar"] = np.broadcast_to(
        rows[0]["alpha_bar"][None, :],
        (sample_count, len(core.CHECKPOINTS)),
    ).copy()
    for row in rows[1:]:
        if not np.array_equal(row["internal_timestep"], rows[0]["internal_timestep"]):
            raise RuntimeError("trace timestep axes differ across seeds")
        if not np.array_equal(row["alpha_bar"], rows[0]["alpha_bar"]):
            raise RuntimeError("trace alpha schedules differ across seeds")
    return result


def load_runtime(
    metadata: Mapping[str, Any],
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
) -> tuple[Any, Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retrospective shifted replay")
    strict.ensure_single_process()
    replay_v1._validate_runtime_lineage(metadata, dit_root, checkpoint, vae_snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    prior_cwd = Path.cwd()
    prior_path = list(sys.path)
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
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to("cuda")
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        vae = AutoencoderKL.from_pretrained(
            str(vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device="cuda", dtype=torch.float32).eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        return model, vae, diffusion
    finally:
        os.chdir(prior_cwd)
        sys.path[:] = prior_path


def decode_drafts(vae: Any, pred: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    flat = np.ascontiguousarray(pred.reshape(-1, 4, 32, 32), dtype=np.float32)
    decoded: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            latent = torch.from_numpy(flat[start : start + batch_size]).to(
                device="cuda", dtype=torch.float32
            )
            rgb = (
                (vae.decode(latent / strict.VAE_SCALING_FACTOR).sample + 1.0) / 2.0
            ).clamp(0.0, 1.0)
            decoded.append(rgb.cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(
        np.concatenate(decoded, axis=0).reshape(
            pred.shape[0], len(core.CHECKPOINTS), 3, strict.IMAGE_SIZE, strict.IMAGE_SIZE
        )
    )


def calibrate_thresholds(
    trace_root: Path, vae: Any, decode_batch_size: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays: list[dict[str, np.ndarray]] = []
    trace_records: list[dict[str, Any]] = []
    for seed in CALIBRATION_SEEDS:
        row, metadata = load_seed(trace_root, seed)
        arrays.append(row)
        trace_records.append(
            {
                "global_seed": seed,
                "trace_manifest_sha256": metadata["trace_manifest_sha256"],
                "trace_archive_sha256": metadata["trace_archive_sha256"],
            }
        )
    combined = combine_seed_arrays(arrays)
    rng_before = strict.cuda_rng_state_sha256()
    decoded = decode_drafts(vae, combined["pred_xstart"], decode_batch_size)
    if strict.cuda_rng_state_sha256() != rng_before:
        raise RuntimeError("calibration VAE decode consumed CUDA randomness")
    severity = core.compute_blur_tiles(decoded).severity.reshape(
        len(CALIBRATION_SEEDS), len(EXPECTED_CLASSES), len(core.CHECKPOINTS)
    )
    # Exactly the frozen 17th/20 checkpoint rule and 19th/20 persistence rule.
    gate = np.sort(severity, axis=0)[16]
    persistence = np.mean(severity, axis=2)
    alarm = np.sort(persistence, axis=0)[18]
    payload: dict[str, Any] = {
        "status": "RETROSPECTIVE_LABEL_FREE_B_CALIBRATION_ONLY",
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "classes_ordered": list(EXPECTED_CLASSES),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "gate_rule": "17th ascending of 20 per class/checkpoint; strict greater",
        "alarm_rule": "19th ascending of 20 per class B_persistence; strict greater",
        "blur_gate_threshold_by_class_checkpoint": gate.tolist(),
        "B_alarm_threshold_by_class": alarm.tolist(),
        "trace_records": trace_records,
        "labels_reviews_endpoints_external_representations_opened": False,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return np.ascontiguousarray(gate), np.ascontiguousarray(alarm), payload


def build_observation(
    arrays: Mapping[str, np.ndarray],
    class_ids: np.ndarray,
    gate: np.ndarray,
    alarm: np.ndarray,
    model: Any,
    vae: Any,
    diffusion: Any,
    decode_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    batch = len(class_ids)
    alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    if not np.array_equal(
        arrays["alpha_bar"], alpha[arrays["internal_timestep"].astype(np.int64)]
    ):
        raise RuntimeError("runtime diffusion schedule differs from saved traces")
    theta = np.zeros(
        (batch, len(core.HEAT_SHIFTS), len(core.CHECKPOINTS), 4, 32, 32),
        dtype=np.float64,
    )
    y = torch.cat(
        (
            torch.from_numpy(class_ids.astype(np.int64)).to("cuda"),
            torch.full((batch,), strict.NULL_CLASS_ID, device="cuda", dtype=torch.long),
        )
    )
    maximum_error = 0.0
    rng_before = strict.cuda_rng_state_sha256()
    with torch.inference_mode():
        for checkpoint_index in range(len(core.CHECKPOINTS)):
            internal_t = int(arrays["internal_timestep"][0, checkpoint_index])
            if not np.all(arrays["internal_timestep"][:, checkpoint_index] == internal_t):
                raise RuntimeError("batched internal timestep mismatch")
            state_first = torch.from_numpy(
                np.ascontiguousarray(arrays["state_before"][:, checkpoint_index])
            ).to(device="cuda", dtype=torch.float32)
            state = torch.cat((state_first, state_first), dim=0)
            current_t = torch.full(
                (2 * batch,),
                int(timestep_map[internal_t]),
                device="cuda",
                dtype=torch.long,
            )
            current = model.forward_with_cfg(
                state, current_t, y=y, cfg_scale=strict.CFG_SCALE
            )[:batch, :4]
            current_np = current.cpu().numpy().astype(np.float64)
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
            maximum_error = max(maximum_error, error)
            if error > 5e-4:
                raise RuntimeError("current replay does not reconstruct saved pred_xstart")
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
                theta[:, scale_index, checkpoint_index] = (
                    -rho
                    * shifted.cpu().numpy().astype(np.float64)
                    / math.sqrt(1.0 - alpha_shifted)
                    + current_np / math.sqrt(1.0 - alpha_current)
                )
        decoded = decode_drafts(vae, arrays["pred_xstart"], decode_batch_size)
    if strict.cuda_rng_state_sha256() != rng_before:
        raise RuntimeError("shifted replay or VAE observation consumed CUDA randomness")
    observed = {
        "decoded_pred_xstart_rgb": decoded,
        "theta": np.ascontiguousarray(theta),
        "p_standard_deviation": np.ascontiguousarray(
            arrays["p_standard_deviation"], dtype=np.float32
        ),
        "transition_innovation": np.ascontiguousarray(
            arrays["transition_innovation"], dtype=np.float32
        ),
        "sampling_step": np.asarray(core.CHECKPOINTS, dtype=np.int16),
        "shifted_internal_timestep": np.asarray(
            core.SHIFTED_INTERNAL_TIMESTEPS, dtype=np.int16
        ),
        "heat_shift": np.asarray(core.HEAT_SHIFTS, dtype=np.float64),
        "effective_nonidentity": np.asarray(
            core.EFFECTIVE_NONIDENTITY, dtype=np.uint8
        ),
        "blur_gate_threshold": np.ascontiguousarray(gate, dtype=np.float64),
        "blur_score_threshold": np.ascontiguousarray(alarm, dtype=np.float64),
        "class_id": np.ascontiguousarray(class_ids, dtype=np.int16),
    }
    core.validate_observer_input(observed)
    return observed, {
        "cuda_rng_unchanged": True,
        "maximum_current_reconstruction_error": maximum_error,
    }


def score_rows(
    observed: Mapping[str, np.ndarray], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    blur = core.compute_blur_tiles(observed["decoded_pred_xstart_rgb"])
    persistence, _ = core.blur_summary_tracks(blur.severity)
    common = {
        "theta": observed["theta"],
        "p_standard_deviation": observed["p_standard_deviation"],
        "transition_innovation": observed["transition_innovation"],
        "local_mask": blur.latent_mask,
        "blur_severity": blur.severity,
        "blur_gate_threshold": observed["blur_gate_threshold"],
        "effective_nonidentity": observed["effective_nonidentity"],
    }
    gated = core.compute_eprocess_tracks(**common, use_state_gate=True)
    no_gate = core.compute_eprocess_tracks(**common, use_state_gate=False)
    one_shot = core.compute_eprocess_tracks(
        **common, use_state_gate=True, one_shot_full_budget=True
    )
    e_score = np.max(gated.running_max_log_e, axis=1)
    no_gate_score = np.max(no_gate.running_max_log_e, axis=1)
    one_shot_score = np.max(one_shot.running_max_log_e, axis=1)
    start = gated.start_time_index
    h = gated.start_remaining_effective_count
    g_score = 0.5 * (start[:, 0] >= 0) * h[:, 0] / 5.0 + 0.5 * (
        start[:, 1] >= 0
    ) * h[:, 1] / 8.0
    rows: list[dict[str, Any]] = []
    class_ids = observed["class_id"].astype(int)
    repeated_seeds = np.repeat(np.asarray(seeds, dtype=np.int64), len(EXPECTED_CLASSES))
    if len(repeated_seeds) != len(class_ids):
        raise RuntimeError("seed/class row axis changed")
    for index in range(len(class_ids)):
        class_slot = EXPECTED_CLASSES.index(int(class_ids[index]))
        rows.append(
            {
                "global_seed": int(repeated_seeds[index]),
                "class_id": int(class_ids[index]),
                "B_persistence": float(persistence[index]),
                "B_alarm": int(
                    persistence[index]
                    > observed["blur_score_threshold"][index]
                ),
                "E_blur_gated_running_max_log": float(e_score[index]),
                "E_blur_gated_alarm": int(gated.alarm[index]),
                "E_no_state_gate_running_max_log": float(no_gate_score[index]),
                "E_first_hit_full_budget_running_max_log": float(
                    one_shot_score[index]
                ),
                "G_start_schedule_diagnostic": float(g_score[index]),
                "T_delta1": int(start[index, 0]),
                "h_delta1": int(h[index, 0]),
                "T_delta4": int(start[index, 1]),
                "h_delta4": int(h[index, 1]),
                "fallback_steps_delta1": int(
                    np.sum(gated.direction_reused[index, 0])
                ),
                "fallback_steps_delta4": int(
                    np.sum(gated.direction_reused[index, 1])
                ),
            }
        )
        if class_slot < 0:  # pragma: no cover - index() already guards this
            raise AssertionError("invalid class slot")
    mechanics = {
        "applied_K": gated.applied_K,
        "start_time_index": gated.start_time_index,
        "start_remaining_effective_count": gated.start_remaining_effective_count,
        "direction_reused": gated.direction_reused.astype(np.uint8),
    }
    return rows, mechanics


def summarize_mechanics(mechanics: Mapping[str, np.ndarray]) -> dict[str, Any]:
    applied = np.asarray(mechanics["applied_K"], dtype=np.float64)
    start = np.asarray(mechanics["start_time_index"], dtype=np.int16)
    h = np.asarray(mechanics["start_remaining_effective_count"], dtype=np.int16)
    reused = np.asarray(mechanics["direction_reused"], dtype=np.uint8)
    summaries: list[dict[str, Any]] = []
    for scale_index, heat_shift in enumerate(core.HEAT_SHIFTS):
        started = h[:, scale_index] > 0
        positive_steps = np.sum(applied[:, scale_index] > 0.0, axis=1)
        total_K = np.sum(applied[:, scale_index], axis=1)
        complete = started & (positive_steps == h[:, scale_index]) & np.isclose(
            total_K, core.TOTAL_K_PER_SCALE, rtol=2e-12, atol=1e-12
        )
        started_steps = int(np.sum(positive_steps[started]))
        fallback_steps = int(np.sum(reused[:, scale_index]))
        summaries.append(
            {
                "heat_shift": float(heat_shift),
                "path_count": int(len(started)),
                "started_path_count": int(np.sum(started)),
                "started_class_count": int(
                    len(
                        set(
                            np.tile(EXPECTED_CLASSES, len(started) // len(EXPECTED_CLASSES))[
                                started
                            ].tolist()
                        )
                    )
                ),
                "complete_started_path_count": int(np.sum(complete)),
                "complete_coverage_fraction_among_started": (
                    None if not np.any(started) else float(np.mean(complete[started]))
                ),
                "started_positive_K_step_count": started_steps,
                "fallback_step_count": fallback_steps,
                "fallback_fraction_among_started_steps": (
                    None if started_steps == 0 else fallback_steps / started_steps
                ),
                "start_time_histogram": {
                    str(value): int(np.sum(start[:, scale_index] == value))
                    for value in sorted(set(start[:, scale_index].tolist()))
                },
            }
        )
    return {
        "quality_or_endpoint_metric": False,
        "labels_reviews_external_representations_used": False,
        "scales": summaries,
    }


def publish_score(args: argparse.Namespace) -> None:
    if not EVALUATION_SEED_MIN <= args.seed_start < args.seed_end <= EVALUATION_SEED_MAX_EXCLUSIVE:
        raise RuntimeError("score seed shard must lie inside the disjoint 270..849 axis")
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite score shard: {output}")
    trace_root = args.trace_root.expanduser().resolve()
    _, first_metadata = load_seed(trace_root, CALIBRATION_SEEDS[0])
    model, vae, diffusion = load_runtime(
        first_metadata, args.dit_root, args.checkpoint, args.vae_snapshot
    )
    gate_by_class, alarm_by_class, calibration = calibrate_thresholds(
        trace_root, vae, args.decode_batch_size
    )
    all_rows: list[dict[str, Any]] = []
    all_mechanics: dict[str, list[np.ndarray]] = {}
    trace_records: list[dict[str, Any]] = []
    observation_receipts: list[dict[str, Any]] = []
    seeds = tuple(range(args.seed_start, args.seed_end))
    for offset in range(0, len(seeds), args.seed_batch_size):
        seed_batch = seeds[offset : offset + args.seed_batch_size]
        arrays_list: list[dict[str, np.ndarray]] = []
        for seed in seed_batch:
            arrays, metadata = load_seed(trace_root, seed)
            if any(
                metadata[key] != first_metadata[key]
                for key in ("classes", "source", "checkpoint", "vae_snapshot")
            ):
                raise RuntimeError(f"runtime lineage changed at seed {seed}")
            arrays_list.append(arrays)
            trace_records.append(
                {
                    "global_seed": seed,
                    "trace_manifest_sha256": metadata["trace_manifest_sha256"],
                    "trace_archive_sha256": metadata["trace_archive_sha256"],
                }
            )
        combined = combine_seed_arrays(arrays_list)
        class_ids = np.tile(np.asarray(EXPECTED_CLASSES, dtype=np.int16), len(seed_batch))
        slots = np.asarray(
            [EXPECTED_CLASSES.index(int(value)) for value in class_ids], dtype=np.int64
        )
        observed, receipt = build_observation(
            combined,
            class_ids,
            gate_by_class[slots],
            alarm_by_class[slots],
            model,
            vae,
            diffusion,
            args.decode_batch_size,
        )
        rows, mechanics = score_rows(observed, seed_batch)
        all_rows.extend(rows)
        for name, value in mechanics.items():
            all_mechanics.setdefault(name, []).append(value)
        receipt["seeds"] = list(seed_batch)
        observation_receipts.append(receipt)
        if args.progress_every and len(all_rows) // len(EXPECTED_CLASSES) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "scored_seeds": len(all_rows) // len(EXPECTED_CLASSES),
                        "total_seeds": len(seeds),
                        "last_seed": seed_batch[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    mechanics = {
        name: np.ascontiguousarray(np.concatenate(chunks, axis=0))
        for name, chunks in all_mechanics.items()
    }
    mechanics_summary = summarize_mechanics(mechanics)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (staging / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        np.savez(staging / "mechanics.npz", **mechanics)
        write_json(staging / "calibration.json", calibration)
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "RETROSPECTIVE_DIT_V22_LABEL_FREE_SCORE_SHARD",
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "seed_start_inclusive": args.seed_start,
            "seed_end_exclusive": args.seed_end,
            "classes_ordered": list(EXPECTED_CLASSES),
            "row_count": len(all_rows),
            "calibration_identity_sha256": calibration["identity_sha256"],
            "trace_records": trace_records,
            "observation_receipts_identity_sha256": canonical_sha256(
                observation_receipts
            ),
            "mechanics_summary": mechanics_summary,
            "scorer_source_sha256": sha256_file(Path(__file__).resolve()),
            "labels_reviews_endpoints_FID_Inception_DINO_CLIP_embeddings_opened": False,
            "intervention_or_rollback_performed": False,
            "files": [
                {
                    "name": name,
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("scores.csv", "mechanics.npz", "calibration.json")
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(all_rows),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def validate_score_shard(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = path.expanduser().resolve()
    expected = {"scores.csv", "mechanics.npz", "calibration.json", "manifest.json", "completion.json"}
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in root.iterdir()} != expected
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise RuntimeError(f"score shard exact tree changed: {root}")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    identity = manifest.get("identity_sha256")
    without = dict(manifest)
    without.pop("identity_sha256", None)
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != "RETROSPECTIVE_DIT_V22_LABEL_FREE_SCORE_SHARD"
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get(
            "labels_reviews_endpoints_FID_Inception_DINO_CLIP_embeddings_opened"
        )
        is not False
        or canonical_sha256(without) != identity
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != identity
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
    ):
        raise RuntimeError("score shard envelope is invalid")
    records = {row["name"]: row for row in manifest.get("files", [])}
    if set(records) != {"scores.csv", "mechanics.npz", "calibration.json"}:
        raise RuntimeError("score shard file records changed")
    for name, record in records.items():
        if (
            record.get("bytes") != (root / name).stat().st_size
            or record.get("sha256") != sha256_file(root / name)
        ):
            raise RuntimeError(f"score shard member changed: {name}")
    with (root / "scores.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise RuntimeError("score columns changed")
        rows = list(reader)
    if len(rows) != manifest.get("row_count"):
        raise RuntimeError("score row count changed")
    return manifest, rows


def tie_auc(positive: np.ndarray, negative: np.ndarray) -> tuple[float, int]:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan"), 0
    comparisons = positive[:, None] - negative[None, :]
    numerator = np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0)
    denominator = comparisons.size
    return float(numerator / denominator), int(denominator)


def class_matched_auc(
    rows: Sequence[Mapping[str, Any]], score: str, positive: Any
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class: list[dict[str, Any]] = []
    for class_id in EXPECTED_CLASSES:
        positives = np.asarray(
            [float(row[score]) for row in rows if row["class_id"] == class_id and positive(row)],
            dtype=np.float64,
        )
        negatives = np.asarray(
            [
                float(row[score])
                for row in rows
                if row["class_id"] == class_id and row["final_severity"] == "clean_good"
            ],
            dtype=np.float64,
        )
        auc, pairs = tie_auc(positives, negatives)
        if pairs:
            numerator += auc * pairs
            denominator += pairs
        per_class.append(
            {
                "class_id": class_id,
                "positive_count": len(positives),
                "clean_good_count": len(negatives),
                "auc": None if not math.isfinite(auc) else auc,
                "comparable_pairs": pairs,
            }
        )
    return {
        "auc": None if denominator == 0 else numerator / denominator,
        "comparable_pairs": denominator,
        "per_class": per_class,
    }


def schedule_exact_concordance(
    rows: Sequence[Mapping[str, Any]], positive: Any
) -> dict[str, Any]:
    strata: dict[tuple[int, int, int, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not (positive(row) or row["final_severity"] == "clean_good"):
            continue
        key = (
            int(row["class_id"]),
            int(row["T_delta1"]),
            int(row["h_delta1"]),
            int(row["T_delta4"]),
            int(row["h_delta4"]),
        )
        strata.setdefault(key, []).append(row)
    numerator = 0.0
    denominator = 0
    informative = 0
    tie_count = 0
    for members in strata.values():
        pos = np.asarray(
            [float(row["E_blur_gated_running_max_log"]) for row in members if positive(row)]
        )
        neg = np.asarray(
            [
                float(row["E_blur_gated_running_max_log"])
                for row in members
                if row["final_severity"] == "clean_good"
            ]
        )
        if len(pos) == 0 or len(neg) == 0:
            continue
        comparisons = pos[:, None] - neg[None, :]
        numerator += float(np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0))
        tie_count += int(np.sum(comparisons == 0.0))
        denominator += int(comparisons.size)
        informative += 1
    return {
        "concordance": None if denominator == 0 else numerator / denominator,
        "comparable_pairs": denominator,
        "informative_exact_strata": informative,
        "tie_fraction": None if denominator == 0 else tie_count / denominator,
        "claim_limit": (
            "retrospective descriptive association only within exact-start-schedule "
            "strata having both labels; not causal and not a rollback gate"
        ),
    }


def publish_evaluation(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite retrospective evaluation: {output}")
    score_rows_all: list[dict[str, str]] = []
    manifests: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    calibration_identity: str | None = None
    for path in args.score_shards:
        manifest, rows = validate_score_shard(path)
        if calibration_identity is None:
            calibration_identity = str(manifest["calibration_identity_sha256"])
        elif calibration_identity != manifest.get("calibration_identity_sha256"):
            raise RuntimeError("score shards use different B calibrations")
        for row in rows:
            key = (int(row["global_seed"]), int(row["class_id"]))
            if key in seen:
                raise RuntimeError("score shards overlap")
            seen.add(key)
            score_rows_all.append(row)
        manifests.append(manifest)
    label_path = args.labels.expanduser().resolve()
    with label_path.open(encoding="utf-8", newline="") as handle:
        label_reader = csv.DictReader(handle)
        labels = {
            (int(row["global_seed"]), int(row["class_id"])): row
            for row in label_reader
        }
    joined: list[dict[str, Any]] = []
    for score_row in score_rows_all:
        key = (int(score_row["global_seed"]), int(score_row["class_id"]))
        label = labels.get(key)
        if label is None:
            raise RuntimeError(f"missing frozen consensus row for {key}")
        joined.append(
            {
                **{name: (int(score_row[name]) if name in {"global_seed", "class_id", "B_alarm", "E_blur_gated_alarm", "T_delta1", "h_delta1", "T_delta4", "h_delta4", "fallback_steps_delta1", "fallback_steps_delta4"} else float(score_row[name])) for name in SCORE_COLUMNS},
                "final_severity": label["final_severity"],
                "blur_component_consensus": label["blur_component_consensus"] == "true",
                "discrete_structure_component_consensus": label[
                    "discrete_structure_component_consensus"
                ]
                == "true",
            }
        )
    primary = lambda row: (
        row["final_severity"] == "clear_bad" and row["blur_component_consensus"]
    )
    all_clear = lambda row: row["final_severity"] == "clear_bad"
    score_names = (
        "B_persistence",
        "E_blur_gated_running_max_log",
        "E_no_state_gate_running_max_log",
        "E_first_hit_full_budget_running_max_log",
        "G_start_schedule_diagnostic",
    )
    result: dict[str, Any] = {
        "status": "RETROSPECTIVE_EXPLORATORY_ONLY_NO_INTERVENTION_AUTHORITY",
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "score_shard_manifest_ids": [row["identity_sha256"] for row in manifests],
        "score_row_count": len(joined),
        "seed_range": [min(key[0] for key in seen), max(key[0] for key in seen)],
        "external_label_file": str(label_path),
        "external_label_file_sha256": sha256_file(label_path),
        "external_labels_used_only_after_label_free_scores_were_immutable": True,
        "FID_Inception_DINO_CLIP_embeddings_used": False,
        "primary_definition": "clear_bad AND blur_component_consensus vs clean_good",
        "primary_counts": {
            "positive": sum(primary(row) for row in joined),
            "clean_good": sum(row["final_severity"] == "clean_good" for row in joined),
        },
        "primary_auc": {
            name: class_matched_auc(joined, name, primary) for name in score_names
        },
        "all_clear_bad_secondary_auc": {
            name: class_matched_auc(joined, name, all_clear) for name in score_names
        },
        "primary_schedule_exact_E_concordance": schedule_exact_concordance(
            joined, primary
        ),
    }
    positives = [row for row in joined if primary(row)]
    negatives = [row for row in joined if row["final_severity"] == "clean_good"]
    result["frozen_operating_points"] = {
        "B": {
            "TPR": None if not positives else sum(row["B_alarm"] for row in positives) / len(positives),
            "FPR": None if not negatives else sum(row["B_alarm"] for row in negatives) / len(negatives),
            "positive_alerts": sum(row["B_alarm"] for row in positives),
            "clean_good_alerts": sum(row["B_alarm"] for row in negatives),
        },
        "E_at_10": {
            "TPR": None if not positives else sum(row["E_blur_gated_alarm"] for row in positives) / len(positives),
            "FPR": None if not negatives else sum(row["E_blur_gated_alarm"] for row in negatives) / len(negatives),
            "positive_alerts": sum(row["E_blur_gated_alarm"] for row in positives),
            "clean_good_alerts": sum(row["E_blur_gated_alarm"] for row in negatives),
            "alpha_is_overall_P_intervention_budget_not_clean_good_FPR": True,
        },
    }
    result["primary_positive_rows"] = [
        {
            "global_seed": row["global_seed"],
            "class_id": row["class_id"],
            **{name: row[name] for name in score_names},
            "B_alarm": row["B_alarm"],
            "E_alarm": row["E_blur_gated_alarm"],
            "start_schedule": [
                row["T_delta1"],
                row["h_delta1"],
                row["T_delta4"],
                row["h_delta4"],
            ],
        }
        for row in positives
    ]
    result["identity_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "results.json", result)
        manifest = {
            "status": "complete",
            "artifact_kind": "RETROSPECTIVE_DIT_V22_EXTERNAL_LABEL_EVALUATION",
            "result_identity_sha256": result["identity_sha256"],
            "result_file_sha256": sha256_file(staging / "results.json"),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def self_test() -> None:
    positive = np.asarray([2.0, 1.0])
    negative = np.asarray([0.0, 1.0])
    auc, pairs = tie_auc(positive, negative)
    if pairs != 4 or not math.isclose(auc, 0.875):
        raise AssertionError("tie-aware AUC synthetic witness changed")
    synthetic = [
        {
            "class_id": 207,
            "global_seed": 1,
            "final_severity": "clear_bad",
            "blur_component_consensus": True,
            "E_blur_gated_running_max_log": 2.0,
            "T_delta1": 2,
            "h_delta1": 3,
            "T_delta4": -1,
            "h_delta4": 0,
        },
        {
            "class_id": 207,
            "global_seed": 2,
            "final_severity": "clean_good",
            "blur_component_consensus": False,
            "E_blur_gated_running_max_log": 1.0,
            "T_delta1": 2,
            "h_delta1": 3,
            "T_delta4": -1,
            "h_delta4": 0,
        },
    ]
    conditional = schedule_exact_concordance(
        synthetic,
        lambda row: row["final_severity"] == "clear_bad"
        and row["blur_component_consensus"],
    )
    if conditional["concordance"] != 1.0 or conditional["comparable_pairs"] != 1:
        raise AssertionError("schedule-exact concordance synthetic witness changed")
    print("retrospective v2.2 pilot self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    score = subs.add_parser("score")
    score.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    score.add_argument("--seed-start", type=int, required=True)
    score.add_argument("--seed-end", type=int, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    score.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    score.add_argument("--vae-snapshot", type=Path, default=DEFAULT_VAE)
    score.add_argument("--seed-batch-size", type=int, default=1)
    score.add_argument("--decode-batch-size", type=int, default=12)
    score.add_argument("--progress-every", type=int, default=20)
    score.set_defaults(func=publish_score)
    evaluate = subs.add_parser("evaluate")
    evaluate.add_argument("--score-shards", type=Path, nargs="+", required=True)
    evaluate.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=publish_evaluation)
    test = subs.add_parser("self-test")
    test.set_defaults(func=lambda _: self_test())
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "seed_batch_size", 1) <= 0 or getattr(args, "decode_batch_size", 1) <= 0:
        raise ValueError("batch sizes must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

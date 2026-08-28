#!/usr/bin/env python3
"""Freeze label-free scientific-v4 B state/alarm thresholds.

Exactly 20 calibration traces per selected class are consumed.  The only
trajectory tensor loaded is the nine-checkpoint ``pred_xstart`` latent (plus
its frozen axes); endpoint PNGs/envelopes, labels, reviews, representations,
innovations, E values, and confirmation rows are never opened.  The 17th
ascending checkpoint B value defines a strict state gate; the 19th ascending
path-mean B value defines a strict B alarm.  These are overall exchangeability
trigger budgets, never clean-good conditional false-positive guarantees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch

try:
    from .dit_scientific_v4_be_contract import (
        CALIBRATION_SEEDS,
        CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        METHOD_LOCK_ID,
        canonical_sha256,
        publish_artifact,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
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
    from dit_scientific_v4_be_contract import (  # type: ignore
        CALIBRATION_SEEDS,
        CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        METHOD_LOCK_ID,
        canonical_sha256,
        publish_artifact,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
    )
    from sample_dit_scientific_v4_be_traces import (  # type: ignore
        load_source_lock,
        load_trace_array_whitelist,
        validate_trace_pool,
    )


CALIBRATOR = "calibrate_dit_scientific_v4_be"
STATE_GATE_ORDER_INDEX = 16
B_ALARM_ORDER_INDEX = 18
LOADED_ARRAYS = ("pred_xstart", "sampling_step", "internal_timestep")
DECODE_BATCH_SIZE = 8


def verify_source(source_lock: Path, manifest: Mapping[str, Any]) -> None:
    by_name = {row["name"]: row for row in manifest["files"]}
    expected = "sources/calibrate_dit_scientific_v4_be.py"
    if by_name.get(expected, {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running calibrator differs from frozen source snapshot")


def load_vae(strict: Any, contract: Mapping[str, Any], args: argparse.Namespace) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen VAE draft decoding")
    vae_snapshot = require_directory(args.vae_snapshot, "VAE snapshot")
    if strict.validate_vae_snapshot(vae_snapshot) != contract["assets"]["vae_snapshot"]:
        raise RuntimeError("runtime VAE differs from frozen asset")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    from diffusers.models import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        str(vae_snapshot), local_files_only=True, use_safetensors=True
    ).to(device="cuda", dtype=torch.float32).eval()


def decode_pred_xstart(
    vae: Any, pred: np.ndarray, *, scaling_factor: float, batch_size: int
) -> np.ndarray:
    flat = np.ascontiguousarray(pred.reshape(-1, 4, 32, 32), dtype=np.float32)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            latent = torch.from_numpy(flat[start : start + batch_size]).to(
                device="cuda", dtype=torch.float32
            )
            decoded = vae.decode(latent / scaling_factor).sample
            rows.append(((decoded + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(
        np.concatenate(rows, axis=0).reshape(pred.shape[0], len(CHECKPOINTS), 3, 256, 256)
    )


def derive_thresholds(
    *,
    severity: np.ndarray,
    classes: tuple[int, ...],
    source_lineage: Mapping[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    if severity.shape != (len(classes), len(CALIBRATION_SEEDS), len(CHECKPOINTS)):
        raise RuntimeError("calibration B tensor has the wrong exact axis")
    if severity.dtype != np.float64 or not np.isfinite(severity).all():
        raise RuntimeError("calibration B tensor must be finite float64")
    rows = []
    for slot, class_id in enumerate(classes):
        values = severity[slot]
        persistence = np.mean(values, axis=1)
        rows.append(
            {
                "class_id": class_id,
                "state_gate_threshold_by_checkpoint": [
                    float(value)
                    for value in np.sort(values, axis=0)[STATE_GATE_ORDER_INDEX]
                ],
                "B_alarm_threshold": float(np.sort(persistence)[B_ALARM_ORDER_INDEX]),
                "state_gate_order_statistic_1_based": 17,
                "B_alarm_order_statistic_1_based": 19,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "LABEL_FREE_B_CALIBRATION_COMPLETE",
        "calibrator": CALIBRATOR,
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "checkpoint_sampling_steps": list(CHECKPOINTS),
        "ordered_global_seeds": list(CALIBRATION_SEEDS),
        "count_per_class": len(CALIBRATION_SEEDS),
        "state_gate_strict_comparison": "B_k > class_checkpoint_threshold",
        "B_alarm_strict_comparison": "B_persistence > class_threshold",
        "state_gate_rank_interpretation": "4/21 at one fixed checkpoint under exchangeability and no ties",
        "B_alarm_overall_trigger_budget": "at most 2/21 under within-class exchangeability; not clean-good FPR",
        "classes": rows,
        "source_lineage": dict(source_lineage),
        "loaded_B_severity_array": {
            "shape": list(severity.shape),
            "dtype": severity.dtype.str,
            "raw_sha256": sha256_array(severity),
        },
        "implementation_source_sha256": implementation_sha256,
        "endpoint_images_envelopes_labels_reviews_external_representations_opened": False,
        "state_sigma_innovation_E_or_confirmation_arrays_loaded": False,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def validate_calibration(payload: Mapping[str, Any]) -> None:
    identity = payload.get("identity_sha256")
    body = dict(payload)
    body.pop("identity_sha256", None)
    if (
        canonical_sha256(body) != identity
        or payload.get("status") != "LABEL_FREE_B_CALIBRATION_COMPLETE"
        or payload.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or payload.get("endpoint_images_envelopes_labels_reviews_external_representations_opened")
        is not False
        or payload.get("state_sigma_innovation_E_or_confirmation_arrays_loaded") is not False
    ):
        raise RuntimeError("label-free calibration payload changed")
    rows = payload.get("classes")
    if not isinstance(rows, list) or len(rows) != 6:
        raise RuntimeError("calibration must contain exactly six class rows")


def run(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, strict, method_core = load_source_lock(source_lock)
    verify_source(source_lock, source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready; freeze a later activation lock")
    validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    trace_manifest, inventory = validate_trace_pool(
        args.trace_pool, contract=contract, protocol=protocol, plan=plan
    )
    classes = tuple(plan["selected_classes"])
    pred_rows: list[np.ndarray] = []
    loaded_records: list[dict[str, Any]] = []
    for class_id in classes:
        class_rows = []
        for seed in CALIBRATION_SEEDS:
            arrays, record = load_trace_array_whitelist(
                args.trace_pool,
                contract=contract,
                plan=plan,
                phase="calibration",
                global_seed=seed,
                class_id=class_id,
                names=LOADED_ARRAYS,
            )
            if not np.array_equal(arrays["sampling_step"], np.asarray(CHECKPOINTS, dtype=np.int16)):
                raise RuntimeError("calibration checkpoint axis changed")
            class_rows.append(arrays["pred_xstart"])
            loaded_records.append(record)
        pred_rows.append(np.stack(class_rows))
    pred = np.ascontiguousarray(np.stack(pred_rows), dtype=np.float32)
    # Decode in seed-major numerical batches, then restore [class,seed,time].
    flat_pred = pred.reshape(len(classes) * len(CALIBRATION_SEEDS), len(CHECKPOINTS), 4, 32, 32)
    vae = load_vae(strict, contract, args)
    rng_before = strict.cuda_rng_state_sha256()
    decoded = decode_pred_xstart(
        vae,
        flat_pred,
        scaling_factor=strict.VAE_SCALING_FACTOR,
        batch_size=DECODE_BATCH_SIZE,
    )
    rng_after = strict.cuda_rng_state_sha256()
    if rng_before != rng_after:
        raise RuntimeError("temporary VAE calibration decode consumed CUDA RNG")
    blur = method_core.compute_blur_tiles(decoded)
    severity = np.ascontiguousarray(
        blur.severity.reshape(len(classes), len(CALIBRATION_SEEDS), len(CHECKPOINTS)),
        dtype=np.float64,
    )
    calibration = derive_thresholds(
        severity=severity,
        classes=classes,
        source_lineage={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "trace_pool_identity_sha256": trace_manifest["identity_sha256"],
            "loaded_calibration_trace_records_identity_sha256": canonical_sha256(loaded_records),
            "loaded_array_names_exactly": list(LOADED_ARRAYS),
            "confirmation_rows_loaded": 0,
            "cuda_rng_unchanged_across_temporary_VAE_decode": True,
            "frozen_VAE_decode_batch_size": DECODE_BATCH_SIZE,
        },
        implementation_sha256=sha256_file(Path(__file__).resolve()),
    )
    validate_calibration(calibration)
    publish_artifact(
        args.output,
        artifact_kind="SCIENTIFIC_V4_LABEL_FREE_B_CALIBRATION",
        payloads={"calibration.json": json.dumps(calibration, indent=2, sort_keys=True) + "\n"},
        manifest_fields={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "trace_pool_identity_sha256": trace_manifest["identity_sha256"],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "labels_reviews_endpoint_or_external_representations_opened": False,
        },
    )
    print(json.dumps({"calibration_identity_sha256": calibration["identity_sha256"]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    result.add_argument("--trace-plan", type=Path, required=True)
    result.add_argument("--trace-pool", type=Path, required=True)
    result.add_argument("--vae-snapshot", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

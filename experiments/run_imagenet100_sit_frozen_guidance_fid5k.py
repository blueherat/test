#!/usr/bin/env python3
"""Run one resumable, resource-audited frozen-guidance ADM FID-5K job."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from experiments.run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_resource_audit,
    )
    from experiments.run_imagenet100_sit_static_pair_fid5k import (
        checkpoint_metadata,
        validate_metadata_pair,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_resource_audit,
    )
    from run_imagenet100_sit_static_pair_fid5k import (
        checkpoint_metadata,
        validate_metadata_pair,
    )
    from train_imagenet100_sit_flow import atomic_json_dump


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ANCHOR = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_OTHER = (
    BASE
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_REFERENCE = (
    BASE / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_OUTPUT = BASE / "finite_guidance_800k_replication/seed0/x_frozen"


def _same_float(left: object, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def valid_frozen_sampling_artifact(
    output_dir: Path,
    *,
    anchor: dict[str, object],
    other: dict[str, object],
    args: argparse.Namespace,
) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_frozen_n{args.num_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    if not valid_resource_audit(
        output_dir / "sampling_resource_audit.json",
        gpu_indices=args.gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        return False
    manifest = load_json(manifest_path)
    expected = {
        "format": "eqvae_imagenet100_sit_frozen_guidance_samples_v1",
        "weights": "ema",
        "requested_samples": args.num_samples,
        "batch_size": args.batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "global_seed": args.global_seed,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    for key, value in {
        "gamma": args.gamma,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
    }.items():
        if not _same_float(manifest.get(key), float(value)):
            mismatches[key] = (manifest.get(key), value)
    for side, metadata in (("anchor", anchor), ("other", other)):
        recorded = manifest.get(side, {})
        for key in (
            "checkpoint_sha256",
            "checkpoint_step",
            "protocol",
            "field_path",
            "prediction_target",
            "denominator_floor",
        ):
            if recorded.get(key) != metadata.get(key):
                mismatches[f"{side}.{key}"] = (
                    recorded.get(key),
                    metadata.get(key),
                )
    sampler = manifest.get("sampler", {})
    sampler_expected = {
        "method": "dopri5",
        "num_output_points": args.num_output_points,
        "precision": "fp32",
        "allow_tf32": True,
    }
    for key, value in sampler_expected.items():
        if sampler.get(key) != value:
            mismatches[f"sampler.{key}"] = (sampler.get(key), value)
    for key, value in (("atol", args.atol), ("rtol", args.rtol)):
        if not _same_float(sampler.get(key), float(value)):
            mismatches[f"sampler.{key}"] = (sampler.get(key), value)
    if mismatches:
        raise ValueError(f"incompatible frozen-guidance artifact: {mismatches}")
    if Path(manifest.get("samples", "")).resolve() != sample_path.resolve():
        raise ValueError("frozen sampling manifest points to a different sample file")
    for key in ("noise_sha256", "label_sha256"):
        if len(str(manifest.get(key, ""))) != 64:
            raise ValueError(f"frozen sampling manifest lacks a valid {key}")
    return True


def valid_frozen_fid_artifact(
    output_dir: Path,
    *,
    args: argparse.Namespace,
) -> bool:
    result_path = output_dir / "fid5k_adm_results.json"
    sample_path = output_dir / f"samples_frozen_n{args.num_samples}.npz"
    if not result_path.is_file():
        return False
    if not valid_resource_audit(
        output_dir / "fid_resource_audit.json",
        gpu_indices=args.gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        return False
    result = load_json(result_path)
    if Path(result.get("reference", "")).resolve() != args.reference.resolve():
        raise ValueError("frozen FID result uses a different reference")
    if Path(result.get("samples", "")).resolve() != sample_path.resolve():
        raise ValueError("frozen FID result uses a different sample file")
    expected = {
        "batch_size": args.fid_batch_size,
        "gpu_memory_fraction": args.fid_gpu_memory_fraction,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible frozen FID artifact: {mismatches}")
    for key in ("fid", "sfid", "inception_score"):
        if not math.isfinite(float(result.get(key, float("nan")))):
            raise ValueError(f"non-finite frozen {key}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.10)
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=8192)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    args = parser.parse_args()

    args.anchor_checkpoint = args.anchor_checkpoint.expanduser().resolve()
    args.other_checkpoint = args.other_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.gpu_indices = parse_gpu_indices(args.cuda_visible_devices)
    if len(args.gpu_indices) != 1:
        raise ValueError("frozen-guidance FID jobs must reserve exactly one GPU")
    if not args.reference.is_file() or not args.adm_python.is_file():
        raise FileNotFoundError("ADM reference statistics or evaluator is missing")
    if not math.isfinite(args.gamma):
        raise ValueError("gamma must be finite")
    if min(
        args.num_samples,
        args.batch_size,
        args.vae_decode_batch_size,
        args.cuda_allocator_limit_gib,
        args.fid_batch_size,
        args.gpu_memory_ceiling_mib,
        args.memory_poll_interval,
    ) <= 0:
        raise ValueError("sample, batch, and memory settings must be positive")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("allocator limit must leave headroom below memory ceiling")
    if not 0 < args.fid_gpu_memory_fraction < 1:
        raise ValueError("FID GPU memory fraction must lie in (0, 1)")

    anchor = checkpoint_metadata(args.anchor_checkpoint, "auto")
    other = checkpoint_metadata(args.other_checkpoint, "auto")
    validate_metadata_pair(
        anchor,
        other,
        allow_step_mismatch=args.allow_step_mismatch,
    )
    if anchor["prediction_target"] != "velocity":
        raise ValueError("frozen-guidance anchor must be a native velocity checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    base_env["PYTHONUNBUFFERED"] = "1"
    if not valid_frozen_sampling_artifact(
        args.output_dir,
        anchor=anchor,
        other=other,
        args=args,
    ):
        sampling_audit = run_logged(
            (
                sys.executable,
                str(REPO_ROOT / "experiments/sample_imagenet100_sit_frozen_guidance_fid.py"),
                "--anchor-checkpoint",
                str(args.anchor_checkpoint),
                "--other-checkpoint",
                str(args.other_checkpoint),
                "--gamma",
                repr(float(args.gamma)),
                "--output-dir",
                str(args.output_dir),
                "--num-samples",
                str(args.num_samples),
                "--batch-size",
                str(args.batch_size),
                "--vae-decode-batch-size",
                str(args.vae_decode_batch_size),
                "--cuda-allocator-limit-gib",
                str(args.cuda_allocator_limit_gib),
                "--global-seed",
                str(args.global_seed),
                "--num-output-points",
                str(args.num_output_points),
                "--atol",
                repr(float(args.atol)),
                "--rtol",
                repr(float(args.rtol)),
                "--precision",
                "fp32",
                "--allow-tf32",
                *(('--allow-step-mismatch',) if args.allow_step_mismatch else ()),
            ),
            args.output_dir / "sampling.log",
            env=base_env,
            monitored_gpu_indices=args.gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=args.output_dir / "sampling_resource_audit.json",
        )
    else:
        print("[reuse] frozen-guidance samples", flush=True)
        sampling_audit = load_json(args.output_dir / "sampling_resource_audit.json")
    if not valid_frozen_sampling_artifact(
        args.output_dir,
        anchor=anchor,
        other=other,
        args=args,
    ):
        raise RuntimeError("frozen sampler produced an invalid artifact")

    sample_path = args.output_dir / f"samples_frozen_n{args.num_samples}.npz"
    if not valid_frozen_fid_artifact(args.output_dir, args=args):
        fid_audit = run_logged(
            (
                str(args.adm_python),
                str(REPO_ROOT / "experiments/compute_adm_fid.py"),
                "--reference",
                str(args.reference),
                "--samples",
                str(sample_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(args.output_dir / "fid5k_adm_results.json"),
            ),
            args.output_dir / "evaluation.log",
            env=fid_environment(
                base_env,
                cuda_visible_devices=args.cuda_visible_devices,
            ),
            monitored_gpu_indices=args.gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=args.output_dir / "fid_resource_audit.json",
        )
    else:
        print("[reuse] frozen-guidance ADM FID", flush=True)
        fid_audit = load_json(args.output_dir / "fid_resource_audit.json")
    if not valid_frozen_fid_artifact(args.output_dir, args=args):
        raise RuntimeError("FID evaluator produced an invalid frozen-guidance artifact")

    manifest = load_json(args.output_dir / "sampling_manifest.json")
    result = load_json(args.output_dir / "fid5k_adm_results.json")
    summary = {
        "protocol": "imagenet100_sit_frozen_guidance_fid5k_v1",
        "formula": manifest["formula"],
        "anchor": anchor,
        "other": other,
        "gamma": float(args.gamma),
        "global_seed": args.global_seed,
        "num_samples": args.num_samples,
        "noise_fingerprint": manifest["noise_sha256"],
        "label_fingerprint": manifest["label_sha256"],
        "total_nfe": int(manifest["nfe"]),
        "total_anchor_forwards": int(manifest["anchor_forwards"]),
        "total_other_forwards": int(manifest["other_forwards"]),
        "sampling_peak_memory_mib": max(
            int(value) for value in sampling_audit["peak_memory_mib"].values()
        ),
        "fid_peak_memory_mib": max(
            int(value) for value in fid_audit["peak_memory_mib"].values()
        ),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
        "samples": str(sample_path),
    }
    atomic_json_dump(summary, args.output_dir / "frozen_guidance_fid5k.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

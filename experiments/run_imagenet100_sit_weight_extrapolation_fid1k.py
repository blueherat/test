#!/usr/bin/env python3
"""Sample and evaluate one weight-extrapolated SiT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from experiments.run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_fid_artifact,
        valid_resource_audit,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_fid_artifact,
        valid_resource_audit,
    )
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_REFERENCE = (
    DEFAULT_BASE
    / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)


def checkpoint_metadata(path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    extrapolation = checkpoint.get("weight_extrapolation")
    if not isinstance(extrapolation, dict):
        raise ValueError("checkpoint is not a weight-extrapolation artifact")
    weights = str(extrapolation.get("weights"))
    if weights not in {"ema", "model"} or weights not in checkpoint:
        raise ValueError("checkpoint has an invalid extrapolated weight type")
    config = checkpoint["config"]
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": int(checkpoint["step"]),
        "protocol": str(checkpoint["protocol"]),
        "model_name": str(config["model_name"]),
        "prediction_target": str(config.get("prediction_target", "velocity")),
        "weights": weights,
        "weight_extrapolation": extrapolation,
    }
    if metadata["prediction_target"] != "velocity":
        raise ValueError("weight extrapolation experiment requires velocity checkpoints")
    return metadata


def valid_sampling_artifact(
    output_dir: Path,
    *,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
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
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "weights": checkpoint["weights"],
        "prediction_target": "velocity",
        "requested_samples": args.num_samples,
        "world_size": 1,
        "per_rank_batch_size": args.batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
        "global_seed": args.global_seed,
        "weight_extrapolation": checkpoint["weight_extrapolation"],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible weight-extrapolation samples: {mismatches}")
    if Path(manifest["samples"]).resolve() != sample_path.resolve():
        raise ValueError("sampling manifest points to a different NPZ")
    if not manifest.get("rank_noise_sha256") or not manifest.get(
        "rank_label_sha256"
    ):
        raise ValueError("sampling manifest lacks pairing fingerprints")
    return True


def main(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.gpu_indices = parse_gpu_indices(args.cuda_visible_devices)
    if len(args.gpu_indices) != 1:
        raise ValueError("this paired FID-1K protocol uses exactly one GPU")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("allocator limit must leave headroom below memory ceiling")
    if not args.reference.is_file() or not args.adm_python.is_file():
        raise FileNotFoundError("ADM reference or evaluator environment is missing")
    metadata = checkpoint_metadata(args.checkpoint)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not valid_sampling_artifact(output_dir, checkpoint=metadata, args=args):
        run_logged(
            (
                args.torchrun,
                "--standalone",
                "--nproc_per_node=1",
                str(REPO_ROOT / "experiments/sample_imagenet100_sit_fid.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(output_dir),
                "--weights",
                str(metadata["weights"]),
                "--num-samples",
                str(args.num_samples),
                "--per-rank-batch-size",
                str(args.batch_size),
                "--vae-decode-batch-size",
                str(args.vae_decode_batch_size),
                "--cuda-allocator-limit-gib",
                str(args.cuda_allocator_limit_gib),
                "--global-seed",
                str(args.global_seed),
                "--precision",
                "fp32",
                "--allow-tf32",
            ),
            output_dir / "sampling.log",
            env=env,
            monitored_gpu_indices=args.gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "sampling_resource_audit.json",
        )
        if not valid_sampling_artifact(output_dir, checkpoint=metadata, args=args):
            raise RuntimeError("sampler completed without a valid paired artifact")
    else:
        print("[reuse] valid weight-extrapolation samples", flush=True)

    if not valid_fid_artifact(
        output_dir,
        reference=args.reference,
        num_samples=args.num_samples,
        fid_batch_size=args.fid_batch_size,
        fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
        gpu_indices=args.gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        run_logged(
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
                str(output_dir / "fid5k_adm_results.json"),
            ),
            output_dir / "evaluation.log",
            env=fid_environment(env, cuda_visible_devices=args.cuda_visible_devices),
            monitored_gpu_indices=args.gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "fid_resource_audit.json",
        )
    metrics = load_json(output_dir / "fid5k_adm_results.json")
    manifest = load_json(output_dir / "sampling_manifest.json")
    result = {
        "protocol": "imagenet100_sit_weight_extrapolation_fid1k_v1",
        "checkpoint": metadata,
        "gamma": float(metadata["weight_extrapolation"]["gamma"]),
        "num_samples": args.num_samples,
        "global_seed": args.global_seed,
        "noise_fingerprint": manifest["rank_noise_sha256"][0],
        "label_fingerprint": manifest["rank_label_sha256"][0],
        "fid": float(metrics["fid"]),
        "sfid": float(metrics["sfid"]),
        "inception_score": float(metrics["inception_score"]),
        "total_nfe": sum(
            int(item["total_nfe_across_batches"])
            for item in manifest.get("rank_sampling_stats", [])
        )
        if manifest.get("rank_sampling_stats")
        else None,
        "sampling_manifest": str(output_dir / "sampling_manifest.json"),
        "fid_result": str(output_dir / "fid5k_adm_results.json"),
    }
    atomic_json_dump(result, output_dir / "weight_extrapolation_fid1k.json")
    print(json.dumps(result, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.3)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=12 * 1024)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    parser.add_argument("--global-seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

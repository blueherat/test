#!/usr/bin/env python3
"""Sample and ADM-evaluate selected paths of a dual-output SiT checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import torch

try:
    from experiments.run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        DEFAULT_REFERENCE,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_fid_artifact,
        valid_resource_audit,
    )
    from experiments.train_imagenet100_sit_dual_output import PROTOCOL
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        DEFAULT_REFERENCE,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_fid_artifact,
        valid_resource_audit,
    )
    from train_imagenet100_sit_dual_output import PROTOCOL
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid5k_dual-output_step450000_seed0"
)
MODES = ("x", "epsilon", "dynamic")


def checkpoint_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
    metadata = {
        "step": int(checkpoint["step"]),
        "protocol": checkpoint["protocol"],
        "model_name": checkpoint["config"]["model_name"],
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
    }
    del checkpoint
    return metadata


def valid_sampling_artifact(
    output_dir: Path,
    *,
    checkpoint: dict[str, object],
    mode: str,
    args: argparse.Namespace,
) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    if not valid_resource_audit(
        output_dir / "sampling_resource_audit.json",
        gpu_indices=args.sampling_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        return False
    manifest = load_json(manifest_path)
    expected = {
        "format": "eqvae_imagenet100_sit_dual_output_samples_v1",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_step": checkpoint["step"],
        "weights": "ema",
        "sampling_mode": mode,
        "requested_samples": args.num_samples,
        "global_seed": args.global_seed,
        "world_size": len(args.sampling_gpu_indices),
        "per_rank_batch_size": args.per_rank_batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
        "cfg_scale": 1.0,
        "guidance": False,
        "same_noise_and_labels_across_modes": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible {mode} sampling artifact: {mismatches}")
    if Path(manifest["samples"]).resolve() != sample_path.resolve():
        raise ValueError(f"{mode} manifest points to a different sample NPZ")
    labels_path = Path(manifest["labels"])
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing {mode} label artifact: {labels_path}")
    return True


def evaluate_mode(
    *,
    mode: str,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    sampling_env: dict[str, str],
) -> dict[str, object]:
    output_dir = args.output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
    if not valid_sampling_artifact(
        output_dir,
        checkpoint=checkpoint,
        mode=mode,
        args=args,
    ):
        sampling_audit = run_logged(
            (
                args.torchrun,
                "--standalone",
                f"--nproc_per_node={len(args.sampling_gpu_indices)}",
                str(REPO_ROOT / "experiments/sample_imagenet100_sit_dual_fid.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(output_dir),
                "--weights",
                "ema",
                "--mode",
                mode,
                "--num-samples",
                str(args.num_samples),
                "--per-rank-batch-size",
                str(args.per_rank_batch_size),
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
            env=sampling_env,
            monitored_gpu_indices=args.sampling_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "sampling_resource_audit.json",
        )
        if not valid_sampling_artifact(
            output_dir,
            checkpoint=checkpoint,
            mode=mode,
            args=args,
        ):
            raise RuntimeError(f"{mode} sampler completed without valid artifacts")
    else:
        print(f"[reuse] valid {mode} samples", flush=True)
        sampling_audit = load_json(output_dir / "sampling_resource_audit.json")

    if not valid_fid_artifact(
        output_dir,
        reference=args.reference,
        num_samples=args.num_samples,
        fid_batch_size=args.fid_batch_size,
        fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
        gpu_indices=args.fid_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
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
                str(output_dir / "fid5k_adm_results.json"),
            ),
            output_dir / "evaluation.log",
            env=fid_environment(
                sampling_env,
                cuda_visible_devices=args.fid_cuda_visible_devices,
            ),
            monitored_gpu_indices=args.fid_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "fid_resource_audit.json",
        )
        if not valid_fid_artifact(
            output_dir,
            reference=args.reference,
            num_samples=args.num_samples,
            fid_batch_size=args.fid_batch_size,
            fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
            gpu_indices=args.fid_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
        ):
            raise RuntimeError(f"{mode} FID evaluator produced invalid artifacts")
    else:
        print(f"[reuse] valid {mode} FID", flush=True)
        fid_audit = load_json(output_dir / "fid_resource_audit.json")

    result = load_json(output_dir / "fid5k_adm_results.json")
    return {
        "mode": mode,
        **checkpoint,
        "weights": "ema",
        "num_samples": args.num_samples,
        "global_seed": args.global_seed,
        "reference": str(args.reference.resolve()),
        "samples": str(sample_path.resolve()),
        "sampling_peak_memory_mib": max(
            int(value) for value in sampling_audit["peak_memory_mib"].values()
        ),
        "fid_peak_memory_mib": max(
            int(value) for value in fid_audit["peak_memory_mib"].values()
        ),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
    }


def save_summary(rows: list[dict[str, object]], output_root: Path) -> dict[str, object]:
    csv_path = output_root / "dual_output_unguided_fid5k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "imagenet100_sit_dual_output_unguided_fid5k_v1",
        "comparison_is_paired": True,
        "pairing": "same checkpoint, EMA, global seed, per-rank RNG, labels, ODE, VAE, reference",
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(summary, output_root / "dual_output_unguided_fid5k.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default=shutil.which("torchrun") or "torchrun")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=64)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=7.5)
    parser.add_argument("--sampling-cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--fid-cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=9 * 1024)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    parser.add_argument("--global-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.sampling_gpu_indices = parse_gpu_indices(args.sampling_cuda_visible_devices)
    args.fid_gpu_indices = parse_gpu_indices(args.fid_cuda_visible_devices)
    if len(args.fid_gpu_indices) != 1:
        raise ValueError("ADM FID must use exactly one visible GPU")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("modes must not contain duplicates")
    if not args.reference.is_file():
        raise FileNotFoundError(f"missing reference: {args.reference}")
    if not args.adm_python.is_file():
        raise FileNotFoundError(f"missing ADM evaluator: {args.adm_python}")
    if min(
        args.num_samples,
        args.per_rank_batch_size,
        args.vae_decode_batch_size,
        args.fid_batch_size,
        args.cuda_allocator_limit_gib,
        args.gpu_memory_ceiling_mib,
        args.memory_poll_interval,
    ) <= 0:
        raise ValueError("sample, batch, and memory settings must be positive")
    if not 0 < args.fid_gpu_memory_fraction < 1:
        raise ValueError("FID GPU memory fraction must be between zero and one")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("allocator limit must leave headroom below memory ceiling")
    checkpoint = checkpoint_metadata(args.checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    sampling_env = os.environ.copy()
    sampling_env["CUDA_VISIBLE_DEVICES"] = args.sampling_cuda_visible_devices
    sampling_env.setdefault("OMP_NUM_THREADS", "1")
    sampling_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = [
        evaluate_mode(
            mode=mode,
            checkpoint=checkpoint,
            args=args,
            sampling_env=sampling_env,
        )
        for mode in args.modes
    ]
    summary = save_summary(rows, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)
    for row in rows:
        for key in ("fid", "sfid", "inception_score"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite {key} for {row['mode']}")


if __name__ == "__main__":
    main()

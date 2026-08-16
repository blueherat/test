#!/usr/bin/env python3
"""Run a paired ADM FID-1K sweep for a frozen SiT Internal Guidance head."""

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
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
        valid_fid_artifact,
        valid_resource_audit,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
    from experiments.train_imagenet100_sit_frozen_internal_v_head import (
        CLEAN_PROTOCOL,
        PROTOCOL,
    )
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
    from train_imagenet100_sit_frozen_internal_v_head import CLEAN_PROTOCOL, PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/checkpoints/step_00050000.pt"
)
DEFAULT_REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_frozen_internal_v_depth8_step50000_ema"
)
DEFAULT_GAMMAS = (0.1, 0.2, 0.4, 0.6, 1.0, 1.5)


def format_float(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def checkpoint_metadata(path: Path, head_weights: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_protocol = checkpoint.get("protocol")
    if checkpoint_protocol not in (PROTOCOL, CLEAN_PROTOCOL):
        raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
    config = checkpoint["config"]
    prediction_target = str(config.get("prediction_target", "velocity"))
    expected_protocol = PROTOCOL if prediction_target == "velocity" else CLEAN_PROTOCOL
    if checkpoint_protocol != expected_protocol:
        raise ValueError("checkpoint protocol and prediction target disagree")
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "step": int(checkpoint["step"]),
        "head_weights": head_weights,
        "source_checkpoint": str(Path(config["source_checkpoint"]).resolve()),
        "source_checkpoint_sha256": str(config["source_checkpoint_sha256"]),
        "source_step": int(config["source_step"]),
        "source_state_key": str(config["source_state_key"]),
        "model_name": str(config["model_name"]),
        "internal_depth": int(config["internal_depth"]),
        "prediction_target": prediction_target,
        "clean_velocity_denominator_floor": float(
            config.get("clean_velocity_denominator_floor", 0.05)
        ),
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
    }
    del checkpoint
    return metadata


def conditions(
    gammas: list[float],
    *,
    include_full: bool = True,
    include_internal: bool = True,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    if include_full:
        rows.append(("full", "full", 0.0))
    if include_internal:
        rows.append(("internal", "internal", 0.0))
    rows.extend(
        (f"extrap_gamma_{format_float(gamma)}", "extrapolation", gamma)
        for gamma in gammas
    )
    return rows


def valid_sampling_artifact(
    output_dir: Path,
    *,
    checkpoint: dict[str, object],
    mode: str,
    gamma: float,
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
    expected_format = (
        "eqvae_imagenet100_sit_frozen_internal_v_head_samples_v1"
        if checkpoint["prediction_target"] == "velocity"
        else "eqvae_imagenet100_sit_frozen_internal_clean_head_samples_v1"
    )
    expected = {
        "format": expected_format,
        "mode": mode,
        "gamma": float(gamma),
        "requested_samples": args.num_samples,
        "world_size": len(args.sampling_gpu_indices),
        "per_rank_batch_size": args.per_rank_batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
        "global_seed": args.global_seed,
        "cfg_scale": 1.0,
        "guidance": False,
        "same_noise_and_labels_across_conditions": True,
        "single_shared_backbone_forward_per_nfe": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    recorded_model = manifest.get("model", {})
    model_expected = {
        "head_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "head_step": checkpoint["step"],
        "head_weights": checkpoint["head_weights"],
        "source_checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
        "source_step": checkpoint["source_step"],
        "source_state_key": checkpoint["source_state_key"],
        "internal_depth": checkpoint["internal_depth"],
        "prediction_target": checkpoint["prediction_target"],
    }
    mismatches.update(
        {
            f"model.{key}": (
                recorded_model.get(
                    key,
                    "velocity" if key == "prediction_target" else None,
                ),
                value,
            )
            for key, value in model_expected.items()
            if recorded_model.get(
                key,
                "velocity" if key == "prediction_target" else None,
            )
            != value
        }
    )
    if mismatches:
        raise ValueError(f"incompatible existing sampling artifact: {mismatches}")
    if Path(manifest["samples"]).resolve() != sample_path.resolve():
        raise ValueError("sampling manifest points to a different NPZ")
    return True


def evaluate_condition(
    *,
    name: str,
    mode: str,
    gamma: float,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    sampling_env: dict[str, str],
) -> dict[str, object]:
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
    if not valid_sampling_artifact(
        output_dir,
        checkpoint=checkpoint,
        mode=mode,
        gamma=gamma,
        args=args,
    ):
        run_logged(
            (
                args.torchrun,
                "--standalone",
                f"--nproc_per_node={len(args.sampling_gpu_indices)}",
                str(
                    REPO_ROOT
                    / "experiments/sample_imagenet100_sit_frozen_internal_v_head_fid.py"
                ),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(output_dir),
                "--head-weights",
                args.head_weights,
                "--mode",
                mode,
                "--gamma",
                str(gamma),
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
            gamma=gamma,
            args=args,
        ):
            raise RuntimeError(f"sampler produced an invalid artifact: {output_dir}")
    else:
        print(f"[reuse] valid samples for {name}", flush=True)

    if not valid_fid_artifact(
        output_dir,
        reference=args.reference,
        num_samples=args.num_samples,
        fid_batch_size=args.fid_batch_size,
        fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
        gpu_indices=args.fid_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        fid_env = fid_environment(
            sampling_env,
            cuda_visible_devices=args.fid_cuda_visible_devices,
        )
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
            env=fid_env,
            monitored_gpu_indices=args.fid_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "fid_resource_audit.json",
        )
    metric = load_json(output_dir / "fid5k_adm_results.json")
    manifest = load_json(output_dir / "sampling_manifest.json")
    sampling_audit = load_json(output_dir / "sampling_resource_audit.json")
    fid_audit = load_json(output_dir / "fid_resource_audit.json")
    return {
        "condition": name,
        "mode": mode,
        "gamma": float(gamma),
        "head_step": checkpoint["step"],
        "head_weights": args.head_weights,
        "num_samples": args.num_samples,
        "global_seed": args.global_seed,
        "fid": float(metric["fid"]),
        "sfid": float(metric["sfid"]),
        "inception_score": float(metric["inception_score"]),
        "sampling_peak_memory_mib": max(
            int(value) for value in sampling_audit["peak_memory_mib"].values()
        ),
        "fid_peak_memory_mib": max(
            int(value) for value in fid_audit["peak_memory_mib"].values()
        ),
        "total_nfe": int(manifest["total_nfe"]),
        "noise_fingerprint": ";".join(manifest["rank_noise_sha256"]),
        "label_fingerprint": ";".join(manifest["rank_label_sha256"]),
        "sample_sha256": sha256_file(sample_path),
    }


def save_summary(
    rows: list[dict[str, object]],
    *,
    checkpoint: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    fingerprints = {
        (row["noise_fingerprint"], row["label_fingerprint"]) for row in rows
    }
    if len(fingerprints) != 1:
        raise ValueError("conditions do not share identical noise and labels")
    target = str(checkpoint["prediction_target"])
    csv_path = output_root / f"frozen_internal_{target}_head_fid1k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    extrapolated = [row for row in rows if row["mode"] == "extrapolation"]
    best_extrapolation = min(extrapolated, key=lambda row: float(row["fid"]))
    summary = {
        "protocol": f"imagenet100_sit_frozen_internal_{target}_head_fid1k_v1",
        "comparison_is_paired": True,
        "pairing": "same shared model, initial noise, labels, ODE, VAE, and ADM reference",
        "checkpoint": checkpoint,
        "best_extrapolation": best_extrapolation,
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(
        summary,
        output_root / f"frozen_internal_{target}_head_fid1k.json",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--head-weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default=shutil.which("torchrun") or "torchrun")
    parser.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument(
        "--include-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-internal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--sampling-cuda-visible-devices", default="1,2")
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--fid-cuda-visible-devices", default="3")
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=10 * 1024)
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
        raise ValueError("ADM FID evaluation requires exactly one visible GPU")
    if not args.gammas or any(
        gamma <= 0 or not math.isfinite(gamma) for gamma in args.gammas
    ):
        raise ValueError("gammas must be finite and positive")
    if len(set(args.gammas)) != len(args.gammas):
        raise ValueError("gammas must not contain duplicates")
    if len({format_float(value) for value in args.gammas}) != len(args.gammas):
        raise ValueError("formatted gamma names collide")
    if not args.reference.is_file() or not args.adm_python.is_file():
        raise FileNotFoundError("ADM reference or evaluator is missing")
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
        raise ValueError("FID GPU memory fraction must lie in (0, 1)")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("allocator limit must leave headroom below memory ceiling")

    checkpoint = checkpoint_metadata(args.checkpoint, args.head_weights)
    args.output_root.mkdir(parents=True, exist_ok=True)
    sampling_env = os.environ.copy()
    sampling_env["CUDA_VISIBLE_DEVICES"] = args.sampling_cuda_visible_devices
    sampling_env.setdefault("OMP_NUM_THREADS", "1")
    sampling_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = [
        evaluate_condition(
            name=name,
            mode=mode,
            gamma=gamma,
            checkpoint=checkpoint,
            args=args,
            sampling_env=sampling_env,
        )
        for name, mode, gamma in conditions(
            args.gammas,
            include_full=args.include_full,
            include_internal=args.include_internal,
        )
    ]
    summary = save_summary(rows, checkpoint=checkpoint, output_root=args.output_root)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

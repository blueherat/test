#!/usr/bin/env python3
"""Run resumable ADM FID-5K sweeps for paired SiT field controls."""

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
    from experiments.imagenet100_sit_static_pair import (
        COMMON_UNIQUE_COMPONENTS,
        CONTROL_MODES,
        WINDOW_CONTROL_MODES,
        X_FLOOR_CONTROL_MODES,
        resolve_field_semantics,
    )
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
    from imagenet100_sit_static_pair import (
        COMMON_UNIQUE_COMPONENTS,
        CONTROL_MODES,
        WINDOW_CONTROL_MODES,
        X_FLOOR_CONTROL_MODES,
        resolve_field_semantics,
    )
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
DEFAULT_ANCHOR_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_OTHER_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_REFERENCE_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00270000.pt"
)
DEFAULT_REFERENCE_STATS = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid5k_static_pair_v_to_jit_x_step400000_seed0"
)
DEFAULT_SCALES = (
    -1.0,
    -0.75,
    -0.5,
    -0.3,
    -0.2,
    -0.1,
    -0.05,
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
)


def format_scale(scale: float) -> str:
    value = format(float(scale), ".8g")
    return value.replace("-", "m").replace(".", "p").replace("+", "")


def checkpoint_metadata(path: Path, requested_field: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["config"]
    semantics = resolve_field_semantics(
        protocol=str(checkpoint.get("protocol")),
        config=config,
        requested_path=requested_field,
    )
    metadata: dict[str, object] = {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": int(checkpoint["step"]),
        "protocol": str(checkpoint["protocol"]),
        "model_name": str(config["model_name"]),
        "field_path": semantics.field_path,
        "prediction_target": semantics.prediction_target,
        "loss_space": config.get("loss_space", "velocity"),
        "denominator_floor": semantics.denominator_floor,
        "global_batch_size": int(config["global_batch_size"]),
        "seed": int(config["seed"]),
        "training_world_size": int(config.get("world_size", 1)),
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
        "official_sit": checkpoint.get("official_sit"),
    }
    del checkpoint
    return metadata


def validate_metadata_pair(
    anchor: dict[str, object],
    other: dict[str, object],
    *,
    allow_step_mismatch: bool = False,
) -> None:
    keys = [
        "model_name",
        "global_batch_size",
        "seed",
        "data_manifest_sha256",
        "official_sit",
    ]
    if not allow_step_mismatch:
        keys.append("checkpoint_step")
    mismatches = {
        key: (anchor[key], other[key])
        for key in keys
        if anchor[key] != other[key]
    }
    if mismatches:
        raise ValueError(f"incompatible pair checkpoints: {mismatches}")


def valid_sampling_artifact(
    output_dir: Path,
    *,
    anchor: dict[str, object],
    other: dict[str, object],
    reference: dict[str, object] | None,
    scale: float,
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
    recorded_control_mode = manifest.get("control_mode", "full_pair")
    expected = {
        "format": (
            "eqvae_imagenet100_sit_common_unique_samples_v1"
            if args.common_unique_component is not None
            else "eqvae_imagenet100_sit_static_pair_samples_v1"
        ),
        "weights": "ema",
        "static_scale": float(scale),
        "requested_samples": args.num_samples,
        "world_size": len(args.sampling_gpu_indices),
        "per_rank_batch_size": args.per_rank_batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
        "inter_batch_sleep": args.inter_batch_sleep,
        "global_seed": args.global_seed,
        "cfg_scale": 1.0,
        "guidance": False,
        "same_noise_and_labels_across_scales": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if recorded_control_mode != args.control_mode:
        mismatches["control_mode"] = (recorded_control_mode, args.control_mode)
    if manifest.get("common_unique_component") != args.common_unique_component:
        mismatches["common_unique_component"] = (
            manifest.get("common_unique_component"),
            args.common_unique_component,
        )
    if bool(manifest.get("allow_step_mismatch", False)) != args.allow_step_mismatch:
        mismatches["allow_step_mismatch"] = (
            bool(manifest.get("allow_step_mismatch", False)),
            args.allow_step_mismatch,
        )
    if bool(manifest.get("allow_reference_step_mismatch", False)) != (
        args.allow_reference_step_mismatch
    ):
        mismatches["allow_reference_step_mismatch"] = (
            bool(manifest.get("allow_reference_step_mismatch", False)),
            args.allow_reference_step_mismatch,
        )
    if args.control_mode != "full_pair" and manifest.get(
        "window_transition_width"
    ) != args.window_transition_width:
        mismatches["window_transition_width"] = (
            manifest.get("window_transition_width"),
            args.window_transition_width,
        )
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
    if reference is not None:
        recorded = manifest.get("reference", {})
        for key in (
            "checkpoint_sha256",
            "checkpoint_step",
            "protocol",
            "field_path",
            "prediction_target",
            "denominator_floor",
        ):
            if recorded.get(key) != reference.get(key):
                mismatches[f"reference.{key}"] = (
                    recorded.get(key),
                    reference.get(key),
                )
    elif manifest.get("reference") is not None:
        mismatches["reference"] = (manifest.get("reference"), None)
    recorded_other_floor = manifest.get("other", {}).get(
        "inference_denominator_floor",
        manifest.get("other", {}).get("denominator_floor"),
    )
    expected_other_floor = (
        float(args.other_inference_denominator_floor)
        if args.other_inference_denominator_floor is not None
        else float(other["denominator_floor"])
    )
    if recorded_other_floor != expected_other_floor:
        mismatches["other.inference_denominator_floor"] = (
            recorded_other_floor,
            expected_other_floor,
        )
    if mismatches:
        raise ValueError(f"incompatible static-pair artifact: {mismatches}")
    if Path(manifest["samples"]).resolve() != sample_path.resolve():
        raise ValueError("sampling manifest points to a different sample NPZ")
    if not manifest.get("rank_noise_sha256") or not manifest.get("rank_label_sha256"):
        raise ValueError("sampling manifest lacks pairing fingerprints")
    return True


def evaluate_scale(
    *,
    scale: float,
    anchor: dict[str, object],
    other: dict[str, object],
    reference: dict[str, object] | None,
    args: argparse.Namespace,
    sampling_env: dict[str, str],
) -> dict[str, object]:
    prefix = (
        args.common_unique_component
        if args.common_unique_component is not None
        else "static"
        if args.control_mode == "full_pair"
        else args.control_mode
    )
    if args.other_inference_denominator_floor is not None:
        prefix += f"_ifloor{format_scale(args.other_inference_denominator_floor)}"
    condition = f"{prefix}_s{format_scale(scale)}"
    output_dir = args.output_root / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"

    if not valid_sampling_artifact(
        output_dir,
        anchor=anchor,
        other=other,
        reference=reference,
        scale=scale,
        args=args,
    ):
        command = (
            args.torchrun,
            "--standalone",
            f"--nproc_per_node={len(args.sampling_gpu_indices)}",
            str(REPO_ROOT / "experiments/sample_imagenet100_sit_static_pair_fid.py"),
            "--anchor-checkpoint",
            str(args.anchor_checkpoint),
            "--anchor-field",
            args.anchor_field,
            "--other-checkpoint",
            str(args.other_checkpoint),
            "--other-field",
            args.other_field,
            "--static-scale",
            repr(float(scale)),
            "--control-mode",
            args.control_mode,
            "--window-transition-width",
            repr(float(args.window_transition_width)),
            "--output-dir",
            str(output_dir),
            "--weights",
            "ema",
            "--num-samples",
            str(args.num_samples),
            "--per-rank-batch-size",
            str(args.per_rank_batch_size),
            "--vae-decode-batch-size",
            str(args.vae_decode_batch_size),
            "--cuda-allocator-limit-gib",
            str(args.cuda_allocator_limit_gib),
            "--inter-batch-sleep",
            str(args.inter_batch_sleep),
            "--global-seed",
            str(args.global_seed),
            "--precision",
            "fp32",
            "--allow-tf32",
        )
        if args.other_inference_denominator_floor is not None:
            command = (
                *command,
                "--other-inference-denominator-floor",
                repr(float(args.other_inference_denominator_floor)),
            )
        if args.allow_step_mismatch:
            command = (*command, "--allow-step-mismatch")
        if args.common_unique_component is not None:
            assert args.reference_checkpoint is not None
            command = (
                *command,
                "--reference-checkpoint",
                str(args.reference_checkpoint),
                "--reference-field",
                args.reference_field,
                "--common-unique-component",
                args.common_unique_component,
            )
            if args.allow_reference_step_mismatch:
                command = (*command, "--allow-reference-step-mismatch")
        sampling_audit = run_logged(
            command,
            output_dir / "sampling.log",
            env=sampling_env,
            monitored_gpu_indices=args.sampling_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "sampling_resource_audit.json",
        )
        if not valid_sampling_artifact(
            output_dir,
            anchor=anchor,
            other=other,
            reference=reference,
            scale=scale,
            args=args,
        ):
            raise RuntimeError(f"sampler produced invalid artifacts for scale {scale:g}")
    else:
        print(f"[reuse] scale={scale:g} samples", flush=True)
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
            raise RuntimeError(f"FID evaluator produced invalid artifacts for {scale:g}")
    else:
        print(f"[reuse] scale={scale:g} FID", flush=True)
        fid_audit = load_json(output_dir / "fid_resource_audit.json")

    manifest = load_json(output_dir / "sampling_manifest.json")
    result = load_json(output_dir / "fid5k_adm_results.json")
    rank_sampling_stats = manifest.get("rank_sampling_stats", [])
    return {
        "condition": condition,
        "control_mode": args.control_mode,
        "common_unique_component": args.common_unique_component,
        "scale": float(scale),
        "region": (
            "common_unique_guidance"
            if args.common_unique_component is not None
            else "beyond_velocity"
            if scale < 0
            else "interpolation"
            if scale <= 1
            else "beyond_x"
        ),
        "anchor_prediction_target": anchor["prediction_target"],
        "other_prediction_target": other["prediction_target"],
        "other_training_denominator_floor": float(other["denominator_floor"]),
        "other_inference_denominator_floor": (
            float(args.other_inference_denominator_floor)
            if args.other_inference_denominator_floor is not None
            else float(other["denominator_floor"])
        ),
        "checkpoint_step": anchor["checkpoint_step"],
        "other_checkpoint_step": other["checkpoint_step"],
        "reference_checkpoint_step": (
            reference["checkpoint_step"] if reference is not None else None
        ),
        "num_samples": args.num_samples,
        "noise_fingerprint": ":".join(manifest["rank_noise_sha256"]),
        "label_fingerprint": ":".join(manifest["rank_label_sha256"]),
        "sampling_peak_memory_mib": max(
            int(value) for value in sampling_audit["peak_memory_mib"].values()
        ),
        "fid_peak_memory_mib": max(
            int(value) for value in fid_audit["peak_memory_mib"].values()
        ),
        "total_nfe": int(manifest.get("total_nfe", 0)),
        "total_model_forwards": int(manifest.get("total_model_forwards", 0)),
        "sampling_elapsed_seconds_max_rank": max(
            (float(row["elapsed_seconds"]) for row in rank_sampling_stats),
            default=0.0,
        ),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
        "samples": str(sample_path.resolve()),
    }


def save_summary(
    rows: list[dict[str, object]],
    *,
    anchor: dict[str, object],
    other: dict[str, object],
    reference: dict[str, object] | None,
    output_root: Path,
) -> dict[str, object]:
    noise_fingerprints = {str(row["noise_fingerprint"]) for row in rows}
    label_fingerprints = {str(row["label_fingerprint"]) for row in rows}
    if len(noise_fingerprints) != 1 or len(label_fingerprints) != 1:
        raise RuntimeError("static scales did not use identical noise and labels")
    csv_path = output_root / "static_pair_v_to_jit_x_fid5k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: float(row["fid"]))
    summary = {
        "protocol": "imagenet100_sit_field_control_fid5k_v2",
        "control_mode": rows[0]["control_mode"],
        "common_unique_component": rows[0]["common_unique_component"],
        "definition": "see sampling_manifest formula for each condition",
        "comparison_is_paired": True,
        "pairing_verified_by_noise_and_label_sha256": True,
        "anchor": anchor,
        "other": other,
        "reference": reference,
        "best": best,
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(summary, output_root / "static_pair_v_to_jit_x_fid5k.json")
    generic_csv = output_root / "field_control_fid5k.csv"
    if generic_csv != csv_path:
        generic_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    atomic_json_dump(summary, output_root / "field_control_fid5k.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--anchor-field", default="auto", choices=("auto", "x", "epsilon", "dynamic"))
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER_CHECKPOINT)
    parser.add_argument("--other-field", default="auto", choices=("auto", "x", "epsilon", "dynamic"))
    parser.add_argument("--other-inference-denominator-floor", type=float)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=DEFAULT_REFERENCE_CHECKPOINT,
    )
    parser.add_argument(
        "--reference-field",
        default="auto",
        choices=("auto", "x", "epsilon", "dynamic"),
    )
    parser.add_argument(
        "--common-unique-component",
        choices=COMMON_UNIQUE_COMPONENTS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_STATS)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default=shutil.which("torchrun") or "torchrun")
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="full_pair")
    parser.add_argument("--window-transition-width", type=float, default=0.01)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--allow-reference-step-mismatch", action="store_true")
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--inter-batch-sleep", type=float, default=0.0)
    parser.add_argument("--sampling-cuda-visible-devices", default="3")
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--fid-cuda-visible-devices", default="3")
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=8 * 1024)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    parser.add_argument("--global-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.anchor_checkpoint = args.anchor_checkpoint.expanduser().resolve()
    args.other_checkpoint = args.other_checkpoint.expanduser().resolve()
    args.reference_checkpoint = args.reference_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.sampling_gpu_indices = parse_gpu_indices(args.sampling_cuda_visible_devices)
    args.fid_gpu_indices = parse_gpu_indices(args.fid_cuda_visible_devices)
    if len(args.sampling_gpu_indices) != 1 or len(args.fid_gpu_indices) != 1:
        raise ValueError("this low-impact paired sweep must use exactly one GPU")
    if args.sampling_gpu_indices != args.fid_gpu_indices:
        raise ValueError("sampling and FID must use the same reserved GPU")
    if len(set(args.scales)) != len(args.scales):
        raise ValueError("scales must not contain duplicates")
    if not args.scales or any(not math.isfinite(value) for value in args.scales):
        raise ValueError("scales must be finite and non-empty")
    if len({format_scale(value) for value in args.scales}) != len(args.scales):
        raise ValueError("formatted scale names collide")
    if not args.reference.is_file() or not args.adm_python.is_file():
        raise FileNotFoundError("ADM reference statistics or evaluator is missing")
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

    anchor = checkpoint_metadata(args.anchor_checkpoint, args.anchor_field)
    other = checkpoint_metadata(args.other_checkpoint, args.other_field)
    validate_metadata_pair(
        anchor,
        other,
        allow_step_mismatch=args.allow_step_mismatch,
    )
    if (
        args.control_mode in X_FLOOR_CONTROL_MODES
        and other["prediction_target"] != "x"
    ):
        raise ValueError(f"{args.control_mode} requires an x-prediction other checkpoint")
    reference = None
    if args.common_unique_component is not None:
        if args.control_mode != "full_pair":
            raise ValueError("common/unique guidance cannot be combined with pair controls")
        if other["prediction_target"] != "x":
            raise ValueError("common/unique guidance requires x400 as the other field")
        reference = checkpoint_metadata(
            args.reference_checkpoint,
            args.reference_field,
        )
        if reference["prediction_target"] != "velocity":
            raise ValueError("common/unique guidance requires a velocity reference field")
        validate_metadata_pair(
            anchor,
            reference,
            allow_step_mismatch=args.allow_reference_step_mismatch,
        )
    elif args.allow_reference_step_mismatch:
        raise ValueError(
            "--allow-reference-step-mismatch requires --common-unique-component"
        )
    if args.other_inference_denominator_floor is not None:
        if other["prediction_target"] != "x":
            raise ValueError("inference denominator override requires an x-prediction other checkpoint")
        if not 0 < args.other_inference_denominator_floor < 0.5:
            raise ValueError("inference denominator floor must lie in (0, 0.5)")
    if args.control_mode in WINDOW_CONTROL_MODES and not (
        0 < args.window_transition_width < float(other["denominator_floor"])
    ):
        raise ValueError(
            "window transition width must lie in (0, other denominator floor)"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    sampling_env = os.environ.copy()
    sampling_env["CUDA_VISIBLE_DEVICES"] = args.sampling_cuda_visible_devices
    sampling_env.setdefault("OMP_NUM_THREADS", "1")
    sampling_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = [
        evaluate_scale(
            scale=scale,
            anchor=anchor,
            other=other,
            reference=reference,
            args=args,
            sampling_env=sampling_env,
        )
        for scale in args.scales
    ]
    summary = save_summary(
        rows,
        anchor=anchor,
        other=other,
        reference=reference,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

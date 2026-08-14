#!/usr/bin/env python3
"""Run the paired 5K tangent projection mechanism experiment end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from experiments.run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
    )
    from experiments.sample_imagenet100_sit_tangent_projection_fid import CONDITIONS
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        absolute_without_resolving_symlinks,
        fid_environment,
        load_json,
        parse_gpu_indices,
        run_logged,
    )
    from sample_imagenet100_sit_tangent_projection_fid import CONDITIONS
    from train_imagenet100_sit_flow import atomic_json_dump


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "tangent_projection_800k_v1"
)
DEFAULT_REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)


def _parse_directions(value: str) -> list[str]:
    directions = [item.strip() for item in value.split(",") if item.strip()]
    if not directions or any(item not in {"x800", "v500"} for item in directions):
        raise argparse.ArgumentTypeError("directions must be x800 and/or v500")
    if len(set(directions)) != len(directions):
        raise argparse.ArgumentTypeError("directions must not contain duplicates")
    return directions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sampling_complete(output_dir: Path, args: argparse.Namespace) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_json(manifest_path)
    expected = {
        "format": "eqvae_imagenet100_sit_tangent_projection_samples_v1",
        "requested_samples": int(args.num_samples),
        "world_size": int(args.sampling_processes),
        "per_rank_batch_size": int(args.per_rank_batch_size),
        "vae_decode_batch_size": int(args.vae_decode_batch_size),
        "cuda_allocator_limit_gib": float(args.cuda_allocator_limit_gib),
        "global_seed": int(args.global_seed),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if int(manifest.get("sampler", {}).get("steps", -1)) != args.heun_steps:
        mismatches["sampler.steps"] = (
            manifest.get("sampler", {}).get("steps"),
            args.heun_steps,
        )
    if mismatches:
        raise ValueError(f"incompatible existing sampling artifact: {mismatches}")
    for condition in CONDITIONS:
        path = output_dir / condition / f"samples_unguided_n{args.num_samples}.npz"
        if not path.is_file():
            return False
    return True


def _fid_complete(output_dir: Path, sample_path: Path, args: argparse.Namespace) -> bool:
    result_path = output_dir / "fid5k_adm_results.json"
    if not result_path.is_file():
        return False
    result = load_json(result_path)
    if Path(result.get("reference", "")).resolve() != args.reference.resolve():
        raise ValueError(f"FID reference mismatch: {result_path}")
    if Path(result.get("samples", "")).resolve() != sample_path.resolve():
        raise ValueError(f"FID sample mismatch: {result_path}")
    for key in ("fid", "sfid", "inception_score"):
        if not math.isfinite(float(result.get(key, float("nan")))):
            return False
    return True


def _sampling_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in args.sampling_gpu_indices
    )
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _run_sampling(direction: str, args: argparse.Namespace) -> Path:
    output_dir = args.output_root / direction
    output_dir.mkdir(parents=True, exist_ok=True)
    if _sampling_complete(output_dir, args):
        print(f"[reuse] {direction} paired samples", flush=True)
        return output_dir
    command = (
        str(args.torchrun),
        "--standalone",
        f"--nproc_per_node={args.sampling_processes}",
        str(REPO_ROOT / "experiments/sample_imagenet100_sit_tangent_projection_fid.py"),
        "--direction",
        direction,
        "--output-dir",
        str(output_dir),
        "--num-samples",
        str(args.num_samples),
        "--per-rank-batch-size",
        str(args.per_rank_batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--heun-steps",
        str(args.heun_steps),
        "--global-seed",
        str(args.global_seed),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--log-every",
        str(args.log_every),
        "--no-allow-tf32",
    )
    if args.sampling_processes > len(args.sampling_gpu_indices):
        command += ("--process-group-backend", "gloo", "--share-visible-gpus")
    run_logged(
        command,
        output_dir / "sampling.log",
        env=_sampling_environment(args),
        monitored_gpu_indices=args.sampling_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
        memory_poll_interval=args.memory_poll_interval,
        resource_audit_path=output_dir / "sampling_resource_audit.json",
    )
    if not _sampling_complete(output_dir, args):
        raise RuntimeError(f"sampler produced incomplete artifacts: {direction}")
    return output_dir


def _evaluate_condition(
    direction: str,
    condition: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    condition_dir = output_dir / condition
    sample_path = condition_dir / f"samples_unguided_n{args.num_samples}.npz"
    if not _fid_complete(condition_dir, sample_path, args):
        env = fid_environment(
            os.environ.copy(),
            cuda_visible_devices=str(args.fid_gpu_index),
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
                str(condition_dir / "fid5k_adm_results.json"),
            ),
            condition_dir / "evaluation.log",
            env=env,
            monitored_gpu_indices=[args.fid_gpu_index],
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=condition_dir / "fid_resource_audit.json",
        )
    result = load_json(condition_dir / "fid5k_adm_results.json")
    return {
        "direction": direction,
        "condition": condition,
        "num_samples": int(args.num_samples),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
        "samples": str(sample_path),
        "sample_sha256": _sha256(sample_path),
    }


def _assert_pairing(output_dirs: dict[str, Path]) -> None:
    if set(output_dirs) != {"x800", "v500"}:
        return
    x_manifest = load_json(output_dirs["x800"] / "sampling_manifest.json")
    v_manifest = load_json(output_dirs["v500"] / "sampling_manifest.json")
    for key in ("rank_noise_sha256", "rank_label_sha256", "rank_seeds"):
        if x_manifest[key] != v_manifest[key]:
            raise RuntimeError(f"x800/v500 pairing mismatch: {key}")
    x_path = Path(x_manifest["sample_paths"]["baseline"])
    v_path = Path(v_manifest["sample_paths"]["baseline"])
    with np.load(x_path) as x_file, np.load(v_path) as v_file:
        if not np.array_equal(x_file["arr_0"], v_file["arr_0"]):
            raise RuntimeError("x800/v500 baseline images are not bitwise identical")


def main(args: argparse.Namespace) -> None:
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.torchrun = args.torchrun.expanduser().resolve()
    if args.sampling_processes is None:
        args.sampling_processes = len(args.sampling_gpu_indices)
    if args.sampling_processes < 1:
        raise ValueError("sampling_processes must be positive")
    if args.sampling_processes < len(args.sampling_gpu_indices):
        raise ValueError("sampling_processes cannot be smaller than visible sampling GPUs")
    if not args.reference.is_file() or not args.adm_python.is_file():
        raise FileNotFoundError("ADM reference statistics or Python environment is missing")
    if not args.torchrun.is_file():
        raise FileNotFoundError(f"missing torchrun: {args.torchrun}")
    output_dirs = {
        direction: _run_sampling(direction, args) for direction in args.directions
    }
    _assert_pairing(output_dirs)
    rows = [
        _evaluate_condition(direction, condition, output_dirs[direction], args)
        for direction in args.directions
        for condition in CONDITIONS
    ]
    baseline_fids = {
        direction: next(
            float(row["fid"])
            for row in rows
            if row["direction"] == direction and row["condition"] == "baseline"
        )
        for direction in args.directions
    }
    for row in rows:
        row["fid_gain_vs_baseline"] = baseline_fids[str(row["direction"])] - float(
            row["fid"]
        )
    _write_csv(rows, args.output_root / "tangent_projection_fid5k.csv")
    manifests = {
        direction: load_json(output_dirs[direction] / "sampling_manifest.json")
        for direction in args.directions
    }
    summary = {
        "format": "eqvae_imagenet100_sit_tangent_projection_fid5k_v1",
        "directions": args.directions,
        "conditions": list(CONDITIONS),
        "num_samples": int(args.num_samples),
        "global_seed": int(args.global_seed),
        "paired_across_directions": set(args.directions) == {"x800", "v500"},
        "geometry": {
            direction: manifests[direction]["geometry_summary"]
            for direction in args.directions
        },
        "results": rows,
    }
    atomic_json_dump(summary, args.output_root / "tangent_projection_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directions",
        type=_parse_directions,
        default=_parse_directions("x800,v500"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--global-seed", type=int, default=20260814)
    parser.add_argument("--heun-steps", type=int, default=100)
    parser.add_argument("--per-rank-batch-size", type=int, default=32)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=9.5)
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=10240)
    parser.add_argument("--sampling-gpus", type=parse_gpu_indices, default=[0, 1, 2, 3])
    parser.add_argument(
        "--sampling-processes",
        type=int,
        default=None,
        help="Logical sampler ranks; may exceed visible GPUs for paired recovery.",
    )
    parser.add_argument("--fid-gpu-index", type=int, default=0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--memory-poll-interval", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--torchrun",
        type=Path,
        default=Path("/home/zhoushunyu/miniconda3/envs/myenv/bin/torchrun"),
    )
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.sampling_gpu_indices = parsed.sampling_gpus
    main(parsed)

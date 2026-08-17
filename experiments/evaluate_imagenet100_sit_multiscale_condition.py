#!/usr/bin/env python3
"""Sample and ADM-evaluate one multiscale-guidance condition atomically."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")
DEFAULT_REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def valid_result(path: Path, *, expected_samples: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        metrics = payload["metrics"]
        sampling = payload["sampling_manifest"]
        if not isinstance(metrics, dict) or not isinstance(sampling, dict):
            return False
        required = ("fid", "sfid", "inception_score")
        if not all(isinstance(metrics.get(name), (int, float)) for name in required):
            return False
        if expected_samples is not None:
            observed = sampling.get("sampling", {}).get("num_samples")
            if int(observed) != int(expected_samples):
                return False
        return bool(sampling.get("noise_sha256") and sampling.get("label_sha256"))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print(json.dumps({"event": "subprocess", "command": command}), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def main(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.condition_json = args.condition_json.expanduser().resolve()
    args.atlas_summary = args.atlas_summary.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = args.adm_python.expanduser().absolute()
    result_path = args.output_dir / "condition_result.json"
    if valid_result(result_path, expected_samples=args.num_samples):
        print(json.dumps({"event": "reuse", "result": str(result_path)}), flush=True)
        return
    for required in (
        args.condition_json,
        args.atlas_summary,
        args.reference,
        args.adm_python,
        args.strong_checkpoint,
        args.external_weak_checkpoint,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / f"samples_n{args.num_samples}.npz"
    fid_path = args.output_dir / "adm_metrics.json"
    sampling_manifest_path = args.output_dir / "sampling_manifest.json"

    sample_command = [
        sys.executable,
        str(REPO_ROOT / "experiments/sample_imagenet100_sit_multiscale_guidance.py"),
        "--condition-json",
        str(args.condition_json),
        "--atlas-summary",
        str(args.atlas_summary),
        "--output-dir",
        str(args.output_dir),
        "--strong-checkpoint",
        str(args.strong_checkpoint),
        "--external-weak-checkpoint",
        str(args.external_weak_checkpoint),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--device",
        args.device,
    ]
    for name, path in args.head:
        sample_command.extend(("--head", f"{name}={path.expanduser().resolve()}"))
    run(sample_command)
    sampling_manifest = read_json(sampling_manifest_path)
    observed_samples = int(sampling_manifest.get("sampling", {}).get("num_samples", -1))
    if observed_samples != args.num_samples or not sample_path.is_file():
        raise RuntimeError("sampler did not produce the expected complete artifact")

    fid_environment = os.environ.copy()
    fid_environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    fid_command = [
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
        str(fid_path),
    ]
    run(fid_command, environment=fid_environment)
    metrics = read_json(fid_path)
    condition = read_json(args.condition_json)
    payload = {
        "format": "eqvae_imagenet100_sit_multiscale_condition_result_v1",
        "condition": condition,
        "condition_json": str(args.condition_json),
        "atlas_summary": str(args.atlas_summary),
        "sampling_manifest": sampling_manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    atomic_json_dump(payload, result_path)
    if not valid_result(result_path, expected_samples=args.num_samples):
        raise RuntimeError("written condition result failed validation")
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "result": str(result_path),
                "fid": metrics["fid"],
            }
        ),
        flush=True,
    )


def parse_head_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("head must use NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("head must use non-empty NAME=PATH")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-json", type=Path, required=True)
    parser.add_argument("--atlas-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strong-checkpoint", type=Path, required=True)
    parser.add_argument("--external-weak-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--head", action="append", type=parse_head_argument, default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-samples", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

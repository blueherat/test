#!/usr/bin/env python3
"""Run weak-head-difference and frozen-x800 depth-4 readout experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "experiments/train_imagenet100_sit_frozen_internal_v_head.py"
EVALUATOR = REPO_ROOT / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"
DATA_ROOT = Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "weak_difference_x800_depth4_readouts_v1"
DEFAULT_CACHE = DATA_ROOT / "imagenet100_cmc_sdvae"
DEFAULT_V800 = DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_V500 = DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
DEFAULT_X800 = (
    DATA_ROOT
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_DEPTH4 = (
    DATA_ROOT
    / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8 = (
    DATA_ROOT
    / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_ATLAS = (
    DATA_ROOT / "multiscale_guidance_study_v1/atlas/atlas_summary.json"
)
DEFAULT_REFERENCE = (
    DATA_ROOT / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")
DEFAULT_HEAD_DIFFERENCE_GAMMAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_READOUT_GAMMAS = (-0.5, 0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0)
TARGETS = ("velocity", "clean", "epsilon")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def gamma_tag(value: float) -> str:
    sign = "m" if value < 0 else ""
    return sign + f"{abs(value):g}".replace(".", "p")


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one floating-point value")
    return result


def valid_result(path: Path, expected_samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        sampling = payload["sampling_manifest"]
        return (
            int(sampling["sampling"]["num_samples"]) == expected_samples
            and all(isinstance(metrics[key], (int, float)) for key in ("fid", "sfid", "inception_score"))
            and bool(sampling["noise_sha256"])
            and bool(sampling["label_sha256"])
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    gpu: int,
    extra_environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        "/home/zhoushunyu/data/eqvae/torchinductor_cache",
    )
    if extra_environment:
        environment.update(extra_environment)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def head_run_dir(args: argparse.Namespace, target: str) -> Path:
    return args.output_root / "runs" / f"x800_depth4_{target}"


def head_checkpoint(args: argparse.Namespace, target: str) -> Path:
    return head_run_dir(args, target) / "checkpoints" / f"step_{args.train_steps:08d}.pt"


def train_heads(args: argparse.Namespace) -> None:
    for target in TARGETS:
        checkpoint = head_checkpoint(args, target)
        if checkpoint.is_file():
            print(json.dumps({"event": "reuse_head", "target": target, "checkpoint": str(checkpoint)}), flush=True)
            continue
        output_dir = head_run_dir(args, target)
        command = [
            sys.executable,
            str(TRAINER),
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(output_dir),
            "--source-checkpoint",
            str(args.x800_checkpoint),
            "--source-state-key",
            "ema",
            "--internal-depth",
            "4",
            "--prediction-target",
            target,
            "--clean-velocity-denominator-floor",
            "0.05",
            "--global-batch-size",
            str(args.train_batch_size),
            "--max-steps",
            str(args.train_steps),
            "--save-every",
            str(args.save_every),
            "--validation-every",
            str(args.validation_every),
            "--validation-batches",
            str(args.validation_batches),
            "--log-every",
            str(args.log_every),
            "--seed",
            str(args.train_seed),
            "--device",
            "cuda:0",
            "--resume",
            "auto",
        ]
        command.append("--compile" if args.compile else "--no-compile")
        run_logged(
            command,
            log_path=args.output_root / "logs" / f"train_x800_depth4_{target}.log",
            gpu=args.gpu,
        )
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not produce {checkpoint}")


def condition_payload(
    *,
    name: str,
    kind: str,
    gamma: float,
    formula: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
        "name": name,
        "evaluation_group": "weak_difference_x800_depth4_readouts_v1",
        "hypothesis_id": kind,
        "kind": kind,
        "gamma": float(gamma),
        "field_formula": formula,
        **(extra or {}),
    }


def evaluate_condition(
    args: argparse.Namespace,
    *,
    family: str,
    condition: dict[str, Any],
    strong_checkpoint: Path,
    heads: dict[str, Path],
) -> dict[str, Any]:
    name = str(condition["name"])
    condition_path = args.output_root / "conditions" / f"{name}.json"
    output_dir = args.output_root / "evaluations" / family / name
    result_path = output_dir / "condition_result.json"
    atomic_json(condition_path, condition)
    if valid_result(result_path, args.num_samples):
        print(json.dumps({"event": "reuse_evaluation", "name": name}), flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(EVALUATOR),
        "--condition-json",
        str(condition_path),
        "--atlas-summary",
        str(args.atlas_summary),
        "--output-dir",
        str(output_dir),
        "--strong-checkpoint",
        str(strong_checkpoint),
        "--external-weak-checkpoint",
        str(args.v500_checkpoint),
        "--reference",
        str(args.reference),
        "--adm-python",
        str(args.adm_python),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.sample_batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.sample_seed),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
        "--device",
        "cuda:0",
    ]
    for head_name, path in sorted(heads.items()):
        command.extend(("--head", f"{head_name}={path}"))
    run_logged(
        command,
        log_path=args.output_root / "logs" / f"evaluate_{name}.log",
        gpu=args.gpu,
    )
    if not valid_result(result_path, args.num_samples):
        raise RuntimeError(f"invalid evaluation result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_head_difference_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    heads = {"depth4_v": args.depth4_head, "depth8_v": args.depth8_head}
    for gamma in args.head_difference_gammas:
        name = f"v800_plus_g_depth8_minus_depth4_g{gamma_tag(gamma)}"
        result = evaluate_condition(
            args,
            family="head_difference",
            condition=condition_payload(
                name=name,
                kind="head_difference",
                gamma=gamma,
                formula="v800 + gamma * (depth8_v - depth4_v)",
                extra={"positive_head": "depth8_v", "negative_head": "depth4_v"},
            ),
            strong_checkpoint=args.v800_checkpoint,
            heads=heads,
        )
        rows.append(result)
    return rows


def run_x800_readout_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        provider = f"x800_depth4_{target}"
        heads = {provider: head_checkpoint(args, target)}
        for gamma in args.readout_gammas:
            name = f"x800_plus_g_strong_minus_depth4_{target}_g{gamma_tag(gamma)}"
            result = evaluate_condition(
                args,
                family=f"x800_{target}",
                condition=condition_payload(
                    name=name,
                    kind="full_gap",
                    gamma=gamma,
                    formula=f"velocity(x800) + gamma * (velocity(x800) - velocity(depth4_{target}))",
                    extra={"provider": provider},
                ),
                strong_checkpoint=args.x800_checkpoint,
                heads=heads,
            )
            rows.append(result)
    return rows


def write_summary(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted((args.output_root / "evaluations").rglob("condition_result.json")):
        if not valid_result(path, args.num_samples):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload["condition"]
        metrics = payload["metrics"]
        sampling = payload["sampling_manifest"]
        records.append(
            {
                "family": path.parent.parent.name,
                "name": condition["name"],
                "kind": condition["kind"],
                "gamma": condition["gamma"],
                "formula": condition.get("field_formula"),
                "provider": condition.get("provider"),
                "positive_head": condition.get("positive_head"),
                "negative_head": condition.get("negative_head"),
                "fid": metrics["fid"],
                "sfid": metrics["sfid"],
                "inception_score": metrics["inception_score"],
                "noise_sha256": sampling["noise_sha256"],
                "label_sha256": sampling["label_sha256"],
                "total_nfe": sampling["total_nfe"],
                "elapsed_seconds": sampling["elapsed_seconds"],
            }
        )
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with (summary_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    grouped_best: dict[str, dict[str, Any]] = {}
    for record in records:
        family = str(record["family"])
        if family not in grouped_best or float(record["fid"]) < float(grouped_best[family]["fid"]):
            grouped_best[family] = record
    atomic_json(
        summary_dir / "summary.json",
        {
            "format": "eqvae_weak_difference_x800_depth4_readouts_summary_v1",
            "num_samples": args.num_samples,
            "records": records,
            "best_by_family": grouped_best,
        },
    )


def validate_inputs(args: argparse.Namespace) -> None:
    required = (
        args.cache_dir / "manifest.json",
        args.v800_checkpoint,
        args.v500_checkpoint,
        args.x800_checkpoint,
        args.depth4_head,
        args.depth8_head,
        args.atlas_summary,
        args.reference,
        args.adm_python,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def main(args: argparse.Namespace) -> None:
    for name in (
        "output_root",
        "cache_dir",
        "v800_checkpoint",
        "v500_checkpoint",
        "x800_checkpoint",
        "depth4_head",
        "depth8_head",
        "atlas_summary",
        "reference",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Keep the ADM environment path itself instead of resolving its Python
    # symlink into the training environment.
    args.adm_python = args.adm_python.expanduser().absolute()
    validate_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "eqvae_weak_difference_x800_depth4_readouts_protocol_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": args.gpu,
        "num_samples": args.num_samples,
        "sample_seed": args.sample_seed,
        "train_seed": args.train_seed,
        "train_steps": args.train_steps,
        "head_difference_formula": "v800 + gamma * (depth8_v - depth4_v)",
        "head_difference_gammas": list(args.head_difference_gammas),
        "x800_readout_formula": "velocity(x800) + gamma * (velocity(x800) - velocity(depth4_target))",
        "readout_targets": list(TARGETS),
        "readout_gammas": list(args.readout_gammas),
        "paired_sampling": True,
    }
    atomic_json(args.output_root / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2), flush=True)

    if args.phase in {"all", "head_difference"}:
        run_head_difference_scan(args)
        write_summary(args)
    if args.phase in {"all", "train_heads"}:
        train_heads(args)
    if args.phase in {"all", "x800_scan"}:
        for target in TARGETS:
            if not head_checkpoint(args, target).is_file():
                raise FileNotFoundError(head_checkpoint(args, target))
        run_x800_readout_scan(args)
        write_summary(args)
    print(json.dumps({"status": "complete", "summary": str(args.output_root / "summary/summary.json")}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--v800-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--v500-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--x800-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--depth4-head", type=Path, default=DEFAULT_DEPTH4)
    parser.add_argument("--depth8-head", type=Path, default=DEFAULT_DEPTH8)
    parser.add_argument("--atlas-summary", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("all", "head_difference", "train_heads", "x800_scan"),
        default="all",
    )
    parser.add_argument("--head-difference-gammas", type=parse_float_list, default=DEFAULT_HEAD_DIFFERENCE_GAMMAS)
    parser.add_argument("--readout-gammas", type=parse_float_list, default=DEFAULT_READOUT_GAMMAS)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--validation-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

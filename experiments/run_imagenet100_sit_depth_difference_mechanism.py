#!/usr/bin/env python3
"""Run causal depth-difference projections and cross-backbone transfer scans."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"
BASE = Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow")
DEFAULT_OUTPUT = BASE / "depth_difference_mechanism_v1"
DEFAULT_V800 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_V500 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
DEFAULT_X800 = (
    BASE / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_V_DEPTH4 = (
    BASE / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
)
DEFAULT_V_DEPTH8 = (
    BASE
    / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_X_DEPTH4 = (
    BASE
    / "weak_difference_x800_depth4_readouts_v1/runs/x800_depth4_velocity/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_X_DEPTH8 = (
    BASE
    / "weak_difference_x800_depth8_readouts_v1/velocity/runs/"
    "x800_depth8_velocity/checkpoints/step_00050000.pt"
)
DEFAULT_ATLAS = BASE / "multiscale_guidance_study_v1/atlas/atlas_summary.json"
DEFAULT_REFERENCE = BASE / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")


def parse_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one scale")
    return parsed


def gamma_tag(value: float) -> str:
    prefix = "m" if value < 0 else ""
    return prefix + f"{abs(value):g}".replace(".", "p")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def valid_result(path: Path, samples: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(payload["sampling_manifest"]["sampling"]["num_samples"]) == samples
            and bool(payload["sampling_manifest"]["noise_sha256"])
            and bool(payload["sampling_manifest"]["label_sha256"])
            and all(
                isinstance(payload["metrics"][name], (int, float))
                for name in ("fid", "sfid", "inception_score")
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_logged(command: list[str], *, log: Path, gpu: int) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log.open("a", encoding="utf-8") as handle:
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


def evaluate(
    args: argparse.Namespace,
    *,
    family: str,
    condition: dict[str, Any],
    strong: Path,
    heads: dict[str, Path],
) -> dict[str, Any]:
    name = str(condition["name"])
    condition_path = args.output_root / "conditions" / f"{name}.json"
    output_dir = args.output_root / "evaluations" / family / name
    result_path = output_dir / "condition_result.json"
    atomic_json(condition_path, condition)
    if not valid_result(result_path, args.num_samples):
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
            str(strong),
            "--external-weak-checkpoint",
            str(args.v500_checkpoint),
            "--reference",
            str(args.reference),
            "--adm-python",
            str(args.adm_python),
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.batch_size),
            "--vae-decode-batch-size",
            str(args.vae_decode_batch_size),
            "--seed",
            str(args.seed),
            "--cuda-allocator-limit-gib",
            str(args.cuda_allocator_limit_gib),
            "--fid-batch-size",
            str(args.fid_batch_size),
            "--fid-gpu-memory-fraction",
            str(args.fid_gpu_memory_fraction),
            "--device",
            "cuda:0",
        ]
        for head_name, head_path in sorted(heads.items()):
            command.extend(("--head", f"{head_name}={head_path}"))
        run_logged(
            command,
            log=args.output_root / "logs" / f"{name}.log",
            gpu=args.gpu,
        )
    if not valid_result(result_path, args.num_samples):
        raise RuntimeError(f"invalid result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def condition(
    *,
    name: str,
    kind: str,
    gamma: float,
    formula: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
        "name": name,
        "evaluation_group": "depth_difference_mechanism_v1",
        "hypothesis_id": kind,
        "kind": kind,
        "gamma": float(gamma),
        "field_formula": formula,
        **extra,
    }


def projection_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    heads = {"depth4_v": args.v_depth4_head, "depth8_v": args.v_depth8_head}
    settings = [("baseline", "full", 0.0)] + [
        (component, component, args.projection_gamma)
        for component in ("full", "parallel", "orthogonal")
    ]
    for label, component_name, gamma in settings:
        name = f"v800_depth_difference_{label}_g{gamma_tag(gamma)}"
        rows.append(
            evaluate(
                args,
                family="v800_projection",
                condition=condition(
                    name=name,
                    kind="head_difference_component",
                    gamma=gamma,
                    formula=(
                        "v800 + gamma * component(depth8_v - depth4_v; "
                        "reference=v800-depth4_v)"
                    ),
                    extra={
                        "positive_head": "depth8_v",
                        "negative_head": "depth4_v",
                        "component": component_name,
                    },
                ),
                strong=args.v800_checkpoint,
                heads=heads,
            )
        )
    return rows


def reverse_projection_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    heads = {"depth4_v": args.v_depth4_head, "depth8_v": args.v_depth8_head}
    settings = [("baseline", "full", 0.0)] + [
        (component, component, args.reverse_projection_gamma)
        for component in ("full", "parallel", "orthogonal")
    ]
    for label, component_name, gamma in settings:
        name = f"v800_full_gap_onto_weak_difference_{label}_g{gamma_tag(gamma)}"
        rows.append(
            evaluate(
                args,
                family="v800_reverse_projection",
                condition=condition(
                    name=name,
                    kind="head_difference_component",
                    gamma=gamma,
                    formula=(
                        "v800 + gamma * component(v800 - depth4_v; "
                        "reference=depth8_v-depth4_v)"
                    ),
                    extra={
                        "positive_head": "depth8_v",
                        "negative_head": "depth4_v",
                        "component": component_name,
                        "projection_orientation": "full_gap_onto_difference",
                    },
                ),
                strong=args.v800_checkpoint,
                heads=heads,
            )
        )
    return rows


def x_transfer_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    heads = {"depth4_v": args.x_depth4_head, "depth8_v": args.x_depth8_head}
    for gamma in args.x_transfer_gammas:
        name = f"x800_plus_g_depth8v_minus_depth4v_g{gamma_tag(gamma)}"
        rows.append(
            evaluate(
                args,
                family="x800_transfer",
                condition=condition(
                    name=name,
                    kind="head_difference",
                    gamma=gamma,
                    formula="velocity(x800) + gamma * (depth8_v - depth4_v)",
                    extra={
                        "positive_head": "depth8_v",
                        "negative_head": "depth4_v",
                    },
                ),
                strong=args.x800_checkpoint,
                heads=heads,
            )
        )
    return rows


def write_summary(args: argparse.Namespace, family: str, rows: list[dict[str, Any]]) -> None:
    records = []
    for payload in rows:
        cond = payload["condition"]
        metrics = payload["metrics"]
        manifest = payload["sampling_manifest"]
        records.append(
            {
                "family": family,
                "name": cond["name"],
                "kind": cond["kind"],
                "component": cond.get("component"),
                "gamma": cond["gamma"],
                "fid": metrics["fid"],
                "sfid": metrics["sfid"],
                "inception_score": metrics["inception_score"],
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
                "total_nfe": manifest["total_nfe"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{family}_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    best = min(records, key=lambda row: float(row["fid"]))
    atomic_json(
        summary_dir / f"{family}_summary.json",
        {"family": family, "num_samples": args.num_samples, "records": records, "best": best},
    )
    print(json.dumps({"family": family, "best": best}, indent=2), flush=True)


def validate(args: argparse.Namespace) -> None:
    paths = [
        args.v800_checkpoint,
        args.v500_checkpoint,
        args.x800_checkpoint,
        args.v_depth4_head,
        args.v_depth8_head,
        args.x_depth4_head,
        args.x_depth8_head,
        args.atlas_summary,
        args.reference,
        args.adm_python,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("v_projection", "v_reverse_projection", "x_transfer"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v800-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--v500-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--x800-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--v-depth4-head", type=Path, default=DEFAULT_V_DEPTH4)
    parser.add_argument("--v-depth8-head", type=Path, default=DEFAULT_V_DEPTH8)
    parser.add_argument("--x-depth4-head", type=Path, default=DEFAULT_X_DEPTH4)
    parser.add_argument("--x-depth8-head", type=Path, default=DEFAULT_X_DEPTH8)
    parser.add_argument("--atlas-summary", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--projection-gamma", type=float, default=0.65)
    parser.add_argument("--reverse-projection-gamma", type=float, default=0.25)
    parser.add_argument(
        "--x-transfer-gammas",
        type=parse_float_list,
        default=(0.0, 0.25, 0.5, 0.65, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0),
    )
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(args: argparse.Namespace) -> None:
    for name in (
        "output_root",
        "v800_checkpoint",
        "v500_checkpoint",
        "x800_checkpoint",
        "v_depth4_head",
        "v_depth8_head",
        "x_depth4_head",
        "x_depth8_head",
        "atlas_summary",
        "reference",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.adm_python = args.adm_python.expanduser().absolute()
    validate(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "v_projection":
        write_summary(args, "v800_projection", projection_scan(args))
    elif args.phase == "v_reverse_projection":
        write_summary(
            args,
            "v800_reverse_projection",
            reverse_projection_scan(args),
        )
    else:
        write_summary(args, "x800_transfer", x_transfer_scan(args))


if __name__ == "__main__":
    main(build_parser().parse_args())

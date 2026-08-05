#!/usr/bin/env python3
"""Sweep RAEv2 internal-guidance scale over paired checkpoints.

This is an orchestration layer over the repository's existing strict protocol:

1. ``experiments/sample_raev2_threeway.py`` generates same-noise samples for
   every checkpoint at one IG scale.
2. ``experiments/evaluate_raev2_samples.py`` evaluates all branches against the
   same ImageNet reference.
3. This script aggregates every scale into one CSV/JSON table and optionally
   reports deltas relative to a named baseline branch.

The script intentionally does not reimplement model loading, guidance, sampling,
decoding, or torch-fidelity metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SCRIPT = ROOT / "experiments" / "sample_raev2_threeway.py"
EVALUATE_SCRIPT = ROOT / "experiments" / "evaluate_raev2_samples.py"
MANIFEST_NAME = "sweep_manifest.json"
COMMAND_HISTORY_NAME = "command_history.jsonl"

DEFAULT_SCALES = "0.8,1.0,1.2,1.4,1.6,1.78,1.9,2.1,2.3"
SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not SAFE_BRANCH_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "branch NAME must contain only letters, digits, '.', '_' or '-'"
        )
    path = Path(raw_path).expanduser().resolve()
    return name, path


def parse_scales(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must be comma-separated floats") from exc
    if not scales:
        raise argparse.ArgumentTypeError("at least one scale is required")
    if any((not math.isfinite(scale)) or scale < 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be finite and non-negative")
    if len(set(scales)) != len(scales):
        raise argparse.ArgumentTypeError("scales must not contain duplicates")
    return scales


def scale_slug(scale: float) -> str:
    text = f"{float(scale):.6f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return f"ig_{text}"


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("devices must be a non-empty comma-separated list")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("devices must not contain duplicates")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict same-noise RAEv2 IG-scale sweep."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "sample", "evaluate", "summarize"),
        default="all",
        help="all: sample+evaluate+summarize; other modes run one stage.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--branch",
        action="append",
        type=parse_branch,
        required=True,
        help="Repeat NAME=CHECKPOINT for every paired branch.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scales", type=parse_scales, default=parse_scales(DEFAULT_SCALES))
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--sampling-seed", type=int, default=20260804)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--state-key",
        choices=("model", "ema"),
        default="model",
        help="The existing short continuation studies use online model weights.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices("0,1,2,3"),
        help="Physical GPU IDs exposed to torchrun, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        help="Defaults to the number of --devices.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--metric-seed", type=int, default=20260804)
    parser.add_argument(
        "--baseline-branch",
        help="Optional branch used for per-scale metric deltas, e.g. flow10.",
    )
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    parser.add_argument(
        "--rerun-sampling",
        action="store_true",
        help="Delete and regenerate samples even when complete archives exist.",
    )
    parser.add_argument(
        "--rerun-metrics",
        action="store_true",
        help="Recompute metrics even when a scale-level metrics.csv exists.",
    )
    parser.add_argument(
        "--delete-samples-after-metrics",
        action="store_true",
        help="Delete large samples.npz files after successful metric evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.config = args.config.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.dino_ckpt_dir = args.dino_ckpt_dir.expanduser().resolve()
    if args.dino_repo_dir is not None:
        args.dino_repo_dir = args.dino_repo_dir.expanduser().resolve()

    if not SAMPLE_SCRIPT.is_file():
        raise FileNotFoundError(SAMPLE_SCRIPT)
    if not EVALUATE_SCRIPT.is_file():
        raise FileNotFoundError(EVALUATE_SCRIPT)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.mode in ("all", "evaluate") and not args.reference.is_file():
        raise FileNotFoundError(args.reference)

    if args.sample_count <= 0 or args.sample_count % 1000:
        raise ValueError("--sample-count must be a positive multiple of 1000")
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")

    names = [name for name, _ in args.branch]
    if len(names) != len(set(names)):
        raise ValueError("branch names must be unique")
    for name, checkpoint in args.branch:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name}: checkpoint not found: {checkpoint}")

    if args.baseline_branch is not None and args.baseline_branch not in names:
        raise ValueError(
            f"--baseline-branch {args.baseline_branch!r} is not among {names}"
        )

    if args.nproc_per_node is None:
        args.nproc_per_node = len(args.devices)
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")
    if args.nproc_per_node > len(args.devices):
        raise ValueError("--nproc-per-node cannot exceed the number of --devices")


def manifest_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": "raev2_ig_scale_sweep_v1",
        "repository_root": str(ROOT),
        "sample_script": str(SAMPLE_SCRIPT),
        "evaluate_script": str(EVALUATE_SCRIPT),
        "config": str(args.config),
        "branches": [
            {"name": name, "checkpoint": str(checkpoint)}
            for name, checkpoint in args.branch
        ],
        "scales": [float(scale) for scale in args.scales],
        "sample_count": int(args.sample_count),
        "per_rank_batch": int(args.per_rank_batch),
        "sampling_seed": int(args.sampling_seed),
        "precision": args.precision,
        "state_key": args.state_key,
        "devices": list(args.devices),
        "nproc_per_node": int(args.nproc_per_node),
        "reference": str(args.reference),
        "metric_batch_size": int(args.metric_batch_size),
        "metric_seed": int(args.metric_seed),
        "baseline_branch": args.baseline_branch,
        "dino_ckpt_dir": str(args.dino_ckpt_dir),
        "dino_repo_dir": (
            None if args.dino_repo_dir is None else str(args.dino_repo_dir)
        ),
    }


def prepare_manifest(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / MANIFEST_NAME
    payload = manifest_payload(args)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise RuntimeError(
                f"{path} already exists with a different protocol. "
                "Use a new --output-root rather than mixing sweeps."
            )
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(args.devices)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    return env


def record_command(
    args: argparse.Namespace,
    *,
    stage: str,
    scale: float,
    command: list[str],
) -> None:
    payload = {
        "stage": stage,
        "ig_scale": float(scale),
        "cwd": str(ROOT),
        "cuda_visible_devices": ",".join(args.devices),
        "command": command,
        "shell": shlex.join(command),
    }
    with (args.output_root / COMMAND_HISTORY_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_command(
    args: argparse.Namespace,
    *,
    stage: str,
    scale: float,
    command: list[str],
) -> None:
    record_command(args, stage=stage, scale=scale, command=command)
    print(f"\n[{stage} | IG={scale:g}]\n{shlex.join(command)}\n", flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=command_env(args), check=True)


def scale_paths(args: argparse.Namespace, scale: float) -> dict[str, Path]:
    root = args.output_root / scale_slug(scale)
    samples_root = root / "samples"
    return {
        "root": root,
        "samples_root": samples_root,
        "metrics_csv": root / "metrics.csv",
        "metrics_json": root / "metrics.json",
    }


def archive_paths(args: argparse.Namespace, scale: float) -> dict[str, Path]:
    samples_root = scale_paths(args, scale)["samples_root"]
    return {
        name: samples_root / name / "samples.npz"
        for name, _ in args.branch
    }


def samples_complete(args: argparse.Namespace, scale: float) -> bool:
    samples_root = scale_paths(args, scale)["samples_root"]
    return all(
        (samples_root / name / "samples.npz").is_file()
        and (samples_root / name / "sampling_summary.json").is_file()
        for name, _ in args.branch
    )


def sampling_command(args: argparse.Namespace, scale: float) -> list[str]:
    paths = scale_paths(args, scale)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.nproc_per_node}",
        str(SAMPLE_SCRIPT),
        "--config",
        str(args.config),
        "--results-dir",
        str(paths["samples_root"]),
        "--sample-count",
        str(args.sample_count),
        "--per-rank-batch",
        str(args.per_rank_batch),
        "--sampling-seed",
        str(args.sampling_seed),
        "--precision",
        args.precision,
        "--state-key",
        args.state_key,
        "--ig-scale",
        str(float(scale)),
        "--dino-ckpt-dir",
        str(args.dino_ckpt_dir),
    ]
    if args.dino_repo_dir is not None:
        command.extend(["--dino-repo-dir", str(args.dino_repo_dir)])
    for name, checkpoint in args.branch:
        command.extend(["--branch", f"{name}={checkpoint}"])
    return command


def evaluation_command(args: argparse.Namespace, scale: float) -> list[str]:
    paths = scale_paths(args, scale)
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--reference",
        str(args.reference),
        "--output",
        str(paths["metrics_csv"]),
        "--batch-size",
        str(args.metric_batch_size),
        "--seed",
        str(args.metric_seed),
    ]
    for name, archive in archive_paths(args, scale).items():
        command.extend(["--branch", f"{name}={archive}"])
    return command


def process_scale(args: argparse.Namespace, scale: float) -> None:
    paths = scale_paths(args, scale)
    paths["root"].mkdir(parents=True, exist_ok=True)

    # A completely evaluated scale is resumable even when sample archives were
    # deliberately deleted to save disk.
    if (
        args.mode == "all"
        and paths["metrics_csv"].is_file()
        and not args.rerun_sampling
        and not args.rerun_metrics
    ):
        print(f"[skip] IG={scale:g}: metrics already complete")
        return

    if args.mode in ("all", "sample"):
        if args.rerun_sampling:
            if paths["samples_root"].exists():
                shutil.rmtree(paths["samples_root"])
            for metric_path in (paths["metrics_csv"], paths["metrics_json"]):
                metric_path.unlink(missing_ok=True)

        if samples_complete(args, scale):
            print(f"[skip] IG={scale:g}: samples already complete")
        else:
            run_command(
                args,
                stage="sample",
                scale=scale,
                command=sampling_command(args, scale),
            )
            if not args.dry_run and not samples_complete(args, scale):
                raise RuntimeError(f"IG={scale:g}: sampling command finished incompletely")

    if args.mode in ("all", "evaluate"):
        # Dry run does not create samples. Print the planned evaluation command
        # without checking whether samples.npz already exists.
        if args.dry_run:
            run_command(
                args,
                stage="evaluate",
                scale=scale,
                command=evaluation_command(args, scale),
            )
            return

        if args.rerun_metrics:
            paths["metrics_csv"].unlink(missing_ok=True)
            paths["metrics_json"].unlink(missing_ok=True)

        if paths["metrics_csv"].is_file():
            print(f"[skip] IG={scale:g}: metrics already complete")
        else:
            missing = [
                str(path)
                for path in archive_paths(args, scale).values()
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "cannot evaluate because sample archives are missing:\n"
                    + "\n".join(missing)
                )
            run_command(
                args,
                stage="evaluate",
                scale=scale,
                command=evaluation_command(args, scale),
            )
            if not args.dry_run and not paths["metrics_csv"].is_file():
                raise RuntimeError(f"IG={scale:g}: metric command produced no CSV")

        if (
            args.delete_samples_after_metrics
            and paths["metrics_csv"].is_file()
            and not args.dry_run
        ):
            deleted: list[str] = []
            for archive in archive_paths(args, scale).values():
                if archive.is_file():
                    deleted.append(str(archive))
                    archive.unlink()
            marker = paths["root"] / "samples_deleted_after_metrics.json"
            marker.write_text(
                json.dumps(
                    {
                        "ig_scale": float(scale),
                        "deleted_archives": deleted,
                        "metrics_csv": str(paths["metrics_csv"]),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )


def choose_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def summarize(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_scales: list[float] = []
    for scale in args.scales:
        metrics_path = scale_paths(args, scale)["metrics_csv"]
        if not metrics_path.is_file():
            missing_scales.append(float(scale))
            continue
        local = pd.read_csv(metrics_path)
        local.insert(0, "ig_scale", float(scale))
        local.insert(1, "scale_slug", scale_slug(scale))
        frames.append(local)

    if not frames:
        if args.dry_run:
            print("[dry-run] no metrics to summarize")
            return pd.DataFrame()
        raise RuntimeError("no completed metrics.csv files were found")

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["branch", "ig_scale"]).reset_index(drop=True)

    fid_col = choose_column(
        frame,
        ("frechet_inception_distance", "fid", "FID"),
    )
    kid_col = choose_column(
        frame,
        ("kernel_inception_distance_mean", "kid_mean", "KID"),
    )
    is_col = choose_column(
        frame,
        ("inception_score_mean", "is_mean", "IS"),
    )

    if args.baseline_branch is not None:
        metric_columns = [
            column for column in (fid_col, kid_col, is_col) if column is not None
        ]
        baseline = frame[frame["branch"].eq(args.baseline_branch)][
            ["ig_scale", *metric_columns]
        ].copy()
        if baseline["ig_scale"].duplicated().any():
            raise RuntimeError("baseline branch has duplicate rows at one scale")
        rename = {
            column: f"{column}__baseline_{args.baseline_branch}"
            for column in metric_columns
        }
        baseline = baseline.rename(columns=rename)
        frame = frame.merge(baseline, on="ig_scale", how="left", validate="many_to_one")
        for column in metric_columns:
            base_column = rename[column]
            frame[f"{column}__delta_vs_{args.baseline_branch}"] = (
                frame[column] - frame[base_column]
            )

    summary_csv = args.output_root / "ig_scale_sweep_summary.csv"
    summary_json = args.output_root / "ig_scale_sweep_summary.json"
    frame.to_csv(summary_csv, index=False)
    frame.to_json(summary_json, orient="records", indent=2, force_ascii=False)

    if fid_col is not None:
        valid = frame.dropna(subset=[fid_col])
        best_indices = valid.groupby("branch", sort=True)[fid_col].idxmin()
        best = valid.loc[best_indices].sort_values("branch").reset_index(drop=True)
        best.insert(0, "selected_by", fid_col)
        best.to_csv(args.output_root / "best_scale_by_branch.csv", index=False)
        best.to_json(
            args.output_root / "best_scale_by_branch.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
        display_columns = [
            column
            for column in ("branch", "ig_scale", fid_col, kid_col, is_col)
            if column is not None
        ]
        print("\nBest observed scale per branch (coarse search):")
        print(best[display_columns].to_string(index=False))

    if missing_scales:
        print(f"\nMissing metric scales: {missing_scales}")
    print(f"\nWrote: {summary_csv}")
    return frame


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_manifest(args)

    if args.mode != "summarize":
        for scale in args.scales:
            process_scale(args, float(scale))

    if args.mode in ("all", "summarize", "evaluate"):
        summarize(args)


if __name__ == "__main__":
    main()
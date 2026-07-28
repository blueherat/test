"""Train one dense paired RAE-LPL trajectory, audit it, and screen all checkpoints."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_lpl_checkpoint_curve"
DEFAULT_LPL_WEIGHT = 0.000732496037420993
ENDPOINTS = tuple(range(500, 5001, 500))
TRAINING_CONFIG = (
    ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml"
)


def branch_path(results: Path, seed: int, objective: str) -> Path:
    return results / f"ditdh_s_ep20_seed{seed}_{objective}_to_s5000"


def branch_complete(branch: Path) -> bool:
    required = [
        branch / "manifest.json",
        branch / "metrics.jsonl",
        *(branch / "checkpoints" / f"step-{step:07d}.pt" for step in ENDPOINTS),
    ]
    return all(path.exists() for path in required)


def train_command(
    *,
    results: Path,
    seed: int,
    objective: str,
    lpl_weight: float,
    devices: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments/run_rae_lpl_authenticity_validation.py"),
        "--mode",
        "train",
        "--prior",
        "ditdh_s_ep20",
        "--objective",
        objective,
        "--seed",
        str(seed),
        "--endpoint",
        "5000",
        "--results-dir",
        str(results),
        "--devices",
        devices,
    ]
    if objective == "full":
        command.extend(["--lpl-weight", str(lpl_weight)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=4102)
    parser.add_argument("--lpl-weight", type=float, default=DEFAULT_LPL_WEIGHT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--screen-sample-count", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=20260715)
    parser.add_argument("--skip-complete-training", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.lpl_weight <= 0:
        raise ValueError("LPL weight must be positive")
    if args.screen_sample_count <= 0 or args.screen_sample_count % 1000 != 0:
        raise ValueError("screen sample count must be a positive multiple of 1000")
    if len(args.devices.split(",")) != 4:
        raise ValueError("dense paired training requires exactly four devices")
    config = OmegaConf.load(TRAINING_CONFIG)
    configured_endpoints = tuple(
        int(step) for step in config.training.checkpoint_offsets
    )
    if configured_endpoints != ENDPOINTS:
        raise ValueError(
            "training config does not contain the required dense checkpoints: "
            f"{configured_endpoints}"
        )

    results = args.results_dir.expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUNBUFFERED"] = "1"

    branches = {
        objective: branch_path(results, args.seed, objective)
        for objective in ("flow", "full")
    }
    for objective, branch in branches.items():
        command = train_command(
            results=results,
            seed=args.seed,
            objective=objective,
            lpl_weight=args.lpl_weight,
            devices=args.devices,
        )
        print(" ".join(command), flush=True)
        if args.print_only:
            continue
        if branch.exists():
            if args.skip_complete_training and branch_complete(branch):
                continue
            raise FileExistsError(
                f"{branch} already exists; refuse to overwrite a scientific run"
            )
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if not branch_complete(branch):
            raise RuntimeError(f"dense checkpoint branch is incomplete: {branch}")

    audit_output = results / f"ditdh_s_ep20_seed{args.seed}_dense_pair_audit.json"
    audit_command = [
        sys.executable,
        str(ROOT / "experiments/audit_rae_lpl_authenticity.py"),
        "--flow",
        str(branches["flow"]),
        "--lpl",
        str(branches["full"]),
        "--output",
        str(audit_output),
    ]
    curve_output = results / f"seed{args.seed}_screen"
    curve_command = [
        sys.executable,
        str(ROOT / "experiments/run_rae_lpl_checkpoint_curve.py"),
        "--flow-branch",
        str(branches["flow"]),
        "--lpl-branch",
        str(branches["full"]),
        "--sample-count",
        str(args.screen_sample_count),
        "--sampling-seed",
        str(args.sampling_seed),
        "--devices",
        args.devices,
        "--output-dir",
        str(curve_output),
        "--skip-existing",
    ]
    for command in (audit_command, curve_command):
        print(" ".join(command), flush=True)
        if not args.print_only:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Versioned launcher for existing, proven RAEv2 IG dynamics diagnostics.

No sampler or Euler update is implemented here.  The launcher computes the
three official IG-active phase ranges, then invokes existing repository tools:

* run_raev2_ig_parameterization_audit.py
* run_raev2_ig_direction_audit.py
* run_raev2_ig_replay_response.py (once for early/middle/late)

Every subdirectory and manifest is suffixed ``reuse_v1`` so stale outputs from
older custom diagnostics cannot be silently reused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for _path in (RAEV2_SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.run_raev2_distribution_auc import load_config  # noqa: E402
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    official_shifted_solver_grid,
)

PROTOCOL = "raev2_ig_dynamics_suite_reuse_v1"
COMPONENTS = ("parameterization", "direction", "replay")
PHASE_NAMES = ("early", "middle", "late")


@dataclass(frozen=True)
class Phase:
    name: str
    start_step: int
    end_step: int
    start_time: float
    end_time: float

    @property
    def active_steps(self) -> int:
        return self.end_step - self.start_step


def parse_components(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("at least one component is required")
    unknown = sorted(set(items) - set(COMPONENTS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown components: {unknown}")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("components must not contain duplicates")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--components", type=parse_components, default=COMPONENTS)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--parameterization-samples", type=int, default=64)
    parser.add_argument("--direction-samples", type=int, default=256)
    parser.add_argument("--replay-samples", type=int, default=32)
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--direction-batch-size", type=int, default=2)
    parser.add_argument(
        "--direction-times",
        default="0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.95",
    )
    parser.add_argument(
        "--direction-scales",
        default="1.0,1.25,1.50,1.78,2.0",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k"),
    )
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must be a non-empty unique comma-separated list")
    return devices


def prepare_fresh_output(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        elif any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to reuse non-empty output root: {output}. "
                "Use a new versioned name or pass --overwrite."
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_phases(
    grid: np.ndarray,
    interval: tuple[float, float],
) -> tuple[Phase, ...]:
    values = np.asarray(grid, dtype=np.float64)
    active = np.asarray(
        [
            step
            for step in range(len(values) - 1)
            if interval[0] <= float(values[step]) <= interval[1]
        ],
        dtype=np.int64,
    )
    if len(active) < 3:
        raise ValueError("official IG interval has fewer than three active steps")
    if not np.array_equal(
        active, np.arange(int(active[0]), int(active[-1]) + 1, dtype=np.int64)
    ):
        raise ValueError("official IG-active steps are not contiguous")
    chunks = tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(active, 3))
    phases = tuple(
        Phase(
            name=PHASE_NAMES[index],
            start_step=int(chunk[0]),
            end_step=int(chunk[-1]) + 1,
            start_time=float(values[int(chunk[0])]),
            end_time=float(values[int(chunk[-1]) + 1]),
        )
        for index, chunk in enumerate(chunks)
    )
    covered = np.concatenate(
        [np.arange(phase.start_step, phase.end_step, dtype=np.int64) for phase in phases]
    )
    if not np.array_equal(covered, active):
        raise AssertionError("phase ranges do not exactly cover official active steps")
    return phases


def torchrun_prefix(python: Path, device_count: int) -> list[str]:
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={device_count}",
    ]


def common_model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--state-key",
        args.state_key,
        "--precision",
        args.precision,
        "--dino-ckpt-dir",
        str(args.dino_ckpt_dir),
        "--dino-repo-dir",
        str(args.dino_repo_dir),
    ]


def build_commands(
    args: argparse.Namespace,
    *,
    output_root: Path,
    phases: tuple[Phase, ...],
    ig_scale: float,
    device_count: int,
) -> list[tuple[str, list[str], Path]]:
    prefix = torchrun_prefix(args.python, device_count)
    common = common_model_args(args)
    commands: list[tuple[str, list[str], Path]] = []

    if "parameterization" in args.components:
        output = output_root / "parameterization_reuse_v1"
        command = prefix + [
            str(args.repo / "experiments/run_raev2_ig_parameterization_audit.py"),
            "--output-dir",
            str(output),
            "--samples",
            str(args.parameterization_samples),
            "--per-rank-batch",
            str(args.per_rank_batch),
            "--ig-scale",
            str(ig_scale),
            "--seed",
            str(args.seed),
            *common,
        ]
        commands.append(("parameterization", command, output))

    if "direction" in args.components:
        output = output_root / "direction_reuse_v1"
        command = prefix + [
            str(args.repo / "experiments/run_raev2_ig_direction_audit.py"),
            "--output-dir",
            str(output),
            "--data-path",
            str(args.data_path),
            "--samples",
            str(args.direction_samples),
            "--batch-size",
            str(args.direction_batch_size),
            "--times",
            args.direction_times,
            "--scales",
            args.direction_scales,
            "--state-key",
            args.state_key,
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
            "--bootstrap-repeats",
            str(args.bootstrap_repeats),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--dino-ckpt-dir",
            str(args.dino_ckpt_dir),
            "--dino-repo-dir",
            str(args.dino_repo_dir),
        ]
        commands.append(("direction", command, output))

    if "replay" in args.components:
        for phase_index, phase in enumerate(phases):
            output = output_root / f"replay_{phase.name}_reuse_v1"
            command = prefix + [
                str(args.repo / "experiments/run_raev2_ig_replay_response.py"),
                "--output-dir",
                str(output),
                "--samples",
                str(args.replay_samples),
                "--per-rank-batch",
                str(args.per_rank_batch),
                "--start-step",
                str(phase.start_step),
                "--end-step",
                str(phase.end_step),
                "--gamma",
                str(ig_scale - 1.0),
                "--seed",
                str(args.seed + 1000 * phase_index),
                "--bootstrap-repeats",
                str(args.bootstrap_repeats),
                *common,
            ]
            commands.append((f"replay_{phase.name}", command, output))
    return commands


def validate_paths(args: argparse.Namespace, commands: Iterable[tuple[str, list[str], Path]]) -> None:
    required = (
        args.repo,
        args.python,
        args.config,
        args.checkpoint,
        args.dino_ckpt_dir,
        args.dino_repo_dir,
    )
    for path in required:
        if not path.expanduser().exists():
            raise FileNotFoundError(path)
    for _name, command, _output in commands:
        script = Path(command[5])
        if not script.is_file():
            raise FileNotFoundError(script)


def run_logged(
    *,
    name: str,
    command: list[str],
    output: Path,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "launcher_reuse_v1.log"
    printable = shlex.join(command)
    print(f"\n[{name}] {printable}", flush=True)
    if dry_run:
        return {
            "name": name,
            "command": command,
            "output": str(output),
            "status": "dry_run",
        }
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(printable + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return {
        "name": name,
        "command": command,
        "output": str(output),
        "log": str(log_path),
        "status": "complete",
    }


def main() -> None:
    args = parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dino_ckpt_dir = args.dino_ckpt_dir.expanduser().resolve()
    args.dino_repo_dir = args.dino_repo_dir.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    devices = selected_devices(args.devices)
    output_root = prepare_fresh_output(args.output_root, overwrite=args.overwrite)

    config = load_config(args.config)
    config.prepare_model_params()
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )
    phases = build_phases(grid.cpu().numpy(), interval)
    ig_scale = (
        float(args.ig_scale)
        if args.ig_scale is not None
        else float(config.guidance.ig.scale)
    )
    commands = build_commands(
        args,
        output_root=output_root,
        phases=phases,
        ig_scale=ig_scale,
        device_count=len(devices),
    )
    validate_paths(args, commands)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["PYTHONPATH"] = f"{args.repo}:{env.get('PYTHONPATH', '')}"
    results = []
    for name, command, output in commands:
        results.append(
            run_logged(
                name=name,
                command=command,
                output=output,
                cwd=args.repo,
                env=env,
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "protocol": PROTOCOL,
        "script_name": Path(__file__).name,
        "custom_sampler_implemented": False,
        "repo": str(args.repo),
        "python": str(args.python),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "state_key": args.state_key,
        "precision": args.precision,
        "devices": devices,
        "seed": args.seed,
        "ig_scale": ig_scale,
        "ig_gamma": ig_scale - 1.0,
        "official_ig_interval": interval,
        "phases": [asdict(phase) | {"active_steps": phase.active_steps} for phase in phases],
        "components": args.components,
        "subtasks": results,
    }
    (output_root / "suite_manifest_reuse_v1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDynamics suite reuse_v1 complete: {output_root}")


if __name__ == "__main__":
    main()
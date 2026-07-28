"""Launch the preregistered 2k->5k floor/static crossover on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
DATASET = Path("/data/shared/imagenet-1k")
BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
SCHEDULE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_path_crossover_train_v2"


@dataclass(frozen=True)
class CrossoverBranch:
    name: str
    early_path: str
    late_path: str
    source_branch: Path
    floor: float

    @property
    def source_checkpoint(self) -> Path:
        return self.source_branch / "checkpoints/step-0002000.pt"


BRANCHES = (
    CrossoverBranch(
        "floor_to_floor",
        "floor",
        "floor",
        SCHEDULE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000",
        0.20,
    ),
    CrossoverBranch(
        "floor_to_static",
        "floor",
        "static",
        SCHEDULE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000",
        0.0,
    ),
    CrossoverBranch(
        "static_to_static",
        "static",
        "static",
        BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000",
        0.0,
    ),
    CrossoverBranch(
        "static_to_floor",
        "static",
        "floor",
        BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000",
        0.20,
    ),
)


def branch_experiment_name(branch: CrossoverBranch, endpoint: int) -> str:
    return f"seed3407_{branch.name}_rank16_s2000_to_{int(endpoint)}"


def training_command(
    branch: CrossoverBranch, *, results: Path, endpoint: int
) -> tuple[list[str], str]:
    manifest = json.loads(
        (branch.source_branch / "manifest.json").read_text(encoding="utf-8")
    )
    name = branch_experiment_name(branch, endpoint)
    mode = "static" if branch.late_path == "static" else "annealed"
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        str(ROOT / "experiments/train_rae_layerwise_path.py"),
        "--config",
        str(CONFIG),
        "--data-path",
        str(DATASET),
        "--results-dir",
        str(results),
        "--experiment-name",
        name,
        "--ckpt",
        str(branch.source_checkpoint),
        "--subspaces",
        str(manifest["subspace_path"]),
        "--subspace-rank",
        "16",
        "--latent-cache",
        str(manifest["latent_cache"]),
        "--path-mode",
        mode,
        "--path-family",
        "power",
        "--path-power",
        "2.0",
        "--path-floor",
        str(branch.floor),
        "--detail-scale",
        "1.0",
        "--global-seed",
        "3407",
        "--max-train-steps",
        str(endpoint),
        "--fork-full-state",
    ]
    if branch.early_path == "static":
        command.append("--isolate-loader-rng")
    return command, name


def validate_sources() -> None:
    for branch in BRANCHES:
        if not branch.source_checkpoint.exists():
            raise FileNotFoundError(branch.source_checkpoint)
        manifest = branch.source_branch / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.endpoint <= 2000:
        raise ValueError("endpoint must be greater than the 2k fork step")
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if len(devices) != len(BRANCHES):
        raise ValueError("exactly four devices are required")
    validate_sources()
    results = args.results.expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    log_root = results / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    processes = []
    for device, branch in zip(devices, BRANCHES):
        command, name = training_command(
            branch, results=results, endpoint=args.endpoint
        )
        print(f"[{branch.name}] CUDA {device}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        environment["PYTHONUNBUFFERED"] = "1"
        handle = (log_root / f"{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((branch.name, process, handle))

    failures = []
    for name, process, handle in processes:
        return_code = process.wait()
        handle.close()
        print(f"[{name}] exit={return_code}", flush=True)
        if return_code:
            failures.append((name, return_code))
    if failures:
        raise RuntimeError(f"crossover training failures: {failures}")


if __name__ == "__main__":
    main()

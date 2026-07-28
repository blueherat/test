"""Launch four paired 2k well-conditioned RAE path candidates on four GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
DATASET = Path("/data/shared/imagenet-1k")
SOURCE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
SOURCE_BRANCH = SOURCE_ROOT / "seed3407_annealed_rank16_s0_to_10000"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    power: float
    floor: float
    alpha: float


CANDIDATES = (
    Candidate("floor005_p1", "power", 1.0, 0.05, 1.0),
    Candidate("floor015_rat05", "rational", 2.0, 0.15, 0.5),
    Candidate("floor030_p2", "power", 2.0, 0.30, 1.0),
    Candidate("floor020_p2", "power", 2.0, 0.20, 1.0),
)


def training_command(
    candidate: Candidate,
    *,
    results: Path,
    endpoint: int,
    seed: int,
) -> tuple[list[str], str]:
    manifest = __import__("json").loads(
        (SOURCE_BRANCH / "manifest.json").read_text(encoding="utf-8")
    )
    name = f"seed{seed}_{candidate.name}_rank16_s0_to_{endpoint}"
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
        str(manifest["source_checkpoint"]),
        "--subspaces",
        str(manifest["subspace_path"]),
        "--subspace-rank",
        "16",
        "--latent-cache",
        str(manifest["latent_cache"]),
        "--path-mode",
        "annealed",
        "--path-family",
        candidate.family,
        "--path-power",
        str(candidate.power),
        "--path-floor",
        str(candidate.floor),
        "--path-alpha",
        str(candidate.alpha),
        "--detail-scale",
        "1.0",
        "--global-seed",
        str(seed),
        "--max-train-steps",
        str(endpoint),
    ]
    return command, name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--endpoint", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if len(devices) < len(CANDIDATES):
        raise ValueError("four devices are required")
    args.results.expanduser().mkdir(parents=True, exist_ok=True)
    log_root = args.results.expanduser() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for device, candidate in zip(devices, CANDIDATES):
        command, name = training_command(
            candidate,
            results=args.results.expanduser(),
            endpoint=args.endpoint,
            seed=args.seed,
        )
        print(f"[{name}] CUDA {device}: {' '.join(command)}", flush=True)
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
        processes.append((name, process, handle))
    failures = []
    for name, process, handle in processes:
        return_code = process.wait()
        handle.close()
        print(f"[{name}] exit={return_code}", flush=True)
        if return_code:
            failures.append((name, return_code))
    if failures:
        raise RuntimeError(f"candidate training failures: {failures}")


if __name__ == "__main__":
    main()

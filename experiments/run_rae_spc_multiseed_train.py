"""Launch the preregistered paired five-seed SPC training study."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
DATASET = Path("/data/shared/imagenet-1k")
CACHE = Path.home() / "data/eqvae/cache/rae_layerwise_path_streams/seed3407_n160000_fp32"
SUBSPACES = Path.home() / "data/eqvae/experiments/rae_layerwise_path/gate1_imagenet_train1024_val256_mid9/subspaces.pt"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_SEEDS = (1201, 2309, 3413, 4517, 5623)


@dataclass(frozen=True)
class Job:
    seed: int
    condition: str

    def name(self, endpoint: int, switch_step: int) -> str:
        if self.condition == "static":
            return f"seed{self.seed}_static_s0_to{endpoint}"
        return (
            f"seed{self.seed}_spc_floor020_p2_rank16_"
            f"switch{switch_step}_s0_to{endpoint}"
        )


def training_command(
    job: Job,
    *,
    results: Path,
    endpoint: int,
    switch_step: int,
) -> list[str]:
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
        job.name(endpoint, switch_step),
        "--subspaces",
        str(SUBSPACES),
        "--subspace-rank",
        "16",
        "--latent-cache",
        str(CACHE),
        "--path-mode",
        "static" if job.condition == "static" else "annealed",
        "--path-family",
        "power",
        "--path-power",
        "2.0",
        "--path-floor",
        "0.0" if job.condition == "static" else "0.20",
        "--detail-scale",
        "1.0",
        "--global-seed",
        str(job.seed),
        "--cache-order-seed",
        str(job.seed),
        "--max-train-steps",
        str(endpoint),
        "--ema-reset-step",
        str(switch_step),
        "--save-steps",
        f"{switch_step},{endpoint}",
        "--isolate-loader-rng",
    ]
    if job.condition == "spc":
        command.extend(
            [
                "--path-switch-step",
                str(switch_step),
                "--path-mode-after-switch",
                "static",
            ]
        )
    return command


def jobs(seeds: tuple[int, ...]) -> list[Job]:
    return [Job(seed, condition) for seed in seeds for condition in ("static", "spc")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    if not seeds or not devices:
        raise ValueError("at least one seed and device are required")
    if not 0 < args.switch_step < args.endpoint:
        raise ValueError("switch step must lie inside the training interval")
    for path in (CONFIG, DATASET, CACHE, SUBSPACES):
        if not path.exists():
            raise FileNotFoundError(path)

    results = args.results.expanduser().resolve()
    log_root = results / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = jobs(seeds)
    if args.dry_run:
        for index, job in enumerate(pending):
            device = devices[index % len(devices)]
            print(
                f"[{job.seed}/{job.condition}] CUDA {device}: "
                + " ".join(
                    training_command(
                        job,
                        results=results,
                        endpoint=args.endpoint,
                        switch_step=args.switch_step,
                    )
                )
            )
        return

    active: dict[int, tuple[Job, subprocess.Popen, object]] = {}
    failures: list[tuple[Job, int]] = []
    while pending or active:
        for device in devices:
            if not pending or device in active:
                continue
            job = pending.pop(0)
            command = training_command(
                job,
                results=results,
                endpoint=args.endpoint,
                switch_step=args.switch_step,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            log_path = log_root / f"{job.name(args.endpoint, args.switch_step)}.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[device] = (job, process, handle)
            print(f"started seed={job.seed} condition={job.condition} cuda={device}", flush=True)
        if not active:
            break
        time.sleep(2)
        for device, (job, process, handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del active[device]
            print(
                f"finished seed={job.seed} condition={job.condition} "
                f"cuda={device} exit={return_code}",
                flush=True,
            )
            if return_code:
                failures.append((job, return_code))
    if failures:
        raise RuntimeError(
            "training failures: "
            + ", ".join(
                f"seed={job.seed}/{job.condition}:exit={code}"
                for job, code in failures
            )
        )


if __name__ == "__main__":
    main()

"""Run NFE audits for completed Imagenette latent-prior runs across GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "experiments/audit_imagenette_latent_prior_tradeoff.py"
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def run_sweep(
    root: Path,
    capacities: Sequence[int],
    seeds: Sequence[int],
    devices: Sequence[int],
    *,
    nfe: int,
    overwrite: bool,
) -> None:
    pending = deque((int(d), int(seed)) for seed in seeds for d in capacities)
    running: dict[int, tuple[subprocess.Popen, object, tuple[int, int]]] = {}
    log_root = root / "audit_logs"
    log_root.mkdir(exist_ok=True)
    failed = False
    while pending or running:
        for device in devices:
            if device in running or not pending or failed:
                continue
            latent_dim, seed = pending.popleft()
            run = root / f"d{latent_dim}_seed{seed}_p0"
            if (run / f"nfe{int(nfe)}_audit.json").is_file() and not overwrite:
                continue
            command = [
                sys.executable,
                str(AUDIT_SCRIPT),
                "--run",
                str(run),
                "--device",
                "cuda:0",
                "--nfe",
                str(nfe),
            ]
            if overwrite:
                command.append("--overwrite")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONPATH"] = str(ROOT)
            handle = (log_root / f"d{latent_dim}_seed{seed}_nfe{nfe}.log").open("w")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, handle, (latent_dim, seed))
            print(f"gpu {device}: audit d{latent_dim} seed{seed}", flush=True)
        time.sleep(2)
        for device, (process, handle, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[device]
            print(f"gpu {device}: audit {job} return={code}", flush=True)
            failed |= code != 0
        if failed:
            for process, _handle, _job in running.values():
                process.terminate()
            for process, handle, _job in running.values():
                process.wait()
                handle.close()
            raise RuntimeError("audit sweep failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capacities", type=parse_ints, default=(16, 64, 256))
    parser.add_argument("--seeds", type=parse_ints, default=(0, 1, 2, 3, 4))
    parser.add_argument("--devices", type=parse_ints, default=(0, 1, 2, 3))
    parser.add_argument("--nfe", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_sweep(
        args.root,
        args.capacities,
        args.seeds,
        args.devices,
        nfe=args.nfe,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

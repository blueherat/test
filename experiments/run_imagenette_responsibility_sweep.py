"""Run independent Imagenette responsibility configs across available GPUs."""

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
TRAIN_SCRIPT = ROOT / "experiments/imagenette_noise_responsibility.py"
DEFAULT_OUTPUT = Path.home() / "data/eqvae/imagenette_noise_responsibility_formal"


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def run_sweep(
    output_root: Path,
    capacities: Sequence[int],
    seeds: Sequence[int],
    devices: Sequence[int],
    *,
    overwrite: bool = False,
    extra_args: Sequence[str] = (),
) -> None:
    if not devices:
        raise ValueError("at least one GPU device is required")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(exist_ok=True)
    pending = deque((int(capacity), int(seed)) for seed in seeds for capacity in capacities)
    running: dict[int, tuple[subprocess.Popen, object, tuple[int, int]]] = {}
    failed = False
    try:
        while pending or running:
            for device in devices:
                if device in running or not pending or failed:
                    continue
                capacity, seed = pending.popleft()
                result_dir = output_root / f"d{capacity}_seed{seed}"
                if (result_dir / "summary.json").is_file() and not overwrite:
                    print(f"skip complete d{capacity} seed{seed}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--output-root",
                    str(output_root),
                    "--latent-dim",
                    str(capacity),
                    "--seed",
                    str(seed),
                    "--device",
                    "cuda:0",
                    *extra_args,
                ]
                if overwrite:
                    command.append("--overwrite")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                environment["PYTHONPATH"] = str(ROOT)
                log_handle = (log_root / f"d{capacity}_seed{seed}.log").open("w")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[device] = (process, log_handle, (capacity, seed))
                print(f"gpu {device}: start d{capacity} seed{seed} pid={process.pid}", flush=True)

            time.sleep(2.0)
            for device, (process, log_handle, job) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                del running[device]
                capacity, seed = job
                print(
                    f"gpu {device}: finish d{capacity} seed{seed} return={return_code}",
                    flush=True,
                )
                if return_code != 0:
                    failed = True
            if failed:
                for process, _log_handle, _job in running.values():
                    process.terminate()
                for process, log_handle, _job in running.values():
                    process.wait()
                    log_handle.close()
                raise RuntimeError("a sweep job failed; inspect output_root/logs")
    except KeyboardInterrupt:
        for process, _log_handle, _job in running.values():
            process.terminate()
        for process, log_handle, _job in running.values():
            process.wait()
            log_handle.close()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacities", type=parse_ints, default=(16, 64, 256))
    parser.add_argument("--seeds", type=parse_ints, default=(0, 1, 2))
    parser.add_argument("--devices", type=parse_ints, default=(0, 1, 2, 3))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extra_args = list(args.extra_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    run_sweep(
        args.output_root,
        args.capacities,
        args.seeds,
        args.devices,
        overwrite=args.overwrite,
        extra_args=extra_args,
    )


if __name__ == "__main__":
    main()

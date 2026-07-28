"""Run decoder-amplification audits across completed Imagenette prior runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/analyze_imagenette_decoder_amplification.py"
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def run(
    root: Path,
    devices: tuple[int, ...],
    *,
    count: int,
    capacities: tuple[int, ...],
    seeds: tuple[int, ...],
    overwrite: bool,
) -> None:
    pending = deque((capacity, seed) for seed in seeds for capacity in capacities)
    running = {}
    logs = root / "decoder_amplification_logs"
    logs.mkdir(exist_ok=True)
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            latent_dim, seed = pending.popleft()
            run_dir = root / f"d{latent_dim}_seed{seed}_p0"
            output = run_dir / "decoder_amplification_audit.json"
            if output.is_file() and not overwrite:
                print(f"already complete: d{latent_dim} seed{seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(SCRIPT),
                "--run",
                str(run_dir),
                "--device",
                "cuda:0",
                "--count",
                str(int(count)),
            ]
            if overwrite:
                command.append("--overwrite")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONPATH"] = str(ROOT)
            handle = (logs / f"d{latent_dim}_seed{seed}.log").open("w")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, handle, (latent_dim, seed))
            print(f"gpu {device}: decoder audit d{latent_dim} seed{seed}", flush=True)
        time.sleep(1)
        for device, (process, handle, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[device]
            print(f"gpu {device}: decoder audit {job} return={code}", flush=True)
            if code:
                for active, active_handle, _active_job in running.values():
                    active.terminate()
                    active.wait()
                    active_handle.close()
                log = logs / f"d{job[0]}_seed{job[1]}.log"
                tail = "\n".join(log.read_text().splitlines()[-60:])
                raise RuntimeError(f"decoder audit failed for {job}:\n{tail}")


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--capacities", default="16,64,256")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.root,
        _csv_ints(args.devices),
        count=args.count,
        capacities=_csv_ints(args.capacities),
        seeds=_csv_ints(args.seeds),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

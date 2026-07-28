"""Run the post-hoc semantic-gap audit across completed prior runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/analyze_imagenette_prior_semantic_gap.py"
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def run(root: Path, devices: tuple[int, ...], overwrite: bool) -> None:
    pending = deque((d, seed) for seed in range(5) for d in (16, 64, 256))
    running = {}
    logs = root / "semantic_audit_logs"
    logs.mkdir(exist_ok=True)
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            latent_dim, seed = pending.popleft()
            run_dir = root / f"d{latent_dim}_seed{seed}_p0"
            if (run_dir / "semantic_gap_audit.json").is_file() and not overwrite:
                continue
            command = [
                sys.executable,
                str(SCRIPT),
                "--run",
                str(run_dir),
                "--device",
                "cuda:0",
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
            print(f"gpu {device}: semantic audit d{latent_dim} seed{seed}", flush=True)
        time.sleep(1)
        for device, (process, handle, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[device]
            print(f"gpu {device}: semantic audit {job} return={code}", flush=True)
            if code:
                for active, active_handle, _active_job in running.values():
                    active.terminate()
                    active.wait()
                    active_handle.close()
                raise RuntimeError("semantic audit failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    run(args.root, devices, args.overwrite)


if __name__ == "__main__":
    main()

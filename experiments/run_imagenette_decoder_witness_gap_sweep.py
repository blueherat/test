"""Run the post-hoc decoder-witness audit on all completed prior runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/analyze_imagenette_decoder_witness_gap.py"
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def run(root: Path, devices: tuple[int, ...], overwrite: bool) -> None:
    pending = deque((capacity, seed) for seed in range(5) for capacity in (16, 64, 256))
    running = {}
    logs = root / "decoder_witness_logs"
    logs.mkdir(exist_ok=True)
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            latent_dim, seed = pending.popleft()
            run_dir = root / f"d{latent_dim}_seed{seed}_p0"
            if (run_dir / "decoder_witness_gap_posthoc.json").is_file() and not overwrite:
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
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[variable] = "4"
            handle = (logs / f"d{latent_dim}_seed{seed}.log").open("w")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, handle, (latent_dim, seed))
            print(f"gpu {device}: witness audit d{latent_dim} seed{seed}", flush=True)
        time.sleep(1)
        for device, (process, handle, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[device]
            print(f"gpu {device}: witness audit {job} return={code}", flush=True)
            if code:
                for active, active_handle, _ in running.values():
                    active.terminate()
                    active.wait()
                    active_handle.close()
                log = logs / f"d{job[0]}_seed{job[1]}.log"
                tail = "\n".join(log.read_text().splitlines()[-80:])
                raise RuntimeError(f"witness audit failed for {job}:\n{tail}")


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

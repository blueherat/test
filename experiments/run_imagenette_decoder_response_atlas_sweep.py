"""Run all frozen-decoder response atlases across four GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/imagenette_decoder_response_atlas.py"
SUMMARY = ROOT / "experiments/summarize_imagenette_decoder_response_atlas.py"
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def run_sweep(
    root: Path,
    devices: tuple[int, ...],
    capacities: tuple[int, ...],
    seeds: tuple[int, ...],
    *,
    count: int,
    paired_count: int,
    pixel_steps: int,
    batch_size: int,
    projection_dim: int,
    projection_seed: int,
    summary_name: str,
    overwrite: bool,
) -> None:
    pending = deque((latent_dim, seed) for seed in seeds for latent_dim in capacities)
    active = {}
    logs = root / "decoder_response_atlas_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            latent_dim, seed = pending.popleft()
            run = root / f"d{latent_dim}_seed{seed}_p0"
            output = run / "decoder_response_atlas.json"
            tensor = run / "decoder_response_atlas.pt"
            if output.is_file() and tensor.is_file() and not overwrite:
                print(f"already complete: d{latent_dim} seed{seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(SCRIPT),
                "--run",
                str(run),
                "--device",
                "cuda:0",
                "--count",
                str(int(count)),
                "--paired-count",
                str(int(paired_count)),
                "--pixel-steps",
                str(int(pixel_steps)),
                "--batch-size",
                str(int(batch_size)),
                "--projection-dim",
                str(int(projection_dim)),
                "--projection-seed",
                str(int(projection_seed)),
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
            active[device] = (process, handle, (latent_dim, seed))
            print(f"gpu {device}: response atlas d{latent_dim} seed{seed}", flush=True)
        time.sleep(1)
        for device, (process, handle, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[device]
            print(f"gpu {device}: response atlas {job} return={code}", flush=True)
            if code:
                for other, other_handle, _other_job in active.values():
                    other.terminate()
                    other.wait()
                    other_handle.close()
                log = logs / f"d{job[0]}_seed{job[1]}.log"
                tail = "\n".join(log.read_text().splitlines()[-80:])
                raise RuntimeError(f"response atlas failed for {job}:\n{tail}")
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--root",
            str(root),
            "--expected-count",
            str(int(count)),
            "--output-name",
            str(summary_name),
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--capacities", default="16,64,256")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--paired-count", type=int, default=128)
    parser.add_argument("--pixel-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=48_271)
    parser.add_argument("--summary-name", default="decoder_response_atlas_summary")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_sweep(
        args.root,
        _csv_ints(args.devices),
        _csv_ints(args.capacities),
        _csv_ints(args.seeds),
        count=args.count,
        paired_count=args.paired_count,
        pixel_steps=args.pixel_steps,
        batch_size=args.batch_size,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        summary_name=args.summary_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

"""Launch full semantic and conditional-detail branches for dual-stream RAE."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
DATASET = Path("/data/shared/imagenet-1k")
CACHE = Path.home() / "data/eqvae/cache/rae_layerwise_path_streams/seed3407_n160000_fp32"
SUBSPACES = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path/"
    "gate1_imagenet_train1024_val256_mid9/subspaces.pt"
)
SOURCE = (
    Path.home()
    / "data/eqvae/stage2_training/fair_ditdh_s_dinov2_original_gbs1024_ep80_4gpu/"
    "checkpoints/ep-0000000.pt"
)
GATE_RESULTS = Path.home() / "data/eqvae/experiments/rae_dual_stream_gate"
RESULTS = Path.home() / "data/eqvae/experiments/rae_dual_stream_full"


def semantic_name(seed: int) -> str:
    return f"seed{int(seed)}_semantic_rank16_s0_to_10000"


def detail_name(seed: int, mode: str) -> str:
    return f"seed{int(seed)}_{mode}_detail_rank16_steps4500"


def semantic_command(seed: int, results: Path) -> list[str]:
    return [
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
        semantic_name(seed),
        "--ckpt",
        str(SOURCE),
        "--subspaces",
        str(SUBSPACES),
        "--subspace-rank",
        "16",
        "--latent-cache",
        str(CACHE),
        "--path-mode",
        "static",
        "--path-power",
        "2",
        "--clean-component",
        "semantic",
        "--global-seed",
        str(seed),
        "--max-train-steps",
        "10000",
    ]


def detail_command(seed: int, mode: str, results: Path) -> list[str]:
    init = GATE_RESULTS / f"seed{seed}_{mode}_steps2000/checkpoint.pt"
    return [
        sys.executable,
        str(ROOT / "experiments/train_rae_dual_stream_gate.py"),
        "--config",
        str(CONFIG),
        "--latent-cache",
        str(CACHE),
        "--subspaces",
        str(SUBSPACES),
        "--results",
        str(results),
        "--experiment-name",
        detail_name(seed, mode),
        "--context-mode",
        mode,
        "--seed",
        str(seed),
        "--steps",
        "4500",
        "--batch-size",
        "32",
        "--val-count",
        "4096",
        "--init-checkpoint",
        str(init),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = tuple(int(item) for item in args.devices.split(",") if item.strip())
    jobs = [
        (semantic_name(3407), semantic_command(3407, args.results)),
        (detail_name(3407, "paired"), detail_command(3407, "paired", args.results)),
        (semantic_name(4211), semantic_command(4211, args.results)),
        (detail_name(3407, "shuffled"), detail_command(3407, "shuffled", args.results)),
    ]
    if len(devices) < len(jobs):
        raise ValueError("full dual-stream screen requires four devices")
    required = [CONFIG, DATASET / "data/train-00000-of-00294.parquet", CACHE / "manifest.json", SUBSPACES, SOURCE]
    required.extend(
        GATE_RESULTS / f"seed3407_{mode}_steps2000/checkpoint.pt" for mode in ("paired", "shuffled")
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    args.results.mkdir(parents=True, exist_ok=True)
    protocol = {
        "semantic_seeds": [3407, 4211],
        "semantic_steps": 10000,
        "semantic_global_batch": 16,
        "detail_seed": 3407,
        "detail_modes": ["paired", "shuffled"],
        "detail_steps": 4500,
        "detail_global_batch": 32,
        "precision": "fp32",
        "endpoint": "semantic + basis @ detail_coefficients equals original final RAE latent",
        "primary_gate": "seed3407 paired dual-stream FID improves static by at least 5% and beats semantic-only and shuffled-detail controls",
        "stop_rule": "failure ends the dual-stream direction",
    }
    protocol_path = args.results / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("refusing to change registered full dual-stream protocol")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)
    log_root = args.results / "logs"
    log_root.mkdir(exist_ok=True)
    processes = []
    for device, (name, command) in zip(devices, jobs):
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
        code = process.wait()
        handle.close()
        print(f"[{name}] exit={code}", flush=True)
        if code:
            failures.append(name)
    if failures:
        raise RuntimeError(f"full dual-stream branches failed: {failures}")


if __name__ == "__main__":
    main()

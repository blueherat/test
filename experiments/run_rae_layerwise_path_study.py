"""Launch paired static/annealed/reverse/random layerwise-path branches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rae_latent_cache import cache_directory, load_cache_manifest


CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
DATASET = Path("/data/shared/imagenet-1k")
START_CHECKPOINT = (
    Path.home()
    / "data/eqvae/stage2_training/fair_ditdh_s_dinov2_original_gbs1024_ep80_4gpu/"
    "checkpoints/ep-0000000.pt"
)
SUBSPACES = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path/"
    "gate1_imagenet_train1024_val256_mid9/subspaces.pt"
)
RESULTS = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CACHE_ROOT = Path.home() / "data/eqvae/cache/rae_layerwise_path_streams"
SEEDS = (3407, 4211, 5821)
CONDITIONS = (
    ("static", "static", False),
    ("annealed", "annealed", False),
    ("reverse", "reverse", False),
    ("random", "annealed", True),
)


def verify_inputs(
    checkpoint: Path,
    subspaces: Path,
    rank: int,
) -> dict[str, object]:
    required = (
        CONFIG,
        DATASET / "data/train-00000-of-00294.parquet",
        checkpoint,
        subspaces,
        ROOT / "external/RAE/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    entries = torch.load(subspaces, map_location="cpu", weights_only=False)["subspaces"]
    if int(rank) not in entries and str(rank) not in entries:
        raise KeyError(f"rank {rank} is absent from {subspaces}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(state["step"]),
        "subspaces": str(subspaces),
        "subspace_rank": int(rank),
    }


def branch_name(seed: int, condition: str, endpoint: int, tag: str = "") -> str:
    name = f"seed{int(seed)}_{condition}_rank16_s0_to_{int(endpoint)}"
    return f"{name}_{tag}" if tag else name


def training_command(
    *,
    seed: int,
    condition: str,
    path_mode: str,
    random_subspace: bool,
    endpoint: int,
    checkpoint: Path,
    subspaces: Path,
    latent_cache: Path,
    results: Path,
    rank: int,
    power: float,
    detail_scale: float,
    tag: str = "",
) -> tuple[list[str], str]:
    name = branch_name(seed, condition, endpoint, tag)
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
        str(checkpoint),
        "--subspaces",
        str(subspaces),
        "--subspace-rank",
        str(int(rank)),
        "--latent-cache",
        str(latent_cache),
        "--path-mode",
        path_mode,
        "--path-power",
        str(float(power)),
        "--detail-scale",
        str(float(detail_scale)),
        "--global-seed",
        str(int(seed)),
        "--max-train-steps",
        str(int(endpoint)),
    ]
    if random_subspace:
        command.append("--random-subspace")
    return command, name


def write_protocol(
    results: Path,
    metadata: dict[str, object],
    *,
    endpoint: int,
    power: float,
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "preregistered_tiny_imagenet_gate",
        "question": "Does a semantic-first middle-guided layerwise path improve generation over the identical static endpoint?",
        "inputs": metadata,
        "seeds": list(SEEDS),
        "conditions": {
            name: {
                "path_mode": mode,
                "random_subspace": random,
                "detail_scale": metadata["random_detail_scale"] if random else 1.0,
            }
            for name, mode, random in CONDITIONS
        },
        "path_power": float(power),
        "endpoint_step": int(endpoint),
        "sample_count": 5000,
        "primary_metrics": ["paired_kid_5000", "fid_5000_proxy"],
        "gate": {
            "annealed_mean_improvement": ">=5% versus static",
            "seed_direction": "3/3",
            "mechanism_control": "reverse worse than annealed",
        "subspace_control": "energy-matched random does not match middle-guided annealed",
        },
        "pairing": "same endpoint, decoder, architecture, checkpoint, data stream, time, noise, optimizer, fp32 and sampling seed",
        "latent_stream": "one fixed fp32 RAE cache per seed, shared by all four conditions",
        "stop_rule": "Gate failure stops direction one; teacher loss cannot override generation metrics.",
    }
    path = results / "layerwise_path_protocol_v3_energy_matched.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(f"refusing to overwrite changed protocol: {path}")
    else:
        path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_seed(
    *,
    seed: int,
    endpoint: int,
    checkpoint: Path,
    subspaces: Path,
    results: Path,
    rank: int,
    power: float,
    random_detail_scale: float,
    devices: tuple[int, ...],
    latent_cache: Path,
    tag: str,
    dry_run: bool,
) -> None:
    if len(devices) < len(CONDITIONS):
        raise ValueError("one device per condition is required for the paired parallel screen")
    processes: list[tuple[str, subprocess.Popen, object]] = []
    log_root = results / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for device, (condition, path_mode, random_subspace) in zip(devices, CONDITIONS):
        command, name = training_command(
            seed=seed,
            condition=condition,
            path_mode=path_mode,
            random_subspace=random_subspace,
            endpoint=endpoint,
            checkpoint=checkpoint,
            subspaces=subspaces,
            latent_cache=latent_cache,
            results=results,
            rank=rank,
            power=power,
            detail_scale=random_detail_scale if random_subspace else 1.0,
            tag=tag,
        )
        print(f"[{name}] CUDA {device}: {' '.join(command)}", flush=True)
        if dry_run:
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(int(device))
        environment["PYTHONUNBUFFERED"] = "1"
        log_path = log_root / f"{name}.log"
        handle = log_path.open("a", encoding="utf-8")
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
        if return_code != 0:
            failures.append(name)
    if failures:
        raise RuntimeError(f"training branches failed: {failures}")


def ensure_latent_cache(
    *,
    seed: int,
    sample_count: int,
    cache_root: Path,
    devices: tuple[int, ...],
    dry_run: bool,
) -> Path:
    path = cache_directory(cache_root, seed, sample_count)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = load_cache_manifest(path)
        if int(manifest["seed"]) != int(seed) or int(manifest["sample_count"]) != sample_count:
            raise ValueError(f"cache metadata mismatch: {manifest_path}")
        return path
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        str(ROOT / "experiments/cache_rae_final_latents.py"),
        "--config",
        str(CONFIG),
        "--data-path",
        str(DATASET),
        "--cache-root",
        str(cache_root),
        "--seed",
        str(int(seed)),
        "--sample-count",
        str(int(sample_count)),
    ]
    print(f"[seed{seed} cache] CUDA {devices}: {' '.join(command)}", flush=True)
    if dry_run:
        return path
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in devices)
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    load_cache_manifest(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "one-seed", "screen"), default="preflight")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--endpoint", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path, default=START_CHECKPOINT)
    parser.add_argument("--subspaces", type=Path, default=SUBSPACES)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--subspace-rank", type=int, default=16)
    parser.add_argument("--path-power", type=float, default=2.0)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    metadata = verify_inputs(args.checkpoint, args.subspaces, args.subspace_rank)
    subspace_payload = torch.load(args.subspaces, map_location="cpu", weights_only=False)
    entry = subspace_payload["subspaces"].get(
        int(args.subspace_rank), subspace_payload["subspaces"].get(str(args.subspace_rank))
    )
    random_detail_scale = (
        float(entry["explained_final_fraction"])
        / (float(args.subspace_rank) / float(entry["basis"].shape[0]))
    ) ** 0.5
    metadata["random_detail_scale"] = random_detail_scale
    write_protocol(
        args.results,
        metadata,
        endpoint=args.endpoint,
        power=args.path_power,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    if args.mode == "preflight":
        return
    config = OmegaConf.load(CONFIG)
    global_batch = int(config.training.global_batch_size)
    source_step = int(metadata["checkpoint_step"])
    updates = int(args.endpoint) - source_step
    if updates <= 0:
        raise ValueError("endpoint must be after the source checkpoint")
    sample_count = updates * global_batch
    seeds = (args.seed,) if args.mode == "one-seed" else SEEDS
    for seed in seeds:
        latent_cache = ensure_latent_cache(
            seed=seed,
            sample_count=sample_count,
            cache_root=args.cache_root,
            devices=devices,
            dry_run=args.dry_run,
        )
        run_seed(
            seed=seed,
            endpoint=args.endpoint,
            checkpoint=args.checkpoint,
            subspaces=args.subspaces,
            results=args.results,
            rank=args.subspace_rank,
            power=args.path_power,
            random_detail_scale=random_detail_scale,
            devices=devices,
            latent_cache=latent_cache,
            tag=args.tag,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

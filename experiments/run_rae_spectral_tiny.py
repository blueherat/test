"""Launch the preregistered paired tiny RAE spectral-loss experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
CONFIG = ROOT / "experiments" / "configs" / "rae_spectral_tiny_ditdh_s_dinov2.yaml"
START_CHECKPOINT = (
    Path.home()
    / "data/eqvae/artifacts/rae_stage2_training/"
    "ditdh_s_dinov2_imagenet256_parquet_2gpu_s20000/checkpoints/step-0005000.pt"
)
DATASET = Path("/data/shared/imagenet-1k")
RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
PAIRED_SEEDS = (3407, 4211, 5821)
TREATMENTS = (("baseline", 0.0), ("partial", 0.5))


def verify_inputs(checkpoint: Path = START_CHECKPOINT) -> dict[str, object]:
    required = [
        CONFIG,
        checkpoint,
        DATASET / "data" / "train-00000-of-00294.parquet",
        RAE_ROOT / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        RAE_ROOT / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing tiny-experiment inputs:\n" + "\n".join(missing))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_keys = {"model", "ema", "optimizer", "scheduler", "step", "epoch"}
    absent = checkpoint_keys.difference(state)
    if absent:
        raise KeyError(f"checkpoint is not full-state; missing {sorted(absent)}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(state["step"]),
        "checkpoint_epoch": int(state["epoch"]),
        "has_optimizer": bool(state["optimizer"]),
        "has_scheduler": state["scheduler"] is not None,
    }


def branch_name(seed: int, treatment: str, start_step: int, end_step: int, tag: str) -> str:
    if tag:
        return f"seed{int(seed)}_{treatment}_s{int(start_step)}_to_{int(end_step)}_{tag}"
    return f"seed{int(seed)}_{treatment}_from_s{int(start_step)}"


def training_command(
    *,
    seed: int,
    treatment: str,
    gamma: float,
    end_step: int,
    device: int,
    checkpoint: Path,
    results: Path,
    tag: str = "",
) -> tuple[list[str], dict[str, str], str]:
    metadata = verify_inputs(checkpoint)
    name = branch_name(seed, treatment, int(metadata["checkpoint_step"]), end_step, tag)
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        str(ROOT / "experiments/train_rae_spectral_tiny.py"),
        "--config",
        str(CONFIG),
        "--data-path",
        str(DATASET),
        "--results-dir",
        str(results),
        "--experiment-name",
        name,
        "--image-size",
        "256",
        "--ckpt",
        str(checkpoint),
        "--global-seed",
        str(int(seed)),
        "--spectral-gamma",
        str(float(gamma)),
        "--max-train-steps",
        str(int(end_step)),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(int(device))
    environment["PYTHONUNBUFFERED"] = "1"
    return command, environment, name


def write_protocol(results: Path, checkpoint_metadata: dict[str, object]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "preregistered_screen",
        "question": "Does direction-only inverse-standard-deviation weighting improve fixed-budget generation?",
        "start": checkpoint_metadata,
        "paired_seeds": list(PAIRED_SEEDS),
        "treatments": {name: gamma for name, gamma in TREATMENTS},
        "primary_endpoint_update": 10000,
        "curve_updates": [5500, 6000, 7000, 10000],
        "primary_screen_metrics": ["paired_kid_5000", "fid_5000_proxy"],
        "secondary_metrics": ["decoder_sensitive_band_error", "raw_velocity_mse"],
        "continue_rule": "At least 2/3 paired seeds improve KID by >=5% without coverage collapse.",
        "confirmation_rule": "Then add two paired seeds; require >=4/5 before 50k FID.",
        "fixed_loss": {
            "bands": "8 equal-cardinality radial DCT bands",
            "normalization": "coefficient-weighted mean one for each sample/time",
            "damping": 1e-4,
            "weight_bounds": [0.2, 2.0],
            "online_updates": False,
        },
        "pairing": "same checkpoint, optimizer/scheduler, data order, augmentation, t, noise, labels and dropout within each seed",
        "caveat": "The checkpoint has no RNG/dataloader cursor; this is a fresh paired branch stream, not an exact continuation of the original trajectory.",
    }
    path = results / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(f"refusing to overwrite changed preregistration: {path}")
    else:
        path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_branch(**kwargs) -> None:
    command, environment, name = training_command(**kwargs)
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=RAE_ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "smoke", "single", "screen"], default="preflight")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--seed", type=int, default=PAIRED_SEEDS[0])
    parser.add_argument("--treatment", choices=[name for name, _ in TREATMENTS], default="baseline")
    parser.add_argument("--end-step", type=int, default=10000)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--checkpoint", type=Path, default=START_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_metadata = verify_inputs(args.checkpoint)
    write_protocol(args.results, checkpoint_metadata)
    print(json.dumps(checkpoint_metadata, indent=2), flush=True)
    if args.mode == "preflight":
        return

    gamma_by_name = dict(TREATMENTS)
    if args.mode == "smoke":
        jobs = [(args.seed, "baseline", 0.0), (args.seed, "partial", 0.5)]
        end_step = int(checkpoint_metadata["checkpoint_step"]) + 2
        tag = "smoke_v3"
    elif args.mode == "single":
        jobs = [(args.seed, args.treatment, gamma_by_name[args.treatment])]
        end_step = args.end_step
        tag = ""
    else:
        jobs = []
        for index, seed in enumerate(PAIRED_SEEDS):
            ordered = TREATMENTS if index % 2 == 0 else tuple(reversed(TREATMENTS))
            jobs.extend((seed, treatment, gamma) for treatment, gamma in ordered)
        end_step = args.end_step
        tag = ""

    for seed, treatment, gamma in jobs:
        kwargs = dict(
            seed=seed,
            treatment=treatment,
            gamma=gamma,
            end_step=end_step,
            device=args.device,
            checkpoint=args.checkpoint,
            results=args.results,
            tag=tag,
        )
        if args.dry_run:
            command, _, name = training_command(**kwargs)
            print(name, "\n ", " ".join(command))
        else:
            run_branch(**kwargs)


if __name__ == "__main__":
    main()

"""Train the low-cost semantic-conditioning gate for RAE detail flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_dual_stream import (  # noqa: E402
    SemanticConditionedDetailDDT,
    split_semantic_coefficients,
)
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402
from stage2.transport import ModelType, create_transport  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--subspaces", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--context-mode", choices=("paired", "shuffled"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--init-checkpoint", type=Path)
    return parser.parse_args()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def context_for_mode(semantic: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "paired":
        return semantic
    if mode == "shuffled":
        if len(semantic) < 2:
            raise ValueError("shuffled context requires batch size at least two")
        return semantic.roll(shifts=1, dims=0)
    raise ValueError(mode)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@torch.inference_mode()
def evaluate(
    model: SemanticConditionedDetailDDT,
    loader: DataLoader,
    basis: torch.Tensor,
    transport,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    torch.manual_seed(int(seed) + 900_001)
    torch.cuda.manual_seed(int(seed) + 900_001)
    totals = {
        "paired_loss": 0.0,
        "shuffled_loss": 0.0,
        "target_energy": 0.0,
        "samples": 0.0,
    }
    for latent, labels in loader:
        latent = latent.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        semantic, coefficients = split_semantic_coefficients(latent, basis)
        time, noise, clean = transport.sample(coefficients)
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * clean + expanded * noise
        target = noise - clean
        paired = model(state, time, labels, semantic)
        shuffled = model(state, time, labels, semantic.roll(shifts=1, dims=0))
        count = len(latent)
        totals["paired_loss"] += float(F.mse_loss(paired, target, reduction="sum"))
        totals["shuffled_loss"] += float(F.mse_loss(shuffled, target, reduction="sum"))
        totals["target_energy"] += float(target.square().sum())
        totals["samples"] += count
    target_energy = totals["target_energy"]
    return {
        "paired_context_normalized_mse": totals["paired_loss"] / target_energy,
        "shuffled_context_normalized_mse": totals["shuffled_loss"] / target_energy,
        "context_usage_gain": 1.0 - totals["paired_loss"] / totals["shuffled_loss"],
        "target_energy_sum": target_energy,
        "sample_count": int(totals["samples"]),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    configure_determinism(int(args.seed))
    output = args.results.expanduser() / args.experiment_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    result_path = output / "result.json"
    if result_path.exists() and checkpoint_path.exists():
        completed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(completed["step"]) >= int(args.steps):
            return

    payload = torch.load(args.subspaces, map_location="cpu", weights_only=False)
    entry = payload["subspaces"].get(int(args.rank), payload["subspaces"].get(str(args.rank)))
    if entry is None:
        raise KeyError(f"rank {args.rank} is absent from {args.subspaces}")
    basis = entry["basis"].to(device=device, dtype=torch.float32)
    if basis.shape != (768, int(args.rank)):
        raise ValueError(f"unexpected basis shape: {tuple(basis.shape)}")

    config = OmegaConf.load(args.config)
    transport_params = OmegaConf.to_container(config.transport.params, resolve=True)
    detail_dimension = int(args.rank) * 16 * 16
    time_shift = math.sqrt(
        float(detail_dimension) / float(config.misc.time_dist_shift_base)
    )
    transport = create_transport(**dict(transport_params), time_dist_shift=time_shift)
    if transport.model_type != ModelType.VELOCITY:
        raise ValueError("dual-stream gate requires velocity flow matching")

    model = SemanticConditionedDetailDDT(detail_channels=int(args.rank)).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    step = 0
    load_path = checkpoint_path if checkpoint_path.exists() else args.init_checkpoint
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], device=device)

    train_count = int(args.steps) * int(args.batch_size)
    consumed = step * int(args.batch_size)
    if consumed > train_count:
        raise ValueError("checkpoint step exceeds requested endpoint")
    train_dataset = CachedRAELatentDataset(
        args.latent_cache, start=consumed, stop=train_count
    )
    val_dataset = CachedRAELatentDataset(
        args.latent_cache, start=train_count, stop=train_count + int(args.val_count)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )
    manifest = {
        "experiment_name": args.experiment_name,
        "context_mode": args.context_mode,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_count": train_count,
        "val_count": int(args.val_count),
        "precision": "fp32",
        "tf32": False,
        "rank": int(args.rank),
        "detail_dimension": detail_dimension,
        "time_dist_shift": time_shift,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "question": "Does paired clean semantic context predict held-out detail flow better than shuffled context?",
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
    }
    atomic_json(output / "manifest.json", manifest)

    model.train()
    started = perf_counter()
    metrics_path = output / "metrics.jsonl"
    for latent, labels in train_loader:
        latent = latent.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with torch.no_grad():
            semantic, coefficients = split_semantic_coefficients(latent, basis)
            time, noise, clean = transport.sample(coefficients)
            expanded = time[:, None, None, None]
            state = (1.0 - expanded) * clean + expanded * noise
            target = noise - clean
            context = context_for_mode(semantic, args.context_mode)
        prediction = model(state, time, labels, context)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0:
            atomic_json(
                output / "pair_fingerprint.json",
                {
                    "latent": tensor_hash(latent),
                    "labels": tensor_hash(labels),
                    "time": tensor_hash(time),
                    "noise": tensor_hash(noise),
                    "clean_coefficients": tensor_hash(clean),
                    "semantic": tensor_hash(semantic),
                    "context": tensor_hash(context),
                    "target": tensor_hash(target),
                    "initial_prediction": tensor_hash(prediction),
                },
            )
        step += 1
        if step % 100 == 0 or step == int(args.steps):
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "target_energy": float(target.square().mean()),
                "normalized_loss": float(loss.detach() / target.square().mean()),
                "grad_norm": float(gradient),
                "elapsed_seconds": perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % 500 == 0 or step == int(args.steps):
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device=device),
                },
                checkpoint_path,
            )
        if step >= int(args.steps):
            break
    if step != int(args.steps):
        raise RuntimeError(f"training ended at step {step}, expected {args.steps}")
    validation = evaluate(model, val_loader, basis, transport, device, int(args.seed))
    result = {**manifest, **validation, "elapsed_seconds": perf_counter() - started}
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

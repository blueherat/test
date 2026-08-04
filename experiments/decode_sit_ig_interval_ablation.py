"""Decode completed SiT interval endpoints into ADM-compatible image NPZs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    _atomic_json,
    _load_condition,
    _open_memmap,
)


PROTOCOL = "sit_ig_interval_decode_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--condition-names", default="")
    parser.add_argument("--log-every", type=int, default=128)
    return parser.parse_args()


def parse_condition_names(value: str, available: list[str]) -> list[str]:
    if not value.strip():
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown conditions: {missing}")
    if len(set(requested)) != len(requested):
        raise ValueError("condition names must be unique")
    return requested


def decode_latents(vae: AutoencoderKL, latent: torch.Tensor) -> np.ndarray:
    scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215],
        device=latent.device,
        dtype=torch.float32,
    ).view(1, 4, 1, 1)
    with torch.inference_mode():
        image = vae.decode(latent.float() / scale).sample
    return (
        image.add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .permute(0, 2, 3, 1)
        .to(device="cpu", dtype=torch.uint8)
        .numpy()
    )


def merge_shards(
    *,
    output_dir: Path,
    condition_name: str,
    samples: int,
    world_size: int,
) -> Path:
    merged_path = output_dir / f".{condition_name}_merged.npy"
    merged = np.lib.format.open_memmap(
        merged_path,
        mode="w+",
        dtype=np.uint8,
        shape=(samples, 256, 256, 3),
    )
    for rank in range(world_size):
        ids = np.arange(rank, samples, world_size, dtype=np.int64)
        shard = np.load(
            output_dir / f".{condition_name}_rank{rank:02d}.npy",
            mmap_mode="r",
        )
        if len(shard) != len(ids):
            raise RuntimeError("decoded shard length mismatch")
        merged[ids] = shard
    merged.flush()
    final_path = output_dir / f"{condition_name}.npz"
    temporary = output_dir / f".{condition_name}.npz.tmp-{os.getpid()}"
    with temporary.open("wb") as handle:
        np.savez(handle, np.load(merged_path, mmap_mode="r"))
    os.replace(temporary, final_path)
    del merged
    merged_path.unlink()
    for rank in range(world_size):
        (output_dir / f".{condition_name}_rank{rank:02d}.npy").unlink()
        (output_dir / f".{condition_name}_progress_rank{rank:02d}.npy").unlink()
    return final_path


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("batch size and log interval must be positive")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_dir = args.run_dir.expanduser().resolve()
    source_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "complete":
        raise RuntimeError("interval endpoint run is incomplete")
    if source_manifest.get("protocol") != "sit_ig_interval_ablation_v1":
        raise RuntimeError("unexpected source protocol")
    samples = int(source_manifest["samples"])
    source_world_size = int(source_manifest["world_size"])
    condition_rows = source_manifest["conditions"]
    available = [str(row["name"]) for row in condition_rows]
    selected = parse_condition_names(args.condition_names, available)
    by_name = {name: index for index, name in enumerate(available)}
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "source_run": str(run_dir),
        "source_checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "samples": samples,
        "world_size": world_size,
        "source_world_size": source_world_size,
        "vae": args.vae,
        "precision": "fp32",
        "conditions": selected,
    }
    if rank == 0:
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = (
                "protocol",
                "source_run",
                "source_checkpoint_sha256",
                "samples",
                "world_size",
                "source_world_size",
                "vae",
                "conditions",
            )
            changed = [key for key in keys if current.get(key) != manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed decode protocol: {changed}")
        else:
            _atomic_json(manifest_path, manifest)
    dist.barrier()
    vae = AutoencoderKL.from_pretrained(
        args.vae,
        local_files_only=True,
    ).to(device=device, dtype=torch.float32)
    vae.eval()
    local_ids = np.arange(rank, samples, world_size, dtype=np.int64)
    for condition_name in selected:
        final_path = output_dir / f"{condition_name}.npz"
        final_exists = [final_path.is_file() if rank == 0 else False]
        dist.broadcast_object_list(final_exists, src=0)
        if final_exists[0]:
            continue
        endpoint = _load_condition(
            run_dir,
            condition_index=by_name[condition_name],
            samples=samples,
            world_size=source_world_size,
        )
        local_endpoint = endpoint[local_ids]
        shard_path = output_dir / f".{condition_name}_rank{rank:02d}.npy"
        shard = _open_memmap(
            shard_path,
            shape=(len(local_ids), 256, 256, 3),
            dtype=np.uint8,
        )
        progress_path = output_dir / f".{condition_name}_progress_rank{rank:02d}.npy"
        progress_existed = progress_path.is_file()
        progress = _open_memmap(
            progress_path,
            shape=(len(local_ids),),
            dtype=np.bool_,
        )
        if not progress_existed:
            progress.fill(False)
            progress.flush()
        for start in range(0, len(local_ids), args.batch_size):
            stop = min(start + args.batch_size, len(local_ids))
            if bool(np.asarray(progress[start:stop]).all()):
                continue
            if bool(np.asarray(progress[start:stop]).any()):
                raise RuntimeError("partially decoded batch cannot be resumed safely")
            latent = torch.from_numpy(local_endpoint[start:stop]).to(device)
            shard[start:stop] = decode_latents(vae, latent)
            shard.flush()
            progress[start:stop] = True
            progress.flush()
            if rank == 0 and (
                stop % args.log_every == 0 or stop == len(local_ids)
            ):
                print(
                    f"[rank 0] {condition_name}: {stop}/{len(local_ids)}",
                    flush=True,
                )
        dist.barrier()
        if rank == 0:
            path = merge_shards(
                output_dir=output_dir,
                condition_name=condition_name,
                samples=samples,
                world_size=world_size,
            )
            print(f"wrote {path}", flush=True)
        dist.barrier()
    if rank == 0:
        final = json.loads(manifest_path.read_text(encoding="utf-8"))
        final["status"] = "complete"
        _atomic_json(manifest_path, final)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

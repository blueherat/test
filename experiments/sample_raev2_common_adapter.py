"""Same-noise sampling for frozen RAEv2 with common residual adapters."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_common_adapter import (  # noqa: E402
    COMMON_ADAPTER_FORMAT,
    CommonResidualAdapter,
    forward_with_internalguidance_common_adapter,
    load_common_adapter_checkpoint,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    tensor_fingerprint,
)
from experiments.sample_raev2_threeway import (  # noqa: E402
    generator_fingerprint,
    load_config,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_branch(value: str) -> tuple[str, Path | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=CHECKPOINT or NAME=zero")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("branch name cannot be empty")
    if raw_path.lower() == "zero":
        return name, None
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument("--adapter-state-key", choices=("adapter", "adapter_ema"), default="adapter")
    parser.add_argument("--zero-hidden-channels", type=int, default=64)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument(
        "--allow-unbalanced-smoke",
        action="store_true",
        help="Allow a non-1000-multiple sample count for engineering checks only.",
    )
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ig-scale", type=float, default=1.78)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def _load_adapter(
    checkpoint_path: Path | None,
    *,
    channels: int,
    zero_hidden_channels: int,
    source_sha256: str,
    source_state_key: str,
    adapter_state_key: str,
    device: torch.device,
) -> tuple[CommonResidualAdapter, dict[str, object]]:
    if checkpoint_path is None:
        adapter = CommonResidualAdapter(
            channels,
            hidden_channels=int(zero_hidden_channels),
        ).to(device)
        return adapter, {
            "kind": "zero",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "branch_update": 0,
            "state_key": None,
        }

    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != COMMON_ADAPTER_FORMAT:
        raise ValueError(f"{checkpoint_path} is not a common-adapter checkpoint")
    metadata = checkpoint.get("common_adapter")
    if not isinstance(metadata, dict):
        raise ValueError(f"{checkpoint_path} has no common_adapter metadata")
    if metadata.get("source_sha256") != source_sha256:
        raise ValueError("adapter and requested source checkpoint hashes differ")
    if metadata.get("source_state_key") != source_state_key:
        raise ValueError("adapter and requested source state keys differ")
    config = checkpoint.get("adapter_config")
    if not isinstance(config, dict):
        raise ValueError("adapter checkpoint has no adapter_config")
    adapter = CommonResidualAdapter(**config).to(device)
    load_common_adapter_checkpoint(
        adapter,
        checkpoint,
        state_key=adapter_state_key,
    )
    result = {
        "kind": "trained",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "branch_update": int(metadata.get("branch_update", 0)),
        "objective": metadata.get("objective"),
        "lpl_variant": metadata.get("lpl_variant"),
        "state_key": adapter_state_key,
    }
    del checkpoint
    return adapter, result


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.sample_count % 1000 and not args.allow_unbalanced_smoke:
        raise ValueError(
            "--sample-count must be a multiple of 1000 unless "
            "--allow-unbalanced-smoke is set"
        )
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    if args.zero_hidden_channels <= 0:
        raise ValueError("--zero-hidden-channels must be positive")
    if args.ig_scale < 0:
        raise ValueError("--ig-scale must be non-negative")
    names = [name for name, _ in args.branch]
    if len(names) != len(set(names)):
        raise ValueError("branch names must be unique")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)
    latent_size = tuple(config.misc.latent_size)
    source_path = args.source_checkpoint.expanduser().resolve()
    source_sha256 = file_sha256(source_path)
    source_checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if args.source_state_key not in source_checkpoint:
        raise KeyError(f"source checkpoint has no {args.source_state_key!r} state")
    source_step = int(source_checkpoint.get("step", 0))

    source_model = instantiate_from_config(config.stage_2).to(device)
    source_model.load_state_dict(
        source_checkpoint[args.source_state_key],
        strict=True,
    )
    source_model.eval()
    source_model.requires_grad_(False)
    del source_checkpoint

    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    rae.requires_grad_(False)
    del rae.encoder
    torch.cuda.empty_cache()

    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    guidance_interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.expanduser().mkdir(parents=True, exist_ok=True)

    for branch_name, adapter_path in args.branch:
        adapter, adapter_metadata = _load_adapter(
            adapter_path,
            channels=int(source_model.in_channels),
            zero_hidden_channels=int(args.zero_hidden_channels),
            source_sha256=source_sha256,
            source_state_key=args.source_state_key,
            adapter_state_key=args.adapter_state_key,
            device=device,
        )
        adapter.eval()
        adapter.requires_grad_(False)
        model_fn = partial(
            forward_with_internalguidance_common_adapter,
            source_model,
            adapter,
        )
        model_kwargs_base = {
            "ig_scale": float(args.ig_scale),
            "ig_interval": guidance_interval,
        }

        output_dir = args.results_dir.expanduser() / branch_name
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.sampling_seed) * world_size + rank)
        initial_generator_sha256 = generator_fingerprint(generator)
        images_local: list[np.ndarray] = []
        first_noise_sha256 = None
        first_label_sha256 = None

        with torch.inference_mode():
            for start in range(0, len(global_ids), args.per_rank_batch):
                ids = global_ids[start : start + args.per_rank_batch]
                batch_size = len(ids)
                noise = torch.randn(
                    batch_size,
                    *latent_size,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                labels = torch.from_numpy(ids % 1000).to(
                    device=device,
                    dtype=torch.long,
                )
                if first_noise_sha256 is None:
                    first_noise_sha256 = tensor_fingerprint(noise)
                    first_label_sha256 = tensor_fingerprint(labels)

                null = torch.full(
                    (batch_size,),
                    int(config.misc.num_classes),
                    device=device,
                    dtype=torch.long,
                )
                sample_noise = torch.cat((noise, noise), dim=0)
                context = torch.cat((labels, null), dim=0)
                model_kwargs = dict(model_kwargs_base)
                model_kwargs.update(context=context, attn_mask=None)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else __import__("contextlib").nullcontext()
                )
                with autocast:
                    latent = sample_fn(
                        sample_noise,
                        model_fn,
                        **model_kwargs,
                    )[-1]
                    latent = latent.chunk(2, dim=0)[0]
                    decoded = rae.decode(latent).clamp(0, 1)
                images_local.append(
                    decoded.mul(255)
                    .permute(0, 2, 3, 1)
                    .to(device="cpu", dtype=torch.uint8)
                    .numpy()
                )

        images_array = np.concatenate(images_local, axis=0)
        np.save(output_dir / f"images-rank{rank:02d}.npy", images_array)
        np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
        audit = {
            "protocol": "raev2_common_adapter_same_noise_v1",
            "branch": branch_name,
            "rank": rank,
            "world_size": world_size,
            "source_checkpoint": str(source_path),
            "source_sha256": source_sha256,
            "source_state_key": args.source_state_key,
            "source_step": source_step,
            "adapter": adapter_metadata,
            "sampling_seed": int(args.sampling_seed),
            "sample_count": int(args.sample_count),
            "balanced_imagenet_classes": args.sample_count % 1000 == 0,
            "per_rank_batch": int(args.per_rank_batch),
            "sampler_steps": int(config.sampler.num_steps),
            "guidance_ig_scale": float(args.ig_scale),
            "guidance_ig_interval": list(guidance_interval),
            "initial_generator_sha256": initial_generator_sha256,
            "first_noise_sha256": first_noise_sha256,
            "first_label_sha256": first_label_sha256,
            "final_generator_sha256": generator_fingerprint(generator),
            "contrast_function_preserved_by_parameterization": True,
        }
        (output_dir / f"sampling_audit_rank{rank}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        dist.barrier()

        if rank == 0:
            all_ids = []
            all_images = []
            for shard_rank in range(world_size):
                all_ids.append(np.load(output_dir / f"ids-rank{shard_rank:02d}.npy"))
                all_images.append(
                    np.load(output_dir / f"images-rank{shard_rank:02d}.npy")
                )
            ids = np.concatenate(all_ids)
            images = np.concatenate(all_images)
            order = np.argsort(ids)
            ids = ids[order]
            images = images[order]
            if not np.array_equal(
                ids,
                np.arange(args.sample_count, dtype=np.int64),
            ):
                raise RuntimeError("distributed sample IDs are incomplete or duplicated")
            archive = output_dir / "samples.npz"
            np.savez(archive, images)
            summary = {
                "protocol": "raev2_common_adapter_same_noise_v1",
                "branch": branch_name,
                "source_checkpoint": str(source_path),
                "source_sha256": source_sha256,
                "source_state_key": args.source_state_key,
                "source_step": source_step,
                "adapter": adapter_metadata,
                "samples": int(images.shape[0]),
                "balanced_imagenet_classes": args.sample_count % 1000 == 0,
                "shape": list(images.shape),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
            }
            (output_dir / "sampling_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for shard_rank in range(world_size):
                (output_dir / f"images-rank{shard_rank:02d}.npy").unlink()
                (output_dir / f"ids-rank{shard_rank:02d}.npy").unlink()
            print(json.dumps(summary, ensure_ascii=False))
        dist.barrier()

        del adapter
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

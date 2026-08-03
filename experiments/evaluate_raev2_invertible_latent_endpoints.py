"""Decode cached RAEv2 endpoints through invertible latent adapters.

Sampling is deliberately outside this script.  All adapter branches reuse the
same cached endpoint latents, so any decoded difference is caused only by
``A^{-1}`` before the frozen RAE decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.latent_equiv_adapter import InvertibleLatentAdapter  # noqa: E402
from experiments.raev2_invertible_latent_lpl import (  # noqa: E402
    INVERTIBLE_LATENT_LPL_FORMAT,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.run_raev2_decoded_distribution_audit import (  # noqa: E402
    feature_statistics,
    fid_between_statistics,
    load_reference_statistics,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    load_config,
)
from experiments.run_raev2_scale_response_study import (  # noqa: E402
    atomic_save_npy,
    extract_inception,
    rank_file,
    scale_key,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("adapter name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--scale", action="append", type=float, dest="scales")
    parser.add_argument("--adapter", action="append", type=parse_named_path, default=[])
    parser.add_argument(
        "--adapter-state-key",
        choices=("adapter", "adapter_ema"),
        default="adapter",
    )
    parser.add_argument("--fid-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def load_adapters(
    named_paths: list[tuple[str, Path]],
    *,
    source_sha256: str,
    source_state_key: str,
    state_key: str,
    device: torch.device,
) -> list[tuple[str, InvertibleLatentAdapter | None, dict[str, Any]]]:
    names = ["identity", *(name for name, _ in named_paths)]
    if len(names) != len(set(names)):
        raise ValueError("adapter names must be unique and cannot be 'identity'")
    loaded: list[tuple[str, InvertibleLatentAdapter | None, dict[str, Any]]] = [
        (
            "identity",
            None,
            {
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "branch_update": 0,
                "training_objective": "identity",
            },
        )
    ]
    for name, raw_path in named_paths:
        path = raw_path.resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != INVERTIBLE_LATENT_LPL_FORMAT:
            raise ValueError(f"{path} is not an invertible latent LPL checkpoint")
        metadata = checkpoint.get("invertible_latent_lpl", {})
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError(f"{path} source checkpoint hash mismatch")
        if metadata.get("source_state_key") != source_state_key:
            raise ValueError(f"{path} source state mismatch")
        config = checkpoint.get("adapter_config")
        if not isinstance(config, dict):
            raise ValueError(f"{path} has no adapter configuration")
        adapter = InvertibleLatentAdapter(**config).to(device)
        adapter.load_state_dict(checkpoint[state_key], strict=True)
        adapter.eval().requires_grad_(False)
        loaded.append(
            (
                name,
                adapter,
                {
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "branch_update": int(metadata["branch_update"]),
                    "training_objective": str(metadata["objective"]),
                },
            )
        )
        del checkpoint
    return loaded


def decode_features(
    *,
    latents: np.ndarray,
    adapter: InvertibleLatentAdapter | None,
    decoder: torch.nn.Module,
    extractor: torch.nn.Module,
    batch_size: int,
    precision: str,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, latents.shape[0], batch_size):
            batch = torch.from_numpy(
                np.asarray(latents[start : start + batch_size], dtype=np.float32)
            ).to(device)
            if adapter is not None:
                batch = adapter.inverse(batch)
            with autocast_context(precision):
                decoded = decoder.decode(batch).float()
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("decoder produced non-finite pixels")
            uint8 = decoded.clamp(0, 1).mul(255).to(torch.uint8)
            parts.append(extract_inception(extractor, uint8))
    return np.concatenate(parts, axis=0)


def add_identity_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    identity = summary.loc[
        summary["branch"] == "identity", ["scale", "fid_to_official"]
    ].rename(columns={"fid_to_official": "identity_fid"})
    result = summary.merge(identity, on="scale", how="left", validate="many_to_one")
    result["fid_delta_vs_identity"] = (
        result["fid_to_official"] - result["identity_fid"]
    )
    return result


def plot_fid_curves(summary: pd.DataFrame, path: Path) -> None:
    scales = sorted(summary["scale"].unique().tolist())
    figure, axes = plt.subplots(
        1, len(scales), figsize=(7.0 * len(scales), 5.2), squeeze=False
    )
    for axis, scale in zip(axes[0], scales, strict=True):
        part = summary[summary["scale"] == scale]
        for objective, style in (("flow", "--"), ("lpl", "-")):
            line = part[part["training_objective"] == objective].sort_values(
                "branch_update"
            )
            if line.empty:
                continue
            axis.plot(
                line["branch_update"],
                line["fid_delta_vs_identity"],
                marker="o",
                linestyle=style,
                linewidth=2,
                label=objective,
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"Internal-guidance scale {scale:g}")
        axis.set_xlabel("adapter update")
        axis.set_ylabel("FID change vs identity")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.decode_batch <= 0:
        raise ValueError("--decode-batch must be positive")
    scales = tuple(sorted(set(args.scales or (1.0, 1.78))))
    if not scales:
        raise ValueError("at least one scale is required")
    install_raev2_decoder_config_compat()
    os.environ["DINO_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    rank, world_size, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    endpoint_dir = args.endpoint_dir.resolve()
    endpoint_manifest = json.loads(
        (endpoint_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source_path = args.source_checkpoint.resolve()
    source_sha256 = file_sha256(source_path) if rank == 0 else ""
    hashes = [source_sha256]
    dist.broadcast_object_list(hashes, src=0)
    source_sha256 = hashes[0]
    if Path(endpoint_manifest["checkpoint"]).resolve() != source_path:
        raise ValueError("endpoint source checkpoint differs from requested source")
    if endpoint_manifest["state_key"] != args.source_state_key:
        raise ValueError("endpoint source state differs from requested source state")
    if int(endpoint_manifest["world_size"]) != world_size:
        raise ValueError("endpoint shards were generated with a different world size")
    available_scales = {float(value) for value in endpoint_manifest["scales"]}
    if any(float(scale) not in available_scales for scale in scales):
        raise ValueError("a requested scale is absent from the endpoint cache")

    adapters = load_adapters(
        args.adapter,
        source_sha256=source_sha256,
        source_state_key=args.source_state_key,
        state_key=args.adapter_state_key,
        device=device,
    )
    config = load_config(args.config.resolve())
    decoder = instantiate_from_config(config.stage_1)
    del decoder.encoder
    decoder = decoder.to(device).eval().requires_grad_(False)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device)

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    rows: list[dict[str, Any]] = []
    for scale in scales:
        condition = scale_key(scale)
        latent_path = rank_file(endpoint_dir / "latents", condition, rank)
        latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
        for name, adapter, metadata in adapters:
            features = decode_features(
                latents=latents,
                adapter=adapter,
                decoder=decoder,
                extractor=extractor,
                batch_size=int(args.decode_batch),
                precision=args.precision,
                device=device,
            )
            feature_path = rank_file(
                output_dir / "inception", f"{condition}_{name}", rank
            )
            atomic_save_npy(feature_path, features)
            rows.append(
                {
                    "scale": float(scale),
                    "condition": condition,
                    "branch": name,
                    **metadata,
                    "rank": rank,
                    "samples": int(features.shape[0]),
                    "feature_path": str(feature_path),
                }
            )
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        reference = load_reference_statistics(args.fid_reference.resolve(), "2048")
        summary_rows = []
        for scale in scales:
            condition = scale_key(scale)
            for name, _, metadata in adapters:
                features = np.concatenate(
                    [
                        np.load(
                            rank_file(
                                output_dir / "inception",
                                f"{condition}_{name}",
                                shard_rank,
                            ),
                            allow_pickle=False,
                        )
                        for shard_rank in range(world_size)
                    ],
                    axis=0,
                )
                statistics = feature_statistics(features)
                summary_rows.append(
                    {
                        "scale": float(scale),
                        "condition": condition,
                        "branch": name,
                        **metadata,
                        "samples": int(features.shape[0]),
                        "fid_to_official": fid_between_statistics(
                            statistics, reference
                        ),
                    }
                )
        summary = add_identity_deltas(
            pd.DataFrame(summary_rows).sort_values(
                ["scale", "training_objective", "branch_update", "branch"]
            )
        )
        pd.DataFrame([row for rank_rows in gathered for row in rank_rows]).to_csv(
            output_dir / "rank_outputs.csv", index=False
        )
        summary.to_csv(output_dir / "endpoint_fid.csv", index=False)
        plot_fid_curves(summary, output_dir / "endpoint_fid_curves.png")
        manifest = {
            "format_version": 1,
            "scope": "fixed_cached_endpoint_inverse_adapter_decode",
            "source_checkpoint": str(source_path),
            "source_sha256": source_sha256,
            "source_state_key": args.source_state_key,
            "endpoint_dir": str(endpoint_dir),
            "endpoint_manifest": endpoint_manifest,
            "fid_reference": str(args.fid_reference.resolve()),
            "scales": list(scales),
            "adapter_state_key": args.adapter_state_key,
            "precision": args.precision,
            "world_size": world_size,
            "same_endpoint_latents_across_branches": True,
            "stage1_decoder_frozen": True,
            "interpretation_limit": "N=1000 FID is an early paired diagnostic, not a final FID estimate.",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(summary.to_string(index=False))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

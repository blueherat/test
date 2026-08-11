"""Measure per-batch adaptive-solver NFE for SiT and dual-output paths.

The script performs no decoding and stores no generated latents.  Every path
reuses the exact same in-memory noise and labels on each rank, so differences
in NFE come only from the evaluated vector field.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from experiments.sample_imagenet100_sit_dual_fid import conditional_dual_velocity
    from experiments.sample_imagenet100_sit_fid import (
        conditional_velocity,
        official_rank_seed,
        official_total_samples,
    )
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.train_imagenet100_sit_dual_output import (
        PROTOCOL as DUAL_PROTOCOL,
        create_dual_output_sit,
    )
    from experiments.train_imagenet100_sit_flow import (
        LATENT_SHAPE,
        NUM_CLASSES,
        load_official_sit_module,
        sha256_file,
    )
except ImportError:
    from sample_imagenet100_sit_dual_fid import conditional_dual_velocity
    from sample_imagenet100_sit_fid import (
        conditional_velocity,
        official_rank_seed,
        official_total_samples,
    )
    from sample_imagenet100_sit_flow import integrate_velocity
    from train_imagenet100_sit_dual_output import (
        PROTOCOL as DUAL_PROTOCOL,
        create_dual_output_sit,
    )
    from train_imagenet100_sit_flow import (
        LATENT_SHAPE,
        NUM_CLASSES,
        load_official_sit_module,
        sha256_file,
    )


DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_BASELINE_CHECKPOINT = (
    DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_DUAL_CHECKPOINT = (
    DATA_ROOT
    / "runs/sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt"
)
DEFAULT_OFFICIAL_SIT_REPO = Path("/home/zhoushunyu/data/research_repos/SiT")
DEFAULT_OUTPUT_DIR = Path(
    "docs/data/imagenet100_sit_dual_endpoint_audit/nfe_per_batch_audit"
)
BASELINE_PROTOCOL = "imagenet100_sit_linear_flow_v1"
MODES = ("velocity", "x", "epsilon", "dynamic")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for mode in MODES:
        values = torch.tensor(
            [float(row["nfe"]) for row in rows if row["mode"] == mode],
            dtype=torch.float64,
        )
        if values.numel() == 0:
            raise ValueError(f"no NFE rows for {mode}")
        quantiles = torch.quantile(
            values, torch.tensor([0.05, 0.5, 0.95], dtype=values.dtype)
        )
        summaries.append(
            {
                "mode": mode,
                "batch_count": int(values.numel()),
                "mean_nfe": float(values.mean().item()),
                "std_nfe": float(values.std(unbiased=False).item()),
                "min_nfe": int(values.min().item()),
                "q05_nfe": float(quantiles[0].item()),
                "q50_nfe": float(quantiles[1].item()),
                "q95_nfe": float(quantiles[2].item()),
                "max_nfe": int(values.max().item()),
            }
        )
    return summaries


def save_nfe_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    values = [
        [float(row["nfe"]) for row in rows if row["mode"] == mode]
        for mode in MODES
    ]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot(values, tick_labels=MODES, showmeans=True)
    axis.set(
        xlabel="vector-field path",
        ylabel="NFE per batch trajectory",
        title="Adaptive Dopri5 effort on paired noise and labels",
    )
    axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_models(args: argparse.Namespace, device: torch.device):
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    dual_checkpoint = torch.load(
        args.dual_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if baseline_checkpoint.get("protocol") != BASELINE_PROTOCOL:
        raise ValueError("unexpected baseline checkpoint protocol")
    if dual_checkpoint.get("protocol") != DUAL_PROTOCOL:
        raise ValueError("unexpected dual checkpoint protocol")
    if baseline_checkpoint.get("official_sit") != source_metadata:
        raise ValueError("baseline checkpoint uses a different SiT source")
    if dual_checkpoint.get("official_sit") != source_metadata:
        raise ValueError("dual checkpoint uses a different SiT source")

    baseline_config = baseline_checkpoint["config"]
    baseline = sit_module.SiT_models[baseline_config["model_name"]](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(baseline_config["cfg_dropout"]),
    )
    baseline.load_state_dict(baseline_checkpoint[args.weights], strict=True)

    dual_config = dual_checkpoint["config"]
    dual = create_dual_output_sit(
        sit_module,
        model_name=dual_config["model_name"],
        cfg_dropout=float(dual_config["cfg_dropout"]),
    )
    dual.load_state_dict(dual_checkpoint[args.weights], strict=True)
    baseline.to(device).eval().requires_grad_(False)
    dual.to(device).eval().requires_grad_(False)
    return baseline, dual, baseline_checkpoint, dual_checkpoint, source_metadata


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    baseline, dual, baseline_checkpoint, dual_checkpoint, source_metadata = load_models(
        args, device
    )
    total_samples = official_total_samples(
        args.num_samples, args.per_rank_batch_size, world_size
    )
    samples_per_rank = total_samples // world_size
    iterations = samples_per_rank // args.per_rank_batch_size
    rank_seed = official_rank_seed(args.global_seed, world_size, rank)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    batches = [
        (
            torch.randn(args.per_rank_batch_size, *LATENT_SHAPE, device=device),
            torch.randint(
                0, NUM_CLASSES, (args.per_rank_batch_size,), device=device
            ),
        )
        for _ in range(iterations)
    ]
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    rows: list[dict[str, object]] = []
    for mode in MODES:
        model = baseline if mode == "velocity" else dual
        for batch_index, (noise, labels) in enumerate(batches):
            if mode == "velocity":
                velocity, counter = conditional_velocity(
                    model, labels, autocast_dtype=autocast_dtype
                )
            else:
                velocity, counter = conditional_dual_velocity(
                    model,
                    labels,
                    mode=mode,
                    gate_activation=dual_checkpoint["config"]["gate_activation"],
                    denominator_floor=float(
                        dual_checkpoint["config"]["denominator_floor"]
                    ),
                    autocast_dtype=autocast_dtype,
                )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            endpoint = integrate_velocity(
                noise.clone(),
                velocity,
                num_output_points=args.num_output_points,
                atol=args.atol,
                rtol=args.rtol,
            )
            torch.cuda.synchronize(device)
            rows.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "batch_index": batch_index,
                    "batch_size": args.per_rank_batch_size,
                    "nfe": int(counter["nfe"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "endpoint_mean": float(endpoint.mean().item()),
                    "endpoint_std": float(endpoint.std().item()),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = output_dir / f"rank_{rank:02d}.json"
    rank_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    if rank == 0:
        all_rows: list[dict[str, object]] = []
        for source_rank in range(world_size):
            all_rows.extend(
                json.loads(
                    (output_dir / f"rank_{source_rank:02d}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        all_rows.sort(
            key=lambda row: (
                MODES.index(str(row["mode"])),
                int(row["rank"]),
                int(row["batch_index"]),
            )
        )
        summaries = summarize_rows(all_rows)
        write_csv(all_rows, output_dir / "nfe_per_batch.csv")
        write_csv(summaries, output_dir / "nfe_distribution_summary.csv")
        save_nfe_plot(all_rows, output_dir / "nfe_distribution.png")
        manifest = {
            "protocol": "imagenet100_sit_nfe_per_batch_audit_v1",
            "baseline_checkpoint_name": args.baseline_checkpoint.name,
            "baseline_checkpoint_sha256": sha256_file(args.baseline_checkpoint),
            "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
            "dual_checkpoint_name": args.dual_checkpoint.name,
            "dual_checkpoint_sha256": sha256_file(args.dual_checkpoint),
            "dual_checkpoint_step": int(dual_checkpoint["step"]),
            "weights": args.weights,
            "official_sit": source_metadata,
            "requested_samples": args.num_samples,
            "padded_samples": total_samples,
            "world_size": world_size,
            "per_rank_batch_size": args.per_rank_batch_size,
            "same_noise_and_labels_across_modes": True,
            "global_seed": args.global_seed,
            "sampler": {
                "method": "dopri5",
                "num_output_points": args.num_output_points,
                "atol": args.atol,
                "rtol": args.rtol,
                "precision": args.precision,
                "allow_tf32": bool(args.allow_tf32),
            },
            "summary": summaries,
        }
        (output_dir / "nfe_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "complete", "summary": summaries}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT
    )
    parser.add_argument("--dual-checkpoint", type=Path, default=DEFAULT_DUAL_CHECKPOINT)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0 or args.per_rank_batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()

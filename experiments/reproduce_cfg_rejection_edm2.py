#!/usr/bin/env python3
"""Reproduce the ImageNet EDM2 path signal used by CFG-Rejection.

This runner turns the official notebook into a resumable command-line
experiment while preserving its model checkpoints, guidance rule, Heun grid,
and seed semantics.  It intentionally records three different quantities:

1. the early-step ranking statistic used by the official notebook, evaluated
   with the notebook's FP32 norm arithmetic;
2. squared EDM2 denoiser-output differences;
3. squared score differences, using score = (D(x, sigma) - x) / sigma**2.

The deterministic Heun sampler in this file is a baseline reproduction only.
It is not a Gaussian path-likelihood-ratio experiment.

Sampling equations are adapted from the official NVIDIA EDM2 implementation
and the official CFG-Rejection notebook.  Their source licenses continue to
apply to those components:
https://github.com/NVlabs/edm2
https://github.com/WSX20003/CFG-Rejection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


PAPER_CLASS_IDS = (
    1,
    207,
    949,
    388,
    483,
    511,
    759,
    888,
    957,
    936,
    850,
    776,
    701,
    681,
    579,
    405,
    289,
    277,
    122,
    16,
    95,
    638,
    106,
    244,
    336,
    437,
    520,
    598,
    629,
    646,
    657,
    667,
    727,
    777,
    803,
    812,
    900,
    980,
    779,
    562,
    965,
    857,
    905,
    652,
    101,
    979,
    914,
    108,
    698,
    926,
)

SMOKE_CLASS_IDS = (207, 388, 949)  # golden retriever, giant panda, strawberry
# Classes shown as qualitative failures or density-analysis examples in the
# paper.  We intentionally omit the paper/notebook's ambiguous "hummingbird"
# entry: standard ImageNet class 94 is hummingbird, while the released 50-class
# list contains class 95 (jacamar).  Neither identifier should silently stand
# in for the other in a locked protocol.
BAD_CASE_CLASS_IDS = (289, 336, 405, 437, 520, 562, 681, 701, 900, 936)

MAIN_MODEL_NAME = "edm2-img512-s-2147483-0.025.pkl"
WEAK_MODEL_NAME = "edm2-img512-xs-uncond-2147483-0.025.pkl"
MODEL_URL_ROOT = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"
MAIN_MODEL_BYTES = 560_565_890
WEAK_MODEL_BYTES = 248_541_796


@dataclass(frozen=True)
class Protocol:
    class_ids: tuple[int, ...]
    seeds: tuple[int, ...]


def parse_int_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges such as 1,4-7."""

    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", token)
        if match:
            start, stop = (int(match.group(1)), int(match.group(2)))
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending range is not allowed: {token}")
            result.extend(range(start, stop + 1))
        else:
            try:
                result.append(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid integer specification: {token}") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer specification is empty")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer specification contains duplicates")
    return tuple(result)


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    if args.protocol == "smoke":
        default_classes = SMOKE_CLASS_IDS
        default_seeds = tuple(range(8))
    elif args.protocol == "pilot":
        default_classes = BAD_CASE_CLASS_IDS
        default_seeds = tuple(range(100))
    elif args.protocol == "paper-10k":
        default_classes = PAPER_CLASS_IDS
        default_seeds = tuple(range(200))
    elif args.protocol == "confirmation-10k":
        # Never render tail grids or inspect these images until every detector,
        # endpoint, and threshold has been locked on discovery/validation data.
        default_classes = PAPER_CLASS_IDS
        default_seeds = tuple(range(10_000, 10_200))
    elif args.protocol == "notebook-500k":
        default_classes = PAPER_CLASS_IDS
        default_seeds = tuple(range(10_000))
    elif args.protocol == "custom":
        if args.classes is None or args.seeds is None:
            raise ValueError("--protocol custom requires both --classes and --seeds")
        default_classes = ()
        default_seeds = ()
    else:  # pragma: no cover - argparse constrains this value.
        raise AssertionError(args.protocol)

    class_ids = args.classes if args.classes is not None else default_classes
    seeds = args.seeds if args.seeds is not None else default_seeds
    if any(class_id < 0 or class_id >= 1_000 for class_id in class_ids):
        raise ValueError("ImageNet class IDs must be in [0, 999]")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return Protocol(tuple(class_ids), tuple(seeds))


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_model(path: Path, expected_bytes: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing checkpoint: {path}\n"
            "Run experiments/download_cross_scale_baselines.sh first, or pass an explicit path."
        )
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"checkpoint has the wrong size: {path} ({actual_bytes:,} != {expected_bytes:,} bytes); "
            "the download may be incomplete"
        )


def add_edm2_to_import_path(edm2_root: Path) -> tuple[Any, Any]:
    required = edm2_root / "training" / "networks_edm2.py"
    if not required.is_file():
        raise FileNotFoundError(f"not an EDM2 checkout: {edm2_root}")
    sys.path.insert(0, str(edm2_root))
    import dnnlib  # type: ignore[import-not-found]
    from torch_utils import distributed as dist  # type: ignore[import-not-found]

    return dnnlib, dist


class StackedRandomGenerator:
    """One independent CUDA generator per sample, matching official EDM2."""

    def __init__(self, device: torch.device, seeds: Sequence[int]) -> None:
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size: Sequence[int], **kwargs: Any) -> torch.Tensor:
        if size[0] != len(self.generators):
            raise ValueError("batch size and number of generators differ")
        return torch.stack(
            [torch.randn(tuple(size[1:]), generator=generator, **kwargs) for generator in self.generators]
        )


@torch.inference_mode()
def instrumented_heun_sampler(
    net: torch.nn.Module,
    gnet: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_steps: int = 32,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    guidance: float = 1.4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the deterministic paper sampler and retain unambiguous path signals."""

    if num_steps < 2:
        raise ValueError("Heun reproduction requires at least two steps")
    device = noise.device
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    batch_size, channels = noise.shape[:2]
    gap_euler = torch.zeros(batch_size, num_steps, channels, dtype=torch.float64, device=device)
    gap_prime = torch.zeros_like(gap_euler)
    official_gap_euler = torch.zeros(
        batch_size, num_steps, channels, dtype=torch.float32, device=device
    )
    official_gap_prime = torch.zeros_like(official_gap_euler)

    def denoise_and_gap(
        x: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denoised = net(x, sigma, labels).to(dtype)
        weak_denoised = gnet(x, sigma, labels).to(dtype)
        difference = denoised - weak_denoised
        # The released notebook takes this norm directly in FP32.  Preserve it
        # separately from the higher-precision diagnostic energies below.
        official_channel_gap = torch.linalg.vector_norm(difference, dim=(-2, -1))
        channel_gap = torch.linalg.vector_norm(difference.to(torch.float64), dim=(-2, -1))
        guided = weak_denoised.lerp(denoised, guidance)
        return guided, official_channel_gap, channel_gap

    x_next = noise.to(dtype) * t_steps[0]
    for step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat = x_next
        guided, current_official_gap, current_gap = denoise_and_gap(x_hat, t_cur)
        official_gap_euler[:, step] = current_official_gap
        gap_euler[:, step] = current_gap
        derivative = (x_hat - guided) / t_cur
        proposal = x_hat + (t_next - t_cur) * derivative

        if step < num_steps - 1:
            guided_prime, current_official_prime_gap, current_prime_gap = denoise_and_gap(proposal, t_next)
            official_gap_prime[:, step] = current_official_prime_gap
            gap_prime[:, step] = current_prime_gap
            derivative_prime = (proposal - guided_prime) / t_next
            x_next = x_hat + (t_next - t_cur) * (0.5 * derivative + 0.5 * derivative_prime)
        else:
            x_next = proposal

    sigma_euler = t_steps[:-1].to(torch.float64)
    sigma_prime = t_steps[1:].to(torch.float64)
    denoiser_l2_euler = torch.sqrt(torch.sum(gap_euler.square(), dim=-1))
    denoiser_l2_prime = torch.sqrt(torch.sum(gap_prime.square(), dim=-1))
    score_l2_euler = denoiser_l2_euler / sigma_euler.square().unsqueeze(0)
    score_l2_prime = torch.zeros_like(denoiser_l2_prime)
    nonzero_prime = sigma_prime > 0
    score_l2_prime[:, nonzero_prime] = (
        denoiser_l2_prime[:, nonzero_prime] / sigma_prime[nonzero_prime].square().unsqueeze(0)
    )

    early_steps = min(5, num_steps)
    combined_official_channel_gap = official_gap_euler + official_gap_prime
    # Match the released notebook's arithmetic order exactly: it transfers the
    # FP32 [batch, step, channel] tensor to CPU, averages channels first, and
    # then averages the first five solver iterations.
    official_notebook_metric = (
        combined_official_channel_gap[:, :early_steps]
        .detach()
        .cpu()
        .mean(dim=-1)
        .mean(dim=-1)
    )
    signals = {
        "sigma_euler": sigma_euler,
        "sigma_prime": sigma_prime,
        "gap_channel_euler": gap_euler,
        "gap_channel_prime": gap_prime,
        "official_gap_channel_euler": official_gap_euler,
        "official_gap_channel_prime": official_gap_prime,
        "official_gap_channel_sum": combined_official_channel_gap,
        "denoiser_l2_euler": denoiser_l2_euler,
        "denoiser_l2_prime": denoiser_l2_prime,
        "score_l2_euler": score_l2_euler,
        "score_l2_prime": score_l2_prime,
        # This is exactly the statistic used in the released ImageNet notebook.
        "official_notebook_metric_tau5": official_notebook_metric,
        # These retain the paper's squared-accumulation form.  Euler evaluations
        # are used once per solver step so Heun's extra model call is not counted
        # as a second diffusion time.
        "denoiser_asd_tau5": denoiser_l2_euler[:, :early_steps].square().sum(dim=1),
        "denoiser_asd_full": denoiser_l2_euler.square().sum(dim=1),
        "score_asd_tau5": score_l2_euler[:, :early_steps].square().sum(dim=1),
        "score_asd_full": score_l2_euler.square().sum(dim=1),
    }
    return x_next, signals


def chunks(items: Sequence[tuple[int, int]], size: int) -> Iterable[Sequence[tuple[int, int]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sample_paths(args: argparse.Namespace, protocol: Protocol) -> None:
    validate_model(args.net, MAIN_MODEL_BYTES)
    validate_model(args.gnet, WEAK_MODEL_BYTES)
    dnnlib, dist = add_edm2_to_import_path(args.edm2_root)
    dist.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "experiment": "cfg_rejection_edm2_reproduction",
        "role": "deterministic_baseline_reproduction_not_path_lr",
        "protocol": args.protocol,
        "class_ids": list(protocol.class_ids),
        "seeds": list(protocol.seeds),
        "sample_count": len(protocol.class_ids) * len(protocol.seeds),
        "model": {
            "main_path": str(args.net.resolve()),
            "main_url": f"{MODEL_URL_ROOT}/{MAIN_MODEL_NAME}",
            "main_bytes": args.net.stat().st_size,
            "weak_path": str(args.gnet.resolve()),
            "weak_url": f"{MODEL_URL_ROOT}/{WEAK_MODEL_NAME}",
            "weak_bytes": args.gnet.stat().st_size,
        },
        "sampler": {
            "name": "EDM2 deterministic Heun",
            "num_steps": args.steps,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "guidance": args.guidance,
            "S_churn": 0,
        },
        "execution": {
            "world_size": world_size,
            "model_batch_per_rank": args.batch,
            "vae_batch_per_rank": args.encoder_batch,
        },
        "metric_definitions": {
            "official_notebook_metric_tau5": (
                "mean over the first five steps and channels of the sum of Euler and Heun-prime "
                "FP32 spatial L2 norms of EDM2 denoiser-output differences; exactly follows the "
                "released notebook's GPU FP32 sum then CPU mean(channel), mean(step) order"
            ),
            "denoiser_asd": "sum over solver steps of squared full denoiser-output L2 differences",
            "score_asd": (
                "schedule-dependent unweighted sum over Euler iterations of squared score-gap L2, "
                "using gap_D / sigma^2; not invariant across schedules or step counts"
            ),
            "tau5_suffix": "the first five solver iterations (early5), not physical time tau=5",
        },
        "sources": {
            "paper": "https://arxiv.org/abs/2505.23343",
            "cfg_rejection_repo": "https://github.com/WSX20003/CFG-Rejection",
            "edm2_repo": "https://github.com/NVlabs/edm2",
            "edm2_revision": git_revision(args.edm2_root),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "world_size": world_size,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if not args.skip_model_hash:
        if rank == 0:
            manifest["model"]["main_sha256"] = sha256_file(args.net)
            manifest["model"]["weak_sha256"] = sha256_file(args.gnet)
        # Other ranks do not need the digest to sample.
    if rank == 0:
        existing_manifest_path = output_dir / "manifest.json"
        if existing_manifest_path.exists():
            with existing_manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            comparable_keys = (
                "schema_version",
                "experiment",
                "role",
                "protocol",
                "class_ids",
                "seeds",
                "model",
                "sampler",
                "execution",
                "metric_definitions",
                "sources",
                "software",
                "runner",
            )
            mismatches = [key for key in comparable_keys if existing.get(key) != manifest.get(key)]
            if mismatches:
                raise RuntimeError(
                    f"output directory already has an incompatible manifest ({', '.join(mismatches)}): "
                    f"{existing_manifest_path}"
                )
        else:
            atomic_json_dump(manifest, existing_manifest_path)
    torch.distributed.barrier()

    with args.net.open("rb") as handle:
        main_payload = pickle.load(handle)
    net = main_payload["ema"].eval().requires_grad_(False).to(device)
    encoder = main_payload.get("encoder")
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name="training.encoders.StandardRGBEncoder")
    # On a fresh machine the encoder initialization downloads the SD-VAE.
    # Let rank 0 populate the shared cache before the remaining ranks load it.
    if rank == 0:
        encoder.init(device)
    torch.distributed.barrier()
    if rank != 0:
        encoder.init(device)
    torch.distributed.barrier()
    if hasattr(encoder, "batch_size"):
        encoder.batch_size = args.encoder_batch
    del main_payload

    with args.gnet.open("rb") as handle:
        weak_payload = pickle.load(handle)
    gnet = weak_payload["ema"].eval().requires_grad_(False).to(device)
    del weak_payload

    all_pairs = [(class_id, seed) for class_id in protocol.class_ids for seed in protocol.seeds]
    rank_pairs = all_pairs[rank::world_size]
    pending_pairs: list[tuple[int, int]] = []
    skipped = 0
    for class_id, seed in rank_pairs:
        image_path = output_dir / "images" / f"class_{class_id:04d}" / f"{seed:06d}.png"
        signal_path = output_dir / "signals" / f"class_{class_id:04d}" / f"{seed:06d}.npz"
        if image_path.is_file() and signal_path.is_file():
            skipped += 1
        else:
            pending_pairs.append((class_id, seed))

    start_time = time.monotonic()
    generated = 0
    for batch_pairs in chunks(pending_pairs, args.batch):
        batch_classes = [item[0] for item in batch_pairs]
        batch_seeds = [item[1] for item in batch_pairs]
        random = StackedRandomGenerator(device, batch_seeds)
        noise = random.randn(
            [len(batch_pairs), net.img_channels, net.img_resolution, net.img_resolution],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.zeros(len(batch_pairs), net.label_dim, dtype=torch.float32, device=device)
        labels[torch.arange(len(batch_pairs), device=device), torch.tensor(batch_classes, device=device)] = 1
        latents, signals = instrumented_heun_sampler(
            net,
            gnet,
            noise,
            labels,
            num_steps=args.steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            guidance=args.guidance,
        )
        images = encoder.decode(latents)

        cpu_signals = {name: value.detach().cpu().numpy() for name, value in signals.items()}
        cpu_images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
        for index, (class_id, seed) in enumerate(batch_pairs):
            image_path = output_dir / "images" / f"class_{class_id:04d}" / f"{seed:06d}.png"
            signal_path = output_dir / "signals" / f"class_{class_id:04d}" / f"{seed:06d}.npz"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.parent.mkdir(parents=True, exist_ok=True)

            image_tmp = image_path.with_suffix(".png.tmp")
            Image.fromarray(cpu_images[index], mode="RGB").save(image_tmp, format="PNG")
            os.replace(image_tmp, image_path)

            signal_tmp = signal_path.with_suffix(".npz.tmp")
            common_signal_names = {"sigma_euler", "sigma_prime"}
            per_sample = {
                name: value if name in common_signal_names else value[index]
                for name, value in cpu_signals.items()
            }
            with signal_tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    class_id=np.int64(class_id),
                    seed=np.int64(seed),
                    **per_sample,
                )
            os.replace(signal_tmp, signal_path)
            generated += 1

        if rank == 0:
            elapsed = time.monotonic() - start_time
            dist.print0(
                f"rank0 generated {generated}/{len(pending_pairs)} new paths "
                f"({skipped} already complete, {elapsed:.1f}s)"
            )

    local_counts = torch.tensor([generated, skipped], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.barrier()
    if rank == 0:
        completion = {
            "generated_this_run": int(local_counts[0].item()),
            "already_complete": int(local_counts[1].item()),
            "total_expected": len(all_pairs),
            "wall_seconds": time.monotonic() - start_time,
            "finished_at_unix": time.time(),
        }
        atomic_json_dump(completion, output_dir / "completion.json")
        dist.print0(json.dumps(completion, indent=2))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    baseline_root = data_root / "baselines"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("smoke", "pilot", "paper-10k", "confirmation-10k", "notebook-500k", "custom"),
        default="smoke",
    )
    parser.add_argument("--classes", type=parse_int_spec, default=None, help="Override class IDs, e.g. 1,207,388")
    parser.add_argument("--seeds", type=parse_int_spec, default=None, help="Override seeds, e.g. 0-31")
    parser.add_argument("--edm2-root", type=Path, default=baseline_root / "edm2")
    parser.add_argument(
        "--net", type=Path, default=baseline_root / "edm2" / "checkpoints" / MAIN_MODEL_NAME
    )
    parser.add_argument(
        "--gnet", type=Path, default=baseline_root / "edm2" / "checkpoints" / WEAK_MODEL_NAME
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--encoder-batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--guidance", type=float, default=1.4)
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved protocol without loading a GPU")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch < 1 or args.encoder_batch < 1:
        parser.error("batch sizes must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    protocol = protocol_from_args(args)
    if args.output_dir is None:
        args.output_dir = (
            Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
            / "cross_scale_evidence"
            / "cfg_rejection_edm2"
            / args.protocol
        )
    summary = {
        "protocol": args.protocol,
        "classes": list(protocol.class_ids),
        "seed_min": min(protocol.seeds),
        "seed_max": max(protocol.seeds),
        "seeds_per_class": len(protocol.seeds),
        "sample_count": len(protocol.class_ids) * len(protocol.seeds),
        "output_dir": str(args.output_dir),
        "main_model": str(args.net),
        "weak_model": str(args.gnet),
    }
    if args.dry_run or int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EDM2 sampling")
    sample_paths(args, protocol)


if __name__ == "__main__":
    main()

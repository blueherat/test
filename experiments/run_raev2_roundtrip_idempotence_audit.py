"""Audit clamp effects and repeated RAEv2 encode/decode round trips.

This is an inference-only fixed-latent experiment.  It evaluates the closure
map ``G(z) = E(clamp(D(z)))`` repeatedly on real, unguided, and guided endpoint
latents.  It also isolates the only direction comparison that is valid inside
the pixel part of the chain: decoder increments before versus after clamp.
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


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    load_config,
)
from experiments.run_raev2_scale_response_study import (  # noqa: E402
    local_ids_for_rank,
    scale_key,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_roundtrip_idempotence_audit_v1"
DEFAULT_INPUT = (
    Path.home()
    / "data/eqvae/experiments/raev2_ig_scale_response/n1000_seed20260801_v1"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/experiments/raev2_roundtrip_idempotence/n1000_seed20260801_v1"
)


def sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    flat = value.float().flatten(1)
    return flat.square().mean(dim=1).add(float(eps)).sqrt()


def sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    result = torch.full_like(numerator, float("nan"))
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def clamp_sample_metrics(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    values = raw.float()
    clamped = values.clamp(0, 1)
    low = (values < 0).flatten(1).float().mean(dim=1)
    high = (values > 1).flatten(1).float().mean(dim=1)
    distortion = sample_rms(clamped - values)
    return {
        "below_zero_fraction": low,
        "above_one_fraction": high,
        "clipped_fraction": low + high,
        "clamp_distortion_rms": distortion,
        "clamp_distortion_over_raw_rms": distortion / sample_rms(values, 1e-30),
        "raw_min": values.flatten(1).min(dim=1).values,
        "raw_max": values.flatten(1).max(dim=1).values,
    }


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distributed(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        raise RuntimeError("launch this audit with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(int(seed) + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return rank, world_size, device


def _load_rae(config: Any, device: torch.device) -> torch.nn.Module:
    install_raev2_decoder_config_compat()
    model = instantiate_from_config(config.stage_1)
    model = model.to(device).requires_grad_(False).eval()
    return model


def _latent_arrays(
    root: Path,
    *,
    rank: int,
    samples: int,
    world_size: int,
    control_scale: float,
    guided_scale: float,
) -> dict[str, np.ndarray]:
    count = len(local_ids_for_rank(samples, rank, world_size))
    conditions = {
        "real": "real",
        "control": scale_key(control_scale),
        "guided": scale_key(guided_scale),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, condition in conditions.items():
        path = root / "latents" / f"{condition}_rank{rank:02d}.npy"
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(value) < count:
            raise RuntimeError(f"not enough rows in {path}: {len(value)} < {count}")
        arrays[name] = value
    return arrays


def _append_condition_rows(
    rows: list[dict[str, object]],
    *,
    condition: str,
    sample_ids: np.ndarray,
    power: int,
    previous: torch.Tensor,
    current: torch.Tensor,
    first_step: torch.Tensor,
    raw: torch.Tensor,
) -> None:
    step = current - previous
    step_rms = sample_rms(step)
    first_rms = sample_rms(first_step, 1e-30)
    previous_rms = sample_rms(previous, 1e-30)
    clamp = clamp_sample_metrics(raw)
    step_cosine = sample_cosine(step, first_step)
    for offset, sample_id in enumerate(sample_ids.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "condition": condition,
                "power": int(power),
                "roundtrip_step_rms": float(step_rms[offset]),
                "roundtrip_step_over_input_rms": float(
                    step_rms[offset] / previous_rms[offset]
                ),
                "roundtrip_step_over_first_step": float(
                    step_rms[offset] / first_rms[offset]
                ),
                "roundtrip_step_vs_first_cosine": float(step_cosine[offset]),
                **{
                    name: float(value[offset]) for name, value in clamp.items()
                },
            }
        )


def _append_pair_rows(
    rows: list[dict[str, object]],
    *,
    sample_ids: np.ndarray,
    power: int,
    control_input: torch.Tensor,
    guided_input: torch.Tensor,
    control_raw: torch.Tensor,
    guided_raw: torch.Tensor,
    control_next: torch.Tensor,
    guided_next: torch.Tensor,
) -> None:
    latent_increment = guided_input - control_input
    pre_increment = guided_raw.float() - control_raw.float()
    post_increment = guided_raw.float().clamp(0, 1) - control_raw.float().clamp(0, 1)
    next_increment = guided_next - control_next
    latent_rms = sample_rms(latent_increment, 1e-30)
    pre_rms = sample_rms(pre_increment, 1e-30)
    post_rms = sample_rms(post_increment)
    next_rms = sample_rms(next_increment)
    pre_post_cosine = sample_cosine(pre_increment, post_increment)
    latent_next_cosine = sample_cosine(latent_increment, next_increment)
    for offset, sample_id in enumerate(sample_ids.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "power": int(power),
                "input_latent_increment_rms": float(latent_rms[offset]),
                "decoder_preclamp_increment_rms": float(pre_rms[offset]),
                "decoder_postclamp_increment_rms": float(post_rms[offset]),
                "next_latent_increment_rms": float(next_rms[offset]),
                "postclamp_over_preclamp_increment": float(
                    post_rms[offset] / pre_rms[offset]
                ),
                "preclamp_postclamp_increment_cosine": float(
                    pre_post_cosine[offset]
                ),
                "next_over_input_latent_increment": float(
                    next_rms[offset] / latent_rms[offset]
                ),
                "input_next_latent_increment_cosine": float(
                    latent_next_cosine[offset]
                ),
            }
        )


@torch.inference_mode()
def run_local(
    *,
    rae: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    local_ids: np.ndarray,
    batch_size: int,
    powers: int,
    precision: str,
    device: torch.device,
    rank: int,
    log_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    total_batches = math.ceil(len(local_ids) / batch_size)
    names = ("real", "control", "guided")
    for batch_index, start in enumerate(range(0, len(local_ids), batch_size)):
        stop = min(start + batch_size, len(local_ids))
        sample_ids = local_ids[start:stop]
        initial = {
            name: torch.from_numpy(
                np.array(arrays[name][start:stop], dtype=np.float32, copy=True)
            ).to(device)
            for name in names
        }
        current = {name: value for name, value in initial.items()}
        first_steps: dict[str, torch.Tensor] = {}
        for power in range(1, powers + 1):
            combined = torch.cat([current[name] for name in names], dim=0)
            with autocast_context(precision):
                raw_combined = rae.decode(combined).float()
                next_combined = rae.encode(raw_combined.clamp(0, 1)).float()
            raw_parts = dict(zip(names, raw_combined.chunk(len(names))))
            next_parts = dict(zip(names, next_combined.chunk(len(names))))
            for name in names:
                if power == 1:
                    first_steps[name] = next_parts[name] - current[name]
                _append_condition_rows(
                    condition_rows,
                    condition=name,
                    sample_ids=sample_ids,
                    power=power,
                    previous=current[name],
                    current=next_parts[name],
                    first_step=first_steps[name],
                    raw=raw_parts[name],
                )
            _append_pair_rows(
                pair_rows,
                sample_ids=sample_ids,
                power=power,
                control_input=current["control"],
                guided_input=current["guided"],
                control_raw=raw_parts["control"],
                guided_raw=raw_parts["guided"],
                control_next=next_parts["control"],
                guided_next=next_parts["guided"],
            )
            current = next_parts
        if rank == 0 and (
            (batch_index + 1) % log_every == 0
            or batch_index + 1 == total_batches
        ):
            print(f"[roundtrip audit] {batch_index + 1}/{total_batches}", flush=True)
    return pd.DataFrame(condition_rows), pd.DataFrame(pair_rows)


def _summarize(conditions: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_summary = (
        conditions.groupby(["condition", "power"], as_index=False)
        .agg(
            sample_count=("sample_id", "count"),
            step_over_first_mean=("roundtrip_step_over_first_step", "mean"),
            step_over_first_median=("roundtrip_step_over_first_step", "median"),
            step_over_input_median=("roundtrip_step_over_input_rms", "median"),
            step_first_cosine_mean=("roundtrip_step_vs_first_cosine", "mean"),
            clipping_fraction_mean=("clipped_fraction", "mean"),
            clipping_fraction_median=("clipped_fraction", "median"),
            clamp_distortion_over_raw_mean=("clamp_distortion_over_raw_rms", "mean"),
        )
    )
    pair_summary = (
        pairs.groupby("power", as_index=False)
        .agg(
            sample_count=("sample_id", "count"),
            post_over_pre_mean=("postclamp_over_preclamp_increment", "mean"),
            post_over_pre_median=("postclamp_over_preclamp_increment", "median"),
            pre_post_cosine_mean=("preclamp_postclamp_increment_cosine", "mean"),
            pre_post_cosine_median=("preclamp_postclamp_increment_cosine", "median"),
            next_over_input_mean=("next_over_input_latent_increment", "mean"),
            next_over_input_median=("next_over_input_latent_increment", "median"),
            input_next_cosine_mean=("input_next_latent_increment_cosine", "mean"),
            input_next_cosine_median=("input_next_latent_increment_cosine", "median"),
        )
    )
    return condition_summary, pair_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--powers", type=int, default=3)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--guided-scale", type=float, default=1.78)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = _distributed(args.seed)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("input scale-response artifact must be complete")
    if int(manifest["world_size"]) != world_size:
        raise RuntimeError("torchrun world size must match the source artifact")
    if args.samples <= 0 or args.samples > int(manifest["samples"]):
        raise ValueError("samples must be within the source artifact")
    if args.powers < 1:
        raise ValueError("powers must be positive")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    config = load_config(args.config.expanduser().resolve())
    arrays = _latent_arrays(
        input_dir,
        rank=rank,
        samples=int(args.samples),
        world_size=world_size,
        control_scale=float(args.control_scale),
        guided_scale=float(args.guided_scale),
    )
    local_ids = local_ids_for_rank(int(args.samples), rank, world_size)
    rae = _load_rae(config, device)
    conditions, pairs = run_local(
        rae=rae,
        arrays=arrays,
        local_ids=local_ids,
        batch_size=int(args.batch_size),
        powers=int(args.powers),
        precision=args.precision,
        device=device,
        rank=rank,
        log_every=int(args.log_every),
    )
    conditions.to_csv(output_dir / f"condition_metrics_rank{rank:02d}.csv", index=False)
    pairs.to_csv(output_dir / f"paired_metrics_rank{rank:02d}.csv", index=False)
    dist.barrier()
    if rank == 0:
        all_conditions = pd.concat(
            [
                pd.read_csv(output_dir / f"condition_metrics_rank{value:02d}.csv")
                for value in range(world_size)
            ],
            ignore_index=True,
        )
        all_pairs = pd.concat(
            [
                pd.read_csv(output_dir / f"paired_metrics_rank{value:02d}.csv")
                for value in range(world_size)
            ],
            ignore_index=True,
        )
        condition_summary, pair_summary = _summarize(all_conditions, all_pairs)
        all_conditions.to_csv(output_dir / "condition_metrics.csv", index=False)
        all_pairs.to_csv(output_dir / "paired_metrics.csv", index=False)
        condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
        pair_summary.to_csv(output_dir / "paired_summary.csv", index=False)
        step_two = condition_summary[condition_summary.power.eq(2)]
        atomic_json(
            output_dir / "summary.json",
            {
                "protocol": PROTOCOL,
                "input_artifact": str(input_dir),
                "samples": int(args.samples),
                "world_size": world_size,
                "precision": args.precision,
                "powers": int(args.powers),
                "control_scale": float(args.control_scale),
                "guided_scale": float(args.guided_scale),
                "second_step_smaller_than_first": bool(
                    (step_two.step_over_first_median < 1.0).all()
                ),
                "measurement_guardrail": (
                    "G includes decoder, clamp, encoder, and latent normalization. "
                    "Only pre/post-clamp directions share a pixel coordinate space; "
                    "cross-space cosine is intentionally not reported."
                ),
            },
        )
        print(condition_summary.to_string(index=False))
        print(pair_summary.to_string(index=False))
        print(f"wrote {output_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

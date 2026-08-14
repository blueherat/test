#!/usr/bin/env python3
"""Measure how nominal AutoGuidance transfers to off-nominal trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_static_pair import output_to_field_velocity
    from experiments.nominal_guidance_transfer import (
        nominal_transfer_metrics,
        sample_cosine,
        sample_rms,
    )
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_static_pair import output_to_field_velocity
    from nominal_guidance_transfer import (
        nominal_transfer_metrics,
        sample_cosine,
        sample_rms,
    )
    from sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ANCHOR = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_X800 = (
    BASE
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT = BASE / "nominal_guidance_transfer_800k_v1"
DEFAULT_TIMES = "0.02,0.05,0.1,0.2,0.4,0.6,0.8,0.9,0.95,0.97,0.99"
DEFAULT_ALPHAS = "0,0.25,0.5,0.75,1"
TRAJECTORIES = ("baseline", "frozen", "replay", "closed")


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed or not all(math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("expected finite comma-separated floats")
    return parsed


def _sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def _input_bank(num_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(num_samples, *LATENT_SHAPE, generator=generator)
    labels = torch.randint(NUM_CLASSES, (num_samples,), generator=generator)
    return noise, labels


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {name: float("nan") for name in ("mean", "std", "q05", "median", "q95")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


class FieldPair:
    """Evaluate strong and weak checkpoints with explicit paired labels."""

    def __init__(
        self,
        anchor_model: torch.nn.Module,
        other_model: torch.nn.Module,
        anchor_semantics,
        other_semantics,
    ) -> None:
        self.anchor_model = anchor_model
        self.other_model = other_model
        self.anchor_semantics = anchor_semantics
        self.other_semantics = other_semantics
        self.anchor_forwards = 0
        self.other_forwards = 0
        self.anchor_examples = 0
        self.other_examples = 0

    @staticmethod
    def _times(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return time_value.to(device=state.device, dtype=state.dtype).expand(len(state))

    def _evaluate(
        self,
        model: torch.nn.Module,
        semantics,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        times = self._times(time_value, state)
        output = model(state, times, labels)
        return output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=semantics,
        ).float()

    def anchor(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        self.anchor_forwards += 1
        self.anchor_examples += len(state)
        return self._evaluate(
            self.anchor_model,
            self.anchor_semantics,
            state,
            time_value,
            labels,
        )

    def other(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        self.other_forwards += 1
        self.other_examples += len(state)
        return self._evaluate(
            self.other_model,
            self.other_semantics,
            state,
            time_value,
            labels,
        )

    def both(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.anchor(state, time_value, labels), self.other(
            state,
            time_value,
            labels,
        )


def _load_pair(args: argparse.Namespace, device: torch.device):
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    anchor_model, anchor_semantics, anchor_meta, anchor_checkpoint = _load_field_model(
        checkpoint_path=args.anchor_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    other_model, other_semantics, other_meta, other_checkpoint = _load_field_model(
        checkpoint_path=args.other_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    assert anchor_model is not None and other_model is not None
    validate_pair_compatibility(
        anchor_checkpoint,
        other_checkpoint,
        anchor_meta,
        other_meta,
        allow_step_mismatch=args.allow_step_mismatch,
    )
    if anchor_semantics.prediction_target != "velocity":
        raise ValueError("the nominal strong field must use native velocity prediction")
    metadata = {
        "anchor": anchor_meta,
        "other": other_meta,
        "anchor_semantics": vars(anchor_semantics),
        "other_semantics": vars(other_semantics),
        "source": source_metadata,
    }
    return (
        FieldPair(anchor_model, other_model, anchor_semantics, other_semantics),
        metadata,
    )


def _integrate_trajectories(
    fields: FieldPair,
    noise: torch.Tensor,
    labels: torch.Tensor,
    output_times: torch.Tensor,
    *,
    gamma: float,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    batch_size = len(noise)
    initial = noise.repeat(4, 1, 1, 1)

    def derivative(time_value: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        baseline, frozen, replay, closed = combined.split(batch_size)
        anchor_states = torch.cat((baseline, frozen, closed))
        anchor_values = fields.anchor(
            anchor_states,
            time_value,
            labels.repeat(3),
        )
        anchor_baseline, anchor_frozen, anchor_closed = anchor_values.split(batch_size)
        other_values = fields.other(
            torch.cat((baseline, closed)),
            time_value,
            labels.repeat(2),
        )
        other_baseline, other_closed = other_values.split(batch_size)
        nominal_gap = anchor_baseline - other_baseline
        return torch.cat(
            (
                anchor_baseline,
                anchor_frozen + float(gamma) * nominal_gap,
                anchor_baseline + float(gamma) * nominal_gap,
                anchor_closed + float(gamma) * (anchor_closed - other_closed),
            )
        )

    trajectory = odeint(
        derivative,
        initial,
        output_times,
        method="dopri5",
        atol=float(atol),
        rtol=float(rtol),
    )
    return trajectory.reshape(
        len(output_times),
        len(TRAJECTORIES),
        batch_size,
        *noise.shape[1:],
    )


def _tensor_metrics_to_rows(
    metrics: dict[str, torch.Tensor],
    *,
    seed: int,
    sample_offset: int,
    time_value: float,
    relation: str,
) -> list[dict[str, object]]:
    cpu = {name: value.detach().cpu() for name, value in metrics.items()}
    rows: list[dict[str, object]] = []
    for local_index in range(len(next(iter(cpu.values())))):
        row: dict[str, object] = {
            "seed": int(seed),
            "sample_index": int(sample_offset + local_index),
            "time": float(time_value),
            "relation": relation,
        }
        for name, value in cpu.items():
            item = value[local_index].item()
            row[name] = bool(item) if value.dtype == torch.bool else float(item)
        rows.append(row)
    return rows


def _aggregate_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        groups.setdefault((float(row["time"]), str(row["relation"])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    identifiers = {"seed", "sample_index", "time", "relation", "valid"}
    for (time_value, relation), rows in sorted(groups.items()):
        valid_rows = [row for row in rows if bool(row.get("valid", True))]
        summary: dict[str, object] = {
            "time": time_value,
            "relation": relation,
            "samples": len(rows),
            "valid_rate": len(valid_rows) / len(rows),
        }
        metric_names = [name for name in rows[0] if name not in identifiers]
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in valid_rows])
            for statistic, value in _summary(values).items():
                summary[f"{name}_{statistic}"] = value
        summary_rows.append(summary)
    return summary_rows


def _diagnose_time(
    fields: FieldPair,
    states: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    sample_offset: int,
    segment_alphas: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    branch_count, batch_size = states.shape[:2]
    if branch_count != len(TRAJECTORIES):
        raise ValueError("unexpected trajectory count")
    flat_states = states.reshape(branch_count * batch_size, *states.shape[2:])
    anchor, other = fields.both(flat_states, time_value, labels.repeat(branch_count))
    anchor = anchor.reshape(branch_count, batch_size, *states.shape[2:])
    other = other.reshape(branch_count, batch_size, *states.shape[2:])
    gaps = anchor - other
    baseline_state = states[0]
    baseline_gap = gaps[0]

    rows: list[dict[str, object]] = []
    for branch_index, relation in enumerate(TRAJECTORIES[1:], start=1):
        state_shift = states[branch_index] - baseline_state
        metrics = nominal_transfer_metrics(
            baseline_gap,
            gaps[branch_index],
            state_shift=state_shift,
        )
        anchor_change = anchor[branch_index] - anchor[0]
        tiny = torch.finfo(anchor.dtype).tiny
        metrics.update(
            {
                "anchor_change_rms": sample_rms(anchor_change),
                "anchor_change_over_nominal_rms": sample_rms(anchor_change)
                / sample_rms(baseline_gap).clamp_min(tiny),
                "anchor_change_cosine_nominal": sample_cosine(
                    anchor_change,
                    baseline_gap,
                ),
            }
        )
        rows.extend(
            _tensor_metrics_to_rows(
                metrics,
                seed=seed,
                sample_offset=sample_offset,
                time_value=float(time_value.item()),
                relation=relation,
            )
        )

    frozen_shift = states[1] - baseline_state
    interior = [alpha for alpha in segment_alphas if alpha not in (0.0, 1.0)]
    interior_gaps: dict[float, torch.Tensor] = {}
    if interior:
        segment_states = torch.cat(
            [baseline_state + float(alpha) * frozen_shift for alpha in interior]
        )
        segment_anchor, segment_other = fields.both(
            segment_states,
            time_value,
            labels.repeat(len(interior)),
        )
        evaluated = (segment_anchor - segment_other).reshape(
            len(interior),
            batch_size,
            *states.shape[2:],
        )
        interior_gaps = dict(zip(interior, evaluated, strict=True))
    segment_rows: list[dict[str, object]] = []
    for alpha in segment_alphas:
        if alpha == 0.0:
            current_gap = baseline_gap
        elif alpha == 1.0:
            current_gap = gaps[1]
        else:
            current_gap = interior_gaps[alpha]
        metrics = nominal_transfer_metrics(
            baseline_gap,
            current_gap,
            state_shift=float(alpha) * frozen_shift,
        )
        segment_rows.extend(
            _tensor_metrics_to_rows(
                metrics,
                seed=seed,
                sample_offset=sample_offset,
                time_value=float(time_value.item()),
                relation=f"segment_alpha_{alpha:g}",
            )
        )
    return rows, segment_rows


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if args.num_samples % args.batch_size != 0:
        raise ValueError("num-samples must be divisible by batch-size")
    diagnostic_times = sorted(set(float(value) for value in args.times))
    if not diagnostic_times or diagnostic_times[0] <= 0 or diagnostic_times[-1] >= 1:
        raise ValueError("diagnostic times must lie strictly inside (0, 1)")
    if any(alpha < 0 or alpha > 1 for alpha in args.segment_alphas):
        raise ValueError("segment alphas must lie in [0, 1]")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_bank, label_bank = _input_bank(args.num_samples, args.seed)
    fields, metadata = _load_pair(args, device)
    output_times = torch.tensor(
        [0.0, *diagnostic_times, 1.0],
        device=device,
        dtype=torch.float32,
    )

    raw_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    endpoint_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in TRAJECTORIES}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, args.num_samples, args.batch_size):
            stop = start + args.batch_size
            noise = noise_bank[start:stop].to(device)
            labels = label_bank[start:stop].to(device)
            trajectory = _integrate_trajectories(
                fields,
                noise,
                labels,
                output_times,
                gamma=args.gamma,
                atol=args.atol,
                rtol=args.rtol,
            )
            for time_index, time_value in enumerate(output_times[1:-1], start=1):
                rows, current_segment_rows = _diagnose_time(
                    fields,
                    trajectory[time_index],
                    time_value,
                    labels,
                    seed=args.seed,
                    sample_offset=start,
                    segment_alphas=args.segment_alphas,
                )
                raw_rows.extend(rows)
                segment_rows.extend(current_segment_rows)
            endpoints = trajectory[-1].cpu()
            for branch_index, name in enumerate(TRAJECTORIES):
                endpoint_chunks[name].append(endpoints[branch_index])
            print(
                f"[{stop:04d}/{args.num_samples}] elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    endpoint_payload = {
        name: torch.cat(chunks) for name, chunks in endpoint_chunks.items()
    }
    endpoint_payload["labels"] = label_bank
    torch.save(endpoint_payload, output_dir / "endpoint_latents.pt")
    _write_csv(raw_rows, output_dir / "nominal_transfer_per_sample.csv")
    _write_csv(_aggregate_rows(raw_rows), output_dir / "nominal_transfer_by_time.csv")
    _write_csv(segment_rows, output_dir / "segment_transfer_per_sample.csv")
    _write_csv(_aggregate_rows(segment_rows), output_dir / "segment_transfer_by_time.csv")
    manifest = {
        "format": "eqvae_imagenet100_sit_nominal_transfer_geometry_v1",
        "formula": {
            "baseline": "z'=S(z,t)",
            "frozen": "z'=S(z,t)+gamma*g(z_baseline,t)",
            "replay": "z'=S(z_baseline,t)+gamma*g(z_baseline,t)",
            "closed": "z'=S(z,t)+gamma*g(z,t)",
            "gap": "g=S-W",
        },
        "anchor_checkpoint": str(args.anchor_checkpoint.expanduser().resolve()),
        "other_checkpoint": str(args.other_checkpoint.expanduser().resolve()),
        "weights": args.weights,
        "gamma": float(args.gamma),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "times": diagnostic_times,
        "segment_alphas": [float(value) for value in args.segment_alphas],
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "allow_tf32": bool(args.allow_tf32),
        "noise_sha256": _sha256(noise_bank),
        "label_sha256": _sha256(label_bank),
        "elapsed_seconds": time.perf_counter() - started,
        "anchor_forwards": fields.anchor_forwards,
        "other_forwards": fields.other_forwards,
        "anchor_examples": fields.anchor_examples,
        "other_examples": fields.other_examples,
        "metadata": metadata,
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--times", type=_parse_floats, default=_parse_floats(DEFAULT_TIMES))
    parser.add_argument(
        "--segment-alphas",
        type=_parse_floats,
        default=_parse_floats(DEFAULT_ALPHAS),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "x800_seed0")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

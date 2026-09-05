#!/usr/bin/env python3
"""Paired distributed sampling for RAEv2 exponential-retiming PFR.

The sampler preserves the official EMA checkpoint, shifted 100-step Euler
grid, decoder, labels, and initial noise.  PFR adds one cheap base-head prefix
query at a dataward time while ordinary full/IG conditions remain exact
anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.raev2_pfr_retiming import (  # noqa: E402
    bridge_latentized_counterfactual_state,
    clean_to_velocity,
    dataward_future_time,
    euler_dataward_query_state,
    evaluate_base_head_only,
    orthogonal_counterfactual_guidance_clean,
    pfr_velocity,
    norm_preserving_certificate_revision,
    project_revision_onto_certificate,
    raev2_ou_degree1_velocity_defect,
    shared_retiming_revision,
    strong_anchored_counterfactual_guidance_clean,
    transport_raev2_state_at_fixed_ou_coordinate,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_pfr_exponential_retiming_v1"
PFR_DATA_TIME_CUTOFF = 0.5
PFR_EARLY_NOISE_TIME_CUTOFF = 0.75
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
    "dinov3l-k7/checkpoint.pt"
)


@dataclass(frozen=True)
class SamplingCondition:
    name: str
    guidance_scale: float
    horizon: float
    revision_scale: float
    method: str = "retiming"
    mid_guidance_scale: float | None = None
    guidance_min_time: float | None = None
    guidance_max_time: float | None = None

    def validate(self) -> None:
        if not self.name or any(char in self.name for char in "/=,:"):
            raise ValueError("condition name must be a safe path component")
        for label, value in (
            ("guidance_scale", self.guidance_scale),
            ("horizon", self.horizon),
            ("revision_scale", self.revision_scale),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.guidance_scale < 0.0 or self.horizon < 0.0:
            raise ValueError("guidance scale and horizon must be non-negative")
        if self.mid_guidance_scale is not None and (
            not math.isfinite(self.mid_guidance_scale)
            or self.mid_guidance_scale < 0.0
        ):
            raise ValueError("mid guidance scale must be finite and non-negative")
        window_values = (self.guidance_min_time, self.guidance_max_time)
        if (window_values[0] is None) != (window_values[1] is None):
            raise ValueError("guidance window requires both minimum and maximum time")
        if window_values[0] is not None:
            minimum, maximum = window_values
            assert minimum is not None and maximum is not None
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("guidance window times must be finite")
            if not 0.0 <= minimum <= maximum <= 1.0:
                raise ValueError("guidance window must satisfy 0 <= min <= max <= 1")
            if self.mid_guidance_scale is not None:
                raise ValueError("piecewise and window guidance cannot be combined")
        if self.method not in {
            "retiming",
            "shared_retiming",
            "ou_common_retiming",
            "ou_polar_retiming",
            "ou_weak_common_retiming",
            "ou_weak_polar_retiming",
            "ou_polar_first_half_retiming",
            "pathwise_first_half_retiming",
            "ou_polar_pathwise_first_half_retiming",
            "bridge_counterfactual",
            "bridge_counterfactual_orthogonal",
        }:
            raise ValueError(f"unknown sampling method: {self.method}")
        if (
            self.method in {"retiming", "shared_retiming"}
            and self.horizon == 0.0
            and self.revision_scale != 0.0
        ):
            raise ValueError("a nonzero revision requires a positive horizon")
        if self.uses_bridge_counterfactual and (
            self.horizon != 0.0 or self.revision_scale != 0.0
        ):
            raise ValueError(
                "bridge_counterfactual has no horizon or revision-scale parameter"
            )

    @property
    def uses_retiming(self) -> bool:
        return (
            self.method
            in {
                "retiming",
                "shared_retiming",
                "ou_common_retiming",
                "ou_polar_retiming",
                "ou_weak_common_retiming",
                "ou_weak_polar_retiming",
                "ou_polar_first_half_retiming",
                "pathwise_first_half_retiming",
                "ou_polar_pathwise_first_half_retiming",
            }
            and self.horizon > 0.0
            and self.revision_scale != 0.0
        )

    @property
    def uses_shared_retiming(self) -> bool:
        return self.method == "shared_retiming" and self.uses_retiming

    @property
    def uses_ou_certificate(self) -> bool:
        return self.method in {
            "ou_common_retiming",
            "ou_polar_retiming",
            "ou_weak_common_retiming",
            "ou_weak_polar_retiming",
            "ou_polar_first_half_retiming",
            "ou_polar_pathwise_first_half_retiming",
        } and self.uses_retiming

    @property
    def uses_weak_ou_certificate(self) -> bool:
        return self.method in {
            "ou_weak_common_retiming",
            "ou_weak_polar_retiming",
        } and self.uses_retiming

    @property
    def uses_polar_ou_certificate(self) -> bool:
        return self.method in {
            "ou_polar_retiming",
            "ou_weak_polar_retiming",
            "ou_polar_first_half_retiming",
            "ou_polar_pathwise_first_half_retiming",
        } and self.uses_retiming

    @property
    def uses_first_half_revision(self) -> bool:
        """Whether PFR is restricted to data time ``u=1-t < 0.5``."""

        return self.method in {
            "ou_polar_first_half_retiming",
            "pathwise_first_half_retiming",
            "ou_polar_pathwise_first_half_retiming",
        }

    @property
    def uses_pathwise_retiming(self) -> bool:
        return self.method in {
            "pathwise_first_half_retiming",
            "ou_polar_pathwise_first_half_retiming",
        } and self.uses_retiming

    def revision_is_active(self, noise_time: float) -> bool:
        if not self.uses_retiming:
            return False
        return (
            noise_time > 1.0 - PFR_DATA_TIME_CUTOFF
            if self.uses_first_half_revision
            else True
        )

    def revision_future_floor(self, ig_minimum_time: float) -> float:
        """Keep a foresight query inside its registered intervention window."""

        if self.uses_first_half_revision:
            return max(ig_minimum_time, 1.0 - PFR_DATA_TIME_CUTOFF)
        return ig_minimum_time

    @property
    def uses_bridge_counterfactual(self) -> bool:
        return self.method in {
            "bridge_counterfactual",
            "bridge_counterfactual_orthogonal",
        }

    @property
    def uses_piecewise_guidance(self) -> bool:
        return self.mid_guidance_scale is not None

    @property
    def uses_window_guidance(self) -> bool:
        return self.guidance_min_time is not None

    def guidance_scale_at(self, noise_time: float) -> float:
        """Return the IG scale for early, middle, and data-side intervals.

        RAE sampling runs from noise time 1 to data time 0.  A piecewise
        condition uses ``guidance_scale`` above 0.75,
        ``mid_guidance_scale`` on (0.5, 0.75], and the strong head alone on
        [0, 0.5].  Scalar conditions preserve the historical behavior.
        """

        if self.uses_window_guidance:
            assert self.guidance_min_time is not None
            assert self.guidance_max_time is not None
            if self.guidance_min_time <= noise_time <= self.guidance_max_time:
                return self.guidance_scale
            return 1.0
        if self.mid_guidance_scale is None:
            return self.guidance_scale
        if noise_time > PFR_EARLY_NOISE_TIME_CUTOFF:
            return self.guidance_scale
        if noise_time > PFR_DATA_TIME_CUTOFF:
            return self.mid_guidance_scale
        return 1.0


DEFAULT_CONDITIONS = (
    SamplingCondition("full", 1.0, 0.0, 0.0),
    SamplingCondition("ig_s1p78", 1.78, 0.0, 0.0),
    SamplingCondition("pfr_h0p015625_r0p5", 1.78, 1.0 / 64.0, 0.5),
    SamplingCondition("pfr_h0p015625_r1", 1.78, 1.0 / 64.0, 1.0),
    SamplingCondition("pfr_h0p03125_r0p5", 1.78, 1.0 / 32.0, 0.5),
    SamplingCondition("pfr_h0p03125_r1", 1.78, 1.0 / 32.0, 1.0),
    SamplingCondition("pfr_h0p03125_rm1", 1.78, 1.0 / 32.0, -1.0),
)


def parse_condition(text: str) -> SamplingCondition:
    parts = text.split(",")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "condition must be NAME,GUIDANCE_SCALE,HORIZON,REVISION_SCALE"
            "[,METHOD]"
        )
    try:
        condition = SamplingCondition(
            name=parts[0],
            guidance_scale=float(parts[1]),
            horizon=float(parts[2]),
            revision_scale=float(parts[3]),
            method=parts[4] if len(parts) == 5 else "retiming",
        )
        condition.validate()
        return condition
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_piecewise_condition(text: str) -> SamplingCondition:
    parts = text.split(",")
    if len(parts) not in {3, 5, 6}:
        raise argparse.ArgumentTypeError(
            "piecewise condition must be NAME,EARLY_SCALE,MID_SCALE"
            "[,HORIZON,REVISION_SCALE[,METHOD]]"
        )
    try:
        condition = SamplingCondition(
            name=parts[0],
            guidance_scale=float(parts[1]),
            horizon=float(parts[3]) if len(parts) >= 5 else 0.0,
            revision_scale=float(parts[4]) if len(parts) >= 5 else 0.0,
            method=parts[5] if len(parts) == 6 else "retiming",
            mid_guidance_scale=float(parts[2]),
        )
        condition.validate()
        return condition
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_window_condition(text: str) -> SamplingCondition:
    parts = text.split(",")
    if len(parts) not in {4, 6, 7}:
        raise argparse.ArgumentTypeError(
            "window condition must be NAME,GUIDANCE_SCALE,T_MIN,T_MAX"
            "[,HORIZON,REVISION_SCALE[,METHOD]]"
        )
    try:
        condition = SamplingCondition(
            name=parts[0],
            guidance_scale=float(parts[1]),
            horizon=float(parts[4]) if len(parts) >= 6 else 0.0,
            revision_scale=float(parts[5]) if len(parts) >= 6 else 0.0,
            method=parts[6] if len(parts) == 7 else "retiming",
            guidance_min_time=float(parts[2]),
            guidance_max_time=float(parts[3]),
        )
        condition.validate()
        return condition
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    config.prepare_model_params()
    return config


def shifted_time_grid(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return shift * base / (1.0 + (shift - 1.0) * base)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def generator_sha256(generator: torch.Generator) -> str:
    return tensor_sha256(generator.get_state())


def sample_condition(
    *,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    condition: SamplingCondition,
    config: Stage2Config,
    time_grid: torch.Tensor,
    global_ids: np.ndarray,
    per_rank_batch: int,
    sampling_seed: int,
    precision: str,
    output_dir: Path,
    rank: int,
    world_size: int,
    horizon_coordinate: str = "raw_time",
    revision_composition: str = "additive",
) -> dict[str, object]:
    condition.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=time_grid.device)
    generator.manual_seed(int(sampling_seed) * world_size + rank)
    initial_rng = generator_sha256(generator)
    images_local: list[np.ndarray] = []
    preview = None
    first_noise_sha256 = None
    first_label_sha256 = None
    full_calls = 0
    prefix_calls = 0
    started = time.perf_counter()
    t_floor = float(config.transport.t_eps)
    ig_min = float(config.guidance.ig.t_min)
    ig_max = float(config.guidance.ig.t_max)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for start in range(0, len(global_ids), per_rank_batch):
            ids = global_ids[start : start + per_rank_batch]
            batch_size = len(ids)
            state = torch.randn(
                batch_size,
                *config.misc.latent_size,
                generator=generator,
                device=time_grid.device,
                dtype=torch.float32,
            )
            labels = torch.from_numpy(ids % 1000).to(
                device=time_grid.device, dtype=torch.long
            )
            if first_noise_sha256 is None:
                first_noise_sha256 = tensor_sha256(state)
                first_label_sha256 = tensor_sha256(labels)
            model_kwargs = {"context": labels, "attn_mask": None}

            for index in range(len(time_grid) - 1):
                current = float(time_grid[index].item())
                following = float(time_grid[index + 1].item())
                step = current - following
                times = torch.full(
                    (batch_size,), current, device=state.device, dtype=torch.float32
                )
                full_clean, base_clean = model(state, times, **model_kwargs)
                full_calls += 1
                full_velocity = clean_to_velocity(
                    full_clean,
                    state,
                    times,
                    denominator_floor=t_floor,
                )

                active = ig_min <= current <= ig_max
                guidance_scale = condition.guidance_scale_at(current)
                if not active or guidance_scale == 1.0:
                    drift = full_velocity
                elif condition.uses_bridge_counterfactual:
                    counterfactual_state = bridge_latentized_counterfactual_state(
                        state,
                        full_clean,
                        base_clean,
                        times,
                        guidance_scale=guidance_scale,
                    )
                    counterfactual_base_clean = evaluate_base_head_only(
                        model,
                        counterfactual_state,
                        times,
                        **model_kwargs,
                    )
                    prefix_calls += 1
                    if condition.method == "bridge_counterfactual_orthogonal":
                        guided_clean = orthogonal_counterfactual_guidance_clean(
                            full_clean,
                            base_clean,
                            counterfactual_base_clean,
                            guidance_scale=guidance_scale,
                        )
                    else:
                        guided_clean = strong_anchored_counterfactual_guidance_clean(
                            full_clean,
                            counterfactual_base_clean,
                            guidance_scale=guidance_scale,
                        )
                    drift = clean_to_velocity(
                        guided_clean,
                        state,
                        times,
                        denominator_floor=t_floor,
                    )
                else:
                    base_velocity = clean_to_velocity(
                        base_clean,
                        state,
                        times,
                        denominator_floor=t_floor,
                    )
                    ordinary_guided_velocity = base_velocity + (
                        guidance_scale * (full_velocity - base_velocity)
                    )
                    if condition.revision_is_active(current):
                        future = dataward_future_time(
                            current,
                            condition.horizon,
                            coordinate=horizon_coordinate,
                            minimum_time=condition.revision_future_floor(ig_min),
                        )
                        future_times = torch.full_like(times, future)
                        future_query_state = (
                            euler_dataward_query_state(
                                state,
                                ordinary_guided_velocity,
                                times,
                                future_times,
                            )
                            if condition.uses_pathwise_retiming
                            else state
                        )
                        use_ou_certificate = (
                            condition.uses_ou_certificate and current > 0.75
                        )
                        if use_ou_certificate:
                            base_future_clean = evaluate_base_head_only(
                                model,
                                future_query_state,
                                future_times,
                                **model_kwargs,
                            )
                            prefix_calls += 1
                            base_future_velocity = clean_to_velocity(
                                base_future_clean,
                                future_query_state,
                                future_times,
                                denominator_floor=t_floor,
                            )
                            future_ou_state = (
                                transport_raev2_state_at_fixed_ou_coordinate(
                                    state, times, future_times
                                )
                            )
                            if condition.uses_weak_ou_certificate:
                                base_future_ou_clean = evaluate_base_head_only(
                                    model,
                                    future_ou_state,
                                    future_times,
                                    **model_kwargs,
                                )
                                prefix_calls += 1
                                base_future_ou_velocity = clean_to_velocity(
                                    base_future_ou_clean,
                                    future_ou_state,
                                    future_times,
                                    denominator_floor=t_floor,
                                )
                                certificate = raev2_ou_degree1_velocity_defect(
                                    base_velocity,
                                    base_future_ou_velocity,
                                    state,
                                    times,
                                    future_times,
                                )
                            else:
                                full_future_clean, _ = model(
                                    future_ou_state, future_times, **model_kwargs
                                )
                                full_calls += 1
                                full_future_velocity = clean_to_velocity(
                                    full_future_clean,
                                    future_ou_state,
                                    future_times,
                                    denominator_floor=t_floor,
                                )
                                certificate = raev2_ou_degree1_velocity_defect(
                                    full_velocity,
                                    full_future_velocity,
                                    state,
                                    times,
                                    future_times,
                                )
                            raw_revision = base_velocity - base_future_velocity
                            if condition.uses_polar_ou_certificate:
                                revision = norm_preserving_certificate_revision(
                                    raw_revision, certificate
                                )
                            else:
                                revision = project_revision_onto_certificate(
                                    raw_revision, certificate
                                )
                            future_velocity = base_velocity - revision
                        elif condition.uses_shared_retiming:
                            full_future_clean, base_future_clean = model(
                                state, future_times, **model_kwargs
                            )
                            full_calls += 1
                            full_future_velocity = clean_to_velocity(
                                full_future_clean,
                                state,
                                future_times,
                                denominator_floor=t_floor,
                            )
                            base_future_velocity = clean_to_velocity(
                                base_future_clean,
                                state,
                                future_times,
                                denominator_floor=t_floor,
                            )
                            revision = shared_retiming_revision(
                                base_velocity,
                                base_future_velocity,
                                full_velocity,
                                full_future_velocity,
                            )
                            future_velocity = base_velocity - revision
                        else:
                            future_clean = evaluate_base_head_only(
                                model,
                                future_query_state,
                                future_times,
                                **model_kwargs,
                            )
                            prefix_calls += 1
                            future_velocity = clean_to_velocity(
                                future_clean,
                                future_query_state,
                                future_times,
                                denominator_floor=t_floor,
                            )
                    else:
                        future_velocity = base_velocity
                    drift = pfr_velocity(
                        full_velocity,
                        base_velocity,
                        future_velocity,
                        guidance_scale=guidance_scale,
                        revision_scale=condition.revision_scale,
                        composition=revision_composition,
                    )
                state = state - step * drift

            decoded = decoder.decode(state).clamp(0, 1)
            if preview is None:
                preview = decoded[: min(16, len(decoded))].float().cpu()
            images_local.append(
                decoded.mul(255)
                .permute(0, 2, 3, 1)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )

    elapsed = time.perf_counter() - started
    images = np.concatenate(images_local, axis=0)
    np.save(output_dir / f"images-rank{rank:02d}.npy", images)
    np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
    if preview is not None:
        save_image(preview, output_dir / f"preview-rank{rank:02d}.png", nrow=4)
    audit = {
        "protocol": PROTOCOL,
        "condition": asdict(condition),
        "rank": rank,
        "world_size": world_size,
        "sampling_seed": sampling_seed,
        "sample_count": int(len(global_ids)),
        "per_rank_batch": per_rank_batch,
        "precision": precision,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "horizon_coordinate": horizon_coordinate,
        "revision_composition": revision_composition,
        "initial_generator_sha256": initial_rng,
        "final_generator_sha256": generator_sha256(generator),
        "first_noise_sha256": first_noise_sha256,
        "first_label_sha256": first_label_sha256,
        "full_model_calls": full_calls,
        "base_prefix_calls": prefix_calls,
        "elapsed_seconds": elapsed,
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(time_grid.device)
        ),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(time_grid.device)
        ),
    }
    (output_dir / f"sampling_audit_rank{rank}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override sampler.num_steps; omitted preserves the official config.",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--horizon-coordinate",
        choices=("raw_time", "log_odds"),
        default="raw_time",
    )
    parser.add_argument(
        "--revision-composition",
        choices=(
            "additive",
            "norm_preserving",
            "orthogonal_norm_preserving",
            "strong_anchored_additive",
            "strong_anchored_norm_preserving",
            "strong_anchored_angular",
        ),
        default="additive",
    )
    parser.add_argument(
        "--condition", action="append", type=parse_condition, dest="conditions"
    )
    parser.add_argument(
        "--piecewise-condition",
        action="append",
        type=parse_piecewise_condition,
        dest="conditions",
    )
    parser.add_argument(
        "--window-condition",
        action="append",
        type=parse_window_condition,
        dest="conditions",
    )
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    conditions = tuple(args.conditions or DEFAULT_CONDITIONS)
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("condition names must be unique")
    if args.sample_count <= 0:
        raise ValueError("sample count must be positive")
    if args.per_rank_batch <= 0:
        raise ValueError("per-rank batch must be positive")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("number of sampling steps must be positive")
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("RAEv2 config or checkpoint is missing")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    config = load_config(args.config)
    decoder = instantiate_from_config(config.stage_1).to(device).eval()
    decoder.requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    num_steps = int(
        config.sampler.num_steps if args.num_steps is None else args.num_steps
    )
    time_grid = shifted_time_grid(num_steps, shift, device)
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        request = {
            "protocol": PROTOCOL,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "state_key": args.state_key,
            "conditions": [asdict(condition) for condition in conditions],
            "sample_count": args.sample_count,
            "per_rank_batch": args.per_rank_batch,
            "sampling_seed": args.sampling_seed,
            "precision": args.precision,
            "allow_tf32": allow_tf32,
            "horizon_coordinate": args.horizon_coordinate,
            "revision_composition": args.revision_composition,
            "world_size": world_size,
            "sampler_steps": num_steps,
            "time_shift": shift,
            "ig_interval": [
                float(config.guidance.ig.t_min),
                float(config.guidance.ig.t_max),
            ],
            "transport_prediction": str(config.transport.prediction),
            "transport_t_eps": float(config.transport.t_eps),
        }
        (args.results_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    dist.barrier()

    for condition in conditions:
        output = args.results_dir / condition.name
        sample_condition(
            model=model,
            decoder=decoder,
            condition=condition,
            config=config,
            time_grid=time_grid,
            global_ids=global_ids,
            per_rank_batch=args.per_rank_batch,
            sampling_seed=args.sampling_seed,
            precision=args.precision,
            output_dir=output,
            rank=rank,
            world_size=world_size,
            horizon_coordinate=args.horizon_coordinate,
            revision_composition=args.revision_composition,
        )
        dist.barrier()
        if rank == 0:
            all_ids = []
            all_images = []
            audits = []
            for shard in range(world_size):
                all_ids.append(np.load(output / f"ids-rank{shard:02d}.npy"))
                all_images.append(np.load(output / f"images-rank{shard:02d}.npy"))
                audits.append(
                    json.loads(
                        (output / f"sampling_audit_rank{shard}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
            ids = np.concatenate(all_ids)
            images = np.concatenate(all_images)
            order = np.argsort(ids)
            ids = ids[order]
            images = images[order]
            if not np.array_equal(ids, np.arange(args.sample_count)):
                raise RuntimeError("distributed sample IDs are incomplete or duplicated")
            archive = output / "samples.npz"
            np.savez(archive, images)
            summary = {
                "protocol": PROTOCOL,
                "condition": asdict(condition),
                "samples": int(len(images)),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
                "checkpoint_step": checkpoint_step,
                "state_key": args.state_key,
                "total_full_model_calls": sum(
                    int(audit["full_model_calls"]) for audit in audits
                ),
                "total_base_prefix_calls": sum(
                    int(audit["base_prefix_calls"]) for audit in audits
                ),
                "max_rank_elapsed_seconds": max(
                    float(audit["elapsed_seconds"]) for audit in audits
                ),
            }
            (output / "sampling_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for shard in range(world_size):
                (output / f"ids-rank{shard:02d}.npy").unlink()
                (output / f"images-rank{shard:02d}.npy").unlink()
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

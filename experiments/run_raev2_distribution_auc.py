"""Test whether RAEv2 internal guidance moves sampler states toward p_t.

The experiment leaves the official RAEv2 model and sampler untouched.  It
captures sampler inputs at selected solver times and compares them with

    p_t = (1 - t) * E(x) + t * epsilon

using a held-out diagonal-LDA linear probe over the full latent tensor.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    validate_full_stage2_checkpoint,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_TIMES = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--packed-data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument(
        "--parquet-data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--time", action="append", type=float, dest="times")
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    config.prepare_model_params()
    if config.transport.prediction != "x":
        raise ValueError("distribution AUC audit requires clean-latent prediction")
    return config


def shifted_solver_grid(num_steps: int, shift: float) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if shift <= 0:
        raise ValueError("time shift must be positive")
    grid = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def match_requested_times(
    requested: tuple[float, ...], grid: torch.Tensor
) -> list[dict[str, float | int]]:
    if not requested:
        raise ValueError("at least one requested time is required")
    matched: list[dict[str, float | int]] = []
    seen_indices: set[int] = set()
    usable = grid[:-1]
    for value in requested:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"requested time must be in (0, 1], got {value}")
        index = int(torch.argmin(torch.abs(usable - float(value))).item())
        if index in seen_indices:
            raise ValueError(
                f"multiple requested times map to solver index {index}; use distinct times"
            )
        seen_indices.add(index)
        actual = float(usable[index].item())
        matched.append(
            {
                "requested_time": float(value),
                "solver_index": index,
                "actual_time": actual,
                "absolute_time_error": abs(actual - float(value)),
            }
        )
    return sorted(matched, key=lambda row: float(row["requested_time"]))


def build_requested_labels(sample_count: int, num_classes: int) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if num_classes <= 1:
        raise ValueError("num_classes must exceed one")
    return np.arange(sample_count, dtype=np.int64) % int(num_classes)


def class_group_split(
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> np.ndarray:
    """Return a test mask while keeping every class wholly in one split."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("at least two represented classes are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    test_classes = max(1, min(unique.size - 1, round(unique.size * test_fraction)))
    return np.isin(labels, shuffled[:test_classes])


def _parquet_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    data_dir = root / "data" if (root / "data").is_dir() else root
    files = sorted(data_dir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet shards under {data_dir}")
    return files


def select_matching_imagenet_rows(
    parquet_root: Path,
    requested_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Choose distinct real ImageNet rows with exactly the requested labels."""

    label_parts = []
    for path in _parquet_files(parquet_root):
        label_parts.append(
            np.asarray(
                pq.read_table(path, columns=["label"]).column("label").to_numpy(),
                dtype=np.int64,
            )
        )
    all_labels = np.concatenate(label_parts)
    order = np.argsort(all_labels, kind="stable")
    counts = np.bincount(all_labels, minlength=int(requested_labels.max()) + 1)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    needed = np.bincount(requested_labels, minlength=counts.size)
    if np.any(needed > counts):
        missing = np.flatnonzero(needed > counts)
        raise ValueError(f"not enough ImageNet rows for labels {missing.tolist()}")

    selected_by_class: dict[int, list[int]] = {}
    for label in np.flatnonzero(needed):
        candidates = order[offsets[label] : offsets[label + 1]].copy()
        rng = np.random.default_rng(int(seed) + 104729 * int(label))
        rng.shuffle(candidates)
        selected_by_class[int(label)] = candidates[: needed[label]].tolist()

    cursors = {label: 0 for label in selected_by_class}
    selected = np.empty(requested_labels.size, dtype=np.int64)
    for index, label_value in enumerate(requested_labels.tolist()):
        label = int(label_value)
        selected[index] = selected_by_class[label][cursors[label]]
        cursors[label] += 1
    if np.unique(selected).size != selected.size:
        raise RuntimeError("real ImageNet row selection contains duplicates")
    return selected


@dataclass
class MomentAccumulator:
    count: int = 0
    total: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        flat = values.detach().float().flatten(1)
        batch_total = flat.sum(dim=0).cpu().double()
        batch_total_sq = flat.square().sum(dim=0).cpu().double()
        if self.total is None:
            self.total = batch_total
            self.total_sq = batch_total_sq
        else:
            self.total.add_(batch_total)
            assert self.total_sq is not None
            self.total_sq.add_(batch_total_sq)
        self.count += int(flat.shape[0])

    def reduced(
        self,
        device: torch.device,
        feature_count: int | None = None,
    ) -> "MomentAccumulator":
        if self.total is None or self.total_sq is None:
            if feature_count is None or feature_count <= 0:
                raise RuntimeError(
                    "feature_count is required to reduce an empty local accumulator"
                )
            local_total = torch.zeros(feature_count, dtype=torch.float64)
            local_total_sq = torch.zeros(feature_count, dtype=torch.float64)
        else:
            local_total = self.total
            local_total_sq = self.total_sq
        count = torch.tensor([self.count], device=device, dtype=torch.float64)
        total = local_total.to(device=device)
        total_sq = local_total_sq.to(device=device)
        if dist.is_initialized():
            dist.all_reduce(count)
            dist.all_reduce(total)
            dist.all_reduce(total_sq)
        return MomentAccumulator(
            count=int(count.item()),
            total=total.cpu(),
            total_sq=total_sq.cpu(),
        )

    def mean_variance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2 or self.total is None or self.total_sq is None:
            raise RuntimeError("at least two samples are required")
        mean = self.total / self.count
        centered = self.total_sq - self.total.square() / self.count
        variance = (centered / (self.count - 1)).clamp_min_(0.0)
        return mean, variance


@dataclass
class HeldoutStates:
    states: dict[float, list[torch.Tensor]] = field(default_factory=dict)
    ids: dict[float, list[np.ndarray]] = field(default_factory=dict)
    sample_shape: tuple[int, ...] | None = None

    def add(self, time_key: float, values: torch.Tensor, sample_ids: np.ndarray) -> None:
        if values.shape[0] != sample_ids.size:
            raise ValueError("state and ID counts differ")
        shape = tuple(values.shape[1:])
        if self.sample_shape is None:
            self.sample_shape = shape
        elif self.sample_shape != shape:
            raise ValueError("held-out state shape changed")
        if values.shape[0] == 0:
            return
        self.states.setdefault(time_key, []).append(
            values.detach().to(device="cpu", dtype=torch.float16).contiguous()
        )
        self.ids.setdefault(time_key, []).append(sample_ids.astype(np.int64, copy=True))

    def tensors(self, time_key: float) -> tuple[torch.Tensor, np.ndarray]:
        if time_key not in self.states:
            if self.sample_shape is None:
                raise KeyError(time_key)
            return (
                torch.empty((0, *self.sample_shape), dtype=torch.float16),
                np.empty(0, dtype=np.int64),
            )
        return (
            torch.cat(self.states[time_key], dim=0),
            np.concatenate(self.ids[time_key]),
        )


def fit_diagonal_lda(
    negative: MomentAccumulator,
    positive: MomentAccumulator,
    ridge_ratio: float,
) -> tuple[torch.Tensor, float, float]:
    if ridge_ratio <= 0:
        raise ValueError("ridge_ratio must be positive")
    mean_neg, var_neg = negative.mean_variance()
    mean_pos, var_pos = positive.mean_variance()
    degrees = negative.count + positive.count - 2
    pooled = (
        (negative.count - 1) * var_neg + (positive.count - 1) * var_pos
    ) / degrees
    positive_scale = pooled[pooled > 0]
    base_scale = (
        float(positive_scale.median().item()) if positive_scale.numel() else 1.0
    )
    ridge = max(float(ridge_ratio) * base_scale, 1e-12)
    weight = (mean_pos - mean_neg) / (pooled + ridge)
    weight = weight.float()
    weight.div_(weight.norm().clamp_min(1e-30))
    intercept = -0.5 * torch.dot(
        weight.double(), (mean_pos + mean_neg)
    ).item()
    return weight, float(intercept), ridge


def score_states(
    states: torch.Tensor,
    weight: torch.Tensor,
    intercept: float,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    if states.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    weight_device = weight.to(device=device)
    scores = []
    for start in range(0, states.shape[0], batch_size):
        batch = states[start : start + batch_size].to(device=device, dtype=torch.float32)
        score = batch.flatten(1).matmul(weight_device).add_(float(intercept))
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def paired_auc(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    if negative_scores.shape != positive_scores.shape:
        raise ValueError("paired score arrays must have identical shapes")
    labels = np.concatenate(
        [np.zeros(negative_scores.size, dtype=np.int8), np.ones(positive_scores.size, dtype=np.int8)]
    )
    return float(roc_auc_score(labels, np.concatenate([negative_scores, positive_scores])))


def bootstrap_paired_auc(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, negative_scores.size, size=negative_scores.size)
        values[repeat] = paired_auc(negative_scores[indices], positive_scores[indices])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def bootstrap_auc_delta(
    p_full_scores: np.ndarray,
    full_scores: np.ndarray,
    p_ig_scores: np.ndarray,
    ig_scores: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(
            0, p_full_scores.size, size=p_full_scores.size
        )
        values[repeat] = paired_auc(
            p_ig_scores[indices], ig_scores[indices]
        ) - paired_auc(p_full_scores[indices], full_scores[indices])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def screening_conclusion(deltas: pd.DataFrame, heldout_pairs: int) -> str:
    if heldout_pairs < 100:
        return "insufficient held-out pairs for a mechanism conclusion"
    non_null = deltas[deltas["requested_time"] < 1.0]
    closer = int((non_null["delta_ci_high"] < 0).sum())
    farther = int((non_null["delta_ci_low"] > 0).sum())
    if closer and farther:
        return (
            "phase-dependent sign reversal: IG is closer to p_t at some times "
            "and farther at others"
        )
    majority = max(1, math.ceil(len(non_null) / 2))
    if closer >= majority:
        return "IG is linearly closer to p_t at a majority of non-null times"
    if farther >= majority:
        return "IG is linearly farther from p_t at a majority of non-null times"
    return "the linear-probe result is inconclusive"


class SamplerStateRecorder:
    """Record inputs to the unmodified official model function at chosen steps."""

    def __init__(
        self,
        model_fn: Callable[..., torch.Tensor],
        matched_times: list[dict[str, float | int]],
        real_batch_size: int,
        callback: Callable[[float, torch.Tensor], None],
    ) -> None:
        self.model_fn = model_fn
        self.by_index = {
            int(item["solver_index"]): float(item["requested_time"])
            for item in matched_times
        }
        self.actual_by_index = {
            int(item["solver_index"]): float(item["actual_time"])
            for item in matched_times
        }
        self.real_batch_size = int(real_batch_size)
        self.callback = callback
        self.calls = 0
        self.captured: set[int] = set()

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        index = self.calls
        if index in self.by_index:
            actual = float(t[0].item())
            expected = self.actual_by_index[index]
            if abs(actual - expected) > 1e-6:
                raise RuntimeError(
                    f"sampler time mismatch at step {index}: {actual} != {expected}"
                )
            if not torch.allclose(t, t[:1].expand_as(t), rtol=0.0, atol=0.0):
                raise RuntimeError("sampler batch contains inconsistent times")
            self.callback(self.by_index[index], x[: self.real_batch_size])
            self.captured.add(index)
        self.calls += 1
        return self.model_fn(x, t, **kwargs)

    def validate(self, expected_calls: int) -> None:
        if self.calls != expected_calls:
            raise RuntimeError(f"sampler made {self.calls} calls, expected {expected_calls}")
        if self.captured != set(self.by_index):
            raise RuntimeError("not all requested sampler states were captured")


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def gather_rank_payload(payload: Any) -> list[Any]:
    if not dist.is_initialized():
        return [payload]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return gathered


def concatenate_rank_scores(
    payloads: list[dict[str, Any]],
    time_key: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.concatenate([item["ids"][time_key] for item in payloads])
    p_scores = np.concatenate([item["p_scores"][time_key] for item in payloads])
    q_scores = np.concatenate([item["q_scores"][time_key] for item in payloads])
    if np.unique(ids).size != ids.size:
        raise RuntimeError("held-out sample IDs are duplicated across ranks")
    order = np.argsort(ids)
    return ids[order], p_scores[order], q_scores[order]


def _branch_states(
    *,
    branch: str,
    ig_scale: float,
    model: torch.nn.Module,
    sample_fn: Callable[..., torch.Tensor],
    config: Stage2Config,
    matched_times: list[dict[str, float | int]],
    local_ids: np.ndarray,
    local_labels: np.ndarray,
    local_test_mask: np.ndarray,
    local_noise: torch.Tensor,
    per_rank_batch: int,
    log_every_batches: int,
    precision: str,
    p_stats: dict[float, MomentAccumulator],
    p_test: HeldoutStates,
    ridge_ratio: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], HeldoutStates]:
    q_stats = {float(item["requested_time"]): MomentAccumulator() for item in matched_times}
    q_test = HeldoutStates()
    model_fn = partial(forward_with_internalguidance, model)
    ig_interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    t1_hasher = hashlib.sha256()

    with torch.inference_mode():
        batch_starts = range(0, local_ids.size, per_rank_batch)
        total_batches = math.ceil(local_ids.size / per_rank_batch)
        for batch_index, start in enumerate(batch_starts):
            end = min(start + per_rank_batch, local_ids.size)
            batch_ids = local_ids[start:end]
            batch_labels = torch.from_numpy(local_labels[start:end]).to(
                device=device, dtype=torch.long
            )
            batch_test = local_test_mask[start:end]
            noise = local_noise[start:end].to(device=device, dtype=torch.float32)
            doubled_noise = torch.cat([noise, noise], dim=0)
            null_labels = torch.full(
                (noise.shape[0],),
                int(config.misc.num_classes),
                device=device,
                dtype=torch.long,
            )
            context = torch.cat([batch_labels, null_labels], dim=0)

            def capture(time_key: float, state: torch.Tensor) -> None:
                train_mask = torch.from_numpy(~batch_test).to(device=state.device)
                test_mask = ~train_mask
                q_stats[time_key].update(state[train_mask])
                q_test.add(time_key, state[test_mask], batch_ids[batch_test])
                if time_key == 1.0:
                    t1_hasher.update(
                        state.detach().float().cpu().contiguous().numpy().tobytes()
                    )

            recorder = SamplerStateRecorder(
                model_fn,
                matched_times,
                real_batch_size=noise.shape[0],
                callback=capture,
            )
            with autocast_context(precision):
                sample_fn(
                    doubled_noise,
                    recorder,
                    context=context,
                    attn_mask=None,
                    ig_scale=float(ig_scale),
                    ig_interval=ig_interval,
                )
            recorder.validate(int(config.sampler.num_steps))
            if (
                dist.get_rank() == 0
                and (
                    (batch_index + 1) % log_every_batches == 0
                    or batch_index + 1 == total_batches
                )
            ):
                print(
                    f"[{branch}] sampler batches {batch_index + 1}/{total_batches}",
                    flush=True,
                )

    payload: dict[str, Any] = {"ids": {}, "p_scores": {}, "q_scores": {}}
    diagnostics: dict[str, Any] = {
        "branch": branch,
        "ig_scale": float(ig_scale),
        "rank_t1_sha256": t1_hasher.hexdigest(),
        "times": {},
    }
    feature_count = math.prod(int(value) for value in config.misc.latent_size)
    for item in matched_times:
        time_key = float(item["requested_time"])
        p_global = p_stats[time_key].reduced(device, feature_count)
        q_global = q_stats[time_key].reduced(device, feature_count)
        weight, intercept, ridge = fit_diagonal_lda(
            p_global, q_global, ridge_ratio=ridge_ratio
        )
        p_states, p_ids = p_test.tensors(time_key)
        q_states, q_ids = q_test.tensors(time_key)
        if not np.array_equal(p_ids, q_ids):
            raise RuntimeError(f"held-out p/q IDs differ at t={time_key}")
        if time_key == 1.0:
            max_abs = (
                float((p_states.float() - q_states.float()).abs().max().item())
                if p_states.numel()
                else 0.0
            )
            if max_abs != 0.0:
                raise RuntimeError(f"t=1 p/q states differ, max_abs={max_abs}")
            diagnostics["t1_p_q_max_abs"] = max_abs
        payload["ids"][time_key] = p_ids
        payload["p_scores"][time_key] = score_states(
            p_states, weight, intercept, device
        )
        payload["q_scores"][time_key] = score_states(
            q_states, weight, intercept, device
        )
        diagnostics["times"][time_key] = {
            "train_p": p_global.count,
            "train_q": q_global.count,
            "test": int(p_ids.size),
            "ridge": ridge,
            "weight_norm": float(weight.norm().item()),
        }
    return payload, diagnostics, q_test


def compare_heldout_state_stores(
    full: HeldoutStates,
    ig: HeldoutStates,
    matched_times: list[dict[str, float | int]],
    device: torch.device,
) -> dict[float, dict[str, float]]:
    comparisons: dict[float, dict[str, float]] = {}
    for item in matched_times:
        time_key = float(item["requested_time"])
        full_states, full_ids = full.tensors(time_key)
        ig_states, ig_ids = ig.tensors(time_key)
        if not np.array_equal(full_ids, ig_ids):
            raise RuntimeError(f"full/IG held-out state IDs differ at t={time_key}")
        full_float = full_states.float()
        ig_float = ig_states.float()
        local = torch.tensor(
            [
                float((ig_float - full_float).square().sum().item()),
                float(full_float.square().sum().item()),
                float(full_float.numel()),
            ],
            device=device,
            dtype=torch.float64,
        )
        local_max = torch.tensor(
            [
                float((ig_float - full_float).abs().max().item())
                if full_float.numel()
                else 0.0
            ],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(local)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        diff_rms = math.sqrt(float(local[0].item()) / max(float(local[2].item()), 1.0))
        full_rms = math.sqrt(float(local[1].item()) / max(float(local[2].item()), 1.0))
        comparisons[time_key] = {
            "q_ig_minus_full_rms": diff_rms,
            "q_full_rms": full_rms,
            "q_ig_vs_full_relative_rms": diff_rms / max(full_rms, 1e-30),
            "q_ig_vs_full_max_abs": float(local_max.item()),
        }
    return comparisons


def _plot_results(results: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    styles = {"full": ("#3569a8", "o"), "ig": ("#c94f3d", "s")}
    for branch in ("full", "ig"):
        frame = results[results["branch"] == branch].sort_values("actual_time")
        color, marker = styles[branch]
        axis.plot(
            frame["actual_time"],
            frame["auc"],
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=7,
            label="Full (IG scale=1.0)" if branch == "full" else "Official IG",
        )
        axis.fill_between(
            frame["actual_time"].to_numpy(),
            frame["auc_ci_low"].to_numpy(),
            frame["auc_ci_high"].to_numpy(),
            color=color,
            alpha=0.16,
        )
    axis.axhline(0.5, color="#333333", linestyle="--", linewidth=1.3, label="Chance")
    axis.set_xlabel("Actual shifted solver time t (sampling proceeds from 1 to 0)")
    axis.set_ylabel("Held-out AUC: real p_t vs sampler q_t")
    axis.set_title("Does RAEv2 Internal Guidance Move q_t Toward p_t?")
    axis.set_ylim(0.45, 1.01)
    axis.grid(True, alpha=0.22)
    axis.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats cannot be negative")

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

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = load_config(config_path)
    requested_times = tuple(args.times or DEFAULT_TIMES)
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    matched_times = match_requested_times(requested_times, grid)
    if not any(float(item["requested_time"]) == 1.0 for item in matched_times):
        raise ValueError("t=1.0 is required as a hard null control")

    labels = build_requested_labels(args.samples, int(config.misc.num_classes))
    test_mask = class_group_split(labels, args.test_fraction, args.seed + 17)
    train_classes = np.unique(labels[~test_mask])
    test_classes = np.unique(labels[test_mask])
    if np.intersect1d(train_classes, test_classes).size:
        raise RuntimeError("ImageNet classes leak across the train/test split")
    if rank == 0:
        real_rows = select_matching_imagenet_rows(
            args.parquet_data_path, labels, args.seed + 31
        )
    else:
        real_rows = np.empty(args.samples, dtype=np.int64)
    rows_tensor = torch.from_numpy(real_rows).to(device=device)
    dist.broadcast(rows_tensor, src=0)
    real_rows = rows_tensor.cpu().numpy().astype(np.int64, copy=True)

    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]
    local_rows = real_rows[local_ids]
    local_test_mask = test_mask[local_ids]
    noise_generator = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + 1_000_003 * rank
    )
    local_noise = torch.randn(
        (local_ids.size, *latent_size),
        generator=noise_generator,
        dtype=torch.float32,
    )

    dataset = DeterministicImageNetPacked(
        args.packed_data_path,
        split="train",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    rae = instantiate_from_config(config.stage_1)
    del rae.decoder
    rae = rae.to(device).eval()
    rae.requires_grad_(False)
    p_stats = {float(item["requested_time"]): MomentAccumulator() for item in matched_times}
    p_test = HeldoutStates()

    with torch.inference_mode():
        batch_starts = range(0, local_ids.size, args.per_rank_batch)
        total_batches = math.ceil(local_ids.size / args.per_rank_batch)
        for batch_index, start in enumerate(batch_starts):
            end = min(start + args.per_rank_batch, local_ids.size)
            image_batch = []
            for source_row, expected_label in zip(
                local_rows[start:end].tolist(), local_labels[start:end].tolist()
            ):
                image, actual_label, _ = dataset[int(source_row)]
                if int(actual_label) != int(expected_label):
                    raise RuntimeError(
                        f"ImageNet label mismatch: row {source_row} has {actual_label}, "
                        f"expected {expected_label}"
                    )
                image_batch.append(image)
            images = torch.stack(image_batch).to(device=device)
            with autocast_context(args.precision):
                clean = rae.encode(images).float()
            noise = local_noise[start:end].to(device=device)
            batch_test = local_test_mask[start:end]
            train_mask = torch.from_numpy(~batch_test).to(device=device)
            test_device_mask = ~train_mask
            batch_ids = local_ids[start:end]
            for item in matched_times:
                key = float(item["requested_time"])
                actual_t = float(item["actual_time"])
                state = (1.0 - actual_t) * clean + actual_t * noise
                p_stats[key].update(state[train_mask])
                p_test.add(key, state[test_device_mask], batch_ids[batch_test])
            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"[p_t] encoder batches {batch_index + 1}/{total_batches}",
                    flush=True,
                )

    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_full_stage2_checkpoint(checkpoint)
    if args.state_key not in checkpoint:
        raise KeyError(f"checkpoint has no {args.state_key!r} state")
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    del checkpoint
    gc.collect()

    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    official_ig_scale = (
        float(args.ig_scale)
        if args.ig_scale is not None
        else float(config.guidance.ig.scale)
    )
    branches = (("full", 1.0), ("ig", official_ig_scale))
    branch_payloads: dict[str, dict[str, Any]] = {}
    branch_diagnostics: dict[str, list[dict[str, Any]]] = {}
    branch_heldout_states: dict[str, HeldoutStates] = {}

    for branch_name, branch_scale in branches:
        payload, diagnostic, heldout_states = _branch_states(
            branch=branch_name,
            ig_scale=branch_scale,
            model=model,
            sample_fn=sample_fn,
            config=config,
            matched_times=matched_times,
            local_ids=local_ids,
            local_labels=local_labels,
            local_test_mask=local_test_mask,
            local_noise=local_noise,
            per_rank_batch=args.per_rank_batch,
            log_every_batches=args.log_every_batches,
            precision=args.precision,
            p_stats=p_stats,
            p_test=p_test,
            ridge_ratio=args.ridge_ratio,
            device=device,
        )
        branch_payloads[branch_name] = payload
        branch_heldout_states[branch_name] = heldout_states
        branch_diagnostics[branch_name] = gather_rank_payload(diagnostic)
        dist.barrier()

    full_t1_hashes = [item["rank_t1_sha256"] for item in branch_diagnostics["full"]]
    ig_t1_hashes = [item["rank_t1_sha256"] for item in branch_diagnostics["ig"]]
    if full_t1_hashes != ig_t1_hashes:
        raise RuntimeError("full and IG branches do not share identical t=1 states")
    trajectory_differences = compare_heldout_state_stores(
        branch_heldout_states["full"],
        branch_heldout_states["ig"],
        matched_times,
        device,
    )
    non_null_differences = [
        values["q_ig_vs_full_max_abs"]
        for key, values in trajectory_differences.items()
        if key < 1.0
    ]
    if official_ig_scale != 1.0 and not any(value > 0.0 for value in non_null_differences):
        raise RuntimeError("IG scale differs from 1 but the captured trajectories are identical")

    all_branch_scores: dict[str, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "full": {},
        "ig": {},
    }
    for branch_name, _ in branches:
        gathered = gather_rank_payload(branch_payloads[branch_name])
        if rank == 0:
            for item in matched_times:
                time_key = float(item["requested_time"])
                all_branch_scores[branch_name][time_key] = concatenate_rank_scores(
                    gathered, time_key
                )

    if rank == 0:
        result_rows = []
        delta_rows = []
        for time_index, item in enumerate(matched_times):
            time_key = float(item["requested_time"])
            for branch_index, (branch_name, branch_scale) in enumerate(branches):
                ids, p_scores, q_scores = all_branch_scores[branch_name][time_key]
                auc = paired_auc(p_scores, q_scores)
                ci_low, ci_high = bootstrap_paired_auc(
                    p_scores,
                    q_scores,
                    args.bootstrap_repeats,
                    args.seed + 1000 * time_index + branch_index,
                )
                result_rows.append(
                    {
                        "branch": branch_name,
                        "ig_scale": branch_scale,
                        **item,
                        "auc": auc,
                        "auc_ci_low": ci_low,
                        "auc_ci_high": ci_high,
                        "heldout_pairs": int(ids.size),
                    }
                )

            full_ids, p_full, q_full = all_branch_scores["full"][time_key]
            ig_ids, p_ig, q_ig = all_branch_scores["ig"][time_key]
            if not np.array_equal(full_ids, ig_ids):
                raise RuntimeError("full and IG comparisons use different held-out IDs")
            full_auc = paired_auc(p_full, q_full)
            ig_auc = paired_auc(p_ig, q_ig)
            delta_low, delta_high = bootstrap_auc_delta(
                p_full,
                q_full,
                p_ig,
                q_ig,
                args.bootstrap_repeats,
                args.seed + 10_000 + time_index,
            )
            delta_rows.append(
                {
                    **item,
                    "auc_full": full_auc,
                    "auc_ig": ig_auc,
                    "auc_delta_ig_minus_full": ig_auc - full_auc,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    **trajectory_differences[time_key],
                }
            )

        results = pd.DataFrame(result_rows).sort_values(["actual_time", "branch"])
        deltas = pd.DataFrame(delta_rows).sort_values("actual_time")
        non_null = deltas[deltas["requested_time"] < 1.0]
        mean_delta = float(non_null["auc_delta_ig_minus_full"].mean())
        screening = screening_conclusion(deltas, int(test_mask.sum()))

        results.to_csv(output_dir / "auc_results.csv", index=False)
        deltas.to_csv(output_dir / "auc_delta_ig_minus_full.csv", index=False)
        np.savez_compressed(
            output_dir / "sample_protocol.npz",
            sample_ids=np.arange(args.samples, dtype=np.int64),
            labels=labels,
            real_source_rows=real_rows,
            test_mask=test_mask,
        )
        score_archive: dict[str, np.ndarray] = {}
        for branch_name, _ in branches:
            for item in matched_times:
                time_key = float(item["requested_time"])
                ids, p_scores, q_scores = all_branch_scores[branch_name][time_key]
                suffix = str(time_key).replace(".", "p")
                score_archive[f"ids_t{suffix}"] = ids
                score_archive[f"p_{branch_name}_t{suffix}"] = p_scores
                score_archive[f"q_{branch_name}_t{suffix}"] = q_scores
        np.savez_compressed(output_dir / "heldout_probe_scores.npz", **score_archive)
        _plot_results(results, output_dir / "auc_vs_time.png")
        manifest = {
            "protocol": "raev2_distribution_auc_v1",
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size": checkpoint_path.stat().st_size,
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_key": args.state_key,
            "samples": args.samples,
            "train_pairs": int((~test_mask).sum()),
            "heldout_pairs": int(test_mask.sum()),
            "train_classes": int(train_classes.size),
            "heldout_classes": int(test_classes.size),
            "split_unit": "ImageNet class; all examples of a class stay in one split",
            "paired_controls": "same class and initial noise for p, full, and IG",
            "classifier": "full-latent diagonal LDA with train-only moments",
            "ridge_ratio": args.ridge_ratio,
            "precision": args.precision,
            "sampler_steps": int(config.sampler.num_steps),
            "time_dist_shift": shift,
            "matched_times": matched_times,
            "official_ig_scale": official_ig_scale,
            "ig_interval": [
                float(config.guidance.ig.t_min),
                float(config.guidance.ig.t_max),
            ],
            "seed": args.seed,
            "world_size": world_size,
            "t1_full_ig_hashes_equal": True,
            "trajectory_differences": trajectory_differences,
            "mean_non_null_auc_delta_ig_minus_full": mean_delta,
            "screening_conclusion": screening,
            "diagnostics": branch_diagnostics,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(results.to_string(index=False))
        print(deltas.to_string(index=False))
        print(json.dumps({"screening_conclusion": screening}, ensure_ascii=False))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Audit RAEv2 predicted-clean latents across Full and IG trajectories.

This is an inference-only 2x2 intervention.  At matched solver states from the
Full and official-IG trajectories, it decodes both the Full-head prediction and
the IG-extrapolated prediction.  Holding either the state or the head fixed
separates an immediate head effect from the accumulated trajectory effect.

The official sampler and ``forward_with_internalguidance`` remain unchanged.
A read-only forward hook observes the model's Full/Base outputs, and all image
metrics are computed after the frozen official RAE decoder.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import validate_full_stage2_checkpoint  # noqa: E402
from experiments.run_raev2_decoded_distribution_audit import (  # noqa: E402
    _decode_features,
    _load_feature_shards,
    feature_probe_scores,
    feature_statistics,
    fid_between_statistics,
    fit_feature_probe,
    load_reference_statistics,
    time_suffix,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    bootstrap_paired_auc,
    build_requested_labels,
    class_group_split,
    load_config,
    match_requested_times,
    paired_auc,
    shifted_solver_grid,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_TIMES = (0.14, 0.2, 0.4, 1.0)
HEADS = ("full", "ig")
STATE_BRANCHES = ("full", "ig")


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
    parser.add_argument("--decoded-reference-run", type=Path, required=True)
    parser.add_argument(
        "--fid-reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--time", action="append", type=float, dest="times")
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--inception-feature", choices=("64", "192", "768", "2048"), default="2048"
    )
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def condition_name(head: str, state_branch: str) -> str:
    if head not in HEADS or state_branch not in STATE_BRANCHES:
        raise ValueError(f"invalid head/state pair: {head}/{state_branch}")
    return f"{head}_on_{state_branch}"


def guided_clean_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    t: torch.Tensor,
    *,
    scale: float,
    interval: tuple[float, float],
) -> torch.Tensor:
    """Reproduce the official IG combination in clean-prediction space."""

    if full.shape != base.shape:
        raise ValueError("Full/Base prediction shapes differ")
    if t.shape != (full.shape[0],):
        raise ValueError("time tensor must contain one value per prediction")
    if not interval[0] < interval[1]:
        raise ValueError("IG interval must be ordered")
    active = ((t >= interval[0]) & (t <= interval[1])).view(
        -1, *([1] * (full.ndim - 1))
    )
    return torch.where(active, base + float(scale) * (full - base), full)


def metric_effect_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Decompose the 2x2 table into current-head, history, and total effects."""

    comparisons = (
        ("head_on_full_state", "ig_on_full", "full_on_full"),
        ("head_on_ig_state", "ig_on_ig", "full_on_ig"),
        ("history_under_full_head", "full_on_ig", "full_on_full"),
        ("history_under_ig_head", "ig_on_ig", "ig_on_full"),
        ("on_policy_total", "ig_on_ig", "full_on_full"),
    )
    rows: list[dict[str, Any]] = []
    for requested_time, group in summary.groupby("requested_time", sort=True):
        by_condition = group.set_index("condition")
        missing = {
            condition
            for _, positive, negative in comparisons
            for condition in (positive, negative)
            if condition not in by_condition.index
        }
        if missing:
            raise ValueError(f"2x2 table is incomplete: {sorted(missing)}")
        for effect, positive, negative in comparisons:
            pos = by_condition.loc[positive]
            neg = by_condition.loc[negative]
            rows.append(
                {
                    "requested_time": float(requested_time),
                    "actual_time": float(pos["actual_time"]),
                    "effect": effect,
                    "positive_condition": positive,
                    "negative_condition": negative,
                    "auc_delta": float(pos["auc"] - neg["auc"]),
                    "auc_separability_delta": float(
                        abs(pos["auc"] - 0.5) - abs(neg["auc"] - 0.5)
                    ),
                    "fid_real_delta": float(pos["fid_real"] - neg["fid_real"]),
                    "fid_reconstruction_delta": float(
                        pos["fid_reconstruction"] - neg["fid_reconstruction"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["actual_time", "effect"], ascending=[False, True])


def _load_reconstruction_reference(
    run_dir: Path,
    *,
    expected_samples: int,
    expected_seed: int,
    expected_feature: str,
    expected_checkpoint: Path,
    expected_state_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "protocol": "raev2_decoded_distribution_audit_v1",
        "samples": expected_samples,
        "seed": expected_seed,
        "inception_feature": expected_feature,
        "state_key": expected_state_key,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"reference manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}"
            )
    reference_checkpoint = Path(manifest["checkpoint"]).expanduser().resolve()
    if reference_checkpoint != expected_checkpoint.expanduser().resolve():
        raise ValueError("reference run used a different stage-2 checkpoint")
    world_size = int(manifest["world_size"])
    ids, labels, test_mask, features, _ = _load_feature_shards(run_dir, world_size)
    key = f"feat_p_{time_suffix(0.0)}"
    if key not in features:
        raise KeyError(f"reference run does not contain {key}")

    examples: dict[int, np.ndarray] = {}
    for rank in range(world_size):
        with np.load(run_dir / f"decoded_features_rank{rank:02d}.npz") as shard:
            ids_key = f"example_ids_p_{time_suffix(0.0)}"
            image_key = f"example_p_{time_suffix(0.0)}"
            for sample_id, image in zip(shard[ids_key].tolist(), shard[image_key]):
                examples[int(sample_id)] = image
    return ids, labels, test_mask, features[key], examples, manifest


class HeadOutputHook:
    """Capture exactly one Full/Base output pair from a top-level model call."""

    def __init__(self, in_channels: int) -> None:
        self.in_channels = int(in_channels)
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.calls = 0

    def clear(self) -> None:
        if self.pending is not None:
            raise RuntimeError("previous Full/Base output was not consumed")

    def __call__(self, _module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if self.pending is not None:
            raise RuntimeError("stage-2 model was called more than once per solver evaluation")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("internal-guidance model did not return Full/Base heads")
        full = output[0][:, : self.in_channels]
        base = output[1][:, : self.in_channels]
        if full.shape != base.shape:
            raise RuntimeError("Full/Base output shapes differ")
        self.pending = (full, base)
        self.calls += 1

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending is None:
            raise RuntimeError("stage-2 hook did not observe a model output")
        value = self.pending
        self.pending = None
        return value


class OnlinePredictedCleanRecorder:
    """Wrap the official model function and decode selected clean predictions."""

    def __init__(
        self,
        *,
        branch: str,
        model_fn: Callable[..., torch.Tensor],
        head_hook: HeadOutputHook,
        decoder_rae: torch.nn.Module,
        extractor: torch.nn.Module,
        matched_times: list[dict[str, float | int]],
        official_ig_scale: float,
        ig_interval: tuple[float, float],
        precision: str,
        example_ids: set[int],
        device: torch.device,
    ) -> None:
        if branch not in STATE_BRANCHES:
            raise ValueError(f"invalid trajectory branch: {branch}")
        self.branch = branch
        self.model_fn = model_fn
        self.head_hook = head_hook
        self.decoder_rae = decoder_rae
        self.extractor = extractor
        self.by_index = {
            int(item["solver_index"]): float(item["requested_time"])
            for item in matched_times
        }
        self.actual_by_index = {
            int(item["solver_index"]): float(item["actual_time"])
            for item in matched_times
        }
        self.official_ig_scale = float(official_ig_scale)
        self.ig_interval = tuple(float(value) for value in ig_interval)
        self.precision = precision
        self.example_ids = example_ids
        self.device = device
        self.features: dict[str, list[np.ndarray]] = defaultdict(list)
        self.examples: dict[tuple[str, float, int], np.ndarray] = {}
        self.latent_sums: dict[tuple[float, str], dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "full_sumsq": 0.0, "base_sumsq": 0.0,
                     "head_gap_sumsq": 0.0, "guided_gap_sumsq": 0.0}
        )
        self.decode_sums: dict[tuple[float, str], dict[str, float]] = defaultdict(
            lambda: {"samples": 0.0, "raw_min": math.inf, "raw_max": -math.inf,
                     "clipped_low_weighted": 0.0, "clipped_high_weighted": 0.0}
        )
        self.calls = 0
        self.captured: set[int] = set()
        self.current_ids: np.ndarray | None = None
        self.expected_calls: int | None = None

    def begin_batch(self, sample_ids: np.ndarray, expected_calls: int) -> None:
        if self.current_ids is not None:
            raise RuntimeError("previous sampler batch was not finalized")
        self.current_ids = sample_ids.astype(np.int64, copy=True)
        self.calls = 0
        self.captured = set()
        self.expected_calls = int(expected_calls)

    def _guided_prediction(
        self, full: torch.Tensor, base: torch.Tensor, t: torch.Tensor, scale: float
    ) -> torch.Tensor:
        return guided_clean_prediction(
            full,
            base,
            t,
            scale=scale,
            interval=self.ig_interval,
        )

    def _record_prediction(
        self,
        requested_time: float,
        head: str,
        prediction: torch.Tensor,
    ) -> None:
        if self.current_ids is None:
            raise RuntimeError("recording outside a sampler batch")
        condition = condition_name(head, self.branch)
        key = f"feat_{condition}_{time_suffix(requested_time)}"
        # Disable the outer stage-2 autocast around Inception.  _decode_features
        # re-enables it only for the decoder, matching the existing decoder audit.
        with torch.autocast(device_type="cuda", enabled=False):
            features, examples, diagnostics = _decode_features(
                self.decoder_rae,
                self.extractor,
                prediction,
                self.current_ids,
                decode_batch=max(1, prediction.shape[0]),
                precision=self.precision,
                example_ids=self.example_ids,
                device=self.device,
            )
        self.features[key].append(features)
        for sample_id, image in examples.items():
            self.examples[(condition, requested_time, int(sample_id))] = image
        aggregate = self.decode_sums[(requested_time, condition)]
        batch_samples = float(prediction.shape[0])
        aggregate["samples"] += batch_samples
        aggregate["raw_min"] = min(aggregate["raw_min"], diagnostics["raw_min"])
        aggregate["raw_max"] = max(aggregate["raw_max"], diagnostics["raw_max"])
        aggregate["clipped_low_weighted"] += (
            diagnostics["clipped_low_fraction"] * batch_samples
        )
        aggregate["clipped_high_weighted"] += (
            diagnostics["clipped_high_fraction"] * batch_samples
        )

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        index = self.calls
        self.head_hook.clear()
        output = self.model_fn(x, t, **kwargs)
        full, base = self.head_hook.pop()
        real_batch_size = full.shape[0]
        if self.current_ids is None or self.current_ids.size != real_batch_size:
            raise RuntimeError("Full/Base batch size does not match current sample IDs")
        if output.shape[0] != 2 * real_batch_size:
            raise RuntimeError("official IG wrapper returned an unexpected batch size")

        branch_scale = float(kwargs["ig_scale"])
        branch_expected = self._guided_prediction(
            full, base, t[:real_batch_size], branch_scale
        )
        official = output[:real_batch_size, : full.shape[1]]
        if not torch.allclose(official, branch_expected, rtol=2e-3, atol=2e-3):
            maximum = float((official - branch_expected).abs().max().item())
            raise RuntimeError(f"manual guidance does not match official output: {maximum}")

        if index in self.by_index:
            requested_time = self.by_index[index]
            actual = float(t[0].item())
            expected = self.actual_by_index[index]
            if abs(actual - expected) > 1e-6:
                raise RuntimeError(
                    f"sampler time mismatch at step {index}: {actual} != {expected}"
                )
            if not torch.allclose(t, t[:1].expand_as(t), rtol=0.0, atol=0.0):
                raise RuntimeError("sampler batch contains inconsistent times")
            guided = self._guided_prediction(
                full, base, t[:real_batch_size], self.official_ig_scale
            )
            values = self.latent_sums[(requested_time, self.branch)]
            values["count"] += float(full.numel())
            values["full_sumsq"] += float(full.detach().float().square().sum().item())
            values["base_sumsq"] += float(base.detach().float().square().sum().item())
            values["head_gap_sumsq"] += float(
                (full - base).detach().float().square().sum().item()
            )
            values["guided_gap_sumsq"] += float(
                (guided - full).detach().float().square().sum().item()
            )
            self._record_prediction(requested_time, "full", full.detach())
            self._record_prediction(requested_time, "ig", guided.detach())
            self.captured.add(index)
        self.calls += 1
        return output

    def finish_batch(self) -> None:
        if self.expected_calls is None or self.calls != self.expected_calls:
            raise RuntimeError(
                f"sampler made {self.calls} calls, expected {self.expected_calls}"
            )
        if self.captured != set(self.by_index):
            raise RuntimeError("not all requested predicted-clean outputs were captured")
        self.current_ids = None
        self.expected_calls = None

    def archive(self, local_ids: np.ndarray) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, chunks in self.features.items():
            value = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            if value.shape[0] != local_ids.size:
                raise RuntimeError(f"feature count mismatch for {key}")
            result[key] = value
        for (condition, requested_time, sample_id), image in self.examples.items():
            suffix = time_suffix(requested_time)
            result[f"example_{condition}_{suffix}_id{sample_id:06d}"] = image
        return result

    def diagnostics(self) -> dict[str, Any]:
        rows = []
        for (requested_time, branch), values in sorted(self.latent_sums.items()):
            count = max(values["count"], 1.0)
            rows.append(
                {
                    "requested_time": requested_time,
                    "state_branch": branch,
                    "full_rms": math.sqrt(values["full_sumsq"] / count),
                    "base_rms": math.sqrt(values["base_sumsq"] / count),
                    "head_gap_rms": math.sqrt(values["head_gap_sumsq"] / count),
                    "guided_minus_full_rms": math.sqrt(
                        values["guided_gap_sumsq"] / count
                    ),
                }
            )
        decode_rows = []
        for (requested_time, condition), values in sorted(self.decode_sums.items()):
            count = max(values["samples"], 1.0)
            decode_rows.append(
                {
                    "requested_time": requested_time,
                    "condition": condition,
                    "raw_min": values["raw_min"],
                    "raw_max": values["raw_max"],
                    "clipped_low_fraction": values["clipped_low_weighted"] / count,
                    "clipped_high_fraction": values["clipped_high_weighted"] / count,
                }
            )
        return {"latent_rows": rows, "decode_rows": decode_rows}


def _load_prediction_shards(
    output_dir: Path, world_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    shards = [
        np.load(output_dir / f"predicted_clean_features_rank{rank:02d}.npz")
        for rank in range(world_size)
    ]
    ids = np.concatenate([shard["ids"] for shard in shards])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(ids.size)):
        raise RuntimeError("prediction shards do not contain every sample exactly once")
    labels = np.concatenate([shard["labels"] for shard in shards])[order]
    test_mask = np.concatenate([shard["test_mask"] for shard in shards])[order].astype(bool)
    keys = sorted(key for key in shards[0].files if key.startswith("feat_"))
    for shard in shards[1:]:
        if sorted(key for key in shard.files if key.startswith("feat_")) != keys:
            raise RuntimeError("prediction feature keys differ across ranks")
    features = {
        key: np.concatenate([shard[key] for shard in shards], axis=0)[order]
        for key in keys
    }
    examples: dict[str, np.ndarray] = {}
    for shard in shards:
        for key in shard.files:
            if key.startswith("example_"):
                examples[key] = shard[key]
    for shard in shards:
        shard.close()
    return ids[order], labels, test_mask, features, examples


def _plot_examples(
    output_dir: Path,
    requested_times: tuple[float, ...],
    reconstruction_examples: dict[int, np.ndarray],
    predicted_examples: dict[str, np.ndarray],
) -> None:
    columns = ("full_on_full", "ig_on_full", "full_on_ig", "ig_on_ig")
    ordered_times = sorted(requested_times, reverse=True)
    for sample_id, reconstruction in sorted(reconstruction_examples.items()):
        available = all(
            f"example_{condition}_{time_suffix(time)}_id{sample_id:06d}"
            in predicted_examples
            for condition in columns
            for time in ordered_times
        )
        if not available:
            continue
        fig, axes = plt.subplots(
            len(ordered_times), 5, figsize=(14.8, 3.0 * len(ordered_times)), squeeze=False
        )
        for row, requested_time in enumerate(ordered_times):
            images = [reconstruction] + [
                predicted_examples[
                    f"example_{condition}_{time_suffix(requested_time)}_id{sample_id:06d}"
                ]
                for condition in columns
            ]
            for column, image in enumerate(images):
                axes[row, column].imshow(image)
                axes[row, column].axis("off")
            axes[row, 0].set_ylabel(f"t={requested_time:g}", rotation=0, labelpad=28)
        titles = ("D(E(x)) reference",) + columns
        for column, title in enumerate(titles):
            axes[0, column].set_title(title)
        fig.suptitle(f"RAEv2 predicted-clean 2x2, sample ID {sample_id}")
        fig.tight_layout()
        fig.savefig(output_dir / f"predicted_clean_example_id{sample_id:04d}.png", dpi=170)
        plt.close(fig)


def _plot_summary(summary: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    colors = {
        "full_on_full": "#3366aa",
        "ig_on_full": "#ee7733",
        "full_on_ig": "#228833",
        "ig_on_ig": "#aa3377",
    }
    markers = {"full_on_full": "o", "ig_on_full": "s", "full_on_ig": "^", "ig_on_ig": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for condition in colors:
        frame = summary[summary["condition"] == condition].sort_values("actual_time")
        axes[0, 0].plot(frame["actual_time"], frame["auc"], marker=markers[condition],
                        color=colors[condition], label=condition)
        axes[0, 1].plot(frame["actual_time"], frame["fid_real"], marker=markers[condition],
                        color=colors[condition], label=condition)
        axes[1, 0].plot(frame["actual_time"], frame["fid_reconstruction"],
                        marker=markers[condition], color=colors[condition], label=condition)
    axes[0, 0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("AUC vs D(E(x))")
    axes[0, 1].set_ylabel("FID to ImageNet")
    axes[1, 0].set_ylabel("FID to D(E(x))")
    total = effects[effects["effect"] == "on_policy_total"].sort_values("actual_time")
    axes[1, 1].plot(total["actual_time"], total["fid_real_delta"], "o-", label="FID real")
    axes[1, 1].plot(total["actual_time"], total["fid_reconstruction_delta"], "s-",
                    label="FID reconstruction")
    axes[1, 1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("On-policy IG - Full (lower is better)")
    for axis in axes.flat:
        axis.set_xlabel("Solver time t (sampling: 1 to 0)")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False)
    fig.suptitle("RAEv2 Predicted-Clean Head/Trajectory Intervention")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.per_rank_batch <= 0:
        raise ValueError("sample and batch counts must be positive")
    if args.example_count < 0:
        raise ValueError("--example-count cannot be negative")
    requested_times = tuple(sorted(set(args.times or DEFAULT_TIMES)))
    if any(not 0.0 < value <= 1.0 for value in requested_times):
        raise ValueError("predicted-clean times must lie in (0, 1]")

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

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    matched_times = match_requested_times(requested_times, grid)
    actual_by_requested = {
        float(row["requested_time"]): float(row["actual_time"])
        for row in matched_times
    }
    checkpoint_path = args.checkpoint.expanduser().resolve()

    labels = build_requested_labels(args.samples, int(config.misc.num_classes))
    test_mask = class_group_split(labels, args.test_fraction, args.seed + 17)
    reference_payload = None
    if rank == 0:
        reference_payload = _load_reconstruction_reference(
            args.decoded_reference_run,
            expected_samples=args.samples,
            expected_seed=args.seed,
            expected_feature=args.inception_feature,
            expected_checkpoint=checkpoint_path,
            expected_state_key=args.state_key,
        )
        reference_ids, reference_labels, reference_test, *_ = reference_payload
        if not np.array_equal(reference_ids, np.arange(args.samples)):
            raise RuntimeError("reconstruction reference IDs are incomplete")
        if not np.array_equal(reference_labels, labels):
            raise RuntimeError("reconstruction reference labels differ")
        if not np.array_equal(reference_test, test_mask):
            raise RuntimeError("reconstruction reference class split differs")
    dist.barrier()

    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]
    local_test_mask = test_mask[local_ids]
    generator = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + 1_000_003 * rank
    )
    local_noise = torch.randn(
        (local_ids.size, *latent_size), generator=generator, dtype=torch.float32
    )
    example_ids = set(range(min(args.example_count, args.samples)))

    # Instantiate the decoder first, discard its encoder, then keep only the
    # frozen decoder alongside the stage-2 model.  No ImageNet training image is
    # loaded in this experiment.
    decoder_rae = instantiate_from_config(config.stage_1)
    del decoder_rae.encoder
    decoder_rae = decoder_rae.to(device).eval()
    decoder_rae.requires_grad_(False)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", [args.inception_feature], verbose=False
    ).to(device).eval()
    extractor.requires_grad_(False)

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    del checkpoint

    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    official_scale = (
        float(args.ig_scale) if args.ig_scale is not None else float(config.guidance.ig.scale)
    )
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    total_batches = math.ceil(local_ids.size / args.per_rank_batch)
    archive: dict[str, np.ndarray] = {
        "ids": local_ids,
        "labels": local_labels,
        "test_mask": local_test_mask,
    }
    diagnostic_payload: dict[str, Any] = {}

    for branch, path_scale in (("full", 1.0), ("ig", official_scale)):
        head_hook = HeadOutputHook(int(model.in_channels))
        handle = model.register_forward_hook(head_hook)
        recorder = OnlinePredictedCleanRecorder(
            branch=branch,
            model_fn=model_fn,
            head_hook=head_hook,
            decoder_rae=decoder_rae,
            extractor=extractor,
            matched_times=matched_times,
            official_ig_scale=official_scale,
            ig_interval=interval,
            precision=args.precision,
            example_ids=example_ids,
            device=device,
        )
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, local_ids.size, args.per_rank_batch)):
                end = min(start + args.per_rank_batch, local_ids.size)
                ids = local_ids[start:end]
                noise = local_noise[start:end].to(device=device)
                batch_labels = torch.from_numpy(local_labels[start:end]).to(device=device)
                null = torch.full(
                    (noise.shape[0],),
                    int(config.misc.num_classes),
                    device=device,
                    dtype=torch.long,
                )
                recorder.begin_batch(ids, int(config.sampler.num_steps))
                with autocast_context(args.precision):
                    sample_fn(
                        torch.cat((noise, noise), dim=0),
                        recorder,
                        context=torch.cat((batch_labels.long(), null), dim=0),
                        attn_mask=None,
                        ig_scale=float(path_scale),
                        ig_interval=interval,
                    )
                recorder.finish_batch()
                if rank == 0 and (
                    (batch_index + 1) % args.log_every_batches == 0
                    or batch_index + 1 == total_batches
                ):
                    print(
                        f"[{branch}_state] batches {batch_index + 1}/{total_batches}",
                        flush=True,
                    )
        handle.remove()
        archive.update(recorder.archive(local_ids))
        diagnostic_payload[branch] = recorder.diagnostics()
        dist.barrier()

    if 1.0 in requested_times:
        suffix = time_suffix(1.0)
        for head in HEADS:
            full_path = archive[f"feat_{condition_name(head, 'full')}_{suffix}"]
            ig_path = archive[f"feat_{condition_name(head, 'ig')}_{suffix}"]
            if not np.array_equal(full_path, ig_path):
                maximum = float(np.max(np.abs(full_path - ig_path)))
                raise RuntimeError(
                    f"t=1 same-state hard control failed for {head}: max feature diff {maximum}"
                )

    np.savez(
        output_dir / f"predicted_clean_features_rank{rank:02d}.npz",
        **archive,
    )
    (output_dir / f"predicted_clean_diagnostics_rank{rank:02d}.json").write_text(
        json.dumps(diagnostic_payload, indent=2), encoding="utf-8"
    )
    del model, sampler, transport, decoder_rae, extractor
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        ids, ordered_labels, ordered_test, features, examples = _load_prediction_shards(
            output_dir, world_size
        )
        if not np.array_equal(ids, np.arange(args.samples)):
            raise RuntimeError("prediction IDs changed after gathering")
        if not np.array_equal(ordered_labels, labels):
            raise RuntimeError("prediction labels changed after gathering")
        if not np.array_equal(ordered_test, test_mask):
            raise RuntimeError("prediction class split changed after gathering")
        assert reference_payload is not None
        _, _, _, reconstruction_features, reconstruction_examples, reference_manifest = (
            reference_payload
        )
        train_mask = ~test_mask
        real_reference = None
        if not args.skip_fid:
            real_reference = load_reference_statistics(
                args.fid_reference.expanduser().resolve(), args.inception_feature
            )
        reconstruction_stats = (
            feature_statistics(reconstruction_features)
            if real_reference is not None
            else None
        )
        rows = []
        score_archive: dict[str, np.ndarray] = {}
        for time_index, requested_time in enumerate(requested_times):
            suffix = time_suffix(requested_time)
            for condition_index, condition in enumerate(
                condition_name(head, state) for state in STATE_BRANCHES for head in HEADS
            ):
                prediction_features = features[f"feat_{condition}_{suffix}"]
                weight, intercept, ridge = fit_feature_probe(
                    reconstruction_features,
                    prediction_features,
                    train_mask,
                    args.ridge_ratio,
                )
                reference_scores = feature_probe_scores(
                    reconstruction_features[test_mask], weight, intercept
                )
                prediction_scores = feature_probe_scores(
                    prediction_features[test_mask], weight, intercept
                )
                auc = paired_auc(reference_scores, prediction_scores)
                ci_low, ci_high = bootstrap_paired_auc(
                    reference_scores,
                    prediction_scores,
                    args.bootstrap_repeats,
                    args.seed + 1000 * time_index + condition_index,
                )
                score_archive[f"reference_{condition}_{suffix}"] = reference_scores
                score_archive[f"prediction_{condition}_{suffix}"] = prediction_scores
                if real_reference is None or reconstruction_stats is None:
                    fid_real = fid_reconstruction = float("nan")
                else:
                    prediction_stats = feature_statistics(prediction_features)
                    fid_real = fid_between_statistics(prediction_stats, real_reference)
                    fid_reconstruction = fid_between_statistics(
                        prediction_stats, reconstruction_stats
                    )
                head, _, state_branch = condition.partition("_on_")
                rows.append(
                    {
                        "requested_time": requested_time,
                        "actual_time": actual_by_requested[requested_time],
                        "condition": condition,
                        "head": head,
                        "state_branch": state_branch,
                        "on_policy": bool(head == state_branch),
                        "auc": auc,
                        "auc_ci_low": ci_low,
                        "auc_ci_high": ci_high,
                        "auc_separability": abs(auc - 0.5),
                        "ridge": ridge,
                        "fid_real": fid_real,
                        "fid_reconstruction": fid_reconstruction,
                        "heldout_pairs": int(test_mask.sum()),
                    }
                )
        summary = pd.DataFrame(rows).sort_values(
            ["actual_time", "state_branch", "head"], ascending=[False, True, True]
        )
        effects = metric_effect_rows(summary)
        summary.to_csv(output_dir / "predicted_clean_summary.csv", index=False)
        effects.to_csv(output_dir / "predicted_clean_effects.csv", index=False)
        np.savez_compressed(output_dir / "predicted_clean_heldout_scores.npz", **score_archive)
        _plot_summary(summary, effects, output_dir / "predicted_clean_curves.png")
        _plot_examples(
            output_dir,
            requested_times,
            reconstruction_examples,
            examples,
        )
        manifest = {
            "protocol": "raev2_predicted_clean_2x2_v1",
            "inference_only": True,
            "official_sampler_unmodified": True,
            "intervention": {
                "states": ["full_trajectory", "official_ig_trajectory"],
                "heads": ["full", "base_plus_scale_times_full_minus_base"],
                "interpretation": (
                    "same-state comparisons isolate the current head; same-head comparisons "
                    "isolate accumulated trajectory history"
                ),
            },
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_key": args.state_key,
            "decoded_reference_run": str(args.decoded_reference_run.expanduser().resolve()),
            "decoded_reference_protocol": reference_manifest["protocol"],
            "samples": args.samples,
            "train_pairs": int(train_mask.sum()),
            "heldout_pairs": int(test_mask.sum()),
            "split_unit": "ImageNet class",
            "seed": args.seed,
            "world_size": world_size,
            "requested_times": requested_times,
            "matched_times": matched_times,
            "ig_scale": official_scale,
            "ig_interval": interval,
            "inception_feature": args.inception_feature,
            "fid_skipped": bool(args.skip_fid),
            "fid_reference": str(args.fid_reference.expanduser().resolve()),
            "important_scope": (
                "Predicted-clean outputs are decoder diagnostics at intermediate solver states. "
                "Only a final generated endpoint would be an official generation FID."
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(summary.to_string(index=False))
        print("\nEffect decomposition (positive - negative):")
        print(effects.to_string(index=False))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

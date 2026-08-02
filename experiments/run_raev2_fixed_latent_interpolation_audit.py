"""Compare actual RAEv2 scale endpoints with fixed-latent chord endpoints.

For each requested scale ``s``, the counterfactual endpoint is

``z_line(s) = z_1 + (s - 1) / (1.78 - 1) * (z_1.78 - z_1)``.

Only the counterfactual latents are decoded.  Completed scale-response
artifacts provide the actual endpoints and their observations.  The script is
inference-only and uses paired samples throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.analyze_raev2_scale_path_geometry import (  # noqa: E402
    low_frequency_features,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.run_raev2_decoded_distribution_audit import (  # noqa: E402
    feature_probe_scores,
    fit_feature_probe,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    bootstrap_paired_auc,
    load_config,
)
from experiments.run_raev2_scale_response_study import (  # noqa: E402
    atomic_save_npy,
    local_ids_for_rank,
    scale_key,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_fixed_latent_interpolation_audit_v1"
DEFAULT_SCALES = (1.2, 1.4, 1.6, 1.78, 2.0, 2.2)


def sample_path_comparison(
    control: torch.Tensor,
    actual: torch.Tensor,
    counterfactual: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare actual and counterfactual paths in one shared coordinate space."""

    if control.shape != actual.shape or control.shape != counterfactual.shape:
        raise ValueError("all path tensors must align")
    actual_step = (actual.float() - control.float()).flatten(1)
    counterfactual_step = (counterfactual.float() - control.float()).flatten(1)
    mismatch = (actual.float() - counterfactual.float()).flatten(1)
    actual_norm = actual_step.norm(dim=1)
    counterfactual_norm = counterfactual_step.norm(dim=1)
    mismatch_norm = mismatch.norm(dim=1)
    denominator = actual_norm * counterfactual_norm
    cosine = torch.full_like(actual_norm, float("nan"))
    valid = denominator > 0
    cosine[valid] = (
        (actual_step[valid] * counterfactual_step[valid]).sum(dim=1)
        / denominator[valid]
    )
    return {
        "actual_counterfactual_mismatch_over_actual_step": mismatch_norm
        / actual_norm.clamp_min(1e-30),
        "counterfactual_over_actual_step_norm": counterfactual_norm
        / actual_norm.clamp_min(1e-30),
        "actual_counterfactual_step_cosine": cosine,
    }


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


def _rank_array(
    root: Path, directory: str, condition: str, rank: int
) -> np.ndarray:
    return np.load(
        root / directory / f"{condition}_rank{rank:02d}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )


def _subset_count(samples: int, rank: int, world_size: int) -> int:
    return len(local_ids_for_rank(samples, rank, world_size))


def _load_ordered_subset(
    root: Path,
    directory: str,
    condition: str,
    *,
    samples: int,
    world_size: int,
) -> np.ndarray:
    parts, ids = [], []
    for rank in range(world_size):
        count = _subset_count(samples, rank, world_size)
        value = _rank_array(root, directory, condition, rank)
        if len(value) < count:
            raise RuntimeError(f"not enough rows for {condition} rank {rank}")
        parts.append(np.asarray(value[:count]))
        ids.append(local_ids_for_rank(samples, rank, world_size))
    joined_ids = np.concatenate(ids)
    order = np.argsort(joined_ids)
    return np.concatenate(parts, axis=0)[order]


def _load_low_frequency_subset(
    root: Path,
    condition: str,
    *,
    samples: int,
    world_size: int,
    grid_size: int,
    chunk_size: int = 8,
) -> np.ndarray:
    parts, ids = [], []
    for rank in range(world_size):
        count = _subset_count(samples, rank, world_size)
        images = _rank_array(root, "decoded", condition, rank)
        features = []
        for start in range(0, count, chunk_size):
            features.append(
                low_frequency_features(
                    images[start : min(start + chunk_size, count)], grid_size
                )
            )
        parts.append(np.concatenate(features))
        ids.append(local_ids_for_rank(samples, rank, world_size))
    joined_ids = np.concatenate(ids)
    order = np.argsort(joined_ids)
    return np.concatenate(parts, axis=0)[order]


def _extract_inception(
    extractor: torch.nn.Module, images: torch.Tensor
) -> torch.Tensor:
    return extractor(images.clamp(0, 1).mul(255).to(torch.uint8))[0].float()


def _append_space_metrics(
    rows: list[dict[str, object]],
    *,
    sample_ids: np.ndarray,
    scale: float,
    space: str,
    control: torch.Tensor,
    actual: torch.Tensor,
    counterfactual: torch.Tensor,
) -> None:
    metrics = sample_path_comparison(control, actual, counterfactual)
    for offset, sample_id in enumerate(sample_ids.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "scale": float(scale),
                "space": space,
                **{
                    name: float(value[offset]) for name, value in metrics.items()
                },
            }
        )


@torch.inference_mode()
def run_local(
    *,
    rae: torch.nn.Module,
    extractor: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    scales: tuple[float, ...],
    samples: int,
    control_scale: float,
    anchor_scale: float,
    batch_size: int,
    precision: str,
    low_frequency_grid: int,
    rank: int,
    world_size: int,
    device: torch.device,
    log_every: int,
) -> pd.DataFrame:
    count = _subset_count(samples, rank, world_size)
    ids = local_ids_for_rank(samples, rank, world_size)
    control_key, anchor_key = scale_key(control_scale), scale_key(anchor_scale)
    z_control = _rank_array(input_dir, "latents", control_key, rank)
    z_anchor = _rank_array(input_dir, "latents", anchor_key, rank)
    decoded_control = _rank_array(input_dir, "decoded", control_key, rank)
    roundtrip_control = _rank_array(input_dir, "roundtrip", control_key, rank)
    inception_control = _rank_array(input_dir, "inception", control_key, rank)
    rows: list[dict[str, object]] = []
    total_batches = math.ceil(count / batch_size)
    for scale in scales:
        condition = scale_key(scale)
        z_actual = _rank_array(input_dir, "latents", condition, rank)
        decoded_actual = _rank_array(input_dir, "decoded", condition, rank)
        roundtrip_actual = _rank_array(input_dir, "roundtrip", condition, rank)
        inception_actual = _rank_array(input_dir, "inception", condition, rank)
        line_inception_parts, line_lowfreq_parts = [], []
        interpolation = (scale - control_scale) / (anchor_scale - control_scale)
        for batch_index, start in enumerate(range(0, count, batch_size)):
            stop = min(start + batch_size, count)
            sample_ids = ids[start:stop]

            def tensor(value: np.ndarray, *, nhwc: bool = False) -> torch.Tensor:
                array = np.array(value[start:stop], dtype=np.float32, copy=True)
                result = torch.from_numpy(array).to(device)
                return result.permute(0, 3, 1, 2) if nhwc else result

            control_latent = tensor(z_control)
            anchor_latent = tensor(z_anchor)
            actual_latent = tensor(z_actual)
            line_latent = control_latent + float(interpolation) * (
                anchor_latent - control_latent
            )
            with autocast_context(precision):
                line_raw = rae.decode(line_latent).float()
                line_image = line_raw.clamp(0, 1)
                line_roundtrip = rae.encode(line_image).float()
            line_inception = _extract_inception(extractor, line_image)
            line_lowfreq = low_frequency_features(
                line_image.permute(0, 2, 3, 1).cpu().numpy(), low_frequency_grid
            )
            line_inception_parts.append(line_inception.cpu().numpy())
            line_lowfreq_parts.append(line_lowfreq)
            _append_space_metrics(
                rows,
                sample_ids=sample_ids,
                scale=scale,
                space="latent",
                control=control_latent,
                actual=actual_latent,
                counterfactual=line_latent,
            )
            _append_space_metrics(
                rows,
                sample_ids=sample_ids,
                scale=scale,
                space="decoded_pixel",
                control=tensor(decoded_control, nhwc=True),
                actual=tensor(decoded_actual, nhwc=True),
                counterfactual=line_image,
            )
            _append_space_metrics(
                rows,
                sample_ids=sample_ids,
                scale=scale,
                space="roundtrip",
                control=tensor(roundtrip_control),
                actual=tensor(roundtrip_actual),
                counterfactual=line_roundtrip,
            )
            _append_space_metrics(
                rows,
                sample_ids=sample_ids,
                scale=scale,
                space="decoded_inception",
                control=tensor(inception_control),
                actual=tensor(inception_actual),
                counterfactual=line_inception,
            )
            if rank == 0 and (
                (batch_index + 1) % log_every == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"[fixed interpolation s={scale:g}] "
                    f"{batch_index + 1}/{total_batches}",
                    flush=True,
                )
        atomic_save_npy(
            output_dir / "line_inception" / f"{condition}_rank{rank:02d}.npy",
            np.concatenate(line_inception_parts).astype(np.float32),
        )
        atomic_save_npy(
            output_dir / "line_lowfreq" / f"{condition}_rank{rank:02d}.npy",
            np.concatenate(line_lowfreq_parts).astype(np.float32),
        )
    return pd.DataFrame(rows)


def _probe_row(
    reference: np.ndarray,
    candidate: np.ndarray,
    test_mask: np.ndarray,
    *,
    scale: float,
    feature_space: str,
    path: str,
    ridge_ratio: float,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    train = ~test_mask
    weight, intercept, ridge = fit_feature_probe(
        reference, candidate, train, ridge_ratio
    )
    ref_scores = feature_probe_scores(reference[test_mask], weight, intercept)
    candidate_scores = feature_probe_scores(candidate[test_mask], weight, intercept)
    labels = np.concatenate(
        [
            np.zeros(len(ref_scores), dtype=np.int8),
            np.ones(len(candidate_scores), dtype=np.int8),
        ]
    )
    auc = float(
        roc_auc_score(labels, np.concatenate([ref_scores, candidate_scores]))
    )
    ci_low, ci_high = bootstrap_paired_auc(
        ref_scores,
        candidate_scores,
        bootstrap_repeats,
        bootstrap_seed,
    )
    return {
        "scale": float(scale),
        "feature_space": feature_space,
        "path": path,
        "auc": auc,
        "auc_separability": 0.5 + abs(auc - 0.5),
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "ridge": float(ridge),
    }


def _merge_line_features(
    output_dir: Path,
    directory: str,
    condition: str,
    *,
    samples: int,
    world_size: int,
) -> np.ndarray:
    parts, ids = [], []
    for rank in range(world_size):
        parts.append(
            np.load(
                output_dir / directory / f"{condition}_rank{rank:02d}.npy",
                allow_pickle=False,
            )
        )
        ids.append(local_ids_for_rank(samples, rank, world_size))
    joined_ids = np.concatenate(ids)
    order = np.argsort(joined_ids)
    return np.concatenate(parts)[order]


def compute_probe_table(
    *,
    input_dir: Path,
    output_dir: Path,
    scales: tuple[float, ...],
    samples: int,
    world_size: int,
    low_frequency_grid: int,
    ridge_ratio: float,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    protocol = np.load(input_dir / "sample_protocol.npz", allow_pickle=False)
    test_mask = np.asarray(protocol["test_mask"][:samples], dtype=bool)
    if test_mask.sum() < 2 or (~test_mask).sum() < 2:
        raise RuntimeError("train/test split is too small")
    reference_inception = _load_ordered_subset(
        input_dir,
        "inception",
        "real",
        samples=samples,
        world_size=world_size,
    ).astype(np.float32)
    reference_lowfreq = _load_low_frequency_subset(
        input_dir,
        "real",
        samples=samples,
        world_size=world_size,
        grid_size=low_frequency_grid,
    )
    rows = []
    for scale in scales:
        condition = scale_key(scale)
        actual_inception = _load_ordered_subset(
            input_dir,
            "inception",
            condition,
            samples=samples,
            world_size=world_size,
        ).astype(np.float32)
        line_inception = _merge_line_features(
            output_dir,
            "line_inception",
            condition,
            samples=samples,
            world_size=world_size,
        )
        actual_lowfreq = _load_low_frequency_subset(
            input_dir,
            condition,
            samples=samples,
            world_size=world_size,
            grid_size=low_frequency_grid,
        )
        line_lowfreq = _merge_line_features(
            output_dir,
            "line_lowfreq",
            condition,
            samples=samples,
            world_size=world_size,
        )
        for space_index, (feature_space, reference, actual, line) in enumerate((
            (
                "decoded_inception",
                reference_inception,
                actual_inception,
                line_inception,
            ),
            ("decoded_lowfreq", reference_lowfreq, actual_lowfreq, line_lowfreq),
        )):
            base_seed = (
                int(bootstrap_seed)
                + int(round(scale * 10_000))
                + 100_000 * space_index
            )
            rows.append(
                _probe_row(
                    reference,
                    actual,
                    test_mask,
                    scale=scale,
                    feature_space=feature_space,
                    path="actual_scale_trajectory",
                    ridge_ratio=ridge_ratio,
                    bootstrap_repeats=bootstrap_repeats,
                    bootstrap_seed=base_seed,
                )
            )
            rows.append(
                _probe_row(
                    reference,
                    line,
                    test_mask,
                    scale=scale,
                    feature_space=feature_space,
                    path="fixed_endpoint_chord",
                    ridge_ratio=ridge_ratio,
                    bootstrap_repeats=bootstrap_repeats,
                    bootstrap_seed=base_seed + 1,
                )
            )
    return pd.DataFrame(rows)


def plot_results(paired: pd.DataFrame, probes: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = {
        "latent": "#4c78a8",
        "roundtrip": "#e45756",
        "decoded_pixel": "#72b7b2",
        "decoded_inception": "#54a24b",
    }
    for space, values in paired.groupby("space"):
        curve = values.groupby("scale").actual_counterfactual_mismatch_over_actual_step.median()
        axes[0].plot(
            curve.index,
            curve.values,
            marker="o",
            linewidth=2,
            color=colors[space],
            label=space,
        )
    axes[0].set_title("Actual endpoint vs fixed endpoint chord")
    axes[0].set_xlabel("Internal-guidance scale")
    axes[0].set_ylabel("Median mismatch / actual path step")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    styles = {
        "actual_scale_trajectory": ("-", "o"),
        "fixed_endpoint_chord": ("--", "s"),
    }
    for (feature_space, path_name), values in probes.groupby(
        ["feature_space", "path"]
    ):
        linestyle, marker = styles[path_name]
        axes[1].plot(
            values.scale,
            values.auc_separability,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            label=f"{feature_space}: {path_name}",
        )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Decoded distribution separability")
    axes[1].set_xlabel("Internal-guidance scale")
    axes[1].set_ylabel("Held-out AUC separability (lower is closer)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--scale", action="append", type=float, dest="scales")
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--anchor-scale", type=float, default=1.78)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--low-frequency-grid", type=int, default=16)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = tuple(sorted(set(args.scales or DEFAULT_SCALES)))
    rank, world_size, device = _distributed(args.seed)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    available = {float(value) for value in manifest["scales"]}
    required = {*scales, float(args.control_scale), float(args.anchor_scale)}
    if not required.issubset(available):
        raise ValueError(f"missing source scales: {sorted(required - available)}")
    if int(manifest["world_size"]) != world_size:
        raise RuntimeError("torchrun world size must match source artifact")
    if args.samples <= 0 or args.samples > int(manifest["samples"]):
        raise ValueError("samples must be within source artifact")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    config = load_config(args.config.expanduser().resolve())
    install_raev2_decoder_config_compat()
    rae = (
        instantiate_from_config(config.stage_1)
        .to(device)
        .requires_grad_(False)
        .eval()
    )
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device)
    paired = run_local(
        rae=rae,
        extractor=extractor,
        input_dir=input_dir,
        output_dir=output_dir,
        scales=scales,
        samples=int(args.samples),
        control_scale=float(args.control_scale),
        anchor_scale=float(args.anchor_scale),
        batch_size=int(args.batch_size),
        precision=args.precision,
        low_frequency_grid=int(args.low_frequency_grid),
        rank=rank,
        world_size=world_size,
        device=device,
        log_every=int(args.log_every),
    )
    paired.to_csv(output_dir / f"paired_metrics_rank{rank:02d}.csv", index=False)
    dist.barrier()
    if rank == 0:
        all_paired = pd.concat(
            [
                pd.read_csv(output_dir / f"paired_metrics_rank{value:02d}.csv")
                for value in range(world_size)
            ],
            ignore_index=True,
        )
        all_paired.to_csv(output_dir / "paired_metrics.csv", index=False)
        paired_summary = (
            all_paired.groupby(["space", "scale"], as_index=False)
            .agg(
                sample_count=("sample_id", "count"),
                mismatch_over_actual_mean=(
                    "actual_counterfactual_mismatch_over_actual_step",
                    "mean",
                ),
                mismatch_over_actual_median=(
                    "actual_counterfactual_mismatch_over_actual_step",
                    "median",
                ),
                counterfactual_over_actual_norm_median=(
                    "counterfactual_over_actual_step_norm",
                    "median",
                ),
                path_cosine_mean=("actual_counterfactual_step_cosine", "mean"),
                path_cosine_median=("actual_counterfactual_step_cosine", "median"),
            )
        )
        paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)
        probes = compute_probe_table(
            input_dir=input_dir,
            output_dir=output_dir,
            scales=scales,
            samples=int(args.samples),
            world_size=world_size,
            low_frequency_grid=int(args.low_frequency_grid),
            ridge_ratio=float(args.ridge_ratio),
            bootstrap_repeats=int(args.bootstrap_repeats),
            bootstrap_seed=int(args.seed) + 10_000,
        )
        probes.to_csv(output_dir / "distribution_probe.csv", index=False)
        plot_results(all_paired, probes, output_dir / "fixed_interpolation_audit.png")
        summary = {
            "protocol": PROTOCOL,
            "input_artifact": str(input_dir),
            "samples": int(args.samples),
            "scales": list(scales),
            "control_scale": float(args.control_scale),
            "anchor_scale": float(args.anchor_scale),
            "precision": args.precision,
            "inference_only": True,
            "same_endpoint_direction_across_counterfactual_scales": True,
            "anchor_latent_identity_check": bool(
                (
                    paired_summary[
                        np.isclose(paired_summary.scale, float(args.anchor_scale))
                        & paired_summary.space.eq("latent")
                    ].mismatch_over_actual_median
                    < 1e-6
                ).all()
            ),
            "anchor_observation_recompute_calibration": {
                str(row.space): float(row.mismatch_over_actual_median)
                for row in paired_summary[
                    np.isclose(paired_summary.scale, float(args.anchor_scale))
                ].itertuples()
                if row.space != "latent"
            },
            "measurement_guardrail": (
                "Actual scale endpoints include recursive trajectory changes; chord "
                "endpoints hold the z1-to-z1.78 endpoint direction fixed. C2ST AUC is "
                "a separability diagnostic, not a metric or causal quality score."
            ),
        }
        temporary = output_dir / "summary.json.tmp"
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir / "summary.json")
        print(paired_summary.to_string(index=False))
        print(probes.to_string(index=False))
        print(f"wrote {output_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

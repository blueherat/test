"""Run the paired no-training RAE path-difference mechanism screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_cycle_direction_intervention import sample_rms  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset, load_cache_manifest  # noqa: E402
from experiments.rae_path_difference_intervention import (  # noqa: E402
    PATH_PAIRS,
    PathDifferenceThresholds,
    component_energy_fraction,
    feature_progress,
    fit_global_direction,
    matched_path_directions,
    path_difference_gate,
    projected_frechet_distance,
    random_unit_directions,
    rms_preserving_lerp,
    spherical_interpolate,
    spatial_components,
    standardized_sliced_wasserstein,
)
from experiments.rae_teacher_rollout_gap import configure_fp32, load_frozen_decoder  # noqa: E402


DEFAULT_PHASE0 = Path.home() / "data/eqvae/experiments/rae_decoder_risk_phase0"
DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_difference"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class PathDifferenceConfig:
    phase0_root: Path = DEFAULT_PHASE0
    audit_cache: Path = DEFAULT_CACHE
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "paired_cal128_test128_seed20260719"
    calibration_count: int = 128
    test_count: int = 128
    clean_reference_offset: int = 1024
    clean_reference_count: int = 128
    batch_size: int = 2
    projection_dim: int = 64
    swd_directions: int = 128
    seed: int = 20_260_719


def _distributed(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        raise RuntimeError("launch this study with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_fp32(int(seed) * world_size + rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def _finish() -> None:
    dist.barrier()
    dist.destroy_process_group()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _endpoint_path(root: Path, source: str) -> Path:
    path = root / f"0b_generated_latents_{source}_n256_s50.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_decoder(branch: Path, device: torch.device) -> torch.nn.Module:
    config = OmegaConf.load(branch / "config.yaml")
    stage_1 = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    return (
        load_frozen_decoder(stage_1)
        .to(device=device, dtype=torch.float32)
        .requires_grad_(False)
        .eval()
    )


def _load_inception_probe(device: torch.device) -> torch.nn.Module:
    from torchvision.models import Inception_V3_Weights, inception_v3

    return (
        inception_v3(weights=Inception_V3_Weights.DEFAULT)
        .to(device=device, dtype=torch.float32)
        .requires_grad_(False)
        .eval()
    )


@torch.no_grad()
def _decode(decoder: torch.nn.Module, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = latent.float()
    if bool(decoder.do_normalization):
        mean = decoder.latent_mean.to(z) if decoder.latent_mean is not None else 0.0
        var = decoder.latent_var.to(z) if decoder.latent_var is not None else 1.0
        z = z * torch.sqrt(var + float(decoder.eps)) + mean
    batch, channels, height, width = z.shape
    tokens = z.reshape(batch, channels, height * width).transpose(1, 2)
    output = decoder.decoder(tokens, drop_cls_token=False).logits
    image = decoder.decoder.unpatchify(output)
    raw = image * decoder.encoder_std.to(image) + decoder.encoder_mean.to(image)
    return raw.float(), raw.clamp(0, 1).float()


@torch.no_grad()
def _inception_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    resized = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = resized.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    std = resized.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.flatten(1))

    handle = model.avgpool.register_forward_hook(capture)
    try:
        logits = model((resized - mean) / std)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one Inception feature tensor, got {len(captured)}")
    return captured[0].float(), logits.float()


@torch.no_grad()
def _decode_probe(
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    latent: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_images = []
    images = []
    features = []
    logits = []
    for start in range(0, len(latent), batch_size):
        stop = min(start + batch_size, len(latent))
        raw, image = _decode(decoder, latent[start:stop].to(device))
        feature, logit = _inception_outputs(inception, image)
        raw_images.append(raw.cpu())
        images.append(image.cpu())
        features.append(feature.cpu())
        logits.append(logit.cpu())
    return (
        torch.cat(raw_images),
        torch.cat(images),
        torch.cat(features),
        torch.cat(logits),
    )


def _condition_alpha(condition: str) -> float:
    if condition.startswith(("own_a", "rms_a", "slerp_a")):
        return int(condition.rsplit("a", maxsplit=1)[1]) / 100.0
    if condition.startswith("good_") or condition.startswith("bad_"):
        return 0.25
    if condition.startswith("component_"):
        return 1.0
    raise ValueError(condition)


def _candidate_specs(
    good: torch.Tensor,
    bad: torch.Tensor,
    calibration_delta: torch.Tensor,
    *,
    seed: int,
) -> tuple[dict[str, tuple[torch.Tensor, str]], dict[str, torch.Tensor]]:
    delta = bad - good
    global_direction = fit_global_direction(calibration_delta)
    good_directions = matched_path_directions(delta, global_direction, seed=seed)
    bad_directions = matched_path_directions(-delta, -global_direction, seed=seed + 17)
    components = spatial_components(delta)
    specs: dict[str, tuple[torch.Tensor, str]] = {
        "own_a0": (good, "good"),
        "own_a25": (good + 0.25 * delta, "good"),
        "own_a50": (good + 0.50 * delta, "good"),
        "own_a75": (good + 0.75 * delta, "bad"),
        "own_a100": (bad, "bad"),
    }
    for alpha, suffix in ((0.25, "25"), (0.50, "50"), (0.75, "75")):
        anchor = "good" if alpha <= 0.5 else "bad"
        specs[f"rms_a{suffix}"] = (rms_preserving_lerp(good, bad, alpha), anchor)
        specs[f"slerp_a{suffix}"] = (spherical_interpolate(good, bad, alpha), anchor)
    for condition in ("shuffled", "random", "global", "opposite"):
        specs[f"good_{condition}"] = (good + 0.25 * good_directions[condition], "good")
        specs[f"bad_{condition}"] = (bad + 0.25 * bad_directions[condition], "bad")
    for name, value in components.items():
        specs[f"component_{name}"] = (good + value, "good")
    return specs, components


def _sample_rows(
    *,
    pair: str,
    condition: str,
    anchor: str,
    candidate: torch.Tensor,
    good: torch.Tensor,
    bad: torch.Tensor,
    raw: torch.Tensor,
    image: torch.Tensor,
    features: torch.Tensor,
    logits: torch.Tensor,
    good_image: torch.Tensor,
    bad_image: torch.Tensor,
    good_features: torch.Tensor,
    bad_features: torch.Tensor,
    labels: torch.Tensor,
    first_index: int,
    component_fraction: torch.Tensor | None,
) -> list[dict[str, object]]:
    anchor_latent = good if anchor == "good" else bad
    step = sample_rms(candidate - anchor_latent)
    relative_step = step / sample_rms(anchor_latent, 1e-12)
    progress = feature_progress(features, good_features, bad_features)
    feature_delta_good = sample_rms(features - good_features)
    feature_delta_bad = sample_rms(features - bad_features)
    image_delta_good = sample_rms(image - good_image)
    image_delta_bad = sample_rms(image - bad_image)
    cosine_good = F.cosine_similarity(features, good_features, dim=1)
    cosine_bad = F.cosine_similarity(features, bad_features, dim=1)
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    offsets = torch.arange(len(labels))
    target_nll = -log_probabilities[offsets, labels]
    target_probability = log_probabilities[offsets, labels].exp()
    target_top1 = logits.argmax(dim=1).eq(labels)
    clipping = ((raw < 0) | (raw > 1)).float().flatten(1).mean(dim=1)
    rows = []
    for index in range(len(candidate)):
        rows.append(
            {
                "pair": pair,
                "sample_index": int(first_index + index),
                "condition": condition,
                "anchor": anchor,
                "alpha": _condition_alpha(condition),
                "step_relative_rms": float(relative_step[index]),
                "feature_progress": float(progress[index]),
                "feature_delta_rms_to_good": float(feature_delta_good[index]),
                "feature_delta_rms_to_bad": float(feature_delta_bad[index]),
                "feature_cosine_to_good": float(cosine_good[index]),
                "feature_cosine_to_bad": float(cosine_bad[index]),
                "image_delta_rms_to_good": float(image_delta_good[index]),
                "image_delta_rms_to_bad": float(image_delta_bad[index]),
                "target_nll": float(target_nll[index]),
                "target_probability": float(target_probability[index]),
                "target_top1": bool(target_top1[index]),
                "decoded_pixel_clipping_fraction": float(clipping[index]),
                "component_energy_fraction": (
                    None if component_fraction is None else float(component_fraction[index])
                ),
            }
        )
    return rows


def _evaluate_pair(
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    good: torch.Tensor,
    bad: torch.Tensor,
    calibration_delta: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair: str,
    first_index: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    specs, components = _candidate_specs(
        good, bad, calibration_delta, seed=seed
    )
    good_raw, good_image, good_features, good_logits = _decode_probe(
        decoder, inception, good, batch_size=batch_size, device=device
    )
    bad_raw, bad_image, bad_features, bad_logits = _decode_probe(
        decoder, inception, bad, batch_size=batch_size, device=device
    )
    feature_payload = {"own_a0": good_features, "own_a100": bad_features}
    rows: list[dict[str, object]] = []
    endpoint_payloads = {
        "own_a0": (good_raw, good_image, good_features, good_logits),
        "own_a100": (bad_raw, bad_image, bad_features, bad_logits),
    }
    for condition, (candidate, anchor) in specs.items():
        if condition in endpoint_payloads:
            raw, image, features, logits = endpoint_payloads[condition]
        else:
            raw, image, features, logits = _decode_probe(
                decoder, inception, candidate, batch_size=batch_size, device=device
            )
            feature_payload[condition] = features
        component_fraction = None
        if condition.startswith("component_"):
            component = components[condition.removeprefix("component_")]
            component_fraction = component_energy_fraction(component, bad - good)
        rows.extend(
            _sample_rows(
                pair=pair,
                condition=condition,
                anchor=anchor,
                candidate=candidate,
                good=good,
                bad=bad,
                raw=raw,
                image=image,
                features=features,
                logits=logits,
                good_image=good_image,
                bad_image=bad_image,
                good_features=good_features,
                bad_features=bad_features,
                labels=labels,
                first_index=first_index,
                component_fraction=component_fraction,
            )
        )
    for condition in specs:
        if condition not in feature_payload:
            feature_payload[condition] = endpoint_payloads[condition][2]
    return pd.DataFrame(rows), feature_payload


def _reference_features(
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    cache: Path,
    *,
    start: int,
    count: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = CachedRAELatentDataset(cache, start=start, stop=start + count)
    latents = torch.stack([dataset[index][0] for index in range(len(dataset))])
    _, _, features, _ = _decode_probe(
        decoder, inception, latents, batch_size=batch_size, device=device
    )
    return features


def _distribution_table(
    output: Path,
    samples: pd.DataFrame,
    reference: torch.Tensor,
    *,
    world_size: int,
    projection_dim: int,
    swd_count: int,
    seed: int,
) -> pd.DataFrame:
    dimension = int(reference.shape[1])
    projection = random_unit_directions(dimension, projection_dim, seed + 401)
    directions = random_unit_directions(dimension, swd_count, seed + 409)
    rows = []
    for rank in range(world_size):
        payload = torch.load(output / f"features_rank{rank:02d}.pt", map_location="cpu", weights_only=True)
        pair = str(payload["pair"])
        features = payload["features"]
        for condition, candidate in features.items():
            selected = samples[(samples.pair == pair) & (samples.condition == condition)]
            if len(selected) != len(candidate):
                raise RuntimeError(f"sample/feature count mismatch for {pair}/{condition}")
            rows.append(
                {
                    "pair": pair,
                    "condition": condition,
                    "alpha": float(selected.alpha.iloc[0]),
                    "sample_count": len(candidate),
                    "projected_frechet": projected_frechet_distance(
                        reference, candidate, projection
                    ),
                    "swd": standardized_sliced_wasserstein(
                        reference, candidate, directions
                    ),
                    "target_nll_mean": float(selected.target_nll.mean()),
                    "target_nll_median": float(selected.target_nll.median()),
                    "target_top1_rate": float(selected.target_top1.mean()),
                    "feature_progress_median": float(selected.feature_progress.median()),
                    "image_delta_good_median": float(selected.image_delta_rms_to_good.median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["pair", "condition"]).reset_index(drop=True)


def _plot(
    distribution: pd.DataFrame,
    samples: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (good, bad) in enumerate(PATH_PAIRS):
        pair = f"{good}_to_{bad}"
        own = distribution[
            (distribution.pair == pair) & distribution.condition.str.startswith("own_a")
        ].sort_values("alpha")
        label = pair.replace("_to_", " -> ")
        axes[0, 0].plot(
            own.alpha,
            own.projected_frechet,
            marker="o",
            color=colors(index),
            label=label,
        )
        axes[0, 1].plot(
            own.alpha,
            own.swd,
            marker="o",
            color=colors(index),
            label=label,
        )
    axes[0, 0].set_title("Own path interpolation: projected Frechet")
    axes[0, 1].set_title("Own path interpolation: standardized SWD")
    for axis in axes[0]:
        axis.set_xlabel("alpha: good -> bad")
        axis.legend(frameon=False)

    progress_conditions = ("own_a25", "good_shuffled", "good_random", "good_global", "good_opposite")
    progress = (
        samples[samples.condition.isin(progress_conditions)]
        .groupby(["pair", "condition"])
        .feature_progress.median()
        .unstack()
        .reindex(index=[f"{a}_to_{b}" for a, b in PATH_PAIRS], columns=progress_conditions)
    )
    progress.plot.bar(ax=axes[1, 0], width=0.82)
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Good-anchor feature progress at alpha=0.25")
    axes[1, 0].set_ylabel("Median paired feature progress")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].legend(frameon=False, ncol=2)

    components = samples[samples.condition.str.startswith("component_")]
    component_summary = (
        components.groupby(["pair", "condition"], as_index=False)
        .agg(
            energy=("component_energy_fraction", "median"),
            progress=("feature_progress", "median"),
        )
    )
    markers = {"component_token_mean": "o", "component_spatial_residual": "s"}
    for index, (good, bad) in enumerate(PATH_PAIRS):
        pair = f"{good}_to_{bad}"
        for condition, marker in markers.items():
            selected = component_summary[
                (component_summary.pair == pair) & (component_summary.condition == condition)
            ].iloc[0]
            axes[1, 1].scatter(
                selected.energy,
                selected.progress,
                marker=marker,
                s=90,
                color=colors(index),
                label=f"{pair}: {condition.removeprefix('component_')}",
            )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Natural-energy spatial decomposition")
    axes[1, 1].set_xlabel("Median latent energy fraction")
    axes[1, 1].set_ylabel("Median paired feature progress")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("RAE paired path-difference intervention", fontsize=18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_geometry_followup(distribution: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    method_specs = {
        "linear": ("own_a", "o"),
        "RMS-preserving": ("rms_a", "s"),
        "spherical": ("slerp_a", "^"),
    }
    for axis, (good, bad) in zip(axes.flat, PATH_PAIRS):
        pair = f"{good}_to_{bad}"
        endpoint = distribution[
            (distribution.pair == pair)
            & distribution.condition.isin(("own_a0", "own_a100"))
        ]
        for label, (prefix, marker) in method_specs.items():
            middle = distribution[
                (distribution.pair == pair)
                & distribution.condition.str.startswith(prefix)
            ]
            selected = pd.concat([endpoint, middle], ignore_index=True).drop_duplicates(
                subset="alpha", keep="last"
            ).sort_values("alpha")
            axis.plot(
                selected.alpha,
                selected.projected_frechet,
                marker=marker,
                linewidth=2,
                label=label,
            )
        axis.set_title(pair.replace("_to_", " -> "))
        axis.set_xlabel("alpha: good -> bad")
        axis.set_ylabel("Projected Frechet (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Post-hoc path geometry controls", fontsize=18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: PathDifferenceConfig) -> Path | None:
    rank, world_size, device = _distributed(config.seed)
    if world_size != len(PATH_PAIRS):
        raise ValueError(f"expected {len(PATH_PAIRS)} GPUs, got {world_size}")
    good_source, bad_source = PATH_PAIRS[rank]
    pair = f"{good_source}_to_{bad_source}"
    output = config.output_root.expanduser().resolve() / config.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    root = config.phase0_root.expanduser().resolve()
    good_payload = torch.load(_endpoint_path(root, good_source), map_location="cpu", weights_only=True)
    bad_payload = torch.load(_endpoint_path(root, bad_source), map_location="cpu", weights_only=True)
    good_all = good_payload["latents"].float()
    bad_all = bad_payload["latents"].float()
    labels = good_payload["labels"].long()
    if not torch.equal(labels, bad_payload["labels"].long()):
        raise RuntimeError(f"labels differ within pair {pair}")
    total = config.calibration_count + config.test_count
    if total > len(good_all) or config.clean_reference_count != config.test_count:
        raise ValueError("endpoint counts are insufficient or clean/test counts differ")
    gathered_labels = [torch.empty_like(labels[:total].to(device)) for _ in range(world_size)]
    dist.all_gather(gathered_labels, labels[:total].to(device))
    if any(not torch.equal(gathered_labels[0], value) for value in gathered_labels[1:]):
        raise RuntimeError("path pairs do not share labels")

    decoder = _load_decoder(Path(str(good_payload["branch"])), device)
    inception = _load_inception_probe(device)
    cache = config.audit_cache.expanduser().resolve()
    manifest = load_cache_manifest(cache)
    if config.clean_reference_offset + config.clean_reference_count > int(manifest["sample_count"]):
        raise ValueError("clean reference slice exceeds cache")

    reference = torch.zeros(
        (config.clean_reference_count, 2048), device=device, dtype=torch.float32
    )
    if rank == 0:
        reference.copy_(
            _reference_features(
                decoder,
                inception,
                cache,
                start=config.clean_reference_offset,
                count=config.clean_reference_count,
                batch_size=config.batch_size,
                device=device,
            ).to(device)
        )
    dist.broadcast(reference, src=0)
    if rank == 0:
        torch.save({"features": reference.cpu()}, output / "clean_reference_features.pt")

    calibration_delta = (
        bad_all[: config.calibration_count] - good_all[: config.calibration_count]
    )
    start = config.calibration_count
    stop = start + config.test_count
    started = perf_counter()
    samples, features = _evaluate_pair(
        decoder,
        inception,
        good_all[start:stop],
        bad_all[start:stop],
        calibration_delta,
        labels[start:stop],
        pair=pair,
        first_index=start,
        batch_size=config.batch_size,
        seed=config.seed + 101 * rank,
        device=device,
    )
    samples.to_csv(output / f"samples_rank{rank:02d}.csv", index=False)
    torch.save(
        {"pair": pair, "features": features},
        output / f"features_rank{rank:02d}.pt",
    )
    print(f"rank{rank} {pair} complete in {(perf_counter() - started) / 60:.1f}m", flush=True)
    dist.barrier()
    if rank != 0:
        _finish()
        return None

    samples = pd.concat(
        [pd.read_csv(output / f"samples_rank{index:02d}.csv") for index in range(world_size)],
        ignore_index=True,
    )
    samples.to_csv(output / "samples.csv", index=False)
    distribution = _distribution_table(
        output,
        samples,
        reference.cpu(),
        world_size=world_size,
        projection_dim=config.projection_dim,
        swd_count=config.swd_directions,
        seed=config.seed,
    )
    distribution.to_csv(output / "distribution_metrics.csv", index=False)
    sample_summary = (
        samples.groupby(["pair", "condition", "alpha"], as_index=False)
        .agg(
            sample_count=("sample_index", "count"),
            step_relative_rms_median=("step_relative_rms", "median"),
            feature_progress_median=("feature_progress", "median"),
            feature_cosine_good_median=("feature_cosine_to_good", "median"),
            image_delta_good_median=("image_delta_rms_to_good", "median"),
            target_nll_mean=("target_nll", "mean"),
            target_top1_rate=("target_top1", "mean"),
            component_energy_fraction_median=("component_energy_fraction", "median"),
        )
    )
    sample_summary.to_csv(output / "sample_summary.csv", index=False)
    gate = path_difference_gate(distribution, samples)
    _plot(distribution, samples, output / "path_difference_intervention.png")
    _plot_geometry_followup(distribution, output / "path_geometry_followup.png")
    result = {
        "config": {
            **asdict(config),
            "phase0_root": str(config.phase0_root),
            "audit_cache": str(config.audit_cache),
            "output_root": str(config.output_root),
        },
        "gate": gate,
        "thresholds": asdict(PathDifferenceThresholds()),
        "protocol": {
            "status": "mechanism screen on previously studied endpoints; pass only authorizes fresh-seed confirmation",
            "endpoint_calibration_indices": [0, config.calibration_count],
            "endpoint_test_indices": [start, stop],
            "clean_reference_indices": [
                config.clean_reference_offset,
                config.clean_reference_offset + config.clean_reference_count,
            ],
            "alphas": list(ALPHAS),
            "controls": ["own", "shuffled", "random", "global", "opposite"],
            "posthoc_geometry_controls": ["RMS-preserving lerp", "spherical interpolation"],
            "posthoc_warning": "geometry controls diagnose midpoint norm contraction and were added after the primary screen",
            "direction_norm": "good/bad local controls match each sample path-difference RMS",
            "distribution_warning": "projected Frechet and SWD use torchvision features and 128 samples; they are screening proxies, not ADM-FID",
            "numerics": "fp32, TF32 disabled, frozen decoder and Inception",
        },
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("\nDistribution metrics:\n", distribution.to_string(index=False), flush=True)
    _finish()
    return output


def parse_args() -> PathDifferenceConfig:
    defaults = PathDifferenceConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-root", type=Path, default=defaults.phase0_root)
    parser.add_argument("--audit-cache", type=Path, default=defaults.audit_cache)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--calibration-count", type=int, default=defaults.calibration_count)
    parser.add_argument("--test-count", type=int, default=defaults.test_count)
    parser.add_argument("--clean-reference-offset", type=int, default=defaults.clean_reference_offset)
    parser.add_argument("--clean-reference-count", type=int, default=defaults.clean_reference_count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--projection-dim", type=int, default=defaults.projection_dim)
    parser.add_argument("--swd-directions", type=int, default=defaults.swd_directions)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    return PathDifferenceConfig(
        phase0_root=args.phase0_root,
        audit_cache=args.audit_cache,
        output_root=args.output_root,
        run_name=args.run_name,
        calibration_count=args.calibration_count,
        test_count=args.test_count,
        clean_reference_offset=args.clean_reference_offset,
        clean_reference_count=args.clean_reference_count,
        batch_size=args.batch_size,
        projection_dim=args.projection_dim,
        swd_directions=args.swd_directions,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())

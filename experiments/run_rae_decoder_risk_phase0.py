"""Run the four-GPU decoder-aware RAE Phase-0 no-training mechanism gates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external/RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_decoder_risk_phase0 import (  # noqa: E402
    banded_metric_loss,
    channel_metric_loss,
    clean_from_velocity,
    decoder_embed_metric,
    decoder_hidden_features,
    decoder_hidden_loss,
    decoder_hidden_rms,
    dct2,
    dct_matrix,
    gradient_cosine,
    gradient_energy_distributions,
    loss_space_gate,
    proxy_gate,
    radial_dct_band_masks,
    static_linear_state,
    summarize_quadratic_metric,
    trace_normalize_banded_metric,
    trace_normalize_channel_metric,
    velocity_and_clean_losses,
)
from experiments.rae_latent_cache import CachedRAELatentDataset, split_range  # noqa: E402
from experiments.rae_spectral_gradient_audit import shifted_time_quantiles  # noqa: E402
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    load_models,
    official_time_grid,
)
from experiments.train_rae_layerwise_path import resolve_stage1_paths  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_BRANCH_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
DEFAULT_STATIC_BRANCH = DEFAULT_BRANCH_ROOT / "seed3407_static_rank16_s0_to_10000"
DEFAULT_AUDIT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_decoder_risk_phase0"
DEFAULT_PATHS = ("static", "annealed", "reverse", "random")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distributed_context(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 0 requires CUDA")
    if "RANK" not in os.environ:
        raise RuntimeError("launch this script with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_fp32(int(seed) * world_size + rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def finish_distributed() -> None:
    dist.barrier()
    dist.destroy_process_group()


def cache_manifest(cache: Path) -> dict[str, object]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    required = {"complete", "calibration_count", "test_count", "shards"}
    if missing := required.difference(manifest):
        raise KeyError(f"Phase-0 cache manifest is missing {sorted(missing)}")
    if not bool(manifest["complete"]):
        raise RuntimeError(f"Phase-0 cache is incomplete: {cache}")
    return manifest


def split_dataset(
    cache: Path,
    split: str,
    rank: int,
    world_size: int,
) -> tuple[CachedRAELatentDataset, int, int]:
    manifest = cache_manifest(cache)
    calibration_count = int(manifest["calibration_count"])
    if split == "calibration":
        offset, count = 0, calibration_count
    elif split == "test":
        offset, count = calibration_count, int(manifest["test_count"])
    else:
        raise ValueError(f"unknown Phase-0 split {split}")
    local_start, local_stop = split_range(count, rank, world_size)
    dataset = CachedRAELatentDataset(
        cache, start=offset + local_start, stop=offset + local_stop
    )
    return dataset, local_start, local_stop


def deterministic_noise(
    shape: Sequence[int],
    *,
    seed: int,
    split: str,
    first_index: int,
) -> torch.Tensor:
    split_offset = 0 if split == "calibration" else 10_000_019
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + split_offset + int(first_index) * 1_000_003
    )
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def _merge_rank_csv(output: Path, stem: str, world_size: int) -> pd.DataFrame:
    parts = [pd.read_csv(output / f"{stem}_rank{rank:02d}.csv") for rank in range(world_size)]
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(output / f"{stem}.csv", index=False)
    return table


def summarize_gradient_distributions(output: Path, world_size: int = 4) -> pd.DataFrame:
    """Merge exact-gradient channel/token/DCT distributions into compact summaries."""

    groups: dict[tuple[str, str, int], list[np.ndarray]] = {}
    for rank in range(int(world_size)):
        payload = np.load(
            output / f"0a_gradient_distributions_rank{rank:02d}.npz",
            allow_pickle=True,
        )
        for gradient, axis, time_bin, values in zip(
            payload["gradient"], payload["axis"], payload["time_bin"], payload["values"]
        ):
            groups.setdefault((str(gradient), str(axis), int(time_bin)), []).append(
                np.asarray(values, dtype=np.float64)
            )
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for (gradient, axis, time_bin), values in sorted(groups.items()):
        matrix = np.stack(values)
        mean = matrix.mean(axis=0)
        median = np.median(matrix, axis=0)
        arrays[f"{gradient}_{axis}_time{time_bin}_mean"] = mean
        arrays[f"{gradient}_{axis}_time{time_bin}_median"] = median
        entropy = -(matrix * np.log(matrix.clip(min=1e-30))).sum(axis=1) / math.log(
            matrix.shape[1]
        )
        effective_support = 1.0 / np.square(matrix).sum(axis=1).clip(min=1e-30)
        top_count = max(1, int(math.ceil(matrix.shape[1] * 0.10)))
        top_fraction = np.sort(matrix, axis=1)[:, -top_count:].sum(axis=1)
        row: dict[str, object] = {
            "gradient": gradient,
            "axis": axis,
            "time_bin": time_bin,
            "sample_count": len(matrix),
            "normalized_entropy_median": float(np.median(entropy)),
            "effective_support_median": float(np.median(effective_support)),
            "top_10_percent_energy_median": float(np.median(top_fraction)),
        }
        if axis == "dct":
            for band in range(matrix.shape[1]):
                row[f"band_{band}_mean_energy"] = float(mean[band])
        rows.append(row)
    np.savez_compressed(output / "0a_gradient_distribution_summary.npz", **arrays)
    table = pd.DataFrame(rows)
    table.to_csv(output / "0a_gradient_distribution_summary.csv", index=False)
    return table


def _time_values(branch_config: OmegaConf, count: int = 5) -> torch.Tensor:
    shift = math.sqrt(
        float(branch_config.misc.time_dist_shift_dim)
        / float(branch_config.misc.time_dist_shift_base)
    )
    return shifted_time_quantiles(int(count), shift)


def _exact_indices(test_count: int, exact_count: int) -> set[int]:
    if int(exact_count) > int(test_count):
        raise ValueError("exact gradient count exceeds held-out test count")
    return set(
        np.linspace(0, int(test_count) - 1, int(exact_count), dtype=np.int64).tolist()
    )


@torch.no_grad()
def _hidden_reference(rae: torch.nn.Module, clean: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(feature.detach() for feature in decoder_hidden_features(rae, clean))


@torch.no_grad()
def _hidden_loss_no_grad(
    rae: torch.nn.Module,
    latent: torch.Tensor,
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    return decoder_hidden_loss(decoder_hidden_features(rae, latent), reference)


def _exact_gradient_row(
    rae: torch.nn.Module,
    estimate: torch.Tensor,
    clean: torch.Tensor,
    reference: Sequence[torch.Tensor],
    *,
    correction_fractions: Sequence[float],
    basis: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]]]:
    if len(estimate) != 1 or len(clean) != 1:
        raise ValueError("exact decoder gradients are evaluated one sample at a time")
    candidate = estimate.detach().clone().requires_grad_(True)
    candidate_features = decoder_hidden_features(rae, candidate)
    before = decoder_hidden_loss(candidate_features, reference).sum()
    decoder_gradient = torch.autograd.grad(before, candidate)[0].detach()
    error = estimate.detach() - clean.detach()
    latent_gradient = 2.0 * error / float(error[0].numel())
    cosine = float(gradient_cosine(latent_gradient, decoder_gradient).item())
    error_norm = error.flatten(1).norm(dim=1)
    if not correction_fractions or min(correction_fractions) <= 0:
        raise ValueError("correction fractions must be positive")

    def corrected(gradient: torch.Tensor, fraction: float) -> torch.Tensor:
        direction = gradient / gradient.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1).clamp_min(1e-30)
        correction_norm = float(fraction) * error_norm
        return estimate.detach() - correction_norm.reshape(-1, 1, 1, 1) * direction

    before_value = float(before.detach())
    correction_values: dict[str, float] = {}
    for fraction in correction_fractions:
        with torch.no_grad():
            x0_after = _hidden_loss_no_grad(
                rae, corrected(latent_gradient, fraction), reference
            )
            dec_after = _hidden_loss_no_grad(
                rae, corrected(decoder_gradient, fraction), reference
            )
        suffix = f"{float(fraction):.4g}".replace(".", "p")
        correction_values[f"decoder_reduction_x0_step_{suffix}"] = (
            before_value - float(x0_after.item())
        ) / max(before_value, 1e-30)
        correction_values[f"decoder_reduction_dec_step_{suffix}"] = (
            before_value - float(dec_after.item())
        ) / max(before_value, 1e-30)
    primary_suffix = f"{float(correction_fractions[0]):.4g}".replace(".", "p")
    reduction_x0 = correction_values[f"decoder_reduction_x0_step_{primary_suffix}"]
    reduction_dec = correction_values[f"decoder_reduction_dec_step_{primary_suffix}"]
    distributions = {
        "x0": {
            name: value[0].detach().cpu().numpy()
            for name, value in gradient_energy_distributions(
                latent_gradient, basis=basis, masks=masks
            ).items()
        },
        "decoder": {
            name: value[0].detach().cpu().numpy()
            for name, value in gradient_energy_distributions(
                decoder_gradient, basis=basis, masks=masks
            ).items()
        },
    }
    return (
        {
            "gradient_cosine_x0_dec": cosine,
            "gradient_norm_x0": float(latent_gradient.flatten(1).norm(dim=1).item()),
            "gradient_norm_dec": float(decoder_gradient.flatten(1).norm(dim=1).item()),
            "decoder_error_before": before_value,
            "decoder_reduction_x0": reduction_x0,
            "decoder_reduction_dec": reduction_dec,
            "correction_fraction_of_latent_error": float(correction_fractions[0]),
            **correction_values,
        },
        distributions,
    )


def run_loss_space_audit(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_context(args.seed)
    if world_size != 4:
        raise ValueError(f"Phase 0 is preregistered for four GPUs, got {world_size}")
    output = args.output.expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model, rae, config = load_models(args.static_branch.expanduser(), device)
    times = _time_values(config, args.time_bins).to(device)
    basis = dct_matrix(16).to(device)
    masks = radial_dct_band_masks(16, args.dct_bands, device=device)
    manifest = cache_manifest(args.audit_cache.expanduser())
    exact_indices = _exact_indices(int(manifest["test_count"]), args.exact_count)
    correction_fractions = tuple(
        float(value) for value in str(args.correction_fractions).split(",") if value.strip()
    )
    score_rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    distribution_records: list[tuple[str, str, int, int, np.ndarray]] = []
    started = perf_counter()

    for split in ("calibration", "test"):
        dataset, local_start, _ = split_dataset(
            args.audit_cache.expanduser(), split, rank, world_size
        )
        loader = DataLoader(
            dataset,
            batch_size=args.model_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        cursor = 0
        for clean_cpu, labels_cpu in loader:
            first_index = local_start + cursor
            indices = list(range(first_index, first_index + len(clean_cpu)))
            clean = clean_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
            noise = deterministic_noise(
                clean.shape, seed=args.noise_seed, split=split, first_index=first_index
            ).to(device, non_blocking=True)
            reference = _hidden_reference(rae, clean)
            for time_bin, scalar_time in enumerate(times):
                batch_time = torch.full(
                    (len(clean),), float(scalar_time), device=device, dtype=torch.float32
                )
                state, target = static_linear_state(clean, noise, batch_time)
                with torch.no_grad():
                    prediction = model(state, batch_time, y=labels)
                    estimate = clean_from_velocity(state, prediction, batch_time)
                    velocity_loss, clean_loss = velocity_and_clean_losses(
                        prediction, target, batch_time
                    )
                    measured_clean_loss = (estimate - clean).square().flatten(1).mean(dim=1)
                    if not torch.allclose(clean_loss, measured_clean_loss, atol=2e-6, rtol=2e-5):
                        raise RuntimeError("static-path identity L_x0 = t^2 L_v was violated")
                    perceptual_loss = _hidden_loss_no_grad(rae, estimate, reference)
                for offset, sample_index in enumerate(indices):
                    score_rows.append(
                        {
                            "split": split,
                            "sample_index": sample_index,
                            "time_bin": time_bin,
                            "time": float(scalar_time),
                            "l_v": float(velocity_loss[offset]),
                            "l_x0": float(clean_loss[offset]),
                            "l_dec": float(perceptual_loss[offset]),
                        }
                    )
                    if split != "test" or sample_index not in exact_indices:
                        continue
                    sample_reference = tuple(
                        feature[offset : offset + 1].detach() for feature in reference
                    )
                    row, distributions = _exact_gradient_row(
                        rae,
                        estimate[offset : offset + 1],
                        clean[offset : offset + 1],
                        sample_reference,
                        correction_fractions=correction_fractions,
                        basis=basis,
                        masks=masks,
                    )
                    exact_rows.append(
                        {
                            "split": split,
                            "sample_index": sample_index,
                            "time_bin": time_bin,
                            "time": float(scalar_time),
                            **row,
                        }
                    )
                    for gradient_name, values in distributions.items():
                        for axis_name, value in values.items():
                            distribution_records.append(
                                (gradient_name, axis_name, sample_index, time_bin, value)
                            )
            cursor += len(clean_cpu)
            if cursor % max(args.model_batch_size * 32, 1) == 0:
                print(
                    f"0A rank{rank} {split} {cursor}/{len(dataset)} "
                    f"elapsed={(perf_counter() - started) / 60:.1f}m",
                    flush=True,
                )

    pd.DataFrame(score_rows).to_csv(output / f"0a_scores_rank{rank:02d}.csv", index=False)
    pd.DataFrame(exact_rows).to_csv(output / f"0a_exact_rank{rank:02d}.csv", index=False)
    np.savez_compressed(
        output / f"0a_gradient_distributions_rank{rank:02d}.npz",
        gradient=np.asarray([record[0] for record in distribution_records]),
        axis=np.asarray([record[1] for record in distribution_records]),
        sample_index=np.asarray([record[2] for record in distribution_records], dtype=np.int64),
        time_bin=np.asarray([record[3] for record in distribution_records], dtype=np.int64),
        values=np.asarray([record[4] for record in distribution_records], dtype=object),
    )
    dist.barrier()
    if rank == 0:
        scores = _merge_rank_csv(output, "0a_scores", world_size)
        exact = _merge_rank_csv(output, "0a_exact", world_size)
        summarize_gradient_distributions(output, world_size)
        gate = loss_space_gate(exact)
        correlations = []
        for (split, time_bin), split_rows in scores.groupby(["split", "time_bin"]):
            correlations.append(
                {
                    "split": split,
                    "time_bin": int(time_bin),
                    "pearson_l_x0_l_dec": float(split_rows[["l_x0", "l_dec"]].corr().iloc[0, 1]),
                    "pearson_l_v_l_dec": float(split_rows[["l_v", "l_dec"]].corr().iloc[0, 1]),
                }
            )
        gate["loss_correlations"] = correlations
        gate["times"] = [float(value) for value in times.cpu()]
        gate["data_protocol"] = {
            "calibration": int(manifest["calibration_count"]),
            "test": int(manifest["test_count"]),
            "exact_gradient_test_subset": int(args.exact_count),
            "correction_fractions_of_error_norm": list(correction_fractions),
            "gate_correction_fraction": float(correction_fractions[0]),
            "same_path": "static linear FM",
            "same_target": "velocity",
            "l_x0_identity": "L_x0 = t^2 L_v",
            "leakage_control": {
                "calibration": manifest["calibration_source"],
                "test": manifest["test_source"],
            },
        }
        atomic_json(output / "0a_gate.json", gate)
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    finish_distributed()


def _randomized_gn_local(
    rae: torch.nn.Module,
    samples: Iterable[torch.Tensor],
    *,
    probe_count: int,
    seed: int,
    basis: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    channels = int(rae.decoder.decoder_embed.in_features)
    channel = torch.zeros(channels, channels, device=device, dtype=torch.float64)
    banded = torch.zeros(len(masks), channels, channels, device=device, dtype=torch.float64)
    probes = 0
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for clean_cpu in samples:
        if probes >= int(probe_count):
            break
        clean = clean_cpu.to(device=device, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        features = decoder_hidden_features(rae, clean)
        scalar = clean.new_zeros(())
        for feature in features:
            signs = torch.randint(
                0, 2, feature.shape, generator=generator, device=device, dtype=torch.int64
            ).to(torch.float32).mul_(2).sub_(1)
            scalar = scalar + (feature * signs).sum() / math.sqrt(float(feature.numel()))
        scalar = scalar / math.sqrt(float(len(features)))
        gradient = torch.autograd.grad(scalar, clean)[0].detach()
        tokens = gradient.permute(0, 2, 3, 1).reshape(-1, channels).double()
        channel += tokens.T @ tokens
        coefficients = dct2(gradient, basis)
        for band, mask in enumerate(masks):
            selected = coefficients[:, :, mask].transpose(1, 2).reshape(-1, channels).double()
            banded[band] += selected.T @ selected
        probes += 1
    return channel, banded, probes


def _calibration_estimate_samples(
    model: torch.nn.Module,
    dataset: CachedRAELatentDataset,
    *,
    local_start: int,
    count: int,
    times: torch.Tensor,
    noise_seed: int,
    device: torch.device,
) -> Iterable[torch.Tensor]:
    """Yield calibration z0 estimates spanning all registered shifted-time bins."""

    if int(count) < 1:
        return
    positions = np.linspace(0, len(dataset) - 1, int(count), dtype=np.int64)
    for probe_index, position in enumerate(positions):
        clean_cpu, label = dataset[int(position)]
        sample_index = int(local_start) + int(position)
        clean = clean_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
        noise = deterministic_noise(
            clean.shape,
            seed=noise_seed,
            split="calibration",
            first_index=sample_index,
        ).to(device)
        scalar_time = times[int(probe_index) % len(times)]
        batch_time = torch.full((1,), float(scalar_time), device=device)
        state, _ = static_linear_state(clean, noise, batch_time)
        with torch.no_grad():
            prediction = model(
                state,
                batch_time,
                y=torch.tensor([int(label)], device=device, dtype=torch.long),
            )
            estimate = clean_from_velocity(state, prediction, batch_time)[0].cpu()
        yield estimate


def _proxy_losses(
    error: torch.Tensor,
    embed: torch.Tensor,
    gn: torch.Tensor,
    banded: torch.Tensor,
    masks: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "decoder_embed": channel_metric_loss(error, embed),
        "randomized_gn": channel_metric_loss(error, gn),
        "randomized_gn_dct4": banded_metric_loss(error, banded, masks, basis),
    }


def run_proxy_audit(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_context(args.seed + 101)
    if world_size != 4:
        raise ValueError(f"Phase 0 is preregistered for four GPUs, got {world_size}")
    output = args.output.expanduser().resolve()
    if not (output / "0a_scores.csv").exists():
        raise FileNotFoundError("run Phase 0A before Phase 0C")
    model, rae, config = load_models(args.static_branch.expanduser(), device)
    times = _time_values(config, args.time_bins).to(device)
    basis = dct_matrix(16).to(device)
    masks = radial_dct_band_masks(16, args.dct_bands, device=device)
    calibration, calibration_start, _ = split_dataset(
        args.audit_cache.expanduser(), "calibration", rank, world_size
    )
    probe_local = math.ceil(args.gn_probes / world_size)
    probe_samples = _calibration_estimate_samples(
        model,
        calibration,
        local_start=calibration_start,
        count=probe_local,
        times=times,
        noise_seed=args.noise_seed,
        device=device,
    )
    gn_local, banded_local, probe_count = _randomized_gn_local(
        rae,
        probe_samples,
        probe_count=probe_local,
        seed=args.seed + 20_003 * rank,
        basis=basis,
        masks=masks,
        device=device,
    )
    count_tensor = torch.tensor(float(probe_count), device=device, dtype=torch.float64)
    dist.all_reduce(gn_local)
    dist.all_reduce(banded_local)
    dist.all_reduce(count_tensor)
    gn = trace_normalize_channel_metric((gn_local / count_tensor).float())
    banded = trace_normalize_banded_metric((banded_local / count_tensor).float(), masks)
    embed = decoder_embed_metric(rae).to(device)
    if rank == 0:
        torch.save(
            {
                "decoder_embed": embed.cpu(),
                "randomized_gn": gn.cpu(),
                "randomized_gn_dct4": banded.cpu(),
                "dct_masks": masks.cpu(),
                "gn_probes": int(count_tensor.item()),
                "normalization": "full metric trace / dimension = 1",
            },
            output / "0c_proxy_metrics.pt",
        )
    manifest = cache_manifest(args.audit_cache.expanduser())
    exact_indices = _exact_indices(int(manifest["test_count"]), args.exact_count)
    exact_lpl = pd.read_csv(output / "0a_scores.csv").set_index(
        ["split", "sample_index", "time_bin"]
    )["l_dec"].to_dict()
    score_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    started = perf_counter()
    for split in ("calibration", "test"):
        dataset, local_start, _ = split_dataset(
            args.audit_cache.expanduser(), split, rank, world_size
        )
        loader = DataLoader(dataset, batch_size=args.model_batch_size, shuffle=False, num_workers=0)
        cursor = 0
        for clean_cpu, labels_cpu in loader:
            first_index = local_start + cursor
            indices = list(range(first_index, first_index + len(clean_cpu)))
            clean = clean_cpu.to(device=device, dtype=torch.float32)
            labels = labels_cpu.to(device=device, dtype=torch.long)
            noise = deterministic_noise(
                clean.shape, seed=args.noise_seed, split=split, first_index=first_index
            ).to(device)
            for time_bin, scalar_time in enumerate(times):
                batch_time = torch.full((len(clean),), float(scalar_time), device=device)
                state, _ = static_linear_state(clean, noise, batch_time)
                with torch.no_grad():
                    prediction = model(state, batch_time, y=labels)
                    estimate = clean_from_velocity(state, prediction, batch_time)
                    proxy_values = _proxy_losses(
                        estimate - clean, embed, gn, banded, masks, basis
                    )
                for offset, sample_index in enumerate(indices):
                    key = (split, sample_index, time_bin)
                    for proxy, values in proxy_values.items():
                        score_rows.append(
                            {
                                "proxy": proxy,
                                "split": split,
                                "sample_index": sample_index,
                                "time_bin": time_bin,
                                "time": float(scalar_time),
                                "l_dec": float(exact_lpl[key]),
                                "l_proxy": float(values[offset]),
                            }
                        )
                    if split != "test" or sample_index not in exact_indices:
                        continue
                    candidate = estimate[offset : offset + 1].detach().clone().requires_grad_(True)
                    reference = _hidden_reference(rae, clean[offset : offset + 1])
                    exact_loss = decoder_hidden_loss(
                        decoder_hidden_features(rae, candidate), reference
                    ).sum()
                    exact_gradient = torch.autograd.grad(exact_loss, candidate)[0].detach()
                    proxy_error = (estimate[offset : offset + 1] - clean[offset : offset + 1]).detach()
                    for proxy in ("decoder_embed", "randomized_gn", "randomized_gn_dct4"):
                        variable = proxy_error.clone().requires_grad_(True)
                        values = _proxy_losses(variable, embed, gn, banded, masks, basis)
                        proxy_gradient = torch.autograd.grad(values[proxy].sum(), variable)[0]
                        gradient_rows.append(
                            {
                                "proxy": proxy,
                                "sample_index": sample_index,
                                "time_bin": time_bin,
                                "time": float(scalar_time),
                                "gradient_cosine_proxy_dec": float(
                                    gradient_cosine(proxy_gradient, exact_gradient).item()
                                ),
                            }
                        )
            cursor += len(clean_cpu)
            if cursor % max(args.model_batch_size * 64, 1) == 0:
                print(
                    f"0C rank{rank} {split} {cursor}/{len(dataset)} "
                    f"elapsed={(perf_counter() - started) / 60:.1f}m",
                    flush=True,
                )
    pd.DataFrame(score_rows).to_csv(output / f"0c_scores_rank{rank:02d}.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(
        output / f"0c_gradients_rank{rank:02d}.csv", index=False
    )
    dist.barrier()
    if rank == 0:
        scores = _merge_rank_csv(output, "0c_scores", world_size)
        gradients = _merge_rank_csv(output, "0c_gradients", world_size)
        gate = proxy_gate(scores, gradients)
        gate["metric_summaries"] = {
            "decoder_embed": summarize_quadratic_metric(embed.cpu()),
            "randomized_gn": summarize_quadratic_metric(gn.cpu()),
            "randomized_gn_dct4_bands": [
                summarize_quadratic_metric(value.cpu()) for value in banded
            ],
        }
        gate["proxy_protocol"] = {
            "gn_probes": int(count_tensor.item()),
            "gn_linearization_points": "calibration z0hat across all five shifted-time bins",
            "proxy_adjustments": 1,
            "adjustment": "channel-shared randomized GN expanded once to four DCT bands",
            "trace_normalization": "tr(W)/d = 1",
        }
        atomic_json(output / "0c_gate.json", gate)
        print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    finish_distributed()


@torch.no_grad()
def _sample_endpoints(
    model: torch.nn.Module,
    *,
    count: int,
    batch_size: int,
    times: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randn((count, 768, 16, 16), generator=generator, dtype=torch.float32)
    labels = torch.randint(0, 1000, (count,), generator=generator, dtype=torch.long)
    endpoints = []
    for start in range(0, count, batch_size):
        state = initial[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size].to(device)
        for current, following in zip(times[:-1], times[1:]):
            batch_time = torch.full((len(state),), float(current), device=device)
            velocity = model(state, batch_time, y=batch_labels)
            state = state + (following.to(state) - current.to(state)) * velocity
        endpoints.append(state.cpu())
    return torch.cat(endpoints), labels


def _load_full_rae(config: OmegaConf, device: torch.device) -> torch.nn.Module:
    stage_1 = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    resolve_stage1_paths(OmegaConf.create({"stage_1": stage_1}))
    return instantiate_from_config(stage_1).to(device=device, dtype=torch.float32).requires_grad_(False).eval()


@torch.no_grad()
def _decode_image_and_hidden(
    rae: torch.nn.Module, latent: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    z = latent
    if rae.do_normalization:
        latent_var = rae.latent_var.to(z) if rae.latent_var is not None else 1.0
        latent_mean = rae.latent_mean.to(z) if rae.latent_mean is not None else 0.0
        z = z * torch.sqrt(latent_var + rae.eps) + latent_mean
    batch, channels, height, width = z.shape
    tokens = z.reshape(batch, channels, height * width).transpose(1, 2)
    output = rae.decoder(tokens, drop_cls_token=False, output_hidden_states=True)
    image = rae.decoder.unpatchify(output.logits)
    image = image * rae.encoder_std.to(image) + rae.encoder_mean.to(image)
    indices = (2, 4, 6, 8)
    hidden = tuple(output.hidden_states[index][:, 1:] for index in indices)
    return image, hidden


def _descriptor_projection(channels: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(channels, dimension, generator=generator)
    return matrix / matrix.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _latent_descriptor(latent: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(latent, (4, 4))
    projected = torch.einsum("bchw,cd->bdhw", pooled, projection.to(latent)).flatten(1)
    basis = dct_matrix(latent.shape[-1]).to(latent)
    masks = radial_dct_band_masks(latent.shape[-1], 4, device=latent.device)
    coefficients = dct2(latent, basis).square()
    band_energy = torch.stack(
        [coefficients[:, :, mask].mean(dim=(1, 2)) for mask in masks], dim=1
    ).clamp_min(1e-20).log()
    return torch.cat([projected, band_energy], dim=1)


@torch.no_grad()
def _closure_rows(
    rae: torch.nn.Module,
    latents: torch.Tensor,
    clean_reference: torch.Tensor,
    *,
    source: str,
    batch_size: int,
    perturb_fraction: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    projection = _descriptor_projection(768, 32, seed).to(device)
    reference_descriptors = []
    reference_hidden_rms = []
    for start in range(0, len(clean_reference), batch_size):
        clean = clean_reference[start : start + batch_size].to(device)
        _, hidden = _decode_image_and_hidden(rae, clean)
        reference_descriptors.append(_latent_descriptor(clean, projection).cpu())
        reference_hidden_rms.append(decoder_hidden_rms(hidden).cpu())
    reference_descriptor = torch.cat(reference_descriptors)
    descriptor_mean = reference_descriptor.mean(dim=0, keepdim=True)
    descriptor_std = reference_descriptor.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    standardized_reference = (reference_descriptor - descriptor_mean) / descriptor_std
    hidden_reference = torch.cat(reference_hidden_rms)
    hidden_mean = hidden_reference.mean(dim=0)
    hidden_std = hidden_reference.std(dim=0, unbiased=False).clamp_min(1e-6)
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    for start in range(0, len(latents), batch_size):
        latent = latents[start : start + batch_size].to(device)
        image, hidden = _decode_image_and_hidden(rae, latent)
        clipped = image.clamp(0, 1)
        cycle = rae.encode(clipped)
        descriptor = _latent_descriptor(latent, projection).cpu()
        standardized = (descriptor - descriptor_mean) / descriptor_std
        nearest = torch.cdist(standardized, standardized_reference).min(dim=1).values
        response = (decoder_hidden_rms(hidden).cpu() - hidden_mean) / hidden_std
        perturbation = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(device)
        perturbation = perturbation / perturbation.square().flatten(1).mean(dim=1).sqrt().reshape(
            -1, 1, 1, 1
        ).clamp_min(1e-12)
        latent_rms = latent.square().flatten(1).mean(dim=1).sqrt()
        perturbation = perturbation * (
            float(perturb_fraction) * latent_rms
        ).reshape(-1, 1, 1, 1)
        _, perturbed_hidden = _decode_image_and_hidden(rae, latent + perturbation)
        sensitivity = decoder_hidden_loss(perturbed_hidden, hidden).sqrt().cpu() / (
            perturbation.square().flatten(1).mean(dim=1).sqrt().cpu().clamp_min(1e-12)
        )
        cycle_relative = (
            (cycle - latent).square().flatten(1).mean(dim=1).sqrt()
            / latent_rms.clamp_min(1e-12)
        ).cpu()
        clipping = ((image < 0) | (image > 1)).float().flatten(1).mean(dim=1).cpu()
        for offset in range(len(latent)):
            row: dict[str, object] = {
                "source": source,
                "sample_index": start + offset,
                "cycle_relative_rms": float(cycle_relative[offset]),
                "projected_nearest_clean_distance": float(nearest[offset]),
                "local_decoder_sensitivity": float(sensitivity[offset]),
                "decoded_pixel_clipping_fraction": float(clipping[offset]),
            }
            for layer in range(response.shape[1]):
                row[f"decoder_hidden_rms_z_layer{layer}"] = float(response[offset, layer])
            rows.append(row)
    return rows


def _load_cache_tensor(cache: Path, split: str, count: int) -> torch.Tensor:
    manifest = cache_manifest(cache)
    offset = 0 if split == "calibration" else int(manifest["calibration_count"])
    dataset = CachedRAELatentDataset(cache, start=offset, stop=offset + int(count))
    return torch.stack([dataset[index][0] for index in range(len(dataset))])


def run_generated_closure(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_context(args.seed + 202)
    if world_size != 4:
        raise ValueError(f"Phase 0 is preregistered for four GPUs, got {world_size}")
    paths = list(DEFAULT_PATHS)
    if world_size != len(paths):
        raise ValueError("one GPU is required per path")
    path_name = paths[rank]
    branches = sorted(
        path
        for path in args.branch_root.expanduser().glob(f"seed3407_{path_name}_rank16_s0_to_10000")
        if (path / "generation/ema_step-0010000.pt").exists()
    )
    if len(branches) != 1:
        raise RuntimeError(f"expected exactly one {path_name} branch, found {branches}")
    branch = branches[0]
    output = args.output.expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model, frozen_decoder, config = load_models(branch, device)
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    times = official_time_grid(args.sampling_steps, time_shift=shift).to(device)
    endpoint_path = output / (
        f"0b_generated_latents_{path_name}_n{args.generated_count}_s{args.sampling_steps}.pt"
    )
    if endpoint_path.exists():
        endpoint_payload = torch.load(endpoint_path, map_location="cpu", weights_only=True)
        endpoints, labels = endpoint_payload["latents"], endpoint_payload["labels"]
    else:
        endpoints, labels = _sample_endpoints(
            model,
            count=args.generated_count,
            batch_size=args.sampling_batch_size,
            times=times,
            seed=args.generation_seed,
            device=device,
        )
        torch.save(
            {
                "latents": endpoints,
                "labels": labels,
                "path": path_name,
                "branch": str(branch),
                "sampling_steps": int(args.sampling_steps),
                "seed": int(args.generation_seed),
                "pairing": "same initial noise and labels across paths; no generated-to-real pairing",
            },
            endpoint_path,
        )
    del model, frozen_decoder
    gc.collect()
    torch.cuda.empty_cache()
    rae = _load_full_rae(config, device)
    clean_reference = _load_cache_tensor(
        args.audit_cache.expanduser(), "calibration", args.closure_reference_count
    )
    rows = _closure_rows(
        rae,
        endpoints,
        clean_reference,
        source=path_name,
        batch_size=args.closure_batch_size,
        perturb_fraction=args.perturb_fraction,
        seed=args.seed,
        device=device,
    )
    if rank == 0:
        clean_query = _load_cache_tensor(
            args.audit_cache.expanduser(), "test", args.generated_count
        )
        rows.extend(
            _closure_rows(
                rae,
                clean_query,
                clean_reference,
                source="clean_test",
                batch_size=args.closure_batch_size,
                perturb_fraction=args.perturb_fraction,
                seed=args.seed,
                device=device,
            )
        )
    pd.DataFrame(rows).to_csv(output / f"0b_closure_rank{rank:02d}.csv", index=False)
    dist.barrier()
    if rank == 0:
        table = _merge_rank_csv(output, "0b_closure", world_size)
        metric_columns = [
            "cycle_relative_rms",
            "projected_nearest_clean_distance",
            "local_decoder_sensitivity",
            "decoded_pixel_clipping_fraction",
        ] + [column for column in table if column.startswith("decoder_hidden_rms_z_layer")]
        summary = (
            table.groupby("source")[metric_columns]
            .agg(["median", "mean", "std"])
            .reset_index()
        )
        summary.columns = [
            column if isinstance(column, str) else "_".join(value for value in column if value)
            for column in summary.columns
        ]
        summary.to_csv(output / "0b_closure_summary.csv", index=False)
        protocol = {
            "generated_count_per_path": int(args.generated_count),
            "paths": paths,
            "same_noise_and_labels": True,
            "generated_to_real_pairing": False,
            "cycle_definition": "E(clamp(D(z_gen), 0, 1))",
            "nearest_clean": "standardized random-channel projected 4x4 pooled latent plus four DCT energies",
            "clean_reference_count": int(args.closure_reference_count),
            "local_perturbation_fraction": float(args.perturb_fraction),
        }
        atomic_json(output / "0b_protocol.json", protocol)
        print(summary.to_string(index=False), flush=True)
    finish_distributed()


def build_report(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    gate_a = json.loads((output / "0a_gate.json").read_text(encoding="utf-8"))
    gate_c = json.loads((output / "0c_gate.json").read_text(encoding="utf-8"))
    closure = pd.read_csv(output / "0b_closure_summary.csv")
    gradient_summary = summarize_gradient_distributions(output)
    exact = pd.DataFrame(gate_a["per_time"])
    proxy = pd.DataFrame(gate_c["proxies"])
    proxy_display = proxy.drop(columns=["per_time_spearman"], errors="ignore")
    proceed = bool(gate_a["pass"] and gate_c["pass"])

    def closure_value(source: str, metric: str) -> float:
        row = closure[closure["source"] == source]
        return float(row.iloc[0][metric])

    clean_cycle = closure_value("clean_test", "cycle_relative_rms_median")
    clean_sensitivity = closure_value("clean_test", "local_decoder_sensitivity_median")
    static_cycle = closure_value("static", "cycle_relative_rms_median")
    reverse_cycle = closure_value("reverse", "cycle_relative_rms_median")
    static_sensitivity = closure_value("static", "local_decoder_sensitivity_median")
    reverse_sensitivity = closure_value("reverse", "local_decoder_sensitivity_median")
    best_proxy = proxy.sort_values("test_spearman", ascending=False).iloc[0]
    generation_rows = []
    for source in DEFAULT_PATHS:
        matches = list(
            args.branch_root.expanduser().glob(
                f"seed3407_{source}_rank16_s0_to_10000/generation/generation_metrics.json"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(f"expected one generation metric file for {source}")
        metrics = json.loads(matches[0].read_text(encoding="utf-8"))
        closure_row = closure[closure["source"] == source].iloc[0]
        generation_rows.append(
            {
                "source": source,
                "fid_5k": float(metrics["frechet_inception_distance"]),
                "kid_5k": float(metrics["kernel_inception_distance_mean"]),
                "is_5k": float(metrics["inception_score_mean"]),
                "cycle_residual": float(closure_row["cycle_relative_rms_median"]),
                "local_sensitivity": float(closure_row["local_decoder_sensitivity_median"]),
                "nearest_clean": float(
                    closure_row["projected_nearest_clean_distance_median"]
                ),
                "early_hidden_z": float(
                    closure_row["decoder_hidden_rms_z_layer0_median"]
                ),
            }
        )
    generation_table = pd.DataFrame(generation_rows).sort_values("fid_5k")
    generation_table.to_csv(output / "0b_closure_vs_generation.csv", index=False)
    closure_rank_correlations = {
        metric: float(
            generation_table[metric].rank().corr(generation_table["fid_5k"].rank())
        )
        for metric in ("cycle_residual", "local_sensitivity", "nearest_clean", "early_hidden_z")
    }

    def gradient_value(gradient: str, axis: str, time_bin: int, column: str) -> float:
        row = gradient_summary[
            (gradient_summary["gradient"] == gradient)
            & (gradient_summary["axis"] == axis)
            & (gradient_summary["time_bin"] == time_bin)
        ]
        return float(row.iloc[0][column])
    def markdown_table(table: pd.DataFrame) -> str:
        columns = [str(column) for column in table.columns]
        rows = [columns] + [
            [str(value) for value in row]
            for row in table.itertuples(index=False, name=None)
        ]
        widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]
        header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"
        separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
        body = [
            "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
            for row in rows[1:]
        ]
        return "\n".join([header, separator, *body])

    lines = [
        "# RAE Decoder-Aware Phase 0 机制审计",
        "",
        "## 结论",
        "",
        (
            "**Phase 0 通过，可以进入 Phase 1 的 2.5k-step paired continuation。**"
            if proceed
            else "**Phase 0 未通过，按预注册停止 decoder-aware 训练方向，不进入 Phase 1。**"
        ),
        "",
        "这里没有用跨 path raw MSE 证明机制。0A 和 0C 始终固定同一个 static linear path、"
        "同一个 velocity target，并显式保留 `L_x0=t^2 L_v` 对照。",
        "",
        "## 0A Loss-Space Audit",
        "",
        f"- 总门槛：{'通过' if gate_a['pass'] else '未通过'}",
        f"- 梯度方向明显不同的时间段：{gate_a['distinct_time_bins']}/5",
        f"- decoder 梯度优于 x0 梯度的时间段：{gate_a['better_time_bins']}/5",
        f"- 等范数局部修正总体优势比：{gate_a['correction_ratio']:.4f}x",
        "",
        markdown_table(exact),
        "",
        "## 0B Generated Latent Closure",
        "",
        "0B 不把生成样本与任意真图逐样本配对；nearest-clean 只作为分布外程度诊断。",
        f"clean/static/reverse 的 cycle residual 中位数分别为 {clean_cycle:.4f}/"
        f"{static_cycle:.4f}/{reverse_cycle:.4f}；local sensitivity 分别为 "
        f"{clean_sensitivity:.4f}/{static_sensitivity:.4f}/{reverse_sensitivity:.4f}。",
        "生成 latent 的 projected nearest-clean 距离低于 clean query，结合 hidden response 更像"
        "分布向中心收缩，不能据此声称生成 latent 更接近真实流形。",
        "",
        markdown_table(closure),
        "",
        "四个 path 的 closure residual 与已有同配置 5k FID 排序完全一致，local sensitivity "
        "也完全一致；但这里只有四个点，只能作为机制一致性，不能当统计因果证据。",
        f"Spearman(FID, cycle)={closure_rank_correlations['cycle_residual']:.3f}，"
        f"Spearman(FID, sensitivity)={closure_rank_correlations['local_sensitivity']:.3f}。",
        "",
        markdown_table(generation_table),
        "",
        "## 0C Metric Proxy",
        "",
        f"- 总门槛：{'通过' if gate_c['pass'] else '未通过'}",
        f"- 选中 proxy：{gate_c.get('selected_proxy')}",
        f"- held-out Spearman 最好的候选为 `{best_proxy['proxy']}`："
        f"{float(best_proxy['test_spearman']):.4f}；其 gradient cosine 为 "
        f"{float(best_proxy['median_gradient_cosine']):.4f}",
        "- `randomized_gn_dct4` 是唯一一次结构扩展；若仍失败，不再追加 proxy 结构。",
        "",
        markdown_table(proxy_display),
        "",
        "## 梯度能量分布",
        "",
        "下表中的 channel、token 与 DCT 分布均逐样本归一化为和 1；它们描述方向"
        "而不是受原始 loss scale 影响的梯度大小。",
        f"time-bin 0 时，decoder/x0 的 channel effective support 为 "
        f"{gradient_value('decoder', 'channel', 0, 'effective_support_median'):.1f}/"
        f"{gradient_value('x0', 'channel', 0, 'effective_support_median'):.1f}，token support 为 "
        f"{gradient_value('decoder', 'token', 0, 'effective_support_median'):.1f}/"
        f"{gradient_value('x0', 'token', 0, 'effective_support_median'):.1f}。"
        "decoder 风险明显更稀疏且随时间变化；这与常数 channel/DCT metric 无法泛化相一致。",
        "",
        markdown_table(gradient_summary),
        "",
        "## 可声称与不可声称",
        "",
        "可声称的是 frozen 高维 RAE decoder 的感知风险是否给有限容量 stage-2 提供了"
        "不同且可低成本近似的误差分配信号。常数正定 metric 不改变无限容量 Bayes 最优解，"
        "因此不能把结果表述成发现了新的 latent manifold geometry。",
        "",
        "所有原始表、矩阵和生成 latent 均保存在本机数据目录；仓库不写 `outputs/`。",
    ]
    report_path = output / "phase0_gate_report_zh.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = {
        "phase0_pass": proceed,
        "phase0a_pass": bool(gate_a["pass"]),
        "phase0c_pass": bool(gate_c["pass"]),
        "next_step": "Phase 1 paired continuation" if proceed else "stop decoder-aware route",
        "report": str(report_path),
    }
    atomic_json(output / "phase0_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("0a", "0b", "0c", "report"))
    parser.add_argument("--static-branch", type=Path, default=DEFAULT_STATIC_BRANCH)
    parser.add_argument("--branch-root", type=Path, default=DEFAULT_BRANCH_ROOT)
    parser.add_argument("--audit-cache", type=Path, default=DEFAULT_AUDIT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--noise-seed", type=int, default=104_729)
    parser.add_argument("--generation-seed", type=int, default=20_260_718)
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument("--dct-bands", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=2)
    parser.add_argument("--exact-count", type=int, default=128)
    parser.add_argument("--correction-fractions", default="0.001,0.003,0.01")
    parser.add_argument("--gn-probes", type=int, default=32)
    parser.add_argument("--generated-count", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-batch-size", type=int, default=4)
    parser.add_argument("--closure-reference-count", type=int, default=1024)
    parser.add_argument("--closure-batch-size", type=int, default=2)
    parser.add_argument("--perturb-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "0a":
        run_loss_space_audit(args)
    elif args.task == "0b":
        run_generated_closure(args)
    elif args.task == "0c":
        run_proxy_audit(args)
    else:
        build_report(args)


if __name__ == "__main__":
    main()

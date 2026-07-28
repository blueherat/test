"""Check whether the latent trust-spectrum crossover survives frozen decoding."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.analyze_rae_predictability_gain import MATCHED_PAIRS  # noqa: E402
from experiments.evaluate_rae_spc_multiseed import DEFAULT_SEEDS, branch_name  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_teacher_rollout_gap import official_time_grid  # noqa: E402
from experiments.run_rae_decoder_risk_phase0 import (  # noqa: E402
    _decode_image_and_hidden,
    _load_full_rae,
)
from experiments.run_rae_latent_trust_rollout import (  # noqa: E402
    rollout_condition_endpoints,
    selected_block_names,
)
from experiments.run_rae_path_gradient_interference import _load_basis  # noqa: E402
from experiments.run_rae_path_schedule_closure import _load_stage2_model  # noqa: E402
from experiments.run_rae_spc_cross_path_study import evaluation_path_kwargs  # noqa: E402
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_BASES = DEFAULT_RESULTS / "evaluation/predictability_basis_v1/bases.pt"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "evaluation/latent_trust_decoder_spotcheck_v1"


@torch.no_grad()
def per_sample_lpips(
    metric: LearnedPerceptualImagePatchSimilarity,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Evaluate LPIPS without accumulating TorchMetrics state across calls."""

    values = metric.net(first, second, normalize=True)
    return values.reshape(len(first), -1).mean(dim=1)


@torch.no_grad()
def paired_image_metrics(
    clean_image: torch.Tensor,
    base_image: torch.Tensor,
    perturbed_image: torch.Tensor,
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> dict[str, torch.Tensor]:
    clean = clean_image.float().clamp(0.0, 1.0)
    base = base_image.float().clamp(0.0, 1.0)
    perturbed = perturbed_image.float().clamp(0.0, 1.0)
    base_clean_l1 = (base - clean).abs().flatten(1).mean(1)
    perturbed_clean_l1 = (perturbed - clean).abs().flatten(1).mean(1)
    base_clean_lpips = per_sample_lpips(lpips, base, clean)
    perturbed_clean_lpips = per_sample_lpips(lpips, perturbed, clean)
    return {
        "image_shift_l1": (perturbed - base).abs().flatten(1).mean(1),
        "image_shift_lpips": per_sample_lpips(lpips, perturbed, base),
        "base_clean_l1": base_clean_l1,
        "perturbed_clean_l1": perturbed_clean_l1,
        "clean_l1_increase": perturbed_clean_l1 - base_clean_l1,
        "base_clean_lpips": base_clean_lpips,
        "perturbed_clean_lpips": perturbed_clean_lpips,
        "clean_lpips_increase": perturbed_clean_lpips - base_clean_lpips,
    }


@torch.no_grad()
def decode_latents(
    rae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    images = []
    for start in range(0, len(latents), int(batch_size)):
        batch = latents[start : start + int(batch_size)].to(device)
        image, _ = _decode_image_and_hidden(rae, batch)
        images.append(image.float().clamp(0.0, 1.0).cpu())
    return torch.cat(images)


def worker_command(args: argparse.Namespace, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--results",
        str(args.results),
        "--bases",
        str(args.bases),
        "--output",
        str(args.output),
        "--seeds",
        str(seed),
        "--endpoint",
        str(args.endpoint),
        "--switch-step",
        str(args.switch_step),
        "--cache-start",
        str(args.cache_start),
        "--count",
        str(args.count),
        "--data-batch-size",
        str(args.data_batch_size),
        "--model-batch-size",
        str(args.model_batch_size),
        "--decode-batch-size",
        str(args.decode_batch_size),
        "--probe-seed",
        str(args.probe_seed),
        "--times",
        args.times,
        "--amplitude",
        str(args.amplitude),
        "--sampling-steps",
        str(args.sampling_steps),
    ]


def _decode_rollout_payloads(
    rae: torch.nn.Module,
    lpips: LearnedPerceptualImagePatchSimilarity,
    clean: torch.Tensor,
    payloads: list[dict[str, object]],
    *,
    seed: int,
    device: torch.device,
    decode_batch_size: int,
) -> list[dict[str, float | int | str]]:
    clean_images = decode_latents(
        rae, clean, device=device, batch_size=decode_batch_size
    )
    rows: list[dict[str, float | int | str]] = []
    for payload in payloads:
        start = int(payload["start"])
        stop = int(payload["stop"])
        endpoints = payload["endpoints"]
        if not isinstance(endpoints, torch.Tensor):
            raise TypeError("rollout endpoints must be a tensor")
        flat_images = decode_latents(
            rae,
            endpoints.flatten(0, 1),
            device=device,
            batch_size=decode_batch_size,
        )
        endpoint_images = flat_images.reshape(
            *endpoints.shape[:2], *flat_images.shape[1:]
        )
        base_image = endpoint_images[0].to(device)
        clean_image = clean_images[start:stop].to(device)
        base_latent = endpoints[0]
        conditions = payload["conditions"]
        deltas = payload["deltas"]
        if not isinstance(conditions, list) or not isinstance(deltas, list):
            raise TypeError("rollout conditions and deltas must be lists")
        for condition_index, (condition, delta) in enumerate(
            zip(conditions, deltas), start=1
        ):
            if not isinstance(condition, dict) or not isinstance(delta, torch.Tensor):
                raise TypeError("malformed rollout condition")
            perturbed_latent = endpoints[condition_index]
            input_energy = delta.square().flatten(1).mean(1)
            endpoint_shift = (perturbed_latent - base_latent).square().flatten(1).mean(1)
            image_metrics = paired_image_metrics(
                clean_image,
                base_image,
                endpoint_images[condition_index].to(device),
                lpips,
            )
            for sample_index in range(stop - start):
                denominator = float(input_energy[sample_index].clamp_min(1e-20))
                row: dict[str, float | int | str] = {
                    "seed": int(seed),
                    "sample_index": start + sample_index,
                    "target_time": float(payload["target_time"]),
                    "actual_time": float(payload["actual_time"]),
                    "basis": str(condition["basis"]),
                    "amplitude": float(condition["amplitude"]),
                    "input_energy": float(input_energy[sample_index]),
                    "endpoint_shift_energy": float(endpoint_shift[sample_index]),
                    "endpoint_shift_gain": float(endpoint_shift[sample_index])
                    / denominator,
                }
                row.update(
                    {
                        name: float(values[sample_index])
                        for name, values in image_metrics.items()
                    }
                )
                rows.append(row)
        print(
            f"seed={seed} decoded samples={start}:{stop} "
            f"time={float(payload['target_time']):.2f}",
            flush=True,
        )
    return rows


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(args.seeds)
    configure_determinism(args.probe_seed)
    root = args.results.expanduser().resolve()
    static_branch = root / branch_name(
        seed, "static", args.endpoint, args.switch_step
    )
    spc_branch = root / branch_name(seed, "spc", args.endpoint, args.switch_step)
    manifest = json.loads(
        (static_branch / "manifest.json").read_text(encoding="utf-8")
    )
    spc_manifest = json.loads(
        (spc_branch / "manifest.json").read_text(encoding="utf-8")
    )
    dataset = CachedRAELatentDataset(
        Path(str(manifest["latent_cache"])),
        start=args.cache_start,
        stop=args.cache_start + args.count,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    clean = torch.stack([sample[0] for sample in samples]).float()
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)

    model, config = _load_stage2_model(
        static_branch, args.endpoint, device, weight_source="model"
    )
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim)
        / float(config.misc.time_dist_shift_base)
    )
    full_times = official_time_grid(args.sampling_steps, time_shift=shift).to(device)
    reference_basis = _load_basis(manifest).to(device)
    basis_payload = torch.load(
        args.bases.expanduser(), map_location="cpu", weights_only=False
    )
    selected = selected_block_names()
    bases = {
        name: basis.to(device=device, dtype=torch.float32)
        for name, basis in basis_payload["blocks"].items()
        if name in selected
    }
    if set(bases) != selected:
        raise KeyError(f"missing selected bases: {sorted(selected.difference(bases))}")
    paths = evaluation_path_kwargs(spc_manifest)
    target_times = tuple(float(value) for value in args.times.split(",") if value)
    payloads: list[dict[str, object]] = []
    for start in range(0, len(clean), args.data_batch_size):
        stop = min(start + args.data_batch_size, len(clean))
        batch_clean = clean[start:stop].to(device)
        batch_labels = labels[start:stop].to(device)
        batch_noise = noise[start:stop].to(device)
        for target_time in target_times:
            actual_time, conditions, deltas, endpoints = rollout_condition_endpoints(
                model,
                batch_clean,
                batch_labels,
                batch_noise,
                reference_basis,
                bases,
                paths,
                full_times,
                target_time=target_time,
                amplitudes=(float(args.amplitude),),
                model_batch_size=args.model_batch_size,
            )
            payloads.append(
                {
                    "start": start,
                    "stop": stop,
                    "target_time": target_time,
                    "actual_time": actual_time,
                    "conditions": conditions,
                    "deltas": [delta.cpu() for delta in deltas],
                    "endpoints": endpoints.cpu(),
                }
            )
        print(f"seed={seed} rolled out {stop}/{len(clean)}", flush=True)

    del model, reference_basis, bases
    gc.collect()
    torch.cuda.empty_cache()
    rae = _load_full_rae(config, device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device).requires_grad_(False).eval()
    rows = _decode_rollout_payloads(
        rae,
        lpips,
        clean,
        payloads,
        seed=seed,
        device=device,
        decode_batch_size=args.decode_batch_size,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"decoder_samples_seed{seed}.csv", index=False)
    print(f"completed seed={seed}", flush=True)


def pair_decoder_ratios(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for high, low in MATCHED_PAIRS:
        high_rows = per_seed[per_seed["basis"] == high].set_index(
            ["seed", "target_time"]
        )
        low_rows = per_seed[per_seed["basis"] == low].set_index(
            ["seed", "target_time"]
        )
        common = high_rows.index.intersection(low_rows.index)
        for seed, target_time in common:
            high_row = high_rows.loc[(seed, target_time)]
            low_row = low_rows.loc[(seed, target_time)]
            row: dict[str, float | int | str] = {
                "higher_predictability_basis": high,
                "lower_predictability_basis": low,
                "seed": int(seed),
                "target_time": float(target_time),
            }
            for field in (
                "endpoint_shift_gain",
                "image_shift_l1",
                "image_shift_lpips",
                "decoder_l1_secant",
                "decoder_lpips_secant",
            ):
                row[f"{field}_ratio"] = float(high_row[field]) / max(
                    float(low_row[field]), 1e-20
                )
            for field in ("clean_l1_increase", "clean_lpips_increase"):
                row[f"{field}_difference"] = float(high_row[field]) - float(
                    low_row[field]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(pairs: pd.DataFrame, output: Path) -> None:
    fig, axes_grid = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
    axes = axes_grid.flatten()
    colors = ("#2678a8", "#2f855a", "#8b5a9f")
    fields = (
        ("endpoint_shift_gain_ratio", "latent endpoint gain ratio"),
        ("image_shift_l1_ratio", "decoded L1 shift ratio"),
        ("image_shift_lpips_ratio", "decoded LPIPS shift ratio"),
        ("decoder_l1_secant_ratio", "decoder L1 secant ratio"),
        ("decoder_lpips_secant_ratio", "decoder LPIPS secant ratio"),
    )
    for axis, (field, title) in zip(axes, fields):
        for (high, low), color in zip(MATCHED_PAIRS, colors):
            frame = pairs[
                (pairs["higher_predictability_basis"] == high)
                & (pairs["lower_predictability_basis"] == low)
            ]
            grouped = frame.groupby("target_time")[field].agg(["mean", "std"])
            axis.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                color=color,
                label=f"{high} / {low}",
            )
        axis.axhline(1.0, color="#888888", linestyle="--", linewidth=1)
        axis.set_xlabel("teacher-path start time")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[-2].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output / "latent_trust_decoder_spotcheck.png", dpi=180)
    plt.close(fig)


def summarize_gate(pairs: pd.DataFrame) -> dict[str, object]:
    rows = []
    for (high, low), frame in pairs.groupby(
        ["higher_predictability_basis", "lower_predictability_basis"]
    ):
        record: dict[str, object] = {
            "higher_predictability_basis": high,
            "lower_predictability_basis": low,
        }
        for target_time, time_frame in frame.groupby("target_time"):
            for metric in ("image_shift_l1_ratio", "image_shift_lpips_ratio"):
                record[f"t{target_time:.2f}_{metric}_mean"] = float(
                    time_frame[metric].mean()
                )
                record[f"t{target_time:.2f}_{metric}_above_one_seeds"] = int(
                    (time_frame[metric] > 1.0).sum()
                )
        rows.append(record)

    times = sorted(pairs["target_time"].unique())
    low_time, high_time = float(times[0]), float(times[-1])
    pair_time = pairs.groupby(
        ["higher_predictability_basis", "lower_predictability_basis", "target_time"]
    )[["image_shift_l1_ratio", "image_shift_lpips_ratio"]].mean()
    ordering = {}
    for metric in ("image_shift_l1_ratio", "image_shift_lpips_ratio"):
        preserved = 0
        for high, low in MATCHED_PAIRS:
            if pair_time.loc[(high, low, high_time), metric] > pair_time.loc[
                (high, low, low_time), metric
            ]:
                preserved += 1
        ordering[metric] = preserved
    high_noise = pairs[pairs["target_time"] == high_time]
    l1_positive = high_noise.groupby(
        ["higher_predictability_basis", "lower_predictability_basis"]
    )["image_shift_l1_ratio"].mean() > 1.0
    lpips_positive = high_noise.groupby(
        ["higher_predictability_basis", "lower_predictability_basis"]
    )["image_shift_lpips_ratio"].mean() > 1.0
    agreement = int((l1_positive == lpips_positive).sum())
    passed = bool(
        ordering["image_shift_l1_ratio"] == len(MATCHED_PAIRS)
        and ordering["image_shift_lpips_ratio"] == len(MATCHED_PAIRS)
        and agreement >= 2
    )
    return {
        "pass": passed,
        "definition": (
            "For all three matched-variance pairs, high-noise decoded ratios must "
            "exceed low-noise decoded ratios in both L1 and LPIPS; L1/LPIPS high-noise "
            "directions must agree for at least two pairs."
        ),
        "low_time": low_time,
        "high_time": high_time,
        "time_order_pairs": ordering,
        "high_noise_l1_lpips_direction_agreement_pairs": agreement,
        "pair_summary": rows,
    }


def summarize_handoff(pairs: pd.DataFrame) -> dict[str, object]:
    """Describe the post-hoc dynamics-to-decoder handoff without changing the gate."""

    times = sorted(float(value) for value in pairs["target_time"].unique())
    low_time, high_time = times[0], times[-1]
    rows = []
    all_pairs_match = True
    for high, low in MATCHED_PAIRS:
        frame = pairs[
            (pairs["higher_predictability_basis"] == high)
            & (pairs["lower_predictability_basis"] == low)
        ]
        low_frame = frame[frame["target_time"] == low_time]
        high_frame = frame[frame["target_time"] == high_time]
        record = {
            "higher_predictability_basis": high,
            "lower_predictability_basis": low,
            "low_time": low_time,
            "high_time": high_time,
            "low_endpoint_ratio_mean": float(
                low_frame["endpoint_shift_gain_ratio"].mean()
            ),
            "low_decoder_l1_secant_ratio_mean": float(
                low_frame["decoder_l1_secant_ratio"].mean()
            ),
            "high_endpoint_ratio_mean": float(
                high_frame["endpoint_shift_gain_ratio"].mean()
            ),
            "high_decoder_l1_secant_ratio_mean": float(
                high_frame["decoder_l1_secant_ratio"].mean()
            ),
            "low_endpoint_below_one_seeds": int(
                (low_frame["endpoint_shift_gain_ratio"] < 1.0).sum()
            ),
            "low_decoder_l1_secant_above_one_seeds": int(
                (low_frame["decoder_l1_secant_ratio"] > 1.0).sum()
            ),
            "high_endpoint_above_one_seeds": int(
                (high_frame["endpoint_shift_gain_ratio"] > 1.0).sum()
            ),
            "decoded_l1_above_one_all_times_seeds": int(
                (
                    frame.groupby("seed")["image_shift_l1_ratio"]
                    .min()
                    .gt(1.0)
                ).sum()
            ),
        }
        pair_matches = bool(
            record["low_endpoint_below_one_seeds"] == len(low_frame)
            and record["low_decoder_l1_secant_above_one_seeds"] == len(low_frame)
            and record["high_endpoint_above_one_seeds"] == len(high_frame)
            and record["decoded_l1_above_one_all_times_seeds"] == len(high_frame)
        )
        record["handoff_pattern"] = pair_matches
        all_pairs_match = all_pairs_match and pair_matches
        rows.append(record)
    return {
        "exploratory_not_preregistered": True,
        "all_pairs_match_handoff_pattern": all_pairs_match,
        "interpretation": (
            "At low noise the higher-predictability direction contracts more during "
            "rollout but has larger decoder secant sensitivity; at high noise rollout "
            "leverage dominates. Decoded differences remain larger across the full path."
        ),
        "pairs": rows,
    }


def latent_decoder_correlations(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, target_time), frame in per_seed.groupby(["seed", "target_time"]):
        log_values = frame[
            ["endpoint_shift_gain", "image_shift_l1", "image_shift_lpips"]
        ].clip(lower=1e-20).map(math.log)
        rows.append(
            {
                "seed": int(seed),
                "target_time": float(target_time),
                "endpoint_to_l1_pearson_log": float(
                    log_values[["endpoint_shift_gain", "image_shift_l1"]]
                    .corr()
                    .iloc[0, 1]
                ),
                "endpoint_to_lpips_pearson_log": float(
                    log_values[["endpoint_shift_gain", "image_shift_lpips"]]
                    .corr()
                    .iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_outputs(
    args: argparse.Namespace, seeds: tuple[int, ...], output: Path
) -> None:
    samples = pd.concat(
        [pd.read_csv(output / f"decoder_samples_seed{seed}.csv") for seed in seeds],
        ignore_index=True,
    )
    endpoint_energy = samples["endpoint_shift_energy"].clip(lower=1e-20)
    samples["decoder_l1_secant"] = samples["image_shift_l1"] / endpoint_energy.pow(
        0.5
    )
    samples["decoder_lpips_secant"] = (
        samples["image_shift_lpips"] / endpoint_energy
    )
    samples.to_csv(output / "decoder_samples.csv", index=False)
    metric_columns = [
        "input_energy",
        "endpoint_shift_energy",
        "endpoint_shift_gain",
        "image_shift_l1",
        "image_shift_lpips",
        "decoder_l1_secant",
        "decoder_lpips_secant",
        "base_clean_l1",
        "perturbed_clean_l1",
        "clean_l1_increase",
        "base_clean_lpips",
        "perturbed_clean_lpips",
        "clean_lpips_increase",
    ]
    per_seed = samples.groupby(
        ["seed", "target_time", "actual_time", "basis", "amplitude"],
        as_index=False,
    )[metric_columns].mean()
    per_seed.to_csv(output / "decoder_per_seed.csv", index=False)
    pairs = pair_decoder_ratios(per_seed)
    pairs.to_csv(output / "matched_pair_decoder_ratios.csv", index=False)
    correlations = latent_decoder_correlations(per_seed)
    correlations.to_csv(output / "latent_decoder_correlations.csv", index=False)
    plot_summary(pairs, output)
    gate = summarize_gate(pairs)
    gate.update(
        {
            "seed_count": len(seeds),
            "sample_count_per_seed": int(args.count),
            "metric": "frozen RAE decoder, pixel L1, TorchMetrics AlexNet LPIPS",
            "paired_reference": "decode(clean latent), not raw ImageNet pixels",
            "secant_normalization": {
                "decoder_l1_secant": "decoded L1 shift / latent endpoint RMS shift",
                "decoder_lpips_secant": "decoded LPIPS shift / latent endpoint MSE shift",
            },
            "exploratory_handoff": summarize_handoff(pairs),
        }
    )
    (output / "summary.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    devices = tuple(int(value) for value in args.devices.split(",") if value)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = list(seeds)
    active: dict[int, tuple[int, subprocess.Popen, object]] = {}
    failures: list[tuple[int, int]] = []
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            seed = pending.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (output / f"worker_seed{seed}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                worker_command(args, seed),
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[device] = (seed, process, handle)
            print(f"started seed={seed} cuda={device}", flush=True)
        time.sleep(2)
        for device, (seed, process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[device]
            print(f"finished seed={seed} exit={code}", flush=True)
            if code:
                failures.append((seed, code))
    if failures:
        raise RuntimeError(f"decoder spot-check failures: {failures}")

    summarize_outputs(args, seeds, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bases", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="1,2,3")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--cache-start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--data-batch-size", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=20_260_730)
    parser.add_argument("--times", default="0.95,0.85,0.3")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count % args.data_batch_size:
        raise ValueError("count must be divisible by data_batch_size")
    if args.summarize_only:
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        summarize_outputs(args, seeds, args.output.expanduser().resolve())
    elif args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()

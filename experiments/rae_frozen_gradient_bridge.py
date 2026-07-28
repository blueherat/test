"""No-training gradient-geometry bridge from small images to RAE checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss  # noqa: E402
from experiments.rae_spectral_gradient_audit import (  # noqa: E402
    RAEAuditConfig,
    dct2_basis,
    load_cached_latents,
    load_validation_labels,
    radial_band_masks,
    random_orthogonal_basis,
    spatial_transform,
)
from experiments.rae_teacher_rollout_gap import configure_fp32  # noqa: E402
from experiments.rae_vector_field_switch_probe import load_stage2  # noqa: E402
from experiments.small_image_gradient_allocation import gradient_metrics  # noqa: E402


@dataclass(frozen=True)
class RAEFrozenGradientConfig:
    experiment_root: Path = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
    output_root: Path = Path.home() / "data/eqvae/experiments/rae_frozen_gradient_bridge"
    seeds: tuple[int, ...] = (3407, 4211, 5821)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2")
    validation_count: int = 8
    batch_size: int = 2
    times: tuple[float, ...] = (0.95, 0.85, 0.70)
    evaluation_seed: int = 20260716
    random_basis_seed: int = 20260747
    save: bool = True


def band_losses(
    error: torch.Tensor,
    basis: torch.Tensor,
    masks: torch.Tensor,
) -> list[torch.Tensor]:
    coefficients = spatial_transform(error, basis)
    squared = coefficients.square()
    denominator = float(error.numel())
    return [
        squared[:, :, mask.to(error.device)].sum() / denominator for mask in masks
    ]


def _selected_parameter_groups(
    model: torch.nn.Module,
) -> tuple[tuple[torch.nn.Parameter, ...], dict[str, list[int]], int]:
    block_indices = []
    for name, _ in model.named_parameters():
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            block_indices.append(int(match.group(1)))
    if not block_indices:
        raise RuntimeError("stage-2 model has no named transformer blocks")
    last_block = max(block_indices)
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if name.startswith(f"blocks.{last_block}.") or name in (
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        ):
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    names = [name for name, _ in selected]
    output = [
        index for index, name in enumerate(names) if name.startswith("final_layer.linear.")
    ]
    last = [
        index for index, name in enumerate(names) if name.startswith(f"blocks.{last_block}.")
    ]
    if not output or not last:
        raise RuntimeError("failed to select last-block and output parameters")
    return (
        tuple(parameter for _, parameter in selected),
        {"last_block": last, "output_head": output},
        last_block,
    )


def _flatten_gradients(
    gradients: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> torch.Tensor:
    return torch.cat([gradients[index].reshape(-1) for index in indices])


def band_gradient_tables(
    gradients: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, float]], torch.Tensor]:
    """Summarize ``[bands, parameters]`` gradients under fixed band weights."""

    if gradients.ndim != 2 or weights.shape != (gradients.shape[0],):
        raise ValueError("gradient matrix and weight vector have incompatible shapes")
    coarse = gradients[0]
    detail = gradients[1:].sum(dim=0)
    weighted_coarse = weights[0] * coarse
    weighted_detail = (weights[1:, None] * gradients[1:]).sum(dim=0)
    aggregate = gradient_metrics(
        coarse,
        detail,
        weighted_coarse,
        weighted_detail,
    )
    unweighted_total = gradients.sum(dim=0)
    weighted_total = (weights[:, None] * gradients).sum(dim=0)
    norms = torch.linalg.vector_norm(gradients, dim=1).clamp_min(1e-20)
    cosine = gradients @ gradients.T / (norms[:, None] * norms[None])
    rows = []
    for band in range(len(gradients)):
        baseline_descent = torch.dot(gradients[band], unweighted_total)
        weighted_descent = torch.dot(gradients[band], weighted_total)
        others = torch.cat((cosine[band, :band], cosine[band, band + 1 :]))
        rows.append(
            {
                "band": int(band),
                "weight": float(weights[band]),
                "gradient_norm": float(norms[band]),
                "descent_baseline": float(baseline_descent),
                "descent_weighted": float(weighted_descent),
                "descent_ratio": float(
                    weighted_descent / baseline_descent.abs().clamp_min(1e-20)
                ),
                "cosine_to_other_mean": float(others.mean()),
                "cosine_to_other_min": float(others.min()),
                "cosine_to_other_max": float(others.max()),
            }
        )
    return aggregate, rows, cosine


def _audit_checkpoint(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    analyzer: DCTDirectionLoss,
    bases: dict[str, torch.Tensor],
    masks: torch.Tensor,
    config: RAEFrozenGradientConfig,
    *,
    seed: int,
    checkpoint_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    model.requires_grad_(False).eval()
    parameters, parameter_groups, last_block = _selected_parameter_groups(model)
    aggregate_rows = []
    band_rows = []
    cosine_rows = []
    for time_value in config.times:
        accumulators: dict[tuple[str, str], torch.Tensor] = {}
        seen = 0
        for start in range(0, len(clean), config.batch_size):
            end = min(start + config.batch_size, len(clean))
            data = clean[start:end]
            batch_noise = noise[start:end]
            batch_labels = labels[start:end]
            time = torch.full(
                (len(data),), float(time_value), device=data.device, dtype=data.dtype
            )
            expanded = time[:, None, None, None]
            state = (1.0 - expanded) * data + expanded * batch_noise
            target = batch_noise - data
            prediction = model(state, time, y=batch_labels)
            losses: list[tuple[str, int, torch.Tensor]] = []
            for basis_name, basis in bases.items():
                losses.extend(
                    (basis_name, band, loss)
                    for band, loss in enumerate(band_losses(prediction - target, basis, masks))
                )
            for loss_index, (basis_name, band, loss) in enumerate(losses):
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=loss_index < len(losses) - 1,
                    create_graph=False,
                )
                for group_name, indices in parameter_groups.items():
                    vector = _flatten_gradients(gradients, indices).detach()
                    key = (basis_name, group_name)
                    if key not in accumulators:
                        accumulators[key] = torch.zeros(
                            (len(masks), vector.numel()),
                            device=vector.device,
                            dtype=vector.dtype,
                        )
                    accumulators[key][band].add_(vector, alpha=len(data) / len(clean))
            seen += len(data)
            del prediction, losses
        if seen != len(clean):
            raise RuntimeError("gradient audit did not consume the requested validation set")
        weight = analyzer.weights(
            torch.tensor([float(time_value)], device=clean.device, dtype=clean.dtype)
        )[0]
        for (basis_name, group_name), gradients in accumulators.items():
            aggregate, bands, cosine = band_gradient_tables(gradients, weight)
            identity = {
                "seed": int(seed),
                "checkpoint_variant": checkpoint_variant,
                "basis": basis_name,
                "time": float(time_value),
                "parameter_group": group_name,
                "sample_count": int(len(clean)),
            }
            aggregate_rows.append({**identity, **aggregate})
            band_rows.extend({**identity, **row} for row in bands)
            for first in range(len(masks)):
                for second in range(len(masks)):
                    cosine_rows.append(
                        {
                            **identity,
                            "band_i": int(first),
                            "band_j": int(second),
                            "cosine": float(cosine[first, second]),
                        }
                    )
            del gradients
        gc.collect()
        if clean.device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {
        "last_block_index": int(last_block),
        "selected_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "last_block_parameter_count": int(
            sum(parameters[index].numel() for index in parameter_groups["last_block"])
        ),
        "output_head_parameter_count": int(
            sum(parameters[index].numel() for index in parameter_groups["output_head"])
        ),
    }
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(cosine_rows),
        metadata,
    )


def audit_seed(
    config: RAEFrozenGradientConfig,
    seed: int,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    configure_fp32(config.evaluation_seed + int(seed))
    device = torch.device(
        device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu"
    )
    root = config.experiment_root.expanduser().resolve()
    branches = {
        "baseline": root / f"seed{seed}_baseline_from_s5000",
        "partial": root / f"seed{seed}_partial_from_s5000",
    }
    manifests = {
        name: json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        for name, branch in branches.items()
    }
    if int(manifests["baseline"]["global_seed"]) != int(seed) or int(
        manifests["partial"]["global_seed"]
    ) != int(seed):
        raise ValueError("branch manifest seed mismatch")
    if float(manifests["baseline"]["gamma"]) != 0.0 or float(
        manifests["partial"]["gamma"]
    ) == 0.0:
        raise ValueError("expected paired baseline and partial branches")

    audit_config = RAEAuditConfig(
        train_count=1,
        validation_count=int(config.validation_count),
    )
    payload = load_cached_latents(audit_config)
    clean = payload["validation"][: config.validation_count].to(device)
    labels = load_validation_labels(
        audit_config.dataset_path,
        payload["validation_indices"][: config.validation_count],
    ).to(device)
    noise_generator = torch.Generator(device="cpu").manual_seed(
        int(config.evaluation_seed)
    )
    noise = torch.randn(clean.shape, generator=noise_generator, dtype=torch.float32).to(device)
    config_yaml = OmegaConf.load(branches["baseline"] / "config.yaml")
    spectral = config_yaml.training.spectral_direction_loss
    analyzer = DCTDirectionLoss(
        int(spectral.spatial_size),
        list(spectral.second_moments),
        gamma=0.5,
        damping=float(spectral.damping),
        min_weight=float(spectral.min_weight),
        max_weight=float(spectral.max_weight),
    ).to(device)
    size = int(spectral.spatial_size)
    masks = radial_band_masks(size, analyzer.band_count)
    bases = {
        "dct": dct2_basis(size).float().to(device),
        "random": random_orthogonal_basis(
            size, seed=config.random_basis_seed
        ).float().to(device),
    }
    aggregate_frames = []
    band_frames = []
    cosine_frames = []
    checkpoint_metadata = {}
    for variant, branch in branches.items():
        model, _, manifest = load_stage2(branch, device)
        aggregate, bands, cosine, metadata = _audit_checkpoint(
            model,
            clean,
            labels,
            noise,
            analyzer,
            bases,
            masks,
            config,
            seed=int(seed),
            checkpoint_variant=variant,
        )
        aggregate_frames.append(aggregate)
        band_frames.append(bands)
        cosine_frames.append(cosine)
        checkpoint_metadata[variant] = {
            **metadata,
            "branch": str(branch),
            "gamma": float(manifest["gamma"]),
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {
        "seed": int(seed),
        "device": str(device),
        "precision": "fp32",
        "tf32": False,
        "validation_indices": [
            int(value) for value in payload["validation_indices"][: config.validation_count]
        ],
        "checkpoints": checkpoint_metadata,
    }
    return (
        pd.concat(aggregate_frames, ignore_index=True),
        pd.concat(band_frames, ignore_index=True),
        pd.concat(cosine_frames, ignore_index=True),
        metadata,
    )


def _audit_task(args):
    config, seed, device = args
    result = audit_seed(config, seed, device)
    print(f"audited frozen RAE gradients seed={seed}", flush=True)
    return result


def run_bridge(
    config: RAEFrozenGradientConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    devices = config.devices or ("cpu",)
    tasks = [
        (config, int(seed), devices[index % len(devices)])
        for index, seed in enumerate(config.seeds)
    ]
    if len(tasks) == 1:
        results = [_audit_task(tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(devices), len(tasks)), mp_context=context
        ) as executor:
            futures = [executor.submit(_audit_task, task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    aggregate = pd.concat([result[0] for result in results], ignore_index=True)
    bands = pd.concat([result[1] for result in results], ignore_index=True)
    cosine = pd.concat([result[2] for result in results], ignore_index=True)
    aggregate = aggregate.sort_values(
        ["seed", "checkpoint_variant", "basis", "time", "parameter_group"]
    ).reset_index(drop=True)
    bands = bands.sort_values(
        ["seed", "checkpoint_variant", "basis", "time", "parameter_group", "band"]
    ).reset_index(drop=True)
    cosine = cosine.sort_values(
        [
            "seed",
            "checkpoint_variant",
            "basis",
            "time",
            "parameter_group",
            "band_i",
            "band_j",
        ]
    ).reset_index(drop=True)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"preregistered_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["experiment_root"] = str(config.experiment_root.expanduser())
        serialized["output_root"] = str(config.output_root.expanduser())
        metadata = {
            "config": serialized,
            "seeds": [result[3] for result in results],
            "scope": "frozen EMA last-block/output-head gradients; no optimizer, encoder, or decoder",
        }
        aggregate.to_csv(result_dir / "aggregate_metrics.csv", index=False)
        bands.to_csv(result_dir / "band_metrics.csv", index=False)
        cosine.to_csv(result_dir / "band_cosine.csv", index=False)
        (result_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return aggregate, bands, cosine, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", default="3407,4211,5821")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--times", default="0.95,0.85,0.70")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    times = tuple(float(value) for value in args.times.split(",") if value.strip())
    config = RAEFrozenGradientConfig(
        experiment_root=args.experiment_root or RAEFrozenGradientConfig.experiment_root,
        output_root=args.output_root or RAEFrozenGradientConfig.output_root,
        seeds=seeds,
        devices=devices or ("cpu",),
        validation_count=args.validation_count,
        batch_size=args.batch_size,
        times=times,
        save=not args.no_save,
    )
    aggregate, _, _, result_dir = run_bridge(config)
    print(f"result_dir={result_dir}")
    high = aggregate[aggregate["time"].ge(0.85)]
    print(
        high.groupby(["checkpoint_variant", "basis", "parameter_group"])[
            [
                "coarse_detail_cosine_unweighted",
                "allocation_multiplier",
                "coarse_descent_ratio",
                "detail_descent_ratio",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()

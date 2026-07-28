"""Measure frozen-decoder response distributions for Imagenette latent priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_imagenette_latent_prior_tradeoff import load_run_config  # noqa: E402
from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    INTERFACE_DIM,
    OrthogonalLatentInterface,
    build_prior,
    deterministic_datasets,
    fixed_orthogonal_basis,
    load_frozen_models,
    sample_prior_coordinates,
    state_dict_sha256,
)
from experiments.imagenette_noise_responsibility import fixed_eval_subset  # noqa: E402
from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    descending_time_grid,
)


DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"
LAYERS = (
    "condition",
    "down0",
    "down1",
    "down2",
    "middle",
    "up2",
    "up1",
    "up0",
    "velocity",
)
REPRESENTATIONS = ("raw", "condition")
PROBE_TIMES = (0.9, 0.5, 0.1)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _stable_seed(name: str, base: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little") % 1_000_000


def decoder_forward_trace(
    model: torch.nn.Module,
    value: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match ``ImagenetteConditionalUNet.forward`` while exposing stage outputs."""
    condition_embedding = model.condition_embedding(condition)
    embedding = model.time_mlp(model_time_embedding(time, model.embedding_dim))
    embedding = embedding + condition_embedding
    trace: dict[str, torch.Tensor] = {"condition": condition_embedding}

    skip0 = model._run(model.down0, model.input(value), embedding)
    trace["down0"] = skip0
    skip1 = model._run(model.down1, model.downsample0(skip0), embedding)
    trace["down1"] = skip1
    skip2 = model._run(model.down2, model.downsample1(skip1), embedding)
    trace["down2"] = skip2
    hidden = model._run(model.middle, model.downsample2(skip2), embedding)
    trace["middle"] = hidden

    hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
    hidden = model.upsample2(hidden)
    hidden = model._run(model.up2, torch.cat([hidden, skip2], dim=1), embedding)
    trace["up2"] = hidden
    hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
    hidden = model.upsample1(hidden)
    hidden = model._run(model.up1, torch.cat([hidden, skip1], dim=1), embedding)
    trace["up1"] = hidden
    hidden = F.interpolate(hidden, size=skip0.shape[-2:], mode="nearest")
    hidden = model.upsample0(hidden)
    hidden = model._run(model.up0, torch.cat([hidden, skip0], dim=1), embedding)
    trace["up0"] = hidden
    velocity = model.output(F.silu(model.output_norm(hidden)))
    trace["velocity"] = velocity
    return velocity, trace


def model_time_embedding(time: torch.Tensor, dimensions: int) -> torch.Tensor:
    # Local import keeps the traced forward visibly tied to the model's public path.
    from experiments.mnist_spectral_rollout_toy import sinusoidal_time_embedding

    return sinusoidal_time_embedding(time, int(dimensions))


class FixedFeatureProjector:
    """Pool spatial activations and apply deterministic Gaussian projections."""

    def __init__(self, output_dim: int = 128, seed: int = 48_271):
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self._matrices: dict[tuple[str, int, int], torch.Tensor] = {}

    @staticmethod
    def flatten(layer: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            return value.flatten(1)
        if value.ndim != 4:
            raise ValueError(f"unsupported activation rank for {layer}: {value.ndim}")
        size = 8 if layer == "velocity" else 4
        return F.adaptive_avg_pool2d(value, (size, size)).flatten(1)

    def __call__(self, layer: str, value: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten(layer, value)
        input_dim = int(flattened.shape[1])
        output_dim = min(self.output_dim, input_dim)
        if output_dim == input_dim:
            return flattened
        key = (str(layer), input_dim, output_dim)
        matrix = self._matrices.get(key)
        if matrix is None:
            generator = torch.Generator(device="cpu").manual_seed(
                _stable_seed(f"{layer}:{input_dim}:{output_dim}", self.seed)
            )
            matrix = torch.randn(
                (input_dim, output_dim), generator=generator, dtype=torch.float64
            ) / math.sqrt(output_dim)
            self._matrices[key] = matrix.float()
        return flattened @ matrix.to(device=flattened.device, dtype=flattened.dtype)


@torch.no_grad()
def rollout_response_features(
    decoder: torch.nn.Module,
    conditions: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    steps: int,
    probe_times: Iterable[float],
    batch_size: int,
    projection_dim: int,
    projection_seed: int,
) -> dict[str, torch.Tensor]:
    """Trace one branch on its own rollout and isolate same-state condition effects."""
    if len(conditions) != len(initial_noise):
        raise ValueError("conditions and initial_noise must have equal length")
    device = next(decoder.parameters()).device
    grid = descending_time_grid(int(steps), device=device)
    selected = {
        int(torch.argmin((grid[:-1] - float(target)).abs())): float(target)
        for target in probe_times
    }
    if len(selected) != len(tuple(probe_times)):
        raise ValueError("probe times map to duplicate rollout steps")
    projector = FixedFeatureProjector(projection_dim, projection_seed)
    collected: dict[str, list[torch.Tensor]] = {}

    for start in range(0, len(conditions), int(batch_size)):
        end = min(start + int(batch_size), len(conditions))
        local_condition = conditions[start:end].to(device, non_blocking=True)
        null_condition = torch.zeros_like(local_condition)
        state = initial_noise[start:end].to(device, non_blocking=True).clone()
        local: dict[str, torch.Tensor] = {}
        for step_index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
            time = torch.full((len(state),), float(current), device=device)
            if step_index in selected:
                velocity, raw_trace = decoder_forward_trace(
                    decoder, state, time, local_condition
                )
                _, null_trace = decoder_forward_trace(
                    decoder, state, time, null_condition
                )
                target_time = selected[step_index]
                for layer in LAYERS:
                    raw = raw_trace[layer]
                    contribution = raw - null_trace[layer]
                    for representation, value in (
                        ("raw", raw),
                        ("condition", contribution),
                    ):
                        key = f"{representation}|{target_time:.1f}|{layer}"
                        local[key] = projector(layer, value).cpu()
            else:
                velocity = decoder(state, time, local_condition)
            state = state + (following - current) * velocity
        if set(local) != {
            f"{representation}|{time:.1f}|{layer}"
            for representation in REPRESENTATIONS
            for time in probe_times
            for layer in LAYERS
        }:
            raise RuntimeError("rollout did not record the complete response atlas")
        for key, value in local.items():
            collected.setdefault(key, []).append(value)

    result = {key: torch.cat(values) for key, values in collected.items()}
    if any(len(value) != len(conditions) for value in result.values()):
        raise RuntimeError("response feature count mismatch")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("non-finite decoder response feature")
    return result


def _covariance(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.double()
    mean = value.mean(dim=0)
    centered = value - mean
    covariance = centered.T @ centered / max(len(value) - 1, 1)
    return mean, covariance


def _psd_sqrt(value: torch.Tensor) -> torch.Tensor:
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    return (eigenvectors * eigenvalues.clamp_min(0.0).sqrt()[None, :]) @ eigenvectors.T


def effective_rank(covariance: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T)).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= 1e-18:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def normalized_sliced_wasserstein(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    directions: int,
    seed: int,
) -> float:
    if real.shape != generated.shape:
        raise ValueError("sliced Wasserstein inputs must have the same shape")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        (real.shape[1], int(directions)), generator=generator, dtype=torch.float64
    )
    projection = F.normalize(projection, dim=0)
    real_projected = torch.sort(real.double() @ projection, dim=0).values
    generated_projected = torch.sort(generated.double() @ projection, dim=0).values
    distance = (real_projected - generated_projected).square().mean().sqrt()
    scale = (real.double() - real.double().mean(dim=0)).square().mean().sqrt()
    return float(distance / scale.clamp_min(1e-12))


def distribution_core_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    if real.ndim != 2 or real.shape != generated.shape or len(real) < 8:
        raise ValueError("response metrics require equal rank-two tensors")
    real_mean, real_covariance = _covariance(real)
    generated_mean, generated_covariance = _covariance(generated)
    real_trace = torch.trace(real_covariance).clamp_min(1e-12)
    mean_square = (real_mean - generated_mean).square().sum()
    covariance_relative = torch.linalg.norm(
        real_covariance - generated_covariance
    ) / torch.linalg.norm(real_covariance).clamp_min(1e-12)
    real_sqrt = _psd_sqrt(real_covariance)
    middle_sqrt = _psd_sqrt(real_sqrt @ generated_covariance @ real_sqrt)
    frechet = (
        mean_square
        + torch.trace(real_covariance)
        + torch.trace(generated_covariance)
        - 2.0 * torch.trace(middle_sqrt)
    ).clamp_min(0.0)
    return {
        "mean_relative_error": float(mean_square.sqrt() / real_trace.sqrt()),
        "covariance_relative_error": float(covariance_relative),
        "normalized_frechet": float(frechet / real_trace),
        "normalized_swd": normalized_sliced_wasserstein(
            real, generated, directions=128, seed=int(seed)
        ),
        "real_effective_rank": effective_rank(real_covariance),
        "generated_effective_rank": effective_rank(generated_covariance),
    }


def grouped_linear_c2st_auc(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    seed: int,
    test_fraction: float = 0.30,
) -> float:
    if real.shape != generated.shape or len(real) < 16:
        raise ValueError("C2ST requires equal feature sets with at least 16 rows")
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(real))
    test_count = max(4, int(round(len(real) * float(test_fraction))))
    test_indices = order[:test_count]
    train_indices = order[test_count:]
    real_np = real.double().numpy()
    generated_np = generated.double().numpy()
    train_x = np.concatenate([real_np[train_indices], generated_np[train_indices]])
    train_y = np.concatenate(
        [np.zeros(len(train_indices), dtype=np.int64), np.ones(len(train_indices), dtype=np.int64)]
    )
    test_x = np.concatenate([real_np[test_indices], generated_np[test_indices]])
    test_y = np.concatenate(
        [np.zeros(len(test_indices), dtype=np.int64), np.ones(len(test_indices), dtype=np.int64)]
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed)),
    )
    classifier.fit(train_x, train_y)
    probability = classifier.predict_proba(test_x)[:, 1]
    auc = float(roc_auc_score(test_y, probability))
    return max(auc, 1.0 - auc)


def response_distribution_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    seed: int,
    real_reference: torch.Tensor | None = None,
) -> dict[str, float]:
    values = distribution_core_metrics(real, generated, seed=int(seed))
    values["linear_c2st_auc"] = grouped_linear_c2st_auc(
        real, generated, seed=int(seed) + 1
    )
    if real_reference is None:
        order = torch.randperm(
            len(real), generator=torch.Generator().manual_seed(int(seed) + 2)
        )
        half = len(real) // 2
        floor_left = real[order[:half]]
        floor_right = real[order[half : 2 * half]]
    else:
        if real_reference.shape != real.shape:
            raise ValueError("real_reference must match the primary real feature shape")
        floor_left = real
        floor_right = real_reference
    floor = distribution_core_metrics(floor_left, floor_right, seed=int(seed) + 3)
    floor["linear_c2st_auc"] = grouped_linear_c2st_auc(
        floor_left, floor_right, seed=int(seed) + 4
    )
    for name, value in floor.items():
        values[f"real_real_{name}"] = float(value)
    values["frechet_over_real_floor"] = values["normalized_frechet"] / max(
        values["real_real_normalized_frechet"], 1e-12
    )
    values["swd_over_real_floor"] = values["normalized_swd"] / max(
        values["real_real_normalized_swd"], 1e-12
    )
    return values


@torch.no_grad()
def paired_shuffle_controls(
    decoder: torch.nn.Module,
    images: torch.Tensor,
    latents: torch.Tensor,
    *,
    probe_times: Iterable[float],
    batch_size: int,
    noise_seed: int,
) -> list[dict[str, float]]:
    if len(images) != len(latents):
        raise ValueError("paired images and latents must have equal length")
    device = next(decoder.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise = torch.randn(images.shape, generator=generator)
    shuffled = torch.roll(latents, shifts=1, dims=0)
    rows = []
    for target_time in probe_times:
        sums = {"matched": 0.0, "shuffled": 0.0, "null": 0.0}
        elements = 0
        for start in range(0, len(images), int(batch_size)):
            end = min(start + int(batch_size), len(images))
            clean = images[start:end].to(device, non_blocking=True)
            local_noise = noise[start:end].to(device, non_blocking=True)
            time = torch.full((len(clean),), float(target_time), device=device)
            expanded = time[:, None, None, None]
            state = (1.0 - expanded) * clean + expanded * local_noise
            target = local_noise - clean
            local_latent = latents[start:end].to(device, non_blocking=True)
            local_shuffled = shuffled[start:end].to(device, non_blocking=True)
            for name, condition in (
                ("matched", local_latent),
                ("shuffled", local_shuffled),
                ("null", torch.zeros_like(local_latent)),
            ):
                prediction = decoder(state, time, condition)
                sums[name] += float(F.mse_loss(prediction, target, reduction="sum"))
            elements += target.numel()
        means = {name: value / elements for name, value in sums.items()}
        rows.append(
            {
                "time": float(target_time),
                "matched_velocity_mse": means["matched"],
                "shuffled_velocity_mse": means["shuffled"],
                "null_velocity_mse": means["null"],
                "shuffled_over_matched": means["shuffled"] / means["matched"],
                "null_over_matched": means["null"] / means["matched"],
            }
        )
    return rows


def load_fixed_images(dataset, indices: Sequence[int], batch_size: int) -> torch.Tensor:
    subset = torch.utils.data.Subset(dataset, [int(index) for index in indices])
    loader = DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=2)
    return torch.cat([images for images, _labels in loader])


def run_atlas(
    run: Path,
    *,
    device_name: str,
    count: int = 256,
    paired_count: int = 128,
    pixel_steps: int = 50,
    batch_size: int = 16,
    projection_dim: int = 128,
    projection_seed: int = 48_271,
    overwrite: bool = False,
) -> Path:
    output_json = run / "decoder_response_atlas.json"
    output_tensor = run / "decoder_response_atlas.pt"
    if output_json.is_file() and output_tensor.is_file() and not overwrite:
        print(f"response atlas already complete: {output_json}", flush=True)
        return output_json
    config = load_run_config(run, device_name)
    configure_fp32(config.prior_seed)
    device = torch.device(device_name)
    _train_dataset, val_dataset = deterministic_datasets(config.data_root, config.image_size)
    _encoder, decoder, frozen = load_frozen_models(config, device)
    cache = torch.load(run / "latent_cache.pt", map_location="cpu", weights_only=True)
    prior_state = torch.load(run / "prior_state.pt", map_location="cpu", weights_only=True)
    prior = build_prior(config, device)
    prior.load_state_dict(prior_state["prior_ema"])
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    interface = OrthogonalLatentInterface(
        config.latent_dim,
        fixed_orthogonal_basis(INTERFACE_DIM, config.basis_seed),
    ).to(device)

    count = min(int(count), len(cache["train_latent"]))
    empirical_order = torch.randperm(
        len(cache["train_latent"]), generator=torch.Generator().manual_seed(83_011)
    )[: 2 * count]
    if len(empirical_order) != 2 * count:
        raise RuntimeError("not enough cached train latents for the real-real control")
    empirical = cache["train_latent"][empirical_order[:count]].float()
    empirical_reference = cache["train_latent"][empirical_order[count:]].float()
    prior_latent = sample_prior_coordinates(
        prior,
        interface,
        count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    ).float()
    initial_noise = torch.randn(
        (count, 3, config.image_size, config.image_size),
        generator=torch.Generator(device="cpu").manual_seed(91_027),
    )
    empirical_features = rollout_response_features(
        decoder,
        empirical,
        initial_noise,
        steps=int(pixel_steps),
        probe_times=PROBE_TIMES,
        batch_size=int(batch_size),
        projection_dim=int(projection_dim),
        projection_seed=int(projection_seed),
    )
    empirical_reference_features = rollout_response_features(
        decoder,
        empirical_reference,
        initial_noise,
        steps=int(pixel_steps),
        probe_times=PROBE_TIMES,
        batch_size=int(batch_size),
        projection_dim=int(projection_dim),
        projection_seed=int(projection_seed),
    )
    prior_features = rollout_response_features(
        decoder,
        prior_latent,
        initial_noise,
        steps=int(pixel_steps),
        probe_times=PROBE_TIMES,
        batch_size=int(batch_size),
        projection_dim=int(projection_dim),
        projection_seed=int(projection_seed),
    )

    rows = []
    for key in sorted(empirical_features):
        representation, time_text, layer = key.split("|")
        metrics = response_distribution_metrics(
            empirical_features[key],
            prior_features[key],
            seed=_stable_seed(key, 62_001 + config.frozen_seed),
            real_reference=empirical_reference_features[key],
        )
        rows.append(
            {
                "representation": representation,
                "time": float(time_text),
                "layer": layer,
                "feature_dim": int(empirical_features[key].shape[1]),
                **metrics,
            }
        )

    paired_count = min(int(paired_count), len(val_dataset))
    paired_subset = fixed_eval_subset(val_dataset, paired_count, seed=2_027)
    paired_indices = torch.as_tensor(paired_subset.indices, dtype=torch.long)
    paired_images = load_fixed_images(
        val_dataset, paired_subset.indices, batch_size=int(batch_size)
    )
    paired_latents = cache["val_latent"][paired_indices].float()
    paired_rows = paired_shuffle_controls(
        decoder,
        paired_images,
        paired_latents,
        probe_times=PROBE_TIMES,
        batch_size=int(batch_size),
        noise_seed=93_017,
    )

    formal_summary = json.loads((run / "summary.json").read_text())
    payload = {
        "run": str(run),
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "modeling_gap": float(formal_summary["modeling_gap"]),
        "count": int(count),
        "paired_count": int(paired_count),
        "pixel_steps": int(pixel_steps),
        "probe_times": list(PROBE_TIMES),
        "projection_dim": int(projection_dim),
        "projection_seed": int(projection_seed),
        "frozen_decoder_sha256": state_dict_sha256(decoder),
        "formal_frozen_decoder_sha256": str(formal_summary["frozen_decoder_sha256"]),
        "frozen_decoder_matches_formal": bool(
            state_dict_sha256(decoder) == formal_summary["frozen_decoder_sha256"]
        ),
        "response_rows": rows,
        "paired_rows": paired_rows,
        "config": asdict(config),
    }
    # Config paths are already recorded in each formal run; keep this audit JSON portable.
    payload["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload["config"].items()
    }
    if not all(
        math.isfinite(float(value))
        for row in rows + paired_rows
        for value in row.values()
        if isinstance(value, (int, float))
    ):
        raise FloatingPointError("non-finite response atlas metric")
    torch.save(
        {
            "empirical_latent": empirical,
            "empirical_reference_latent": empirical_reference,
            "prior_latent": prior_latent,
            "initial_noise_sha256": hashlib.sha256(initial_noise.numpy().tobytes()).hexdigest(),
            "empirical_features": empirical_features,
            "empirical_reference_features": empirical_reference_features,
            "prior_features": prior_features,
        },
        output_tensor,
    )
    _write_json(output_json, payload)
    print(json.dumps({"output": str(output_json), "rows": len(rows)}), flush=True)
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--paired-count", type=int, default=128)
    parser.add_argument("--pixel-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=48_271)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_atlas(
        args.run,
        device_name=args.device,
        count=args.count,
        paired_count=args.paired_count,
        pixel_steps=args.pixel_steps,
        batch_size=args.batch_size,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

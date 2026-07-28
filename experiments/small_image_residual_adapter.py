"""Frozen-baseline projected residual adapters for small-image flow matching.

The baseline velocity field is never updated.  A zero-initialized small adapter
may only change non-protected output groups.  The rollout-aware variant also
trains on one stop-gradient self-generated state and matches normalized band
drift there.  This is a preregistered mechanism gate, not an ImageNet result.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    TinyVelocityUNet,
    configure_fp32,
    evaluate_rollouts,
    evaluate_teacher_path,
    shifted_uniform,
    train_feature_classifier,
)
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    OrthogonalDirectionLoss,
    load_small_image_tensors,
)
from experiments.small_image_training_rescue import (  # noqa: E402
    _toy_config,
    summarize_variants,
)


@dataclass(frozen=True)
class ResidualAdapterConfig:
    study_dir: Path
    output_root: Path = (
        Path.home() / "data/eqvae/experiments/small_image_residual_adapter"
    )
    bases: tuple[str, ...] = ("dct", "pca")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    adapter_width: int = 12
    offpath_loss_weight: float = 0.5
    drift_loss_weight: float = 0.25
    offpath_step: float = 0.1
    sample_count: int = 1024
    variants: tuple[str, ...] = (
        "teacher_residual",
        "rollout_drift_residual",
    )
    save: bool = True


class ProjectedResidualField(nn.Module):
    """Frozen baseline plus an adapter confined to non-protected output groups."""

    def __init__(
        self,
        baseline: nn.Module,
        analyzer: OrthogonalDirectionLoss,
        *,
        adapter_width: int,
        depth: int,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.adapter = TinyVelocityUNet(int(adapter_width), int(depth))
        self.spatial_size = int(analyzer.spatial_size)
        self.residual_scale = float(residual_scale)
        self.register_buffer("basis", analyzer.basis.detach().clone(), persistent=True)
        self.register_buffer(
            "detail_mask",
            analyzer.group_index.ne(0).to(torch.float32),
            persistent=True,
        )
        self.baseline.requires_grad_(False).eval()

    def project_detail(self, value: torch.Tensor) -> torch.Tensor:
        coefficients = value.flatten(1) @ self.basis.to(value.dtype)
        coefficients = coefficients * self.detail_mask.to(value.dtype)[None]
        return (coefficients @ self.basis.to(value.dtype).T).reshape_as(value)

    def field_delta(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.project_detail(self.adapter(value, time))

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.baseline.eval()
        with torch.no_grad():
            baseline = self.baseline(value, time)
        return baseline + self.residual_scale * self.field_delta(value, time)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self


def detail_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
) -> torch.Tensor:
    error = analyzer.transform(prediction - target).square()
    detail = analyzer.group_index.to(error.device).ne(0)
    if not bool(detail.any()):
        raise ValueError("detail loss requires at least one non-protected direction")
    weights = analyzer.weights(time).to(error.dtype)
    return torch.mean(error[:, detail] * weights[:, detail])


def normalized_band_drift_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
) -> torch.Tensor:
    """Squared normalized errors in ``E[z_b * v_b]`` for detail groups."""

    state_coefficients = analyzer.transform(state)
    prediction_coefficients = analyzer.transform(prediction)
    target_coefficients = analyzer.transform(target)
    errors = prediction_coefficients - target_coefficients
    group_index = analyzer.group_index.to(state.device)
    losses = []
    for band in range(1, int(analyzer.band_count)):
        selected = group_index.eq(band)
        if not bool(selected.any()):
            continue
        band_state = state_coefficients[:, selected]
        band_target = target_coefficients[:, selected]
        numerator = torch.mean(band_state * errors[:, selected])
        scale = torch.sqrt(
            torch.mean(band_state.square()) * torch.mean(band_target.square())
        ).clamp_min(1e-6)
        losses.append((numerator / scale).square())
    if not losses:
        raise ValueError("drift loss requires at least one non-protected band")
    return torch.stack(losses).mean()


def self_generated_state(
    model: nn.Module,
    state: torch.Tensor,
    time: torch.Tensor,
    *,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one detached Euler step in the data/sampling direction."""

    step = torch.minimum(torch.full_like(time, float(max_step)), 0.5 * time)
    with torch.no_grad():
        velocity = model(state, time)
        generated = state - step[:, None, None, None] * velocity
    return generated, time - step


def residual_variant_terms(variant: str) -> tuple[bool, bool]:
    """Return whether a variant uses off-path MSE and normalized drift."""

    terms = {
        "teacher_residual": (False, False),
        "rollout_drift_residual": (True, True),
        "offpath_mse_residual": (True, False),
        "drift_only_residual": (False, True),
    }
    if variant not in terms:
        raise ValueError(f"unknown residual-adapter variant: {variant}")
    return terms[variant]


def train_residual_adapters(
    clean: torch.Tensor,
    baseline: nn.Module,
    analyzer: OrthogonalDirectionLoss,
    study_config,
    config: ResidualAdapterConfig,
    *,
    seed: int,
) -> tuple[dict[str, ProjectedResidualField], pd.DataFrame]:
    device = clean.device
    baseline = copy.deepcopy(baseline).to(device).eval()
    baseline.requires_grad_(False)
    torch.manual_seed(int(seed) + 29_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 29_001)
    initial = ProjectedResidualField(
        baseline,
        analyzer,
        adapter_width=config.adapter_width,
        depth=study_config.depth,
    ).to(device)
    if not config.variants or len(set(config.variants)) != len(config.variants):
        raise ValueError("residual adapter variants must be nonempty and unique")
    for variant in config.variants:
        residual_variant_terms(variant)
    models = {
        name: initial if index == 0 else copy.deepcopy(initial)
        for index, name in enumerate(config.variants)
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.adapter.parameters(),
            lr=study_config.learning_rate,
            weight_decay=1e-4,
        )
        for name, model in models.items()
    }
    generator = torch.Generator(device=device).manual_seed(int(seed) + 101)
    rows: list[dict[str, float | int | str]] = []
    log_every = max(1, int(study_config.steps) // 40)
    for step_index in range(1, int(study_config.steps) + 1):
        indices = torch.randint(
            len(clean),
            (int(study_config.batch_size),),
            device=device,
            generator=generator,
        )
        data = clean[indices]
        noise = torch.randn(data.shape, device=device, generator=generator)
        time = shifted_uniform(
            len(data), 1.0, device=device, generator=generator
        )
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            model.train()
            prediction = model(state, time)
            teacher_loss = detail_weighted_mse(
                prediction, target, analyzer, time
            )
            offpath_loss = torch.zeros((), device=device)
            drift_loss = torch.zeros((), device=device)
            objective = teacher_loss
            use_offpath_mse, use_drift = residual_variant_terms(name)
            if use_offpath_mse or use_drift:
                generated, generated_time = self_generated_state(
                    model, state, time, max_step=config.offpath_step
                )
                generated_prediction = model(generated, generated_time)
                if use_offpath_mse:
                    offpath_loss = detail_weighted_mse(
                        generated_prediction, target, analyzer, generated_time
                    )
                    objective = objective + float(config.offpath_loss_weight) * offpath_loss
                if use_drift:
                    drift_loss = normalized_band_drift_loss(
                        generated_prediction, target, generated, analyzer
                    )
                    objective = objective + float(config.drift_loss_weight) * drift_loss
            optimizers[name].zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizers[name].step()
            if (
                step_index == 1
                or step_index % log_every == 0
                or step_index == study_config.steps
            ):
                rows.append(
                    {
                        "step": int(step_index),
                        "variant": name,
                        "objective": float(objective.detach()),
                        "teacher_detail_loss": float(teacher_loss.detach()),
                        "offpath_detail_loss": float(offpath_loss.detach()),
                        "offpath_drift_loss": float(drift_loss.detach()),
                    }
                )
    return models, pd.DataFrame(rows)


@torch.no_grad()
def protection_error(
    model: ProjectedResidualField,
    value: torch.Tensor,
    time: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
) -> float:
    coefficients = analyzer.transform(model.field_delta(value, time))
    coarse = analyzer.group_index.to(value.device).eq(0)
    return float(coefficients[:, coarse].abs().max())


def _gain_retention(
    baseline_ratio: float,
    weighted_ratio: float,
    candidate_ratio: float,
) -> float:
    available = float(baseline_ratio) - float(weighted_ratio)
    if available <= 1e-12:
        return float("nan")
    return (float(baseline_ratio) - float(candidate_ratio)) / available


def run_residual_pair(
    config: ResidualAdapterConfig,
    *,
    source_run_dir: Path,
    device_name: str,
    result_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    study_config = _load_study_config(config.study_dir)
    source_summary = json.loads(
        (source_run_dir / "summary.json").read_text(encoding="utf-8")
    )
    seed = int(source_summary["seed"])
    basis = str(source_summary["basis"])
    configure_fp32(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu"
    )
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        study_config.test_size,
        seed,
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    raw_models, analyzer, _ = _load_run(source_run_dir, study_config, device)
    residual_models, history = train_residual_adapters(
        train,
        raw_models["baseline"],
        analyzer,
        study_config,
        config,
        seed=seed,
    )
    models: dict[str, nn.Module] = {**raw_models, **residual_models}
    for model in models.values():
        model.eval()
    teacher, teacher_bands = evaluate_teacher_path(
        models,
        test,
        analyzer,
        study_config.eval_times,
        seed,
        study_config.batch_size,
    )
    classifier, accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        loaded["test_labels"].to(device),
        epochs=study_config.classifier_epochs,
        batch_size=study_config.classifier_batch_size,
        seed=seed,
    )
    if abs(float(accuracy) - float(source_summary["classifier_accuracy"])) > 0.01:
        raise RuntimeError("residual-adapter classifier differs by more than one point")
    rollout, rollout_bands, samples = evaluate_rollouts(
        models,
        test,
        classifier,
        analyzer,
        _toy_config(study_config, seed, device, config.sample_count),
        loaded["normalization"],
    )
    references = {name: "baseline" for name in models}
    summary = summarize_variants(
        teacher, teacher_bands, rollout, references=references
    )
    summary.insert(0, "seed", seed)
    summary.insert(0, "basis", basis)
    summary.insert(0, "dataset", study_config.dataset)
    summary["classifier_accuracy"] = float(accuracy)
    values = summary.set_index("variant")
    for candidate in residual_models:
        summary.loc[summary["variant"].eq(candidate), "detail_gain_retention"] = (
            _gain_retention(
                float(values.loc["baseline", "high_nonzero_mse_ratio"]),
                float(values.loc["weighted", "high_nonzero_mse_ratio"]),
                float(values.loc[candidate, "high_nonzero_mse_ratio"]),
            )
        )
    protection = {
        name: protection_error(
            model,
            test[: min(32, len(test))],
            torch.full(
                (min(32, len(test)),), 0.9, device=device, dtype=test.dtype
            ),
            analyzer,
        )
        for name, model in residual_models.items()
    }
    adapter_parameters = sum(
        parameter.numel()
        for parameter in next(iter(residual_models.values())).adapter.parameters()
    )
    baseline_parameters = sum(
        parameter.numel() for parameter in raw_models["baseline"].parameters()
    )
    metadata: dict[str, object] = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "source_run_dir": str(source_run_dir),
        "source_baseline_hash": source_summary["baseline_hash"],
        "classifier_accuracy": float(accuracy),
        "protection_error": protection,
        "adapter_parameters": int(adapter_parameters),
        "baseline_parameters": int(baseline_parameters),
        "parameter_ratio": float(adapter_parameters / baseline_parameters),
        "scope": "frozen baseline; projected nonzero-group residual adapter",
    }
    if max(protection.values()) >= 1e-5:
        raise RuntimeError("projected residual adapter changed the protected group")
    if config.save:
        if result_dir is None:
            raise ValueError("result_dir is required when saving")
        run_dir = result_dir / f"{basis}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        history.to_csv(run_dir / "history.csv", index=False)
        teacher.to_csv(run_dir / "teacher_summary.csv", index=False)
        teacher_bands.to_csv(run_dir / "teacher_bands.csv", index=False)
        rollout.to_csv(run_dir / "rollout_summary.csv", index=False)
        rollout_bands.to_csv(run_dir / "rollout_bands.csv", index=False)
        summary.to_csv(run_dir / "variant_summary.csv", index=False)
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "adapters": {
                    name: model.adapter.state_dict()
                    for name, model in residual_models.items()
                },
                "samples": {
                    name: value
                    for name, value in samples.items()
                    if name in residual_models
                },
                "normalization": loaded["normalization"],
            },
            run_dir / "state.pt",
        )
        summary["run_dir"] = str(run_dir)
    for model in models.values():
        model.cpu()
    analyzer.cpu()
    del train, test, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, metadata


def _run_group(
    tasks: Sequence[tuple[ResidualAdapterConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    results = []
    for config, source_run_dir, device, result_dir in tasks:
        summary, metadata = run_residual_pair(
            config,
            source_run_dir=source_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = summary.set_index("variant")
        ratios = ", ".join(
            f"{name}={selected.loc[name, 'feature_fid_over_reference']:.3f}"
            for name in config.variants
        )
        print(
            f"done {metadata['dataset']} {metadata['basis']} seed={metadata['seed']}: "
            f"raw={selected.loc['weighted', 'feature_fid_over_reference']:.3f}, "
            f"{ratios}, protect={max(metadata['protection_error'].values()):.2e}",
            flush=True,
        )
        results.append((summary, metadata))
    return results


def run_residual_study(
    config: ResidualAdapterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    config = ResidualAdapterConfig(
        **{**asdict(config), "study_dir": config.study_dir.expanduser().resolve()}
    )
    study_config = _load_study_config(config.study_dir)
    source = pd.read_csv(config.study_dir / "study_summary.csv")
    source = source[source["basis"].isin(config.bases)].copy()
    missing = set(config.bases) - set(source["basis"])
    if missing:
        raise ValueError(f"source study is missing bases: {sorted(missing)}")
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = (
            config.output_root.expanduser()
            / f"{study_config.dataset}_residual_adapter_preregistered_{timestamp}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["study_dir"] = str(config.study_dir)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
    devices = config.devices or ("cpu",)
    tasks = [
        (config, Path(row.run_dir), devices[index % len(devices)], result_dir)
        for index, row in enumerate(source.itertuples())
    ]
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    results: list[tuple[pd.DataFrame, dict[str, object]]] = []
    if len(devices) == 1:
        results = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in grouped if group]
            for future in as_completed(futures):
                results.extend(future.result())
    summary = pd.concat([item[0] for item in results], ignore_index=True)
    summary = summary.sort_values(["basis", "seed", "variant"]).reset_index(drop=True)
    metadata = pd.DataFrame([item[1] for item in results]).sort_values(
        ["basis", "seed"]
    )
    if result_dir is not None:
        summary.to_csv(result_dir / "study_summary.csv", index=False)
        metadata.to_csv(result_dir / "metadata.csv", index=False)
    return summary, metadata, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--adapter-width", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument(
        "--variants", default="teacher_residual,rollout_drift_residual"
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    config = ResidualAdapterConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or ResidualAdapterConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        adapter_width=args.adapter_width,
        sample_count=args.sample_count,
        variants=variants,
        save=not args.no_save,
    )
    summary, metadata, result_dir = run_residual_study(config)
    print(f"result_dir={result_dir}")
    selected = summary[summary["variant"].isin(("weighted", *variants))]
    print(
        selected.groupby(["basis", "variant"])[
            [
                "feature_fid_over_reference",
                "feature_swd_over_reference",
                "latent_swd_over_reference",
                "high_group0_mse_ratio",
                "high_nonzero_mse_ratio",
                "detail_gain_retention",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )
    print(
        metadata[["basis", "seed", "parameter_ratio", "protection_error"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

"""Predictive training-side rescue of the small-image transport reversal.

The study reuses frozen raw baseline/weighted checkpoints and trains two
mechanism-driven interventions on the exact same dataset split:

* coarse_protected: group 0 keeps unit loss weight while groups 1-7 retain
  their inverse-variance relative weighting;
* split_baseline/split_weighted: parameter-matched independent coarse and
  detail velocity paths remove cross-subspace parameter competition.

MNIST/FashionMNIST remain mechanism probes, not ImageNet quality claims.
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
from typing import Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    MNISTToyConfig,
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


@dataclass(frozen=True)
class TrainingRescueConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_training_rescue"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    split_width: int = 17
    sample_count: int = 1024
    save: bool = True


def coarse_protected_weights(
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
) -> torch.Tensor:
    """Keep group 0 at one and preserve detail relative weights and mean one."""

    weights = analyzer.weights(time)
    coarse = analyzer.group_index.to(weights.device).eq(0)
    detail = ~coarse
    detail_count = int(detail.sum())
    if int(coarse.sum()) == 0 or detail_count == 0:
        raise ValueError("coarse protection requires nonempty group 0 and detail groups")
    detail_sum = weights[:, detail].sum(dim=1, keepdim=True).clamp_min(1e-12)
    detail_scale = float(detail_count) / detail_sum
    protected = weights * detail_scale
    protected[:, coarse] = 1.0
    return protected


def coefficient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
    *,
    protected: bool,
) -> torch.Tensor:
    coefficient_error = analyzer.transform(prediction - target)
    if protected:
        weights = coarse_protected_weights(analyzer, time).to(coefficient_error.dtype)
        return torch.mean(coefficient_error.square() * weights)
    return torch.mean(coefficient_error.square())


class SplitVelocityField(nn.Module):
    """Independent parameter-matched paths for group 0 and groups 1-7."""

    def __init__(
        self,
        analyzer: OrthogonalDirectionLoss,
        *,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.coarse = TinyVelocityUNet(width, depth)
        self.detail = TinyVelocityUNet(width, depth)
        self.spatial_size = analyzer.spatial_size
        self.register_buffer("basis", analyzer.basis.detach().clone(), persistent=True)
        self.register_buffer(
            "coarse_mask",
            analyzer.group_index.eq(0).to(torch.float32),
            persistent=True,
        )

    def project(self, value: torch.Tensor, *, coarse: bool) -> torch.Tensor:
        coefficients = value.flatten(1) @ self.basis.to(value.dtype)
        mask = self.coarse_mask if coarse else 1.0 - self.coarse_mask
        projected = (coefficients * mask.to(value.dtype)[None]) @ self.basis.to(value.dtype).T
        return projected.reshape_as(value)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.project(self.coarse(value, time), coarse=True) + self.project(
            self.detail(value, time), coarse=False
        )

    def clip_grad_norm_(self, max_norm: float) -> None:
        # Clip independently so one path cannot suppress the other through global clipping.
        torch.nn.utils.clip_grad_norm_(self.coarse.parameters(), float(max_norm))
        torch.nn.utils.clip_grad_norm_(self.detail.parameters(), float(max_norm))


def parameter_counts(width: int, split_width: int, depth: int) -> dict[str, int]:
    raw = sum(parameter.numel() for parameter in TinyVelocityUNet(width, depth).parameters())
    split = 2 * sum(
        parameter.numel()
        for parameter in TinyVelocityUNet(split_width, depth).parameters()
    )
    return {"raw": int(raw), "split": int(split)}


def train_rescue_fields(
    clean: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    study_config,
    *,
    split_width: int,
    seed: int,
) -> tuple[dict[str, nn.Module], pd.DataFrame]:
    """Train only the preregistered rescue variants on a shared batch stream."""

    device = clean.device
    # Recreate the raw model initialization used by the source paired study.
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    protected_model = TinyVelocityUNet(study_config.width, study_config.depth).to(device)

    torch.manual_seed(int(seed) + 13_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 13_001)
    split_initial = SplitVelocityField(
        analyzer,
        width=int(split_width),
        depth=study_config.depth,
    ).to(device)
    models: dict[str, nn.Module] = {
        "coarse_protected": protected_model,
        "split_baseline": split_initial,
        "split_weighted": copy.deepcopy(split_initial),
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=study_config.learning_rate,
            weight_decay=1e-4,
        )
        for name, model in models.items()
    }
    generator = torch.Generator(device=device).manual_seed(int(seed) + 101)
    rows: list[dict[str, float | int | str]] = []
    log_every = max(1, int(study_config.steps) // 40)
    for step in range(1, int(study_config.steps) + 1):
        indices = torch.randint(
            len(clean),
            (int(study_config.batch_size),),
            device=device,
            generator=generator,
        )
        data = clean[indices]
        noise = torch.randn(data.shape, device=device, generator=generator)
        time = shifted_uniform(
            len(data),
            1.0,
            device=device,
            generator=generator,
        )
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            prediction = model(state, time)
            loss = coefficient_loss(
                prediction,
                target,
                analyzer,
                time,
                protected=name != "split_baseline",
            )
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            if isinstance(model, SplitVelocityField):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if step == 1 or step % log_every == 0 or step == study_config.steps:
                rows.append(
                    {
                        "step": int(step),
                        "variant": name,
                        "training_objective": float(loss.detach()),
                        "raw_mse": float(F.mse_loss(prediction.detach(), target)),
                    }
                )
    return models, pd.DataFrame(rows)


def _toy_config(study_config, seed: int, device: torch.device, sample_count: int) -> MNISTToyConfig:
    return MNISTToyConfig(
        data_root=study_config.data_root,
        train_size=study_config.train_size,
        test_size=study_config.test_size,
        sample_count=min(int(sample_count), int(study_config.test_size)),
        batch_size=study_config.batch_size,
        steps=study_config.steps,
        learning_rate=study_config.learning_rate,
        width=study_config.width,
        depth=study_config.depth,
        gamma=study_config.gamma,
        band_count=study_config.band_count,
        ode_steps=study_config.ode_steps,
        classifier_epochs=study_config.classifier_epochs,
        classifier_batch_size=study_config.classifier_batch_size,
        eval_times=study_config.eval_times,
        seed=int(seed),
        device=str(device),
        save=False,
    )


def summarize_variants(
    teacher: pd.DataFrame,
    teacher_bands: pd.DataFrame,
    rollout: pd.DataFrame,
    references: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if references is None:
        references = {
            "baseline": "baseline",
            "weighted": "baseline",
            "coarse_protected": "baseline",
            "split_baseline": "split_baseline",
            "split_weighted": "split_baseline",
        }
    teacher_pivot = teacher.pivot(index="time", columns="variant", values="velocity_mse")
    band_pivot = teacher_bands.pivot(
        index=["time", "band"], columns="variant", values="velocity_mse"
    )
    rollout_values = rollout.set_index("variant")
    rows = []
    for variant, reference in references.items():
        teacher_ratio = teacher_pivot[variant] / teacher_pivot[reference].clip(lower=1e-12)
        band_ratio = band_pivot[variant] / band_pivot[reference].clip(lower=1e-12)
        high = band_ratio.index.get_level_values("time") >= 0.7
        group0 = band_ratio.index.get_level_values("band") == 0
        rows.append(
            {
                "variant": variant,
                "reference_variant": reference,
                "teacher_ratio_all": float(teacher_ratio.mean()),
                "teacher_ratio_low_mid": float(teacher_ratio[teacher_ratio.index <= 0.5].mean()),
                "teacher_ratio_high": float(teacher_ratio[teacher_ratio.index >= 0.7].mean()),
                "high_group0_mse_ratio": float(band_ratio[high & group0].mean()),
                "high_nonzero_mse_ratio": float(band_ratio[high & ~group0].mean()),
                "feature_fid": float(rollout_values.loc[variant, "feature_fid"]),
                "feature_fid_over_reference": float(
                    rollout_values.loc[variant, "feature_fid"]
                    / max(float(rollout_values.loc[reference, "feature_fid"]), 1e-12)
                ),
                "feature_swd_over_reference": float(
                    rollout_values.loc[variant, "feature_swd"]
                    / max(float(rollout_values.loc[reference, "feature_swd"]), 1e-12)
                ),
                "latent_swd_over_reference": float(
                    rollout_values.loc[variant, "latent_swd"]
                    / max(float(rollout_values.loc[reference, "latent_swd"]), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_rescue_pair(
    config: TrainingRescueConfig,
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
    rescue_models, history = train_rescue_fields(
        train,
        analyzer,
        study_config,
        split_width=config.split_width,
        seed=seed,
    )
    models = {**raw_models, **rescue_models}
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
        raise RuntimeError("rescue classifier differs from source by more than one point")
    toy_config = _toy_config(study_config, seed, device, config.sample_count)
    rollout, rollout_bands, samples = evaluate_rollouts(
        models,
        test,
        classifier,
        analyzer,
        toy_config,
        loaded["normalization"],
    )
    summary = summarize_variants(teacher, teacher_bands, rollout)
    summary.insert(0, "seed", seed)
    summary.insert(0, "basis", basis)
    summary.insert(0, "dataset", study_config.dataset)
    summary["classifier_accuracy"] = float(accuracy)
    counts = parameter_counts(study_config.width, config.split_width, study_config.depth)
    metadata: dict[str, object] = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "source_run_dir": str(source_run_dir),
        "source_baseline_hash": source_summary["baseline_hash"],
        "classifier_accuracy": float(accuracy),
        "raw_parameters": counts["raw"],
        "split_parameters": counts["split"],
        "split_parameter_ratio": counts["split"] / counts["raw"],
    }
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
                "models": {
                    name: model.state_dict() for name, model in rescue_models.items()
                },
                "samples": samples,
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
    tasks: Sequence[tuple[TrainingRescueConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    results = []
    for config, source_run_dir, device, result_dir in tasks:
        summary, metadata = run_rescue_pair(
            config,
            source_run_dir=source_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = summary.set_index("variant")
        print(
            f"done {metadata['dataset']} {metadata['basis']} seed={metadata['seed']}: "
            f"raw={selected.loc['weighted', 'feature_fid_over_reference']:.3f}, "
            f"protected={selected.loc['coarse_protected', 'feature_fid_over_reference']:.3f}, "
            f"split={selected.loc['split_weighted', 'feature_fid_over_reference']:.3f}"
        )
        results.append((summary, metadata))
    return results


def run_training_rescue_study(
    config: TrainingRescueConfig,
) -> tuple[pd.DataFrame, Path | None]:
    config = TrainingRescueConfig(
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
            / f"{study_config.dataset}_training_rescue_preregistered_{timestamp}"
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
    if result_dir is not None:
        summary.to_csv(result_dir / "study_summary.csv", index=False)
        pd.DataFrame([item[1] for item in results]).sort_values(
            ["basis", "seed"]
        ).to_csv(result_dir / "metadata.csv", index=False)
    return summary, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--split-width", type=int, default=17)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = TrainingRescueConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or TrainingRescueConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        split_width=args.split_width,
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    summary, result_dir = run_training_rescue_study(config)
    print(f"result_dir={result_dir}")
    selected = summary[
        summary["variant"].isin(("weighted", "coarse_protected", "split_weighted"))
    ]
    print(
        selected.groupby(["basis", "variant"])[
            [
                "feature_fid_over_reference",
                "high_group0_mse_ratio",
                "high_nonzero_mse_ratio",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()

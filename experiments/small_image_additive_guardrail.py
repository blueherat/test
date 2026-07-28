"""Additive coarse-risk guardrail with a matched total-loss-scale control."""

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
import torch.nn.functional as F


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
class AdditiveGuardrailConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_additive_guardrail"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    sample_count: int = 1024
    save: bool = True


def additive_guardrail_weights(
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
) -> torch.Tensor:
    """Restore group 0 to one without changing any detail coefficient weight."""

    weights = analyzer.weights(time).clone()
    coarse = analyzer.group_index.to(weights.device).eq(0)
    if int(coarse.sum()) == 0 or int((~coarse).sum()) == 0:
        raise ValueError("guardrail requires nonempty coarse and detail groups")
    weights[:, coarse] = 1.0
    return weights


def time_scale_control_weights(
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
) -> torch.Tensor:
    """Match guardrail total weight while preserving the original directions."""

    original = analyzer.weights(time)
    target_mean = additive_guardrail_weights(analyzer, time).mean(dim=1, keepdim=True)
    return original * target_mean


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    time: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    error = analyzer.transform(prediction - target).square()
    if variant == "additive_guardrail":
        weights = additive_guardrail_weights(analyzer, time)
    elif variant == "time_scale_control":
        weights = time_scale_control_weights(analyzer, time)
    else:
        raise ValueError(f"unknown guardrail variant: {variant}")
    return torch.mean(error * weights.to(error.dtype))


def train_guardrail_fields(
    clean: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    study_config,
    *,
    seed: int,
) -> tuple[dict[str, TinyVelocityUNet], pd.DataFrame]:
    device = clean.device
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    initial = TinyVelocityUNet(study_config.width, study_config.depth).to(device)
    models = {
        "additive_guardrail": initial,
        "time_scale_control": copy.deepcopy(initial),
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
            len(data), 1.0, device=device, generator=generator
        )
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            prediction = model(state, time)
            loss = _weighted_loss(prediction, target, analyzer, time, name)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
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


def run_guardrail_pair(
    config: AdditiveGuardrailConfig,
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
    guardrail_models, history = train_guardrail_fields(
        train, analyzer, study_config, seed=seed
    )
    models = {**raw_models, **guardrail_models}
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
        raise RuntimeError("guardrail classifier differs from source by more than one point")
    rollout, rollout_bands, samples = evaluate_rollouts(
        models,
        test,
        classifier,
        analyzer,
        _toy_config(study_config, seed, device, config.sample_count),
        loaded["normalization"],
    )
    references = {
        "baseline": "baseline",
        "weighted": "baseline",
        "additive_guardrail": "baseline",
        "time_scale_control": "baseline",
    }
    summary = summarize_variants(
        teacher, teacher_bands, rollout, references=references
    )
    summary.insert(0, "seed", seed)
    summary.insert(0, "basis", basis)
    summary.insert(0, "dataset", study_config.dataset)
    summary["classifier_accuracy"] = float(accuracy)
    metadata: dict[str, object] = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "source_run_dir": str(source_run_dir),
        "source_baseline_hash": source_summary["baseline_hash"],
        "classifier_accuracy": float(accuracy),
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
                    name: model.state_dict() for name, model in guardrail_models.items()
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
    tasks: Sequence[tuple[AdditiveGuardrailConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    results = []
    for config, source_run_dir, device, result_dir in tasks:
        summary, metadata = run_guardrail_pair(
            config,
            source_run_dir=source_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = summary.set_index("variant")
        print(
            f"done {metadata['dataset']} {metadata['basis']} seed={metadata['seed']}: "
            f"raw={selected.loc['weighted', 'feature_fid_over_reference']:.3f}, "
            f"guard={selected.loc['additive_guardrail', 'feature_fid_over_reference']:.3f}, "
            f"scale={selected.loc['time_scale_control', 'feature_fid_over_reference']:.3f}"
        )
        results.append((summary, metadata))
    return results


def run_additive_guardrail_study(
    config: AdditiveGuardrailConfig,
) -> tuple[pd.DataFrame, Path | None]:
    config = AdditiveGuardrailConfig(
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
            / f"{study_config.dataset}_additive_guardrail_preregistered_{timestamp}"
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
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = AdditiveGuardrailConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or AdditiveGuardrailConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    summary, result_dir = run_additive_guardrail_study(config)
    print(f"result_dir={result_dir}")
    selected = summary[
        summary["variant"].isin(
            ("weighted", "additive_guardrail", "time_scale_control")
        )
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

"""Parameter-matched split-path test of the coarse/detail Pareto mechanism."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    configure_fp32,
    evaluate_rollouts,
    evaluate_teacher_path,
    shifted_uniform,
    train_feature_classifier,
)
from experiments.small_image_additive_guardrail import (  # noqa: E402
    additive_guardrail_weights,
    time_scale_control_weights,
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
    SplitVelocityField,
    _toy_config,
    parameter_counts,
    summarize_variants,
)


@dataclass(frozen=True)
class SplitGuardrailConfig:
    study_dir: Path
    rescue_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_split_guardrail"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    split_width: int = 17
    sample_count: int = 1024
    save: bool = True


def branchwise_losses(
    model: SplitVelocityField,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disjoint coarse/detail losses and coefficient-space raw MSE."""

    coarse_prediction = analyzer.transform(model.coarse(state, time))
    detail_prediction = analyzer.transform(model.detail(state, time))
    target_coefficients = analyzer.transform(target)
    coarse = analyzer.group_index.to(state.device).eq(0)
    detail = ~coarse
    if variant == "split_baseline":
        weights = torch.ones_like(target_coefficients)
    elif variant == "split_additive_guardrail":
        weights = additive_guardrail_weights(analyzer, time).to(target_coefficients.dtype)
    elif variant == "split_time_scale_control":
        weights = time_scale_control_weights(analyzer, time).to(target_coefficients.dtype)
    else:
        raise ValueError(f"unknown split variant: {variant}")
    denominator = float(target_coefficients.numel())
    coarse_error = coarse_prediction[:, coarse] - target_coefficients[:, coarse]
    detail_error = detail_prediction[:, detail] - target_coefficients[:, detail]
    coarse_loss = torch.sum(coarse_error.square() * weights[:, coarse]) / denominator
    detail_loss = torch.sum(detail_error.square() * weights[:, detail]) / denominator
    raw_mse = (
        coarse_error.square().sum() + detail_error.square().sum()
    ) / denominator
    return coarse_loss, detail_loss, raw_mse


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_split_guardrail_fields(
    clean: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    study_config,
    *,
    split_width: int,
    seed: int,
) -> tuple[dict[str, SplitVelocityField], pd.DataFrame, dict[str, str]]:
    device = clean.device
    torch.manual_seed(int(seed) + 13_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 13_001)
    initial = SplitVelocityField(
        analyzer, width=int(split_width), depth=study_config.depth
    ).to(device)
    models = {
        "split_baseline": initial,
        "split_additive_guardrail": copy.deepcopy(initial),
        "split_time_scale_control": copy.deepcopy(initial),
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
            coarse_loss, detail_loss, raw_mse = branchwise_losses(
                model, state, target, time, analyzer, name
            )
            optimizers[name].zero_grad(set_to_none=True)
            coarse_loss.backward()
            detail_loss.backward()
            model.clip_grad_norm_(1.0)
            optimizers[name].step()
            if step == 1 or step % log_every == 0 or step == study_config.steps:
                rows.append(
                    {
                        "step": int(step),
                        "variant": name,
                        "training_objective": float(
                            (coarse_loss + detail_loss).detach()
                        ),
                        "raw_mse": float(raw_mse.detach()),
                    }
                )
    hashes = {
        name: _state_hash(model.coarse) for name, model in models.items()
    }
    if hashes["split_baseline"] != hashes["split_additive_guardrail"]:
        raise RuntimeError("baseline and additive coarse paths are not bitwise identical")
    return models, pd.DataFrame(rows), hashes


def run_split_guardrail_pair(
    config: SplitGuardrailConfig,
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
    split_models, history, hashes = train_split_guardrail_fields(
        train,
        analyzer,
        study_config,
        split_width=config.split_width,
        seed=seed,
    )
    models: dict[str, nn.Module] = {**raw_models, **split_models}
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
        raise RuntimeError("split guardrail classifier differs by more than one point")
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
        "split_baseline": "split_baseline",
        "split_additive_guardrail": "split_baseline",
        "split_time_scale_control": "split_baseline",
    }
    summary = summarize_variants(
        teacher, teacher_bands, rollout, references=references
    )
    summary.insert(0, "seed", seed)
    summary.insert(0, "basis", basis)
    summary.insert(0, "dataset", study_config.dataset)
    summary["classifier_accuracy"] = float(accuracy)
    old_summary = pd.read_csv(
        config.rescue_dir / f"{basis}_seed{seed}" / "variant_summary.csv"
    ).set_index("variant")
    new_split_fid = float(
        summary.set_index("variant").loc["split_baseline", "feature_fid"]
    )
    old_split_fid = float(old_summary.loc["split_baseline", "feature_fid"])
    counts = parameter_counts(study_config.width, config.split_width, study_config.depth)
    metadata: dict[str, object] = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "source_run_dir": str(source_run_dir),
        "source_baseline_hash": source_summary["baseline_hash"],
        "classifier_accuracy": float(accuracy),
        "coarse_hash_baseline": hashes["split_baseline"],
        "coarse_hash_guardrail": hashes["split_additive_guardrail"],
        "coarse_hash_equal": hashes["split_baseline"]
        == hashes["split_additive_guardrail"],
        "p14_split_baseline_feature_fid": old_split_fid,
        "p16_split_baseline_feature_fid": new_split_fid,
        "p16_over_p14_split_baseline_fid": new_split_fid / max(old_split_fid, 1e-12),
        "raw_parameters": counts["raw"],
        "split_parameters": counts["split"],
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
                    name: model.state_dict() for name, model in split_models.items()
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
    tasks: Sequence[tuple[SplitGuardrailConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    results = []
    for config, source_run_dir, device, result_dir in tasks:
        summary, metadata = run_split_guardrail_pair(
            config,
            source_run_dir=source_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = summary.set_index("variant")
        print(
            f"done {metadata['dataset']} {metadata['basis']} seed={metadata['seed']}: "
            f"raw={selected.loc['weighted', 'feature_fid_over_reference']:.3f}, "
            f"guard={selected.loc['split_additive_guardrail', 'feature_fid_over_reference']:.3f}, "
            f"scale={selected.loc['split_time_scale_control', 'feature_fid_over_reference']:.3f}, "
            f"hash_equal={metadata['coarse_hash_equal']}"
        )
        results.append((summary, metadata))
    return results


def run_split_guardrail_study(
    config: SplitGuardrailConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    config = SplitGuardrailConfig(
        **{
            **asdict(config),
            "study_dir": config.study_dir.expanduser().resolve(),
            "rescue_dir": config.rescue_dir.expanduser().resolve(),
        }
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
            / f"{study_config.dataset}_split_guardrail_preregistered_{timestamp}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        for key in ("study_dir", "rescue_dir", "output_root"):
            serialized[key] = str(serialized[key])
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
    parser.add_argument("--rescue-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--split-width", type=int, default=17)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = SplitGuardrailConfig(
        study_dir=args.study_dir,
        rescue_dir=args.rescue_dir,
        output_root=args.output_root or SplitGuardrailConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        split_width=args.split_width,
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    summary, metadata, result_dir = run_split_guardrail_study(config)
    print(f"result_dir={result_dir}")
    print(
        summary[
            summary["variant"].isin(
                (
                    "weighted",
                    "split_additive_guardrail",
                    "split_time_scale_control",
                )
            )
        ]
        .groupby(["basis", "variant"])[
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
    print(
        metadata.groupby("basis")[
            ["coarse_hash_equal", "p16_over_p14_split_baseline_fid"]
        ]
        .agg(["mean", "min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()

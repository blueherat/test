"""Full-width detail path plus a small protected coarse-path mechanism test."""

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
import torch.nn.functional as F
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
class AsymmetricGuardrailConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_asymmetric_guardrail"
    bases: tuple[str, ...] = ("dct", "pca")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    detail_width: int = 24
    coarse_width: int = 12
    wide_width: int = 27
    sample_count: int = 1024
    save: bool = True


class AsymmetricVelocityField(nn.Module):
    def __init__(
        self,
        analyzer: OrthogonalDirectionLoss,
        *,
        detail_width: int,
        coarse_width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.coarse = TinyVelocityUNet(coarse_width, depth)
        self.detail = TinyVelocityUNet(detail_width, depth)
        self.register_buffer("basis", analyzer.basis.detach().clone(), persistent=True)
        self.register_buffer(
            "coarse_mask",
            analyzer.group_index.eq(0).to(torch.float32),
            persistent=True,
        )

    def project(self, value: torch.Tensor, *, coarse: bool) -> torch.Tensor:
        coefficients = value.flatten(1) @ self.basis.to(value.dtype)
        mask = self.coarse_mask if coarse else 1.0 - self.coarse_mask
        return (
            (coefficients * mask.to(value.dtype)[None]) @ self.basis.to(value.dtype).T
        ).reshape_as(value)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.project(self.coarse(value, time), coarse=True) + self.project(
            self.detail(value, time), coarse=False
        )

    def clip_grad_norm_(self, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(self.coarse.parameters(), float(max_norm))
        torch.nn.utils.clip_grad_norm_(self.detail.parameters(), float(max_norm))


def asymmetric_parameter_counts(
    detail_width: int,
    coarse_width: int,
    wide_width: int,
    depth: int,
) -> dict[str, int]:
    count = lambda width: sum(  # noqa: E731
        parameter.numel() for parameter in TinyVelocityUNet(width, depth).parameters()
    )
    return {
        "asymmetric": int(count(detail_width) + count(coarse_width)),
        "wide": int(count(wide_width)),
    }


def asymmetric_losses(
    model: AsymmetricVelocityField,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    *,
    weighted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_prediction = analyzer.transform(model.coarse(state, time))
    detail_prediction = analyzer.transform(model.detail(state, time))
    target_coefficients = analyzer.transform(target)
    coarse = analyzer.group_index.to(state.device).eq(0)
    detail = ~coarse
    detail_weights = (
        analyzer.weights(time).to(target_coefficients.dtype)
        if weighted
        else torch.ones_like(target_coefficients)
    )
    denominator = float(target_coefficients.numel())
    coarse_error = coarse_prediction[:, coarse] - target_coefficients[:, coarse]
    detail_error = detail_prediction[:, detail] - target_coefficients[:, detail]
    coarse_loss = coarse_error.square().sum() / denominator
    detail_loss = (
        detail_error.square() * detail_weights[:, detail]
    ).sum() / denominator
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


def train_asymmetric_fields(
    clean: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    study_config,
    config: AsymmetricGuardrailConfig,
    *,
    seed: int,
) -> tuple[dict[str, nn.Module], pd.DataFrame, dict[str, str]]:
    device = clean.device
    torch.manual_seed(int(seed) + 17_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 17_001)
    asymmetric_initial = AsymmetricVelocityField(
        analyzer,
        detail_width=config.detail_width,
        coarse_width=config.coarse_width,
        depth=study_config.depth,
    ).to(device)
    torch.manual_seed(int(seed) + 19_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 19_001)
    wide_initial = TinyVelocityUNet(config.wide_width, study_config.depth).to(device)
    models: dict[str, nn.Module] = {
        "asym_baseline": asymmetric_initial,
        "asym_weighted": copy.deepcopy(asymmetric_initial),
        "wide_baseline": wide_initial,
        "wide_weighted": copy.deepcopy(wide_initial),
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
            optimizers[name].zero_grad(set_to_none=True)
            if isinstance(model, AsymmetricVelocityField):
                coarse_loss, detail_loss, raw_mse = asymmetric_losses(
                    model,
                    state,
                    target,
                    time,
                    analyzer,
                    weighted=name == "asym_weighted",
                )
                coarse_loss.backward()
                detail_loss.backward()
                model.clip_grad_norm_(1.0)
                objective = coarse_loss + detail_loss
            else:
                prediction = model(state, time)
                if name == "wide_weighted":
                    objective = analyzer(prediction, target, time)[0].mean()
                else:
                    objective = F.mse_loss(prediction, target)
                raw_mse = F.mse_loss(prediction.detach(), target)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if step == 1 or step % log_every == 0 or step == study_config.steps:
                rows.append(
                    {
                        "step": int(step),
                        "variant": name,
                        "training_objective": float(objective.detach()),
                        "raw_mse": float(raw_mse.detach()),
                    }
                )
    hashes = {
        name: _state_hash(model.coarse)
        for name, model in models.items()
        if isinstance(model, AsymmetricVelocityField)
    }
    if hashes["asym_baseline"] != hashes["asym_weighted"]:
        raise RuntimeError("asymmetric coarse paths are not bitwise identical")
    return models, pd.DataFrame(rows), hashes


def run_asymmetric_pair(
    config: AsymmetricGuardrailConfig,
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
    intervention_models, history, hashes = train_asymmetric_fields(
        train, analyzer, study_config, config, seed=seed
    )
    models = {**raw_models, **intervention_models}
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
        raise RuntimeError("asymmetric classifier differs by more than one point")
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
        "asym_baseline": "asym_baseline",
        "asym_weighted": "asym_baseline",
        "wide_baseline": "wide_baseline",
        "wide_weighted": "wide_baseline",
    }
    summary = summarize_variants(
        teacher, teacher_bands, rollout, references=references
    )
    summary.insert(0, "seed", seed)
    summary.insert(0, "basis", basis)
    summary.insert(0, "dataset", study_config.dataset)
    summary["classifier_accuracy"] = float(accuracy)
    values = summary.set_index("variant")
    counts = asymmetric_parameter_counts(
        config.detail_width,
        config.coarse_width,
        config.wide_width,
        study_config.depth,
    )
    metadata: dict[str, object] = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "source_run_dir": str(source_run_dir),
        "classifier_accuracy": float(accuracy),
        "coarse_hash_baseline": hashes["asym_baseline"],
        "coarse_hash_weighted": hashes["asym_weighted"],
        "coarse_hash_equal": hashes["asym_baseline"] == hashes["asym_weighted"],
        "asymmetric_parameters": counts["asymmetric"],
        "wide_parameters": counts["wide"],
        "parameter_ratio": counts["asymmetric"] / counts["wide"],
        "asym_baseline_over_raw_baseline_fid": float(
            values.loc["asym_baseline", "feature_fid"]
            / max(float(values.loc["baseline", "feature_fid"]), 1e-12)
        ),
        "wide_baseline_over_raw_baseline_fid": float(
            values.loc["wide_baseline", "feature_fid"]
            / max(float(values.loc["baseline", "feature_fid"]), 1e-12)
        ),
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
                    name: model.state_dict()
                    for name, model in intervention_models.items()
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
    tasks: Sequence[tuple[AsymmetricGuardrailConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    results = []
    for config, source_run_dir, device, result_dir in tasks:
        summary, metadata = run_asymmetric_pair(
            config,
            source_run_dir=source_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = summary.set_index("variant")
        print(
            f"done {metadata['dataset']} {metadata['basis']} seed={metadata['seed']}: "
            f"raw={selected.loc['weighted', 'feature_fid_over_reference']:.3f}, "
            f"asym={selected.loc['asym_weighted', 'feature_fid_over_reference']:.3f}, "
            f"wide={selected.loc['wide_weighted', 'feature_fid_over_reference']:.3f}, "
            f"hash_equal={metadata['coarse_hash_equal']}"
        )
        results.append((summary, metadata))
    return results


def run_asymmetric_guardrail_study(
    config: AsymmetricGuardrailConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    config = AsymmetricGuardrailConfig(
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
            / f"{study_config.dataset}_asymmetric_guardrail_preregistered_{timestamp}"
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
    parser.add_argument("--detail-width", type=int, default=24)
    parser.add_argument("--coarse-width", type=int, default=12)
    parser.add_argument("--wide-width", type=int, default=27)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = AsymmetricGuardrailConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or AsymmetricGuardrailConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        detail_width=args.detail_width,
        coarse_width=args.coarse_width,
        wide_width=args.wide_width,
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    summary, metadata, result_dir = run_asymmetric_guardrail_study(config)
    print(f"result_dir={result_dir}")
    print(
        summary[
            summary["variant"].isin(("weighted", "asym_weighted", "wide_weighted"))
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
            ["coarse_hash_equal", "asym_baseline_over_raw_baseline_fid"]
        ]
        .agg(["mean", "min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()

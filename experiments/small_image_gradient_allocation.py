"""Held-out gradient allocation audit for coarse/detail transport tasks."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import configure_fp32  # noqa: E402
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    OrthogonalDirectionLoss,
    load_small_image_tensors,
)


@dataclass(frozen=True)
class GradientAllocationConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_gradient_allocation"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    audit_count: int = 256
    batch_size: int = 128
    times: tuple[float, ...] = (0.7, 0.9)
    save: bool = True


def gradient_metrics(
    coarse_unweighted: torch.Tensor,
    detail_unweighted: torch.Tensor,
    coarse_weighted: torch.Tensor,
    detail_weighted: torch.Tensor,
) -> dict[str, float]:
    def norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value)

    def cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.dot(first, second) / (
            norm(first) * norm(second)
        ).clamp_min(1e-20)

    baseline_total = coarse_unweighted + detail_unweighted
    weighted_total = coarse_weighted + detail_weighted
    unweighted_ratio = norm(detail_unweighted) / norm(coarse_unweighted).clamp_min(1e-20)
    weighted_ratio = norm(detail_weighted) / norm(coarse_weighted).clamp_min(1e-20)
    coarse_descent_baseline = torch.dot(coarse_unweighted, baseline_total)
    coarse_descent_weighted = torch.dot(coarse_unweighted, weighted_total)
    detail_descent_baseline = torch.dot(detail_unweighted, baseline_total)
    detail_descent_weighted = torch.dot(detail_unweighted, weighted_total)
    return {
        "coarse_detail_cosine_unweighted": float(
            cosine(coarse_unweighted, detail_unweighted)
        ),
        "coarse_detail_cosine_weighted": float(
            cosine(coarse_weighted, detail_weighted)
        ),
        "detail_over_coarse_norm_unweighted": float(unweighted_ratio),
        "detail_over_coarse_norm_weighted": float(weighted_ratio),
        "allocation_multiplier": float(weighted_ratio / unweighted_ratio.clamp_min(1e-20)),
        "coarse_descent_baseline": float(coarse_descent_baseline),
        "coarse_descent_weighted": float(coarse_descent_weighted),
        "coarse_descent_ratio": float(
            coarse_descent_weighted / coarse_descent_baseline.abs().clamp_min(1e-20)
        ),
        "detail_descent_baseline": float(detail_descent_baseline),
        "detail_descent_weighted": float(detail_descent_weighted),
        "detail_descent_ratio": float(
            detail_descent_weighted / detail_descent_baseline.abs().clamp_min(1e-20)
        ),
        "coarse_gradient_norm": float(norm(coarse_unweighted)),
        "detail_gradient_norm": float(norm(detail_unweighted)),
    }


def _flatten_group(
    gradients: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> torch.Tensor:
    return torch.cat([gradients[index].reshape(-1) for index in indices])


def _parameter_groups(model: torch.nn.Module) -> dict[str, list[int]]:
    names = [name for name, _ in model.named_parameters()]
    output_head = [
        index
        for index, name in enumerate(names)
        if name in ("output.weight", "output.bias")
    ]
    if not output_head:
        raise RuntimeError("failed to identify final output convolution")
    return {
        "all": list(range(len(names))),
        "shared_trunk": [index for index in range(len(names)) if index not in output_head],
        "output_head": output_head,
    }


def _losses(
    model: torch.nn.Module,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = model(state, time)
    squared_error = analyzer.transform(prediction - target).square()
    weights = analyzer.weights(time).to(squared_error.dtype)
    coarse = analyzer.group_index.to(state.device).eq(0)
    detail = ~coarse
    denominator = float(squared_error.numel())
    return (
        squared_error[:, coarse].sum() / denominator,
        squared_error[:, detail].sum() / denominator,
        (squared_error[:, coarse] * weights[:, coarse]).sum() / denominator,
        (squared_error[:, detail] * weights[:, detail]).sum() / denominator,
    )


def audit_checkpoint(
    model: torch.nn.Module,
    analyzer: OrthogonalDirectionLoss,
    clean: torch.Tensor,
    *,
    basis: str,
    seed: int,
    checkpoint_variant: str,
    config: GradientAllocationConfig,
) -> pd.DataFrame:
    parameters = tuple(model.parameters())
    groups = _parameter_groups(model)
    generator = torch.Generator(device=clean.device).manual_seed(int(seed) + 23_003)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    rows: list[dict[str, float | int | str]] = []
    for time_value in config.times:
        time_value = float(time_value)
        for batch_index, start in enumerate(range(0, len(clean), config.batch_size)):
            data = clean[start : start + config.batch_size]
            batch_noise = noise[start : start + config.batch_size]
            time = torch.full((len(data),), time_value, device=clean.device)
            expanded = time[:, None, None, None]
            state = (1.0 - expanded) * data + expanded * batch_noise
            target = batch_noise - data
            losses = _losses(model, state, target, time, analyzer)
            gradients = []
            for index, loss in enumerate(losses):
                gradients.append(
                    torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=index < len(losses) - 1,
                        create_graph=False,
                    )
                )
            for layer_group, indices in groups.items():
                vectors = [
                    _flatten_group(gradient, indices).detach() for gradient in gradients
                ]
                rows.append(
                    {
                        "basis": basis,
                        "seed": int(seed),
                        "checkpoint_variant": checkpoint_variant,
                        "time": time_value,
                        "batch": int(batch_index),
                        "parameter_group": layer_group,
                        **gradient_metrics(*vectors),
                    }
                )
    return pd.DataFrame(rows)


def audit_run(
    config: GradientAllocationConfig,
    source_run_dir: Path,
    device_name: str,
) -> pd.DataFrame:
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
    clean = loaded["test"][: config.audit_count].to(device)
    models, analyzer, _ = _load_run(source_run_dir, study_config, device)
    frames = []
    for variant in ("baseline", "weighted"):
        models[variant].train(False)
        frames.append(
            audit_checkpoint(
                models[variant],
                analyzer,
                clean,
                basis=basis,
                seed=seed,
                checkpoint_variant=variant,
                config=config,
            )
        )
    for model in models.values():
        model.cpu()
    analyzer.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def _run_group(
    tasks: Sequence[tuple[GradientAllocationConfig, Path, str]],
) -> list[pd.DataFrame]:
    frames = []
    for config, source_run_dir, device in tasks:
        frame = audit_run(config, source_run_dir, device)
        frames.append(frame)
        print(f"audited gradients {source_run_dir.name}")
    return frames


def summarize_gradient_allocation(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["basis", "seed", "checkpoint_variant", "parameter_group"],
            as_index=False,
        )[
            [
                "coarse_detail_cosine_unweighted",
                "coarse_detail_cosine_weighted",
                "detail_over_coarse_norm_unweighted",
                "detail_over_coarse_norm_weighted",
                "allocation_multiplier",
                "coarse_descent_ratio",
                "detail_descent_ratio",
            ]
        ]
        .mean()
    )


def run_gradient_allocation_study(
    config: GradientAllocationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    config = GradientAllocationConfig(
        **{**asdict(config), "study_dir": config.study_dir.expanduser().resolve()}
    )
    source = pd.read_csv(config.study_dir / "study_summary.csv")
    source = source[source["basis"].isin(config.bases)].copy()
    missing = set(config.bases) - set(source["basis"])
    if missing:
        raise ValueError(f"source study is missing bases: {sorted(missing)}")
    devices = config.devices or ("cpu",)
    tasks = [
        (config, Path(row.run_dir), devices[index % len(devices)])
        for index, row in enumerate(source.itertuples())
    ]
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    frames: list[pd.DataFrame] = []
    if len(devices) == 1:
        frames = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in grouped if group]
            for future in as_completed(futures):
                frames.extend(future.result())
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_gradient_allocation(metrics)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = (
            config.output_root.expanduser()
            / f"gradient_allocation_preregistered_{timestamp}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["study_dir"] = str(config.study_dir)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "gradient_metrics.csv", index=False)
        summary.to_csv(result_dir / "gradient_summary.csv", index=False)
    return metrics, summary, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--audit-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bases = tuple(value.strip() for value in args.bases.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = GradientAllocationConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or GradientAllocationConfig.output_root,
        bases=bases,
        devices=devices or ("cpu",),
        audit_count=args.audit_count,
        batch_size=args.batch_size,
        save=not args.no_save,
    )
    _, summary, result_dir = run_gradient_allocation_study(config)
    print(f"result_dir={result_dir}")
    print(
        summary.groupby(["basis", "checkpoint_variant", "parameter_group"])[
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

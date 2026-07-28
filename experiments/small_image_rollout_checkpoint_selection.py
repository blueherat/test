"""Prospective train-only rollout checkpoint-selection experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

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
    shifted_uniform,
    train_feature_classifier,
)
from experiments.small_image_basis_mechanism import _load_study_config  # noqa: E402
from experiments.small_image_basis_transport import (  # noqa: E402
    DATASETS,
    _toy_config,
    build_direction_analyzer,
    load_small_image_tensors,
)
from experiments.small_image_signed_leverage import (  # noqa: E402
    canonical_band_energies,
    differentiable_euler_sample,
    endpoint_moment_loss,
)


METRICS = ("feature_fid", "feature_swd", "decoded_pixel_swd", "latent_swd")
STRUCTURED_BASES = ("dct", "pca")


@dataclass(frozen=True)
class RolloutSelectionConfig:
    study_dir: Path
    output_root: Path = (
        Path.home()
        / "data/eqvae/experiments/small_image_rollout_checkpoint_selection"
    )
    bases: tuple[str, ...] = ("dct", "pca", "random")
    training_seeds: tuple[int, ...] = (5, 6, 7, 8, 9)
    evaluation_seeds: tuple[int, ...] = (7101, 7102, 7103)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    checkpoints: tuple[int, ...] = (200, 300, 400, 500, 600, 700, 800, 900, 1000)
    selection_count: int = 256
    selection_seed: int = 6101
    test_size: int = 4096
    sample_count: int = 4096
    evaluation_batch_size: int = 256
    ode_steps: int = 50
    save: bool = True


def _index_hash(indices: torch.Tensor) -> str:
    return hashlib.sha256(indices.cpu().contiguous().numpy().tobytes()).hexdigest()


def select_checkpoint(
    history: Sequence[Mapping[str, float | int]],
) -> tuple[int, float]:
    if not history:
        raise ValueError("selection history must not be empty")
    ordered = sorted(history, key=lambda row: (float(row["proxy_loss"]), int(row["step"])))
    return int(ordered[0]["step"]), float(ordered[0]["proxy_loss"])


def _heldout_train_reference(
    dataset: str,
    data_root: Path,
    used_indices: torch.Tensor,
    normalization: Mapping[str, float],
    *,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_class = DATASETS[dataset][0]
    raw = dataset_class(root=str(data_root), train=True, download=False).data
    used = torch.zeros(len(raw), dtype=torch.bool)
    used[used_indices.cpu()] = True
    available = torch.arange(len(raw))[~used]
    if int(count) > len(available):
        raise ValueError("selection_count exceeds held-out training images")
    generator = torch.Generator().manual_seed(int(seed))
    selected = available[
        torch.randperm(len(available), generator=generator)[: int(count)]
    ]
    pixels = raw[selected].float().unsqueeze(1) / 255.0
    normalized = (
        pixels - float(normalization["mean"])
    ) / float(normalization["std"])
    return normalized, selected


@torch.no_grad()
def _proxy_losses(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference_energy: torch.Tensor,
    basis: torch.Tensor,
    group_index: torch.Tensor,
    *,
    ode_steps: int,
) -> dict[str, float]:
    losses = {}
    for name, model in models.items():
        generated = differentiable_euler_sample(
            model, initial, ode_steps=int(ode_steps)
        )
        loss, _ = endpoint_moment_loss(
            generated, reference_energy, basis, group_index
        )
        losses[name] = float(loss)
    return losses


def paired_rollout_rows(
    rollout: pd.DataFrame,
    *,
    dataset: str,
    training_seed: int,
    evaluation_seed: int,
    bases: Sequence[str],
) -> list[dict[str, float | int | str]]:
    required = {"variant", *METRICS}
    missing = required - set(rollout.columns)
    if missing:
        raise ValueError(f"rollout is missing columns: {sorted(missing)}")
    indexed = rollout.set_index("variant")
    rows: list[dict[str, float | int | str]] = []

    def add(basis: str, comparison: str, numerator: str, denominator: str) -> None:
        if numerator not in indexed.index or denominator not in indexed.index:
            raise ValueError(f"missing rollout variants: {numerator}, {denominator}")
        for metric in METRICS:
            first = float(indexed.loc[numerator, metric])
            second = float(indexed.loc[denominator, metric])
            rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "evaluation_seed": int(evaluation_seed),
                    "basis": basis,
                    "comparison": comparison,
                    "metric": metric,
                    "numerator": first,
                    "denominator": second,
                    "delta": first - second,
                    "ratio": first / max(second, 1e-12),
                }
            )

    add(
        "baseline",
        "selected_vs_final",
        "baseline_selected",
        "baseline_final",
    )
    for basis in bases:
        add(
            basis,
            "selected_vs_final",
            f"{basis}_selected",
            f"{basis}_final",
        )
        add(
            basis,
            "selected_vs_selected_baseline",
            f"{basis}_selected",
            "baseline_selected",
        )
        add(
            basis,
            "final_vs_final_baseline",
            f"{basis}_final",
            "baseline_final",
        )
    return rows


def summarize_paired_metrics(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "basis", "comparison", "metric", "training_seed"]
    seed_summary = (
        paired.groupby(keys, as_index=False)
        .agg(
            evaluation_seeds=("evaluation_seed", "nunique"),
            ratio=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            delta=("delta", "mean"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    aggregate_keys = ["dataset", "basis", "comparison", "metric"]
    aggregate = (
        seed_summary.groupby(aggregate_keys, as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            ratio_median=("ratio", "median"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            seed_win_rate=("ratio", lambda values: float((values < 1.0).mean())),
            practical_seed_win_rate=(
                "ratio", lambda values: float((values <= 0.98).mean())
            ),
        )
        .sort_values(aggregate_keys)
        .reset_index(drop=True)
    )
    return seed_summary, aggregate


def evaluate_gates(aggregate: pd.DataFrame) -> dict[str, object]:
    feature = aggregate[aggregate["metric"].eq("feature_fid")].copy()

    def row(dataset: str, basis: str, comparison: str) -> pd.Series:
        selected = feature[
            feature["dataset"].eq(dataset)
            & feature["basis"].eq(basis)
            & feature["comparison"].eq(comparison)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one aggregate row for {dataset}/{basis}/{comparison}"
            )
        return selected.iloc[0]

    datasets = sorted(feature["dataset"].unique())
    h1_rows = []
    for dataset in datasets:
        for basis in ("baseline", *STRUCTURED_BASES):
            value = row(dataset, basis, "selected_vs_final")
            h1_rows.append(
                {
                    "dataset": dataset,
                    "basis": basis,
                    "ratio": float(value["ratio_mean"]),
                    "seed_win_rate": float(value["seed_win_rate"]),
                    "pass": bool(
                        value["ratio_mean"] <= 0.98
                        and value["seed_win_rate"] >= 0.6
                    ),
                    "no_degradation": bool(value["ratio_mean"] <= 1.02),
                }
            )
    h2_rows = []
    for dataset in datasets:
        for basis in STRUCTURED_BASES:
            value = row(dataset, basis, "selected_vs_selected_baseline")
            h2_rows.append(
                {
                    "dataset": dataset,
                    "basis": basis,
                    "ratio": float(value["ratio_mean"]),
                    "seed_win_rate": float(value["seed_win_rate"]),
                    "pass": bool(
                        value["ratio_mean"] <= 0.98
                        and value["seed_win_rate"] >= 0.6
                    ),
                }
            )
    h1 = all(item["pass"] and item["no_degradation"] for item in h1_rows)
    h2 = all(item["pass"] for item in h2_rows)
    return {
        "h1_rollout_selection_prevents_late_drift": h1,
        "h2_selection_makes_spectral_weighting_better_than_baseline": h2,
        "h1_rows": h1_rows,
        "h2_rows": h2_rows,
    }


def _build_models(
    study_config,
    analyzers: Mapping[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    baseline = TinyVelocityUNet(study_config.width, study_config.depth).to(device)
    models = {"baseline": baseline}
    for basis in analyzers:
        models[basis] = copy.deepcopy(baseline)
    return models


def _run_seed(
    config: RolloutSelectionConfig,
    training_seed: int,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    study_dir = config.study_dir.expanduser().resolve()
    study_config = _load_study_config(study_dir)
    checkpoints = tuple(sorted(set(int(value) for value in config.checkpoints)))
    if checkpoints[0] < 1 or checkpoints[-1] != study_config.steps:
        raise ValueError("checkpoints must be positive and include the final step")
    configure_fp32(int(training_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name
        if torch.cuda.is_available() or not device_name.startswith("cuda")
        else "cpu"
    )
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        int(config.test_size),
        int(training_seed),
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    selection_reference, selection_indices = _heldout_train_reference(
        study_config.dataset,
        study_config.data_root,
        loaded["train_indices"],
        loaded["normalization"],
        count=int(config.selection_count),
        seed=int(config.selection_seed) + int(training_seed),
    )
    if set(selection_indices.tolist()) & set(loaded["train_indices"].tolist()):
        raise RuntimeError("selection images overlap minibatch update images")
    selection_reference = selection_reference.to(device)
    analyzers = {
        basis: build_direction_analyzer(
            train,
            basis,
            band_count=study_config.band_count,
            gamma=study_config.gamma,
            seed=int(training_seed),
        )[0].to(device)
        for basis in config.bases
    }
    models = _build_models(study_config, analyzers, device)
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=study_config.learning_rate, weight_decay=1e-4
        )
        for name, model in models.items()
    }
    canonical = analyzers["dct"]
    reference_energy = canonical_band_energies(
        selection_reference,
        canonical.basis,
        canonical.group_index,
        canonical.band_count,
    )
    selection_generator = torch.Generator(device=device).manual_seed(
        int(config.selection_seed) + 10_000 + int(training_seed)
    )
    selection_initial = torch.randn(
        selection_reference.shape,
        generator=selection_generator,
        device=device,
    )
    training_generator = torch.Generator(device=device).manual_seed(
        int(training_seed) + 101
    )
    history: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in models
    }
    selected_states: dict[str, dict[str, torch.Tensor]] = {}
    selected_losses: dict[str, float] = {}
    selected_steps: dict[str, int] = {}
    checkpoint_set = set(checkpoints)

    for step in range(1, int(study_config.steps) + 1):
        indices = torch.randint(
            len(train),
            (int(study_config.batch_size),),
            device=device,
            generator=training_generator,
        )
        data = train[indices]
        noise = torch.randn(data.shape, device=device, generator=training_generator)
        time = shifted_uniform(
            len(data), 1.0, device=device, generator=training_generator
        )
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            prediction = model(state, time)
            loss = (
                F.mse_loss(prediction, target)
                if name == "baseline"
                else analyzers[name](prediction, target, time)[0].mean()
            )
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
        if step in checkpoint_set:
            losses = _proxy_losses(
                models,
                selection_initial,
                reference_energy,
                canonical.basis,
                canonical.group_index,
                ode_steps=int(config.ode_steps),
            )
            for name, proxy_loss in losses.items():
                history[name].append({"step": int(step), "proxy_loss": proxy_loss})
                if name not in selected_losses or proxy_loss < selected_losses[name]:
                    selected_losses[name] = proxy_loss
                    selected_steps[name] = int(step)
                    selected_states[name] = {
                        key: value.detach().cpu().clone()
                        for key, value in models[name].state_dict().items()
                    }

    final_states = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        for name, model in models.items()
    }
    for name in models:
        expected_step, expected_loss = select_checkpoint(history[name])
        if selected_steps[name] != expected_step or selected_losses[name] != expected_loss:
            raise RuntimeError(f"online selection disagrees with history for {name}")

    classifier, classifier_accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        loaded["test_labels"].to(device),
        epochs=study_config.classifier_epochs,
        batch_size=study_config.classifier_batch_size,
        seed=int(training_seed),
    )
    evaluation_models: dict[str, torch.nn.Module] = {}
    for name in models:
        for checkpoint_name, states in (
            ("final", final_states),
            ("selected", selected_states),
        ):
            model = TinyVelocityUNet(study_config.width, study_config.depth).to(device)
            model.load_state_dict(states[name])
            model.eval()
            evaluation_models[f"{name}_{checkpoint_name}"] = model

    rollout_frames = []
    paired_rows: list[dict[str, float | int | str]] = []
    for evaluation_seed in config.evaluation_seeds:
        toy_config = replace(
            _toy_config(
                study_config,
                seed=int(evaluation_seed),
                device=str(device),
                output_root=config.output_root,
            ),
            sample_count=min(int(config.sample_count), len(test)),
            batch_size=int(config.evaluation_batch_size),
            ode_steps=int(config.ode_steps),
        )
        rollout, _, _ = evaluate_rollouts(
            evaluation_models,
            test,
            classifier,
            canonical,
            toy_config,
            loaded["normalization"],
        )
        rollout.insert(0, "dataset", study_config.dataset)
        rollout.insert(1, "training_seed", int(training_seed))
        rollout.insert(2, "evaluation_seed", int(evaluation_seed))
        rollout_frames.append(rollout)
        paired_rows.extend(
            paired_rollout_rows(
                rollout,
                dataset=study_config.dataset,
                training_seed=int(training_seed),
                evaluation_seed=int(evaluation_seed),
                bases=config.bases,
            )
        )

    history_rows = []
    for name, values in history.items():
        for value in values:
            history_rows.append(
                {
                    "dataset": study_config.dataset,
                    "training_seed": int(training_seed),
                    "variant": name,
                    **value,
                    "selected": int(value["step"]) == selected_steps[name],
                }
            )
    metadata = {
        "dataset": study_config.dataset,
        "training_seed": int(training_seed),
        "classifier_accuracy": float(classifier_accuracy),
        "selected_steps": selected_steps,
        "selected_proxy_losses": selected_losses,
        "update_indices_hash": _index_hash(loaded["train_indices"]),
        "selection_indices_hash": _index_hash(selection_indices),
        "test_indices_hash": _index_hash(loaded["test_indices"]),
        "update_selection_overlap": 0,
    }
    return (
        pd.concat(rollout_frames, ignore_index=True),
        pd.DataFrame(paired_rows),
        {"metadata": metadata, "history": history_rows},
    )


def run_rollout_selection_study(
    config: RolloutSelectionConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    Path | None,
]:
    if set(config.bases) != {"dct", "pca", "random"}:
        raise ValueError("prospective study requires dct, pca, and random bases")
    devices = config.devices or ("cpu",)
    tasks = [
        (config, int(seed), devices[index % len(devices)])
        for index, seed in enumerate(config.training_seeds)
    ]
    if len(tasks) == 1:
        results = [_run_seed(*tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(tasks), len(devices)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_seed, *task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    rollout = pd.concat([result[0] for result in results], ignore_index=True)
    paired = pd.concat([result[1] for result in results], ignore_index=True)
    seed_summary, aggregate = summarize_paired_metrics(paired)
    gates = evaluate_gates(aggregate)
    history = pd.DataFrame(
        [row for result in results for row in result[2]["history"]]
    ).sort_values(["training_seed", "variant", "step"])
    metadata = [result[2]["metadata"] for result in results]
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"prospective_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        for key in ("study_dir", "output_root"):
            serialized[key] = str(serialized[key])
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        (result_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (result_dir / "gates.json").write_text(
            json.dumps(gates, indent=2), encoding="utf-8"
        )
        rollout.to_csv(result_dir / "rollout_metrics.csv", index=False)
        paired.to_csv(result_dir / "paired_metrics.csv", index=False)
        seed_summary.to_csv(result_dir / "seed_summary.csv", index=False)
        aggregate.to_csv(result_dir / "aggregate_summary.csv", index=False)
        history.to_csv(result_dir / "selection_history.csv", index=False)
    return rollout, paired, seed_summary, aggregate, gates, result_dir


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-seeds", default="5,6,7,8,9")
    parser.add_argument("--evaluation-seeds", default="7101,7102,7103")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--checkpoints", default="200,300,400,500,600,700,800,900,1000")
    parser.add_argument("--selection-count", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=6101)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = RolloutSelectionConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or RolloutSelectionConfig.output_root,
        training_seeds=_integers(args.training_seeds),
        evaluation_seeds=_integers(args.evaluation_seeds),
        devices=_strings(args.devices) or ("cpu",),
        checkpoints=_integers(args.checkpoints),
        selection_count=args.selection_count,
        selection_seed=args.selection_seed,
        test_size=args.test_size,
        sample_count=args.sample_count,
        evaluation_batch_size=args.evaluation_batch_size,
        ode_steps=args.ode_steps,
        save=not args.no_save,
    )
    _, _, _, aggregate, gates, result_dir = run_rollout_selection_study(config)
    feature = aggregate[aggregate["metric"].eq("feature_fid")]
    print(feature.round(4).to_string(index=False))
    print(json.dumps(gates, indent=2, ensure_ascii=False))
    print(f"result_dir={result_dir}")


if __name__ == "__main__":
    main()

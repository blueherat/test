"""Replay paired training and audit endpoint moments along the optimization path."""

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

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    TinyVelocityUNet,
    configure_fp32,
    shifted_uniform,
)
from experiments.small_image_basis_mechanism import _load_study_config  # noqa: E402
from experiments.small_image_basis_transport import (  # noqa: E402
    DATASETS,
    _state_hash,
    build_direction_analyzer,
    load_small_image_tensors,
)
from experiments.small_image_signed_leverage import (  # noqa: E402
    canonical_band_energies,
    differentiable_euler_sample,
    endpoint_moment_loss,
)


@dataclass(frozen=True)
class TrainingPathConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_training_path"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    training_seeds: tuple[int, ...] = (3, 4)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    checkpoints: tuple[int, ...] = (0, 25, 50, 100, 200, 400, 600, 800, 1000)
    endpoint_count: int = 128
    endpoint_seed: int = 5101
    ode_steps: int = 50
    save: bool = True


@torch.no_grad()
def endpoint_snapshot(
    models: dict[str, torch.nn.Module],
    initial: torch.Tensor,
    reference_energy: torch.Tensor,
    basis: torch.Tensor,
    group_index: torch.Tensor,
    *,
    ode_steps: int,
) -> list[dict[str, float | str]]:
    rows = []
    for variant, model in models.items():
        generated = differentiable_euler_sample(
            model, initial, ode_steps=int(ode_steps)
        )
        loss, log_gap = endpoint_moment_loss(
            generated, reference_energy, basis, group_index
        )
        row: dict[str, float | str] = {
            "variant": variant,
            "endpoint_moment_loss": float(loss),
        }
        for band, value in enumerate(log_gap):
            row[f"band{band}_log_gap"] = float(value)
        rows.append(row)
    return rows


def summarize_training_path(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "basis",
        "training_seed",
        "step",
        "variant",
        "endpoint_moment_loss",
        "band0_log_gap",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing columns: {sorted(missing)}")
    index = ["dataset", "basis", "training_seed", "step"]
    pivot = metrics.pivot(index=index, columns="variant")
    rows = pd.DataFrame(index=pivot.index).reset_index()
    for metric in ("endpoint_moment_loss", "band0_log_gap"):
        rows[f"baseline_{metric}"] = pivot[(metric, "baseline")].to_numpy()
        rows[f"weighted_{metric}"] = pivot[(metric, "weighted")].to_numpy()
        rows[f"weighted_minus_baseline_{metric}"] = (
            rows[f"weighted_{metric}"] - rows[f"baseline_{metric}"]
        )
    return rows.sort_values(["dataset", "training_seed", "basis", "step"]).reset_index(
        drop=True
    )


def _run_pair(
    config: TrainingPathConfig,
    basis_name: str,
    training_seed: int,
    device_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    study_dir = config.study_dir.expanduser().resolve()
    study_config = _load_study_config(study_dir)
    checkpoints = tuple(sorted(set(int(value) for value in config.checkpoints)))
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] != study_config.steps:
        raise ValueError("checkpoints must include 0 and the original final training step")
    configure_fp32(int(training_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu"
    )
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        study_config.test_size,
        int(training_seed),
        download=False,
    )
    train = loaded["train"].to(device)
    dataset_class = DATASETS[study_config.dataset][0]
    raw_train = dataset_class(
        root=str(study_config.data_root), train=True, download=False
    ).data
    used = torch.zeros(len(raw_train), dtype=torch.bool)
    used[loaded["train_indices"]] = True
    available = torch.arange(len(raw_train))[~used]
    reference_generator = torch.Generator().manual_seed(int(training_seed) + 51_001)
    reference_indices = available[
        torch.randperm(len(available), generator=reference_generator)[
            : int(config.endpoint_count)
        ]
    ]
    normalization = loaded["normalization"]
    endpoint_pixels = raw_train[reference_indices].float().unsqueeze(1) / 255.0
    endpoint_reference = (
        endpoint_pixels - float(normalization["mean"])
    ) / float(normalization["std"])
    endpoint_reference = endpoint_reference.to(device)
    analyzer, _ = build_direction_analyzer(
        train,
        basis_name,
        band_count=study_config.band_count,
        gamma=study_config.gamma,
        seed=int(training_seed),
    )
    analyzer = analyzer.to(device)
    baseline = TinyVelocityUNet(study_config.width, study_config.depth).to(device)
    weighted = copy.deepcopy(baseline)
    models = {"baseline": baseline, "weighted": weighted}
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=study_config.learning_rate, weight_decay=1e-4
        )
        for name, model in models.items()
    }
    canonical_basis = build_direction_analyzer(
        train,
        "dct",
        band_count=study_config.band_count,
        gamma=study_config.gamma,
        seed=int(training_seed),
    )[0].basis.to(device)
    canonical_groups = build_direction_analyzer(
        train,
        "dct",
        band_count=study_config.band_count,
        gamma=study_config.gamma,
        seed=int(training_seed),
    )[0].group_index.to(device)
    reference_energy = canonical_band_energies(
        endpoint_reference,
        canonical_basis,
        canonical_groups,
        study_config.band_count,
    )
    endpoint_generator = torch.Generator(device=device).manual_seed(
        int(config.endpoint_seed) + int(training_seed)
    )
    initial = torch.randn(
        endpoint_reference.shape, generator=endpoint_generator, device=device
    )
    training_generator = torch.Generator(device=device).manual_seed(
        int(training_seed) + 101
    )
    rows: list[dict[str, float | int | str]] = []

    def record(step: int) -> None:
        snapshots = endpoint_snapshot(
            models,
            initial,
            reference_energy,
            canonical_basis,
            canonical_groups,
            ode_steps=int(config.ode_steps),
        )
        for snapshot in snapshots:
            snapshot.update(
                {
                    "dataset": study_config.dataset,
                    "basis": basis_name,
                    "training_seed": int(training_seed),
                    "step": int(step),
                    "endpoint_count": int(config.endpoint_count),
                }
            )
            rows.append(snapshot)

    record(0)
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
                else analyzer(prediction, target, time)[0].mean()
            )
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
        if step in checkpoint_set:
            record(step)

    expected_state = torch.load(
        study_dir / f"{basis_name}_seed{training_seed}" / "state.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected_models = {}
    for name, state in expected_state["models"].items():
        expected = TinyVelocityUNet(study_config.width, study_config.depth)
        expected.load_state_dict(state)
        expected_models[name] = expected
    reproduced_hashes = {name: _state_hash(models, name) for name in models}
    expected_hashes = {name: _state_hash(expected_models, name) for name in expected_models}
    hash_match = {name: reproduced_hashes[name] == expected_hashes[name] for name in models}
    metadata = {
        "dataset": study_config.dataset,
        "basis": basis_name,
        "training_seed": int(training_seed),
        "reproduced_hashes": reproduced_hashes,
        "expected_hashes": expected_hashes,
        "hash_match": hash_match,
    }
    return pd.DataFrame(rows), metadata


def run_training_path_probe(
    config: TrainingPathConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], Path | None]:
    devices = config.devices or ("cpu",)
    tasks = [
        (config, basis, int(seed), devices[index % len(devices)])
        for index, (basis, seed) in enumerate(
            (basis, seed) for basis in config.bases for seed in config.training_seeds
        )
    ]
    if len(tasks) == 1:
        results = [_run_pair(*tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(tasks), len(devices)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_pair, *task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    metrics = pd.concat([result[0] for result in results], ignore_index=True)
    metadata = [result[1] for result in results]
    summary = summarize_training_path(metrics)
    if not all(all(item["hash_match"].values()) for item in metadata):
        mismatches = [item for item in metadata if not all(item["hash_match"].values())]
        raise RuntimeError(f"training replay did not reproduce saved checkpoints: {mismatches}")
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"path_{timestamp}"
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
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        summary.to_csv(result_dir / "summary.csv", index=False)
    return metrics, summary, metadata, result_dir


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--training-seeds", default="3,4")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--checkpoints", default="0,25,50,100,200,400,600,800,1000")
    parser.add_argument("--endpoint-count", type=int, default=128)
    parser.add_argument("--endpoint-seed", type=int, default=5101)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = TrainingPathConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or TrainingPathConfig.output_root,
        bases=_strings(args.bases),
        training_seeds=_integers(args.training_seeds),
        devices=_strings(args.devices) or ("cpu",),
        checkpoints=_integers(args.checkpoints),
        endpoint_count=args.endpoint_count,
        endpoint_seed=args.endpoint_seed,
        ode_steps=args.ode_steps,
        save=not args.no_save,
    )
    _, summary, _, result_dir = run_training_path_probe(config)
    columns = [
        "dataset",
        "basis",
        "training_seed",
        "step",
        "weighted_minus_baseline_endpoint_moment_loss",
        "baseline_band0_log_gap",
        "weighted_band0_log_gap",
    ]
    print(summary[columns].round(5).to_string(index=False))
    print(f"result_dir={result_dir}")


if __name__ == "__main__":
    main()

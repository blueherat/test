"""Matched-spectrum direction-basis study on MNIST and FashionMNIST.

DCT, train-only PCA, and random orthogonal bases receive exactly the same
time-dependent loss eigenvalues.  Only the orientation of the output-risk
metric changes, isolating basis/data/architecture alignment from weight
strength.
"""

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

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import FashionMNIST, MNIST


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    MNISTToyConfig,
    configure_fp32,
    estimate_band_second_moments,
    evaluate_rollouts,
    evaluate_teacher_path,
    resolve_device,
    train_feature_classifier,
    train_paired_velocity_fields,
)
from experiments.rae_spectral_direction_loss import (  # noqa: E402
    DCTDirectionLoss,
    bounded_coefficient_mean_one,
    dct_matrix,
    radial_band_index,
)


DATASETS = {
    "mnist": (MNIST, Path("/data/shared/mnist")),
    "fashion_mnist": (FashionMNIST, Path("/data/shared/fashion_mnist")),
}


@dataclass(frozen=True)
class BasisStudyConfig:
    dataset: str = "fashion_mnist"
    data_root: Path = Path("/data/shared/fashion_mnist")
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_basis_transport"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    train_size: int = 8192
    test_size: int = 1024
    sample_count: int = 1024
    batch_size: int = 128
    steps: int = 1000
    learning_rate: float = 2e-4
    width: int = 24
    depth: int = 2
    gamma: float = 0.5
    band_count: int = 8
    ode_steps: int = 50
    classifier_epochs: int = 3
    classifier_batch_size: int = 256
    eval_times: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    save: bool = True


class OrthogonalDirectionLoss(nn.Module):
    """Direction loss with an explicit orthogonal basis and component spectrum."""

    def __init__(
        self,
        basis: torch.Tensor,
        component_moments: torch.Tensor,
        group_index: torch.Tensor,
        *,
        gamma: float,
        damping: float = 1e-4,
        min_weight: float = 0.2,
        max_weight: float = 2.0,
    ) -> None:
        super().__init__()
        basis = torch.as_tensor(basis, dtype=torch.float32)
        component_moments = torch.as_tensor(component_moments, dtype=torch.float32)
        group_index = torch.as_tensor(group_index, dtype=torch.long)
        if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
            raise ValueError("basis must be a square matrix with directions in columns")
        if component_moments.shape != (basis.shape[1],) or torch.any(component_moments <= 0):
            raise ValueError("component moments must be positive and match the basis")
        if group_index.shape != component_moments.shape or torch.any(group_index < 0):
            raise ValueError("group index must match component moments")
        identity = torch.eye(basis.shape[0], dtype=basis.dtype)
        if not torch.allclose(basis.T @ basis, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("basis must be orthogonal")
        self.dimension = int(basis.shape[0])
        self.spatial_size = int(round(self.dimension**0.5))
        if self.spatial_size**2 != self.dimension:
            raise ValueError("the current image audit requires a square spatial basis")
        self.band_count = int(group_index.max().item()) + 1
        self.gamma = float(gamma)
        self.damping = float(damping)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("component_moments", component_moments, persistent=True)
        self.register_buffer("group_index", group_index, persistent=True)
        self.register_buffer(
            "band_counts",
            torch.bincount(group_index, minlength=self.band_count).float(),
            persistent=True,
        )

    def residual_variance(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        moments = self.component_moments.to(device=time.device, dtype=time.dtype)[None]
        denominator = (1.0 - time).square() * moments + time.square()
        return moments / denominator.clamp_min(1e-12)

    def weights(self, time: torch.Tensor) -> torch.Tensor:
        raw = (self.residual_variance(time) + self.damping).pow(-self.gamma)
        counts = torch.ones(self.dimension, device=time.device, dtype=time.dtype)
        return bounded_coefficient_mean_one(
            raw,
            counts,
            self.min_weight,
            self.max_weight,
        )

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (
            1,
            self.spatial_size,
            self.spatial_size,
        ):
            raise ValueError(
                f"expected [B,1,{self.spatial_size},{self.spatial_size}], got {tuple(value.shape)}"
            )
        return value.flatten(1) @ self.basis.to(device=value.device, dtype=value.dtype)

    def band_mse(self, error: torch.Tensor) -> torch.Tensor:
        coefficient_mse = self.transform(error).square()
        index = self.group_index.to(error.device)
        sums = torch.zeros(
            (len(error), self.band_count), device=error.device, dtype=error.dtype
        )
        sums.scatter_add_(1, index[None].expand(len(error), -1), coefficient_mse)
        return sums / self.band_counts.to(device=error.device, dtype=error.dtype)[None]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have equal shape")
        error = prediction - target
        coefficient_mse = self.transform(error).square()
        weights = self.weights(time).to(coefficient_mse.dtype)
        loss = torch.mean(coefficient_mse * weights, dim=1)
        return loss, {
            "raw_mse": error.square().flatten(1).mean(dim=1),
            "band_mse": self.band_mse(error),
            "component_weights": weights,
            "residual_variance": self.residual_variance(time).to(coefficient_mse.dtype),
        }


def _fixed_subset(length: int, count: int, seed: int) -> torch.Tensor:
    if not 1 <= int(count) <= int(length):
        raise ValueError(f"subset count must lie in [1, {length}]")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(length), generator=generator)[: int(count)]


def load_small_image_tensors(
    dataset: str,
    data_root: str | Path,
    train_size: int,
    test_size: int,
    seed: int,
    *,
    download: bool = True,
) -> dict[str, torch.Tensor | dict[str, float]]:
    name = dataset.strip().lower()
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    dataset_class = DATASETS[name][0]
    train_set = dataset_class(root=str(data_root), train=True, download=download)
    test_set = dataset_class(root=str(data_root), train=False, download=download)
    train_index = _fixed_subset(len(train_set), train_size, seed)
    test_index = _fixed_subset(len(test_set), test_size, seed + 1)
    train_images = train_set.data[train_index].float().unsqueeze(1) / 255.0
    test_images = test_set.data[test_index].float().unsqueeze(1) / 255.0
    train_labels = train_set.targets[train_index].long()
    test_labels = test_set.targets[test_index].long()
    mean = float(train_images.mean())
    std = float(train_images.std(unbiased=False).clamp_min(1e-6))
    return {
        "train": (train_images - mean) / std,
        "test": (test_images - mean) / std,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "normalization": {"mean": mean, "std": std},
        "train_indices": train_index,
        "test_indices": test_index,
    }


def dct_pixel_basis(size: int) -> torch.Tensor:
    matrix = dct_matrix(int(size))
    atoms = torch.einsum("ui,vj->uvij", matrix, matrix)
    return atoms.reshape(int(size) ** 2, int(size) ** 2).T.contiguous()


def _pca_basis(train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = train.flatten(1).float().cpu()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors.contiguous(), eigenvalues.clamp_min(0.0)


def _random_basis(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(50_003 + int(seed))
    matrix = torch.randn((int(dimension), int(dimension)), generator=generator)
    basis, triangular = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(triangular) >= 0, 1.0, -1.0)
    return basis * signs[None]


def build_direction_analyzer(
    train: torch.Tensor,
    basis_name: str,
    *,
    band_count: int,
    gamma: float,
    seed: int,
) -> tuple[OrthogonalDirectionLoss, dict[str, float | str]]:
    size = int(train.shape[-1])
    if train.shape[1:] != (1, size, size):
        raise ValueError("expected one-channel square images")
    band_moments = estimate_band_second_moments(train, int(band_count))
    dct_groups = radial_band_index(size, int(band_count)).flatten()
    component_moments = band_moments[dct_groups]
    name = basis_name.strip().lower()
    if name == "dct":
        basis = dct_pixel_basis(size)
        groups = dct_groups
    else:
        order = torch.argsort(component_moments, stable=True)
        component_moments = component_moments[order]
        groups = dct_groups[order]
        if name == "pca":
            basis, eigenvalues = _pca_basis(train)
            pca_rank = float(
                eigenvalues.sum().square()
                / eigenvalues.square().sum().clamp_min(1e-12)
            )
        elif name == "random":
            basis = _random_basis(size * size, int(seed))
            pca_rank = float("nan")
        else:
            raise ValueError(f"unknown basis: {basis_name}")
    analyzer = OrthogonalDirectionLoss(
        basis,
        component_moments,
        groups,
        gamma=float(gamma),
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    )
    probe_times = torch.tensor([0.1, 0.5, 0.9])
    weights = analyzer.weights(probe_times)
    metadata: dict[str, float | str] = {
        "basis": name,
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_condition_max": float((weights.max(dim=1).values / weights.min(dim=1).values).max()),
    }
    if name == "pca":
        metadata["pca_effective_rank"] = pca_rank
    return analyzer, metadata


def _state_hash(models: Mapping[str, nn.Module], variant: str) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(models[variant].state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _toy_config(config: BasisStudyConfig, *, seed: int, device: str, output_root: Path) -> MNISTToyConfig:
    return MNISTToyConfig(
        data_root=config.data_root,
        output_root=output_root,
        train_size=config.train_size,
        test_size=config.test_size,
        sample_count=config.sample_count,
        batch_size=config.batch_size,
        steps=config.steps,
        learning_rate=config.learning_rate,
        width=config.width,
        depth=config.depth,
        gamma=config.gamma,
        band_count=config.band_count,
        ode_steps=config.ode_steps,
        classifier_epochs=config.classifier_epochs,
        classifier_batch_size=config.classifier_batch_size,
        eval_times=config.eval_times,
        seed=int(seed),
        device=str(device),
        save=False,
    )


def _ratio_summary(teacher: pd.DataFrame, rollout: pd.DataFrame) -> dict[str, float]:
    pivot = teacher.pivot(index="time", columns="variant", values="velocity_mse")
    ratios = pivot["weighted"] / pivot["baseline"]
    result = {
        "teacher_ratio_all": float(ratios.mean()),
        "teacher_ratio_low_mid": float(ratios[ratios.index <= 0.5].mean()),
        "teacher_ratio_high": float(ratios[ratios.index >= 0.7].mean()),
    }
    values = rollout.set_index("variant")
    for metric in (
        "latent_swd",
        "decoded_pixel_swd",
        "feature_swd",
        "feature_fid",
    ):
        result[f"rollout_{metric}_ratio"] = float(
            values.loc["weighted", metric] / max(values.loc["baseline", metric], 1e-12)
        )
    return result


def run_basis_pair(
    config: BasisStudyConfig,
    *,
    basis_name: str,
    seed: int,
    device: str,
    study_dir: Path | None = None,
) -> dict[str, object]:
    configure_fp32(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    actual_device = resolve_device(device)
    loaded = load_small_image_tensors(
        config.dataset,
        config.data_root,
        config.train_size,
        config.test_size,
        int(seed),
        download=False,
    )
    train = loaded["train"].to(actual_device)
    test = loaded["test"].to(actual_device)
    train_labels = loaded["train_labels"].to(actual_device)
    test_labels = loaded["test_labels"].to(actual_device)
    analyzer, basis_metadata = build_direction_analyzer(
        train,
        basis_name,
        band_count=config.band_count,
        gamma=config.gamma,
        seed=int(seed),
    )
    analyzer = analyzer.to(actual_device)
    toy_config = _toy_config(
        config,
        seed=int(seed),
        device=str(actual_device),
        output_root=study_dir or config.output_root,
    )
    models, history = train_paired_velocity_fields(train, toy_config, analyzer)
    for model in models.values():
        model.eval()
    teacher, teacher_bands = evaluate_teacher_path(
        models,
        test,
        analyzer,
        config.eval_times,
        int(seed),
        config.batch_size,
    )
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=config.classifier_epochs,
        batch_size=config.classifier_batch_size,
        seed=int(seed),
    )
    rollout, rollout_bands, samples = evaluate_rollouts(
        models,
        test,
        classifier,
        analyzer,
        toy_config,
        loaded["normalization"],
    )
    summary: dict[str, object] = {
        "dataset": config.dataset,
        "basis": basis_name,
        "seed": int(seed),
        "classifier_accuracy": float(classifier_accuracy),
        "baseline_hash": _state_hash(models, "baseline"),
        **basis_metadata,
        **_ratio_summary(teacher, rollout),
    }

    run_dir = None
    if config.save:
        if study_dir is None:
            raise ValueError("study_dir is required when saving")
        run_dir = study_dir / f"{basis_name}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        history.to_csv(run_dir / "history.csv", index=False)
        teacher.to_csv(run_dir / "teacher_summary.csv", index=False)
        teacher_bands.to_csv(run_dir / "teacher_bands.csv", index=False)
        rollout.to_csv(run_dir / "rollout_summary.csv", index=False)
        rollout_bands.to_csv(run_dir / "rollout_bands.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "models": {name: model.state_dict() for name, model in models.items()},
                "samples": samples,
                "analyzer": analyzer.state_dict(),
                "normalization": loaded["normalization"],
                "train_indices": loaded["train_indices"],
                "test_indices": loaded["test_indices"],
            },
            run_dir / "state.pt",
        )
        summary["run_dir"] = str(run_dir)
    del train, test, train_labels, test_labels, classifier, analyzer, models
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _run_group(tasks: Sequence[tuple[BasisStudyConfig, str, int, str, Path | None]]) -> list[dict[str, object]]:
    rows = []
    for config, basis_name, seed, device, study_dir in tasks:
        row = run_basis_pair(
            config,
            basis_name=basis_name,
            seed=seed,
            device=device,
            study_dir=study_dir,
        )
        print(
            f"done {config.dataset} {basis_name} seed={seed}: "
            f"teacher={row['teacher_ratio_all']:.3f}, "
            f"feature_fid={row['rollout_feature_fid_ratio']:.3f}"
        )
        rows.append(row)
    return rows


def run_basis_study(config: BasisStudyConfig = BasisStudyConfig()) -> tuple[pd.DataFrame, Path | None]:
    dataset = config.dataset.strip().lower()
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {config.dataset}")
    if set(config.bases) != {"dct", "pca", "random"}:
        raise ValueError("the preregistered study requires dct, pca, and random bases")
    # Download once before worker processes start to avoid concurrent archive writes.
    dataset_class = DATASETS[dataset][0]
    dataset_class(root=str(config.data_root), train=True, download=True)
    dataset_class(root=str(config.data_root), train=False, download=True)

    study_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = (
            config.output_root.expanduser()
            / f"{dataset}_preregistered_v2_deterministic_{timestamp}"
        )
        study_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["data_root"] = str(config.data_root)
        serialized["output_root"] = str(config.output_root.expanduser())
        (study_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )

    devices = config.devices or ("cpu",)
    tasks = []
    for index, (basis_name, seed) in enumerate(
        (basis_name, seed) for basis_name in config.bases for seed in config.seeds
    ):
        tasks.append((config, basis_name, int(seed), devices[index % len(devices)], study_dir))
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)

    rows: list[dict[str, object]] = []
    if len(devices) == 1:
        rows = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in grouped if group]
            for future in as_completed(futures):
                rows.extend(future.result())
    summary = pd.DataFrame(rows).sort_values(["basis", "seed"]).reset_index(drop=True)
    if study_dir is not None:
        summary.to_csv(study_dir / "study_summary.csv", index=False)
    return summary, study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fashion_mnist")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    default_root = DATASETS[args.dataset][1]
    config = replace(
        BasisStudyConfig(),
        dataset=args.dataset,
        data_root=args.data_root or default_root,
        devices=devices or ("cpu",),
        save=not args.no_save,
    )
    if args.quick:
        config = replace(
            config,
            bases=("dct", "pca", "random"),
            seeds=(0,),
            devices=(devices[0],) if devices else ("cpu",),
            train_size=128,
            test_size=64,
            sample_count=64,
            batch_size=16,
            steps=4,
            width=8,
            depth=1,
            ode_steps=10,
            classifier_epochs=1,
            classifier_batch_size=32,
        )
    summary, result_dir = run_basis_study(config)
    print(f"result_dir={result_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

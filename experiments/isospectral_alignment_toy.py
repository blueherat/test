"""Show why equal loss spectra need not imply equal optimization dynamics.

For a linear residual ``r(theta) = A theta - b`` and output metric ``W``,

    L_W(theta) = 0.5 * r(theta)^T W r(theta)

has parameter Hessian ``A^T W A``.  Rotating ``W`` while preserving all of
its eigenvalues generally changes this effective Hessian because the output
Jacobian ``A`` is not isotropic.  This is the smallest exact analogue of the
DCT/PCA/random matched-spectrum experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.small_image_basis_transport import dct_pixel_basis


DEFAULT_OUTPUT_ROOT = (
    Path.home() / "data/eqvae/experiments/isospectral_alignment_toy"
)


@dataclass(frozen=True)
class IsospectralToyConfig:
    spatial_size: int = 8
    singular_condition: float = 10**1.5
    weight_condition: float = 10.0
    steps: int = 2_000
    step_fraction: float = 0.9
    seed: int = 0
    output_root: Path = DEFAULT_OUTPUT_ROOT
    save: bool = True


def _orthogonal_matrix(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(
        (int(dimension), int(dimension)),
        generator=generator,
        dtype=torch.float64,
    )
    basis, triangular = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(triangular) >= 0, 1.0, -1.0)
    return basis * signs[None]


def build_isospectral_problem(
    config: IsospectralToyConfig,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Construct aligned and random output metrics with identical spectra."""

    size = int(config.spatial_size)
    if size < 2:
        raise ValueError("spatial_size must be at least two")
    if config.singular_condition <= 1.0 or config.weight_condition <= 1.0:
        raise ValueError("condition numbers must exceed one")
    dimension = size * size
    # The repository DCT helper is fp32; re-orthogonalize in fp64 so the
    # matched-spectrum audit is limited by float64 roundoff, not basis storage.
    output_basis, _ = torch.linalg.qr(dct_pixel_basis(size).double())
    parameter_basis = _orthogonal_matrix(dimension, config.seed + 11)
    random_output_basis = _orthogonal_matrix(dimension, config.seed + 17)

    singular_values = torch.logspace(
        0.0,
        -math.log10(float(config.singular_condition)),
        dimension,
        dtype=torch.float64,
    )
    weight_eigenvalues = torch.logspace(
        math.log10(2.0),
        math.log10(2.0 / float(config.weight_condition)),
        dimension,
        dtype=torch.float64,
    )
    jacobian = (
        output_basis
        @ torch.diag(singular_values)
        @ parameter_basis.T
    )
    target = torch.randn(
        dimension,
        generator=torch.Generator().manual_seed(config.seed + 23),
        dtype=torch.float64,
    )
    observation = jacobian @ target
    metrics = {
        "aligned": output_basis
        @ torch.diag(weight_eigenvalues)
        @ output_basis.T,
        "random": random_output_basis
        @ torch.diag(weight_eigenvalues)
        @ random_output_basis.T,
    }
    return {
        "jacobian": jacobian,
        "target": target,
        "observation": observation,
        "weight_eigenvalues": weight_eigenvalues,
        "metrics": metrics,
    }


def effective_hessian(jacobian: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return jacobian.T @ metric @ jacobian


def spectral_condition(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("matrix must be positive definite")
    return float(eigenvalues[-1] / eigenvalues[0])


def run_gradient_descent(
    jacobian: torch.Tensor,
    observation: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, torch.Tensor]:
    theta = torch.zeros_like(target)
    rows: list[dict[str, float | int]] = []
    checkpoints = {0, int(steps)}
    checkpoints.update(
        int(round(value))
        for value in torch.linspace(0, int(steps), 21).tolist()
    )
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-15)
    for step in range(int(steps) + 1):
        if step in checkpoints:
            residual = jacobian @ theta - observation
            rows.append(
                {
                    "step": step,
                    "weighted_loss": float(0.5 * residual @ metric @ residual),
                    "relative_parameter_error": float(
                        torch.linalg.vector_norm(theta - target) / target_norm
                    ),
                }
            )
        if step == int(steps):
            break
        residual = jacobian @ theta - observation
        theta -= float(learning_rate) * (jacobian.T @ metric @ residual)
    return pd.DataFrame(rows), theta


def joint_rotation_error(
    jacobian: torch.Tensor,
    observation: torch.Tensor,
    metric: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    """Rotate data/Jacobian and metric together; the problem must not change."""

    rotation = _orthogonal_matrix(jacobian.shape[0], seed)
    rotated_jacobian = rotation @ jacobian
    rotated_observation = rotation @ observation
    rotated_metric = rotation @ metric @ rotation.T
    original_hessian = effective_hessian(jacobian, metric)
    rotated_hessian = effective_hessian(rotated_jacobian, rotated_metric)
    return {
        "hessian_max_abs_error": float(
            (original_hessian - rotated_hessian).abs().max()
        ),
        "observation_norm_error": float(
            abs(
                torch.linalg.vector_norm(observation)
                - torch.linalg.vector_norm(rotated_observation)
            )
        ),
    }


def run_study(
    config: IsospectralToyConfig = IsospectralToyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], Path | None]:
    problem = build_isospectral_problem(config)
    jacobian = problem["jacobian"]
    target = problem["target"]
    observation = problem["observation"]
    metrics = problem["metrics"]
    assert isinstance(jacobian, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(observation, torch.Tensor)
    assert isinstance(metrics, dict)

    hessians = {
        name: effective_hessian(jacobian, metric)
        for name, metric in metrics.items()
    }
    largest_curvature = max(
        float(torch.linalg.eigvalsh(hessian)[-1])
        for hessian in hessians.values()
    )
    learning_rate = float(config.step_fraction) / largest_curvature

    spectrum_reference = torch.linalg.eigvalsh(metrics["aligned"])
    spectrum_error = float(
        (
            spectrum_reference
            - torch.linalg.eigvalsh(metrics["random"])
        )
        .abs()
        .max()
    )
    summary_rows = []
    histories = []
    for name, metric in metrics.items():
        history, _ = run_gradient_descent(
            jacobian,
            observation,
            target,
            metric,
            steps=config.steps,
            learning_rate=learning_rate,
        )
        history.insert(0, "basis", name)
        histories.append(history)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessians[name])
        summary_rows.append(
            {
                "basis": name,
                "weight_condition": spectral_condition(metric),
                "effective_hessian_condition": spectral_condition(hessians[name]),
                "effective_curvature_min": float(hessian_eigenvalues[0]),
                "effective_curvature_max": float(hessian_eigenvalues[-1]),
                "final_weighted_loss": float(history.iloc[-1]["weighted_loss"]),
                "final_relative_parameter_error": float(
                    history.iloc[-1]["relative_parameter_error"]
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    history = pd.concat(histories, ignore_index=True)
    audit = {
        "weight_spectrum_max_abs_error": spectrum_error,
        "learning_rate": learning_rate,
        **joint_rotation_error(
            jacobian,
            observation,
            metrics["aligned"],
            seed=config.seed + 29,
        ),
    }

    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"seed{config.seed}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        summary.to_csv(result_dir / "summary.csv", index=False)
        history.to_csv(result_dir / "history.csv", index=False)
        (result_dir / "audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    return summary, history, audit, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = IsospectralToyConfig(
        seed=args.seed,
        steps=args.steps,
        save=not args.no_save,
    )
    summary, _, audit, result_dir = run_study(config)
    print(f"result_dir={result_dir}")
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

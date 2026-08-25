#!/usr/bin/env python3
"""Track whether maximizing AdvFD keeps improving its generator correction.

AdvFD trains its inner representation to expose a larger feature Frechet
discrepancy.  The outer generator, however, needs a field that improves the
data distribution.  On analytic toys this script measures both quantities
along the exact same critic optimization trajectory, including reverse-KL
descent per unit particle displacement and finite-step invertibility.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    learned_pullback_field,
)
from experiments.frechet_residual_score_toy import (
    field_diagnostics,
    finite_pushforward_kl,
)
from experiments.run_advfd_monoflow_audit import (
    build_regime,
    monoflow_diagnostics,
)
from experiments.run_advfd_quotient_gradient_audit import (
    UPDATE_MODES,
    UpdateMode,
    critic_feature_statistics,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    parse_floats,
    parse_ints,
    parse_strings,
    population_advfd,
)


TRAJECTORY_MODES = {
    **UPDATE_MODES,
    "real_mean_stopgrad": UpdateMode("official_mean_only", True, True),
    "real_mean_quotient": UpdateMode("official_mean_only", False, False),
    "real_covariance_stopgrad": UpdateMode(
        "official_covariance_only", True, True
    ),
    "real_covariance_quotient": UpdateMode(
        "official_covariance_only", False, False
    ),
    "pooled_mean_stopgrad": UpdateMode("pooled_mean_only", True, True),
    "pooled_mean_quotient": UpdateMode("pooled_mean_only", False, False),
    "pooled_covariance_stopgrad": UpdateMode(
        "pooled_covariance_only", True, True
    ),
    "pooled_covariance_quotient": UpdateMode(
        "pooled_covariance_only", False, False
    ),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_critic(
    critic,
    target,
    source,
    *,
    config: CriticConfig,
    mode_name: str,
    step: int,
    quadrature_order: int,
    sample_count: int,
    sample_seed: int,
    displacement_rms_values: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    with torch.no_grad():
        objective, _ = population_advfd(
            critic,
            target,
            source,
            order=config.quadrature_order,
            whitening_epsilon=config.whitening_epsilon,
            detach_real=config.detach_real,
            objective_mode=config.objective_mode,
            detach_calibration=config.detach_calibration,
        )
    context = build_feature_force_context(
        critic,
        target,
        source,
        order=quadrature_order,
        whitening_epsilon=config.whitening_epsilon,
        objective_mode=config.objective_mode,
    )
    field = learned_pullback_field(critic, context, mode="transpose")
    diagnostics = field_diagnostics(
        target, source, field, quadrature_order=quadrature_order
    )
    diagnostics.update(
        monoflow_diagnostics(
            critic,
            context,
            target,
            source,
            sample_count=sample_count,
            sample_seed=sample_seed,
        )
    )
    diagnostics.update(
        critic_feature_statistics(
            critic, target, source, order=quadrature_order
        )
    )
    velocity_rms = diagnostics["velocity_rms"]
    diagnostics["kl_descent_per_velocity_rms"] = (
        -diagnostics["reverse_kl_derivative"] / max(velocity_rms, 1e-30)
    )
    rows: list[dict[str, float | int | str]] = []
    for displacement_rms in displacement_rms_values:
        if velocity_rms > 1e-14:
            finite = finite_pushforward_kl(
                target,
                source,
                field,
                step_size=displacement_rms / velocity_rms,
                quadrature_order=quadrature_order,
            )
        else:
            finite = {
                "kl_before": float("nan"),
                "kl_after": float("nan"),
                "kl_change": float("nan"),
                "positive_jacobian_fraction": float("nan"),
                "minimum_jacobian_determinant": float("nan"),
            }
        rows.append(
            {
                "mode": mode_name,
                "step": step,
                "advfd": float(objective),
                "displacement_rms": displacement_rms,
                **diagnostics,
                **finite,
            }
        )
    return rows


def train_trajectory(
    target,
    source,
    *,
    mode_name: str,
    seed: int,
    checkpoints: tuple[int, ...],
    quadrature_order: int,
    sample_count: int,
    displacement_rms_values: tuple[float, ...],
    whitening_epsilon: float,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    mode = TRAJECTORY_MODES[mode_name]
    config = CriticConfig(
        steps=max(checkpoints),
        whitening_epsilon=whitening_epsilon,
        quadrature_order=min(quadrature_order, 16),
        objective_mode=mode.objective_mode,
        detach_real=mode.detach_real,
        detach_calibration=mode.detach_calibration,
    )
    seed_everything(seed)
    critic = FeatureCritic(config).to(device=device, dtype=target.means.dtype)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_set = set(checkpoints)
    rows: list[dict[str, float | int | str]] = []
    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            rows.extend(
                evaluate_critic(
                    critic,
                    target,
                    source,
                    config=config,
                    mode_name=mode_name,
                    step=step,
                    quadrature_order=quadrature_order,
                    sample_count=sample_count,
                    sample_seed=seed + 100_003 * step,
                    displacement_rms_values=displacement_rms_values,
                )
            )
        if step == max(checkpoints):
            break
        optimizer.zero_grad(set_to_none=True)
        objective, _ = population_advfd(
            critic,
            target,
            source,
            order=config.quadrature_order,
            whitening_epsilon=config.whitening_epsilon,
            detach_real=config.detach_real,
            objective_mode=config.objective_mode,
            detach_calibration=config.detach_calibration,
        )
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
        optimizer.step()
    return rows


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigma: float,
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    quadrature_order: int,
    sample_count: int,
    displacement_rms_values: tuple[float, ...],
    whitening_epsilon: float,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("checkpoints must begin at zero")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be sorted and unique")
    unknown = set(modes) - set(TRAJECTORY_MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    output_root.mkdir(parents=True)

    rows: list[dict[str, float | int | str]] = []
    for regime_name in regimes:
        target_clean, source_clean = build_regime(regime_name, device=device)
        target = target_clean.convolve_isotropic(noise_sigma)
        source = source_clean.convolve_isotropic(noise_sigma)
        for mode_name in modes:
            for seed in seeds:
                print(
                    f"regime={regime_name} mode={mode_name} seed={seed}",
                    flush=True,
                )
                trajectory = train_trajectory(
                    target,
                    source,
                    mode_name=mode_name,
                    seed=seed,
                    checkpoints=checkpoints,
                    quadrature_order=quadrature_order,
                    sample_count=sample_count,
                    displacement_rms_values=displacement_rms_values,
                    whitening_epsilon=whitening_epsilon,
                    device=device,
                )
                for row in trajectory:
                    rows.append(
                        {
                            "regime": regime_name,
                            "noise_sigma": noise_sigma,
                            "seed": seed,
                            **row,
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "inner_trajectory.csv", index=False)
    aggregate = frame.groupby(
        ["regime", "mode", "step", "displacement_rms"], as_index=False
    ).agg(
        advfd=("advfd", "mean"),
        kl_descent_per_velocity_rms=("kl_descent_per_velocity_rms", "mean"),
        score_cosine=("score_cosine", "mean"),
        kl_change=("kl_change", "mean"),
        positive_jacobian_fraction=("positive_jacobian_fraction", "mean"),
        velocity_effective_fraction=("velocity_effective_fraction", "mean"),
        feature_rms=("source_feature_rms", "mean"),
        feature_effective_rank=("pooled_feature_effective_rank", "mean"),
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)

    smallest_displacement = min(displacement_rms_values)
    selected = aggregate[aggregate["displacement_rms"] == smallest_displacement]
    figure, axes = plt.subplots(len(regimes), 3, figsize=(13.5, 4.0 * len(regimes)))
    if len(regimes) == 1:
        axes = axes[None, :]
    for row_index, regime_name in enumerate(regimes):
        regime_frame = selected[selected["regime"] == regime_name]
        for mode_name in modes:
            mode_frame = regime_frame[regime_frame["mode"] == mode_name]
            axes[row_index, 0].plot(
                mode_frame["step"], mode_frame["advfd"], marker="o", label=mode_name
            )
            axes[row_index, 1].plot(
                mode_frame["step"],
                mode_frame["kl_descent_per_velocity_rms"],
                marker="o",
                label=mode_name,
            )
            axes[row_index, 2].plot(
                mode_frame["step"],
                mode_frame["velocity_effective_fraction"],
                marker="o",
                label=mode_name,
            )
        axes[row_index, 0].set_title(f"{regime_name}: critic objective")
        axes[row_index, 1].set_title(f"{regime_name}: KL descent / field RMS")
        axes[row_index, 2].set_title(f"{regime_name}: field effective fraction")
        for axis in axes[row_index]:
            axis.set_xlabel("critic steps")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / "inner_trajectory.png", dpi=180)
    plt.close(figure)

    peak_rows = []
    for (regime_name, mode_name), group in selected.groupby(["regime", "mode"]):
        best = group.loc[group["kl_descent_per_velocity_rms"].idxmax()]
        final = group.loc[group["step"].idxmax()]
        peak_rows.append(
            {
                "regime": regime_name,
                "mode": mode_name,
                "peak_correctability_step": int(best["step"]),
                "peak_correctability": float(best["kl_descent_per_velocity_rms"]),
                "final_correctability": float(final["kl_descent_per_velocity_rms"]),
                "peak_advfd": float(best["advfd"]),
                "final_advfd": float(final["advfd"]),
            }
        )
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(output_root / "peak_correctability.csv", index=False)
    summary = {
        "protocol": "advfd_inner_trajectory_audit_v1",
        "regimes": list(regimes),
        "noise_sigma": noise_sigma,
        "modes": list(modes),
        "seeds": list(seeds),
        "checkpoints": list(checkpoints),
        "displacement_rms_values": list(displacement_rms_values),
        "all_modes_peak_before_final": bool(
            (peaks["peak_correctability_step"] < max(checkpoints)).all()
        ),
        "all_advfd_non_decreasing_on_average": bool(
            all(
                (group.sort_values("step")["advfd"].diff().dropna() >= -1e-10).all()
                for _, group in selected.groupby(["regime", "mode"])
            )
        ),
        "all_recorded_objectives_finite": bool(
            frame["advfd"].map(lambda value: math.isfinite(float(value))).all()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--regimes", type=parse_strings, default=("rotated_ring", "shape_only")
    )
    parser.add_argument("--noise-sigma", type=float, default=0.4)
    parser.add_argument("--modes", type=parse_strings, default=tuple(UPDATE_MODES))
    parser.add_argument(
        "--seeds", type=parse_ints, default=(20260824, 20260825, 20260826)
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_ints,
        default=(0, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000),
    )
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument(
        "--displacement-rms-values", type=parse_floats, default=(0.001, 0.003, 0.01)
    )
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigma=args.noise_sigma,
        modes=args.modes,
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        quadrature_order=args.quadrature_order,
        sample_count=args.sample_count,
        displacement_rms_values=args.displacement_rms_values,
        whitening_epsilon=args.whitening_epsilon,
        device=torch.device(args.device),
    )

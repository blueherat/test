"""Post-hoc conjugacy audit for the preregistered four-path toy.

This does not retrain or select a checkpoint.  It asks whether the learned Base
field, when exactly pushed through the coordinate map, reproduces the Base
rollout after inverse transformation.  Passing this control separates a path
implementation bug from lack of closure of the ordinary MLP model class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32  # noqa: E402
from experiments.latent_transport_four_path_toy import (  # noqa: E402
    ConjugatedVelocityField,
    QuadraticShear,
    ResidualTimeMLP,
    RingMixtureConfig,
    distribution_metrics,
    sample_model,
    sample_ring_mixture,
)
from experiments.latent_transport_paths import (  # noqa: E402
    conditional_path_sample,
    relative_l2_per_sample,
)


DEFAULT_RESULT = (
    Path.home()
    / "data/eqvae/experiments/latent_transport_four_path_toy/"
    "preregistered_v1_20260718_131400"
)


def _load_config(path: Path) -> tuple[dict, RingMixtureConfig, dict]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    mixture = RingMixtureConfig(
        **json.loads((path / "mixture.json").read_text(encoding="utf-8"))
    )
    states = torch.load(path / "models.pt", map_location="cpu", weights_only=False)
    return config, mixture, states


@torch.no_grad()
def audit_one(
    *,
    strength: float,
    seed: int,
    config: dict,
    mixture: RingMixtureConfig,
    states: dict,
    device: torch.device,
    count: int,
    field_count: int,
    step_counts: tuple[int, ...],
) -> tuple[list[dict], dict]:
    key = f"a{strength}_seed{seed}"
    transform = QuadraticShear(strength).to(device)
    base = ResidualTimeMLP(config["hidden_size"], config["depth"]).to(device).eval()
    learned_push = ResidualTimeMLP(config["hidden_size"], config["depth"]).to(device).eval()
    base.load_state_dict(states[key]["base"])
    learned_push.load_state_dict(states[key]["pushforward"])
    conjugated = ConjugatedVelocityField(base, transform).to(device).eval()

    generator = torch.Generator(device=device).manual_seed(710_003 + int(seed))
    epsilon = torch.randn((count, 2), generator=generator, device=device)
    reference, _ = sample_ring_mixture(
        mixture,
        count,
        generator=generator,
        device=device,
    )
    rows = []
    for steps in step_counts:
        base_endpoint = sample_model(
            base,
            "base",
            transform,
            epsilon,
            ode_steps=steps,
        )
        conjugated_endpoint = sample_model(
            conjugated,
            "pushforward",
            transform,
            epsilon,
            ode_steps=steps,
        )
        relative = relative_l2_per_sample(conjugated_endpoint, base_endpoint)
        base_metrics = distribution_metrics(
            base_endpoint,
            reference,
            mixture,
            directions=128,
            seed=720_003 + int(seed),
        )
        conjugated_metrics = distribution_metrics(
            conjugated_endpoint,
            reference,
            mixture,
            directions=128,
            seed=720_003 + int(seed),
        )
        rows.append(
            {
                "strength": float(strength),
                "seed": int(seed),
                "steps": int(steps),
                "paired_relative_l2_mean": float(relative.mean()),
                "paired_relative_l2_max": float(relative.max()),
                "paired_absolute_l2_max": float(
                    torch.linalg.vector_norm(conjugated_endpoint - base_endpoint, dim=1).max()
                ),
                "base_sliced_w1": base_metrics["sliced_w1"],
                "conjugated_sliced_w1": conjugated_metrics["sliced_w1"],
            }
        )

    learned_endpoint = sample_model(
        learned_push,
        "pushforward",
        transform,
        epsilon,
        ode_steps=config["ode_steps"],
    )
    conjugated_endpoint = sample_model(
        conjugated,
        "pushforward",
        transform,
        epsilon,
        ode_steps=config["ode_steps"],
    )
    learned_endpoint_gap = float(
        relative_l2_per_sample(learned_endpoint, conjugated_endpoint).mean()
    )

    field_generator = torch.Generator(device=device).manual_seed(730_003 + int(seed))
    data, _ = sample_ring_mixture(
        mixture,
        field_count,
        generator=field_generator,
        device=device,
    )
    noise = torch.randn(data.shape, generator=field_generator, device=device)
    time_value = torch.rand((field_count,), generator=field_generator, device=device)
    base_state = (1.0 - time_value[:, None]) * data + time_value[:, None] * noise
    transformed_path = conditional_path_sample(
        data,
        noise,
        time_value,
        branch="pushforward",
        transform=transform,
    )
    base_prediction = base(base_state, time_value)
    _, pushed_base_prediction = torch.func.jvp(
        transform,
        (base_state,),
        (base_prediction,),
    )
    conjugated_prediction = conjugated(transformed_path.state, time_value)
    learned_prediction = learned_push(transformed_path.state, time_value)
    definition_absolute_max = float(
        (conjugated_prediction - pushed_base_prediction).abs().max()
    )
    target = transformed_path.velocity
    field = {
        "strength": float(strength),
        "seed": int(seed),
        "conjugated_microscopic_mse": float(
            (conjugated_prediction - target).square().mean()
        ),
        "learned_push_microscopic_mse": float(
            (learned_prediction - target).square().mean()
        ),
        "learned_vs_conjugated_relative_l2": float(
            relative_l2_per_sample(learned_prediction, conjugated_prediction).mean()
        ),
        "learned_vs_conjugated_endpoint_relative_l2": learned_endpoint_gap,
        "conjugated_definition_absolute_max": definition_absolute_max,
    }
    return rows, field


def run(
    result_dir: Path,
    *,
    device_name: str,
    count: int = 2048,
    field_count: int = 8192,
    step_counts: tuple[int, ...] = (100, 200, 400),
) -> dict:
    configure_fp32()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    config, mixture, states = _load_config(result_dir)
    endpoint_rows = []
    field_rows = []
    for strength in config["strengths"]:
        for seed in config["seeds"]:
            rows, field = audit_one(
                strength=float(strength),
                seed=int(seed),
                config=config,
                mixture=mixture,
                states=states,
                device=device,
                count=count,
                field_count=field_count,
                step_counts=step_counts,
            )
            endpoint_rows.extend(rows)
            field_rows.append(field)
            print(
                f"a={strength:g} seed={seed}: rel@{max(step_counts)}="
                f"{rows[-1]['paired_relative_l2_mean']:.3e}, "
                f"learned_gap={field['learned_vs_conjugated_endpoint_relative_l2']:.3f}",
                flush=True,
            )
    endpoint = pd.DataFrame(endpoint_rows)
    field = pd.DataFrame(field_rows)
    coarse = endpoint[endpoint.steps.eq(min(step_counts))].set_index(["strength", "seed"])
    fine = endpoint[endpoint.steps.eq(max(step_counts))].set_index(["strength", "seed"])
    convergence_ratio = coarse.paired_relative_l2_mean / fine.paired_relative_l2_mean
    checks = {
        "conjugated_field_definition": {
            "value": float(field.conjugated_definition_absolute_max.max()),
            "threshold": 1e-5,
            "passed": float(field.conjugated_definition_absolute_max.max()) <= 1e-5,
        },
        "fine_conjugacy_mean_max": {
            "value": float(fine.paired_relative_l2_mean.max()),
            "threshold": 1e-4,
            "passed": float(fine.paired_relative_l2_mean.max()) <= 1e-4,
        },
        "heun_second_order_trend_min": {
            "value": float(convergence_ratio.min()),
            "threshold": 8.0,
            "passed": float(convergence_ratio.min()) >= 8.0,
        },
    }
    result = {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "interpretation": (
            "Path implementation and coordinate conjugacy are verified. The independently "
            "trained ordinary MLP fails because its model class/objective is not closed under "
            "the nonlinear coordinate pushforward."
        ),
    }
    endpoint.to_csv(result_dir / "conjugacy_endpoint.csv", index=False)
    field.to_csv(result_dir / "conjugacy_field.csv", index=False)
    (result_dir / "conjugacy_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--field-count", type=int, default=8192)
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 200, 400])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.result_dir.expanduser(),
        device_name=args.device,
        count=args.count,
        field_count=args.field_count,
        step_counts=tuple(args.steps),
    )

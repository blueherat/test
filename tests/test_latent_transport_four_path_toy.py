from dataclasses import replace

import pandas as pd
import torch

from experiments.latent_transport_four_path_toy import (
    BRANCHES,
    ConjugatedVelocityField,
    FourPathToyConfig,
    QuadraticShear,
    RingMixtureConfig,
    acceptance_report,
    distribution_metrics,
    train_one,
)
from experiments.latent_transport_paths import conditional_path_sample, jvp_relative_error


def test_quadratic_shear_has_exact_inverse_and_jvp():
    generator = torch.Generator().manual_seed(0)
    value = torch.randn((128, 2), generator=generator, dtype=torch.float64)
    direction = torch.randn((128, 2), generator=generator, dtype=torch.float64)
    transform = QuadraticShear(1.25)
    torch.testing.assert_close(transform.inverse(transform(value)), value)
    error = jvp_relative_error(transform, value, direction, step=1e-5)
    assert float(error.max()) < 1e-8


def test_conjugated_field_matches_analytic_quadratic_pushforward():
    class BaseField(torch.nn.Module):
        def forward(self, state, time_value):
            del time_value
            return torch.stack((2.0 * state[:, 0], -state[:, 1]), dim=1)

    transform = QuadraticShear(0.7)
    field = ConjugatedVelocityField(BaseField(), transform)
    generator = torch.Generator().manual_seed(21)
    base_state = torch.randn((64, 2), generator=generator, dtype=torch.float64)
    transformed_state = transform(base_state)
    time_value = torch.rand((64,), generator=generator, dtype=torch.float64)
    base_velocity = BaseField()(base_state, time_value)
    expected = torch.stack(
        (
            base_velocity[:, 0],
            base_velocity[:, 1]
            + 2.0 * transform.strength * base_state[:, 0] * base_velocity[:, 0],
        ),
        dim=1,
    )
    torch.testing.assert_close(field(transformed_state, time_value), expected)


def test_zero_strength_makes_all_four_conditional_paths_identical():
    generator = torch.Generator().manual_seed(1)
    data = torch.randn((64, 2), generator=generator, dtype=torch.float64)
    noise = torch.randn((64, 2), generator=generator, dtype=torch.float64)
    time = torch.linspace(0.05, 0.95, len(data), dtype=torch.float64)
    transform = QuadraticShear(0.0)
    paths = {
        branch: conditional_path_sample(
            data,
            noise,
            time,
            branch=branch,
            transform=transform,
        )
        for branch in BRANCHES
    }
    for branch in BRANCHES[1:]:
        torch.testing.assert_close(paths[branch].state, paths["base"].state)
        torch.testing.assert_close(paths[branch].velocity, paths["base"].velocity)


def test_distribution_metrics_distinguish_reference_from_shift():
    generator = torch.Generator().manual_seed(2)
    reference = torch.randn((512, 2), generator=generator)
    same = distribution_metrics(
        reference,
        reference.clone(),
        RingMixtureConfig(),
        directions=32,
        seed=3,
    )
    shifted = distribution_metrics(
        reference + 2.0,
        reference,
        RingMixtureConfig(),
        directions=32,
        seed=3,
    )
    assert same["sliced_w1"] == 0.0
    assert same["coordinate_w1"] == 0.0
    assert shifted["sliced_w1"] > 0.5
    assert shifted["mean_l2"] > 2.0


def test_acceptance_requires_valid_base_gap_and_pushforward_recovery():
    config = FourPathToyConfig(strengths=(1.0,), primary_strength=1.0)
    endpoint_rows = []
    pair_rows = []
    for seed in config.seeds:
        for branch, value in (
            ("base", 0.10),
            ("gaussian_straight", 0.20),
            ("matched_chord", 0.16),
            ("pushforward", 0.105),
        ):
            endpoint_rows.append(
                {
                    "strength": 1.0,
                    "seed": seed,
                    "branch": branch,
                    "sliced_w1": value,
                    "mode_coverage": 8,
                }
            )
        pair_rows.append(
            {
                "seed": seed,
                "has_coordinate_gap": True,
                "pushforward_recovers_half": True,
            }
        )
    solver = pd.DataFrame({"endpoint_relative_l2": [0.001]})
    runs = pd.DataFrame(
        {
            "initial_max_parameter_gap": [0.0],
            "cycle_relative_l2_max": [0.0],
            "jvp_finite_difference_relative_l2_max": [1e-5],
        }
    )
    report = acceptance_report(
        pd.DataFrame(endpoint_rows),
        solver,
        runs,
        pd.DataFrame(pair_rows),
        config,
    )
    assert report["method_passed"] is True
    assert report["decision"] == "phase3a_pass_small_image_authorized"


def test_quick_train_one_returns_all_branches_on_cpu():
    config = replace(
        FourPathToyConfig(),
        strengths=(1.0,),
        seeds=(0,),
        devices=("cpu",),
        hidden_size=8,
        depth=1,
        batch_size=16,
        steps=2,
        eval_every=1,
        eval_count=32,
        sample_count=32,
        sliced_directions=8,
        ode_steps=2,
        solver_steps=4,
        solver_count=16,
        save=False,
    )
    result = train_one(
        RingMixtureConfig(),
        config,
        strength=0.0,
        seed=0,
        device_name="cpu",
    )
    assert {row["branch"] for row in result["endpoint"]} == set(BRANCHES)
    assert {row["branch"] for row in result["teacher"]} == set(BRANCHES)
    assert {row["branch"] for row in result["solver"]} == set(BRANCHES)
    assert all(torch.isfinite(value).all() for state in result["states"].values() for value in state.values())
    base_state = result["states"]["base"]
    for branch in BRANCHES[1:]:
        for name, value in base_state.items():
            torch.testing.assert_close(result["states"][branch][name], value)
    endpoint_sliced = {row["branch"]: row["sliced_w1"] for row in result["endpoint"]}
    assert len(set(endpoint_sliced.values())) == 1

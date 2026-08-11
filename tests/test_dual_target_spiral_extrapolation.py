from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import evaluate_dual_target_spiral_extrapolation as sweep
import run_dual_target_closed_loop_toy as core


def build_tiny_suite() -> tuple[core.SpiralGaussianMixture, core.ModelSuite]:
    device = torch.device("cpu")
    distribution = core.SpiralGaussianMixture(
        2, components=8, component_std=0.08, seed=3, device=device
    )
    suite = core.build_model_suite(
        ambient_dim=2,
        hidden_dim=16,
        depth=2,
        time_dim=8,
        mode_dim=4,
        model_ids=("D0_xeps", "D4_safe"),
        lr=1e-3,
        weight_decay=0.0,
        seed=5,
        device=device,
    )
    suite.models["D0_xeps"].eval()
    return distribution, suite


def test_scale_endpoints_equal_existing_d0_branches() -> None:
    distribution, suite = build_tiny_suite()
    state = torch.randn(17, 2)
    time_value = torch.linspace(0.0, 1.0, len(state))
    for scale, condition in ((0.0, "D0_eps_shared"), (1.0, "D0_x_shared")):
        actual = sweep.scaled_dual_velocity(
            suite=suite,
            state=state,
            time_value=time_value,
            scale_value=scale,
            denominator_floor=1e-3,
            endpoint_mode="raw",
        )
        expected, _ = core.condition_field(
            condition,
            suite=suite,
            distribution=distribution,
            state=state,
            time_value=time_value,
            denominator_floor=1e-3,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_batched_scale_sampler_matches_individual_heun() -> None:
    distribution, suite = build_tiny_suite()
    generator = torch.Generator().manual_seed(11)
    initial_noise = torch.randn(13, 2, generator=generator)
    batched = sweep.sample_scale_sweep_heun(
        suite=suite,
        initial_noise=initial_noise,
        scales=(0.0, 1.0),
        steps=8,
        denominator_floor=1e-3,
        endpoint_mode="raw",
    )
    for scale, condition in ((0.0, "D0_eps_shared"), (1.0, "D0_x_shared")):
        expected, _ = core.sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=8,
            denominator_floor=1e-3,
            snapshot_times=(),
        )
        torch.testing.assert_close(batched[scale], expected, rtol=0.0, atol=0.0)


def test_endpoint_override_is_a_separate_protocol() -> None:
    _, suite = build_tiny_suite()
    state = torch.randn(2, 2)
    time_value = torch.tensor([0.0, 1.0])
    output = suite.models["D0_xeps"](state, time_value)
    velocity_x, velocity_epsilon = core.endpoint_velocities(
        state=state,
        time_value=time_value,
        clean_prediction=output["x"],
        epsilon_prediction=output["eps"],
        denominator_floor=1e-3,
    )
    actual = sweep.scaled_dual_velocity(
        suite=suite,
        state=state,
        time_value=time_value,
        scale_value=1.5,
        denominator_floor=1e-3,
        endpoint_mode="override",
    )
    torch.testing.assert_close(actual[0], velocity_x[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], velocity_epsilon[1], rtol=0.0, atol=0.0)


def test_d4_static_anchors_equal_own_branch_fields() -> None:
    distribution, suite = build_tiny_suite()
    state = torch.randn(17, 2)
    time_value = torch.linspace(0.0, 1.0, len(state))
    for scale, condition in ((0.0, "D4_eps_own"), (1.0, "D4_x_own")):
        actual = sweep.scaled_dual_velocity(
            suite=suite,
            state=state,
            time_value=time_value,
            scale_value=scale,
            denominator_floor=1e-3,
            endpoint_mode="override",
            head_source="d4",
        )
        expected, _ = core.condition_field(
            condition,
            suite=suite,
            distribution=distribution,
            state=state,
            time_value=time_value,
            denominator_floor=1e-3,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_d4_static_anchor_rollouts_equal_own_branches() -> None:
    distribution, suite = build_tiny_suite()
    generator = torch.Generator().manual_seed(17)
    initial_noise = torch.randn(13, 2, generator=generator)
    generated = sweep.sample_scale_sweep_heun(
        suite=suite,
        initial_noise=initial_noise,
        scales=(0.0, 1.0),
        steps=8,
        denominator_floor=1e-3,
        endpoint_mode="override",
        head_source="d4",
    )
    for scale, condition in ((0.0, "D4_eps_own"), (1.0, "D4_x_own")):
        expected, _ = core.sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=8,
            denominator_floor=1e-3,
            snapshot_times=(),
        )
        torch.testing.assert_close(generated[scale], expected, rtol=0.0, atol=0.0)


def test_scale_tag_is_stable() -> None:
    assert sweep.scale_tag(1.0) == "scale_1"
    assert sweep.scale_tag(1.78) == "scale_1p78"
    assert sweep.scale_tag(0.0) == "scale_0"

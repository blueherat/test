from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from experiments.audit_prediction_target_rank_spectra import (
    covariance_spectrum,
    spectrum_summary,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    RankOutputMLP,
    analytic_native_output,
    analytic_skip_velocity_via_target,
    condition_predictions,
    stamp_seed_metadata,
)
from experiments.run_prediction_target_operator_hybrid_from_rank_baseline import (
    hybrid_velocity,
    project_data_subspace,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import CurvedEmbedding


def test_rank_output_model_has_requested_affine_output_rank() -> None:
    torch.manual_seed(3)
    model = RankOutputMLP(32, hidden=24, output_rank=7, depth=4, time_dim=8)
    assert torch.linalg.matrix_rank(model.output.weight).item() == 7

    state = torch.randn(128, 32)
    time = torch.rand(128)
    output = model(state, time)
    centered = output - output.mean(dim=0, keepdim=True)
    assert torch.linalg.matrix_rank(centered, tol=1e-5).item() <= 7


def test_analytic_native_targets_recover_identical_velocity() -> None:
    generator = torch.Generator().manual_seed(5)
    state = torch.randn(11, 19, generator=generator)
    clean_residual = torch.randn(11, 19, generator=generator)
    time = torch.linspace(0.05, 0.95, 11)
    expected = (state - clean_residual) / time[:, None]

    for target in ("x", "v", "eps"):
        native = analytic_native_output(clean_residual, state, time, target, 1e-6)
        actual = analytic_skip_velocity_via_target(
            clean_residual, state, time, target, 1e-6
        )
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert native.shape == state.shape


def test_native_x_and_analytic_skip_have_identical_updates() -> None:
    torch.manual_seed(7)
    x_model = RankOutputMLP(16, hidden=20, output_rank=5, depth=3, time_dim=8)
    skip_model = copy.deepcopy(x_model)
    x_optimizer = torch.optim.AdamW(x_model.parameters(), lr=1e-3)
    skip_optimizer = torch.optim.AdamW(skip_model.parameters(), lr=1e-3)
    state = torch.randn(32, 16)
    time = torch.rand(32).mul(0.9).add(0.05)
    target_velocity = torch.randn(32, 16)

    for model, optimizer, condition in (
        (x_model, x_optimizer, "native_x"),
        (skip_model, skip_optimizer, "analytic_skip"),
    ):
        optimizer.zero_grad(set_to_none=True)
        _native, velocity, _clean = condition_predictions(
            model=model,
            condition=condition,
            state=state,
            time=time,
            clip=0.02,
        )
        F.mse_loss(velocity, target_velocity).backward()
        optimizer.step()

    for x_value, skip_value in zip(x_model.parameters(), skip_model.parameters()):
        torch.testing.assert_close(x_value, skip_value, atol=0, rtol=0)


def test_spectrum_tail_matches_best_rank_affine_residual() -> None:
    torch.manual_seed(11)
    left = torch.randn(200, 3)
    right = torch.randn(3, 12)
    values = left @ right + torch.randn(12)
    spectrum = covariance_spectrum(values)
    summary = spectrum_summary(spectrum, rank=3)
    assert summary["effective_rank"] <= 3.01
    assert summary["tail_fraction"] < 1e-10


def test_operator_hybrid_exactly_recovers_paired_velocity_on_linear_data() -> None:
    embedding = CurvedEmbedding(
        24,
        curvature=0.0,
        frequency_scale=4.0,
        seed=13,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
    )
    generator = torch.Generator().manual_seed(17)
    intrinsic = torch.randn(9, 2, generator=generator)
    clean = embedding.embed(intrinsic)
    eps = torch.randn(9, 24, generator=generator)
    time = torch.linspace(0.05, 0.95, 9)
    state = (1.0 - time[:, None]) * clean + time[:, None] * eps
    true_velocity = eps - clean
    projector_complement_clean = clean - project_data_subspace(clean, embedding)
    torch.testing.assert_close(
        projector_complement_clean,
        torch.zeros_like(projector_complement_clean),
        atol=3e-5,
        rtol=0,
    )

    native_hybrid = project_data_subspace(true_velocity, embedding)
    recovered = hybrid_velocity(
        native_hybrid,
        state,
        time,
        embedding,
        clip=1e-6,
    )
    torch.testing.assert_close(recovered, true_velocity, atol=3e-5, rtol=3e-5)


def test_seed_metadata_distinguishes_experiment_and_setting_seeds() -> None:
    rows = [{"seed": 99, "value": 1.0}]
    stamp_seed_metadata(rows, experiment_seed=7, setting_seed=101)
    assert rows == [{"seed": 7, "value": 1.0, "setting_seed": 101}]

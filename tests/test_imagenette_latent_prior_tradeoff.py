import subprocess
import sys
from pathlib import Path

import torch

from experiments.imagenette_latent_prior_tradeoff import (
    LatentPriorTradeoffConfig,
    OrthogonalLatentInterface,
    build_prior,
    covariance_statistics,
    fixed_orthogonal_basis,
    sample_prior_coordinates,
    train_prior,
)


def test_fixed_basis_is_orthogonal_and_interface_roundtrips():
    basis = fixed_orthogonal_basis(32, seed=3)
    torch.testing.assert_close(basis.T @ basis, torch.eye(32), atol=2e-6, rtol=0)
    interface = OrthogonalLatentInterface(7, basis)
    coordinates = torch.randn(11, 7, generator=torch.Generator().manual_seed(5))
    embedded = interface.embed(coordinates)
    recovered = interface.recover(embedded)
    projected = interface.project(torch.randn(11, 32))
    torch.testing.assert_close(recovered, coordinates, atol=2e-6, rtol=0)
    torch.testing.assert_close(interface.project(projected), projected, atol=2e-6, rtol=0)


def test_prior_model_is_capacity_independent():
    first = LatentPriorTradeoffConfig(
        latent_dim=16, prior_width=16, prior_depth=1, device="cpu"
    )
    second = LatentPriorTradeoffConfig(
        latent_dim=256, prior_width=16, prior_depth=1, device="cpu"
    )
    model_first = build_prior(first, torch.device("cpu"))
    model_second = build_prior(second, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in model_first.parameters()) == sum(
        parameter.numel() for parameter in model_second.parameters()
    )
    for left, right in zip(model_first.state_dict().values(), model_second.state_dict().values()):
        assert torch.equal(left, right)


def test_training_streams_and_initialization_match_across_capacities():
    basis = fixed_orthogonal_basis(256, seed=7)
    common = dict(
        frozen_seed=2,
        prior_width=16,
        prior_depth=1,
        prior_steps=2,
        prior_batch_size=8,
        log_every=1,
        eval_batch_size=8,
        device="cpu",
        save=False,
    )
    config16 = LatentPriorTradeoffConfig(latent_dim=16, **common)
    config64 = LatentPriorTradeoffConfig(latent_dim=64, **common)
    generator = torch.Generator().manual_seed(9)
    train16, val16 = torch.randn(32, 16, generator=generator), torch.randn(16, 16, generator=generator)
    train64, val64 = torch.randn(32, 64, generator=generator), torch.randn(16, 64, generator=generator)
    _, _, metadata16 = train_prior(
        train16, val16, OrthogonalLatentInterface(16, basis), config16
    )
    _, _, metadata64 = train_prior(
        train64, val64, OrthogonalLatentInterface(64, basis), config64
    )
    assert metadata16["prior_initial_sha256"] == metadata64["prior_initial_sha256"]
    assert metadata16["prior_parameters"] == metadata64["prior_parameters"]
    assert metadata16["stream_indices_first_32_sha256"] == metadata64["stream_indices_first_32_sha256"]
    assert metadata16["stream_base_noise_first_32_sha256"] == metadata64["stream_base_noise_first_32_sha256"]
    assert metadata16["stream_time_first_32_sha256"] == metadata64["stream_time_first_32_sha256"]


class GaussianScaleVelocity(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, state, time):
        path_scale = (1.0 - time) * self.scale + time
        return ((1.0 - self.scale) / path_scale)[:, None] * state


def test_descending_prior_euler_recovers_scaled_gaussian_moments():
    basis = torch.eye(8)
    interface = OrthogonalLatentInterface(8, basis)
    model = GaussianScaleVelocity(0.4)
    samples = sample_prior_coordinates(
        model,
        interface,
        20_000,
        200,
        seed=13,
        batch_size=1_000,
    )
    assert abs(float(samples.mean())) < 0.01
    assert abs(float(samples.std()) - 0.4) < 0.01


def test_covariance_metrics_detect_exact_match_and_rank_loss():
    real = torch.randn(1_000, 8, generator=torch.Generator().manual_seed(17))
    exact = covariance_statistics(real, real.clone())
    assert exact["latent_covariance_relative_error"] == 0.0
    assert abs(exact["latent_covariance_eigenvalue_overlap"] - 1.0) < 1e-12
    collapsed = real.clone()
    collapsed[:, 4:] = 0
    changed = covariance_statistics(real, collapsed)
    assert changed["latent_covariance_relative_error"] > 0.5
    assert changed["generated_latent_effective_rank"] < changed["real_latent_effective_rank"]


def test_script_entrypoint_help_loads_repo_modules():
    script = Path(__file__).resolve().parents[1] / "experiments/imagenette_latent_prior_tradeoff.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

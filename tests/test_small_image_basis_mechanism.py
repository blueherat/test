import torch

from experiments.mnist_spectral_rollout_toy import TinyVelocityUNet
from experiments.small_image_basis_mechanism import (
    reference_component_drift,
    reference_component_energy,
    rollout_snapshots,
)


def test_reference_energy_and_drift_match_monte_carlo_path():
    count = 100_000
    generator = torch.Generator().manual_seed(0)
    clean = 1.7 * torch.randn((count, 3), generator=generator) + 0.4
    noise = torch.randn((count, 3), generator=generator)
    clean_energy = clean.square().mean(dim=0)
    time = 0.63
    state = (1.0 - time) * clean + time * noise
    velocity = noise - clean

    empirical_energy = state.square().mean(dim=0)
    empirical_drift = 2.0 * (state * velocity).mean(dim=0)

    assert torch.allclose(
        empirical_energy,
        reference_component_energy(clean_energy, time),
        atol=0.025,
        rtol=0.015,
    )
    assert torch.allclose(
        empirical_drift,
        reference_component_drift(clean_energy, time),
        atol=0.035,
        rtol=0.02,
    )


def test_image_rollout_snapshots_are_grid_aligned_and_finite():
    model = TinyVelocityUNet(width=4, depth=1).eval()
    initial = torch.randn((8, 1, 28, 28), generator=torch.Generator().manual_seed(1))
    snapshots = rollout_snapshots(
        model,
        initial,
        (0.9, 0.5, 0.1),
        ode_steps=10,
        batch_size=4,
    )

    assert set(snapshots) == {0.9, 0.5, 0.1}
    assert all(value.shape == initial.shape for value in snapshots.values())
    assert torch.isfinite(torch.stack(list(snapshots.values()))).all()

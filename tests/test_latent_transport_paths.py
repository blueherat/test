import torch

from experiments.latent_equiv_adapter import InvertibleLatentAdapter
from experiments.latent_transport_paths import (
    ScaledAdditiveCouplingTransform,
    bridge_commutation_defect,
    conditional_path_sample,
    jvp_relative_error,
    relative_l2_per_sample,
)


class LinearMap(torch.nn.Module):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.matrix.T

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.matrix, value.T).T


class NonlinearShear(torch.nn.Module):
    def __init__(self, strength: float):
        super().__init__()
        self.strength = float(strength)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.unbind(dim=-1)
        return torch.stack((first, second + self.strength * first.square()), dim=-1)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.unbind(dim=-1)
        return torch.stack((first, second - self.strength * first.square()), dim=-1)


def _paired_values(dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    data = torch.randn((16, 2), generator=generator, dtype=dtype)
    noise = torch.randn((16, 2), generator=generator, dtype=dtype)
    time = torch.linspace(0.05, 0.95, len(data), dtype=dtype)
    return data, noise, time


def test_identity_map_makes_all_four_branches_identical():
    data, noise, time = _paired_values()
    identity = LinearMap(torch.eye(2, dtype=data.dtype))
    samples = {
        branch: conditional_path_sample(
            data,
            noise,
            time,
            branch=branch,
            transform=identity,
        )
        for branch in ("base", "gaussian_straight", "matched_chord", "pushforward")
    }

    for sample in samples.values():
        torch.testing.assert_close(sample.state, samples["base"].state)
        torch.testing.assert_close(sample.velocity, samples["base"].velocity)
    assert float(bridge_commutation_defect(data, noise, time, identity).max()) < 1e-12


def test_affine_linear_map_commutes_with_chord_but_not_standard_gaussian_source():
    data, noise, time = _paired_values()
    linear = LinearMap(torch.diag(torch.tensor([2.0, 0.5], dtype=data.dtype)))
    gaussian = conditional_path_sample(
        data,
        noise,
        time,
        branch="gaussian_straight",
        transform=linear,
    )
    chord = conditional_path_sample(
        data,
        noise,
        time,
        branch="matched_chord",
        transform=linear,
    )
    pushforward = conditional_path_sample(
        data,
        noise,
        time,
        branch="pushforward",
        transform=linear,
    )

    torch.testing.assert_close(chord.state, pushforward.state)
    torch.testing.assert_close(chord.velocity, pushforward.velocity)
    assert not torch.allclose(gaussian.source_endpoint, chord.source_endpoint)
    assert float(bridge_commutation_defect(data, noise, time, linear).max()) < 1e-12


def test_nonlinear_map_has_chord_defect_and_pushforward_analytic_velocity():
    data, noise, time = _paired_values()
    strength = 0.7
    transform = NonlinearShear(strength)
    pushforward = conditional_path_sample(
        data,
        noise,
        time,
        branch="pushforward",
        transform=transform,
    )
    expanded_time = time[:, None]
    base_state = (1.0 - expanded_time) * data + expanded_time * noise
    base_velocity = noise - data
    expected_velocity = torch.stack(
        (
            base_velocity[:, 0],
            base_velocity[:, 1]
            + 2.0 * strength * base_state[:, 0] * base_velocity[:, 0],
        ),
        dim=-1,
    )

    torch.testing.assert_close(pushforward.state, transform(base_state))
    torch.testing.assert_close(pushforward.velocity, expected_velocity)
    assert float(bridge_commutation_defect(data, noise, time, transform).mean()) > 1e-3


def test_pushforward_path_has_exact_transformed_endpoints():
    data, noise, _ = _paired_values()
    transform = NonlinearShear(0.4)
    for time, expected in ((0.0, transform(data)), (1.0, transform(noise))):
        sample = conditional_path_sample(
            data,
            noise,
            torch.tensor(time, dtype=data.dtype),
            branch="pushforward",
            transform=transform,
        )
        torch.testing.assert_close(sample.state, expected)


def test_jvp_matches_central_finite_difference():
    data, noise, _ = _paired_values()
    error = jvp_relative_error(
        NonlinearShear(0.6),
        data,
        noise - data,
        step=1e-5,
    )
    assert float(error.max()) < 1e-8


def test_real_additive_coupling_architecture_is_numerically_invertible():
    generator = torch.Generator().manual_seed(11)
    adapter = InvertibleLatentAdapter(channels=4, hidden_channels=8, blocks=3).double()
    with torch.no_grad():
        for block in adapter.blocks:
            block.net.net[-1].weight.normal_(std=0.02, generator=generator)
            block.net.net[-1].bias.normal_(std=0.02, generator=generator)
    value = torch.randn((3, 4, 5, 5), generator=generator, dtype=torch.float64)
    reconstructed = adapter.inverse(adapter(value))
    error = relative_l2_per_sample(reconstructed, value)
    assert float(error.detach().max()) < 1e-12


def test_scaled_additive_coupling_is_exact_and_matches_endpoints():
    generator = torch.Generator().manual_seed(19)
    adapter = InvertibleLatentAdapter(channels=4, hidden_channels=8, blocks=3).double()
    with torch.no_grad():
        for block in adapter.blocks:
            block.net.net[-1].weight.normal_(std=0.02, generator=generator)
            block.net.net[-1].bias.normal_(std=0.02, generator=generator)
    value = torch.randn((2, 4, 4, 4), generator=generator, dtype=torch.float64)
    identity = ScaledAdditiveCouplingTransform(adapter, 0.0)
    full = ScaledAdditiveCouplingTransform(adapter, 1.0)
    torch.testing.assert_close(identity(value), value)
    torch.testing.assert_close(full(value), adapter(value))
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        transform = ScaledAdditiveCouplingTransform(adapter, scale)
        torch.testing.assert_close(transform.inverse(transform(value)), value, atol=1e-12, rtol=1e-12)


def test_invalid_shapes_are_rejected():
    data, noise, time = _paired_values()
    try:
        conditional_path_sample(
            data,
            noise[:, :1],
            time,
            branch="base",
        )
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("shape mismatch must raise ValueError")

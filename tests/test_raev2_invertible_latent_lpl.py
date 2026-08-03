import torch

from experiments.latent_equiv_adapter import InvertibleLatentAdapter
from experiments.raev2_invertible_latent_lpl import (
    adapter_config,
    cycle_metrics,
    make_reparameterized_path,
    trainable_parameter_boundary,
)


def test_adapter_starts_as_exact_identity_and_has_exact_inverse():
    adapter = InvertibleLatentAdapter(channels=4, hidden_channels=8, blocks=3)
    latent = torch.randn(2, 4, 5, 5)

    transformed = adapter(latent)
    recovered = adapter.inverse(transformed)

    torch.testing.assert_close(transformed, latent, atol=0.0, rtol=0.0)
    torch.testing.assert_close(recovered, latent, atol=0.0, rtol=0.0)
    metrics = cycle_metrics(adapter, latent)
    assert metrics["cycle_max_abs"].item() == 0.0
    assert metrics["forward_relative_mse"].item() == 0.0


def test_reparameterized_path_matches_raev2_linear_path_at_identity():
    adapter = InvertibleLatentAdapter(channels=4, hidden_channels=8, blocks=2)
    clean = torch.randn(3, 4, 2, 2)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.2, 0.5, 0.9])

    path = make_reparameterized_path(
        adapter,
        clean,
        noise,
        time,
        t_eps=0.05,
    )
    time_scale = time.view(3, 1, 1, 1)

    torch.testing.assert_close(path.transformed_clean, clean)
    torch.testing.assert_close(
        path.noisy_transformed,
        (1.0 - time_scale) * clean + time_scale * noise,
    )
    torch.testing.assert_close(path.target_velocity, noise - clean)


def test_path_loss_backpropagates_through_frozen_predictor_to_adapter():
    adapter = InvertibleLatentAdapter(channels=4, hidden_channels=8, blocks=2)
    frozen_predictor = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
    frozen_predictor.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    clean = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.3, 0.7])

    path = make_reparameterized_path(
        adapter,
        clean,
        noise,
        time,
        t_eps=0.05,
    )
    transformed_prediction = frozen_predictor(path.noisy_transformed)
    recovered_prediction = adapter.inverse(transformed_prediction)
    loss = (recovered_prediction - clean).square().mean()
    loss.backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in adapter.parameters()
    )
    assert all(parameter.grad is None for parameter in frozen_predictor.parameters())
    boundary = trainable_parameter_boundary(
        adapter,
        (frozen_predictor,),
        optimizer,
    )
    assert boundary["trainable_parameters"] > 0
    assert boundary["frozen_parameters"] == sum(
        parameter.numel() for parameter in frozen_predictor.parameters()
    )


def test_adapter_config_is_recoverable_from_module():
    adapter = InvertibleLatentAdapter(channels=6, hidden_channels=10, blocks=4)
    assert adapter_config(adapter) == {
        "channels": 6,
        "hidden_channels": 10,
        "blocks": 4,
    }

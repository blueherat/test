import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from experiments.rae_decoder_risk_phase0 import (
    Phase0GateThresholds,
    banded_metric_loss,
    channel_metric_loss,
    clean_from_velocity,
    decoder_embed_metric,
    decoder_hidden_features,
    decoder_hidden_loss,
    dct2,
    dct_matrix,
    gradient_energy_distributions,
    idct2,
    loss_space_gate,
    proxy_gate,
    radial_dct_band_masks,
    static_linear_state,
    trace_normalize_banded_metric,
    velocity_and_clean_losses,
)


class _FakeDecoder(torch.nn.Module):
    def __init__(self, channels: int = 3, hidden: int = 5):
        super().__init__()
        self.decoder_embed = torch.nn.Linear(channels, hidden, bias=False)

    def forward(self, tokens, **kwargs):
        del kwargs
        embedded = self.decoder_embed(tokens)
        cls = embedded.new_zeros(len(tokens), 1, embedded.shape[-1])
        state = torch.cat([cls, embedded], dim=1)
        return SimpleNamespace(hidden_states=(state, state.square()))


def test_static_path_losses_are_exactly_t_squared_velocity_loss():
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(3, 4, 2, 2, generator=generator)
    noise = torch.randn(3, 4, 2, 2, generator=generator)
    time = torch.tensor([0.2, 0.5, 0.9])
    state, target = static_linear_state(clean, noise, time)
    prediction = target + torch.randn(target.shape, generator=generator) * 0.1
    estimate = clean_from_velocity(state, prediction, time)
    velocity_loss, clean_loss = velocity_and_clean_losses(prediction, target, time)

    measured_clean = (estimate - clean).square().flatten(1).mean(dim=1)
    assert torch.allclose(clean_loss, measured_clean, atol=1e-7, rtol=1e-6)
    assert torch.allclose(clean_loss, time.square() * velocity_loss, atol=1e-7, rtol=1e-6)


def test_dct_round_trip_and_identity_metrics_preserve_mse():
    generator = torch.Generator().manual_seed(11)
    error = torch.randn(5, 3, 4, 4, generator=generator)
    basis = dct_matrix(4)
    masks = radial_dct_band_masks(4, 4)
    identity = torch.eye(3)
    banded = trace_normalize_banded_metric(identity.repeat(4, 1, 1), masks)

    assert torch.allclose(idct2(dct2(error, basis), basis), error, atol=2e-6, rtol=2e-6)
    expected = error.square().flatten(1).mean(dim=1)
    assert torch.allclose(channel_metric_loss(error, identity), expected, atol=1e-6)
    assert torch.allclose(
        banded_metric_loss(error, banded, masks, basis), expected, atol=2e-6, rtol=2e-6
    )


def test_gradient_distributions_are_normalized():
    gradient = torch.randn(2, 5, 4, 4, generator=torch.Generator().manual_seed(19))
    distributions = gradient_energy_distributions(
        gradient,
        basis=dct_matrix(4),
        masks=radial_dct_band_masks(4, 4),
    )
    assert distributions["channel"].shape == (2, 5)
    assert distributions["token"].shape == (2, 16)
    assert distributions["dct"].shape == (2, 4)
    for value in distributions.values():
        assert torch.allclose(value.sum(dim=1), torch.ones(2), atol=1e-6)


def test_loss_space_gate_requires_distinct_and_consistent_decoder_signal():
    rows = []
    for time_bin in range(5):
        for _ in range(4):
            rows.append(
                {
                    "time_bin": time_bin,
                    "gradient_cosine_x0_dec": 0.7 if time_bin < 3 else 0.9,
                    "decoder_reduction_x0": 0.10,
                    "decoder_reduction_dec": 0.13 if time_bin < 3 else 0.105,
                }
            )
    result = loss_space_gate(pd.DataFrame(rows))
    assert result["pass"] is True
    assert result["distinct_time_bins"] == 3
    assert result["better_time_bins"] == 3


def test_proxy_gate_uses_heldout_rank_correlation_gradient_and_split_gap():
    scores = []
    gradients = []
    random = np.random.default_rng(23)
    for proxy, noise in (("good", 0.01), ("bad", 2.0)):
        for split in ("calibration", "test"):
            target = np.linspace(0.0, 1.0, 64)
            predicted = target + random.normal(scale=noise, size=len(target))
            scores.extend(
                {
                    "proxy": proxy,
                    "split": split,
                    "l_dec": float(left),
                    "l_proxy": float(right),
                }
                for left, right in zip(target, predicted)
            )
        gradients.extend(
            {"proxy": proxy, "gradient_cosine_proxy_dec": value}
            for value in ([0.8] * 8 if proxy == "good" else [0.2] * 8)
        )
    result = proxy_gate(pd.DataFrame(scores), pd.DataFrame(gradients))
    assert result["pass"] is True
    assert result["selected_proxy"] == "good"


def test_stricter_gate_can_reject_same_rows():
    rows = pd.DataFrame(
        {
            "time_bin": np.repeat(np.arange(5), 2),
            "gradient_cosine_x0_dec": 0.75,
            "decoder_reduction_x0": 0.1,
            "decoder_reduction_dec": 0.12,
        }
    )
    strict = Phase0GateThresholds(correction_advantage=0.5)
    assert loss_space_gate(rows, strict)["pass"] is False


def test_decoder_embed_metric_averages_spatial_normalization_exactly():
    decoder = _FakeDecoder(channels=3, hidden=2)
    with torch.no_grad():
        decoder.decoder_embed.weight.copy_(
            torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        )
    variance = torch.tensor(
        [
            [[1.0, 4.0], [9.0, 16.0]],
            [[4.0, 4.0], [4.0, 4.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    rae = SimpleNamespace(
        decoder=decoder,
        do_normalization=True,
        latent_var=variance,
        eps=0.0,
    )
    scale = variance.sqrt().reshape(3, -1)
    expected = (decoder.decoder_embed.weight.T @ decoder.decoder_embed.weight) * (
        scale @ scale.T / scale.shape[1]
    )
    expected = expected * (3.0 / torch.trace(expected))
    assert torch.allclose(decoder_embed_metric(rae), expected, atol=1e-6)


def test_decoder_hidden_loss_keeps_gradient_to_normalized_latent():
    decoder = _FakeDecoder(channels=3, hidden=4)
    rae = SimpleNamespace(
        decoder=decoder,
        do_normalization=False,
        reshape_to_2d=True,
    )
    reference_latent = torch.zeros(2, 3, 2, 2)
    candidate_latent = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(31))
    candidate_latent.requires_grad_(True)
    with torch.no_grad():
        reference = decoder_hidden_features(rae, reference_latent)
    loss = decoder_hidden_loss(decoder_hidden_features(rae, candidate_latent), reference).sum()
    gradient = torch.autograd.grad(loss, candidate_latent)[0]
    assert gradient.shape == candidate_latent.shape
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0

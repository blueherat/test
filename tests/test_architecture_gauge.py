import math

import torch

from experiments.architecture_gauge import (
    GaugeSpec,
    ProbeConfig,
    ProbeTrainingConfig,
    _flow_batch,
    build_probe,
    evaluate_probe,
    exact_equivalence_table,
    make_gauge,
)


def test_all_gauges_are_invertible_and_norm_preserving():
    torch.manual_seed(0)
    z = torch.randn(5, 8, 8, 8)
    specs = [
        GaugeSpec("identity"),
        GaugeSpec("roll", kind="roll", shift_x=2, shift_y=-1),
        GaugeSpec("channel", kind="channel_givens", strength=0.7, seed=3),
        GaugeSpec("allpass", kind="fourier_allpass", strength=0.8, radius=3),
        GaugeSpec("haar", kind="block_haar"),
    ]
    table = exact_equivalence_table(z, specs, device="cpu")
    assert table["inverse_rel_l2"].max() < 2e-6
    assert table["norm_rel_error"].max() < 2e-6
    assert table["pairwise_distance_rel_error"].max() < 2e-6
    assert table["paired_noise_rel_error"].max() < 2e-6


def test_allpass_preserves_total_power_spectrum():
    torch.manual_seed(1)
    z = torch.randn(3, 4, 8, 8)
    spec = GaugeSpec("allpass", kind="fourier_allpass", strength=1.1, radius=2)
    table = exact_equivalence_table(z, [spec], device="cpu")
    assert float(table.loc[0, "total_psd_rel_error"]) < 2e-6


def test_gauge_inverse_is_not_a_learned_approximation():
    z = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    spec = GaugeSpec("channel", kind="channel_givens", strength=0.35, seed=4)
    gauge = make_gauge(spec)
    assert torch.allclose(gauge.inverse(gauge.forward(z)), z, atol=1e-4, rtol=1e-6)


def test_flow_velocity_recovers_clean_latent_through_gauge_inverse():
    torch.manual_seed(2)
    z = torch.randn(4, 6, 8, 8)
    epsilon = torch.randn_like(z)
    t = torch.tensor([0.1, 0.35, 0.65, 0.9])
    gauge = make_gauge(
        GaugeSpec("allpass", kind="fourier_allpass", strength=0.7, radius=2)
    )

    noisy_y, velocity_y = _flow_batch(z, t, epsilon, gauge)
    alpha = torch.cos(0.5 * math.pi * t).view(-1, 1, 1, 1)
    sigma = torch.sin(0.5 * math.pi * t).view(-1, 1, 1, 1)
    recovered_z = gauge.inverse(alpha * noisy_y - sigma * velocity_y)

    assert torch.allclose(recovered_z, z, atol=2e-6, rtol=2e-6)


def test_full_validation_evaluation_covers_every_sample_once():
    latents = torch.randn(11, 4, 4, 4)
    probe = build_probe(
        latents.shape[1],
        ProbeConfig("local", kind="local", hidden=8, depth=1),
    )
    training = ProbeTrainingConfig(
        batch_size=4,
        eval_batches=1,
        eval_full_dataset=True,
        time_bins=3,
    )
    _, rows = evaluate_probe(
        probe,
        latents,
        make_gauge(GaugeSpec("identity")),
        training,
        torch.device("cpu"),
    )

    assert sum(row["count"] for row in rows) == len(latents)

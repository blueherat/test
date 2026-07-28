from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.rae_lpl_error_geometry import (
    feature_normalization_decomposition,
    finite_difference_feature_gain,
    paired_amplitudes,
    raw_feature_layer_losses,
    sample_rms,
    scale_direction_to_rms,
    shuffled_direction,
    summarize_geometry_rows,
    unit_rms_direction,
)


def test_equal_rms_controls_preserve_direction_and_target_norm() -> None:
    generator = torch.Generator().manual_seed(7)
    direction = torch.randn(3, 5, 4, 4, generator=generator)
    target = torch.tensor([0.1, 0.2, 0.3])
    scaled = scale_direction_to_rms(direction, target)

    torch.testing.assert_close(sample_rms(unit_rms_direction(direction)), torch.ones(3))
    torch.testing.assert_close(sample_rms(scaled), target)
    cosine = torch.nn.functional.cosine_similarity(
        direction.flatten(1), scaled.flatten(1), dim=1
    )
    torch.testing.assert_close(cosine, torch.ones(3))


def test_paired_amplitudes_use_smaller_error_and_local_cap() -> None:
    clean = torch.ones(2, 3, 2, 2)
    flow_error = torch.stack((torch.ones(3, 2, 2) * 0.3, torch.ones(3, 2, 2) * 0.1))
    lpl_error = torch.stack((torch.ones(3, 2, 2) * 0.2, torch.ones(3, 2, 2) * 0.4))
    realistic, local = paired_amplitudes(
        clean, flow_error, lpl_error, local_fraction=0.05
    )

    torch.testing.assert_close(realistic, torch.tensor([0.2, 0.1]))
    torch.testing.assert_close(local, torch.tensor([0.05, 0.05]))


def test_shuffled_direction_preserves_rms_but_disrupts_layout() -> None:
    direction = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    shuffled = shuffled_direction(direction)
    torch.testing.assert_close(sample_rms(shuffled), sample_rms(direction))
    assert not torch.equal(shuffled, direction)


def test_finite_difference_recovers_linear_decoder_gain() -> None:
    generator = torch.Generator().manual_seed(11)
    direction = unit_rms_direction(torch.randn(2, 3, 2, 2, generator=generator))
    step = torch.tensor([1e-3, 2e-3])
    matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])

    def features(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tokens = value.permute(0, 2, 3, 1)
        transformed = torch.einsum("bhwc,dc->bhwd", tokens, matrix)
        return (transformed,)

    delta = direction * step[:, None, None, None]
    plus = features(delta)
    minus = features(-delta)
    layer_gain, total_gain = finite_difference_feature_gain(plus, minus, step)
    expected = raw_feature_layer_losses(features(direction), features(torch.zeros_like(direction)))

    torch.testing.assert_close(layer_gain, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(total_gain, expected[:, 0], rtol=2e-5, atol=2e-5)


def test_normalization_decomposition_exposes_prediction_variance_denominator() -> None:
    reference = torch.tensor([[[1.0], [-1.0]]])
    candidate = 2.0 * reference
    metrics = feature_normalization_decomposition(
        (candidate,), (reference,), eps=0.0
    )

    target = metrics["target_normalized"][0, 0]
    prediction = metrics["prediction_normalized"][0, 0]
    symmetric = metrics["symmetric_normalized"][0, 0]
    torch.testing.assert_close(prediction, target / 4.0)
    torch.testing.assert_close(symmetric, target / 2.5)
    torch.testing.assert_close(
        metrics["prediction_over_target_variance_gmean"][0, 0],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        metrics["centered_channel_cosine"][0, 0], torch.tensor(1.0)
    )


def test_summary_gate_distinguishes_decoder_geometry_from_latent_mse() -> None:
    rows = []
    for seed in range(4):
        for sample in range(8):
            actual = 1.0 + 0.2 * sample
            rows.append(
                {
                    "training_seed": seed,
                    "sample_index": sample,
                    "noise_to_signal_ratio": 1.0,
                    "flow_latent_mse": 1.0 + (sample % 2) * 0.01,
                    "lpl_latent_mse": 1.01 + (sample % 2) * 0.01,
                    "flow_actual_raw_loss": actual,
                    "lpl_actual_raw_loss": 0.8 * actual,
                    "flow_local_gain": actual,
                    "lpl_local_gain": 0.8 * actual,
                    "random_local_gain": 1.1 * actual,
                    "shuffled_lpl_local_gain": 1.0 * actual,
                    "flow_fd_gain": actual,
                    "lpl_fd_gain": 0.8 * actual,
                    "random_fd_gain": 1.1 * actual,
                    "flow_quadratic_prediction": actual,
                    "lpl_quadratic_prediction": 0.8 * actual,
                    "flow_local_strict_gain": actual,
                    "lpl_local_strict_gain": 0.8 * actual,
                    "flow_fd_strict_gain": actual,
                    "lpl_fd_strict_gain": 0.8 * actual,
                    "flow_actual_strict_lpl": actual,
                    "lpl_actual_strict_lpl": 0.8 * actual,
                    "flow_strict_quadratic_prediction": actual,
                    "lpl_strict_quadratic_prediction": 0.8 * actual,
                }
            )
    seed_table, gate = summarize_geometry_rows(pd.DataFrame(rows))

    assert len(seed_table) == 4
    assert np.isclose(seed_table["local_gain_lpl_over_flow_gmean"], 0.8).all()
    assert gate["improved_seed_count"] == 4
    assert gate["quadratic_prediction_to_actual_raw_spearman"] > 0.99
    assert gate["mechanism_supported"] is True

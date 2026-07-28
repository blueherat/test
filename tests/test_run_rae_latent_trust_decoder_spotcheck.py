from __future__ import annotations

import pandas as pd
import pytest
import torch

from experiments.analyze_rae_predictability_gain import MATCHED_PAIRS
from experiments.run_rae_latent_trust_decoder_spotcheck import (
    pair_decoder_ratios,
    paired_image_metrics,
    summarize_gate,
)


class SquaredDistanceNet(torch.nn.Module):
    def forward(
        self, first: torch.Tensor, second: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor:
        assert normalize
        return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)


class FakeLpips:
    def __init__(self) -> None:
        self.net = SquaredDistanceNet()


def test_paired_image_metrics_use_clean_and_base_references() -> None:
    clean = torch.zeros((2, 3, 4, 4))
    base = torch.full_like(clean, 0.25)
    perturbed = torch.full_like(clean, 0.5)
    metrics = paired_image_metrics(clean, base, perturbed, FakeLpips())
    torch.testing.assert_close(metrics["image_shift_l1"], torch.full((2,), 0.25))
    torch.testing.assert_close(metrics["base_clean_l1"], torch.full((2,), 0.25))
    torch.testing.assert_close(metrics["perturbed_clean_l1"], torch.full((2,), 0.5))
    torch.testing.assert_close(metrics["clean_l1_increase"], torch.full((2,), 0.25))
    torch.testing.assert_close(metrics["image_shift_lpips"], torch.full((2,), 0.0625))


def _synthetic_per_seed() -> pd.DataFrame:
    rows = []
    for seed in range(5):
        for target_time in (0.3, 0.85, 0.95):
            values: dict[str, float] = {}
            for high, low in MATCHED_PAIRS:
                high_scale = 0.8 if target_time == 0.3 else 1.5 + target_time
                values[high] = high_scale
                values[low] = 1.0
            for basis, scale in values.items():
                rows.append(
                    {
                        "seed": seed,
                        "target_time": target_time,
                        "basis": basis,
                        "endpoint_shift_gain": scale,
                        "image_shift_l1": scale,
                        "image_shift_lpips": scale,
                        "decoder_l1_secant": scale,
                        "decoder_lpips_secant": scale,
                        "clean_l1_increase": scale - 1.0,
                        "clean_lpips_increase": scale - 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_decoder_gate_recognizes_consistent_time_crossover() -> None:
    ratios = pair_decoder_ratios(_synthetic_per_seed())
    assert len(ratios) == 5 * 3 * len(MATCHED_PAIRS)
    low = ratios[ratios["target_time"] == 0.3]
    high = ratios[ratios["target_time"] == 0.95]
    assert (low["image_shift_l1_ratio"] < 1.0).all()
    assert (high["image_shift_lpips_ratio"] > 1.0).all()
    gate = summarize_gate(ratios)
    assert gate["pass"] is True
    assert gate["high_noise_l1_lpips_direction_agreement_pairs"] == pytest.approx(3)

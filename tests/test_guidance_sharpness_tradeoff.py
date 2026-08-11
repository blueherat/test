from __future__ import annotations

import pandas as pd

from experiments.analyze_guidance_sharpness_tradeoff import (
    candidate_table,
    contrast_match_table,
    is_windowed,
    method_family,
    summarize_contrast_regions,
)
import numpy as np


def test_method_family_and_window_parsing() -> None:
    assert method_family("ag_early_w3_mid03_07") == "ag_early"
    assert method_family("ig_w2.5_mid025_075") == "ig"
    assert method_family("ptg_w1.5") == "ptg"
    assert is_windowed("ig_w3_mid03_07")
    assert not is_windowed("ig_w3")


def test_candidate_table_applies_fixed_gate_then_minimizes_swd() -> None:
    reference = pd.Series(
        {
            "mean_adjacent_log_density_contrast": 5.0,
            "intrinsic_bridge_rate": 0.05,
            "occupied_components": 32,
        }
    )
    frame = pd.DataFrame(
        [
            {
                "condition": "ig_w2",
                "latent_swd": 0.020,
                "mean_adjacent_log_density_contrast": 4.0,
                "intrinsic_bridge_rate": 0.07,
                "occupied_components": 32,
            },
            {
                "condition": "ig_w2.5_mid03_07",
                "latent_swd": 0.030,
                "mean_adjacent_log_density_contrast": 4.6,
                "intrinsic_bridge_rate": 0.07,
                "occupied_components": 32,
            },
            {
                "condition": "ig_w3_mid03_07",
                "latent_swd": 0.035,
                "mean_adjacent_log_density_contrast": 5.1,
                "intrinsic_bridge_rate": 0.06,
                "occupied_components": 32,
            },
        ]
    )
    selected = candidate_table(frame, reference)
    assert selected.condition.tolist() == ["ig_w2.5_mid03_07"]


def test_contrast_match_prioritizes_reference_contrast_after_safety_gate() -> None:
    reference = pd.Series(
        {
            "mean_adjacent_log_density_contrast": 5.0,
            "intrinsic_bridge_rate": 0.05,
            "occupied_components": 32,
        }
    )
    frame = pd.DataFrame(
        [
            {
                "condition": "ig_w2.5_mid03_07",
                "latent_swd": 0.029,
                "mean_adjacent_log_density_contrast": 4.6,
                "intrinsic_bridge_rate": 0.07,
                "occupied_components": 32,
            },
            {
                "condition": "ig_w2.75_mid03_07",
                "latent_swd": 0.030,
                "mean_adjacent_log_density_contrast": 5.04,
                "intrinsic_bridge_rate": 0.072,
                "occupied_components": 32,
            },
            {
                "condition": "ig_w3_mid03_07",
                "latent_swd": 0.031,
                "mean_adjacent_log_density_contrast": 5.01,
                "intrinsic_bridge_rate": 0.10,
                "occupied_components": 32,
            },
        ]
    )
    selected = contrast_match_table(frame, reference)
    assert selected.condition.tolist() == ["ig_w2.75_mid03_07"]


def test_contrast_profile_regions_cover_every_gap_once() -> None:
    profile = np.arange(31, dtype=np.float64)
    result = summarize_contrast_regions(profile)
    chunks = np.array_split(profile, 3)
    assert result["inner_contrast"] == float(chunks[0].mean())
    assert result["middle_contrast"] == float(chunks[1].mean())
    assert result["outer_contrast"] == float(chunks[2].mean())
    assert result["all_contrast"] == float(profile.mean())

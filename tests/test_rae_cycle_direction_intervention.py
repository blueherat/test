from __future__ import annotations

import pandas as pd
import torch

from experiments.rae_cycle_direction_intervention import (
    GENERATED_SOURCES,
    cycle_direction_gate,
    cyclic_derangement,
    interpolate_direction,
    matched_intervention_directions,
    sample_rms,
    select_global_alpha,
)


def test_matched_directions_have_equal_rms_and_derangement() -> None:
    residuals = torch.randn(
        7, 5, 3, 3, generator=torch.Generator().manual_seed(7)
    )
    directions = matched_intervention_directions(residuals, seed=11)
    target = sample_rms(residuals)
    for direction in directions.values():
        torch.testing.assert_close(sample_rms(direction), target)
    torch.testing.assert_close(directions["own"], residuals)
    torch.testing.assert_close(directions["opposite"], -residuals)
    permutation = cyclic_derangement(len(residuals), 11)
    assert not torch.any(permutation == torch.arange(len(residuals)))


def test_interpolation_uses_signed_cycle_direction() -> None:
    latent = torch.zeros(2, 3, 2, 2)
    direction = torch.ones_like(latent)
    torch.testing.assert_close(
        interpolate_direction(latent, direction, 0.25),
        torch.full_like(latent, 0.25),
    )
    torch.testing.assert_close(
        interpolate_direction(latent, -direction, 0.5),
        torch.full_like(latent, -0.5),
    )


def test_alpha_selection_uses_all_path_guardrails_and_smallest_tie() -> None:
    rows = []
    for source in GENERATED_SOURCES:
        for alpha, ratio, cosine in (
            (0.25, 0.85, 0.995),
            (0.50, 0.70, 0.990),
            (1.00, 0.60, 0.970),
        ):
            rows.extend(
                {
                    "source": source,
                    "condition": "own",
                    "sample_index": index,
                    "alpha": alpha,
                    "cycle_ratio_to_base": ratio,
                    "inception_cosine_to_base": cosine,
                }
                for index in range(4)
            )
    selected, summary = select_global_alpha(pd.DataFrame(rows))
    assert selected == 0.5
    assert summary.loc[summary.alpha == 1.0, "eligible"].item() is False


def _gate_rows() -> pd.DataFrame:
    rows = []

    def add(source: str, condition: str, cycle: float, hidden: float, cosine: float = 0.99):
        for index in range(4):
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "sample_index": index,
                    "cycle_relative_rms": cycle,
                    "cycle_ratio_to_base": cycle,
                    "inception_cosine_to_base": cosine,
                    "primary_hidden_rms_z": hidden,
                }
            )

    add("clean_test", "baseline", 0.4, 0.0, 1.0)
    add("clean_test", "own", 0.3, 0.0, 0.99)
    for source in GENERATED_SOURCES:
        base_cycle = 1.2 if source == "reverse" else 1.0
        base_hidden = 6.0 if source == "reverse" else 2.0
        add(source, "baseline", base_cycle, base_hidden, 1.0)
        add(source, "own", 0.7, 2.0 if source == "reverse" else 1.0)
        add(source, "shuffled", 0.9, base_hidden)
        add(source, "random", 0.95, base_hidden)
        add(source, "opposite", 1.3, 7.0 if source == "reverse" else 2.5)
    return pd.DataFrame(rows)


def test_cycle_direction_gate_requires_causal_specificity() -> None:
    result = cycle_direction_gate(_gate_rows())
    assert result["pass"] is True
    assert result["checks"]["own_beats_controls"] is True
    assert result["checks"]["opposite_worsens"] is True

    failed = _gate_rows()
    failed.loc[
        (failed.source.isin(GENERATED_SOURCES)) & (failed.condition == "shuffled"),
        "cycle_relative_rms",
    ] = 0.69
    failed.loc[
        (failed.source.isin(GENERATED_SOURCES)) & (failed.condition == "shuffled"),
        "cycle_ratio_to_base",
    ] = 0.69
    result = cycle_direction_gate(failed)
    assert result["pass"] is False
    assert result["checks"]["own_beats_controls"] is False

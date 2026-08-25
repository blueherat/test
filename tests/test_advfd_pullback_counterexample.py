from __future__ import annotations

import pandas as pd
import pytest
import torch

from experiments.run_advfd_pullback_counterexample import (
    analytic_pseudoinverse_velocity,
    exponential_pullback_field,
    run,
)


def test_pseudoinverse_pullback_matches_closed_form_everywhere() -> None:
    concentration = 1.3
    shift = 0.75
    states = torch.linspace(-4.0, 5.0, 101, dtype=torch.float64)[:, None]
    observed = exponential_pullback_field(
        concentration, shift, "pseudoinverse"
    )(states, False)
    expected = torch.full_like(
        observed, analytic_pseudoinverse_velocity(concentration, shift)
    )
    torch.testing.assert_close(observed, expected, atol=1e-11, rtol=1e-11)


def test_pullback_audit_separates_transpose_and_pseudoinverse(tmp_path) -> None:
    output = tmp_path / "audit"
    run(
        output,
        shift=0.75,
        concentrations=(0.5, 2.0),
        displacement_rms=(1e-4,),
        quadrature_order=32,
        device=torch.device("cpu"),
    )
    frame = pd.read_csv(output / "pullback_counterexample.csv")
    high = frame[frame["concentration"] == 2.0].set_index("pullback")
    assert high.loc["pseudoinverse", "score_cosine"] == pytest.approx(1.0)
    assert high.loc["pseudoinverse", "field_effective_mass"] == pytest.approx(1.0)
    assert high.loc["pseudoinverse", "feature_tracking_relative_error"] < 1e-10
    assert high.loc["transpose", "score_cosine"] < 0.01
    assert high.loc["transpose", "feature_tracking_relative_error"] > 1.0
    assert (output / "pullback_counterexample.png").is_file()

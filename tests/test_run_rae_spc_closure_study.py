from __future__ import annotations

import pandas as pd

from experiments.run_rae_spc_closure_study import summarize_closure


def test_closure_decision_detects_systematic_degradation() -> None:
    rows = []
    for seed in range(5):
        for condition, shift in (("static", 0.0), ("spc", 0.1)):
            rows.append(
                {
                    "training_seed": seed,
                    "condition": condition,
                    "cycle_relative_rms": 1.0 + shift,
                    "local_decoder_sensitivity": 2.0 + shift,
                }
            )
    paired, decision = summarize_closure(pd.DataFrame(rows))
    assert len(paired) == 5
    assert not decision["closure_not_systematically_worse"]


def test_closure_decision_accepts_mixed_paired_effects() -> None:
    rows = []
    for seed in range(5):
        shift = -0.1 if seed < 3 else 0.1
        for condition, value in (("static", 0.0), ("spc", shift)):
            rows.append(
                {
                    "training_seed": seed,
                    "condition": condition,
                    "cycle_relative_rms": 1.0 + value,
                    "local_decoder_sensitivity": 2.0 + value,
                }
            )
    _, decision = summarize_closure(pd.DataFrame(rows))
    assert decision["closure_not_systematically_worse"]

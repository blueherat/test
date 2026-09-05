from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_ou_conditional_score_identity import (
    bridge_signal,
    default_mixtures,
    evaluate_case,
)


def test_translated_standard_gaussian_is_exact_degree1_null() -> None:
    row = evaluate_case(
        default_mixtures()[0],
        time=0.1,
        future_time=0.2,
        samples=512,
        quadrature_order=32,
        rng=np.random.default_rng(3),
    )
    assert row["conditional_identity_residual_max_abs"] < 1e-12
    assert row["degree1_fixed_coordinate_defect_rms"] < 1e-12
    assert row["degree2_fixed_coordinate_defect_rms"] > 1e-3


def test_nonlinear_mixture_obeys_conditional_score_identity() -> None:
    row = evaluate_case(
        default_mixtures()[2],
        time=0.2,
        future_time=0.3,
        samples=1024,
        quadrature_order=80,
        rng=np.random.default_rng(4),
    )
    assert row["conditional_identity_residual_rms"] < 1e-10
    assert row["degree1_fixed_coordinate_defect_rms"] > 1e-3


def test_bridge_signal_is_strictly_increasing() -> None:
    values = [bridge_signal(time) for time in (0.01, 0.1, 0.5, 0.9)]
    assert values == sorted(values)

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from experiments.frechet_residual_score_toy import (
    field_diagnostics,
    pearson_field,
    pooled_fisher_divergence,
    pooled_fisher_field,
    score_field,
)
from experiments.run_advfd_divergence_transport_audit import (
    gaussian_shift_pair,
    run,
)


def test_gaussian_shift_discrepancies_and_fields_have_expected_limits() -> None:
    target, source = gaussian_shift_pair(3.0, device=torch.device("cpu"))
    pooled = pooled_fisher_divergence(target, source, quadrature_order=64)
    assert 0.0 < pooled < 4.0

    score = field_diagnostics(target, source, score_field(target, source), quadrature_order=64)
    pearson = field_diagnostics(
        target, source, pearson_field(target, source), quadrature_order=64
    )
    pooled_field = field_diagnostics(
        target, source, pooled_fisher_field(target, source), quadrature_order=64
    )
    assert score["score_cosine"] == pytest.approx(1.0)
    assert pearson["velocity_rms"] > 1e5 * score["velocity_rms"]
    source_mean = source.moments().mean[None, :]
    typical_pooled = pooled_fisher_field(target, source)(source_mean, False)
    typical_score = score_field(target, source)(source_mean, False)
    assert float(typical_pooled.abs().max()) < 0.01 * float(
        typical_score.abs().max()
    )
    assert pooled_field["score_cosine"] < 0.5


def test_audit_writes_complete_artifacts(tmp_path) -> None:
    output = tmp_path / "audit"
    run(
        output,
        shifts=(0.5, 1.5),
        fixed_step_size=1e-5,
        matched_displacement_rms=1e-5,
        quadrature_order=32,
        device=torch.device("cpu"),
    )
    frame = pd.read_csv(output / "divergence_transport_audit.csv")
    assert len(frame) == 2 * 3 * 2
    assert set(frame["field"]) == {
        "score_reverse_kl",
        "real_fisher_pearson",
        "pooled_fisher_triangular",
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["protocol"] == "advfd_divergence_transport_audit_v1"
    assert (output / "divergence_transport_audit.png").is_file()

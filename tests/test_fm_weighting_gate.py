import numpy as np
import torch

from experiments.fm_weighting_gate import (
    GateTreatment,
    gate_weight,
    treatment_weight_diagnostics,
)
from experiments.nonlinear_fm_whitening_toy import MixtureFMConfig, residual_weight_normalizer


def _residual_variance(problem: MixtureFMConfig, count: int = 1001) -> torch.Tensor:
    t = torch.linspace(problem.t_min, problem.t_max, count, dtype=torch.float64)[:, None]
    variance = torch.tensor(problem.variance, dtype=torch.float64)[None]
    return variance / ((1.0 - t).square() * variance + t.square())


def test_direction_weight_has_unit_mean_at_every_time():
    problem = MixtureFMConfig()
    residual_variance = _residual_variance(problem)
    treatment = GateTreatment("direction", 0.65)
    normalizer = residual_weight_normalizer(problem, treatment.gamma, 1e-4)
    weight = gate_weight(
        residual_variance,
        treatment,
        global_normalizer=normalizer,
        damping=1e-4,
    )
    assert torch.allclose(weight.mean(dim=1), torch.ones(len(weight), dtype=weight.dtype))


def test_time_weight_is_constant_across_directions_and_globally_normalized():
    problem = MixtureFMConfig()
    residual_variance = _residual_variance(problem, 4097)
    treatment = GateTreatment("time", 0.5)
    normalizer = residual_weight_normalizer(problem, treatment.gamma, 1e-4)
    weight = gate_weight(
        residual_variance,
        treatment,
        global_normalizer=normalizer,
        damping=1e-4,
    )
    assert torch.allclose(weight, weight[:, :1].expand_as(weight))
    assert np.isclose(float(weight.mean()), 1.0, atol=1e-12)


def test_full_weight_is_exact_product_of_direction_and_time_terms():
    problem = MixtureFMConfig()
    residual_variance = _residual_variance(problem)
    gamma = 0.75
    normalizer = residual_weight_normalizer(problem, gamma, 1e-4)
    kwargs = dict(global_normalizer=normalizer, damping=1e-4)
    direction = gate_weight(residual_variance, GateTreatment("direction", gamma), **kwargs)
    temporal = gate_weight(residual_variance, GateTreatment("time", gamma), **kwargs)
    full = gate_weight(residual_variance, GateTreatment("full", gamma), **kwargs)
    assert torch.allclose(full, direction * temporal, atol=1e-12, rtol=1e-12)


def test_weight_diagnostic_exposes_only_the_intended_axis():
    table = treatment_weight_diagnostics(
        MixtureFMConfig(),
        [GateTreatment("direction", 0.5), GateTreatment("time", 0.5)],
    ).set_index("mode")
    assert table.loc["direction", "per_time_mean_cv"] < 1e-12
    assert table.loc["direction", "mean_within_time_cv"] > 0.0
    assert table.loc["time", "per_time_mean_cv"] > 0.0
    assert table.loc["time", "mean_within_time_cv"] < 1e-12

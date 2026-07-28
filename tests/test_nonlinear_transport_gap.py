import numpy as np
import pandas as pd
import torch

from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    ResidualMLP,
    sample_latent_reference,
)
from experiments.nonlinear_transport_gap import (
    rollout_snapshots,
    shared_state_audit,
    summarize_pair,
)


def test_rollout_snapshots_returns_exact_requested_grid_times():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.9),
        decoder_gain=(1.0, 1.0),
    )
    initial = torch.randn((32, 2), generator=torch.Generator().manual_seed(0))
    snapshots = rollout_snapshots(
        problem,
        initial,
        (0.9, 0.5, 0.1, 0.0),
        model=None,
        ode_steps=20,
        oracle=True,
    )

    assert set(snapshots) == {0.9, 0.5, 0.1, 0.0}
    assert all(value.shape == initial.shape for value in snapshots.values())
    assert torch.isfinite(torch.stack(list(snapshots.values()))).all()


def test_identical_fields_have_unit_shared_state_ratios():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.9),
        decoder_gain=(1.0, 1.0),
    )
    model = ResidualMLP(problem.dimension, hidden_size=8, depth=1)
    clone = ResidualMLP(problem.dimension, hidden_size=8, depth=1)
    clone.load_state_dict(model.state_dict())
    count = 48
    initial = torch.randn((count, 2), generator=torch.Generator().manual_seed(1))
    clean = sample_latent_reference(problem, count, seed=2)
    states = rollout_snapshots(
        problem,
        initial,
        (0.9, 0.5, 0.1, 0.0),
        model=model,
        ode_steps=20,
    )
    state, direction = shared_state_audit(
        problem,
        model,
        clone,
        clean,
        initial,
        states,
        states,
        (0.9, 0.5, 0.1),
        (0.0, 0.5, 1.0),
        device="cpu",
    )
    endpoint = pd.DataFrame(
        [
            {"variant": variant, "metric": metric, "value": value}
            for variant in ("baseline", "weighted")
            for metric, value in (
                ("mean_coordinate_w1", 0.4),
                ("covariance_rel_fro", 0.3),
                ("sliced_w1", 0.2),
            )
        ]
    )
    summary = summarize_pair(state, direction, endpoint)

    assert np.isclose(summary["teacher_field_mse_ratio"], 1.0)
    assert np.isclose(summary["middle_offpath_field_ratio"], 1.0)
    assert np.isclose(summary["middle_offpath_log_ratio_slope"], 0.0, atol=1e-10)
    assert np.isclose(summary["high_teacher_coordinate_drift_ratio"], 1.0)
    assert summary["strong_middle_crossover"] is False

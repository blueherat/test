from pathlib import Path

import pandas as pd
import torch

from experiments.run_advfd_cross_component_trajectory_audit import run


def test_cross_component_trajectory_audit_smoke(tmp_path: Path) -> None:
    output = tmp_path / "cross"
    run(
        output,
        regimes=("rotated_ring",),
        noise_sigma=0.4,
        training_modes=("real_covariance_quotient",),
        seeds=(41,),
        checkpoints=(0, 1),
        quadrature_order=8,
        sample_count=16,
        whitening_epsilon=1e-3,
        device=torch.device("cpu"),
    )

    frame = pd.read_csv(output / "cross_component_trajectory.csv")
    assert set(frame["generator_component"]) == {"mean", "covariance", "full"}
    assert set(frame["step"]) == {0, 1}
    assert (output / "final_correctability.csv").is_file()
    assert (output / "summary.json").is_file()

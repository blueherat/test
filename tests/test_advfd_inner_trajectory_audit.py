from pathlib import Path

import pandas as pd
import torch

from experiments.run_advfd_inner_trajectory_audit import run


def test_inner_trajectory_audit_smoke(tmp_path: Path) -> None:
    output = tmp_path / "trajectory"
    run(
        output,
        regimes=("rotated_ring",),
        noise_sigma=0.4,
        modes=("real_stopgrad", "real_quotient"),
        seeds=(31,),
        checkpoints=(0, 1, 2),
        quadrature_order=8,
        sample_count=32,
        displacement_rms_values=(0.0001, 0.001),
        whitening_epsilon=1e-3,
        device=torch.device("cpu"),
    )

    frame = pd.read_csv(output / "inner_trajectory.csv")
    assert set(frame["mode"]) == {"real_stopgrad", "real_quotient"}
    assert set(frame["step"]) == {0, 1, 2}
    assert set(frame["displacement_rms"]) == {0.0001, 0.001}
    assert frame["kl_descent_per_velocity_rms"].notna().all()
    assert (output / "peak_correctability.csv").is_file()
    assert (output / "summary.json").is_file()


def test_inner_trajectory_audit_accepts_component_objectives(
    tmp_path: Path,
) -> None:
    output = tmp_path / "components"
    modes = (
        "real_mean_quotient",
        "real_covariance_quotient",
        "pooled_mean_quotient",
        "pooled_covariance_quotient",
    )
    run(
        output,
        regimes=("shape_only",),
        noise_sigma=0.4,
        modes=modes,
        seeds=(37,),
        checkpoints=(0, 1),
        quadrature_order=8,
        sample_count=16,
        displacement_rms_values=(0.001,),
        whitening_epsilon=1e-3,
        device=torch.device("cpu"),
    )

    frame = pd.read_csv(output / "inner_trajectory.csv")
    assert set(frame["mode"]) == set(modes)

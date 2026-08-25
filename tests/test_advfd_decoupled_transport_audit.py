from pathlib import Path

import pandas as pd
import torch

from experiments.run_advfd_decoupled_transport_audit import run


def test_decoupled_transport_audit_smoke(tmp_path: Path) -> None:
    output = tmp_path / "transport"
    run(
        output,
        regimes=("rotated_ring",),
        noise_sigma=0.4,
        protocols=("score", "full_quotient_full", "full_quotient_mean"),
        seeds=(43,),
        rounds=1,
        displacement_rms=0.001,
        critic_steps=1,
        quadrature_order=8,
        whitening_epsilon=1e-3,
        device=torch.device("cpu"),
    )

    frame = pd.read_csv(output / "transport_curves.csv")
    assert set(frame["protocol"]) == {
        "score",
        "full_quotient_full",
        "full_quotient_mean",
    }
    assert set(frame["round"]) == {0, 1}
    assert (output / "aggregate.csv").is_file()
    assert (output / "summary.json").is_file()

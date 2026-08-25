from pathlib import Path

import pandas as pd
import torch

from experiments.run_advfd_monoflow_audit import build_regime
from experiments.run_advfd_quotient_gradient_audit import (
    affine_gauge_diagnostic,
    run,
)


def test_quotient_gradient_removes_common_translation_gauge() -> None:
    device = torch.device("cpu")
    target, source = build_regime("rotated_ring", device=device)
    rows = affine_gauge_diagnostic(
        target,
        source,
        feature_dim=4,
        order=10,
        whitening_epsilon=1e-3,
        seed=17,
        device=device,
    )
    by_mode = {row["mode"]: row for row in rows}

    assert abs(by_mode["real_stopgrad"]["finite_translation_change"]) < 1e-8
    assert abs(by_mode["pooled_stopgrad"]["finite_translation_change"]) < 1e-8
    assert by_mode["real_stopgrad"]["translation_gradient_norm"] > 1e-4
    assert by_mode["pooled_stopgrad"]["translation_gradient_norm"] > 1e-4
    assert by_mode["real_quotient"]["translation_gradient_norm"] < 1e-8
    assert by_mode["pooled_quotient"]["translation_gradient_norm"] < 1e-8


def test_quotient_audit_smoke(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    run(
        output,
        regimes=("rotated_ring",),
        noise_sigmas=(0.4,),
        modes=("real_stopgrad", "real_quotient"),
        seeds=(23,),
        critic_steps=2,
        quadrature_order=8,
        sample_count=32,
        displacement_rms=0.001,
        whitening_epsilon=1e-3,
        device=torch.device("cpu"),
    )

    learned = pd.read_csv(output / "learned_critic_audit.csv")
    gauges = pd.read_csv(output / "affine_gauge_gradients.csv")
    assert set(learned["mode"]) == {"real_stopgrad", "real_quotient"}
    assert set(gauges["mode"]) == {
        "real_stopgrad",
        "real_quotient",
        "pooled_stopgrad",
        "pooled_quotient",
    }
    assert (output / "summary.json").is_file()

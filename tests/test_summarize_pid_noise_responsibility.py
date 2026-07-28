import json

import pandas as pd
import pytest

from experiments.summarize_pid_noise_responsibility import (
    load_screen,
    model_gate_table,
    profile_table,
)


def _write_model(root, model: str, values: tuple[float, float]) -> None:
    model_root = root / model
    model_root.mkdir()
    rows = []
    for mode in ("teacher_forced", "real_rollout"):
        for seed in (0, 1):
            for sample in (0, 1):
                for timestep, value in zip((0.999, 0.866), values):
                    rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "sample_index": sample,
                            "timestep": timestep,
                            "delta_shuffle": value,
                        }
                    )
    pd.DataFrame(rows).to_csv(model_root / "paired_rows.csv", index=False)
    pd.DataFrame(
        [{"absolute_rms_max": 0.0, "relative_rms_max": 0.0}]
    ).to_csv(model_root / "identity_controls.csv", index=False)
    pd.DataFrame(
        [{"metric": "loss_real", "max_absolute_difference": 0.0}]
    ).to_csv(model_root / "batch_order_controls.csv", index=False)
    (model_root / "provenance.json").write_text(
        json.dumps({"checkpoint_sha256": f"sha-{model}"})
    )


def test_summary_preserves_curve_shape_and_checks_controls(tmp_path):
    _write_model(tmp_path, "semantic", (0.4, 0.02))
    _write_model(tmp_path, "vae", (0.4, 0.2))
    paired, controls = load_screen(tmp_path, ["semantic", "vae"])
    profile = profile_table(paired)
    gates = model_gate_table(paired, controls)

    assert controls["controls_exact"].all()
    assert len(profile) == 8
    first = profile[profile["timestep"] == 0.999]
    assert first["delta_relative_to_first"].eq(1.0).all()
    teacher = gates[gates["mode"] == "teacher_forced"].set_index("model")
    assert teacher.loc["semantic", "second_to_first_ratio"] == pytest.approx(0.05)
    assert teacher.loc["vae", "second_to_first_ratio"] == pytest.approx(0.5)
    assert gates["automatic_gate_pass"].all()


def test_negative_seed_mean_fails_model_gate(tmp_path):
    _write_model(tmp_path, "bad", (-0.1, 0.01))
    paired, controls = load_screen(tmp_path, ["bad"])
    gates = model_gate_table(paired, controls)
    assert not gates["automatic_gate_pass"].any()

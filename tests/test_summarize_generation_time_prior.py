import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.summarize_generation_time_prior import gate_table


def test_gate_rejects_high_noise_when_images_are_worse(tmp_path):
    rows = []
    for mode, fid, swd in (("all_time", 25.0, 0.08), ("high_noise", 35.0, 0.076)):
        stage2 = tmp_path / "stage2"
        stage2.mkdir(exist_ok=True)
        stage2_mode = stage2 / f"{mode}_seed0_date"
        stage2_mode.mkdir(exist_ok=True)
        pd.DataFrame(
            [{"branch": "real", "feature_fid": 20.0 if mode == "all_time" else 30.0}]
        ).to_csv(stage2_mode / "rollout.csv", index=False)
        run = stage2 / "none_seed0_date"
        run.mkdir(exist_ok=True)
        pd.DataFrame(
            [{"branch": "real", "feature_fid": 150.0}]
        ).to_csv(run / "rollout.csv", index=False)
        rows.append(
            {
                "seed": 0,
                "mode": mode,
                "stage2_run": str(stage2 / f"{mode}_seed0_date"),
                "image_feature_fid": fid,
                "latent_swd": swd,
                "latent_class_entropy": 2.3,
                "image_class_entropy": 2.3,
            }
        )
    gates = gate_table(pd.DataFrame(rows))
    assert gates.loc[0, "two_stage_pass"]
    assert not gates.loc[0, "quality_guardrail"]
    assert not gates.loc[0, "method_value"]
    assert not gates.loc[0, "end_to_end_gate_pass"]
    assert gates.loc[0, "all_prior_fid_increment"] == 5.0
    assert gates.loc[0, "high_prior_fid_increment"] == 5.0


def test_summary_script_entrypoint_loads_repo_modules():
    script = Path(__file__).resolve().parents[1] / "experiments/summarize_generation_time_prior.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

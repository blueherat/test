import json

import pandas as pd

from experiments.summarize_generation_time_bottleneck import load_runs, seed_gate_table


def _write_run(root, mode, seed, real, shuffled, classifier=0.99):
    run = root / f"{mode}_seed{seed}_date"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"seed": seed}))
    times = (0.99, 0.85, 0.55, 0.3)
    if mode == "high_noise":
        deltas = (1.0, 0.8, 0.0, 0.0)
    elif mode == "none":
        deltas = (0.0, 0.0, 0.0, 0.0)
    else:
        deltas = (0.2, 0.2, 0.1, 0.1)
    pd.DataFrame(
        {"mode": mode, "timestep": times, "delta_shuffle_mean": deltas}
    ).to_csv(run / "teacher_profile.csv", index=False)
    pd.DataFrame(
        [
            {"mode": mode, "branch": "real", "source_class_match": real},
            {"mode": mode, "branch": "shuffle", "source_class_match": shuffled},
        ]
    ).to_csv(run / "rollout.csv", index=False)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "classifier_accuracy": classifier,
                "identity_absolute_max": 0.0,
                "identity_relative_max": 0.0,
            }
        )
    )


def test_gate_accepts_expected_two_seed_pattern(tmp_path):
    for seed in (0, 1):
        root = tmp_path / f"seed{seed}"
        _write_run(root, "all_time", seed, 0.82, 0.10)
        _write_run(root, "high_noise", seed, 0.80, 0.10)
        _write_run(root, "low_noise", seed, 0.15, 0.10)
        _write_run(root, "none", seed, 0.10, 0.10)
    tables = load_runs([tmp_path / "seed0", tmp_path / "seed1"])
    gates = seed_gate_table(tables)
    assert len(gates) == 2
    assert gates["prior_gate_pass"].all()
    assert (gates["high_over_all"] > 0.9).all()

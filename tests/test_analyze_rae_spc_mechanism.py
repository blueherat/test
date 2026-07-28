from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_spc_mechanism import (
    attach_generation_metrics,
    build_gradient_relation_table,
    build_loss_pairs,
    build_seed_mechanism_table,
    summarize_mechanism,
)


def _gradient_rows() -> pd.DataFrame:
    rows = []
    for seed in (1, 2, 3, 4, 5):
        for condition in ("static", "spc"):
            for step in (2000, 5000):
                for time in (0.3, 0.1):
                    if condition == "static":
                        semantic, basis = 1.0, 1.0
                    elif step == 2000:
                        semantic, basis = 0.8, 1.5
                    else:
                        semantic, basis = 0.9, 1.1
                    for group in ("last_block", "output_head"):
                        rows.append(
                            {
                                "training_seed": seed,
                                "condition": condition,
                                "checkpoint_step": step,
                                "time": time,
                                "parameter_group": group,
                                "semantic_loss": semantic,
                                "basis_loss": basis,
                                "basis_over_semantic_norm": 0.5,
                                "semantic_basis_cosine": 0.1,
                                "semantic_descent_ratio": 1.0,
                            }
                        )
    return pd.DataFrame(rows)


def test_mechanism_predictions_pass_for_constructed_pattern() -> None:
    gradient = _gradient_rows()
    per_time, mean_time = build_loss_pairs(gradient)
    seeds = build_seed_mechanism_table(mean_time)
    relations = build_gradient_relation_table(gradient)
    summary = summarize_mechanism(seeds, relations)

    assert len(per_time) == 20
    assert seeds["p1_capacity_reallocation"].all()
    assert seeds["p2_basis_catchup_30pct"].all()
    assert summary["mechanism_gate_pass"] is True


def test_generation_metrics_are_paired_by_seed() -> None:
    _, mean_time = build_loss_pairs(_gradient_rows())
    seeds = build_seed_mechanism_table(mean_time)
    metrics = pd.DataFrame(
        [
            {
                "seed": seed,
                "condition": condition,
                "frechet_inception_distance": seed + (1 if condition == "static" else 0),
            }
            for seed in reversed((1, 2, 3, 4, 5))
            for condition in ("spc", "static")
        ]
    )
    merged = attach_generation_metrics(seeds, metrics)
    assert (merged["delta_fid"] == -1).all()

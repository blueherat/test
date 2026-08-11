from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.summarize_internal_guidance_multiseed import (
    aggregate,
    consistency_table,
    paired_deltas,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for seed, baseline, guided in ((1, 0.2, 0.21), (2, 0.3, 0.29)):
        for condition, swd, bridge, contrast in (
            ("ig_w1", baseline, 0.16, 2.0),
            ("ig_w2.3_mid03_07", guided, 0.08, 4.0),
        ):
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "latent_swd": swd,
                    "pushforward_swd": swd,
                    "intrinsic_bridge_rate": bridge,
                    "mean_adjacent_log_density_contrast": contrast,
                    "component_jsd_y": 0.002,
                    "occupied_components": 32,
                }
            )
    return pd.DataFrame(rows)


def test_paired_deltas_use_same_seed_baseline() -> None:
    deltas = paired_deltas(sample_frame())
    assert np.allclose(deltas.latent_swd_delta, [0.01, -0.01])
    assert (deltas.intrinsic_bridge_rate_delta < 0).all()
    assert (deltas.mean_adjacent_log_density_contrast_delta > 0).all()


def test_aggregate_and_consistency_count_seeds() -> None:
    frame = sample_frame()
    summary = aggregate(frame)
    assert set(summary.seeds) == {2}
    consistency = consistency_table(paired_deltas(frame)).iloc[0]
    assert consistency.bridge_improved_seeds == 2
    assert consistency.contrast_improved_seeds == 2
    assert consistency.latent_swd_improved_seeds == 1
    assert consistency.latent_swd_within_10pct_seeds == 2
    assert consistency.all_modes_retained_seeds == 2

import math

import pandas as pd

from experiments.small_image_seed_factorial import (
    factorial_effects,
    summarize_conditions,
)


def _factorial_rows():
    rows = []
    for data_seed in (3, 4):
        for init_seed in (3, 4):
            for stream_seed in (3, 4):
                log_ratio = (
                    0.2 * (-1 if data_seed == 3 else 1)
                    + 0.4 * (-1 if init_seed == 3 else 1)
                    - 0.1 * (-1 if stream_seed == 3 else 1)
                )
                for rollout_seed in (11, 12):
                    ratio = math.exp(log_ratio)
                    rows.append(
                        {
                            "data_seed": data_seed,
                            "init_seed": init_seed,
                            "stream_seed": stream_seed,
                            "rollout_seed": rollout_seed,
                            "metric": "feature_fid",
                            "baseline": 2.0,
                            "weighted": 2.0 * ratio,
                            "ratio": ratio,
                            "delta": 2.0 * (ratio - 1.0),
                        }
                    )
    return pd.DataFrame(rows)


def test_factorial_effects_recover_log_ratio_contrasts():
    conditions = summarize_conditions(_factorial_rows())
    effects = factorial_effects(conditions).set_index("term")
    assert math.isclose(effects.loc["data", "log_ratio_effect"], 0.4)
    assert math.isclose(effects.loc["init", "log_ratio_effect"], 0.8)
    assert math.isclose(effects.loc["stream", "log_ratio_effect"], -0.2)
    assert abs(effects.loc["data:init", "log_ratio_effect"]) < 1e-12

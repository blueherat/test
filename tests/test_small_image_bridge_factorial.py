import math

import pandas as pd

from experiments.small_image_bridge_factorial import bridge_factorial_effects


def test_bridge_factorial_effects_recover_known_contrasts():
    rows = []
    for noise_seed in (3, 4):
        for time_seed in (3, 4):
            noise_code = -1 if noise_seed == 3 else 1
            time_code = -1 if time_seed == 3 else 1
            log_ratio = -0.4 * noise_code + 0.1 * time_code
            rows.append(
                {
                    "noise_seed": noise_seed,
                    "time_seed": time_seed,
                    "metric": "feature_fid",
                    "ratio_mean": math.exp(log_ratio),
                }
            )
    effects = bridge_factorial_effects(pd.DataFrame(rows)).set_index("term")
    assert math.isclose(effects.loc["noise", "log_ratio_effect"], -0.8)
    assert math.isclose(effects.loc["time", "log_ratio_effect"], 0.2)
    assert abs(effects.loc["noise:time", "log_ratio_effect"]) < 1e-12

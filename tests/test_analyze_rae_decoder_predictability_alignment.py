from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_rae_decoder_predictability_alignment import (
    standardized_regression,
)


def test_regression_recovers_predictability_after_variance_control() -> None:
    variance = np.linspace(0.5, 3.0, 24)
    predictability = np.tile(np.linspace(0.1, 0.9, 8), 3)
    target = np.exp(0.4 * np.log(variance) + 1.2 * predictability)
    table = pd.DataFrame(
        {
            "val_final_variance_per_dimension": variance,
            "val_r2": predictability,
            "decoder_gain": target,
        }
    )
    result = standardized_regression(table, "decoder_gain")
    assert result["predictability_beta"] > 0.0
    assert result["variance_beta"] > 0.0
    assert result["combined_r2"] > 0.999

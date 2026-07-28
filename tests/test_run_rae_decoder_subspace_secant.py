from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_predictability_gain import MATCHED_PAIRS
from experiments.run_rae_decoder_subspace_secant import pair_ratios, summarize


def test_decoder_subspace_summary_tracks_all_matched_pairs() -> None:
    rows = []
    metrics = (
        "decoder_embed_gain",
        "decoder_pixel_mse_gain",
        "decoder_l1_secant",
        "decoder_lpips_secant",
        "decoder_hidden_1_mse_gain",
        "decoder_hidden_4_mse_gain",
    )
    bases = {basis for pair in MATCHED_PAIRS for basis in pair}
    higher = {pair[0] for pair in MATCHED_PAIRS}
    for basis in bases:
        for sign in (-1.0, 1.0):
            scale = 2.0 if basis in higher else 1.0
            row = {"basis": basis, "fraction": 0.01, "sign": sign}
            row.update({metric: scale for metric in metrics})
            rows.append(row)
    ratios = pair_ratios(pd.DataFrame(rows))
    result = summarize(ratios, count=8)
    assert result["all_three_pairs"] is True
    assert result["higher_predictability_more_sensitive_l1_and_lpips_pairs"] == 3

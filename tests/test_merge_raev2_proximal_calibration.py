import numpy as np
import pandas as pd
import pytest

from experiments.merge_raev2_proximal_calibration import analyze


def sample_frame():
    return pd.DataFrame([
        dict(step_index=k, sample_id=i, source_row=100+i, label=i,
             risk_before=4.+i, risk_after=3.+i, risk_gain=1.,
             J=.5, nonexpansive_slack=.5)
        for k in range(3) for i in range(6)
    ])


def test_paired_whole_image_summary_and_simultaneous_intervals():
    steps, summary = analyze(sample_frame(), 3)
    assert summary["step_gain_bonferroni_positive_count"] == 3
    assert summary["all_step_mean_risk_nonincreasing"]
    expected = 3*(np.sqrt(6.5)-np.sqrt(5.5))
    assert summary["unweighted_local_rms_source_sum"]["gain"] == pytest.approx(expected)
    assert np.all(steps.risk_gain_simultaneous_lower == 1.)


@pytest.mark.parametrize("mutation", ["missing_step", "duplicate", "changed_source", "changed_label"])
def test_reject_incomplete_or_mispaired_evidence(mutation):
    frame = sample_frame()
    if mutation == "missing_step":
        frame = frame[frame.step_index != 2]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "changed_source":
        frame.loc[0, "source_row"] = 999
    else:
        frame.loc[0, "label"] = 999
    with pytest.raises(ValueError):
        analyze(frame, 3)

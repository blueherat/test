import numpy as np
import pandas as pd
import pytest

from experiments.analyze_raev2_proximal_global_calibration import (
    FAMILY_SIZE, analyze, confidence,
)


def example():
    rows, metadata = [], []
    for step, (prediction_scale, lam) in enumerate(((1.2, .95), (.8, 1.0), (1.2, .95))):
        metadata.append(dict(step_index=step, effective_lambda=lam, active=lam < 1))
        a = 1 / lam - 1
        for sample, y in enumerate((-1., 1., -1., 1.)):
            prediction = prediction_scale * y
            residual = prediction - y
            corrected = prediction * lam
            before, after = residual ** 2, (corrected - y) ** 2
            j = 2 * a * y * residual - (a * y) ** 2
            second_difference = 2 * residual * y + before
            rows.append(dict(step_index=step, sample_id=sample, source_row=100 + sample, label=sample,
                risk_before=before, risk_after=after, risk_gain=before - after, J=j,
                nonexpansive_slack=before - after - j, h_target_mse=(a * y) ** 2,
                applied_correction_mse=(corrected - prediction) ** 2,
                target_second_moment=y ** 2, predicted_second_moment=prediction ** 2,
                residual_cross_moment=residual * y,
                w2_nonincrease_certificate=((1 + lam) ** 2 - 4) * y ** 2 + (1 + lam) ** 2 * second_difference,
                w2_nonovershoot_certificate=(lam ** 2 - 1) * y ** 2 + lam ** 2 * second_difference,
                diagonal_risk_after=before + .01, diagonal_risk_gain=-.01, diagonal_J=-.02))
    return pd.DataFrame(rows), pd.DataFrame(metadata)


def test_primary_admission_keeps_inactive_negative_certificates_as_identity():
    frame, metadata = example()
    steps, summary = analyze(frame, metadata, expected_steps=3, expected_samples=4)
    assert summary["eligible_for_fixed_1k_test"]
    assert summary["active_step_count"] == 2
    assert summary["statistical_protocol"]["bonferroni_family_size"] == FAMILY_SIZE == 100
    inactive = steps[~steps.active].iloc[0]
    assert inactive.w2_nonincrease_certificate_mean < 0
    assert inactive.identity_verified
    assert inactive.primary_certificate_pass
    assert summary["original_diagonal_secondary"]["negative_D_simultaneous_upper_count"] == 3
    assert not summary["original_diagonal_secondary"]["used_for_selection"]


@pytest.mark.parametrize("mutation", ["missing_step", "duplicate", "source", "label", "inactive_changed", "lambda"])
def test_reject_broken_pairing_or_identity(mutation):
    frame, metadata = example()
    if mutation == "missing_step":
        frame = frame[frame.step_index != 2]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "source":
        frame.loc[0, "source_row"] = 999
    elif mutation == "label":
        frame.loc[0, "label"] = 2
    elif mutation == "inactive_changed":
        frame.loc[frame.step_index == 1, "risk_gain"] = 1e-30
    elif mutation == "lambda":
        metadata.loc[0, "effective_lambda"] = .96
    with pytest.raises(ValueError):
        analyze(frame, metadata, expected_steps=3, expected_samples=4)


def test_an_uncertain_active_step_is_never_dropped_or_replaced_by_diagonal():
    frame, metadata = example()
    # Inflate image-to-image uncertainty while keeping the saved formula true.
    mask = (frame.step_index == 2) & (frame.sample_id == 0)
    frame.loc[mask, "residual_cross_moment"] -= 2
    lam = metadata.loc[2, "effective_lambda"]
    difference = 2 * frame.loc[mask, "residual_cross_moment"] + frame.loc[mask, "risk_before"]
    b = frame.loc[mask, "target_second_moment"]
    frame.loc[mask, "w2_nonincrease_certificate"] = ((1 + lam) ** 2 - 4) * b + (1 + lam) ** 2 * difference
    frame.loc[mask, "w2_nonovershoot_certificate"] = (lam ** 2 - 1) * b + lam ** 2 * difference
    steps, summary = analyze(frame, metadata, expected_steps=3, expected_samples=4)
    assert len(steps) == 3
    assert not summary["eligible_for_fixed_1k_test"]
    assert summary["primary_certificate"]["unconfirmed_active_steps"] == [2]
    assert not summary["time_window_selected"]
    assert not summary["coefficients_changed"]


def test_standard_error_uses_images_not_coordinates_or_times():
    values = np.array([1., 2., 4., 8.])
    result = confidence(values, 3.48)
    assert result["se"] == pytest.approx(values.std(ddof=1) / 2)


import pandas as pd
import pytest

from experiments.run_pag_cifar10_direction_audit import (
    calibration_policy,
    parse_float_list,
    parse_int_list,
    validate_protocol,
)


def test_list_parsers() -> None:
    assert parse_int_list("1, 3,5") == (1, 3, 5)
    assert parse_float_list("0, 1.0,1.5") == (0.0, 1.0, 1.5)


def test_protocol_rejects_selection_leakage_and_invalid_schedule() -> None:
    validate_protocol(
        samples=8,
        calibration_samples=4,
        batch_size=2,
        timesteps=(100, 500),
        scales=(0.0, 1.0, 2.0),
        train_timesteps=1000,
    )
    with pytest.raises(ValueError):
        validate_protocol(
            samples=8,
            calibration_samples=8,
            batch_size=2,
            timesteps=(100, 500),
            scales=(0.0, 1.0, 2.0),
            train_timesteps=1000,
        )
    with pytest.raises(ValueError):
        validate_protocol(
            samples=8,
            calibration_samples=4,
            batch_size=2,
            timesteps=(1000,),
            scales=(0.0, 1.0, 2.0),
            train_timesteps=1000,
        )


def test_calibration_policy_selects_without_using_evaluation_gain() -> None:
    rows = []
    for split in ("calibration", "evaluation"):
        for layer, calibration_gain, evaluation_gain in (
            ("early", 0.2, -0.1),
            ("late", 0.1, 0.4),
        ):
            gain = calibration_gain if split == "calibration" else evaluation_gain
            for dataset_index in range(3):
                rows.append(
                    {
                        "split": split,
                        "dataset_index": dataset_index,
                        "timestep": 500,
                        "layer": layer,
                        "scale_s": 2.0,
                        "pag_gamma": 1.0,
                        "gain_over_full": gain,
                    }
                )
    policy = calibration_policy(pd.DataFrame(rows))
    assert policy.loc[0, "layer"] == "early"
    assert policy.loc[0, "calibration_gain_mean"] == pytest.approx(0.2)
    assert policy.loc[0, "evaluation_gain_mean"] == pytest.approx(-0.1)


def test_positive_extrapolation_policy_excludes_interpolation() -> None:
    frame = pd.DataFrame(
        [
            {
                "split": split,
                "dataset_index": index,
                "timestep": 500,
                "layer": "mid",
                "scale_s": scale,
                "pag_gamma": scale - 1.0,
                "gain_over_full": gain,
            }
            for split in ("calibration", "evaluation")
            for index in range(3)
            for scale, gain in ((0.5, 0.5), (1.5, 0.1))
        ]
    )
    policy = calibration_policy(frame, family="positive_extrapolation")
    assert policy.loc[0, "policy_family"] == "positive_extrapolation"
    assert policy.loc[0, "scale_s"] == pytest.approx(1.5)

import numpy as np
import torch

from experiments.noise_responsibility_profile import (
    ResponsibilityBatch,
    aggregate_profile,
    derangement,
    identity_control_error,
    radial_frequency_mse,
    responsibility_rows,
)


def test_derangement_is_deterministic_and_has_no_fixed_points():
    first = derangement(32, seed=7)
    second = derangement(32, seed=7)
    torch.testing.assert_close(first, second)
    assert torch.all(first != torch.arange(32))
    assert torch.equal(torch.sort(first).values, torch.arange(32))


def test_paired_responsibility_has_expected_sign_and_aggregation():
    target = torch.zeros((6, 2, 4, 4))
    real = torch.zeros_like(target)
    null = torch.ones_like(target)
    shuffle = torch.full_like(target, 0.5)
    timestep = torch.tensor([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
    rows = responsibility_rows(
        ResponsibilityBatch(
            timestep=timestep,
            target=target,
            predictions={"real": real, "null": null, "shuffle": shuffle},
        )
    )
    np.testing.assert_allclose(rows.delta_null, 1.0)
    np.testing.assert_allclose(rows.delta_shuffle, 0.25)
    np.testing.assert_allclose(rows.gain_null, 1.0)
    np.testing.assert_allclose(rows.gain_shuffle, 1.0)
    profile = aggregate_profile(rows)
    assert profile["count"].tolist() == [3, 3]
    np.testing.assert_allclose(profile.delta_shuffle_mean, 0.25)
    np.testing.assert_allclose(profile.delta_shuffle_positive_rate, 1.0)


def test_identity_control_is_zero_for_repeated_prediction():
    value = torch.randn((4, 3, 8, 8), generator=torch.Generator().manual_seed(8))
    metrics = identity_control_error(value, value.clone())
    assert metrics == {"absolute_rms_max": 0.0, "relative_rms_max": 0.0}


def test_frequency_mse_separates_constant_and_checkerboard_errors():
    target = torch.zeros((1, 1, 16, 16))
    constant = torch.ones_like(target)
    grid_y, grid_x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    checkerboard = ((grid_y + grid_x) % 2).mul(2).sub(1).float()[None, None]
    constant_bands = radial_frequency_mse(constant, target)
    checkerboard_bands = radial_frequency_mse(checkerboard, target)
    assert constant_bands[0, 0] > 100 * constant_bands[0, -1]
    assert checkerboard_bands[0, -1] > 100 * checkerboard_bands[0, 0]


def test_batch_validation_rejects_missing_or_mismatched_branches():
    target = torch.zeros((2, 1, 4, 4))
    batch = ResponsibilityBatch(
        timestep=torch.zeros(2),
        target=target,
        predictions={"real": target, "null": target},
    )
    try:
        batch.validate()
    except ValueError as error:
        assert "shuffle" in str(error)
    else:
        raise AssertionError("missing shuffle branch was accepted")

import torch

from experiments.run_raev2_internal_guidance_audit import (
    internal_guidance_prediction,
    relative_rms,
)


def test_internal_guidance_prediction_endpoints_and_extrapolation() -> None:
    full = torch.tensor([[3.0, 5.0]])
    base = torch.tensor([[1.0, 1.0]])

    torch.testing.assert_close(internal_guidance_prediction(full, base, 0.0), base)
    torch.testing.assert_close(internal_guidance_prediction(full, base, 1.0), full)
    torch.testing.assert_close(
        internal_guidance_prediction(full, base, 2.0),
        torch.tensor([[5.0, 9.0]]),
    )


def test_internal_guidance_prediction_rejects_shape_mismatch() -> None:
    try:
        internal_guidance_prediction(torch.zeros(1, 2), torch.zeros(1, 3), 1.0)
    except ValueError as error:
        assert "identical shapes" in str(error)
    else:
        raise AssertionError("shape mismatch must raise ValueError")


def test_relative_rms_is_scale_free() -> None:
    reference = torch.tensor([[3.0, 4.0]])
    error = 0.5 * reference
    torch.testing.assert_close(relative_rms(error, reference), torch.tensor([0.5]))

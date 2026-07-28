from __future__ import annotations

import torch

from experiments.raev2_training_core import (
    branch_epoch,
    infer_source_steps_per_epoch,
    official_flow_loss_map,
    predicted_clean_latent,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)


class XPredictionTransport:
    def compute_loss(self, output, target_velocity, noisy_latent, time):
        scale = time.reshape((time.shape[0],) + (1,) * (noisy_latent.ndim - 1))
        predicted_velocity = (noisy_latent - output) / scale.clamp_min(0.05)
        return (predicted_velocity - target_velocity).square()


def test_x_prediction_is_already_the_clean_latent() -> None:
    noisy = torch.randn(2, 3, 4, 4)
    output = torch.randn_like(noisy)
    time = torch.tensor([0.2, 0.8])
    actual = predicted_clean_latent(
        output,
        prediction="x",
        noisy_latent=noisy,
        time=time,
    )
    assert actual is output


def test_velocity_prediction_conversion_remains_available_for_audit() -> None:
    clean = torch.randn(2, 3, 4, 4)
    velocity = torch.randn_like(clean)
    time = torch.tensor([0.2, 0.8])
    scale = time.reshape(2, 1, 1, 1)
    noisy = clean + scale * velocity
    actual = predicted_clean_latent(
        velocity,
        prediction="velocity",
        noisy_latent=noisy,
        time=time,
    )
    torch.testing.assert_close(actual, clean)


def test_official_dual_output_loss_keeps_primary_and_base_terms() -> None:
    transport = XPredictionTransport()
    clean = torch.randn(2, 3, 4, 4)
    time = torch.tensor([0.2, 0.8])
    noise = torch.randn_like(clean)
    scale = time.reshape(2, 1, 1, 1)
    noisy = (1 - scale) * clean + scale * noise
    target_velocity = (noisy - clean) / scale.clamp_min(0.05)
    primary = clean.clone()
    base = clean + 0.1

    total, details = official_flow_loss_map(
        transport,
        (primary, base),
        target_velocity=target_velocity,
        noisy_latent=noisy,
        time=time,
        base_model_coeff=0.25,
    )
    expected = transport.compute_loss(
        primary, target_velocity, noisy, time
    ) + 0.25 * transport.compute_loss(base, target_velocity, noisy, time)
    torch.testing.assert_close(total, expected)
    assert set(details) == {"primary", "base"}
    split_primary, split_base = split_internal_guidance_output((primary, base))
    assert split_primary is primary
    assert split_base is base


def test_source_scheduler_epoch_length_is_inferred_from_raev2_checkpoint() -> None:
    assert infer_source_steps_per_epoch(100_080, 80) == 1_251
    assert branch_epoch(80, 0, 1_251) == 80
    assert branch_epoch(80, 1_250, 1_251) == 80
    assert branch_epoch(80, 1_251, 1_251) == 81


def test_full_checkpoint_validation_rejects_model_only_payload() -> None:
    complete = {
        "step": 100_080,
        "epoch": 80,
        "model": {},
        "ema": {},
        "optimizer": {},
        "scheduler": {},
    }
    validate_full_stage2_checkpoint(complete)

    incomplete = dict(complete)
    incomplete.pop("optimizer")
    try:
        validate_full_stage2_checkpoint(incomplete)
    except ValueError as error:
        assert "optimizer" in str(error)
    else:
        raise AssertionError("model-only checkpoint was accepted")

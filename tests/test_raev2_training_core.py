from __future__ import annotations

import torch

from experiments.raev2_training_core import (
    branch_epoch,
    infer_source_steps_per_epoch,
    official_flow_loss_map,
    predicted_clean_latent,
    split_internal_guidance_output,
    synchronize_loaded_gmuon_param_groups,
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


def test_loaded_gmuon_param_group_aliases_are_repaired_before_resave() -> None:
    class FakeMuon:
        def __init__(self) -> None:
            self._muon_param_groups = [{"lr": 2e-4, "params": ["weight"]}]
            self._combined_param_groups = [{"lr": 2e-4, "params": ["weight"]}]
            self.scalar_optimizer = None
            self.state = {"weight": {"momentum_buffer": torch.ones(1)}}

        @property
        def param_groups(self):
            return self._combined_param_groups

        def state_dict(self):
            return {
                "state": {},
                "param_groups": [
                    {key: value for key, value in group.items()}
                    for group in self.param_groups
                ],
            }

    class FakeAdamW:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 2e-5, "params": ["bias"]}]
            self.state = {"bias": {"step": torch.tensor(1.0)}}

        def state_dict(self):
            return {
                "state": {},
                "param_groups": [
                    {key: value for key, value in group.items()}
                    for group in self.param_groups
                ],
            }

    class FakeComposite:
        def __init__(self) -> None:
            self._muon = FakeMuon()
            self._adamw = FakeAdamW()
            self.param_groups = (
                self._muon.param_groups + self._adamw.param_groups
            )

        def state_dict(self):
            return {
                "muon": self._muon.state_dict(),
                "adamw": self._adamw.state_dict(),
            }

    optimizer = FakeComposite()
    loaded_state = {
        "muon": {
            "state": {0: {"momentum_buffer": torch.ones(1)}},
            "param_groups": [{"lr": 2e-5, "params": [0]}],
        },
        "adamw": {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"lr": 2e-5, "params": [0]}],
        },
    }
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 2e-5]

    audit = synchronize_loaded_gmuon_param_groups(optimizer, loaded_state)

    assert audit == {
        "composite_gmuon": True,
        "aliases_repaired": True,
        "learning_rates": [2e-5, 2e-5],
        "resaved_learning_rates": [2e-5, 2e-5],
        "muon_state_entries": 1,
        "adamw_state_entries": 1,
    }
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-5]
    saved = optimizer.state_dict()
    assert saved["muon"]["param_groups"][0]["lr"] == 2e-5
    assert saved["adamw"]["param_groups"][0]["lr"] == 2e-5

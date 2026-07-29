from __future__ import annotations

import io
import json

import torch
import numpy as np
from PIL import Image

from experiments.raev2_training_core import (
    DeterministicImageNetPacked,
    PACKED_IMAGENET_FORMAT,
    branch_epoch,
    infer_source_steps_per_epoch,
    load_permutation_index,
    official_flow_loss_map,
    predicted_clean_latent,
    split_internal_guidance_output,
    summarize_lpl_calibration,
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


def test_lpl_calibration_includes_zero_contribution_outside_gate() -> None:
    summary = summarize_lpl_calibration(
        flow_sum=8.0,
        flow_count=8,
        active_lpl_sum=200.0,
        active_lpl_count=2,
        target_lpl_over_flow=0.2,
    )

    assert summary["flow_mean"] == 1.0
    assert summary["conditional_lpl_mean"] == 100.0
    assert summary["global_gated_lpl_mean"] == 25.0
    assert summary["gate_rate"] == 0.25
    assert summary["recommended_lpl_weight"] == 0.008


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


def test_permutation_index_rejects_incomplete_or_duplicate_maps(tmp_path) -> None:
    valid_path = tmp_path / "valid.npy"
    np.save(valid_path, np.array([2, 0, 1], dtype=np.int64))
    actual = load_permutation_index(valid_path, expected_length=3)
    np.testing.assert_array_equal(actual, [2, 0, 1])

    duplicate_path = tmp_path / "duplicate.npy"
    np.save(duplicate_path, np.array([0, 0, 2], dtype=np.int64))
    try:
        load_permutation_index(duplicate_path, expected_length=3)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate index map was accepted")


def test_packed_imagenet_preserves_bytes_labels_and_index_map(tmp_path) -> None:
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    encoded_images = []
    for color in ((255, 0, 0), (0, 255, 0)):
        image = Image.new("RGB", (8, 8), color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded_images.append(buffer.getvalue())

    data_path = shard_dir / "train-00000.bin"
    data_path.write_bytes(b"".join(encoded_images))
    offsets = np.array(
        [0, len(encoded_images[0]), sum(map(len, encoded_images))],
        dtype=np.int64,
    )
    labels = np.array([3, 7], dtype=np.int32)
    np.save(shard_dir / "train-00000.offsets.npy", offsets)
    np.save(shard_dir / "train-00000.labels.npy", labels)
    manifest = {
        "format": PACKED_IMAGENET_FORMAT,
        "version": 1,
        "split": "train",
        "total_rows": 2,
        "shards": [
            {
                "rows": 2,
                "data_file": "train/train-00000.bin",
                "offsets_file": "train/train-00000.offsets.npy",
                "labels_file": "train/train-00000.labels.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    index_map = tmp_path / "index.npy"
    np.save(index_map, np.array([1, 0], dtype=np.int64))

    dataset = DeterministicImageNetPacked(
        tmp_path,
        split="train",
        image_size=8,
        horizontal_flip=False,
        index_map_path=index_map,
        max_open_shards=1,
    )
    first_image, first_label, first_index = dataset[0]
    second_image, second_label, second_index = dataset[1]

    assert (first_label, first_index) == (7, 0)
    assert (second_label, second_index) == (3, 1)
    torch.testing.assert_close(first_image[:, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(second_image[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    dataset.close()

import json

import pytest
import torch
from omegaconf import OmegaConf

from experiments.evaluate_rae_spectral_generation import (
    _sampling_provenance_contract,
    _write_sampling_provenance,
    endpoint_checkpoint,
    prepare_sampling_config,
    sample_branch,
    sample_folder_name,
)


def test_sample_folder_name_versions_interleaved_label_protocol():
    assert sample_folder_name(5000, 10000, 50) == (
        "fixed_seed20260715_5000_step10000_labels-interleaved-v3-provenance"
    )
    assert sample_folder_name(5000, 10000, 25) == (
        "fixed_seed20260715_5000_step10000_labels-interleaved-v3-provenance_25steps"
    )
    assert sample_folder_name(5000, 10000, 50, state_key="model") == (
        "fixed_seed20260715_5000_step10000_labels-interleaved-v3-provenance_state-model"
    )


def test_zero_update_branch_uses_manifest_official_source(tmp_path):
    source = tmp_path / "official.pt"
    source.touch()
    branch = tmp_path / "branch"
    branch.mkdir()
    (branch / "manifest.json").write_text(
        json.dumps({"source_checkpoint": str(source)}),
        encoding="utf-8",
    )

    assert endpoint_checkpoint(branch, 0) == source


def test_complete_sample_archive_is_reused_without_resampling(
    tmp_path, monkeypatch
):
    branch = tmp_path / "branch"
    checkpoint_dir = branch / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-0000500.pt"
    torch.save({"ema": {"weight": torch.tensor([1.0])}}, checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    sampling_checkpoint = tmp_path / "ema.pt"
    torch.save({"weight": torch.tensor([1.0])}, sampling_checkpoint)
    folder = (
        branch
        / "generation"
        / sample_folder_name(5000, 500, 50, sampling_seed=123)
    )
    folder.mkdir(parents=True)
    folder.with_suffix(".npz").write_bytes(b"archive")
    monkeypatch.setattr(
        "experiments.evaluate_rae_spectral_generation.prepare_sampling_config",
        lambda *_args, **_kwargs: (config, sampling_checkpoint),
    )
    contract = _sampling_provenance_contract(
        checkpoint=checkpoint,
        sampling_checkpoint=sampling_checkpoint,
        config=config,
        sample_count=5000,
        steps=50,
        sampling_seed=123,
        state_key="ema",
        processes=4,
        per_process_batch=4,
    )
    _write_sampling_provenance(folder, contract)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("a complete archive must not be resampled")

    monkeypatch.setattr(
        "experiments.evaluate_rae_spectral_generation.subprocess.run",
        fail_if_called,
    )

    actual = sample_branch(
        branch,
        endpoint=500,
        sample_count=5000,
        steps=50,
        devices="0,1,2,3",
        processes=4,
        per_process_batch=4,
        sampling_seed=123,
    )

    assert actual == folder


def test_complete_archive_without_provenance_is_rejected(tmp_path, monkeypatch):
    branch = tmp_path / "branch"
    checkpoint_dir = branch / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-0000500.pt"
    torch.save({"ema": {"weight": torch.tensor([1.0])}}, checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    sampling_checkpoint = tmp_path / "ema.pt"
    torch.save({"weight": torch.tensor([1.0])}, sampling_checkpoint)
    folder = (
        branch
        / "generation"
        / sample_folder_name(5000, 500, 50, sampling_seed=123)
    )
    folder.mkdir(parents=True)
    folder.with_suffix(".npz").write_bytes(b"archive")
    monkeypatch.setattr(
        "experiments.evaluate_rae_spectral_generation.prepare_sampling_config",
        lambda *_args, **_kwargs: (config, sampling_checkpoint),
    )

    with pytest.raises(RuntimeError, match="no sampling provenance"):
        sample_branch(
            branch,
            endpoint=500,
            sample_count=5000,
            steps=50,
            devices="0,1,2,3",
            processes=4,
            per_process_batch=4,
            sampling_seed=123,
        )


def test_complete_archive_with_stale_checkpoint_provenance_is_rejected(
    tmp_path, monkeypatch
):
    branch = tmp_path / "branch"
    checkpoint_dir = branch / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-0000500.pt"
    torch.save({"ema": {"weight": torch.tensor([1.0])}}, checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    sampling_checkpoint = tmp_path / "ema.pt"
    torch.save({"weight": torch.tensor([1.0])}, sampling_checkpoint)
    folder = (
        branch
        / "generation"
        / sample_folder_name(5000, 500, 50, sampling_seed=123)
    )
    folder.mkdir(parents=True)
    folder.with_suffix(".npz").write_bytes(b"archive")
    contract = _sampling_provenance_contract(
        checkpoint=checkpoint,
        sampling_checkpoint=sampling_checkpoint,
        config=config,
        sample_count=5000,
        steps=50,
        sampling_seed=123,
        state_key="ema",
        processes=4,
        per_process_batch=4,
    )
    _write_sampling_provenance(folder, contract)
    torch.save({"ema": {"weight": torch.tensor([2.0])}}, checkpoint)
    monkeypatch.setattr(
        "experiments.evaluate_rae_spectral_generation.prepare_sampling_config",
        lambda *_args, **_kwargs: (config, sampling_checkpoint),
    )

    with pytest.raises(RuntimeError, match="sampling provenance mismatch"):
        sample_branch(
            branch,
            endpoint=500,
            sample_count=5000,
            steps=50,
            devices="0,1,2,3",
            processes=4,
            per_process_batch=4,
            sampling_seed=123,
        )


def test_sampling_config_replaces_stale_materialized_state(tmp_path):
    branch = tmp_path / "branch"
    branch.mkdir()
    config = {
        "stage_2": {"ckpt": "source.pt"},
        "sampler": {"params": {"num_steps": 50}},
        "guidance": {"method": "cfg", "scale": 1.0},
        "training": {"unused": True},
        "eval": {"unused": True},
    }
    OmegaConf.save(OmegaConf.create(config), branch / "config.yaml")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"ema": {"weight": torch.tensor([1.0])}}, checkpoint)

    _, materialized = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )
    first = torch.load(materialized, map_location="cpu", weights_only=True)
    torch.testing.assert_close(first["weight"], torch.tensor([1.0]))

    torch.save({"ema": {"weight": torch.tensor([2.0])}}, checkpoint)
    _, materialized_again = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )
    second = torch.load(materialized_again, map_location="cpu", weights_only=True)

    assert materialized_again == materialized
    torch.testing.assert_close(second["weight"], torch.tensor([2.0]))
    provenance = json.loads(
        materialized.with_suffix(".source.json").read_text(encoding="utf-8")
    )
    assert provenance["state_key"] == "ema"
    assert provenance["source_checkpoint"] == str(checkpoint.resolve())
    assert provenance["materialized_checkpoint_sha256"]


def test_sampling_config_repairs_corrupt_materialized_state(tmp_path):
    branch = tmp_path / "branch"
    branch.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "stage_2": {"ckpt": "source.pt"},
                "sampler": {"params": {"num_steps": 50}},
                "guidance": {"method": "cfg", "scale": 1.0},
            }
        ),
        branch / "config.yaml",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"ema": {"weight": torch.tensor([3.0])}}, checkpoint)
    _, materialized = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )
    materialized.write_bytes(b"corrupt")

    _, repaired = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )

    restored = torch.load(repaired, map_location="cpu", weights_only=True)
    torch.testing.assert_close(restored["weight"], torch.tensor([3.0]))


def test_sampling_config_repairs_corrupt_provenance_sidecar(tmp_path):
    branch = tmp_path / "branch"
    branch.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "stage_2": {"ckpt": "source.pt"},
                "sampler": {"params": {"num_steps": 50}},
                "guidance": {"method": "cfg", "scale": 1.0},
            }
        ),
        branch / "config.yaml",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"ema": {"weight": torch.tensor([4.0])}}, checkpoint)
    _, materialized = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )
    sidecar = materialized.with_suffix(".source.json")
    sidecar.write_text("{", encoding="utf-8")

    _, repaired = prepare_sampling_config(
        branch, checkpoint, 50, state_key="ema"
    )

    restored = torch.load(repaired, map_location="cpu", weights_only=True)
    torch.testing.assert_close(restored["weight"], torch.tensor([4.0]))
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["materialized_checkpoint_sha256"]

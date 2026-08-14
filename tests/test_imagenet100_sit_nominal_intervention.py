from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import run_imagenet100_sit_nominal_donor_fid as donor_runner
from experiments.imagenet100_sit_static_pair import FieldSemantics, LEGACY_PROTOCOL
from experiments.sample_imagenet100_sit_nominal_intervention_fid import (
    conditional_nominal_intervention_velocity,
)
from experiments.sample_imagenet100_sit_nominal_donor_fid import (
    conditional_donor_guidance_velocity,
)
from experiments import run_imagenet100_sit_nominal_intervention_fid5k as fid_runner
from experiments.run_imagenet100_sit_fid_curve import valid_resource_audit


class _ScaleModel(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(
        self,
        state: torch.Tensor,
        times: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del times, labels
        return self.scale * state


SEMANTICS = FieldSemantics(
    protocol=LEGACY_PROTOCOL,
    field_path="auto",
    prediction_target="velocity",
    denominator_floor=0.001,
    gate_activation=None,
)


def _derivative(mode: str) -> torch.Tensor:
    labels = torch.tensor([3])
    field, _ = conditional_nominal_intervention_velocity(
        _ScaleModel(2.0),
        _ScaleModel(1.0),
        labels,
        anchor_semantics=SEMANTICS,
        other_semantics=SEMANTICS,
        mode=mode,
        gamma=1.0,
        autocast_dtype=None,
    )
    baseline = torch.tensor([[1.0, 0.0]])
    current = torch.tensor([[2.0, 3.0]])
    return field(torch.tensor(0.5), torch.cat((baseline, current)))


def test_nominal_intervention_modes_have_expected_coupled_derivatives() -> None:
    expected_current = {
        "frozen": torch.tensor([[5.0, 6.0]]),
        "replay": torch.tensor([[3.0, 0.0]]),
        "gain_only": torch.tensor([[6.0, 6.0]]),
        "direction_only": torch.tensor([[5.0, 9.0]]),
        "closed": torch.tensor([[6.0, 9.0]]),
    }
    for mode, expected in expected_current.items():
        derivative = _derivative(mode)
        torch.testing.assert_close(derivative[:1], torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(derivative[1:], expected)


def test_donor_guidance_uses_donor_gap_and_target_current_anchor() -> None:
    target_labels = torch.tensor([3])
    donor_labels = torch.tensor([7])
    field, _ = conditional_donor_guidance_velocity(
        _ScaleModel(2.0),
        _ScaleModel(1.0),
        target_labels,
        donor_labels,
        anchor_semantics=SEMANTICS,
        other_semantics=SEMANTICS,
        gamma=0.5,
        autocast_dtype=None,
    )
    target_baseline = torch.tensor([[1.0, 0.0]])
    donor = torch.tensor([[4.0, 2.0]])
    current = torch.tensor([[2.0, 3.0]])

    derivative = field(
        torch.tensor(0.5),
        torch.cat((target_baseline, donor, current)),
    )

    torch.testing.assert_close(derivative[0:1], torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(derivative[1:2], torch.tensor([[8.0, 4.0]]))
    # Current strong field (4, 6) plus half the donor gap (4, 2).
    torch.testing.assert_close(derivative[2:3], torch.tensor([[6.0, 7.0]]))


def test_sampling_artifact_accepts_sampler_manifest_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 800_000,
        "protocol": "sit_flow_v1",
        "field_path": "auto",
        "prediction_target": "velocity",
        "denominator_floor": 0.001,
    }
    other = {**metadata, "checkpoint_sha256": "b" * 64}
    sample_path = tmp_path / "samples_frozen_n8.npz"
    sample_path.touch()
    manifest = {
        "format": "eqvae_imagenet100_sit_nominal_intervention_samples_v1",
        "mode": "frozen",
        "weights": "ema",
        "requested_samples": 8,
        "batch_size": 2,
        "vae_decode_batch_size": 1,
        "global_seed": 0,
        "gamma": 1.0,
        "allocator_limit_gib": 4.0,
        "anchor": metadata,
        "other": other,
        "sampler": {
            "method": "dopri5",
            "num_output_points": 16,
            "precision": "fp32",
            "allow_tf32": True,
            "atol": 1e-6,
            "rtol": 1e-3,
        },
        "samples": str(sample_path),
        "noise_sha256": "c" * 64,
        "label_sha256": "d" * 64,
    }
    (tmp_path / "sampling_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        mode="frozen",
        num_samples=8,
        batch_size=2,
        vae_decode_batch_size=1,
        global_seed=0,
        gamma=1.0,
        cuda_allocator_limit_gib=4.0,
        gpu_indices=[0],
        gpu_memory_ceiling_mib=8192,
        num_output_points=16,
        atol=1e-6,
        rtol=1e-3,
    )
    monkeypatch.setattr(fid_runner, "valid_resource_audit", lambda *args, **kwargs: True)

    assert fid_runner._valid_sampling_artifact(
        tmp_path,
        anchor=metadata,
        other=other,
        args=args,
    )


@pytest.mark.parametrize(
    ("runner", "sample_name", "result_name", "mode_key", "mode"),
    (
        (
            fid_runner,
            "samples_replay_n8.npz",
            "fid5k_adm_results.json",
            "mode",
            "replay",
        ),
        (
            donor_runner,
            "samples_paired_n8.npz",
            "fid_adm_results.json",
            "donor_mode",
            "paired",
        ),
    ),
)
def test_fid_artifact_is_rejected_after_samples_are_regenerated(
    tmp_path: Path,
    monkeypatch,
    runner,
    sample_name: str,
    result_name: str,
    mode_key: str,
    mode: str,
) -> None:
    sample_path = tmp_path / sample_name
    manifest_path = tmp_path / "sampling_manifest.json"
    result_path = tmp_path / result_name
    reference = tmp_path / "reference.npz"
    for path in (sample_path, manifest_path, reference):
        path.touch()
    result_path.write_text(
        json.dumps(
            {
                "reference": str(reference),
                "samples": str(sample_path),
                "batch_size": 8,
                "gpu_memory_fraction": 0.1,
                "fid": 1.0,
                "sfid": 2.0,
                "inception_score": 3.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "valid_resource_audit", lambda *args, **kwargs: True)
    args = argparse.Namespace(
        **{
            mode_key: mode,
            "num_samples": 8,
            "reference": reference,
            "fid_batch_size": 8,
            "fid_gpu_memory_fraction": 0.1,
            "gpu_indices": [0],
            "gpu_memory_ceiling_mib": 8192,
        }
    )

    assert runner._valid_fid_artifact(tmp_path, args=args)

    newer = result_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(sample_path, ns=(newer, newer))
    assert not runner._valid_fid_artifact(tmp_path, args=args)


def test_resource_audit_can_be_reused_on_a_different_single_gpu(tmp_path: Path) -> None:
    path = tmp_path / "resource_audit.json"
    path.write_text(
        json.dumps(
            {
                "violation": None,
                "return_code": 0,
                "memory_ceiling_mib": 8192,
                "monitored_gpu_indices": [2],
                "peak_memory_mib": {"2": 3000},
            }
        ),
        encoding="utf-8",
    )

    assert not valid_resource_audit(
        path,
        gpu_indices=[0],
        memory_ceiling_mib=8192,
    )
    assert valid_resource_audit(
        path,
        gpu_indices=[0],
        memory_ceiling_mib=8192,
        allow_gpu_index_mismatch=True,
    )


@pytest.mark.parametrize(
    ("mode", "same_noise", "same_class"),
    (
        ("paired", True, True),
        ("same_noise_other_class", True, False),
        ("other_noise_same_class", False, True),
        ("other_noise_other_class", False, False),
    ),
)
def test_donor_fingerprint_contract(
    mode: str,
    same_noise: bool,
    same_class: bool,
) -> None:
    target_noise = "a" * 64
    target_label = "b" * 64
    manifest = {
        "target_noise_sha256": target_noise,
        "donor_noise_sha256": target_noise if same_noise else "c" * 64,
        "target_label_sha256": target_label,
        "donor_label_sha256": target_label if same_class else "d" * 64,
    }
    donor_runner._validate_donor_fingerprints(manifest, mode)

    manifest["donor_noise_sha256"] = (
        "c" * 64 if same_noise else target_noise
    )
    with pytest.raises(ValueError, match="noise fingerprint"):
        donor_runner._validate_donor_fingerprints(manifest, mode)

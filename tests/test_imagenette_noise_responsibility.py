import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.imagenette_noise_responsibility import (
    CompactImagenetteEncoder,
    ImagenetteConditionalUNet,
    ImagenetteResponsibilityConfig,
    aggregate_paired_rows,
    apply_condition_dropout,
    build_fresh_models,
    conditional_euler_sample,
    curve_summary,
    within_class_derangement,
)


def test_preregistered_capacities_and_dropout_are_enforced():
    ImagenetteResponsibilityConfig(latent_dim=16).validate()
    ImagenetteResponsibilityConfig(latent_dim=64).validate()
    ImagenetteResponsibilityConfig(latent_dim=256).validate()
    try:
        ImagenetteResponsibilityConfig(latent_dim=32).validate()
    except ValueError as error:
        assert "latent_dim" in str(error)
    else:
        raise AssertionError("an unregistered capacity was accepted")


def test_condition_dropout_uses_exact_zero_and_expected_rate():
    latent = torch.randn(20_000, 16, generator=torch.Generator().manual_seed(1))
    dropped, mask = apply_condition_dropout(
        latent,
        0.1,
        generator=torch.Generator().manual_seed(2),
    )
    assert 0.09 < float(mask.float().mean()) < 0.11
    assert torch.equal(dropped[mask], torch.zeros_like(dropped[mask]))
    assert torch.equal(dropped[~mask], latent[~mask])


def test_condition_projection_has_fixed_rms_and_zero_is_exact_null():
    torch.manual_seed(3)
    model = ImagenetteConditionalUNet(16, width=8)
    latent = torch.randn(7, 16)
    projected = model.condition_embedding(latent)
    null = model.condition_embedding(torch.zeros_like(latent))
    torch.testing.assert_close(projected.square().mean(dim=1), torch.ones(7), atol=2e-4, rtol=0)
    assert torch.equal(null, torch.zeros_like(null))


def test_null_condition_normalization_has_finite_backward():
    torch.manual_seed(31)
    model = ImagenetteConditionalUNet(16, width=8)
    torch.nn.init.normal_(model.output.weight, std=0.01)
    state = torch.randn(4, 3, 16, 16)
    time = torch.rand(4)
    condition = torch.randn(4, 16, requires_grad=True)
    condition.data[:2].zero_()
    loss = model(state, time, condition).square().mean()
    loss.backward()
    assert torch.isfinite(condition.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_all_timesteps_can_use_condition():
    torch.manual_seed(4)
    model = ImagenetteConditionalUNet(16, width=8)
    torch.nn.init.normal_(model.output.weight, std=0.01)
    state = torch.randn(2, 3, 64, 64)
    first = torch.zeros(2, 16)
    second = torch.randn(2, 16)
    for time_value in (0.05, 0.5, 0.95):
        time = torch.full((2,), time_value)
        null_output = model(state, time, first)
        real_output = model(state, time, second)
        assert not torch.equal(null_output, real_output)


def test_encoder_shapes_and_normalizes_each_sample():
    torch.manual_seed(5)
    encoder = CompactImagenetteEncoder(16, width=8)
    latent = encoder(torch.randn(4, 3, 64, 64))
    assert latent.shape == (4, 16)
    torch.testing.assert_close(latent.mean(dim=1), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(latent.square().mean(dim=1), torch.ones(4), atol=2e-4, rtol=0)


def test_shared_initialization_is_identical_across_capacities():
    small_config = ImagenetteResponsibilityConfig(
        latent_dim=16, encoder_width=8, model_width=8, seed=13
    )
    large_config = ImagenetteResponsibilityConfig(
        latent_dim=256, encoder_width=8, model_width=8, seed=13
    )
    _, _, small_hashes = build_fresh_models(small_config, torch.device("cpu"))
    _, _, large_hashes = build_fresh_models(large_config, torch.device("cpu"))
    assert small_hashes == large_hashes


def test_within_class_derangement_has_no_leak_or_fixed_point():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
    permutation = within_class_derangement(labels, seed=7)
    assert torch.all(permutation != torch.arange(len(labels)))
    assert torch.equal(labels[permutation], labels)


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, state, time, condition):
        return self.velocity.expand_as(state)


def test_descending_euler_sign_recovers_clean_endpoint():
    clean = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(8))
    noise = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(9))
    model = ConstantVelocity(noise - clean)
    recovered = conditional_euler_sample(model, noise.clone(), torch.zeros(2, 1), steps=7)
    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=0)


def test_aggregation_and_curve_regions_preserve_paired_signs():
    paired = pd.DataFrame(
        {
            "time": [0.1, 0.1, 0.5, 0.5, 0.9, 0.9],
            "loss_real": np.zeros(6),
            "loss_null": np.ones(6),
            "loss_shuffle": np.ones(6),
            "loss_within_class": np.ones(6),
            "delta_null": [1, 1, 2, 2, 3, 3],
            "delta_shuffle": [1, 1, 2, 2, 3, 3],
            "delta_within_class": [1, 1, 2, 2, 3, 3],
        }
    )
    profile = aggregate_paired_rows(paired, ["time"])
    frequency = pd.concat(
        [profile.assign(band=band) for band in ("low", "mid", "high")],
        ignore_index=True,
    )
    summary = curve_summary(profile, frequency)
    total = summary[(summary.source == "total") & (summary.metric == "delta_shuffle")].iloc[0]
    assert total.low_noise_fraction == 1 / 6
    assert total.mid_noise_fraction == 2 / 6
    assert total.high_noise_fraction == 3 / 6


def test_script_entrypoint_help_loads_repo_modules():
    script = Path(__file__).resolve().parents[1] / "experiments/imagenette_noise_responsibility.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

import subprocess
import sys
from pathlib import Path

import torch

from experiments.imagenette_decoder_aware_prior import (
    clean_estimate_from_velocity,
    decoder_condition_loss,
    decoder_feature_loss,
    decoder_response_features,
)
from experiments.imagenette_noise_responsibility import ImagenetteConditionalUNet


def _decoder() -> ImagenetteConditionalUNet:
    torch.manual_seed(3)
    decoder = ImagenetteConditionalUNet(latent_dim=4, width=8).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def test_clean_estimate_recovers_clean_for_exact_velocity():
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn(7, 4, generator=generator)
    noise = torch.randn(7, 4, generator=generator)
    time = torch.rand(7, generator=generator).clamp_min(0.05)
    state = (1.0 - time[:, None]) * clean + time[:, None] * noise
    recovered = clean_estimate_from_velocity(state, noise - clean, time)
    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=0)


def test_decoder_response_features_are_paired_and_differentiable():
    decoder = _decoder()
    generator = torch.Generator().manual_seed(7)
    state = torch.randn(2, 3, 16, 16, generator=generator)
    time = torch.tensor([0.6, 0.9])
    reference_condition = torch.randn(2, 4, generator=generator)
    candidate_condition = reference_condition.clone().requires_grad_(True)
    layer_names = ("middle", "up1", "up0")

    with torch.no_grad():
        reference = decoder_response_features(
            decoder, state, time, reference_condition, layer_names=layer_names
        )
    candidate = decoder_response_features(
        decoder, state, time, candidate_condition, layer_names=layer_names
    )
    equal_loss = decoder_feature_loss(candidate, reference)
    torch.testing.assert_close(equal_loss, torch.zeros_like(equal_loss), atol=0, rtol=0)

    perturbed = decoder_response_features(
        decoder,
        state,
        time,
        candidate_condition + 0.2,
        layer_names=layer_names,
    )
    loss = decoder_feature_loss(perturbed, reference).mean()
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert candidate_condition.grad is not None
    assert torch.isfinite(candidate_condition.grad).all()
    assert float(candidate_condition.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_condition_loss_is_zero_for_equal_conditions_and_has_gradient():
    decoder = _decoder()
    reference = torch.randn(3, 4, generator=torch.Generator().manual_seed(11))
    candidate = reference.clone().requires_grad_(True)
    exact = decoder_condition_loss(decoder, candidate, reference)
    torch.testing.assert_close(exact, torch.zeros_like(exact), atol=0, rtol=0)
    changed = decoder_condition_loss(decoder, candidate + 0.1, reference).mean()
    changed.backward()
    assert candidate.grad is not None
    assert float(candidate.grad.norm()) > 0.0


def test_script_entrypoint_help_loads_repo_modules():
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "imagenette_decoder_aware_prior.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

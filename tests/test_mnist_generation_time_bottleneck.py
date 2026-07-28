import subprocess
import sys
from pathlib import Path

import torch

from experiments.mnist_generation_time_bottleneck import (
    CompactConditionEncoder,
    ConditionalVelocityUNet,
    _sample_training_batch,
    condition_gate,
    covariance_regularizer,
)


def test_condition_gates_have_exact_endpoints_and_are_complements():
    time = torch.tensor([0.0, 0.55, 0.65, 0.75, 1.0])
    high = condition_gate(time, "high_noise")
    low = condition_gate(time, "low_noise")
    assert torch.equal(condition_gate(time, "all_time"), torch.ones_like(time))
    assert torch.equal(condition_gate(time, "none"), torch.zeros_like(time))
    assert torch.equal(high[[0, 1]], torch.zeros(2))
    assert torch.equal(high[[-2, -1]], torch.ones(2))
    assert torch.allclose(high + low, torch.ones_like(time))


def test_condition_is_exactly_ignored_outside_high_noise_window():
    torch.manual_seed(0)
    model = ConditionalVelocityUNet(4, width=8, depth=1, mode="high_noise")
    state = torch.randn(2, 1, 28, 28)
    time = torch.tensor([0.2, 0.4])
    first = model(state, time, torch.randn(2, 4))
    second = model(state, time, torch.randn(2, 4))
    assert torch.equal(first, second)


def test_high_noise_condition_changes_output_after_nonzero_head_init():
    torch.manual_seed(1)
    model = ConditionalVelocityUNet(4, width=8, depth=1, mode="high_noise")
    torch.nn.init.normal_(model.output.weight, std=0.01)
    state = torch.randn(2, 1, 28, 28)
    time = torch.tensor([0.9, 0.9])
    first = model(state, time, torch.zeros(2, 4))
    second = model(state, time, torch.ones(2, 4))
    assert not torch.equal(first, second)


def test_encoder_and_covariance_regularizer_are_finite():
    encoder = CompactConditionEncoder(8, width=8)
    latent = encoder(torch.randn(16, 1, 28, 28))
    loss = covariance_regularizer(latent)
    assert latent.shape == (16, 8)
    assert torch.isfinite(loss)


def test_script_entrypoint_can_load_repo_modules():
    script = Path(__file__).resolve().parents[1] / "experiments/mnist_generation_time_bottleneck.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_data_stream_is_unchanged_by_separate_latent_random_draws():
    clean = torch.randn(32, 1, 28, 28, generator=torch.Generator().manual_seed(5))
    first_data = torch.Generator().manual_seed(7)
    second_data = torch.Generator().manual_seed(7)
    latent_stream = torch.Generator().manual_seed(11)
    first_batch = _sample_training_batch(clean, 8, first_data)
    torch.randn((8, 4), generator=latent_stream)
    second_batch = _sample_training_batch(clean, 8, second_data)
    for first, second in zip(first_batch, second_batch):
        assert torch.equal(first, second)

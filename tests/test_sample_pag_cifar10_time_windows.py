import torch

from experiments.sample_pag_cifar10_time_windows import (
    SamplingPolicy,
    default_policies,
    policy_scale_tensor,
    validate_sampling_protocol,
)


def test_default_policy_windows_have_expected_sign_and_boundaries() -> None:
    policies = {policy.name: policy for policy in default_policies()}
    assert policies["baseline_full"].scale_at(900) == 1.0
    assert policies["pag_all_s1.25"].scale_at(100) == 1.25
    assert policies["pag_high_t800_s1.25"].scale_at(799) == 1.0
    assert policies["pag_high_t800_s1.25"].scale_at(800) == 1.25
    assert policies["pag_low_t500_s1.25"].scale_at(500) == 1.25
    assert policies["pag_low_t500_s1.25"].scale_at(501) == 1.0
    assert policies["interpolate_low_t300_s0.5"].scale_at(300) == 0.5


def test_policy_scale_tensor_matches_policy_major_state_order() -> None:
    policies = (
        SamplingPolicy("full", 1.0, 0, 999),
        SamplingPolicy("high", 1.25, 800, 999),
    )
    actual = policy_scale_tensor(
        policies,
        900,
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 1.0, 1.0, 1.25, 1.25, 1.25]))


def test_sampling_protocol_is_valid() -> None:
    validate_sampling_protocol(
        samples=8,
        batch_size=2,
        inference_steps=10,
        policies=default_policies(),
        train_timesteps=1000,
    )

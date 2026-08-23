import math

import torch

from experiments.advfd_cleanroom.audit_pmf_residual_generator_vjp import (
    cosine,
    gradient_sketch,
    method_sigmas,
    parameter_group,
    relative_l2,
)


def test_parameter_group_preserves_transformer_depth() -> None:
    assert (
        parameter_group("net.shared_blocks.7.attn.q_proj.weight")
        == "net.shared_blocks.7"
    )
    assert parameter_group("net.x_embedder.proj1.weight") == "net.x_embedder"


def test_gradient_sketch_is_deterministic_and_tracks_exact_norm() -> None:
    first = torch.linspace(-1.0, 1.0, 101)
    second = torch.linspace(2.0, 3.0, 37)
    named = [
        ("net.shared_blocks.0.weight", first, first.numel()),
        ("net.final_layer.weight", second, second.numel()),
    ]
    left = gradient_sketch(named, budget=64, seed=19)
    right = gradient_sketch(named, budget=64, seed=19)
    expected_norm = math.sqrt(
        float(first.double().square().sum() + second.double().square().sum())
    )
    assert torch.equal(left["sketch"], right["sketch"])
    assert left["group_sketches"].keys() == right["group_sketches"].keys()
    assert math.isclose(left["exact_norm"], expected_norm, rel_tol=1e-12)


def test_pair_metrics_have_expected_limits() -> None:
    first = torch.tensor([1.0, 2.0, 3.0])
    assert math.isclose(cosine(first, first), 1.0)
    assert math.isclose(relative_l2(first, first), 0.0)
    assert math.isclose(cosine(first, -first), -1.0)
    assert math.isclose(relative_l2(first, -first), 2.0)


def test_zero_ratio_uses_clean_features_only() -> None:
    requested = (0.1, 0.3, 0.7)
    assert method_sigmas("zero_ratio", requested) == (0.0,)
    assert method_sigmas("shared_dsm", requested) == requested

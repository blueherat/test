import math

import torch

from experiments.advfd_cleanroom.audit_pmf_residual_generator_vjp import (
    cosine,
    gradient_sketch,
    method_sigmas,
    parameter_group,
    relative_l2,
)
from experiments.advfd_cleanroom.run_pmf_residual_score_posttrain import (
    apply_moment_tangent_projection,
    projected_adam_step,
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


def test_frozen_moment_projection_applies_affine_normal_field() -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    field = torch.tensor([[4.0, -1.0], [2.0, 7.0]])
    projection = {
        "source_mean": torch.tensor([0.5, 1.0]),
        "translation": torch.tensor([0.2, -0.3]),
        "symmetric_linear": torch.tensor([[2.0, 0.5], [0.5, -1.0]]),
    }
    tangent, diagnostics = apply_moment_tangent_projection(
        features, field, projection
    )
    expected_normal = projection["translation"] + (
        features - projection["source_mean"]
    ) @ projection["symmetric_linear"].mT
    torch.testing.assert_close(tangent, field - expected_normal)
    assert diagnostics["full_field_sample_norm"] > 0
    assert diagnostics["tangent_field_sample_norm"] > 0
    assert diagnostics["normal_field_sample_norm"] > 0


def test_projected_adam_step_projects_the_realized_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    parameter.grad = torch.tensor([2.0, -1.0, 0.25])
    constraint = torch.tensor([1.0, 2.0, -1.0])
    before = parameter.detach().clone()
    diagnostics = projected_adam_step(
        [parameter],
        [constraint],
        [{}],
        step=1,
        learning_rate=0.1,
    )
    realized_direction = (before - parameter.detach()) / 0.1
    assert abs(float(realized_direction @ constraint)) < 1e-6
    assert abs(diagnostics["adam_constraint_cosine_after"]) < 1e-6
    assert diagnostics["adam_removed_energy_fraction"] > 0

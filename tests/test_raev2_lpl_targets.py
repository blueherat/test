import torch

from experiments.raev2_lpl_targets import (
    lpl_prediction_targets,
    parse_guidance_scales,
    positive_parallel_projection,
    substitute_prediction_gradient,
)


def test_parse_guidance_scales_validates_values() -> None:
    assert parse_guidance_scales("1.0, 1.39,1.78") == (1.0, 1.39, 1.78)
    try:
        parse_guidance_scales("")
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty scale list must fail")


def test_full_and_full_base_targets_preserve_predictions() -> None:
    full = torch.tensor([[3.0], [5.0]])
    base = torch.tensor([[1.0], [2.0]])
    indices = torch.tensor([4, 5])

    predictions, scales = lpl_prediction_targets(
        (full, base),
        target="full",
        guidance_scale=1.78,
        multiscale_scales=(1.0, 1.39, 1.78),
        sample_indices=indices,
    )
    assert predictions == (full,)
    assert scales is None

    predictions, scales = lpl_prediction_targets(
        (full, base),
        target="full_base",
        guidance_scale=1.78,
        multiscale_scales=(1.0, 1.39, 1.78),
        sample_indices=indices,
    )
    torch.testing.assert_close(predictions[0], torch.tensor([[3.0], [2.0]]))
    assert scales is None


def test_guided_target_matches_sampling_formula() -> None:
    full = torch.tensor([[3.0], [5.0]])
    base = torch.tensor([[1.0], [1.0]])
    predictions, scales = lpl_prediction_targets(
        (full, base),
        target="guided",
        guidance_scale=1.5,
        multiscale_scales=(1.0, 1.5),
        sample_indices=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(predictions[0], torch.tensor([[4.0], [7.0]]))
    torch.testing.assert_close(scales, torch.tensor([1.5, 1.5]))


def test_guided_common_preserves_forward_value_and_uses_common_gradient() -> None:
    full = torch.tensor([[3.0], [5.0]], requires_grad=True)
    base = torch.tensor([[1.0], [1.0]], requires_grad=True)
    predictions, scales = lpl_prediction_targets(
        (full, base),
        target="guided_common",
        guidance_scale=1.5,
        multiscale_scales=(1.0, 1.5),
        sample_indices=torch.tensor([0, 1]),
    )

    torch.testing.assert_close(predictions[0], torch.tensor([[4.0], [7.0]]))
    torch.testing.assert_close(scales, torch.tensor([1.5, 1.5]))
    predictions[0].sum().backward()
    torch.testing.assert_close(full.grad, torch.full_like(full, 0.5))
    torch.testing.assert_close(base.grad, torch.full_like(base, 0.5))


def test_multiscale_assignment_is_index_deterministic() -> None:
    full = torch.tensor([[3.0], [3.0], [3.0], [3.0]])
    base = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    predictions, scales = lpl_prediction_targets(
        (full, base),
        target="guided_multiscale",
        guidance_scale=1.78,
        multiscale_scales=(1.0, 1.5, 2.0),
        sample_indices=torch.tensor([3, 4, 5, 6]),
    )
    torch.testing.assert_close(scales, torch.tensor([1.0, 1.5, 2.0, 1.0]))
    torch.testing.assert_close(
        predictions[0],
        torch.tensor([[3.0], [4.0], [5.0], [3.0]]),
    )


def test_positive_parallel_projection_removes_conflict_and_tangent() -> None:
    auxiliary = torch.tensor([[[-2.0, 3.0]], [[4.0, 5.0]]])
    reference = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]])

    projected, details = positive_parallel_projection(auxiliary, reference)

    torch.testing.assert_close(projected, torch.tensor([[[0.0, 0.0]], [[4.0, 0.0]]]))
    torch.testing.assert_close(details["conflict_fraction"], torch.tensor([1.0, 0.0]))
    assert bool(
        ((projected.flatten(1) * reference.flatten(1)).sum(1) >= 0).all()
    )


def test_substitute_prediction_gradient_preserves_value_and_uses_replacement() -> None:
    prediction = torch.tensor([[2.0, -1.0]], requires_grad=True)
    original_loss = (prediction.square()).sum()
    replacement = torch.tensor([[3.0, 4.0]])

    substituted = substitute_prediction_gradient(
        original_loss,
        prediction,
        replacement,
    )

    torch.testing.assert_close(substituted, original_loss.detach())
    substituted.backward()
    torch.testing.assert_close(prediction.grad, replacement)


def test_parallel_substitution_cannot_increase_reference_quadratic_first_order() -> None:
    prediction = torch.tensor([[2.0, -1.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    reference_loss = (prediction - target).square().mean()
    auxiliary_loss = 3.0 * prediction[0, 1] - prediction[0, 0]
    auxiliary_gradient = torch.autograd.grad(
        auxiliary_loss,
        prediction,
        retain_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_loss,
        prediction,
        retain_graph=True,
    )[0]
    projected, _ = positive_parallel_projection(
        auxiliary_gradient,
        reference_gradient,
    )
    surrogate = substitute_prediction_gradient(
        auxiliary_loss,
        prediction,
        projected,
    )

    gradient = torch.autograd.grad(surrogate, prediction)[0]
    first_order_reference_change = -(gradient * reference_gradient).sum()
    assert float(first_order_reference_change) <= 1e-7


def test_detached_gradient_bridge_matches_direct_parameter_gradient() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2, bias=False)
    decoder = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.SiLU(),
        torch.nn.Linear(4, 3),
    )
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 3)

    direct_prediction = model(inputs)
    direct_loss = (decoder(direct_prediction) - targets).square().mean()
    direct_gradient = torch.autograd.grad(direct_loss, model.weight)[0]

    probe_prediction = model(inputs).detach().requires_grad_(True)
    probe_loss = (decoder(probe_prediction) - targets).square().mean()
    output_gradient = torch.autograd.grad(probe_loss, probe_prediction)[0]
    replay_prediction = model(inputs)
    replay_loss = substitute_prediction_gradient(
        probe_loss.detach(),
        replay_prediction,
        output_gradient,
    )
    replay_gradient = torch.autograd.grad(replay_loss, model.weight)[0]

    torch.testing.assert_close(replay_gradient, direct_gradient)

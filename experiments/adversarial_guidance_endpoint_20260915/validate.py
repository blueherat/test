"""Gradient and objective correctness checks, not image-quality experiments."""

import argparse
from pathlib import Path
import torch
from .discrete_adjoint import ordinary_rollout, recomputed_rollout
from .objectives import pair_logits, guided_generator_loss
from experiments.guidance_loss_50k_20260914.config import atomic, sha


def validate():
    torch.manual_seed(2026091503)
    torch.set_num_threads(2)
    parameter = torch.nn.Parameter(torch.randn(3, 3, dtype=torch.float64) * .1)
    initial = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    tail = torch.tensor([[1.1, .2, 0.], [0., .8, .3], [.1, 0., 1.2]], dtype=torch.float64)

    def step(index, state):
        strong = torch.sin(state @ tail)
        weak = torch.tanh(state @ parameter) if index < 3 else 0.
        return state + .1 * (strong - weak)

    reference = ordinary_rollout(step, 6, initial)
    actual = recomputed_rollout(step, 6, initial, (parameter,))
    torch.testing.assert_close(actual, reference, rtol=0., atol=0.)
    expected_grad = torch.autograd.grad(reference.square().mean(), (initial, parameter))
    actual_grad = torch.autograd.grad(actual.square().mean(), (initial, parameter))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-12, atol=1e-12)
    # An independent directional finite difference checks the entire sampler.
    direction = torch.randn_like(parameter)
    direction /= direction.norm()
    epsilon = 1e-5
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.copy_(original + epsilon * direction)
        positive = ordinary_rollout(step, 6, initial).square().mean()
        parameter.copy_(original - epsilon * direction)
        negative = ordinary_rollout(step, 6, initial).square().mean()
        parameter.copy_(original)
    finite_difference = (positive - negative) / (2 * epsilon)
    derivative = (actual_grad[1] * direction).sum()
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-7, atol=1e-9)

    # Detaching at the switch-off boundary really removes all head gradients.
    detached_boundary = ordinary_rollout(step, 3, initial).detach().requires_grad_(True)
    detached_final = ordinary_rollout(lambda i, x: step(i + 3, x), 3, detached_boundary)
    missing = torch.autograd.grad(detached_final.square().mean(), parameter, allow_unused=True)[0]
    assert missing is None and actual_grad[1].norm() > 0

    # Four-way optimal source posterior must reduce to P/R for arbitrary G/Q
    # and unequal source sampling frequencies.
    densities = torch.rand(11, 4, dtype=torch.float64) + .01
    priors = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    posterior_logits = (densities * priors).log()
    probabilities = pair_logits(posterior_logits, priors).softmax(1)
    expected_probability = densities[:, 0] / (densities[:, 0] + densities[:, 3])
    torch.testing.assert_close(probabilities[:, 0], expected_probability, rtol=1e-12, atol=1e-12)
    auxiliary_changed = posterior_logits.clone()
    auxiliary_changed[:, 1:3] += torch.tensor([1e4, -1e4], dtype=torch.float64)
    torch.testing.assert_close(guided_generator_loss(auxiliary_changed, priors),
                               guided_generator_loss(posterior_logits, priors), rtol=0., atol=0.)
    # The simple common-suffix equivalence is a statement about densities with
    # a shared Jacobian, not arbitrary fixed feature moments.
    return dict(
        kind="engineering_validation_not_generation_quality", complete=True,
        discrete_adjoint_max_abs_error=max(float((a - b).abs().max()) for a, b in zip(actual_grad, expected_grad)),
        directional_finite_difference_abs_error=float((derivative - finite_difference).abs()),
        head_gradient_norm=float(actual_grad[1].norm()),
        truncated_suffix_head_gradient_is_none=missing is None,
        auxiliary_source_pair_probability_max_error=float((probabilities[:, 0] - expected_probability).abs().max()),
        sources={str(path.resolve()): sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    atomic(args.output, result)
    print(result, flush=True)

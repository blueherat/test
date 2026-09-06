import inspect

import pytest
import torch
from torch.func import functional_call

from experiments.raev2_distribution_guidance import (
    AffineTokenGuidance, conditional_energy_training_objective,
    energy_distance_u_statistic,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_zero_gate_preserves_supplied_official_baseline_bits_and_initial_gradient(dtype):
    rng = torch.Generator().manual_seed(4201)
    full = torch.randn(2, 1024, 2, 3, generator=rng, dtype=dtype)
    base = torch.randn(full.shape, generator=rng, dtype=dtype)
    module = AffineTokenGuidance().to(dtype=dtype)
    assert sum(p.numel() for p in module.parameters()) == 2049
    assert all(torch.count_nonzero(p) == 0 for p in module.parameters())
    # Caller supplies the official active window or the official Full-only
    # branch. No alternate arithmetic is used to reconstruct that baseline.
    for official in (base + 1.78 * (full - base), full):
        result = module(full, base, official)
        assert result.shape == full.shape and result.dtype == dtype
        assert torch.equal(result.contiguous().view(torch.uint8), official.contiguous().view(torch.uint8))
    module(full, base, full).sum().backward()
    expected = (full - base).sum()
    torch.testing.assert_close(module.b.grad, expected)
    assert module.u.grad.abs().sum() > 0 and module.v.grad.abs().sum() > 0


def test_gate_and_state_gradients_match_finite_differences():
    module = AffineTokenGuidance(2).double()
    state = torch.tensor([[[[.2, -.4]], [[.7, .1]]]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([.03, -.02], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([-.04, .05], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(.01, dtype=torch.float64, requires_grad=True)

    def response(z, weight_full, weight_base, b):
        full, base = z.sin(), .6 * z.cos()
        official = base + 1.78 * (full - base)
        return functional_call(module, {"u": weight_full, "v": weight_base, "b": b}, (full, base, official))

    assert torch.autograd.gradcheck(response, (state, u, v, bias), eps=1e-6, atol=1e-7, rtol=1e-5)
    full = state.sin().clone().requires_grad_(True)
    base = state.cos().clone().requires_grad_(True)
    official = (state * 2).clone().requires_grad_(True)
    with torch.no_grad():
        module.u.copy_(u)
        module.v.copy_(v)
        module.b.copy_(bias)
    assert torch.autograd.gradcheck(module, (full, base, official), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_affine_gate_is_shared_per_token_without_time_inputs():
    module = AffineTokenGuidance(2).double()
    with torch.no_grad():
        module.u.copy_(torch.tensor([1., 2.]))
        module.v.copy_(torch.tensor([3., 4.]))
        module.b.fill_(5.)
    full = torch.tensor([[[[1., 2.]], [[3., 4.]]]], dtype=torch.float64)
    base = torch.tensor([[[[5., 6.]], [[7., 8.]]]], dtype=torch.float64)
    torch.testing.assert_close(module.gate(full, base), torch.tensor([[[[55., 65.]]]], dtype=torch.float64))
    assert list(inspect.signature(module.forward).parameters) == ["full", "base", "official"]
    assert list(inspect.signature(module.gate).parameters) == ["full", "base"]


def test_full_energy_known_1d_negative_value_and_repulsion_gradients():
    generated = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[1.], [3.]], dtype=torch.float64, requires_grad=True)
    objective = energy_distance_u_statistic(generated, real)
    # Cross mean=1.5, both within off-diagonal means=2. Unbiased U values
    # need not be nonnegative, even though population energy is nonnegative.
    torch.testing.assert_close(objective, torch.tensor(-1., dtype=torch.float64))
    gen_grad, real_grad = torch.autograd.grad(objective, (generated, real))
    torch.testing.assert_close(gen_grad, torch.tensor([[0.], [-1.]], dtype=torch.float64))
    torch.testing.assert_close(real_grad, torch.tensor([[1.], [0.]], dtype=torch.float64))
    attraction_grad = torch.tensor([[-1.], [0.]], dtype=torch.float64)
    torch.testing.assert_close(gen_grad - attraction_grad, torch.tensor([[1.], [-1.]], dtype=torch.float64))


def test_energy_unequal_counts_and_feature_gradients():
    generated = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[1.], [3.], [5.]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(energy_distance_u_statistic(generated, real), torch.zeros((), dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.autograd.gradcheck(energy_distance_u_statistic, (generated, real), eps=1e-6, atol=1e-7)
    # Direct cdist also gives finite zero gradients for exactly coincident rows.
    duplicated = torch.ones(3, 2, dtype=torch.float64, requires_grad=True)
    value = energy_distance_u_statistic(duplicated, torch.ones(2, 2, dtype=torch.float64))
    assert value == 0
    assert torch.equal(torch.autograd.grad(value, duplicated)[0], torch.zeros_like(duplicated))


def test_conditional_two_generated_one_real_value_and_both_gradients():
    generated = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[3.]], dtype=torch.float64)
    value = conditional_energy_training_objective(generated, torch.tensor([7, 7]), real, torch.tensor([7]))
    # 2*mean(3,1) - mean_offdiag(2,2) = 2. Real-real is omitted.
    torch.testing.assert_close(value, torch.tensor(2., dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(value, generated)[0], torch.tensor([[0.], [-2.]], dtype=torch.float64))


def test_conditional_classes_receive_equal_weights_despite_unequal_counts():
    generated = torch.tensor([[0.], [2.], [0.], [2.], [4.]], dtype=torch.float64)
    real = torch.tensor([[3.], [5.], [99.]], dtype=torch.float64)
    value = conditional_energy_training_objective(generated, torch.tensor([0, 0, 1, 1, 1]), real, torch.tensor([0, 1, 9]))
    # Class0=2; class1=2*mean(5,3,1)-mean_offdiag(2,4,2)=10/3.
    torch.testing.assert_close(value, torch.tensor(8/3, dtype=torch.float64))


def test_shape_count_dtype_and_label_errors():
    module = AffineTokenGuidance(2)
    full = torch.zeros(1, 2, 1, 1)
    for args in ((full, full[:, :1], full), (full, full, full.double()),
                 (full.flatten(1), full.flatten(1), full.flatten(1))):
        with pytest.raises(ValueError):
            module(*args)
    with pytest.raises(ValueError):
        AffineTokenGuidance(0)
    with pytest.raises(ValueError):
        module.double()(full, full, full)
    for gen, real in ((torch.ones(1, 2), torch.ones(2, 2)),
                      (torch.ones(2, 2), torch.ones(1, 2)),
                      (torch.ones(2, 2), torch.ones(2, 3)),
                      (torch.ones(2, 0), torch.ones(2, 0)),
                      (torch.ones(2, 2), torch.ones(2, 2).double()),
                      (torch.full((2, 2), float("nan")), torch.ones(2, 2))):
        with pytest.raises(ValueError):
            energy_distance_u_statistic(gen, real)
    with pytest.raises(TypeError):
        energy_distance_u_statistic(torch.ones(2, 2).long(), torch.ones(2, 2).long())
    with pytest.raises(ValueError, match="at least 2 generated"):
        conditional_energy_training_objective(torch.ones(2, 1), torch.tensor([0, 1]), torch.ones(2, 1), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="at least 1 real"):
        conditional_energy_training_objective(torch.ones(2, 1), torch.tensor([0, 0]), torch.ones(1, 1), torch.tensor([1]))
    with pytest.raises(ValueError, match="one label"):
        conditional_energy_training_objective(torch.ones(2, 1), torch.tensor([0]), torch.ones(1, 1), torch.tensor([0]))
    with pytest.raises(TypeError, match="integer"):
        conditional_energy_training_objective(torch.ones(2, 1), torch.zeros(2), torch.ones(1, 1), torch.tensor([0]))

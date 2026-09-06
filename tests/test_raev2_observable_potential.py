import pytest
import torch

from experiments.raev2_observable_potential import (
    ScalarGuidancePotential,
    observable_error_loss,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    # Tiny convolutions are much faster with one thread; restore global state.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_potential(*, nonzero=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(8101)
        model = ScalarGuidancePotential(
            latent_channels=2, spatial_size=(2, 2), hidden_channels=3, num_classes=4,
        ).double()
    if nonzero:
        with torch.no_grad():
            model.readout.weight.copy_(torch.tensor([.2, -.15, .1], dtype=torch.float64).reshape(1, 3, 1, 1))
    return model


def inputs(batch=2):
    rng = torch.Generator().manual_seed(8102)
    z = torch.randn(batch, 2, 2, 2, dtype=torch.float64, generator=rng)
    time = torch.linspace(.2, .8, batch, dtype=torch.float64)
    labels = torch.arange(batch, dtype=torch.long)
    return z, time, labels


def test_frozen_production_structure_and_zero_correction_preserve_official_mix():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(8103)
        model = ScalarGuidancePotential()
    assert sum(parameter.numel() for parameter in model.parameters()) == 608000
    assert model.readout.bias is None
    assert torch.count_nonzero(model.position) == torch.count_nonzero(model.readout.weight) == 0
    assert model.input_conv.kernel_size == (1, 1)
    assert [layer.kernel_size for layer in model.hidden_convs] == [(3, 3), (3, 3)]
    assert not any(isinstance(layer, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
                   for layer in model.modules())
    rng = torch.Generator().manual_seed(8104)
    z = torch.randn(2, 1024, 16, 16, generator=rng)
    full = torch.randn(z.shape, generator=rng)
    base = torch.randn(z.shape, generator=rng)
    with torch.no_grad():
        correction = model.clean_correction(z, torch.tensor([.8, .05]), torch.tensor([2, 999]))
    assert not z.requires_grad and not correction.requires_grad
    assert torch.count_nonzero(correction) == 0
    for official in (base + 1.78 * (full - base), full):
        guided = official + correction
        assert torch.equal(guided.contiguous().view(torch.uint8), official.contiguous().view(torch.uint8))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_input_gradient_matches_directional_finite_difference_and_gradcheck():
    model = tiny_potential(nonzero=True)
    z, time, labels = inputs(1)
    z.requires_grad_()
    correction = model.clean_correction(z, time, labels, create_graph=True)
    direction = torch.linspace(-.4, .7, z.numel(), dtype=z.dtype).reshape_as(z)
    epsilon = 1e-5
    finite_difference = (model(z + epsilon * direction, time, labels)
                         - model(z - epsilon * direction, time, labels)) / (2 * epsilon)
    torch.testing.assert_close((correction * direction).flatten(1).sum(1), finite_difference,
                               atol=1e-10, rtol=1e-7)
    assert torch.autograd.gradcheck(
        lambda state: model.clean_correction(state, time, labels, create_graph=True),
        (z,), eps=1e-6, atol=1e-8, rtol=1e-5,
    )


def test_training_loss_reaches_zero_readout_through_input_derivative():
    model = tiny_potential()
    z, time, labels = inputs()
    with torch.no_grad():
        # enable_grad inside the method must also work for create_graph=True.
        correction = model.clean_correction(z, time, labels, create_graph=True)
    assert correction.requires_grad and not z.requires_grad
    target, official = z.cos(), z.sin()
    loss = observable_error_loss(correction, target, official)
    loss.backward()
    assert model.readout.weight.grad is not None
    assert torch.isfinite(model.readout.weight.grad).all()
    assert model.readout.weight.grad.abs().sum() > 0
    assert model.input_conv.weight.grad is not None
    assert torch.count_nonzero(model.input_conv.weight.grad) == 0

    index = (0, 0, 0, 0)
    analytic = model.readout.weight.grad[index].clone()
    epsilon = 1e-5
    with torch.no_grad():
        model.readout.weight[index] = epsilon
        plus = observable_error_loss(model.clean_correction(z, time, labels), target, official)
        model.readout.weight[index] = -epsilon
        minus = observable_error_loss(model.clean_correction(z, time, labels), target, official)
        model.readout.weight[index] = 0
    torch.testing.assert_close(analytic, (plus - minus) / (2 * epsilon), atol=1e-11, rtol=1e-7)


def test_nonzero_readout_allows_hidden_parameter_training_gradients():
    model = tiny_potential(nonzero=True)
    z, time, labels = inputs()
    correction = model.clean_correction(z, time, labels, create_graph=True)
    observable_error_loss(correction, z.cos(), z.sin()).backward()
    for name in ("input_conv.weight", "hidden_convs.0.weight", "position",
                 "class_embedding.weight", "time_mlp.0.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_correction_hessian_is_symmetric_and_preserves_upstream_input_graph():
    model = tiny_potential(nonzero=True)
    z, time, labels = inputs(1)
    z.requires_grad_()
    jacobian = torch.autograd.functional.jacobian(
        lambda state: model.clean_correction(state, time, labels, create_graph=True), z,
    ).reshape(z.numel(), z.numel())
    torch.testing.assert_close(jacobian, jacobian.T, atol=1e-12, rtol=1e-10)
    assert jacobian.norm() > 0

    upstream = z.detach().clone().requires_grad_()
    state = 1.7 * upstream
    correction = model.clean_correction(state, time, labels, create_graph=True)
    state_gradient, upstream_gradient = torch.autograd.grad(correction.sum(), (state, upstream))
    torch.testing.assert_close(upstream_gradient, 1.7 * state_gradient, atol=1e-12, rtol=1e-10)
    assert upstream_gradient.norm() > 0


def test_batch_separability_in_value_and_derivatives():
    model = tiny_potential(nonzero=True)
    z, time, labels = inputs()
    z.requires_grad_()
    value = model(z, time, labels)
    correction = model.clean_correction(z, time, labels, create_graph=True)
    for i in range(len(z)):
        torch.testing.assert_close(value[i:i+1], model(z[i:i+1], time[i:i+1], labels[i:i+1]),
                                   atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(correction[i:i+1], model.clean_correction(z[i:i+1], time[i:i+1], labels[i:i+1]),
                                   atol=1e-12, rtol=1e-10)
    cross_gradient, = torch.autograd.grad(correction[0].sum(), z)
    assert torch.count_nonzero(cross_gradient[1]) == 0
    torch.testing.assert_close(model(z, .4, labels), model(z, torch.full_like(time, .4), labels))


def test_loss_dimension_mean_importance_weights_and_detached_targets():
    correction = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[2., 1.], [4., -2.]], dtype=torch.float64, requires_grad=True)
    official = torch.tensor([[.5, -.5], [1., 1.]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2., .5], dtype=torch.float64, requires_grad=True)
    value = observable_error_loss(correction, target, official, sample_weights=weights)
    # Per-image losses are -1 and 8.5; importance averaging gives 1.125.
    torch.testing.assert_close(value, torch.tensor(1.125, dtype=torch.float64))
    value.backward()
    torch.testing.assert_close(correction.grad, (correction.detach() - (target.detach() - official.detach()))
                               * weights.detach()[:, None] / correction.numel())
    assert target.grad is None and official.grad is None and weights.grad is None
    half_loss = observable_error_loss(correction.detach().bfloat16(), target.detach().bfloat16(),
                                      official.detach().bfloat16())
    assert half_loss.dtype == torch.float32


def test_inference_mode_is_explicitly_unsupported():
    model = tiny_potential()
    z, time, labels = inputs()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        model.clean_correction(z, time, labels)
    with torch.inference_mode():
        inference_z = z.clone()
    with pytest.raises(RuntimeError, match="inference tensors"):
        model.clean_correction(inference_z, time, labels)


def test_input_and_weight_contracts():
    model = tiny_potential()
    z, time, labels = inputs()
    for bad_z, bad_t, bad_y in ((z[:, :, :1], time, labels), (z.float(), time, labels),
                                (z, time[:, None], labels), (z, -0.1, labels),
                                (z, float("nan"), labels), (z, time, torch.tensor([0, 4]))):
        with pytest.raises(ValueError):
            model(bad_z, bad_t, bad_y)
    with pytest.raises(TypeError, match="torch.long"):
        model(z, time, labels.float())
    for weights in (torch.ones(2, 1), torch.tensor([1., -1.]), torch.tensor([1., float("nan")])):
        with pytest.raises(ValueError):
            observable_error_loss(z, z, z, sample_weights=weights)
    with pytest.raises(ValueError):
        observable_error_loss(z, z[:, :, :1], z)

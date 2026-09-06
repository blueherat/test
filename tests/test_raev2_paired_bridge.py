import pytest
import torch

from experiments.raev2_paired_bridge import (
    PairedBridgeField, bridge_midpoint, construct_bridge, native_euler,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_field(*, nonzero=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(92301)
        field = PairedBridgeField(latent_channels=3, spatial_size=(2, 2), num_classes=4)
    if nonzero:
        with torch.no_grad():
            field.readout.weight.copy_(torch.tensor([
                [0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.1, 0.2, 0.5],
            ]).reshape(3, 3, 1, 1))
    return field


def inputs():
    clean = torch.linspace(-1.1, 1.3, 24).reshape(2, 3, 2, 2)
    noise = torch.linspace(0.7, -0.5, 24).reshape_as(clean)
    g = 0.6 * clean + 0.1
    return clean, noise, g, torch.tensor([0.8, 0.03]), torch.tensor([0.6, 0.01]), torch.tensor([0, 2])


def test_fixed_production_structure_and_parameter_count():
    field = PairedBridgeField()
    assert sum(p.numel() for p in field.parameters()) == 3652224
    assert field.class_embedding.embedding_dim == 64
    assert field.condition_mlp[0].in_features == 192
    assert field.condition_mlp[0].out_features == 128
    assert field.condition_mlp[2].out_features == 1024
    assert len(field.blocks) == 2
    for block in field.blocks:
        assert block.depthwise.groups == 1024
        assert block.depthwise.kernel_size == (3, 3)
        assert block.pointwise.in_channels == block.pointwise.out_channels == 1024
    assert field.readout.in_channels == field.readout.out_channels == 1024
    assert torch.count_nonzero(field.position) == 0
    assert torch.count_nonzero(field.readout.weight) == torch.count_nonzero(field.readout.bias) == 0
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.LayerNorm, torch.nn.BatchNorm2d))
                   for m in field.modules())


def test_native_euler_exact_arithmetic_and_active_floor_target():
    clean, noise, g, t, s, _ = inputs()
    bridge = construct_bridge(clean, noise, g, t, s, torch.tensor([0.25, 0.75]))
    b = lambda x: x[:, None, None, None]
    z = (1 - b(t)) * clean + b(t) * noise
    expected_y = z - b(t - s) * ((z - g) / b(t.clamp_min(0.05)))
    expected_w = (1 - b(s)) * clean + b(s) * noise
    assert torch.equal(bridge["z"], z)
    assert torch.equal(bridge["Y"], expected_y)
    assert torch.equal(bridge["W"], expected_w)
    assert torch.equal(bridge["R"], expected_w - expected_y)
    torch.testing.assert_close(bridge["beta"], torch.tensor([0.25, 0.4]))
    torch.testing.assert_close(bridge["target"][0], (clean - g)[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(bridge["target"][1], (clean - g)[1], atol=1e-5, rtol=1e-5)
    expected_active = 1.02 * clean[1] - 0.02 * noise[1] - g[1]
    torch.testing.assert_close(bridge["target"][1], expected_active, atol=1e-6, rtol=1e-5)
    assert torch.equal(construct_bridge(clean, noise, g, t, s, 0)["U"], expected_y)
    assert torch.equal(construct_bridge(clean, noise, g, t, s, 1)["U"], expected_w)


def test_zero_field_is_identity_in_both_midpoint_modes():
    field = tiny_field()
    clean, noise, g, t, s, labels = inputs()
    bridge = construct_bridge(clean, noise, g, t, s, 0.4)
    with torch.inference_mode():
        assert torch.count_nonzero(field(bridge["U"], t, s, 0.4, labels)) == 0
        for locked in (False, True):
            result = bridge_midpoint(field, bridge["Y"], t, s, labels, tau_locked_zero=locked)
            assert torch.equal(result.contiguous().view(torch.uint8),
                               bridge["Y"].contiguous().view(torch.uint8))


def test_zero_readout_receives_parameter_gradient_then_trunk_trains():
    field = tiny_field()
    clean, noise, g, t, s, labels = inputs()
    bridge = construct_bridge(clean, noise, g, t, s, 0.6)
    prediction = field(bridge["U"], t, s, 0.6, labels)
    (prediction - bridge["target"]).square().mean().backward()
    assert field.readout.weight.grad is not None
    assert torch.isfinite(field.readout.weight.grad).all()
    assert field.readout.weight.grad.abs().sum() > 0
    assert field.readout.bias.grad.abs().sum() > 0
    assert torch.count_nonzero(field.blocks[0].pointwise.weight.grad) == 0
    field = tiny_field(nonzero=True)
    (field(bridge["U"], t, s, 0.6, labels) - bridge["target"]).square().mean().backward()
    for name in ("blocks.0.pointwise.weight", "blocks.1.depthwise.weight",
                 "class_embedding.weight", "condition_mlp.0.weight", "position"):
        grad = dict(field.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_batch_separation_and_current_state_derivative():
    field = tiny_field(nonzero=True).double()
    u = inputs()[0].double().requires_grad_()
    t, s, labels = inputs()[3:]
    t, s = t.double(), s.double()
    tau = torch.tensor([0.2, 0.7], dtype=torch.float64)
    result = field(u, t, s, tau, labels)
    for i in range(2):
        expected = field(u[i:i+1], t[i:i+1], s[i:i+1], tau[i:i+1], labels[i:i+1])
        torch.testing.assert_close(result[i:i+1], expected, atol=1e-12, rtol=1e-10)
    gradient, = torch.autograd.grad(result[0].sum(), u)
    assert gradient[0].abs().sum() > 0
    assert torch.count_nonzero(gradient[1]) == 0


def test_full_channel_path_can_represent_nonsymmetric_linear_map():
    field = tiny_field()
    with torch.no_grad():
        for p in field.parameters():
            p.zero_()
        matrix = torch.tensor([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])
        field.readout.weight.copy_(matrix[:, :, None, None])
    u = inputs()[0]
    expected = torch.einsum("ij,bjhw->bihw", matrix, u)
    torch.testing.assert_close(field(u, 0.8, 0.6, 0.3, inputs()[-1]), expected)
    changed = u.clone()
    changed[:, 2] += 1
    difference = field(changed, 0.8, 0.6, 0.3, inputs()[-1]) - expected
    torch.testing.assert_close(difference[:, 2], torch.ones_like(difference[:, 2]))
    assert not torch.equal(matrix, matrix.T)  # A vector field need not be a potential gradient.


def test_gaussian_zero_initial_velocity_has_nonzero_midpoint_response():
    y = torch.tensor([-1.5, 0.0, 2.0]).reshape(3, 1, 1, 1)
    t, s = 0.8, 0.6
    a, b = 2.0, 0.4  # Independent Y,R variances.
    calls = []

    def gaussian(u, t, s, tau, labels):
        calls.append((u.clone(), tau.clone()))
        beta = (t - s) / t.clamp_min(0.05)
        coefficient = tau * b / (a + tau.square() * b)
        return (coefficient / beta)[:, None, None, None] * u

    result = bridge_midpoint(gaussian, y, t, s, torch.zeros(3, dtype=torch.long))
    factor = 1 + 0.5 * b / (a + 0.25 * b)
    torch.testing.assert_close(result, factor * y)
    assert len(calls) == 2 and torch.count_nonzero(calls[0][1]) == 0
    assert torch.equal(calls[1][0], y)  # k1=0, nevertheless second velocity is nonzero.
    assert not torch.equal(result, y)
    exact_factor = (1 + b / a) ** 0.5
    assert abs(factor - exact_factor) > 1e-6  # Finite midpoint is not exact oracle transport.
    locked = bridge_midpoint(gaussian, y, t, s, torch.zeros(3, dtype=torch.long), tau_locked_zero=True)
    assert torch.equal(locked, y)


def test_affine_midpoint_uses_current_midpoint_and_locks_only_tau():
    y = torch.tensor([-1.0, 0.5]).reshape(2, 1, 1, 1)
    t, s = torch.tensor([0.8, 0.03]), torch.tensor([0.6, 0.01])
    labels = torch.tensor([0, 1])
    a, b, d = 0.4, -0.2, 0.6
    for locked in (False, True):
        calls = []

        def affine(u, t, s, tau, labels):
            calls.append((u.clone(), tau.clone()))
            beta = ((t - s) / t.clamp_min(0.05))[:, None, None, None]
            return (a * u + b + d * tau[:, None, None, None]) / beta

        result = bridge_midpoint(affine, y, t, s, labels, tau_locked_zero=locked)
        expected_mid = y + 0.5 * (a * y + b)
        expected = y + a * expected_mid + b + (0 if locked else 0.5 * d)
        torch.testing.assert_close(calls[1][0], expected_mid)
        torch.testing.assert_close(result, expected)
        assert not torch.equal(calls[1][0], y)
        assert torch.equal(calls[1][1], torch.full_like(t, 0 if locked else 0.5))


@pytest.mark.parametrize("t,s,tau", [
    (0., 0., 0.5), (0.5, 0.6, 0.5), (1.1, 0.2, 0.5),
    (0.5, -0.1, 0.5), (float("nan"), 0.2, 0.5),
    (0.5, 0.2, float("inf")), (0.5, 0.2, -0.1), (0.5, 0.2, 1.1),
])
def test_invalid_times(t, s, tau):
    field = tiny_field()
    clean, noise, g, _, _, labels = inputs()
    with pytest.raises(ValueError):
        field(clean, t, s, tau, labels)
    with pytest.raises(ValueError):
        construct_bridge(clean, noise, g, t, s, tau)


def test_shape_dtype_label_and_callable_contracts():
    field = tiny_field()
    clean, noise, g, t, s, labels = inputs()
    for u, bad_labels in ((clean[:, :, :1], labels), (clean.double(), labels),
                          (clean, torch.tensor([0, 4])), (clean, labels[:, None])):
        with pytest.raises(ValueError):
            field(u, t, s, 0.4, bad_labels)
    with pytest.raises(TypeError):
        field(clean, t, s, 0.4, labels.float())
    with pytest.raises(ValueError):
        native_euler(clean.double(), g.double(), t, s)
    with pytest.raises(ValueError):
        construct_bridge(clean, noise[:1], g, t, s, 0.4)
    with pytest.raises(ValueError):
        native_euler(clean, g, t[:, None], s)
    with pytest.raises(ValueError):
        bridge_midpoint(lambda *args: torch.zeros(1), clean, t, s, labels)
    with pytest.raises(TypeError):
        bridge_midpoint(field, clean, t, s, labels, tau_locked_zero=1)

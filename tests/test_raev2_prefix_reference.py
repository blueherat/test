"""Check the audit reference's precision semantics on the actual block family.

A small untrained architecture is a numerical fixture, not a new experimental
guidance head or quality sample. In particular this catches the native mask's
float64 overflow and RMSNorm's otherwise hidden float32 cast.
"""
from types import SimpleNamespace
import torch
from experiments.audit_raev2_prefix_ratio64k_gradient import PrefixReference
from experiments.raev2_prefix_ratio_features import prefix_features
from stage2.models.DDT import DiTwDDTHeadIG


def test_reference_is_double_and_matches_native_input_gradient():
    torch.manual_seed(202609107)
    native = DiTwDDTHeadIG(input_size=4, in_channels=4, hidden_size=[32, 64],
        depth=[8, 2], num_heads=[4, 4], base_model_depth=8,
        cond_arch=SimpleNamespace(num_t_tokens=4, num_c_tokens=8)).eval().requires_grad_(False)
    reference = PrefixReference(native)
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    times = torch.tensor([.9987, .4])
    labels = torch.tensor([4, 971])
    coefficient = torch.randn(64, dtype=torch.float64)
    f32 = prefix_features(native, x, times, labels).double()@coefficient
    gradient32, = torch.autograd.grad(f32.sum(), x)
    xd = x.detach().double().requires_grad_(True)
    f64 = prefix_features(reference, xd, times.double(), labels)@coefficient
    gradient64, = torch.autograd.grad(f64.sum(), xd)
    assert f64.dtype == gradient64.dtype == torch.float64
    assert torch.isfinite(f64).all() and torch.isfinite(gradient64).all()
    torch.testing.assert_close(f32, f64, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gradient32.double(), gradient64, atol=2e-6, rtol=1e-4)
    direction = gradient64/gradient64.flatten(1).norm(dim=1)[:, None, None, None]
    with torch.no_grad():
        hi = prefix_features(reference, xd+.0001*direction, times.double(), labels)@coefficient
        lo = prefix_features(reference, xd-.0001*direction, times.double(), labels)@coefficient
    torch.testing.assert_close((hi-lo)/.0002, gradient64.flatten(1).norm(dim=1), atol=1e-7, rtol=1e-7)

import pytest
import torch

from experiments.audit_raev2_guidance_normal_noise import (
    affine_geometry, domain_metrics, mirror_normal_noise, normal_part,
    official_guided, project_clean,
)


def geometry():
    generator = torch.Generator().manual_seed(30)
    mean = torch.randn(5, 2, 3, generator=generator, dtype=torch.float64)
    variance = torch.rand(5, 2, 3, generator=generator, dtype=torch.float64)+0.2
    unit, anchor = affine_geometry(mean, variance)
    return mean, variance, unit, anchor


def test_projection_enforces_actual_denormalized_channel_sum_and_pythagoras():
    mean, variance, unit, anchor = geometry()
    generator = torch.Generator().manual_seed(31)
    value = torch.randn(4, 5, 2, 3, generator=generator, dtype=torch.float64)
    projected = project_clean(value, unit, anchor)
    raw = projected*(variance+1e-5).sqrt()+mean
    torch.testing.assert_close(raw.sum(1), torch.zeros_like(raw[:, 0]), atol=2e-15, rtol=0)
    target = project_clean(torch.randn(value.shape, generator=generator, dtype=value.dtype), unit, anchor)
    torch.testing.assert_close((value-target).square().sum(),
                               (projected-target).square().sum()+(value-projected).square().sum())
    # This would fail for the incorrect unweighted normalized-channel centering.
    assert projected.mean(1).abs().max() > 0.1


def test_reflection_involution_preserves_tangent_and_flips_only_centered_normal():
    _, _, unit, anchor = geometry()
    value = torch.randn(2, 5, 2, 3, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    for time in (0., .37, 1.):
        reflected = mirror_normal_noise(value, time, unit, anchor)
        torch.testing.assert_close(mirror_normal_noise(reflected, time, unit, anchor), value)
        torch.testing.assert_close(normal_part(reflected-(1-time)*anchor, unit),
                                   -normal_part(value-(1-time)*anchor, unit))
        torch.testing.assert_close(reflected-normal_part(reflected, unit),
                                   value-normal_part(value, unit))


def test_even_predictor_has_zero_symmetry_change_and_projection_removes_known_error():
    _, _, unit, anchor = geometry()
    clean = project_clean(torch.zeros(2, 5, 2, 3, dtype=torch.float64), unit, anchor).float()
    full = (clean.double()+0.5*unit).float()
    base = (clean.double()+0.1*unit).float()
    metrics = domain_metrics(clean, clean, full, base, full, base, .5, unit, anchor)
    assert torch.all(metrics["full_mirror_tangent_change_energy"] == 0)
    assert torch.all(metrics["full_average_projected_reference_error_gain"] > 0)
    torch.testing.assert_close(metrics["gap_normal_fraction"], torch.ones(2, dtype=torch.float64))
    assert torch.max(metrics["full_average_projected_reference_error_energy"]) < 1e-14


def test_official_guidance_exact_interval_and_fp32_contract():
    full = torch.tensor([1.25], dtype=torch.float32)
    base = torch.tensor([.125], dtype=torch.float32)
    assert torch.equal(official_guided(full, base, .1), full+.78*(full-base))
    assert torch.equal(official_guided(full, base, .099), full)
    with pytest.raises(ValueError):
        official_guided(full.to(torch.bfloat16), base.to(torch.bfloat16), .5)

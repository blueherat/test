from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_radius_guidance import (  # noqa: E402
    GROUPINGS,
    MODES,
    decompose_depth_direction,
    radius_guided_clean,
)


def predictions() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(47)
    return (
        torch.randn(3, 5, 2, 4, generator=generator),
        torch.randn(3, 5, 2, 4, generator=generator),
    )


@pytest.mark.parametrize("grouping,dims", [("token", (1,)), ("global", (1, 2, 3))])
def test_split_is_orthogonal_and_energy_additive(grouping, dims) -> None:
    full, base = predictions()
    split = decompose_depth_direction(full, base, grouping=grouping)
    torch.testing.assert_close(split.radial + split.tangent, full - base)
    torch.testing.assert_close(
        (full * split.tangent).sum(dim=dims),
        torch.zeros_like(full.sum(dim=dims)),
        atol=4e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        split.direction_squared_norm,
        (split.radial.square() + split.tangent.square()).sum(dim=dims, keepdim=True),
    )


@pytest.mark.parametrize("grouping,dims", [("token", (1,)), ("global", (1, 2, 3))])
def test_retraction_preserves_full_radius_and_ordinary_direction(grouping, dims) -> None:
    full, base = predictions()
    actual, telemetry = radius_guided_clean(
        full, base, guidance_scale=1.78, mode="retracted",
        grouping=grouping, return_telemetry=True,
    )
    ordinary = radius_guided_clean(full, base, guidance_scale=1.78)
    torch.testing.assert_close(actual.square().sum(dims), full.square().sum(dims))
    actual_unit = actual / actual.square().sum(dims, keepdim=True).sqrt()
    ordinary_unit = ordinary / ordinary.square().sum(dims, keepdim=True).sqrt()
    torch.testing.assert_close(actual_unit, ordinary_unit)
    torch.testing.assert_close(
        telemetry["relative_radius_ratio"],
        torch.ones_like(telemetry["relative_radius_ratio"]),
    )
    # The removed displacement is normal to the sphere at its closest point.
    displacement = ordinary - actual
    torch.testing.assert_close(
        displacement / actual,
        (ordinary.square().sum(dims, keepdim=True).sqrt()
         / actual.square().sum(dims, keepdim=True).sqrt() - 1).expand_as(actual),
        atol=2e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_tangent_radius_increase_has_only_quadratic_gain(grouping) -> None:
    full, base = predictions()
    dims = (1,) if grouping == "token" else (1, 2, 3)
    split = decompose_depth_direction(full, base, grouping=grouping)
    actual = radius_guided_clean(
        full, base, guidance_scale=1.78, mode="tangent", grouping=grouping,
    )
    torch.testing.assert_close(
        actual.square().sum(dims, keepdim=True),
        split.reference_squared_norm + 0.78**2 * split.tangent.square().sum(dims, keepdim=True),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_zero_reference_falls_back_to_ordinary_per_group(mode, grouping) -> None:
    full, base = predictions()
    if grouping == "token":
        full[:, :, 0, 0] = 0
        mask = torch.zeros_like(full, dtype=torch.bool)
        mask[:, :, 0, 0] = True
    else:
        full[0] = 0
        mask = torch.zeros_like(full, dtype=torch.bool)
        mask[0] = True
    actual, telemetry = radius_guided_clean(
        full, base, guidance_scale=1.78, mode=mode,
        grouping=grouping, return_telemetry=True,
    )
    torch.testing.assert_close(actual[mask], (-0.78 * base)[mask], rtol=0, atol=0)
    assert all(torch.isfinite(value).all() for value in telemetry.values())


def test_zero_retraction_candidate_chooses_full_radius_anchor() -> None:
    full = torch.tensor([[[[1.0]], [[2.0]]]])
    base = 2 * full  # beta = 2 gives ordinary = 0 exactly.
    guided, telemetry = radius_guided_clean(
        full, base, guidance_scale=2, mode="retracted", return_telemetry=True,
    )
    assert torch.equal(guided, full)
    assert torch.equal(telemetry["degenerate_ordinary"], torch.ones(1, 1, 1, 1))


@pytest.mark.parametrize("mode", MODES)
def test_scale_one_is_exact_fp32_full_even_for_bf16_inputs(mode) -> None:
    full, base = (value.bfloat16() for value in predictions())
    actual = radius_guided_clean(full, base, guidance_scale=1, mode=mode)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, full.float())


def test_ordinary_uses_fp32_subtraction_and_matches_clean_anchor_exactly() -> None:
    full, base = (value.bfloat16() for value in predictions())
    actual = radius_guided_clean(full, base, guidance_scale=1.78)
    expected = full.float() + (1.78 - 1) * (full.float() - base.float())
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    unchecked = radius_guided_clean(full, base, guidance_scale=1.78, check_finite=False)
    assert torch.equal(actual, unchecked)


def test_identical_heads_leave_no_guidance_direction_or_nonfinite_telemetry() -> None:
    full, _ = predictions()
    full[0] = 0
    for mode in MODES:
        actual, telemetry = radius_guided_clean(
            full, full, guidance_scale=1.78, mode=mode, return_telemetry=True,
        )
        torch.testing.assert_close(actual, full)
        assert not telemetry["radial_energy_fraction"].any()
        assert all(torch.isfinite(value).all() for value in telemetry.values())


def test_validation_rejects_nonfinite_and_ambiguous_inputs() -> None:
    full, base = predictions()
    for scale in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            radius_guided_clean(full, base, guidance_scale=scale)
    for value in (float("nan"), float("inf")):
        bad = full.clone()
        bad[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            radius_guided_clean(bad, base, guidance_scale=1.78)
    with pytest.raises(ValueError, match="grouping"):
        radius_guided_clean(full, base, guidance_scale=1.78, grouping="spatial")
    with pytest.raises(ValueError, match="mode"):
        radius_guided_clean(full, base, guidance_scale=1.78, mode="unknown")
    with pytest.raises(ValueError, match="share"):
        radius_guided_clean(full.flatten(1), base.flatten(1), guidance_scale=1.78)

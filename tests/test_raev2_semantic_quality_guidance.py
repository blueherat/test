from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_semantic_quality_guidance import (
    MODES, semantic_orthogonal_residual, semantic_quality_clean,
)
from experiments.sample_raev2_semantic_quality_guidance import evaluate_four_corners, parse_args, scale_at_time


def corners():
    # Dyadic numbers make the corner identities exact in FP32.
    return tuple(torch.tensor([[value, value * 2, value * -3]]) for value in (1.0, 0.25, -0.5, 2.0))


@pytest.mark.parametrize("ig,cfg,index", ((1, 1, 0), (0, 1, 1), (1, 0, 2), (0, 0, 3)))
def test_bilinear_recovers_all_four_corners(ig, cfg, index):
    values = corners()
    actual = semantic_quality_clean(*values, ig_scale=ig, cfg_scale=cfg, mode="bilinear")
    assert torch.equal(actual, values[index])


def test_bilinear_mixed_difference_and_axis_anchors():
    fc, bc, fu, bu = corners()
    arguments = dict(ig_scale=1.5, cfg_scale=1.25)
    additive = semantic_quality_clean(fc, bc, fu, bu, mode="combined", **arguments)
    bilinear = semantic_quality_clean(fc, bc, fu, bu, mode="bilinear", **arguments)
    assert torch.equal(bilinear - additive, 0.5 * 0.25 * (fc - bc - fu + bu))
    for mode in ("combined", "bilinear", "semantic_orthogonal"):
        clean = semantic_quality_clean(fc, bc, fu, bu, mode=mode, ig_scale=1.5, cfg_scale=1.0)
        assert torch.equal(clean, fc + 0.5 * (fc - bc))
    for mode in ("combined", "bilinear", "full_cfg"):
        clean = semantic_quality_clean(fc, bc, fu, bu, mode=mode, ig_scale=1.0, cfg_scale=1.25)
        assert torch.equal(clean, fc + 0.25 * (fc - fu))


def test_orthogonal_residual_handles_zero_parallel_and_transverse_directions():
    ig = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    cfg = torch.tensor([[3.0, 4.0], [5.0, 0.0], [3.0, 4.0]])
    residual = semantic_orthogonal_residual(ig, cfg)
    assert torch.equal(residual, torch.tensor([[3.0, 4.0], [0.0, 0.0], [0.0, 4.0]]))
    assert torch.equal((residual * ig).sum(dim=1), torch.zeros(3))
    assert bool((residual.norm(dim=1) <= cfg.norm(dim=1)).all())
    fc = torch.ones_like(ig)
    clean, telemetry = semantic_quality_clean(
        fc, fc - ig, fc - cfg, torch.zeros_like(fc),
        ig_scale=1.5, cfg_scale=2.0, mode="semantic_orthogonal", return_telemetry=True,
    )
    assert torch.equal(clean, fc + 0.5 * ig + residual)
    assert all(bool(torch.isfinite(value).all()) for value in telemetry.values())


@pytest.mark.parametrize("mode", MODES)
def test_bfloat16_inputs_are_subtracted_and_combined_in_fp32(mode):
    values = tuple(torch.tensor([[value, -value]], dtype=torch.bfloat16) for value in (1.0, 0.00390625, -0.0078125, 0.125))
    expected = semantic_quality_clean(*(value.float() for value in values), ig_scale=1.78, cfg_scale=1.3, mode=mode)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = semantic_quality_clean(*values, ig_scale=1.78, cfg_scale=1.3, mode=mode)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_four_corner_query_has_fixed_conditional_null_layout():
    state = torch.arange(12.0).reshape(2, 2, 3)
    times, labels = torch.tensor([0.8, 0.8]), torch.tensor([3, 9])
    calls = []

    def model(x, t, context, attn_mask):
        calls.append((x, t, context, attn_mask))
        full = x + context[:, None, None]
        return full, full + 10000

    fc, bc, fu, bu = evaluate_four_corners(model, state, times, labels, null_label=1000)
    assert len(calls) == 1
    assert torch.equal(calls[0][0], torch.cat((state, state)))
    assert torch.equal(calls[0][1], torch.cat((times, times)))
    assert torch.equal(calls[0][2], torch.tensor([3, 9, 1000, 1000]))
    assert torch.equal(fc, state + labels[:, None, None])
    assert torch.equal(fu, state + 1000)
    assert torch.equal(bc, fc + 10000)
    assert torch.equal(bu, fu + 10000)


def test_independent_axis_windows_and_invalid_scale(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--mode", "combined", "--cfg-min-time", "0.4"])
    assert args.sample_count == 1000 and args.num_steps == 100
    assert scale_at_time(1.78, args.ig_min_time, args.ig_max_time, 0.2) == 1.78
    assert scale_at_time(1.3, args.cfg_min_time, args.cfg_max_time, 0.2) == 1.0
    assert scale_at_time(1.3, 0.4, 1.0, 0.4) == 1.3
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--mode", "combined", "--cfg-scale", "nan"])

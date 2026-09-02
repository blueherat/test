from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import experiments.sample_imagenet100_sit_foresight_fixed_point as sampler


def test_internal_guidance_fields_select_scheduled_head_and_share_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_model_velocity(
        _model,
        _semantics,
        state: torch.Tensor,
        _time: torch.Tensor,
        _labels: torch.Tensor,
        *,
        autocast_dtype,
    ) -> torch.Tensor:
        assert autocast_dtype is None
        return state + 10.0

    def fake_source_with_heads(
        _model,
        state: torch.Tensor,
        _times: torch.Tensor,
        _labels: torch.Tensor,
        *,
        heads,
        source_semantics,
    ):
        assert source_semantics == "velocity"
        assert set(heads) == {"depth4_v", "depth10_v"}
        return (
            state + 10.0,
            {
                "depth4_v": torch.full_like(state, 2.0),
                "depth10_v": torch.full_like(state, 6.0),
            },
            {},
        )

    monkeypatch.setattr(sampler, "_model_velocity", fake_model_velocity)
    monkeypatch.setattr(
        sampler,
        "evaluate_source_with_heads",
        fake_source_with_heads,
    )
    heads = {
        "depth4_v": SimpleNamespace(
            prediction_target="velocity", depth=4, checkpoint="d4.pt"
        ),
        "depth10_v": SimpleNamespace(
            prediction_target="velocity", depth=10, checkpoint="d10.pt"
        ),
    }
    fields = sampler.build_ig_fields(
        torch.nn.Identity(),
        torch.tensor([3, 7]),
        strong_semantics="velocity",
        heads=heads,
        depths=(4, 10),
        gamma=0.45,
        autocast_dtype=None,
    )
    state = torch.zeros(2, 1, 1, 1)

    early = fields.guided(torch.tensor(0.25), state)
    late = fields.guided(torch.tensor(0.75), state)
    torch.testing.assert_close(early, torch.full_like(state, 13.6))
    torch.testing.assert_close(late, torch.full_like(state, 11.8))
    torch.testing.assert_close(
        fields.target(torch.tensor(0.25), state),
        torch.full_like(state, 10.0),
    )
    torch.testing.assert_close(
        fields.reference(torch.tensor(0.75), state),
        torch.full_like(state, 6.0),
    )
    assert fields.counters == {
        "strong_backbone_forwards": 1,
        "weak_probe_backbone_forwards": 1,
        "guided_shared_backbone_forwards": 2,
    }
    assert fields.metadata["provider"] == "scheduled_internal_guidance"
    assert fields.metadata["reference_choice"] == "weak"


def test_parse_ig_depths() -> None:
    assert sampler.parse_depths("4,10") == (4, 10)
    with pytest.raises(sampler.argparse.ArgumentTypeError):
        sampler.parse_depths("4,4")


def test_best_depth4_piecewise_gamma_is_applied_to_guided_and_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_model_velocity(
        _model,
        _semantics,
        state: torch.Tensor,
        _time: torch.Tensor,
        _labels: torch.Tensor,
        *,
        autocast_dtype,
    ) -> torch.Tensor:
        assert autocast_dtype is None
        return torch.full_like(state, 10.0)

    def fake_source_with_heads(
        _model,
        state: torch.Tensor,
        _times: torch.Tensor,
        _labels: torch.Tensor,
        *,
        heads,
        source_semantics,
    ):
        assert source_semantics == "velocity"
        assert set(heads) == {"depth4_v"}
        return torch.full_like(state, 10.0), {"depth4_v": torch.full_like(state, 2.0)}, {}

    monkeypatch.setattr(sampler, "_model_velocity", fake_model_velocity)
    monkeypatch.setattr(sampler, "evaluate_source_with_heads", fake_source_with_heads)
    fields = sampler.build_ig_fields(
        torch.nn.Identity(),
        torch.tensor([3, 7]),
        strong_semantics="velocity",
        heads={
            "depth4_v": SimpleNamespace(
                prediction_target="velocity", depth=4, checkpoint="d4.pt"
            )
        },
        depths=(4,),
        gamma=1.0,
        autocast_dtype=None,
        gamma_segments=((0.25, 0.6), (0.5, 0.7), (1.0, 0.0)),
    )
    state = torch.zeros(2, 1, 1, 1)
    for time_value, gamma in ((0.1, 0.6), (0.25, 0.7), (0.75, 0.0)):
        guided = fields.guided(torch.tensor(time_value), state)
        strong = fields.target(torch.tensor(time_value), state)
        reference = fields.reference(torch.tensor(time_value), state)
        torch.testing.assert_close(guided, torch.full_like(state, 10.0 + 8.0 * gamma))
        torch.testing.assert_close(strong - reference, torch.full_like(state, 8.0 * gamma))


def test_parse_best_depth4_gamma_segments() -> None:
    assert sampler.parse_gamma_segments(".25:.6,.5:.7,1:0") == (
        (0.25, 0.6),
        (0.5, 0.7),
        (1.0, 0.0),
    )
    with pytest.raises(sampler.argparse.ArgumentTypeError):
        sampler.parse_gamma_segments(".25:.6,.5:.7")


def test_sampler_exposes_historical_per_batch_rng_mode() -> None:
    args = sampler.build_parser().parse_args(
        [
            "--family",
            "ig",
            "--method",
            "closed",
            "--sample-rng-mode",
            "per_batch",
        ]
    )
    assert args.sample_rng_mode == "per_batch"

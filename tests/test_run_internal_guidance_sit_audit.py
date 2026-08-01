import json

import numpy as np
import pandas as pd
import torch

from experiments.run_internal_guidance_sit_audit import (
    checkpoint_state_dict,
    parse_float_list,
    run_local_audit,
    run_rollout_audit,
    summarize_results,
    validate_protocol,
)


def test_parse_float_list() -> None:
    assert parse_float_list("0.2, 0.5,0.8") == (0.2, 0.5, 0.8)


def test_validate_protocol_requires_full_baseline_and_nested_rollout_times() -> None:
    validate_protocol(
        times=(0.2, 0.5),
        rollout_times=(0.5,),
        scales=(0.5, 1.0, 1.4),
        samples=4,
        batch_size=2,
        rollout_samples=2,
        rollout_batch_size=1,
        rollout_steps=10,
    )
    try:
        validate_protocol(
            times=(0.2, 0.5),
            rollout_times=(0.8,),
            scales=(0.5, 1.4),
            samples=4,
            batch_size=2,
            rollout_samples=2,
            rollout_batch_size=1,
            rollout_steps=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid protocol must be rejected")


def test_checkpoint_state_dict_selects_ema_and_strips_ddp_prefix() -> None:
    checkpoint = {
        "ema": {"module.weight": torch.ones(2, 2)},
        "model": {"module.weight": torch.zeros(2, 2)},
        "steps": 123,
    }
    state, metadata = checkpoint_state_dict(checkpoint, state_key="ema")
    assert list(state) == ["weight"]
    torch.testing.assert_close(state["weight"], torch.ones(2, 2))
    assert metadata["steps"] == 123
    assert metadata["stripped_module_prefix"] is True


class ConstantDualVelocity(torch.nn.Module):
    def forward(self, state, time, labels):
        del time, labels
        return torch.full_like(state, 0.75), torch.full_like(state, 0.5), None


def test_end_to_end_audit_tables_detect_a_better_guided_direction() -> None:
    model = ConstantDualVelocity()
    clean = torch.zeros((4, 1, 2, 2))
    noise = torch.ones_like(clean)
    labels = torch.arange(4, dtype=torch.long)
    indices = [10, 11, 12, 13]
    scales = (0.0, 1.0, 2.0)

    local, sweep = run_local_audit(
        model,
        clean,
        noise,
        labels,
        indices,
        times=(0.5,),
        scales=scales,
        batch_size=2,
        device=torch.device("cpu"),
    )
    rollout = run_rollout_audit(
        model,
        clean,
        noise,
        labels,
        indices,
        rollout_samples=4,
        rollout_batch_size=2,
        rollout_times=(0.5,),
        rollout_steps=2,
        scales=scales,
        device=torch.device("cpu"),
    )

    assert len(local) == 4
    assert bool(local["positive_alignment"].all())
    np.testing.assert_allclose(local["scale_star"], 2.0)
    scale_two = sweep[sweep["scale"] == 2.0]
    np.testing.assert_allclose(scale_two["mse"], 0.0)
    persistent_two = rollout[
        (rollout["mode"] == "persistent") & (rollout["scale"] == 2.0)
    ]
    assert bool((persistent_two["gain_over_full"] > 0).all())

    summary = summarize_results(local, sweep, rollout)
    serialized = json.dumps(summary, allow_nan=False)
    assert "local_endpoint_correlations" in serialized


def test_summary_skips_undefined_spearman_correlations() -> None:
    local = pd.DataFrame(
        {
            "time": [0.5] * 3,
            "alignment_cosine": [1.0] * 3,
            "positive_alignment": [True] * 3,
            "scale_star": [2.0] * 3,
            "oracle_relative_gain": [1.0] * 3,
            "full_mse": [0.25] * 3,
            "base_mse": [1.0] * 3,
        }
    )
    sweep = pd.DataFrame(
        {
            "dataset_index": [0, 1, 2],
            "time": [0.5] * 3,
            "scale": [2.0] * 3,
            "gain_over_full": [0.5] * 3,
        }
    )
    rollout = pd.DataFrame(
        {
            "dataset_index": [0, 1, 2],
            "mode": ["persistent"] * 3,
            "start_time": [0.5] * 3,
            "scale": [2.0] * 3,
            "gain_over_full": [0.5] * 3,
            "endpoint_delta_rms": [0.1] * 3,
        }
    )
    summary = summarize_results(local, sweep, rollout)
    assert summary["local_endpoint_correlations"] == []
    json.dumps(summary, allow_nan=False)

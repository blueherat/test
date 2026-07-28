import torch
import pandas as pd

from experiments.run_pid_noise_responsibility import (
    advance_student_state,
    batch_order_control_table,
    flow_noisy_state,
    output_to_x0,
    paired_predictions,
)


def test_flow_noisy_state_endpoints_and_midpoint():
    x0 = torch.ones((2, 1, 2, 2))
    noise = -torch.ones_like(x0)
    result = flow_noisy_state(x0, noise, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(result[0], x0[0])
    torch.testing.assert_close(result[1], torch.zeros_like(result[1]))


def test_velocity_output_round_trip():
    x0 = torch.randn((2, 3, 4, 4), generator=torch.Generator().manual_seed(4))
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(5))
    timestep = torch.tensor([0.2, 0.8])
    x_t = flow_noisy_state(x0, noise, timestep)
    velocity = noise - x0
    recovered = output_to_x0(x_t, velocity, timestep, "velocity")
    torch.testing.assert_close(recovered, x0)


def test_paired_predictions_share_state_and_repeat_real_branch():
    x_t = torch.ones((1, 1, 2, 2))
    timestep = torch.tensor([0.5])
    latents = {
        "real": torch.full_like(x_t, 1.0),
        "null": torch.full_like(x_t, 2.0),
        "shuffle": torch.full_like(x_t, 3.0),
    }
    seen_x = []

    def predict(state, _time, latent):
        seen_x.append(state.clone())
        return latent

    predictions, identity = paired_predictions(
        x_t=x_t,
        timestep=timestep,
        latents=latents,
        predict=predict,
        prediction_type="x0",
    )
    assert set(predictions) == {"real", "null", "shuffle"}
    assert len(seen_x) == 4
    for state in seen_x:
        torch.testing.assert_close(state, x_t)
    assert identity == {"absolute_rms_max": 0.0, "relative_rms_max": 0.0}


def test_student_state_transitions_match_sde_and_ode_definitions():
    x_t = torch.full((1, 1, 2, 2), 0.75)
    x0 = torch.full_like(x_t, 0.25)
    noise = torch.full_like(x_t, -1.0)
    current = torch.tensor([0.5])
    following = torch.tensor([0.25])
    sde = advance_student_state(
        x_t, x0, current, following, sample_type="sde", noise=noise
    )
    torch.testing.assert_close(sde, torch.full_like(sde, -0.0625))
    ode = advance_student_state(x_t, x0, current, following, sample_type="ode")
    torch.testing.assert_close(ode, torch.full_like(ode, 0.5))


def test_batch_order_control_ignores_rollout_rows():
    metrics = {
        "loss_real": 1.0,
        "loss_null": 2.0,
        "loss_shuffle": 3.0,
        "delta_null": 1.0,
        "delta_shuffle": 2.0,
    }
    teacher = {"seed": 0, "mode": "teacher_forced", "sample_index": 4, "timestep": 0.5, **metrics}
    rollout = {"seed": 0, "mode": "real_rollout", "sample_index": 4, "timestep": 0.5, **metrics}
    reordered = {"seed": 0, "sample_index": 4, "timestep": 0.5, **metrics}
    result = batch_order_control_table(
        pd.DataFrame([teacher, rollout]), pd.DataFrame([reordered]), seed=0
    )
    assert len(result) == 5
    assert result.max_absolute_difference.max() == 0.0

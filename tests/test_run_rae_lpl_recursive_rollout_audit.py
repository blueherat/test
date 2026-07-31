import torch

from experiments.run_rae_lpl_recursive_rollout_audit import (
    model_state_from_checkpoint,
    shifted_time_grid,
    velocity_endpoint_rollout,
)
from experiments.run_raev2_endpoint_rollout_audit import (
    shifted_time_grid as raev2_shifted_time_grid,
)


class PerfectVelocity(torch.nn.Module):
    def __init__(self, clean: torch.Tensor, noise: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", noise - clean)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        y: torch.Tensor,
    ) -> torch.Tensor:
        del state, time, y
        return self.velocity


def test_old_and_raev2_rollout_grids_are_identical() -> None:
    for start_time in (0.5, 0.75):
        for num_steps in (1, 4, 16):
            actual = shifted_time_grid(
                start_time,
                num_steps=num_steps,
                shift=8.0,
            )
            expected = raev2_shifted_time_grid(
                start_time,
                num_steps=num_steps,
                shift=8.0,
            )
            torch.testing.assert_close(actual, expected)


def test_perfect_velocity_reaches_clean_for_every_rollout_length() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((2, 3, 4, 4), generator=generator)
    noise = torch.randn((2, 3, 4, 4), generator=generator)
    labels = torch.tensor([1, 2])
    start_time = 0.75
    initial = (1.0 - start_time) * clean + start_time * noise
    model = PerfectVelocity(clean, noise)

    for num_steps in (1, 4, 16):
        final, path = velocity_endpoint_rollout(
            model=model,
            initial_state=initial,
            clean=clean,
            noise=noise,
            labels=labels,
            start_time=start_time,
            num_steps=num_steps,
            time_shift=8.0,
        )
        torch.testing.assert_close(final, clean, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(
            path["state_path_error_rms"],
            torch.zeros_like(path["state_path_error_rms"]),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            path["endpoint_error_rms"],
            torch.zeros_like(path["endpoint_error_rms"]),
            atol=2e-6,
            rtol=0,
        )


def test_model_state_loader_accepts_wrapped_and_materialized_weights() -> None:
    weight = torch.tensor([1.0])
    wrapped, wrapped_key = model_state_from_checkpoint(
        {"ema": {"weight": weight}},
        state_key="auto",
    )
    raw, raw_key = model_state_from_checkpoint(
        {"weight": weight},
        state_key="auto",
    )
    assert wrapped_key == "ema"
    assert raw_key == "raw"
    torch.testing.assert_close(wrapped["weight"], weight)
    torch.testing.assert_close(raw["weight"], weight)

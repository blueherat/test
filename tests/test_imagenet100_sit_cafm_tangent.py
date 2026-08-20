from __future__ import annotations

import torch
import torch.nn as nn

from experiments.imagenet100_sit_cafm_tangent import (
    TangentJVP,
    explicit_spatial_time_jvp,
    local_lsgan_optimum,
    lsgan_tangent_losses,
)
from experiments.audit_imagenet100_sit_cafm_predictivity import direction_rows
from experiments.train_imagenet100_sit_cafm_tangent_critic import (
    prepare_teacher_batch,
)
from experiments.train_imagenet100_sit_flow import NUM_CLASSES


class QuadraticCritic(nn.Module):
    def forward(self, state, time_value, labels):
        label_term = labels.float() * 0.01
        return state.flatten(1).square().sum(1) + time_value.square() + label_term


def test_tangent_jvp_matches_explicit_gradient_and_is_linear():
    torch.manual_seed(4)
    critic = QuadraticCritic()
    wrapper = TangentJVP(critic)
    state = torch.randn(3, 2, 2)
    time_value = torch.rand(3)
    labels = torch.tensor([1, 2, 3])
    velocity_a = torch.randn_like(state)
    velocity_b = torch.randn_like(state)
    dt_a = torch.randn_like(time_value)
    dt_b = torch.randn_like(time_value)

    value, tangent = wrapper(
        state,
        time_value,
        labels,
        torch.stack((velocity_a, velocity_b, velocity_a + velocity_b)),
        torch.stack((dt_a, dt_b, dt_a + dt_b)),
    )
    explicit_value, explicit_a, _ = explicit_spatial_time_jvp(
        critic,
        state.clone(),
        time_value.clone(),
        labels,
        velocity_a,
        dt_a,
    )
    assert torch.allclose(value[0], explicit_value)
    assert torch.allclose(tangent[0], explicit_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(tangent[2], tangent[0] + tangent[1], atol=1e-6, rtol=1e-6)


def test_local_lsgan_solution_is_stationary_and_precision_weighted():
    mean = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
    generator = torch.tensor([-0.4, 0.1, -0.25], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.2, 0.03, 0.0], [0.03, 2.0, 0.2], [0.0, 0.2, 8.0]],
        dtype=torch.float64,
    )
    solution = local_lsgan_optimum(mean, covariance, generator)
    gradient = solution.gradient.clone().requires_grad_(True)
    offset = solution.offset.clone().requires_grad_(True)
    residual = mean - generator

    # Exact population objective: E[(a^T V+b-1)^2] plus the fake term.
    real_mean = gradient.dot(mean) + offset - 1.0
    fake = gradient.dot(generator) + offset + 1.0
    loss = gradient @ covariance @ gradient + real_mean.square() + fake.square()
    grad_a, grad_b = torch.autograd.grad(loss, (gradient, offset))
    assert grad_a.norm().item() < 1e-10
    assert grad_b.abs().item() < 1e-10
    assert torch.allclose(loss.detach(), solution.discriminator_loss, atol=1e-10)

    precision_direction = torch.linalg.solve(covariance, residual)
    cosine = torch.nn.functional.cosine_similarity(
        solution.gradient, precision_direction, dim=0
    )
    assert cosine.item() > 1.0 - 1e-12


def test_lsgan_labels_and_centering_match_official_orientation():
    values = torch.tensor([2.0, -1.0])
    real = torch.tensor([1.0, 1.0])
    fake = torch.tensor([-1.0, -1.0])
    losses = lsgan_tangent_losses(
        values,
        real,
        fake,
        centering_scale=1e-3,
    )
    assert losses["real"].item() == 0.0
    assert losses["fake"].item() == 0.0
    assert torch.allclose(losses["total"], 1e-3 * values.square().mean())


def test_ab_audit_recovers_exact_mean_and_sample_ls_scale():
    samples = 8
    strong = torch.zeros(samples, 1, 1, 2)
    residual = torch.full_like(strong, 2.0)
    gradient = torch.ones_like(strong)
    direction = torch.ones_like(strong)
    bank = {
        "strong_velocity": strong,
        "real_velocity": strong + residual,
        "critic_gradient": gradient,
        "critic_action": (gradient * residual).flatten(1).sum(1),
        "time": torch.linspace(0.05, 0.95, samples),
    }
    rows, arrays = direction_rows("control", direction, bank)
    overall = next(row for row in rows if row["time_bin"] == "overall")
    assert overall["A_mean"] == 4.0
    assert overall["B_mean"] == 2.0
    assert overall["gamma_hat_mean"] == 2.0
    assert overall["gamma_hat_sample_ls"] == 2.0
    assert arrays["action"].shape == (samples,)


def test_explicit_class_dropout_uses_the_null_class():
    moments = torch.zeros(3, 8, 32, 32)
    labels = torch.tensor([1, 2, 3])
    generator = torch.Generator().manual_seed(8)
    _, _, retained, _ = prepare_teacher_batch(
        moments,
        labels,
        device=torch.device("cpu"),
        generator=generator,
        class_dropout_probability=0.0,
    )
    assert torch.equal(retained, labels)

    _, _, dropped, _ = prepare_teacher_batch(
        moments,
        labels,
        device=torch.device("cpu"),
        generator=generator,
        class_dropout_probability=1.0,
    )
    assert torch.equal(dropped, torch.full_like(labels, NUM_CLASSES))

"""CPU checks for guidance RL, local adjoints, and estimator limitations.

No image model, GPU workload, RL training, or speedup benchmark is run.
"""
from pathlib import Path
import json
import numpy as np
import torch


OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_rl_20260923"


def heun_local_gradient():
    """Full nonlinear Heun gradient equals detached-state local field updates."""
    dtype = torch.float64
    theta = torch.tensor([[.23, -.14], [.08, .19]], dtype=dtype, requires_grad=True)
    scales = torch.tensor([.75, -.2, 0., 1.1, .4], dtype=dtype, requires_grad=True)
    initial = torch.tensor([[.8, -.5], [-.4, .9], [.3, .6]], dtype=dtype)
    strong_matrix = torch.tensor([[.25, -.3], [.2, -.1]], dtype=dtype)
    target = torch.tensor([.2, -.15], dtype=dtype)
    queries, outputs = [], []

    def strong(x, t):
        return x @ strong_matrix.T + .12 * torch.sin(1.3 * x + t)

    def field(x, t, i):
        prediction = strong(x, t)
        feature = torch.tanh(x + t)
        weak = feature @ theta.T
        value = prediction + scales[i] * (prediction - weak)
        queries.append((prediction.detach(), feature.detach(), i))
        outputs.append(value)
        return value

    x = initial
    step = 1 / len(scales)
    for i in range(len(scales)):
        first = field(x, i * step, i)
        predicted = x + step * first
        second = field(predicted, (i + 1) * step, i)
        x = x + step * .5 * (first + second)
    loss = ((x - target).square().sum(-1) + .07 * x.sin().sum(-1)).mean()
    exact = torch.autograd.grad(loss, (theta, scales), retain_graph=True)
    terminal = torch.autograd.grad(loss, x, retain_graph=True)[0]
    cotangents = torch.autograd.grad(loss, outputs)
    local_loss = theta.new_zeros(())
    naive_loss = theta.new_zeros(())
    for (prediction, feature, index), cotangent in zip(queries, cotangents):
        local = prediction + scales[index] * (prediction - feature @ theta.T)
        local_loss = local_loss + (local * cotangent.detach()).sum()
        naive_loss = naive_loss + (local * (step * .5 * terminal.detach())).sum()
    local_gradient = torch.autograd.grad(local_loss, (theta, scales), retain_graph=True)
    naive_gradient = torch.autograd.grad(naive_loss, (theta, scales))

    flatten = lambda parts: torch.cat([value.flatten() for value in parts])
    exact_flat = flatten(exact)
    local_flat = flatten(local_gradient)
    naive_flat = flatten(naive_gradient)
    error = float((exact_flat - local_flat).abs().max())
    naive_relative_error = float((exact_flat - naive_flat).norm() / exact_flat.norm())
    assert error < 1e-12
    assert naive_relative_error > .01
    assert abs(float(exact[1][2])) > 1e-5
    correction_rows = []
    residual = exact_flat - naive_flat
    for probability in [1., .5, .1, .02]:
        absent = naive_flat
        present = naive_flat + residual / probability
        expected = (1 - probability) * absent + probability * present
        variance = ((1 - probability) * (absent - exact_flat).square().sum()
                    + probability * (present - exact_flat).square().sum())
        predicted_variance = (1 / probability - 1) * residual.square().sum()
        assert float((expected - exact_flat).abs().max()) < 1e-12
        assert abs(float(variance - predicted_variance)) < 1e-12
        correction_rows.append(dict(
            exact_backward_probability=probability,
            expectation_max_error=float((expected - exact_flat).abs().max()),
            added_gradient_variance=float(variance),
            predicted_added_variance=float(predicted_variance),
        ))
    return dict(
        steps=len(scales), field_queries=len(outputs), batch=len(initial),
        exact_loss=float(loss.detach()),
        local_vs_full_max_error=error,
        direct_terminal_cotangent_relative_error=naive_relative_error,
        scale_gradients=exact[1].tolist(),
        unbiased_random_exact_correction=correction_rows,
        point="Exact query cotangents absorb strong input Jacobians; terminal gradient alone does not.",
    )


def linear_score_variance():
    """Exact variance is (d+1)||g-h||^2 for centered linear control variates."""
    rng = np.random.default_rng(20260923)
    rows = []
    for dimension in [2, 16, 64, 256]:
        noise = rng.standard_normal((20000, dimension))
        score = noise[:, :1] * noise
        score[:, 0] -= 1
        empirical = float(np.mean(np.sum(score**2, axis=1)))
        theoretical = dimension + 1
        assert abs(empirical / theoretical - 1) < .06
        rows.append(dict(
            action_dimension=dimension,
            score_mean_squared_gradient_error=empirical,
            theoretical_error=theoretical,
            critic_80_percent_gradient_error=empirical * .04,
            theoretical_critic_error=theoretical * .04,
        ))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Identical on-policy values do not identify the action derivative.
    # Q(a)=a, Qhat(a)=a-epsilon*sin(a/epsilon^2).
    eps = .01
    critic = dict(
        epsilon=eps, maximum_value_error_bound=eps,
        true_value_at_zero=0., fitted_value_at_zero=0.,
        true_action_gradient=1., fitted_action_gradient=1 - 1 / eps,
        point="Perfect value fit at the deterministic action can give the opposite actor update.",
    )
    # R(a)=-(a^2-1)^2, a~N(mu,sigma^2); stochastic optimum differs from sigma=0.
    smoothing = []
    for sigma in [0., .1, .3, .5, .6]:
        positive_mean_optimum = np.sqrt(max(1 - 3 * sigma**2, 0))
        deterministic_reward = -(positive_mean_optimum**2 - 1)**2
        smoothing.append(dict(
            exploration_sigma=sigma, positive_optimal_mean=float(positive_mean_optimum),
            deterministic_reward_at_mean=float(deterministic_reward),
        ))
    result = dict(
        kind="CPU mathematical audits, not image training or measured acceleration",
        heun_local_gradient=heun_local_gradient(),
        critic_value_counterexample=critic,
        gaussian_policy_variance=linear_score_variance(),
        exploration_objective_shift=smoothing,
    )
    (OUT / "rl_adjoint_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

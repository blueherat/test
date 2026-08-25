import torch

from experiments.advfd_cleanroom.audit_pmf_critic_surrogate_direction import (
    central_difference,
    control_variate_moments,
    fd_components,
    gradient_norm,
    parameter_norm,
    population_moments,
    project_moments,
    shift_along_gradient,
)


def test_population_moments_match_direct_covariance() -> None:
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], dtype=torch.float64
    )
    mean, covariance = population_moments(features)
    centered = features - features.mean(dim=0)
    expected = centered.T @ centered / features.shape[0]
    torch.testing.assert_close(mean, features.mean(dim=0))
    torch.testing.assert_close(covariance, expected)


def test_recalibrated_fd_is_invariant_to_common_translation() -> None:
    real = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float64
    )
    fake = torch.tensor(
        [[-0.5, 0.3], [1.5, -0.1], [0.2, 1.5]], dtype=torch.float64
    )
    shift = torch.tensor([13.0, -7.0], dtype=torch.float64)
    before = fd_components(real, fake, epsilon=1e-3)
    after = fd_components(real + shift, fake + shift, epsilon=1e-3)
    torch.testing.assert_close(torch.stack(before), torch.stack(after))


def test_control_variate_moments_recover_paired_affine_change() -> None:
    baseline = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float64
    )
    matrix = torch.tensor([[1.2, 0.1], [-0.2, 0.8]], dtype=torch.float64)
    shift = torch.tensor([0.4, -0.7], dtype=torch.float64)
    perturbed = baseline @ matrix + shift
    anchor = population_moments(baseline)
    estimated = control_variate_moments(anchor, baseline, perturbed)
    expected = population_moments(perturbed)
    torch.testing.assert_close(estimated[0], expected[0])
    torch.testing.assert_close(estimated[1], expected[1])


def test_project_moments_matches_projected_features() -> None:
    features = torch.tensor(
        [[-1.0, 0.0, 2.0], [1.0, 1.0, -1.0], [0.0, 2.0, 3.0]],
        dtype=torch.float64,
    )
    projection = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=torch.float64
    )
    expected = population_moments(features @ projection)
    actual = project_moments(population_moments(features), projection)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_parameter_gradient_shift_and_restore() -> None:
    module = torch.nn.Linear(2, 1, bias=True).double()
    parameters = list(module.parameters())
    loss = sum(parameter.square().sum() for parameter in parameters)
    loss.backward()
    initial = [parameter.detach().clone() for parameter in parameters]
    grad_l2 = float(gradient_norm(parameters))
    model_l2 = float(parameter_norm(parameters))
    distance = 1e-3 * model_l2
    shift_along_gradient(parameters, distance=distance, gradient_l2=grad_l2)
    shift_along_gradient(parameters, distance=-distance, gradient_l2=grad_l2)
    for parameter, expected in zip(parameters, initial):
        torch.testing.assert_close(parameter, expected)


def test_central_difference_tracks_each_component() -> None:
    plus = {
        "recalibrated_ema_quotient": {
            "mean": 3.0,
            "covariance": 7.0,
            "full": 10.0,
        },
        "implemented_surrogate": {"mean": 5.0, "covariance": 9.0, "full": 14.0},
    }
    minus = {
        "recalibrated_ema_quotient": {
            "mean": 1.0,
            "covariance": 3.0,
            "full": 4.0,
        },
        "implemented_surrogate": {"mean": 1.0, "covariance": 1.0, "full": 2.0},
    }
    derivative = central_difference(plus, minus, 2.0)
    assert derivative["recalibrated_ema_quotient"] == {
        "mean": 0.5,
        "covariance": 1.0,
        "full": 1.5,
    }
    assert derivative["implemented_surrogate"] == {
        "mean": 1.0,
        "covariance": 2.0,
        "full": 3.0,
    }

import torch

from experiments.isospectral_alignment_toy import (
    IsospectralToyConfig,
    build_isospectral_problem,
    effective_hessian,
    joint_rotation_error,
    run_study,
)


def test_equal_weight_spectra_can_have_different_effective_hessians():
    config = IsospectralToyConfig(spatial_size=4, steps=20, save=False)
    problem = build_isospectral_problem(config)
    metrics = problem["metrics"]
    jacobian = problem["jacobian"]

    aligned_spectrum = torch.linalg.eigvalsh(metrics["aligned"])
    random_spectrum = torch.linalg.eigvalsh(metrics["random"])
    assert torch.allclose(aligned_spectrum, random_spectrum, atol=1e-10, rtol=1e-10)

    aligned_hessian = effective_hessian(jacobian, metrics["aligned"])
    random_hessian = effective_hessian(jacobian, metrics["random"])
    assert not torch.allclose(aligned_hessian, random_hessian, atol=1e-5, rtol=1e-5)


def test_joint_output_rotation_is_a_true_change_of_coordinates():
    config = IsospectralToyConfig(spatial_size=4, steps=20, save=False)
    problem = build_isospectral_problem(config)
    audit = joint_rotation_error(
        problem["jacobian"],
        problem["observation"],
        problem["metrics"]["aligned"],
        seed=41,
    )
    assert audit["hessian_max_abs_error"] < 1e-10
    assert audit["observation_norm_error"] < 1e-10


def test_default_orientation_changes_conditioning_and_convergence():
    config = IsospectralToyConfig(spatial_size=4, steps=1_000, save=False)
    summary, _, audit, result_dir = run_study(config)
    rows = summary.set_index("basis")
    assert result_dir is None
    assert audit["weight_spectrum_max_abs_error"] < 1e-10
    assert (
        rows.loc["aligned", "effective_hessian_condition"]
        > rows.loc["random", "effective_hessian_condition"]
    )
    assert (
        rows.loc["aligned", "final_relative_parameter_error"]
        > rows.loc["random", "final_relative_parameter_error"]
    )

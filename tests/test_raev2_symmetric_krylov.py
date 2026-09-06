import pytest
import torch

from experiments.raev2_symmetric_krylov import symmetric_krylov


def dense_indefinite(size, seed=719):
    generator = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))
    values = torch.linspace(-7., 4., size, dtype=torch.float64)
    return (basis * values) @ basis.T


def test_dense_indefinite_full_dimension_matches_exact_top_eigenpair():
    matrix = dense_indefinite(12)
    calls = []

    def operator(vector):
        calls.append(vector.clone())
        return matrix @ vector

    result = symmetric_krylov(operator, torch.arange(1., 13., dtype=torch.float64), 12, (4, 8, 12))
    assert len(calls) == result["matvec_calls"] == result["iterations"] == 12
    assert [item["iterations"] for item in result["history"]] == [4, 8, 12]
    assert result["termination"] == "dimension_exhausted"
    assert result["ritz_value"] == pytest.approx(4., abs=2e-12)
    assert result["rayleigh"] == pytest.approx(4., abs=2e-12)
    vector = result["vector"]
    assert torch.linalg.vector_norm(matrix @ vector - result["rayleigh"] * vector) < 2e-12
    assert result["residual_norm"] < 2e-12
    torch.testing.assert_close(result["basis"] @ result["basis"].T, torch.eye(12, dtype=torch.float64), atol=2e-13, rtol=0)
    torch.testing.assert_close(result["abasis"], result["basis"] @ matrix.T, atol=2e-13, rtol=0)


def test_fixed_8_16_24_checkpoints_use_cached_images_and_converge():
    matrix = dense_indefinite(32)
    generator = torch.Generator().manual_seed(831)
    start = torch.randn(32, generator=generator, dtype=torch.float64)
    calls = 0

    def operator(vector):
        nonlocal calls
        calls += 1
        return matrix @ vector

    result = symmetric_krylov(operator, start)
    assert calls == result["iterations"] == 24
    assert result["termination"] == "max_iterations"
    assert [item["iterations"] for item in result["history"]] == [8, 16, 24]
    previous = -float("inf")
    for item in result["history"]:
        count = item["iterations"]
        basis = result["basis"][:count]
        values, coefficients = torch.linalg.eigh(basis @ matrix @ basis.T)
        vector = coefficients[:, -1] @ basis
        direct_residual = torch.linalg.vector_norm(matrix @ vector - values[-1] * vector)
        assert item["rayleigh"] == pytest.approx(float(values[-1]), abs=2e-12)
        assert item["residual_norm"] == pytest.approx(float(direct_residual), abs=2e-12)
        assert item["rayleigh"] >= previous - 2e-12
        previous = item["rayleigh"]
    assert result["rayleigh"] == pytest.approx(4., abs=2e-5)


def test_nonsymmetric_jacobian_is_symmetrized_in_operator_not_only_ritz_matrix():
    symmetric = dense_indefinite(9)
    generator = torch.Generator().manual_seed(78)
    raw = torch.randn(9, 9, generator=generator, dtype=torch.float64)
    jacobian = symmetric + 4 * (raw - raw.T)
    assert torch.linalg.matrix_norm(jacobian - jacobian.T) > 1
    result = symmetric_krylov(lambda v: (jacobian @ v + jacobian.T @ v) / 2,
                             torch.ones(9, dtype=torch.float64), 9)
    assert result["rayleigh"] == pytest.approx(4., abs=2e-12)
    assert result["residual_norm"] < 2e-12
    assert result["projected_asymmetry_norm"] < 2e-12
    torch.testing.assert_close(result["abasis"], result["basis"] @ symmetric.T, atol=2e-12, rtol=0)


def test_rank_one_positive_direction_found_despite_negative_random_rayleigh():
    generator = torch.Generator().manual_seed(202609081)
    start = torch.randn(128, generator=generator, dtype=torch.float64)
    direction = torch.zeros(128, dtype=torch.float64)
    direction[0] = 1

    def operator(vector):
        return -vector + 6 * torch.dot(direction, vector) * direction

    initial_rayleigh = torch.dot(start, operator(start)) / torch.dot(start, start)
    assert initial_rayleigh < 0
    result = symmetric_krylov(operator, start)
    assert result["iterations"] == result["matvec_calls"] == 2
    assert result["termination"] == "numerical_breakdown"
    assert result["rayleigh"] == pytest.approx(5., abs=2e-12)
    assert result["residual_norm"] < 2e-12
    assert [item["iterations"] for item in result["history"]] == [2]


def test_negative_exact_ritz_pair_can_miss_an_orthogonal_positive_eigenspace():
    diagonal = torch.tensor([5., -1., -1., -1.], dtype=torch.float64)
    start = torch.tensor([0., 1., 0., 0.], dtype=torch.float64)
    result = symmetric_krylov(lambda v: diagonal * v, start)
    assert diagonal.max() > 0
    assert result["rayleigh"] == -1
    assert result["residual_norm"] == 0
    assert result["iterations"] == 1
    assert "does not certify" in result["spectral_scope"]


def test_fp32_matvec_with_fp64_basis_preserves_shape_and_discards_graphs():
    matrix = dense_indefinite(20).float()
    generator = torch.Generator().manual_seed(917)
    start = torch.randn(4, 5, generator=generator)
    shapes = []

    def operator(vector):
        assert vector.dtype == torch.float64
        shapes.append(vector.shape)
        differentiable_input = vector.float().detach().requires_grad_(True)
        return (matrix @ differentiable_input.flatten()).reshape_as(vector)

    result = symmetric_krylov(operator, start, 20, (8, 16, 20))
    assert shapes == [start.shape] * 20
    assert result["vector"].shape == start.shape
    assert result["vector"].dtype == result["basis"].dtype == torch.float64
    assert not any(result[key].requires_grad for key in ("vector", "basis", "abasis"))
    assert result["orthogonality_error"] < 2e-13
    direct = matrix.double() @ result["vector"].flatten()
    assert result["rayleigh"] == pytest.approx(float(torch.linalg.eigvalsh(matrix.double())[-1]), abs=2e-6)
    assert torch.linalg.vector_norm(direct - result["rayleigh"] * result["vector"].flatten()) < 2e-6


def test_zero_operator_breakdown_and_invalid_inputs_are_explicit():
    result = symmetric_krylov(torch.zeros_like, torch.ones(6))
    assert result["iterations"] == 1
    assert result["rayleigh"] == result["residual_norm"] == 0
    with pytest.raises(ValueError, match="nonzero"):
        symmetric_krylov(torch.clone, torch.zeros(6))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        symmetric_krylov(lambda v: v * float("nan"), torch.ones(6))
    with pytest.raises(ValueError, match="dimension"):
        symmetric_krylov(lambda v: v[:2], torch.ones(6))

import torch
from pathlib import Path

from experiments.d2c_prior_decoder_audit import (
    D2CAuditConfig,
    covariance_relative_error_from_gram,
    effective_rank_from_gram,
    high_dimensional_latent_metrics,
    sample_empirical_covariance_gaussian,
    split_fit_evaluation_records,
)


def test_fit_and_evaluation_records_are_disjoint_and_ordered():
    records = [bytes([index]) for index in range(10)]
    fit, evaluation = split_fit_evaluation_records(records, 4, 5)
    assert fit == records[:4]
    assert evaluation == records[4:9]
    assert set(fit).isdisjoint(evaluation)


def test_factorized_gaussian_recovers_mean_and_covariance():
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(8_000, 3, generator=generator)
    transform = torch.tensor(
        [[2.0, 0.0, 0.0], [1.1, 0.6, 0.0], [-0.4, 0.2, 0.3]]
    )
    fit = base @ transform.T + torch.tensor([1.0, -2.0, 0.5])
    samples = sample_empirical_covariance_gaussian(
        fit, 8_000, seed=11, batch_size=512
    )
    assert torch.linalg.norm(samples.mean(0) - fit.mean(0)) < 0.06
    assert covariance_relative_error_from_gram(fit, samples) < 0.05


def test_factorized_gaussian_preserves_nonvector_shape():
    values = torch.randn(200, 2, 3, 4, generator=torch.Generator().manual_seed(13))
    samples = sample_empirical_covariance_gaussian(values, 50, seed=17)
    assert samples.shape == (50, 2, 3, 4)
    assert torch.isfinite(samples).all()


def test_gram_metrics_are_exact_for_identical_values():
    values = torch.randn(128, 64, generator=torch.Generator().manual_seed(19))
    assert covariance_relative_error_from_gram(values, values.clone()) < 1e-7
    rank = effective_rank_from_gram(values)
    assert 1.0 < rank <= 64.1


def test_projected_high_dimensional_metrics_detect_shift():
    values = torch.randn(600, 8, 8, 8, generator=torch.Generator().manual_seed(23))
    exact = high_dimensional_latent_metrics(
        values, values.clone(), seed=29, max_samples=400, projection_dimensions=64
    )
    shifted = high_dimensional_latent_metrics(
        values, values + 1.5, seed=29, max_samples=400, projection_dimensions=64
    )
    assert exact["standardized_swd"] == 0.0
    assert exact["covariance_relative_error_gram"] < 1e-7
    assert shifted["standardized_swd"] > 0.5
    assert shifted["projected_c2st_accuracy"] > 0.8


def test_config_rejects_invalid_counts_without_assets(tmp_path):
    config = D2CAuditConfig(
        d2c_repo=tmp_path,
        checkpoint=tmp_path / "model.ckpt",
        count=1,
    )
    try:
        config.validate(require_checkpoint=False)
    except ValueError as error:
        assert "count" in str(error)
    else:
        raise AssertionError("invalid D2C audit count was accepted")


def test_tensor_only_cache_payload_is_weights_only_loadable(tmp_path):
    path = Path(tmp_path) / "cache.pt"
    payload = {
        "fit": torch.randn(2, 3),
        "metadata": {"torch_version": str(torch.__version__), "fp32": True},
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["metadata"]["torch_version"] == str(torch.__version__)

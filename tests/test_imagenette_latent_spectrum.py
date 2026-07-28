import torch

from experiments.analyze_imagenette_latent_spectrum import spectrum_metrics


def test_spectrum_metrics_detect_rank_two_data() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(2_000, 2, generator=generator)
    values = torch.cat([values, torch.zeros(2_000, 6)], dim=1)
    metrics = spectrum_metrics(values)
    assert 1.9 < metrics["effective_rank"] <= 2.0
    assert metrics["k90"] == 2
    assert metrics["k99"] == 2
    assert metrics["top16_fraction"] == 1.0


def test_spectrum_metrics_rejects_zero_variance() -> None:
    try:
        spectrum_metrics(torch.ones(8, 4))
    except ValueError as error:
        assert "zero variance" in str(error)
    else:
        raise AssertionError("zero-variance input should fail")

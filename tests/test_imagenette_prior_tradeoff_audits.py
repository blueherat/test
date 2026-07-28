import pandas as pd
import torch

from experiments.analyze_imagenette_prior_semantic_gap import class_direction_basis
from experiments.summarize_imagenette_latent_prior_tradeoff import nfe_audit_summary


def test_class_direction_basis_has_at_most_class_count_minus_one_dimensions():
    generator = torch.Generator().manual_seed(3)
    labels = torch.arange(10).repeat_interleave(20)
    embedding = torch.randn(200, 16, generator=generator)
    embedding += torch.nn.functional.one_hot(labels, 10).float() @ torch.randn(
        10, 16, generator=generator
    )
    mean, centroids, basis = class_direction_basis(embedding, labels)
    assert mean.shape == (16,)
    assert centroids.shape == (10, 16)
    assert basis.shape == (16, 9)
    torch.testing.assert_close(basis.T @ basis, torch.eye(9), atol=2e-6, rtol=0)


def test_nfe_audit_summary_checks_complete_order_and_exact_replay():
    rows = []
    for seed in range(5):
        for latent_dim, fid in ((16, 100.0), (64, 105.0), (256, 120.0)):
            row = {
                "latent_dim": latent_dim,
                "frozen_seed": seed,
                "audit_nfe_fid": fid + seed,
                "audit_nfe_minus_formal_fid": -0.3,
                "formal_saved_feature_fid_abs_diff": 0.0,
                "regenerated100_metric_abs_diff": 0.0,
                "frozen_decoder_matches_formal": True,
            }
            if latent_dim == 256 and seed == 2:
                row["independent_nfe100_fid_abs_diff"] = 0.0
            rows.append(row)
    summary = nfe_audit_summary(pd.DataFrame(rows))
    assert summary["complete"]
    assert summary["nfe200_order_16_better_64_better_256_seed_count"] == 5
    assert summary["independent_d256_seed2_nfe100_exact"]
    assert summary["regenerated100_metric_max_abs_diff"] == 0.0

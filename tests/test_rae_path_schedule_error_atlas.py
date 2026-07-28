import torch

from experiments.rae_layerwise_path import random_detail_basis
from experiments.run_rae_path_schedule_error_atlas import (
    basis_projection,
    component_error_metrics,
)


def test_component_error_metrics_separate_semantic_and_basis_errors():
    clean = torch.randn(3, 8, 4, 4)
    target = torch.randn_like(clean)
    basis = random_detail_basis(8, 2, seed=5)
    basis_error = basis_projection(torch.randn_like(clean), basis)
    prediction = target + basis_error
    metrics = component_error_metrics(
        prediction,
        target,
        clean,
        basis,
        time=0.5,
        semantic_factor=1.0,
        basis_factor=0.25,
    )
    assert float(metrics["semantic_velocity_relative"].max()) < 1e-5
    assert bool((metrics["basis_velocity_relative"] > 0).all())
    assert bool((metrics["basis_endpoint_relative"] > 0).all())

import pandas as pd
import pytest
import torch

from experiments.small_image_residual_trust_region import (
    select_scale,
    unseen_pool_slices,
)


def test_unseen_pool_slices_are_ordered_and_disjoint():
    values = torch.arange(12)
    indices = torch.arange(100, 112)
    validation, test, validation_indices, test_indices = unseen_pool_slices(
        values,
        indices,
        source_count=4,
        validation_count=3,
        test_count=5,
    )
    torch.testing.assert_close(validation, torch.tensor([4, 5, 6]))
    torch.testing.assert_close(test, torch.tensor([7, 8, 9, 10, 11]))
    assert set(validation_indices.tolist()).isdisjoint(test_indices.tolist())
    assert set(indices[:4].tolist()).isdisjoint(validation_indices.tolist())


def test_select_scale_uses_fid_and_breaks_ties_toward_smaller_scale():
    frame = pd.DataFrame(
        {
            "variant": ["scale_0.00", "scale_0.25", "scale_0.50"],
            "feature_fid": [2.0, 1.0, 1.0],
        }
    )
    assert select_scale(frame, (0.0, 0.25, 0.5)) == pytest.approx(0.25)

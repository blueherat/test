from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_proximal_calibration import (
    CleanBank, bank_batches, coupled_step_predictions, load_clean_bank, time_indices, validate_bank_pair,
)
from experiments.raev2_proximal_calibration import DiagonalMoments, heldout_step_metrics


def test_shards_cover_all_official_steps_without_time_selection():
    shards = [time_indices(index, 4) for index in range(4)]
    assert all(len(shard) == 25 for shard in shards)
    assert sorted(index for shard in shards for index in shard) == list(range(100))


def test_bank_selection_uses_experiment_ids_and_independence_uses_real_rows(tmp_path):
    banks = []
    for split, offset in (("train", 10), ("heldout", 20)):
        root = tmp_path / split
        root.mkdir()
        for rank in range(2):
            ids = np.arange(rank, 6, 2, dtype=np.int64)
            np.savez(root / f"clean_rank{rank:02d}.npz", ids=ids, rows=ids + offset,
                     labels=ids % 4, latents=np.broadcast_to(ids[:, None, None, None], (3, 2, 1, 1)).astype(np.float16))
        banks.append(load_clean_bank(root, samples=4))
    train, heldout = banks
    assert np.array_equal(train.ids, np.arange(4))
    assert np.array_equal(train.latents[:, 0, 0, 0], np.arange(4))
    validate_bank_pair(train, heldout, num_classes=4)
    overlap = CleanBank(heldout.latents, heldout.ids, train.rows, heldout.labels, heldout.metadata)
    with pytest.raises(ValueError, match="actual ImageNet rows"):
        validate_bank_pair(train, overlap, num_classes=4)


def test_step_target_uses_same_clean_noise_coupling_and_official_ig():
    clean = torch.tensor([[[[2.0]]], [[[4.0]]]])
    noise = torch.tensor([[[[-1.0]]], [[[1.0]]]])
    labels = torch.tensor([0, 1])
    seen = []

    def model(state, times, context, attn_mask):
        seen.append(state.clone())
        return 2 * state, state

    for t, s in ((0.7, 0.4), (0.07, 0.0)):
        predicted, target = coupled_step_predictions(model, clean, noise, labels, t, s)
        state = (1 - t) * clean + t * noise
        guided = 2 * state + 0.78 * state if t >= 0.1 else 2 * state
        assert torch.equal(seen[-1], state)
        assert torch.equal(target, (1 - s) * clean + s * noise)
        torch.testing.assert_close(predicted - target, ((t - s) / t) * (guided - clean))


def test_oracle_step_fit_is_exact_null_on_balanced_gaussian_moment_design():
    # Cartesian symmetric quadrature has exact Gaussian first/second moments.
    clean = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    noise = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
    t, s = 0.7, 0.4
    state = (1 - t) * clean + t * noise
    target = (1 - s) * clean + s * noise
    oracle = (1 - t) / ((1 - t) ** 2 + t ** 2) * state
    predicted = state - (t - s) * (state - oracle) / t
    moments = DiagonalMoments()
    moments.update(target, predicted - target)
    fit = moments.fit()
    assert torch.equal(fit.offset, torch.zeros_like(fit.offset))
    assert torch.equal(fit.slope, torch.zeros_like(fit.slope))
    assert bool((fit.covariance < 0).all())
    metrics = heldout_step_metrics(predicted, target, fit)
    assert torch.equal(metrics["risk_gain"], torch.zeros_like(metrics["risk_gain"]))

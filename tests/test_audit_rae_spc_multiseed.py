from __future__ import annotations

import torch

from experiments.audit_rae_spc_multiseed import small_state_equal


def test_small_checkpoint_state_pairing() -> None:
    state = {
        "step": 2000,
        "branch_start_step": 0,
        "scheduler": {"last_epoch": 2000},
        "rng_cpu": torch.arange(8, dtype=torch.uint8),
        "rng_cuda": [torch.arange(6, dtype=torch.uint8)],
    }
    checks = small_state_equal(state, state)
    assert all(checks.values())
    changed = dict(state, rng_cpu=torch.ones(8, dtype=torch.uint8))
    assert not small_state_equal(state, changed)["rng_cpu"]

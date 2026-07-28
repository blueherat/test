from __future__ import annotations

from pathlib import Path

from experiments.run_rae_path_gradient_interference import _path_kwargs
from experiments.run_rae_spc_gradient_study import worker_command


def test_switched_manifest_resolves_late_static_mode() -> None:
    manifest = {
        "path_mode": "annealed",
        "path_switch_step": 2000,
        "path_mode_after_switch": "static",
        "path_power": 2.0,
        "path_family": "power",
        "path_floor": 0.2,
        "path_alpha": 1.0,
        "detail_scale": 1.0,
    }
    assert _path_kwargs(manifest, step=1999)["mode"] == "annealed"
    assert _path_kwargs(manifest, step=2000)["mode"] == "static"
    assert _path_kwargs(manifest, step=5000)["mode"] == "static"
    assert _path_kwargs(manifest, step=1999, mode_override="static")["mode"] == "static"


def test_gradient_worker_uses_fixed_probe_arguments() -> None:
    command = worker_command(
        Path("/tmp/results"),
        Path("/tmp/output"),
        seed=1201,
        condition="spc",
        endpoint=5000,
        switch_step=2000,
        cache_start=100288,
        count=32,
        batch_size=4,
        probe_seed=20260730,
    )
    joined = " ".join(command)
    assert "--worker" in command
    assert "--condition spc" in joined
    assert "--probe-seed 20260730" in joined

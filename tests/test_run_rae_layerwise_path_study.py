from __future__ import annotations

from experiments.run_rae_layerwise_path_study import branch_name, training_command


def test_branch_name_is_stable() -> None:
    assert branch_name(3, "annealed", 10000) == "seed3_annealed_rank16_s0_to_10000"


def test_random_condition_adds_random_subspace_flag(tmp_path) -> None:
    command, name = training_command(
        seed=3,
        condition="random",
        path_mode="annealed",
        random_subspace=True,
        endpoint=10,
        checkpoint=tmp_path / "start.pt",
        subspaces=tmp_path / "subspaces.pt",
        latent_cache=tmp_path / "cache",
        results=tmp_path,
        rank=16,
        power=2.0,
        detail_scale=2.5,
    )
    assert name == "seed3_random_rank16_s0_to_10"
    assert "--random-subspace" in command
    assert command[command.index("--latent-cache") + 1] == str(tmp_path / "cache")
    assert command[command.index("--detail-scale") + 1] == "2.5"
    assert command[command.index("--path-mode") + 1] == "annealed"

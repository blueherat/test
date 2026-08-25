from pathlib import Path

import pytest
import torch

from experiments.advfd_cleanroom.train_fresh_pmf_critic_tournament import (
    covariance_contribution_summary,
    paired_train_eval_paths,
)


def make_png_names(root: Path, label: str, names: list[str]) -> tuple[str, Path]:
    folder = root / label
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"not decoded by this test")
    return label, folder


def test_paired_train_eval_paths_are_shared_and_disjoint(tmp_path: Path) -> None:
    names = [f"{index:04d}.png" for index in range(12)]
    first = make_png_names(tmp_path, "static", names)
    second = make_png_names(tmp_path, "advfd", names + ["extra.png"])

    train, evaluation, manifest = paired_train_eval_paths(
        [first, second], train_count=7, eval_count=5, seed=17
    )

    train_names = {path.name for path in train["static"]}
    eval_names = {path.name for path in evaluation["static"]}
    assert train_names.isdisjoint(eval_names)
    assert train_names == {path.name for path in train["advfd"]}
    assert eval_names == {path.name for path in evaluation["advfd"]}
    assert manifest["train_count"] == 7
    assert manifest["eval_count"] == 5


def test_paired_train_eval_paths_reject_insufficient_common_files(
    tmp_path: Path,
) -> None:
    first = make_png_names(tmp_path, "static", ["a.png", "b.png", "c.png"])
    second = make_png_names(tmp_path, "advfd", ["a.png", "b.png", "d.png"])
    with pytest.raises(ValueError, match="paired PNG"):
        paired_train_eval_paths([first, second], train_count=2, eval_count=2, seed=1)


def test_covariance_contribution_summary_identity_is_zero_rank_safe() -> None:
    mean = torch.zeros(4, dtype=torch.float64)
    covariance = torch.eye(4, dtype=torch.float64)
    summary = covariance_contribution_summary(
        mean, covariance, covariance, epsilon=1e-3
    )
    assert summary["participation_rank"] == pytest.approx(0.0, abs=1e-12)
    assert summary["top_k_share"]["1"] == pytest.approx(0.0, abs=1e-12)


def test_covariance_contribution_summary_detects_single_active_mode() -> None:
    mean = torch.zeros(4, dtype=torch.float64)
    real = torch.eye(4, dtype=torch.float64)
    fake = torch.eye(4, dtype=torch.float64)
    fake[0, 0] = 4.0
    summary = covariance_contribution_summary(mean, real, fake, epsilon=1e-8)
    assert summary["participation_rank"] == pytest.approx(1.0, rel=1e-6)
    assert summary["top_k_share"]["1"] == pytest.approx(1.0, rel=1e-6)

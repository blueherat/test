from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_raev2_official_samples import parse_branch  # noqa: E402


def test_parse_branch_requires_existing_archive(tmp_path: Path) -> None:
    archive = tmp_path / "samples.npz"
    archive.touch()
    assert parse_branch(f"pfr={archive}") == ("pfr", archive.resolve())
    with pytest.raises(Exception):
        parse_branch("missing-separator")
    with pytest.raises(Exception):
        parse_branch(f"pfr={tmp_path / 'missing.npz'}")

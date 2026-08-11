from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prepare_imagenet100_sdvae_index import (
    build_subset_indices,
    read_wnids,
)


def test_build_subset_indices_uses_class_major_cache_order() -> None:
    counts = np.asarray([2, 3, 1, 4], dtype=np.int64)

    indices, labels = build_subset_indices(counts, [2, 0, 3])

    assert indices.tolist() == [5, 0, 1, 6, 7, 8, 9]
    assert labels.tolist() == [0, 1, 1, 2, 2, 2, 2]


def test_read_wnids_rejects_duplicate_entries(tmp_path: Path) -> None:
    entries = [f"n{index:08d}" for index in range(99)] + ["n00000000"]
    path = tmp_path / "classes.txt"
    path.write_text("\n".join(entries), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        read_wnids(path)

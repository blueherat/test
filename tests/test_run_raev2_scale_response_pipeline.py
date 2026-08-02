from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_raev2_scale_response_pipeline import available_memory_gib


def test_available_memory_gib_reads_memavailable(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       67108864 kB\nMemAvailable:   33554432 kB\n",
        encoding="utf-8",
    )
    assert available_memory_gib(meminfo) == 32.0


def test_available_memory_gib_rejects_missing_field(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 67108864 kB\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MemAvailable"):
        available_memory_gib(meminfo)

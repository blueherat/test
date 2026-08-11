from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dual_spiral_package",
    ROOT / "experiments" / "package_dual_target_closed_loop_spiral_results.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_collect_data_excludes_checkpoints(tmp_path: Path) -> None:
    setting = tmp_path / "seed1" / "D2_H8"
    aggregate = tmp_path / "aggregate"
    setting.mkdir(parents=True)
    aggregate.mkdir()
    (setting / "config.json").write_text("{}\n", encoding="utf-8")
    (setting / "endpoint_metrics.csv").write_text("condition,swd\na,1\n")
    (setting / "checkpoint.pt").write_bytes(b"weights")
    (aggregate / "endpoint_seed_summary.csv").write_text("condition,swd\na,1\n")

    paths = MODULE.collect_data_files(tmp_path)

    assert {path.name for path in paths} == {
        "config.json",
        "endpoint_metrics.csv",
        "endpoint_seed_summary.csv",
    }
    assert all(path.name != "checkpoint.pt" for path in paths)


def test_figure_order_and_description_are_deterministic(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    setting = tmp_path / "seed7" / "D512_H128"
    aggregate.mkdir()
    setting.mkdir(parents=True)
    (aggregate / "spiral_endpoint_geometry_atlas.png").write_bytes(b"png")
    (setting / "endpoint_scatter.png").write_bytes(b"png")

    paths = MODULE.collect_figure_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "aggregate/spiral_endpoint_geometry_atlas.png",
        "seed7/D512_H128/endpoint_scatter.png",
    ]
    assert "seed=7, D512_H128" in MODULE.describe_figure(paths[1], tmp_path)

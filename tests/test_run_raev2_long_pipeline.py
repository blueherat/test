import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.run_raev2_long_pipeline import (
    checkpoint_branch_step,
    checkpoint_global_step,
    checkpoint_steps,
    evaluation_complete,
    merge_evaluations,
    verify_same_noise_protocol,
    write_curve,
    write_flow_curve,
)


def test_checkpoint_name_parsing_and_schedule() -> None:
    path = Path("branch-0000150-global-0100230.pt")
    assert checkpoint_branch_step(path) == 150
    assert checkpoint_global_step(path) == 100_230
    assert checkpoint_steps(150, 10) == tuple(range(10, 151, 10))


def test_script_entrypoint_imports_repo_without_pythonpath(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_raev2_long_pipeline.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_requests_two_gib_memory_reserve() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "launch_raev2_long_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "--min-free-gib 2.0" in launcher
    assert "--per-rank-batch 16" in launcher
    assert "Flow evaluation must finish before LPL resumes" in launcher


def test_flow_evaluation_waiter_orders_sampling_evaluation_and_resume() -> None:
    waiter = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "wait_evaluate_raev2_flow_then_resume.sh"
    ).read_text(encoding="utf-8")
    sampling = waiter.index("expected_summaries")
    evaluation = waiter.index("evaluate_raev2_flow_curve.py")
    resume = waiter.index("launch_raev2_long_pipeline.sh")
    assert sampling < evaluation < resume


def test_curve_writer_adds_objective_and_step(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "branch": "official",
                "frechet_inception_distance": 10.0,
                "kernel_inception_distance_mean": 0.01,
                "inception_score_mean": 100.0,
            },
            {
                "branch": "flow_s0010",
                "frechet_inception_distance": 9.9,
                "kernel_inception_distance_mean": 0.009,
                "inception_score_mean": 101.0,
            },
            {
                "branch": "lpl_s0010",
                "frechet_inception_distance": 9.7,
                "kernel_inception_distance_mean": 0.008,
                "inception_score_mean": 102.0,
            },
        ]
    ).to_csv(source, index=False)

    output_csv = tmp_path / "curve.csv"
    output_png = tmp_path / "curve.png"
    write_curve(source, output_csv=output_csv, output_png=output_png)

    curve = pd.read_csv(output_csv)
    assert curve[["objective", "branch_update"]].values.tolist() == [
        ["flow", 10],
        ["lpl", 10],
        ["official", 0],
    ]
    assert output_png.stat().st_size > 0


def test_flow_curve_writer_orders_steps_and_draws_plot(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "branch": "flow_s0020",
                "frechet_inception_distance": 9.8,
                "kernel_inception_distance_mean": 0.008,
                "inception_score_mean": 102.0,
            },
            {
                "branch": "official",
                "frechet_inception_distance": 10.0,
                "kernel_inception_distance_mean": 0.01,
                "inception_score_mean": 100.0,
            },
            {
                "branch": "flow_s0010",
                "frechet_inception_distance": 9.9,
                "kernel_inception_distance_mean": 0.009,
                "inception_score_mean": 101.0,
            },
        ]
    ).to_csv(source, index=False)

    output_csv = tmp_path / "flow_curve.csv"
    output_png = tmp_path / "flow_curve.png"
    write_flow_curve(source, output_csv=output_csv, output_png=output_png)

    curve = pd.read_csv(output_csv)
    assert curve["branch_update"].tolist() == [0, 10, 20]
    assert output_png.stat().st_size > 0


def _write_sampling_artifacts(
    directory: Path,
    *,
    branch: str,
    noise_suffix: str = "",
) -> None:
    directory.mkdir(parents=True)
    archive = directory / "samples.npz"
    archive.write_bytes(b"placeholder")
    (directory / "sampling_summary.json").write_text(
        json.dumps(
            {
                "branch": branch,
                "archive": str(archive),
                "archive_sha256": f"sha-{branch}",
                "samples": 5000,
            }
        ),
        encoding="utf-8",
    )
    for rank in range(4):
        (directory / f"sampling_audit_rank{rank}.json").write_text(
            json.dumps(
                {
                    "protocol": "raev2_same_noise_v1",
                    "world_size": 4,
                    "sampling_seed": 0,
                    "sample_count": 5000,
                    "per_rank_batch": 32,
                    "sampler_steps": 100,
                    "guidance_cfg_scale": 1.0,
                    "guidance_ig_scale": 1.78,
                    "guidance_ig_t_min": 0.1,
                    "initial_generator_sha256": f"initial-{rank}",
                    "first_noise_sha256": f"noise-{rank}{noise_suffix}",
                    "first_label_sha256": f"labels-{rank}",
                    "first_labels": [rank],
                    "final_generator_sha256": f"final-{rank}",
                }
            ),
            encoding="utf-8",
        )


def test_same_noise_audit_detects_branch_mismatch(tmp_path: Path) -> None:
    official = tmp_path / "official"
    flow = tmp_path / "flow"
    _write_sampling_artifacts(official, branch="official")
    _write_sampling_artifacts(flow, branch="flow")
    verify_same_noise_protocol({"official": official, "flow": flow})

    audit = flow / "sampling_audit_rank2.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["first_noise_sha256"] = "different"
    audit.write_text(json.dumps(payload), encoding="utf-8")
    try:
        verify_same_noise_protocol({"official": official, "flow": flow})
    except ValueError as error:
        assert "first_noise_sha256" in str(error)
    else:
        raise AssertionError("same-noise mismatch was not detected")


def test_per_branch_evaluation_can_be_validated_and_merged(
    tmp_path: Path,
) -> None:
    inputs = {}
    for branch in ("official", "flow_s0010"):
        sample_dir = tmp_path / "samples" / branch
        _write_sampling_artifacts(sample_dir, branch=branch)
        metrics = tmp_path / "metrics" / f"{branch}.csv"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "branch": branch,
                    "sample_sha256": f"sha-{branch}",
                    "sample_count": 5000,
                    "frechet_inception_distance": 10.0,
                }
            ]
        ).to_csv(metrics, index=False)
        assert evaluation_complete(
            metrics,
            branch=branch,
            sample_directory=sample_dir,
        )
        inputs[branch] = metrics

    merged = tmp_path / "combined.csv"
    merge_evaluations(inputs, merged)
    assert pd.read_csv(merged)["branch"].tolist() == [
        "official",
        "flow_s0010",
    ]

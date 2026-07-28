from __future__ import annotations

import json
from pathlib import Path

from experiments.summarize_rae_lpl_authenticity import summarize


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def provenance_fields() -> dict[str, str]:
    return {
        "sampling_provenance_protocol": "strict-sampling-provenance-v1",
        "label_sampler_version": "interleaved-v3-provenance",
        "endpoint_checkpoint_sha256": "endpoint",
        "sampling_checkpoint_sha256": "sampling",
        "sampling_provenance_sha256": "provenance",
        "sample_npz_sha256": "samples",
    }


def write_seed(results: Path, seed: int, flow_fid: float, lpl_fid: float) -> None:
    prior = "ditdh_s_ep20"
    for objective, weight in (("flow", 0.0), ("full", 0.001)):
        branch = results / f"{prior}_seed{seed}_{objective}_to_s5000"
        write_json(
            branch / "manifest.json",
            {
                "source_checkpoint_sha256": "source",
                "lpl_weight": weight,
            },
        )
        row = {
            "step": 5000,
            "total_loss": 1.0 + weight * 100.0,
            "flow_loss": 1.0,
            "lpl_batch_contribution": 0.0 if objective == "flow" else 100.0,
            "eligible_rate": 0.2,
            "grad_norm": 0.5,
            "clip_rate": 0.0,
        }
        (branch / "metrics.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    write_json(
        results / f"{prior}_seed{seed}_pair_audit_s5000.json",
        {"passed": True, "errors": []},
    )
    write_json(
        results / f"eval_{prior}_seed{seed}_pair_s5000_n5000_seed20260715.json",
        [
            {
                "branch": "flow",
                "frechet_inception_distance": flow_fid,
                "kernel_inception_distance_mean": 0.02,
                "inception_score_mean": 10.0,
                **provenance_fields(),
            },
            {
                "branch": "lpl",
                "frechet_inception_distance": lpl_fid,
                "kernel_inception_distance_mean": 0.01,
                "inception_score_mean": 11.0,
                **provenance_fields(),
            },
        ],
    )


def test_summary_refuses_to_call_incomplete_evidence_strong(tmp_path: Path) -> None:
    write_seed(tmp_path, 4101, 20.0, 18.0)
    official = tmp_path / "official.json"
    write_json(
        official,
        {
            "branch": "official",
            "frechet_inception_distance": 19.0,
            "kernel_inception_distance_mean": 0.015,
            "inception_score_mean": 10.5,
            **provenance_fields(),
        },
    )

    result = summarize(
        tmp_path,
        prior="ditdh_s_ep20",
        seeds=[4101, 4102, 4103],
        endpoint=5000,
        sampling_seed=20260715,
        official_evaluation=official,
    )

    assert result["completed_seeds"] == 1
    assert not result["fixed_seed_training_gate_passed"]
    assert not result["strong_reproduction_complete"]
    assert result["rows"][0]["fid_lpl_minus_flow"] == -2.0


def test_fixed_seed_gate_requires_all_three_paired_improvements(tmp_path: Path) -> None:
    for seed in (4101, 4102, 4103):
        write_seed(tmp_path, seed, 20.0, 18.0)
    official = tmp_path / "official.json"
    write_json(
        official,
        {
            "branch": "official",
            "frechet_inception_distance": 19.0,
            "kernel_inception_distance_mean": 0.015,
            "inception_score_mean": 10.5,
            **provenance_fields(),
        },
    )

    result = summarize(
        tmp_path,
        prior="ditdh_s_ep20",
        seeds=[4101, 4102, 4103],
        endpoint=5000,
        sampling_seed=20260715,
        official_evaluation=official,
    )

    assert result["fixed_seed_training_gate_passed"]
    assert not result["strong_reproduction_complete"]
    assert result["fid_improvements"] == 3
    assert result["kid_improvements"] == 3


def test_fixed_seed_gate_rejects_legacy_samples_without_provenance(
    tmp_path: Path,
) -> None:
    for seed in (4101, 4102, 4103):
        write_seed(tmp_path, seed, 20.0, 18.0)
    evaluation = (
        tmp_path
        / "eval_ditdh_s_ep20_seed4102_pair_s5000_n5000_seed20260715.json"
    )
    rows = json.loads(evaluation.read_text(encoding="utf-8"))
    rows[1].pop("sampling_provenance_sha256")
    write_json(evaluation, rows)
    official = tmp_path / "official.json"
    write_json(
        official,
        {
            "branch": "official",
            "frechet_inception_distance": 19.0,
            "kernel_inception_distance_mean": 0.015,
            "inception_score_mean": 10.5,
            **provenance_fields(),
        },
    )

    result = summarize(
        tmp_path,
        prior="ditdh_s_ep20",
        seeds=[4101, 4102, 4103],
        endpoint=5000,
        sampling_seed=20260715,
        official_evaluation=official,
    )

    assert not result["all_sampling_provenance_valid"]
    assert not result["fixed_seed_training_gate_passed"]


def test_strong_gate_requires_three_provenance_valid_sampling_seeds(
    tmp_path: Path,
) -> None:
    for seed in (4101, 4102, 4103):
        write_seed(tmp_path, seed, 20.0, 18.0)
    source = (
        tmp_path
        / "eval_ditdh_s_ep20_seed4102_pair_s5000_n5000_seed20260715.json"
    )
    rows = json.loads(source.read_text(encoding="utf-8"))
    for sampling_seed in (20260716, 20260717):
        write_json(
            tmp_path
            / (
                "eval_ditdh_s_ep20_seed4102_pair_s5000_n5000_"
                f"seed{sampling_seed}.json"
            ),
            rows,
        )
    official = tmp_path / "official.json"
    write_json(
        official,
        {
            "branch": "official",
            "frechet_inception_distance": 19.0,
            "kernel_inception_distance_mean": 0.015,
            "inception_score_mean": 10.5,
            **provenance_fields(),
        },
    )

    result = summarize(
        tmp_path,
        prior="ditdh_s_ep20",
        seeds=[4101, 4102, 4103],
        endpoint=5000,
        sampling_seed=20260715,
        official_evaluation=official,
        stability_training_seed=4102,
        stability_sampling_seeds=[20260715, 20260716, 20260717],
    )

    assert result["fixed_seed_training_gate_passed"]
    assert result["multi_sampling_seed_gate_evaluated"]
    assert result["multi_sampling_seed_gate_passed"]
    assert result["strong_reproduction_complete"]


def test_official_sampling_seed_comparison_preserves_metric_tradeoff(
    tmp_path: Path,
) -> None:
    for seed in (4101, 4102, 4103):
        write_seed(tmp_path, seed, 20.0, 18.0)
    source = (
        tmp_path
        / "eval_ditdh_s_ep20_seed4102_pair_s5000_n5000_seed20260715.json"
    )
    paired_rows = json.loads(source.read_text(encoding="utf-8"))
    for sampling_seed in (20260716, 20260717):
        write_json(
            tmp_path
            / (
                "eval_ditdh_s_ep20_seed4102_pair_s5000_n5000_"
                f"seed{sampling_seed}.json"
            ),
            paired_rows,
        )

    official_paths = []
    for sampling_seed in (20260715, 20260716, 20260717):
        path = tmp_path / f"official_{sampling_seed}.json"
        write_json(
            path,
            {
                "branch": "official",
                "sampling_seed": sampling_seed,
                "frechet_inception_distance": 19.0,
                "kernel_inception_distance_mean": 0.005,
                "inception_score_mean": 12.0,
                **provenance_fields(),
            },
        )
        official_paths.append(path)

    result = summarize(
        tmp_path,
        prior="ditdh_s_ep20",
        seeds=[4101, 4102, 4103],
        endpoint=5000,
        sampling_seed=20260715,
        official_evaluation=official_paths[0],
        stability_training_seed=4102,
        stability_sampling_seeds=[20260715, 20260716, 20260717],
        official_stability_evaluations=official_paths[1:],
    )

    comparison = result["official_sampling_seed_comparison"]
    assert comparison["evaluated"]
    assert comparison["all_lpl_fid_better_than_official"]
    assert not comparison["all_lpl_kid_better_than_official"]
    assert not comparison["all_lpl_is_better_than_official"]
    assert len(comparison["rows"]) == 3

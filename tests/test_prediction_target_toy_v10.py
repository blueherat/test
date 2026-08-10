from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from experiments import run_prediction_target_extrapolation_toy_v10_final as v10


def make_embedding(*, dimension: int = 8, seed: int = 11) -> v10.ControlledEmbedding:
    return v10.ControlledEmbedding(
        dimension,
        curvature=0.3,
        frequency_scale=3.0,
        seed=seed,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
        energy_match=False,
        calibration_samples=16,
        data_jitter=0.015,
    )


def test_batched_sampler_is_conditionwise_identical_to_sequential_sampler() -> None:
    device = torch.device("cpu")
    embedding = make_embedding()
    locator = v10.SpiralLocator(points=64, device=device)
    models = v10.v4.build_same_init_models(
        embedding.D, 16, 3, 8, device, seed=17
    )
    conditions = [
        v10.Condition("x", "x"),
        v10.Condition("v", "v"),
        v10.Condition("eps", "eps"),
        v10.Condition("xv_raw", "xv", 0.2),
        v10.Condition("xeps_raw", "xeps", -0.1),
        v10.Condition("xv_abs", "xv_absnorm", 0.03),
        v10.Condition("xeps_rel", "xeps_rel", 0.04),
        v10.Condition("xv_shuffle", "xv_shuffle", 0.1),
        v10.Condition("xeps_gauss", "xeps_gausscov", 0.1),
        v10.Condition("xv_random", "xv_random", 0.1),
        v10.Condition("xv_curve", "xv_curve", 0.1),
        v10.Condition("xeps_ridge", "xeps_ridge", 0.1),
        v10.Condition("xv_ambient", "xv_ambient", 0.1),
        v10.Condition("xeps_curve_rel", "xeps_curve_rel", 0.03),
        v10.Condition("xv_ridge_abs", "xv_ridge_absnorm", 0.02),
        v10.Condition("xv_window", "xv", 0.2, t_low=0.4, t_high=0.7),
    ]
    common = dict(
        models=models,
        emb=embedding,
        locator=locator,
        sample_count=8,
        batch_size=4,
        steps=4,
        t_max=0.9,
        t_min=0.1,
        clip=0.02,
        seed=29,
        device=device,
        diag_stride=1,
    )

    sequential_samples: list[np.ndarray] = []
    sequential_diagnostics: list[dict] = []
    for condition in conditions:
        samples, diagnostics = v10.sample_condition_recursive_v10(
            condition=condition, **common
        )
        sequential_samples.append(samples)
        sequential_diagnostics.extend(diagnostics)

    batched_samples, batched_diagnostics = v10.sample_conditions_batched_v10(
        conditions=conditions, **common
    )

    for sequential, batched in zip(sequential_samples, batched_samples):
        np.testing.assert_array_equal(batched, sequential)
    assert batched_diagnostics == sequential_diagnostics
    sequential_by_key = {
        (row["condition"], row["step"]): row for row in sequential_diagnostics
    }
    batched_by_key = {
        (row["condition"], row["step"]): row for row in batched_diagnostics
    }
    assert batched_by_key == sequential_by_key


def _train_small_triplet(
    output_dir: Path,
    *,
    fixed_steps: int,
    resume: bool,
) -> dict[str, torch.nn.Module]:
    embedding = make_embedding(dimension=6, seed=31)
    return v10.train_triplet_v10(
        emb=embedding,
        hidden=8,
        depth=3,
        time_dim=4,
        training_mode="fixed",
        fixed_steps=fixed_steps,
        max_steps=fixed_steps,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=10.0,
        loss_space="v",
        t_min=0.1,
        t_max=0.9,
        clip=0.02,
        jitter=0.015,
        val_times=(0.3, 0.7),
        val_samples_per_time=8,
        val_batch_size=8,
        val_every=2,
        patience_evals=20,
        min_rel_improve=0.0,
        scheduler="constant",
        warmup_steps=0,
        seed=37,
        device=torch.device("cpu"),
        output_dir=output_dir,
        resume_training=resume,
        checkpoint_every=1,
    )


def test_training_resume_is_bitwise_identical_to_uninterrupted_training(
    tmp_path: Path,
) -> None:
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted = _train_small_triplet(
        uninterrupted_dir, fixed_steps=4, resume=False
    )
    _train_small_triplet(resumed_dir, fixed_steps=2, resume=False)
    resumed = _train_small_triplet(resumed_dir, fixed_steps=4, resume=True)

    for target in ("x", "v", "eps"):
        uninterrupted_state = uninterrupted[target].state_dict()
        resumed_state = resumed[target].state_dict()
        assert uninterrupted_state.keys() == resumed_state.keys()
        for name in uninterrupted_state:
            assert torch.equal(uninterrupted_state[name], resumed_state[name]), (
                target,
                name,
            )

    checkpoint = torch.load(
        resumed_dir / "training_checkpoint_v10.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["step"] == 4
    assert (resumed_dir / "models_v10.pt").is_file()


def test_raw_gamma_conditions_are_available_to_visual_atlas() -> None:
    marker = np.zeros((2, 2), dtype=np.float32)
    cache = {
        "v": marker,
        "x": marker,
        "xv_g0p3": marker,
        "xv_g1": marker,
    }
    entries = v10._atlas_entries_for_pair(
        "xv",
        cache=cache,
        path_alphas=(0.0, 1.0),
        raw_gammas=(0.3, 1.0),
    )
    assert [(alpha, name) for alpha, name, _ in entries] == [
        (0.0, "v"),
        (1.0, "x"),
        (1.3, "xv_g0p3"),
        (2.0, "xv_g1"),
    ]


def _write_worker_metrics(path: Path, *, seed: int, x_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "D",
                "curvature",
                "hidden",
                "seed",
                "training_source",
                "condition",
                "kind",
                "strength",
                "t_low",
                "t_high",
                "swd2",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "D": 8,
                "curvature": 0.5,
                "hidden": 16,
                "seed": seed,
                "training_source": "v4",
                "condition": "x",
                "kind": "x",
                "strength": 0.0,
                "t_low": 0.0,
                "t_high": 1.0,
                "swd2": x_value,
            }
        )


def test_three_seed_aggregation_reports_mean_std_and_sem(tmp_path: Path) -> None:
    for seed, value in zip((11, 12, 13), (1.0, 2.0, 3.0)):
        _write_worker_metrics(
            tmp_path / f"worker_seed{seed}" / "generation_metrics_v10_all.csv",
            seed=seed,
            x_value=value,
        )

    combined_path, summary_path = v10.aggregate_worker_outputs(tmp_path)
    combined = v10.load_csv(combined_path)
    summary = v10.load_csv(summary_path)
    assert len(combined) == 3
    assert len(summary) == 1
    row = summary[0]
    assert row["seed_count"] == "3"
    assert row["seeds"] == "11,12,13"
    assert float(row["swd2_mean"]) == 2.0
    assert float(row["swd2_std"]) == 1.0
    np.testing.assert_allclose(float(row["swd2_sem"]), 1.0 / np.sqrt(3.0))


def test_aggregation_rejects_duplicate_seed_condition_rows(tmp_path: Path) -> None:
    _write_worker_metrics(
        tmp_path / "worker_seed11_a" / "generation_metrics_v10_all.csv",
        seed=11,
        x_value=1.0,
    )
    _write_worker_metrics(
        tmp_path / "worker_seed11_b" / "generation_metrics_v10_all.csv",
        seed=11,
        x_value=2.0,
    )
    try:
        v10.aggregate_worker_outputs(tmp_path)
    except ValueError as error:
        assert "duplicate seed/condition" in str(error)
    else:
        raise AssertionError("duplicate seed/condition rows must not be averaged")

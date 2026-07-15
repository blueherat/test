import pandas as pd

from experiments.gauge_large_validation import (
    LargeValidationConfig,
    _add_time_holm_significance,
    _confidence_summary,
    add_large_identity_ratios,
    build_tasks,
    task_name,
)


def test_large_grid_has_280_unique_paired_tasks():
    config = LargeValidationConfig()
    tasks = build_tasks(config)
    names = [task_name(task) for task in tasks]

    assert len(tasks) == 2 * 4 * 7 * 5
    assert len(set(names)) == len(names)


def test_identity_ratios_are_paired_within_train_size_probe_seed_and_step():
    rows = []
    for train_count, identity_loss in ((1024, 2.0), (2048, 4.0)):
        for seed in (0, 1):
            rows.extend(
                [
                    {
                        "train_count": train_count,
                        "probe": "local_rf5",
                        "seed": seed,
                        "step": 10,
                        "gauge": "identity",
                        "relative_mse": identity_loss + seed,
                    },
                    {
                        "train_count": train_count,
                        "probe": "local_rf5",
                        "seed": seed,
                        "step": 10,
                        "gauge": "candidate",
                        "relative_mse": 0.9 * (identity_loss + seed),
                    },
                ]
            )

    ratios = add_large_identity_ratios(pd.DataFrame(rows))
    candidates = ratios[ratios["gauge"] == "candidate"]

    assert candidates["loss_ratio_to_identity"].round(8).eq(0.9).all()


def test_time_holm_controls_curve_and_full_screen_families():
    rows = []
    for probe in ("local_rf5", "global_attn"):
        for time_bin in range(2):
            for seed, ratio in enumerate((0.90, 0.91, 0.89, 0.92, 0.88)):
                rows.append(
                    {
                        "train_count": 1024,
                        "probe": probe,
                        "gauge": "candidate",
                        "time_bin": time_bin,
                        "t_center": time_bin + 0.5,
                        "logsnr": -float(time_bin),
                        "seed": seed,
                        "loss_ratio_to_identity": ratio if time_bin == 0 else 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    summary = _confidence_summary(
        frame,
        ["train_count", "probe", "gauge", "time_bin", "t_center", "logsnr"],
        "loss_ratio_to_identity",
    )
    result = _add_time_holm_significance(frame, summary)

    signal = result[result["time_bin"] == 0]
    null = result[result["time_bin"] == 1]
    assert signal["holm_screen_better"].all()
    assert null["p_value"].isna().all()
    assert (signal["p_holm_screen"] >= signal["p_holm_within_curve"]).all()

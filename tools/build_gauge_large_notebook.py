from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/gauge_large_validation.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        markdown(
            """
            # Large-Scale Orthogonal Gauge Validation

            ## tl;dr

            Across 280 paired ImageNet runs, no non-identity static gauge improves held-out probe
            risk after Holm correction. The numerically best setting, `haar_2x2/local_rf17`, is
            only about 0.11% better at both train sizes and is not multiplicity-corrected
            significant. Haar is instead reliably worse for small/global probes, with a penalty
            that shrinks over training and with larger receptive field. The experiment therefore
            supports finite-horizon, architecture-dependent coordinate sensitivity, but not a
            useful alternative latent gauge or a complete explanation of generation quality.
            """
        ),
        markdown(
            """
            ## Context & Methods

            The encoder, decoder, latent samples, validation images, noise, time samples,
            initialization, optimizer, and training budget are paired across gauges. Train data
            come from ImageNet-1k `train`; held-out evaluation uses ImageNet-1k `validation`, so
            there is no split leakage. The 1024-image train set is a strict subset of the
            2048-image set.

            ### Key Assumptions

            - Probe loss is a screening metric, not a substitute for FID.
            - H2 tables include Holm correction within each train size. The pre-specified
              all-pass locality pattern remains the primary mechanism test.
            - Time-bin claims use Holm correction over the full exploratory screen, not just one
              selected curve.
            - The 1024 set is nested inside the 2048 set; agreement is a consistency check, not
              an independent data replication.
            - Two thousand steps test a substantially longer finite horizon than the smoke run,
              but do not replace a later 20k-50k confirmation if a signal survives.
            """
        ),
        markdown("### 1. Setup"),
        code(
            """
            import json
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from experiments.gauge_large_validation import (
                LargeValidationConfig,
                aggregate_results,
                large_gauges,
                load_latent_cache,
                mechanism_evidence_tables,
                plot_final_seed_summary,
                plot_learning_summary,
                plot_locality_summary,
                plot_time_summary,
                task_completion_table,
            )
            from experiments.architecture_gauge import exact_equivalence_table
            """
        ),
        markdown("### 2. Data and experiment contract"),
        code(
            """
            RESULT_DIR = Path.home() / "data/eqvae/artifacts/gauge_large_validation"
            CONFIG = LargeValidationConfig(
                train_counts=(1024, 2048),
                val_count=512,
                steps=2000,
                eval_steps=(0, 100, 500, 1000, 2000),
                batch_size=32,
                seeds=(0, 1, 2, 3, 4),
                hidden=16,
                time_bins=8,
                dataset_path="/data/shared/imagenet-1k",
            )

            saved_config = json.loads((RESULT_DIR / "config.json").read_text())
            completion = task_completion_table(RESULT_DIR, config=CONFIG)
            display(pd.DataFrame([saved_config]))
            display(
                completion.groupby(["train_count", "complete"], as_index=False)
                .size()
                .rename(columns={"size": "task_count"})
            )
            assert len(completion) == 280
            assert completion["complete"].all(), "Large validation is incomplete. Resume the runner first."
            """
        ),
        markdown("### 3. Exact orthogonal gate"),
        code(
            """
            CACHE_PATH = Path.home() / "data/eqvae/cache/gauge_large/rae_dinov2_imagenet_2048_512.pt"
            cache = load_latent_cache(CACHE_PATH)
            exact = exact_equivalence_table(
                cache["val_latents"][:16],
                large_gauges(),
                device="cuda:0",
            )
            display(exact.round(8))
            assert exact[["inverse_rel_l2", "norm_rel_error", "paired_noise_rel_error"]].to_numpy().max() < 1e-5
            """
        ),
        markdown(
            """
            ## Results

            ### 4. Aggregate paired runs
            """
        ),
        code(
            """
            tables = aggregate_results(RESULT_DIR)
            evidence = mechanism_evidence_tables(tables)
            print("history rows:", len(tables["history"]))
            print("final paired rows:", len(tables["final"]))
            print("time-bin rows:", len(tables["time_ratios"]))
            """
        ),
        markdown(
            """
            ### 5. H2: does any gauge beat native identity?

            The y-axis is percentage change relative to the paired identity run. Negative is
            better. Error bars are two-sided 95% t intervals over five seeds; the focused delta
            scale is intentional and the zero line is the exact identity benchmark.
            """
        ),
        code(
            """
            plot_final_seed_summary(tables["final_summary"]);

            h2_columns = [
                "train_count", "probe", "gauge", "mean", "ci95_low", "ci95_high",
                "relative_improvement", "seed_wins", "p_value", "p_holm",
                "h2_statistically_supported",
            ]
            for train_count in CONFIG.train_counts:
                print(f"train={train_count}: best paired candidates")
                display(evidence["h2"][evidence["h2"]["train_count"] == train_count][h2_columns].head(12))
            """
        ),
        markdown("### 6. Does the ranking reproduce at train=1024 and 2048?"),
        code(
            """
            size_comparison = evidence["h2"].pivot_table(
                index=["probe", "gauge"],
                columns="train_count",
                values="mean",
                aggfunc="first",
            ).reset_index()
            size_comparison["ratio_difference_2048_minus_1024"] = (
                size_comparison[2048] - size_comparison[1024]
            )
            display(size_comparison.sort_values([1024, 2048]).round(6))
            """
        ),
        markdown(
            """
            ### 7. Pre-specified higher-order locality test

            This subtracts the global probe's gauge ratio from each local probe's ratio within
            the same train size, gauge, and seed. A positive penalty that is strongest at RF5 and
            shrinks toward zero as RF grows is the predicted locality signature.
            """
        ),
        code(
            """
            plot_locality_summary(evidence["locality"]);
            display(
                evidence["locality"][
                    ["train_count", "gauge", "probe", "receptive_field", "mean", "ci95_low", "ci95_high"]
                ].round(6)
            )
            """
        ),
        markdown("### 8. Finite-horizon behavior"),
        code(
            """
            for probe in ("local_rf5", "local_rf9", "local_rf17", "global_attn"):
                plot_learning_summary(tables["learning_summary"], probe=probe);

            display(
                evidence["finite_horizon"]
                .query("gauge != 'identity'")
                .sort_values("distance_shrink", ascending=False)
                .head(20)
                .round(6)
            )
            """
        ),
        markdown(
            """
            ### 9. Identity-neighborhood direction and time-bin conflict

            `allpass_r3` shows an exploratory sign change over time in the global probe, but no
            bin survives correction over the full time-bin screen. The robust time-localized
            signal is Haar degradation near the noisy endpoint, not all-pass improvement.
            """
        ),
        code(
            """
            display(evidence["directional"].round(6))

            plot_time_summary(tables["time_summary"], probe="global_attn");

            supported_conflicts = evidence["time_conflicts"].query("supported_sign_conflict")
            print("supported time-bin sign conflicts:", len(supported_conflicts))
            display(supported_conflicts.round(6))

            time_columns = [
                "train_count", "probe", "gauge", "time_bin", "t_center", "mean",
                "ci95_low", "ci95_high", "p_holm_within_curve", "p_holm_screen",
                "holm_screen_significant",
            ]
            display(
                tables["time_summary"]
                .query("probe == 'global_attn' and gauge in ['allpass_r3', 'haar_2x2']")
                [time_columns]
                .round(6)
            )
            """
        ),
        markdown(
            """
            ## Takeaways

            The following cell applies explicit project-management gates. `statistically detected`
            means the five-seed CI excludes identity and the within-train-size Holm-adjusted
            p-value is below 0.05; `method-scale` additionally means at least 2% lower held-out
            probe risk. Neither label establishes FID improvement.
            """
        ),
        code(
            """
            for train_count in CONFIG.train_counts:
                rows = evidence["h2"][evidence["h2"]["train_count"] == train_count]
                detected = rows[rows["h2_statistically_supported"]]
                method_scale = detected[detected["relative_improvement"] >= 0.02]
                best = rows.iloc[0]
                print(
                    f"train={train_count}: best={best['probe']}/{best['gauge']}, "
                    f"mean ratio={best['mean']:.6f}, 95% CI=[{best['ci95_low']:.6f}, {best['ci95_high']:.6f}], "
                    f"seed wins={int(best['seed_wins'])}/5"
                )
                print(
                    f"  statistically detected candidates={len(detected)}; "
                    f"method-scale candidates={len(method_scale)}"
                )

            locality_r3 = evidence["locality"].query("gauge == 'allpass_r3'")
            display(locality_r3.round(6))
            print(
                "Interpret H2 only after checking cross-size replication, the pre-specified locality "
                "pattern, multiple-comparison risk, and eventual matched-compute generation metrics."
            )
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build_notebook()

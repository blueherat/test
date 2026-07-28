"""Relate frozen-decoder sensitivity to variance and cross-layer predictability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_ATLAS = (
    DEFAULT_RESULTS / "evaluation/decoder_subspace_atlas_v1/decoder_subspace_secant_samples.csv"
)
DEFAULT_BASIS_METRICS = (
    DEFAULT_RESULTS / "evaluation/predictability_basis_v1/basis_block_metrics.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "evaluation/decoder_predictability_alignment_v1"


def standardized_regression(
    table: pd.DataFrame, outcome: str
) -> dict[str, float | str]:
    values = table[
        ["val_final_variance_per_dimension", "val_r2", outcome]
    ].dropna()
    variance = np.log(
        values["val_final_variance_per_dimension"].to_numpy().clip(1e-20)
    )
    predictability = values["val_r2"].to_numpy()
    target = np.log(values[outcome].to_numpy().clip(1e-20))

    def standardize(array: np.ndarray) -> np.ndarray:
        return (array - array.mean()) / array.std(ddof=0).clip(min=1e-20)

    design = np.column_stack(
        [np.ones(len(values)), standardize(variance), standardize(predictability)]
    )
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = design @ coefficients
    total = np.square(target - target.mean()).sum()
    combined_r2 = 1.0 - float(np.square(target - prediction).sum() / max(total, 1e-20))

    def single_r2(feature: np.ndarray) -> float:
        single = np.column_stack([np.ones(len(feature)), standardize(feature)])
        estimate = single @ np.linalg.lstsq(single, target, rcond=None)[0]
        return 1.0 - float(np.square(target - estimate).sum() / max(total, 1e-20))

    return {
        "outcome": outcome,
        "block_count": int(len(values)),
        "variance_beta": float(coefficients[1]),
        "predictability_beta": float(coefficients[2]),
        "variance_only_r2": single_r2(variance),
        "predictability_only_r2": single_r2(predictability),
        "combined_r2": combined_r2,
    }


def family_leaveout_regression(table: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for family in sorted(table["basis_family"].unique()):
        row = standardized_regression(table[table["basis_family"] != family], outcome)
        row["left_out_family"] = family
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_atlas(samples: pd.DataFrame) -> pd.DataFrame:
    largest_fraction = float(samples["fraction"].max())
    selected = samples[samples["fraction"] == largest_fraction]
    metric_columns = [
        column for column in samples if column.startswith("decoder_")
    ]
    return selected.groupby("basis", as_index=False)[metric_columns].mean()


def plot_alignment(table: pd.DataFrame, regressions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    colors = {"absolute": "#2678a8", "fractional": "#2f855a", "pca": "#8b5a9f"}
    outcomes = (
        ("decoder_pixel_mse_gain", "pixel MSE gain"),
        ("decoder_lpips_secant", "LPIPS / latent MSE"),
    )
    for axis, (outcome, label) in zip(axes[:2], outcomes):
        for family, frame in table.groupby("basis_family"):
            axis.scatter(
                frame["val_r2"],
                frame[outcome],
                color=colors.get(family, "#777777"),
                label=family,
                s=48,
                alpha=0.85,
            )
        axis.set_yscale("log")
        axis.set_xlabel("held-out cross-layer predictability R2")
        axis.set_ylabel(label)
        axis.set_title(f"Decoder sensitivity vs predictability: {label}")
        axis.grid(alpha=0.25)
    display = regressions.set_index("outcome").loc[
        ["decoder_pixel_mse_gain", "decoder_lpips_secant"]
    ]
    x = np.arange(len(display))
    width = 0.25
    axes[2].bar(x - width, display["variance_only_r2"], width, label="variance only")
    axes[2].bar(x, display["predictability_only_r2"], width, label="predictability only")
    axes[2].bar(x + width, display["combined_r2"], width, label="combined")
    axes[2].set_xticks(x, ["pixel MSE", "LPIPS"])
    axes[2].set_ylabel("log-sensitivity regression R2")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("What explains decoder anisotropy?")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output / "decoder_predictability_alignment.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    samples = pd.read_csv(args.atlas.expanduser())
    basis_metrics = pd.read_csv(args.basis_metrics.expanduser())
    atlas = aggregate_atlas(samples)
    table = basis_metrics.merge(atlas, on="basis", validate="one_to_one")
    table = table[table["basis_family"].isin(["absolute", "fractional", "pca"])]
    outcomes = [
        "decoder_embed_gain",
        "decoder_pixel_mse_gain",
        "decoder_l1_secant",
        "decoder_lpips_secant",
        "decoder_hidden_1_mse_gain",
        "decoder_hidden_2_mse_gain",
        "decoder_hidden_3_mse_gain",
        "decoder_hidden_4_mse_gain",
    ]
    regressions = pd.DataFrame(
        [standardized_regression(table, outcome) for outcome in outcomes]
    )
    leaveout = pd.concat(
        [family_leaveout_regression(table, outcome) for outcome in outcomes],
        ignore_index=True,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "decoder_basis_alignment.csv", index=False)
    regressions.to_csv(output / "decoder_alignment_regressions.csv", index=False)
    leaveout.to_csv(output / "decoder_alignment_family_leaveout.csv", index=False)
    plot_alignment(table, regressions, output)
    pixel = regressions.set_index("outcome").loc["decoder_pixel_mse_gain"]
    lpips = regressions.set_index("outcome").loc["decoder_lpips_secant"]
    summary = {
        "block_count": int(len(table)),
        "regression": regressions.to_dict(orient="records"),
        "interpretation_gate": {
            "predictability_positive_pixel": bool(pixel["predictability_beta"] > 0.0),
            "predictability_positive_lpips": bool(lpips["predictability_beta"] > 0.0),
            "combined_improves_pixel_r2_by_005": bool(
                pixel["combined_r2"] - pixel["variance_only_r2"] >= 0.05
            ),
            "combined_improves_lpips_r2_by_005": bool(
                lpips["combined_r2"] - lpips["variance_only_r2"] >= 0.05
            ),
        },
        "caveat": (
            "Exploratory 24-block atlas from one frozen decoder and one held-out latent "
            "sample split; basis construction used disjoint ImageNet train data."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--basis-metrics", type=Path, default=DEFAULT_BASIS_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

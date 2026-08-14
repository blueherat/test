#!/usr/bin/env python3
"""Summarize 800K causal tests of nominal-path frozen guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ROOT = BASE / "nominal_guidance_transfer_800k_v1"
DEFAULT_PRIOR = (
    REPO_ROOT
    / "docs/data/imagenet100_sit_800k_compact_replication/compact_replication_summary.json"
)
FAMILIES = ("x800", "v500")


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_5k_rows(path: Path, seed: int) -> list[dict[str, object]]:
    payload = _load(path)
    seed_entry = next(
        entry for entry in payload["seed_metrics"] if int(entry["sample_seed"]) == seed
    )
    conditions = seed_entry["conditions"]
    rows: list[dict[str, object]] = []
    mapping = {
        "x800": ("x_frozen", "x_closed"),
        "v500": ("vweak_frozen", "vweak_closed"),
    }
    for family, (frozen_key, closed_key) in mapping.items():
        for condition, key in (
            ("baseline", "baseline"),
            ("frozen", frozen_key),
            ("closed", closed_key),
        ):
            source = conditions[key]
            rows.append(
                {
                    "family": family,
                    "sample_seed": seed,
                    "num_samples": 5000,
                    "condition": condition,
                    "fid": float(source["fid"]),
                    "sfid": float(source["sfid"]),
                    "inception_score": float(source["inception_score"]),
                    "noise_fingerprint": source["noise_fingerprint"],
                    "label_fingerprint": source["label_fingerprint"],
                    "source": source["source"],
                }
            )
    return rows


def _intervention_row(
    root: Path,
    *,
    family: str,
    mode: str,
    num_samples: int,
    seed: int,
    section: str = "causal_screen",
) -> dict[str, object]:
    directory = root / section / family / f"{mode}_n{num_samples}_seed{seed}"
    payload = _load(directory / "nominal_intervention_fid5k.json")
    if payload["mode"] != mode or int(payload["num_samples"]) != num_samples:
        raise ValueError(f"intervention metadata mismatch: {directory}")
    return {
        "family": family,
        "sample_seed": seed,
        "num_samples": num_samples,
        "condition": mode,
        "fid": float(payload["fid"]),
        "sfid": float(payload["sfid"]),
        "inception_score": float(payload["inception_score"]),
        "noise_fingerprint": payload["noise_fingerprint"],
        "label_fingerprint": payload["label_fingerprint"],
        "source": str(directory),
    }


def _donor_row(
    root: Path,
    *,
    family: str,
    mode: str,
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    directory = root / "causal_screen" / family / f"donor_{mode}_n{num_samples}_seed{seed}"
    payload = _load(directory / "nominal_donor_fid.json")
    if payload["donor_mode"] != mode or int(payload["num_samples"]) != num_samples:
        raise ValueError(f"donor metadata mismatch: {directory}")
    return {
        "family": family,
        "sample_seed": seed,
        "num_samples": num_samples,
        "condition": mode,
        "fid": float(payload["fid"]),
        "sfid": float(payload["sfid"]),
        "inception_score": float(payload["inception_score"]),
        "noise_fingerprint": payload["target_noise_fingerprint"],
        "label_fingerprint": payload["target_label_fingerprint"],
        "donor_noise_fingerprint": payload["donor_noise_fingerprint"],
        "donor_label_fingerprint": payload["donor_label_fingerprint"],
        "source": str(directory),
    }


def _assert_paired(frame: pd.DataFrame, *, context: str) -> None:
    for (_, seed, num_samples), group in frame.groupby(
        ["family", "sample_seed", "num_samples"]
    ):
        if group.noise_fingerprint.nunique() != 1 or group.label_fingerprint.nunique() != 1:
            raise ValueError(f"unpaired target inputs in {context}, seed={seed}")


def build_tables(
    root: Path,
    *,
    prior: Path,
    replay_seeds: tuple[int, ...],
    screen_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_rows: list[dict[str, object]] = []
    for seed in replay_seeds:
        replay_rows.extend(_prior_5k_rows(prior, seed))
        for family in FAMILIES:
            replay_rows.append(
                _intervention_row(
                    root,
                    family=family,
                    mode="replay",
                    num_samples=5000,
                    seed=seed,
                )
            )
    replay = pd.DataFrame(replay_rows)
    _assert_paired(replay, context="5K replay table")

    projection = pd.DataFrame(
        [
            _intervention_row(
                root,
                family=family,
                mode=mode,
                num_samples=1000,
                seed=screen_seed,
            )
            for family in FAMILIES
            for mode in ("frozen", "gain_only", "direction_only", "closed")
        ]
    )
    _assert_paired(projection, context="1K projection table")

    donor = pd.DataFrame(
        [
            _donor_row(
                root,
                family=family,
                mode=mode,
                num_samples=1000,
                seed=screen_seed,
            )
            for family in FAMILIES
            for mode in (
                "paired",
                "same_noise_other_class",
                "other_noise_same_class",
                "other_noise_other_class",
            )
        ]
    )
    _assert_paired(donor, context="1K donor table")
    return replay, projection, donor


def build_formal_projection_table(
    root: Path,
    *,
    prior: Path,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Combine paired prior controls with formal 5K projection interventions."""

    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.extend(_prior_5k_rows(prior, seed))
        for family in FAMILIES:
            for mode in ("gain_only", "direction_only"):
                rows.append(
                    _intervention_row(
                        root,
                        family=family,
                        mode=mode,
                        num_samples=5000,
                        seed=seed,
                        section="causal_formal",
                    )
                )
    table = pd.DataFrame(rows)
    _assert_paired(table, context="formal 5K projection table")
    return table


def _plot_grouped(
    table: pd.DataFrame,
    output: Path,
    *,
    order: tuple[str, ...],
    title: str,
) -> None:
    figure, axes = plt.subplots(1, len(FAMILIES), figsize=(7 * len(FAMILIES), 5), sharey=True)
    for axis, family in zip(axes, FAMILIES, strict=True):
        family_table = table.loc[table.family == family]
        summary = family_table.groupby("condition").fid.agg(["mean", "std"])
        values = [float(summary.loc[condition, "mean"]) for condition in order]
        errors = [
            0.0 if pd.isna(summary.loc[condition, "std"])
            else float(summary.loc[condition, "std"])
            for condition in order
        ]
        bars = axis.bar(order, values, yerr=errors, capsize=3, color="#2f6fa3")
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_title(family)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel(f"ADM FID-N={int(table.num_samples.iloc[0])}")
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prior-summary", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--replay-seeds", default="0,1")
    parser.add_argument(
        "--formal-seeds",
        default="",
        help="Optional comma-separated seeds for completed formal 5K interventions.",
    )
    parser.add_argument("--screen-seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    prior = args.prior_summary.expanduser().resolve()
    replay_seeds = tuple(int(item) for item in args.replay_seeds.split(",") if item)
    formal_seeds = tuple(int(item) for item in args.formal_seeds.split(",") if item)
    replay, projection, donor = build_tables(
        root,
        prior=prior,
        replay_seeds=replay_seeds,
        screen_seed=args.screen_seed,
    )
    formal_projection = None
    if formal_seeds:
        formal_projection = build_formal_projection_table(
            root,
            prior=prior,
            seeds=formal_seeds,
        )
    output = root / "causal_screen" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output / "replay_fid5k.csv", index=False)
    projection.to_csv(output / "projection_fid1k.csv", index=False)
    donor.to_csv(output / "donor_fid1k.csv", index=False)
    if formal_projection is not None:
        formal_projection.to_csv(output / "projection_fid5k.csv", index=False)
    _plot_grouped(
        replay,
        output / "replay_fid5k.png",
        order=("baseline", "frozen", "replay", "closed"),
        title="Does the strong field feedback make frozen guidance work?",
    )
    _plot_grouped(
        projection,
        output / "projection_fid1k.png",
        order=("frozen", "gain_only", "direction_only", "closed"),
        title="Restoring gain or direction reevaluation",
    )
    _plot_grouped(
        donor,
        output / "donor_fid1k.png",
        order=(
            "paired",
            "same_noise_other_class",
            "other_noise_same_class",
            "other_noise_other_class",
        ),
        title="Which nominal-trajectory information does frozen guidance need?",
    )
    if formal_projection is not None:
        _plot_grouped(
            formal_projection,
            output / "projection_fid5k.png",
            order=("baseline", "frozen", "gain_only", "direction_only", "closed"),
            title="Formal gain-versus-direction intervention",
        )
    payload = {
        "protocol": "imagenet100_sit_nominal_transfer_causal_summary_v1",
        "replay_seeds": list(replay_seeds),
        "formal_seeds": list(formal_seeds),
        "screen_seed": args.screen_seed,
        "comparisons_are_paired_only_within_equal_sample_count": True,
        "replay_rows": replay.to_dict(orient="records"),
        "projection_rows": projection.to_dict(orient="records"),
        "donor_rows": donor.to_dict(orient="records"),
        "formal_projection_rows": (
            formal_projection.to_dict(orient="records")
            if formal_projection is not None
            else []
        ),
    }
    atomic_json_dump(payload, output / "summary.json")
    print("\nReplay FID-5K\n", replay.to_string(index=False), flush=True)
    print("\nProjection FID-1K\n", projection.to_string(index=False), flush=True)
    print("\nDonor FID-1K\n", donor.to_string(index=False), flush=True)
    if formal_projection is not None:
        print(
            "\nFormal projection FID-5K\n",
            formal_projection.to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()

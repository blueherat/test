"""Build a leakage-audited transport-risk atlas from completed experiments.

The prospective score deliberately uses only gradients evaluated at the baseline
checkpoint. Endpoint metrics and field-splice results are retained as targets or
post-hoc diagnostics; they never enter the prospective score.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT_ROOT = Path.home() / "data/eqvae/experiments"
PROSPECTIVE_FEATURES = (
    "allocation_collapse",
    "gradient_decoupling",
    "detail_pressure",
)
TARGET_COLUMNS = (
    "rollout_feature_fid_ratio",
    "endpoint_log_damage",
    "endpoint_harm",
)
FIXED_HARM_THRESHOLD = 0.05
PRACTICAL_HARM_RATIO = 1.02


@dataclass(frozen=True)
class AtlasInput:
    dataset: str
    study_dir: Path
    gradient_dir: Path


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate keys {list(keys)}: {examples}")


def _finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values in {list(columns)}")


def build_prospective_atlas(
    study_summary: pd.DataFrame,
    gradient_summary: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Join endpoint targets to baseline-only optimization geometry.

    ``gradient_summary`` may contain audits of trained weighted checkpoints, but
    those rows are removed before any feature is computed. This is the central
    leakage invariant of the atlas.
    """

    study_required = (
        "basis",
        "seed",
        "teacher_ratio_all",
        "rollout_feature_fid_ratio",
        "weight_mean",
        "weight_min",
        "weight_max",
    )
    gradient_required = (
        "basis",
        "seed",
        "checkpoint_variant",
        "parameter_group",
        "coarse_detail_cosine_unweighted",
        "allocation_multiplier",
        "coarse_descent_ratio",
    )
    _require_columns(study_summary, study_required, "study_summary")
    _require_columns(gradient_summary, gradient_required, "gradient_summary")

    targets = study_summary.loc[:, list(study_required)].copy()
    _require_unique(targets, ("basis", "seed"), "study_summary")
    gradients = gradient_summary[
        gradient_summary["checkpoint_variant"].eq("baseline")
        & gradient_summary["parameter_group"].eq("all")
    ].copy()
    if gradients.empty:
        raise ValueError("gradient_summary has no baseline/all rows")
    _require_unique(gradients, ("basis", "seed"), "baseline gradient summary")

    feature_source = gradients[
        [
            "basis",
            "seed",
            "coarse_detail_cosine_unweighted",
            "allocation_multiplier",
            "coarse_descent_ratio",
        ]
    ].copy()
    _finite(
        feature_source,
        (
            "coarse_detail_cosine_unweighted",
            "allocation_multiplier",
            "coarse_descent_ratio",
        ),
        "baseline gradient summary",
    )

    atlas = targets.merge(
        feature_source,
        on=["basis", "seed"],
        how="inner",
        validate="one_to_one",
    )
    if len(atlas) != len(targets) or len(atlas) != len(feature_source):
        target_keys = set(map(tuple, targets[["basis", "seed"]].to_numpy()))
        feature_keys = set(map(tuple, feature_source[["basis", "seed"]].to_numpy()))
        raise ValueError(
            "study and gradient keys differ: "
            f"targets_only={sorted(target_keys - feature_keys)[:5]}, "
            f"gradients_only={sorted(feature_keys - target_keys)[:5]}"
        )

    atlas.insert(0, "dataset", str(dataset))
    atlas["allocation_collapse"] = np.maximum(
        1.0 - atlas["coarse_descent_ratio"].astype(float), 0.0
    )
    atlas["gradient_decoupling"] = np.clip(
        (1.0 - atlas["coarse_detail_cosine_unweighted"].astype(float)) / 2.0,
        0.0,
        1.0,
    )
    atlas["detail_pressure"] = np.maximum(
        np.log(np.maximum(atlas["allocation_multiplier"].astype(float), 1.0)),
        0.0,
    )
    atlas["optimization_risk_score"] = (
        atlas["allocation_collapse"]
        * atlas["gradient_decoupling"]
        * np.log1p(atlas["detail_pressure"])
    )
    atlas["endpoint_log_damage"] = np.log(
        atlas["rollout_feature_fid_ratio"].astype(float).clip(lower=1e-12)
    )
    atlas["endpoint_harm"] = atlas["rollout_feature_fid_ratio"].gt(1.0)
    atlas["practical_endpoint_harm"] = atlas["rollout_feature_fid_ratio"].gt(
        PRACTICAL_HARM_RATIO
    )
    atlas["teacher_mse_improved"] = atlas["teacher_ratio_all"].lt(1.0)
    atlas["isospectral_signature"] = atlas.apply(
        lambda row: (
            round(float(row["weight_mean"]), 6),
            round(float(row["weight_min"]), 6),
            round(float(row["weight_max"]), 6),
        ),
        axis=1,
    )
    return atlas.sort_values(["dataset", "seed", "basis"]).reset_index(drop=True)


def _spearman(first: pd.Series, second: pd.Series) -> float:
    if len(first) < 2 or first.nunique() < 2 or second.nunique() < 2:
        return float("nan")
    return float(first.rank(method="average").corr(second.rank(method="average")))


def _roc_auc(scores: pd.Series, labels: pd.Series) -> float:
    labels = labels.astype(bool).to_numpy()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scores.astype(float).rank(method="average").to_numpy()
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _pairwise_concordance(scores: pd.Series, targets: pd.Series) -> float:
    score_values = scores.astype(float).to_numpy()
    target_values = targets.astype(float).to_numpy()
    correct = 0.0
    comparisons = 0
    for first in range(len(score_values)):
        for second in range(first + 1, len(score_values)):
            target_delta = target_values[first] - target_values[second]
            if math.isclose(target_delta, 0.0, abs_tol=1e-12):
                continue
            score_delta = score_values[first] - score_values[second]
            comparisons += 1
            if math.isclose(score_delta, 0.0, abs_tol=1e-12):
                correct += 0.5
            elif score_delta * target_delta > 0:
                correct += 1.0
    return correct / comparisons if comparisons else float("nan")


def evaluate_prospective_score(atlas: pd.DataFrame) -> dict[str, object]:
    """Evaluate the fixed baseline-only score without fitting on endpoint data."""

    required = (
        "dataset",
        "basis",
        "seed",
        "optimization_risk_score",
        "endpoint_log_damage",
        "endpoint_harm",
    )
    _require_columns(atlas, required, "atlas")
    score = atlas["optimization_risk_score"]
    damage = atlas["endpoint_log_damage"]
    labels = atlas["endpoint_harm"]
    practical_labels = atlas["practical_endpoint_harm"]
    fixed_prediction = score.gt(FIXED_HARM_THRESHOLD)
    basis_means = (
        atlas.groupby(["dataset", "basis"], as_index=False)[
            ["optimization_risk_score", "endpoint_log_damage"]
        ]
        .mean()
    )
    centered = atlas.copy()
    for column in ("optimization_risk_score", "endpoint_log_damage"):
        centered[f"centered_{column}"] = centered[column] - centered.groupby(
            ["dataset", "basis"]
        )[column].transform("mean")

    control_comparisons = 0
    control_correct = 0.0
    for _, rows in atlas.groupby(["dataset", "seed"], sort=True):
        indexed = rows.set_index("basis")
        if "random" not in indexed.index:
            continue
        for structured in ("dct", "pca"):
            if structured not in indexed.index:
                continue
            score_delta = float(
                indexed.loc[structured, "optimization_risk_score"]
                - indexed.loc["random", "optimization_risk_score"]
            )
            damage_delta = float(
                indexed.loc[structured, "endpoint_log_damage"]
                - indexed.loc["random", "endpoint_log_damage"]
            )
            control_comparisons += 1
            if math.isclose(score_delta, 0.0, abs_tol=1e-12):
                control_correct += 0.5
            elif score_delta * damage_delta > 0:
                control_correct += 1.0
    result: dict[str, object] = {
        "row_count": int(len(atlas)),
        "dataset_count": int(atlas["dataset"].nunique()),
        "seed_count": int(atlas[["dataset", "seed"]].drop_duplicates().shape[0]),
        "spearman": _spearman(score, damage),
        "roc_auc": _roc_auc(score, labels),
        "practical_harm_ratio": PRACTICAL_HARM_RATIO,
        "practical_roc_auc": _roc_auc(score, practical_labels),
        "pairwise_concordance": _pairwise_concordance(score, damage),
        "between_basis_spearman": _spearman(
            basis_means["optimization_risk_score"],
            basis_means["endpoint_log_damage"],
        ),
        "within_basis_seed_spearman": _spearman(
            centered["centered_optimization_risk_score"],
            centered["centered_endpoint_log_damage"],
        ),
        "structured_vs_random_pair_accuracy": (
            control_correct / control_comparisons
            if control_comparisons
            else float("nan")
        ),
        "structured_vs_random_comparisons": control_comparisons,
        "fixed_harm_threshold": FIXED_HARM_THRESHOLD,
        "fixed_sign_accuracy": float(fixed_prediction.eq(labels).mean()),
        "fixed_practical_sign_accuracy": float(
            fixed_prediction.eq(practical_labels).mean()
        ),
        "feature_columns": list(PROSPECTIVE_FEATURES),
        "target_columns": list(TARGET_COLUMNS),
        "leakage_audit_passed": not bool(
            set(PROSPECTIVE_FEATURES) & set(TARGET_COLUMNS)
        ),
    }
    by_dataset: dict[str, dict[str, float | int]] = {}
    for dataset, rows in atlas.groupby("dataset", sort=True):
        by_dataset[str(dataset)] = {
            "rows": int(len(rows)),
            "spearman": _spearman(
                rows["optimization_risk_score"], rows["endpoint_log_damage"]
            ),
            "roc_auc": _roc_auc(
                rows["optimization_risk_score"], rows["endpoint_harm"]
            ),
            "fixed_sign_accuracy": float(
                rows["optimization_risk_score"]
                .gt(FIXED_HARM_THRESHOLD)
                .eq(rows["endpoint_harm"])
                .mean()
            ),
        }
    result["by_dataset"] = by_dataset
    result["gates"] = {
        "spearman_ge_0_70": bool(result["spearman"] >= 0.70),
        "roc_auc_ge_0_80": bool(result["roc_auc"] >= 0.80),
        "fixed_sign_accuracy_ge_0_80": bool(
            result["fixed_sign_accuracy"] >= 0.80
        ),
    }
    result["all_gates_passed"] = all(result["gates"].values())
    return result


def summarize_by_basis(atlas: pd.DataFrame) -> pd.DataFrame:
    return (
        atlas.groupby(["dataset", "basis"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            optimization_risk_score=("optimization_risk_score", "mean"),
            allocation_collapse=("allocation_collapse", "mean"),
            gradient_decoupling=("gradient_decoupling", "mean"),
            teacher_ratio=("teacher_ratio_all", "mean"),
            endpoint_fid_ratio=("rollout_feature_fid_ratio", "mean"),
            endpoint_harm_rate=("endpoint_harm", "mean"),
        )
        .sort_values(["dataset", "optimization_risk_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _read_dataset_name(study_dir: Path) -> str:
    config_path = study_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing study config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config.get("dataset")
    if not dataset:
        raise ValueError(f"study config has no dataset: {config_path}")
    return str(dataset)


def discover_atlas_inputs(
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> list[AtlasInput]:
    """Discover the newest deterministic study and its matching gradient audit."""

    experiment_root = experiment_root.expanduser().resolve()
    study_root = experiment_root / "small_image_basis_transport"
    gradient_root = experiment_root / "small_image_gradient_allocation"
    studies_by_dataset: dict[str, Path] = {}
    for summary_path in study_root.glob("*/study_summary.csv"):
        study_dir = summary_path.parent
        if "deterministic" not in study_dir.name:
            continue
        dataset = _read_dataset_name(study_dir)
        current = studies_by_dataset.get(dataset)
        if current is None or summary_path.stat().st_mtime > (
            current / "study_summary.csv"
        ).stat().st_mtime:
            studies_by_dataset[dataset] = study_dir

    gradients_by_study: dict[str, Path] = {}
    for config_path in gradient_root.glob("*/config.json"):
        gradient_dir = config_path.parent
        if not (gradient_dir / "gradient_summary.csv").exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_name = Path(str(config.get("study_dir", ""))).name
        if not source_name:
            continue
        current = gradients_by_study.get(source_name)
        if current is None or config_path.stat().st_mtime > (
            current / "config.json"
        ).stat().st_mtime:
            gradients_by_study[source_name] = gradient_dir

    inputs = []
    for dataset, study_dir in sorted(studies_by_dataset.items()):
        gradient_dir = gradients_by_study.get(study_dir.name)
        if gradient_dir is not None:
            inputs.append(AtlasInput(dataset, study_dir, gradient_dir))
    if not inputs:
        raise FileNotFoundError(
            f"no matched deterministic studies and gradient audits under {experiment_root}"
        )
    return inputs


def load_atlas(inputs: Sequence[AtlasInput]) -> pd.DataFrame:
    frames = []
    for item in inputs:
        frames.append(
            build_prospective_atlas(
                pd.read_csv(item.study_dir / "study_summary.csv"),
                pd.read_csv(item.gradient_dir / "gradient_summary.csv"),
                dataset=item.dataset,
            )
        )
    atlas = pd.concat(frames, ignore_index=True)
    _require_unique(atlas, ("dataset", "basis", "seed"), "combined atlas")
    return atlas.sort_values(["dataset", "seed", "basis"]).reset_index(drop=True)


def save_atlas(
    atlas: pd.DataFrame,
    evaluation: dict[str, object],
    inputs: Sequence[AtlasInput],
    output_root: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_root.expanduser() / f"retrospective_v1_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    atlas_to_save = atlas.copy()
    atlas_to_save["isospectral_signature"] = atlas_to_save[
        "isospectral_signature"
    ].map(str)
    atlas_to_save.to_csv(result_dir / "atlas_rows.csv", index=False)
    summarize_by_basis(atlas).to_csv(result_dir / "basis_summary.csv", index=False)
    (result_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "scope": "retrospective calibration; not an independent prospective claim",
        "prospective_score_uses": list(PROSPECTIVE_FEATURES),
        "prospective_score_excludes": list(TARGET_COLUMNS),
        "inputs": [
            {
                **asdict(item),
                "study_dir": str(item.study_dir),
                "gradient_dir": str(item.gradient_dir),
            }
            for item in inputs
        ],
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result_dir


def _parse_inputs(values: Sequence[Sequence[str]]) -> list[AtlasInput]:
    return [
        AtlasInput(dataset, Path(study_dir).expanduser(), Path(gradient_dir).expanduser())
        for dataset, study_dir, gradient_dir in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("DATASET", "STUDY_DIR", "GRADIENT_DIR"),
        default=[],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "transport_risk_atlas",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--strict-gates", action="store_true")
    args = parser.parse_args()

    inputs = _parse_inputs(args.input) if args.input else discover_atlas_inputs(
        args.experiment_root
    )
    atlas = load_atlas(inputs)
    evaluation = evaluate_prospective_score(atlas)
    print(summarize_by_basis(atlas).round(4).to_string(index=False))
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    if args.save:
        result_dir = save_atlas(atlas, evaluation, inputs, args.output_root)
        print(f"result_dir={result_dir}")
    if args.strict_gates and not evaluation["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

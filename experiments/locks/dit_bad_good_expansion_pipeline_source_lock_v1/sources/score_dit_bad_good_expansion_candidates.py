#!/usr/bin/env python3
"""Apply candidate-v5 A/B/S_UNION unchanged to expansion seeds 130..249.

The original scorer intentionally accepts only the frozen 30..129 cohort.
This expansion wrapper keeps the score formula, discovery normalizers, source
hash requirements, sampler lineage, and negative controls unchanged while
requiring the exact disjoint 130..249 x {207,602,795} Cartesian product.
It first proves that the final expansion label lock exists by validating only
its aggregate manifest/completion and byte hashes; row labels are not decoded.
It never reads calibration thresholds, alerts, reviewer drafts, or row labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .dit_bad_good_expansion_contract import (
        CLASSES,
        EXPANSION_SEEDS,
        LABELS,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
    from .score_dit_bad_good_frozen_candidates import (
        frozen_old_score,
        validate_feature_product,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from dit_bad_good_expansion_contract import (
        CLASSES,
        EXPANSION_SEEDS,
        LABELS,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
    from score_dit_bad_good_frozen_candidates import (
        frozen_old_score,
        validate_feature_product,
    )


KEYS = ["sample_index", "run_index", "global_seed", "class_slot", "class_id"]


def validate_final_label_lock_without_reading_rows(root: Path) -> dict[str, Any]:
    """Prove labels are final using lock metadata/hashes, not row decoding."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"final expansion label lock is missing: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    manifest_identity = require_canonical_identity(manifest, "expansion label manifest")
    counts = manifest.get("counts")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("consensus_identity_sha256")
        != manifest.get("consensus_identity_sha256")
        or completion.get("locked_row_count") != 360
        or manifest.get("candidate_protocol_identity_sha256")
        != validate_candidate_lock()["identity_sha256"]
        or not isinstance(manifest.get("raw_consensus_identity_sha256"), str)
        or not isinstance(manifest.get("blind_pack_identity_sha256"), str)
        or not isinstance(counts, dict)
        or set(counts) != set(LABELS)
        or sum(int(counts[label]) for label in LABELS) != 360
    ):
        raise RuntimeError("final expansion label lock metadata is invalid")
    members = {str(item.get("name")): item for item in manifest.get("files", [])}
    if "adjudication_locked.json" not in members:
        raise RuntimeError("label lock is raw consensus, not final adjudication")
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"final label-lock member changed: {path}")
    return {
        "manifest_identity_sha256": manifest_identity,
        "consensus_identity_sha256": manifest["consensus_identity_sha256"],
        "blind_pack_identity_sha256": manifest["blind_pack_identity_sha256"],
        "row_level_labels_decoded": False,
    }


def _validate_expansion_lineage(
    protocol: dict[str, Any],
    primary_manifest: dict[str, Any],
    posterior_manifest: dict[str, Any],
    primary_inventory: dict[str, Any],
    posterior_inventory: dict[str, Any],
) -> None:
    primary_hash = protocol["source_products"]["primary_label_free"][
        "analysis_source_sha256"
    ]
    if (
        primary_manifest.get("analysis_source_sha256") != primary_hash
        or posterior_manifest.get("imported_validation_helper_sha256") != primary_hash
    ):
        raise RuntimeError("posterior reductions do not import the frozen primary helper")
    primary_identities = primary_manifest.get("trace_identity_sha256_ordered")
    posterior_identities = posterior_manifest.get("trace_identity_sha256_ordered")
    if (
        not isinstance(primary_identities, list)
        or primary_identities != posterior_identities
        or len(primary_identities) != len(EXPANSION_SEEDS)
    ):
        raise RuntimeError("feature products do not bind the same 120-run expansion")
    primary_runs = primary_inventory.get("trace_runs")
    posterior_runs = posterior_inventory.get("trace_runs")
    if not isinstance(primary_runs, list) or not isinstance(posterior_runs, list):
        raise RuntimeError("feature product lacks trace-run inventory")
    core_fields = (
        "cfg_epsilon_channels",
        "cfg_scale",
        "classes",
        "completion_sha256",
        "global_seed",
        "identity_sha256",
        "manifest_sha256",
        "source_snapshot_sha256",
        "trace_sha256",
    )
    primary_core = [{key: run.get(key) for key in core_fields} for run in primary_runs]
    posterior_core = [{key: run.get(key) for key in core_fields} for run in posterior_runs]
    if primary_core != posterior_core:
        raise RuntimeError("primary and posterior products came from different traces")
    if [run.get("identity_sha256") for run in primary_runs] != primary_identities:
        raise RuntimeError("manifest trace identities differ from source inventory")
    if tuple(run.get("global_seed") for run in primary_runs) != EXPANSION_SEEDS:
        raise RuntimeError("trace inventory is not ordered seeds 130..249")
    expected_snapshots = protocol["sampler_lineage_contract"][
        "source_snapshot_sha256"
    ]
    for run in primary_runs:
        if (
            run.get("cfg_scale") != 4.0
            or run.get("cfg_epsilon_channels") != 3
            or tuple(run.get("classes", ())) != CLASSES
            or run.get("source_snapshot_sha256") != expected_snapshots
        ):
            raise RuntimeError("expansion sampler/source lineage differs from discovery")


def _expected_rows() -> set[tuple[int, int, int]]:
    return {
        (seed, slot, class_id)
        for seed in EXPANSION_SEEDS
        for slot, class_id in enumerate(CLASSES)
    }


def score(
    candidate_lock: Path,
    expansion_label_lock: Path,
    primary_root: Path,
    posterior_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = validate_candidate_lock(candidate_lock)
    expansion_protocol = validate_expansion_lock()
    validate_pipeline_source_lock(Path(__file__).name)
    label_lineage = validate_final_label_lock_without_reading_rows(expansion_label_lock)
    source = protocol["source_products"]
    primary_manifest, primary, primary_inventory = validate_feature_product(
        primary_root.resolve(), source["primary_label_free"]["analysis_source_sha256"]
    )
    posterior_manifest, posterior, posterior_inventory = validate_feature_product(
        posterior_root.resolve(), source["posterior_label_free"]["analysis_source_sha256"]
    )
    _validate_expansion_lineage(
        protocol,
        primary_manifest,
        posterior_manifest,
        primary_inventory,
        posterior_inventory,
    )
    if len(primary) != 360 or len(posterior) != 360:
        raise RuntimeError("expansion feature products must each contain 360 rows")
    if not np.array_equal(primary[KEYS].to_numpy(), posterior[KEYS].to_numpy()):
        raise RuntimeError("primary/posterior expansion sample axes differ")
    observed_rows = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in primary[["global_seed", "class_slot", "class_id"]].itertuples(
            index=False
        )
    }
    if (
        observed_rows != _expected_rows()
        or primary[["global_seed", "class_id"]].duplicated().any()
        or posterior[["global_seed", "class_id"]].duplicated().any()
        or tuple(primary["sample_index"].astype(int)) != tuple(range(360))
        or tuple(sorted(primary["global_seed"].astype(int).unique()))
        != EXPANSION_SEEDS
        or tuple(sorted(primary["class_id"].astype(int).unique())) != CLASSES
    ):
        raise RuntimeError("feature cohort is not exact expansion Cartesian product")

    feature_a = protocol["single_feature_backups"]["A"]["feature"]
    feature_b = protocol["single_feature_backups"]["B"]["feature"]
    evidence_controls = protocol["negative_controls"][
        "exact_path_evidence_running_maxima"
    ]
    if feature_b not in primary:
        raise RuntimeError("primary feature product lacks frozen feature B")
    if feature_a not in posterior or any(name not in posterior for name in evidence_controls):
        raise RuntimeError("posterior feature product lacks frozen A/control columns")

    result = primary[KEYS + ["trace_dir", "endpoint_png_path"]].copy()
    result["cohort_role"] = "inferential_expansion"
    result["A_posterior_logstd_concentration_jump"] = posterior[feature_a].to_numpy(float)
    result["B_withheld_channel_predx0_cusum"] = primary[feature_b].to_numpy(float)
    result["old_fixed_predicted_clean_score_control"] = frozen_old_score(
        protocol, primary_root.resolve(), primary
    )
    for name in ("z_A_low_is_bad", "z_B_high_is_bad", "S_INTERSECTION", "S_UNION"):
        result[name] = np.nan
    references = protocol["normalization"]["class_reference"]
    for class_id in CLASSES:
        mask = result["class_id"].eq(class_id)
        statistics = references[str(class_id)]["statistics"]
        a = statistics["A_low_is_bad"]
        b = statistics["B_high_is_bad"]
        result.loc[mask, "z_A_low_is_bad"] = (
            -result.loc[mask, "A_posterior_logstd_concentration_jump"]
            - float(a["median"])
        ) / float(a["scale"])
        result.loc[mask, "z_B_high_is_bad"] = (
            result.loc[mask, "B_withheld_channel_predx0_cusum"]
            - float(b["median"])
        ) / float(b["scale"])
        result.loc[mask, "S_INTERSECTION"] = np.minimum(
            result.loc[mask, "z_A_low_is_bad"], result.loc[mask, "z_B_high_is_bad"]
        )
        result.loc[mask, "S_UNION"] = np.maximum(
            result.loc[mask, "z_A_low_is_bad"], result.loc[mask, "z_B_high_is_bad"]
        )
    for name in evidence_controls:
        short = name.replace("__full_maximum", "")
        values = posterior[name].to_numpy(float)
        result[f"control_{short}"] = values
        result[f"control_{short}_trigger_alpha0p10"] = values >= np.log(10.0)
        result[f"control_{short}_trigger_alpha0p05"] = values >= np.log(20.0)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("expansion score output contains non-finite numbers")
    if result.columns.astype(str).str.lower().str.contains(
        "label|review|consensus|severity|adjudic"
    ).any():
        raise RuntimeError("expansion score output leaked visual supervision")

    summary = {
        "schema_version": 1,
        "status": "COMPLETE_LABEL_FREE_FROZEN_CANDIDATE_EXPANSION_SCORES",
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "expansion_protocol_identity_sha256": expansion_protocol["identity_sha256"],
        "expansion_final_label_lock": label_lineage,
        "sample_count": 360,
        "classes": list(CLASSES),
        "seeds": list(EXPANSION_SEEDS),
        "cohort_role": "inferential_expansion",
        "labels_read_or_emitted": False,
        "thresholds_read_or_reestimated": False,
        "alerts_read_or_emitted": False,
        "formula_changed": False,
        "normalizers_reestimated": False,
        "primary_scores": ["z_A_low_is_bad", "z_B_high_is_bad", "S_UNION"],
        "descriptive_retired_score": "S_INTERSECTION",
        "primary_manifest_identity_sha256": primary_manifest["identity_sha256"],
        "posterior_manifest_identity_sha256": posterior_manifest["identity_sha256"],
    }
    return result, summary


def publish(
    candidate_lock: Path,
    expansion_label_lock: Path,
    primary_root: Path,
    posterior_root: Path,
    output: Path,
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite expansion score output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        frame, summary = score(
            candidate_lock, expansion_label_lock, primary_root, posterior_root
        )
        frame.to_csv(staging / "frozen_candidate_scores_label_free.csv", index=False)
        write_json(staging / "summary.json", summary)
        shutil.copy2(Path(__file__).resolve(), staging / "scorer_source.py")
        helper = Path(__file__).resolve().with_name("dit_bad_good_expansion_contract.py")
        shutil.copy2(helper, staging / "expansion_contract_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": summary[
                "candidate_protocol_identity_sha256"
            ],
            "cohort": "expansion_seed130_249",
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "scores_file_sha256": sha256_file(
                    staging / "frozen_candidate_scores_label_free.csv"
                ),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    protocol = validate_candidate_lock()
    validate_expansion_lock()
    assert tuple(protocol["fresh_confirmation"]["classes"]) == CLASSES
    assert len(_expected_rows()) == 360
    values_a = np.asarray([1.0, -2.0, 0.0])
    values_b = np.asarray([0.0, -3.0, 4.0])
    assert np.array_equal(np.maximum(values_a, values_b), np.asarray([1.0, -2.0, 4.0]))
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=None)
    parser.add_argument("--expansion-label-lock", type=Path)
    parser.add_argument("--primary-root", type=Path)
    parser.add_argument("--posterior-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if (
        args.expansion_label_lock is None
        or args.primary_root is None
        or args.posterior_root is None
        or args.output is None
    ):
        parser.error(
            "--expansion-label-lock, --primary-root, --posterior-root and --output are required"
        )
    candidate = args.candidate_lock or (
        Path(__file__).resolve().parents[1]
        / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
    )
    output = publish(
        candidate,
        args.expansion_label_lock,
        args.primary_root,
        args.posterior_root,
        args.output,
    )
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exploratory inventory replication screen including label-free visual tracks.

This is a deliberately separate v2 analysis.  It imports the frozen v1
screening/statistical implementation by an exact source hash, adds the two
preterminal ``pred_xstart`` visual products, and keeps the same split:

* original seeds 50..129: choose one scalar reduction and direction per track;
* expansion seeds 130..249: evaluate those in-memory-frozen choices;
* mild/disputed endpoints: excluded;
* output: aggregate track tables only, never sample labels/scores/ranks/paths.

The visual extractor was label-free.  This script validates its complete
envelope, formulas/catalog timing, exact sample axis, ordered trace lineage,
and absence of image/label payloads before any label lock is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPECTED_BASE_SOURCE_SHA256 = (
    "c23ef50713e3a310bf8a6bb7b0541563717b5979cca035f3b15394f68aaaca90"
)
EXPECTED_VISUAL_EXTRACTOR_SHA256 = (
    "452ae0e61fe36d027036e0d74c232fbcfbd7cb462d3749db92e062a104d0e398"
)


def _sha256_file_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_base() -> tuple[ModuleType, Path]:
    source = Path(__file__).resolve().with_name(
        "explore_dit_bad_good_inventory_replication.py"
    )
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"frozen v1 screening source is unavailable: {source}")
    observed = _sha256_file_local(source)
    if observed != EXPECTED_BASE_SOURCE_SHA256:
        raise RuntimeError(
            "frozen v1 screening source changed: "
            f"{observed} != {EXPECTED_BASE_SOURCE_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_dit_inventory_replication_frozen_v1", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v1 source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


BASE, BASE_SOURCE_PATH = _load_frozen_base()

KEYS = BASE.KEYS
GOOD = BASE.GOOD
BAD = BASE.BAD
MILD = BASE.MILD
CLASSES = BASE.CLASSES
DISCOVERY_SEEDS = BASE.DISCOVERY_SEEDS
REPLICATION_SEEDS = BASE.REPLICATION_SEEDS
DEFAULT_PERMUTATION_DRAWS = BASE.DEFAULT_PERMUTATION_DRAWS
DEFAULT_PERMUTATION_SEED = BASE.DEFAULT_PERMUTATION_SEED

EXPECTED_PRIMARY_FEATURES = BASE.EXPECTED_PRIMARY_FEATURES
EXPECTED_POSTERIOR_FEATURES = BASE.EXPECTED_POSTERIOR_FEATURES
EXPECTED_NORMALIZED_POSTERIOR_FEATURES = BASE.EXPECTED_NORMALIZED_POSTERIOR_FEATURES
EXPECTED_VISUAL_FEATURES = 40
EXPECTED_BASE_FEATURES = BASE.EXPECTED_TOTAL_FEATURES
EXPECTED_TOTAL_FEATURES = 6_884
EXPECTED_BASE_TRACKS = 192
EXPECTED_VISUAL_TRACKS = 8
EXPECTED_TOTAL_TRACKS = 200
EXPECTED_ORIGINAL_VISUAL_FULL_SEEDS = tuple(range(30, 130))
EXPECTED_REPLICATION_VISUAL_FULL_SEEDS = REPLICATION_SEEDS
EXPECTED_VISUAL_STEPS = tuple(range(69, 150, 10))
EXPECTED_VISUAL_INTERNAL_TIMESTEPS = tuple(range(180, 99, -10))
EXPECTED_VISUAL_EXPERIMENT = "dit_predxstart_preterminal_visual_tracks_label_free"
VISUAL_IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
)
EXPECTED_VISUAL_TRACK_NAMES = {
    "decoded_local_blur_severity",
    "decoded_edge_tangle",
    "resnet18_target_log_odds",
    "decoder_clipping_fraction",
    "decoded_coherent_edge_jump",
    "resnet18_embedding_cosine_jump",
    "resnet18_target_log_odds_drop",
    "resnet18_target_cam_jump",
}
EXPECTED_SUPERVISION_AUDIT = {
    "labels_read_or_emitted": False,
    "reviews_read": False,
    "candidate_scores_read": False,
    "calibration_thresholds_read": False,
    "alerts_read": False,
    "auc_or_selection_computed": False,
}
FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

DEFAULT_BASE = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
DEFAULT_ORIGINAL_LABELS = BASE.DEFAULT_ORIGINAL_LABELS
DEFAULT_EXPANSION_LABELS = BASE.DEFAULT_EXPANSION_LABELS
DEFAULT_ORIGINAL_PRIMARY = BASE.DEFAULT_ORIGINAL_PRIMARY
DEFAULT_ORIGINAL_POSTERIOR = BASE.DEFAULT_ORIGINAL_POSTERIOR
DEFAULT_EXPANSION_PRIMARY = BASE.DEFAULT_EXPANSION_PRIMARY
DEFAULT_EXPANSION_POSTERIOR = BASE.DEFAULT_EXPANSION_POSTERIOR
DEFAULT_ORIGINAL_NORMALIZED_POSTERIOR = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_v1/normalized_posterior_label_free_v2"
)
DEFAULT_EXPANSION_NORMALIZED_POSTERIOR = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/normalized_posterior_label_free_v2"
)
DEFAULT_ORIGINAL_VISUAL = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/predxstart_visual_label_free_v1"
)
DEFAULT_EXPANSION_VISUAL = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/predxstart_visual_label_free_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/"
    "inventory_replication_exploratory_visual_v2"
)
FROZEN_PERMUTATION_DRAWS = 100_000
FROZEN_PERMUTATION_SEED = 20260827
FROZEN_OUTPUT = DEFAULT_OUTPUT
FROZEN_LABEL_CONSENSUS_IDENTITIES = {
    "original": "21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a",
    "expansion": "fc478b7ae04b67869d0dfca3b63f169a5266bacc1c8701e1ae20368516e793fd",
}
FROZEN_FEATURE_MANIFEST_IDENTITIES = {
    "original_primary": "a0aa2d1c8ffb8a2be2b95cc6cba7710a282f22ff3d5f24d246e8ea23b6e01044",
    "original_posterior": "e7abf3a8968f986779c101a05be8066754f4fa154cafe31ee74223789c4568ab",
    "original_normalized_posterior": "0eb3c0a94ecbf3bf6e44bff8a6df09210e856e6b6d6bf591b93bd05583e7c221",
    "original_predxstart_visual": "7554c9bf8a15375e83df50a4308a00176a5e2dc5deb8f378a5611cd8f3c322cd",
    "expansion_primary": "e9e441c379796a2066ef672987c36dd52bc77b59578e3133b8372743c1593a31",
    "expansion_posterior": "df902c7ee526da2e2f1a12d483338d595af4393133729028f5e2835a5658c9cf",
    "expansion_normalized_posterior": "db9022e9779dca0884108215bd6da845a5327b88b0b25ed48eb921e6c6b5d572",
    "expansion_predxstart_visual": "bddd0e9d17fcce2f1c87a02373991a83b6091b6b46e05f8fe7f7566718913696",
}
FROZEN_EXECUTION_CONTRACT = {
    "label_consensus_identity_sha256": FROZEN_LABEL_CONSENSUS_IDENTITIES,
    "feature_manifest_identity_sha256": FROZEN_FEATURE_MANIFEST_IDENTITIES,
    "permutation_draws": FROZEN_PERMUTATION_DRAWS,
    "permutation_seed": FROZEN_PERMUTATION_SEED,
    "expected_scalar_feature_count": EXPECTED_TOTAL_FEATURES,
    "expected_unique_track_count": EXPECTED_TOTAL_TRACKS,
    "output": str(FROZEN_OUTPUT),
    "cli_override_policy": "fail_closed_on_any_difference",
}


def _canonical_self_hash(payload: Mapping[str, Any], key: str) -> str:
    copied = dict(payload)
    copied.pop(key, None)
    return BASE.canonical_sha256(copied)


def _array_record(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "raw_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _require_exact_file_set(root: Path, manifest_members: dict[str, Any]) -> None:
    entries = list(root.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RuntimeError(f"visual product contains a directory/symlink: {root}")
    actual = {path.name for path in entries}
    expected = set(manifest_members) | {"manifest.json", "completion.json"}
    if actual != expected:
        raise RuntimeError(f"visual product member set changed: {actual} != {expected}")
    images = sorted(path.name for path in entries if path.suffix.lower() in FORBIDDEN_IMAGE_SUFFIXES)
    if images:
        raise RuntimeError(f"visual product unexpectedly emits images: {images}")


def _require_clean_supervision_audit(value: Any, description: str) -> None:
    if not isinstance(value, dict) or value != EXPECTED_SUPERVISION_AUDIT:
        raise RuntimeError(f"{description} supervision audit is not strictly label-free")


def validate_visual_product(
    root: Path,
    cohort_seeds: tuple[int, ...],
    full_seeds: tuple[int, ...],
    product_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate a visual product without opening any endpoint or label file."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"visual product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = BASE.read_json(manifest_path)
    completion = BASE.read_json(completion_path)
    summary = BASE.read_json(root / "summary.json")
    protocol = BASE.read_json(root / "protocol_snapshot.json")
    provenance = BASE.read_json(root / "provenance.json")
    inventory = BASE.read_json(root / "source_inventory.json")
    manifest_id = BASE.require_identity(manifest, f"{product_name} manifest")
    members = BASE.validate_manifest_members(root, manifest)
    _require_exact_file_set(root, members)
    required_members = {
        "analysis_source.py",
        "feature_catalog.csv",
        "protocol_snapshot.json",
        "provenance.json",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
        "time_series.npz",
    }
    if set(members) != required_members:
        raise RuntimeError(f"{product_name} payload schema changed")
    if (
        manifest.get("status") != "complete"
        or manifest.get("experiment") != EXPECTED_VISUAL_EXPERIMENT
        or manifest.get("analysis_source_sha256") != EXPECTED_VISUAL_EXTRACTOR_SHA256
        or manifest.get("analysis_source_sha256")
        != members["analysis_source.py"].get("sha256")
        or manifest.get("protocol_snapshot_sha256")
        != members["protocol_snapshot.json"].get("sha256")
        or manifest.get("source_inventory_sha256")
        != members["source_inventory.json"].get("sha256")
        or manifest.get("provenance_sha256")
        != members["provenance.json"].get("sha256")
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest_id
        or completion.get("manifest_file_sha256") != BASE.sha256_file(manifest_path)
        or completion.get("summary_file_sha256")
        != BASE.sha256_file(root / "summary.json")
        or completion.get("payload_sha256")
        != _canonical_self_hash(completion, "payload_sha256")
    ):
        raise RuntimeError(f"{product_name} manifest/completion envelope is invalid")

    _require_clean_supervision_audit(
        protocol.get("supervision_policy"), f"{product_name} protocol"
    )
    _require_clean_supervision_audit(
        provenance.get("supervision_audit"), f"{product_name} provenance"
    )
    if (
        protocol.get("status") != "LABEL_FREE_OBSERVATION_ONLY"
        or protocol.get("experiment") != EXPECTED_VISUAL_EXPERIMENT
        or protocol.get("decode", {}).get("saved_images") is not False
        or provenance.get("decoded_images_saved") is not False
        or provenance.get("offline_only") is not True
        or summary.get("decoded_images_saved") is not False
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("status") != "COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION"
        or summary.get("experiment") != EXPECTED_VISUAL_EXPERIMENT
    ):
        raise RuntimeError(f"{product_name} is not a sealed label/image-free product")

    catalog = pd.read_csv(root / "feature_catalog.csv")
    frame = pd.read_csv(root / "sample_features.csv")
    required_catalog = {
        "feature",
        "track",
        "family",
        "reduction",
        "latest_required_sampling_step",
        "latest_required_internal_timestep",
        "observation_timing",
        "preterminal_actionable",
        "track_length",
        "uses_realized_innovation",
        "checkpoint_sampling_steps",
        "checkpoint_internal_timesteps",
    }
    if not required_catalog.issubset(catalog.columns):
        raise RuntimeError(f"{product_name} visual catalog schema is incomplete")
    catalog = catalog.copy()
    catalog["feature"] = catalog["feature"].astype(str)
    catalog["track"] = catalog["track"].astype(str)
    catalog["preterminal_actionable"] = BASE._as_bool(
        catalog["preterminal_actionable"], f"{product_name} actionable flag"
    )
    catalog["uses_realized_innovation"] = BASE._as_bool(
        catalog["uses_realized_innovation"], f"{product_name} innovation flag"
    )
    catalog["latest_required_sampling_step"] = pd.to_numeric(
        catalog["latest_required_sampling_step"], errors="raise"
    ).astype(int)
    catalog["latest_required_internal_timestep"] = pd.to_numeric(
        catalog["latest_required_internal_timestep"], errors="raise"
    ).astype(int)
    features = catalog["feature"].tolist()
    expected_step_string = ",".join(map(str, EXPECTED_VISUAL_STEPS))
    expected_internal_string = ",".join(map(str, EXPECTED_VISUAL_INTERNAL_TIMESTEPS))
    if (
        len(catalog) != EXPECTED_VISUAL_FEATURES
        or catalog["feature"].duplicated().any()
        or catalog["track"].nunique() != EXPECTED_VISUAL_TRACKS
        or set(catalog["track"]) != EXPECTED_VISUAL_TRACK_NAMES
        or not catalog.groupby("track").size().eq(5).all()
        or not catalog["latest_required_sampling_step"].eq(149).all()
        or not catalog["latest_required_internal_timestep"].eq(100).all()
        or not catalog["preterminal_actionable"].all()
        or catalog["uses_realized_innovation"].any()
        or not catalog["observation_timing"].eq(
            "before_transition_at_latest_checkpoint"
        ).all()
        or not catalog["checkpoint_sampling_steps"].eq(expected_step_string).all()
        or not catalog["checkpoint_internal_timesteps"].eq(
            expected_internal_string
        ).all()
    ):
        raise RuntimeError(f"{product_name} visual feature/timing contract changed")
    if (
        tuple(frame.columns[: len(VISUAL_IDENTIFIER_COLUMNS)])
        != VISUAL_IDENTIFIER_COLUMNS
        or tuple(frame.columns[len(VISUAL_IDENTIFIER_COLUMNS) :]) != tuple(features)
        or any(
            name in frame.columns
            for name in ("label", "primary_label", "raw_consensus_label")
        )
        or frame.duplicated(list(KEYS)).any()
    ):
        raise RuntimeError(f"{product_name} sample table schema/axis is invalid")
    full_axis = {(seed, class_id) for seed in full_seeds for class_id in CLASSES}
    observed_axis = set(map(tuple, frame[list(KEYS)].to_numpy()))
    if observed_axis != full_axis or len(frame) != len(full_axis):
        raise RuntimeError(f"{product_name} full sample axis differs")
    values = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{product_name} contains non-finite visual features")
    cohort = frame[
        frame["global_seed"].isin(cohort_seeds) & frame["class_id"].isin(CLASSES)
    ].copy()
    cohort_axis = {(seed, class_id) for seed in cohort_seeds for class_id in CLASSES}
    if set(map(tuple, cohort[list(KEYS)].to_numpy())) != cohort_axis:
        raise RuntimeError(f"{product_name} lacks the exact analysis cohort")

    ordered_seeds = inventory.get("ordered_seeds")
    ordered_classes = inventory.get("ordered_classes")
    runs = inventory.get("trace_runs")
    if (
        ordered_seeds != list(full_seeds)
        or ordered_classes != list(CLASSES)
        or not isinstance(runs, list)
        or len(runs) != len(full_seeds)
        or summary.get("ordered_seeds") != list(full_seeds)
        or summary.get("ordered_classes") != list(CLASSES)
        or summary.get("trace_count") != len(full_seeds)
        or summary.get("sample_count") != len(full_axis)
        or summary.get("scalar_feature_count") != EXPECTED_VISUAL_FEATURES
        or summary.get("level_track_count") != 4
        or summary.get("jump_track_count") != 4
        or summary.get("time_series_array_count") != 16
        or summary.get("selected_sampling_steps") != list(EXPECTED_VISUAL_STEPS)
        or summary.get("selected_internal_timesteps")
        != list(EXPECTED_VISUAL_INTERNAL_TIMESTEPS)
    ):
        raise RuntimeError(f"{product_name} summary/source inventory differs")
    identities: list[str] = []
    for expected_seed, run in zip(full_seeds, runs, strict=True):
        if (
            not isinstance(run, dict)
            or run.get("global_seed") != expected_seed
            or run.get("classes") != list(CLASSES)
        ):
            raise RuntimeError(f"{product_name} ordered trace record changed")
        for field in (
            "identity_sha256",
            "manifest_sha256",
            "completion_sha256",
            "trace_sha256",
            "scientific_fingerprint_sha256",
        ):
            value = run.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"{product_name} trace binding lacks {field}")
        identities.append(run["identity_sha256"])
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"{product_name} repeats trace identities")
    source_bindings = inventory.get("input_label_free_source_inventories")
    if not isinstance(source_bindings, list) or len(source_bindings) != 1:
        raise RuntimeError(f"{product_name} source analysis binding changed")
    binding = source_bindings[0]
    source_inventory_path = Path(str(binding.get("path", ""))).expanduser().resolve()
    forbidden = ("label_lock", "consensus", "review", "adjudicat")
    if any(token in str(source_inventory_path).lower() for token in forbidden):
        raise RuntimeError(f"{product_name} source path looks supervised")
    if (
        not source_inventory_path.is_file()
        or source_inventory_path.is_symlink()
        or binding.get("sha256") != BASE.sha256_file(source_inventory_path)
        or binding.get("manifest_sha256")
        != BASE.sha256_file(source_inventory_path.parent / "manifest.json")
        or binding.get("completion_sha256")
        != BASE.sha256_file(source_inventory_path.parent / "completion.json")
        or binding.get("trace_run_count") != len(full_seeds)
    ):
        raise RuntimeError(f"{product_name} upstream label-free binding changed")

    time_records = inventory.get("time_series_arrays")
    with np.load(root / "time_series.npz", allow_pickle=False) as archive:
        if (
            not isinstance(time_records, dict)
            or set(archive.files) != set(time_records)
            or len(archive.files) != 16
            or any("label" in name.lower() for name in archive.files)
        ):
            raise RuntimeError(f"{product_name} time-series payload schema changed")
        for name in archive.files:
            array = archive[name]
            if not np.isfinite(array).all() or _array_record(array) != time_records[name]:
                raise RuntimeError(f"{product_name} time-series array changed: {name}")

    trace_identity_ordered_sha256 = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cohort[list(KEYS) + features], catalog, {
        "manifest_identity_sha256": manifest_id,
        "manifest_file_sha256": BASE.sha256_file(manifest_path),
        "sample_features_filename": "sample_features.csv",
        "sample_features_sha256": BASE.sha256_file(root / "sample_features.csv"),
        "feature_catalog_sha256": BASE.sha256_file(root / "feature_catalog.csv"),
        "analysis_source_sha256": manifest.get("analysis_source_sha256"),
        "source_inventory_sha256": manifest.get("source_inventory_sha256"),
        "scientific_fingerprint_sha256": inventory.get(
            "scientific_fingerprint_sha256"
        ),
        "full_product_sample_count": len(frame),
        "selected_cohort_sample_count": len(cohort),
        "scalar_feature_count": len(features),
        "track_count": int(catalog["track"].nunique()),
        "trace_identity_ordered_sha256": trace_identity_ordered_sha256,
        "trace_identity_count": len(identities),
        "decoded_images_saved": False,
        "labels_read_or_emitted": False,
        "latest_required_sampling_step": 149,
        "latest_required_internal_timestep": 100,
        "observation_timing": "before_transition_at_latest_checkpoint",
    }


def merge_visual_product(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    visual: pd.DataFrame,
    visual_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_names = set(catalog["feature"].astype(str))
    visual_names = visual_catalog["feature"].astype(str).tolist()
    if existing_names.intersection(visual_names):
        raise RuntimeError("visual feature names overlap the frozen base inventory")
    if set(catalog["track"].astype(str)).intersection(
        visual_catalog["track"].astype(str)
    ):
        raise RuntimeError("visual track names overlap the frozen base inventory")
    merged = features.merge(
        visual[list(KEYS) + visual_names],
        on=list(KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise RuntimeError("visual and frozen-base sample axes differ")
    vc = visual_catalog.copy()
    vc.insert(0, "source_product", "predxstart_visual")
    return merged, pd.concat([catalog, vc], ignore_index=True, sort=False)


def _payload_record(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": BASE.sha256_file(path),
    }


def publish_visual_v2(
    output: Path,
    tables: dict[str, pd.DataFrame],
    lineage: dict[str, Any],
    draws: int,
    permutation_seed: int,
    cohort_summary: dict[str, Any],
    *,
    synthetic_test: bool = False,
) -> None:
    output = output.expanduser().resolve()
    if not synthetic_test and (
        draws != FROZEN_PERMUTATION_DRAWS
        or permutation_seed != FROZEN_PERMUTATION_SEED
        or output != FROZEN_OUTPUT.expanduser().resolve()
    ):
        raise RuntimeError("publisher arguments differ from the frozen execution contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        filenames = {
            "preterminal_step149": "primary_preterminal_step149_track_winners.csv",
            "preterminal_step199": "secondary_preterminal_step199_track_winners.csv",
            "retrospective": "retrospective_track_winners.csv",
        }
        for tier, table in tables.items():
            table.to_csv(staging / filenames[tier], index=False)
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        shutil.copyfile(
            BASE_SOURCE_PATH,
            staging / "explore_dit_bad_good_inventory_replication.py",
        )
        methodology = {
            "schema_version": 2,
            "status": "EXPLORATORY_NEXT_CANDIDATE_GENERATION_ONLY",
            "frozen_execution_contract": FROZEN_EXECUTION_CONTRACT,
            "warning": (
                "Feature and direction were selected after screening the original "
                "cohort. The disjoint expansion estimates replication, but this "
                "full-inventory exercise is exploratory and cannot confirm a newly "
                "named detector."
            ),
            "labels": {"positive": BAD, "negative": GOOD, "excluded": MILD},
            "selection": (
                "Within each tier and trajectory track, maximize original-cohort "
                "within-class pair-weighted directional AUC; deterministic ties use "
                "macro AUC then lexicographic feature name."
            ),
            "visual_product": {
                "semantic_role": (
                    "fixed label-free diagnostic witnesses decoded from pred_xstart; "
                    "ResNet target log-odds is not a quality posterior"
                ),
                "latest_required_sampling_step": 149,
                "latest_required_internal_timestep": 100,
                "observation_timing": "before_transition_at_latest_checkpoint",
                "decoded_images_saved": False,
            },
            "tiers": {
                "primary_preterminal_step149": (
                    "preterminal_actionable=true and latest_required_sampling_step<=149"
                ),
                "secondary_preterminal_step199": (
                    "preterminal_actionable=true and latest_required_sampling_step<=199; "
                    "intentionally includes the <=149 candidate set"
                ),
                "retrospective": (
                    "preterminal_actionable=false or latest_required_sampling_step>199"
                ),
            },
            "replication_test": {
                "statistic": "within-class pair-count-weighted AUC, frozen direction",
                "null": "fixed bad count, labels permuted separately within each class",
                "alternative": "AUC greater than chance in the frozen direction",
                "draws": draws,
                "rng": "numpy.default_rng(PCG64)",
                "rng_seed": permutation_seed,
                "monte_carlo_p": "(1 + exceedances)/(1 + draws)",
                "multiple_testing": (
                    "BH and Holm jointly across every emitted track-tier winner; tiers "
                    "overlap, so identical or related hypotheses may appear more than once"
                ),
            },
            "output_privacy": (
                "aggregate track tables only; no sample key, seed, row label, row score, "
                "rank, endpoint, image, or trace path is emitted"
            ),
        }
        BASE.write_json(staging / "methodology.json", methodology)
        inventory_count = sum(
            int(lineage[name]["scalar_feature_count"])
            for name in (
                "original_primary",
                "original_posterior",
                "original_normalized_posterior",
                "original_predxstart_visual",
            )
        )
        summary = {
            "schema_version": 2,
            "status": "COMPLETE_EXPLORATORY_INVENTORY_REPLICATION_VISUAL_V2",
            "exploratory_not_confirmatory": True,
            "cohorts": cohort_summary,
            "inventory": {
                "scalar_feature_count": inventory_count,
                "catalog_track_count": len(
                    set().union(*(set(table["track"]) for table in tables.values()))
                ),
                "winner_counts": {tier: len(table) for tier, table in tables.items()},
            },
            "permutation_draws": draws,
            "permutation_seed": permutation_seed,
            "row_level_payload_emitted": False,
            "image_payload_emitted": False,
            "frozen_execution_contract_enforced": not synthetic_test,
        }
        BASE.write_json(staging / "summary.json", summary)
        payload_names = [
            *filenames.values(),
            "analysis_source.py",
            "explore_dit_bad_good_inventory_replication.py",
            "methodology.json",
            "summary.json",
        ]
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "status": "complete",
            "experiment": "dit_bad_good_inventory_replication_exploratory_visual_v2",
            "exploratory_not_confirmatory": True,
            "frozen_execution_contract": FROZEN_EXECUTION_CONTRACT,
            "input_lineage": lineage,
            "files": [_payload_record(staging / name) for name in payload_names],
        }
        manifest["identity_sha256"] = BASE.canonical_sha256(manifest)
        BASE.write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_file_sha256": BASE.sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "source_file_sha256": BASE.sha256_file(staging / "analysis_source.py"),
            "frozen_base_source_sha256": BASE.sha256_file(
                staging / "explore_dit_bad_good_inventory_replication.py"
            ),
            "row_level_payload_emitted": False,
            "image_payload_emitted": False,
        }
        BASE.write_json(staging / "completion.json", completion)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def synthetic_self_test() -> None:
    rng = np.random.default_rng(97)
    records: list[dict[str, Any]] = []
    for cohort_shift in (0, 100):
        for class_id in CLASSES:
            for index in range(10):
                is_bad = index < 2
                records.append(
                    {
                        "global_seed": cohort_shift + index,
                        "class_id": class_id,
                        "label": BAD if is_bad else GOOD,
                        "base__mean": rng.normal(),
                        "visual__mean": (4.0 if is_bad else 0.0)
                        + rng.normal(0.0, 0.02),
                    }
                )
    frame = pd.DataFrame(records)
    discovery = frame[frame["global_seed"] < 100].reset_index(drop=True)
    replication = frame[frame["global_seed"] >= 100].reset_index(drop=True)
    catalog = pd.DataFrame(
        [
            {
                "source_product": "primary",
                "feature": "base__mean",
                "track": "base",
                "family": "synthetic",
                "reduction": "mean",
                "latest_required_sampling_step": 100,
                "preterminal_actionable": True,
            },
            {
                "source_product": "predxstart_visual",
                "feature": "visual__mean",
                "track": "visual",
                "family": "synthetic_visual",
                "reduction": "mean",
                "latest_required_sampling_step": 149,
                "preterminal_actionable": True,
            },
        ]
    )
    tables = BASE.analyze(
        discovery, replication, catalog, draws=999, permutation_seed=17
    )
    primary = tables["preterminal_step149"]
    visual = primary[primary["track"] == "visual"].iloc[0]
    assert visual["source_product"] == "predxstart_visual"
    assert visual["direction"] == "bad_high"
    assert visual["replication_pair_weighted_auc"] == 1.0
    temporary_parent = Path(tempfile.mkdtemp(prefix="inventory-visual-v2-selftest-"))
    try:
        output = temporary_parent / "published"
        fake_lineage = {
            "original_primary": {"scalar_feature_count": 1},
            "original_posterior": {"scalar_feature_count": 0},
            "original_normalized_posterior": {"scalar_feature_count": 0},
            "original_predxstart_visual": {"scalar_feature_count": 1},
        }
        try:
            publish_visual_v2(
                output,
                tables,
                fake_lineage,
                draws=999,
                permutation_seed=17,
                cohort_summary={},
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("formal publisher accepted non-frozen arguments")
        publish_visual_v2(
            output,
            tables,
            fake_lineage,
            draws=999,
            permutation_seed=17,
            cohort_summary={"synthetic": {"binary_included": len(frame)}},
            synthetic_test=True,
        )
        manifest = BASE.read_json(output / "manifest.json")
        completion = BASE.read_json(output / "completion.json")
        assert BASE.require_identity(manifest, "visual-v2 synthetic manifest") == completion[
            "manifest_identity_sha256"
        ]
        members = BASE.validate_manifest_members(output, manifest)
        assert set(members).isdisjoint({"sample_features.csv", "labels.csv"})
        assert BASE.read_json(output / "summary.json")["row_level_payload_emitted"] is False
        try:
            publish_visual_v2(
                output, tables, fake_lineage, 999, 17, {}, synthetic_test=True
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("visual-v2 publisher overwrote an existing result")
    finally:
        shutil.rmtree(temporary_parent)
    print("synthetic visual-v2 self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-label-lock", type=Path, default=DEFAULT_ORIGINAL_LABELS)
    parser.add_argument("--expansion-label-lock", type=Path, default=DEFAULT_EXPANSION_LABELS)
    parser.add_argument("--original-primary", type=Path, default=DEFAULT_ORIGINAL_PRIMARY)
    parser.add_argument("--original-posterior", type=Path, default=DEFAULT_ORIGINAL_POSTERIOR)
    parser.add_argument("--expansion-primary", type=Path, default=DEFAULT_EXPANSION_PRIMARY)
    parser.add_argument("--expansion-posterior", type=Path, default=DEFAULT_EXPANSION_POSTERIOR)
    parser.add_argument(
        "--original-normalized-posterior-root",
        type=Path,
        default=DEFAULT_ORIGINAL_NORMALIZED_POSTERIOR,
    )
    parser.add_argument(
        "--expansion-normalized-posterior-root",
        type=Path,
        default=DEFAULT_EXPANSION_NORMALIZED_POSTERIOR,
    )
    parser.add_argument("--original-visual-root", type=Path, default=DEFAULT_ORIGINAL_VISUAL)
    parser.add_argument(
        "--expansion-visual-root", type=Path, default=DEFAULT_EXPANSION_VISUAL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--permutation-draws", type=int, default=FROZEN_PERMUTATION_DRAWS
    )
    parser.add_argument("--permutation-seed", type=int, default=FROZEN_PERMUTATION_SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if (
        args.permutation_draws != FROZEN_PERMUTATION_DRAWS
        or args.permutation_seed != FROZEN_PERMUTATION_SEED
        or args.output.expanduser().resolve() != FROZEN_OUTPUT.expanduser().resolve()
    ):
        raise ValueError(
            "draws, seed, and output must exactly match the frozen execution contract"
        )

    # Every label-free product is validated and merged before a label file is opened.
    op, opc, op_lineage = BASE.validate_feature_product(
        args.original_primary, DISCOVERY_SEEDS, "original primary"
    )
    oq, oqc, oq_lineage = BASE.validate_feature_product(
        args.original_posterior, DISCOVERY_SEEDS, "original posterior"
    )
    ep, epc, ep_lineage = BASE.validate_feature_product(
        args.expansion_primary, REPLICATION_SEEDS, "expansion primary"
    )
    eq, eqc, eq_lineage = BASE.validate_feature_product(
        args.expansion_posterior, REPLICATION_SEEDS, "expansion posterior"
    )
    on, onc, on_lineage = BASE.validate_feature_product(
        args.original_normalized_posterior_root,
        DISCOVERY_SEEDS,
        "original normalized posterior",
        sample_filename="sample_features_label_free.csv",
        require_supervision_audit=True,
    )
    en, enc, en_lineage = BASE.validate_feature_product(
        args.expansion_normalized_posterior_root,
        REPLICATION_SEEDS,
        "expansion normalized posterior",
        sample_filename="sample_features_label_free.csv",
        require_supervision_audit=True,
    )
    ov, ovc, ov_lineage = validate_visual_product(
        args.original_visual_root,
        DISCOVERY_SEEDS,
        EXPECTED_ORIGINAL_VISUAL_FULL_SEEDS,
        "original pred_xstart visual",
    )
    ev, evc, ev_lineage = validate_visual_product(
        args.expansion_visual_root,
        REPLICATION_SEEDS,
        EXPECTED_REPLICATION_VISUAL_FULL_SEEDS,
        "expansion pred_xstart visual",
    )

    observed_feature_manifest_identities = {
        "original_primary": op_lineage["manifest_identity_sha256"],
        "original_posterior": oq_lineage["manifest_identity_sha256"],
        "original_normalized_posterior": on_lineage["manifest_identity_sha256"],
        "original_predxstart_visual": ov_lineage["manifest_identity_sha256"],
        "expansion_primary": ep_lineage["manifest_identity_sha256"],
        "expansion_posterior": eq_lineage["manifest_identity_sha256"],
        "expansion_normalized_posterior": en_lineage["manifest_identity_sha256"],
        "expansion_predxstart_visual": ev_lineage["manifest_identity_sha256"],
    }
    if observed_feature_manifest_identities != FROZEN_FEATURE_MANIFEST_IDENTITIES:
        raise RuntimeError("a feature product differs from the frozen manifest identities")

    BASE.validate_catalog_replication(opc, epc, "primary")
    BASE.validate_catalog_replication(oqc, eqc, "posterior")
    BASE.validate_catalog_replication(onc, enc, "normalized posterior")
    BASE.validate_catalog_replication(ovc, evc, "pred_xstart visual")
    if (
        op_lineage["scalar_feature_count"] != EXPECTED_PRIMARY_FEATURES
        or ep_lineage["scalar_feature_count"] != EXPECTED_PRIMARY_FEATURES
        or oq_lineage["scalar_feature_count"] != EXPECTED_POSTERIOR_FEATURES
        or eq_lineage["scalar_feature_count"] != EXPECTED_POSTERIOR_FEATURES
        or on_lineage["scalar_feature_count"] != EXPECTED_NORMALIZED_POSTERIOR_FEATURES
        or en_lineage["scalar_feature_count"] != EXPECTED_NORMALIZED_POSTERIOR_FEATURES
        or ov_lineage["scalar_feature_count"] != EXPECTED_VISUAL_FEATURES
        or ev_lineage["scalar_feature_count"] != EXPECTED_VISUAL_FEATURES
        or ov_lineage["track_count"] != EXPECTED_VISUAL_TRACKS
        or ev_lineage["track_count"] != EXPECTED_VISUAL_TRACKS
        or op_lineage["feature_catalog_sha256"] != ep_lineage["feature_catalog_sha256"]
        or oq_lineage["feature_catalog_sha256"] != eq_lineage["feature_catalog_sha256"]
        or on_lineage["feature_catalog_sha256"] != en_lineage["feature_catalog_sha256"]
        or ov_lineage["feature_catalog_sha256"] != ev_lineage["feature_catalog_sha256"]
        or op_lineage["trace_identity_ordered_sha256"]
        != oq_lineage["trace_identity_ordered_sha256"]
        or ep_lineage["trace_identity_ordered_sha256"]
        != eq_lineage["trace_identity_ordered_sha256"]
        or ep_lineage["trace_identity_ordered_sha256"]
        != en_lineage["trace_identity_ordered_sha256"]
        or op_lineage["trace_identity_ordered_sha256"]
        != ov_lineage["trace_identity_ordered_sha256"]
        or ep_lineage["trace_identity_ordered_sha256"]
        != ev_lineage["trace_identity_ordered_sha256"]
        or op_lineage["trace_identity_count"] != 100
        or oq_lineage["trace_identity_count"] != 100
        or on_lineage["trace_identity_count"] not in (len(DISCOVERY_SEEDS), 100)
        or ov_lineage["trace_identity_count"] != 100
        or ep_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or eq_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or en_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or ev_lineage["trace_identity_count"] != len(REPLICATION_SEEDS)
        or op_lineage["analysis_source_sha256"] != ep_lineage["analysis_source_sha256"]
        or oq_lineage["analysis_source_sha256"] != eq_lineage["analysis_source_sha256"]
        or oq_lineage["imported_validation_helper_sha256"]
        != eq_lineage["imported_validation_helper_sha256"]
        or oq_lineage["imported_validation_helper_sha256"]
        != op_lineage["analysis_source_sha256"]
        or on_lineage["analysis_source_sha256"] != en_lineage["analysis_source_sha256"]
        or ov_lineage["analysis_source_sha256"] != EXPECTED_VISUAL_EXTRACTOR_SHA256
        or ev_lineage["analysis_source_sha256"] != EXPECTED_VISUAL_EXTRACTOR_SHA256
        or ov_lineage["scientific_fingerprint_sha256"]
        != ev_lineage["scientific_fingerprint_sha256"]
    ):
        raise RuntimeError("discovery and replication extractor lineages differ")
    if (
        on_lineage["trace_identity_count"] == op_lineage["trace_identity_count"]
        and on_lineage["trace_identity_ordered_sha256"]
        != op_lineage["trace_identity_ordered_sha256"]
    ):
        raise RuntimeError("original normalized posterior came from different traces")

    original_base, base_catalog = BASE.merge_products(op, oq, opc, oqc, on, onc)
    expansion_base, expansion_base_catalog = BASE.merge_products(
        ep, eq, epc, eqc, en, enc
    )
    if (
        len(base_catalog) != EXPECTED_BASE_FEATURES
        or base_catalog["track"].nunique() != EXPECTED_BASE_TRACKS
    ):
        raise RuntimeError("frozen v1 base inventory is not 6844 features/192 tracks")
    original_features, catalog = merge_visual_product(
        original_base, base_catalog, ov, ovc
    )
    expansion_features, expansion_catalog = merge_visual_product(
        expansion_base, expansion_base_catalog, ev, evc
    )
    BASE.validate_catalog_replication(catalog, expansion_catalog, "visual-v2 combined")
    if (
        len(catalog) != EXPECTED_TOTAL_FEATURES
        or catalog["feature"].nunique() != EXPECTED_TOTAL_FEATURES
        or catalog["track"].nunique() != EXPECTED_TOTAL_TRACKS
        or len(original_features) != len(DISCOVERY_SEEDS) * len(CLASSES)
        or len(expansion_features) != len(REPLICATION_SEEDS) * len(CLASSES)
    ):
        raise RuntimeError("combined inventory is not the frozen 6884-feature/200-track set")

    # Label locks are opened only after the label-free 6884-feature inventory is fixed.
    original_labels, original_label_lineage = BASE.validate_label_lock(
        args.original_label_lock,
        DISCOVERY_SEEDS,
        "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
    )
    expansion_labels, expansion_label_lineage = BASE.validate_label_lock(
        args.expansion_label_lock,
        REPLICATION_SEEDS,
        "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN",
    )
    observed_label_consensus_identities = {
        "original": original_label_lineage["consensus_identity_sha256"],
        "expansion": expansion_label_lineage["consensus_identity_sha256"],
    }
    if observed_label_consensus_identities != FROZEN_LABEL_CONSENSUS_IDENTITIES:
        raise RuntimeError("a label consensus differs from the frozen identities")
    discovery = BASE.join_labels(original_features, original_labels)
    replication = BASE.join_labels(expansion_features, expansion_labels)
    tables = BASE.analyze(
        discovery,
        replication,
        catalog,
        draws=args.permutation_draws,
        permutation_seed=args.permutation_seed,
    )
    lineage = {
        "frozen_v1_screen_source_sha256": EXPECTED_BASE_SOURCE_SHA256,
        "original_label_lock": original_label_lineage,
        "expansion_label_lock": expansion_label_lineage,
        "original_primary": op_lineage,
        "original_posterior": oq_lineage,
        "original_normalized_posterior": on_lineage,
        "original_predxstart_visual": ov_lineage,
        "expansion_primary": ep_lineage,
        "expansion_posterior": eq_lineage,
        "expansion_normalized_posterior": en_lineage,
        "expansion_predxstart_visual": ev_lineage,
    }
    cohort_summary = {
        "discovery": {
            "total_locked": len(original_labels),
            "binary_included": len(discovery),
            "counts": original_label_lineage["counts"],
            "class_count": len(CLASSES),
        },
        "independent_replication": {
            "total_locked": len(expansion_labels),
            "binary_included": len(replication),
            "counts": expansion_label_lineage["counts"],
            "class_count": len(CLASSES),
        },
    }
    publish_visual_v2(
        args.output,
        tables,
        lineage,
        args.permutation_draws,
        args.permutation_seed,
        cohort_summary,
    )
    print(f"published aggregate exploratory visual-v2 screen: {args.output.resolve()}")


if __name__ == "__main__":
    main()

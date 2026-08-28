#!/usr/bin/env python3
"""Fail-closed, aggregate-only operational audit of two frozen DiT candidates.

This audit intentionally does *not* select a feature, direction, window, or
combination.  It evaluates two candidates that were already named by sealed
exploratory screens:

* primary: ``decoded_local_blur_severity__mean`` (higher is riskier), and
* secondary: ``pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump``
  (lower is riskier).

The real-data path is controlled entirely by a canonical frozen protocol JSON.
The CLI exposes no scientific or publication override.  The result contains
aggregate statistics only: no seed, sample key, row label, row score, rank,
image, endpoint, or trace path is published.

Important: this is a post-selection operational audit on cohorts already used
by exploratory screening.  Its permutation and conformal-style summaries are
descriptive diagnostics, not a fresh confirmatory test and not a Ville test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


KEYS = ("global_seed", "class_id")
GOOD = "clean_good"
BAD = "clear_bad"
MILD = "mild_or_disputed"
LABELS = (GOOD, BAD, MILD)
CLASSES = (207, 602, 795)
DISCOVERY_SEEDS = tuple(range(50, 130))
EXPANSION_SEEDS = tuple(range(130, 250))

EXPECTED_HELPER_SHA256 = (
    "c23ef50713e3a310bf8a6bb7b0541563717b5979cca035f3b15394f68aaaca90"
)
EXPECTED_LABEL_CONSENSUS = {
    "discovery": "21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a",
    "expansion": "fc478b7ae04b67869d0dfca3b63f169a5266bacc1c8701e1ae20368516e793fd",
}
EXPECTED_FEATURE_MANIFESTS = {
    "discovery_primary": "a0aa2d1c8ffb8a2be2b95cc6cba7710a282f22ff3d5f24d246e8ea23b6e01044",
    "expansion_primary": "e9e441c379796a2066ef672987c36dd52bc77b59578e3133b8372743c1593a31",
    "discovery_visual": "7554c9bf8a15375e83df50a4308a00176a5e2dc5deb8f378a5611cd8f3c322cd",
    "expansion_visual": "bddd0e9d17fcce2f1c87a02373991a83b6091b6b46e05f8fe7f7566718913696",
}
EXPECTED_SCREEN_MANIFESTS = {
    "blur_screen": "a5d5b13317fdc6e5881398da1c91e92a140c6c1f77c9cd15f7f8a57119e1527c",
    "c3_screen": "04458d39bf8ab34242b1287140f8a6e83cf712afdc49de35f76ab6d774fe55a9",
}
EXPECTED_DRAWS = 100_000
EXPECTED_ALPHAS = (0.05, 0.10, 0.20)
EXPECTED_SCREEN_MEMBER = "primary_preterminal_step149_track_winners.csv"
EXPECTED_PROTOCOL_STATUS = "FROZEN_DUAL_CANDIDATE_OPERATIONAL_AUDIT_AMENDMENT_V1_1"
SUPERSEDED_PROTOCOL_IDENTITY_SHA256 = (
    "d6bad25c50ea40819f29000783bae572c5af3b165d62f9274d7400f3c7c0c27c"
)
SUPERSEDED_PROTOCOL_FILE_SHA256 = (
    "fc3484603df9d43c9480d42d6a5d9d7ebffaf82f87995c3d7c28d6cae651a8e9"
)
EXPECTED_VISUAL_EXTRACTOR_SHA256 = (
    "452ae0e61fe36d027036e0d74c232fbcfbd7cb462d3749db92e062a104d0e398"
)
EXPECTED_VISUAL_EXPERIMENT = "dit_predxstart_preterminal_visual_tracks_label_free"
EXPECTED_VISUAL_FEATURES = 40
EXPECTED_VISUAL_TRACKS = 8
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
VISUAL_IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
)
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
BLUR_STEPS = (69, 79, 89, 99, 109, 119, 129, 139, 149)
BLUR_INTERNAL_TIMESTEPS = (180, 170, 160, 150, 140, 130, 120, 110, 100)
C3_Q2_STEPS = tuple(range(100, 150))
EXPECTED_TIME_SERIES: dict[str, dict[str, Any]] = {
    "discovery_primary": {
        "npz_sha256": "6f8bbb0fa5d4b42b7f5509ac9398cf0ccc1612eeab119f1cbac580d95b16cb0d",
        "row_count": 300,
        "arrays": {
            "global_seed": {
                "shape": [300],
                "dtype": "int64",
                "raw_sha256": "8078caead813002ed470df147625ac7e24e0b22f87c415e4c4818619fad4ef02",
            },
            "class_id": {
                "shape": [300],
                "dtype": "int16",
                "raw_sha256": "604ebde08d52055e6f5a07dcafe0ba4d93c1f2be069d8f8236fc4e0e9e2e9d98",
            },
            "sampling_step_250": {
                "shape": [250],
                "dtype": "int16",
                "raw_sha256": "73a6bd8e22ccd6317c7918f34a73b9a5ebaaf679f1d74af561251c905ec276bb",
            },
            "internal_timestep_250": {
                "shape": [250],
                "dtype": "int16",
                "raw_sha256": "1ae8e9470a17622397f15a6b981c1cc4de802532d6a934289c3a7b88c653ce27",
            },
            "pred_xstart_alpha_compensated_gradient_energy_c3": {
                "shape": [300, 250],
                "dtype": "float64",
                "raw_sha256": "6a59bd99d07fc2a20b588dcc5c4417e4551408b26f4c1de4eda9773eb6cf9eae",
            },
        },
    },
    "expansion_primary": {
        "npz_sha256": "4f58d2805f177dbed089b5286b2ba76e9e6e43405d1ae69b32184e5ceae8f0f0",
        "row_count": 360,
        "arrays": {
            "global_seed": {
                "shape": [360],
                "dtype": "int64",
                "raw_sha256": "c517a141babe985cd083b29b76c6145f8c22b28cc890fe5a53eb36b27c50a6e1",
            },
            "class_id": {
                "shape": [360],
                "dtype": "int16",
                "raw_sha256": "8f76d55e7bf6ea7d74de86a1ebbce1a92597387083de5dd05a57a807dbd2083b",
            },
            "sampling_step_250": {
                "shape": [250],
                "dtype": "int16",
                "raw_sha256": "73a6bd8e22ccd6317c7918f34a73b9a5ebaaf679f1d74af561251c905ec276bb",
            },
            "internal_timestep_250": {
                "shape": [250],
                "dtype": "int16",
                "raw_sha256": "1ae8e9470a17622397f15a6b981c1cc4de802532d6a934289c3a7b88c653ce27",
            },
            "pred_xstart_alpha_compensated_gradient_energy_c3": {
                "shape": [360, 250],
                "dtype": "float64",
                "raw_sha256": "78ff0a2703aa1d917a868ead3430f3948bc7679c337bdda4b558cfea3a7c767c",
            },
        },
    },
    "discovery_visual": {
        "npz_sha256": "9c8dbbc30bce973548314a3bef50c7f9fed5f386d380fae576075646f048811c",
        "row_count": 300,
        "arrays": {
            "global_seed": {
                "shape": [300],
                "dtype": "int64",
                "raw_sha256": "8078caead813002ed470df147625ac7e24e0b22f87c415e4c4818619fad4ef02",
            },
            "class_id": {
                "shape": [300],
                "dtype": "int16",
                "raw_sha256": "604ebde08d52055e6f5a07dcafe0ba4d93c1f2be069d8f8236fc4e0e9e2e9d98",
            },
            "selected_sampling_step": {
                "shape": [9],
                "dtype": "int16",
                "raw_sha256": "bece1004441c32eea8bbce159c7f3efce5b25a06786aa2a853c072e21769b756",
            },
            "selected_internal_timestep": {
                "shape": [9],
                "dtype": "int16",
                "raw_sha256": "0c878fe46e56de57ed3cc2cac0a8f5e5b4ea1e8e568475f57be9ad9c2c94863a",
            },
            "decoded_local_blur_severity": {
                "shape": [300, 9],
                "dtype": "float64",
                "raw_sha256": "b68b6f9f90156ee4046bf86b227ea8c6ede6fc08f89e9704c13f3009dd918937",
            },
        },
    },
    "expansion_visual": {
        "npz_sha256": "a7786a4a4d34470f0f4571ef422e418ad82eeff794d6219c53eb669aba0f53be",
        "row_count": 360,
        "arrays": {
            "global_seed": {
                "shape": [360],
                "dtype": "int64",
                "raw_sha256": "c517a141babe985cd083b29b76c6145f8c22b28cc890fe5a53eb36b27c50a6e1",
            },
            "class_id": {
                "shape": [360],
                "dtype": "int16",
                "raw_sha256": "8f76d55e7bf6ea7d74de86a1ebbce1a92597387083de5dd05a57a807dbd2083b",
            },
            "selected_sampling_step": {
                "shape": [9],
                "dtype": "int16",
                "raw_sha256": "bece1004441c32eea8bbce159c7f3efce5b25a06786aa2a853c072e21769b756",
            },
            "selected_internal_timestep": {
                "shape": [9],
                "dtype": "int16",
                "raw_sha256": "0c878fe46e56de57ed3cc2cac0a8f5e5b4ea1e8e568475f57be9ad9c2c94863a",
            },
            "decoded_local_blur_severity": {
                "shape": [360, 9],
                "dtype": "float64",
                "raw_sha256": "13f22f7f99f18a6e9e957185aab1bfaf9b6477b1524c242dd742f83fbbf667ba",
            },
        },
    },
}

BLUR = "decoded_local_blur_severity_mean"
C3 = "alpha_compensated_gradient_energy_c3_q2_jump"
CANDIDATE_ORDER = (BLUR, C3)
CANDIDATES: dict[str, dict[str, Any]] = {
    BLUR: {
        "role": "primary",
        "feature": "decoded_local_blur_severity__mean",
        "source_product": "predxstart_visual",
        "track": "decoded_local_blur_severity",
        "family": "decoded_pixels",
        "reduction": "mean over selected checkpoints",
        "direction": "bad_high",
        "risk_transform": "raw",
        "screen_key": "blur_screen",
        "catalog": {
            "feature_index": 0,
            "availability": "pre_innovation_current_model_output",
            "latest_required_sampling_step": 149,
            "latest_required_internal_timestep": 100,
            "observation_timing": "before_transition_at_latest_checkpoint",
            "preterminal_actionable": True,
            "track_length": 9,
            "uses_realized_innovation": False,
            "checkpoint_sampling_steps": "69,79,89,99,109,119,129,139,149",
            "checkpoint_internal_timesteps": "180,170,160,150,140,130,120,110,100",
            "formula_source": "protocol_snapshot.json",
            "deployment_note": (
                "fixed decoded-pred_xstart diagnostic; no endpoint, future innovation, "
                "label, or quality posterior is used"
            ),
        },
        "protocol_formula": (
            "q_j=mean(l_j^2)/(mean(g_j^2)+1e-12); "
            "B=-log(percentile_25(q over active tiles)+1e-12)"
        ),
    },
    C3: {
        "role": "secondary",
        "feature": (
            "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
        ),
        "source_product": "primary",
        "track": "pred_xstart_alpha_compensated_gradient_energy_c3",
        "family": "predicted_clean",
        "reduction": "q2_max_positive_jump",
        "direction": "bad_low",
        "risk_transform": "negative_raw",
        "screen_key": "c3_screen",
        "catalog": {
            "track_formula": (
                "alpha_bar[k] * (mean vertical-difference^2 + mean "
                "horizontal-difference^2), pred_xstart channel 3"
            ),
            "feature_formula": (
                "max_positive_jump over "
                "pred_xstart_alpha_compensated_gradient_energy_c3[100:150] "
                "under fixed phase q2"
            ),
            "track_length": 250,
            "availability": "predictable",
            "latest_required_sampling_step": 149,
            "latest_required_internal_timestep": 100,
            "observation_timing": "before_transition_at_latest_step",
            "preterminal_actionable": True,
            "uses_realized_innovation": False,
            "deployment_note": "none",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("identity_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_identity(value: dict[str, Any], description: str) -> str:
    observed = value.get("identity_sha256")
    expected = canonical_sha256(value)
    if observed != expected:
        raise RuntimeError(f"{description} canonical identity mismatch")
    return expected


def require_exact_keys(
    value: dict[str, Any], expected: set[str], description: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise RuntimeError(
            f"{description} keys differ; missing={missing}, extra={extra}"
        )


def _absolute_real_path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{description} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{description} must be absolute")
    return path.resolve()


def _require_sha(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} is not a lowercase SHA-256")
    return value


def validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = protocol_path.expanduser().resolve()
    if not protocol_path.is_file() or protocol_path.is_symlink():
        raise RuntimeError("protocol must be a real, non-symlink file")
    protocol = read_json(protocol_path)
    require_exact_keys(
        protocol,
        {
            "schema_version",
            "status",
            "scientific_role",
            "runtime",
            "cohorts",
            "feature_products",
            "exploratory_screens",
            "candidates",
            "statistics",
            "time_series_descriptive",
            "publication",
            "claim_limits",
            "amendment_lineage",
            "identity_sha256",
        },
        "protocol",
    )
    protocol_id = require_identity(protocol, "protocol")
    if protocol.get("schema_version") != 1 or protocol.get("status") != EXPECTED_PROTOCOL_STATUS:
        raise RuntimeError("protocol is not a frozen operational-audit v1 protocol")
    if protocol.get("scientific_role") != "post_selection_operational_audit_not_confirmation":
        raise RuntimeError("protocol scientific role changed")
    amendment = protocol["amendment_lineage"]
    if amendment != {
        "status": "MECHANICAL_VALIDATOR_FIX_AFTER_FAIL_CLOSED_BEFORE_LABEL_LOAD",
        "superseded_protocol_identity_sha256": SUPERSEDED_PROTOCOL_IDENTITY_SHA256,
        "superseded_protocol_file_sha256": SUPERSEDED_PROTOCOL_FILE_SHA256,
        "failure_stage": "discovery_visual feature-envelope validation",
        "failure_message": "discovery_visual has invalid ordered trace identities",
        "root_cause": (
            "visual-v2 lineage is represented by source_inventory trace_runs, not "
            "manifest.trace_identity_sha256_ordered expected by the frozen base validator"
        ),
        "sole_change": (
            "use an embedded strict visual-v2 validator with exact file set, supervision, "
            "source-inventory trace lineage, all-array NPZ hashes, axes, and feature replay"
        ),
        "labels_loaded": False,
        "labels_joined_to_features": False,
        "statistics_started": False,
        "output_published": False,
        "scientific_rules_changed": False,
    }:
        raise RuntimeError("protocol amendment/failure lineage changed")

    runtime = protocol["runtime"]
    require_exact_keys(
        runtime,
        {
            "audit_source_path",
            "audit_source_sha256",
            "pinned_inventory_helper_path",
            "pinned_inventory_helper_sha256",
        },
        "protocol.runtime",
    )
    source_path = _absolute_real_path(runtime["audit_source_path"], "audit source")
    actual_source = Path(__file__).resolve()
    if source_path != actual_source or not actual_source.is_file() or actual_source.is_symlink():
        raise RuntimeError("protocol audit source path does not name this real source file")
    expected_source_sha = _require_sha(runtime["audit_source_sha256"], "audit source SHA")
    observed_source_sha = sha256_file(actual_source)
    if observed_source_sha != expected_source_sha:
        raise RuntimeError(
            f"audit source changed: expected {expected_source_sha}, got {observed_source_sha}"
        )

    helper_path = _absolute_real_path(
        runtime["pinned_inventory_helper_path"], "inventory helper"
    )
    helper_sha = _require_sha(
        runtime["pinned_inventory_helper_sha256"], "inventory helper SHA"
    )
    if (
        helper_sha != EXPECTED_HELPER_SHA256
        or not helper_path.is_file()
        or helper_path.is_symlink()
        or sha256_file(helper_path) != EXPECTED_HELPER_SHA256
    ):
        raise RuntimeError("pinned inventory validation helper changed")

    cohorts = protocol["cohorts"]
    require_exact_keys(cohorts, {"discovery", "expansion"}, "protocol.cohorts")
    expected_cohorts = {
        "discovery": (50, 130, 240, EXPECTED_LABEL_CONSENSUS["discovery"]),
        "expansion": (130, 250, 360, EXPECTED_LABEL_CONSENSUS["expansion"]),
    }
    for name, (start, stop, count, consensus_id) in expected_cohorts.items():
        item = cohorts[name]
        require_exact_keys(
            item,
            {
                "label_lock_path",
                "label_consensus_identity_sha256",
                "global_seed_start_inclusive",
                "global_seed_stop_exclusive",
                "trajectory_count",
                "classes",
            },
            f"protocol.cohorts.{name}",
        )
        if (
            item["global_seed_start_inclusive"] != start
            or item["global_seed_stop_exclusive"] != stop
            or item["trajectory_count"] != count
            or item["classes"] != list(CLASSES)
            or item["label_consensus_identity_sha256"] != consensus_id
        ):
            raise RuntimeError(f"frozen cohort specification changed: {name}")
        _absolute_real_path(item["label_lock_path"], f"{name} label lock")

    products = protocol["feature_products"]
    if set(products) != set(EXPECTED_FEATURE_MANIFESTS):
        raise RuntimeError("feature-product keys changed")
    for name, expected_id in EXPECTED_FEATURE_MANIFESTS.items():
        item = products[name]
        require_exact_keys(item, {"root", "manifest_identity_sha256"}, f"product {name}")
        _absolute_real_path(item["root"], f"feature product {name}")
        if item["manifest_identity_sha256"] != expected_id:
            raise RuntimeError(f"feature-product identity changed: {name}")

    screens = protocol["exploratory_screens"]
    if set(screens) != set(EXPECTED_SCREEN_MANIFESTS):
        raise RuntimeError("exploratory-screen keys changed")
    for name, expected_id in EXPECTED_SCREEN_MANIFESTS.items():
        item = screens[name]
        require_exact_keys(
            item,
            {"root", "manifest_identity_sha256", "candidate_table_member"},
            f"screen {name}",
        )
        _absolute_real_path(item["root"], f"screen {name}")
        if (
            item["manifest_identity_sha256"] != expected_id
            or item["candidate_table_member"] != EXPECTED_SCREEN_MEMBER
        ):
            raise RuntimeError(f"exploratory-screen pin changed: {name}")

    if protocol["candidates"] != CANDIDATES:
        raise RuntimeError("candidate formula, timing, role, or direction changed")

    time_series = protocol["time_series_descriptive"]
    require_exact_keys(
        time_series,
        {"role", "npz_member", "products", "fixed_grids", "inference_policy"},
        "protocol.time_series_descriptive",
    )
    if (
        time_series["role"] != "post_selection_descriptive_only"
        or time_series["npz_member"] != "time_series.npz"
        or time_series["products"] != EXPECTED_TIME_SERIES
        or time_series["fixed_grids"]
        != {
            "blur_level_sampling_steps": list(BLUR_STEPS),
            "c3_q2_level_sampling_steps": list(C3_Q2_STEPS),
            "c3_q2_adjacent_jump_sampling_steps": list(C3_Q2_STEPS[1:]),
            "c3_adjacent_jump_formula": "G[k] - G[k-1] within q2 only",
        }
        or time_series["inference_policy"]
        != "no p-values, thresholds, window selection, candidate selection, or score combination"
    ):
        raise RuntimeError("fixed descriptive time-series contract changed")

    statistics = protocol["statistics"]
    require_exact_keys(
        statistics,
        {"labels", "block_permutation", "conformal_risk", "group_score_summary"},
        "protocol.statistics",
    )
    if statistics["labels"] != {
        "positive": BAD,
        "negative": GOOD,
        "excluded_from_auc": MILD,
        "mild_retained_in_seed_block_permutation": True,
    }:
        raise RuntimeError("label roles changed")
    permutation = statistics["block_permutation"]
    require_exact_keys(
        permutation,
        {
            "draws",
            "seed",
            "scheme",
            "alternative",
            "monte_carlo_p_formula",
            "multiple_testing",
        },
        "protocol.statistics.block_permutation",
    )
    if (
        permutation["draws"] != EXPECTED_DRAWS
        or not isinstance(permutation["seed"], int)
        or isinstance(permutation["seed"], bool)
        or permutation["seed"] < 0
        or permutation["scheme"]
        != "permute intact three-class label-and-mild vectors across global_seed"
        or permutation["alternative"]
        != "class_matched_pair_weighted_auc_greater_than_chance_in_frozen_direction"
        or permutation["monte_carlo_p_formula"] != "(1+exceedances)/(1+draws)"
        or permutation["multiple_testing"] != "Holm across exactly the two frozen candidates"
    ):
        raise RuntimeError("block-permutation specification changed")
    conformal = statistics["conformal_risk"]
    require_exact_keys(
        conformal,
        {
            "calibration_population",
            "class_specific",
            "risk_p_formula",
            "alert_rule",
            "alphas",
            "alpha_semantics",
        },
        "protocol.statistics.conformal_risk",
    )
    if (
        conformal["calibration_population"] != "discovery clean_good only"
        or conformal["class_specific"] is not True
        or conformal["risk_p_formula"]
        != "(1 + count(calibration_risk >= test_risk)) / (n_calibration + 1)"
        or conformal["alert_rule"] != "risk_p <= alpha"
        or tuple(conformal["alphas"]) != EXPECTED_ALPHAS
        or conformal["alpha_semantics"]
        != "upper-tail conformal-style operational risk level; not Ville alpha"
    ):
        raise RuntimeError("conformal-risk specification changed")
    group_spec = statistics["group_score_summary"]
    require_exact_keys(
        group_spec,
        {"minimum_group_size", "quantiles", "small_group_policy"},
        "protocol.statistics.group_score_summary",
    )
    if (
        not isinstance(group_spec["minimum_group_size"], int)
        or group_spec["minimum_group_size"] < 2
        or group_spec["quantiles"] != [0.10, 0.25, 0.50, 0.75, 0.90]
        or group_spec["small_group_policy"] != "counts_only_statistics_null"
    ):
        raise RuntimeError("group-summary privacy policy changed")

    publication = protocol["publication"]
    require_exact_keys(
        publication,
        {
            "output",
            "atomic",
            "no_overwrite",
            "aggregate_only",
            "row_level_payload_emitted",
            "sealed_members",
        },
        "protocol.publication",
    )
    output = _absolute_real_path(publication["output"], "publication output")
    if (
        output == Path("/")
        or output.parent == Path("/")
        or publication["atomic"] is not True
        or publication["no_overwrite"] is not True
        or publication["aggregate_only"] is not True
        or publication["row_level_payload_emitted"] is not False
        or publication["sealed_members"]
        != ["audit_source.py", "pinned_inventory_helper.py", "protocol_snapshot.json"]
    ):
        raise RuntimeError("publication safety contract changed")
    claim_limits = protocol["claim_limits"]
    if claim_limits != {
        "post_selection": True,
        "not_fresh_confirmation": True,
        "no_combined_score": True,
        "no_intervention_authorized": True,
        "conformal_alpha_is_not_ville_alpha": True,
    }:
        raise RuntimeError("claim limits changed")
    return protocol, {
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_identity_sha256": protocol_id,
        "source_path": str(actual_source),
        "source_sha256": observed_source_sha,
        "helper_path": str(helper_path),
        "helper_sha256": helper_sha,
        "output": str(output),
    }


def load_pinned_helper(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_dit_inventory_pinned_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct pinned-helper import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def precheck_manifest_identity(
    root: Path, expected_identity: str, description: str
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{description} must be a real directory")
    manifest = read_json(root / "manifest.json")
    observed = require_identity(manifest, f"{description} manifest")
    if observed != expected_identity:
        raise RuntimeError(f"{description} manifest identity changed")
    return manifest


def validate_screen(
    helper: ModuleType,
    root: Path,
    expected_identity: str,
    candidate_name: str,
    member_name: str,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = precheck_manifest_identity(root, expected_identity, candidate_name + " screen")
    members = helper.validate_manifest_members(root, manifest)
    completion = read_json(root / "completion.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("exploratory_not_confirmatory") is not True
        or member_name not in members
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("manifest_identity_sha256") != expected_identity
        or completion.get("row_level_payload_emitted") is not False
    ):
        raise RuntimeError(f"invalid sealed exploratory screen: {candidate_name}")
    table = pd.read_csv(root / member_name)
    spec = CANDIDATES[candidate_name]
    matches = table[
        table["feature"].astype(str).eq(spec["feature"])
        & table["track"].astype(str).eq(spec["track"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"screen does not name exactly one frozen candidate: {candidate_name}")
    row = matches.iloc[0]
    expected_row = {
        "tier": "preterminal_step149",
        "source_product": spec["source_product"],
        "track": spec["track"],
        "family": spec["family"],
        "feature": spec["feature"],
        "reduction": spec["reduction"],
        "direction": spec["direction"],
        "latest_required_sampling_step": 149,
        "preterminal_actionable": True,
        "exploratory_not_confirmatory": True,
    }
    for key, expected in expected_row.items():
        if key not in row.index:
            raise RuntimeError(f"screen candidate row lacks {key}: {candidate_name}")
        observed = row[key]
        if isinstance(expected, bool):
            observed_bool = str(observed).lower() == "true"
            if observed_bool != expected:
                raise RuntimeError(f"screen candidate {key} changed: {candidate_name}")
        elif isinstance(expected, int):
            if int(observed) != expected:
                raise RuntimeError(f"screen candidate {key} changed: {candidate_name}")
        elif str(observed) != expected:
            raise RuntimeError(f"screen candidate {key} changed: {candidate_name}")
    return {
        "manifest_identity_sha256": expected_identity,
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "candidate_table_member": member_name,
        "candidate_table_sha256": sha256_file(root / member_name),
    }


def _catalog_value_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(observed, (bool, np.bool_)):
            return bool(observed) == expected
        return str(observed).lower() == str(expected).lower()
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(observed) == expected and float(observed) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(observed) == str(expected)


def validate_catalog_candidate(
    catalog: pd.DataFrame, candidate_name: str, description: str
) -> None:
    spec = CANDIDATES[candidate_name]
    matches = catalog[catalog["feature"].astype(str).eq(spec["feature"])]
    if len(matches) != 1:
        raise RuntimeError(f"{description} lacks exactly one candidate feature")
    row = matches.iloc[0]
    base = {
        "feature": spec["feature"],
        "track": spec["track"],
        "family": spec["family"],
        "reduction": spec["reduction"],
    }
    expected_fields = {**base, **spec["catalog"]}
    for key, expected in expected_fields.items():
        if key not in row.index or not _catalog_value_equal(row[key], expected):
            observed = None if key not in row.index else row[key]
            raise RuntimeError(
                f"{description} candidate catalog field changed: {key}; "
                f"expected={expected!r}, observed={observed!r}"
            )


def _canonical_sha256_without(value: dict[str, Any], key: str) -> str:
    copied = dict(value)
    copied.pop(key, None)
    return canonical_sha256(copied)


def _array_inventory_record(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "raw_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _require_visual_exact_file_set(
    root: Path, manifest_members: dict[str, Any]
) -> None:
    entries = list(root.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RuntimeError(f"visual product contains a directory/symlink: {root}")
    actual = {path.name for path in entries}
    expected = set(manifest_members) | {"manifest.json", "completion.json"}
    if actual != expected:
        raise RuntimeError(f"visual product member set changed: {actual} != {expected}")
    images = sorted(
        path.name for path in entries if path.suffix.lower() in FORBIDDEN_IMAGE_SUFFIXES
    )
    if images:
        raise RuntimeError(f"visual product unexpectedly emits images: {images}")


def _require_clean_supervision_audit(value: Any, description: str) -> None:
    if not isinstance(value, dict) or value != EXPECTED_SUPERVISION_AUDIT:
        raise RuntimeError(f"{description} supervision audit is not strictly label-free")


def validate_visual_feature_product(
    helper: ModuleType,
    root: Path,
    cohort_seeds: tuple[int, ...],
    full_seeds: tuple[int, ...],
    product_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Strict visual-v2 envelope validation, embedded to avoid an unpinned import."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"visual product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    summary = read_json(root / "summary.json")
    visual_protocol = read_json(root / "protocol_snapshot.json")
    provenance = read_json(root / "provenance.json")
    inventory = read_json(root / "source_inventory.json")
    manifest_id = require_identity(manifest, f"{product_name} manifest")
    members = helper.validate_manifest_members(root, manifest)
    _require_visual_exact_file_set(root, members)
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
        or manifest.get("provenance_sha256") != members["provenance.json"].get("sha256")
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest_id
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("summary_file_sha256") != sha256_file(root / "summary.json")
        or completion.get("payload_sha256")
        != _canonical_sha256_without(completion, "payload_sha256")
    ):
        raise RuntimeError(f"{product_name} manifest/completion envelope is invalid")

    _require_clean_supervision_audit(
        visual_protocol.get("supervision_policy"), f"{product_name} protocol"
    )
    _require_clean_supervision_audit(
        provenance.get("supervision_audit"), f"{product_name} provenance"
    )
    if (
        visual_protocol.get("status") != "LABEL_FREE_OBSERVATION_ONLY"
        or visual_protocol.get("experiment") != EXPECTED_VISUAL_EXPERIMENT
        or visual_protocol.get("decode", {}).get("saved_images") is not False
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
    catalog["preterminal_actionable"] = helper._as_bool(
        catalog["preterminal_actionable"], f"{product_name} actionable flag"
    )
    catalog["uses_realized_innovation"] = helper._as_bool(
        catalog["uses_realized_innovation"], f"{product_name} innovation flag"
    )
    catalog["latest_required_sampling_step"] = pd.to_numeric(
        catalog["latest_required_sampling_step"], errors="raise"
    ).astype(int)
    catalog["latest_required_internal_timestep"] = pd.to_numeric(
        catalog["latest_required_internal_timestep"], errors="raise"
    ).astype(int)
    features = catalog["feature"].tolist()
    expected_steps = ",".join(map(str, BLUR_STEPS))
    expected_internal = ",".join(map(str, BLUR_INTERNAL_TIMESTEPS))
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
        or not catalog["checkpoint_sampling_steps"].eq(expected_steps).all()
        or not catalog["checkpoint_internal_timesteps"].eq(expected_internal).all()
    ):
        raise RuntimeError(f"{product_name} visual feature/timing contract changed")
    if (
        tuple(frame.columns[: len(VISUAL_IDENTIFIER_COLUMNS)])
        != VISUAL_IDENTIFIER_COLUMNS
        or tuple(frame.columns[len(VISUAL_IDENTIFIER_COLUMNS) :]) != tuple(features)
        or any(
            name in frame.columns for name in ("label", "primary_label", "raw_consensus_label")
        )
        or frame.duplicated(list(KEYS)).any()
    ):
        raise RuntimeError(f"{product_name} sample table schema/axis is invalid")
    full_axis = {(seed, class_id) for seed in full_seeds for class_id in CLASSES}
    if set(map(tuple, frame[list(KEYS)].to_numpy())) != full_axis or len(frame) != len(
        full_axis
    ):
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
        or summary.get("selected_sampling_steps") != list(BLUR_STEPS)
        or summary.get("selected_internal_timesteps")
        != list(BLUR_INTERNAL_TIMESTEPS)
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
            _require_sha(run.get(field), f"{product_name} trace binding {field}")
        identities.append(run["identity_sha256"])
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"{product_name} repeats trace identities")
    source_bindings = inventory.get("input_label_free_source_inventories")
    if not isinstance(source_bindings, list) or len(source_bindings) != 1:
        raise RuntimeError(f"{product_name} source analysis binding changed")
    binding = source_bindings[0]
    if not isinstance(binding, dict):
        raise RuntimeError(f"{product_name} source binding is not an object")
    source_inventory_path = Path(str(binding.get("path", ""))).expanduser().resolve()
    forbidden = ("label_lock", "consensus", "review", "adjudicat")
    if any(token in str(source_inventory_path).lower() for token in forbidden):
        raise RuntimeError(f"{product_name} source path looks supervised")
    upstream_manifest = source_inventory_path.parent / "manifest.json"
    upstream_completion = source_inventory_path.parent / "completion.json"
    if (
        not source_inventory_path.is_file()
        or source_inventory_path.is_symlink()
        or not upstream_manifest.is_file()
        or upstream_manifest.is_symlink()
        or not upstream_completion.is_file()
        or upstream_completion.is_symlink()
        or binding.get("sha256") != sha256_file(source_inventory_path)
        or binding.get("manifest_sha256") != sha256_file(upstream_manifest)
        or binding.get("completion_sha256") != sha256_file(upstream_completion)
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
            array = np.asarray(archive[name])
            if not np.isfinite(array).all() or _array_inventory_record(array) != time_records[name]:
                raise RuntimeError(f"{product_name} time-series array changed: {name}")

    trace_identity_ordered_sha256 = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cohort[list(KEYS) + features], catalog, {
        "manifest_identity_sha256": manifest_id,
        "manifest_file_sha256": sha256_file(manifest_path),
        "sample_features_filename": "sample_features.csv",
        "sample_features_sha256": sha256_file(root / "sample_features.csv"),
        "feature_catalog_sha256": sha256_file(root / "feature_catalog.csv"),
        "analysis_source_sha256": manifest.get("analysis_source_sha256"),
        "source_inventory_sha256": manifest.get("source_inventory_sha256"),
        "scientific_fingerprint_sha256": inventory.get("scientific_fingerprint_sha256"),
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


def validate_visual_protocol_formula(root: Path, expected_formula: str) -> str:
    path = root / "protocol_snapshot.json"
    protocol = read_json(path)
    try:
        observed = protocol["pixel_features"]["local_blur"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("visual protocol lacks the local-blur formula") from error
    if observed != expected_formula:
        raise RuntimeError("visual local-blur formula changed")
    return sha256_file(path)


def raw_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def validate_time_series_product(
    root: Path,
    product_key: str,
    selected_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a sealed NPZ and return only the fixed selected track and axes."""

    if product_key not in EXPECTED_TIME_SERIES:
        raise RuntimeError(f"unknown time-series product: {product_key}")
    expected = EXPECTED_TIME_SERIES[product_key]
    root = root.expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    members = {
        item.get("name"): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    member = members.get("time_series.npz")
    path = root / "time_series.npz"
    if (
        member is None
        or member.get("sha256") != expected["npz_sha256"]
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected["npz_sha256"]
    ):
        raise RuntimeError(f"sealed time_series.npz changed: {product_key}")
    loaded: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name, array_spec in expected["arrays"].items():
            if name not in archive.files:
                raise RuntimeError(f"time-series array missing: {product_key}/{name}")
            array = np.asarray(archive[name])
            if (
                list(array.shape) != array_spec["shape"]
                or str(array.dtype) != array_spec["dtype"]
                or raw_array_sha256(array) != array_spec["raw_sha256"]
            ):
                raise RuntimeError(f"time-series array changed: {product_key}/{name}")
            loaded[name] = array.copy()
    seeds = loaded["global_seed"].astype(int, copy=False)
    classes = loaded["class_id"].astype(int, copy=False)
    if len(seeds) != expected["row_count"] or len(classes) != expected["row_count"]:
        raise RuntimeError(f"time-series row axis changed: {product_key}")
    full_axis = pd.DataFrame({"global_seed": seeds, "class_id": classes})
    if full_axis.duplicated(list(KEYS)).any():
        raise RuntimeError(f"time-series axis has duplicate keys: {product_key}")
    if product_key.startswith("discovery_"):
        expected_full_seeds = tuple(range(30, 130))
    else:
        expected_full_seeds = EXPANSION_SEEDS
    expected_full_axis = {
        (seed, class_id) for seed in expected_full_seeds for class_id in CLASSES
    }
    if set(map(tuple, full_axis[list(KEYS)].to_numpy())) != expected_full_axis:
        raise RuntimeError(f"time-series full key axis changed: {product_key}")
    if product_key.endswith("_primary"):
        if (
            loaded["sampling_step_250"].astype(int).tolist() != list(range(250))
            or loaded["internal_timestep_250"].astype(int).tolist()
            != list(range(249, -1, -1))
        ):
            raise RuntimeError(f"primary time axis changed: {product_key}")
        values = loaded["pred_xstart_alpha_compensated_gradient_energy_c3"]
        value_name = "pred_xstart_alpha_compensated_gradient_energy_c3"
        sampling_steps = loaded["sampling_step_250"].astype(int)
        internal_timesteps = loaded["internal_timestep_250"].astype(int)
    else:
        if (
            loaded["selected_sampling_step"].astype(int).tolist() != list(BLUR_STEPS)
            or loaded["selected_internal_timestep"].astype(int).tolist()
            != list(BLUR_INTERNAL_TIMESTEPS)
        ):
            raise RuntimeError(f"visual time axis changed: {product_key}")
        values = loaded["decoded_local_blur_severity"]
        value_name = "decoded_local_blur_severity"
        sampling_steps = loaded["selected_sampling_step"].astype(int)
        internal_timesteps = loaded["selected_internal_timestep"].astype(int)
    if not np.isfinite(values).all():
        raise RuntimeError(f"time-series target array is non-finite: {product_key}")
    selected_mask = np.isin(seeds, np.asarray(selected_seeds, dtype=int))
    selected_axis = full_axis.loc[selected_mask].copy()
    selected_axis["_source_row"] = np.flatnonzero(selected_mask)
    selected_axis = selected_axis.sort_values(list(KEYS)).reset_index(drop=True)
    expected_selected_axis = {
        (seed, class_id) for seed in selected_seeds for class_id in CLASSES
    }
    if set(map(tuple, selected_axis[list(KEYS)].to_numpy())) != expected_selected_axis:
        raise RuntimeError(f"time-series selected key axis changed: {product_key}")
    selected_values = values[selected_axis["_source_row"].to_numpy(dtype=int)]
    return {
        "keys": selected_axis[list(KEYS)].reset_index(drop=True),
        "values": selected_values,
        "sampling_steps": sampling_steps,
        "internal_timesteps": internal_timesteps,
    }, {
        "time_series_npz_sha256": expected["npz_sha256"],
        "target_array_name": value_name,
        "target_array_raw_sha256": expected["arrays"][value_name]["raw_sha256"],
        "selected_row_count": len(selected_axis),
        "sampling_step_count": len(sampling_steps),
        "axes_validated_by_raw_sha256": True,
    }


def precheck_label_advertisement(
    root: Path, expected_consensus: str, description: str
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    manifest_id = require_identity(manifest, description + " label manifest")
    if manifest.get("consensus_identity_sha256") != expected_consensus:
        raise RuntimeError(f"{description} label manifest advertises another consensus")
    return {
        "manifest_identity_sha256": manifest_id,
        "consensus_identity_sha256": expected_consensus,
    }


def load_and_validate_inputs(
    protocol: dict[str, Any], runtime: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    helper_path = Path(runtime["helper_path"])
    helper = load_pinned_helper(helper_path)

    # First validate both sealed aggregate screens.  No endpoint or row-level
    # label/score is read by this step.
    screen_lineage: dict[str, Any] = {}
    for candidate_name in CANDIDATE_ORDER:
        screen_key = CANDIDATES[candidate_name]["screen_key"]
        screen_spec = protocol["exploratory_screens"][screen_key]
        screen_lineage[screen_key] = validate_screen(
            helper,
            Path(screen_spec["root"]),
            screen_spec["manifest_identity_sha256"],
            candidate_name,
            screen_spec["candidate_table_member"],
        )

    # Precheck every feature manifest identity before reading any feature table.
    for name, item in protocol["feature_products"].items():
        precheck_manifest_identity(
            Path(item["root"]), item["manifest_identity_sha256"], name
        )

    product_frames: dict[str, pd.DataFrame] = {}
    product_catalogs: dict[str, pd.DataFrame] = {}
    product_lineage: dict[str, Any] = {}
    for cohort_name, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("expansion", EXPANSION_SEEDS),
    ):
        for product_name in ("primary", "visual"):
            key = f"{cohort_name}_{product_name}"
            item = protocol["feature_products"][key]
            if product_name == "primary":
                frame, catalog, lineage = helper.validate_feature_product(
                    Path(item["root"]),
                    seeds,
                    key,
                    require_supervision_audit=False,
                )
            else:
                full_seeds = (
                    tuple(range(30, 130))
                    if cohort_name == "discovery"
                    else EXPANSION_SEEDS
                )
                frame, catalog, lineage = validate_visual_feature_product(
                    helper,
                    Path(item["root"]),
                    seeds,
                    full_seeds,
                    key,
                )
            if lineage["manifest_identity_sha256"] != item["manifest_identity_sha256"]:
                raise RuntimeError(f"validated feature manifest changed: {key}")
            summary = read_json(Path(item["root"]) / "summary.json")
            if product_name == "visual" and (
                summary.get("labels_read_or_emitted") is not False
                or summary.get("decoded_images_saved") is not False
            ):
                raise RuntimeError(f"visual feature product is not label/image-free: {key}")
            product_frames[key] = frame
            product_catalogs[key] = catalog
            product_lineage[key] = lineage

    helper.validate_catalog_replication(
        product_catalogs["discovery_primary"],
        product_catalogs["expansion_primary"],
        "primary",
    )
    helper.validate_catalog_replication(
        product_catalogs["discovery_visual"],
        product_catalogs["expansion_visual"],
        "predxstart visual",
    )
    for key in ("discovery_primary", "expansion_primary"):
        validate_catalog_candidate(product_catalogs[key], C3, key)
    for key in ("discovery_visual", "expansion_visual"):
        validate_catalog_candidate(product_catalogs[key], BLUR, key)
        product_lineage[key]["visual_protocol_snapshot_sha256"] = (
            validate_visual_protocol_formula(
                Path(protocol["feature_products"][key]["root"]),
                CANDIDATES[BLUR]["protocol_formula"],
            )
        )
    if (
        product_lineage["discovery_primary"]["feature_catalog_sha256"]
        != product_lineage["expansion_primary"]["feature_catalog_sha256"]
        or product_lineage["discovery_visual"]["feature_catalog_sha256"]
        != product_lineage["expansion_visual"]["feature_catalog_sha256"]
        or product_lineage["discovery_primary"]["trace_identity_ordered_sha256"]
        != product_lineage["discovery_visual"]["trace_identity_ordered_sha256"]
        or product_lineage["expansion_primary"]["trace_identity_ordered_sha256"]
        != product_lineage["expansion_visual"]["trace_identity_ordered_sha256"]
    ):
        raise RuntimeError("candidate products do not share the frozen cohort traces/catalogs")

    time_series_products: dict[str, dict[str, Any]] = {}
    for cohort_name, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("expansion", EXPANSION_SEEDS),
    ):
        for product_name in ("primary", "visual"):
            key = f"{cohort_name}_{product_name}"
            time_series_products[key], time_lineage = validate_time_series_product(
                Path(protocol["feature_products"][key]["root"]), key, seeds
            )
            product_lineage[key].update(time_lineage)

    cohort_features: dict[str, pd.DataFrame] = {}
    cohort_time_series: dict[str, dict[str, Any]] = {}
    for cohort_name in ("discovery", "expansion"):
        primary_feature = CANDIDATES[C3]["feature"]
        visual_feature = CANDIDATES[BLUR]["feature"]
        left = product_frames[f"{cohort_name}_primary"][list(KEYS) + [primary_feature]]
        right = product_frames[f"{cohort_name}_visual"][list(KEYS) + [visual_feature]]
        merged = left.merge(right, on=list(KEYS), how="inner", validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise RuntimeError(f"candidate feature axes differ: {cohort_name}")
        merged = merged.sort_values(list(KEYS)).reset_index(drop=True)
        primary_series = time_series_products[f"{cohort_name}_primary"]
        visual_series = time_series_products[f"{cohort_name}_visual"]
        if (
            not merged[list(KEYS)].equals(primary_series["keys"])
            or not merged[list(KEYS)].equals(visual_series["keys"])
        ):
            raise RuntimeError(f"feature/time-series row axes differ: {cohort_name}")
        c3_series = primary_series["values"]
        blur_series = visual_series["values"]
        replay_c3 = np.maximum(
            0.0, np.max(np.diff(c3_series[:, 100:150], axis=1), axis=1)
        )
        replay_blur = np.mean(blur_series, axis=1)
        if not np.allclose(
            replay_c3,
            merged[primary_feature].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"c3 scalar feature does not replay from NPZ: {cohort_name}")
        if not np.allclose(
            replay_blur,
            merged[visual_feature].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"blur scalar feature does not replay from NPZ: {cohort_name}")
        cohort_features[cohort_name] = merged
        cohort_time_series[cohort_name] = {
            BLUR: visual_series,
            C3: primary_series,
        }

    # Only now inspect the locked labels.  The advertised consensus identities
    # were checked before the helper is allowed to parse consensus rows.
    label_lineage: dict[str, Any] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    advertised_labels: dict[str, dict[str, Any]] = {}
    expected_label_status = {
        "discovery": "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
        "expansion": "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN",
    }
    for cohort_name in ("discovery", "expansion"):
        item = protocol["cohorts"][cohort_name]
        advertised_labels[cohort_name] = precheck_label_advertisement(
            Path(item["label_lock_path"]),
            item["label_consensus_identity_sha256"],
            cohort_name,
        )
    for cohort_name, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("expansion", EXPANSION_SEEDS),
    ):
        item = protocol["cohorts"][cohort_name]
        labels, validated = helper.validate_label_lock(
            Path(item["label_lock_path"]), seeds, expected_label_status[cohort_name]
        )
        if (
            validated["consensus_identity_sha256"]
            != item["label_consensus_identity_sha256"]
            or validated["manifest_identity_sha256"]
            != advertised_labels[cohort_name]["manifest_identity_sha256"]
        ):
            raise RuntimeError(f"validated label lock changed: {cohort_name}")
        label_frames[cohort_name] = labels
        label_lineage[cohort_name] = validated

    joined: dict[str, pd.DataFrame] = {}
    for cohort_name in ("discovery", "expansion"):
        frame = cohort_features[cohort_name].merge(
            label_frames[cohort_name], on=list(KEYS), how="inner", validate="one_to_one"
        )
        if len(frame) != len(label_frames[cohort_name]):
            raise RuntimeError(f"feature/label axes differ: {cohort_name}")
        for candidate_name, spec in CANDIDATES.items():
            raw = frame[spec["feature"]].to_numpy(dtype=np.float64)
            if not np.isfinite(raw).all():
                raise RuntimeError(f"non-finite candidate values: {candidate_name}")
            frame[candidate_name + "__raw"] = raw
            frame[candidate_name + "__risk"] = raw if spec["direction"] == "bad_high" else -raw
        joined[cohort_name] = frame.sort_values(list(KEYS)).reset_index(drop=True)
        if not joined[cohort_name][list(KEYS)].equals(
            cohort_time_series[cohort_name][BLUR]["keys"]
        ):
            raise RuntimeError(f"joined labels changed time-series row order: {cohort_name}")

    return joined["discovery"], joined["expansion"], {
        "screens": screen_lineage,
        "feature_products": product_lineage,
        "label_locks": label_lineage,
    }, cohort_time_series


def auc_bad_high(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    bad = np.asarray(scores[labels == BAD], dtype=np.float64)
    good = np.sort(np.asarray(scores[labels == GOOD], dtype=np.float64))
    pairs = len(bad) * len(good)
    if pairs == 0:
        return float("nan"), 0
    left = np.searchsorted(good, bad, side="left")
    right = np.searchsorted(good, bad, side="right")
    wins = float(np.sum(left + 0.5 * (right - left)))
    return wins / pairs, pairs


def class_matched_auc(
    risk: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    class_rows: list[dict[str, Any]] = []
    for class_id in CLASSES:
        mask = classes == class_id
        y = labels[mask]
        auc, pairs = auc_bad_high(risk[mask], y)
        n_bad = int(np.sum(y == BAD))
        n_good = int(np.sum(y == GOOD))
        n_mild = int(np.sum(y == MILD))
        if pairs == 0 or not math.isfinite(auc):
            raise RuntimeError(f"class {class_id} lacks bad-good pairs")
        numerator += auc * pairs
        denominator += pairs
        class_rows.append(
            {
                "class_id": class_id,
                "auc": float(auc),
                "bad_count": n_bad,
                "good_count": n_good,
                "mild_excluded_count": n_mild,
                "pair_count": pairs,
            }
        )
    return {
        "pair_weighted_auc": float(numerator / denominator),
        "macro_auc": float(np.mean([row["auc"] for row in class_rows])),
        "pair_count": int(denominator),
        "per_class": class_rows,
    }


def _seed_class_cubes(
    expansion: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    expected_axis = {(seed, class_id) for seed in EXPANSION_SEEDS for class_id in CLASSES}
    if (
        expansion.duplicated(list(KEYS)).any()
        or set(map(tuple, expansion[list(KEYS)].to_numpy())) != expected_axis
    ):
        raise RuntimeError("expansion is not the exact seed-by-class block axis")
    ordered = expansion.sort_values(list(KEYS)).reset_index(drop=True)
    seeds = sorted(ordered["global_seed"].astype(int).unique().tolist())
    if seeds != list(EXPANSION_SEEDS):
        raise RuntimeError("expansion seed axis changed")
    label_code = {GOOD: 0, BAD: 1, MILD: 2}
    labels = np.empty((len(seeds), len(CLASSES)), dtype=np.int8)
    risks = np.empty((len(seeds), len(CLASSES), len(CANDIDATE_ORDER)), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        block = ordered[ordered["global_seed"] == seed].sort_values("class_id")
        if block["class_id"].astype(int).tolist() != list(CLASSES):
            raise RuntimeError("a seed does not contain the exact three-class block")
        labels[seed_index] = [label_code[value] for value in block["label"]]
        for candidate_index, candidate_name in enumerate(CANDIDATE_ORDER):
            risks[seed_index, :, candidate_index] = block[
                candidate_name + "__risk"
            ].to_numpy(dtype=np.float64)
    return labels, risks, seeds


def _uniform_permutation_batch(
    rng: np.random.Generator, batch_size: int, item_count: int
) -> np.ndarray:
    """Draw exact independent uniform permutations; one permutation per row."""

    result = np.empty((batch_size, item_count), dtype=np.int16)
    for row in range(batch_size):
        result[row] = rng.permutation(item_count)
    return result


def block_permutation_pvalues(
    expansion: pd.DataFrame,
    observed_auc: np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int = 1_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Seed-vector block permutation preserving all three label states.

    A single seed permutation is shared across classes and candidates in every
    draw.  Therefore each intact (class207, class602, class795) label vector,
    including every mild status, moves together to another seed.  Scores remain
    fixed.  Mild-assigned positions are excluded only when the AUC numerator is
    evaluated.
    """

    if draws < 1 or batch_size < 1:
        raise ValueError("draws and batch_size must be positive")
    labels, risks, seeds = _seed_class_cubes(expansion)
    if observed_auc.shape != (len(CANDIDATE_ORDER),):
        raise ValueError("observed_auc has the wrong shape")
    n_seed = len(seeds)
    pair_denominator = 0
    class_counts: list[dict[str, int]] = []
    win_matrices: list[list[np.ndarray]] = []
    for candidate_index in range(len(CANDIDATE_ORDER)):
        candidate_wins: list[np.ndarray] = []
        for class_index, _class_id in enumerate(CLASSES):
            score = risks[:, class_index, candidate_index]
            diff = score[:, None] - score[None, :]
            candidate_wins.append((diff > 0).astype(np.float64) + 0.5 * (diff == 0))
        win_matrices.append(candidate_wins)
    for class_index, class_id in enumerate(CLASSES):
        codes = labels[:, class_index]
        n_bad = int(np.sum(codes == 1))
        n_good = int(np.sum(codes == 0))
        n_mild = int(np.sum(codes == 2))
        if n_bad == 0 or n_good == 0:
            raise RuntimeError(f"class {class_id} lacks a permutable bad-good pair")
        pair_denominator += n_bad * n_good
        class_counts.append(
            {"class_id": class_id, "bad": n_bad, "good": n_good, "mild": n_mild}
        )
    threshold = np.asarray(observed_auc, dtype=np.float64) * pair_denominator
    exceedances = np.zeros(len(CANDIDATE_ORDER), dtype=np.int64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        permutations = _uniform_permutation_batch(rng, size, n_seed)
        assigned = labels[permutations]
        numerators = np.zeros((size, len(CANDIDATE_ORDER)), dtype=np.float64)
        for class_index, _class_id in enumerate(CLASSES):
            assigned_class = assigned[:, :, class_index]
            bad_count = int(np.sum(labels[:, class_index] == 1))
            bad_positions = np.argwhere(assigned_class == 1)[:, 1].reshape(size, bad_count)
            good_mask = assigned_class == 0
            for candidate_index in range(len(CANDIDATE_ORDER)):
                win_rows = win_matrices[candidate_index][class_index][bad_positions]
                numerators[:, candidate_index] += np.sum(
                    win_rows * good_mask[:, None, :], axis=(1, 2)
                )
        exceedances += np.sum(
            numerators >= threshold[None, :] - 1e-12, axis=0
        ).astype(np.int64)
        completed += size
    pvalues = (exceedances + 1.0) / (draws + 1.0)
    return pvalues, {
        "draws": draws,
        "rng": "numpy.default_rng(PCG64)",
        "rng_seed": seed,
        "seed_block_count": n_seed,
        "class_counts_including_mild": class_counts,
        "pair_denominator_after_mild_exclusion": pair_denominator,
        "exceedances": {
            candidate: int(exceedances[index])
            for index, candidate in enumerate(CANDIDATE_ORDER)
        },
    }


def holm_adjust_two(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.shape != (2,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm adjustment requires exactly two valid p-values")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(ranked * np.array([2.0, 1.0]))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def conformal_risk_pvalues(calibration_risk: np.ndarray, test_risk: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_risk, dtype=np.float64)
    test = np.asarray(test_risk, dtype=np.float64)
    if calibration.ndim != 1 or len(calibration) == 0 or not np.isfinite(calibration).all():
        raise ValueError("calibration risk must be a non-empty finite vector")
    if test.ndim != 1 or not np.isfinite(test).all():
        raise ValueError("test risk must be a finite vector")
    descending = np.sort(calibration)
    # Number of calibration values >= each test value, with ties included.
    count_ge = len(descending) - np.searchsorted(descending, test, side="left")
    return (1.0 + count_ge.astype(np.float64)) / (len(descending) + 1.0)


def candidate_auc_rows(
    discovery: pd.DataFrame, expansion: pd.DataFrame
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    observed_expansion: list[float] = []
    for candidate_name in CANDIDATE_ORDER:
        spec = CANDIDATES[candidate_name]
        result: dict[str, Any] = {
            "candidate": candidate_name,
            "role": spec["role"],
            "feature": spec["feature"],
            "direction": spec["direction"],
            "risk_transform": spec["risk_transform"],
            "latest_required_sampling_step": 149,
        }
        for cohort_name, frame in (("discovery", discovery), ("expansion", expansion)):
            summary = class_matched_auc(
                frame[candidate_name + "__risk"].to_numpy(dtype=np.float64),
                frame["label"].to_numpy(),
                frame["class_id"].to_numpy(dtype=int),
            )
            result[cohort_name] = summary
            if cohort_name == "expansion":
                observed_expansion.append(summary["pair_weighted_auc"])
        rows.append(result)
    return rows, np.asarray(observed_expansion, dtype=np.float64)


def conformal_operating_points(
    discovery: pd.DataFrame,
    expansion: pd.DataFrame,
    alphas: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    operating_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    expansion_labels = expansion["label"].to_numpy()
    for candidate_name in CANDIDATE_ORDER:
        pvalues = np.empty(len(expansion), dtype=np.float64)
        for class_id in CLASSES:
            calibration = discovery[
                discovery["class_id"].eq(class_id) & discovery["label"].eq(GOOD)
            ][candidate_name + "__risk"].to_numpy(dtype=np.float64)
            mask = expansion["class_id"].eq(class_id).to_numpy()
            pvalues[mask] = conformal_risk_pvalues(
                calibration,
                expansion.loc[mask, candidate_name + "__risk"].to_numpy(dtype=np.float64),
            )
            calibration_rows.append(
                {
                    "candidate": candidate_name,
                    "class_id": class_id,
                    "clean_good_calibration_count": len(calibration),
                    "minimum_attainable_risk_p": 1.0 / (len(calibration) + 1.0),
                    "calibration_source": "discovery_clean_good_only",
                }
            )
        for alpha in alphas:
            alert = pvalues <= alpha
            bad_mask = expansion_labels == BAD
            good_mask = expansion_labels == GOOD
            mild_mask = expansion_labels == MILD
            binary_mask = bad_mask | good_mask
            operating_rows.append(
                {
                    "candidate": candidate_name,
                    "alpha": alpha,
                    "alpha_semantics": "conformal_style_risk_not_ville",
                    "bad_alert_count": int(np.sum(alert & bad_mask)),
                    "bad_count": int(np.sum(bad_mask)),
                    "tpr": float(np.mean(alert[bad_mask])),
                    "good_alert_count": int(np.sum(alert & good_mask)),
                    "good_count": int(np.sum(good_mask)),
                    "fpr": float(np.mean(alert[good_mask])),
                    "mild_alert_count": int(np.sum(alert & mild_mask)),
                    "mild_count": int(np.sum(mild_mask)),
                    "mild_intervention_rate": float(np.mean(alert[mild_mask])),
                    "binary_alert_count": int(np.sum(alert & binary_mask)),
                    "binary_count": int(np.sum(binary_mask)),
                    "binary_intervention_rate": float(np.mean(alert[binary_mask])),
                    "all_alert_count": int(np.sum(alert)),
                    "all_locked_count": len(alert),
                    "all_locked_intervention_rate": float(np.mean(alert)),
                }
            )
    return pd.DataFrame(operating_rows), pd.DataFrame(calibration_rows)


def _descriptive_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(np.min(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def group_score_summary(
    discovery: pd.DataFrame, expansion: pd.DataFrame, minimum_group_size: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort_name, frame in (("discovery", discovery), ("expansion", expansion)):
        group_defs: list[tuple[str, int | None, str, pd.Series]] = []
        for label in LABELS:
            group_defs.append(("pooled_class", None, label, frame["label"].eq(label)))
        for class_id in CLASSES:
            for label in LABELS:
                group_defs.append(
                    (
                        "within_class",
                        class_id,
                        label,
                        frame["class_id"].eq(class_id) & frame["label"].eq(label),
                    )
                )
        for scope, class_id, label, mask in group_defs:
            for candidate_name in CANDIDATE_ORDER:
                count = int(mask.sum())
                row: dict[str, Any] = {
                    "cohort": cohort_name,
                    "scope": scope,
                    "class_id": class_id,
                    "label_group": label,
                    "candidate": candidate_name,
                    "count": count,
                    "statistics_suppressed": count < minimum_group_size,
                }
                for representation in ("raw", "risk"):
                    names = (
                        "mean",
                        "std",
                        "minimum",
                        "q10",
                        "q25",
                        "median",
                        "q75",
                        "q90",
                        "maximum",
                    )
                    if count >= minimum_group_size:
                        stats = _descriptive_stats(
                            frame.loc[mask, candidate_name + "__" + representation].to_numpy(
                                dtype=np.float64
                            )
                        )
                        for name in names:
                            row[representation + "_" + name] = stats[name]
                    else:
                        for name in names:
                            row[representation + "_" + name] = None
                rows.append(row)
    return pd.DataFrame(rows)


def trajectory_descriptive_summary(
    discovery: pd.DataFrame,
    expansion: pd.DataFrame,
    time_series: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Fixed-grid trajectory descriptions that do not feed any inference."""

    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        cohort_name: str,
        frame: pd.DataFrame,
        candidate_name: str,
        value_kind: str,
        raw_values: np.ndarray,
        sampling_step: int,
        internal_timestep: int,
        from_sampling_step: int | None,
    ) -> None:
        spec = CANDIDATES[candidate_name]
        raw = np.asarray(raw_values, dtype=np.float64)
        if raw.shape != (len(frame),) or not np.isfinite(raw).all():
            raise RuntimeError("invalid fixed-grid trajectory vector")
        oriented_risk = raw if spec["direction"] == "bad_high" else -raw
        auc = class_matched_auc(
            oriented_risk,
            frame["label"].to_numpy(),
            frame["class_id"].to_numpy(dtype=int),
        )
        row: dict[str, Any] = {
            "cohort": cohort_name,
            "candidate": candidate_name,
            "track": spec["track"],
            "value_kind": value_kind,
            "from_sampling_step": from_sampling_step,
            "sampling_step": sampling_step,
            "internal_timestep": internal_timestep,
            "fixed_direction": spec["direction"],
            "class_matched_pair_weighted_auc_in_fixed_direction": auc[
                "pair_weighted_auc"
            ],
            "macro_auc_in_fixed_direction": auc["macro_auc"],
            "bad_good_pair_count": auc["pair_count"],
            "post_selection_descriptive_only": True,
            "used_in_permutation_conformal_or_threshold": False,
        }
        by_class = {int(item["class_id"]): item for item in auc["per_class"]}
        for class_id in CLASSES:
            row[f"auc_class_{class_id}_in_fixed_direction"] = by_class[class_id]["auc"]
        labels = frame["label"].to_numpy()
        for label in LABELS:
            mask = labels == label
            values = raw[mask]
            if len(values) == 0:
                raise RuntimeError(f"trajectory summary label group is empty: {label}")
            row[f"{label}_count"] = len(values)
            row[f"{label}_raw_mean"] = float(np.mean(values))
            row[f"{label}_raw_median"] = float(np.median(values))
        rows.append(row)

    for cohort_name, frame in (("discovery", discovery), ("expansion", expansion)):
        blur = time_series[cohort_name][BLUR]
        if (
            blur["sampling_steps"].astype(int).tolist() != list(BLUR_STEPS)
            or blur["internal_timesteps"].astype(int).tolist()
            != list(BLUR_INTERNAL_TIMESTEPS)
        ):
            raise RuntimeError("blur fixed time grid changed")
        for index, step in enumerate(BLUR_STEPS):
            append_row(
                cohort_name=cohort_name,
                frame=frame,
                candidate_name=BLUR,
                value_kind="level",
                raw_values=blur["values"][:, index],
                sampling_step=step,
                internal_timestep=BLUR_INTERNAL_TIMESTEPS[index],
                from_sampling_step=None,
            )

        c3 = time_series[cohort_name][C3]
        sampling_axis = c3["sampling_steps"].astype(int)
        internal_axis = c3["internal_timesteps"].astype(int)
        if (
            sampling_axis.tolist() != list(range(250))
            or internal_axis.tolist() != list(range(249, -1, -1))
        ):
            raise RuntimeError("c3 fixed time grid changed")
        for step in C3_Q2_STEPS:
            append_row(
                cohort_name=cohort_name,
                frame=frame,
                candidate_name=C3,
                value_kind="raw_G_level",
                raw_values=c3["values"][:, step],
                sampling_step=step,
                internal_timestep=int(internal_axis[step]),
                from_sampling_step=None,
            )
        for step in C3_Q2_STEPS[1:]:
            append_row(
                cohort_name=cohort_name,
                frame=frame,
                candidate_name=C3,
                value_kind="raw_G_adjacent_jump",
                raw_values=c3["values"][:, step] - c3["values"][:, step - 1],
                sampling_step=step,
                internal_timestep=int(internal_axis[step]),
                from_sampling_step=step - 1,
            )
    return pd.DataFrame(rows)


def _payload_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish(
    *,
    output: Path,
    protocol_path: Path,
    source_path: Path,
    helper_path: Path,
    auc_rows: list[dict[str, Any]],
    block_pvalues: np.ndarray,
    holm_pvalues: np.ndarray,
    block_details: dict[str, Any],
    operating_points: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    lineage: dict[str, Any],
    protocol_lineage: dict[str, Any],
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        metric_rows: list[dict[str, Any]] = []
        for index, row in enumerate(auc_rows):
            flat: dict[str, Any] = {
                "candidate": row["candidate"],
                "role": row["role"],
                "feature": row["feature"],
                "direction": row["direction"],
                "risk_transform": row["risk_transform"],
                "latest_required_sampling_step": row["latest_required_sampling_step"],
                "discovery_pair_weighted_auc": row["discovery"]["pair_weighted_auc"],
                "discovery_macro_auc": row["discovery"]["macro_auc"],
                "discovery_pair_count": row["discovery"]["pair_count"],
                "expansion_pair_weighted_auc": row["expansion"]["pair_weighted_auc"],
                "expansion_macro_auc": row["expansion"]["macro_auc"],
                "expansion_pair_count": row["expansion"]["pair_count"],
                "expansion_seed_vector_block_permutation_p_one_sided": float(
                    block_pvalues[index]
                ),
                "holm_p_across_two_frozen_candidates": float(holm_pvalues[index]),
                "post_selection_not_confirmatory": True,
            }
            for cohort_name in ("discovery", "expansion"):
                by_class = {
                    int(item["class_id"]): item for item in row[cohort_name]["per_class"]
                }
                for class_id in CLASSES:
                    item = by_class[class_id]
                    flat[f"{cohort_name}_auc_class_{class_id}"] = item["auc"]
                    flat[f"{cohort_name}_bad_count_class_{class_id}"] = item["bad_count"]
                    flat[f"{cohort_name}_good_count_class_{class_id}"] = item["good_count"]
                    flat[f"{cohort_name}_mild_excluded_count_class_{class_id}"] = item[
                        "mild_excluded_count"
                    ]
            metric_rows.append(flat)
        pd.DataFrame(metric_rows).to_csv(staging / "candidate_metrics.csv", index=False)
        operating_points.to_csv(staging / "conformal_operating_points.csv", index=False)
        calibration_summary.to_csv(staging / "conformal_calibration_summary.csv", index=False)
        group_summary.to_csv(staging / "group_score_summary.csv", index=False)
        trajectory_summary.to_csv(staging / "trajectory_descriptive_fixed_grid.csv", index=False)
        shutil.copyfile(source_path, staging / "audit_source.py")
        shutil.copyfile(helper_path, staging / "pinned_inventory_helper.py")
        shutil.copyfile(protocol_path, staging / "protocol_snapshot.json")
        methodology = {
            "schema_version": 1,
            "status": "POST_SELECTION_OPERATIONAL_AUDIT_ONLY",
            "candidate_policy": (
                "two candidates, formulas, directions, timing, and roles are frozen; "
                "no combined score is constructed"
            ),
            "auc": (
                "mild excluded; within each class compare every clear_bad risk with "
                "every clean_good risk, count ties as one half, then weight class AUCs "
                "by their bad-good pair counts"
            ),
            "block_permutation": {
                "scheme": (
                    "move each intact three-class label vector, including mild states, "
                    "between global seeds; use the same permutation for both candidates"
                ),
                "alternative": "larger frozen-direction risk AUC than permuted",
                "monte_carlo_p": "(1+exceedances)/(1+draws)",
                "multiple_testing": "Holm across exactly two candidates",
                **block_details,
            },
            "conformal_risk": {
                "calibration": "discovery clean_good only, separately by class",
                "risk_p": "(1 + count(calibration_risk >= test_risk))/(n_calibration+1)",
                "application": "the same discovery calibration is applied to expansion",
                "warning": (
                    "alpha is an upper-tail conformal-style operational risk level, not "
                    "Ville alpha or an anytime-valid trigger budget.  Because these "
                    "features were selected using the discovery cohort, no fresh formal "
                    "conditional-FPR guarantee is claimed."
                ),
            },
            "trajectory_descriptive": {
                "blur": "all nine frozen checkpoints k=69,79,...,149",
                "c3_level": "every q2 raw-G level k=100..149",
                "c3_jump": "every within-q2 adjacent raw-G jump ending k=101..149",
                "statistics": (
                    "fixed-direction class-matched AUC plus pooled-by-label raw mean and median"
                ),
                "inference_policy": (
                    "descriptive only; no p-value, threshold, window selection, feature "
                    "selection, candidate selection, or score combination"
                ),
            },
            "privacy": (
                "aggregate outputs only; small group descriptive statistics are suppressed; "
                "no row key, seed, label, score, rank, image, endpoint, or trace path is emitted"
            ),
            "claim_limit": (
                "both cohorts participated in exploratory candidate naming; these p-values "
                "and operating points do not constitute an independent confirmation"
            ),
        }
        write_json(staging / "methodology.json", methodology)
        summary = {
            "schema_version": 1,
            "status": "COMPLETE_DUAL_CANDIDATE_POST_SELECTION_OPERATIONAL_AUDIT",
            "candidate_count": 2,
            "candidate_order": list(CANDIDATE_ORDER),
            "no_combined_score": True,
            "permutation_draws": block_details["draws"],
            "permutation_seed": block_details["rng_seed"],
            "conformal_alphas": list(EXPECTED_ALPHAS),
            "conformal_alpha_is_ville_alpha": False,
            "post_selection_not_confirmatory": True,
            "row_level_payload_emitted": False,
            "images_or_endpoints_read_or_emitted": False,
            "trajectory_descriptive_row_count": len(trajectory_summary),
            "trajectory_descriptive_used_for_inference": False,
        }
        write_json(staging / "summary.json", summary)
        payload_names = [
            "candidate_metrics.csv",
            "conformal_operating_points.csv",
            "conformal_calibration_summary.csv",
            "group_score_summary.csv",
            "trajectory_descriptive_fixed_grid.csv",
            "methodology.json",
            "summary.json",
            "audit_source.py",
            "pinned_inventory_helper.py",
            "protocol_snapshot.json",
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "dit_dual_candidate_operational_audit_v1",
            "post_selection_not_confirmatory": True,
            "aggregate_only": True,
            "row_level_payload_emitted": False,
            "input_lineage": lineage,
            "protocol_lineage": protocol_lineage,
            "files": [_payload_record(staging / name) for name in payload_names],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "audit_source_sha256": sha256_file(staging / "audit_source.py"),
            "pinned_inventory_helper_sha256": sha256_file(
                staging / "pinned_inventory_helper.py"
            ),
            "protocol_snapshot_sha256": sha256_file(staging / "protocol_snapshot.json"),
            "aggregate_only": True,
            "row_level_payload_emitted": False,
        }
        write_json(staging / "completion.json", completion)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def synthetic_self_test() -> None:
    rng = np.random.default_rng(81)
    discovery_records: list[dict[str, Any]] = []
    expansion_records: list[dict[str, Any]] = []
    # Bad states are aligned across classes at the seed-vector level so a
    # perfect frozen-direction signal has a very small block-permutation p.
    for cohort_name, records, seed_offset in (
        ("discovery", discovery_records, 0),
        ("expansion", expansion_records, 100),
    ):
        for seed_index in range(20):
            for class_id in CLASSES:
                label = BAD if seed_index < 4 else (MILD if seed_index < 6 else GOOD)
                blur_risk = (5.0 if label == BAD else (-1.0 if label == MILD else 0.0))
                blur_risk += rng.normal(0.0, 0.01)
                c3_risk = (4.0 if label == BAD else (-1.0 if label == MILD else 0.0))
                c3_risk += rng.normal(0.0, 0.01)
                records.append(
                    {
                        "global_seed": seed_offset + seed_index,
                        "class_id": class_id,
                        "label": label,
                        BLUR + "__raw": blur_risk,
                        BLUR + "__risk": blur_risk,
                        C3 + "__raw": -c3_risk,
                        C3 + "__risk": c3_risk,
                    }
                )
    discovery = pd.DataFrame(discovery_records)
    expansion = pd.DataFrame(expansion_records)
    # The production permutation helper expects expansion seeds 130..249.
    # Exercise its general math on a production-shaped synthetic copy with
    # 120 blocks (six repetitions of the 20-block construction).
    expanded: list[pd.DataFrame] = []
    for repeat in range(6):
        part = expansion.copy()
        part["global_seed"] = 130 + repeat * 20 + (part["global_seed"] - 100)
        expanded.append(part)
    expansion120 = pd.concat(expanded, ignore_index=True)
    # Add tiny deterministic offsets so repeated copies do not introduce ties.
    expansion120[BLUR + "__risk"] += (
        expansion120["global_seed"].to_numpy(dtype=float) - 130.0
    ) * 1e-7
    expansion120[BLUR + "__raw"] = expansion120[BLUR + "__risk"]
    expansion120[C3 + "__risk"] += (
        expansion120["global_seed"].to_numpy(dtype=float) - 130.0
    ) * 1e-7
    expansion120[C3 + "__raw"] = -expansion120[C3 + "__risk"]
    # Recreate aligned label vectors with the production 120-seed axis.
    expansion120["label"] = np.where(
        ((expansion120["global_seed"] - 130) % 20) < 4,
        BAD,
        np.where(((expansion120["global_seed"] - 130) % 20) < 6, MILD, GOOD),
    )
    auc_rows, observed = candidate_auc_rows(discovery, expansion120)
    assert np.allclose(observed, 1.0)
    pvalues, details = block_permutation_pvalues(
        expansion120, observed, draws=999, seed=17, batch_size=100
    )
    assert np.all(pvalues <= 0.01), pvalues
    assert details["seed_block_count"] == 120
    adjusted = holm_adjust_two(np.array([0.01, 0.04]))
    assert np.allclose(adjusted, [0.02, 0.04])
    calibration = np.array([0.0, 1.0, 2.0, 2.0])
    test = np.array([3.0, 2.0, 1.5, -1.0])
    assert np.allclose(
        conformal_risk_pvalues(calibration, test),
        np.array([0.2, 0.6, 0.6, 1.0]),
    )
    operating, calibration_summary = conformal_operating_points(
        discovery, expansion120, EXPECTED_ALPHAS
    )
    assert len(operating) == len(CANDIDATE_ORDER) * len(EXPECTED_ALPHAS)
    assert len(calibration_summary) == len(CANDIDATE_ORDER) * len(CLASSES)
    groups = group_score_summary(discovery, expansion120, minimum_group_size=5)
    suppressed = groups[groups["count"] < 5]
    stat_columns = [column for column in groups if column.startswith(("raw_", "risk_"))]
    assert suppressed[stat_columns].isna().all().all()
    synthetic_time_series: dict[str, dict[str, Any]] = {}
    for cohort_name, frame in (("discovery", discovery), ("expansion", expansion120)):
        blur_level = frame[BLUR + "__raw"].to_numpy(dtype=np.float64)
        c3_risk = frame[C3 + "__risk"].to_numpy(dtype=np.float64)
        blur_values = np.column_stack(
            [blur_level + 0.001 * index for index in range(len(BLUR_STEPS))]
        )
        c3_values = np.column_stack(
            [(-c3_risk) + 0.0001 * step for step in range(250)]
        )
        synthetic_time_series[cohort_name] = {
            BLUR: {
                "values": blur_values,
                "sampling_steps": np.asarray(BLUR_STEPS),
                "internal_timesteps": np.asarray(BLUR_INTERNAL_TIMESTEPS),
            },
            C3: {
                "values": c3_values,
                "sampling_steps": np.arange(250),
                "internal_timesteps": np.arange(249, -1, -1),
            },
        }
    trajectory = trajectory_descriptive_summary(
        discovery, expansion120, synthetic_time_series
    )
    assert len(trajectory) == 2 * (len(BLUR_STEPS) + 50 + 49)
    assert not any("permutation_p" in column for column in trajectory.columns)

    temporary_parent = Path(tempfile.mkdtemp(prefix="dual-candidate-audit-selftest-"))
    try:
        source = temporary_parent / "source.py"
        helper = temporary_parent / "helper.py"
        protocol = temporary_parent / "protocol.json"
        source.write_text("# synthetic source\n", encoding="utf-8")
        helper.write_text("# synthetic helper\n", encoding="utf-8")
        synthetic_protocol: dict[str, Any] = {
            "schema_version": 1,
            "status": "synthetic",
        }
        synthetic_protocol["identity_sha256"] = canonical_sha256(synthetic_protocol)
        write_json(protocol, synthetic_protocol)
        output = temporary_parent / "published"
        publish(
            output=output,
            protocol_path=protocol,
            source_path=source,
            helper_path=helper,
            auc_rows=auc_rows,
            block_pvalues=pvalues,
            holm_pvalues=holm_adjust_two(pvalues),
            block_details=details,
            operating_points=operating,
            calibration_summary=calibration_summary,
            group_summary=groups,
            trajectory_summary=trajectory,
            lineage={"synthetic": True},
            protocol_lineage={"synthetic": True},
        )
        manifest = read_json(output / "manifest.json")
        completion = read_json(output / "completion.json")
        assert require_identity(manifest, "synthetic manifest") == completion[
            "manifest_identity_sha256"
        ]
        expected_members = {
            "candidate_metrics.csv",
            "conformal_operating_points.csv",
            "conformal_calibration_summary.csv",
            "group_score_summary.csv",
            "trajectory_descriptive_fixed_grid.csv",
            "methodology.json",
            "summary.json",
            "audit_source.py",
            "pinned_inventory_helper.py",
            "protocol_snapshot.json",
        }
        assert {item["name"] for item in manifest["files"]} == expected_members
        for item in manifest["files"]:
            member = output / item["name"]
            assert member.stat().st_size == item["bytes"]
            assert sha256_file(member) == item["sha256"]
        metrics = pd.read_csv(output / "candidate_metrics.csv")
        forbidden_columns = {"global_seed", "sample_key", "row_label", "row_score", "rank"}
        assert not forbidden_columns.intersection(metrics.columns)
        try:
            publish(
                output=output,
                protocol_path=protocol,
                source_path=source,
                helper_path=helper,
                auc_rows=auc_rows,
                block_pvalues=pvalues,
                holm_pvalues=holm_adjust_two(pvalues),
                block_details=details,
                operating_points=operating,
                calibration_summary=calibration_summary,
                group_summary=groups,
                trajectory_summary=trajectory,
                lineage={},
                protocol_lineage={},
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("publisher overwrote an existing result")
    finally:
        shutil.rmtree(temporary_parent)
    print("synthetic self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="canonical frozen protocol; the only real-data configuration input",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        if args.protocol is not None:
            raise ValueError("--self-test cannot be combined with --protocol")
        synthetic_self_test()
        return
    if args.protocol is None:
        raise ValueError("--protocol is required outside --self-test")
    protocol, protocol_lineage = validate_protocol(args.protocol)
    discovery, expansion, lineage, time_series = load_and_validate_inputs(
        protocol, protocol_lineage
    )
    auc_rows, observed_expansion = candidate_auc_rows(discovery, expansion)
    permutation = protocol["statistics"]["block_permutation"]
    block_pvalues, block_details = block_permutation_pvalues(
        expansion,
        observed_expansion,
        draws=permutation["draws"],
        seed=permutation["seed"],
    )
    holm_pvalues = holm_adjust_two(block_pvalues)
    operating, calibration = conformal_operating_points(
        discovery,
        expansion,
        tuple(protocol["statistics"]["conformal_risk"]["alphas"]),
    )
    groups = group_score_summary(
        discovery,
        expansion,
        protocol["statistics"]["group_score_summary"]["minimum_group_size"],
    )
    trajectory = trajectory_descriptive_summary(
        discovery, expansion, time_series
    )
    publish(
        output=Path(protocol_lineage["output"]),
        protocol_path=Path(protocol_lineage["protocol_path"]),
        source_path=Path(protocol_lineage["source_path"]),
        helper_path=Path(protocol_lineage["helper_path"]),
        auc_rows=auc_rows,
        block_pvalues=block_pvalues,
        holm_pvalues=holm_pvalues,
        block_details=block_details,
        operating_points=operating,
        calibration_summary=calibration,
        group_summary=groups,
        trajectory_summary=trajectory,
        lineage=lineage,
        protocol_lineage=protocol_lineage,
    )
    print(f"published aggregate-only operational audit: {protocol_lineage['output']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-unseal bad/good metric discovery on the 64-path DiT pool.

This program is deliberately exploratory.  The source pool's visual labels and
the original likelihood-ratio readout have already been unsealed, so every
feature emitted here is a hypothesis generator, never a confirmatory result or
an intervention trigger.

The analysis keeps three time-availability classes separate:

* ``predictable``: available before the current Gaussian innovation is drawn;
* ``online_causal``: available after the current transition/prediction change;
* ``retrospective``: uses the endpoint or a whole-path reduction.

The central mechanistic screen is based on the model's raw ``pred_xstart``
latent.  It computes per-channel amplitude-normalized Dirichlet energy, raw
amplitude, alpha-compensated gradient energy, temporal instability, and state,
epsilon, learned-variance, innovation, cross-scale, and endpoint controls.  It
also records one explicitly post-hoc two-phase score discovered in this pool.
That score must be frozen and tested on new blind trajectories before it can be
called a detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments/configs/bad_good_metric_discovery_v1.json"
DEFAULT_CONSENSUS = (
    ROOT
    / "experiments/annotations/dit_cross_prefix_pool64_consensus_lock_v1/consensus_locked.json"
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae")
)
DEFAULT_POOL = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/dit_imagenet256_t60_cross_prefix_validation"
)
DEFAULT_MAPPING = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/dit_imagenet256_t60_cross_prefix_blind_review"
    / "class0207_poolv1_blind_b2ab6bf/private_seal/blind_mapping_private.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_discovery"
    / "dit_class0207_legacy64_v7"
)

POSITIVE = "clear_overall_structural_bad"
NEGATIVE = "not_clear_overall_structural_bad"
EXCLUDED = "uncertain"
PHASES = (
    ("q0", 0, 50, "internal t=249..200"),
    ("q1", 50, 100, "internal t=199..150"),
    ("q2", 100, 150, "internal t=149..100"),
    ("q3", 150, 200, "internal t=99..50"),
    ("q4", 200, 250, "internal t=49..0"),
)
EPS = 1e-8


@dataclass(frozen=True)
class SampleBinding:
    global_index: int
    shard_index: int
    local_index: int
    stream_seed: int
    runner_blind_id: str
    public_blind_id: str
    label: str
    png_path: Path


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def require_label_counts(labels: Iterable[str]) -> None:
    values = list(labels)
    observed = {label: values.count(label) for label in (POSITIVE, NEGATIVE, EXCLUDED)}
    expected = {POSITIVE: 4, NEGATIVE: 59, EXCLUDED: 1}
    if observed != expected or len(values) != 64:
        raise RuntimeError(f"legacy label counts changed: {observed} != {expected}")


def validate_locked_label_lineage(consensus_path: Path, mapping_path: Path) -> None:
    consensus_root = consensus_path.parent
    consensus_manifest_path = consensus_root / "manifest.json"
    consensus_completion_path = consensus_root / "completion.json"
    for path in (consensus_manifest_path, consensus_completion_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"consensus lock artifact is missing or indirect: {path}")
    manifest = load_json(consensus_manifest_path)
    completion = load_json(consensus_completion_path)
    consensus_document = load_json(consensus_path)
    consensus_hash = sha256_file(consensus_path)
    manifest_hash = sha256_file(consensus_manifest_path)
    if (
        manifest.get("status") != "LOCKED_COMPLETE_CONSENSUS_BEFORE_EVIDENCE_UNSEAL"
        or completion.get("complete") is not True
        or manifest.get("consensus", {}).get("file_sha256") != consensus_hash
        or completion.get("consensus_file_sha256") != consensus_hash
        or completion.get("manifest_file_sha256") != manifest_hash
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
    ):
        raise RuntimeError("consensus manifest/completion/hash binding is invalid")
    for key, filename in (("review_A", "review_A_locked.json"), ("review_B", "review_B_locked.json")):
        review_path = consensus_root / filename
        if (
            not review_path.is_file()
            or review_path.is_symlink()
            or manifest.get(key, {}).get("file_sha256") != sha256_file(review_path)
        ):
            raise RuntimeError(f"locked consensus review binding is invalid: {filename}")

    mapping_root = mapping_path.parent
    mapping_completion_path = mapping_root / "completion.json"
    commitment_path = mapping_root / "blind_mapping_commitment_frozen.json"
    for path in (mapping_completion_path, commitment_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"mapping lock artifact is missing or indirect: {path}")
    mapping = load_json(mapping_path)
    mapping_completion = load_json(mapping_completion_path)
    if (
        mapping_completion.get("complete") is not True
        or mapping_completion.get("entry_count") != 64
        or mapping_completion.get("mapping_file_sha256") != sha256_file(mapping_path)
        or mapping_completion.get("mapping_identity_sha256") != mapping.get("identity_sha256")
        or mapping_completion.get("mapping_commitment_file_sha256")
        != sha256_file(commitment_path)
        or mapping_completion.get("mapping_commitment_identity_sha256")
        != mapping.get("blind_mapping_commitment_identity_sha256")
    ):
        raise RuntimeError("private blind mapping completion/hash binding is invalid")
    public_manifest_identities = {
        manifest.get("blind_pack_manifest_identity_sha256"),
        consensus_document.get("blind_pack_manifest_identity_sha256"),
        mapping.get("public_manifest_identity_sha256"),
        mapping_completion.get("public_manifest_identity_sha256"),
    }
    if (
        len(public_manifest_identities) != 1
        or not isinstance(next(iter(public_manifest_identities)), str)
        or len(next(iter(public_manifest_identities))) != 64
    ):
        raise RuntimeError("consensus and private mapping do not bind the same blind pack")


def build_bindings(pool: Path, consensus_path: Path, mapping_path: Path) -> list[SampleBinding]:
    validate_locked_label_lineage(consensus_path, mapping_path)
    consensus = load_json(consensus_path)
    mapping = load_json(mapping_path)
    rows = consensus.get("rows")
    entries = mapping.get("entries")
    if not isinstance(rows, list) or not isinstance(entries, list):
        raise RuntimeError("consensus or private mapping has no row list")
    label_by_public: dict[str, str] = {}
    for row in rows:
        blind_id = row.get("blind_id")
        label = row.get("primary_overall_structural_quality")
        if not isinstance(blind_id, str) or label not in {POSITIVE, NEGATIVE, EXCLUDED}:
            raise RuntimeError("invalid consensus row")
        if blind_id in label_by_public:
            raise RuntimeError(f"duplicate consensus blind id: {blind_id}")
        label_by_public[blind_id] = label
    require_label_counts(label_by_public.values())

    bindings: list[SampleBinding] = []
    seen_global: set[int] = set()
    for entry in entries:
        global_index = entry.get("global_index")
        shard_index = entry.get("shard_index")
        local_index = entry.get("local_index")
        public_id = entry.get("public_blind_id")
        runner_id = entry.get("runner_blind_id")
        if (
            type(global_index) is not int
            or type(shard_index) is not int
            or type(local_index) is not int
            or not isinstance(public_id, str)
            or not isinstance(runner_id, str)
            or public_id not in label_by_public
            or global_index in seen_global
        ):
            raise RuntimeError("invalid or duplicate private mapping entry")
        shard_matches = sorted(pool.glob(f"*shard{shard_index:02d}of08*/"))
        if len(shard_matches) != 1:
            raise RuntimeError(f"cannot resolve shard {shard_index}: {shard_matches}")
        png_path = shard_matches[0] / "blind_images" / f"{runner_id}.png"
        if not png_path.is_file() or png_path.is_symlink():
            raise RuntimeError(f"missing endpoint PNG: {png_path}")
        results = load_json(shard_matches[0] / "results.json")
        records = results.get("branch_records")
        if not isinstance(records, list) or local_index not in range(len(records)):
            raise RuntimeError(f"invalid branch record index in shard {shard_index}")
        record = records[local_index]
        if (
            record.get("global_index") != global_index
            or record.get("local_index") != local_index
            or record.get("blind_id") != runner_id
        ):
            raise RuntimeError("mapping and shard branch record disagree")
        stream_seed = record.get("stream_seed")
        if type(stream_seed) is not int:
            raise RuntimeError("branch record lacks an integer stream seed")
        bindings.append(
            SampleBinding(
                global_index=global_index,
                shard_index=shard_index,
                local_index=local_index,
                stream_seed=stream_seed,
                runner_blind_id=runner_id,
                public_blind_id=public_id,
                label=label_by_public[public_id],
                png_path=png_path.resolve(),
            )
        )
        seen_global.add(global_index)
    bindings.sort(key=lambda item: item.global_index)
    if [item.global_index for item in bindings] != list(range(64)):
        raise RuntimeError("private mapping no longer covers global indices 0..63")
    require_label_counts(item.label for item in bindings)
    return bindings


def mean_square_spatial_gradient(array: np.ndarray) -> np.ndarray:
    """Return summed horizontal/vertical mean-square finite differences.

    Input shape is ``[..., C, H, W]``.  Output retains every leading dimension
    before C and averages over C and the respective valid spatial supports.
    """

    if array.ndim < 3:
        raise ValueError("spatial gradient input is too small")
    vertical = np.diff(array, axis=-2)
    horizontal = np.diff(array, axis=-1)
    axes = tuple(range(array.ndim - 3, array.ndim))
    return np.mean(vertical * vertical, axis=axes) + np.mean(
        horizontal * horizontal, axis=axes
    )


def per_channel_dirichlet(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log normalized Dirichlet energy, log variance, and raw energy.

    Input is ``[B,T,C,H,W]``.  The normalized energy for channel c is

        (mean (vertical differences)^2 + mean (horizontal differences)^2)
        / (spatial variance + EPS).

    All three returned arrays have shape ``[B,T,C]``.
    """

    if array.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W], got {array.shape}")
    variance = np.var(array, axis=(-2, -1))
    vertical = np.diff(array, axis=-2)
    horizontal = np.diff(array, axis=-1)
    energy = np.mean(vertical * vertical, axis=(-2, -1)) + np.mean(
        horizontal * horizontal, axis=(-2, -1)
    )
    log_dirichlet = np.log(energy / (variance + EPS) + EPS)
    return log_dirichlet, np.log(variance + EPS), energy


def tile_reduction(energy: np.ndarray, grid: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Max tile mean and max/sum concentration for ``[B,T,C,H,W]`` energy."""

    if energy.ndim != 5 or energy.shape[-2] % grid or energy.shape[-1] % grid:
        raise ValueError("tile energy has incompatible shape")
    tile_h = energy.shape[-2] // grid
    tile_w = energy.shape[-1] // grid
    values = []
    for row in range(grid):
        for column in range(grid):
            tile = energy[
                ...,
                row * tile_h : (row + 1) * tile_h,
                column * tile_w : (column + 1) * tile_w,
            ]
            values.append(np.mean(tile, axis=(-3, -2, -1)))
    stacked = np.stack(values, axis=-1)
    return np.max(stacked, axis=-1), np.max(stacked, axis=-1) / (
        np.sum(stacked, axis=-1) + EPS
    )


def image_features(rgb: np.ndarray) -> dict[str, float]:
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("endpoint image must be uint8 RGB 256x256")
    value = rgb.astype(np.float64) / 255.0
    gray = 0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(gx, gy)
    laplacian = ndimage.laplace(gray, mode="reflect")
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - np.mean(gray)))) ** 2
    yy, xx = np.indices(gray.shape)
    radius = np.sqrt(
        (yy - (gray.shape[0] - 1) / 2.0) ** 2
        + (xx - (gray.shape[1] - 1) / 2.0) ** 2
    ) / (min(gray.shape) / 2.0)
    patch_laplacian = []
    for row in range(4):
        for column in range(4):
            patch = laplacian[
                row * 64 : (row + 1) * 64,
                column * 64 : (column + 1) * 64,
            ]
            patch_laplacian.append(float(np.var(patch)))
    patches = np.asarray(patch_laplacian)
    histogram = np.histogram(gray, bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    histogram /= np.sum(histogram)
    positive = histogram > 0
    saturation = (np.max(value, axis=-1) - np.min(value, axis=-1)) / (
        np.max(value, axis=-1) + EPS
    )
    return {
        "endpoint_gray_std": float(np.std(gray)),
        "endpoint_gradient_mean": float(np.mean(gradient)),
        "endpoint_gradient_q90": float(np.quantile(gradient, 0.90)),
        "endpoint_gradient_q99": float(np.quantile(gradient, 0.99)),
        "endpoint_laplacian_variance": float(np.var(laplacian)),
        "endpoint_laplacian_abs_mean": float(np.mean(np.abs(laplacian))),
        "endpoint_edge_density_0p05": float(np.mean(gradient > 0.05)),
        "endpoint_edge_density_0p10": float(np.mean(gradient > 0.10)),
        "endpoint_fft_power_radius_gt_0p4": float(
            np.sum(spectrum[radius > 0.4]) / (np.sum(spectrum) + EPS)
        ),
        "endpoint_fft_power_radius_gt_0p5": float(
            np.sum(spectrum[radius > 0.5]) / (np.sum(spectrum) + EPS)
        ),
        "endpoint_gaussian_residual_sigma_0p5_rms": float(
            np.sqrt(
                np.mean(
                    (gray - ndimage.gaussian_filter(gray, 0.5, mode="reflect")) ** 2
                )
            )
        ),
        "endpoint_patch_laplacian_q25": float(np.quantile(patches, 0.25)),
        "endpoint_patch_laplacian_cv": float(
            np.std(patches) / (np.mean(patches) + EPS)
        ),
        "endpoint_gray_entropy": float(-np.sum(histogram[positive] * np.log(histogram[positive]))),
        "endpoint_saturation_mean": float(np.mean(saturation)),
    }


def phase_means(
    features: dict[str, float], name: str, values: np.ndarray, *, length: int = 250
) -> None:
    if values.ndim != 1 or len(values) not in {length, length - 1}:
        raise ValueError(f"unexpected track shape for {name}: {values.shape}")
    for phase, start, stop, _ in PHASES:
        bounded_stop = min(stop, len(values))
        if start < bounded_stop:
            features[f"{name}_{phase}_mean"] = float(
                np.nanmean(values[start:bounded_stop])
            )


def track_reductions(features: dict[str, float], name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid track for {name}")
    delta = np.diff(values)
    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_maximum"] = float(np.max(values))
    features[f"{name}_minimum"] = float(np.min(values))
    features[f"{name}_terminal"] = float(values[-1])
    features[f"{name}_total_variation"] = float(np.sum(np.abs(delta)))
    features[f"{name}_max_positive_jump"] = float(max(0.0, np.max(delta)))
    features[f"{name}_max_negative_jump"] = float(max(0.0, np.max(-delta)))
    features[f"{name}_peak_to_terminal"] = float(np.max(values) - values[-1])


def robust_reference_z(track: np.ndarray, reference_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if track.ndim != 2 or reference_mask.shape != (len(track),):
        raise ValueError("invalid robust-reference inputs")
    reference = track[reference_mask]
    median = np.median(reference, axis=0)
    mad = np.median(np.abs(reference - median), axis=0)
    scale = 1.4826 * mad + 1e-6
    z = (track - median) / scale
    return z, {
        "reference_count": int(np.sum(reference_mask)),
        "median": median.tolist(),
        "mad": mad.tolist(),
        "scale_floor": 1e-6,
        "mad_consistency_factor": 1.4826,
    }


def feature_family(name: str) -> str:
    if name.startswith("endpoint_"):
        return "endpoint_image_retrospective"
    if name.startswith("legacy_"):
        return "posthoc_legacy_hypothesis"
    if name.startswith("pred_temporal_"):
        return "predicted_clean_online_causal"
    if name.startswith("pred_") or name.startswith("alpha_compensated_pred_"):
        return "predicted_clean_predictable"
    if name.startswith("state_") or name.startswith("epsilon_"):
        return "state_epsilon_controls"
    if name.startswith("variance_head_") or name.startswith("operational_logstd_"):
        return "reverse_kernel_predictable"
    if name.startswith("innovation_") or name.startswith("transition_"):
        return "innovation_online_causal"
    if name.startswith("cross_") or name.startswith("raw_K_"):
        return "cross_scale_predictable_or_adapted"
    if name.startswith("mixture_") or name.startswith("component_"):
        return "likelihood_ratio_retrospective"
    return "other"


FULL_PATH_SUFFIXES = (
    "_maximum",
    "_minimum",
    "_terminal",
    "_total_variation",
    "_max_positive_jump",
    "_max_negative_jump",
    "_peak_to_terminal",
)
PHASE_BY_NAME = {phase: (start, stop) for phase, start, stop, _ in PHASES}


def _base_availability(name: str) -> str:
    if (
        name.startswith("innovation_")
        or name.startswith("transition_")
        or name.startswith("pred_temporal_")
        or name.startswith("cross_theta_innovation_")
    ):
        return "online_causal"
    return "predictable"


def feature_timing(name: str) -> dict[str, Any]:
    """Describe when a scalar is first available for a pre-terminal decision.

    A phase mean from a predictable track is available when the last model
    output in that phase has been computed.  A phase mean from an adapted
    track is available one decision step later, after its last transition or
    adjacent-prediction change has been observed.  Whole-path reductions are
    retrospective even when their underlying pointwise track is predictable.
    """

    if name.startswith("legacy_two_phase_"):
        return {
            "availability": "predictable",
            "latest_required_sampling_step": 149,
            "latest_required_internal_timestep": 100,
            "observation_timing": "before_transition_at_latest_step",
            "preterminal_actionable": True,
        }
    legacy_special = {
        "legacy_q1_pred_roughness_reference_z": ("predictable", 99),
        "legacy_q1_pred_temporal_instability_reference_z": ("online_causal", 100),
        "legacy_q2_pred_amplitude_reference_z": ("predictable", 149),
    }
    if name in legacy_special:
        kind, step = legacy_special[name]
        return {
            "availability": kind,
            "latest_required_sampling_step": step,
            "latest_required_internal_timestep": 249 - step,
            "observation_timing": "before_transition_at_latest_step",
            "preterminal_actionable": True,
        }
    if name.startswith("endpoint_") or name.startswith("mixture_") or name.startswith(
        "component_"
    ):
        return {
            "availability": "retrospective",
            "latest_required_sampling_step": 249,
            "latest_required_internal_timestep": 0,
            "observation_timing": "endpoint_or_whole_path",
            "preterminal_actionable": False,
        }

    phase_match = re.search(r"_(q[0-4])_mean$", name)
    if phase_match is not None:
        _, stop = PHASE_BY_NAME[phase_match.group(1)]
        base = _base_availability(name)
        track_has_249_rows = (
            name.startswith("innovation_")
            or name.startswith("transition_")
            or name.startswith("pred_temporal_")
            or name.startswith("operational_logstd_")
        )
        effective_stop = min(stop, 249) if track_has_249_rows else stop
        decision_step = effective_stop - 1 if base == "predictable" else effective_stop
        if decision_step >= 249:
            return {
                "availability": "retrospective",
                "latest_required_sampling_step": min(decision_step, 250),
                "latest_required_internal_timestep": 0 if decision_step == 249 else None,
                "observation_timing": "endpoint_or_after_final_transition",
                "preterminal_actionable": False,
            }
        return {
            "availability": base,
            "latest_required_sampling_step": decision_step,
            "latest_required_internal_timestep": 249 - decision_step,
            "observation_timing": "before_transition_at_latest_step",
            "preterminal_actionable": True,
        }

    # The remaining emitted track summaries consume their entire 61-, 249-,
    # or 250-row track.  A bare ``_mean`` is also a whole-path reduction; phase
    # means were handled above first.
    if name.endswith("_mean") or name.endswith(FULL_PATH_SUFFIXES):
        return {
            "availability": "retrospective",
            "latest_required_sampling_step": 249,
            "latest_required_internal_timestep": 0,
            "observation_timing": "whole_path_reduction",
            "preterminal_actionable": False,
        }
    raise RuntimeError(f"feature timing is not explicitly classified: {name}")


def evaluate_features(frame: pd.DataFrame) -> pd.DataFrame:
    include = frame["primary_label"].isin([POSITIVE, NEGATIVE]).to_numpy()
    target = (frame.loc[include, "primary_label"] == POSITIVE).astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    excluded_columns = {
        "global_index",
        "shard_index",
        "local_index",
        "stream_seed",
        "runner_blind_id",
        "public_blind_id",
        "primary_label",
        "endpoint_png_path",
    }
    for name in frame.columns:
        if name in excluded_columns:
            continue
        values = pd.to_numeric(frame.loc[include, name], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if not np.all(finite) or len(np.unique(values)) < 2:
            continue
        raw_auc = float(roc_auc_score(target, values))
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        oriented = direction * values
        timing = feature_timing(name)
        rows.append(
            {
                "feature": name,
                "family": feature_family(name),
                **timing,
                "N_clear_bad": int(np.sum(target)),
                "N_not_clear_bad": int(np.sum(1 - target)),
                "raw_auc_bad_high": raw_auc,
                "exploratory_orientation": "higher_is_bad" if direction > 0 else "lower_is_bad",
                "oriented_auc": float(roc_auc_score(target, oriented)),
                "oriented_average_precision": float(average_precision_score(target, oriented)),
                "cliffs_delta_raw": float(2.0 * raw_auc - 1.0),
                "clear_bad_mean": float(np.mean(values[target == 1])),
                "clear_bad_median": float(np.median(values[target == 1])),
                "not_clear_bad_mean": float(np.mean(values[target == 0])),
                "not_clear_bad_median": float(np.median(values[target == 0])),
                "p_value_computed": False,
                "confirmatory_claim_allowed": False,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["oriented_auc", "oriented_average_precision", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def selected_auc_curves(
    tracks: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, np.ndarray]:
    include = np.isin(labels, [POSITIVE, NEGATIVE])
    target = (labels[include] == POSITIVE).astype(int)
    curves: dict[str, np.ndarray] = {}
    for name in (
        "pred_dirichlet_log_mean",
        "pred_log_spatial_variance_mean",
        "alpha_compensated_pred_gradient_energy",
        "pred_temporal_instability",
        "operational_logstd_mean",
        "innovation_z2_mean",
    ):
        values = tracks[name][include]
        curves[name] = np.asarray(
            [roc_auc_score(target, values[:, step]) for step in range(values.shape[1])],
            dtype=np.float64,
        )
    return curves


def save_auc_figure(curves: dict[str, np.ndarray], path: Path) -> None:
    labels = {
        "pred_dirichlet_log_mean": "pred-x0 normalized roughness (predictable)",
        "pred_log_spatial_variance_mean": "pred-x0 amplitude (predictable)",
        "alpha_compensated_pred_gradient_energy": "alpha-compensated gradient (predictable)",
        "pred_temporal_instability": "pred-x0 temporal instability (online)",
        "operational_logstd_mean": "stochastic reverse-kernel log std (predictable)",
        "innovation_z2_mean": "realized innovation energy (online)",
    }
    figure, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    for name, curve in curves.items():
        axis.plot(np.arange(len(curve)), curve, linewidth=1.7, label=labels[name])
    for _, start, stop, _ in PHASES[1:]:
        axis.axvline(start, color="#777777", linewidth=0.7, alpha=0.5)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlim(0, 249)
    axis.set_xlabel("sampling step k (0 = internal t249; 249 = internal t0)")
    axis.set_ylabel("exploratory per-step ROC-AUC, bad-high orientation fixed as plotted")
    axis.set_title("Legacy 64-path discovery: discrimination changes over sampling time")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_two_phase_figure(frame: pd.DataFrame, path: Path) -> None:
    include = frame["primary_label"].isin([POSITIVE, NEGATIVE])
    data = frame.loc[include].copy()
    colors = data["primary_label"].map({NEGATIVE: "#5a78b5", POSITIVE: "#d1495b"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    axes[0].scatter(
        data["legacy_q1_pred_roughness_reference_z"],
        data["legacy_q2_pred_amplitude_reference_z"],
        c=colors,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
    )
    axes[0].set_xlabel("q1 predicted-clean normalized roughness (reference z)")
    axes[0].set_ylabel("q2 predicted-clean amplitude (reference z)")
    axes[0].set_title("Post-hoc two-phase hypothesis")
    axes[0].grid(alpha=0.18)
    bad = data[data["primary_label"] == POSITIVE]
    good = data[data["primary_label"] == NEGATIVE]
    axes[1].hist(
        good["legacy_two_phase_predicted_clean_score"],
        bins=16,
        color="#5a78b5",
        alpha=0.68,
        label=f"not-clear-bad (N={len(good)})",
    )
    axes[1].scatter(
        bad["legacy_two_phase_predicted_clean_score"],
        np.full(len(bad), 0.12),
        color="#d1495b",
        marker="x",
        s=80,
        linewidths=2.0,
        label=f"clear bad (N={len(bad)})",
    )
    axes[1].set_xlabel("two-phase score (post-hoc; not a validated cutoff)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Same-pool separation cannot validate the score")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def inspect_trace_source(shard: Path) -> dict[str, Any]:
    results_path = shard / "results.json"
    manifest_path = shard / "manifest.json"
    completion_path = shard / "completion.json"
    trace_path = shard / "trace_private.npz"
    for path in (results_path, manifest_path, completion_path, trace_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or indirect shard artifact: {path}")
    results = load_json(results_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = results.get("protocol_identity_sha256")
    recorded = results.get("private_trace", {}).get("sha256")
    actual = sha256_file(trace_path)
    if (
        completion.get("complete") is not True
        or not isinstance(protocol_identity, str)
        or len(protocol_identity) != 64
        or manifest.get("identity_sha256") != completion.get("manifest_identity_sha256")
        or sha256_file(manifest_path) != completion.get("manifest_file_sha256")
        or sha256_file(results_path) != completion.get("results_file_sha256")
        or recorded != actual
        or completion.get("private_trace_sha256") != actual
        or completion.get("protocol_identity_sha256") != protocol_identity
    ):
        raise RuntimeError(f"shard completion/hash/protocol binding is invalid: {shard}")
    return {
        "shard": str(shard.resolve()),
        "results_sha256": sha256_file(results_path),
        "completion_sha256": sha256_file(completion_path),
        "trace_sha256": actual,
        "trace_bytes": trace_path.stat().st_size,
        "protocol_identity_sha256": protocol_identity,
        "trace_schema": results.get("private_trace", {}).get("schema"),
        "trace_keys": results.get("private_trace", {}).get("keys"),
    }


def require_archive_array(
    archive: np.lib.npyio.NpzFile,
    schema: dict[str, Any],
    name: str,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if name not in archive.files or name not in schema:
        raise RuntimeError(f"trace array is absent from archive/schema: {name}")
    value = archive[name]
    recorded = schema[name]
    if (
        value.dtype != expected_dtype
        or value.shape != expected_shape
        or recorded.get("dtype") != value.dtype.name
        or recorded.get("shape") != list(value.shape)
        or not np.isfinite(value).all()
    ):
        raise RuntimeError(
            f"trace array schema/dtype/shape/finite check failed for {name}: "
            f"dtype={value.dtype}, shape={value.shape}"
        )
    return value


def extract(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    bindings = build_bindings(args.pool_dir, args.consensus, args.mapping)
    binding_by_global = {item.global_index: item for item in bindings}
    shards = sorted(args.pool_dir.glob("class0207_poolv1_shard*of08_*"))
    if len(shards) != 8:
        raise RuntimeError(f"expected eight shards, found {len(shards)}")
    source_records = [inspect_trace_source(shard) for shard in shards]
    protocol_ids = {record["protocol_identity_sha256"] for record in source_records}
    if len(protocol_ids) != 1:
        raise RuntimeError(f"source shards disagree on protocol identity: {protocol_ids}")

    rows: list[dict[str, Any]] = []
    tracks_by_global: dict[int, dict[str, np.ndarray]] = {}
    for shard_index, shard in enumerate(shards):
        trace_path = shard / "trace_private.npz"
        with np.load(trace_path, allow_pickle=False) as archive:
            source = source_records[shard_index]
            schema = source["trace_schema"]
            trace_keys = source["trace_keys"]
            if (
                not isinstance(schema, dict)
                or not isinstance(trace_keys, list)
                or set(archive.files) != set(trace_keys)
                or set(schema) != set(trace_keys)
            ):
                raise RuntimeError(f"trace archive/schema key set changed: {trace_path}")
            global_indices = require_archive_array(
                archive, schema, "branch_global_index", np.dtype(np.int16), (8,)
            ).astype(int)
            stream_seeds = require_archive_array(
                archive, schema, "branch_stream_seed", np.dtype(np.int64), (8,)
            )
            internal = require_archive_array(
                archive, schema, "full_internal_timestep", np.dtype(np.int16), (250,)
            ).astype(int)
            alpha_schedule = require_archive_array(
                archive,
                schema,
                "full_internal_alpha_bar",
                np.dtype(np.float64),
                (250,),
            )
            if not np.array_equal(internal, np.arange(249, -1, -1)):
                raise RuntimeError(f"full timestep order changed in {trace_path}")
            if (
                alpha_schedule.shape != (250,)
                or np.any(alpha_schedule <= 0.0)
                or np.any(alpha_schedule >= 1.0)
                or np.any(np.diff(alpha_schedule) >= 0.0)
            ):
                raise RuntimeError("internal-t alpha-bar schedule is invalid")
            # ``full_internal_alpha_bar`` is a lookup table indexed by internal
            # t=0..249, whereas every saved state/prediction row is in sampling
            # order internal t=249..0.  Join through the explicit timestep axis.
            alpha = alpha_schedule[internal]
            if np.any(np.diff(alpha) <= 0.0):
                raise RuntimeError("sampling-order alpha-bar must increase from t249 to t0")
            step_shape = (8, 250, 4, 32, 32)
            state = require_archive_array(
                archive, schema, "state_before", np.dtype(np.float32), step_shape
            ).astype(np.float64)
            pred = require_archive_array(
                archive, schema, "pred_xstart", np.dtype(np.float32), step_shape
            ).astype(np.float64)
            pstd = require_archive_array(
                archive, schema, "p_standard_deviation", np.dtype(np.float32), step_shape
            ).astype(np.float64)
            innovation = require_archive_array(
                archive, schema, "transition_innovation", np.dtype(np.float32), step_shape
            ).astype(np.float64)
            decoded = require_archive_array(
                archive,
                schema,
                "decoded_images",
                np.dtype(np.float32),
                (8, 3, 256, 256),
            )
            if np.any(pstd <= 0.0):
                raise RuntimeError(f"nonpositive reverse standard deviation in {trace_path}")

            pred_dirichlet_c, pred_logvar_c, pred_energy_c = per_channel_dirichlet(pred)
            state_dirichlet_c, _, _ = per_channel_dirichlet(state)
            epsilon = (
                state
                - np.sqrt(alpha)[None, :, None, None, None] * pred
            ) / np.sqrt(1.0 - alpha)[None, :, None, None, None]
            epsilon_dirichlet_c, _, _ = per_channel_dirichlet(epsilon)
            variance_head_logstd = np.log(pstd)
            variance_head_dirichlet_c, _, _ = per_channel_dirichlet(
                variance_head_logstd
            )
            pred_delta = np.diff(pred, axis=1)
            pred_delta_energy = np.mean(pred_delta * pred_delta, axis=(2, 3, 4))
            pred_pair_variance = np.var(pred[:, :-1], axis=(2, 3, 4)) + np.var(
                pred[:, 1:], axis=(2, 3, 4)
            )
            pred_instability = pred_delta_energy / (pred_pair_variance + EPS)
            # The upstream sampler draws a t=0 tensor for RNG-stream fidelity,
            # but the nonzero mask discards it.  Realized-transition features
            # therefore contain only the 249 stochastic transitions k=0..248.
            innovation_effective = innovation[:, :-1]
            innovation_energy = innovation_effective * innovation_effective
            innovation_tile_max, innovation_tile_concentration = tile_reduction(
                innovation_energy
            )
            innovation_z2_mean = np.mean(innovation_energy, axis=(2, 3, 4))
            variance_head_logstd_mean = np.mean(
                variance_head_logstd, axis=(2, 3, 4)
            )
            operational_logstd_mean = variance_head_logstd_mean[:, :-1]
            transition_nll_no_constant = (
                0.5 * innovation_z2_mean + operational_logstd_mean
            )
            alpha_gradient = alpha[None, :] * np.mean(pred_energy_c, axis=2)

            theta = require_archive_array(
                archive,
                schema,
                "theta",
                np.dtype(np.float64),
                (8, 61, 4, 32, 32),
            )
            current_eps = require_archive_array(
                archive,
                schema,
                "epsilon_current_reconstructed",
                np.dtype(np.float32),
                (8, 61, 4, 32, 32),
            ).astype(np.float64)
            shifted_eps = require_archive_array(
                archive,
                schema,
                "epsilon_shifted",
                np.dtype(np.float32),
                (8, 61, 4, 32, 32),
            ).astype(np.float64)
            evidence_full_index = require_archive_array(
                archive,
                schema,
                "evidence_full_step_index",
                np.dtype(np.int16),
                (61,),
            ).astype(int)
            evidence_internal = require_archive_array(
                archive,
                schema,
                "evidence_internal_timestep",
                np.dtype(np.int16),
                (61,),
            ).astype(int)
            if not np.array_equal(internal[evidence_full_index], evidence_internal):
                raise RuntimeError("evidence rows no longer join to the full sampling axis")
            epsilon_join_error = float(
                np.max(np.abs(epsilon[:, evidence_full_index] - current_eps))
            )
            if epsilon_join_error > 1e-5:
                raise RuntimeError(
                    f"alpha/timestep epsilon reconstruction audit failed: {epsilon_join_error}"
                )
            raw_K = require_archive_array(
                archive, schema, "component_raw_K", np.dtype(np.float64), (8, 61, 34)
            )
            component_L = require_archive_array(
                archive, schema, "component_L", np.dtype(np.float64), (8, 61, 34)
            )
            component_log_e = require_archive_array(
                archive, schema, "component_log_e", np.dtype(np.float64), (8, 61, 34)
            )
            mixture_log_e = require_archive_array(
                archive, schema, "mixture_path_log_e", np.dtype(np.float64), (8, 61)
            )
            if not np.array_equal(raw_K[..., 0::2], raw_K[..., 1::2]):
                raise RuntimeError("signed raw-K pairs are no longer exactly equal")
            cross_theta_rms = np.sqrt(np.mean(theta * theta, axis=(2, 3, 4)))
            cross_epsdiff_rms = np.sqrt(
                np.mean((current_eps - shifted_eps) ** 2, axis=(2, 3, 4))
            )
            current_flat = current_eps.reshape(8, 61, -1)
            shifted_flat = shifted_eps.reshape(8, 61, -1)
            cross_eps_cosine = np.sum(current_flat * shifted_flat, axis=2) / (
                np.linalg.norm(current_flat, axis=2)
                * np.linalg.norm(shifted_flat, axis=2)
                + EPS
            )
            theta_energy = theta * theta
            _, cross_theta_tile_concentration = tile_reduction(theta_energy)
            effective_evidence = evidence_internal > 0
            theta_effective_flat = theta[:, effective_evidence].reshape(8, 60, -1)
            innovation_aligned = innovation[:, evidence_full_index[effective_evidence]].reshape(
                8, 60, -1
            )
            cross_theta_innovation_cosine = np.sum(
                theta_effective_flat * innovation_aligned, axis=2
            ) / (
                np.linalg.norm(theta_effective_flat, axis=2)
                * np.linalg.norm(innovation_aligned, axis=2)
                + EPS
            )
            raw_K_base = raw_K[..., 0::2]
            raw_K_global = raw_K_base[..., 0]
            raw_K_tiles = raw_K_base[..., 1:]
            raw_K_tile_mean = np.mean(raw_K_tiles, axis=2)
            raw_K_tile_concentration = np.max(raw_K_tiles, axis=2) / (
                np.sum(raw_K_tiles, axis=2) + EPS
            )

            for local_index, global_index in enumerate(global_indices.tolist()):
                binding = binding_by_global[global_index]
                if (
                    binding.shard_index != shard_index
                    or binding.local_index != local_index
                    or int(stream_seeds[local_index]) != binding.stream_seed
                ):
                    raise RuntimeError("trace branch ordering disagrees with locked mapping")
                # torchvision.save_image uses ``mul(255).add_(0.5)`` followed
                # by a uint8 cast.  NumPy ``round`` uses bankers' rounding and
                # differs at exact half-integers, so reproduce the upstream
                # add-half/floor rule byte-for-byte instead of using a fuzzy
                # pixel tolerance.
                rgb_from_trace = np.floor(
                    (np.clip(decoded[local_index], -1.0, 1.0) + 1.0) * 127.5
                    + 0.5
                ).astype(np.uint8).transpose(1, 2, 0)
                with Image.open(binding.png_path) as image:
                    image.load()
                    rgb_from_png = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if not np.array_equal(rgb_from_trace, rgb_from_png):
                    raise RuntimeError(
                        f"decoded trace and endpoint PNG differ for global {global_index}"
                    )
                features: dict[str, Any] = {
                    "global_index": global_index,
                    "shard_index": shard_index,
                    "local_index": local_index,
                    "stream_seed": binding.stream_seed,
                    "runner_blind_id": binding.runner_blind_id,
                    "public_blind_id": binding.public_blind_id,
                    "primary_label": binding.label,
                    "endpoint_png_path": str(binding.png_path),
                }
                features.update(image_features(rgb_from_png))
                sample_tracks: dict[str, np.ndarray] = {
                    "pred_dirichlet_log_mean": np.mean(
                        pred_dirichlet_c[local_index], axis=1
                    ),
                    "pred_log_spatial_variance_mean": np.mean(
                        pred_logvar_c[local_index], axis=1
                    ),
                    "alpha_compensated_pred_gradient_energy": alpha_gradient[local_index],
                    "state_dirichlet_log_mean": np.mean(
                        state_dirichlet_c[local_index], axis=1
                    ),
                    "epsilon_dirichlet_log_mean": np.mean(
                        epsilon_dirichlet_c[local_index], axis=1
                    ),
                    "variance_head_logstd_dirichlet_log_mean": np.mean(
                        variance_head_dirichlet_c[local_index], axis=1
                    ),
                    "variance_head_logstd_mean": variance_head_logstd_mean[local_index],
                    "variance_head_logstd_spatial_sd": np.std(
                        variance_head_logstd[local_index], axis=(1, 2, 3)
                    ),
                    "operational_logstd_mean": operational_logstd_mean[local_index],
                    "pred_temporal_instability": pred_instability[local_index],
                    "innovation_z2_mean": innovation_z2_mean[local_index],
                    "innovation_tile_max_z2": innovation_tile_max[local_index],
                    "innovation_tile_concentration": innovation_tile_concentration[local_index],
                    "transition_nll_per_dim_no_constant": transition_nll_no_constant[local_index],
                    "cross_theta_rms": cross_theta_rms[local_index],
                    "cross_epsdiff_rms": cross_epsdiff_rms[local_index],
                    "cross_eps_cosine": cross_eps_cosine[local_index],
                    "cross_theta_tile_concentration": cross_theta_tile_concentration[local_index],
                    "cross_theta_innovation_cosine": cross_theta_innovation_cosine[local_index],
                    "raw_K_global": raw_K_global[local_index],
                    "raw_K_tile_mean": raw_K_tile_mean[local_index],
                    "raw_K_tile_concentration": raw_K_tile_concentration[local_index],
                    "mixture_log_e": mixture_log_e[local_index],
                }
                for channel in range(4):
                    sample_tracks[f"pred_dirichlet_log_c{channel}"] = pred_dirichlet_c[
                        local_index, :, channel
                    ]
                    sample_tracks[f"pred_log_spatial_variance_c{channel}"] = pred_logvar_c[
                        local_index, :, channel
                    ]
                for name, values in sample_tracks.items():
                    if len(values) in {249, 250}:
                        phase_means(features, name, values)
                    if len(values) in {60, 61, 249, 250}:
                        track_reductions(features, name, values)
                features["component_any_running_max_log_e"] = float(
                    np.max(component_log_e[local_index])
                )
                features["component_any_terminal_max_log_e"] = float(
                    np.max(component_log_e[local_index, -1])
                )
                features["component_max_positive_segment_log_lr"] = float(
                    max(
                        _maximum_positive_segment(component_L[local_index, :, component])
                        for component in range(component_L.shape[2])
                    )
                )
                rows.append(features)
                tracks_by_global[global_index] = sample_tracks

    rows.sort(key=lambda row: int(row["global_index"]))
    if [row["global_index"] for row in rows] != list(range(64)):
        raise RuntimeError("extracted rows do not cover global indices 0..63")
    frame = pd.DataFrame(rows)
    track_names = sorted(next(iter(tracks_by_global.values())))
    tracks = {
        name: np.stack(
            [tracks_by_global[index][name] for index in range(64)], axis=0
        )
        for name in track_names
    }
    return frame, tracks, source_records


def _maximum_positive_segment(values: np.ndarray) -> float:
    current = 0.0
    best = 0.0
    for value in values:
        current = max(0.0, current + float(value))
        best = max(best, current)
    return best


def add_legacy_two_phase_score(
    frame: pd.DataFrame, tracks: dict[str, np.ndarray]
) -> dict[str, Any]:
    labels = frame["primary_label"].to_numpy()
    reference = labels == NEGATIVE
    rough_z, rough_reference = robust_reference_z(
        tracks["pred_dirichlet_log_mean"], reference
    )
    amplitude_z, amplitude_reference = robust_reference_z(
        tracks["pred_log_spatial_variance_mean"], reference
    )
    instability_z, instability_reference = robust_reference_z(
        tracks["pred_temporal_instability"], reference
    )
    q1_roughness = np.mean(rough_z[:, 50:100], axis=1)
    q2_amplitude = np.mean(amplitude_z[:, 100:150], axis=1)
    q1_instability = np.mean(instability_z[:, 50:100], axis=1)
    two_phase = (q1_roughness + q2_amplitude) / math.sqrt(2.0)
    frame["legacy_q1_pred_roughness_reference_z"] = q1_roughness
    frame["legacy_q1_pred_temporal_instability_reference_z"] = q1_instability
    frame["legacy_q2_pred_amplitude_reference_z"] = q2_amplitude
    frame["legacy_two_phase_predicted_clean_score"] = two_phase
    return {
        "status": "POSTHOC_SAME_POOL_REFERENCE_AND_FEATURE_SELECTION",
        "reference_label": NEGATIVE,
        "reference_warning": (
            "Reference statistics and the feature/window choice use the already-unsealed "
            "legacy pool. They are saved only so the exact formula can be frozen for a "
            "future blind dataset."
        ),
        "q1_pred_roughness": rough_reference,
        "q2_pred_amplitude": amplitude_reference,
        "q1_pred_temporal_instability_control": instability_reference,
        "score_formula": (
            "(mean_{k=50..99} z_ref(pred_dirichlet_log_mean[k]) + "
            "mean_{k=100..149} z_ref(pred_log_spatial_variance_mean[k])) / sqrt(2)"
        ),
        "available_before_transition": "internal t=100 -> 99",
        "validated": False,
        "intervention_authorized": False,
    }


def source_inventory(
    args: argparse.Namespace, source_records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "consensus": {
            "path": str(args.consensus.resolve()),
            "sha256": sha256_file(args.consensus),
            "manifest_sha256": sha256_file(args.consensus.parent / "manifest.json"),
            "completion_sha256": sha256_file(args.consensus.parent / "completion.json"),
        },
        "private_mapping": {
            "path": str(args.mapping.resolve()),
            "sha256": sha256_file(args.mapping),
            "completion_sha256": sha256_file(args.mapping.parent / "completion.json"),
            "commitment_sha256": sha256_file(
                args.mapping.parent / "blind_mapping_commitment_frozen.json"
            ),
        },
        "pool": str(args.pool_dir.resolve()),
        "shards": source_records,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def publish(args: argparse.Namespace) -> Path:
    protocol = load_json(args.protocol)
    if protocol.get("status") != "EXPLORATORY_POST_UNSEAL_DO_NOT_USE_FOR_CONFIRMATION":
        raise RuntimeError("discovery protocol status changed")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.staging.", dir=args.output_dir.parent)
    )
    try:
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        shutil.copyfile(args.protocol, staging / "protocol_snapshot.json")
        frame, tracks, source_records = extract(args)
        reference = add_legacy_two_phase_score(frame, tracks)
        results = evaluate_features(frame)
        frame.to_csv(staging / "sample_features.csv", index=False)
        results.to_csv(staging / "univariate_feature_results.csv", index=False)
        labels = frame["primary_label"].to_numpy(dtype="<U40")
        np.savez_compressed(
            staging / "selected_time_series.npz",
            global_index=frame["global_index"].to_numpy(np.int16),
            primary_label=labels,
            internal_timestep=np.arange(249, -1, -1, dtype=np.int16),
            **tracks,
        )
        curves = selected_auc_curves(tracks, labels)
        np.savez_compressed(staging / "selected_auc_curves.npz", **curves)
        save_auc_figure(curves, staging / "selected_auc_curves.png")
        save_two_phase_figure(frame, staging / "legacy_two_phase_score.png")

        included = frame["primary_label"].isin([POSITIVE, NEGATIVE])
        target = (frame.loc[included, "primary_label"] == POSITIVE).astype(int)
        candidate_values = frame.loc[
            included, "legacy_two_phase_predicted_clean_score"
        ].to_numpy(float)
        candidate_auc = float(roc_auc_score(target, candidate_values))
        family_winners = (
            results.sort_values("oriented_auc", ascending=False)
            .groupby("family", sort=True, as_index=False)
            .first()[
                [
                    "family",
                    "feature",
                    "availability",
                    "oriented_auc",
                    "exploratory_orientation",
                ]
            ]
            .to_dict(orient="records")
        )
        summary = {
            "schema_version": 6,
            "status": "EXPLORATORY_POST_UNSEAL_DO_NOT_USE_FOR_CONFIRMATION",
            "sample_counts": {
                POSITIVE: int(np.sum(frame["primary_label"] == POSITIVE)),
                NEGATIVE: int(np.sum(frame["primary_label"] == NEGATIVE)),
                EXCLUDED: int(np.sum(frame["primary_label"] == EXCLUDED)),
            },
            "metric_count_evaluated": int(len(results)),
            "multiplicity_warning": (
                "All features and time windows in this output are discovery-only. No p-value "
                "was computed and no same-pool AUC is confirmatory. The four positives all "
                "belong to the global-blur/blocky subtype."
            ),
            "supersession": (
                "This v7 analysis supersedes legacy64_v1 through legacy64_v6. "
                "v1/v2 "
                "misjoined the internal-t alpha-bar lookup table to the reversed sampling "
                "axis and mislabeled whole-path reductions as online/predictable. The main "
                "two-phase roughness-plus-amplitude score did not depend on alpha-bar, but "
                "all reconstructed-epsilon and alpha-compensated-gradient results from v1/v2 "
                "are invalid. v3 fixed those issues but still treated the masked t=0 random "
                "draw as a realized transition and did not fully validate label/trace lock "
                "lineage; v4 removed t=0 from innovation/NLL features and failed closed on the "
                "recorded hashes and schemas. v5 additionally records the exact q4 availability "
                "of 249-row operational tracks rather than using the nominal 250-row phase stop. "
                "v6 adds immutable analysis-source and protocol snapshots because the working "
                "tree is not yet committed. v7 also cross-binds the consensus, private mapping, "
                "and public blind-pack manifest identity rather than validating those locks only "
                "within their separate directories."
            ),
            "endpoint_label_consistency_check": {
                "endpoint_laplacian_variance_raw_auc_bad_high": float(
                    results.loc[
                        results["feature"] == "endpoint_laplacian_variance",
                        "raw_auc_bad_high",
                    ].iloc[0]
                ),
                "interpretation": (
                    "The visual positives are uniformly blurry, so low endpoint sharpness "
                    "checks label consistency but does not establish pre-terminal detection."
                ),
            },
            "legacy_emerging_candidate": {
                "name": "two_phase_predicted_clean_instability",
                "same_pool_auc": candidate_auc,
                "same_pool_auc_role": "hypothesis_generation_only",
                "formula_and_reference": reference,
                "interpretation": (
                    "The four blur-labeled paths show an early predictable rough/unstable "
                    "predicted-clean latent followed by a mid-trajectory amplitude overshoot. "
                    "New classes and new seeds must decide whether this is a real path-risk "
                    "signal or a class/composition coincidence."
                ),
            },
            "family_winners_for_inventory_not_selection": family_winners,
            "negative_result_retained": (
                "The previously frozen fixed high-noise likelihood-ratio mixture remains "
                "retired; this discovery analysis does not reopen or retune its alarm."
            ),
            "next_gate": (
                "Lock independent visual labels on a larger multi-class discovery set; rerun "
                "the exact custom-batch sampler with full trajectories; fit no combination "
                "until at least 40 clear bad events exist; then freeze one score for a new "
                "blind confirmation set."
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inventory = source_inventory(args, source_records)
        (staging / "source_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_files = []
        for path in sorted(staging.iterdir()):
            if path.name in {"manifest.json", "completion.json"}:
                continue
            artifact_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": 6,
            "experiment": "dit_bad_good_metric_discovery_legacy64_v7",
            "role": "POST_UNSEAL_HYPOTHESIS_GENERATION_ONLY",
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "protocol_sha256": sha256_file(args.protocol),
            "files": artifact_files,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_file_sha256": sha256_file(staging / "summary.json"),
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        (staging / "completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(args.output_dir)
        return args.output_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    rng = np.random.default_rng(17)
    array = rng.normal(size=(3, 5, 4, 8, 8))
    log_d, log_v, energy = per_channel_dirichlet(array)
    scaled_log_d, scaled_log_v, scaled_energy = per_channel_dirichlet(3.7 * array)
    if log_d.shape != (3, 5, 4) or not np.allclose(log_d, scaled_log_d, atol=1e-9):
        raise AssertionError("Dirichlet normalization is not scale invariant")
    if not np.allclose(scaled_log_v - log_v, 2.0 * math.log(3.7), atol=1e-9):
        raise AssertionError("log spatial variance scaling changed")
    if not np.allclose(scaled_energy, energy * 3.7**2, rtol=1e-12, atol=1e-12):
        raise AssertionError("raw gradient energy scaling changed")
    tile_max, tile_concentration = tile_reduction(array * array)
    if tile_max.shape != (3, 5) or tile_concentration.shape != (3, 5):
        raise AssertionError("tile reduction shape changed")
    if np.any(tile_concentration <= 0.0) or np.any(tile_concentration > 1.0):
        raise AssertionError("tile concentration escaped (0,1]")
    test_frame = pd.DataFrame(
        {
            "primary_label": [POSITIVE, POSITIVE, NEGATIVE, NEGATIVE],
            "global_index": [0, 1, 2, 3],
            "pred_log_spatial_variance_mean_q2_mean": [3.0, 4.0, 1.0, 2.0],
        }
    )
    result = evaluate_features(test_frame)
    if len(result) != 1 or result.iloc[0]["raw_auc_bad_high"] != 1.0:
        raise AssertionError("AUC evaluator changed")
    assert feature_timing("pred_log_spatial_variance_mean_q2_mean") == {
        "availability": "predictable",
        "latest_required_sampling_step": 149,
        "latest_required_internal_timestep": 100,
        "observation_timing": "before_transition_at_latest_step",
        "preterminal_actionable": True,
    }
    assert feature_timing("innovation_z2_mean_q1_mean")["latest_required_sampling_step"] == 100
    assert feature_timing("cross_epsdiff_rms_max_negative_jump")["availability"] == "retrospective"
    assert feature_timing("pred_temporal_instability_q4_mean")["preterminal_actionable"] is False
    assert feature_timing("pred_temporal_instability_q4_mean")[
        "latest_required_sampling_step"
    ] == 249
    assert feature_timing("operational_logstd_mean_q4_mean")[
        "latest_required_sampling_step"
    ] == 248
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("protocol", "consensus", "mapping"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{name} is missing or indirect: {path}")
        setattr(args, name, path)
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    output = publish(args)
    summary = load_json(output / "summary.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": summary["status"],
                "sample_counts": summary["sample_counts"],
                "metric_count_evaluated": summary["metric_count_evaluated"],
                "legacy_candidate_auc_hypothesis_only": summary[
                    "legacy_emerging_candidate"
                ]["same_pool_auc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

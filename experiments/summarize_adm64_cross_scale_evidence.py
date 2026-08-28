#!/usr/bin/env python3
"""Strictly summarize observe-only ADM64 cross-scale path evidence.

This is a read-only consumer of an output directory produced by
``observe_adm64_cross_scale_evidence.py``.  Before aggregating anything it
validates the manifest identity, frozen source identities, completion record,
every self-hashed signal, every PNG, and byte-decoded equality with the pure-P
baseline through the original runner's validators.

The manifest cap is always reported as the primary evidence definition.  An
optional, command-line-predeclared ``--cap-grid`` reconstructs alternative KL
caps exactly from the stored sufficient statistics

    K = raw_conditional_kl,
    R = raw_innovation_projection,
    gamma = min(1, sqrt(kappa / K)),
    Delta log E = gamma * R - gamma**2 * K.

Those reconstructions are explicitly exploratory sensitivity diagnostics.
They do not replace the manifest primary and must not be selected after
examining endpoint labels while retaining the primary anytime-valid claim.
No image or visual-quality conclusion is produced by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Support package imports and ``python experiments/file.py``.
    from . import observe_adm64_cross_scale_evidence as observer
except ImportError:  # pragma: no cover - exercised by the CLI entry point.
    import observe_adm64_cross_scale_evidence as observer


SUMMARY_SCHEMA_VERSION = 1
PRIMARY_ROLE = "manifest_primary_observe_only"
EXPLORATORY_ROLE = "exploratory_cap_sensitivity_diagnostic_only"
QUANTILES = (
    ("q05", 0.05),
    ("q10", 0.10),
    ("q25", 0.25),
    ("median", 0.50),
    ("q75", 0.75),
    ("q90", 0.90),
    ("q95", 0.95),
)


def parse_cap_grid(value: str) -> tuple[float, ...]:
    """Parse a finite, positive, duplicate-free cap grid in declared order."""

    caps: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            cap = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid KL cap: {token!r}") from exc
        if not math.isfinite(cap) or cap <= 0:
            raise argparse.ArgumentTypeError("every KL cap must be finite and strictly positive")
        caps.append(cap)
    if not caps:
        raise argparse.ArgumentTypeError("--cap-grid is empty")
    if len(caps) != len(set(caps)):
        raise argparse.ArgumentTypeError("--cap-grid contains duplicate values")
    return tuple(caps)


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _as_unique_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"manifest {name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RuntimeError(f"manifest {name} must contain integers")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise RuntimeError(f"manifest {name} contains duplicates")
    return result


def _require_finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{name} must be finite and strictly positive")
    return result


def validate_source_run(
    run_dir: Path,
    guided_diffusion_root: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    observer.Protocol,
    tuple[observer.ComponentSpec, ...],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run the frozen runner's complete validation path, then load signals."""

    manifest = load_json_object(run_dir / "manifest.json")
    manifest_identity = manifest.get("identity_sha256")
    if not isinstance(manifest_identity, str) or manifest_identity != observer._canonical_payload_sha(
        manifest, "identity_sha256"
    ):
        raise RuntimeError("observe-run manifest identity hash is invalid")
    if manifest.get("schema_version") != observer.SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported observe schema: {manifest.get('schema_version')!r} != "
            f"{observer.SCHEMA_VERSION}"
        )
    if manifest.get("experiment") != observer.EXPERIMENT:
        raise RuntimeError(f"unexpected observe experiment: {manifest.get('experiment')!r}")
    if manifest.get("role") != "observe_only_no_intervention_no_rejection_no_rollback_no_resampling":
        raise RuntimeError("manifest is not the frozen observe-only experiment")

    class_ids = _as_unique_int_tuple(manifest.get("class_ids"), "class_ids")
    seeds = _as_unique_int_tuple(manifest.get("seeds"), "seeds")
    protocol = observer.Protocol(class_ids, seeds)
    expected_pairs = protocol.pairs
    if manifest.get("sample_count") != len(expected_pairs):
        raise RuntimeError("manifest sample_count does not match the class/seed Cartesian product")
    expected_pair_sha = observer.sha256_json([[class_id, seed] for class_id, seed in expected_pairs])
    if manifest.get("pair_set_sha256") != expected_pair_sha:
        raise RuntimeError("manifest pair_set_sha256 is invalid")

    current_runner = Path(observer.__file__).resolve()
    current_runner_sha = observer.sha256_file(current_runner)
    runner_record = manifest.get("runner")
    if not isinstance(runner_record, dict) or runner_record.get("sha256") != current_runner_sha:
        raise RuntimeError(
            "the frozen observe runner differs from the source recorded by this run; "
            "strict schema validation is not safe"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("manifest sources record is missing")
    evidence_record = sources.get("evidence_primitives")
    evidence_path = current_runner.with_name("adm64_path_evidence.py")
    if not isinstance(evidence_record, dict) or evidence_record.get("sha256") != observer.sha256_file(
        evidence_path
    ):
        raise RuntimeError("current evidence primitives differ from the frozen manifest source")

    source_root = (
        guided_diffusion_root.resolve()
        if guided_diffusion_root is not None
        else Path(str(sources.get("guided_diffusion_root", ""))).resolve()
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"guided-diffusion source root is unavailable: {source_root}")
    if observer.git_revision(source_root) != sources.get("guided_diffusion_revision"):
        raise RuntimeError("guided-diffusion git revision differs from the manifest")
    if observer.git_tracked_dirty(source_root) != sources.get("guided_diffusion_tracked_dirty"):
        raise RuntimeError("guided-diffusion tracked-dirty state differs from the manifest")
    if observer.sha256_python_tree(source_root / "guided_diffusion") != sources.get(
        "guided_diffusion_python_tree_sha256"
    ):
        raise RuntimeError("guided-diffusion Python source tree differs from the manifest")

    primary = manifest.get("primary_evidence_definition")
    if not isinstance(primary, dict):
        raise RuntimeError("manifest primary_evidence_definition is missing")
    if primary.get("internal_checkpoints_reverse_order") != list(
        observer.EVIDENCE_INTERNAL_TIMESTEPS
    ):
        raise RuntimeError("manifest checkpoint schedule differs from the frozen runner")
    if primary.get("checkpoint_count") != len(observer.EVIDENCE_INTERNAL_TIMESTEPS):
        raise RuntimeError("manifest checkpoint_count is invalid")
    max_conditional_kl = _require_finite_positive(
        primary.get("max_conditional_kl_per_component_checkpoint"),
        "manifest primary KL cap",
    )
    alpha = _require_finite_positive(primary.get("alpha"), "manifest alpha")
    if alpha >= 1:
        raise RuntimeError("manifest alpha must be strictly less than one")
    if not observer._close(float(primary.get("log_e_crossing_threshold", math.nan)), -math.log(alpha)):
        raise RuntimeError("manifest crossing threshold is inconsistent with alpha")

    manifest_components = primary.get("components")
    if not isinstance(manifest_components, list) or not manifest_components:
        raise RuntimeError("manifest must contain at least one evidence component")
    heat_shifts = tuple(
        _require_finite_positive(record.get("additive_heat_shift"), "additive heat shift")
        for record in manifest_components
        if isinstance(record, dict)
    )
    if len(heat_shifts) != len(manifest_components):
        raise RuntimeError("manifest component record is malformed")

    original_alpha_bar, timestep_map = observer.original_schedule_and_timestep_map(source_root)
    schedule_record = primary.get("original_schedule")
    if not isinstance(schedule_record, dict):
        raise RuntimeError("manifest original-schedule record is missing")
    alpha_sha = hashlib.sha256(
        np.ascontiguousarray(original_alpha_bar, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    if schedule_record.get("alpha_bar_float64_bytes_sha256") != alpha_sha:
        raise RuntimeError("reconstructed alpha_bar differs from the manifest")
    if primary.get("spaced_timestep_map") != timestep_map.tolist():
        raise RuntimeError("reconstructed respaced timestep map differs from the manifest")
    components = observer.build_component_specs(original_alpha_bar, timestep_map, heat_shifts)
    reconstructed_components = [
        observer._mapping_manifest_record(observer.EVIDENCE_INTERNAL_TIMESTEPS, component)
        for component in components
    ]
    if reconstructed_components != manifest_components:
        raise RuntimeError("reconstructed heat-shift components differ from the manifest")
    expected_weight = 1.0 / len(components)
    if any(
        not math.isclose(component.mixture_weight, expected_weight, rel_tol=0.0, abs_tol=1e-15)
        for component in components
    ):
        raise RuntimeError("this summarizer requires the frozen fixed equal-weight heat mixture")

    checkpoints = manifest.get("checkpoints")
    baseline_record = manifest.get("pure_p_baseline")
    if not isinstance(checkpoints, dict) or not isinstance(baseline_record, dict):
        raise RuntimeError("manifest checkpoint or pure-P baseline record is missing")
    baseline = observer.load_baseline_reference(
        Path(str(baseline_record.get("root", ""))),
        protocol,
        expected_model_sha256=str(checkpoints.get("diffusion", {}).get("sha256", "")),
        expected_classifier_sha256=str(checkpoints.get("classifier", {}).get("sha256", "")),
    )
    expected_baseline = {
        "manifest_identity_sha256": baseline.manifest_identity_sha256,
        "runner_sha256": baseline.runner_sha256,
        "pair_set_sha256": baseline.pair_set_sha256,
    }
    if any(baseline_record.get(key) != value for key, value in expected_baseline.items()):
        raise RuntimeError("resolved pure-P baseline identity differs from the observe manifest")

    completion = observer.validate_existing_completion(
        run_dir / "completion.json",
        manifest_identity_sha256=manifest_identity,
        pair_set_sha256=expected_pair_sha,
        total_expected=len(expected_pairs),
    )
    if completion is None:
        raise RuntimeError("strict summarization requires completion.json")
    if completion.get("interventions") != 0:
        raise RuntimeError("completion record is not observe-only")
    generated = completion.get("generated_this_run")
    already = completion.get("already_complete")
    if not isinstance(generated, int) or not isinstance(already, int) or generated + already != len(
        expected_pairs
    ):
        raise RuntimeError("completion generated/already-complete accounting is inconsistent")

    validated_pairs = observer.validate_observed_output_set(
        run_dir,
        baseline,
        expected_pairs,
        manifest_identity,
        current_runner_sha,
        components,
        max_conditional_kl,
        alpha,
        require_all=True,
    )
    if validated_pairs != set(expected_pairs):
        raise AssertionError("strict output validator returned the wrong pair set")

    signals: list[dict[str, Any]] = []
    for pair in expected_pairs:
        signal_path = observer.signal_pair_path(run_dir, pair)
        signal = load_json_object(signal_path)
        # The full output validator above already called this.  Repeating the
        # payload validator here keeps the loaded in-memory object inside the
        # same trust boundary used for aggregation.
        observer.validate_signal_payload(
            signal,
            pair,
            manifest_identity,
            current_runner_sha,
            baseline.manifest_identity_sha256,
            components,
            max_conditional_kl,
            alpha,
        )
        signals.append(signal)

    validation = {
        "manifest_identity_sha256": manifest_identity,
        "manifest_self_hash_validated": True,
        "completion_validated": True,
        "completion_has_self_hash": False,
        "completion_note": (
            "schema-2 completion.json has no self-hash; its manifest identity, pair-set identity, "
            "complete flag, and counts were cross-validated"
        ),
        "signal_payload_self_hashes_validated": len(signals),
        "png_metadata_and_decoded_hashes_validated": len(signals),
        "pure_p_decoded_pixel_identity_validated": len(signals),
        "frozen_observe_runner_sha256": current_runner_sha,
        "frozen_evidence_primitives_sha256": evidence_record["sha256"],
        "guided_diffusion_source_tree_validated": True,
        "expected_pair_count": len(expected_pairs),
        "validated_pair_count": len(validated_pairs),
    }
    return manifest, completion, protocol, components, signals, validation


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("distribution input must be one-dimensional")
    if array.size == 0:
        result: dict[str, float | int | None] = {
            "count": 0,
            "min": None,
            **{name: None for name, _ in QUANTILES},
            "max": None,
            "mean": None,
            "population_std": None,
        }
        return result
    if not np.isfinite(array).all():
        raise RuntimeError("cannot summarize non-finite values")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        **{name: float(np.quantile(array, quantile)) for name, quantile in QUANTILES},
        "max": float(array.max()),
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
    }


def optional_distribution(values: Iterable[float | None]) -> dict[str, Any]:
    raw = list(values)
    defined = [float(value) for value in raw if value is not None]
    result = distribution(defined)
    result["undefined_count"] = len(raw) - len(defined)
    return result


def flatten_distribution(prefix: str, values: Iterable[float]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in distribution(values).items()}


def path_standardized_innovation(projections: Sequence[float], kls: Sequence[float]) -> float | None:
    total_kl = math.fsum(kls)
    if total_kl <= 0:
        return None
    return math.fsum(projections) / math.sqrt(2.0 * total_kl)


def checkpoint_standardized_innovations(
    projections: Sequence[float],
    kls: Sequence[float],
) -> list[float]:
    result: list[float] = []
    for projection, conditional_kl in zip(projections, kls):
        if conditional_kl < 0:
            raise RuntimeError("negative conditional KL reached standardized-innovation code")
        if conditional_kl > 0:
            result.append(projection / math.sqrt(2.0 * conditional_kl))
        elif abs(projection) > 2e-14:
            raise RuntimeError("zero-KL checkpoint has a nonzero innovation projection")
    return result


def log_equal_weight_mixture(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot mix an empty component set")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values) / len(values))


def reconstruct_component(
    events: Sequence[dict[str, Any]],
    cap: float,
    threshold: float,
) -> dict[str, Any]:
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("reconstruction cap must be finite and positive")
    cumulative = 0.0
    running_max = 0.0
    first_crossing: int | None = None
    reconstructed_events: list[dict[str, Any]] = []
    for expected_index, event in enumerate(events):
        if event.get("checkpoint_index") != expected_index:
            raise RuntimeError("component events are not in checkpoint order")
        raw_kl = float(event["raw_conditional_kl"])
        raw_projection = float(event["raw_innovation_projection"])
        if raw_kl < 0:
            if raw_kl >= -1e-14:
                raw_kl = 0.0
            else:
                raise RuntimeError("raw conditional KL is negative")
        scale = 1.0 if raw_kl == 0.0 else min(1.0, math.sqrt(cap / raw_kl))
        applied_kl = scale * scale * raw_kl
        projection = scale * raw_projection
        increment = projection - applied_kl
        cumulative += increment
        running_max = max(running_max, cumulative)
        crossed = cumulative >= threshold
        if crossed and first_crossing is None:
            first_crossing = expected_index
        reconstructed_events.append(
            {
                "checkpoint_index": expected_index,
                "internal_timestep": int(event["internal_timestep"]),
                "scale": scale,
                "applied_conditional_kl": applied_kl,
                "innovation_projection": projection,
                "log_lr_increment": increment,
                "cumulative_log_e": cumulative,
                "running_max_log_e": running_max,
                "crossed_threshold_at_checkpoint": crossed,
            }
        )
    return {
        "events": reconstructed_events,
        "final_log_e": cumulative,
        "running_max_log_e": running_max,
        "first_crossing_checkpoint_index": first_crossing,
    }


def assert_primary_component_reconstruction(
    stored: dict[str, Any],
    reconstructed: dict[str, Any],
) -> None:
    for stored_event, rebuilt_event in zip(stored["events"], reconstructed["events"]):
        comparisons = {
            "tempering_scale": rebuilt_event["scale"],
            "applied_conditional_kl": rebuilt_event["applied_conditional_kl"],
            "innovation_projection": rebuilt_event["innovation_projection"],
            "log_lr_increment": rebuilt_event["log_lr_increment"],
            "cumulative_log_e": rebuilt_event["cumulative_log_e"],
            "running_max_log_e": rebuilt_event["running_max_log_e"],
        }
        if any(
            not observer._close(float(stored_event.get(key, math.nan)), expected)
            for key, expected in comparisons.items()
        ):
            raise RuntimeError(
                "raw sufficient statistics do not reconstruct the stored primary component"
            )
        if stored_event.get("crossed_threshold_at_checkpoint") != rebuilt_event[
            "crossed_threshold_at_checkpoint"
        ]:
            raise RuntimeError("primary component crossing reconstruction mismatch")
    if not observer._close(
        float(stored.get("final_cumulative_log_e", math.nan)), reconstructed["final_log_e"]
    ) or not observer._close(
        float(stored.get("running_max_log_e", math.nan)), reconstructed["running_max_log_e"]
    ):
        raise RuntimeError("primary component endpoint reconstruction mismatch")
    if stored.get("first_crossing_checkpoint_index") != reconstructed[
        "first_crossing_checkpoint_index"
    ]:
        raise RuntimeError("primary component first-crossing reconstruction mismatch")


def reconstruct_mixture(
    component_reconstructions: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    if not component_reconstructions:
        raise ValueError("mixture needs at least one component")
    checkpoint_count = len(component_reconstructions[0]["events"])
    if any(len(component["events"]) != checkpoint_count for component in component_reconstructions):
        raise RuntimeError("component reconstructions have different checkpoint counts")
    events: list[dict[str, Any]] = []
    running_max = 0.0
    first_crossing: int | None = None
    for checkpoint_index in range(checkpoint_count):
        values = [
            float(component["events"][checkpoint_index]["cumulative_log_e"])
            for component in component_reconstructions
        ]
        log_e = log_equal_weight_mixture(values)
        running_max = max(running_max, log_e)
        crossed = log_e >= threshold
        if crossed and first_crossing is None:
            first_crossing = checkpoint_index
        events.append(
            {
                "checkpoint_index": checkpoint_index,
                "internal_timestep": int(
                    component_reconstructions[0]["events"][checkpoint_index]["internal_timestep"]
                ),
                "log_e_mixture": log_e,
                "running_max_log_e": running_max,
                "crossed_threshold_at_checkpoint": crossed,
            }
        )
    return {
        "events": events,
        "final_log_e": log_equal_weight_mixture(
            [float(component["final_log_e"]) for component in component_reconstructions]
        ),
        "running_max_log_e": running_max,
        "first_crossing_checkpoint_index": first_crossing,
    }


def assert_primary_mixture_reconstruction(
    stored: dict[str, Any],
    reconstructed: dict[str, Any],
) -> None:
    for stored_event, rebuilt_event in zip(stored["events"], reconstructed["events"]):
        if not observer._close(
            float(stored_event.get("log_e_mixture", math.nan)), rebuilt_event["log_e_mixture"]
        ) or not observer._close(
            float(stored_event.get("running_max_log_e_mixture", math.nan)),
            rebuilt_event["running_max_log_e"],
        ):
            raise RuntimeError("raw sufficient statistics do not reconstruct the stored mixture")
        if stored_event.get("crossed_threshold_at_checkpoint") != rebuilt_event[
            "crossed_threshold_at_checkpoint"
        ]:
            raise RuntimeError("primary mixture crossing reconstruction mismatch")
    if not observer._close(
        float(stored.get("final_log_e", math.nan)), reconstructed["final_log_e"]
    ) or not observer._close(
        float(stored.get("running_max_log_e", math.nan)), reconstructed["running_max_log_e"]
    ):
        raise RuntimeError("primary mixture endpoint reconstruction mismatch")
    if stored.get("first_crossing_checkpoint_index") != reconstructed[
        "first_crossing_checkpoint_index"
    ]:
        raise RuntimeError("primary mixture first-crossing reconstruction mismatch")


def component_summary_row(
    signal: dict[str, Any],
    component: observer.ComponentSpec,
    stored: dict[str, Any],
    reconstructed: dict[str, Any],
    cap: float,
    role: str,
    signal_path: Path,
    *,
    include_distributions: bool,
    is_manifest_primary_cap: bool,
) -> dict[str, Any]:
    events = stored["events"]
    rebuilt_events = reconstructed["events"]
    active_mask = [bool(event["shifted_model_evaluated"]) for event in events]
    active_count = sum(active_mask)
    identity_count = len(events) - active_count
    expected_active = int(np.count_nonzero(
        component.mapping.shifted_timestep != component.mapping.current_timestep
    ))
    if active_count != expected_active or identity_count != len(events) - expected_active:
        raise RuntimeError("sample active/identity checkpoint counts differ from the frozen mapping")
    raw_kls = [float(event["raw_conditional_kl"]) for event in events]
    raw_projections = [float(event["raw_innovation_projection"]) for event in events]
    applied_kls = [float(event["applied_conditional_kl"]) for event in rebuilt_events]
    applied_projections = [float(event["innovation_projection"]) for event in rebuilt_events]
    active_raw_kls = [value for value, active in zip(raw_kls, active_mask) if active]
    active_raw_projections = [
        value for value, active in zip(raw_projections, active_mask) if active
    ]
    active_applied_kls = [value for value, active in zip(applied_kls, active_mask) if active]
    active_applied_projections = [
        value for value, active in zip(applied_projections, active_mask) if active
    ]
    capped = [raw_kl > cap for raw_kl, active in zip(raw_kls, active_mask) if active]
    raw_z = checkpoint_standardized_innovations(active_raw_projections, active_raw_kls)
    applied_z = checkpoint_standardized_innovations(
        active_applied_projections, active_applied_kls
    )
    if len(raw_z) == len(applied_z) and any(
        not observer._close(left, right) for left, right in zip(raw_z, applied_z)
    ):
        raise RuntimeError("raw/applied per-checkpoint standardized innovations disagree")
    first = reconstructed["first_crossing_checkpoint_index"]
    row: dict[str, Any] = {
        "analysis_role": role,
        "is_manifest_primary_cap_value": is_manifest_primary_cap,
        "class_id": int(signal["class_id"]),
        "seed": int(signal["seed"]),
        "component_index": component.index,
        "component_id": component.component_id,
        "additive_heat_shift": component.additive_heat_shift,
        "mixture_weight": component.mixture_weight,
        "conditional_kl_cap": cap,
        "checkpoint_count": len(events),
        "active_checkpoint_count": active_count,
        "identity_checkpoint_count": identity_count,
        "positive_raw_kl_checkpoint_count": len(raw_z),
        "cap_saturated_checkpoint_count": int(sum(capped)),
        "cap_saturation_rate_active": float(sum(capped) / active_count) if active_count else None,
        "cumulative_raw_conditional_kl": math.fsum(raw_kls),
        "cumulative_applied_conditional_kl": math.fsum(applied_kls),
        "cumulative_raw_innovation_projection": math.fsum(raw_projections),
        "cumulative_applied_innovation_projection": math.fsum(applied_projections),
        "raw_path_standardized_innovation": path_standardized_innovation(
            raw_projections, raw_kls
        ),
        "applied_path_standardized_innovation": path_standardized_innovation(
            applied_projections, applied_kls
        ),
        "final_log_e": float(reconstructed["final_log_e"]),
        "running_max_log_e": float(reconstructed["running_max_log_e"]),
        "crossed_threshold_ever": first is not None,
        "first_crossing_checkpoint_index": first,
        "first_crossing_internal_timestep": (
            int(events[first]["internal_timestep"]) if first is not None else None
        ),
        "signal_path": str(signal_path),
    }
    if include_distributions:
        row.update(flatten_distribution("raw_kl_all", raw_kls))
        row.update(flatten_distribution("raw_kl_active", active_raw_kls))
        row.update(flatten_distribution("raw_innovation_projection_active", active_raw_projections))
        row.update(
            flatten_distribution(
                "applied_innovation_projection_active", active_applied_projections
            )
        )
        row.update(flatten_distribution("raw_standardized_innovation_active", raw_z))
        row.update(flatten_distribution("applied_standardized_innovation_active", applied_z))
    return row


def mixture_summary_row(
    signal: dict[str, Any],
    reconstructed: dict[str, Any],
    component_reconstructions: Sequence[dict[str, Any]],
    cap: float,
    role: str,
    *,
    is_manifest_primary_cap: bool,
) -> dict[str, Any]:
    first = reconstructed["first_crossing_checkpoint_index"]
    internal_timesteps = signal["internal_timesteps"]
    component_firsts = [
        component["first_crossing_checkpoint_index"] for component in component_reconstructions
    ]
    return {
        "analysis_role": role,
        "is_manifest_primary_cap_value": is_manifest_primary_cap,
        "class_id": int(signal["class_id"]),
        "seed": int(signal["seed"]),
        "conditional_kl_cap": cap,
        "component_count": len(signal["components"]),
        "mixture_policy": "fixed_equal_weight_arithmetic_mixture_of_e_processes",
        "final_log_e": float(reconstructed["final_log_e"]),
        "running_max_log_e": float(reconstructed["running_max_log_e"]),
        "crossed_threshold_ever": first is not None,
        "first_crossing_checkpoint_index": first,
        "first_crossing_internal_timestep": (
            int(internal_timesteps[first]) if first is not None else None
        ),
        "component_crossing_count": int(
            sum(value is not None for value in component_firsts)
        ),
    }


def summarize_signals(
    run_dir: Path,
    signals: Sequence[dict[str, Any]],
    components: Sequence[observer.ComponentSpec],
    primary_cap: float,
    alpha: float,
    cap_grid: Sequence[float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    threshold = -math.log(alpha)
    primary_component_rows: list[dict[str, Any]] = []
    primary_mixture_rows: list[dict[str, Any]] = []
    cap_component_rows: list[dict[str, Any]] = []
    cap_mixture_rows: list[dict[str, Any]] = []

    for signal in signals:
        pair = int(signal["class_id"]), int(signal["seed"])
        signal_path = observer.signal_pair_path(run_dir, pair)
        primary_reconstructed: list[dict[str, Any]] = []
        for component, stored in zip(components, signal["components"]):
            rebuilt = reconstruct_component(stored["events"], primary_cap, threshold)
            assert_primary_component_reconstruction(stored, rebuilt)
            primary_reconstructed.append(rebuilt)
            primary_component_rows.append(
                component_summary_row(
                    signal,
                    component,
                    stored,
                    rebuilt,
                    primary_cap,
                    PRIMARY_ROLE,
                    signal_path,
                    include_distributions=True,
                    is_manifest_primary_cap=True,
                )
            )
        primary_mix = reconstruct_mixture(primary_reconstructed, threshold)
        assert_primary_mixture_reconstruction(signal["mixture"], primary_mix)
        primary_mixture_rows.append(
            mixture_summary_row(
                signal,
                primary_mix,
                primary_reconstructed,
                primary_cap,
                PRIMARY_ROLE,
                is_manifest_primary_cap=True,
            )
        )

        for cap in cap_grid:
            reconstructed_components: list[dict[str, Any]] = []
            same_as_primary = cap == primary_cap
            for component, stored in zip(components, signal["components"]):
                rebuilt = reconstruct_component(stored["events"], cap, threshold)
                reconstructed_components.append(rebuilt)
                cap_component_rows.append(
                    component_summary_row(
                        signal,
                        component,
                        stored,
                        rebuilt,
                        cap,
                        EXPLORATORY_ROLE,
                        signal_path,
                        include_distributions=False,
                        is_manifest_primary_cap=same_as_primary,
                    )
                )
            cap_mix = reconstruct_mixture(reconstructed_components, threshold)
            cap_mixture_rows.append(
                mixture_summary_row(
                    signal,
                    cap_mix,
                    reconstructed_components,
                    cap,
                    EXPLORATORY_ROLE,
                    is_manifest_primary_cap=same_as_primary,
                )
            )
    return primary_component_rows, primary_mixture_rows, cap_component_rows, cap_mixture_rows


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("rank input must be a finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pairwise_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def correlation_report(
    component_rows: Sequence[dict[str, Any]],
    components: Sequence[observer.ComponentSpec],
) -> dict[str, Any]:
    by_pair: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in component_rows:
        by_pair[(int(row["class_id"]), int(row["seed"]))][int(row["component_index"])] = row
    expected_indices = {component.index for component in components}
    if any(set(records) != expected_indices for records in by_pair.values()):
        raise RuntimeError("component correlation table has an incomplete path")
    pairs = sorted(by_pair)
    labels = [f"{component.component_id}:delta={component.additive_heat_shift:g}" for component in components]
    metrics = (
        "final_log_e",
        "running_max_log_e",
        "cumulative_applied_conditional_kl",
        "cap_saturation_rate_active",
        "raw_path_standardized_innovation",
        "applied_path_standardized_innovation",
    )
    reports: dict[str, Any] = {}
    for metric in metrics:
        columns = [
            [by_pair[pair][component.index][metric] for pair in pairs]
            for component in components
        ]
        if any(any(value is None for value in column) for column in columns):
            reports[metric] = {
                "pearson": None,
                "spearman": None,
                "note": "one or more components have undefined values",
            }
            continue
        numeric = [[float(value) for value in column] for column in columns]
        pearson = [
            [pairwise_correlation(left, right) for right in numeric] for left in numeric
        ]
        ranked = [average_ranks(column).tolist() for column in numeric]
        spearman = [
            [pairwise_correlation(left, right) for right in ranked] for left in ranked
        ]
        reports[metric] = {"pearson": pearson, "spearman": spearman}
    return {
        "sample_count": len(pairs),
        "component_labels": labels,
        "matrices": reports,
        "interpretation": (
            "Descriptive pooled correlations only. Shared seed noise couples classes; no p-values "
            "or independent-sample inference are implied. Null entries mean a constant or undefined column."
        ),
    }


def grouped_primary_rows(
    group_name: str,
    mixture_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    components: Sequence[observer.ComponentSpec],
) -> list[dict[str, Any]]:
    if group_name not in {"class_id", "seed"}:
        raise ValueError("group_name must be class_id or seed")
    mixture_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    component_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in mixture_rows:
        mixture_groups[int(row[group_name])].append(row)
    for row in component_rows:
        component_groups[int(row[group_name])].append(row)
    results: list[dict[str, Any]] = []
    for group_value in sorted(mixture_groups):
        mixtures = mixture_groups[group_value]
        rows = component_groups[group_value]
        result: dict[str, Any] = {
            group_name: group_value,
            "sample_count": len(mixtures),
            "mixture_crossing_count": int(sum(bool(row["crossed_threshold_ever"]) for row in mixtures)),
            "mixture_crossing_rate": float(
                sum(bool(row["crossed_threshold_ever"]) for row in mixtures) / len(mixtures)
            ),
            **flatten_distribution("mixture_final_log_e", (row["final_log_e"] for row in mixtures)),
            **flatten_distribution(
                "mixture_running_max_log_e", (row["running_max_log_e"] for row in mixtures)
            ),
        }
        for component in components:
            selected = [row for row in rows if row["component_index"] == component.index]
            if len(selected) != len(mixtures):
                raise RuntimeError("grouped component table is incomplete")
            prefix = component.component_id
            result[f"{prefix}_additive_heat_shift"] = component.additive_heat_shift
            result[f"{prefix}_crossing_count"] = int(
                sum(bool(row["crossed_threshold_ever"]) for row in selected)
            )
            result[f"{prefix}_crossing_rate"] = float(
                sum(bool(row["crossed_threshold_ever"]) for row in selected) / len(selected)
            )
            result[f"{prefix}_mean_final_log_e"] = float(
                np.mean([row["final_log_e"] for row in selected])
            )
            result[f"{prefix}_mean_running_max_log_e"] = float(
                np.mean([row["running_max_log_e"] for row in selected])
            )
            result[f"{prefix}_mean_cumulative_applied_kl"] = float(
                np.mean([row["cumulative_applied_conditional_kl"] for row in selected])
            )
            result[f"{prefix}_mean_cap_saturation_rate_active"] = float(
                np.mean([row["cap_saturation_rate_active"] for row in selected])
            )
        results.append(result)
    return results


def crossing_histogram(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, int] = defaultdict(int)
    by_internal_timestep: dict[str, int] = defaultdict(int)
    not_crossed = 0
    for row in rows:
        checkpoint = row["first_crossing_checkpoint_index"]
        internal_timestep = row["first_crossing_internal_timestep"]
        if checkpoint is None:
            if internal_timestep is not None:
                raise RuntimeError("non-crossing row has a first-crossing timestep")
            not_crossed += 1
            continue
        if internal_timestep is None:
            raise RuntimeError("crossing row lacks its first-crossing timestep")
        by_checkpoint[str(int(checkpoint))] += 1
        by_internal_timestep[str(int(internal_timestep))] += 1
    return {
        "crossed_count": len(rows) - not_crossed,
        "not_crossed_count": not_crossed,
        "by_checkpoint_index": dict(sorted(by_checkpoint.items(), key=lambda item: int(item[0]))),
        "by_internal_timestep": dict(
            sorted(by_internal_timestep.items(), key=lambda item: int(item[0]), reverse=True)
        ),
    }


def global_primary_summary(
    mixture_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    components: Sequence[observer.ComponentSpec],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": len(mixture_rows),
        "mixture": {
            "crossing_count": int(sum(bool(row["crossed_threshold_ever"]) for row in mixture_rows)),
            "crossing_rate": float(
                sum(bool(row["crossed_threshold_ever"]) for row in mixture_rows)
                / len(mixture_rows)
            ),
            "final_log_e": distribution(row["final_log_e"] for row in mixture_rows),
            "running_max_log_e": distribution(row["running_max_log_e"] for row in mixture_rows),
            "first_crossing_histogram": crossing_histogram(mixture_rows),
        },
        "components": {},
    }
    for component in components:
        selected = [row for row in component_rows if row["component_index"] == component.index]
        result["components"][component.component_id] = {
            "additive_heat_shift": component.additive_heat_shift,
            "active_checkpoint_count": selected[0]["active_checkpoint_count"],
            "identity_checkpoint_count": selected[0]["identity_checkpoint_count"],
            "crossing_count": int(sum(bool(row["crossed_threshold_ever"]) for row in selected)),
            "crossing_rate": float(
                sum(bool(row["crossed_threshold_ever"]) for row in selected) / len(selected)
            ),
            "cap_saturation_rate_active": distribution(
                row["cap_saturation_rate_active"] for row in selected
            ),
            "cumulative_raw_conditional_kl": distribution(
                row["cumulative_raw_conditional_kl"] for row in selected
            ),
            "cumulative_applied_conditional_kl": distribution(
                row["cumulative_applied_conditional_kl"] for row in selected
            ),
            "raw_path_standardized_innovation": optional_distribution(
                row["raw_path_standardized_innovation"] for row in selected
            ),
            "applied_path_standardized_innovation": optional_distribution(
                row["applied_path_standardized_innovation"] for row in selected
            ),
            "final_log_e": distribution(row["final_log_e"] for row in selected),
            "running_max_log_e": distribution(row["running_max_log_e"] for row in selected),
            "first_crossing_histogram": crossing_histogram(selected),
        }
    return result


def eta_squared_by_seed(mixture_rows: Sequence[dict[str, Any]], metric: str) -> float | None:
    values = np.asarray([float(row[metric]) for row in mixture_rows], dtype=np.float64)
    if values.size < 2:
        return None
    overall = float(values.mean())
    total = float(np.square(values - overall).sum())
    if total == 0:
        return None
    groups: dict[int, list[float]] = defaultdict(list)
    for row in mixture_rows:
        groups[int(row["seed"])].append(float(row[metric]))
    between = math.fsum(
        len(group) * (float(np.mean(group)) - overall) ** 2 for group in groups.values()
    )
    return float(between / total)


def seed_cluster_report(
    protocol: observer.Protocol,
    mixture_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    cluster_sizes = [sum(int(row["seed"]) == seed for row in mixture_rows) for seed in protocol.seeds]
    paired = bool(manifest.get("rng", {}).get("paired_across_classes"))
    return {
        "cluster_key": "seed",
        "seed_count": len(protocol.seeds),
        "class_count": len(protocol.class_ids),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "complete_class_by_seed_cartesian_product": all(
            size == len(protocol.class_ids) for size in cluster_sizes
        ),
        "manifest_reuses_seed_noise_across_classes": paired,
        "descriptive_seed_eta_squared": {
            "mixture_final_log_e": eta_squared_by_seed(mixture_rows, "final_log_e"),
            "mixture_running_max_log_e": eta_squared_by_seed(mixture_rows, "running_max_log_e"),
        },
        "inference_warning": (
            "The same public seed owns the initial and reverse Gaussian innovations across classes. "
            "Rows sharing a seed are therefore clustered, not independent. Use a seed-cluster or "
            "hierarchical bootstrap stratified by class for uncertainty; the eta-squared values above "
            "are descriptive hints only and are unstable with few seeds."
        ),
    }


def exploratory_cap_summary(
    cap_grid: Sequence[float],
    component_rows: Sequence[dict[str, Any]],
    mixture_rows: Sequence[dict[str, Any]],
    components: Sequence[observer.ComponentSpec],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cap in cap_grid:
        mixtures = [row for row in mixture_rows if row["conditional_kl_cap"] == cap]
        rows = [row for row in component_rows if row["conditional_kl_cap"] == cap]
        result: dict[str, Any] = {
            "analysis_role": EXPLORATORY_ROLE,
            "conditional_kl_cap": cap,
            "sample_count": len(mixtures),
            "mixture_crossing_count": int(sum(bool(row["crossed_threshold_ever"]) for row in mixtures)),
            "mixture_crossing_rate": float(
                sum(bool(row["crossed_threshold_ever"]) for row in mixtures) / len(mixtures)
            ),
            "mixture_final_log_e": distribution(row["final_log_e"] for row in mixtures),
            "mixture_running_max_log_e": distribution(
                row["running_max_log_e"] for row in mixtures
            ),
            "mixture_first_crossing_histogram": crossing_histogram(mixtures),
            "components": {},
        }
        for component in components:
            selected = [row for row in rows if row["component_index"] == component.index]
            result["components"][component.component_id] = {
                "additive_heat_shift": component.additive_heat_shift,
                "crossing_count": int(
                    sum(bool(row["crossed_threshold_ever"]) for row in selected)
                ),
                "crossing_rate": float(
                    sum(bool(row["crossed_threshold_ever"]) for row in selected) / len(selected)
                ),
                "mean_cap_saturation_rate_active": float(
                    np.mean([row["cap_saturation_rate_active"] for row in selected])
                ),
                "cumulative_applied_conditional_kl": distribution(
                    row["cumulative_applied_conditional_kl"] for row in selected
                ),
                "final_log_e": distribution(row["final_log_e"] for row in selected),
                "running_max_log_e": distribution(
                    row["running_max_log_e"] for row in selected
                ),
                "first_crossing_histogram": crossing_histogram(selected),
            }
        results.append(result)
    return results


def table_json(role: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "analysis_role": role,
        "row_count": len(rows),
        "rows": list(rows),
    }


def write_table_pair(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    stem: str,
    role: str,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    atomic_csv_dump(rows, csv_path)
    atomic_json_dump(table_json(role, rows), json_path)
    return csv_path, json_path


def prepare_output_dir(run_dir: Path, output_dir: Path) -> None:
    run_resolved = run_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == run_resolved or run_resolved in output_resolved.parents:
        raise ValueError("--output-dir must be outside --run-dir to keep the source run read-only")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing a non-empty summary output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="derived-output directory outside run-dir; defaults to a sibling named <run>_summary",
    )
    parser.add_argument(
        "--cap-grid",
        type=parse_cap_grid,
        default=(),
        help=(
            "comma-separated positive caps declared before reading the run; outputs are exploratory "
            "sensitivity diagnostics and never replace the manifest primary"
        ),
    )
    parser.add_argument(
        "--guided-diffusion-root",
        type=Path,
        default=None,
        help="optional relocated checkout; its revision and Python-tree hash must match the manifest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"observe run directory does not exist: {run_dir}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir.with_name(run_dir.name + "_summary")
    )
    prepare_output_dir(run_dir, output_dir)

    manifest, completion, protocol, components, signals, validation = validate_source_run(
        run_dir, args.guided_diffusion_root
    )
    primary = manifest["primary_evidence_definition"]
    primary_cap = float(primary["max_conditional_kl_per_component_checkpoint"])
    alpha = float(primary["alpha"])
    cap_grid = tuple(args.cap_grid)
    (
        primary_component_rows,
        primary_mixture_rows,
        cap_component_rows,
        cap_mixture_rows,
    ) = summarize_signals(
        run_dir,
        signals,
        components,
        primary_cap,
        alpha,
        cap_grid,
    )

    class_rows = grouped_primary_rows(
        "class_id", primary_mixture_rows, primary_component_rows, components
    )
    seed_rows = grouped_primary_rows(
        "seed", primary_mixture_rows, primary_component_rows, components
    )
    output_files: list[Path] = []
    for rows, stem, role in (
        (primary_component_rows, "primary_sample_components", PRIMARY_ROLE),
        (primary_mixture_rows, "primary_sample_mixtures", PRIMARY_ROLE),
        (class_rows, "primary_by_class", PRIMARY_ROLE),
        (seed_rows, "primary_by_seed", PRIMARY_ROLE),
    ):
        output_files.extend(write_table_pair(rows, output_dir, stem, role))
    if cap_grid:
        output_files.extend(
            write_table_pair(
                cap_component_rows,
                output_dir,
                "exploratory_cap_grid_sample_components",
                EXPLORATORY_ROLE,
            )
        )
        output_files.extend(
            write_table_pair(
                cap_mixture_rows,
                output_dir,
                "exploratory_cap_grid_sample_mixtures",
                EXPLORATORY_ROLE,
            )
        )

    script_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "analysis_name": "adm64_cross_scale_evidence_strict_summary",
        "analysis_script": {
            "path": str(script_path),
            "sha256": observer.sha256_file(script_path),
        },
        "source_run": str(run_dir),
        "source_manifest_identity_sha256": manifest["identity_sha256"],
        "source_completion": {
            "complete": completion["complete"],
            "total_expected": completion["total_expected"],
            "total_complete": completion["total_complete"],
            "interventions": completion["interventions"],
        },
        "validation": validation,
        "primary": {
            "analysis_role": PRIMARY_ROLE,
            "conditional_kl_cap": primary_cap,
            "alpha": alpha,
            "log_e_crossing_threshold": -math.log(alpha),
            "heat_shifts": [component.additive_heat_shift for component in components],
            "mixture_weights": [component.mixture_weight for component in components],
            "checkpoint_count": len(observer.EVIDENCE_INTERNAL_TIMESTEPS),
            "aggregate": global_primary_summary(
                primary_mixture_rows, primary_component_rows, components
            ),
            "component_correlations": correlation_report(primary_component_rows, components),
        },
        "seed_clustering": seed_cluster_report(protocol, primary_mixture_rows, manifest),
        "exploratory_cap_grid": {
            "analysis_role": EXPLORATORY_ROLE,
            "predeclared_command_line_values": list(cap_grid),
            "grid_sha256": observer.sha256_json(list(cap_grid)),
            "fixed_equal_weight_heat_mixture": True,
            "reconstruction_formula": (
                "gamma=1 if K=0 else min(1,sqrt(kappa/K)); "
                "Delta_log_E=gamma*R-gamma^2*K"
            ),
            "primary_replacement_allowed": False,
            "validity_warning": (
                "A cap chosen after inspecting endpoint labels or these same-path crossing outcomes "
                "is not the manifest primary and cannot inherit its original anytime-valid claim. "
                "Use a preregistered cap, independent calibration/holdout, or a predeclared cap mixture."
            ),
            "aggregate": exploratory_cap_summary(
                cap_grid, cap_component_rows, cap_mixture_rows, components
            ) if cap_grid else [],
        },
        "quality_scope": (
            "This summary contains path-evidence diagnostics only. It performs no visual inspection, "
            "uses no artifact labels, and makes no image-quality or bad-case conclusion."
        ),
        "definitions": {
            "active_checkpoint": "shifted_original_timestep differs from current and shifted U-Net was evaluated",
            "identity_checkpoint": "discrete heat mapping is identity; Q=P and K=R=increment=0",
            "cap_saturation": "an active checkpoint with raw K strictly greater than the stated cap",
            "checkpoint_standardized_innovation": "projection/sqrt(2*conditional_KL), defined only for KL>0",
            "path_standardized_innovation": "sum(projection)/sqrt(2*sum(conditional_KL))",
            "running_max_initial_value": 0.0,
            "mixture": "fixed equal-weight arithmetic mixture in E-space, computed by log-mean-exp",
        },
    }
    summary["derived_files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": observer.sha256_file(path)}
        for path in output_files
    }
    summary["payload_sha256"] = observer._canonical_payload_sha(summary, "payload_sha256")
    atomic_json_dump(summary, output_dir / "summary.json")

    print(
        json.dumps(
            {
                "source_run": str(run_dir),
                "output_dir": str(output_dir),
                "validated_samples": len(signals),
                "primary_cap": primary_cap,
                "primary_mixture_crossings": summary["primary"]["aggregate"]["mixture"][
                    "crossing_count"
                ],
                "exploratory_cap_grid": list(cap_grid),
                "quality_conclusions": 0,
                "summary_payload_sha256": summary["payload_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

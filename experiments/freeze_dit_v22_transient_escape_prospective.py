#!/usr/bin/env python3
"""Freeze the label-free prospective h10 transient-escape experiment.

The freezer is deliberately allowed to inspect only mechanical manifests and
the frozen JSON protocol.  It never opens endpoint pixels, review products,
quality tables, or external representations.  Candidate prefixes are chosen by
a fixed hash over seed/class/step metadata; the generated suffix runner and all
512 fresh RNG streams are bound before any prospective GPU output exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/dit_v22_transient_escape_prospective_v1.json"
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
DEFAULT_OLD_SUFFIX_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence"
)
DEFAULT_SOURCE_LOCK = ROOT / "experiments/locks/dit_v22_repairability_execution_source_lock_v1"
DEFAULT_OUTPUT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_prospective_v1_outputs"
)
DEFAULT_RECEIPT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_prospective_v1_receipts"
)

LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_2"
EXPECTED_POOL_ID = "d7f98049c43383fede2d0eec685c2d75d09470bf623aa31cdfff0ec000bfd022"
NEW_RUNNER_NAME = "intervene_dit_v22_transient_escape_suffix"
NEW_EXPERIMENT = "dit_v22_transient_escape_prospective_suffix"
NEW_RNG_NAMESPACE = "eqvae-dit-v22-h10-max-nonconformity-prospective-v1"
SLOT_NAMESPACE = "eqvae-dit-v22-h10-anonymous-slot-v1"
RANDOM_CONTROL_NAMESPACE = "eqvae-dit-v22-h10-random-control-v1"
CLASS_SLOTS = {207: 0, 602: 1, 795: 2}
TRACE_FILENAME = "trace.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hash(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return canonical_sha256(payload)


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    if value.get(key) != self_hash(value, key):
        raise RuntimeError(f"self hash failed: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_source_lock(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != "DIT_V22_REPAIRABILITY_EXECUTION_SOURCE_LOCK_V1":
        raise RuntimeError("unexpected frozen execution source lock")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("source lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            path.is_symlink()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"source lock member changed: {name}")
    return manifest


def trace_output_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and row.get("relative_path") == TRACE_FILENAME
    ]
    if len(matches) != 1:
        raise RuntimeError("trace manifest does not contain exactly one trace.npz")
    return dict(matches[0])


def load_trace_catalog(trace_root: Path, pool_id: str) -> dict[int, dict[str, Any]]:
    if pool_id != EXPECTED_POOL_ID:
        raise RuntimeError("third-pool identity changed")
    catalog: dict[int, dict[str, Any]] = {}
    for seed in range(250, 850):
        root = trace_root / f"third_pool_v1_seed{seed}"
        manifest_path = root / "manifest.json"
        completion_path = root / "completion.json"
        trace_path = root / TRACE_FILENAME
        manifest = load_json(manifest_path)
        if manifest.get("identity_sha256") != canonical_sha256(manifest.get("identity")):
            raise RuntimeError(f"trace manifest identity hash failed for seed {seed}")
        identity = manifest.get("identity", {})
        protocol = identity.get("protocol", {}) if isinstance(identity, dict) else {}
        if (
            manifest.get("status") != "complete"
            or protocol.get("global_torch_seed") != seed
            or protocol.get("class_ids_ordered") != [207, 602, 795]
            or protocol.get("sampling_steps") != 250
            or identity.get("quality_score") is not None
            or identity.get("selection") is not None
        ):
            raise RuntimeError(f"trace identity/scope changed for seed {seed}")
        trace_record = trace_output_record(manifest)
        if (
            trace_path.is_symlink()
            or not trace_path.is_file()
            or trace_record.get("bytes") != trace_path.stat().st_size
        ):
            raise RuntimeError(f"trace payload changed for seed {seed}")
        checkpoint = identity.get("checkpoint", {})
        vae = identity.get("vae_snapshot", {})
        source = identity.get("source", {})
        catalog[seed] = {
            "root": str(root.resolve()),
            "global_seed": seed,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "completion_file_sha256": sha256_file(completion_path),
            "trace_npz_sha256": trace_record["sha256"],
            "trace_npz_bytes": trace_record["bytes"],
            "checkpoint_sha256": checkpoint.get("sha256"),
            "vae_revision": vae.get("revision"),
            "source_revision": source.get("revision"),
        }
    if set(catalog) != set(range(250, 850)):
        raise AssertionError("third-pool seed axis changed")
    return catalog


def discover_prior_suffixes(root: Path) -> tuple[list[dict[str, Any]], set[tuple[int, int, int]], set[int]]:
    manifests: list[dict[str, Any]] = []
    exact: set[tuple[int, int, int]] = set()
    fresh_seeds: set[int] = set()
    paths = sorted(root.glob("dit_v22_repairability*/**/manifest.json"))
    for path in paths:
        try:
            value = load_json(path)
        except (json.JSONDecodeError, RuntimeError):
            continue
        if value.get("runner") != "intervene_dit_v22_custom_trace_suffix":
            continue
        target = value.get("target", {})
        rollback = value.get("rollback", {})
        seed = target.get("global_seed")
        class_id = target.get("class_id")
        step = rollback.get("sampling_step_index_zero_based")
        if not all(type(item) is int for item in (seed, class_id, step)):
            raise RuntimeError(f"malformed prior suffix target: {path}")
        exact.add((seed, class_id, step))
        streams = value.get("branches", {}).get("fresh_stream_seeds", [])
        if len(streams) != 4:
            raise RuntimeError(f"prior suffix stream inventory changed: {path}")
        values = [row.get("seed") for row in streams]
        if not all(type(item) is int for item in values):
            raise RuntimeError(f"malformed prior suffix stream: {path}")
        fresh_seeds.update(values)
        manifests.append(
            {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "manifest_identity_sha256": value.get("identity_sha256"),
                "global_seed": seed,
                "class_id": class_id,
                "rollback_step": step,
                "rng_namespace": value.get("rng", {}).get("namespace"),
                "fresh_stream_seeds": values,
            }
        )
    if len(manifests) != 34 or len(exact) != 32 or len(fresh_seeds) != 128:
        raise RuntimeError(
            "prior suffix inventory changed; expected 34 manifests, 32 exact jobs, 128 streams"
        )
    return manifests, exact, fresh_seeds


def selection_score(salt: str, seed: int, class_id: int, step: int) -> str:
    payload = f"{salt}|seed={seed:03d}|class={class_id:04d}|step={step}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def branch_seed(
    namespace: str,
    trace_identity: str,
    seed: int,
    step: int,
    slot: int,
    attempt: int,
) -> int:
    payload = (
        f"{namespace}\0{trace_identity}\0{seed}\0{step}\0{slot}\0{attempt}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def anonymous_slots(trace_identity: str, seed: int, class_id: int) -> dict[int, str]:
    ranked = []
    for attempt in range(1, 5):
        digest = hashlib.sha256(
            f"{SLOT_NAMESPACE}\0{trace_identity}\0{seed}\0{class_id}\0{attempt}".encode(
                "ascii"
            )
        ).hexdigest()
        ranked.append((digest, attempt))
    return {attempt: "ABCD"[position] for position, (_, attempt) in enumerate(sorted(ranked))}


def random_control_slot(trace_identity: str, seed: int, class_id: int) -> str:
    digest = hashlib.sha256(
        f"{RANDOM_CONTROL_NAMESPACE}\0{trace_identity}\0{seed}\0{class_id}".encode(
            "ascii"
        )
    ).digest()
    return "ABCD"[int.from_bytes(digest[:8], "big") % 4]


def select_jobs(
    config: Mapping[str, Any],
    catalog: Mapping[int, Mapping[str, Any]],
    excluded: set[tuple[int, int, int]],
    output_root: Path,
) -> tuple[str, list[dict[str, Any]]]:
    step = int(config["model_scope"]["rollback_sampling_step_zero_based"])
    quotas = {int(key): int(value) for key, value in config["prefix_population"]["class_quotas"].items()}
    if quotas != {207: 16, 602: 16, 795: 96} or sum(quotas.values()) != 128:
        raise RuntimeError("frozen class quotas changed")
    salt = (
        f"{config['prefix_population']['selection_namespace']}|pool={EXPECTED_POOL_ID}|"
        f"rollback={step}"
    )
    candidates = sorted(
        (
            selection_score(salt, seed, class_id, step),
            seed,
            class_id,
        )
        for seed in catalog
        for class_id in sorted(CLASS_SLOTS)
        if (seed, class_id, step) not in excluded
    )
    counts = {class_id: 0 for class_id in quotas}
    used_seeds: set[int] = set()
    selected: list[tuple[str, int, int]] = []
    for digest, seed, class_id in candidates:
        if counts[class_id] >= quotas[class_id] or seed in used_seeds:
            continue
        selected.append((digest, seed, class_id))
        counts[class_id] += 1
        used_seeds.add(seed)
        if counts == quotas:
            break
    if counts != quotas or len(selected) != 128 or len(used_seeds) != 128:
        raise RuntimeError("hash-only selection did not fill the frozen design")
    jobs: list[dict[str, Any]] = []
    for job_index, (digest, seed, class_id) in enumerate(selected):
        trace = dict(catalog[seed])
        trace_path = Path(trace["root"]) / TRACE_FILENAME
        if sha256_file(trace_path) != trace["trace_npz_sha256"]:
            raise RuntimeError(f"selected trace payload changed for seed {seed}")
        slot = CLASS_SLOTS[class_id]
        slot_map = anonymous_slots(trace["manifest_identity_sha256"], seed, class_id)
        inverse_slots = {letter: attempt for attempt, letter in slot_map.items()}
        random_slot = random_control_slot(trace["manifest_identity_sha256"], seed, class_id)
        streams = [
            branch_seed(
                NEW_RNG_NAMESPACE,
                trace["manifest_identity_sha256"],
                seed,
                step,
                slot,
                attempt,
            )
            for attempt in range(1, 5)
        ]
        jobs.append(
            {
                "job_index": job_index,
                "selection_sha256": digest,
                "global_seed": seed,
                "class_id": class_id,
                "class_slot": slot,
                "rollback_sampling_step": step,
                "trace_dir": trace["root"],
                "trace_identity_sha256": trace["manifest_identity_sha256"],
                "trace_manifest_file_sha256": trace["manifest_file_sha256"],
                "trace_completion_file_sha256": trace["completion_file_sha256"],
                "trace_npz_sha256": trace["trace_npz_sha256"],
                "physical_attempt_to_anonymous_slot": {
                    str(key): value for key, value in sorted(slot_map.items())
                },
                "anonymous_slot_to_physical_attempt": {
                    key: value for key, value in sorted(inverse_slots.items())
                },
                "hash_random_control_slot": random_slot,
                "hash_random_control_attempt": inverse_slots[random_slot],
                "fresh_stream_seeds": streams,
                "outdir": str(
                    output_root
                    / f"job{job_index:03d}_seed{seed}_class{class_id}_step{step}"
                ),
            }
        )
    return salt, jobs


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"runner transformation expected one occurrence, found {count}: {old[:80]!r}")
    return text.replace(old, new)


def replace_count(text: str, old: str, new: str, expected: int) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"runner transformation expected {expected} occurrences, found {count}: {old[:80]!r}"
        )
    return text.replace(old, new)


def prospective_runner_source(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'RUNNER_NAME = "intervene_dit_v22_custom_trace_suffix"',
        f'RUNNER_NAME = "{NEW_RUNNER_NAME}"',
    )
    text = replace_once(
        text,
        'EXPERIMENT = "dit_v22_custom_trace_suffix_repairability"',
        f'EXPERIMENT = "{NEW_EXPERIMENT}"',
    )
    text = replace_once(
        text,
        'RNG_NAMESPACE = "eqvae-dit-v22-custom-trace-suffix-v1"',
        f'RNG_NAMESPACE = "{NEW_RNG_NAMESPACE}"',
    )
    text = replace_once(
        text,
        'PILOT_LOCK_KIND = "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2"',
        'PILOT_LOCK_KIND = "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2"\n'
        f'PROSPECTIVE_LOCK_KIND = "{LOCK_KIND}"',
    )
    text = replace_once(
        text,
        '"role": "EXPLORATORY_INTERNAL_TRIGGER_SELECTED_SUFFIX_REPAIRABILITY",',
        '"role": "PROSPECTIVE_SYMMETRIC_INTERNAL_BRANCH_SELECTOR_VALIDATION",',
    )
    text = replace_once(text, '"posthoc_exploratory": True,', '"posthoc_exploratory": False,')
    text = replace_once(text, '"method_claim_eligible": False,', '"method_claim_eligible": True,')
    text = replace_once(
        text,
        '"selection_provenance": "supplied before this runner; runner does not inspect scores or labels",',
        '"selection_provenance": "hash-only frozen prospective lock; runner does not inspect images, labels, scores, B, E, O, FID, or embeddings",',
    )
    old_timing = '''"trigger_decision_time": (
                "external to this generic runner and may be later than the restored state; "
                "the v1.2 pilot observes the complete step149 innovation before deciding, "
                "then restores this saved pre-transition state"
            ),'''
    new_timing = '''"trigger_decision_time": (
                "no quality trigger is used; the prospective lock fixes step149 before GPU execution, "
                "and restores the saved pre-transition state for four symmetric fresh scouts"
            ),'''
    text = replace_once(text, old_timing, new_timing)
    old_scope = '''"statistical_scope": {
            "conditional_Ville_bound_applicable": False,
            "TV_bound_applicable": False,
            "retry_cost_bound_applicable": False,
            "repairability_only": True,
        },'''
    new_scope = '''"statistical_scope": {
            "conditional_Ville_bound_applicable": False,
            "TV_bound_applicable": False,
            "retry_cost_bound_applicable": False,
            "prospective_branch_selector_validation": True,
            "success_claim_requires_sealed_internal_selection_and_external_blind_evaluation": True,
        },'''
    text = replace_once(text, old_scope, new_scope)

    prospective_binding_source = r'''

def _prospective_binding(
    prospective_lock: Path,
    *,
    trace: SavedTrace,
    target_slot: int,
    target_class: int,
    rollback_sampling_step: int,
    outdir: Path,
) -> dict[str, Any]:
    """Fail closed unless this exact job belongs to the frozen prospective lock."""

    root = prospective_lock.expanduser().resolve()
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"prospective lock must be a real symlink-free directory: {root}")
    manifest = _load_self_hashed_json(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != PROSPECTIVE_LOCK_KIND or manifest.get("status") != "complete":
        raise RuntimeError("prospective lock kind/status changed")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("prospective lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != strict.sha256_file(path)
        ):
            raise RuntimeError(f"prospective lock member changed: {name}")
    protocol = _load_self_hashed_json(root / "protocol.json", "identity_sha256")
    if (
        protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256")
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
        or protocol.get("fresh_rng", {}).get("namespace") != RNG_NAMESPACE
    ):
        raise RuntimeError("prospective protocol identity/scope changed")
    matches = [
        row
        for row in protocol.get("jobs", [])
        if isinstance(row, dict)
        and row.get("global_seed") == trace.seed
        and row.get("class_id") == target_class
        and row.get("class_slot") == target_slot
        and row.get("rollback_sampling_step") == rollback_sampling_step
    ]
    if len(matches) != 1:
        raise RuntimeError("requested suffix job is absent or ambiguous in the prospective lock")
    job = matches[0]
    if Path(job.get("trace_dir", "")).resolve() != trace.root or Path(job.get("outdir", "")).absolute() != outdir.absolute():
        raise RuntimeError("requested trace/output path differs from the prospective job")
    if job.get("trace_identity_sha256") != trace.identity_sha256:
        raise RuntimeError("prospective trace identity changed")
    trace_file_hashes = {
        "trace_manifest_file_sha256": strict.sha256_file(
            trace.root / custom_trace.MANIFEST_NAME
        ),
        "trace_completion_file_sha256": strict.sha256_file(
            trace.root / custom_trace.COMPLETION_NAME
        ),
        "trace_npz_sha256": strict.sha256_file(
            trace.root / custom_trace.TRACE_NAME
        ),
    }
    if any(job.get(key) != value for key, value in trace_file_hashes.items()):
        raise RuntimeError("prospective input-trace file identity changed")
    expected_streams = [
        _branch_seed(
            trace.identity_sha256,
            global_seed=trace.seed,
            rollback_sampling_step=rollback_sampling_step,
            target_slot=target_slot,
            attempt_index=index,
        )
        for index in range(1, BRANCH_COUNT)
    ]
    if expected_streams != job.get("fresh_stream_seeds"):
        raise RuntimeError("prospective fresh stream derivation differs from the frozen job")
    return {
        "lock_path": str(root),
        "lock_identity_sha256": manifest["identity_sha256"],
        "protocol_identity_sha256": protocol["identity_sha256"],
        "job_index": job["job_index"],
        "selection_sha256": job["selection_sha256"],
        "trace_identity_sha256": trace.identity_sha256,
        **trace_file_hashes,
        "fresh_stream_seeds": expected_streams,
        "physical_attempt_to_anonymous_slot": job["physical_attempt_to_anonymous_slot"],
        "hash_random_control_attempt": job["hash_random_control_attempt"],
        "png_label_quality_B_E_O_FID_or_embedding_used": False,
    }
'''
    text = replace_once(
        text,
        "\n\ndef _with_upstream_imports(root: Path, callback: Any) -> Any:",
        prospective_binding_source + "\n\ndef _with_upstream_imports(root: Path, callback: Any) -> Any:",
    )

    canonical_old = '''    if getattr(args, "pilot_lock", None) is not None:
        command.extend(["--pilot-lock", str(args.pilot_lock)])
    return command'''
    canonical_new = '''    if getattr(args, "pilot_lock", None) is not None:
        command.extend(["--pilot-lock", str(args.pilot_lock)])
    command.extend(["--prospective-lock", str(args.prospective_lock)])
    return command'''
    text = replace_once(text, canonical_old, canonical_new)

    build_old = '''    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    seeds = ['''
    build_new = '''    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    prospective_binding = _prospective_binding(
        args.prospective_lock,
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
        outdir=args.outdir,
    )
    seeds = ['''
    text = replace_once(text, build_old, build_new)

    dry_old = '''    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    internal_t, transition_count, stochastic_count = _validate_rollback('''
    dry_new = '''    pilot_binding = _pilot_binding(
        getattr(args, "pilot_lock", None),
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
    )
    prospective_binding = _prospective_binding(
        args.prospective_lock,
        trace=trace,
        target_slot=args.target_slot,
        target_class=target_class,
        rollback_sampling_step=args.rollback_sampling_step,
        outdir=args.outdir,
    )
    internal_t, transition_count, stochastic_count = _validate_rollback('''
    text = replace_once(text, dry_old, dry_new)
    text = replace_count(
        text,
        '        "pilot_binding": pilot_binding,',
        '        "pilot_binding": pilot_binding,\n        "prospective_binding": prospective_binding,',
        2,
    )

    parse_old = '''    parser.add_argument(
        "--pilot-lock",
        type=Path,
        help="optional frozen v1.2 pilot lock; when supplied, target and rollback must be selected",
    )
    parser.add_argument("--dit-root", type=Path)'''
    parse_new = '''    parser.add_argument(
        "--pilot-lock",
        type=Path,
        help="optional legacy pilot lock; unused by the prospective protocol",
    )
    parser.add_argument("--prospective-lock", type=Path)
    parser.add_argument("--dit-root", type=Path)'''
    text = replace_once(text, parse_old, parse_new)
    text = replace_once(
        text,
        '''        "rollback_sampling_step",
        "dit_root",''',
        '''        "rollback_sampling_step",
        "prospective_lock",
        "dit_root",''',
    )
    resolve_old = '''    if args.pilot_lock is not None:
        args.pilot_lock = args.pilot_lock.expanduser().resolve()
    if args.outdir.is_symlink():'''
    resolve_new = '''    if args.pilot_lock is not None:
        args.pilot_lock = args.pilot_lock.expanduser().resolve()
    args.prospective_lock = args.prospective_lock.expanduser().resolve()
    if args.outdir.is_symlink():'''
    text = replace_once(text, resolve_old, resolve_new)
    input_old = '''    input_paths = [args.trace_dir, args.dit_root, args.checkpoint, args.vae_snapshot]
    if args.pilot_lock is not None:
        input_paths.append(args.pilot_lock)'''
    input_new = '''    input_paths = [
        args.trace_dir,
        args.prospective_lock,
        args.dit_root,
        args.checkpoint,
        args.vae_snapshot,
    ]
    if args.pilot_lock is not None:
        input_paths.append(args.pilot_lock)'''
    text = replace_once(text, input_old, input_new)
    return text


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_existing_lock(root: Path) -> dict[str, Any]:
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != LOCK_KIND or manifest.get("status") != "complete":
        raise RuntimeError("existing prospective lock has wrong kind/status")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("existing prospective lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if record != file_record(path, root):
            raise RuntimeError(f"existing prospective lock member changed: {name}")
    return manifest


def freeze(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    source_lock = args.source_lock.expanduser().resolve()
    output_lock = args.output_lock.expanduser().absolute()
    output_root = args.output_root.expanduser().absolute()
    receipt_root = args.receipt_root.expanduser().absolute()
    if output_lock.exists():
        manifest = validate_existing_lock(output_lock)
        print(json.dumps({"status": "validated", "lock": str(output_lock), "identity_sha256": manifest["identity_sha256"]}, indent=2))
        return

    config = load_json(config_path)
    if (
        config.get("experiment") != "dit_v22_transient_escape_prospective_v1"
        or config.get("status_before_gpu_execution") != "FROZEN_HYPOTHESIS_NOT_YET_TESTED"
    ):
        raise RuntimeError("prospective config scope changed")
    source_manifest = validate_source_lock(source_lock)
    pool_manifest = load_self_hashed(trace_root / "pool_manifest.json", "identity_sha256")
    pool_id = str(pool_manifest["identity_sha256"])
    catalog = load_trace_catalog(trace_root, pool_id)
    prior_manifests, excluded, old_fresh_streams = discover_prior_suffixes(args.old_suffix_root.expanduser().resolve())
    salt, jobs = select_jobs(config, catalog, excluded, output_root)
    new_streams = [seed for job in jobs for seed in job["fresh_stream_seeds"]]
    if len(new_streams) != 512 or len(set(new_streams)) != 512:
        raise RuntimeError("new prospective fresh streams are not 512-way unique")
    overlap = sorted(set(new_streams) & old_fresh_streams)
    if overlap:
        raise RuntimeError(f"new prospective fresh streams collide with old streams: {overlap}")

    protocol: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_PROTOCOL_V1",
        "status": "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES",
        "config_sha256": sha256_file(config_path),
        "pool": {
            "root": str(trace_root),
            "identity_sha256": pool_id,
            "manifest_file_sha256": sha256_file(trace_root / "pool_manifest.json"),
            "completion_file_sha256": sha256_file(trace_root / "pool_completion.json"),
            "endpoint_pixels_labels_quality_tables_or_external_embeddings_opened": False,
            "boundary": "existing prefixes may have appeared in historical review products; only the new suffix draws and their branch-selection test are prospective",
        },
        "selection": {
            "salt": salt,
            "algorithm": config["prefix_population"]["selection_rule"],
            "class_quotas": config["prefix_population"]["class_quotas"],
            "unique_global_seeds": len({job["global_seed"] for job in jobs}),
            "job_count": len(jobs),
            "forbidden_inputs": config["prefix_population"]["selection_must_not_read"],
            "prior_exact_suffix_jobs_excluded": [list(key) for key in sorted(excluded)],
        },
        "method": config["internal_selector"],
        "external_judge_boundary": config["external_judge_only"],
        "decision_rule": config["decision_rule"],
        "fresh_rng": {
            "namespace": NEW_RNG_NAMESPACE,
            "new_stream_count": len(new_streams),
            "new_streams_unique": True,
            "old_stream_count": len(old_fresh_streams),
            "old_new_intersection": overlap,
            "new_streams_sha256": canonical_sha256(new_streams),
        },
        "lineage": {
            "source_lock": str(source_lock),
            "source_lock_identity_sha256": source_manifest["identity_sha256"],
            "dit_root": "/data/users/zhoushunyu/eqvae/baselines/DiT",
            "checkpoint": "/data/users/zhoushunyu/eqvae/baselines/DiT/pretrained_models/DiT-XL-2-256x256.pt",
            "vae_snapshot": "/home/zhoushunyu/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493",
        },
        "outputs": {
            "root": str(output_root),
            "receipt_root": str(receipt_root),
            "all_five_branches_retained_for_blind_counterfactual_evaluation": True,
            "operational_budget_statement": "four scouts to h10 and only the selected scout beyond h10; running all four to endpoint here is evaluation overhead",
        },
        "prior_suffix_manifest_inventory": prior_manifests,
        "jobs": jobs,
    }
    protocol["identity_sha256"] = self_hash(protocol, "identity_sha256")

    output_lock.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_lock.name}.tmp-", dir=output_lock.parent))
    try:
        (staging / "sources").mkdir()
        shutil.copy2(config_path, staging / "frozen_config.json")
        generated = prospective_runner_source(
            source_lock / "sources/intervene_dit_v22_custom_trace_suffix.py"
        )
        runner_path = staging / "sources/intervene_dit_v22_transient_escape_suffix.py"
        runner_path.write_text(generated, encoding="utf-8")
        for name in (
            "reproduce_dit_imagenet256.py",
            "sample_dit_imagenet256_custom.py",
            "trace_dit_imagenet256_custom_batch.py",
        ):
            shutil.copy2(source_lock / "sources" / name, staging / "sources" / name)
        for live_name in (
            "run_dit_v22_transient_escape_prospective_shard.py",
            "extract_dit_v22_transient_escape_internal.py",
        ):
            live = ROOT / "experiments" / live_name
            if not live.is_file():
                raise RuntimeError(f"required prospective source is missing: {live}")
            shutil.copy2(live, staging / "sources" / live_name)
        write_json(staging / "protocol.json", protocol)
        preflight = {
            "schema_version": 1,
            "protocol_identity_sha256": protocol["identity_sha256"],
            "jobs": len(jobs),
            "unique_global_seeds": len({job["global_seed"] for job in jobs}),
            "class_counts": {
                str(class_id): sum(job["class_id"] == class_id for job in jobs)
                for class_id in sorted(CLASS_SLOTS)
            },
            "new_fresh_streams": len(new_streams),
            "new_fresh_streams_unique": len(set(new_streams)) == len(new_streams),
            "old_new_fresh_stream_intersection": overlap,
            "new_runner_rng_namespace_present": (
                f'RNG_NAMESPACE = "{NEW_RNG_NAMESPACE}"' in generated
            ),
            "png_review_label_quality_or_external_embedding_opened": False,
        }
        preflight["identity_sha256"] = self_hash(preflight, "identity_sha256")
        write_json(staging / "preflight.json", preflight)
        members = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": LOCK_KIND,
            "status": "complete",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "preflight_identity_sha256": preflight["identity_sha256"],
            "config_sha256": sha256_file(staging / "frozen_config.json"),
            "files": [file_record(path, staging) for path in members],
        }
        manifest["identity_sha256"] = self_hash(manifest, "identity_sha256")
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_lock)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "frozen",
                "lock": str(output_lock),
                "identity_sha256": manifest["identity_sha256"],
                "protocol_identity_sha256": protocol["identity_sha256"],
                "jobs": len(jobs),
                "class_counts": preflight["class_counts"],
                "new_fresh_streams": len(new_streams),
                "old_new_stream_overlap": overlap,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--old-suffix-root", type=Path, default=DEFAULT_OLD_SUFFIX_ROOT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--output-lock", type=Path, default=DEFAULT_OUTPUT_LOCK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    freeze(parse_args())

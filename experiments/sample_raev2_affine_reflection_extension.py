#!/usr/bin/env python3
"""Hash-bound seed extension of the unchanged affine-reflection sampler.

Four additional independent 1K cohorts retain every historical sampling formula.
Cohort zero is reserved for parity and official cost baselines on the original
1K seed; rerunning its reflection candidate is forbidden by this wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SOURCE = ROOT / "experiments/sample_raev2_affine_reflection.py"
BASE_SHA256 = "37426ed1865537a14f05777ed909d90113aeee224fc0f4f0cf8f4e71d557b069"
PARENT_PLAN_SHA256 = "1f7e448c3d336661e39cad671097be9afcc85d7d0354b7776ab833ec7ae6ce1c"
HISTORICAL_SEED = 202609131
PROTOCOL = "raev2_fixed_1k_cohort_affine_reflection_extension_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_json(record):
    path = Path(record["path"]).expanduser().resolve()
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"frozen JSON hash mismatch: {path}")
    return json.loads(path.read_text()), path


def normalized_freeze(plan):
    result = {}
    for name, identity in plan["frozen_files"].items():
        path = Path(name)
        path = (ROOT / path if not path.is_absolute() else path).resolve()
        digest = identity["sha256"] if isinstance(identity, dict) else identity
        if str(path) in result and result[str(path)] != digest:
            raise ValueError("conflicting identities for one frozen file")
        result[str(path)] = digest
    return result


def validate_extension(manifest_record, cohort_index):
    """Check admission without loading model weights or calling CUDA."""
    if type(cohort_index) is not int or cohort_index not in (0, 1, 2, 3, 4):
        raise ValueError("cohort index must be one of 0,1,2,3,4")
    if sha256_file(BASE_SOURCE) != BASE_SHA256:
        raise ValueError("historical reflection sampler changed")
    manifest, manifest_path = verified_json(manifest_record)
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected extension protocol")
    if manifest.get("historical_sampler_sha256") != BASE_SHA256:
        raise ValueError("manifest does not bind the historical sampler")
    wrapper = Path(__file__).resolve()
    wrapper_sha = sha256_file(wrapper)
    if manifest.get("extension_wrapper_sha256") != wrapper_sha:
        raise ValueError("manifest does not bind this wrapper")
    parent_record = manifest["historical_plan"]
    if parent_record.get("sha256") != PARENT_PLAN_SHA256:
        raise ValueError("original 1K plan identity required")
    parent, _ = verified_json(parent_record)
    if parent["cohort"] != {"seed": HISTORICAL_SEED, "n": 1000, "batch_size": 8}:
        raise ValueError("historical cohort changed")
    parent_freeze = normalized_freeze(parent)
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, list) or [row.get("index") for row in cohorts] != [0, 1, 2, 3, 4]:
        raise ValueError("one historical and four additional cohorts must be fixed together")
    seeds = [row.get("seed") for row in cohorts]
    if (seeds[0] != HISTORICAL_SEED or len(set(seeds)) != 5
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
        raise ValueError("cohort zero needs the original seed; four distinct new seeds required")
    verified = []
    for row in cohorts:
        plan, path = verified_json(row["plan"])
        if plan.get("cohort") != {"seed": row["seed"], "n": 1000, "batch_size": 8}:
            raise ValueError("every sample cohort must remain exactly 1000 images in B8")
        if plan.get("extension_cohort_index") != row["index"] or plan.get("historical_plan") != parent_record:
            raise ValueError("cohort plan index or historical identity mismatch")
        if plan.get("source_freeze_complete") is not True:
            raise ValueError("cohort source freeze must be complete")
        freeze = normalized_freeze(plan)
        if any(freeze.get(name) != digest for name, digest in parent_freeze.items()):
            raise ValueError("all original frozen sources/artifacts must retain their identities")
        if freeze.get(str(wrapper)) != wrapper_sha:
            raise ValueError("wrapper must be included in every cohort frozen_files")
        document = plan["protocol_document"]
        document_path = Path(document["path"]).expanduser().resolve()
        if sha256_file(document_path) != document["sha256"] or freeze.get(str(document_path)) != document["sha256"]:
            raise ValueError("new protocol document must be hash-bound and frozen")
        verified.append((plan, path, row))
    return manifest, manifest_path, verified[cohort_index]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--extension-manifest-sha256", required=True)
    parser.add_argument("--cohort-index", type=int, choices=(0, 1, 2, 3, 4), required=True)
    args, remaining = parser.parse_known_args(argv)
    record = {"path": str(args.extension_manifest.expanduser().resolve()),
              "sha256": args.extension_manifest_sha256}
    manifest, _, (plan, plan_path, selected) = validate_extension(record, args.cohort_index)
    from experiments import sample_raev2_affine_reflection as original
    if original.COUNT != 1000 or original.BATCH_SIZE != 8:
        raise ValueError("historical cohort or microbatch constants changed")
    original.SEED = selected["seed"]
    original.PLAN = plan_path
    original.PROTOCOL_DOCUMENT = Path(plan["protocol_document"]["path"]).expanduser().resolve()
    original.PROTOCOL = PROTOCOL
    original.SOURCE_FILES = (*original.SOURCE_FILES, str(Path(__file__).relative_to(ROOT)))
    parsed = original.parse_args(remaining)
    if selected["index"] == 0 and parsed.mode not in ("official", "parity"):
        raise ValueError("historical cohort zero permits only parity and official cost baselines")
    original_atomic_json = original.atomic_json

    def write_with_extension_identity(path, payload):
        if path.name == "request.json":
            payload = {**payload, "extension_manifest": record,
                       "extension_cohort_index": selected["index"],
                       "historical_plan": manifest["historical_plan"],
                       "extension_changes": "seed and frozen plan/provenance only; unchanged N1000 sampler and reflection"}
        return original_atomic_json(path, payload)

    original.atomic_json = write_with_extension_identity
    sys.argv = [str(Path(__file__)), *remaining]
    original.main()


if __name__ == "__main__":
    main()

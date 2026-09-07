#!/usr/bin/env python3
"""Run one frozen additional 1K cohort through the unchanged spatial-ball code.

The extension changes only the frozen seed, plan identity and protocol label.
It never changes COUNT, the sampler functions, projection, calibration or gains.
Four cohort plans are declared together in a hash-bound extension manifest;
each cohort still requires its own completed 16-image parity run.

CLI: --extension-manifest PATH --extension-manifest-sha256 SHA
     --cohort-index {1,2,3,4} [original sampler arguments]
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

BASE_SOURCE = ROOT / "experiments/sample_raev2_spatial_energy_balls.py"
BASE_SHA256 = "e68c4d5197f40ba4cbffff54349edd619a172e171b8265682def8e108cc0ab34"
PARENT_PLAN_SHA256 = "9c7972ac459b5511f293553e86e43671d0bca8eda7d5e5359f5ebcb03ef62025"
PROTOCOL = "raev2_fixed_1k_cohort_spatial_energy_balls_extension_v1"


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


def validate_extension(manifest_record, cohort_index):
    """CPU-only admission checks; all four seeds/plans are fixed before sampling."""
    if type(cohort_index) is not int or cohort_index not in (1, 2, 3, 4):
        raise ValueError("additional cohort index must be one of 1,2,3,4")
    if sha256_file(BASE_SOURCE) != BASE_SHA256:
        raise ValueError("historical sampler source changed")
    manifest, manifest_path = verified_json(manifest_record)
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected extension protocol")
    if manifest.get("historical_sampler_sha256") != BASE_SHA256:
        raise ValueError("manifest does not bind the unchanged historical sampler")
    wrapper = Path(__file__).resolve()
    if manifest.get("extension_wrapper_sha256") != sha256_file(wrapper):
        raise ValueError("manifest does not bind this extension wrapper")
    parent_record = manifest["historical_plan"]
    if parent_record.get("sha256") != PARENT_PLAN_SHA256:
        raise ValueError("extension must identify the original 1K plan")
    parent, _ = verified_json(parent_record)
    cohorts = manifest.get("additional_cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != 4:
        raise ValueError("exactly four additional 1K cohorts must be frozen together")
    if [row.get("index") for row in cohorts] != [1, 2, 3, 4]:
        raise ValueError("cohort indices must be exactly 1,2,3,4 in that order")
    seeds = [row.get("seed") for row in cohorts]
    if (any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
            or len(set(seeds)) != 4 or parent["cohort"]["seed"] in seeds):
        raise ValueError("four distinct new valid seeds are required")
    verified = []
    for row in cohorts:
        plan, plan_path = verified_json(row["plan"])
        if plan.get("reference") != parent["reference"]:
            raise ValueError("calibration reference must remain exactly unchanged")
        if plan.get("cohort", {}).get("n") != 1000 or plan["cohort"].get("seed") != row["seed"]:
            raise ValueError("every interacting cohort must remain exactly 1000 images")
        if plan.get("extension_cohort_index") != row["index"]:
            raise ValueError("cohort plan index mismatch")
        if plan.get("historical_plan") != parent_record:
            raise ValueError("cohort plan must bind its original historical plan")
        document = plan["protocol_document"]
        if sha256_file(Path(document["path"]).expanduser().resolve()) != document["sha256"]:
            raise ValueError("extension protocol document changed")
        verified.append((plan, plan_path, row))
    selected = verified[cohort_index - 1]
    return manifest, manifest_path, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--extension-manifest-sha256", required=True)
    parser.add_argument("--cohort-index", type=int, choices=(1, 2, 3, 4), required=True)
    wrapper_args, remaining = parser.parse_known_args(argv)
    record = {"path": str(wrapper_args.extension_manifest.expanduser().resolve()),
              "sha256": wrapper_args.extension_manifest_sha256}
    manifest, _, (_, plan_path, selected) = validate_extension(record, wrapper_args.cohort_index)
    from experiments import sample_raev2_spatial_energy_balls as original
    if original.COUNT != 1000 or original.BATCH_SIZE != 8:
        raise ValueError("historical cohort/microbatch constants changed")
    original.SEED = selected["seed"]
    original.PLAN = plan_path
    original.PLAN_SHA256 = selected["plan"]["sha256"]
    original.PROTOCOL = PROTOCOL
    original.SOURCE_FILES = (*original.SOURCE_FILES, str(Path(__file__).relative_to(ROOT)))
    original_atomic_json = original.atomic_json

    def write_with_extension_identity(path, payload):
        if path.name == "request.json":
            payload = {**payload, "extension_manifest": record,
                       "extension_cohort_index": selected["index"],
                       "historical_plan": manifest["historical_plan"],
                       "extension_changes": "seed and frozen plan/provenance only; unchanged N1000 sampler and calibration"}
        return original_atomic_json(path, payload)

    original.atomic_json = write_with_extension_identity
    sys.argv = [str(Path(__file__)), *remaining]
    original.main()


if __name__ == "__main__":
    main()

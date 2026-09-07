#!/usr/bin/env python3
"""CPU-only, lossless merge of one screened 1K and four paired new 1K blocks.

The JSON plan binds each input's request, summary, batch manifest and samples by
SHA256. All arms within a family must have matching *actual* noise/label hashes
in each block. It writes pooled5k and new4k archives plus evaluator argv; it never
launches a model or chooses a method/seed/baseline from quality results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np


PROTOCOL = "raev2_paired_1k_extension_merge_v1"
COUNT = 1000
IMAGE_SHAPE = (256, 256, 3)
COST_FIELDS = (
    "trajectory_through_decode_wall_seconds", "total_wall_seconds_before_summary",
    "elapsed_seconds", "trajectory_wall_seconds", "sampling_wall_including_output_seconds",
    "decode_and_uint8_wall_seconds", "full_model_calls", "full_sample_evaluations",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(1024 * 1024):
            digest.update(part)
    return digest.hexdigest()


def raw_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checked_file(record):
    require(isinstance(record, dict) and set(record) >= {"path", "sha256"}, "artifact identity missing")
    path = Path(record["path"]).expanduser().resolve()
    require(path.is_file(), f"missing input: {path}")
    require(re.fullmatch("[0-9a-f]{64}", record["sha256"]) is not None, f"invalid digest: {path}")
    require(sha256(path) == record["sha256"], f"input SHA256 changed: {path}")
    if "size_bytes" in record:
        require(path.stat().st_size == record["size_bytes"], f"input size changed: {path}")
    return path


def json_value(document, path):
    value = document
    for key in path.split("."):
        require(isinstance(value, dict) and key in value, f"missing request field: {path}")
        value = value[key]
    return value


def read_images(path, ids):
    with np.load(path, allow_pickle=False) as archive:
        require("arr_0" in archive.files and set(archive.files) <= {"arr_0", "ids", "labels"},
                f"unexpected archive entries: {path}")
        images = archive["arr_0"]
        require(images.dtype == np.uint8 and images.shape == (len(ids), *IMAGE_SHAPE),
                f"expected uint8 NHWC 256px RGB: {path}")
        for name in ("ids", "labels"):
            if name in archive.files:
                observed = archive[name]
                require(observed.dtype == np.int64 and np.array_equal(observed, ids),
                        f"wrong local {name}: {path}")
    return images


def load_source(source, block, arm_name):
    """Validate a durable source without mutating it; return metadata and pixels."""
    paths = {name: checked_file(source[name]) for name in
             ("request", "summary", "batch_manifest", "samples")}
    request = json.loads(paths["request"].read_text())
    summary = json.loads(paths["summary"].read_text())
    require(summary.get("complete") is True, f"incomplete source: {arm_name}/{block['id']}")
    for document in (request, summary):
        require(document.get("seed") == block["seed"], "request/summary seed mismatch")
    require(request.get("sample_count") == summary.get("samples") == COUNT, "every block must be complete 1K")
    require(summary.get("mode") == request.get("mode"), "request/summary mode mismatch")
    require(summary.get("num_steps", request.get("num_steps")) == request.get("num_steps"), "step count mismatch")
    require(summary.get("archive_sha256") == source["samples"]["sha256"], "summary archive identity mismatch")
    for field, expected in source.get("expected_request_fields", {}).items():
        require(json_value(request, field) == expected, f"frozen request value changed: {field}")
    for key, path_name in (("request", "request"), ("batch_manifest", "batch_manifest"),
                           ("sample_archive", "samples"), ("samples_npz", "samples")):
        if isinstance(summary.get(key), dict):
            require(summary[key].get("sha256") == source[path_name]["sha256"], f"summary {key} SHA mismatch")
    ids = np.arange(COUNT, dtype=np.int64)
    labels_hash = summary.get("global_labels_sha256", summary.get("labels_sha256"))
    require(labels_hash == raw_hash(ids), "labels must be all 1000 classes in ascending order")
    noise_hash = summary.get("global_noise_sha256", summary.get("noise_sha256"))
    require(isinstance(noise_hash, str) and re.fullmatch("[0-9a-f]{64}", noise_hash), "missing actual noise hash")
    images = read_images(paths["samples"], ids)
    batch_document = json.loads(paths["batch_manifest"].read_text())
    batches = batch_document["batches"]
    require(isinstance(batches, list) and len(batches) == 125, "expected 125 full B8 batches")
    batch_pairing = []
    for index, record in enumerate(batches):
        start, stop = 8 * index, 8 * (index + 1)
        expected_ids = ids[start:stop]
        observed_ids = record.get("global_ids", list(range(record.get("start", -1), record.get("stop", -1))))
        require(observed_ids == expected_ids.tolist(), "batch identity/order mismatch")
        require(record["labels_sha256"] == raw_hash(expected_ids), "batch label hash mismatch")
        if "archive" in record:
            batch_path = checked_file(record["archive"])
        else:
            batch_path = checked_file({"path": str(paths["samples"].parent / record["path"]),
                                       "sha256": record["archive_sha256"]})
        batch_images = read_images(batch_path, expected_ids)
        require(np.array_equal(batch_images, images[start:stop]), "merged source differs from durable batches")
        require(re.fullmatch("[0-9a-f]{64}", record["noise_sha256"]) is not None, "invalid batch noise hash")
        batch_pairing.append({"ids": expected_ids.tolist(), "noise_sha256": record["noise_sha256"],
                              "labels_sha256": record["labels_sha256"]})
    costs = {}
    for key in COST_FIELDS:
        if key in summary:
            value = summary[key]
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    and value >= 0, f"invalid cost: {key}")
            costs[key] = value
    record = {"block_id": block["id"], "role": block["role"], "seed": block["seed"],
              "source": source, "mode": request["mode"], "num_steps": request["num_steps"],
              "noise_sha256": noise_hash, "labels_sha256": labels_hash,
              "noise_rng_state_sha256": summary.get("noise_rng_state_sha256"),
              "batch_pairing": batch_pairing, "observed_costs": costs,
              "source_pixels_sha256": raw_hash(images),
              "label_evidence": "archive labels when present plus bound summary and all 125 batch label hashes"}
    return record, request, images


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL, "wrong merge plan protocol")
    require(isinstance(plan.get("family"), str) and plan["family"], "family name required")
    blocks = plan["blocks"]
    require(len(blocks) == 5 and [b["role"] for b in blocks] == ["screen"] + ["confirmation"] * 4,
            "plan must contain the existing screen then all four new blocks")
    require(len({b["id"] for b in blocks}) == len({b["seed"] for b in blocks}) == 5, "duplicate block ID or seed")
    arms = list(blocks[0]["arms"])
    require(len(arms) >= 2 and all(re.fullmatch(r"[A-Za-z0-9_-]+", arm) for arm in arms), "safe paired arm names required")
    require(all(set(b["arms"]) == set(arms) for b in blocks), "missing arm in a block")
    fields = plan.get("invariant_request_fields", ["mode", "num_steps", "batch_size"])
    require(set(fields) >= {"mode", "num_steps", "batch_size"}, "same arm must retain mode, K and batch size across all five blocks")
    return arms, fields


def sum_costs(records):
    # Do not substitute elapsed for T/W or silently sum an incomplete boundary.
    return {key: {"sum": sum(r["observed_costs"][key] for r in records) if all(key in r["observed_costs"] for r in records) else None,
                  "missing_blocks": [r["block_id"] for r in records if key not in r["observed_costs"]]}
            for key in COST_FIELDS}


def merge(plan_path, output_dir):
    started, cpu_started = time.perf_counter(), time.process_time()
    plan_path, output_dir = Path(plan_path).resolve(), Path(output_dir).resolve()
    plan_record = artifact(plan_path)
    plan = json.loads(plan_path.read_text())
    arms, invariant_fields = validate_plan(plan)
    require(not output_dir.exists(), "output directory already exists; partial output must be reviewed, never overwritten")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "request.json", {"protocol": PROTOCOL, "plan": plan_record, "plan_contents": plan,
               "source": artifact(__file__), "started_utc": datetime.now(timezone.utc).isoformat(),
               "selection": "no image, block, or arm selection; preserve manifest order and all pixels"})
    result = {"protocol": PROTOCOL, "family": plan["family"], "complete": False, "arms": {},
              "cost_match_claim": False, "performance_goal_claim": False,
              "limits": ["5K includes a previously screened 1K and is descriptive; new4k excludes those images.",
                         "Four new blocks must have been fixed before inspecting their FID; this merger does not prove chronology.",
                         "Noise hashes are checked against paired metadata and batches; unsaved full noise is not regenerated.",
                         "Moment-guided methods retain their original 1K cohort interaction size; 5K pools five such cohorts.",
                         "Recorded historical timing boundaries may differ; no missing timing is imputed."]}
    pairing_reference = {}
    archive_sources = set()
    evaluation_argv = ["python", "experiments/evaluate_raev2_official_samples.py", "--output",
                       str(output_dir / "official_evaluation.csv")]
    try:
        for arm in arms:
            records, invariant_reference = [], None
            used_noise_hashes = set()
            temporary = output_dir / f".{arm}.pixels.npy"
            bank = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(5000, *IMAGE_SHAPE))
            for index, block in enumerate(plan["blocks"]):
                source = block["arms"][arm]
                source_path = str(Path(source["samples"]["path"]).resolve())
                require(source_path not in archive_sources, "same source archive used more than once in family")
                archive_sources.add(source_path)
                record, request, pixels = load_source(source, block, arm)
                require(record["noise_sha256"] not in used_noise_hashes, "noise payload repeated across independent blocks")
                used_noise_hashes.add(record["noise_sha256"])
                invariant = {field: json_value(request, field) for field in invariant_fields}
                if invariant_reference is None:
                    invariant_reference = invariant
                require(invariant == invariant_reference, f"method changed across blocks: {arm}")
                pairing = {key: record[key] for key in ("seed", "noise_sha256", "labels_sha256", "noise_rng_state_sha256", "batch_pairing")}
                if block["id"] not in pairing_reference:
                    pairing_reference[block["id"]] = pairing
                require(pairing == pairing_reference[block["id"]], f"actual inputs not paired: {block['id']}/{arm}")
                bank[index * COUNT:(index + 1) * COUNT] = pixels
                records.append(record)
            bank.flush()
            outputs = {}
            for name, offset, selected in (("pooled5k", 0, records), ("new4k", COUNT, records[1:])):
                directory = output_dir / arm / name
                directory.mkdir(parents=True)
                count = len(selected) * COUNT
                ids = np.arange(count, dtype=np.int64)
                labels = ids % COUNT
                archive_path = directory / "samples.npz"
                np.savez(archive_path, bank[offset:], ids=ids, labels=labels,
                         block_index=np.repeat(np.arange(offset // COUNT, 5, dtype=np.int64), COUNT),
                         local_ids=np.tile(np.arange(COUNT, dtype=np.int64), len(selected)))
                # Official evaluator reads arr_0. Other arrays preserve the exact mapping.
                summary = {"protocol": PROTOCOL, "complete": True, "samples": count, "arm": arm,
                           "subset": name, "invariant_request_fields": invariant_reference,
                           "samples_npz": artifact(archive_path), "pixels_sha256": raw_hash(bank[offset:]),
                           "labels_sha256": raw_hash(labels), "source_blocks": selected,
                           "costs_by_unchanged_source_field": sum_costs(selected)}
                write_json(directory / "summary.json", summary)
                outputs[name] = {"samples": summary["samples_npz"], "summary": artifact(directory / "summary.json")}
                if name == "pooled5k":
                    evaluation_argv.extend(["--branch", f"{plan['family']}_{arm}_{name}={archive_path}"])
            del bank
            temporary.unlink()
            result["arms"][arm] = outputs
        result.update(complete=True, pairing_blocks=pairing_reference, evaluation_argv=evaluation_argv,
                      new4k_fid_command="experiments/evaluate_raev2_1k_extension_features.py; reuses pooled5k official features, no repeated extraction",
                      wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started,
                      gpu_model_calls=0, fid_performed=False)
        write_json(output_dir / "summary.json", result)
        return result
    except BaseException as error:
        write_json(output_dir / "failure.json", {"complete": False, "error": f"{type(error).__name__}: {error}",
                   "wall_seconds": time.perf_counter() - started, "partial_outputs_retained": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.plan, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

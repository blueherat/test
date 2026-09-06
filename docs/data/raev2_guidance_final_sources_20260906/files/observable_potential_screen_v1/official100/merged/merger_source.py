#!/usr/bin/env python3
"""Strict CPU merge of one complete paired observable-potential sample cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SAMPLER_PROTOCOL = "raev2_observable_potential_sampling_v1"
PROTOCOL = "raev2_observable_potential_merge_v1"
COHORT_SIZE = 1000
BATCH_SIZE = 8
IMAGE_SHAPE = (256, 256, 3)
REQUIRED_SOURCES = {"experiments/sample_raev2_observable_potential.py",
                    "experiments/train_raev2_observable_potential.py",
                    "experiments/raev2_observable_potential.py",
                    "external/RAEv2/src/utils/guidance_utils.py"}
COUNTERS = ("stage2_forward_calls", "stage2_sample_forwards",
            "potential_forward_input_gradient_calls", "potential_sample_input_gradients",
            "potential_parameter_backward_calls", "decoder_forward_calls", "decoder_sample_forwards")
TIMES = ("trajectory_wall_seconds", "decode_and_uint8_wall_seconds",
         "sampling_wall_including_output_seconds", "potential_checkpoint_load_seconds",
         "common_backbone_and_decoder_load_seconds", "noise_generation_copy_and_audit_wall_seconds",
         "total_wall_seconds_excluding_imports_and_final_summary_write",
         "total_cpu_seconds_excluding_imports_and_final_summary_write")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_artifact(record, *, expected_path=None, cache=None):
    path = Path(record["path"]).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"unexpected artifact location: {path}")
    cache = {} if cache is None else cache
    key = str(path)
    if key not in cache:
        cache[key] = artifact(path)
    actual = cache[key]
    if record.get("sha256") != actual["sha256"] or record.get("size_bytes") != actual["size_bytes"]:
        raise ValueError(f"artifact hash/size mismatch: {path}")
    return path


def expected_batch_ids(rank, world):
    return [np.arange(k*BATCH_SIZE, (k+1)*BATCH_SIZE, dtype=np.int64)
            for k in range(rank, COHORT_SIZE // BATCH_SIZE, world)]


def read_images(path, expected_ids):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"arr_0", "ids", "labels"}:
            raise ValueError("sample archive requires exactly arr_0, ids, labels")
        images, ids, labels = archive["arr_0"], archive["ids"], archive["labels"]
    if images.dtype != np.uint8 or images.shape != (len(expected_ids), *IMAGE_SHAPE):
        raise ValueError("sample images must be uint8 [N,256,256,3]")
    if ids.dtype != np.int64 or not np.array_equal(ids, expected_ids):
        raise ValueError("archive global IDs differ from the fixed batch partition")
    if labels.dtype != np.int64 or not np.array_equal(labels, ids % 1000):
        raise ValueError("archive labels must equal global_id % 1000")
    return images


def request_signature(request):
    # Only locations of worker-local artifacts and worker device metadata differ.
    omitted = {"shard_index", "sources", "paired_noise_audit", "cuda_device", "cuda_visible_devices"}
    result = {key: value for key, value in request.items() if key not in omitted}
    result["source_hashes"] = {key: {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}
                               for key, value in request["sources"].items()}
    return result


def check_counters(summary, *, samples, steps, mode):
    batch_calls = samples // BATCH_SIZE
    expected = {"stage2_forward_calls": batch_calls*steps, "stage2_sample_forwards": samples*steps,
                "potential_forward_input_gradient_calls": batch_calls*steps if mode == "potential" else 0,
                "potential_sample_input_gradients": samples*steps if mode == "potential" else 0,
                "potential_parameter_backward_calls": 0, "decoder_forward_calls": batch_calls,
                "decoder_sample_forwards": samples}
    for key, value in expected.items():
        if type(summary.get(key)) is not int or summary[key] != value:
            raise ValueError(f"incorrect sampling counter: {key}")
    if summary.get("stage2_nfe_per_sample") != steps:
        raise ValueError("incorrect stage2 NFE per sample")
    for key in TIMES:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid worker timing: {key}")


def merge(shard_dirs, output_dir):
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    directories = [Path(path).expanduser().resolve() for path in shard_dirs]
    output = Path(output_dir).expanduser().resolve()
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("provide each shard directory exactly once")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite existing output")
    cache, workers, signature, noise_signature = {}, [], None, None
    seen_ranks = set()
    images = np.empty((COHORT_SIZE, *IMAGE_SHAPE), dtype=np.uint8)
    seen_ids = np.zeros(COHORT_SIZE, dtype=np.int64)
    for directory in directories:
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (summary.get("complete") is not True or summary.get("protocol") != SAMPLER_PROTOCOL
                or summary.get("mode") not in ("official", "potential")
                or summary.get("image_sampling_performed") is not True
                or summary.get("fid_performed") is not False or summary.get("training_performed") is not False):
            raise ValueError("expected a complete sampling shard, not benchmark/training/FID")
        request_path = verify_artifact(summary["request"], expected_path=directory / "request.json", cache=cache)
        request = json.loads(request_path.read_text())
        rank, world = request.get("shard_index"), request.get("num_shards")
        if (type(rank) is not int or type(world) is not int or not 1 <= world <= COHORT_SIZE // BATCH_SIZE
                or not 0 <= rank < world or rank in seen_ranks or len(directories) != world):
            raise ValueError("missing, duplicate or invalid shard ranks")
        seen_ranks.add(rank)
        if (request.get("protocol") != SAMPLER_PROTOCOL or request.get("batch_size") != BATCH_SIZE
                or request.get("sample_count") != COHORT_SIZE or request.get("mode") != summary["mode"]
                or request.get("seed") != summary.get("seed")
                or request.get("num_steps") != summary.get("num_steps")):
            raise ValueError("request/summary sampling protocol mismatch")
        steps = request["num_steps"]
        if type(steps) is not int or steps <= 0 or (request["mode"] == "potential" and steps != 100):
            raise ValueError("invalid frozen candidate or official step count")
        grid = request.get("time_grid")
        if (not isinstance(grid, list) or len(grid) != steps+1 or grid[0] != 1 or grid[-1] != 0
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in grid)
                or not all(0 <= b < a <= 1 for a, b in zip(grid[:-1], grid[1:]))):
            raise ValueError("invalid complete Euler time grid")
        current_signature = request_signature(request)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError("shards differ in mode, seed, sources, checkpoints, grid or protocol")
        for name in ("config", "baseline_checkpoint", "potential_checkpoint", "decoder_checkpoint", "normalization_stats"):
            verify_artifact(request[name], cache=cache)
        if not REQUIRED_SOURCES.issubset(request["sources"]):
            raise ValueError("missing required source snapshots")
        for relative, record in request["sources"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("invalid source snapshot path")
            verify_artifact(record, expected_path=directory / "sources" / relative, cache=cache)
        global_noise_hash = request.get("global_noise_sha256")
        if not digest_string(global_noise_hash) or summary.get("global_noise_sha256") != global_noise_hash:
            raise ValueError("global noise hash mismatch")
        if request.get("global_labels_sha256") != array_sha256(np.arange(COHORT_SIZE, dtype=np.int64)):
            raise ValueError("global label digest mismatch")
        audit_path = verify_artifact(request["paired_noise_audit"], expected_path=directory / "paired_noise_audit.npz", cache=cache)
        with np.load(audit_path, allow_pickle=False) as archive:
            if set(archive.files) != {"first_noise", "rng_state"}:
                raise ValueError("invalid paired noise audit keys")
            first, rng = archive["first_noise"], archive["rng_state"]
            if first.dtype != np.float32 or first.shape != (1024, 16, 16) or not np.isfinite(first).all():
                raise ValueError("invalid first noise tensor")
            if rng.dtype != np.uint8 or rng.ndim != 1 or rng.size == 0:
                raise ValueError("invalid paired noise RNG state")
            noise = (global_noise_hash, array_sha256(first), array_sha256(rng))
        if noise_signature is None:
            noise_signature = noise
        elif noise_signature != noise:
            raise ValueError("shards disagree on paired cohort/first noise/final RNG state")
        batches = expected_batch_ids(rank, world)
        ids = np.concatenate(batches)
        if (summary.get("samples") != len(ids) or summary.get("global_cohort_size") != COHORT_SIZE
                or summary.get("global_ids") != ids.tolist()):
            raise ValueError("summary sample IDs/count do not match its assigned shard")
        check_counters(summary, samples=len(ids), steps=steps, mode=request["mode"])
        sample_path = verify_artifact(summary["sample_archive"], expected_path=directory / "samples.npz", cache=cache)
        if summary.get("archive_sha256") != summary["sample_archive"]["sha256"]:
            raise ValueError("summary sample archive hash mismatch")
        shard_images = read_images(sample_path, ids)
        manifest_path = verify_artifact(summary["batch_manifest"], expected_path=directory / "batch_manifest.json", cache=cache)
        manifest = json.loads(manifest_path.read_text()).get("batches")
        if not isinstance(manifest, list) or len(manifest) != len(batches):
            raise ValueError("missing or extra batch manifest entries")
        batch_trajectory, batch_decode = 0.0, 0.0
        for index, (record, batch_ids) in enumerate(zip(manifest, batches)):
            if record.get("global_ids") != batch_ids.tolist():
                raise ValueError("batch manifest IDs differ from fixed B8 partition")
            if not digest_string(record.get("noise_sha256")) or not digest_string(record.get("endpoint_sha256")):
                raise ValueError("missing batch noise or endpoint fingerprint")
            expected_path = directory / "batches" / f"{int(batch_ids[0]):06d}_{int(batch_ids[-1])+1:06d}.npz"
            batch_path = verify_artifact(record["archive"], expected_path=expected_path, cache=cache)
            batch_images = read_images(batch_path, batch_ids)
            if not np.array_equal(batch_images, shard_images[index*BATCH_SIZE:(index+1)*BATCH_SIZE]):
                raise ValueError("shard image archive differs from its hashed batch archives")
            for key in ("trajectory_wall_seconds", "decode_and_uint8_wall_seconds"):
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid per-batch timing")
            batch_trajectory += record["trajectory_wall_seconds"]
            batch_decode += record["decode_and_uint8_wall_seconds"]
        for actual, key in ((batch_trajectory, "trajectory_wall_seconds"), (batch_decode, "decode_and_uint8_wall_seconds")):
            if not math.isclose(actual, summary[key], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("worker timing differs from sum of completed batches")
        seen_ids[ids] += 1
        images[ids] = shard_images
        workers.append({"shard_index": rank, "directory": str(directory), "request": artifact(request_path),
                        "summary": artifact(summary_path), "samples": len(ids),
                        "sample_archive": summary["sample_archive"], "batch_manifest": summary["batch_manifest"],
                        "cuda_device": request.get("cuda_device"),
                        "counters": {key: summary[key] for key in COUNTERS},
                        "timings": {key: summary[key] for key in TIMES},
                        "peak_gpu_memory_allocated_bytes": summary["peak_gpu_memory_allocated_bytes"]})
    if seen_ranks != set(range(len(directories))) or not np.all(seen_ids == 1):
        raise ValueError("must cover global IDs 0..999 exactly once, one image per class")
    workers.sort(key=lambda item: item["shard_index"])
    output.mkdir(parents=True, exist_ok=True)
    source = output / "merger_source.py"
    source.write_bytes(Path(__file__).read_bytes())
    request = {"protocol": PROTOCOL, "sampler_protocol": SAMPLER_PROTOCOL,
               "common_request": signature, "workers": workers, "merger_source": artifact(source),
               "global_noise_sha256": noise_signature[0], "first_noise_sha256": noise_signature[1],
               "final_noise_rng_sha256": noise_signature[2],
               "noise_audit_boundary": "CPU verifies stored cohort digest agreement and first-noise/RNG bytes; it does not regenerate the full CUDA noise cohort.",
               "output_order": "global IDs 0..999, labels global_id % 1000", "fid_performed": False}
    atomic_json(output / "request.json", request)
    sample_path = output / "samples.npz"
    temporary = sample_path.with_suffix(".tmp")
    ids = np.arange(COHORT_SIZE, dtype=np.int64)
    with temporary.open("wb") as file:
        np.savez(file, images, ids=ids, labels=ids % 1000)
    temporary.replace(sample_path)
    summary = {"protocol": PROTOCOL, "complete": True, "mode": signature["mode"], "seed": signature["seed"],
               "num_steps": signature["num_steps"], "samples": COHORT_SIZE, "classes": 1000,
               "samples_per_class": 1, "shards": len(workers), "sample_archive": artifact(sample_path),
               "request": artifact(output / "request.json"), "global_noise_sha256": noise_signature[0],
               "counters_sum": {key: sum(worker["counters"][key] for worker in workers) for key in COUNTERS},
               "worker_timings_seconds_sum": {key: sum(worker["timings"][key] for worker in workers) for key in TIMES},
               "worker_total_wall_seconds_max_not_makespan": max(worker["timings"]["total_wall_seconds_excluding_imports_and_final_summary_write"] for worker in workers),
               "worker_peak_gpu_memory_allocated_bytes_max": max(worker["peak_gpu_memory_allocated_bytes"] for worker in workers),
               "cost_boundary": "Worker wall/CPU sums measure aggregate process cost, not elapsed makespan or CUDA kernel time. Shared training/validation/preparation/benchmark costs are not multiplied by shard count and are not included here.",
               "merge_wall_seconds_excluding_final_summary_write": time.perf_counter()-wall_started,
               "merge_cpu_seconds_excluding_final_summary_write": time.process_time()-cpu_started,
               "fid_performed": False, "gpu_used_for_merge": False}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.shards, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()

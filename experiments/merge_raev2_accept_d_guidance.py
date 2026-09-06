#!/usr/bin/env python3
"""Validate complete accept-D proposal histories and merge a paired 1K cohort.

No score or image ranking is used. Each output is exactly the recorded first
proposal or first Bernoulli-accepted proposal for each of the 1000 classes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_history(records, ids):
    """Check class coverage, consecutive trials and the first-success rule."""
    histories = defaultdict(list)
    seen_slots = set()
    seen_class_batches = set()
    for record in records:
        index = record["class_id"]
        require(index in ids and record["label"] == index, "unexpected class/label")
        key = (record["proposal_batch"], record["slot"])
        require(key not in seen_slots, "duplicate proposal batch slot")
        seen_slots.add(key)
        require((index, key[0]) not in seen_class_batches, "duplicate class in proposal batch")
        seen_class_batches.add((index, key[0]))
        require(0 <= key[1] < 8 and key[0] >= 0, "invalid proposal batch slot")
        d, uniform, log_d = record["D"], record["uniform"], record["log_D"]
        require(all(math.isfinite(x) for x in (d, uniform, log_d)), "nonfinite acceptance evidence")
        require(0 < d <= 1 and 0 <= uniform < 1, "invalid probability/uniform")
        require(math.isclose(math.log(d), log_d, rel_tol=0, abs_tol=1e-12), "D/log_D mismatch")
        require(record["accepted"] is (uniform < d), "Bernoulli decision mismatch")
        histories[index].append(record)
    require(set(histories) == set(ids), "incomplete class coverage")
    for index, history in histories.items():
        require([r["attempt"] for r in history] == list(range(1, len(history)+1)), "nonconsecutive attempts")
        require(history[-1]["accepted"] and not any(r["accepted"] for r in history[:-1]), "not first acceptance")
        require(history[0]["noise_source"] == "initial_bank", "first proposal did not use paired noise bank")
        require(all(r["noise_source"] == "retry_generator" for r in history[1:]), "retry source mismatch")
    return dict(histories)


def request_identity(request):
    keys = ("protocol", "world_size", "seed", "batch_size", "num_steps", "ig_scale", "ig_interval",
            "sampling_precision", "pixel_arithmetic", "forward_layout", "classifier_precision", "acceptance",
            "attempt_limit", "forced_acceptance", "time_grid", "frozen_train_m", "train_m_not_used_for_acceptance")
    result = {key: request[key] for key in keys}
    for key in ("probe", "noise_bank", "noise_manifest", "config", "checkpoint", "feature_request"):
        result[key] = request[key]["sha256"]
    result["source_sha256"] = {key: item["sha256"] for key, item in request["source_snapshots"].items()}
    result["decoder_sha256"] = {key: item["sha256"] for key, item in request["decoder_artifacts"].items()}
    result["dino_sha256"] = request["dino_weights"]["sha256"]
    return result


def validate_shard(path, rank, bank, identity):
    import torch

    request, summary = read_json(path / "request.json"), read_json(path / "summary.json")
    ids = [index for index in range(1000) if (index // 8) % 4 == rank]
    require(request_identity(request) == identity, "worker protocol identities differ")
    require(summary["complete"] and summary["rank"] == rank and summary["world_size"] == 4, "incomplete/wrong shard")
    require(summary["class_ids"] == ids and request["assigned_class_ids"] == ids, "incorrect shard class assignment")
    require(summary["samples"] == len(ids) and summary["seed"] == identity["seed"], "shard count/seed mismatch")
    require(request["retry_noise_seed"] == identity["seed"]+1000000+rank, "incorrect or reused retry RNG stream")
    require(request["acceptance_uniform_seed"] == identity["seed"]+2000000+rank, "incorrect or reused uniform RNG stream")
    require(not request["forced_acceptance"] and request["attempt_limit"] is None, "rejection was capped")
    require(not summary["fid_read_or_computed"] and not summary["forced_acceptance"], "method used forbidden result-dependent selection")
    for key, filename in (("first_proposals", "first_proposals.npz"),
                          ("accepted_samples", "accepted_samples.npz"), ("proposals_jsonl", "proposals.jsonl")):
        item = summary[key]
        require(Path(item["path"]).name == filename, f"{key} summary points to wrong archive")
        require(sha256(path / filename) == item["sha256"], f"{key} archive hash mismatch")
    require(summary["probe_sha256"] == identity["probe"], "probe hash mismatch")
    require(summary["noise_bank_sha256"] == identity["noise_bank"], "noise bank hash mismatch")
    require(summary["noise_manifest_sha256"] == identity["noise_manifest"], "noise manifest hash mismatch")
    for item in request["source_snapshots"].values():
        require(sha256(path / "sources" / Path(item["path"]).name) == item["sha256"], "source snapshot hash mismatch")
    records = [json.loads(line) for line in (path / "proposals.jsonl").read_text().splitlines() if line]
    histories = validate_history(records, ids)
    require([(r["proposal_batch"], r["slot"]) for r in records] == sorted((r["proposal_batch"], r["slot"]) for r in records),
            "proposal history not in execution order")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["proposal_batch"]].append(record)
    require(sorted(grouped) == list(range(summary["proposal_batches"])), "missing proposal batch")
    uniform_rng = torch.Generator(device="cpu").manual_seed(request["acceptance_uniform_seed"])
    pending = deque(ids)
    padding = 0
    for batch_index, batch_records in grouped.items():
        batch_path = path / "batches" / f"batch_{batch_index:06d}.npz"
        digest = sha256(batch_path)
        with np.load(batch_path, allow_pickle=False) as archive:
            images, class_ids = archive["arr_0"], archive["class_ids"]
            require(images.dtype == np.uint8 and images.shape == (8, 256, 256, 3), "invalid full batch images")
            require(class_ids.shape == (8,), "invalid batch class vector")
            active = class_ids >= 0
            expected_ids = [pending.popleft() for _ in range(min(8, len(pending)))]
            require(class_ids.tolist() == expected_ids + [-1]*(8-len(expected_ids)), "proposal order differs from fixed FIFO")
            require(np.array_equal(active, archive["valid"]), "dummy mask mismatch")
            require(np.array_equal(archive["model_labels"], np.where(active, class_ids, 0)), "model labels mismatch")
            require(np.all(class_ids[~active] == -1), "unknown dummy class")
            require([r["slot"] for r in batch_records] == np.flatnonzero(active).tolist(), "batch records omit/duplicate active slots")
            uniforms = torch.rand(len(batch_records), generator=uniform_rng, device="cpu", dtype=torch.float64).numpy()
            require(np.array_equal(uniforms, [r["uniform"] for r in batch_records]), "acceptance uniform RNG replay mismatch")
            for record in batch_records:
                slot = record["slot"]
                require(record["archive"] == str(batch_path.relative_to(path)) and record["archive_sha256"] == digest,
                        "proposal batch provenance mismatch")
                require(record["class_id"] == int(class_ids[slot]), "proposal class differs from generated class")
                require(record["pixels_sha256"] == array_hash(images[slot]), "proposal pixel hash mismatch")
                for key in ("noise_sha256", "endpoint_sha256"):
                    require(record[key] == str(archive[key][slot]), f"proposal {key} mismatch")
                if not record["accepted"]:
                    pending.append(record["class_id"])
            padding += int((~active).sum())
    require(not pending, "FIFO has unfinished classes")
    state = torch.load(path / "rng_and_queue_state.pt", map_location="cpu", weights_only=True)
    require(torch.equal(state["uniform_rng"], uniform_rng.get_state()), "final uniform RNG state mismatch")
    require(not state["queue"]["pending"] and state["queue"]["inflight"] is None
            and state["queue"]["accepted"] == ids, "final queue state mismatch")
    valid = len(records)
    expected_counts = {"valid_proposals": valid, "padding_proposals": padding,
                       "full_model_calls": len(grouped)*100,
                       "full_sample_evaluations_including_padding": (valid+padding)*100,
                       "decoder_sample_evaluations_including_padding": valid+padding,
                       "classifier_evaluations": valid, "acceptance_uniforms_drawn": valid,
                       "initial_bank_samples_used": len(ids),
                       "retry_noise_samples_drawn_including_padding": valid+padding-len(ids)}
    for key, value in expected_counts.items():
        require(summary[key] == value, f"incorrect computational accounting: {key}")
    require(summary["attempts_by_class"] == {str(i): len(histories[i]) for i in ids}, "attempt counts mismatch")
    outputs = {}
    for kind, archive_name, record_name, position in (
            ("first", "first_proposals.npz", "first_proposal_records.json", 0),
            ("accepted", "accepted_samples.npz", "accepted_proposal_records.json", -1)):
        selected_records = read_json(path / record_name)["records"]
        require(selected_records == [histories[i][position] for i in ids], "selected output differs from acceptance history")
        with np.load(path / archive_name, allow_pickle=False) as archive:
            images = archive["arr_0"]
            require(images.dtype == np.uint8 and images.shape == (len(ids), 256, 256, 3), "invalid selected images")
            require(np.array_equal(archive["ids"], ids) and np.array_equal(archive["labels"], ids), "selected class coverage mismatch")
            for image, record in zip(images, selected_records):
                require(array_hash(image) == record["pixels_sha256"], "selected image hash mismatch")
            outputs[kind] = images.copy()
    for index in ids:
        require(histories[index][0]["noise_sha256"] == array_hash(bank[index]), "first proposal paired noise mismatch")
    return {"ids": ids, "histories": histories, "outputs": outputs, "counts": expected_counts,
            "summary": summary, "request_sha256": sha256(path / "request.json"), "summary_sha256": sha256(path / "summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), f"refusing to overwrite {out}")
    request = read_json(source / "shard_00_of_04/request.json")
    identity = request_identity(request)
    require(identity["world_size"] == 4 and identity["num_steps"] == 100 and identity["batch_size"] == 8, "wrong fixed protocol")
    manifest_path, bank_path = Path(request["noise_manifest"]["path"]), Path(request["noise_bank"]["path"])
    require(sha256(manifest_path) == identity["noise_manifest"] and sha256(bank_path) == identity["noise_bank"], "paired noise artifact changed")
    manifest = read_json(manifest_path)
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    require(bank.shape == (1000, 1024, 16, 16) and bank.dtype == np.float32, "paired noise shape/dtype mismatch")
    digest = hashlib.sha256()
    for start in range(0, 1000, 8):
        digest.update(np.ascontiguousarray(bank[start:start+8]).tobytes())
    require(digest.hexdigest() == manifest["noise_sha256"], "paired raw noise mismatch")
    shards = [validate_shard(source / f"shard_{rank:02d}_of_04", rank, bank, identity) for rank in range(4)]
    require(sorted(index for shard in shards for index in shard["ids"]) == list(range(1000)), "global class coverage mismatch")
    out.mkdir(parents=True, exist_ok=True)
    output_artifacts = {}
    for kind, filename in (("first", "first_proposals.npz"), ("accepted", "accepted_samples.npz")):
        images = np.empty((1000, 256, 256, 3), dtype=np.uint8)
        for shard in shards:
            images[shard["ids"]] = shard["outputs"][kind]
        path = out / filename
        with path.with_suffix(".tmp").open("wb") as file:
            np.savez(file, images, ids=np.arange(1000, dtype=np.int64), labels=np.arange(1000, dtype=np.int64))
        path.with_suffix(".tmp").replace(path)
        output_artifacts[kind] = {"path": str(path), "sha256": sha256(path), "pixels_sha256": array_hash(images)}
    counts = {key: sum(shard["counts"][key] for shard in shards) for key in shards[0]["counts"]}
    histories = {index: history for shard in shards for index, history in shard["histories"].items()}
    attempts = [len(histories[index]) for index in range(1000)]
    summary = {"protocol": "raev2_accept_D_paired_merge_v1", "complete": True, "samples": 1000,
               "identity": identity, "first_noise_sha256": manifest["noise_sha256"],
               "labels_sha256": array_hash(np.arange(1000, dtype=np.int64)), "outputs": output_artifacts,
               "counts": counts, "first_attempt_accepted": sum(n == 1 for n in attempts),
               "attempts_by_class": attempts, "maximum_attempts": max(attempts),
               "mean_valid_proposals_per_image": counts["valid_proposals"]/1000,
               "actual_model_nfe_per_image": counts["full_sample_evaluations_including_padding"]/1000,
               "model_nfe_ratio_to_paired_baseline": counts["full_sample_evaluations_including_padding"] / 100000,
               "acceptance_rate_valid": 1000 / counts["valid_proposals"],
               "forced_acceptance": False, "fid_used_for_selection": False,
               "verification": "complete proposal histories, first-success rule, all noise/pixel/source hashes, batch labels and NFE counts",
               "shards": [{"rank": rank, "request_sha256": item["request_sha256"], "summary_sha256": item["summary_sha256"],
                           "elapsed_seconds": item["summary"]["elapsed_seconds"]} for rank, item in enumerate(shards)],
               "merger_source_sha256": sha256(Path(__file__))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out / "merger_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in summary.items() if key != "attempts_by_class"}, indent=2))


if __name__ == "__main__":
    main()

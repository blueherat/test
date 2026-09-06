from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import merge_raev2_observable_potential as merger


def save_json(path, payload):
    path.write_text(json.dumps(payload))
    return merger.artifact(path)


def save_images(path, ids, *, value_offset=0, labels=None, dtype=np.uint8):
    images = np.broadcast_to((ids+value_offset).astype(dtype)[:, None, None, None],
                             (len(ids), *merger.IMAGE_SHAPE)).copy()
    np.savez(path, images, ids=ids, labels=ids % 1000 if labels is None else labels)
    return merger.artifact(path)


def make_shards(tmp_path, monkeypatch, *, count=1000, world=2):
    # Keep the real 1K/5K ID and class populations, with small CPU image payloads.
    monkeypatch.setattr(merger, "IMAGE_SHAPE", (2, 2, 3))
    tmp_path.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "weights.bin"
    external.write_bytes(b"frozen artifact")
    identity = merger.artifact(external)
    directories = []
    for rank in range(world):
        directory = tmp_path / f"shard{rank}"
        directory.mkdir()
        (directory / "batches").mkdir()
        sources = {}
        for relative in merger.REQUIRED_SOURCES:
            path = directory / "sources" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
            sources[relative] = merger.artifact(path)
        noise_path = directory / "paired_noise_audit.npz"
        np.savez(noise_path, first_noise=np.zeros((1024,16,16), np.float32), rng_state=np.arange(16, dtype=np.uint8))
        batches = merger.expected_batch_ids(rank, world, count=count)
        records = []
        for batch_ids in batches:
            path = directory / "batches" / f"{int(batch_ids[0]):06d}_{int(batch_ids[-1])+1:06d}.npz"
            records.append({"global_ids": batch_ids.tolist(), "noise_sha256": "a"*64,
                            "endpoint_sha256": "b"*64, "archive": save_images(path, batch_ids),
                            "trajectory_wall_seconds": 1.25, "decode_and_uint8_wall_seconds": .25})
        ids = np.concatenate(batches)
        manifest = save_json(directory / "batch_manifest.json", {"batches": records})
        sample = save_images(directory / "samples.npz", ids)
        request = {"protocol": merger.SAMPLER_PROTOCOL, "mode": "potential", "seed": 918, "num_steps": 100,
                   "batch_size": 8, "sample_count": count, "shard_index": rank, "num_shards": world,
                   "sources": sources, "time_grid": np.linspace(1,0,101).tolist(),
                   "global_noise_sha256": "c"*64,
                   "global_labels_sha256": merger.array_sha256(np.arange(count,dtype=np.int64) % 1000),
                   "paired_noise_audit": merger.artifact(noise_path), "cuda_device": "test-only CPU fixture"}
        request.update({key: identity for key in ("config", "baseline_checkpoint", "potential_checkpoint",
                                                 "decoder_checkpoint", "normalization_stats")})
        request_record = save_json(directory / "request.json", request)
        summary = {"complete": True, "protocol": merger.SAMPLER_PROTOCOL, "mode": "potential", "seed": 918,
                   "num_steps": 100, "image_sampling_performed": True, "fid_performed": False,
                   "training_performed": False, "request": request_record,
                   "global_noise_sha256": "c"*64, "samples": len(ids), "global_cohort_size": count,
                   "global_ids": ids.tolist(), "stage2_forward_calls": len(batches)*100,
                   "stage2_sample_forwards": len(ids)*100,
                   "potential_forward_input_gradient_calls": len(batches)*100,
                   "potential_sample_input_gradients": len(ids)*100,
                   "potential_parameter_backward_calls": 0, "decoder_forward_calls": len(batches),
                   "decoder_sample_forwards": len(ids), "stage2_nfe_per_sample": 100,
                   "sample_archive": sample, "archive_sha256": sample["sha256"], "batch_manifest": manifest,
                   "peak_gpu_memory_allocated_bytes": 0}
        summary.update({key: 2.0 for key in merger.TIMES})
        summary.update(trajectory_wall_seconds=1.25*len(batches),
                       decode_and_uint8_wall_seconds=.25*len(batches),
                       sampling_wall_including_output_seconds=1.5*len(batches),
                       total_wall_seconds_excluding_imports_and_final_summary_write=1.5*len(batches)+6)
        save_json(directory / "summary.json", summary)
        directories.append(directory)
    return directories


@pytest.fixture
def shards(tmp_path, monkeypatch):
    return make_shards(tmp_path / "cohort1k", monkeypatch)


def rewrite_request(directory, update):
    path = directory / "request.json"
    request = json.loads(path.read_text())
    update(request)
    record = save_json(path, request)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["request"] = record
    save_json(summary_path, summary)


def rewrite_summary(directory, update):
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    update(summary)
    save_json(path, summary)


def test_merge_sorts_global_ids_and_accounts_worker_cost_without_gpu(shards, tmp_path):
    result = merger.merge(shards[::-1], tmp_path / "merged", count=1000)
    assert result["complete"] and result["gpu_used_for_merge"] is False
    assert result["samples"] == 1000 and result["samples_per_class"] == 1
    assert result["counters_sum"]["stage2_sample_forwards"] == 100000
    assert result["worker_timings_seconds_sum"]["trajectory_wall_seconds"] == 125*1.25
    assert result["worker_total_wall_seconds_max_not_makespan"] == 63*1.5+6
    assert result["protocol"] == "raev2_observable_potential_merge_v1"
    with np.load(result["sample_archive"]["path"], allow_pickle=False) as archive:
        assert archive["arr_0"].shape == (1000,*merger.IMAGE_SHAPE)
        assert np.array_equal(archive["ids"], np.arange(1000))
        assert np.array_equal(archive["labels"], archive["ids"])
        assert np.array_equal(archive["arr_0"][:,0,0,0], np.arange(1000).astype(np.uint8))


@pytest.mark.parametrize("mutation", ("missing", "duplicate_rank", "seed", "global_noise", "source", "counter"))
def test_reject_incomplete_or_incompatible_shards(shards, tmp_path, mutation):
    if mutation == "missing":
        shards = shards[:1]
    elif mutation == "duplicate_rank":
        rewrite_request(shards[1], lambda r: r.update(shard_index=0))
    elif mutation == "seed":
        rewrite_request(shards[1], lambda r: r.update(seed=919))
        rewrite_summary(shards[1], lambda r: r.update(seed=919))
    elif mutation == "global_noise":
        rewrite_request(shards[1], lambda r: r.update(global_noise_sha256="d"*64))
        rewrite_summary(shards[1], lambda r: r.update(global_noise_sha256="d"*64))
    elif mutation == "source":
        request = json.loads((shards[1]/"request.json").read_text())
        Path(next(iter(request["sources"].values()))["path"]).write_text("tampered")
    else:
        rewrite_summary(shards[1], lambda r: r.update(stage2_sample_forwards=799))
    with pytest.raises(ValueError):
        merger.merge(shards, tmp_path / "rejected", count=1000)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mutation", ("batch_pixels", "labels", "dtype"))
def test_rehashed_batch_or_archive_cannot_hide_image_mismatch(shards, tmp_path, mutation):
    directory = shards[0]
    manifest_path = directory / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    batch = manifest["batches"][0]
    ids = np.array(batch["global_ids"], dtype=np.int64)
    if mutation == "batch_pixels":
        batch["archive"] = save_images(Path(batch["archive"]["path"]), ids, value_offset=1)
        record = save_json(manifest_path, manifest)
        rewrite_summary(directory, lambda s: s.update(batch_manifest=record))
    else:
        path = directory / "samples.npz"
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["ids"]
        record = save_images(path, ids, labels=ids+1 if mutation == "labels" else None,
                             dtype=np.float32 if mutation == "dtype" else np.uint8)
        rewrite_summary(directory, lambda s: s.update(sample_archive=record, archive_sha256=record["sha256"]))
    with pytest.raises(ValueError):
        merger.merge(shards, tmp_path / "rejected", count=1000)


def test_merge_5k_cohort_binds_count_and_repeated_class_labels(tmp_path, monkeypatch):
    shards = make_shards(tmp_path / "cohort5k", monkeypatch, count=5000)
    result = merger.merge(shards[::-1], tmp_path / "merged", count=5000)
    assert result["samples"] == 5000 and result["samples_per_class"] == 5
    assert result["counters_sum"]["decoder_sample_forwards"] == 5000
    assert result["counters_sum"]["stage2_sample_forwards"] == 500000
    request = json.loads(Path(result["request"]["path"]).read_text())
    assert request["sample_count"] == request["common_request"]["sample_count"] == 5000
    assert [worker["samples"] for worker in request["workers"]] == [2504, 2496]
    with np.load(result["sample_archive"]["path"], allow_pickle=False) as archive:
        assert archive["arr_0"].shape == (5000, *merger.IMAGE_SHAPE)
        assert np.array_equal(archive["ids"], np.arange(5000))
        assert np.array_equal(archive["labels"], np.arange(5000) % 1000)
        assert np.array_equal(np.bincount(archive["labels"]), np.full(1000, 5))
    with pytest.raises(ValueError, match="sample count"):
        merger.merge(shards, tmp_path / "wrong_count", count=1000)
    # Hashing IDs was correct only for the historical 1K population.
    rewrite_request(shards[0], lambda r: r.update(
        global_labels_sha256=merger.array_sha256(np.arange(5000, dtype=np.int64))))
    with pytest.raises(ValueError, match="global label digest"):
        merger.merge(shards, tmp_path / "bad_labels", count=5000)


def test_merge_rejects_mixed_1k_5k_requests(shards, tmp_path, monkeypatch):
    larger = make_shards(tmp_path / "cohort5k", monkeypatch, count=5000)
    for count in (1000, 5000):
        output = tmp_path / f"mixed{count}"
        with pytest.raises(ValueError, match="sample count"):
            merger.merge([shards[0], larger[1]], output, count=count)
        assert not output.exists()


@pytest.mark.parametrize("field,value", (("sample_count", 5000), ("sample_count", 1000.0),
                                         ("global_cohort_size", 5000), ("global_cohort_size", 1000.0)))
def test_merge_rejects_request_or_summary_count_mismatch(shards, tmp_path, field, value):
    if field == "sample_count":
        rewrite_request(shards[0], lambda r: r.update({field: value}))
    else:
        rewrite_summary(shards[0], lambda s: s.update({field: value}))
    with pytest.raises(ValueError, match="sample count"):
        merger.merge(shards, tmp_path / "rejected", count=1000)


def test_merge_count_cli_and_partition_limits(tmp_path):
    common = ["--shards", "shard0", "--output-dir", str(tmp_path)]
    assert merger.parse_args(common).num_samples == 1000
    assert merger.parse_args([*common, "--num-samples", "5000"]).num_samples == 5000
    assert merger.expected_batch_ids(624, 625, count=5000)[0].tolist() == list(range(4992, 5000))
    for count in ("0", "-1000", "999", "1008", "5001"):
        with pytest.raises(SystemExit):
            merger.parse_args([*common, "--num-samples", count])
    for count in (0, -1000, 999, 1008, 5001, 1000.0, True):
        with pytest.raises(ValueError, match="positive multiple"):
            merger.merge([], tmp_path, count=count)

"""Identity/order safeguards for appending independent samples to a screened 1K."""
import hashlib
import json

import numpy as np
import pytest

from experiments import merge_raev2_1k_extension as module


def write_json(path, value):
    path.write_text(json.dumps(value))
    return module.artifact(path)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "IMAGE_SHAPE", (2, 2, 3))
    blocks = []
    for index in range(5):
        block = {"id": f"block{index}", "role": "screen" if index == 0 else "confirmation",
                 "seed": index + 101, "arms": {}}
        for arm_index, name in enumerate(("official", "candidate")):
            directory = tmp_path / f"{index}-{name}"
            directory.mkdir()
            ids = np.arange(1000, dtype=np.int64)
            pixels = np.broadcast_to(((ids + 17 * index + arm_index) % 256).astype(np.uint8)[:, None, None, None],
                                     (1000, 2, 2, 3)).copy()
            archive = directory / "samples.npz"
            np.savez(archive, pixels, ids=ids, labels=ids)
            manifest = []
            for start in range(0, 1000, 8):
                batch_file = directory / f"{start}.npz"
                np.savez(batch_file, pixels[start:start+8])
                manifest.append({"global_ids": ids[start:start+8].tolist(),
                                 "labels_sha256": module.raw_hash(ids[start:start+8]),
                                 "noise_sha256": digest((index, start)), "archive": module.artifact(batch_file)})
            request = {"mode": name, "seed": block["seed"], "sample_count": 1000, "num_steps": 100, "batch_size": 8}
            summary = {"complete": True, "samples": 1000, "seed": block["seed"], "mode": name,
                       "archive_sha256": module.sha256(archive), "noise_sha256": digest(index),
                       "labels_sha256": module.raw_hash(ids), "elapsed_seconds": 12.0}
            block["arms"][name] = {"request": write_json(directory / "request.json", request),
                                    "summary": write_json(directory / "summary.json", summary),
                                    "batch_manifest": write_json(directory / "batch_manifest.json", {"batches": manifest}),
                                    "samples": module.artifact(archive)}
        blocks.append(block)
    plan = {"protocol": module.PROTOCOL, "family": "test", "blocks": blocks}
    return tmp_path, plan


def test_preserves_all_old_and_new_pixels_and_labels(inputs):
    root, plan = inputs
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    result = module.merge(plan_path, root / "out")
    assert result["complete"] and result["gpu_model_calls"] == 0
    for name in ("official", "candidate"):
        with np.load(root / "out" / name / "pooled5k" / "samples.npz") as pooled, np.load(root / "out" / name / "new4k" / "samples.npz") as new:
            assert np.array_equal(pooled["arr_0"][1000:], new["arr_0"])
            assert np.array_equal(pooled["labels"], np.tile(np.arange(1000), 5))
            assert np.array_equal(new["ids"], np.arange(4000))
            assert np.array_equal(new["block_index"], np.repeat(np.arange(1, 5), 1000))
            for index, block in enumerate(plan["blocks"]):
                with np.load(block["arms"][name]["samples"]["path"]) as source:
                    assert np.array_equal(pooled["arr_0"][1000*index:1000*(index+1)], source["arr_0"])
    summary = json.loads((root / "out" / "official" / "pooled5k" / "summary.json").read_text())
    assert summary["costs_by_unchanged_source_field"]["elapsed_seconds"]["sum"] == 60
    assert summary["costs_by_unchanged_source_field"]["total_wall_seconds_before_summary"]["sum"] is None


def test_rejects_actual_pairing_mismatch(inputs):
    root, plan = inputs
    source = plan["blocks"][1]["arms"]["candidate"]
    path = module.Path(source["summary"]["path"])
    value = json.loads(path.read_text())
    value["noise_sha256"] = digest("wrong")
    source["summary"] = write_json(path, value)
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match="actual inputs not paired"):
        module.merge(plan_path, root / "out")
    assert (root / "out" / "failure.json").is_file()


def test_rejects_changed_old_artifact(inputs):
    root, plan = inputs
    source = plan["blocks"][0]["arms"]["official"]
    module.Path(source["samples"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 changed"):
        module.load_source(source, plan["blocks"][0], "official")


def test_rejects_missing_block_or_duplicate_seed():
    plan = {"protocol": module.PROTOCOL, "family": "test", "blocks": []}
    with pytest.raises(ValueError, match="existing screen"):
        module.validate_plan(plan)
    plan["blocks"] = [{"id": str(i), "seed": 7, "role": "screen" if i == 0 else "confirmation", "arms": {"a": {}, "b": {}}} for i in range(5)]
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_plan(plan)


def test_rejects_heterogeneous_baseline_step_count(inputs):
    root, plan = inputs
    source = plan["blocks"][1]["arms"]["official"]
    path = module.Path(source["request"]["path"])
    value = json.loads(path.read_text())
    value["num_steps"] = 201
    source["request"] = write_json(path, value)
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match="method changed"):
        module.merge(plan_path, root / "out")

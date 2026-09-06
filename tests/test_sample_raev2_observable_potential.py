from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import sample_raev2_observable_potential as sampler


def test_zero_correction_exact_euler_and_official_denominator_floor():
    z = torch.tensor([[[[.75, -1.5]]]], dtype=torch.float32)
    g = torch.tensor([[[[-.25, .125]]]], dtype=torch.float32)
    for current, following in ((.5, .45), (.07476635, 0.), (.03, .02)):
        expected = z-(current-following)*((z-g)/max(current, .05))
        baseline = sampler.euler_update(z, g, None, current, following)
        assert torch.equal(baseline, expected)
        assert torch.equal(sampler.euler_update(z, g, torch.zeros_like(z), current, following), baseline)
    corrected = sampler.euler_update(z, g, torch.ones_like(z), .03, .02)
    assert torch.equal(corrected, z-.01*((z-g-1)/.05))
    with pytest.raises(ValueError, match="next"):
        sampler.euler_update(z, g, None, .1, .1)


def test_step_uses_potential_at_last_ig_off_query_under_no_grad(monkeypatch):
    z = torch.ones((2, 2, 1, 1))
    labels = torch.tensor([0, 1])
    calls = []

    def clean_forward(model, state, times, y):
        assert not torch.is_grad_enabled()
        assert torch.equal(y, labels)
        calls.append("baseline")
        return state*.25

    class Potential:
        def clean_correction(self, state, times, y):
            assert not torch.is_grad_enabled()
            assert torch.all(times < .1)
            calls.append("potential")
            return state*.125

    monkeypatch.setattr(sampler, "clean_forward", clean_forward)
    with torch.no_grad():
        result = sampler.sampling_step(None, Potential(), z, labels, .074, 0, use_potential=True)
    assert calls == ["baseline", "potential"]
    assert torch.equal(result, sampler.euler_update(z, z*.25, z*.125, .074, 0))
    calls.clear()
    sampler.sampling_step(None, None, z, labels, .074, 0, use_potential=False)
    assert calls == ["baseline"]
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        sampler.sampling_step(None, None, z, labels, .074, 0, use_potential=False)


def test_native_bf16_guidance_arithmetic_is_preserved():
    full = torch.tensor([1.03125, 1.03125], dtype=torch.bfloat16).reshape(2, 1, 1, 1)
    base = torch.tensor([.3125, .3125], dtype=torch.bfloat16).reshape_as(full)
    times = torch.tensor([.5, .074])
    result = sampler.official_clean_from_heads(full, base, times)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result[0], (base+1.78*(full-base))[0])
    assert torch.equal(result[1], full[1])
    assert not torch.equal(result[0].float(), (base.float()+1.78*(full.float()-base.float()))[0])


def test_noise_is_identical_across_partition_and_independent_of_global_rng():
    noise, final_rng = sampler.paired_noise(918, torch.device("cpu"), count=24, latent_shape=(2, 1, 1))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(729)
        torch.randn(111)
        repeated, repeated_rng = sampler.paired_noise(918, torch.device("cpu"), count=24, latent_shape=(2, 1, 1))
    assert torch.equal(noise, repeated)
    assert torch.equal(final_rng, repeated_rng)
    for shards in (1, 2, 3):
        reconstructed = torch.empty_like(noise)
        seen = []
        for rank in range(shards):
            local, _ = sampler.paired_noise(918, torch.device("cpu"), count=24, latent_shape=(2, 1, 1))
            for ids in sampler.assigned_batches(rank, shards, count=24):
                reconstructed[ids] = local[ids]
                seen.extend(ids.tolist())
        assert sorted(seen) == list(range(24))
        assert torch.equal(reconstructed, noise)


def test_fixed_candidate_grid_and_cost_baseline_cli(tmp_path):
    common = ["--output-dir", str(tmp_path), "--potential-checkpoint", "final.pt"]
    assert sampler.parse_args([*common, "--mode", "official", "--num-steps", "154"]).num_steps == 154
    for mode in ("potential", "benchmark"):
        with pytest.raises(SystemExit):
            sampler.parse_args([*common, "--mode", mode, "--num-steps", "154"])
    with pytest.raises(SystemExit):
        sampler.parse_args([*common, "--mode", "benchmark", "--num-shards", "2"])


def test_sample_count_cli_and_count_dependent_shard_limits(tmp_path):
    common = ["--output-dir", str(tmp_path), "--potential-checkpoint", "final.pt"]
    for mode in ("official", "potential", "benchmark"):
        assert sampler.parse_args([*common, "--mode", mode]).num_samples == 1000
    for mode in ("official", "potential"):
        args = sampler.parse_args([*common, "--mode", mode, "--num-samples", "5000",
                                  "--num-shards", "625", "--shard-index", "624"])
        assert args.num_samples == 5000 and args.num_shards == 625
    for count in ("0", "-1000", "999", "1008", "5001"):
        with pytest.raises(SystemExit):
            sampler.parse_args([*common, "--mode", "potential", "--num-samples", count])
    for extra in (("--num-shards", "126"), ("--num-samples", "5000", "--num-shards", "626"),
                  ("--num-samples", "5000", "--num-shards", "625", "--shard-index", "625")):
        with pytest.raises(SystemExit):
            sampler.parse_args([*common, "--mode", "potential", *extra])
    with pytest.raises(SystemExit):
        sampler.parse_args([*common, "--mode", "benchmark", "--num-samples", "5000"])


@pytest.mark.parametrize("count", (1000, 5000))
def test_complete_cohort_partition_and_actual_class_label_digest(count):
    for shards in (1, 3, count // 8):
        batches = [batch for rank in range(shards)
                   for batch in sampler.assigned_batches(rank, shards, count=count)]
        ids = np.concatenate(batches)
        assert np.array_equal(np.sort(ids), np.arange(count))
        assert all(len(batch) == 8 and np.array_equal(np.diff(batch), np.ones(7)) for batch in batches)
    labels = np.arange(count, dtype=np.int64) % 1000
    assert np.array_equal(np.bincount(labels), np.full(1000, count // 1000))
    expected = sampler.tensor_sha256(torch.from_numpy(labels))
    assert sampler.global_labels_sha256(count=count) == expected
    legacy_ids_hash = sampler.tensor_sha256(torch.arange(count, dtype=torch.long))
    assert (expected == legacy_ids_hash) == (count == 1000)


@pytest.mark.parametrize("count", (1000, 5000))
def test_noise_requests_one_complete_cuda_shape_without_gpu(monkeypatch, count):
    calls = []
    device = torch.device("cuda:0")

    class Generator:
        def __init__(self, *, device):
            calls.append(("generator", device))

        def manual_seed(self, seed):
            calls.append(("seed", seed))
            return self

        def get_state(self):
            return torch.arange(4, dtype=torch.uint8)

    def randn(shape, *, generator, device, dtype):
        assert isinstance(generator, Generator)
        calls.append(("draw", shape, device, dtype))
        return torch.empty(0)  # Avoid allocating the production cohort or accessing CUDA.

    monkeypatch.setattr(torch, "Generator", Generator)
    monkeypatch.setattr(torch, "randn", randn)
    _, rng = sampler.paired_noise(918, device, count=count)
    assert calls == [("generator", device), ("seed", 918),
                     ("draw", (count, 1024, 16, 16), device, torch.float32)]
    assert torch.equal(rng, torch.arange(4, dtype=torch.uint8))


def test_sample_passes_5k_count_to_partition_and_wraps_labels(tmp_path, monkeypatch):
    noise = torch.arange(5000, dtype=torch.float32).reshape(5000, 1, 1, 1)
    labels_seen = []

    def step(model, potential, state, labels, current, following, *, use_potential):
        assert use_potential and torch.equal(state, noise[1000:1008])
        labels_seen.append(labels.clone())
        return state

    class Decoder:
        def decode(self, state):
            return torch.full((len(state), 3, 256, 256), .5, dtype=torch.bfloat16)

    monkeypatch.setattr(sampler, "sampling_step", step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    args = SimpleNamespace(shard_index=125, num_shards=625, mode="potential", num_steps=100)
    result = sampler.sample(None, None, Decoder(), noise, torch.linspace(1, 0, 101).tolist(),
                            args, tmp_path, torch.device("cpu"), count=5000)
    assert result["global_cohort_size"] == 5000 and result["samples"] == 8
    assert result["global_ids"] == list(range(1000, 1008))
    assert len(labels_seen) == result["stage2_forward_calls"] == 100
    assert all(torch.equal(labels, torch.arange(8)) for labels in labels_seen)
    with np.load(result["sample_archive"]["path"], allow_pickle=False) as archive:
        assert archive["ids"].tolist() == list(range(1000, 1008))
        assert archive["labels"].tolist() == list(range(8))
    with pytest.raises(ValueError, match="complete requested"):
        sampler.sample(None, None, None, noise[:1000], [], args, tmp_path,
                       torch.device("cpu"), count=5000)


def test_bf16_pixel_export_and_adm_npz_format(tmp_path):
    decoded = torch.full((1, 3, 256, 256), .5, dtype=torch.bfloat16)
    images = sampler.native_uint8(decoded)
    assert images.shape == (1, 256, 256, 3) and images.dtype == np.uint8
    assert np.all(images == 127)
    with pytest.raises(ValueError, match="BF16"):
        sampler.native_uint8(decoded.float())
    path = tmp_path / "samples.npz"
    sampler.save_npz(path, images, ids=np.array([7]), labels=np.array([7]))
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive["arr_0"], images)
        assert archive["ids"].tolist() == [7]


def training_fixture(tmp_path):
    def save(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return sampler.artifact(path)
    config = {"path": "config", "sha256": "config"}
    baseline = {"path": "baseline", "sha256": "baseline"}
    decoder = {"path": "decoder", "sha256": "decoder"}
    stats = {"path": "stats", "sha256": "stats"}
    bank_request = save("bank_request.json", {"identity": {
        "normalization_stats": stats, "decoder_weights_instantiated_but_not_used": decoder}})
    bank = save("bank_summary.json", {"complete": True, "request": bank_request})
    source_hashes = {"raev2_observable_potential.py": "potential-source"}
    training = {"mode": "train", "protocol": sampler.TRAINING_PROTOCOL, "updates": sampler.UPDATES,
                "config": config, "baseline_checkpoint": baseline, "bank": bank,
                "sources": {name: {"sha256": digest} for name, digest in source_hashes.items()}}
    train_request = save("request.json", training)
    checkpoint_identity = {"path": "final.pt", "sha256": "final"}
    save("summary.json", {"complete": True, "mode": "train", "updates": sampler.UPDATES,
                          "checkpoint": checkpoint_identity, "request": train_request})
    checkpoint = {"protocol": sampler.TRAINING_PROTOCOL, "updates": sampler.UPDATES,
                  "request": train_request["path"], "request_sha256": train_request["sha256"]}
    arguments = {"config_identity": config, "baseline_identity": baseline,
                 "source_hashes": source_hashes, "decoder_identity": decoder, "stats_identity": stats}
    return checkpoint, checkpoint_identity, arguments


def test_training_chain_rejects_different_sources_stats_decoder_or_unfinished_fit(tmp_path):
    checkpoint, identity, arguments = training_fixture(tmp_path)
    assert sampler.verify_training_chain(checkpoint, identity, **arguments)["training_request"]["sha256"] == checkpoint["request_sha256"]
    for key in ("config_identity", "baseline_identity", "decoder_identity", "stats_identity"):
        bad = dict(arguments)
        bad[key] = {"sha256": "different"}
        with pytest.raises(ValueError):
            sampler.verify_training_chain(checkpoint, identity, **bad)
    with pytest.raises(ValueError, match="source changed"):
        sampler.verify_training_chain(checkpoint, identity, **{**arguments, "source_hashes": {}})
    summary = tmp_path / "summary.json"
    content = json.loads(summary.read_text())
    content["complete"] = False
    summary.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="completed training"):
        sampler.verify_training_chain(checkpoint, identity, **arguments)


def test_complete_rollout_benchmark_counts_reset_order_and_no_image_sampling(tmp_path, monkeypatch):
    calls = []
    start_values = []

    def step(model, potential, state, labels, current, following, *, use_potential):
        if current == 1:
            start_values.append(state.clone())
        calls.append(use_potential)
        return state + (2 if use_potential else 1)

    monkeypatch.setattr(sampler, "sampling_step", step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    noise = torch.arange(1000, dtype=torch.float32).reshape(1000, 1, 1, 1)
    grid = torch.linspace(1, 0, 101).tolist()
    result = sampler.benchmark(None, None, noise, grid, tmp_path, torch.device("cpu"), count=1000)
    assert len(calls) == result["stage2_forward_calls"] == 800
    assert sum(calls) == result["potential_forward_input_gradient_calls"] == 400
    assert result["stage2_sample_forwards"] == 6400
    assert result["decoder_forward_calls"] == 0 and result["image_sampling_performed"] is False
    assert all(torch.equal(value, noise[:8]) for value in start_values) and len(start_values) == 8
    assert [calls[k] for k in range(0, 800, 100)] == [False, True, True, False, False, True, True, False]
    assert result["official_mean_seconds_per_100_steps_batch8"] > 0
    assert result["potential_mean_seconds_per_100_steps_batch8"] > 0
    assert not (tmp_path / "samples.npz").exists()
    for count, cohort in ((5000, noise), (1000, noise[:8])):
        with pytest.raises(ValueError, match="complete 1000"):
            sampler.benchmark(None, None, cohort, grid, tmp_path, torch.device("cpu"), count=count)

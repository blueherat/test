import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.sample_raev2_accept_d_guidance import (
    BATCH, ClassProposalQueue, assigned_class_ids, draw_acceptance, parse_args,
    score_decoded_images, validate_probe_admission,
)
from experiments.extract_raev2_posterior_features import unit_l2_cls
from experiments.raev2_posterior_probe import probe_probabilities


def test_four_rank_partition_preserves_every_official_initial_batch():
    shards = [assigned_class_ids(rank, 4) for rank in range(4)]
    assert [len(ids) for ids in shards] == [256, 248, 248, 248]
    assert sorted(sum(shards, [])) == list(range(1000))
    for rank, ids in enumerate(shards):
        queue = ClassProposalQueue(ids)
        for start in range(0, len(ids), BATCH):
            batch = queue.next_batch()
            assert batch == ids[start:start+BATCH]
            assert all(queue.attempts[index] == 1 for index in batch)
            assert batch[0]//BATCH % 4 == rank
            queue.finish_batch(batch, [False]*BATCH)
        assert list(queue.pending) == ids


def test_rejection_never_forces_acceptance_and_dummy_is_discarded():
    queue = ClassProposalQueue([9])
    for _ in range(1001):
        batch = queue.next_batch()
        assert batch == [9]+[-1]*7
        queue.finish_batch(batch, [False]+[True]*7)
        assert not queue.done and not queue.accepted
    assert queue.attempts[9] == 1001
    batch = queue.next_batch()
    queue.finish_batch(batch, [True]*8)
    assert queue.done and queue.accepted == {9}


def test_fifo_accepts_once_and_requeues_only_rejected_classes():
    queue = ClassProposalQueue(range(16))
    first = queue.next_batch()
    queue.finish_batch(first, [True, False]*4)
    assert queue.next_batch() == list(range(8, 16))
    queue.finish_batch(list(range(8, 16)), [True]*8)
    last = queue.next_batch()
    assert last == [1, 3, 5, 7, -1, -1, -1, -1]
    queue.finish_batch(last, [True]*8)
    assert queue.done and queue.accepted == set(range(16))
    assert all(queue.attempts[index] == (2 if index in [1, 3, 5, 7] else 1) for index in range(16))


def test_raw_probability_threshold_and_continuous_valid_only_uniform_stream():
    seed = 9107
    generator = torch.Generator().manual_seed(seed)
    expected = torch.rand(7, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).numpy()
    probability = np.array([0., 1., .2, .8, .1, .7, .5])
    first_u, first_a = draw_acceptance(probability[:4], generator)
    last_u, last_a = draw_acceptance(probability[4:], generator)
    np.testing.assert_array_equal(np.r_[first_u, last_u], expected)
    np.testing.assert_array_equal(np.r_[first_a, last_a], expected < probability)
    assert not first_a[0] and first_a[1]
    boundary_generator = torch.Generator().manual_seed(seed)
    _, at_boundary = draw_acceptance(expected, boundary_generator)
    assert not at_boundary.any()  # strict uniform<D, never <=


def test_discrete_rejection_samples_q_times_d_without_normalizer_or_temperature():
    proposal_generator = np.random.default_rng(482)
    proposal = proposal_generator.choice(2, size=250000, p=[.7, .3])
    probability = np.array([.25, .75])[proposal]
    _, accepted = draw_acceptance(probability, torch.Generator().manual_seed(981))
    observed = np.bincount(proposal[accepted], minlength=2)/accepted.sum()
    target = np.array([.7, .3])*np.array([.25, .75])
    target /= target.sum()
    np.testing.assert_allclose(observed, target, rtol=0, atol=.006)
    assert abs(accepted.mean()-.4) < .006


def test_score_uses_genuine_cls_and_exact_frozen_cpu_probability():
    class Backbone:
        def forward_features(self, inputs):
            return {"x_norm_clstoken": torch.cat([inputs.mean((2, 3)), torch.ones(1, 1)], dim=1)}

    class Encoder:
        model = Backbone()

        @staticmethod
        def preprocess(inputs):
            return inputs/255

        def forward_features(self, inputs):
            raise AssertionError("wrapper pooled patch CLS must not be used")

    pixels = np.array([[[[0, 64, 255]]], [[[32, 128, 0]]]], dtype=np.uint8)
    fit = {"weight": np.array([.1, -.2, .3, .4]), "bias": -.15, "train_m": .999}
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        observed, hashes = score_decoded_images(Encoder(), pixels, fit, torch.device("cpu"))
        raw = torch.stack([torch.cat([torch.from_numpy(image.copy()).float().mean((0, 1))/255, torch.ones(1)]) for image in pixels])
        expected = probe_probabilities(unit_l2_cls(raw).numpy(), fit)
        for key in ("logits", "log_D", "D"):
            np.testing.assert_array_equal(observed[key], expected[key])
        assert all(set(item) == {"raw_cls_fp32_sha256", "unit_cls_fp64_sha256"} for item in hashes)
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_flags


def test_admission_requires_matching_independent_full_target_certificate(tmp_path):
    probe = tmp_path/"probe.pt"
    probe.write_bytes(b"frozen probe")
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    frozen = {"probe_sha256": digest, "frozen_before_loading_test_features": True}
    summary = {"probe": {"probe_sha256": digest}, "complete": True, "formal_pass": True,
               "empirical_bernstein_upper": -.1}
    for name, payload in (("model_freeze.json", frozen), ("summary.json", summary)):
        (tmp_path/name).write_text(json.dumps(payload))
    review = {"probe_sha256": digest, "complete": True, "formal_pass": True,
              "empirical_bernstein_full_target_upper": -.05}
    for key, path in (("probe_artifact", probe), ("model_freeze_artifact", tmp_path/"model_freeze.json"),
                      ("audit_summary_artifact", tmp_path/"summary.json")):
        review[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    review_path = tmp_path/"independent_review.json"
    review_path.write_text(json.dumps(review))
    assert validate_probe_admission(probe, review_path)["independent_review_contents"] == review
    review["empirical_bernstein_full_target_upper"] = .01
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="full-target"):
        validate_probe_admission(probe, review_path)
    review["empirical_bernstein_full_target_upper"] = -.05
    review["probe_artifact"]["sha256"] = "bad"
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_probe_admission(probe, review_path)


def test_independent_replication_seed_is_supported_without_method_controls():
    args = parse_args(["--rank", "1", "--output-dir", "/tmp/output", "--noise-bank", "/tmp/noise.npy",
                       "--noise-manifest", "/tmp/manifest.json", "--seed", "202609069"])
    assert args.seed == 202609069 and args.world_size == 4
    assert not any(key in vars(args) for key in ("temperature", "attempt_cap", "coefficient", "time_window"))

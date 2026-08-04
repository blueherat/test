import numpy as np

from experiments.evaluate_sit_ig_interval_ablation import (
    atomic_npy,
    load_statistics,
    reference_statistics,
)


class FIDStatistics:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


class FakeADM:
    FIDStatistics = FIDStatistics


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def read_activations(self, path):
        del path
        self.calls += 1
        values = np.ones((4, 2), dtype=np.float64)
        return values, values

    def read_statistics(self, path, activations):
        del path, activations
        return (
            FIDStatistics(np.array([1.0, 2.0]), np.eye(2)),
            FIDStatistics(np.array([3.0]), np.eye(1)),
        )


def test_reference_statistics_are_cached_and_reloadable(tmp_path):
    evaluator = FakeEvaluator()
    cache = tmp_path / "stats.npz"
    reference = tmp_path / "images.npz"
    np.savez(reference, arr_0=np.zeros((1, 2, 2, 3), dtype=np.uint8))
    first = reference_statistics(
        evaluator,
        reference=reference,
        cache=cache,
        adm_evaluator=FakeADM,
    )
    assert evaluator.calls == 1
    assert cache.is_file()
    second = load_statistics(cache, FakeADM)
    np.testing.assert_allclose(first[0].mu, second[0].mu)
    np.testing.assert_allclose(first[1].sigma, second[1].sigma)


def test_embedded_reference_statistics_skip_image_feature_extraction(tmp_path):
    evaluator = FakeEvaluator()
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        mu=np.array([1.0, 2.0]),
        sigma=np.eye(2),
        mu_s=np.array([3.0]),
        sigma_s=np.eye(1),
    )
    stats = reference_statistics(
        evaluator,
        reference=reference,
        cache=tmp_path / "stats.npz",
        adm_evaluator=FakeADM,
    )
    assert evaluator.calls == 0
    np.testing.assert_allclose(stats[0].mu, [1.0, 2.0])


def test_atomic_npy_round_trip(tmp_path):
    path = tmp_path / "features.npy"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    atomic_npy(path, expected)
    np.testing.assert_array_equal(np.load(path), expected)

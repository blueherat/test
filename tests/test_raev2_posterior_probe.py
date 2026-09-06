import numpy as np
import pytest
from scipy.special import expit

from experiments.raev2_posterior_probe import (
    certificate_observations, fit_posterior_probe, map_objective_and_gradient,
    summarize_certificate, validate_features,
)


def test_map_objective_gradient_and_fixed_prior():
    features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    labels = np.array([1., 1., 0., 0.])
    point = np.array([.2, -.3, .4])
    value, gradient = map_objective_and_gradient(point, features, labels)
    epsilon = 1e-6
    numerical = np.array([(map_objective_and_gradient(point+epsilon*np.eye(3)[i], features, labels)[0]
                           -map_objective_and_gradient(point-epsilon*np.eye(3)[i], features, labels)[0])/(2*epsilon)
                          for i in range(3)])
    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-7)
    logits = features @ point[:2]+point[2]
    expected = -(labels*np.log(expit(logits))+(1-labels)*np.log(expit(-logits))).sum()+.5*np.dot(point[:2],point[:2])
    assert abs(value-expected) < 1e-12


def test_identical_domains_fit_constant_half_and_zero_certificate():
    features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    fit = fit_posterior_probe(features, features)
    assert np.array_equal(fit["weight"], np.zeros(2))
    assert fit["bias"] == 0 and fit["train_m"] == .5
    observations = certificate_observations(features, features, fit)
    assert np.array_equal(observations["certificate"], np.zeros(4))
    summary = summarize_certificate(observations)
    assert summary["C_mean"] == 0 and not summary["formal_pass"]
    assert summary["empirical_bernstein_upper"] == 0


def test_known_tilt_certificate_bounds_actual_forward_kl_change():
    features = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    p, q = np.array([.6, .3, .1]), np.array([.2, .3, .5])
    fit = {"weight": np.array([.7, .2]), "bias": -.1, "train_m": .47}
    observations = certificate_observations(features, features, fit)
    d = observations["real"]["D"]
    accepted = q*d/np.dot(q,d)
    actual_kl_change = np.dot(p,np.log(p/accepted))-np.dot(p,np.log(p/q))
    certificate = np.dot(q,d/fit["train_m"]-1)-np.dot(p,np.log(d/fit["train_m"]))
    assert actual_kl_change <= certificate+1e-14
    assert certificate < 0
    bounds = observations["bounds"]
    assert np.all(observations["certificate"] >= bounds["certificate_min"])
    assert np.all(observations["certificate"] <= bounds["certificate_max"])


def test_feature_normalization_is_not_silently_changed_by_probe():
    with pytest.raises(ValueError, match="unit-L2"):
        validate_features(np.array([[2., 0.]]))


def test_empirical_bernstein_uses_global_range_and_preselected_bound():
    observations = {"certificate": np.array([-.3, -.2, -.1, -.2]),
                    "bounds": {"certificate_min": -3., "certificate_max": 4.}}
    summary = summarize_certificate(observations)
    x = observations["certificate"]
    expected = x.mean()+np.sqrt(2*x.var(ddof=1)*np.log(40)/4)+7*7*np.log(40)/9
    assert abs(summary["empirical_bernstein_upper"]-expected) < 1e-12
    assert not summary["formal_pass"]

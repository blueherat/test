"""Exact finite-step counterexample and rank-one Gaussian covariance checks."""
import unittest
import numpy as np
import torch
from experiments.raev2_directional_variance import colored_noise


class DirectionalVarianceTests(unittest.TestCase):
    def test_compensating_drift_breaks_exact_bridge_moments(self):
        variance, t, s = 4., .6, .5
        a, q = 1-t, (t-s)/t
        current = a*a*variance+t*t
        posterior_slope = a*variance/current
        posterior_variance = variance*t*t/current
        mean_slope = s/t+q*posterior_slope
        noise_variance = q*q*posterior_variance
        target = (1-s)**2*variance+s*s
        self.assertAlmostEqual(mean_slope**2*current+noise_variance, target, places=14)
        compensated = (mean_slope-.5*noise_variance/current)**2*current+noise_variance
        self.assertAlmostEqual(compensated, 1.2064, places=14)
        self.assertGreater(abs(compensated-target), .04)

    def test_colored_noise_has_rank_one_covariance(self):
        dimension, kappa = 4, 7.3
        points = torch.cat([torch.eye(dimension)*2, -torch.eye(dimension)*2]).reshape(8, 4, 1, 1)
        direction = torch.tensor([1., 2., -3., 4.]).reshape(1, 4, 1, 1).expand_as(points)
        actual, active = colored_noise(points, direction, torch.zeros_like(direction), kappa)
        self.assertTrue(active.all())
        matrix = actual.flatten(1).double().numpy()
        u = np.array([1., 2., -3., 4.])/np.sqrt(30)
        expected = np.eye(dimension)+(kappa-1)*np.outer(u, u)
        np.testing.assert_allclose(matrix.T@matrix/8, expected, rtol=2e-7, atol=2e-7)
        self.assertGreater(np.linalg.eigvalsh(expected).min(), 0)
        unit, _ = colored_noise(points, direction, torch.zeros_like(direction), 1.)
        self.assertTrue(torch.equal(points, unit))
        zero, active = colored_noise(points, direction, direction, kappa)
        self.assertFalse(active.any())
        self.assertTrue(torch.equal(points, zero))

    def test_gaussian_likelihood_ratio_and_sample_mean_optimum(self):
        dimension, kappa = 3, 4.2
        u = np.array([1., 2., 3.])/np.sqrt(14)
        errors = np.array([[1., 2., 0.], [2., -1., 3.], [3., 4., 1.]])
        m = np.array([.8, 1.1, .5])
        beta = (errors@u)**2/m
        shape = np.eye(dimension)+(kappa-1)*np.outer(u, u)
        direct = np.array([.5*(np.linalg.slogdet(shape)[1]+e@(np.linalg.inv(shape)-np.eye(dimension))@e/v) for e, v in zip(errors, m)])
        expected = .5*(np.log(kappa)+beta*(1/kappa-1))
        np.testing.assert_allclose(direct, expected, rtol=1e-14, atol=1e-14)
        optimum = beta.mean()
        risk = lambda h: .5*(h+beta.mean()*np.expm1(-h))
        h = np.log(optimum)
        self.assertLess(risk(h), risk(h+.1))
        self.assertLess(risk(h), risk(h-.1))


if __name__ == '__main__':
    unittest.main()

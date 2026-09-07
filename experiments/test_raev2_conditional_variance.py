"""Check fixed-mean variance optimum, analytic derivatives and zero-head parity."""
import math
import unittest
import numpy as np
from scipy.optimize import minimize
import torch
from experiments.raev2_conditional_variance import ConditionalVarianceRisk, ConditionalVarianceHead, PLAN, query_indices


class ConditionalVarianceTests(unittest.TestCase):
    def test_recovers_conditional_residual_means_with_imperfect_mean(self):
        # Two equally frequent states. Squared errors include nonzero mean bias;
        # the optimum must recover MSE (2 and 8), not centered variance (1 and 4).
        error = np.array([0., 2., 0., 4.])
        x = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        target = error**2 + 1e-12
        m0 = target.mean()
        risk = ConditionalVarianceRisk(x, target/m0, 0.)
        result = minimize(risk, np.zeros(2), jac=True, method='BFGS', options={'gtol': 1e-11})
        expected = np.array([2., 8.])+1e-12
        np.testing.assert_allclose(m0*np.exp(result.x), expected, rtol=1e-8)
        expected_gain = .5*(np.log(expected).mean()-np.log(m0))
        self.assertAlmostEqual(risk.losses(result.x).mean(), expected_gain, places=12)
        self.assertLess(expected_gain, 0)

    def test_gradient_direction_and_strict_convexity(self):
        rng = np.random.default_rng(203)
        x = rng.normal(size=(31, 7))
        y = np.exp(rng.normal(size=31))
        theta = rng.normal(size=7)*.1
        direction = rng.normal(size=7)
        risk = ConditionalVarianceRisk(x, y, .05)
        _, gradient = risk(theta)
        step = 1e-5
        fd = (risk(theta+step*direction)[0]-risk(theta-step*direction)[0])/(2*step)
        self.assertAlmostEqual(float(gradient@direction), fd, places=8)
        curvature_fd = (risk(theta+step*direction)[1]-risk(theta-step*direction)[1])/(2*step)
        expected = .5*x.T@(y*np.exp(-x@theta)*(x@direction))/len(x)+.05*direction
        np.testing.assert_allclose(curvature_fd, expected, rtol=1e-8, atol=1e-9)
        self.assertGreater(float(direction@curvature_fd), .05*float(direction@direction))

    def test_zero_head_keeps_existing_scalar_noise_arithmetic(self):
        head = ConditionalVarianceHead.__new__(ConditionalVarianceHead)
        torch.nn.Module.__init__(head)
        head.weight = torch.randn(9, dtype=torch.float64)
        head.bias = torch.tensor(.3, dtype=torch.float64)
        head.mean = torch.randn(9, dtype=torch.float64)
        head.scale = torch.tensor(.81, dtype=torch.float64)
        features, noise = torch.randn(8, 9), torch.randn(8, 16, 3, 3)
        for mse0, q in [(1.034567891, .001261), (.000934539, 1.), (.03835, .25)]:
            coefficient, h = head.noise_coefficient(features, mse0, q, zero=True)
            self.assertTrue(torch.equal(h, torch.zeros_like(h)))
            # Existing sampler forms the scalar q*sqrt(mse) before multiplying
            # FP32 noise; the new vector coefficient must preserve this order.
            expected = q*math.sqrt(mse0)*noise
            self.assertTrue(torch.equal(coefficient[:, None, None, None]*noise, expected))

    def test_all_query_times_balanced_and_native_batches_preserved(self):
        for split in ['train', 'validation']:
            indices = query_indices(split)
            counts = np.bincount(indices, minlength=100)
            np.testing.assert_array_equal(counts, PLAN[split]['count']//800)
            self.assertEqual(len(indices)*8, PLAN[split]['count'])


if __name__ == '__main__':
    unittest.main()

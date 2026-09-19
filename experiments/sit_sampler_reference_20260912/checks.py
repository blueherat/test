"""Fixed-noise-level KL identity and its limited scope."""
import numpy as np


def analytic_checks():
    records=[]
    for mean,variance in ((1.,2.),(-.7,.3),(0.,1.)):
        derivative=-mean**2+(1-1/variance)*(1-variance)
        fisher=mean**2+(variance-1)**2/variance
        np.testing.assert_allclose(derivative,-fisher,rtol=1e-14,atol=1e-14)
        records.append(dict(mean=mean,variance=variance,kl_derivative=derivative,fisher=fisher))
    return dict(passed=True,gaussian_identity=records,arbitrary_reference_can_leave_positive_kl_unchanged=True,
        note='Fixed-noise exact-density transport only; not a generated-FID guarantee or CFG conditional-KL theorem.')

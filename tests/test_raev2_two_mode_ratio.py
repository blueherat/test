import numpy as np
from experiments.raev2_two_mode_ratio import posterior_difference


def test_gaussian_denoiser_replacement_including_endpoints():
    for p,q in [(125.,118.),(.51,.55),(.1,10.),(1.,1.)]:
        for t in [1.,.999,.92,.7,.4,.1,.001,0.]:
            a=1-t
            prior_denoiser=a*q/(t*t+a*a*q)
            target_denoiser=a*p/(t*t+a*a*p)
            np.testing.assert_allclose(prior_denoiser+posterior_difference(t,p,q),target_denoiser,atol=1e-14)


def test_dense_covariance_score_difference_to_clean_formula():
    n=4
    dc=np.ones((n,n))/n
    cp=125*dc+.51*(np.eye(n)-dc)
    cq=118*dc+.55*(np.eye(n)-dc)
    t=.83;a=1-t;x=np.array([1.,-.5,.3,.2])
    delta_score=(np.linalg.inv(a*a*cq+t*t*np.eye(n))-np.linalg.inv(a*a*cp+t*t*np.eye(n)))@x
    expected=t*t/a*delta_score
    actual=posterior_difference(t,125,118)*(dc@x)+posterior_difference(t,.51,.55)*((np.eye(n)-dc)@x)
    np.testing.assert_allclose(actual,expected,atol=1e-14)

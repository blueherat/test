import numpy as np
from experiments.raev2_two_mode_ratio import posterior_difference,fit_euler_prior_variance,euler_terminal_variance


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


def test_guided_finite_sampler_exactly_matches_target_second_moment():
    u=np.linspace(1.,0.,101)
    grid=8*u/(1+7*u)
    for p,q in [(124.,116.),(.518,.549),(.001,100.)]:
        pp=fit_euler_prior_variance(p,grid)
        qq=fit_euler_prior_variance(q,grid)
        np.testing.assert_allclose(euler_terminal_variance(qq,grid),q,rtol=1e-11)
        variance=1.
        for t,s in zip(grid[:-1],grid[1:]):
            a=1-t
            native=a*qq/(t*t+a*a*qq)
            guided=native+posterior_difference(t,pp,qq)
            variance*=(1-(t-s)*(1-guided)/t)**2
        np.testing.assert_allclose(variance,p,rtol=1e-11)

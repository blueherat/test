"""Independent distribution-level checks for fixed-mean reverse variance."""
import numpy as np


def test_exact_gaussian_marginals_on_full_shifted_grid():
    u=np.linspace(1.,0.,101)
    grid=8*u/(1+7*u)
    for target in [.01,.1,1.,10.]:
        actual=1.
        for t,s in zip(grid[:-1],grid[1:]):
            vt=(1-t)**2*target+t*t
            gain=(1-t)*target/vt
            posterior_mse=target*t*t/vt
            r=s/t
            q=(t-s)/t
            actual=(r+q*gain)**2*actual+q*q*posterior_mse
            np.testing.assert_allclose(actual,(1-s)**2*target+s*s,rtol=1e-12,atol=1e-14)


def test_imperfect_mean_variance_is_unique_nll_minimizer():
    # X~N(0,2), Z=.4X+.6eps; fixed predictor G=.9Z+.2 is imperfect.
    a,t,s,var=.4,.6,.5,2.
    vz=a*a*var+t*t
    gain=a*var/vz
    posterior=var*t*t/vz
    estimated_mse=posterior+(.9-gain)**2*vz+.2**2
    direct_mse=var-2*.9*a*var+.9**2*vz+.2**2
    np.testing.assert_allclose(estimated_mse,direct_mse)
    q=(t-s)/t
    opt=q*q*estimated_mse
    def nll(v): return .5*np.log(v)+q*q*direct_mse/(2*v)
    assert all(nll(opt)<nll(opt*factor) for factor in [.1,.5,.9,1.1,2.,10.])

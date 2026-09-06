import numpy as np
import torch
from experiments.raev2_image_critic_guidance import ExchangeablePosterior


def test_covariance_operator_matches_full_gaussian_conditioning():
    rng=np.random.default_rng(829)
    n,c=4,3
    k0=rng.normal(size=(c,c)); k0=k0@k0.T+.2*np.eye(c)
    kr=rng.normal(size=(c,c)); kr=kr@kr.T+.1*np.eye(c)
    p0=np.ones((n,n))/n
    prior=np.kron(p0,k0)+np.kron(np.eye(n)-p0,kr)
    op=ExchangeablePosterior.__new__(ExchangeablePosterior)
    for tag,k in [('0',k0),('r',kr)]:
        e,v=np.linalg.eigh(k)
        setattr(op,'e'+tag,torch.from_numpy(e))
        setattr(op,'v'+tag,torch.from_numpy(v))
    g=rng.normal(size=(1,n,c))
    tensor=torch.from_numpy(g.transpose(0,2,1).reshape(1,c,2,2))
    for t in [1.,.7,.05]:
        posterior=np.linalg.inv(np.linalg.inv(prior)+(1-t)**2/t**2*np.eye(n*c))
        mse=.17
        expected=posterior@g.reshape(-1)*mse/(np.trace(posterior)/(n*c))
        actual=op.apply(tensor,t,mse).flatten(2).transpose(1,2).numpy().reshape(-1)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
        assert float(g.reshape(-1)@actual)>0


def test_linear_reward_gaussian_tilt_normalizer_and_mean():
    # Direct numerical integration, independent of the Gaussian mean-shift code.
    grid=np.linspace(-12,12,200001)
    m,var,reward=.3,.7,.9
    weight=np.exp(-.5*(grid-m)**2/var+reward*grid)
    mean=np.trapezoid(grid*weight,grid)/np.trapezoid(weight,grid)
    np.testing.assert_allclose(mean,m+var*reward,atol=1e-12)

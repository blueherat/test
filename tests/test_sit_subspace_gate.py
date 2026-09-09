import torch
from experiments.cache_sit_subspace_gate import quadratic
from experiments.train_sit_subspace_gate import quadratic_loss


def test_cached_quadratic_matches_dense_velocity_loss_and_gate_gradient():
    torch.manual_seed(83)
    p=torch.linalg.qr(torch.randn(17,4,dtype=torch.float64))[0]
    ex,ee=torch.randn(2,13,17,dtype=torch.float64)
    full=quadratic(ex,ee);top=quadratic(ex@p,ee@p);q=torch.stack([top,full-top],dim=1)
    g=torch.rand(13,2,dtype=torch.float64,requires_grad=True)
    delta=ex-ee;parallel=(delta@p)@p.T
    dense=ee+g[:,:1]*parallel+g[:,1:]*(delta-parallel)
    expected=dense.square().sum(1)
    actual=quadratic_loss(g,q,True)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    dg=torch.autograd.grad(actual.sum(),g,retain_graph=True)[0]
    de=torch.autograd.grad(expected.sum(),g)[0]
    torch.testing.assert_close(dg,de,atol=1e-10,rtol=1e-10)

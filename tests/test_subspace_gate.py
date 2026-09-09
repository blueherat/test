import torch
import pytest
from experiments.train_subspace_gate_pilot import Gate, mix, core
from experiments.subspace_gate_distribution import make_distribution


def test_split_initialization_matches_scalar_for_orthogonal_projector():
    torch.manual_seed(10)
    model=Gate(8)
    p=torch.linalg.qr(torch.randn(8,2))[0]
    z,vx,ve=torch.randn(3,19,8)
    t=torch.linspace(.001,.999,19)
    scalar=mix(model,'scalar',p,z,t,vx,ve)
    split=mix(model,'pca',p,z,t,vx,ve)
    torch.testing.assert_close(scalar,split,atol=1e-6,rtol=1e-6)


def test_independent_orthogonal_oracle_cannot_worsen_pointwise_risk():
    torch.manual_seed(11)
    p=torch.linalg.qr(torch.randn(13,3))[0]
    vx,ve,target=torch.randn(3,101,13)
    xp,ep,tp=vx@p,ve@p,target@p
    xn,en,tn=vx-xp@p.T,ve-ep@p.T,target-tp@p.T
    g=core.analytic_scalar_gate(vx,ve,target,clip=True)
    gp=core.analytic_scalar_gate(xp,ep,tp,clip=True)
    gn=core.analytic_scalar_gate(xn,en,tn,clip=True)
    scalar=ve+g*(vx-ve)
    split=(ep+gp*(xp-ep))@p.T+en+gn*(xn-en)
    assert ((split-target).square().sum(1)<=(scalar-target).square().sum(1)+1e-5).all()


def test_curved_geometry_does_not_silently_use_linear_bayes_oracle():
    cfg=dict(ambient_dim=16,seed=71,data_jitter=.015,quadrature_points=64,
        locator_points=128,frequency_scale=6.,scale_mode='unit_rms',curvature=.5,bayes_batch_chunk=32)
    dist=make_distribution(cfg,torch.device('cpu'))
    x,u,_=dist.sample(31,generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(dist.decode_intrinsic(x),u,atol=2e-6,rtol=2e-6)
    assert dist.off_subspace_rms(x).max()<2e-6
    with pytest.raises(NotImplementedError):
        dist.bayes_velocity(x,torch.full((31,),.5))

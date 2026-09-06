import torch
from experiments.raev2_transport_projection import projected_guidance


def test_velocity_coordinate_matches_optimized_weak_scale():
    rng=torch.Generator().manual_seed(12)
    z,f,b=[torch.randn(2,4,3,generator=rng,dtype=torch.float64) for _ in range(3)]
    t=.7
    vf=(f-z)/t; vb=(b-z)/t
    scale=(vf*vb).sum((1,2),keepdim=True)/vb.square().sum((1,2),keepdim=True)
    expected=z+t*(vf+.78*(vf-scale*vb))
    got=projected_guidance(z,f,b,f+.78*(f-b),t,coordinate='velocity')
    torch.testing.assert_close(got,expected,rtol=2e-6,atol=2e-6)


def test_noise_coordinate_matches_optimized_noise_scale():
    rng=torch.Generator().manual_seed(13)
    z,f,b=[torch.randn(2,4,3,generator=rng,dtype=torch.float64) for _ in range(3)]
    t=.7; a=1-t
    ef=(z-a*f)/t; eb=(z-a*b)/t
    scale=(ef*eb).sum((1,2),keepdim=True)/eb.square().sum((1,2),keepdim=True)
    expected=(z-t*(ef+.78*(ef-scale*eb)))/a
    got=projected_guidance(z,f,b,f+.78*(f-b),t,coordinate='noise')
    torch.testing.assert_close(got,expected,rtol=2e-6,atol=2e-6)


def test_projection_constraint_and_noise_endpoint_are_finite():
    rng=torch.Generator().manual_seed(11)
    z,f,b=[torch.randn(2,4,3,generator=rng) for _ in range(3)]
    for t in [1.,.8,.1]:
        for coord in ['velocity','noise']:
            out=projected_guidance(z,f,b,f+.78*(f-b),t,coordinate=coord)
            ref=b-z if coord=='velocity' else z-(1-t)*b
            residual=(out-f)/.78
            assert torch.isfinite(out).all()
            torch.testing.assert_close((residual*ref).sum((1,2)),torch.zeros(2),atol=2e-5,rtol=0.)

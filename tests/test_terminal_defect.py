"""Analytic adversarial controls for the signed temporal decomposition."""
import numpy as np
from experiments.terminal_defect_spiral import defect_statistics


def test_cancellation_is_not_detected_by_local_squares():
    # Every local error is nonzero, but the endpoint gap vanishes exactly.
    features=np.array([[[0.]],[[3.]],[[0.]]])
    result=defect_statistics(features)
    assert result['terminal_feature_gap_squared']==0
    assert result['diagonal_defect_energy']==18
    assert result['cross_time_energy']==-18
    assert result['cancellation_fraction']==1


def test_partial_local_repair_can_increase_terminal_error():
    old=defect_statistics(np.array([[[0.]],[[3.]],[[0.]]]))
    # d1=-3,d2=3 become d1=-1,d2=3: lower diagonal risk, higher endpoint risk.
    new=defect_statistics(np.array([[[2.]],[[3.]],[[0.]]]))
    assert new['diagonal_defect_energy']<old['diagonal_defect_energy']
    assert new['terminal_feature_gap_squared']>old['terminal_feature_gap_squared']


def test_noncommuting_affine_sampler_telescopes_at_finite_amplitude():
    rng=np.random.default_rng(63)
    noise,clean=rng.normal(size=(2,31,2))
    matrices=[np.array([[1.,2.],[0.,1.]]),np.array([[1.,0.],[-3.,1.]])]
    offsets=[np.array([1.,-2.]),np.array([.3,.7])]
    values=[]
    for k in range(3):
        z=(1-k/2)*noise+(k/2)*clean
        for j in range(k,2):
            z=z@matrices[j].T+offsets[j]
        values.append(np.concatenate([z,np.sin(z),z*z],axis=1))
    r=defect_statistics(np.stack(values))
    assert r['telescoping_max_abs']<1e-12
    assert r['energy_identity_abs']<1e-9


def test_unbiased_gram_equals_distinct_pair_enumeration():
    f=np.random.default_rng(37).normal(size=(4,7,5))
    d=f[:-1]-f[1:]
    expected=sum(d[:,i]@d[:,j].T for i in range(7) for j in range(7) if i!=j)/42
    np.testing.assert_allclose(defect_statistics(f)['unbiased_gram'],expected,atol=1e-14)

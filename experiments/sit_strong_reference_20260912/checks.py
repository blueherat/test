"""Ideal finite-dimensional projection identity, not a model/FID diagnostic."""
import numpy as np


def analytic_checks():
    rng=np.random.default_rng(2026121115)
    q,_=np.linalg.qr(rng.normal(size=(19,5)))
    u=rng.normal(size=(19,4));e=rng.normal(size=(19,4));s=u+e
    project=lambda value:q@(q.T@value)
    native,teacher=project(u),project(s)
    records=[]
    for a in (.8*6/7,.8,1.25):
        gn=s+a*(s-native);gd=s+a*(s-teacher)
        observed=np.sum((gn-u)**2)-np.sum((gd-u)**2)
        expected=a*(a+2)*np.sum(project(e)**2)
        np.testing.assert_allclose(observed,expected,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(project(gd-u),project(e),rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(project(gn-u),(1+a)*project(e),rtol=1e-13,atol=1e-13)
        records.append(dict(amount=a,risk_difference=float(observed),identity=float(expected)))
    # Time-zero-to-one linear interpolation, scalar learned S(z,t)=k*z.
    # Its generated endpoint is N(0,exp(2*k)); re-noised endpoints have a
    # different exact FM velocity. This does not rely on nonconservative fields.
    k=.3;v=np.exp(2*k);t=.4
    fitted=(t*v-(1-t))/(t*t*v+(1-t)**2)
    assert abs(fitted-k)>.01
    return dict(passed=True,projection_risk_identity=records,
        exact_teacher_with_unrestricted_capacity_cancels_guidance=True,
        field_vs_generated_example=dict(strong_coefficient=k,endpoint_variance=float(v),time=t,
            generated_data_velocity_coefficient=float(fitted)),
        note='Ideal linear population projection, fixed state measure and time; actual shared nonlinear head need not attain it. No quality guarantee.')

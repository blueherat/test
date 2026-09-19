import numpy as np


def action(points,weights):
    p=np.asarray(weights,dtype=float);p=p/p.sum();x=np.asarray(points,dtype=float)
    mean=p@x;variance=p@(x-mean)**2;third=p@(x-mean)**3
    delta=0. if variance==0 else 2*np.sqrt(variance)*np.sinh(np.arcsinh(third/(2*variance**1.5))/3)
    return mean+delta


def analytic_checks():
    errors=[]
    for probability in (.001,.01,.1,.25,.5,.8,.99,.999):
        points=np.array([-2.,3.]);weights=np.array([1-probability,probability])
        tilted=weights**(1/3);expected=tilted@points/tilted.sum()
        actual=action(points,weights);errors.append(abs(actual-expected))
        assert abs(weights@(actual-points)**3)<1e-10
    assert max(errors)<1e-11
    assert abs(action([-3,-1,1,3],[.1,.4,.4,.1]))<1e-14
    x=np.array([-3.,0.,1.]);p=np.array([.1,.6,.3]);tilted=p**(1/3)
    difference=abs(action(x,p)-tilted@x/tilted.sum());assert difference>.01
    return dict(passed=True,two_point_odds_cube_root_max_error=max(errors),
        symmetric_multimodal_zero=True,three_point_tempered_mean_counterexample_error=difference,
        no_universal_tempered_density_claim=True)

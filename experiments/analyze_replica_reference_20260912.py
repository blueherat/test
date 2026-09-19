"""Exact finite-prior calculations and reusable theory data, no image tuning."""
from pathlib import Path
import csv
import json
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import softmax
from experiments.sit_replica_reference_20260912.checks import analytic_checks

OUT = Path('docs/data/replica_reference_20260912')


def transition(points, weights, sigma, order):
    nodes, masses = hermgauss(order)
    dim = points.shape[1]
    if dim == 1:
        noise = nodes[:, None]*np.sqrt(2); w = masses/np.sqrt(np.pi)
    else:
        xx, yy = np.meshgrid(nodes, nodes, indexing='ij')
        noise = np.stack([xx.ravel(), yy.ravel()], 1)*np.sqrt(2)
        w = (masses[:, None]*masses[None, :]).ravel()/np.pi
    rows = []
    for point in points:
        y = point+sigma*noise
        logits = np.log(weights)-((y[:, None, :]-points[None, :, :])**2).sum(-1)/(2*sigma**2)
        rows.append(w@softmax(logits, axis=-1))
    return np.stack(rows)


def estimates(points, weights, sigma, order, y):
    P = transition(points, weights, sigma, order)
    B = np.linalg.solve(2*np.eye(len(points))-P, points)
    A = np.linalg.solve(np.eye(len(points))+P, 2*points)
    probs = softmax(np.log(weights)-((y[:, None, :]-points[None, :, :])**2).sum(-1)/(2*sigma**2), axis=-1)
    return dict(P=P, B=B, A=A, probs=probs, mse=probs@points,
                consistent=probs@B, average=probs@A)


def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check = analytic_checks()
    points = np.array([[-1.], [1.]])
    weights = np.array([.5, .5])
    rows=[]
    for sigma in (.35, .8, 1.6):
        y=np.linspace(-2.5, 2.5, 401)[:, None]
        v=estimates(points, weights, sigma, 160, y)
        eta=float(v['P'][0, 0]-v['P'][0, 1])
        assert np.max(abs(v['consistent']-v['mse']/(2-eta)))<1e-10
        assert np.max(abs(v['average']-2*v['mse']/(1+eta)))<1e-10
        for i in range(len(y)):
            rows.append(dict(sigma=sigma,y=float(y[i,0]),eta=eta,
                mse=float(v['mse'][i,0]),consistent=float(v['consistent'][i,0]),
                average=float(v['average'][i,0]),
                consistent_guided=float(1.8*v['mse'][i,0]-.8*v['consistent'][i,0]),
                average_guided=float(1.8*v['mse'][i,0]-.8*v['average'][i,0])))
    write_csv('two_point_curves.csv', rows)
    # A converged, genuinely Gaussian corruption counterexample to score integrability.
    points=np.array([[-1.3,-.2],[.4,1.6],[1.1,-.9]])
    weights=np.array([.2,.5,.3]);y=np.array([[.25,.15]]);sigma=.9
    curls=[]
    for order in (60, 100):
        v=estimates(points,weights,sigma,order,y);p=v['probs'][0]
        entry={'quadrature_order':order}
        for name,key in (('consistent','B'),('average','A')):
            covariance=(v[key].T*p)@points-np.outer(p@v[key],p@points)
            jacobian=covariance/sigma**2
            entry[name+'_score_curl']=float((jacobian[1,0]-jacobian[0,1])/sigma**2)
        entry['detailed_balance_error']=float(np.max(abs(weights[:,None]*v['P']-(weights[:,None]*v['P']).T)))
        curls.append(entry)
    assert abs(curls[-1]['consistent_score_curl'])>1e-4
    assert max(abs(curls[0][k]-curls[1][k]) for k in ('consistent_score_curl','average_score_curl'))<1e-6
    check['gaussian_corruption_curl_counterexample']=curls
    check['two_point_guidance_support_overshoot']=max(r['consistent_guided'] for r in rows)-1
    check['interpretation']='Risk identities are exact. Function regularization need not yield a conservative score or any fixed smoothed prior.'
    (OUT/'theory_checks.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()

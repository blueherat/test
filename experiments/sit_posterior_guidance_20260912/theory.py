"""Small analytic checks of the hypothesis, independent of image quality."""
import json
import numpy as np
from .catalog import ROOT


def softmax(x):
    p=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return p/p.sum(axis=-1,keepdims=True)


def run():
    rng=np.random.default_rng(2026120940)
    logits=rng.normal(size=(32,7));negative=rng.normal(size=(32,7))
    p,q=softmax(logits),softmax(negative)
    guided=softmax(2*logits-negative)
    expected=p*p/q;expected/=expected.sum(-1,keepdims=True)
    np.testing.assert_allclose(guided,expected,atol=1e-14)
    values=rng.normal(size=(32,7,3))
    means=np.einsum('bi,bij->bj',guided,values)
    assert np.all(means>=values.min(1)-1e-14) and np.all(means<=values.max(1)+1e-14)
    p2=np.array([.25,.75]);q2=np.array([.5,.5]);v=np.array([-1.,1.])
    alpha=2.
    linear=float(((1+alpha)*p2-alpha*q2)@v)
    probs=softmax((1+alpha)*np.log(p2)-alpha*np.log(q2))
    nonlinear=float(probs@v)
    assert linear>1 and -1<nonlinear<1
    # The local modulation construction has no general equivalence to full CFG.
    # Even in one attention row the initial derivative differs from p-q.
    a=logits-negative
    tangent=p*(a-(p*a).sum(-1,keepdims=True))
    eps=1e-5
    numerical=(softmax(logits+eps*a)-softmax(logits-eps*a))/(2*eps)
    np.testing.assert_allclose(tangent,numerical,atol=1e-9)
    assert np.linalg.norm(tangent-(p-q))>.1
    result=dict(passed=True,probability_ratio_exact=True,row_sum_error=float(abs(guided.sum(-1)-1).max()),
        finite_dictionary_convex_combination=True,two_mode_linear_mean=linear,two_mode_odds_mean=nonlinear,
        analytic_derivative_checked=True,not_equivalent_to_linear_attention=True,
        limitations=['Attention values are not certified clean-image modes.',
            'A block-local null condition is not the network unconditional posterior.',
            'Convexity inside attention does not guarantee bounded final images or improve FID.'])
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'theory_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__=='__main__':
    run()

"""Exact rational checks for the posterior-composition argument; no model calls."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path


def normalized(values):
    total=sum(values)
    return [v/total for v in values]


def combine(strong,weak,alpha=1):
    return normalized([s**(1+alpha)/w**alpha for s,w in zip(strong,weak)])


def mean(weights,support):
    return sum(p*x for p,x in zip(weights,support))


def moments(weights,support):
    m=mean(weights,support)
    return m,mean(weights,[x*x for x in support])-m*m


def main():
    target=[F(1,2),F(1,3),F(1,6)]
    factors=[F(2),F(3),F(5)]
    likelihoods=([F(1)]*3,[F(2),F(3),F(4)],
                 [F(1,7),F(2,11),F(3,5)],[F(8),F(1,13),F(1,17)])
    cancellation=[]
    for eta_s,eta_w in ((1,2),(2,3)):
        alpha=eta_s//(eta_w-eta_s)
        prior_s=normalized([p/c**eta_s for p,c in zip(target,factors)])
        prior_w=normalized([p/c**eta_w for p,c in zip(target,factors)])
        guided=combine(prior_s,prior_w,alpha)
        assert guided==target
        for likelihood in likelihoods:
            rs=normalized([p*l for p,l in zip(prior_s,likelihood)])
            rw=normalized([p*l for p,l in zip(prior_w,likelihood)])
            rg=combine(rs,rw,alpha)
            expected=normalized([p*l for p,l in zip(target,likelihood)])
            assert rg==expected
            ratios=[g*w**alpha/s**(1+alpha) for g,s,w in zip(rg,rs,rw)]
            assert len(set(ratios))==1
            cancellation.append(dict(eta_s=eta_s,eta_w=eta_w,alpha=alpha,
                likelihood=likelihood,posterior=rg,exact=True))
    rw=[F(1,3)]*3
    rs=[F(1,2),F(1,3),F(1,6)]
    rg=combine(rs,rw)
    xs,ys=[F(0),F(1),F(0)],[F(0),F(0),F(1)]
    ms=(mean(rs,xs),mean(rs,ys));mw=(mean(rw,xs),mean(rw,ys));mg=(mean(rg,xs),mean(rg,ys))
    determinant=(ms[0]-mw[0])*(mg[1]-ms[1])-(ms[1]-mw[1])*(mg[0]-ms[0])
    assert rg==[F(9,14),F(2,7),F(1,14)] and determinant==F(-1,126)
    support=list(map(F,(-2,-1,0,1,2)))
    weak=[F(1,5)]*5
    strong_a=list(map(F,('0.10','0.15','0.20','0.25','0.30')))
    strong_b=list(map(F,('0.13','0.03','0.38','0.13','0.33')))
    assert moments(strong_a,support)==moments(strong_b,support)==(F(1,2),F(7,4))
    ga=mean(combine(strong_a,weak),support)
    gb=mean(combine(strong_b,weak),support)
    assert ga==F(8,9) and gb==F(25,36) and ga!=gb
    result=dict(passed=True,arithmetic='exact fractions',model_calls=0,generated_images=0,
        shared_bias_and_common_likelihood=cancellation,
        off_affine_line=dict(strong_mean=ms,weak_mean=mw,guided_mean=mg,determinant=determinant),
        insufficient_first_two_moments=dict(support=support,weak=weak,strong_a=strong_a,strong_b=strong_b,
            common_strong_moments=moments(strong_a,support),weak_moments=moments(weak,support),
            guided_mean_a=ga,guided_mean_b=gb),research_goal_achieved=False)
    out=Path(__file__).resolve().parents[1]/'docs/data/ig_hypothesis_posterior_20260910'
    out.mkdir(parents=True,exist_ok=True)
    data=json.dumps(result,default=str,indent=2)+'\n'
    (out/'exact_checks.json').write_text(data)
    print(data)


if __name__=='__main__':main()

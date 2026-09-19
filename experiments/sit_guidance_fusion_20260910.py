"""Explicit compositions of selected guidance mechanisms and matched controls.

Sampling uses the frozen portfolio's Heun/history/RNG implementation. Only the
field dispatcher is extended; original operators are delegated unchanged.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments import small_sit_guidance_tuning_20260910 as angular

ORIGINAL_EVALUATE=old.evaluate
IG_STRENGTHS=(.6,.7,.8,.9)
CFG_APG_STRENGTHS=(1.5,2.,2.5,3.)
CFG_DIRECTION_STRENGTHS=(1.,1.25,1.5,1.75)
BETAS=(-.75,-.5,-.25)
MIXES=(.25,.5,.75)
SECANTS=(.25,.5,.75)
FAMILIES={
    'ig_local_residual':dict(source='ig',title='局部弱参考＋可预测残差',
        fixed=dict(locality=2.),parameter='residual',values=(.25,.5,1.),strengths=IG_STRENGTHS,
        rationale='保留局部化带来的上下文差，再移除原始gap的浅层可预测分量；检验两类修正能否互补。',
        equation='u=N_D(S-W_local-r*C_original)',
        limitation='C只拟合原始gap，不能视作局部化gap的正确回归；交互可能破坏已确认方向。'),
    'ig_local_metric':dict(source='ig',title='局部弱参考＋通道椭球预算',
        fixed=dict(locality=2.),parameter='channel_ridge',values=(.1,.5,2.),strengths=IG_STRENGTHS,
        rationale='先形成跨空间上下文方向，再限制通道度量下的局部离群增量；方向来源与幅度约束分别控制。',
        equation='u=channel_metric(N_D(S-W_local),m,r)',
        limitation='通道空间协方差不是真实误差协方差，预算可能压掉有用修正。'),
    'ig_local_apg':dict(source='ig',title='局部弱参考＋反向动量/径向投影',
        fixed=dict(locality=2.),parameter='apg_beta',values=BETAS,strengths=IG_STRENGTHS,
        rationale='对新的上下文差方向应用因果反向动量与clean正交投影，检验径向放大是否限制局部参考收益。',
        equation='d=N_D(S-W_local); M=d+beta*Mprev; u=P_m_perp cap(M,2||D||)',
        limitation='APG从CFG迁到IG未获理论保证；投影可能移除局部化带来的有效方向。'),
    'cfg_apg_rescale':dict(source='cfg',title='APG＋通道方差回缩',
        fixed=dict(apg_beta=-.5),parameter='channel_rescale',values=MIXES,strengths=CFG_APG_STRENGTHS,
        rationale='APG调整时间与径向方向，通道回缩控制有限clean预测的中心方差；检验二者是否互补。',
        equation='u=APG(Dc); G=m+b*a*u; G=(1-r)G+r*channel_rescale(G,m)',
        limitation='回缩可能重新改变APG的径向几何；不宣称复合操作仍保留各组件的所有性质。'),
    'cfg_secant_apg':dict(source='cfg',title='条件割线＋APG',
        fixed=dict(apg_beta=-.5),parameter='condition_scale',values=SECANTS,strengths=CFG_APG_STRENGTHS,
        rationale='先在条件embedding内选择方向，再用APG控制重复历史增益；分开条件响应与时间几何。',
        equation='d=N_Dc(S(eu+r(ec-eu))-U); u=APG(d)',
        limitation='条件embedding插值没有一般概率解释；APG可能抹掉割线方向的收益。'),
    'cfg_secant_rescale':dict(source='cfg',title='条件割线＋通道方差回缩',
        fixed=dict(condition_scale=.5),parameter='channel_rescale',values=MIXES,strengths=CFG_DIRECTION_STRENGTHS,
        rationale='改变条件响应方向后独立校准每通道有限预测方差，检验输出增益是否仍限制效果。',
        equation='u=N_Dc(S(eu+.5(ec-eu))-U); rescale(m+b*a*u)',
        limitation='方向归一化与通道回缩会相互作用；需要各自原生组件曲线对照。'),
    'cfg_secant_apg_rescale':dict(source='cfg',title='条件割线＋APG＋通道方差回缩',
        fixed=dict(condition_scale=.5,apg_beta=-.5),parameter='channel_rescale',values=MIXES,strengths=CFG_APG_STRENGTHS,
        rationale='固定次序为条件方向→历史/径向控制→通道方差校准，检查三种误差控制是否有增量收益。',
        equation='d=secant_norm; u=APG(d); rescale(m+b*a*u)',
        limitation='更多组件可能只是重复缩放；必须超过调优后的单组件及两组件曲线才能认领融合价值。'),
}


def configurations():
    output=[]
    def add(family,source,strength,theta=0.,key='native_ig',role='component',parameters=None,solver='heun64'):
        index=sum(c['family']==family for c in output)
        output.append(dict(arm=f'{family}_{index:02d}',family=family,key=key,source=source,
            strength=float(strength),theta=float(theta),solver=solver,cutoff=.5 if source=='ig' else .75,
            role=role,parameters=parameters or {}))
    add('strong','ig',0.,role='native')
    add('ig_dopri','ig',.7,solver='dopri5',role='native')
    for strength in (.5,.6,.7,.8,.9,1.):add('ig_native','ig',strength,role='native')
    for strength in (.5,.6,.7,.8,.9,1.):add('ig_local','ig',strength,2.,key='ig_local_attention')
    for strength in (.5,.6,.7,.8,.9,1.):add('ig_adg','ig',strength,key='paper_adg')
    for strength in IG_STRENGTHS:
        for value in (.25,.5,1.):add('ig_residual','ig',strength,value,key='ig_partial_residual')
    for strength in IG_STRENGTHS:
        for value in (.1,.5,2.):add('ig_metric','ig',strength,value,key='ig_channel_metric')
    for strength in IG_STRENGTHS:
        for beta in BETAS:add('ig_apg','ig',strength,key='composed_guidance',parameters=dict(apg_beta=beta))
    for family,recipe in FAMILIES.items():
        if recipe['source']!='ig':continue
        for strength in recipe['strengths']:
            for theta in recipe['values']:
                add(family,'ig',strength,theta,key='composed_guidance',role='fusion',
                    parameters=dict(recipe['fixed'],**{recipe['parameter']:theta}))
    for strength in (.75,1.,1.25,1.5,1.75,2.,2.5,3.):add('cfg_native','cfg',strength,key='native_cfg',role='native')
    for strength in CFG_APG_STRENGTHS:
        for beta in BETAS:add('cfg_apg','cfg',strength,beta,key='cfg_apg_momentum')
    for strength in CFG_DIRECTION_STRENGTHS:
        for value in SECANTS:add('cfg_secant','cfg',strength,value,key='cfg_condition_secant')
    for strength in CFG_DIRECTION_STRENGTHS:
        for value in MIXES:add('cfg_rescale','cfg',strength,value,key='cfg_channel_rescale')
    for family,recipe in FAMILIES.items():
        if recipe['source']!='cfg':continue
        for strength in recipe['strengths']:
            for theta in recipe['values']:
                add(family,'cfg',strength,theta,key='composed_guidance',role='fusion',
                    parameters=dict(recipe['fixed'],**{recipe['parameter']:theta}))
    assert len(output)==184 and len({c['arm'] for c in output})==184
    assert all(sum(c['family']==family for c in output)==12 for family in FAMILIES)
    assert sum(c['role']=='fusion' for c in output)==84
    return output


def diagnose(z,q,d,value,amount):
    correction=value-q.s;off=correction-old.projection(correction,d)
    return torch.stack(((old.norm(correction)/(amount*old.norm(d)).clamp_min(old.EPS)).flatten(),
        (old.squared(off)/old.squared(correction).clamp_min(old.EPS)).flatten(),
        (old.norm(z+q.b*value)/old.norm(q.m).clamp_min(old.EPS)).flatten(),
        z.new_zeros((len(z),)),(old.norm(d).flatten()<=old.EPS).to(z.dtype)),-1)


def evaluate(rt,z,time_value,config,amount,ctx):
    if config['key'] not in ('composed_guidance','paper_adg') or amount==0:
        return ORIGINAL_EVALUATE(rt,z,time_value,config,amount,ctx)
    q=old.Queries(rt,z,time_value,config['key'],ctx)
    d=q.s-q.null()[0] if config['source']=='cfg' else q.d
    info=dict(gap=d,clean=q.m)
    if config['key']=='paper_adg':
        value,_=angular.adg_field(z,q.s,q.w,q.t,amount)
        info['diagnostics']=diagnose(z,q,d,value,amount)
        return value,info
    params=config['parameters'];u=d
    if 'locality' in params:
        weak=q.call(kind='weak',perturb=('ig_local_attention',params['locality']))
        u=q.s-weak
        if params.get('residual',0.):u=u-params['residual']*q.predictable()
        u=old.match(u,d)
    else:assert not params.get('residual',0.),'Residual fusion is defined after the local reference only.'
    if 'condition_scale' in params and params['condition_scale']!=1.:
        assert config['source']=='cfg'
        conditional,unconditional=q.class_embeddings()
        changed=q.call(kind='full',embedding=unconditional+params['condition_scale']*(conditional-unconditional))
        u=old.match(changed-q.null()[0],d)
    if params.get('channel_ridge') is not None:
        u=old.channel_metric(u,q.m,params['channel_ridge'])
    if params.get('apg_beta') is not None:
        momentum=u+params['apg_beta']*ctx.history.get('momentum',torch.zeros_like(u))
        info['momentum']=momentum
        bounded=old.cap(momentum,2*old.norm(d))
        u=bounded-old.projection(bounded,q.m)
    if params.get('channel_rescale',0.):
        mix=params['channel_rescale'];guided=q.m+q.b*amount*u
        mean=guided.mean((2,3),keepdim=True)
        sd=(q.m-q.m.mean((2,3),keepdim=True)).square().mean((2,3),keepdim=True).sqrt()
        gd=(guided-mean).square().mean((2,3),keepdim=True).sqrt()
        calibrated=mean+(guided-mean)*(sd/gd.clamp_min(old.EPS))
        clean=(1-mix)*guided+mix*calibrated
        u=(clean-q.m)/(q.b*amount)
    value=q.s+amount*u
    info['diagnostics']=diagnose(z,q,d,value,amount)
    return value,info


def install():
    assert old.evaluate in (ORIGINAL_EVALUATE,evaluate)
    old.evaluate=evaluate


def sample(rt,noise,labels,config,*,zero=False):
    from experiments.lifting_scale_sweep_20260909 import array_sha
    install()
    seed=int(array_sha(noise.detach().cpu().numpy())[:15],16)
    return old.sample(rt,noise,labels,config,batch_seed=seed,zero=zero)


@torch.inference_mode()
def limiting_checks(rt,noise,labels):
    """Nontrivial formula/order checks against previously frozen components."""
    rt.labels=labels;ctx=old.StepContext(shadow=noise.clone())
    ctx.history['momentum']=torch.cos(noise)
    checks=[]
    cases=[
        ('ig_local_attention','ig',2.,dict(locality=2.,residual=0.)),
        ('ig_local_attention','ig',2.,dict(locality=2.,channel_ridge=None)),
        ('ig_local_attention','ig',2.,dict(locality=2.,apg_beta=None)),
        ('cfg_apg_momentum','cfg',-.5,dict(apg_beta=-.5,channel_rescale=0.)),
        ('cfg_apg_momentum','cfg',-.5,dict(apg_beta=-.5,condition_scale=1.)),
        ('cfg_condition_secant','cfg',.5,dict(condition_scale=.5,channel_rescale=0.)),
        ('cfg_channel_rescale','cfg',.75,dict(condition_scale=1.,channel_rescale=.75)),
    ]
    for key,source,theta,params in cases:
        for tv in (0.,.375):
            config=dict(arm='limit',key=key,source=source,theta=theta,strength=.8,
                solver='heun64',cutoff=.5 if source=='ig' else .75)
            expected,reference=ORIGINAL_EVALUATE(rt,noise,tv,config,.8,ctx)
            changed=dict(config,key='composed_guidance',parameters=params)
            actual,info=evaluate(rt,noise,tv,changed,.8,ctx)
            assert torch.equal(actual,expected),(key,tv,float((actual-expected).abs().max()))
            if 'momentum' in reference:assert torch.equal(info['momentum'],reference['momentum'])
            checks.append(dict(reference=key,t=tv,parameters=params,field_exact=True))
    return checks


def cpu_checks():
    configs=configurations()
    assert len(configs)==184
    # Channel rescaling must preserve the guided channel mean. Its mix=0 limit
    # and APG's P_perp geometry are checked separately from model sampling.
    generator=torch.Generator().manual_seed(202610066)
    m=torch.randn((3,4,8,8),generator=generator,dtype=torch.float64)
    d=torch.randn(m.shape,generator=generator,dtype=m.dtype)
    guided=m+.7*d
    mean=guided.mean((2,3),keepdim=True)
    target_std=m.std((2,3),keepdim=True,unbiased=False)
    calibrated=mean+(guided-mean)*target_std/guided.std((2,3),keepdim=True,unbiased=False)
    torch.testing.assert_close(calibrated.mean((2,3)),guided.mean((2,3)),rtol=0,atol=1e-15)
    torch.testing.assert_close(calibrated.std((2,3),unbiased=False),m.std((2,3),unbiased=False),rtol=0,atol=1e-15)
    perpendicular=d-old.projection(d,m)
    assert old.inner(perpendicular,m).abs().max()<1e-10
    return dict(passed=True,configurations=184,fusion_configurations=84,fusion_families=7,
        component_and_native_controls=100,channel_means_preserved=True,channel_variances_matched=True,
        apg_orthogonal_projection=True)


if __name__=='__main__':
    import json
    print(json.dumps(cpu_checks()))

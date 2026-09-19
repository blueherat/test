"""Five appended hypotheses and explicit controls on the existing paired 1K bank."""
from experiments import run_sit_control_50ideas_20260910 as previous

STRENGTHS = (.5, 1.25, 2., 2.75)
EVENTS = (8, 16, 24, 32, 40)
WRITE_EVENTS = (8, 24, 40)
NOISE_SEED = previous.NOISE_SEED
IDEAS = [
    dict(id=54, key='curl_refine', title='非保守响应触发主采样细分',
         values=(.004, .008, .016), parameter='h times typical skew threshold', semantic=False,
         claim='用完整受控场的反对称响应分配主积分精度；不直接把旋度当成应删除的语义误差。'),
    dict(id=55, key='future_moment_apg', title='带语义约束的未来幅度投影',
         values=(.1, 1., 10.), parameter='normalized future-moment penalty', semantic=True,
         claim='在APG平行/正交输入平面内识别真实未来响应，再限制未来幅度变化并实际验证类别进展。'),
    dict(id=56, key='future_eta', title='真实续生成选择APG平行保留量',
         values=(0., .1, 1.), parameter='future-moment penalty', semantic=True,
         claim='比较实际8步APG候选块的统一16步null终点，从eta=0,.5,1中逐样本选择。'),
    dict(id=57, key='marginal_release', title='边际条件价值驱动的撤条件',
         values=(0., .002, .01), parameter='minimum eight-step continuation value', semantic=True,
         claim='在null结局已有足够类别置信度时，比较继续条件和撤条件的配对未来价值，执行不可逆null接管。'),
    dict(id=58, key='terminal_horizon', title='统一终点检验的前瞻时距选择',
         values=(0., .002, .01), parameter='minimum common-terminal q improvement', semantic=True,
         claim='三个物理时距分别提出根校正，用同一个完整null终点语义读出验收和选时距。'),
]


def configurations():
    rows = []
    for config in previous.configurations():
        if (config['family'] in ('strong', 'ig_local', 'cfg_apg') or
                config['family'] == 'cfg_native' and config['strength'] in STRENGTHS):
            rows.append(dict(config, parameters=dict(config['parameters'], inherited_exact=True)))

    def add(family, key, amount, theta=0., *, semantic=False, idea=None, **parameters):
        index = sum(c['family'] == family for c in rows)
        rows.append(dict(arm=f'{family}_{index:02d}', family=family, key=key,
            strength=float(amount), theta=float(theta), role='candidate' if idea else 'control',
            source='cfg', solver='heun64', cutoff=.75, idea_id=idea,
            external_semantics=semantic, parameters=parameters))

    for amount in STRENGTHS:
        for beta in (-.75, -.5, -.25):
            for radius in (2.5, 5., 10.):
                add('clean_apg', 'clean_apg', amount, beta, radius=radius)
        for eta in (0., .5, 1.):
            add('projection_apg', 'projection_apg', amount, eta)
        for steps in (96, 128):
            add('uniform_refinement', 'uniform_refinement', amount, steps, main='cfg')
        for threshold in (.001, .003, .01):
            add('embedded_refinement', 'embedded_refinement', amount, threshold, main='cfg')
        for scale in (.5, 1., 2.):
            add('erk_guid', 'erk_guid', amount, scale, main='cfg', stiffness_threshold=0.)
        add('curl_gain_shrink', 'curl_gain_shrink', amount, .008, main='cfg')
        add('semantic_direction', 'semantic_direction', amount, 1., semantic=True)
        add('moment_without_semantic', 'moment_without_semantic', amount, 1., semantic=True)
        for release_time in (.25, .5, .75):
            add('fixed_null_release', 'fixed_null_release', amount, release_time)
        add('probability_null_release', 'probability_null_release', amount, .5, semantic=True)
        for horizon in (.125, .25, .5):
            add('fixed_horizon_verified', 'fixed_horizon_verified', amount, horizon, semantic=True)
    assert len(rows) == 141
    for method in IDEAS:
        for amount in STRENGTHS:
            for theta in method['values']:
                add(f'i{method["id"]}_{method["key"]}', method['key'], amount, theta,
                    semantic=method['semantic'], idea=method['id'],
                    main='cfg' if method['id'] == 54 else 'projected',
                    events=list(EVENTS if method['id'] in (54, 57) else WRITE_EVENTS),
                    future_steps=16, swept_parameter=method['parameter'])
    assert len(rows) == 201 and len({c['arm'] for c in rows}) == 201
    assert all(sum(c['idea_id'] == m['id'] for c in rows) == 12 for m in IDEAS)
    return rows

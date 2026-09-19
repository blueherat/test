"""Five hypotheses, four strengths, and controls with explicit provenance."""
from __future__ import annotations
import json
from .data import ROOT, METHODS

OLD = ROOT.parent/'sit_control_output_50ideas_20260910/control_screen_1k'
STRENGTHS = (.5, 1.25, 2., 2.75)
NOISE_SEED = 202610100
IDEAS = {
    'clt': dict(id=59, title='保均值协方差的高阶平滑', route='AG/IG机制'),
    'factor': dict(id=60, title='保区域边际的依赖消除', route='AG/IG机制'),
    'local_label': dict(id=61, title='保总体先验的类别概率粗化', route='CFG'),
    'conditional_refresh': dict(id=62, title='条件生成流中的随机更新', route='CFG'),
    'defensive': dict(id=63, title='包含原分布的弱模型', route='AG/IG机制'),
}
FAMILIES = IDEAS


def configurations():
    original = json.loads((OLD/'request.json').read_text())['configs']
    selected = ('strong_00', 'ig_local_00', 'cfg_native_04', 'cfg_apg_07')
    rows = [dict(c, parameters=dict(c['parameters'], inherited_exact=True))
            for c in original if c['arm'] in selected]
    assert len(rows) == 4
    rows.append(dict(arm='ig_native_matched_00', family='ig_native', key='native_ig',
        strength=.8, theta=0., role='native', source='ig', solver='heun64', cutoff=.5,
        idea_id=None, external_semantics=False, parameters=dict(inherited_exact=True)))

    def add(key, amount, theta=0., *, method=None, idea=None, family=None, **parameters):
        family = family or (f'i{idea}_{key}' if idea else key)
        index = sum(c['family'] == family for c in rows)
        source = 'cfg' if key != 'measure_ag' or method in ('local_label', 'random_label') else 'ig'
        rows.append(dict(arm=f'{family}_{index:02d}', family=family, key=key,
            strength=float(amount), theta=float(theta), role='candidate' if idea else 'control',
            source=source, solver='heun64', cutoff=.75 if source == 'cfg' else .5,
            idea_id=idea, external_semantics=False,
            parameters=dict(method=method, **parameters)))

    for method in METHODS:
        idea = IDEAS.get(method, {}).get('id')
        for amount in STRENGTHS:
            add('measure_ag', amount, method=method, idea=idea,
                family=f'i{idea}_{method}' if idea else 'weak_'+method)
    for amount in STRENGTHS:
        for rho in (.9, .97):
            add('conditional_refresh', amount, rho, idea=62, event=32, inverse_steps=32)
        add('gaussian_refresh', amount, .9, event=32)
        add('conditional_refresh', amount, 1., family='roundtrip_control', event=32, inverse_steps=32)
        add('hard_pair', amount, method='local', family='hard_pair_control')
    for rho in (.9, .97):
        add('conditional_refresh', 0., rho, family='unguided_refresh', event=32, inverse_steps=32)
    assert len(rows) == 59 and len({c['arm'] for c in rows}) == len(rows)
    assert {c['idea_id'] for c in rows if c['role'] == 'candidate'} == {59, 60, 61, 62, 63}
    return rows

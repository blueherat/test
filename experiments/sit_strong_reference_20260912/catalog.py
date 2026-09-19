from experiments.sit_sampler_reference_20260912 import catalog as parent

WORK, OLD, MEASURE = parent.WORK, parent.OLD, parent.MEASURE
PARENT_ROOT = parent.ROOT
ROOT = PARENT_ROOT.parent / 'sit_strong_reference_20260912'
PROTOCOL = WORK / 'docs/SIT_STRONG_REFERENCE_PROTOCOL_20260912_ZH.md'
STAGE, NOISE_SEED = 'strong_reference_screen_1k', parent.NOISE_SEED
DATA_SEED, DATA_SAMPLES = parent.DATA_SEED, parent.DATA_SAMPLES
STEPS, BATCH, LR, EMA, SEED = parent.STEPS, parent.BATCH, parent.LR, parent.EMA, parent.SEED
METHOD_SPECS = {
    'ig_teacher': dict(dataset='real', target='strong_prediction'),
    'ig_strong': dict(dataset='strong', target='flow_matching'),
    'ig_teacherstrong': dict(dataset='strong', target='strong_prediction'),
    'cfg_strong': dict(dataset='strong', target='flow_matching'),
}
METHODS = tuple(METHOD_SPECS)
REUSED_METHODS = ('ig_real', 'ig_ig', 'cfg_real', 'cfg_cfg')
FAMILIES = {s+'_strong_reference': dict(title='冻结strong的预测或生成分布作为参考目标', source=s,
    hypothesis='Teacher fitting can avoid amplifying shared readable bias; fitting generated data targets a different velocity field.',
    limitation='A projection identity is not a density law or a quality guarantee; direct prediction fitting has a prior repository analogue.')
    for s in ('ig','cfg')}


def planned():
    rows=[]
    for source, variants in (('ig',('original','real','teacher','strong','teacherstrong','ig')),
                             ('cfg',('original','real','strong','cfg'))):
        for variant in variants:
            method=source+'_'+variant
            rows.append(dict(arm=method+'_00', family=source+'_strong_reference', key='strong_reference',
                source=source, strength=.8 if source=='ig' else 1.25, theta=0., solver='heun64',
                cutoff=.5 if source=='ig' else .75, role='candidate' if method in METHODS else 'control',
                idea_id=None, external_semantics=False,
                parameters=dict(head_method=method, no_extra_backbone=True,
                    **METHOD_SPECS.get(method,dict(dataset=variant,target='flow_matching')))))
    original=next(r for r in rows if r['arm']=='ig_original_00')
    rows.append(dict(original,arm='ig_half_00',strength=.4,parameters=dict(original['parameters'],fixed_half_strength=True)))
    rows += [r for r in parent.configurations() if r['arm'] in ('strong_00','cfg_apg_07','ig_adg_reference_00')]
    assert len(rows)==14 and len({r['arm'] for r in rows})==14
    return rows


def configurations():
    return [r for r in parent.configurations() if r['arm'] in ('ig_local_00','cfg_native_04')] + planned()


def head_folder(method):
    return (ROOT if method in METHODS else PARENT_ROOT) / 'training' / method

from experiments.sit_strong_reference_20260912 import catalog as previous
from experiments.sit_sampler_reference_20260912 import catalog as reused

WORK, OLD, MEASURE = previous.WORK, previous.OLD, previous.MEASURE
PARENT_ROOT = reused.ROOT
ROOT = PARENT_ROOT.parent / 'cfg_posterior_reference_20260912'
PROTOCOL = WORK / 'docs/CFG_POSTERIOR_REFERENCE_PROTOCOL_20260912_ZH.md'
STAGE, NOISE_SEED = 'posterior_reference_screen_1k', previous.NOISE_SEED
STEPS, BATCH, LR, EMA, SEED = previous.STEPS, previous.BATCH, previous.LR, previous.EMA, previous.SEED
TEACHER_LABEL_SEED = 2026121216
METHOD_SPECS = {
    'cfg_posterior': dict(dataset='real', target='strong_prediction', teacher_label='paired'),
    'cfg_independent': dict(dataset='real', target='strong_prediction', teacher_label='independent'),
}
METHODS = tuple(METHOD_SPECS)
REUSED_METHODS = ('cfg_real','cfg_cfg')
FAMILIES = {'cfg_posterior_reference':dict(title='用配对条件teacher训练null参考', source='cfg',
    hypothesis='A label-blind reference fitted to paired conditional predictions learns their posterior average; independent labels target a different field.',
    limitation='Exact conditional means imply the marginal identity; finite teacher and readout errors remain, and no endpoint quality guarantee follows.')}


def planned():
    rows=[]
    for variant in ('original','real','posterior','independent','cfg'):
        method='cfg_'+variant
        rows.append(dict(arm=method+'_00', family='cfg_posterior_reference', key='posterior_reference',
            source='cfg', strength=1.25, theta=0., solver='heun64',cutoff=.75,
            role='candidate' if variant=='posterior' else 'control',idea_id=None,external_semantics=False,
            parameters=dict(head_method=method,no_extra_backbone=True,
                **METHOD_SPECS.get(method,dict(dataset=variant,target='flow_matching')))))
    original=rows[0]
    rows.append(dict(original,arm='cfg_half_00',strength=.625,parameters=dict(original['parameters'],fixed_half_strength=True)))
    rows += [r for r in previous.planned() if r['arm']=='cfg_apg_07']
    assert len(rows)==7 and len({r['arm'] for r in rows})==7
    return rows


def configurations():
    return [r for r in previous.configurations() if r['arm'] in ('strong_00','ig_local_00','cfg_native_04','ig_original_00')] + planned()


def head_folder(method):
    return (ROOT if method in METHODS else PARENT_ROOT)/'training'/method

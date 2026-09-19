from pathlib import Path
import json

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_guided_objective_20260912')
MEASURE = ROOT.parent / 'sit_measure_guidance_20260912'
OLD = ROOT.parent / 'sit_control_output_50ideas_20260910/control_screen_1k'
STAGE = 'objective_screen_1k'
PROTOCOL = WORK / 'docs/SIT_GUIDED_OBJECTIVE_PROTOCOL_20260912_ZH.md'
NOISE_SEED = 202610100
METHODS = ('ig_native', 'ig_direct', 'cfg_native', 'cfg_direct')
STEPS, BATCH, LR, EMA, SEED = 1500, 32, 1e-4, .995, 2026120913
FAMILIES = {source + '_direct': dict(
    title='按最终引导预测训练参考读出', source=source,
    hypothesis='A reference trained for standalone denoising can leave predictable bias in the composed guided predictor.',
    limitation='A lower teacher-state quadratic risk does not guarantee a better generated distribution.')
    for source in ('ig', 'cfg')}


def planned():
    rows = []
    for source in ('ig', 'cfg'):
        for target in ('original', 'native', 'direct'):
            method = source + '_' + target
            rows.append(dict(arm=method + '_00', family=method, key='objective_readout',
                source=source, strength=.8 if source == 'ig' else 1.25, theta=0.,
                solver='heun64', cutoff=.5 if source == 'ig' else .75,
                role='candidate' if target == 'direct' else 'control',
                idea_id=None, external_semantics=False,
                parameters=dict(head_method=method, target=target, no_extra_backbone=True)))
    old = json.loads((OLD / 'request.json').read_text())['configs']
    apg = next(c for c in old if c['arm'] == 'cfg_apg_07')
    rows.append(dict(apg, parameters=dict(apg['parameters'], inherited_exact=True)))
    rows.append(dict(arm='ig_adg_reference_00', family='ig_adg_reference', key='paper_adg',
        source='ig', strength=.8, theta=0., solver='heun64', cutoff=.5, role='control',
        idea_id=None, external_semantics=False, parameters=dict(inherited_exact=True)))
    return rows


def configurations():
    old = json.loads((OLD / 'request.json').read_text())['configs']
    names = ('strong_00', 'ig_local_00', 'cfg_native_04')
    anchors = [dict(c, parameters=dict(c['parameters'], inherited_exact=True)) for c in old if c['arm'] in names]
    return anchors + planned()

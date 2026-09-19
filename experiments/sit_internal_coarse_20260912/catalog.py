from pathlib import Path
import json

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_internal_coarse_20260912')
MEASURE = ROOT.parent/'sit_measure_guidance_20260912'
OLD = ROOT.parent/'sit_control_output_50ideas_20260910/control_screen_1k'
STAGE = 'internal_screen_1k'
PROTOCOL = WORK/'docs/SIT_INTERNAL_COARSE_PROTOCOL_20260912_ZH.md'
NOISE_SEED = 202610100
METHODS = ('native', 'clt')
STEPS, BATCH, LR, EMA = 1500, 32, .0001, .995
SEED = 2026120913
FAMILIES = {'ig_internal_clt': dict(idea_id=59,
    title='将固定高阶粗化分布用于原有内部预测头',
    hypothesis='An explicitly coarsened training measure can define an internal weak readout without a second backbone.',
    limitation='A restricted head need not fit the target distribution; a score-ratio identity does not guarantee endpoint quality.')}


def planned():
    rows=[]
    for method in ('original', *METHODS):
        rows.append(dict(arm='ig_internal_'+method+'_00', family='ig_internal_'+method,
            key='internal_head_ig', source='ig', strength=.8, theta=0., solver='heun64', cutoff=.5,
            role='candidate' if method=='clt' else 'control', idea_id=59 if method=='clt' else None,
            parameters=dict(head_method=method, depth=4, no_norm_matching=True)))
    return rows


def configurations():
    rows=json.loads((OLD/'request.json').read_text())['configs']
    names=('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')
    anchors=[dict(c,parameters=dict(c['parameters'],inherited_exact=True)) for c in rows if c['arm'] in names]
    return anchors+planned()

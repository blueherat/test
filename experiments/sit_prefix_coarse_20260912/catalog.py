from pathlib import Path
import json

WORK=Path('/home/zhoushunyu/eqvae')
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/sit_prefix_coarse_20260912')
MEASURE=ROOT.parent/'sit_measure_guidance_20260912'
OLD=ROOT.parent/'sit_control_output_50ideas_20260910/control_screen_1k'
STAGE='prefix_screen_1k'
PROTOCOL=WORK/'docs/SIT_PREFIX_COARSE_PROTOCOL_20260912_ZH.md'
NOISE_SEED=202610100
METHODS=('native','clt')
STEPS,BATCH,LR,EMA=1500,32,.0001,.995
SEED=2026120913
FAMILIES={'ig_prefix_clt':dict(idea_id=59,title='同深度弱前缀学习高阶粗化分布',
    hypothesis='A fixed readout can miss a moment-preserving distribution change; adapt the weak nonlinear features while preserving the strong model.',
    limitation='Affine-regression invariance is only a limiting case; four-block features are nonlinear and finite training is not exact density fitting.')}


def anchors():
    rows=json.loads((OLD/'request.json').read_text())['configs']
    names=('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')
    return [dict(c,parameters=dict(c['parameters'],inherited_exact=True)) for c in rows if c['arm'] in names]


def planned():
    rows=[]
    for method in ('original',*METHODS):
        rows.append(dict(arm='ig_prefix_'+method+'_00',family='ig_prefix_'+method,
            key='weak_prefix_ig',source='ig',strength=.8,theta=0.,solver='heun64',cutoff=.5,
            role='candidate' if method=='clt' else 'control',idea_id=59 if method=='clt' else None,
            parameters=dict(head_method=method,depth=4,no_norm_matching=True)))
    local=next(c for c in anchors() if c['family']=='ig_local')
    rows.insert(1,dict(local,arm='ig_prefix_local_00',role='control'))
    return rows


def configurations():return anchors()+planned()

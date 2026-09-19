from pathlib import Path
import json

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_posterior_guidance_20260912')
OLD = ROOT.parent/'sit_control_output_50ideas_20260910/control_screen_1k'
STAGE = 'routing_screen_1k'
NOISE_SEED = 202610100
FAMILIES = {'routing_odds': {'title':'先改变对应关系概率，再聚合value', 'idea_id':64,
    'hypothesis':'Part of extrapolation distortion comes from changing averages instead of explanation probabilities.',
    'limitation':'Attention is an unverified proxy for explanatory modes; local null modulation is not a full unconditional posterior.'}}


def configurations():
    original = json.loads((OLD/'request.json').read_text())['configs']
    names = ('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')
    rows = [dict(c,parameters=dict(c['parameters'],inherited_exact=True))
        for c in original if c['arm'] in names]
    for key in ('routing_odds','routing_linear','routing_temperature','embedding_extrapolation','manual_kernel'):
        candidate = key == 'routing_odds'
        rows.append(dict(arm=key+'_00',family=key,key=key,source='cfg',strength=1. if key!='manual_kernel' else 0.,
            theta=0.,solver='heun64',cutoff=.75,role='candidate' if candidate else 'control',
            idea_id=64 if candidate else None,external_semantics=False,
            parameters=dict(all_blocks=True,shared_values=True,local_null_modulation=True)))
    assert len(rows)==9 and len({c['arm'] for c in rows})==9
    return rows

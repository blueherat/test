from pathlib import Path
from experiments.sit_guided_objective_20260912 import catalog as parent

WORK, OLD, MEASURE = parent.WORK, parent.OLD, parent.MEASURE
ROOT = parent.ROOT.parent / 'sit_sampler_reference_20260912'
PROTOCOL = WORK / 'docs/SIT_SAMPLER_REFERENCE_PROTOCOL_20260912_ZH.md'
STAGE, NOISE_SEED = 'sampler_reference_screen_1k', 202610100
DATA_SEED, DATA_SAMPLES = 2026121014, 2000
STEPS, BATCH, LR, EMA, SEED = 1500, 32, 1e-4, .995, 2026120913
METHODS = tuple(s+'_'+d for s in ('ig','cfg') for d in ('real','ig','cfg'))
FAMILIES = {s+'_sampler_reference': dict(title='生成器来源交叉的参考训练', source=s,
    hypothesis='A negative reference fitted to the sampler output can encode its overrepresented errors.',
    limitation='Synthetic negative guidance is known; shallow readouts may not learn sampler densities.')
    for s in ('ig','cfg')}


def planned():
    rows = []
    for source in ('ig','cfg'):
        for dataset in ('original','real','ig','cfg'):
            method = source+'_'+dataset
            rows.append(dict(arm=method+'_00', family=source+'_sampler_reference',
                key='sampler_reference', source=source, strength=.8 if source=='ig' else 1.25,
                theta=0., solver='heun64', cutoff=.5 if source=='ig' else .75,
                role='candidate' if dataset in ('ig','cfg') else 'control',
                idea_id=None, external_semantics=False,
                parameters=dict(head_method=method, dataset=dataset, no_extra_backbone=True)))
    # Same-bank published references are re-run, not compared across sample banks.
    rows += [c for c in parent.planned() if c['arm'] in ('cfg_apg_07','ig_adg_reference_00')]
    return rows


def configurations():
    anchors = [c for c in parent.configurations() if c['arm'] in ('strong_00','ig_local_00','cfg_native_04')]
    return anchors + planned()

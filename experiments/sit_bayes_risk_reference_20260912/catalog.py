from experiments.sit_reference_compilation_20260912 import catalog as previous
WORK,OLD,MEASURE=previous.WORK,previous.OLD,previous.MEASURE
parent=previous.parent
ROOT=previous.ROOT.parent/'sit_bayes_risk_reference_20260912'
PROTOCOL=WORK/'docs/SIT_BAYES_RISK_REFERENCE_PROTOCOL_20260912_ZH.md'
STAGE,CONFIRM='risk_screen_1k','risk_confirm_5k'
NOISE_SEED,TRAIN_SEED,CONFIRM_SEED=2026121228,2026121229,2026121230
METHODS=('ig_square','ig_quartic')
STEPS,BATCH,LR,EMA=3000,32,1e-4,.995
FAMILIES={**previous.FAMILIES,'bayes_risk':dict(title='由损失定义的后验弱参考',
    hypothesis='Quartic regression changes the conditional Bayes action; for two-point posteriors it flattens odds by a cube root, while symmetric posteriors have the same mean and quartic action.',
    limitation='Separable quartic actions need not be a score or a tempered joint posterior. Finite readout capacity and tail sensitivity can defeat the proposed guidance.')}


def planned():
    rows=[dict(arm=method+'_00',family='bayes_risk',key='bayes_risk_reference',source='ig',
        strength=.8,theta=0.,solver='heun64',cutoff=.5,role='candidate' if method=='ig_quartic' else 'control',
        idea_id=None,external_semantics=False,parameters=dict(method=method,no_extra_backbone=True)) for method in METHODS]
    rows += [dict(r,role='control') for r in previous.planned() if r['arm'] in
        ('ig_original_00','ig_ig_00','ig_adg_reference_00')]
    assert len(rows)==5
    return rows


def configurations():
    return [r for r in previous.configurations() if r['arm'] in
        ('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')]+planned()

from experiments.sit_reference_compilation_20260912 import catalog as previous
WORK, OLD, MEASURE = previous.WORK, previous.OLD, previous.MEASURE
parent = previous.parent
ROOT = previous.ROOT.parent/'sit_replica_reference_20260912'
PROTOCOL = WORK/'docs/SIT_REPLICA_REFERENCE_PROTOCOL_20260912_ZH.md'
STAGE, CONFIRM = 'replica_screen_1k', 'replica_confirm_5k'
NOISE_SEED, TRAIN_SEED, CONFIRM_SEED = 2026121235, 2026121236, 2026121237
METHODS = ('ig_pair_mse', 'ig_pair_consistent', 'ig_pair_average', 'ig_pair_shuffled')
STEPS, BATCH, LR, EMA = 3000, 16, 1e-4, .995
FAMILIES = {**previous.FAMILIES, 'replica': dict(
    title='真实图像重复加噪定义的参考风险',
    hypothesis='Same-image prediction consistency induces the resolvent (2I-T)^-1 m of the posterior resampling operator. Averaging predictions before squared loss gives the opposite operator 2(I+T)^-1 m.',
    limitation='Function smoothing is not density smoothing; general multidimensional readouts need not be scores. A frozen strong model need not share this regularization bias.')}


def planned():
    rows = [dict(arm=method+'_00', family='replica', key='replica_reference', source='ig',
        strength=.8, theta=0., solver='heun64', cutoff=.5,
        role='candidate' if method=='ig_pair_consistent' else 'control',
        idea_id=None, external_semantics=False,
        parameters=dict(method=method, no_extra_backbone=True)) for method in METHODS]
    rows += [dict(r, role='control') for r in previous.planned() if r['arm'] in
        ('ig_original_00', 'ig_ig_00', 'ig_adg_reference_00')]
    assert len(rows)==7
    return rows


def configurations():
    return [r for r in previous.configurations() if r['arm'] in
        ('strong_00', 'ig_local_00', 'cfg_native_04', 'cfg_apg_07')]+planned()

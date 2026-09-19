from experiments.sit_strong_reference_20260912 import catalog as parent

WORK, OLD, MEASURE = parent.WORK, parent.OLD, parent.MEASURE
ROOT = parent.ROOT.parent / 'sit_reference_compilation_20260912'
PREFIX = ROOT.parent / 'sit_prefix_coarse_20260912'
PROTOCOL = WORK / 'docs/SIT_REFERENCE_COMPILATION_PROTOCOL_20260912_ZH.md'
STAGE, CONFIRM = 'compiled_screen_1k', 'compiled_confirm_5k'
NOISE_SEED, CONFIRM_SEED = 2026121220, 2026121222
DATA_SEED, TRAIN_SEED = 2026121219, 2026121221
METHODS = ('ig_shallow', 'ig_deep', 'cfg_categorical')
STEPS, BATCH, LR, EMA = 3000, 32, 1e-4, .995
CODEWORDS, CODEBOOK_PATCHES, KMEANS_ITERATIONS = 256, 131072, 30
TRAJECTORIES, TRAIN_TRAJECTORIES = 1000, 800
FAMILIES = {
    'ig_compiled': dict(title='将已有效的弱参考离线编译为共享读出',
        hypothesis='A known useful weak function may be recoverable from frozen internal features without executing its separate prefix at inference.',
        limitation='Distillation and path stability are standard; lower teacher MSE alone does not establish preserved generation quality.'),
    'cfg_categorical': dict(title='干净latent候选的概率更新与均值外推',
        hypothesis='Tempering a learned common candidate posterior avoids extrapolating candidate means beyond their support.',
        limitation='Marginal categorical guidance is known in discrete diffusion and does not establish a joint continuous endpoint distribution; the strong anchor removes the full-support guarantee.')}


def planned():
    rows = []
    for method in ('ig_shallow', 'ig_deep', 'cfg_probability', 'cfg_mean'):
        source = method.split('_')[0]
        rows.append(dict(arm=method+'_00', family='ig_compiled' if source=='ig' else 'cfg_categorical',
            key='compiled_reference', source=source, strength=.8 if source=='ig' else 1.25,
            theta=0., solver='heun64', cutoff=.5 if source=='ig' else .75,
            role='control' if method=='cfg_mean' else 'candidate', idea_id=None, external_semantics=False,
            parameters=dict(method=method, no_extra_backbone=True)))
    names = ('ig_original_00', 'ig_ig_00', 'ig_adg_reference_00',
             'cfg_original_00', 'cfg_cfg_00', 'cfg_apg_07')
    rows += [dict(r, role='control', parameters=dict(r['parameters'], inherited_reference=True))
             for r in parent.planned() if r['arm'] in names]
    assert len(rows)==10 and len({r['arm'] for r in rows})==10
    return rows


def configurations():
    anchors=[dict(r,parameters=dict(r['parameters'],inherited_reference=True))
        for r in parent.configurations() if r['arm'] in ('strong_00','ig_local_00','cfg_native_04')]
    return anchors+planned()

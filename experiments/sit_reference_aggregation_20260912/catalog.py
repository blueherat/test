from experiments.sit_reference_compilation_20260912 import catalog as previous

WORK, OLD, MEASURE = previous.WORK,previous.OLD,previous.MEASURE
PARENT_ROOT, PREFIX = previous.ROOT,previous.PREFIX
parent=previous.parent
ROOT=PARENT_ROOT.parent/'sit_reference_aggregation_20260912'
PROTOCOL=WORK/'docs/SIT_REFERENCE_AGGREGATION_PROTOCOL_20260912_ZH.md'
STAGE,CONFIRM='aggregation_screen_1k','aggregation_confirm_5k'
DATA_SEED,NOISE_SEED,TRAIN_SEED,CONFIRM_SEED=2026121223,2026121224,2026121225,2026121226
METHODS=('ig_teachercontinuation','ig_studentaggregation')
STEPS,BATCH,LR,EMA=3000,32,1e-4,.995
TRAJECTORIES,TRAIN_TRAJECTORIES,CODEWORDS=1000,800,previous.CODEWORDS
FAMILIES={**previous.FAMILIES,'ig_aggregation':dict(title='固定教师的学生状态聚合',
    hypothesis='Teacher-state distillation can miss states induced by the compiled student; querying the same teacher on paired student rollouts may preserve its useful guidance more faithfully.',
    limitation='Dataset aggregation is an existing imitation-learning principle; no endpoint or FID guarantee follows from empirical regression alone.')}


def planned():
    rows=[]
    for method in METHODS:
        rows.append(dict(arm=method+'_00',family='ig_aggregation',key='reference_aggregation',source='ig',
            strength=.8,theta=0.,solver='heun64',cutoff=.5,role='candidate',idea_id=None,external_semantics=False,
            parameters=dict(method=method,no_extra_backbone=True)))
    rows += [dict(r,role='control') for r in previous.planned() if r['arm'] in
        ('ig_shallow_00','ig_original_00','ig_ig_00','ig_adg_reference_00')]
    assert len(rows)==6
    return rows


def configurations():
    return [r for r in previous.configurations() if r['arm'] in
        ('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')]+planned()

"""Frozen ten-mechanism catalog: five strengths by four regeneration times."""
from __future__ import annotations

CFG_STRENGTHS = (.75, 1.25, 1.75, 2.25, 2.75)
IG_STRENGTHS = (.4, .6, .8, 1., 1.2)
PROBE_TIMES = (.5, .65, .8, .9)
IDEAS = [
    dict(id=1, key='cfg_cycle_gain', source='cfg', title='按再生成方向增益限制CFG', coefficient=1.,
         hypothesis='沿条件差的小扰动若被再生成放大，减少这一状态上的额外CFG。'),
    dict(id=2, key='cfg_cycle_transport', source='cfg', title='用再生成传输后的条件方向引导', coefficient=.5,
         hypothesis='条件方向经再生成的响应可能比原始方向更能保留，混入其等范数方向。'),
    dict(id=3, key='cfg_cycle_antidrift', source='cfg', title='抑制与再生成漂移同向的CFG分量', coefficient=.5,
         hypothesis='只抑制会继续沿条件分支自身漂移推进的额外引导，不压缩整个条件差。'),
    dict(id=4, key='cfg_cycle_common_drift', source='cfg', title='校正条件与null的共同横向漂移', coefficient=.25,
         hypothesis='强正相关的共同漂移可能包含共享偏置；只修正其与当前条件差正交的部分。'),
    dict(id=5, key='cfg_cycle_curvature', source='cfg', title='按同噪声再生成非线性限制CFG', coefficient=1.,
         hypothesis='同时间同噪声下的二阶响应揭示局部线性外推失效，独立于一阶放大率。'),
    dict(id=6, key='ig_cycle_agreement', source='ig', title='按再生成前后的强弱差方向一致性门控', coefficient=1.,
         hypothesis='原强弱差与回灌后强弱差方向相反时，降低原差分的外推。'),
    dict(id=7, key='ig_cycle_extrapolate', source='ig', title='对回灌造成的强弱差旋转作有限外推', coefficient=.5,
         hypothesis='回灌系统改变差分方向时，有限反向外推可能抵消这部分变化；需反号对照。'),
    dict(id=8, key='ig_cycle_weakgain', source='ig', title='限制弱头相对强头的额外响应放大', coefficient=1.,
         hypothesis='弱参考在同一扰动上比strong更强的放大可能使差分失稳，按其超额增益门控。'),
    dict(id=9, key='ig_cycle_antithetic', source='ig', title='按相反噪声下强弱差的一致性门控', coefficient=1.,
         hypothesis='回灌强弱差若依赖噪声符号，其作为可外推参考可能不可靠；不直接平均替换gap。'),
    dict(id=10, key='ig_cycle_twohop', source='ig', title='按两轮回灌的方向保持和增长门控', coefficient=1.,
         hypothesis='首轮稳定但第二轮迅速放大或旋转的差分，应与持续稳定差分区别使用。'),
]


def configurations():
    out = []

    def add(family, key, source, strength, theta=0., *, role='control', idea=None,
            reference='native', steps=64, parameters=None):
        i = sum(row['family'] == family for row in out)
        out.append(dict(arm=f'{family}_{i:02d}', family=family, key=key, source=source,
            strength=float(strength), theta=float(theta), role=role, idea_id=idea,
            reference=reference, solver=f'heun{steps}', cutoff=.75 if source == 'cfg' else .5,
            parameters=dict(parameters or {})))

    add('strong', 'native', 'ig', 0.)
    for a in (.2, *IG_STRENGTHS):
        add('ig_native', 'native', 'ig', a)
        add('ig_mlp', 'native', 'ig', a, reference='mlp')
    for a in IG_STRENGTHS:
        add('ig_mlp_adg', 'adg', 'ig', a, reference='mlp')
    for a in (.25, .5, *CFG_STRENGTHS):
        add('cfg_native', 'native', 'cfg', a)
    for a in CFG_STRENGTHS:
        for beta in (-.5, -.25):
            add('cfg_apg', 'apg', 'cfg', a, beta)
        add('cfg_ctrl', 'ctrl', 'cfg', a, .2, parameters={'decay': 5.})
    for steps in (96, 128):
        add('cfg_more_steps', 'native', 'cfg', 1.25, steps=steps)
        add('ig_mlp_more_steps', 'native', 'ig', .8, reference='mlp', steps=steps)
    # Identical-probe amplitude controls isolate direction changes, not runtime overhead.
    for idea_id in (2, 3, 4, 7):
        row = IDEAS[idea_id-1]
        for tau in (.65, .8):
            add(f'norm_i{idea_id:02d}', row['key'], row['source'],
                1.25 if row['source'] == 'cfg' else .8, tau,
                reference='mlp' if row['source'] == 'ig' else 'native',
                parameters={'norm_only': True})
    for idea_id in (3, 4, 7):
        row = IDEAS[idea_id-1]
        add(f'reverse_i{idea_id:02d}', row['key'], row['source'],
            1.25 if row['source'] == 'cfg' else .8, .8,
            reference='mlp' if row['source'] == 'ig' else 'native',
            parameters={'reverse': True})
    for source in ('cfg', 'ig'):
        for tau in (.5, .9):
            add(f'{source}_probe_only', 'probe_only', source, 1.25 if source == 'cfg' else .8,
                tau, reference='mlp' if source == 'ig' else 'native')

    for row in IDEAS:
        for a in CFG_STRENGTHS if row['source'] == 'cfg' else IG_STRENGTHS:
            for tau in PROBE_TIMES:
                add(f'i{row["id"]:02d}_{row["key"]}', row['key'], row['source'], a, tau,
                    role='candidate', idea=row['id'], reference='mlp' if row['source'] == 'ig' else 'native')
    assert len({row['arm'] for row in out}) == len(out)
    assert sum(row['role'] == 'candidate' for row in out) == 200
    assert all(sum(row['idea_id'] == idea['id'] for row in out) == 20 for idea in IDEAS)
    # Show complete six-round trajectories for anchors and all ten ideas early;
    # do not put fifty-nine six-round controls ahead of the first candidate.
    anchors = [next(c for c in out if c['family'] == family and c['strength'] == strength)
               for family, strength in (('strong', 0.), ('ig_mlp', .8),
                                        ('cfg_native', 1.25), ('ig_native', .8))]
    anchor_ids = {c['arm'] for c in anchors}
    controls = [c for c in out if c['role'] != 'candidate' and c['arm'] not in anchor_ids]
    queues = {}
    for row in IDEAS:
        members = [c for c in out if c['idea_id'] == row['id']]
        center = next(c for c in members if c['theta'] == .8 and
                      c['strength'] == (1.25 if row['source'] == 'cfg' else .8))
        queues[row['id']] = [center]+[c for c in members if c is not center]
    scheduled = list(anchors)
    candidate_count = 0
    for setting in range(20):
        for idea_id in (1, 6, 2, 7, 3, 8, 4, 9, 5, 10):
            scheduled.append(queues[idea_id][setting])
            candidate_count += 1
            if candidate_count % 4 == 0 and controls:
                scheduled.append(controls.pop(0))
    scheduled.extend(controls)
    assert len(scheduled) == len(out) and {c['arm'] for c in scheduled} == {c['arm'] for c in out}
    return scheduled


FAMILIES = {row['key']: row for row in IDEAS}

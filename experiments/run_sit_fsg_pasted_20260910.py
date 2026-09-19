"""Seven pasted hypotheses, ten settings each, with matched controls.

Run --check before freezing; --pipeline waits for the existing 20-idea queue.
"""
from __future__ import annotations

STRENGTHS = (.75, 1.25, 1.75, 2.25, 2.75)
EVENT_STEPS = (8, 24)
FUTURE_STEPS = 16

METHODS = [
    dict(id='A1', key='golden_tube', title='完整未来残差的软持续性约束', values=(.25, 1.),
         parameter='suffix_weight', solver='heun64', external_semantics=False,
         objective='||R_t||^2 + theta*(||R_s1(Phi_u(t,s1)x)||^2+||R_s2(Phi_u(t,s2)x)||^2)',
         prior='20项#18/#19仅相关；无同终点后缀约束实验',
         caveat='严格连续不变性在可逆C1流下推出两场相等；此处仅测试三个离散时刻的软约束。'),
    dict(id='A2', key='full_path', title='完整续生成的多时距轨迹一致性', values=(.25, 1.),
         parameter='intermediate_weight', solver='heun64', external_semantics=False,
         objective='||R_T||^2+theta*sum_{q=.25,.5,.75}||Phi_c(t,s_q)x-Phi_u(t,s_q)x||^2',
         prior='20项#3已测试两个短区间；本次加入真正跑到t=1的轨迹',
         caveat='轨迹接近不保证目标类别、真实感或少曲率；不将残差相关性当作因果证明。'),
    dict(id='A3', key='terminal_inverse', title='冻结条件终点的未来影响匹配', values=(.5, 1.),
         parameter='radius_multiplier', solver='heun64', external_semantics=False,
         objective='min_delta ||Phi_u(t,1)(x+delta)-stopgrad(Phi_c(t,1)x)||^2',
         prior='RAEv2锚定单步/多步逆映射；20项#8是转置梯度而非求逆',
         caveat='这是冻结目标的单边逆问题，不保证同起点R下降；若完全成功，可能只昂贵地复制条件终点。'),
    dict(id='A4', key='stochastic_coupling', title='共享布朗噪声的随机未来一致性', values=(1., 2.),
         parameter='coupled_paths', solver='sde_tail64', external_semantics=False,
         objective='mean_xi ||Phi_c_SDE(t,1)(x;xi)-Phi_u_SDE(t,1)(x;xi)||^2',
         prior='20项#9/#16读取局部导数和协方差；未比较实际随机续生成核',
         caveat='固定共享噪声耦合只给W2平方上界；1或2条轨迹不等于识别了完整分布。'),
    dict(id='B1', key='semantic_agreement', title='分类语义读出的完整未来一致性', values=(1., 2.),
         parameter='semantic_readout', solver='heun64', external_semantics=True,
         objective='||Psi_c(Phi_c(t,1)x)-Psi_c(Phi_u(t,1)x)||^2',
         prior='20项#14仅池化归一化latent，不能称真正语义；本次使用冻结ConvNeXt分类器',
         caveat='使用额外预训练分类器和内部解码；相同的低类别置信度仍是假一致。'),
    dict(id='B2', key='contrastive_hinge', title='目标与竞争类别的未来间隔', values=(.25, 1.),
         parameter='hinge_weight', solver='heun64', external_semantics=False,
         objective='d(U,C)+theta*relu(.25+(d(U,C)-d(U,N))/scale)^2',
         prior='20项#13已有竞争类别距离项；本次为完整未来和有界hinge，FSG附录B已做错配条件诊断',
         caveat='正反事实间隔仍不充分证明目标语义正确；竞争类按冻结embedding余弦选取。'),
    dict(id='B3', key='robust_future', title='扰动邻域下的完整未来一致性', values=(.005, .02),
         parameter='perturbation_rms', solver='heun64', external_semantics=False,
         objective='mean_{sign=+-1} ||R_t(x+sign*theta*(1-t)*xi)||^2',
         prior='20项#4同一思想、短区间版本已跑；本次把future延伸到t=1',
         caveat='邻域内两场一致不代表邻域位于正确语义basin；需独立目标分类验证。'),
]


def configurations():
    rows = []

    def add(family, key, strength, theta=0., role='control', solver='heun64',
            source='cfg', method_id=None, external_semantics=False, parameters=None):
        index = sum(row['family'] == family for row in rows)
        rows.append(dict(arm=f'{family}_{index:02d}', family=family, key=key,
                         strength=float(strength), theta=float(theta), role=role,
                         solver=solver, source=source, cutoff=.5 if source == 'ig' else .75,
                         idea_id=method_id, external_semantics=external_semantics,
                         parameters=parameters or {}))

    add('strong', 'native_ig', 0., role='native', source='ig')
    add('ig_local', 'ig_local_attention', .8, 2., source='ig')
    for a in (.25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.):
        add('cfg_native', 'native_cfg', a, role='native')
    for a in (1., 1.5, 2., 2.5, 3.):
        for beta in (-.75, -.5, -.25):
            add('cfg_apg', 'cfg_apg_momentum', a, beta)
    for a in STRENGTHS:
        add('cfg_sde_native', 'native_sde', a, role='native', solver='sde_tail64')
        for theta in (.5, 1.):
            add('full_root_control', 'full_root', a, theta)
        for h in (.0625, .125):
            add('fsg_operator_control', 'fsg_operator_control', a, h)
        for target in (.5, .8):
            add('semantic_contract_control', 'semantic_contract', a, target,
                external_semantics=True)
    for a in (1.25, 2., 2.75):
        for gain in (.1, .2):
            add('cfg_smc_control', 'smc_control', a, gain, parameters={'lambda': 5.})
    for method in METHODS:
        for a in STRENGTHS:
            for theta in method['values']:
                add(f'{method["id"].lower()}_{method["key"]}', method['key'], a, theta,
                    role='candidate', solver=method['solver'], method_id=method['id'],
                    external_semantics=method['external_semantics'])
    assert len(rows) == 140 and len({row['arm'] for row in rows}) == 140
    assert all(sum(row['idea_id'] == method['id'] for row in rows) == 10 for method in METHODS)
    return rows


if __name__ == '__main__':
    from experiments.sit_fsg_pasted_20260910.pipeline import main
    main()

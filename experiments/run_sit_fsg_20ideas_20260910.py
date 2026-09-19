"""Twenty FSG follow-up hypotheses and their complete, fixed 10-point grids.

Run with --pipeline inside tmux; --status reads the unattended queue.
Published algorithms are controls, never counted among the twenty candidates.
"""
from __future__ import annotations

SOURCES={
 'fsg':'https://arxiv.org/html/2510.21512v1',
 'mp':'https://arxiv.org/html/2601.21892v1',
 'ctrl':'https://arxiv.org/html/2603.03281v2',
 'path':'https://arxiv.org/html/2608.29107v1',
 'fds':'https://arxiv.org/html/2604.04646v1',
 'repr':'https://arxiv.org/html/2601.22468v1',
 'bridge':'https://arxiv.org/html/2606.03119v1',
 'e2po':'https://arxiv.org/html/2605.15803v1',
 'mambo':'https://arxiv.org/html/2508.03442v4',
 'saddle':'https://arxiv.org/html/2511.21863v1',
 'entropy':'https://arxiv.org/html/2602.09651v1',
 'swg':'https://arxiv.org/html/2411.10257v3',
 'apg':'https://arxiv.org/html/2410.02416v2',
}

# Each row changes an explicit objective, constraint, or information hypothesis.
# The shared 2-D finite-difference solver is infrastructure, not an extra idea.
IDEAS=[
 dict(id=1,key='anchor_root',route='A',title='保留条件结局的同起点等式校准',parameter='conditional_anchor',values=(.1,1.),
      equation='min_delta ||R(z+delta)||^2 + lambda ||C_H(z+delta)-C_H(z)||^2',
      hypothesis='条件与无条件接近时，应尽量保留原来的条件结局，避免通过共同漂移获得无意义的小残差。',
      delta='两端均在新状态重新计算，增加条件结局保持项；不同于旧固定目标的单边逆映射。',
      prior='FSG_ANCHORED_INVERSE_PROBE_20260908_ZH.md',refs=('fsg','mp'),
      risk='条件预测本身可能错误；保持项过强也会阻止有益的模式切换。'),
 dict(id=2,key='noise_shell_root',route='A',title='保留种子噪声长度的等式校准',parameter='trust_multiplier',values=(.5,1.),
      equation='min_delta ||R(z+delta)||^2; delta tangent to n=z-t*C; retract n to its original norm',
      hypothesis='写入类别可以主要改变噪声方向；保留当前估计噪声的长度可能减轻径向先验漂移。',
      delta='先由真实条件/null续生成残差选方向，再在信号中心周围重投影；不是对最终guidance作球面缩放。',
      prior='FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md',refs=('fsg','mp'),
      risk='估计noise不是真实独立高斯；保长度不等于保高斯分布或多样性。'),
 dict(id=3,key='horizon_consensus',route='A',title='同时满足近、远两段续生成等式',parameter='long_weight',values=(.25,.75),
      equation='min_delta (1-w)||R_H/2(z+delta)||^2+w||R_H(z+delta)||^2',
      hypothesis='只解一个远期终点可能利用抵消误差；同时约束两个时距能检验写入信息是否在途中也可读。',
      delta='一个状态更新共同降低两个续生成残差；不是增加同一算子的Picard或Anderson次数。',
      prior='FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md',refs=('fsg','path'),
      risk='两个约束可能冲突；短区间仍可能压过有效的远期语义。'),
 dict(id=4,key='neighborhood_root',route='A',title='小扰动邻域内仍成立的等式',parameter='noise_rms',values=(.005,.02),
      equation='min_delta sum_{s in {-1,+1}} ||R(z+delta+s*sigma*(1-t)*xi)||^2/2',
      hypothesis='可读取的类别信息应在小噪声扰动下稳定；只在一个精确latent上相等可能过于脆弱。',
      delta='优化扰动下的条件/null续生成一致性；不同于旧IG对扰动处gap取平均。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','e2po'),
      risk='邻域扰动可能跨语义模式；鲁棒化可能牺牲细节。'),
 dict(id=5,key='patch_bottleneck',route='A',title='优先解决尚未承载条件的空间区域',parameter='softmax_temperature',values=(1.,4.),
      equation='min_delta sum_p softmax(beta*e_p/mean(e))*||R_p(z+delta)||^2',
      hypothesis='全局等式残差可能掩盖少数尚未决定的区域；按基点patch残差加权可把写入预算给这些区域。',
      delta='权重作用于状态搜索的未来等式损失，且每次局部搜索内固定；不是旧patch输出增量裁剪。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','bridge'),
      risk='高残差区域也可能只是噪声或模型误差；没有语义分割监督。'),
 dict(id=6,key='cycle_defect',route='A',title='扣除无条件往返的数值漂移后写入',parameter='defect_subtraction',values=(.5,1.),
      equation='delta=(L_CU(z)-z)-rho*(L_UU(z)-z), followed by actual-R acceptance',
      hypothesis='有用的写入应来自条件差异；纯null离散往返造成的漂移应被独立扣除。',
      delta='旧实验只比较同场往返对照，本项显式扣除同场缺陷并检查同起点残差是否下降。',
      prior='FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md',refs=('fsg',),
      risk='数值缺陷与条件响应并不严格可加；扣除可能删除偶然有益的数值修正。'),
 dict(id=7,key='condition_continuum',route='A',title='类别强度中间点也应读出同一结局',parameter='intermediate_condition',values=(.25,.75),
      equation='min_delta ||C_H-U_H||^2+||M_H-U_H||^2, M uses e_u+r(e_c-e_u)',
      hypothesis='若latent已充分承载条件，弱化外部条件也应保持续生成；仅两个端点相等可能隐藏非线性条件响应。',
      delta='把中间条件作为额外一致性约束，更新latent；不同于旧CFG条件割线直接替换guidance方向。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','e2po'),
      risk='类别embedding插值未必对应有效概率条件，可能错误限制正常的非线性。'),
 dict(id=8,key='endpoint_cotangent',route='A',title='把未来类别证据作为梯度拉回当前状态',parameter='horizon_fraction',values=(.125,.25),
      equation='delta proportional to Q Q^T J(U_H)^T [C(U_H,s)-U(U_H,s)]',
      hypothesis='未来的类别证据是需要经Jacobian转置拉回的协向量；直接搬运未来速度差会忽略表示坐标变化。',
      delta='用有限差分计算未来null流的转置作用在两维搜索子空间中的分量；不同于旧Strong目标的正向响应求逆。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','path'),
      risk='未来CFG gap只在理想模型下与类别对数后验梯度成正比；两维投影会遗漏方向。'),
 dict(id=9,key='covariance_agreement',route='B',title='条件信息的一阶、二阶读出同时一致',parameter='derivative_weight',values=(.1,1.),
      equation='min_delta ||m_c-m_u||^2+lambda*||J(m_c-m_u)xi||^2',
      hypothesis='两个预测均值相同仍可能有不同的不确定性；读取实际输入Jacobian以检验更强的条件承载。',
      delta='从真实网络扰动查询估计方向导数，而不从两个均值臆造协方差；旧IG协方差来自训练误差缓存。',
      prior='IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md',refs=('fsg','fds','entropy'),
      risk='一个方向探针不能识别完整后验；Tweedie协方差解释要求理想线性高斯加噪模型且t>0。'),
 dict(id=10,key='shared_ambiguity',route='B',title='从等式成立但共同模糊的状态中脱离',parameter='ambiguity_weight',values=(.1,1.),
      equation='choose delta in {0,+r*q,-r*q} minimizing ||R||^2/||R0||^2 + lambda*conditional_divergence',
      hypothesis='条件/null一起落在模糊区域也会接近；一致性需要与条件分支自身的歧义度共同检查。',
      delta='将FDS的已知散度信号用于FSG候选状态的联合选择；不是把FDS本身算作新方法。',
      prior='IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md',refs=('fsg','fds','saddle'),
      risk='降低局部条件散度不保证FID；随机迹估计有方差，可能偏好收缩。'),
 dict(id=11,key='renoise_readability',route='B',title='重新加噪后无条件分支仍能读取预测内容',parameter='future_fraction',values=(.25,.5),
      equation='min_delta ||m_u(s*m_c(z+delta)+(1-s)*xi,s)-m_c(z)||^2',
      hypothesis='稳定的条件内容应进入预测信号，而不只依附于当前特定noise排列；重新加噪提供独立读取测试。',
      delta='优化新noise下null读取的固定条件目标；不同于旧IG保持noise长度的旋转平均。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','bridge','repr'),
      risk='单次再加噪会改变实例细节；强制保持全部latent内容可能过强。'),
 dict(id=12,key='conditional_idempotence',route='B',title='条件预测应是稳定的自重建对象',parameter='future_fraction',values=(.25,.5),
      equation='min_delta ||m_c(s*m_c(z+delta)+(1-s)*n_c(z+delta),s)-m_c(z+delta)||^2',
      hypothesis='即使无需与null完全相同，条件预测也应能沿自身signal/noise分解被再次识别。',
      delta='把条件分支自己的重建循环设为latent搜索目标；不同于旧直接外推未来IG gap。',
      prior='IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md',refs=('repr','mp'),
      risk='稳定的错误图像也可能满足幂等性；目标本身不能辨别真实图像与模型伪固定点。'),
 dict(id=13,key='rival_margin',route='B',title='写入目标类别时同时排除最近的竞争类别',parameter='margin_weight',values=(.25,1.),
      equation='choose delta minimizing ||U_H-C_H||^2 - lambda*||U_H-V_rival,H||^2',
      hypothesis='目标/null接近可能仍处于细类别边界；与竞争类别的距离可检验写入是否具有辨别性。',
      delta='每类使用冻结embedding余弦最近的另一类，比较未来margin；不同于旧随机类别均值负参考。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','entropy'),
      risk='embedding最近不一定语义最近；latent距离是margin代理，不是校准类别概率。'),
 dict(id=14,key='content_quotient',route='B',title='只要求内容读出一致，保留通道外观自由度',parameter='pool_size',values=(4.,8.),
      equation='min_delta ||P(normalize_channels(C_H))-P(normalize_channels(U_H))||^2',
      hypothesis='像素级等式可能不必要地约束色调和纹理；去通道均值/尺度后的粗空间结构可能足够承载内容。',
      delta='更改等式的输出空间，再搜索latent；没有修改最终图像方差，也没有使用外部表示模型。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','repr'),
      risk='粗空间归一化不等于语义表示，可能忽略关键颜色条件或放过伪一致。'),
 dict(id=15,key='orbit_agreement',route='B',title='允许两分支速度不同但走向同一条轨道',parameter='phase_penalty',values=(.1,1.),
      equation='min_delta min_tau ||C_H-U_H-tau*U_end||^2+lambda*tau^2*||U_end||^2',
      hypothesis='一部分条件/null差异可能仅是沿相同轨道的进度差；不必把这部分解释成未写入的内容。',
      delta='在未来null切向方向消去有惩罚的时间相位，再更新latent；不同于APG对当前clean径向投影。',
      prior='FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md',refs=('fsg','ctrl'),
      risk='一阶时间相位仅局部有效，也可能错误删除真正的类别方向。'),
 dict(id=16,key='information_volume',route='B',title='用条件相对不确定性体积识别有信息的状态',parameter='covariance_ridge',values=(.1,1.),
      equation='choose delta minimizing logdet(Sigma_c^Q+ridge*I)-logdet(Sigma_u^Q+ridge*I)',
      hypothesis='条件信息可以体现在后验体积缩小，未必要求预测均值完全相等；检验信息增益而非均值匹配。',
      delta='两维真实Jacobian读出构造受限协方差代理；不是旧固定均值的全局方差最优化。',
      prior='RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md',refs=('entropy','fds'),
      risk='仅为两个方向上的高斯代理；非对称/负特征值需投影并记录，不能当作精确互信息。'),
 dict(id=17,key='layer_interaction',route='B',title='移除前后层重复注入条件产生的非线性交互',parameter='interaction_subtraction',values=(.5,1.),
      equation='I=V_all-V_early-V_late+V_null; v=C+a*(D-rho*I)',
      hypothesis='类别信息若已进入前半网络，后半再次注入可能产生过强交互；二因素差分隔离这部分响应。',
      delta='在同一个Full模型中干预前4层与后续层的类别嵌入；不同于旧强弱深度与条件的gap相减。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('fsg','e2po'),
      risk='分层混用类别/null未在训练中出现；交互也可能正是组合语义所需。'),
 dict(id=18,key='transported_memory',route='B',title='只补写当前状态中尚未保留的类别方向',parameter='redundancy_subtraction',values=(.5,1.),
      equation='carry previous write through J(Phi_u); v=C+a*(D-rho*projection(D,memory))',
      hypothesis='已写入的类别方向若能被null自然搬运，后续重复强化可能浪费预算并放大伪影。',
      delta='记忆保存累计实际guidance写入，并通过null有限差分运输；不是旧gap EMA或双轨影子状态。',
      prior='IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md',refs=('fsg','ctrl'),
      risk='一个累计向量不能代表完整语义记忆，方向运输的线性近似可能失真。'),
 dict(id=19,key='commitment_latch',route='B',title='可恢复的条件写入结束判据',parameter='agreement_threshold',values=(.05,.15),
      equation='disable extra CFG after two R_H/(H*||C||) tests below tau; re-enable above 2*tau',
      hypothesis='一旦未来续生成已连续两次接近，就可以停止额外写入；遗忘时应重新开启，而非固定时间永久撤除。',
      delta='未来一致性、连续两次确认和迟滞共同控制额外CFG；仍保持条件主场，不称永久null接管。',
      prior='FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md',refs=('fsg','ctrl','entropy'),
      risk='低残差可能是共同模糊；关闭的是额外CFG，不能证明不再需要外部条件。'),
 dict(id=20,key='spatial_condition_interaction',route='B',title='突出类别与空间组织之间的交互信息',parameter='interaction_mix',values=(.25,.75),
      equation='D_bind=D(z)-T^-1 D(Tz); v=C+a*((1-rho)*D+rho*D_bind)',
      hypothesis='类别信息的一部分依赖长程空间绑定；局部统计保留但空间关系破坏后的条件差可用于隔离它。',
      delta='对输入4x4 latent块作固定per-image置换，对条件/null做双差；不同于旧弱头value置换或局部注意力参考。',
      prior='SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md',refs=('swg','fsg'),
      risk='置换状态偏离训练分布；隔离出的交互可能是模型对破坏输入的任意响应。'),
]

STRENGTHS=(.75,1.25,1.75,2.25,2.75)

def configurations():
    output=[]
    def add(family,key,source,strength,theta=0.,role='control',solver='heun64',idea_id=None):
        index=sum(c['family']==family for c in output)
        output.append(dict(arm=f'{family}_{index:02d}',family=family,key=key,source=source,
            strength=float(strength),theta=float(theta),solver=solver,
            cutoff=.5 if source=='ig' else .75,role=role,idea_id=idea_id,parameters={}))
    add('strong','native_ig','ig',0.,role='native')
    for a in (.5,.6,.7,.8,.9,1.):add('ig_native','native_ig','ig',a,role='native')
    add('ig_dopri','native_ig','ig',.7,role='native',solver='dopri5')
    add('ig_local','ig_local_attention','ig',.8,2.)
    for a in (.25,.5,.75,1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.):add('cfg_native','native_cfg','cfg',a,role='native')
    for a in (1.,1.5,2.,2.5,3.):
        for beta in (-.75,-.5,-.25):add('cfg_apg','cfg_apg_momentum','cfg',a,beta)
    for a in STRENGTHS:
        for h in (.0625,.125):add('fsg_operator_control','fsg_operator_control','cfg',a,h)
    for a in STRENGTHS:
        for r in (.5,1.):add('plain_root_control','plain_root_control','cfg',a,r)
    for idea in IDEAS:
        for a in STRENGTHS:
            for value in idea['values']:
                add(f'f{idea["id"]:02d}_{idea["key"]}',idea['key'],'cfg',a,value,role='candidate',idea_id=idea['id'])
    assert len(IDEAS)==20 and len(output)==256 and len({c['arm'] for c in output})==256
    assert all(sum(c['idea_id']==i['id'] for c in output)==10 for i in IDEAS)
    return output

if __name__=='__main__':
    from experiments.sit_fsg_followup_20260910.pipeline import main
    main()

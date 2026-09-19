"""Fifty hypotheses plus three polished priorities, twelve settings each, 1K FID.

This catalog records testable mechanisms, including explicit revisits of earlier
hypotheses and adaptations of known algorithms. It does not count them as fifty
novel research contributions. All model weights stay frozen.
"""
from __future__ import annotations

STRENGTHS = (.5, 1.25, 2., 2.75)
FUTURE_STEPS = 8
EVENT_STEPS = (8, 24)
EXTENDED_EVENTS = (8, 16, 24, 32, 40)
NOISE_SEED = 202610100

SOURCES = {
    'fsg': ('FSG, NeurIPS 2025', 'https://arxiv.org/html/2510.21512v1'),
    'ctrl': ('CFG-Ctrl, CVPR 2026', 'https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_CFG-Ctrl_Control-Based_Classifier-Free_Diffusion_Guidance_CVPR_2026_paper.pdf'),
    'fm': ('Flow Matching, ICLR 2023', 'https://arxiv.org/html/2210.02747v2'),
    'sde': ('Score SDE, ICLR 2021', 'https://arxiv.org/html/2011.13456v2'),
    'adjoint': ('AdjointDPM, ICLR 2024', 'https://arxiv.org/html/2307.10711v3'),
    'classifier': ('Classifier guidance, NeurIPS 2021', 'https://arxiv.org/html/2105.05233v4'),
    'clf': ('Ames et al., CLF/CBF QP', 'https://arxiv.org/html/1609.06408v2'),
    'rls': ('Goel et al., RLS with forgetting', 'https://arxiv.org/abs/2003.03523'),
    'anderson': ('Walker and Ni, Anderson acceleration', 'https://doi.org/10.1137/10078356X'),
    'deft': ('DEFT, NeurIPS 2024', 'https://arxiv.org/html/2406.01781v3'),
    'apg': ('Adaptive Projected Guidance', 'https://arxiv.org/html/2410.02416v2'),
}

IDEAS = []


def idea(key, title, group, parameter, values, objective, reasoning, prior, difference,
         risk, refs=('fsg',), *, semantic=False, sde=False, extended=False):
    IDEAS.append(dict(id=len(IDEAS)+1, key=key, title=title, group=group,
        parameter=parameter, values=tuple(float(v) for v in values), objective=objective,
        reasoning=reasoning, prior=prior, difference=difference, risk=risk, references=refs,
        external_semantics=semantic, solver='sde_tail64' if sde else 'heun64',
        events=EXTENDED_EVENTS if extended else EVENT_STEPS))


idea('terminal_root', '完整终点等式的直接反馈', 'A 未来等式',
     '写入半径倍率', (.25, .5, 1.),
     'min ||C_t(x+delta)-U_t(x+delta)||^2',
     '把完整续生成的差作为输出，用实际有限差分识别输入响应，再执行有界状态校正。若局部CFG差只是远期误差的粗代理，直接减少远期差仍可能改善有限模型采样。',
     '七方法 full_root 对照；旧二十项 plain_root 为短区间。',
     '保留核心猜想作为明确复验；本轮8步未来、扩大半径网格，并与16步完整根对照比较。',
     '共同错误和终点时距退化仍可能存在；以FID决定该近似是否有用。')
idea('rolling_root', '固定长度滚动前瞻等式', 'A 未来等式',
     '固定时间长度H', (.125, .25, .5),
     'min ||Phi_c(t,t+H)(x+delta)-Phi_u(t,t+H)(x+delta)||^2',
     '完整残差随剩余时间自然收缩；固定物理前瞻长度让不同事件的输出具有更接近的比较尺度，同时检验语义信息所需的有效时距。',
     '旧二十项采用剩余时间比例的短前瞻。',
     'H为固定时间长度，在每个事件滚动终点；并非只给同一个损失乘常数。',
     '固定H可能越过最相关的语义形成阶段；末端截断时仍会退化。')
idea('path_energy', '完整路径上的多时距一致性', 'A 未来等式',
     '路径项权重', (.25, 1., 4.),
     'min ||R_T||^2 + theta*mean_{q=.25,.5,.75}||R_q||^2',
     '终点相等可能利用中途抵消。联合限制三个中间分歧，可能使写入更均匀、减轻后续采样对粗校准的放大。',
     '七方法A2；旧二十项#3短时距。',
     '保留原猜想，使用三个权重和统一新bank；检查更强路径约束是否存在适度最优点。',
     '两条共同错误路径也能一致；过强约束可能消除有益的条件分支自由度。')
idea('golden_tube', '无条件推进后的黄金管约束', 'A 未来等式',
     '后缀一致性权重', (.25, 1., 4.),
     'min ||R_t||^2 + theta*mean_s ||C_s(U_{t:s}x)-U_t(x)||^2',
     '即使残差反弹不等于原终点遗忘，重新查询条件分支时仍出现分歧，可能说明状态依赖的分支兼容性不足；离散软管可以作为改善稳健性的正则。',
     '七方法A1；960状态诊断未发现大于10%的明显反弹。',
     '不因该诊断淘汰；扩大软管权重，并直接比较完整FID。',
     '严格连续可逆极限会回到速度相等；是否有用取决于有限精度下的偏置。')
idea('terminal_tangent', '终点误差与未来切向联合控制', 'A 未来等式',
     '未来切向权重', (.1, .5, 2.),
     'min ||R_t||^2 + theta*||(1-t) J_C(v_c-v_u)||^2',
     'R小而其未来变化率大时，单次校正可能位于敏感位置；同时约束终点误差和未来切向，可能减少后续放大。实际状态更新仍识别整个联合输出的响应。',
     '第二份分析的J_C gap猜想；旧#8使用J_U转置的证据拉回。',
     '把向量推进J_C gap作为真实有限差分输出，而非把J_C误当控制输入矩阵。',
     '完整J可逆时不改变精确零集合，但加权、有限精度和两项平衡可能改变实际结果。')
idea('rolling_invariance', '滚动前瞻误差的物质导数', 'A 未来等式',
     '导数项权重', (.1, .5, 2.),
     'min ||R_H||^2 + theta*||H D_u R_H||^2, H=.25',
     '滚动终点会带来额外边界项。用实际前进状态和时间的有限差分读取D_u R_H，检验稳定一个可比较的滚动输出是否优于只减小当前差。',
     '第二份分析中的Golden invariance；此前未实现完整滚动边界项。',
     '查询(t+eps,x+eps*v_u)，并把未来终点同步移动；包含移动边界作用。',
     '物质导数来自指定被动流，不能单独证明受控轨迹稳定；联合输出由实际输入差分校正。')
idea('terminal_inverse', '冻结条件终点的单边影响匹配', 'A 未来等式',
     '写入半径倍率', (.25, .5, 1.),
     'min ||U_t(x+delta)-stopgrad(C_t(x))||^2',
     '如果条件终点可信，允许两分支一起移动可能浪费好目标。固定目标后只修改null未来，直接检验状态能否承接已知条件结局。',
     '七方法A3和RAEv2锚定逆映射。',
     '保留并扩大半径范围；和同起点根、语义目标分别比较。',
     '条件目标可能错误，也可能用昂贵计算复制普通条件生成；8步逆响应有离散误差。')
idea('conditional_anchor', '同起点根与条件结局保持', 'A 未来等式',
     '条件锚定权重', (.1, 1., 10.),
     'min ||R_t(x+delta)||^2 + theta*||C_t(x+delta)-C_t(x)||^2',
     '既保留同起点一致性，又惩罚条件分支随校准共同漂移。中间权重允许必要修正，可能比完全冻结或完全放任两端更合适。',
     '旧二十项#1采用短未来。',
     '真实完整终点上的折中，检验过去短区间负结果是否来自目标近视。',
     '过大的保持项会让状态几乎不动；错误教师也会被保留。')
idea('renoise_contract', '重加噪视图的终点承接', 'A 未来等式',
     '重建视图时间比例', (.25, .5, .75),
     'min ||U_s(s*m_c(x)+(1-s)*xi)-stopgrad(C_t(x0))||^2',
     '若状态中的条件信息来自可读的信号估计，将该信号放到新的噪声视图后仍应帮助null到达同一条件结局。固定xi使比较只来自候选状态变化。',
     '旧二十项#11的单步重加噪可读性。',
     '用完整null终点对冻结完整条件目标；不是只比较两个单步clean预测。',
     '重加噪会损失实例细节，单个固定视图也可能被过拟合。')
idea('condition_continuum', '条件强度连续读出的一致性', 'A 未来等式',
     '条件embedding插值lambda', (.25, .5, .75),
     'min ||C_t-U_t||^2 + ||C_t^lambda-U_t||^2',
     '只在全条件和null两个端点重合，可能仍隐藏中间条件读出的不稳定；加入一个中间embedding，检验信息写入是否跨条件读出强度兼容。',
     '旧二十项#7短区间版本。',
     '三个lambda分别作用于完整未来，实际修改embedding后恢复全部hooks。',
     '插值embedding未必对应训练过的语义，可能增加不必要约束。')

idea('semantic_bernoulli', '目标语义一致性与latent弱约束', 'B 语义与几何',
     'latent一致性权重', (0., .1, 1.),
     'min Hellinger(q_c(C),q_c(U))^2 + theta*||R/scale||^2',
     '类别成功与失败构成二元语义读出，可以放松不相关像素差；少量latent约束可能防止两个分支用异常图片获得相同置信度。',
     '七方法B1的目标二元读出。',
     '检验语义目标与完整latent目标的连续折中，而非只替换一种norm。',
     '两个低置信度分支仍会一致；分类器是额外信息源。', ('fsg','classifier'), semantic=True)
idea('semantic_distribution', '全类别软语义分布一致性', 'B 语义与几何',
     '概率温度', (.5, 1., 2.),
     'min ||sqrt(softmax(log p100(C)/theta))-sqrt(softmax(log p100(U)/theta))||^2',
     '二元目标概率丢掉竞争类别结构。完整100类分布可以约束语义歧义的形状，温度控制是否强调最强竞争模式。',
     '七方法B1的100类Hellinger版本。',
     '将温度作为明确结构参数，比较硬类别和软竞争分布的作用。',
     '分类器类别空间不能完整描述纹理与关系属性；可能共同偏向错误类。', ('classifier','fsg'), semantic=True)
idea('semantic_margin', '固定目标对最强竞争类的终点间隔', 'B 语义与几何',
     '目标log概率间隔', (.5, 1., 2.),
     'min relu(theta-log p_c(U)+max_{j!=c}log p_j(U))^2',
     '目标概率上升未必超过最强竞争者。直接要求固定类别间隔，把控制输出与最终类别判别连接，并在达标后停止额外写入。',
     '此前只有目标置信度阈值对照。',
     '目标为终点最强竞争者的log概率差，不是C/U相等，也不是冻结embedding邻近类距离。',
     '优化分类器间隔可能损害感知质量；需由FID选择适度间隔。', ('classifier','clf'), semantic=True)
idea('semantic_teacher', '冻结条件语义教师与目标类别混合', 'B 语义与几何',
     '目标one-hot混合权重', (0., .5, 1.),
     'min ||sqrt(p100(U))-sqrt((1-theta)*stopgrad(p100(C0))+theta*onehot(c))||^2',
     '条件教师可能提供有用的竞争类结构，却也可能不够自信；在冻结软教师与明确类别目标之间插值，测试承接语义而不锁定完整图片。',
     '单边终点逆映射、语义一致性和直接分类器控制已有。',
     '冻结的是语义分布，加入可调的类别纠正；避免移动教师。',
     'theta=1偏向分类器极值，theta=0可能复制错误教师。', ('adjoint','classifier'), semantic=True)
idea('contrastive_future', '单个竞争未来的有界间隔', 'B 语义与几何',
     '竞争hinge权重', (.1, .5, 2.),
     'min d(U,C)+theta*relu(.25+d(U,C)-d(U,N))^2',
     '在减少正对残差的同时，要求null未来离目标条件比离竞争条件更近；可能排除一部分对所有条件都不敏感的状态。',
     '七方法B2；旧二十项#13及FSG错配条件分析。',
     '保留有界hinge形式，扩大竞争权重；由FID判断正间隔是否有实际收益。',
     '正间隔不保证正确类别，但可以作为有用的软选择偏置。')
idea('multi_rival_future', '多个竞争未来中的最困难间隔', 'B 语义与几何',
     '间隔阈值', (.1, .5, 1.),
     'min d(U,C)+relu(theta+d(U,C)-min_{j in top3}d(U,N_j))^2',
     '单个负类可能过于容易。选择embedding最接近的三个负类并使用最困难者，可以检验局部语义区分是否更可靠。',
     '已有单竞争类未来间隔。',
     '竞争者集合固定在当前label，候选状态上重新生成三个真实未来。',
     'embedding邻近并不等于视觉混淆；额外条件未来增加成本。')
idea('whitened_root', '条件终点协方差度量中的一致性', 'B 语义与几何',
     '协方差ridge', (.01, .1, 1.),
     'min ||Sigma_C^(-1/2)(C_t(x+delta)-U_t(x+delta))||^2',
     'latent通道的幅度和相关性不同，欧氏残差可能被高方差通道主导。冻结条件终点通道协方差，检验对较小但结构性偏差的校正。',
     '旧50项有channel metric，但作用于guidance方向。',
     '度量作用于完整未来控制输出，输入响应也在同一度量下识别。',
     '通道白化不等于语义白化；小特征值可能放大噪声，需要ridge。')
idea('coarse_detail_root', '粗结构优先的未来等式', 'B 语义与几何',
     '高频残差权重', (0., .25, 1.),
     'min ||lowpass(R)||^2 + theta*||R-lowpass(R)||^2',
     '类别轮廓可能先于细节可确定；允许高频未来不同，可能减少过度约束纹理带来的分布压缩。',
     '旧50项对guidance频率处理；旧#14池化latent代理。',
     '控制的是完整终点的频带差，并明确不把低频直接称为语义。',
     '类别也可能依赖细纹理，高频放松可能损害条件辨识。')
idea('patch_bottleneck', '最大局部分歧驱动的未来反馈', 'B 语义与几何',
     'patch权重温度', (.5, 2., 8.),
     'min sum_patch softmax(theta*E_patch(R0))*||R_patch||^2',
     '少数对象区域的分歧可能被背景平均掩盖。对基点分歧大的patch提高权重，检验局部瓶颈是否主导全图质量。',
     '旧二十项#5采用短区间。',
     '以完整未来残差冻结patch权重，候选求解不能通过改权重作弊。',
     '最大分歧也可能是无害纹理，过集中会引入局部伪影。')
idea('noise_tangent_root', '保噪声半径的未来状态写入', 'B 语义与几何',
     '写入半径倍率', (.25, .5, 1.),
     'min ||R_t||^2 in tangent(n_c); retract estimated noise to original norm',
     '将校正限制在估计噪声球面的切向并重投影，检验保留种子尺度是否能降低根求解造成的径向先验偏移。',
     '旧二十项#2短区间noise-shell。',
     '完整未来输出选方向，实际重投影后再计算非线性目标并接受。',
     '保长度不等于保高斯分布，估计noise也不是真实隐变量。')

idea('antithetic_root', '对称邻域的完整未来一致性', 'C 稳健与随机',
     '扰动RMS系数', (.003, .01, .03),
     'min mean_{s=+-1}||R_t(x+s*theta*(1-t)*xi)||^2',
     '单点一致性可能依赖脆弱状态。正负相同扰动的平均目标消除部分一阶噪声，检验附近状态是否同样兼容两条未来。',
     '七方法B3和旧二十项#4。',
     '保留完整未来猜想，新增更小扰动档以测试是否此前正则过强。',
     '两分支可在整个邻域共同错误；效果由FID判断。')
idea('worst_neighbor', '最坏邻域未来残差控制', 'C 稳健与随机',
     '扰动RMS系数', (.003, .01, .03),
     'min max_{s=0,+1,-1}||R_t(x+s*theta*(1-t)*xi)||^2',
     '平均稳健性可能掩盖一侧特别脆弱的状态；最坏邻居目标更关注局部不稳定方向，可能减少少数严重失败。',
     '已有邻域平均一致性。',
     '每次候选重新选最坏分支，目标为有限场景鲁棒控制。',
     'max不光滑会使局部线性模型失准，也可能过于保守。')
idea('coupled_sde', '共享随机未来的耦合一致性', 'C 稳健与随机',
     'Monte Carlo路径数', (1, 2, 4),
     'min mean_k ||C_SDE(x;xi_k)-U_SDE(x;xi_k)||^2',
     '真实采样有随机未来时，单个确定终点并不足够。增加固定共享布朗路径，检验跨未来随机性的相容性是否比点映射更有效。',
     '七方法A4已有1/2路径。',
     '保留真实SDE尾部，增加4路径档，和同SDE原生/语义对照比较。',
     '固定耦合成本不是完整分布距离；路径数有限且计算昂贵。', ('sde','fsg'), sde=True)
idea('sde_moments', '随机未来的均值与方差一致性', 'C 稳健与随机',
     '方差项权重', (0., .25, 1.),
     'min ||mean(C)-mean(U)||^2 + theta*||std(C)-std(U)||^2, K=2',
     '逐噪声配对可能限制了不必要的随机数对应关系；比较均值和逐坐标方差，允许不同路径对应却具有相似未来不确定性。',
     '旧#9/#16局部后验协方差代理；七方法A4配对终点。',
     '对真实SDE终点估计矩，未把均值预测冒称协方差。',
     '两条路径的方差估计很粗，只覆盖低阶矩。', ('sde','deft'), sde=True)
idea('sliced_sde', '随机未来的一维投影分布匹配', 'C 稳健与随机',
     '固定投影方向数', (4, 8, 16),
     'min mean_q ||sort(<q,C_k>)-sort(<q,U_k>)||^2, K=4',
     '在投影方向上对经验分布排序匹配，可以放松共享噪声的一一对应；检验比均值方差更丰富的分布约束。',
     '此前只有固定耦合和局部协方差。',
     '四条真实未来、多个固定随机投影；是一维经验运输代价，不宣称等于完整高维W2。',
     '小样本和少数投影可能漏掉语义差异。', ('sde','deft'), sde=True)
idea('sde_semantic_success', '随机未来的平均目标成功读出', 'C 稳健与随机',
     '平均目标置信度阈值', (.4, .6, .8),
     'min relu(theta-mean_{k=1,2} q_c(U_SDE(x;xi_k)))^2',
     '直接控制停止guidance后的随机未来是否支持目标类别；两个未来的平均置信度比单个确定终点更接近剩余成功率的软代理。',
     'Doob/committor理论；现有确定终点置信度对照。',
     '实际SDE Monte Carlo加现有冻结分类器，不声称分类器置信度就是真实committor。',
     '概率校准与两路径估计误差都可能较大；分类器引入额外信息。', ('deft','classifier'), semantic=True, sde=True)
idea('robust_semantic_success', '最坏扰动下的固定语义要求', 'C 稳健与随机',
     '扰动RMS系数', (.003, .01, .03),
     'min max_{s=+-1} relu(.6-q_c(U_t(x+s*theta*(1-t)*xi)))^2',
     '稳健地让两分支相等仍可能共同错误；把邻域要求直接施加于目标类别读出，检验语义边界余量是否有用。',
     '七方法B3只要求两分支一致。',
     '完整null未来的最坏目标置信度，允许条件/null终点不同。',
     '可能牺牲多样性换取分类鲁棒性；需要FID检验。', ('classifier','clf'), semantic=True)
idea('bayes_calibration', 'Bayes条件差的独立读出校准代理', 'C 稳健与随机',
     'clean读出混合权重', (0., .5, 1.),
     'min ||Q^T(v_c-v_u)-kappa*D_Q log q_proxy(c|x,t)||^2',
     '标准Gaussian路径中真实条件速度差与含噪类别后验梯度成比例。现有clean分类器提供一个可计算的独立语义代理，检验它与模型条件差的局部不一致是否能指导有用校正。',
     '第二份分析的Bayes差分校准等式，尚无真实含噪分类器。',
     'q_proxy为条件/null单步clean读出的概率混合，明确近似；真实查询两个方向导数，不把gap本身当模型误差。',
     '代理不等于训练含噪后验，残差不能被解释成精确容量误差；仍保留为实验idea。', ('fm','classifier'), semantic=True)
idea('readout_transport', '局部语义读出的时间输运一致性', 'C 稳健与随机',
     '输运一致性权重', (.1, .5, 2.),
     'min relu(.6-q_c(U_t(x)))^2 + theta*(q_proxy(U_short(x),s)-q_proxy(x,t))^2',
     '完整null终点语义具有被动输运不变性，便宜的单步读出却可能随时间改口。联合终点目标和近似读出一致性，检验减少这种改口是否帮助条件承载。',
     '重加噪一致性与完整终点语义目标已有，未联合时间读出。',
     '输运的是单步clean分类代理，避免对精确半群恒等式重复优化。',
     '近似读出的合理变化也会被惩罚，可能妨碍模式选择。', ('fm','classifier'), semantic=True)
idea('refinement_consistency', '粗细未来网格的一致性校准', 'C 稳健与随机',
     '网格差异权重', (.25, 1., 4.),
     'min ||R_8||^2 + theta*||R_16-R_8||^2',
     '粗未来目标可能通过数值误差被优化得很好。显式压低8步与16步残差差异，检验跨求解分辨率的可靠性是否提高主轨迹质量。',
     'RAEv2逆映射细化后误差回升；七方法采用更细网格诊断。',
     '把数值细化差作为实际优化输出，而非仅在结果后报告。',
     '两种网格仍可能共有偏差，额外计算是否值得需比较实际成本。')

idea('root_cotangent', '完整未来残差的转置梯度反馈', 'D 自适应控制',
     '写入半径倍率', (.25, .5, 1.),
     'delta proportional to -Q J_Q^T R, bounded and accepted on actual loss',
     '求逆可能放大小响应方向中的噪声。转置反馈优先沿有可靠输出敏感性的方向下降，检验对有限模型更稳健的方向选择。',
     '旧#8转置未来类别证据；已有Gauss-Newton根。',
     '转置的是完整根残差对输入的真实响应，与未来向量J_C gap不同。',
     '可能收敛慢，梯度也不保证FID改善。', ('fsg','adjoint'))
idea('root_sliding', '带边界层的输出滑模反馈', 'D 自适应控制',
     '归一化边界层宽度', (.03, .1, .3),
     'delta proportional to -Q J_Q^T tanh(R/epsilon)',
     '符号型输出反馈减少大残差分量对更新的支配；经输入响应转置映回latent，保留控制方向信息，并用边界层减少抖振。',
     'CFG-Ctrl已有滑模；此前只有作者原始离散算法对照。',
     '滑模面是完整未来输出，反馈经真实输入响应映射；不是直接减原始gap的sign。',
     '有限差分响应不准时方向仍可能失败，最终用实际损失和FID检验。', ('ctrl','clf'))
idea('root_integral', '完整未来输出的积分反馈', 'D 自适应控制',
     '积分增益', (.25, 1., 4.),
     'I <- .8 I + dt*R; solve J_Q a = -(R+theta I)',
     '单次校准可能留下同方向持久误差。固定终点坐标中的有泄漏积分保留此前未消除的分量，检验是否能克服有限半径造成的持续偏差。',
     'CFG-Ctrl局部PD；旧记忆搬运针对guidance而非完整输出。',
     '跨五个事件积累完整终点残差，输入方向仍每次实际识别。',
     'R本身随时间变化，积分可能过时；泄漏与实际接受限制累积过冲。', ('ctrl','clf'), extended=True)
idea('root_derivative', '完整未来输出的滤波微分反馈', 'D 自适应控制',
     '微分时间系数', (.02, .1, .5),
     'D <- .5 D + .5*(R-R_prev)/dt; solve J_Q a = -(R+theta D)',
     '快速增长的未来误差可能值得提前抑制。实际时间间隔归一化和低通滤波，检验预测误差趋势是否比只看当前值更有用。',
     'CFG-Ctrl在局部gap上已有差分控制。',
     '使用固定终点残差、实际dt和多事件更新，不将作者未除dt的算法冒称同一个控制器。',
     '自然时距收缩也影响D，噪声可被放大；这是需实验判断的偏置。', ('ctrl','clf'), extended=True)
idea('adaptive_trust', '实际下降比驱动的自适应信赖半径', 'D 自适应控制',
     '目标实际下降比例', (.05, .2, .5),
     'radius adapts from actual/predicted reduction of the full-future residual',
     '固定写入半径忽略局部未来模型的可信度。根据预测下降是否兑现增减下一次半径，可能减少高噪阶段错误的大更新。',
     '已有固定半径与实际接受；没有跨事件下降比控制。',
     '保留相同输出，显式记录模型预测、实际下降和有界半径状态。',
     '局部目标可能可靠但与FID无关；半径适应也可能过慢。', ('clf','fsg'), extended=True)
idea('recursive_semantic_response', '可观察语义响应的在线递推辨识', 'D 自适应控制',
     '遗忘系数', (.5, .8, .95),
     'K=P a/(theta+a^T P a); M <- M+(Delta y-M a)K^T; P <- (P-K a^T P)/theta; fixed Q per trajectory',
     '理想生成场误差不可直接观测，但一次实际写入引起的终点语义变化可以测量。保留固定输入坐标并递推响应，可在更少新探针下改进后续写入方向。',
     '第二份分析的可观察输入响应idea；RLS是已知辨识工具。',
     '五个事件共享固定二维Q，每次刷新一个方向并以实际输出差作递推更新；控制固定目标概率。',
     '响应随t和状态变化，缺乏激励会导致估计偏差；不声称辨识模型容量误差。', ('rls','classifier'), semantic=True, extended=True)
idea('output_disturbance', '非线性输出缺陷的扰动补偿', 'D 自适应控制',
     '缺陷补偿增益', (.25, .5, .9),
     'd_hat = R(x+delta)-[R(x)+J delta]; second correction targets R+theta*d_hat',
     '局部线性响应未解释的输出变化可直接观测。用第一次候选的非线性缺陷修正第二次写入，检验低阶扰动观测能否改善有限差分控制。',
     '已有单次Gauss-Newton与拒绝机制。',
     '同一时刻、同一输出上做两次候选，缺陷来自真实查询而非假定理想速度。',
     '局部缺陷可能不持续，补偿可能加剧过冲；两次实际结果均保留验收。', ('clf','fsg'))
idea('bounded_deadzone', '有界误差容忍的输出死区', 'D 自适应控制',
     '相对死区宽度', (.05, .15, .3),
     'min ||sign(R)*relu(abs(R)-theta*scale0)||^2',
     '模型和数值输出有误差时，追逐每个小残差可能浪费预算并过度约束。死区保留明显差异，允许不确定的小误差。',
     '已有精确平方根目标，未系统扫输出容忍区。',
     '死区作用于真实未来残差分量，尺度在当前基点冻结。',
     '小幅但语义关键的残差也可能被忽略；并非已证明的误差界。', ('clf','fsg'))
idea('performance_funnel', '超越时距收缩的预设误差漏斗', 'D 自适应控制',
     '漏斗时间幂指数', (1.25, 2., 3.),
     'rho_t=scale_first*((1-t)/(1-t_first))^theta; penalize abs(R)>rho_t',
     '普通完整残差可随剩余时间自然变小；要求比线性时距更快进入收缩漏斗，测试是否能推动真正的状态校正。',
     '第二份分析指出终点退化；尚未测试时间漏斗。',
     '阈值由首次事件固定，跨五个事件变化，不是每次把当前损失重新归一化为零。',
     '漏斗可能不可达或过强，不能据其满足与否代替生成质量。', ('clf','ctrl'), extended=True)
idea('semantic_guard_qp', '保条件质量的最小语义干预', 'D 自适应控制',
     '允许条件概率下降', (0., .02, .05),
     'min ||a||^2 subject to J_qU a >= .2*(.6-qU), J_qC a >= -theta',
     '提高null目标概率时，不希望同时明显损害条件分支的目标读出。局部两约束最小范数解提供可解释的干预折中。',
     'CLF/CBF QP为已知框架，当前只有无约束语义目标。',
     '用实际分类器方向导数构造二维约束，显式处理不可行和非线性验收。',
     '分类器质量不是图像质量；有限半径和线性化可能使约束不可满足。', ('clf','classifier'), semantic=True)

idea('future_input_metric', '按null未来形变代价选择根更新', 'E 输入与调度',
     '未来形变惩罚', (.01, .1, 1.),
     'min ||R+J_R Q a||^2 + theta*||J_U Q a||^2',
     '相同latent步长可能导致不同终点改动。把未来形变作为输入代价，可以优先选择达到一致性但较少扰动原结局的方向。',
     '此前控制残差的输出度量与条件锚定。',
     '正则作用于实际J_U输入响应，而非只惩罚latent距离或条件终点漂移。',
     '保留原null未来可能妨碍条件写入；局部形变不等于多样性损失。', ('adjoint','clf'))
idea('semantic_style_guard', '语义达标与类无关矩保持', 'E 输入与调度',
     '终点通道矩保持权重', (.01, .1, 1.),
     'min relu(.6-q_c(U))^2 + theta*||moments(U)-stopgrad(moments(U0))||^2',
     '固定语义目标不必锁定完整图片，但可以温和保留原终点的通道均值与尺度，检验是否减少提高分类分数时的外观崩坏。',
     '固定语义目标与完整图像锚定已有。',
     '只锚定八个通道矩，留下空间细节自由度，不把这些矩声称为全部风格。',
     '通道矩也可能包含类别信息，保持过强会阻碍正确模式切换。', ('classifier','adjoint'), semantic=True)
idea('richer_input_subspace', '扩大可识别的未来输入子空间', 'E 输入与调度',
     '输入子空间维数', (2, 4, 8),
     'min ||R(x+Q a)||^2 with Q built from gap, coarse gap and fixed random directions',
     '目标可能合理而二维输入不足以控制它。扩大正交输入方向，检验此前负结果是否来自可控性瓶颈。',
     '旧二十项和七方法主要使用二维gap/随机方向。',
     '同一目标、实际逐方向查询响应；维度增长只算一个idea的参数。',
     '增加计算也增加过拟合代理的能力，必须同时报告成本。', ('fsg','adjoint'))
idea('two_stage_mpc', '两次未来写入的滚动控制', 'E 输入与调度',
     '后续写入代价', (.1, 1., 10.),
     'plan delta0 now and delta1 at next quarter; min ||U_plan-C0||^2 + theta*||delta1||^2; execute delta0 only',
     '只优化当前写入可能看不到未来可以更容易修正的方向。规划当前和未来两次动作、只执行第一步，再重新规划，检验预算的时间分配。',
     '固定终点逆问题已有；MPC是已知控制思想。',
     '真实null续生成中插入两次虚拟状态写入，四个系数联合识别；未执行的未来动作不伪装成实际采样效果。',
     '规划的未来动作可能从未兑现，粗模型和短规划会造成乐观偏差。', ('clf','adjoint'), extended=True)
idea('event_triggered_root', '由未来误差触发的按需校正', 'E 输入与调度',
     '相对触发阈值', (.5, .8, 1.1),
     'calibrate only when normalized terminal RMS exceeds theta times its first-event value',
     '固定事件可能在无需修正时浪费写入。观察经时距归一化的未来误差，只对超阈值状态更新，检验计算分配与少干预的收益。',
     '已有固定事件校准和旧#19短区间承诺迟滞。',
     '完整未来输出控制是否触发状态校正，五个检查事件逐样本决定。',
     '探针仍有成本，误差小并不意味着语义正确。', ('clf','fsg'), extended=True)
idea('semantic_handoff', '达到固定语义要求后交给null', 'E 输入与调度',
     '接管置信度阈值', (.4, .6, .8),
     'calibrate q_c(U_t) toward theta, then use unconditional dynamics for accepted samples',
     '如果状态已使null完整未来读出正确语义，持续CFG可能只增加过强化。达到阈值后逐样本停止常规条件引导，直接检验写入后接管。',
     'RAEv2已有初始写入后null接管；当前七方法仍在事件间走CFG。',
     '小SiT上由实际完整终点分类读出决定何时接管，保留所有样本而非过滤。',
     '粗未来与分类器误差会导致过早接管；接管读出本身有额外成本。', ('classifier','fsg'), semantic=True, extended=True)
idea('golden_handoff', '完整黄金残差的迟滞接管', 'E 输入与调度',
     '归一化残差接管阈值', (.2, .4, .6),
     'two low-R confirmations latch null; re-enable CFG when normalized R exceeds 1.5*threshold',
     '即使语义含义尚未被证明，完整一致性仍可作为无需持续条件输入的经验信号。两次确认与迟滞避免单次偶然小残差。',
     '旧二十项#19用短未来；第二份分析的黄金不变性。',
     '完整终点残差、逐样本状态机和可恢复null接管；继续保留该猜想。',
     '接管后R变化不等于遗忘；条件探针仍存在，不称完全不再查询条件。', ('fsg','ctrl'), extended=True)
idea('full_cycle', '完整条件前向与null逆向的有界往返', 'E 输入与调度',
     '往返更新倍率', (.25, .5, 1.),
     'delta = Phi_u(T,t)(Phi_c(t,T)x)-x; bounded correction with full-root acceptance',
     '用实际多步逆向null流承接完整条件终点，直接实现比局部影响匹配更接近双流复合的写入算子。',
     'FSG双流固定点；已有短步往返、RAEv2锚定逆映射。',
     '完整未来8步Heun前向与反向，单列同场数值缺陷并保存；不修改旧FSG对照。',
     '逆向可能放大误差，往返自身并非恒等；由完整残差验收和FID判断。')
idea('anderson_root', '完整未来根迭代的有界割线加速', 'E 输入与调度',
     '加速系数ridge', (.01, .1, 1.),
     'two root updates plus depth-1 residual mixing; accept best actual candidate',
     '单次局部校准可能不足，但重复同方向更新又收敛慢。利用连续两次残差变化作有界混合，检验更有效的根求解是否改善采样。',
     'Anderson是已有加速方法；旧FSG已有Picard类迭代。',
     '使用完整未来非线性目标检验加速候选；基线另含普通完整根。',
     '更准确的坏根仍可能更差，加速系数也可能不稳定。', ('anderson','fsg'))
idea('bayes_curl', '条件差的局部可积性约束', 'E 输入与调度',
     '反对称响应惩罚', (.1, 1., 10.),
     'min ||R_t||^2 + theta*||Q^T(J_g-J_g^T)Q||^2',
     '标准Gaussian路径的真实条件差是标量后验势的梯度乘时间系数，其空间Jacobian应对称。非零旋度是可观测的结构不一致，可能帮助选择较可信的写入位置。',
     '旧协方差方法对称化Jacobian，但没有直接控制被丢弃的反对称部分。',
     '保留原始有限差分投影中的反对称项，与完整终点根联合优化。',
     '两个投影方向只能看到局部旋度，低旋度也不证明模型准确；作为经验正则检验。', ('fm','fsg'))

idea('semantic_golden_set', '固定语义要求限定的黄金集合', 'P 三条重点',
     '绝对语义项权重', (.1, 1., 10.),
     'L=||R/scale||^2+theta*relu(.6-q_c(U_t(x)))^2',
     '保留FSG同起点等式，但要求共同终点还达到固定的目标语义阈值。对theta>0，L=0当且仅当两端一致且指定语义读出达标；终点时距收缩不能自动消除语义项。',
     '完整根、语义一致性与固定目标控制均已有；本轮明确联合绝对成功条件。',
     '区别于#11的两分支语义彼此相等：即便两端同样低置信度，本目标仍为正。完整输入Jacobian为[J_C-J_U; Dq(U)J_U]。',
     '严格结论针对固定分类读出和指定数值未来；真实语义可靠性、可达性及FID改善仍需验证。',
     ('fsg','classifier','clf'), semantic=True)
idea('verified_semantic_step', '粗细未来共同确认的语义写入', 'P 三条重点',
     '数值不一致收紧倍率', (1., 1.5, 2.),
     'accept iff Delta q8 > theta*(abs(q16(x)-q8(x))+abs(q16(x_new)-q8(x_new)))',
     '三角不等式直接给出Delta q16 >= Delta q8-两个端点的网格差。因此theta>=1时，每次接受都严格提高实际16步未来的同一语义读出，无需假定可观测差是真实ODE的误差上界。',
     '粗细残差一致性#30与语义输出控制。',
     '8步未来提出方向，16步未来只作独立分辨率验收；保证的是两种已计算映射上的单次语义进展。',
     '16步仍不是连续真解，分类分数也不是FID；保守验收可能拒绝有益更新。',
     ('adjoint','classifier'), semantic=True)
idea('finite_semantic_control', '无输入方向假设的最小干预搜索', 'P 三条重点',
     '期望损失下降比例', (.05, .15, .3),
     'evaluate {0,+-r*q1/2,+-r*q2/2,+-r*q1,+-r*q2}; choose min-norm point meeting L_new<=(1-theta)L0, else best improving point',
     '未知控制方向或局部线性失准时，直接比较有限允许输入集合的实际固定终点损失。集合包含零输入，故被选损失不增；有满足下降要求的候选时，选其中最小范数者。',
     '局部线性控制、有限候选选择和最小干预均为已有工具。',
     '九个实际null未来不依赖Jacobian方向、可逆性或不可观察扰动界；结论在这个明确有限集合内可穷尽核验。',
     '有限集合可能错过有效方向，额外查询可能不划算；保留选择前后全部候选损失以核对。',
     ('classifier','clf'), semantic=True)


def configurations():
    rows = []
    def add(family, key, strength, theta=0., role='control', source='cfg',
            solver='heun64', semantic=False, idea_id=None, parameters=None):
        index = sum(row['family'] == family for row in rows)
        rows.append(dict(arm=f'{family}_{index:02d}', family=family, key=key,
            strength=float(strength), theta=float(theta), role=role, source=source,
            solver=solver, cutoff=.5 if source == 'ig' else .75, idea_id=idea_id,
            external_semantics=semantic, parameters=parameters or {}))
    add('strong', 'native_ig', 0., role='native', source='ig')
    add('ig_local', 'ig_local_attention', .8, 2., source='ig')
    for a in (.25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.):
        add('cfg_native', 'native_cfg', a, role='native')
    for a in (1., 1.5, 2., 2.5, 3.):
        for beta in (-.75, -.5, -.25):
            add('cfg_apg', 'cfg_apg_momentum', a, beta)
    for a in STRENGTHS:
        add('cfg_sde_native', 'native_sde', a, role='native', solver='sde_tail64')
        for h in (.0625, .125):
            add('fsg_operator_control', 'fsg_operator_control', a, h)
        for radius in (.5, 1.):
            add('full_root16_control', 'full_root', a, radius)
        for gain in (.1, .2):
            add('cfg_smc_control', 'smc_control', a, gain, parameters={'lambda': 5.})
        for target in (.5, .8):
            add('semantic_direct', 'semantic_contract', a, target, semantic=True)
            add('sde_semantic_direct', 'sde_semantic_direct', a, target,
                solver='sde_tail64', semantic=True)
    assert len(rows) == 73
    for method in IDEAS:
        for a in STRENGTHS:
            for theta in method['values']:
                add(f'i{method["id"]:02d}_{method["key"]}', method['key'], a, theta,
                    role='candidate', solver=method['solver'], semantic=method['external_semantics'],
                    idea_id=method['id'], parameters={'events': list(method['events']),
                    'future_steps': FUTURE_STEPS, 'swept_parameter': method['parameter']})
    assert len(IDEAS) == 53 and len(rows) == 709
    assert len({row['arm'] for row in rows}) == len(rows)
    assert all(sum(row['idea_id'] == m['id'] for row in rows) == 12 for m in IDEAS)
    return rows


if __name__ == '__main__':
    from experiments.sit_control_50_20260910.pipeline import main
    main()

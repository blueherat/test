# APG、非保守响应与未来控制

本研究把三个不同的问题分开检验：模型条件差是否具有明显的非保守响应，APG删除的平行分量实际改变了什么，以及有限步积分误差是否已成为当前采样的主要瓶颈。32个独立新噪声种子的实测支持前两个问题值得继续研究，却尚未显示64步Heun的局部积分误差具有很大的平均语义影响。因此，新增实验同时保留积分方向和未来语义方向，并用不同对照识别它们的收益来源。

新增第54–58项各安排12组1K配置，另有141组对照，合计201组。它们沿用已有53-idea队列的1K输入，排在原709组及其最终审计之后。机制检查使用独立输入、没有FID；本文中的机制数值不能被解释为新增方法已经改善图像质量。完整配置、成本约定与自动续跑要求见[冻结协议](SIT_APG_EXTENSION_PROTOCOL_20260911_ZH.md)。

**已有方法与本轮研究的边界。**

APG在clean预测空间处理条件差，结合负动量、norm截断及相对条件clean预测的投影，减轻高引导量下的过度放大。三个操作都是已有方法；本轮把clean历史与velocity历史区别开，并研究投影方向对真实未来的影响。[^apg]

非保守性也不是首次被观察到。原始CFG论文已指出学习到的向量场不一定保守；QCSBM直接研究score Jacobian的反对称部分，并以训练正则约束它。密度正确性又不同于Euclidean保守性：满足加权散度条件的附加场可以保留同一密度演化，不能仅由curl大小决定采样好坏。[^cfg][^qcsbm][^gauge]

数值误差引导已有很接近的工作。ERK-Guid利用Heun/Euler差异估计刚性及主要变化方向，再形成无需额外网络查询的校正；其局部特征向量推导使用对称Jacobian的理想结构。第54项研究的是实测神经场的反对称响应是否适合分配主积分精度，必须与ERK及普通嵌入误差分配比较。[^erk] UAI 2024的自适应时间网格则用score误差和光滑性信息优化离散点，说明“更多步”与“更好的模型误差权重”本来就不能混为一谈。[^timesteps]

未来语义控制同样不是空白。DEFT研究终点条件下的Doob变换，DCFG使用额外评价器动态调节CFG，REG把条件guidance置于未来奖励及轨迹联合分布框架中。它们限制了“考虑未来”“学习强度”“用终点分数”这些表述能够承载的新颖性；本轮的新实验单位应是具体输入子空间、实际验收规则和配对撤条件比较。[^deft][^dcfg][^reg]

|本轮方向|继承的工作|本轮可检验的具体变化|
|---|---|---|
|54 非保守响应触发主积分细分|旧#50 curl正则、旧#30未来网格一致性、ERK与自适应求解|把完整受控场的反对称响应用于分配主轨迹步数，并与同探针缩小gain比较|
|55 未来幅度投影|APG、旧#40语义约束、#41未来输入几何、#42通道moment|在APG的平行/正交平面识别未来moment与语义响应，并实际验收二者|
|56 未来选择平行保留量|APG固定η、DCFG有限候选、旧#53直接搜索|比较真正执行的8步候选块，选择η而非只选择整个CFG倍数|
|57 边际条件价值撤条件|旧#46概率接管、条件时机、最优停止|在同一物理区间与同一后缀网格上比较继续条件和撤条件|
|58 统一终点选择H|FSG多时距、旧#2固定H、#52网格验收、#53候选选择|各H只负责提出输入，排名统一使用完整null未来的同一个q|

这些是五个可证伪的实验方向，不是五项已建立的新贡献。是否值得进一步凝练为论文，需要同时检查FID、总计算、对照差异和机制预测是否成立。

**统一记号与应保留的理论区分。**

本项目采用噪声到数据的时间t∈[0,1]，条件场与null场分别记作v_c、v_u，原始条件差为g=v_c−v_u。额外CFG量为a，实际主场为

\[
F_a=v_c+a g=v_u+(1+a)g.
\]

clean条件预测为m_c=x+(1−t)v_c。无动量、无截断的APG投影场可写为

\[
g_\parallel=\frac{\langle g,m_c\rangle}{\|m_c\|^2}m_c,
\quad g_\perp=g-g_\parallel,
\quad F_{a,\eta}=v_c+a(g_\perp+\eta g_\parallel).
\]

η=0删除平行分量，η=1恢复原始CFG。这里的“平行”只描述给定latent度量中的局部几何，不等同于“没有语义”，正交也不保证“纯语义”。投影满足一阶内积为零，但有限更新仍有二阶能量变化：‖m_c+a g_⊥‖²=‖m_c‖²+a²‖g_⊥‖²。

在标准各向同性Gaussian路径的规范速度表达中，理想条件差与条件后验的空间梯度成比例。因此，固定坐标下的反对称Jacobian是可观察的结构偏离；但它只识别导数的不对称部分，既不等于速度误差的大小，也不是一个已经分离出的“错误旋转向量”。常量误差可以很大却没有curl，小振幅高频误差也可以拥有很大的curl。

进一步，把完整场的Jacobian写成J_F=S+A，S为对称部分，A为反对称部分。局部扰动满足

\[
\frac{d}{dt}\|\delta x\|^2=2\delta x^T S\delta x.
\]

A不会直接贡献瞬时长度增长，却会改变扰动方向，进而与S及时间变化共同影响有限时段的放大。纯旋转在连续时间可以保存各向同性Gaussian分布，而显式Heun仍可能引入离散径向漂移。这个例子为“调积分而非直接惩罚curl”提供了动机，却不能自动证明真实SiT的主要误差来自旋转。

APG投影本身也可能产生非保守场。例如二维g=(1,0)、m_c=x时，投影后的g_⊥具有非零curl，却仍满足x·g_⊥=0。这说明必须分别测量raw gap、投影后的gap及完整受控场，而不能用同一个curl指标把APG的几何操作与模型结构误差混在一起。相关恒等式、加权散度反例和旋转积分例子已有[独立数值核对](FSG_RELEASE_CURL_AND_APG_REVIEW_20260911_ZH.md)。

还有一个值得保留的跨分支问题。设两个密度p_u、p_c的score为s_u、s_c，附加场r_u、r_c分别满足∇·(p_i r_i)=0；它们各自在对应分支中可以不改变密度。若形式上指定一个可归一化的瞬时倾斜密度π_w∝p_u^(1−w)p_c^w，并组合r_w=(1−w)r_u+w r_c，则直接展开散度得到

\[
\frac{\nabla\cdot(\pi_w r_w)}{\pi_w}
=w(1-w)(r_u-r_c)\cdot(s_c-s_u).
\]

因此，分支各自无害的偏移，组合后未必仍对指定密度无害；重要的可能是偏移与条件几何的耦合，而不只是curl范数。这里的π_w只是用于检验兼容性的指定瞬时密度，不能冒称真实CFG整条路径的边缘分布，REG关于边缘倾斜的分析尤其要求保留这个限定。[^reg] 该恒等式已在前述独立Gaussian与类别混合例子中核对；真实SiT没有可直接获得的p_i和r_i，所以本轮没有把一个可计算的curl代理冒称为该密度偏差的测量。

**实测设计与可重复证据。**

机制实验冻结SiT-S/2 800K checkpoint、VAE、分类器、代码与新输入。噪声seed为202611101；32个新噪声对应32个不同类别，按B4、4个GPU rank执行。沿a=0、1.25、2.75的CFG主轨迹，在t=0.125、0.25、0.375、0.5、0.625保存状态，合计480个重复观测状态、120个批次快照。

每个状态做三类检查。第一类用4对独立单位Rademacher方向p、q作中心有限差分，估计多个场的反对称响应和总Jacobian响应，并在同一点额外比较三个固定CFG量。第二类分别施加零、parallel、orthogonal、raw gap方向的等范数状态脉冲，各自运行完整8步与16步null未来，记录目标类别概率、margin、top-1、latent能量及8维通道moment。第三类在同一个长度1/64的区间上比较1、2、4个Heun子步，再用同一16步null后缀比较终点读出；另外保留匹配区间的null对照。

有限差分估计依据独立各向同性方向的恒等式。若空间维度为d，且p、q的协方差均为I/d，则

\[
\mathbb E\left[d^2(p^T Jq-q^T Jp)^2\right]=\|J-J^T\|_F^2,
\quad
\mathbb E\left[\frac d2(\|Jp\|^2+\|Jq\|^2)\right]=\|J\|_F^2.
\]

报告先分别聚合分子、分母，再取比值κ²。只有有限差分趋于真实导数时，分子分母的上述恒等式才适用；有限探针的比值本身有偏。κ²不是概率，不是谱范数，也不是“错误占比”；超过1并不矛盾。

同一个种子的时间与强度重复观测不能当成独立样本。本文区间以32个种子/类别对为聚类单位进行4000次bootstrap，完整保留簇内重复状态；类别又是无放回抽取，因此这些区间主要表达探索性不确定度，不作为总体显著性检验。没有进行多重比较校正。

全部原始响应和效果分别保存于[curl观测CSV](data/apg_mechanism_extension_20260911/curl_observations.csv)和[效果观测CSV](data/apg_mechanism_extension_20260911/effect_observations.csv)，原始状态及全部终点latent另有带哈希的NPZ。全程累计57,248次B4完整模型前向，内部解码和分类各6,240张；最慢rank的正式研究循环约95.33秒，模型加载及队列切换另外耗时。[汇总与哈希](data/apg_mechanism_extension_20260911/summary.json)记录可复核范围。

**发现一：raw gap的非保守响应明显，APG投影并没有系统地消除它。**

下表是沿各自CFG轨迹访问状态后的κ²=‖J−Jᵀ‖²_F/‖J‖²_F聚合估计。投影场只含无动量、无clip的η=0投影，不能外推为完整带历史APG的Jacobian。

|轨迹额外引导a|条件场|raw gap|投影后的gap|完整CFG场|完整投影场|
|--:|--:|--:|--:|--:|--:|
|0|0.0566|0.8810|0.8935|0.0566|0.0566|
|1.25|0.0585|1.0463|1.0613|0.1449|0.1435|
|2.75|0.0640|1.1477|1.1272|0.3201|0.3109|

raw gap与投影gap的比值很接近，且两者明显高于条件场。一个自然解释是：相减后较大的共同场部分被抵消，条件差的结构偏离变得更突出；这并不说明相减后的向量误差绝对值一定更大。APG的作用也不能概括为把场“修成保守场”。

只比较不同a的轨迹，会把引导量与访问状态变化混在一起。本研究因此在同一状态上重新构造a=0、1.25、2.75的完整场：例如在a=0轨迹状态上，三种完整场的κ²分别约0.0566、0.1300、0.2741。固定点上J_F=J_c+aJ_g的组合已经能显著改变完整场的结构，不能把整张表完全归因于状态迁移。另一方面，固定点上仅把g整体乘常量，其归一化κ²理论上不变；这里比较的是J_c+aJ_g，二者不是同一个实验。

差分半径为0.001·max(‖x‖,64)，并复查一个方向的双倍半径。导数相对不一致的中位数为0.194%，90分位为5.65%，99分位为39.1%；39/480个状态超过10%。在保留的441个较稳定状态上，raw gap的聚合κ²仍约0.9845，投影gap约0.9885，因此明显非保守响应不完全由最不稳定的差分点造成。但四对方向仍很少，不能据此宣称得到了完整Jacobian或每个样本可靠的curl排序。[场汇总](data/apg_mechanism_extension_20260911/field_summary.csv)、[差分稳定子集](data/apg_mechanism_extension_20260911/fd_reliable_subset.csv)。

**发现二：parallel更容易改变未来幅度，但并非没有语义收益。**

每个方向使用相同欧氏脉冲长度r=(2/64)max(a,0.5)‖g‖，然后完全撤条件运行16步null未来。表中的Δq是完整ImageNet-1K分类器目标类别概率的绝对变化，ΔE是终点latent逐坐标平方均值变化。它们不是FID、成功率或感知饱和度。

|a|orthogonal的平均Δq|parallel的平均Δq|orthogonal的平均ΔE|parallel的平均ΔE|
|--:|--:|--:|--:|--:|
|0|+0.004535|+0.000517|−0.001390|+0.004247|
|1.25|+0.015896|+0.006743|−0.001726|+0.014044|
|2.75|+0.020473|+0.002270|−0.002109|+0.028247|

跨强度与时间聚合，orthogonal的平均Δq约0.01363，聚类bootstrap区间为[0.01060,0.01688]；parallel约0.00318，区间[0.00097,0.00580]。parallel有62.1%的状态提高q，orthogonal为71.7%。尤其在a=2.75时，parallel的平均增益区间跨零，提示它的作用有明显状态差异。

这与APG的幅度控制直觉相容，也直接反对“所有parallel分量都应视为无用”的解释。但是，等范数比较有一个重要边界：实测raw gap中的parallel范数通常只占约13%–16%；将它单位化后再施加同样长度，会比原始gap里的自然parallel分量更大。该实验检验方向的局部因果作用，不能当成真实CFG分解后各项贡献的线性加和。第56项因此进一步比较不归一化、真正执行的APG多步块。

8步和16步null未来的增益符号一致率，raw gap与orthogonal均为94.2%，parallel为92.5%；Spearman相关约0.966、0.964、0.930。短未来并非完全无效，但仍会产生方向判断差异。本轮新方案统一用16步完整null未来作关键验收，避免直接把8步读出当作真值；16步本身也仍是数值近似。[分量汇总](data/apg_mechanism_extension_20260911/component_summary.csv)、[网格比较](data/apg_mechanism_extension_20260911/future_grid_agreement.csv)。

**发现三：当前64步主积分的局部误差可测，但平均语义影响很小。**

在同一个h=1/64区间内，以4个Heun子步作为数值参考，单步CFG的平均latent RMS误差随a从0增至2.75，大约由1.0×10⁻⁵增至4.4×10⁻⁵；相对主步增量约0.061%–0.232%。改成2个子步后误差明显缩小。这里的4子步是较细参考，不是连续ODE真解。

进一步比较粗步与4子步之后接同一16步null未来的q，得到下面的平均变化：

|a|细分CFG主步后的平均Δq|探索性95%聚类区间|
|--:|--:|---|
|0|+0.000014|[−0.000004,+0.000035]|
|1.25|+0.000018|[−0.000018,+0.000057]|
|2.75|+0.000047|[−0.000101,+0.000186]|

在已检查的时间点，单步积分细化的平均语义变化远小于方向脉冲的平均影响，且区间均包含零。这不能排除多个主步误差积累、少数困难样本或不同步数下的效应；它只能说明，不能仅凭“curl很大”就宣布积分误差是当前设置的主导瓶颈。

在固定时间/强度层内排名后，CFG反对称响应估计与局部误差的相关约0.388，总Jacobian响应与局部误差约0.360。差距不大，四对探针也不足以确证谁更有预测力。第54项必须证明它比普通嵌入误差或总预算加密更划算；如果没有，其理论动机仍可成立，而当前实现应被视为没有实用增益。[细分汇总](data/apg_mechanism_extension_20260911/refinement_summary.csv)、[局部相关](data/apg_mechanism_extension_20260911/local_error_correlations.csv)。

**第54项：把非保守响应变成主积分精度的分配信号。**

候选在5个事件读取完整CFG场的随机反对称响应，估计

\[
s_t=h\sqrt{\widehat{\|J_F-J_F^T\|_F^2}/(4d)}.
\]

这是“典型旋转响应×步长”的无量纲代理，不是最坏方向的稳定性界。若超过阈值，就把该样本随后8个主步各分成两个Heun子步；若差分半径检查不可靠，则回退到普通嵌入误差规则。这样保留引导场本身，单独改变数值执行精度。

其可证伪预测是：触发的样本/区间应该比平均区间更受益于细分，并在实际总计算下优于均匀更多步、普通嵌入误差及同探针缩小gain。ERK对照则检验已有无需额外查询的数值校正是否已经足够。随机探针代价很高：即使没有发生细分，本实现也可能比96步原生CFG更贵，因此应以FID–实际GPU秒比较，不把“自适应”当作效率结论。

**第55项：让APG投影尊重未来moment，同时保护可观察语义。**

设U_t为固定16步null终点映射，q为目标类别读出，ν为终点latent通道均值和标准差。理想局部输入响应是

\[
M=Dq(U_t)J_{U_t},\qquad N=D\nu(U_t)J_{U_t}.
\]

相较于只要求m_cᵀu=0，更接近“控制未来幅度”的条件是Nu较小；它涉及未来映射的Jacobian，而不是当前clean向量自身。但直接限制全部未来latent会压制需要的语义变化，所以ν只选8个明确的moment输出，并另加语义保护。

实际实现限定在Q=[单位g_⊥,单位g_∥]的二维平面内，写入δ=r_xQc。两个方向的对称有限差分识别M_Q与N_Q；把N_Q按Frobenius范数归一化后，在明确的259点有限集合中最小化

\[
\|c-c_{\rm raw}\|^2+\lambda\|\bar N_Qc\|^2,
\quad M_Qc\ge\tfrac12\max(M_Qc_{\rm raw},0),\quad\|c\|\le1.
\]

这里求的是有限集合内的最优解，不声称求出了连续二次规划的精确解。原始方向、投影方向和零输入都明确包含在集合中。范数归一化使统一改变moment输出单位时，线性罚项不发生任意变化；它不能保证不同moment之间的相对权重是最优的。

上述M_Q、N_Q是关于无量纲系数c的响应，已经包含写入半径r_x，即分别近似r_xMQ与r_xNQ，避免把单位状态位移的导数与有限控制系数混用。

提案随后用真实16步null未来复查。只有当q严格高于零输入、保留原始gap同半径脉冲至少一半的正q收益，并且实际moment变化不大于原始脉冲时才接受。保存全部有限候选目标、可行标记、提案和真实验收读出，以便穷尽核对。保证仅适用于这个时间点、这个数值未来及这个分类器，不保证后续持续条件采样的最终q或FID单调改善。

专门对照使用相同输入平面、同样未来查询和分类器直接选语义方向；另一个对照移除语义保护但保留moment限制。如果效果只能由分类器信息解释，则不能归因于未来幅度几何。如果moment限制显著损失FID或类别成功，也可能说明这8个latent统计量并不是应保持的属性。

**第56项：用真正执行的多步未来决定保留多少parallel。**

对η∈{0,0.5,1}，从同一个事件状态分别执行8步实际APG场，得到z_η，再由相同起止时刻的16步null未来计算q_η与ν_η。以η=0为参考，选择

\[
\eta^*=\arg\max_{\eta:q_\eta\ge q_0}
\left[q_\eta-\lambda\frac{\|\nu_\eta-\nu_0\|^2}{\max(\|\nu_0\|^2,\epsilon)}\right].
\]

所选z_η直接成为主轨迹下一块状态，候选阶段没有只优化一个不会执行的虚拟动作。有限集合包含固定APGη=0，因此本次所比较的假设撤条件终点q不低于这个参考；全部候选分数和选择索引逐批保存。

它检验的是固定投影丢失parallel是否可以按实际未来纠正。因为比较的是真实多步块，绕开了等范数脉冲相对自然分量放大的问题。也因为测试三条分支及其未来，成本显著高于固定η；固定η三个对照、直接语义对照与总计算账目都不可省略。

**第57项：把撤条件改写成配对的边际价值问题。**

若z_s沿条件前缀前进，完整null终点记为Y(s)=U_s(z_s)，在精确确定性流下

\[
\frac{dY}{ds}=J_{U_s}(v_c-v_u),
\quad
\frac{dq(Y(s))}{ds}=Dq\,J_{U_s}(v_c-v_u).
\]

这与“沿前缀重新计算两分支终点差”有实质不同：它问继续提供条件的边际收益，而不是要求两条分支相等。若前缀使用CFG，对应差为F_a−v_u。FSG残差沿前缀的重算仍有其用途，但精确流下的终点相等会随着剩余时间趋零而自然成立，不能单独作为语义成功证据。[^fsg]

数值实现必须处理后缀网格。对U_t(x)直接重新取16个后缀步，再与t+h处重新取16步的结果相减，含有被动null自身的网格变化；本研究观察到该被动q差可达约0.02的尾部量级。第57项因此先在同一8步区间内分别推进“继续当前条件控制”和null，然后接完全相同的16步null后缀，用q_keep−q_null作为有限区间价值。

仅在q_null≥0.5且该价值不超过阈值时，才锁定null接管；以后不再恢复条件。这是带成功门槛的局部策略，不是全局最优停止：继续价值可以非单调，早停还可能错过后续收益。机制数据的一步配对比较确实出现部分负价值状态，但不能直接预测8步停止策略的FID。

它与Adaptive Guidance也需明确区分。该既有工作根据分支对齐程度，减少额外CFG并继续conditional生成；本实验研究的是撤掉类别输入后的真正null续生成，并支付显式未来探针成本。[^adaptive] 所以当前实现是质量/条件承接的实验，不能凭“撤条件”三个字宣称推理加速。固定撤条件时间和纯概率阈值接管分别作为对照。

**第58项：让物理前瞻H负责提出动作，由统一终点负责选择。**

若直接用‖M_Hᵀe_H‖评价时距，只要把某个输出统一乘常数c_H，评分就会乘c_H²，而根集合和控制问题的含义并没有相应改变。固定H与固定积分步数也不同：同样8步可能跨越不同物理长度，带来不同数值误差，不能把H大小直接解释为信息价值。

候选H∈{0.125,0.25,0.5}各自定义条件/null未来残差R_H。在raw gap单位方向上，通过同样写入半径的中心差分建立一维响应j_H，提出有界系数

\[
\alpha_H=\operatorname{clip}_{[-1,1]}
\left(-\frac{\langle R_H,j_H\rangle}{\max(\|j_H\|^2,\epsilon)}\right),
\qquad \delta_H=r_x\alpha_H\frac{g}{\|g\|}.
\]

三个提案和零输入都通过同一个完整16步null终点读出q(U_t(x+δ_H))评价。只有超过共同基线和预设最小增益的候选才能被选中，最终取q最大的候选；有限比较不依赖不同时距残差的任意幅度单位。保存每个H的系数、长度、共同q和选择索引。

这只是统一评价单位的有限控制器，不证明被选H拥有最大的普遍信息价值，也不排除分类器过拟合。每个固定H都有完全相同提案和末端验收的对照，可以区分“某个固定H本来就足够”与“按状态选择有额外价值”。三个提案全部付费；方法若只提高q而不改善FID或成本，也不能算实践上的进展。

**正确的APG参考与后续判据。**

原冻结队列的APG采用velocity residual历史，β固定，norm上界为当前gap范数的两倍。论文clean历史在本项目变量中满足

\[
m_k^D=b_kg_k+\beta m_{k-1}^D,\quad b_k=1-t_k,
\quad
m_k^V=g_k+\beta\frac{b_{k-1}}{b_k}m_{k-1}^V.
\]

所以把clean历史转为velocity后，历史系数随b变化；固定velocity β与论文clean β一般不等价。新增clean APG对照严格保留clean历史及固定半径，旧适配仍完整保留。两者的差异属于基线校正与参数化审查，不作为新idea计数。负动量也不能简单解释为通用去噪器：它衰减持续方向，却可能放大交替变化，因此还需实际FID比较。

每个方向均以四档引导量乘三个结构参数展开。1K筛选沿用同一bank，后续可以逐样本比较，也能与原53项对照；它同时继承网格选优与小样本FID的局限。内部q、moment和curl只用于机制解释或方法本身，不代替最终质量判断。

第54项需要超过普通计算加密与ERK的解释力；第55项需要超过相同分类器和方向预算的直接语义控制；第56项需要展示按状态选择η比固定η更有效；第57项需要比固定时机及纯q门槛更好；第58项需要比所有单独固定H对照更值得其额外计算。若某项不满足相应条件，应保留该实验结果并缩小其研究主张，而不是回过头重命名对照或隐藏负结果。

实现与复核入口为[机制实验](../experiments/sit_apg_mechanism_20260911/study.py)、[分析脚本](../experiments/sit_apg_mechanism_20260911/analyze.py)、[采样实现](../experiments/sit_apg_mechanism_20260911/core.py)与[逐批选择审计](../experiments/sit_apg_mechanism_20260911/checks.py)。动态执行情况和正式FID结果记录在[筛选结果](SIT_APG_EXTENSION_RESULTS_20260911_ZH.md)，不写入这份冻结研究说明。

**来源。**

[^apg]: Seyedmorteza Sadat, Otmar Hilliges, Romann M. Weber. [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://arxiv.org/html/2410.02416v2). ICLR 2025；v2，2025-06-03。§4、Algorithm 1；clean空间投影、norm截断、负动量的来源。
[^cfg]: Jonathan Ho, Tim Salimans. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). 2022。§3.2；CFG场组合及学习场非保守性的讨论。
[^qcsbm]: Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Chun-Yi Lee. [On Investigating the Conservative Property of Score-Based Generative Models](https://proceedings.mlr.press/v202/chao23a.html). ICML 2023。Jacobian反对称响应及训练正则。
[^gauge]: Christian Horvat, Jean-Pascal Pfister. [On Gauge Freedom, Conservativity and Intrinsic Dimensionality Estimation in Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d553f0e0abb80e2a60328d634583bd2e-Abstract-Conference.html). ICLR 2024。§3、加权散度与密度演化中的规范自由度。
[^erk]: Inho Kong, Sojin Lee, Youngjoon Hong, Hyunwoo J. Kim. [Error as Signal: Stiffness-aware Diffusion Sampling via Embedded Runge-Kutta Guidance](https://arxiv.org/abs/2603.03692). ICLR 2026；v2，2026-04-19。§4、Appendix A与Algorithm 1；实际Heun/Euler响应估计、对称Jacobian理论前提与采样算法。
[^timesteps]: Yuzhu Chen, Fengxiang He, Shi Fu, Xinmei Tian, Dacheng Tao. [Adaptive Time-Stepping Schedules for Diffusion Models](https://proceedings.mlr.press/v244/chen24c.html). UAI 2024。§4–5；模型误差与时间离散选择的关系。
[^deft]: Alexander Denker et al. [DEFT: Efficient Fine-Tuning of Diffusion Models by Learning the Generalised h-transform](https://proceedings.neurips.cc/paper_files/paper/2024/hash/22d258dfbdf840ccbf266bbc545dd95f-Abstract-Conference.html). NeurIPS 2024。终点条件与未来价值控制的既有背景。
[^dcfg]: Pinelopi Papalampidi et al. [Dynamic Classifier-Free Diffusion Guidance via Online Feedback](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cef8b22f26953d8bdaf93bb64b7cc72f-Abstract-Conference.html). ICLR 2026。在线评价器与动态guidance选择；与本轮未来块及APG分量选择的边界。
[^reg]: Zhengqi Gao, Kaiwen Zha, Tianyuan Zhang, Zihui Xue, Duane S. Boning. [REG: Rectified Gradient Guidance for Conditional Diffusion Models](https://proceedings.mlr.press/v267/gao25t.html). ICML 2025。§3–4；终点/逐时边缘倾斜的局限、联合分布目标与未来奖励近似。
[^fsg]: Kaibo Wang, Jianda Mao, Tong Wu, Yang Xiang. [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/html/2510.21512v1). NeurIPS 2025。方法名Foresight Guidance，简称FSG；双流未来与时距设计，与本文沿前缀撤条件分析的关系。
[^adaptive]: Angela Castillo et al. [Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models](https://arxiv.org/abs/2312.12487). 2023。§5；基于对齐减少额外CFG并保留conditional生成。

一手全文保存在`readings/apg_extension_20260911`及此前的`readings/apg_release_curl_20260911`，各有URL和SHA256清单。机制数值来源于本项目实际checkpoint及上述固定输入，而非论文表格。本文未复制论文的图像质量数字来替代本项目验证。

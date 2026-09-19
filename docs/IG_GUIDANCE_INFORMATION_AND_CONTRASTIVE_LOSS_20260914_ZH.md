**研究记录：AG 所需的信息是什么，以及怎样把它变成输出头的训练目标**

2026-09-14。回应“中间表征可能保留最终去噪损失没有充分利用的信息”这一假设。本文区分已有文献、由标准恒等式得到的推导、尚待生成实验验证的假设。上轮两个分布目标已实现并进入空闲 GPU 队列；本文新推导的来源对比目标只有可运行 loss 原型和 CPU 代数核验，尚未完成真实模型训练，也未加入该冻结队列。

**当前判断。** 值得把问题从“弱头需要多大”推进到“弱头被要求估计什么”。更具体的候选信息是：**在相同带噪输入下，真实端点与当前生成端点分别支持什么去噪方向，两者有什么系统差异。** 可以保留 x/v 作为可观测监督，却让目标从单个条件均值变成两个来源的条件均值之差。它不要求强弱误差逐点共线，也不要求真实分布加一个严格共享污染分布。

这个判断不等于“已经解释了 MLP 为何有效”。SiT 的既有结果支持某个内部读出配置有用；RAEv2 的结果则明确限制了泛化：继续降低弱头 MSE，并未使其选定配置在新 5K 上超过原 IG。两者比较过的引导系数不同，不能把这项差异独立归因于结构或训练目标。[SiT 5K](CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)、[RAEv2 收敛](RAEV2_CONTEXT_20K_RESULTS_20260914_ZH.md)、[RAEv2 5K](RAEV2_CONTEXT_5K_RESULTS_20260914_ZH.md)。

**先把容量上的“矛盾”拆清楚。** 冻结网络中，深层特征是浅层特征的确定变换，但这并不证明它严格丢失了某种 AG 信息。信息可能被保留，却变得难以让受限的小头读出。Predictive V-information 专门区分 Shannon 信息和给定计算能力下可利用的信息；它允许计算改变信息的可读性。因此，更适合我们的假设是“对特定任务、小头可读的信息随深度变化”，而不是未经验证的全局信息压缩。[^vinfo]

REPA 的实验发现，把表征对齐限制在前若干层比约束更深层更有利，作者提出后续层可专注细节。这给“层间任务分工”提供背景，不证明我们的冻结模型丢失了 AG 信息，也不证明把语义特征变强一定使 AG 变好。[^repa]

SSG 对头容量已有直接证据：JiT-B/16 上，仅线性头的无 CFG FID 为 36.44，基线为 25.42；一个 Transformer block 为 9.47，两个为 9.14。它支持线性读出在这个配置里不足，同时说明“稍大一点立即失去 guidance”也不成立。论文把线性头失败与 patch 边界伪影联系起来；不能仅凭这些数据推导一个普遍的最佳深度。[^ssg]

如果新头真正复制了强模型的同一个函数，强减弱当然为零。但“参数量相同”不保证函数相同；相同训练来源、相同目标、足够信息和充分优化等条件都不可省略。另一方面，某些高斯问题中线性场就足够，故“线性必然无效”也不是一般定理。更换监督对象后，即使弱头容量充分，它也不必收敛到强模型的预测。

还有一个需要避免的误解：单个位置的 MSE 输出是条件均值，未必显式表示所有后验信息；但一个处处精确的 score 函数，在通常的连通、正密度条件下已决定整个密度。不能宣称精确 x/v 函数之外必然存在某种不可替代的神秘 AG 信息。这里研究的是**有限表征、有限读出容量下，什么监督能把有用的分布对比读出来**。

**必须让训练对象指向端点，而不是采样过程的身份。** 以下令 P 为真实端点总体分布，G 为固定强模型、固定采样器的实际生成端点分布。类别 c 固定，之后可以逐类汇总。严格区分三个对象：

|对象|含义|能否直接互换|
|---|---|---|
|实际采样状态分布 \(r_t^\theta\)|运行某个 ODE/SDE/离散求解器到 t 的状态|不能与重新加噪分布自动等同|
|端点重新加噪分布 \(g_t\)|先采 \(X\sim G\)，再过共同的噪声通道|由 G 和选定通道确定|
|网络输出 \(S_\theta(z,t)\)|冻结神经网络在输入上的数值|不自动等于 \(g_t\) 的精确速度|

你的提醒是必要的：取 \(Z_t=Z_0+a(t)\)，其中 \(a(0)=a(1)=0\) 且中途非零，它与静止路径有相同起终分布，中间分布却不同。分类器可以完美利用某些路径差异，但那些差异不必对应需要修复的最终生成问题。

因此本候选只从干净端点抽样，然后对两来源使用**同一**通道

\[
Z_t=tX+(1-t)\epsilon,\qquad Y=X-\epsilon,\quad 0<t<1.
\tag{1}
\]

如果 \(P=G\)，它们在每个 t 的重新加噪分布必然相同，即使两种实际采样路径不同。共同高斯通道在总体精确分布层面具有可辨识性；但有限样本分类失败、少量时刻近似相同，都不能证明端点相同。

这也明确地区别于仓库中已试过的来源后验调节：旧方案曾比较实际强弱轨迹，并在线调 guidance 系数。这里来源标签定义在固定端点，标签只用于离线学习向量，不在采样时充当局部增益。[旧来源后验协议](IG_SOURCE_POSTERIOR_PROTOCOL_20260912_ZH.md)、[校准修订](IG_SOURCE_CALIBRATION_AMENDMENT_20260912_ZH.md)。

**可以精确定义一种“AG 有用信息”：来源条件下的后验位置差。** 令 B=1 表示真实端点，B=0 表示生成端点，两者先验各半；令 \(I=(Z_t,t,c)\)。定义

\[
\begin{aligned}
\eta(I)&=\Pr(B=1\mid I)=\frac{p_t}{p_t+g_t},\\
\mu_P(I)&=\mathbb E[X\mid I,B=1],&
\mu_G(I)&=\mathbb E[X\mid I,B=0],\\
v_P(I)&=\mathbb E[Y\mid I,B=1],&
v_G(I)&=\mathbb E[Y\mid I,B=0].
\end{aligned}
\tag{2}
\]

由式 (1)，共同高斯通道给出

\[
\boxed{
\delta(I):=v_P-v_G
=\frac{\mu_P-\mu_G}{1-t}
=\frac{1-t}{t}\nabla_z\log\frac{p_t(z)}{g_t(z)}
}.
\tag{3}
\]

因此我们寻找的不是“这张图像质量几分”或抽象的语义强弱，而是：**在当前噪声掩盖了细节之后，真实样本和生成样本仍然支持哪些不同的 clean 解释，以及这个差异指向哪里。** 它可能来自模式比例、形状、局部纹理或模型过量生成的组合；理论不预先把它等同于某个频段。

令混合条件均值 \(m=\eta v_P+(1-\eta)v_G\)。进一步有

\[
\boxed{
C(I):=\operatorname{Cov}(B,Y\mid I)
=\eta(1-\eta)\delta(I)
=\frac{1-t}{t}\nabla_z\eta(I).
}
\tag{4}
\]

这给出可操作的定义：来源标签与去噪目标的**条件协方差**直接对应一个引导方向。一般普通去噪回归估计 m 或某一来源的均值；它没有被直接要求估计这个对比。最后一句是任务定义上的区别，不是声称 MSE 完全无法间接承载该信息。

**这与信息论的联系不仅是命名。** 在等价的信噪比坐标 \(U_\gamma=\sqrt{\gamma}X+\epsilon\) 下，由 \(B\to X\to U_\gamma\) 的 Markov 关系，

\[
I(B;U_\gamma)=I(X;U_\gamma)-I(X;U_\gamma\mid B).
\]

将 I-MMSE 恒等式分别用于右侧两项，可得下面的直接推论；期望均取真实／生成各半的混合通道，均值与 \(\eta_\gamma\) 在该坐标重新定义：[^immse]

\[
\boxed{
\frac{d}{d\gamma}I(B;U_\gamma)
=\frac12\mathbb E\!\left[
\eta_\gamma(1-\eta_\gamma)
\|\mu_P(U_\gamma)-\mu_G(U_\gamma)\|^2
\right].
}
\tag{5}
\]

证明的剩余一步只是全方差公式：

\[
\operatorname{MMSE}(X\mid U_\gamma)
-\operatorname{MMSE}(X\mid U_\gamma,B)
=\mathbb E[\eta_\gamma(1-\eta_\gamma)\|\mu_P-\mu_G\|^2].
\tag{6}
\]

也就是说，**随着噪声减少而变得可辨识的端点来源信息，正好对应“知道来源后，最优去噪均值能改善多少”**。这比“浅层比较 coarse”多了一层约束：希望读出的应当是与端点来源差异相关的预测信息。式 (5) 是标准 I-MMSE 的应用，不是本文新发现的基础定理，更不证明某个 Transformer 层必然保存了它。没有计划为此增加一整套互信息局部探针。

**新 loss 的首选：用来源中心化回归直接学习 \(\delta\)。** 先离线估计 \(\hat\eta(I)\)。选择一个不看 B 的基线 \(b(I)\)，可先用冻结强场 S；若需要更好的方差控制，可单独拟合混合均值 \(\hat m\)。训练时全部停止梯度。小头 \(D_\phi(H(I))\) 直接预测向量差，使用

\[
\boxed{
\mathcal L_{\rm contrast}(\phi)=
\mathbb E\left\|
Y-b(I)-\big(B-\hat\eta(I)\big)D_\phi(H(I))
\right\|^2.
}
\tag{7}
\]

它仍使用 diffusion/FM 的随机训练对，却已不是“让 weak 也把 X 或 v 预测好”的同一目标。B 不是类别 c，而是端点来自哪个分布；小头推理时不接收 B。

为什么该 loss 对应式 (3)？在 \(\hat\eta=\eta\)、允许 D 访问完整 I 且二阶矩存在的总体情况下，

\[
\begin{aligned}
\mathbb E[(B-\eta)^2\mid I]&=\eta(1-\eta),\\
\mathbb E[(B-\eta)(Y-b)\mid I]&=\eta(1-\eta)(v_P-v_G).
\end{aligned}
\tag{8}
\]

故条件二次风险的最优解为 \(D^*=\delta\)。特别地，只要来源后验正确，b 可以有误差；这里不需要把 S 偷换成“自身生成分布的精确速度”。当小头只接收 H 时，理想来源后验下学到的是 \(\delta\) 在该函数族内、权重为 \(\eta(1-\eta)\) 的最佳平方误差逼近，不是自动得到完整 \(\delta\)。若分类器也只能见 H，来源后验的近似还会带来额外偏差。

这是 Robinson residualization / R-learner 的标准代数形式在端点来源对比中的应用。原方法来自条件效应估计；这里 B 是数据来源标记，不把“改变来源”解释成因果干预。[^rlearner]

**为什么不能只说“训练个真假分类器，然后取梯度”。** 判别器引导已经研究了真实／生成密度比；DLSM 又明确使用去噪监督约束条件 likelihood score，后续工作也分析 CE 很低但引导梯度仍错误的情况。[^dg][^dlsm][^improved][^sobolev]

本候选的实际区别是：分类器只提供训练期标量 \(\hat\eta\)，向量由已有主干特征上的小头直接回归；不在采样时通过分类器与主干反向传播，不增加第二个独立 diffusion 主干。分类器与可选混合均值头的拟合、交叉拟合会增加离线成本，不能宣称训练免费。

新颖性必须克制。对二来源分类，\(\nabla\log p(B\mid I)=(B-\eta)\nabla\operatorname{logit}\eta\)；代入 DLSM 后，会出现与式 (7) 很接近的残差结构。**因此“对比去噪 loss”不能作为已确认的全新发明。** 值得验证的剩余问题是：独立估计来源概率、直接向量读出、共享浅层特征，能否在既定 IG 计算预算下更有效地提取端点偏差。是否已有完全相同实现仍未确认，文献检索未发现同名方法不能当作首创证明。

**来源估计不准时，能说到什么程度。** 令

\[
a=\eta-\hat\eta,\qquad b_{\rm err}=m-\hat m,\qquad V=\eta(1-\eta).
\]

如果式 (7) 的基线用 \(\hat m\)，总体最优解满足

\[
D^*_{\hat\eta,\hat m}
=\frac{V\delta+a b_{\rm err}}{V+a^2},\qquad
D^*_{\hat\eta,\hat m}-\delta
=\frac{a b_{\rm err}-a^2\delta}{V+a^2}.
\tag{9}
\]

在来源重叠足够且两项 nuisance 都接近真实值时，误差是一项乘积加二阶收缩项。不能将它表述为“任意基线下均一阶鲁棒”，也不能把式 (7) 的 \(\delta\) 估计称为严格双重鲁棒：仅 m 正确而 \(\eta\) 错误时，仍有收缩。若直接用误差不小的 S 作基线，来源后验误差可能一阶进入。

有限样本必须按**干净端点**划分 nuisance 训练折和目标训练折，再各自反复加噪。同一端点的不同噪声版本分到两折，不能解决端点记忆。旧队列里的离线权重分类器在端点表征上区分来源，且标签 1 表示生成；这里需要每个噪声时刻的 \(\eta(I)\)，标签 1 表示真实。两者不能直接复用或混用符号。

**同一推导给出的稳健备选：回归协方差，避免除以接近零的重叠。** 如果 R-loss 在有限样本下对低重叠区域很不稳定，可以把估计对象明确换成式 (4)，使用

\[
\mathcal L_{\rm cov}(C_\phi)=
\mathbb E\left\|
C_\phi(H(I))-
(B-\hat\eta(I))(Y-\hat m(I))
\right\|^2.
\tag{10}
\]

其条件目标满足

\[
\mathbb E[(B-\hat\eta)(Y-\hat m)\mid I]
=V\delta+a b_{\rm err}.
\tag{11}
\]

这里的协方差目标具有明确的双重鲁棒性：任一 nuisance 精确，乘积偏差消失。采样可研究 \(S+4C_\phi\)，因为 \(4V\le1\)，在理想 \(S=v_G\) 的情况下它是朝向 \(v_P\) 的逐点插值。这个不等式限制的是相对修正倍数，不保证向量范数有界；也不证明该 ODE 精确采到 P。它有意放弃对低重叠区域的完整 \(\delta\) 修复，属于不同的保守目标。

这不是准备同时扫多种门控函数。先比较式 (7) 与匹配的普通 FM / 旧直接 guided-MSE；式 (10) 仅作为由估计困难导出的固定备选。若 nuisance 根本没有可泛化来源信号，应记录这个局限，而不是增加大量校准温度、层数与时间系数。

**它如何与原来的 AG 弱模型相接。** 最直接的实现为

\[
v_{\rm new}=S+D_\phi(H).
\tag{12}
\]

这是一个内部向量修正头。它不自动是“独立弱生成器的 score”。若坚持训练原来输出 W 的架构，可把式 (7) 中 D 替换为 \(\alpha(S-W_\phi(H))\)：

\[
\mathcal L_W=
\mathbb E\left\|
Y-b-(B-\hat\eta)\alpha(S-W_\phi(H))
\right\|^2.
\tag{13}
\]

理想最优条件是 \(\alpha(S-W^*)=v_P-v_G\)，最终采样仍为 \(S+\alpha(S-W)\)。两种参数化在有限头容量下并不等价：式 (13) 还要求小头表示 S 的相应部分，式 (12) 只拟合差值。不能靠形式上的 \(W=S-D/\alpha\) 就宣称构造出了一个独立、规范化且跨时间一致的弱分布。

这个区别也回答“头足够大后是否外推必然为零”：旧的同目标复制过程可能使 W 接近 S；新的目标只有在 \(P=G\) 时才要求该端点修正为零。**我们有意让监督对象不同，而不是靠永久欠拟合制造差值。**

仍有一项不可省略的采样近似。式 (7) 学习的是 \(v_P-v_G\)，但有限网络 S 不必等于 \(v_G\)。因此

\[
S+\delta=v_P+(S-v_G).
\tag{14}
\]

它不会自动消除网络场与自身端点重新加噪场的差异。只有在相应一致性、全时间使用以及精确积分等理想条件下，完整修正才可推出 P 的生成结果。现有 IG 时间窗口、有限步求解器与有限小头都会破坏这种直接保证。本文不从中间场关系跳到最终 FID 必然改善。

**与已经失败的 loss 的区别要实质化。** 仓库之前的 guided-MSE 使用真实数据训练 \(S+\alpha(S-W)\) 逼近 Y；其总体修正是 \(v_P-S\)。1K 中 IG direct 64.7947 对匹配 native 64.8497，仅有极小变化，且未超过当时另一控制；CFG direct 45.6553 也未超过匹配 native 45.4384。[旧实验](SIT_GUIDED_OBJECTIVE_RESULTS_20260912_ZH.md)。

式 (7) 使用真实和生成两来源、估计来源后验、中心化回归，目标是 \(v_P-v_G\)。它不是把旧的反射 target 改个名字。但式 (14) 同时说明，新目标也不是显然优于 \(v_P-S\)；潜在收益应来自差值估计的有限容量或泛化性质，必须由匹配的真实生成实验证明。二者应保留直接对照，不能删掉失败历史来制造新颖性。

**第三类样本——正在训练的弱模型端点——该怎样用。** 本轮建议先只用固定的 P/G 定义修正，保留弱模型端点作之后的固定一轮检验。若直接把弱模型端点 W 混入生成负例，分类器学习的就变成 P 相对于 \(Q=(G+W)/2\) 的差异；此时式 (3) 对应 \(v_P-v_Q\)，不能仍然写成 \(v_P-v_G\)。

若目标是修正“已经带 IG 的完整生成器”，更清楚的做法是冻结这个采样器，用其端点建立新的 G，再固定进行一轮训练与全新采样比较。每轮都有清楚的端点对象；不是一边训练弱模型一边无记录地改变负例分布。把第三类样本作为保持原目标不变的 proposal 则需要额外重要性比值，当前没有充分理由引入这项复杂度。

这条新 loss 属于 IG / 共享内部修正主线。把 B 换成类别再推导一次，会回到已知的条件 score 对比，不能作为纯 CFG 的新贡献。用实际 CFG 采样器的端点作为 G 可以研究后训练修正，但那应明确称为 CFG 加内部头，而不是训练免费的纯 CFG 改法。

**已经交付的代码与实验顺序。** 上轮的“保留原分布成分的平滑参考”与“增加生成过量部分权重的参考”已经实现，协议和源码冻结。SiT-S/2 先做固定 3K 训练、配对 1K 筛选；包含真实数据头、自生成头、原 Context、原 IG、纯 Gaussian 及半强度、类内打乱权重等相应控制。只有一个最优通过者进入独立 5K。推理每图仍为 128 次完整前向、无额外 prefix，新头仅读共享特征。[执行协议](WEAK_REFERENCE_LOSS_PROTOCOL_20260914_ZH.md)、[执行状态与结果](WEAK_REFERENCE_LOSS_RESULTS_20260914_ZH.md)。

队列先等待当前 RAEv2→IG+SG 监督任务，再等待一张 GPU 连续三次空闲。CPU 已核验原 FM 分支、Gaussian 条件监督恒等式、整图混合、正权重和梯度；共享特征、真实模型轨迹与调用次数的 GPU 核验是队列第一项，未完成前不声称实现已通过真实模型验证。后台启动后不需要本对话持续轮询。

新来源对比目标的 [loss 原型](../experiments/endpoint_contrast_20260914/objectives.py) 已实现式 (7)/(10)，可接现有向量头；nuisance 停止梯度。[代数核验](../experiments/endpoint_contrast_20260914/check_identities.py) 验证了式 (3)/(4)/(9)/(11)，并用独立积分核验式 (5)，误差约 \(1.65\times10^{-11}\)。[核验记录](data/endpoint_contrast_20260914/algebra_checks.json) 明确标记未做图像质量实验。这只是公式与梯度实现审计，没有进行 toy 模型训练。

真正值得下一轮验证的是一个窄假设：**在同一冻结表征和同一推理预算下，直接监督端点来源的预测差，比继续提高单来源去噪拟合更适合生成修正。** 若它在控制下没有收益，就不能继续把结果解释为“浅层隐藏着特殊信息”，而应放下这条具体目标。既有两个已授权候选先按固定队列运行，本轮新推导不会悄悄改写其比较对象。

**来源与适用边界**

[^vinfo]: Yilun Xu 等，[A Theory of Usable Information under Computational Constraints](https://arxiv.org/html/2002.10689)，ICLR 2020，定义与 §3。支持计算受限的可利用信息框架，不证明 IG 的层间机制。
[^repa]: Sihyun Yu 等，[Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/html/2410.06940)，§4.2 的 alignment depth。本文引用层间分工假设，不将论文机制迁移为本仓库已验证结论。
[^ssg]: Zixuan Fu 等，[A Frozen Pixel-Space Diffusion Model Can Guide Itself with Its Own Samples](https://arxiv.org/html/2607.29122v1)，Table 6 与 adapter capacity。表中 Rel. FLOPs 是 adapter 训练相对完整训练的代价，不能当作推理 FLOPs。
[^immse]: Dongning Guo、Shlomo Shamai、Sergio Verdú，[Mutual Information and Minimum Mean-square Error in Gaussian Channels](https://arxiv.org/html/cs/0412108v1)，IEEE TIT 2005，Theorem 2。式 (5) 是其向量结论与条件互信息链式法则的直接推论；要求有限二阶矩。
[^rlearner]: Xinkun Nie、Stefan Wager，[Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://arxiv.org/html/1712.04912)，§2，式 (1)–(4)。残差化、交叉拟合与 R-loss 已有文献；不将其因果识别假设套到来源标签上。
[^dg]: Dongjun Kim 等，[Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models](https://proceedings.mlr.press/v202/kim23i.html)，ICML 2023。真实／生成样本的判别器引导已有先例。
[^dlsm]: Chen-Hao Chao 等，[Denoising Likelihood Score Matching for Conditional Score-based Data Generation](https://arxiv.org/html/2203.14206)，ICLR 2022，式 (9)–(11)。本候选与其存在实质代数联系，不能只改实现就认领新的 score 理论。
[^improved]: [Improving Discriminator Guidance in Diffusion Models](https://arxiv.org/html/2503.16117v1)，§4–5，式 (15)/(17)，2025。本文只引用其 CE 与梯度误差的区别及去噪梯度监督，不将其收益外推到内部头。
[^sobolev]: Chenghan Xie、Jose Blanchet、Renyuan Xu，[Sobolev Regularized Score Difference Estimation in Diffusion Models](https://arxiv.org/html/2608.18237v2)，2026，score difference 的 Sobolev 估计。分类函数近似与梯度近似的误差需分别处理。

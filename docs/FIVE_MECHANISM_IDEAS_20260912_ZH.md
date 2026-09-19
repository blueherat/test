# 从弱分布和采样概率过程出发的五个假设

值得继续追问的不是“怎样修改guidance向量”，而是：**模型到底丢失了哪一种统计结构，为什么强弱密度的对比能够补偿这种丢失？对于CFG，怎样提高条件选择能力，同时保留生成过程中应有的随机性？**

这两个问题分别对应分布估计和采样过程。下面提出五个可失败的假设。每个假设先指定一个概率对象，再推导独立于FID的预测，最后给出干预和生成实验。它们并不是五个已证实的原创方法；其中59和62最值得优先投入，60和63需要较强的机制证据，61与既有条件对比方法较近。不能把五个实验编号等同于五项论文贡献。

现有实验尚不足以把IG归结为高斯平滑。此前低噪声depth4的热平移拟合解释约68.38%的gap能量，而只用原score与状态做两参数标定已经解释约65.50%；中间噪声某些点标定对照更好。[既有诊断](CFG_IG_RESEARCH_DIRECTIONS_20260912_ZH.md)说明，“看起来像noise shift”还不是独特机制。强弱误差共线也不作为下面任何结论的前提。

## 共同的概率边界

AG与IG提供了强弱模型对比的既有起点。[^ag][^ig] 设强、弱模型在同一噪声水平的理想密度为p与q。若其输出确实是这些密度的score，则

\[
g(x)=s_p(x)-s_q(x)=\nabla_x\log\frac{p(x)}{q(x)}.
\]

“沿强弱比大于1的地方走”不是严格的运动描述。比值大于1是当前位置的密度比较；g描述比值增大的方向。在比值小于1的地方g也可能有用，在比值大于1的地方g也可能为零。没有误差模型时，沿该方向走不能保证更接近真实数据分布。

固定噪声下，p的score加上αg是

\[
\nabla\log \rho_\alpha,\qquad
\rho_\alpha(x)=Z_\alpha^{-1}p(x)^{1+\alpha}q(x)^{-\alpha}.
\]

但把这一式子沿所有噪声水平送进ODE/SDE，最终样本一般不服从相应的终点乘幂密度。CFG的predictor-corrector分析、Feynman–Kac校正和Gibbs-like guidance已经明确讨论这一问题。[^pcg][^fkc][^gibbs] 因而本文把“瞬时密度比的解释”和“实际采样分布的结论”分开；下面的静态界不冒充最终FID保证。

更根本的假设形式是

\[
q_t \approx \mathcal C(p_t),
\]

其中粗化算子必须有明确含义。至少应回答：它保持哪些统计量，删除哪些信息，是否与前向加噪相容，在哪些分布上完全不起作用，以及怎样把它与高斯平滑、普通训练不足区分开。高斯卷积只是其中一种选择。Self-Guidance已经直接使用原模型在不同噪声水平的对比，不能把这种做法重新命名为新贡献。[^sg]

## 59：弱模型丢失的主要是高阶结构，而非均值或协方差

**机制假设。** 有限表示能力可能较早学会一、二阶统计，却较晚学会多峰结构、偏斜、尾部与高阶依赖。这样得到的弱分布可以与强分布具有相同均值和协方差，仍然把有意义的结构“平均掉”。这比“弱模型多了一点噪声”更具体，也可能解释为什么简单幅度标定会冒充平滑拟合。

设X₁,…,Xₘ独立同分布于一个类别的p，均值为μ。定义

\[
\mathcal C_m p=
\operatorname{Law}\!\left[
\mu+\frac{1}{\sqrt m}\sum_{i=1}^{m}(X_i-\mu)
\right].
\]

这不是通常的mixup。mixup除以m，因此协方差变为Σ/m；这里除以√m，因此

\[
\mathbb E[\mathcal C_m X]=\mu,\quad
\operatorname{Cov}(\mathcal C_m X)=\Sigma,\quad
\kappa_r(\mathcal C_m X)=m^{1-r/2}\kappa_r(X),\ r>2.
\]

其中κᵣ是r阶累积量张量。m=4时，三阶累积量减半，四阶累积量减为四分之一；任何Gaussian分布都是不动点。中心极限定理与熵收敛提供了既有数学背景，但这些数学事实本身不是新的guidance理论。[^clt]

这个算子还有一个很强的约束：它与独立Gaussian加噪相容。令H_q表示加方差q的噪声，则

\[
\mathcal C_m(H_qp)=H_q(\mathcal C_mp).
\]

原因是m个独立噪声相加再除以√m，噪声协方差仍是qI。因而可以在干净数据上定义一个固定弱分布，训练它在所有时间的速度；不必在每个时间另外拟合一个不相容的“弱密度”。

**能解释什么。** 如果strong仍存在高阶结构估计不足，而shallow head更严重，那么二者的对比可能强化尚未充分形成的结构。低噪声时strong已能够恢复这些结构，而weak仍有高阶缺失，此时gap可以不小，但继续强化不再有益。最后一句包含“strong偏差随噪声变化”的额外假设；它不能仅由算子恒等式推出。

**一个必须保留的反例。** 两个非常窄的对称峰经过m=2变换，会在−√2a、0、√2a产生三个峰。因此“模态数单调减少”不成立，“CLT变换总能改善guidance”也不成立。本轮解析检查已经复现2峰变3峰。真正的预测是高阶累积量缩小、低阶统计保持；强弱比能否对应质量修复仍需实验。

**干预。** 用m=4构造训练样本，短训一个完整弱模型。强模型冻结。与同初始化、同训练步数的native弱模型、普通四图平均、均方扰动匹配的Gaussian加噪弱模型共同比较。若clt与普通mixup都有效而差异很小，主要解释可能是数据扰动或弱模型难度；若Gaussian加噪更好，则不支持高阶结构路线的独特价值。

**关键失败条件。** 即使弱模型已经在互斥验证数据上学会clt目标，仍无法比同预算native/heat/plain_mix带来更好的质量—成本关系，或真实IG的gap并不更像这类粗化，则停止把它当IG机制。不能用“训练MSE下降”替代生成质量证据。

**与旧工作的区别。** 这里改变数据的高阶统计，而不是移动噪声水平、给score加滤波器，或修改attention温度。最接近的是SG的噪声对比、AG的容量退化，以及一般mixup。实质差异只能来自“保持二阶统计的具体算子能预测与解释guidance成败”，不是平均操作本身。

## 60：弱模型遗漏的是区域之间的依赖

**机制假设。** 一张图的多个局部区域可以分别合理，组合起来却不构成合理对象。浅层预测可能充分利用局部信息，却不足以建模区域之间的关系。如果如此，weak的关键错误不是低分辨率，也不是总体方差，而是联合分布中的依赖关系不足。

把潜变量分成固定位置的区域X_A₁,…,X_Aₖ，定义

\[
\mathcal Fp(x\mid c)=\prod_{j=1}^{K}p(x_{A_j}\mid c).
\]

它保留每个区域在数据总体中的精确边际，而移除区域间依赖。相应的密度对比是

\[
\log p(x\mid c)-\log\mathcal Fp(x\mid c).
\]

这是逐点的依赖对比，其在p下的期望是total correlation。不能把逐点值都当成正数，更不能从期望的非负性推出逐样本guidance一定有效。

在独立的前向Gaussian加噪下，

\[
\mathcal F(H_qp)=H_q(\mathcal Fp),
\]

因为加噪不把独立区域重新耦合。这提供了一个跨时间一致的弱数据分布。最简单的二元例子是：两个变量各自正负等概率，但同号联合概率为.9。factor分布保留两个边际，四种组合各.25；密度比强化同号、抑制异号。任何只匹配单变量均值和方差的解释都看不到这项区别。

**干预。** 四个同类独立样本各提供一个固定位置的16×16潜变量象限。这样每个象限的总体边际保持不变，而象限来源独立。训练弱模型去拟合这个已知分布，再使用普通强弱对比采样。与既有#40局部attention弱参照及同预算native弱模型比较。

这项实验并不声称接缝是合理图像，也不认为每张合成图的局部细节被保持；它故意制造“局部边际合理、联合关系错误”的负分布。若改用随机位置拼接，空间边际也改变，会混入另一个机制，因此本轮固定区域位置。

**预测。** 如果IG主要在修复长程依赖，已学会factor分布的weak应能提供与浅层IG相关的修正；受益应更集中于结构组合，而非单纯颜色方差。单独的FID改善只能说明方法可能有用。还需要对象结构与局部边际的分离证据，才能声称找到了机制。

**失败条件与近邻。** 如果收益可以被局部attention退化、普通模型短训或边界伪影解释，这条路线不构成新贡献。PAG、局部attention、token稀疏引导已经改变信息交换；它们必须作为近邻工作，而不是被忽略。[^sparse] 本轮的不同点是用明确的训练分布消除依赖，进而检验“遗漏哪类信息”这一因果假设。

## 61：CFG的负参照应表达条件的不确定性

**机制假设。** null条件把所有类别汇总在一起，但实际生成错误可能集中在少数容易混淆的类别之间。与远离当前类别的负参照对比，可能改变大量与区分类别无关的统计特征。需要检验的是负分布的类别拓扑，而不是在向量空间搜索一个更好的投影。

构造一个类别Markov核Q，定义

\[
q(x\mid c)=\sum_d Q_{cd}p(x\mid d).
\]

本轮把100个类别确定性地配成50对，P表示互换配对类别的置换，令Q=(I+P)/2。Q双随机，所以在均匀类别先验下

\[
\frac1{100}\sum_cq(x\mid c)=
\frac1{100}\sum_cp(x\mid c).
\]

总体图像边际没有改变，变化只在条件之间如何共享概率质量。条件混合也与前向加噪交换，因此存在一个固定的弱训练分布。

两类情形揭示了它与“拿另一个类别做负向量”的区别。令q=(p_c+p_d)/2，则

\[
s_q(x)=r_c(x)s_c(x)+r_d(x)s_d(x),\qquad
r_d(x)=\frac{p_d(x)}{p_c(x)+p_d(x)}.
\]

这个权重是概率混合必然产生的责任概率，不是人为设置的分量系数。直接平均两个score通常对应几何平均，而不是密度混合。本轮通过训练一个完整模型表示q，不在采样时用局部指标拟合r。

**干预。** 比较由原模型类别embedding近邻形成的配对、随机配对，以及直接用近邻类别score作负参照的hard-pair对照。所有图像类别先验和弱模型训练预算保持一致。近邻表固定于生成之前；embedding近邻是否确实对应视觉混淆，是待检验条件。

**解释性预测。** 如果有效性取决于“局部类别不确定性”，近邻概率混合应优于随机配对；若只是hard-negative方向有效，直接hard-pair可能同样好甚至更好。如果所有方法都只复现已知ICG，则没有新增机制。

**新颖性风险较高。** ICG已经研究独立条件作为负参照；一般层次条件与负提示同样相关。[^icg] 本轮可能的差异是双随机类别粗化、总体边际保持，以及实际训练概率混合而非随机替代score。没有这些干预证据，不能把它包装成新的CFG理论。

## 62：CFG丢失的随机性应在正确的生成坐标中恢复

**机制假设。** CFG不仅选择条件，也会收缩已有随机性。直接给中间潜变量加Gaussian噪声，通常不保持条件分布；它可能把合理结构推离模型的生成分布。因此问题不是“加多少噪声”，而是选择一个在正确条件分布上本来就平衡的随机转移。

令Φ_t^c把标准Gaussian初始噪声映射到仅用条件模型生成的时间t状态，定义其实际推送分布

\[
\mu_t^c=(\Phi_t^c)_\#\mathcal N(0,I).
\]

这里使用的是模型的实际条件生成流，不要求它等于真实数据加噪分布。假设该流可逆，先把状态映回初始噪声坐标，执行标准OU更新，再推回：

\[
u=(\Phi_t^c)^{-1}(x),\qquad
u'=\rho u+\sqrt{1-\rho^2}\,\xi,\qquad
x'=\Phi_t^c(u'),\quad \xi\sim\mathcal N(0,I).
\]

由于OU保持标准Gaussian分布，这个共轭转移K_t保持μ_t^c；ρ=1时精确过程为恒等变换。更一般地，若ν≪μ_t^c，Markov核的数据处理不等式给出

\[
D_{\rm KL}(\nu K_t\Vert\mu_t^c)
\leq D_{\rm KL}(\nu\Vert\mu_t^c).
\]

因此它能减少相对于未加强条件生成分布的偏离。这不是FID改善定理：CFG的一部分偏离本来就是有益的，K_t也可能抹掉这部分收益。值得检验的假设是，适量恢复随机性是否比在像素/潜变量坐标直接扰动更少损伤结构。

**干预。** 在固定t=.5处插入一次共轭更新，ρ固定为.9或.97；其余使用普通CFG。反向和正向流均为32步RK4。另设置ρ=1的完整往返对照，以分离数值往返误差；设置未加CFG时的更新，检验分布保持近似；设置具有同一类均值及完整经验协方差的Gaussian更新，检验非线性生成坐标是否提供超出二阶统计的价值。

Gaussian对照不是逐样本RMS匹配，也没有投影guidance分量。它用训练池固定的低秩经验协方差加VAE posterior对角方差，定义另一个全局Markov核。仿射Gaussian生成器下，两种正确构造应该一致；非Gaussian生成器才可能产生可辨别的差异。

**失败条件。** 若ρ=1往返误差足以解释结果，停止将变化归因于随机性恢复；若普通Gaussian更新或增加主采样计算同样有效，不能声称非线性坐标必要；若它只提高多样性却降低总体质量，最多是另一种质量—多样性权衡。

**与已有工作的边界。** Gibbs-like guidance、PCG已经研究随机校正；noise-space采样与NoiseRefine也已存在。[^gibbs][^pcg][^noise] 本轮不能认领“加噪声”“去噪重采样”或“操作初始噪声”本身。可能的新意在于：以冻结条件生成流定义具有明确不变分布的中间随机核，并用完整协方差对照分离二阶与非线性结构。一个核需要256次额外完整前向，成本必须公开，暂时不是便宜方法。

## 63：弱模型的危险可能来自密度空洞

**机制假设。** “弱模型更差”包含两种完全不同的错误：它可以覆盖更宽，也可以遗漏合法模式。对遗漏合法模式的weak做强弱比引导，会把weak的盲区当成特别值得强化的位置。强弱模型的局部误差不必共线，gap的范数也不能识别这两种情况。

令b为一个退化分布，定义包含原分布的weak

\[
q_\varepsilon=(1-\varepsilon)p+\varepsilon b,\quad 0<\varepsilon<1.
\]

则q_ε≥(1−ε)p，因此

\[
\frac{p}{q_\varepsilon}\leq\frac1{1-\varepsilon}.
\]

在固定噪声水平，令ρ_α∝p(p/q_ε)^α。由Jensen不等式，

\[
Z_\alpha=\mathbb E_p\exp\!\left(\alpha\log\frac p{q_\varepsilon}\right)
\geq\exp\big(\alpha D_{\rm KL}(p\Vert q_\varepsilon)\big)\geq1,
\]

所以

\[
\frac{\rho_\alpha}{p}\leq(1-\varepsilon)^{-\alpha},
\qquad
D_{\rm KL}(\rho_\alpha\Vert p)\leq-\alpha\log(1-\varepsilon).
\]

这给出一个明确机制：weak无法仅因漏掉某个合法模式而给它无限大的相对强化。它不保证不删除模式，不给出score差范数界，也不保证离散生成轨迹或FID。density ratio有界与梯度有界并不是一回事。

**干预。** 固定ε=.5，把真实训练分布与clt退化分布按整张图混合，训练单个弱模型；与clt-only、同预算native弱模型比较。采样时使用这个完整弱模型，不把两条score做固定权重平均。若mixed weak只把gap普遍缩小，其效果应能由clt-only的强度曲线解释；只有弱模型盲区与尾部异常得到针对性修复，才支持密度空洞假设。

**学习误差边界。** 上式严格成立于理想p和q_ε。实际强模型的生成分布不等于真实训练池，短训weak也不精确，因此本轮不能声称已获得该界的数值认证。可以先检验因果预测；如果需要正式保证，后续必须控制两模型相对真实混合分布的密度误差。

**新颖性边界。** defensive mixture是重要性采样中的成熟思想，以上不等式不是原创数学。[^defensive] 可能的研究贡献是证明AG/IG的某类失败确实来自弱分布盲区，并通过显式训练分布干预加以修复。63和59共享clt退化资产，不能声称它们是完全独立的证据或把同一正结果重复计数。

## 从假设到可审阅实验

本轮八个弱模型均从同一small SiT 800K EMA初始化，完整短训1500步，训练图25600、互斥验证图3200，采用固定最终EMA。所有训练分布、种子、中心、类别配对和数据哈希在生成FID前固定。[训练协议](SIT_MEASURE_GUIDANCE_TRAINING_20260912_ZH.md)给出完整预算和入口修订。

生成阶段使用既有配对1K输入，以便与已有CFG/APG曲线比较，并重跑几个关键基线以检查入口复现。59、60、63首先是AG/IG机制干预：使用独立完整弱模型，需要第二次主干前向，不能直接宣称是低成本IG。61和62属于独立CFG方向。对四个训练候选与四个训练对照使用相同强度网格，62另有两个固定相关系数及必要转移核对照。不会因某个1K较好就自动追加5K。

每项判断同时看三层证据：

1. 概率对象和实现是否相符：数据干预的不变量、实际输出形状、零强度退化、旧基线逐元素复现及全部计算计数。
2. 机制是否得到区分：干预模型在自己的互斥分布上是否学会目标；是否优于普通短训、高斯加噪、普通平均、随机类别拓扑、数值往返或Gaussian转移等相应替代解释。
3. 生成是否值得投入：同源强基线的调优曲线、实际训练与推理成本、FID/sFID/IS，以及后续独立样本确认。小样本最低FID不是机制证据或统计显著性。

解释性的理论必须允许出现这样的结果：公式都对、实现也对，但真实模型不符合假设，因此方向失败。若只剩“某个参数点FID更低”，只能保留一个经验候选，不能反过来宣称它证实了上面的故事。

## 来源

[^ag]: Tero Karras et al. [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/html/2406.02507v1), 2024，本文阅读v1。强弱模型对比与数据分布几何的既有起点。
[^ig]: Xingyu Zhou et al. [Guiding a Diffusion Transformer with the Internal Dynamics of Itself](https://arxiv.org/abs/2512.24176), 2025–2026。内部中间监督与IG的既有工作。
[^sg]: Tiancheng Li et al. [Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/html/2412.05827v5), v5，2025-09-26。跨噪声水平对比及SG-prev。
[^clt]: Andrew R. Barron. [Entropy and the Central Limit Theorem](https://www.stat.yale.edu/~arb4/publications_files/EntropyAndTheCentralLimitTheoremAnnalsProbability.pdf), The Annals of Probability 14(1), 1986, 336–342。概率粗化的数学背景；本文使用的累积量和加噪交换关系另作直接推导。
[^pcg]: Arwen Bradley and Preetum Nakkiran. [Classifier-Free Guidance is a Predictor-Corrector](https://machinelearning.apple.com/research/classifier-free-guidance), 2024–2025。CFG与乘幂终点分布不等价，及随机校正视角。
[^fkc]: Marta Skreta et al. [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://arxiv.org/abs/2503.02819), 2025。跨噪声目标与SMC校正，列作已存在路线。
[^gibbs]: Badr Moufad et al. [Conditional Diffusion Models with Classifier-Free Gibbs-like Guidance](https://arxiv.org/html/2505.21101v1), v1，2025-05-27。CFG缺失项与Gibbs式更新。
[^icg]: Seyedmorteza Sadat et al. [No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models](https://arxiv.org/abs/2407.02687), 2024，ICLR 2025版本。独立条件与time-step guidance。
[^sparse]: Felix Krause et al. [Guiding Token-Sparse Diffusion Models](https://compvis.github.io/sparse-guidance/), 2026，作者项目页。以token信息减少构造弱参照的近邻路线。
[^noise]: Donghoon Ahn et al. [A Noise is Worth Diffusion Guidance](https://openreview.net/pdf?id=xEWooSOgaz), ICLR 2026。初始噪声上的guidance蒸馏，列为已有工作，不计作本文候选。
[^defensive]: Art B. Owen et al. [Optimal mixture weights in multiple importance sampling](https://artowen.su.domains/reports/optwtsmis.pdf)，作者预印本中defensive mixture段落；全文下载受限，本文只将其作为已有思想的来源，所用界已在正文完整推导。

文献检索还发现[高维CFG形变研究](https://arxiv.org/abs/2602.00716)，其v4于2026-05-08更新，已讨论方差收缩与负引导窗口。本文没有把“CFG造成方差收缩”或“改变schedule”计作新贡献。当前检索没有证明以上五个具体构造均未被发表；其新颖性须围绕最接近工作及实验证据继续收窄。

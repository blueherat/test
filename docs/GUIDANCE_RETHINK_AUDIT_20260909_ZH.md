# Guidance 新方向筛选笔记

目前没有筛出同时具备明确方法、可信改进动机和足够新颖性的候选。“条件满足后减弱引导”与“只改变指定条件、保持其他内容”已撤下。本笔记保存重新推演时得到的排除依据，不把问题表述当作方法，不构成图像实验的正结果，也不改变已有 PFR、IG、lifting 实验的状态。

研究中的一个核心区分是：定义了一个合法目标分布，不等于证明这个目标比基础模型更接近真实图像分布；准确实现了这个目标，也不等于改善 FID。仓库已经存在局部 NLL 改善而 FID 恶化的解析反例，以及多种局部校准无法转化为生成质量收益的记录。新的方法必须说清它要改变哪一种生成错误，以及为什么这个改变与图像质量有关。[本地理论与实证边界](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)

下面四个入口都经过了初步检索与数学筛选。它们还不能作为新方法提交。

| 入口 | 最初吸引力 | 直接障碍 | 判断 |
|---|---|---|---|
| 把信息放进样本之间的耦合 | 改变整批生成的组织方式，跳出单点向量外推 | 保持单图边缘时，单图总体分布不变；粒子 guidance 已有系统研究 | 不能把批次多样性当作单图质量突破 |
| 显式选择生成模式，再细化图像 | 将模式选择与模式内生成分开，避免 guidance 同时压缩二者 | Kaleido 已有显式模式选择与辅助 latent 的路线；模式内锐化也未必提高质量 | 大方向有先例，尚无差异充分的新算子 |
| 引导跨区域依赖，而非局部纹理 | 局部都像真图的组合仍可能整体不合理；可规定明确的独立参考分布 | SWG 已增强长程依赖；联合与边缘分布的比值引导也有先例；增强依赖会改变局部边缘 | 不是一个靠互信息命名即可成立的新 idea |
| 在模式之间修正 score 的曲率 | 试图减少不合理的模式插值，不额外训练弱模型 | Laplacian Score Sharpening 已给出直接有限差分方法；曲率不是普遍的错误标签 | 不将局部二阶查询重新包装成突破 |

以上文献重合的依据分别见 Particle Guidance、Kaleido、SWG、Joint Diffusion Sampling 与 Laplacian Score Sharpening。[^1][^2][^3][^4][^5] 本次对 Kaleido 只核对了摘要和模式选择的公开说明，不声称完成其模型与实验的全文审计；其存在已足以排除“首次将模式选择单独建模”这一表述。SWG 的正式版本和源码差异已有[仓库精读](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md)。

**耦合为什么不能免费改善单图分布。** 设批次为 \((X_1,\ldots,X_n)\sim Q\)，每个边缘都等于固定分布 \(p\)。对任意可积单图函数 \(f\)，仍有

\[
\mathbb E_Q\frac1n\sum_i f(X_i)=\mathbb E_p f(X).
\]

改变耦合可以改变覆盖、重复率和估计方差；但没有改变任意一张图的边缘律，因此没有改变总体单图 FID 所比较的那个分布。有限样本 FID 是非线性统计量，其数值可能变化，不能据此宣称单图生成规律更好。若允许改变边缘，则必须重新说明目标与质量的关系。这是基本概率推导，不是对粒子方法的整体否定。仓库已有[PG 与保边缘粒子 guidance 的完整审查](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md)。

**把条件写进保持高斯分布的噪声变换，也有一个识别边界。** 若固定条件 \(c\) 下的变换 \(T_c\) 满足 \((T_c)_\#\mathcal N(0,I)=\mathcal N(0,I)\)，而生成器 \(F\) 不读取 \(c\)，那么

\[
\operatorname{Law}(F(T_c(\epsilon)))=F_\#\mathcal N(0,I)
\]

不依赖 \(c\)。例如只依赖类别的正交旋转无法通过一个不读取类别的固定生成器创造类别条件分布。如果 \(T_c\) 还根据噪声自适应选择，正交矩阵的逐点正交性不保证整个变换保高斯。若生成器本身读取条件，则其条件输出当然可以不同，但此时不能把这种不同归因于该保分布旋转。这说明“信息放在噪声的某个位置”必须有可识别的概率后果，不能只靠逐样本轨迹变化来支撑。

**增强依赖的确能写成方法，但尚没有质量理由。** 对固定坐标分块 \(x=(x_1,\ldots,x_m)\)，设参考分布为

\[
q(x\mid c)=\prod_j p_j(x_j\mid c),\qquad
R(x,c)=\log\frac{p(x\mid c)}{q(x\mid c)}.
\]

在 \(p\) 下取期望得到条件总相关性；单张图上的 \(R\) 是点态对数比值，可能为负，不能直接称为“这张图的互信息”。可以提出倾斜目标

\[
r_\beta(x\mid c)\propto p(x\mid c)e^{\beta R(x,c)},
\]

但它有三个未解决的问题。第一，真实数据中的关联不全是需要增强的关联：背景捷径、常见姿态、颜色共现同样会贡献比值。第二，即便参考精确保持各块边缘，倾斜后的分布一般也不保持它们。第三，在每个噪声层直接组合 joint 与 marginal scores，并不自动采样上述端点倾斜目标；这又回到仓库已讨论过的加噪与非线性倾斜不交换问题，而不是一个新的半群理论。[^6]

一个四状态反例足以检查第二点。令 \((X_1,X_2)\in\{0,1\}^2\)，按 \(00,01,10,11\) 排列：

\[
p=(.6,.1,.1,.2),\quad q=(.49,.21,.21,.09).
\]

取 \(\beta=1\)，归一化 \(p^2/q\) 得

\[
r=(.57651246,.03736655,.03736655,.34875445).
\]

第一坐标为 0 的概率由 .7 变成 .613879。对称例子 \(p=(.45,.05,.05,.45)\) 的边缘恰好不变，但这是对称性带来的特殊情况，不能推广。解析数值保存在[检查结果](data/guidance_rethink_20260909/analytic_checks.json)，只用于反例，不是生成质量实验。

**模式选择与模式内质量需要不同证据。** 假设各模式支撑互不重叠，且 \(p(x)=\sum_k a_kp_k(x)\)。端点温度变换 \(p^\beta\) 会让模式权重变成

\[
\widetilde a_k\propto a_k^\beta\int p_k(x)^\beta\,dx.
\]

因此锐化不仅改变模式内部，还会同时偏好原本质量更大的模式与更集中的模式。可以通过逐模式归一化构造保权重目标，但真实图像的模式定义、归属与归一化常数都未知。更根本地，若 \(p\) 已经是真实目标，锐化它没有普遍的分布质量改善保证。不能仅以“减少模式插值”作为生成正确性的替代指标，也不能把所有新颖的组合都判定成错误。

**为什么不再承诺统一解释所有弱参考。** 设当前预测误差为 \(e\)，候选引导方向为 \(d\)。平方误差的一阶变化由 \(\langle e,d\rangle\) 决定，而强弱差值本身不提供真实 \(e\)。方向相同、弱模型更差、或者局部预测更自洽，都不是识别该内积符号的充分条件。进一步，从局部风险到端点质量还需要传播与目标关系。已有 AG/SWG 的特定误差族有解释价值，但不能当作所有 IG、AG、注意力扰动方法的共同定律。[仓库误差兼容性审查](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md)

对未知方法，当前可合理要求的是明确的、可失败的改进预测，而不是不存在的普遍质量定理。一个候选应能说明：普通 guidance 留下了什么具体错误；哪个新算子能改变它；什么结果能把这个解释与单纯增加 guidance 强度区分开；取得这种区别需要多少真实计算。理论用于约束设计和识别失败，不能取代与调优基线的生成质量比较。

本轮没有形成达到这些要求的新方法，也没有新增 GPU 训练或图像采样。结论仅限于上面的重合与反例；没有声称穷尽 guidance 的研究空间。尤其不能把这些排除依据再变成一套无限机制检测清单：下一项值得报告的进展应当是一个具体方法及其独立改进理由。

[^1]: Corso et al., [Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/612a7948f3294a02a63d970566ca8536-Paper-Conference.pdf), ICLR 2024，特别是联合目标与保边缘讨论。
[^2]: [Kaleido Diffusion: Improving Conditional Diffusion Models with Autoregressive Latent Modeling](https://arxiv.org/abs/2405.21048), 2024；本次仅作模式选择路线的先例核对。
[^3]: Adaloglou et al., [Guiding a diffusion model with itself using sliding windows](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf), BMVC 2025。
[^4]: Raymond et al., [Joint Diffusion Sampling via Positive-Unlabeled Guidance for Multi-Modal Data](https://openreview.net/pdf?id=1o89PN84NH), ICML 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences；多模态联合采样先例，不是 ImageNet 生成质量结论。
[^5]: [Laplacian Score Sharpening for Mitigating Hallucination in Diffusion Models](https://arxiv.org/html/2511.07496v1), 2025 预印本，§3 的向量 Laplacian 与有限差分方法。
[^6]: Bradley and Nakkiran, [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/html/2408.09000v2), arXiv 2024；局部幂乘 score 与实际 CFG 终点律的区别。相关问题也已见[本仓库半群笔记](SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md)。

文献来源为上述六项。它们分别支撑特定先例与适用边界；本笔记中的有限状态例子、保分布变换结论、模式权重计算为独立推导，不是论文所报告的图像实验。

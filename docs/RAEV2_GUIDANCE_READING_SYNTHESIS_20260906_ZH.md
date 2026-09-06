# RAEv2 guidance：扩展阅读与设计判断

**最终边界：按用户要求已完成四轮，在第4轮后收束；公平总成本≥5%目标未实现，全部研究与数据索引进入Git归档。** 见[最后五轮台账](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md)。阅读应改变机制判断和下一项决定性检验，不以篇数替代进度。

**最后两篇阅读带来明确的停止判断。** [FP-Diffusion与第3轮](RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md)的Gaussian族证明：RAE有限时间完全加噪边界下，score PDE自洽性不能单独识别真实数据目标；原论文的真实边界与唯一性假设仍有价值。[Moser Flow与第4轮](RAEV2_GUIDANCE_READING_MOSER_20260906_ZH.md)则提供正确的有限密度source结构，但正密度圆例展示，有限Galerkin在规定路径上的方程成立不保证实际路径相同，经验Gram还可能不可识别。两篇对应固定CPU裁决和独立复核，没有追加训练或FID。至此累计52篇一手论文；阅读范围逐篇注明，未把篇数视作成功指标。

**由阅读继续推导的设计结构：有限步配对桥流。** [理论与独立核验](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)把原生 Euler 新制造的边缘失配写成有限 source，再以真实 teacher 一步输出与真实下一时刻状态的配对插值定义辅助条件速度。精确校准后，相对真实边缘的 KL 不放大；可逆时是等号，不能宣称消除了已有误差。条件残差 covariance 的散度解释了为何起点均值修正为零时，有限流仍可非零。它同时涵盖有限模型均值偏差，不等于再做一次旧 X−G 起点回归。固定辅助网络已完成训练、验证与实际 rollout：速度回归未胜过零预测，有限边缘统计有部分正面信号，固定1K现已完成：candidate38.316312相对推理成本匹配official107的38.372179仅改善0.145592%，未达标，当前实现停止。见[筛查结果](RAEV2_PAIRED_BRIDGE_SCREEN_RESULTS_20260906_ZH.md)、[机制实验结果](RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md)与[固定 1K 筛查协议](RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md)。允许固定预算的小规模机制实验，不以完整高维 KL 可估计性作一刀切门槛。

**最新实验反馈：仿射反射的固定1K筛查已完成并结束。** 官方100、反射100、成本规则选定的官方201之 FID 分别为 `38.2515918863 / 38.2886408205 / 38.5581217308`。候选相对较贵201改善0.698895%，但比更便宜的原100恶化0.096856%；这不是成功证据。独立重算FID、配对输入、批次像素与成本选择通过。保留群平均的理论结构和实际BF16不变性边界，不追加规模或搜索参数。这个决定不是将1K≥5%恢复为必要条件。见[完整结果与研究投入](RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md)。

日期：2026-09-06。响应用户“必须多找论文读，好的论文可以给你启发”。本轮扩展阅读已覆盖五十二篇一手论文，范围以各笔记明确列出的正文、证明和实验附录为准，范围扩展到了基础分布输运、去噪估计与流形几何理论。未将摘要浏览记作深读，也未把论文中的改进自动迁移成 RAEv2 结论。

**Stochastic Interpolants 进一步连接了误差目标和分布动力学。** [JMLR 正式版、证明及作者代码精读](RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md)允许相关端点配对；新增独立 Gaussian 噪声提供内部平滑和 score 标签。相同正扩散的两条动力学有精确 Fisher 耗散项，因此可用准确边缘上的漂移误差控制 target-to-model KL；无扩散的当前确定性桥不能直接继承。最优扩散比例依赖真实 excess-risk，不是两份含不可约噪声的训练loss比值。正式版转换到二次损失时缺少因子2，固定Gaussian反例与独立复核支持更正，但耗散机制及最优比例仍保留。数值例有正面生成证据，Flowers的不同扩散却使用不同Heun步数，正文未报告系统FID比较；不据此改当前冻结实验。28项归档已由root逐项核验。

**补读原始 Flow Matching，核查当前结构真正继承了什么。** [正文、三个定理及数值附录精读](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md)区分 conditional regression 与有限网络实现：固定、非独立的 (Y,W) 配对仍可由弱连续性和 L2 正交分解支持桥速度回归，但不能直接照搬原文的正密度假设，也不自动保证 ODE 唯一性或 midpoint 精确运输。配对残差可跨 τ 复用，节省 teacher 调用；论文没有给出标签方差减少的保证。当前风险相对零预测为负，必须保留这个实证反证，不能用 CFM 定理替代泛化证据。17 项原文及笔记归档已逐项独立核验。

**两篇新增论文补足实际分布反馈的估计结构。** [Sobolev score-difference / Discriminator Flow 精读](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md)分别说明：分类器的函数值与输入梯度需要同时控制；负例要随当前生成轨迹刷新。前者的 CE＋Dirichlet 目标给出受限函数类中的 H¹ 控制及有限正则偏差，九组解析分布的梯度 MSE 均有改善；后者官方代码确实逐批生成当前路径状态，不依赖“终点再加噪等于实际中间分布”的假设。但图像 DF 使用 IPM critic，不能当作校准的 log 密度比；Sobolev 的公开 ECG 代码也与正文 noisy CE 目标及 epsilon 尺度不完全对应。只保留梯度正则与分布刷新这两个设计依据，不照抄手选幅度或忽略训练采样成本。两篇未提供当前 RAEv2 的质量收益证据；原文、关键证明和固定版本代码共 102 项归档已独立校验。

**三篇进一步把正确结构与真正偏差分开。** [When Scores Learn Geometry](RAEV2_GUIDANCE_READING_SCORE_GEOMETRY_20260906_ZH.md)明确区分几何恢复与密度恢复所需的误差尺度；温度缩放可在条件下留下几何、消去密度漂移，但其目标是内禀均匀，并非自动改善真实数据FID。[Manifold Data](RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md)保留有限噪声的曲率耦合：切向后验二阶矩经第二基本形式产生正常的法向均值偏移，系数1/2来自几何展开。圆例说明固定off-manifold点的posterior Jacobian趋向投影的导数，不普遍趋向正交切投影。[DPS的Feynman–Kac分析](RAEV2_GUIDANCE_READING_DPS_REACTION_20260906_ZH.md)则追踪这种局部近似怎样积累为终点密度偏差；独立扩展显式加入有限denoiser的跨噪声PDE残差，避免照抄oracle协方差公式。三篇的证明、数值设定及版本问题分别记录，不重启已失败的投影、value或粒子权重路线。

**基础去噪理论给出了有来源的尺度与局部结构。** [Vincent 2011 / Alain–Bengio 2014](RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md)连接观测噪声、Bayes 重建与密度导数：固定 Gaussian 协方差 Σ、同空间 L2 最优重建满足 `r−y=Σ∇log p_Y`、`J_r=Cov(X|Y)Σ⁻¹`。这不是任意 autoencoder 的性质，也不要求 Jacobian 是收缩或幂等。非线性 encoder 不与条件期望交换，笔记中的完整反例说明完美 image-L2 decoder 的 cycle 残差仍可能在 latent score 为零处非零。实际 RAE 的隐藏混合噪声与 L1/LPIPS/GAN 需另外处理；此次阅读保留“明确观测模型导出自然系数”的机制入口，没有将现有 cycle 包装为额外 score。

**新的四篇把“先固定目标，再学习动态”推得更具体。** [Learn to Guide / Adversarial Schedules](RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md)用真实加噪边缘与单步 guided 核定义训练目标，时间系数由网络学习；这不等于手工 schedule，也不同于旧 X−G 回归。逐原图 self-consistency 在非退化 Gaussian 中受数据处理不等式限制，边缘一致性则允许合法的概率运输。前作 ImageNet-64 相对 LIG 的 FID 2.11→1.99 是正面证据，但伴随训练时间分布和 checkpoint 选择；后作改善对齐/偏好分数，主表 FID 并未胜过强基线。[PG / EDDY](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md)另将目标扩展到联合律：PG 的条件期望势给出自然 Doob 时间规律，保边缘还需单点平衡；EDDY 的单粒子 Stein 恒等式不能直接推出从 IID 创造新联合分布。两条线都保留结构启发，没有据文献启动新训练、粒子排斥或方差校准。

**新增三篇把数值误差与参考分布分别说清。** [Fitted CFG / ERK-Guid](RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md)分别从终端模式解有限步系数、从已有 solver 差分估计误差方向。前者在指定 Gaussian 判别法向上唯一导出 `r^(1+w)−r`，但 RAE 公共法向不满足该模式，同类 Full/Base 也不是条件/无条件对；后者有 ImageNet 低步数同 NFE 的明确收益，却仍手选幅度和阈值，对 Euler100 引入 Heun 查询并非免费。对 [ICG / TSG](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md)的独立推导进一步指出，独立随机类别的平均 score 对应几何参考而非混合 marginal，随机 embedding 的二阶平均偏置与采样方差也不同。TSG100 对 unguided200 的正面 FID 仍保留，但不据此把随机性当作自然误差校正，或启动新强度/窗口试验。这三篇均未给出当前 RAEv2 公平成本下至少 5% 改善的完成证据。

**反射平均已有直接的群结构先例。** Lu 等的 SPDM 在推理中对变换后的网络输出作有限群平均，Chen 等进一步将不变分布下的 score 风险分成群平均后的风险与非等变分量能量。本地 affine 法向反射先中心化，再投影输出到切空间，才与该算子对应；两元素群自然给出 1/2 权重，但只消除整体法向奇分量，并未积分掉全部 Gaussian nuisance。输出投影不能直接继承相对原官方速度的 Lipschitz 非增，路径 W1 上界也不是实际终点 FID 比较。作者 SPDM 实现的外层 NFE 未乘内部 2/4/8 次群查询，需要按真实调用核算成本。见[两篇正文、证明及代码精读](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md)。

**ERG 补足了信息扰动的一个精确机制。** logits 缩放单调改变逐行检索熵，其 value 输出导数是 value–logit 协方差；固定 K 且 V=K 或满足兼容正定度量时，Hopfield 更新还有明确能量下降界，Q 不必等于 K。一般独立 Q/K/V 不自动继承该势梯度解释，检索熵也不决定真实 denoising 误差方向。论文补充的官方 DiT 重跑自报 FID `2.38±0.05→2.15±0.03`，不能抹掉这些正面结果；但层区间、启动时刻与强度经实验选择，官方材料仅有伪代码，两次 denoiser NFE 也不等于 RAEv2 共享 Base/Full 的实测同成本。[精读与源码边界](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md)保留其结构启发，没有据此开启参数或窗口扫描。

**随后两篇把“信息删除”与“误差修复”进一步分开。** [线性 CFG / CPC](RAEV2_GUIDANCE_READING_LINEAR_CPC_20260906_ZH.md)提供精确的公共均值与后验协方差对比分解；自然谱时间权重来自 Gaussian corruption，整体 guidance 强度及实验窗口仍需选择。将同类别 Full/Base 当成条件/无条件对、把公共平移当成中心 covariance 收缩，都会错用其机制。[SWG 正式版与作者代码](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md)说明 crop 从输入开始删除远处信息，与共享全局 encoder 的 decoder JQ 不同；Gaussian-convolved oracle 的解析证明也不保证 crop/weight decay 产生共线放大误差。

这两篇导向一个具体后续：在固定 teacher joint distribution 上，`C=E[(X−F)·(F−W)]/D` 排除了不可约 posterior noise 对该线性交叉量的总体偏置。正 C 是“弱支放大强支同类条件均值误差”的必要方向证据，且可由真实 paired X 估计；简单的两残差正余弦则有共享目标噪声的混杂。[全部 80 条 teacher 记录的冻结实验](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_PROTOCOL_20260906_ZH.md)现已[完成](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_RESULTS_20260906_ZH.md)：10 个时刻的 C_W 均值全部为正，59/80 条逐行正；同一集合下原 Base gap 的 10 个均值全部为负。新方向的风险修复成分很小，等时间经验风险的解析共同标量最多降低该风险 0.00204468%，这不是 FID 改善，也没有用 oracle 系数选择 gain 或窗口。

另有一项[独立注意力几何推导](RAEV2_ATTENTION_CONTRAST_GEOMETRY_20260906_ZH.md)：中心化 query 能消除公共 key 偏置，而保留重心的 query 外推放大跨 query 对比，两者对应不同误差假设；它们都没有自动给出合理部署权重。只记录可证伪条件，没有另启新 attention 试验。

| 继续新增的原始论文 | 可迁移的机制问题 | 阅读记录 |
|---|---|---|
| [FP-Diffusion](https://proceedings.mlr.press/v202/lai23d.html)，ICML2023 | PDE自洽性、正确边界与目标可识别性必须同时具备 | [第3轮](RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md) |
| [Moser Flow](https://proceedings.neurips.cc/paper/2021/hash/93a27b0bd99bac3e68a440b48aa421ab-Abstract.html)，NeurIPS2021 | 有限密度source、全空间输运与有限场估计的区别 | [第4轮](RAEV2_GUIDANCE_READING_MOSER_20260906_ZH.md) |
| [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://jmlr.org/papers/v26/23-1605.html)，JMLR 2025 | 相关配对、同边缘的ODE/SDE及Fisher耗散；score可获得性和excess-risk决定可实现边界 | [Stochastic Interpolants](RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md) |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747v2)，ICLR 2023，归档 arXiv v2 | 条件回归的正交分解、非独立配对的弱连续性延伸；目标正确不等于有限网络或两次查询有效 | [原始 Flow Matching](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md) |
| [Sobolev Regularized Score Difference Estimation in Diffusion Models](https://arxiv.org/abs/2608.18237v1)，ICML 2026 accepted，归档 arXiv v1 | 直接约束所需输入梯度；正则偏差、有限样本常数、代码目标与理论对应关系均须保留 | [实际分布反馈](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md) |
| [Unifying GANs and Score-Based Diffusion as Generative Particle Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf)，NeurIPS 2023 | 当前生成轨迹提供反馈负例；IPM critic 不等于密度比，逐批刷新与输入反传均有成本 | [实际分布反馈](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md) |
| [When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](https://proceedings.iclr.cc/paper_files/paper/2026/file/4406cafe40d3ca4a7511a8996b6bd1b6-Paper-Conference.pdf)，ICLR 2026 | 几何与密度有不同误差尺度；有限噪声投影解释含曲率，温度缩放的uniform目标及实际时间重标需说清 | [score几何](RAEV2_GUIDANCE_READING_SCORE_GEOMETRY_20260906_ZH.md) |
| [Diffusion Model for Manifold Data: Score Decomposition, Curvature, and Statistical Complexity](https://arxiv.org/abs/2603.20645v1)，2026预印本v1 | 精确保留法向位置与切向后验的曲率耦合；on-support项不必切向，统计率证明缺口不影响可直接重建的分解 | [有限噪声曲率](RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md) |
| [Diffusion-Based Posterior Sampling: A Feynman-Kac Analysis of Bias and Stability](https://arxiv.org/abs/2605.06538v1)，2026预印本v1 | 反应项的空间变化连接局部guidance近似与终点密度；有限模型多出动力学残差，原文部分一般公式需修正 | [路径反应项](RAEV2_GUIDANCE_READING_DPS_REACTION_20260906_ZH.md) |
| [A Connection Between Score Matching and Denoising Autoencoders](https://www-labs.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)，Neural Computation 2011 | 条件 kernel score 的后验均值等于平滑边缘 score；固定 Gaussian 同空间平方损失给出由噪声决定的残差尺度 | [去噪与 score](RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md) |
| [What Regularized Auto-Encoders Learn from the Data-Generating Distribution](https://www.jmlr.org/papers/v15/alain14a.html)，JMLR 2014 | 非参数最优重建的小噪声密度导数关系；有限模型、非线性跨空间 cycle 与投影性质均需区别 | [去噪与 score](RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md) |
| [Towards Understanding the Mechanisms of Classifier-Free Guidance](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf)，NeurIPS 2025 | 对比后验 covariance 决定相对增强/抑制；同类双头不能直接继承条件/无条件语义，全 Jacobian 方案也不是免费 | [线性 / CPC](RAEV2_GUIDANCE_READING_LINEAR_CPC_20260906_ZH.md) |
| [Guiding a diffusion model with itself using sliding windows](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)，BMVC 2025 | 可控信息缺失与可控同向误差不同；需要 paired risk 方向证据，生成指标与更多计算的代价须另验 | [SWG / WD](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md) |
| [Entropy Rectifying Guidance for Diffusion and Flow Models](https://papers.nips.cc/paper_files/paper/2025/file/414f4c9fe9653e5de98fad6964d50315-Paper-Conference.pdf)，NeurIPS 2025 | 熵恒等式一般成立，Hopfield 下降须兼容 K/V；有超过 5% 的自报 FID 收益，但未推出免扫参系数或共享双头的公平成本优势 | [ERG](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md) |
| [Diffusion Models under Group Transformations](https://proceedings.mlr.press/v258/lu25a.html)，AISTATS 2025 | 推理群平均直接构造等变预测，权重由有限群决定；结构保证不等于质量保证，内部群查询必须计入成本 | [反射与群平均](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md) |
| [Robustness and Structure Preservation in Flow-Based Generative Models via Wasserstein Path-Space Divergences](https://arxiv.org/abs/2410.01244v2)，2026 修订预印本 | 精确正交误差分解可覆盖任意待修正场；flow 的 W1 上界需路径与稳定性条件，额外输出投影须另证 | [反射与群平均](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md) |
| [Guidance Breaks the Fitted Operator: A Terminal-Fitted Repair for Classifier-Free Guidance](https://arxiv.org/abs/2607.07665v1)，2026 预印本；另读作者 7 月 15 日修订稿 | 指定终端模式唯一决定一个系数，无新增幅度/窗口；证明针对理想 guided ODE 数值误差，实证尚无 ImageNet/RAE，不能直接套同类双头 | [终端系数与 ERK](RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md) |
| [Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](https://arxiv.org/abs/2603.03692v1)，ICLR 2026 作者版 | 复用 Heun 误差方向有低步数 ImageNet 收益；主方向对齐是前提，实际幅度/阈值仍手选，强 32 步 CFG/AG 的 FID 未改善 | [终端系数与 ERK](RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md) |
| [No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models](https://studios.disneyresearch.com/app/uploads/2025/04/No-Training-No-Problem-Rethinking-Diffusion-Guidance-for-Diffusion-Models-Paper.pdf)，ICLR 2025 | 独立类别平均与 marginal score 不同；TSG 同调用数对照有正面结果，embedding 平均偏置、方差、手选强度/层/区间须分开 | [ICG / TSG](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md) |
| [Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/612a7948f3294a02a63d970566ca8536-Paper-Conference.pdf)，ICLR 2024 | 联合目标的条件期望给出自然 Doob 时间势；固定多样性约束后还需边缘平衡，实际图像核/强度/成本不由此自动决定 | [PG / EDDY](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md) |
| [Diverse Sampling in Diffusion Models with Marginal Preserving Particle Guidance](https://arxiv.org/abs/2605.06553v1)，2026 预印本 | 单粒子 Stein 恒等式正确；精确 IID 构造保持整个乘积律，原联合证明与公式有可定位边界；工程多样性收益不是相对真实数据的同成本 FID | [PG / EDDY](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md) |
| [Learn to Guide Your Diffusion Model](https://proceedings.iclr.cc/paper_files/paper/2026/file/d199e76b714c2611051e1b9e4bd882f1-Paper-Conference.pdf)，ICLR 2026 | 学习有限步分布匹配有超过 5% 的 ImageNet-64 正面结果；逐 X self-consistency 比所需边缘约束更强，训练分布/核和选择成本需计入 | [学习一致性](RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md) |
| [Adversarial Learning of Classifier-Free Guidance Schedules](https://arxiv.org/abs/2608.14038v1)，2026 预印本 | 判别器监督真实 teacher 边缘的一步 pushforward，推理只保留状态 guidance MLP；实际正则/奖励改变纯 KL 目标，主表赢对齐而未赢 FID | [学习一致性](RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md) |

**最新三篇带来两个具体机制。** [PAG / SEG 阅读](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md)对照正式论文、附录与作者代码：PAG 删除跨 token 的 value 检索；SEG 的无限平滑把所有 query 换为均值。后者可独立证明是 query 的 Frobenius 投影，也是注意力概率的反向 KL 重心，但不继承原文有缺口的一般曲率保证。由此固定了仅改全部两个 DDT decoder blocks、保留完整 encoder 与 Full 读出的[结构诊断](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md)，没有搜索层、平滑尺度或 guidance 增益。该诊断现已[完成并独立复核](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md)：全部 160 条通过原生 Full/Base 逐位对照，弱分支保留非均匀 key 选择；相对原 gap 的汇总余弦为 0.18254，95.3291% 的扰动能量位于各条原 gap 的正交方向。这说明新增了不同的结构响应，尚未证明同类误差放大或质量纠偏。

[Characteristic Guidance 阅读](RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md)给出另一条思路：从 clean 目标分布推导整条加噪路径。在 RAE 坐标下，无投影隐式方程等价于两个移位查询的 Full/Base clean 共识；独立 Gaussian 推导表明，可归一化的 power target 在 `w=1.78` 时给出线性 fixed-point 收敛，且不要求两协方差可交换。现有 792 个同 gap 方向的 Jacobian 组合均为正，仅通过一个必要方向检查；固定 Gaussian 算例同时显示近纯噪声端可能极慢。没有据此声称实际 RAE 头满足 exact-score 条件，或开始带工程正则和窗口的求根采样。

| 新增原始论文 | 可保留的机制与限制 | 阅读记录 |
|---|---|---|
| [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09184.pdf)，ECCV 2024 | softmax 后 attention 换为单位矩阵，保留每个位置自身的 V；不是删掉整个 block，原文层与强度仍经选择 | [PAG / SEG](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md) |
| [Smoothed Energy Guidance](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3-Paper-Conference.pdf)，NeurIPS 2024 | 平滑 query 的线性恒等式成立；无限平滑具有精确 KL 重心解释，不能推广成全网络曲率或质量保证 | [PAG / SEG](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md) |
| [Characteristic Guidance](https://proceedings.mlr.press/v235/zheng24f.html)，ICML 2024 | 不同查询上的 clean 共识提供目标路径结构；Gaussian 可解性与迭代成本须分开，原文高强度改善不是最佳基线胜出 | [共识与 Gaussian 检验](RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md) |

**最新实验反馈：空间能量球的固定 1K 筛查为阴性。** 阅读与两套历史 bank 分析导出了固定 DC/AC 两个球，而不是手工 guidance schedule；14 项 CPU 检查与 16 图原生 parity 通过后，三组配对 1K 已完成。官方、同参考全局球、两个空间球 FID 分别为 38.1598808139、38.1611443514、38.3169688602；候选相对官方恶化 0.411658%。成本规则在查看 FID 前确定官方 100 步为本轮推理预算对照，完整历史参考准备成本仍未闭合。该实现结束，不扫描半径、分量数或时间窗口。结果明确了下一步理论需要补足的联系：单步潜变量分布距离与整条受控轨迹及解码后质量之间，仍有未解决的机制问题。见[完整结果](RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md)。

**最先改变的是下一次实验要回答的问题。** 当前 `X−G` 势函数校正的有限实现，在旧配对 1K 上仅改善 0.082633%，随后固定候选的配对 5K 相对 official100 仅改善 0.00359028%、相对 official105 恶化 0.20444746%。[规模审计已完成并结束这条候选线](RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md)，不追加种子、50K 或扩大训练；补空间诊断也不足以支持扩宽它。文献提示应区分：强模型的剩余误差、现有 IG 有用的偏置、以及实际轨迹已经积累的分布偏差。三者不是同一个回归标签。这不是已识别失败原因，而是新的可证伪问题。

| 原始论文 | 本轮重点获得的机制判断 | 详细阅读记录 |
|---|---|---|
| [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/html/2406.02507v2)，NeurIPS 2024 | 同类错误的放大比“弱模型更差”更关键；错配退化可使最优 guidance 回到关闭 | [强弱模型](RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md) |
| [Guiding a Diffusion Transformer with the Internal Dynamics of Itself](https://arxiv.org/html/2512.24176v2)，2025/26 | 中间监督、推理外推、偏置训练标签是三个不同设计；目标不必总是原始 X | [强弱模型](RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md) |
| [Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective](https://arxiv.org/html/2503.10638v2)，AAAI 2026 | 分类器解释与实际 flow 配对不同；近邻配对可能悄悄改变目标样本权重 | [几何与分类器](RAEV2_GUIDANCE_READING_GEOMETRY_20260906_ZH.md) |
| [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/html/2408.09000v2)，TMLR 2025 | 随机 CFG 的 predictor/corrector 分解有精确系数；ODE、SDE 与 powered density 不同 | [几何与分类器](RAEV2_GUIDANCE_READING_GEOMETRY_20260906_ZH.md) |
| [Analytic Distribution of Classifier-Free Guidance for Schedule Design](https://arxiv.org/html/2607.19725v1)，2026 预印本 | 瞬时乘积 score 还带来整条路径的密度修正；其 schedule 不由该定理唯一决定 | [实际分布与配对](RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md) |
| [Emergence of Distortions in High-Dimensional Guided Diffusion Models](https://arxiv.org/html/2602.00716v1)，2026 预印本 | exact-score guidance 本身也能改变类均值与类内协方差；高维极限条件需逐项核对 | [实际分布与配对](RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md) |
| [On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v3)，2025 | 精确 guidance 由目标端点分布与配对共同决定；学习辅助场有条件回归入口 | [实际分布与配对](RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md) |
| [Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models](https://proceedings.mlr.press/v202/kim23i/kim23i.pdf)，ICML 2023 | 假样本终点再加噪，与实际 rollout 边缘并不自动一致；missing-score 推导需要自洽条件 | [判别器 guidance](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md) |
| [Improving Discriminator Guidance in Diffusion Models](https://arxiv.org/html/2503.16117v2)，ECML/PKDD 2025 | 分类函数准确不保证输入梯度准确；纯梯度 MSE 部分与旧 residual 校正同类，不能换名重跑 | [判别器 guidance](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md) |
| [Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](https://arxiv.org/abs/2508.12880v4)，ICLR 2026 | mask 的平均结构响应与其随机方差要分开；轻度删块保留 suffix，不能被旧 head-swap 直接否定 | [随机弱模型](RAEV2_GUIDANCE_READING_STOCHASTIC_WEAK_20260906_ZH.md) |
| [Sobolev Descent](https://proceedings.mlr.press/v89/mroueh19a.html)，AISTATS 2019 | 从两样本缺口与当前分布的 Jacobian Gram 导出最小代价的特征输运；加入原有漂移和移动目标时必须重推 | [Sobolev 与移动目标](RAEV2_GUIDANCE_READING_SOBOLEV_20260906_ZH.md) |

后续新增六篇，已影响评估和下一步设计：

| 原始论文 | 具体作用 | 阅读记录 |
|---|---|---|
| [Effectively Unbiased FID and Inception Score and Where to Find Them](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)，CVPR 2020 | 同规模估计的模型相关偏差不自动相消；本地历史特征否定 1K≥5% 是最终目标必要条件 | [规模审计](RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md) |
| [Adjoint Matching](https://arxiv.org/html/2409.08861v5)，ICLR 2025 | 终点重加权需要处理起点归一化；转置 Jacobian 传播终点信息 | [终点控制](RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md) |
| [DEFT](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)，NeurIPS 2024 | 小网络可摊销有新终点条件的修正；sameclass denoising residual 并非新目标 | [终点控制](RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md) |
| [Value Gradient Guidance for Flow Matching Alignment](https://arxiv.org/html/2512.05116v1)，NeurIPS 2025 | 确定性 HJB 可围绕实际基准速度定义；控制能量不等于终点 KL 或真实 FID | [终点控制](RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md) |
| [Adaptive Diffusion Guidance via Stochastic Optimal Control](https://arxiv.org/html/2505.19367v2)，AISTATS 2026 | 优化当前强度还需计入未来状态变化；提高类别置信度的实证 FID 仍可恶化 | [自适应控制](RAEV2_GUIDANCE_READING_ADAPTIVE_CONTROL_20260906_ZH.md) |
| [Manifold-Optimal Guidance](https://arxiv.org/html/2603.11509v1)，2026 预印本 | 局部二次问题有解析解，仍需独立识别其度量与错误机制 | [自适应控制](RAEV2_GUIDANCE_READING_ADAPTIVE_CONTROL_20260906_ZH.md) |

再新增四篇，继续核对具体坐标、实现与对照：

| 原始论文 | 具体作用 | 阅读记录 |
|---|---|---|
| [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://arxiv.org/html/2410.02416v2)，ICLR 2025 | APG 的 clean 投影有增益解释，有限正交更新不保持范数；clipping/momentum 仍依经验设定 | [投影与零初始化](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md) |
| [CFG-Zero⋆](https://arxiv.org/html/2503.18886v2)，2025 | velocity 投影迁移到 clean 时必须保留状态项；零初速度假说可在固定端点证伪 | [投影与零初始化](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md) |
| [C²FG: Control Classifier-Free Guidance via Score Discrepancy Analysis](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_C2FG_Control_Classifier-Free_Guidance_via_Score_Discrepancy_Analysis_CVPR_2026_paper.pdf)，CVPR 2026 | score 差异的衰减上界不确定实际差异单调性或最优系数；正文与附录明确保留参数选择 | [差异与评价](RAEV2_GUIDANCE_READING_DISCREPANCY_EVALUATION_20260906_ZH.md) |
| [Guidance Matters](https://arxiv.org/pdf/2602.22570v1)，ICLR 2026 | 偏好分数可能主要反映旧 CFG 强度；有符号局部投影可诊断，时间平均不保证终点等价 | [差异与评价](RAEV2_GUIDANCE_READING_DISCREPANCY_EVALUATION_20260906_ZH.md) |

新增五篇进一步分开几何、运输与数值误差：

| 原始论文 | 对下一步的具体影响 | 阅读记录 |
|---|---|---|
| [CFG++](https://proceedings.iclr.cc/paper_files/paper/2025/file/4c9477b9e2c7ec0ad3f4f15077aaf85a-Paper-Conference.pdf)，ICLR 2025 正式版 | Flow 一阶重组在 RAE Euler 下等价于步长相关的旧 gap 增益；时间残差机制仍须独立检验 | [CFG++](RAEV2_GUIDANCE_READING_CFGPLUSPLUS_20260906_ZH.md) |
| [Angle Domain Guidance](https://arxiv.org/html/2506.11039v1)，ICML 2025 | 限制 clean 预测长度的对象要明确；范数界不等于严格保长，原始 latent 球壳不等于 posterior mean 球壳 | [ADG 与真实范数检验](RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md) |
| [PTQD](https://papers.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)，NeurIPS 2023 | 实际合成后的误差可作回归校准；确定性 ODE 无随机方差预算可吸收其残差 | [量化误差](RAEV2_GUIDANCE_READING_QUANTIZATION_20260906_ZH.md) |
| [Q-Diffusion](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)，ICCV 2023 | 成对校准数据与直接优化合成误差并不相同；两支误差的联合协方差决定放大 | [量化误差](RAEV2_GUIDANCE_READING_QUANTIZATION_20260906_ZH.md) |
| [Covariance Mismatch in Diffusion Models](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)，EPFL 2024 预印本 | 方向信息量由数据/噪声的相对方差决定；纯坐标白化不改变广义谱，主要方案改变训练路径 | [协方差与后验可辨识性](RAEV2_GUIDANCE_READING_COVARIANCE_MISMATCH_20260906_ZH.md) |

**这轮阅读已带来两个缓存检验。** 固定 160 个历史 teacher/rollout 条目的 BF16 heads，混合阶段舍入误差约为 gap 的 1.06%–1.07%，主要来自最终加法；约 90% 的减法精确，反对“固定 head 相减灾难性相消”解释，但不能排除 head 前向误差放大。[数值诊断](RAEV2_GUIDANCE_MIX_NUMERICS_20260906_ZH.md)未改精度或追加采样。另一方面，全部固定 1K 真实潜变量半径 CV 为 8.835%，不能仅据名义维数把它当作固定 Gaussian 球壳；独立 Gaussian 推导又表明真实 latent 与 posterior mean 的范数应因条件不确定性而不同。[几何检验](RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md)不据此选择范数/角度常数。下一问转向可重复的方向性分布偏差；尚无新 guidance 或 FID 收益结论。

**方向性问题随后得到正向证据。** 预先固定全部通道、空间 DC 与其正交补 AC 的[两套 5K 能量分析](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md)显示：IG 总量接近真实数据，实际是 DC 缺少约 6%–7% 与 AC 多出约 5.7%–5.8% 的抵消。跨库完整残差余弦为 0.87447/0.99754，去除重复源图的等类权敏感性分析结果相近。这不是 centered covariance、完整频谱或质量因果结论。

该发现支持一个具体结构扩展：[两个正交空间能量球协议](RAEV2_SPATIAL_ENERGY_BALLS_PROTOCOL_20260906_ZH.md)。将旧全局球换为固定 DC/AC 两个全通道球，借助原始 bridge 的精确二阶矩公式确定预算；在实际完整 cohort 的 Euler successor 上只收缩超额分量。它是闭凸集投影，保留单步 W₂/非扩张保证，无额外 gain 或窗口；并未证明整条受控轨迹或 FID 改善。实现、原生 parity 和一次冻结 1K 对照现已完成；同参考 global-ball 与成本规则选定的官方臂用于区分结构、算术和预算变化。质量结果为阴性，见本页开头；历史 decoder reversal 并不能因此升级为本次失败的已证实原因。

**终点控制阅读已经落到真实干预，并得到失败结果。** 先由历史原图交叉原型冻结终点类中心对齐量，再从完整 100 步 FP32 suffix 的转置 Jacobian 解出唯一最小能量系数。固定 8 图全保留，连续 FP32 平均响应仅为线性预测的 23.5503%；原生 BF16/uint8 输出平均响应为预测的 −15.1659%，方向相反。没有调 λ、删图或设时间窗口；该实现不进入蒸馏/部署。此试验仅检验有限响应，没有计算 FID；即使 ψ 上升，也还需证明它改善独立真实分布误差。见[冻结协议](RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md)及[完整结果](RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md)。

**两项后续判断已完成。** [终点质量连接推导](RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md)给出准确边界：直接沿原型作特征平移，可以减少一个类均值误差分量；实际 suffix 的 Gram 改变方向后，还需额外的带符号交叉量 H>0。一个正定 Gram 的精确线性反例能补齐全部 ψ 缺口，却使类均值误差和 FID 都从 5 升到 9。另一方面，[固定噪声端点缓存检验](RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md)利用 Gaussian 期望解析计算零场风险，得到官方场减零场风险为 −1.18912893/维，1000 类估计全部负；它反对在这份 checkpoint 上套用“初始 velocity 比零差”的解释，不启动 zero-init 或时间窗口搜索。这两项都没有新 GPU、模型或 FID 调用。

**评估规则的更正保留，5K 结果已经闭合。** 历史两套 5K 有超过 5% 的收益而十个平衡 1K 子集全部不足 5%，说明旧必要条件过强；本次阴性结果不推翻这一点。原候选三组均已完成，FID 分别为 official100 `6.9748978477`、potential100 `6.9746474293`、official105 `6.9604170334`。中断后复用前两臂完整样本，只续做候选 CPU merge、首次运行 official105，再统一评价；没有训练、改系数、挑 seed 或重采样前两臂。105 步实测轨迹成本比候选高 0.34737%，其预算在看 FID 前由成本冻结。完整准备仍为 `660.5193496611901 s + U`、`U≥0` 未闭合，不能称为总成本公平达标。结果及配对 IF 的局部近似边界见[结果归档](RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md)；[原协议](RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md)保持冻结。

**三条可继续发展的机制线索。**

1. **先分清要消除的误差与要保留的偏置。** 同状态令 `τ=G−F`，则 `X−F=(X−G)+τ`。无限函数类下，拟合 `X−G` 后加到 G 得到真实 posterior mean，拟合 `X−F` 后加到 G 则得到该均值加 τ。两者目标不同；“保留 τ”并不保证其终端作用或质量得以保留。下一次监督目标设计须明确这个选择，不先固定扩大旧网络。
2. **从实际访问的分布识别纠偏方向。** 给实际 q_s 加 `∇log(p_s/q_s)`，在同一个快照上对 KL 导数有确定的负贡献，无需假设现有向量场就是 q_s 的 score。错用“生成终点再加噪”的参考则可能使方向反转。此线索仍缺少有限数据下梯度精度、随轨迹变化的反馈、自然有限步规则及成本证明，尚未批准判别器训练。
3. **将弱模型的平均误差结构与随机探索分开。** 随机删块可看成平均结构响应的廉价估计；稳定条件下，Euler 的额外随机方差是 `h²` 量级，而非 Langevin 的 `h` 量级。若平均响应本身有益，蒸馏该平均弱参考是可能的成本设计；若只是在重调现有 gap 幅度、必须依赖手工窗口或误差不兼容，应停止。

**理论阅读已经带来可检验的反例，而不只是警告。**

- 在独立 Gaussian bridge 中，可以构造前后两段误差原本相消的冻结场：有限梯度校正让 bridge 积分误差从 `2/3` 降到 `1/3`，却让终点 `W₂²` 从 0 升到 `1/4`。这说明剩余问题包括跨时刻误差抵消；没有将它直接认定为 RAEv2 的失败原因。
- 在另一个解析 Gaussian 路径快照中，实际边缘比值方向的 KL 导数为 `−2.25`，终点再加噪比值方向为 `+3`。这个反例具体指出需要使用哪一种负例分布；它没有证明有限判别器可准确学到该梯度。
- 对随机弱模型，可以保持平均响应不变而改变其方差，或让平均为零但方差很大。两者区分了“平均结构误差提供引导”与“随机性本身提供收益”的解释。

**公平比较影响了哪些论文方法的直接采用。** 不把修复过强 CFG 当作超过同成本最优基线。例如解析 schedule 主表最佳 FID 为 67.87，对照最佳常数 CFG 为 67.67；S² 的 ImageNet 表相对普通 CFG 改善 5.58%，相对表内更强基线仅 3.33%，并且没有该组同总成本 FID。判别器方法则需要显式计算假样本库、训练、完整特征网络输入反传的代价。详细数字和原文冲突均保留在各笔记中。这些限制不抹去其机制或实证价值，但决定了本任务不能直接认领哪些结论。

**基础论文进一步导出的结构。** Sobolev Descent 给出两样本缺口与当前输入 Jacobian Gram 的关系。将它扩展到移动的真实 bridge 与现有基速度 b 后，本文独立推导了

\[
\dot\delta_s=r_s-D_{q_s}a_s,
\qquad
\delta_s=\mathbb E_{p_s}\Phi_s-\mathbb E_{q_s}\Phi_s.
\]

其中 r 是真实目标矩的变化率减去当前基速度产生的矩变化率，而不是只用静态缺口 δ；显含时间的测试函数还要保留 `∂sΦ` 项。若 r 在 Gram 值域中，最小能量校正为 `u=J D_q⁺ r`，从相同初始矩出发可在理想连续反馈下保持这些矩。该结构无需把 F/B 当作 exact score，也无需手工指定一个时间反馈率。

这不是已经通过的采样方案：精确矩速度匹配不会消除已有的初始/离散偏移；有限函数、估计、反馈、成本与所选矩的质量含义仍需解决。它明确了后续优先研究的问题：**在实际轨迹上，哪些可复现的分布缺口对应真实生成缺陷，现有速度如何继续制造这些缺口，以及可实现的纠正方向能否同时满足这些动态约束。** 仅有可分辨性、MMD 或局部回归下降仍不足以开始新一轮宽度或系数搜索。

**当前实验状态。** [补空间检验](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md)已完成并独立复核：全部 100000 项旧 residual MSE 逐位一致，独立汇总最大差 `3.55e−15`。它检出一个遗漏方向，但两个预先固定方向的同库变分能量只有旧 proxy 改善的 0.4002%，不能解释完整补空间或 FID。随后固定 5K 质量审计仍为阴性，旧有限势候选不再投入额外质量实验。新增的完整终点伴随 8 图试验也未通过原生响应验证；这些结果均不宣称所有规模或所有终点控制无效。旧路线没有新增训练或质量试验；随后有限配对桥的新结构已完成一次固定训练与机制实验，正在准备固定 1K 筛查。全程没有图像筛选。至少 5% 的公平成本提升目标仍未完成；新的理论阅读需导出新的机制证据，不能将同一代理优化改名重跑。

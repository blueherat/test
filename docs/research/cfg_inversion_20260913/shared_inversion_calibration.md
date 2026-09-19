# 真实图反演重建能否校准可迁移的 CFG / IG

2026-09-13；有界文献核查，未运行 GPU、未训练、未修改采样器。这里的“可迁移”指：校准结束后，从新的标准 Gaussian 噪声生成，不再提供某张参考图及其专属参数；类别或文本条件仍可以存在。

**结论：共享参数可以承接真实图监督，但本次没有找到“仅用公共反演器的重建误差，学一个全局 CFG / null 参数，就解决反演器偏差并改善新噪声生成”的直接先例。** 强弱模型联合重建、留一反演器验证也没有在下列原文中出现。这个否定只覆盖本次定向检索，不能当作完整 novelty 声明。

## 1. 最接近的原始证据

| 原文与日期 | 实际被优化的对象 / 目标 | 能否迁到新噪声；与本题的界线 |
|---|---|---|
| [Invertible Consistency Distillation for Text-Guided Image Editing in Around 7 Steps](https://arxiv.org/html/2406.14539v3)，首版 2024-06-20，核对 v3 2024-12-02 | §3.3 同一 teacher 初始化正反两个共享模型，训练 teacher consistency 和双向 preservation；preservation **只在 w=1** 计算。Appendix A 使用 rank-64 LoRA。 | 是共享生成模型；§3.4 另外评估随机生成，Appendix B 报生成指标。不是只学 null，也不是凭 cycle loss 单独保证质量。其选择 guidance 的部分是已知动态 CFG，本任务不沿该支线继续。 |
| [Few-Shot Image Generation by Conditional Relaxing Diffusion Inversion](https://arxiv.org/html/2407.07249v1)，2024-07-09，ECCV 2024；对照 [正式 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11133.pdf) §3.2–3.4 | 冻结 denoiser，给每张少样本图学习专属 score 张量 G_i；目标含 clean 图与一步重建，另加 G_i 与群体均值的距离。Algorithm 1 的监督状态来自 **Gaussian corruption**，不是公共 ODE 反演轨迹。 | 做少样本域生成，但使用样本专属张量及其随机扰动。不能当作“纯 ODE no-op → 全局 CFG”的先例，也不能把这里的 inversion 名称直接等同于本题操作。 |
| [PortraitGen: Exemplar-Driven GRPO with Dual-Reward Guidance for Photorealistic Portrait Generation](https://arxiv.org/html/2606.26930v1)，2026-06-25，预印本 | §4 用 BELM 提取真实图轨迹，真实 exemplar 与模型随机样本组成 GRPO group；真正训练信号是两种 reward。§5.1 冻结 FLUX.1-dev 主干，训练共享 LoRA。 | 是真实图反演进入共享模型后训练、再进行文本生图的直接实例；**不是以重建误差作为质量奖励**。§4 对确定性 BELM 轨迹与随机策略概率的衔接没有给出充分的 off-policy 推导，不能直接复制为无偏目标。 |
| [ReNeg: Learning Negative Embedding with Reward Guidance](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_ReNeg_Learning_Negative_Embedding_with_Reward_Guidance_CVPR_2025_paper.pdf)，首版 [2024-12-27](https://arxiv.org/abs/2412.19637)，CVPR 2025 | §4 学全局 negative embedding，另有逐样本版本；优化依据 reward。正文前置的 reconstruction loss 是基础 denoising 训练介绍，不能误读为 ReNeg 的反演重建目标。 | 明确支持共享负向 embedding 在新噪声 / 新 prompt 上使用，以及同 embedding 空间的模型迁移。因此“把 null 参数共享起来”本身已经不新；尚缺的是本题的真实图反演校准目标。 |

iCD 的可借鉴点是“重建约束与独立的生成目标共同训练”，PortraitGen 的可借鉴点是“反演可把真实图送入共享后训练”，ReNeg 的可借鉴点是“很小的负向参数也能影响全局生成”。三者不能拼成已经证明的 identity-to-quality 定理。

## 2. 公共反演器偏差为什么没有被 leave-one-out 自动解决

以下为本笔记推导。统一 FM 时间为 t=0 噪声、t=1 图像；G_phi 是待校准 guidance 的完整生成映射，I_r 是固定 reference 反演器。真实图 latent 为 x。

\[
L_{\rm rec}(\phi;r)=\mathbb E_{x\sim P_{\rm data}}\|G_\phi(I_r(x))-x\|^2.
\]

若 I_r=G_r^{-1} 精确且候选包含 G_r，那么 G_r 的损失构造上为零，即使 G_r 从 Gaussian 生成的分布很差。损失测量的是与 reference 的配对关系是否一致。把它减去 reference 自己的重建误差，不会移除这个偏好。

将多个 r 混合，只把一个 reference 的偏好变成多个配对的折中。**留出真实图 ID** 检查样本过拟合；**留出反演器** 检查对配对来源的敏感性；两者都不证明逆噪声分布就是标准 Gaussian。独立训练 checkpoint 还可能给同一图分配不同 latent 坐标，不能把逆噪声直接平均当真值。

另一端，若反演也随 phi 更新到 I_phi=G_phi^{-1}，那么 G_phi(I_phi(x))=x 对任何可逆 G_phi 成立。联合优化强弱两模型的 cycle 也允许双方共同移动而保持互逆。更精确的 inversion 可以减少数值污染，但不能把这个代数恒等式变成质量监督。

## 3. 只保留一个数学上有明确目标的可实施构造

**把共享负向 embedding 看作一个很小的 flow 参数，用真实图反演所定义的数据似然校准；identity reconstruction 留作反演有效性的检查。** 这不是“重建损失已经足够”的答案，而是说明要从 image identity 走向 from-prior 质量，最少还缺哪项约束。它属于参数校准研究，不是免训练采样改进；本次不启动。

冻结现有 SiT、正向类别 embedding、CFG 总强度 w 与求解设置，只学习跨图片共享的低维参数 phi：

\[
e^-_\phi=e_\varnothing+B\phi,\qquad
v_\phi(z,t,c)=w v_\theta(z,t,e_c)+(1-w)v_\theta(z,t,e^-_\phi).
\]

B 可先取从冻结 label embedding 表确定的少数正交方向；不能用留出图拟合 B。这里改变的是负向条件表示，不是时间 schedule 或既有控制器的分解。仓库 `y_embedder` 输出接口能够接入任意 embedding；仅说明模型接口可实现，不恢复已经停止的条件混合实验。

对真实图 latent x，用**同一个候选场**反演并同时累计体积变化。令 s=1-t：

\[
y_0=x,\quad \frac{dy_s}{ds}=-v_\phi(y_s,1-s,c),\quad
a_0=0,\quad \frac{da_s}{ds}=\nabla_y\cdot v_\phi(y_s,1-s,c).
\]

得到 z_phi=y_1 后，优化其实际 from-prior 分布的负对数似然：

\[
L_{\rm data}(\phi)=\mathbb E_{x,c}\left[
\tfrac12\|z_\phi\|^2+\tfrac d2\log(2\pi)+a_1
\right]+\lambda\|\phi\|^2.
\]

这是可逆连续流的变量替换公式，**不是本文发明的新 likelihood 原理**。散度项不能删：仅最小化逆噪声范数会奖励把所有真实图挤向高先验密度处，而遗漏压缩体积的代价。CFG 场不必是保守梯度场才能定义此 pushforward 密度；需要的是相应流的存在、唯一性、可逆性及可积 Jacobian。

该目标不使用公共 I_r 的配对标签，故避免了“复现哪位反演老师”的结构偏好。校准后直接取新的 z~N(0,I)，输出 G_phi(z,c)，不提供真实图、不存每图 null 参数。对 IG 若改学弱支路的参数，也必须重新定义其实际场和密度；本笔记不另开第二构造。

**实现成本与判据：** 可以用固定随机向量的 Hutchinson 迹估计累计 a；每个场评估还需输入导数，完整参数梯度涉及更高阶求导及反演轨迹反传，并不便宜。对低维 phi 可先只做少量固定候选的无参数梯度 likelihood 评估，但也须保留输入导数。64/128 等网格精化和不同迹向量检查数值误差，latent roundtrip 只检验反演是否可靠；不能以它给候选排序。公式是连续流目标，不能把粗 Heun 的结果标成精确离散 likelihood。

训练图与验证图按源 ID 分开，最终效益仍由独立新噪声的分布质量、语义和多样性读数确认。似然改善不保证 FID 或感知质量改善；小参数族还可能没有足够表达能力。若坚持**唯一目标只能是“保持原图不变”**，则应止于算子验证，暂不能声称它足以训练出更优的全局 CFG。

## 4. Novelty 边界与本轮决策

- 共享 negative embedding：已与 ReNeg 碰撞；新贡献不能只写“学习全局 null”。
- 双向重建 / cycle 联合训练：已与 iCD 等碰撞；只有 cycle 不提供目标分布。
- 真实图 inversion 支持共享后训练：PortraitGen 已提供实例，但其监督是 reward，不能声称同一目标。
- 学习 inverse-noise adapter 再接冻结生成器：仓库已核过 INC，不重做该构造。这里保持 Gaussian prior、改 guidance 参数，与 INC 的改 prior 路径不同，但“对流做数据似然参数拟合”仍属于已有标准思想。详见 [旧审计](../cfg_transport_search_20260913/prior_and_hybrid_audit.md)。

因此，**目前否决“公共反演器重建误差单独校准全局 CFG”的充分性声明**；可保留的研究问题是：在固定 backbone 上，真实图反演定义的正规数据目标，是否能有效校准极小的负向条件参数，并同时保留 identity 数值可靠性和新噪声生成质量。没有现成收益保证，未宣布 novel，未启动实验。

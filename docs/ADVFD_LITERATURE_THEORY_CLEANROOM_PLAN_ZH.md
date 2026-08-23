# AdvFD 文献、理论与论文优先复现协议

更新时间：2026-08-23

## 1. 研究目标与方法学边界

本阶段只回答三件事：

1. AdvFD 的理论对象究竟是什么，它与 FD-Loss、Fisher GAN、McGAN、MMD-GAN
   及近期 distribution-matching post-training 的边界在哪里；
2. 只依据论文文字和公式，能否独立实现并复现 AdvFD 最关键的现象；
3. 论文描述中哪些歧义或理论缺口会在独立实现时真实造成问题，官方实现如何处理这些问题。

本阶段不以提出新方法或刷最终 FID 为目标。完成“论文优先实现 -> 问题登记 ->
官方实现逐项比较”后暂停，与用户讨论后再决定下一阶段。

短预算仅用于发现明显实现错误，不以 1K/5K 的弱信号结案。正式阶段检查至少运行到
25K，并把训练停止点与官方 125K cosine 调度 horizon 分离；若仍有改善或无法排除，
继续到官方 pMF-B ablation 使用的 50 epochs（62.5K）。

### 1.1 不能声称真正的 blind reproduction

官方 AdvFD 仓库此前已经被克隆并做过一次初步浏览，因此研究者不可能诚实地声称
自己从未见过官方代码。本阶段采用更严格、可审计的替代协议：

- 独立实现不复制、不导入、不改写官方 AdvFD 代码；
- 在独立实现和第一轮结果完成前，不再打开官方 AdvFD 源码；
- 所有公式、超参数和工程选择只引用论文正文/附录；
- 论文没有规定的选择必须先写入 `ambiguities` 记录，不能事后按官方实现倒推；
- 独立结果冻结后，才解封官方仓库并做逐项差异审计。

因此本实验应称为 **paper-only, provenance-clean reproduction**，而不是 blind
reproduction。

## 2. 一手文献账本

### 2.1 主线论文

| 工作 | 核心对象 | 与本研究的关系 |
|---|---|---|
| [FD-Loss](https://arxiv.org/abs/2604.28190) | 在冻结表示中直接最小化生成/真实分布的 Gaussian Fréchet 距离 | AdvFD 的静态基线与 EMA 矩估计来源 |
| [AdvFD](https://arxiv.org/abs/2608.11205) | 静态 FD 加一个最大化 FD 的可学习表示，并用真实特征 whitening 校准 | 必须严格复现的目标工作 |
| [JiT](https://arxiv.org/abs/2511.13720) | 一步 clean-prediction ImageNet 生成器 | AdvFD 的主要生成器骨架 |
| [pMF](https://arxiv.org/abs/2601.22158) | 一步生成器 | AdvFD 的第二类骨架；用于验证跨骨架性 |
| [AMFD](https://arxiv.org/abs/2607.26860) | 以 amortized conditional moment estimator 构造动态矩目标 | 已占据“神经条件矩/动态矩”邻域，不能把简单组合包装成新意 |
| [RDM/iRDM](https://arxiv.org/abs/2607.02375) | 多冻结表示、MMD/全矩分布匹配 | 已占据“更多表示/更高阶矩”邻域 |

### 2.1.1 与 DMD 系列的边界

| 工作 | 核心机制 | 与 AdvFD 的边界 |
|---|---|---|
| [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html) | 用 target-score 与 fake-score 之差构造近似 reverse-KL 生成器梯度，并用 teacher trajectory regression 稳定训练 | 需要 teacher/fake score estimators；不是直接优化图像表示矩 |
| [DMD2](https://papers.nips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | 用 two-time-scale update 改善 fake-score 跟踪，移除昂贵 regression，并加入 GAN loss 与 rollout 对齐 | 已明确处理自适应 fake critic 不准和 mode collapse；“再加一个快慢 critic”不是空白 |
| [DMDX/ADM](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Adversarial_Distribution_Matching_for_Diffusion_Distillation_Towards_Efficient_Image_and_ICCV_2025_paper.html) | 以 diffusion-based discriminator 对抗匹配 real/fake score estimator 的 latent prediction，并结合像素/latent adversarial distillation | 已占据“DMD + adversarial distribution matching”组合，不应与 AdvFD 简单拼接 |
| [DP-DMD](https://arxiv.org/abs/2602.03139) | 将早期步骤用于 target-prediction 保多样性，后续步骤做 DMD，并阻断 DMD 梯度回到第一步 | 直接针对 reverse-KL 的 mode-seeking；说明 precision/recall 与多样性必须进入 adaptive post-training 对照 |

DMD 的梯度来自两个扩散 score 的差，目标是把 student distribution 蒸馏到 teacher/data
distribution；AdvFD 从已经可用的一步生成器出发，不训练 fake score estimator，而在图像表示
的一、二阶矩上做 teacher-free min-max post-training。两者共同的只是“自适应分布反馈”这一
高层结构，不能共享同一个理论结论。

AdvFD 论文已经在 JiT-B 的 SIM FD-loss 基线上测试 DMD-style guidance：FID 从 `1.00`
恶化到 `1.77`，held-out FD-r3 从 `8.45` 改善到 `6.75`，而 AdvFD 为
`0.79 / 6.03`。这说明 score guidance 可能改善部分 held-out 表示却损害 Inception FID，
也再次要求同时报告 FID、held-out representations 与 diversity，而不能把 DMD 当成一个
尚未尝试的直接替代。

### 2.2 理论祖先

| 工作 | 已有结论 | 对 AdvFD 的约束 |
|---|---|---|
| [McGAN](https://proceedings.mlr.press/v70/mroueh17a.html) | 在可学习特征中匹配均值与协方差 | “学习表示中的一二阶矩匹配”并非新概念 |
| [Fisher GAN](https://arxiv.org/abs/1705.09675) | 用真实/生成混合二阶矩约束 critic；满容量时为对称 \(\chi^2\) 距离 | pooled normalization 的均值分支已有严格祖先 |
| [MMD-GAN](https://arxiv.org/abs/1705.08584) | 对抗学习 MMD kernel/feature map | “对抗学习分布比较空间”不是 AdvFD 独有 |
| [On Gradient Regularizers for MMD GANs](https://proceedings.neurips.cc/paper/2018/hash/07f75d9144912970de5a09f5a305e10c-Abstract.html) | 证明 learned-kernel MMD critic 需要函数输入梯度约束才能得到连续、可用的生成损失 | 用 Lipschitz/gradient penalty 阻止 selective amplification 是必要基线，不是本项目可独占的新意 |
| [Fréchet-GAN](https://arxiv.org/abs/2003.11774) | 直接以 Fréchet 型目标训练 GAN | 需要区分其统计估计和 AdvFD 的 post-training 配方 |
| [Chi-square GAN](https://proceedings.mlr.press/v80/tao18b.html) | 连接 IPM、MMD 与 \(\chi^2\) divergence | 对称密度比解释已有成熟文献 |
| [Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://proceedings.mlr.press/v119/liu20m.html) | 在训练 split 上学习最大化检验功效的深核，在不相交 split 上正式检验 | 说明 adaptive discrepancy 的 train/held-out 分离已有严格统计先例 |
| [Two-sample Testing Using Deep Learning](https://proceedings.mlr.press/v108/kirchler20a.html) | 若无辅助数据，则拆分样本以分别学习表示和计算统计量 | cross-fit 的“数据拆分”本身不能作为新贡献 |

### 2.3 固定的代码版本

| 仓库 | 本地路径 | 审计 commit |
|---|---|---|
| AdvFD | `/data/users/zhoushunyu/research_repos/AdvFD` | `4e4cfed944e4fc38a75fae3ea7701ae9e5587060` |
| FD-Loss | `/data/users/zhoushunyu/research_repos/FD-Loss` | `5c03b8112fec8b9432631e4ce053c0d918cc24bc` |
| AMFD | `/data/users/zhoushunyu/research_repos/AMFD` | `0835e4f8ff79b5b652c9cab0988fb603b4995f70` |
| DMD2 | `/data/users/zhoushunyu/research_repos/DMD2` | `8d8fa55633d47cfb81bbc7a892e7248f9518763f` |

这些版本用于复现实验和源码对照；后续不以未登记的上游更新覆盖已有结论。

## 3. FD-Loss：先把静态基线写准确

对冻结表示 \(\phi(x)\in\mathbb R^d\)，记真实和生成特征矩为

\[
(\mu_p,\Sigma_p),\qquad(\mu_q,\Sigma_q).
\]

平方 Gaussian 2-Wasserstein/Fréchet 距离为

\[
D_{\mathrm{FD}}^\phi(p,q)
=\|\mu_p-\mu_q\|_2^2
+\operatorname{Tr}\!\left(
\Sigma_p+\Sigma_q
-2(\Sigma_p^{1/2}\Sigma_q\Sigma_p^{1/2})^{1/2}
\right).
\]

FD-Loss 的关键工程贡献不是改写这个公式，而是把“用于估计总体矩的样本量”和
“当前反向传播 batch”解耦。EMA 版本维护

\[
\mu_t=\beta\,\operatorname{sg}(\mu_{t-1})+(1-\beta)\mu_B,
\]

\[
M_t=\beta\,\operatorname{sg}(M_{t-1})+(1-\beta)M_B,
\qquad
\Sigma_t=M_t-\mu_t\mu_t^\top.
\]

历史矩被 detach，只有当前生成 batch 有梯度。多表示目标逐项归一化：

\[
\widetilde L_i=
\frac{D_i}{\operatorname{sg}(D_i)+c},\qquad c=0.01,
\]

然后等权相加。论文默认 \(\beta=0.999\)，并用 base generator 的 50k 样本
warm-start 生成特征矩。

## 4. AdvFD：论文明确给出的目标

AdvFD 在静态目标之外加入可学习表示 \(\psi_\omega\)：

\[
\min_\theta
D_{\rm static}(p,q_\theta)
+\lambda_{\rm adv}D_{\rm adv}(p,q_\theta;\omega),
\]

\[
\max_\omega D_{\rm adv}(p,q_\theta;\omega).
\]

G-step 固定所有表示，只更新生成器；D-step 固定生成器并对生成样本 detach，只更新
自适应表示。论文用 gradient clipping 和 AdamW 限制 D-step 的有限步长。

### 4.1 为什么 raw adaptive FD 必然有缩放退化

若把特征整体放大为 \(c\psi\)，则均值放大 \(c\)，协方差放大 \(c^2\)，因此

\[
D_{\rm FD}^{c\psi}(p,q)=c^2D_{\rm FD}^{\psi}(p,q).
\]

critic 无需发现新差异，只需放大坐标即可增加目标。

### 4.2 real-feature whitening

AdvFD 用真实分布矩校准可学习特征：

\[
\bar\psi(x)=
(\psi(x)-\mu_p)(\Sigma_p+\varepsilon I)^{-1/2}.
\]

在总体矩、满秩、\(\varepsilon=0\) 的理想条件下，真实特征满足

\[
\mathbb E_p\bar\psi=0,\qquad
\operatorname{Cov}_p(\bar\psi)=I.
\]

此时 adaptive FD 化为

\[
D_{\rm adv}
=\|\mu_q\|^2
+\operatorname{Tr}(I+\Sigma_q-2\Sigma_q^{1/2})
=\|\mu_q\|^2+
\sum_i(\sqrt{\lambda_i(\Sigma_q)}-1)^2.
\]

论文证明的是有限训练 horizon 内，带裁剪 AdamW 的参数轨迹处于初始化的有界邻域。
这不是 min-max 收敛定理，也不是该内层最大化定义了有限总体距离的证明。

## 5. 从 real whitening 推出的更精确理论

以下均为本项目推导，不是 AdvFD 论文已证明的结论。

### 5.1 标量均值分支的满容量极限是 Pearson \(\chi^2(q\|p)\)

设 \(q\ll p\)，密度比 \(r=dq/dp\)。考虑所有满足

\[
\mathbb E_p f=0,\qquad\mathbb E_p f^2=1
\]

的标量特征。因为

\[
\mathbb E_q f
=\mathbb E_p[rf]
=\mathbb E_p[(r-1)f],
\]

Cauchy-Schwarz 给出

\[
\sup_f(\mathbb E_q f)^2
=\mathbb E_p[(r-1)^2]
=\chi^2_{\rm Pearson}(q\|p).
\]

最优 witness 与 \(r-1\) 成比例。因此，理想的 real-whitened adversarial mean
不是模糊的“学习几何”，而是在 critic 函数类内逼近单向 Pearson 密度比残差。

对向量特征 \(f=(f_1,\ldots,f_d)\) 且
\(\operatorname{Cov}_p(f)=I\)，各分量构成
\(L_2^0(p)\) 中的正交组：

\[
\|\mathbb E_q f\|^2
=\sum_i\langle r-1,f_i\rangle_{L_2(p)}^2
\leq\chi^2_{\rm Pearson}(q\|p).
\]

这说明增加 feature dimension 的意义是扩展可观测子空间，而不是无上限累加同一个
密度比方向。

### 5.2 小分布偏移时，FD 的均值与协方差分支分别观测什么

令

\[
q=p(1+\delta h),\qquad \mathbb E_p h=0,
\]

并令 real-whitened 特征 \(f\) 满足 \(\mathbb E_p f=0\)、
\(\mathbb E_pff^\top=I\)。定义

\[
a=\mathbb E_p[h f],
\qquad
B=\mathbb E_p[h(ff^\top-I)].
\]

则

\[
\mu_q=\delta a,
\qquad
\Sigma_q=I+\delta B+O(\delta^2).
\]

在单位阵附近展开矩阵平方根可得

\[
D_{\rm adv}
=\delta^2\left(
\|a\|^2+\frac14\|B\|_F^2
\right)+O(\delta^3).
\]

因此在局部 regime 中：

- mean branch 寻找密度比残差在特征上的线性投影；
- covariance branch 寻找密度比残差与二次特征 \(ff^\top-I\) 的相关性。

这也给出一个高信息量消融：分别训练 mean-only 和 covariance-only adaptive branch，
判断 AdvFD 的实际收益来自哪一种 witness。

### 5.3 real whitening 没有消除 support/memorization 退化

real whitening 只约束 critic 在 \(p\) 下的一二阶矩。如果 \(q\) 在 \(p\) 的零测集上有
质量，则 \(\chi^2(q\|p)=\infty\)。在有限经验样本上，真实图和生成图几乎总是两个
不相交的离散支持。一个高容量 critic 可以：

1. 在真实训练样本上保持均值零、协方差单位阵；
2. 在生成训练样本附近输出任意大的特征；
3. 继续增大经验 adaptive FD，而不违反 real whitening。

因此 real whitening 精确消除了全局可逆仿射坐标退化，却没有单独保证：

- 经验 critic 对 held-out real/fake 泛化；
- 内层 supremum 有限；
- 不会利用 fake-only 区域或样本记忆；
- 结果与 critic 学习率、步数、初始化、batch 无关。

参数裁剪只让一次具体优化轨迹在有限步数内有界，不能替代上述统计/函数空间约束。

## 6. pooled normalization 与 Fisher GAN：必须承认的理论祖先

令混合分布 \(m=(p+q)/2\)。对标量 critic 施加

\[
\mathbb E_m f^2=1.
\]

则

\[
\sup_f(\mathbb E_p f-\mathbb E_q f)^2
=\int\frac{(p-q)^2}{m}
=2\int\frac{(p-q)^2}{p+q}.
\]

右侧是对称 triangular discrimination 的固定倍数，且在支持分离时仍有限。这正是
Fisher GAN 的 Fisher IPM 满容量理论（差一个是否取平方根的记号），不是本项目的新
发现。

Fisher GAN 还已给出有限维神经特征下的 pooled Mahalanobis mean matching：

\[
(\mu_p-\mu_q)^\top
\left[\tfrac12M_p+\tfrac12M_q+\gamma I\right]^{-1}
(\mu_p-\mu_q).
\]

所以“把 AdvFD 的 real whitening 换成 pooled whitening”本身不足以构成 solid
contribution。仍值得检验的窄问题是：

> 在向量特征的完整 Gaussian Fréchet/Bures 目标中，pooled affine calibration 是否能
> 同时保留 adaptive covariance witness，并降低 real-only AdvFD 的 support-sensitive
> critic hacking？

这应称为 **Fisher-calibrated Fréchet control** 的候选，而不是一个全新 IPM。

## 7. 当前候选研究问题（按优先级）

### H1：adaptive critic 的统计泛化，而不只是数值缩放

AdvFD 论文展示 whitening 防止 feature norm 爆炸，但没有充分回答训练 adaptive FD
与 held-out adaptive FD 是否同步。应同时记录：

- D-step train adaptive FD；
- 独立 held-out real/fake batch 上的 adaptive FD；
- real 与 fake whitened feature RMS、谱和有效秩；
- critic 对样本身份的可分辨性；
- train-to-heldout gap 随 D-step 数和 critic LR 的变化。

若 train FD 持续上升而 held-out FD 停滞/反向，则真正瓶颈是 adaptive two-sample
overfitting，而不只是 scale hacking。

### H2：real-only 与 pooled calibration 的 support 稳定性

先在解析/二维分布上验证，再在冻结生成器特征上验证：

- raw adaptive FD：应出现坐标缩放退化；
- real-whitened FD：应消除全局 affine scaling，但允许 fake-only witness 增长；
- pooled-whitened mean：应复现 Fisher IPM 的有界对称距离；
- pooled-whitened full FD：检查 covariance witness、泛化与生成器梯度是否更稳定。

这里的贡献潜力取决于 full Fréchet covariance 部分是否带来 Fisher GAN 均值分支无法
解释的稳定收益。

### H3：mean witness 与 covariance witness 的分工

AdvFD 的完整 FD 混合了两类统计。必须做：

- adaptive mean-only；
- adaptive covariance-only；
- full adaptive FD；
- 同参数量 frozen critic；
- 与 static FD-Loss 对照。

若 mean-only 已复现全部收益，理论主线更接近 Fisher/密度比；若 covariance-only 有独立
收益，才支持“adaptive Gaussian geometry”而非普通 critic 的说法。

### H4：cross-fitted adaptive representation

critic-search batch 与 generator-loss batch 分离：critic 在 A 上最大化，在未参与 D-step
的 B 上计算 G-step adaptive FD。它直接测试 adaptive objective 是否泛化。该方法与普通
GAN 的 minibatch 交替不同。learned deep-kernel / deep two-sample testing 已明确使用
不相交 split 来避免“先学习表示、再在同一数据上报告统计量”的自适应过拟合，因此
**样本拆分或 cross-fit 本身不是新意**。仍待检验的窄问题是：生成器后训练中，使用
held-out adaptive Fréchet witness 是否能在保留生成梯度的同时减少 critic-search bias，
以及完整 covariance/Bures 分支是否产生超出既有 mean/kernel test 的收益。

### H5：函数空间约束是必做基线，不是新贡献

AdvFD 的 D-step 只裁剪参数梯度范数。这个操作限制单次 optimizer update，却不直接
限制 critic 对输入图像的 Jacobian、Lipschitz 常数或 witness 的局部变化。MMD-GAN 的
后续理论已经指出：当 kernel/feature map 被对抗学习时，仅有参数约束不足以保证所得
分布损失连续；显式控制 critic 的输入梯度可以稳定生成训练，并给出 Wasserstein 拓扑
下的连续性结果。

因此以下对照属于已有理论要求，不能作为本项目 novelty：

- 对真实样本或 pooled real/fake 样本施加输入梯度约束；
- spectral normalization / spectral parameterization；
- 对 adaptive witness activation 或 feature residual 施加范数约束。

它们仍然必须进入实验，因为若任一既有正则化已经消除谱塌缩并复现 AdvFD 收益，便
不应再为同一现象发明新名称。

### H6：从最大训练 FD 转向最大 held-out test power

当前 D-step 最大化同一批样本上的 adaptive FD。在高维、global batch 远小于 feature
dimension 时，critic 可能优先放大经验协方差噪声，而不是可重复的分布差异。deep
kernel two-sample testing 的标准做法不是单纯最大化训练 MMD，而是最大化标准化检验
功效，并在独立样本上计算最终统计量。

对应到 adaptive Fréchet critic，可检验的候选目标是

\[
T_{\mathrm{FD}}(\psi)
=
\frac{D_{\mathrm{FD}}(\hat p_A,\hat q_A;\psi)}
{\widehat{\operatorname{SE}}\!left[D_{\mathrm{FD}}(\hat p_A,\hat q_A;\psi)\right]+\epsilon},
\]

并要求在未参与 critic 更新的 split (B) 上仍保持较大的 FD。这里的关键不是
“cross-fit”本身，而是区分：

- 大但只存在于训练 batch 的经验 witness；
- 统计量较小但跨 batch 可重复的 population witness。

在实现任何 studentized objective 前，先用现有 checkpoint 测 train/held-out FD、谱和
有效秩；只有观察到明确的 critic-search generalization gap，才值得推导 FD 的 influence
variance 或采用 bootstrap/EMA 方差近似。

### 暂不作为主线

- 直接拼接 AMFD 与 AdvFD：邻域已拥挤，属于方法拼装；
- 增加更多冻结 encoder：RDM/iRDM 已系统覆盖；
- 只调 critic LR、更新频率或 clip：可能改善工程结果，但理论贡献不足；
- 把 pooled whitening 单独宣称为新方法：Fisher GAN 已覆盖其均值理论。

## 8. 论文优先复现阶段

### 阶段 A：独立数学核与可杀死的 toy

不使用官方 AdvFD 代码，实现并测试：

1. 对称 PSD 矩阵平方根与 Gaussian FD；
2. FD-Loss EMA 一、二阶矩，历史状态 detach、当前 batch 保留梯度；
3. raw、real-whitened、pooled-whitened 三种 adaptive FD；
4. affine invariance、raw scaling degeneracy、梯度路径单测；
5. 离散分布上 Pearson \(\chi^2\) 与 Fisher \(\chi^2\) 等式数值验证；
6. 支持分离 toy 上 real-only 与 pooled calibration 的 boundedness 对照。

### 阶段 B：论文配方的静态 FD-Loss 基线

先使用论文中参数最少且公开采样代码更直接的 pMF-B checkpoint 和 ImageNet-256；
pMF-B 通过后再把同一 clean-room 核迁移到 JiT-B：

- 1 NFE、论文相同数据与 class protocol；
- SIM = SigLIP + Inception + MAE；
- EMA feature moments，\(\beta=0.999\)；
- 50k base samples warm-start；
- global batch 1024，AdamW \((0.9,0.95)\)；
- pMF generator peak LR \(10^{-6}\)，5 epoch warmup，cosine decay；
- online generator 权重用于正式比较；
- 先做短 pilot 验证损失、显存和吞吐，再估算完整 125k steps 成本。

如果本机资源不足以完成论文完整规模，必须明确区分：

- protocol-faithful scaled reproduction；
- full-scale numerical reproduction。

不得用短训结果冒充论文正式复现。

FD-Loss 附录给出的精确 static representation 规格为：

| representation | identifier | dimension | pooling | input |
|---|---|---:|---|---:|
| Inception-v3 | `torch-fidelity` Inception v3 | 2048 | global average | 299 |
| MAE | `vit_large_patch16_224.mae` | 1024 | CLS token | 224 |
| SigLIP2 | `vit_so400m_patch16_siglip_256.v2_webli` | 1152 | CLS token | 224 |

训练精度为 BF16，矩统计和 eigensolver 保持 FP32。当前机器缓存中的
`facebook/vit-mae-base` 和 `google/siglip2-base-patch16-256` 不是可直接替代上述
timm checkpoint 的证据；formal SIM 必须加载精确 identifier 并先做 feature 数值审计。

### 阶段 C：论文描述的 AdvFD

在阶段 B 通过后加入 Inception 初始化的 adaptive representation：

- real-feature whitening，\(\varepsilon=10^{-3}\)；
- adaptive EMA \(\beta=0.99\)；
- pMF critic LR \(2\times10^{-6}\)；
- \(\lambda_{adv}=0.05\)；
- 每两个 G-step 一个 D-step；
- step 1000 启用，4000 steps 线性 warmup；
- critic gradient clip 1；
- real feature gradients detach；
- 论文 static、raw adaptive、real-whitened adaptive 三个必要对照。

第一轮必须额外保存 H1/H3 的诊断量，但这些诊断不得改变论文复现本身的优化目标。

当前 5K-step Inception-only 前缀是缩放筛查：static 保持完整 2048 维，adaptive 暂投影
到 512 维，global batch 为 32。它用于决定是否值得支付完整 SIM/2048-adaptive 的成本，
不能被标记为 full numerical reproduction。

完整 SIM、2048 维 adaptive Inception 的官方代码复现先以 10K steps 检查实现、数值稳定性
和第一段质量趋势。10K 仍只处理约 72 万张样本曝光，远小于论文 pMF 配方的约 1.28 亿，
因此“10K 没有明显优势”本身不是停止证据。除非 10K 出现经代码、采样和指标复核后的
明确反向现象，否则下一轮应从同一个原始 pMF-B checkpoint 分别启动 AdvFD 与 static
SIM 的 25K fresh runs，并使用相同数据顺序、global batch、warmup、总步数与 cosine
schedule。不能把 10K run 的 optimizer 直接续到 25K：它的学习率已按 10K 总长度衰减
到零，重启 schedule 会改变实验定义。25K 配对运行应保留 5K 间隔 checkpoint，以区分
短程优化噪声、critic 晚启动效应和持续质量趋势。

### 阶段 D：问题冻结后再解封官方实现

独立实现和结果冻结后，逐项比较：

1. 数据预处理与 class/noise sampling；
2. generator output、CFG 与 one-step sampler；
3. feature taps、token pooling、归一化；
4. covariance 定义（biased/unbiased、中心化顺序、\(\varepsilon\)）；
5. EMA 更新与 detach 位置；
6. G/D update ordering、频率、warmup；
7. optimizer、EMA model、mixed precision、DDP all-reduce；
8. 论文表格和脚本默认值是否一致。

对每个差异分类：

- 论文明确、独立实现错误；
- 论文歧义、官方实现补充；
- 论文与官方实现不一致；
- 纯数值/性能实现差异；
- 会改变科学结论的行为差异。

完成该审计后暂停，不自动进入新方法训练。

## 9. 验收与停止条件

本阶段成功不等于必须复现论文的最终 0.79 FID。验收为：

1. 数学核通过独立单测和解析分布验证；
2. 静态 FD-Loss 在 paper-only 实现中呈健康下降，生成质量不出现明显崩坏；
3. raw adaptive branch 复现缩放退化，whitening 明显压制该退化；
4. AdvFD 相对 static FD-Loss 至少出现方向一致且可重复的改善信号；
5. 论文未说明的关键选择及其影响有完整记录；
6. 解封后能解释独立实现与官方实现的主要差异。

若 paper-only AdvFD 稳定地产生与论文完全相反的现象，必须先检查数学、数据、采样、
EMA、DDP 和指标流程；确认是真实现象后停止，不通过继续调参把反证抹掉。

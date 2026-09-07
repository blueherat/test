# 相邻生成方向：保边缘的多图／视频／多视角概率耦合

日期：2026-09-07。用途：响应用户对 guidance 之外、training-free／极少训练生成方向的探索。本记录只做文献与理论审查、一个 CPU 代数检查；不修改当前冻结的 5K 补测，不启动新 GPU 作业。

**裁决：这是有价值的新任务定义，但本次没有找到足以推荐立项的全新强方法。** “保每张图的原模型边缘，只改变联合一致性”理论上成立；初始噪声耦合、保高斯的几何噪声搬运已经有直接工作。唯一进一步审查的候选是**根据已有状态，耦合下一步尚未抽取的新噪声**。其保边缘性质非常干净，但属于经典适应耦合；便宜的 Procrustes 设计只保证局部目标，不能保证最终语义／几何一致性。不能将这两层保证拼接成一个尚不存在的终点定理。

## 1. 先区分要改变的分布

固定每个视图的外部条件 \(c_i\) 和生成器 \(G_i\)，令 \(Z_i\sim\mathcal N(0,I)\)，\(X_i=G_i(Z_i;c_i)\)。选择 \((Z_1,\ldots,Z_m)\) 的联合分布而保持每个 \(Z_i\) 的完整向量边缘，就保持每个 \(X_i\) 的原模型输出分布。它允许改变 \(\mathbb E C(X_1,\ldots,X_m)\)，例如跨视图对应区域差异、连贯视频片段的代价。这一基本结论已经由 [Couple to Control，Prop.3.2](https://arxiv.org/html/2605.11311v1#S3.SS2)明确给出。

由此独立得到的研究边界是：

- 单图总体 FID 是模型边缘分布与真实边缘分布的函数，因此不变。有限样本 FID 的期望／波动可以因相关采样而改变，不能据此宣布单图分布改善。
- 保留每帧噪声的白高斯性质，**不等于**保留一个联合视频模型的输出边缘。视频模型的第 \(i\) 帧通常依赖整段噪声；改变整段噪声联合律已经改变该模型的输入分布。
- 要保住固定条件下的原边缘，必须保持 \(G_i,c_i\) 本身。通过跨图 attention、共享预测或更改条件来加强一致性，超出这里的定理。
- 若严格一致要求 \(\phi_i(X_i)=\phi_j(X_j)\) 几乎处处，则必要条件是 \((\phi_i)_\#p_i=(\phi_j)_\#p_j\)。否则任何保边缘耦合都有
  \[
  \mathbb E\|\phi_i(X_i)-\phi_j(X_j)\|^2
  \ge W_2^2((\phi_i)_\#p_i,(\phi_j)_\#p_j)>0.
  \]
  互相矛盾的视角／身份边缘不能仅靠更聪明地相关噪声而兼容。

因此合适的任务是“同一生成任务下，多次输出之间的关系”，不是换个名字继续追求 RAEv2 单图 FID。当前 ImageNet class-conditional RAEv2 也没有直接提供相机条件分布，不能默认它已拥有所需多视角边缘。

## 2. 文献给出的实际空间

| 一手论文及本次读取范围 | 与方向的关系 | 新颖性／适用性裁决 |
|---|---|---|
| [Couple to Control: Joint Initial Noise Design in Diffusion Models](https://arxiv.org/html/2605.11311v1)，2026 预印本。§3、Prop.3.1/3.2、Thm.3.3、A.3/A.6，B.1 配置与 B.3 背景实验设置；未逐表复核全部性能。 | 系统研究保高斯边缘的初始联合噪声，包括负相关、子空间控制与离线学习耦合矩阵。 | “只改联合先验”已有直接覆盖。其线性特征／高斯耦合最优性不能升级成任意非线性生成器的语义最优性。背景实验包含 inversion 与后续 noise 优化，不能把那一应用直接视作精确保边缘的实例。 |
| [How I Warped Your Noise](https://arxiv.org/html/2504.03072v1)，ICLR 2024；arXiv 上传日期为 2025，不把它误记成 2025 首发。§2、B/C、实验适用范围与 latent-space 限制。 | 把像素视为连续白噪声在区域上的积分，依据外部运动场搬运相关性。 | 用 image diffusion 做一致视频编辑的训练自由应用已经存在；“噪声插值＋方差归一化”并不等价于保完整空间白噪声。 |
| [Infinite-Resolution Integral Noise Warping](https://arxiv.org/html/2411.01212v1)，ICLR 2025。§2、Thm.1、Algorithm 1、A，以及 grid/particle 实现限制。 | 用 Brownian bridge 增量实现积分噪声的无限细分极限，省去昂贵上采样。 | 白噪声保证依赖不重叠区域／合法分区；正文给出 injective warp 的充分条件。折叠、遮挡和离散几何退化不是免费解决的。 |
| [Go-with-the-Flow](https://arxiv.org/html/2501.08331v4)，CVPR 2025。§3、Gaussianity 命题、附录算法证明，§4 实验设置。 | 线性复杂度噪声 warping；分别有 image diffusion 的 training-free 视频应用，以及 finetuned video 模型。 | 视频主结果不是极少训练：原文列 40 GPU-days、30,000 iterations、rank-2048 LoRA。不能把其 image 分支的免训练属性转写到视频主模型。 |
| [Coupled Diffusion Sampling for Training-Free Multi-View Image Editing](https://arxiv.org/html/2510.14981v1)，2025 预印本；另见[作者 CVPR 2026 稿](https://www.jiajunwu.com/papers/coupleddiff_cvpr.pdf)。本次实际读取 v1 §3–4。 | 联合运行 2D 编辑模型与多视角模型，通过能量项让轨迹接近。 | 它已经覆盖“免训练耦合两个生成模型”这一应用口号，但使用分布倾斜，所读 v1 没有证明各原模型边缘不变。不能仅因标题含 coupling 就把它当概率论意义上的保边缘耦合。 |
| [Comparing noisy neural population dynamics using optimal transport distances](https://proceedings.iclr.cc/paper_files/paper/2025/file/51484744337f4bf5fea0e4dd92ddab0b-Paper-Conference.pdf)，ICLR 2025。§3、§4.3、A/B 的 Gaussian 过程构造。 | 每时刻对共同白噪声作正交旋转，构成因果耦合并优化轨迹距离；应用于 diffusion 轨迹比较。 | 正交 fresh-noise 耦合及其 Gaussian 过程优化也不是新的数学框架。该文目标是比较过程，而非声称实现视觉一致生成。 |
| [Eberle: Reflection Couplings and Contraction Rates for Diffusions](https://wt.iam.uni-bonn.de/fileadmin/WT/Inhalt/people/Andreas_Eberle/Preprints/ReflectionCouplingsAndContractionRatesForDiffusions.pdf)，作者预印本。实际读引言、reflection/synchronous 构造与收缩假设；未宣称读完全部 32 页证明。 | 历史状态决定的正交矩阵作用于 Brownian 增量，仍产生标准 Brownian motion。 | state-dependent Brownian coupling 属于经典工具。其特定距离／曲率条件下的收缩，不等于一般 diffusion 生成的感知一致性改善。 |

上表是有边界的阅读与查重，不是对全部相关论文的新颖性穷尽证明。另检索到 Jansons–Metcalfe 的[最优 Kolmogorov 耦合与控制](https://doi.org/10.1112/S1461157000001297)，已经将耦合设计写成控制问题；本次仅核对摘要、问题构造及结尾正交控制式，不将其计为完整阅读。

EDDY 已在仓库[既有粒子笔记](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md)审过。本次复核原 Claim 2 与附录的条件边界，没有把同一条研究线重新命名立项。单粒子 Stein 零散度并不能替代所需的联合律方程。

## 3. 唯一残余候选：根据过去选择下一步噪声的耦合

以下是本次独立整理的离散版本，保证对象直接是**所用随机采样器**，不先假设 score 完美。

设第 \(i\) 个原采样器的更新为
\[
X^i_{n+1}=F^i_n(X^i_n,\xi^i_n;c_i),\qquad
\xi^i_n\sim\mathcal N(0,I_d),
\]
且其新噪声独立于过去。令 \(\mathcal F_n\) 包含所有视图的已有状态与历史。根据 \(\mathcal F_n\) 选择可测矩阵 \(L^i_n\in\mathbb R^{d\times r}\)，要求
\[
L^i_n(L^i_n)^\top=I_d.
\]
**选择完矩阵后**才抽 \(\eta_n\sim\mathcal N(0,I_r)\)，独立于 \(\mathcal F_n\)，并令 \(\xi^i_n=L^i_n\eta_n\)。于是
\[
\mathcal L(\xi^i_n\mid\mathcal F_n)=\mathcal N(0,I_d),
\quad
\Pr(X^i_{n+1}\in A\mid\mathcal F_n)=K^i_n(X^i_n,A).
\]
对 \(n\) 归纳，各视图的整个离散路径边缘等于原采样器。跨视图联合律可以变化。这个结论不依赖 \(F^i_n\) 是否是真实 posterior mean，也不需要取步长趋零；前提是固定的单视图更新核本身不读取其他视图，且上述条件噪声确实实现。

连续版本可令 \(dB^i_t=L^i_t dW_t\)，矩阵可预测且 \(L^i_t(L^i_t)^\top=I\)。其二次变差为 \(tI\)，由 Lévy 刻画仍是 Brownian motion；要推出连续 SDE 的路径律相同，还需该单视图方程的存在与分布唯一性。该原理与上表经典耦合文献一致，**不作为新定理宣称**。

这条机制比“看见当前噪声后再将它转向目标”可靠。简单反例：\(Z\sim\mathcal N(0,1)\)、\(Q(Z)=\operatorname{sign}Z\)，尽管 \(Q(Z)^2=1\)，仍有 \(Q(Z)Z=|Z|\)，均值 \(\sqrt{2/\pi}\ne0\)。所以“每个 realization 的矩阵都正交”不够；矩阵与本次新噪声的条件独立性是关键。基于已生成状态预测 optical flow，再重用产生该状态的旧噪声，也不能直接套这条保证。

多于两视图时也不能独立决定所有最佳两两相关系数。例如
\[
R=\begin{pmatrix}1&1&-1\\1&1&1\\-1&1&1\end{pmatrix}
\]
的最小特征值为 \(-1\)，不是合法协方差。共同因子 \(L\) 给出 \(LL^\top\succeq0\)，可以保证存在性，但不会同时满足所有互相矛盾的几何约束。

## 4. Procrustes 可以保证什么，不能保证什么

两条轨迹的下一步 fresh noise 对某个已固定的线性一致性表征的响应为 \(A\eta\) 与 \(BQ\eta\)。最小化其均方差等价于
\[
\min_{Q^\top Q=I}\|A-BQ\|_F^2.
\]
若 \(B^\top A=U\Sigma V^\top\)，则 \(Q_\star=UV^\top\)，最小值为
\[
\|A\|_F^2+\|B\|_F^2-2\|B^\top A\|_*.
\]
它自然导出一个无手调 gain 的矩阵选择。在线性表征、加性高斯更新中，这是固定当前状态后下一步均方一致性代价的精确最小化；在光滑非线性表征的 Itô 展开中，它只控制当下生成元中依赖跨噪声协方差的部分。离散大步长时，使用 Jacobian 线性化更只有近似意义。

**下一步最优并不推出终点最优。** 取两个合法的一维两步过程：
\[
X_1=Z,\quad Y_1=\rho Z+\sqrt{1-\rho^2}E,\quad
X_2=X_1,\quad Y_2=-Y_1,
\]
其中 \(Z,E\) 独立标准高斯。所有 \(X_1,Y_1,X_2,Y_2\) 的边缘均为标准高斯，但
\[
\mathbb E(X_1-Y_1)^2=2-2\rho,
\qquad
\mathbb E(X_2-Y_2)^2=2+2\rho.
\]
局部最优 \(\rho=1\) 把第一步误差降到零，却把终点误差从独立采样的 2 提高到 4；\(\rho=-1\) 则终点误差为零。该反例不是说任何视觉任务都会如此，而是否定“局部对齐自动保证终点一致”的一般论证。

若想得到终点保证，应优化未来联合 value function，而不是当前距离。写
\[
V_n(x,y)=\inf_{\text{未来合法耦合策略}}
\mathbb E[C(X_N,Y_N)\mid X_n=x,Y_n=y].
\]
Bellman 更新允许保边缘的真正终点控制；连续形式的控制项依赖 \(D^2_{xy}V\)，而不是随手选的当前 feature 距离。问题在于真实生成器的 \(V\) 未知，估计它可能需要大量后续 rollout、Jacobian／Hessian 或额外训练。本次没有找到满足低成本且具可验证误差界的替代量，不能将这份计算责任藏在“自然最优耦合”几个字后面。

有一个完全可解的窄特例：已知正交几何变换 \(H\)，两视图更新核严格满足 \(F^2(Hx,H\xi;c_2)=H F^1(x,\xi;c_1)\)。匹配初始状态并取 \(\xi^2=H\xi^1\)，即可逐步保持 \(Y=HX\)，一致性代价为零。但这个结论本质上要求生成器已有相应等变性，且与已知几何噪声 warping 高度重叠；它没有解决真实模型缺乏等变性的问题。

## 5. 落到实际生成时的四个障碍

**RAEv2 当前是确定性 ODE。** 只有初始随机量，没有可自由耦合的后续 Brownian 增量。在保持其确定性更新不变的条件下，后续噪声耦合没有作用；增加噪声就是换采样器。形式上，将速度 \(v\) 改成 \(v+a\nabla\log q_t\) 并加 \(\sqrt{2a}\,dB\) 可保真实 ODE 诱导的 \(q_t\)，但当前并不知道这个 guided、离散误差累积后的实际 \(q_t\) 的 score。拿训练路径的 score 代替，不能保护原 RAEv2 边缘。反之，从一个已经固定的随机 DDPM／SDE 离散采样器出发，第三节的核保证不要求该模型完美，只保证与那个相同随机基线一致。

**几何未知时，保边缘仍可能成立，但联合改善失去保证。** 用过去的状态估计几何并选择 fresh-noise 因子，在满足可预测性时不会破坏第三节的核证明；估错 optical flow、遮挡关系或对象对应，则可能稳定地产生错误的一致性。若几何来自外部给定视频，noise-warping 文献已有强基线；若几何来自当前生成状态，需要明确由谁、在什么噪声水平、花多少额外计算得到。

**小矩阵最优不代表实际开销小。** 全 latent 的 \(d\times d\) Procrustes／矩阵平方根不可默认可行。局部置换、分块正交或部分等距映射可以降低成本，但其约束范围以及跨块／遮挡一致性须重新说明；给定 warp 下的高效白噪声分配也已有 InfRes、GWTF。真实 decoder feature Jacobian 的额外前反传不能漏记。浮点实现的 \(LL^\top\approx I\) 只能给近似保证，不能把代数等式直接称为 BF16 sampler 的严格分布等式。

**一致性指标会退化。** 相同条件下完全共享所有随机量，可得到一模一样的输出；这虽然一致，却不是有运动的视频或正确的新视角。评价必须同时固定用户要求的运动／视角／编辑内容，考察联合一致性、任务完成和跨独立场景的覆盖。每场景作为一个独立统计单元，不能把其相关帧都当独立样本获得虚假的置信度。无需多生成完整图片再选；相关生成的全部视图就是任务输出。

## 6. 本次结论与可复用的内容

不推荐立项的已覆盖版本：重新提出保高斯的初始 noise covariance、常量／子空间相关系数、已知 optical flow 下的 integral noise warping；或者把经典正交 Brownian coupling／Procrustes 改名。

唯一审查的残余候选“根据过去状态耦合下一步 fresh noise”，作为**已有随机 image sampler 的保边缘接口**可以保留；目前缺少兼具新颖性、低成本与终点一致性保证的设计，故不列为待跑 GPU 方法。不因它数学漂亮就让研究再进入局部代理改进的循环。

可复用的研究判断是：新任务应把目标写为受固定边缘约束的联合生成成本，检查边缘是否相容，再找便宜且与终点任务相关的可预测结构。保边缘定理解决“会不会损坏单图分布”；它本身没有回答“能否生成正确的共同场景”。

## 7. 证据与复现

文献归档：`R/adjacent_coupling_review_v1/`，其中 R 为当前 restart 实验根。`manifest.json` 保存六份成功下载的原文身份及一次作者 PDF 链接 404；该失败保留，随后通过正式 ICLR proceedings 恢复第七份原文，身份位于 `causal_ot_conference_recovery.json`。OpenReview 页面遇到浏览器验证时使用 arXiv／正式 proceedings，并未将搜索摘要当作完整原文。每篇的实际阅读范围已列在表格中。

CPU 复现代码：[audit_raev2_adjacent_coupling_toy.py](../experiments/audit_raev2_adjacent_coupling_toy.py)，只依赖 NumPy；代码 SHA256 `f2ace921810d593e96ad7877606e2d1b41b33676e0a44e0267f0d6403c2cb383`。结果 `R/adjacent_coupling_review_v1/toy_checks.json` 已完成：

- 固定 2×2 Procrustes 的代价 5，与核范数下界吻合；同一例独立／未旋转共享创新的代价均为 15。这只验证代数。
- 不合法三视图相关矩阵的特征值为 −1、2、2。
- 自适应旋转当前 noise 的半高斯反例、局部最优却终点恶化的两步高斯反例均以解析矩验证。
- 无 Monte Carlo 估计、无模型调用、无生成图像、无 GPU 调用。结果不作为真实视频／多视角效果证据。

复现时给一个新的输出路径，脚本拒绝覆盖既有结果：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/zhoushunyu/miniconda3/envs/myenv/bin/python \
experiments/audit_raev2_adjacent_coupling_toy.py --output /tmp/raev2_adjacent_coupling_toy_new.json
```

# 反射平均的文献入口：有限群投影与 Gaussian nuisance 积分

日期：2026-09-06。先核对原有 32 篇索引，再读以下两篇不重复的一手论文，完成后已纳入 34 篇总表。**最贴近实现的先例是 Lu 等的推理期群平均，最贴近风险保证的是 Chen 等的正交误差分解。** 两篇均不直接证明当前 RAEv2 候选的 FID 改善。本文阅读工作没有 GPU、模型前向、训练、采样或新实验。

## 一手来源及读到的结论

**Lu, Szabados, Yu，*Diffusion Models under Group Transformations*，AISTATS 2025，PMLR 258:3538–3546。** [正式页面](https://proceedings.mlr.press/v258/lu25a.html)与完整 33 页 PDF 已归档，重点阅读 §3–6、相关证明及附录 G–I。§5.1 Eq.9 的 frame averaging 直接在推理时使用

\[
\mathcal S f(x,y)=\frac1{|\mathcal G|}\sum_{A\in\mathcal G}A^\top f(Ax,Ay).
\]

它不要求网络重新训练成等变架构。固定正交群、兼容的前向过程和先验使精确 score 等变，群平均将任意近似网络输出变成等变场；随机采样若要求逐条轨迹也等变，还须耦合随机增量。确定性 ODE 没有这一随机增量条件。论文的必要条件分析还保留了保持边缘分布的无散度差项，因此“等变漂移”是便于设计的充分结构，并非一切分布对称生成器的唯一形式。

其正向价值是：已知对称性可以通过有限个明确查询强制实现，平均权重由群决定，而非从图像质量调出来。局限是它保证对称结构，未保证每个模型的回归误差或终点 FID 单调下降。图像实验用旋转/翻转群，和这里的法向 nuisance 反射不同；本地迁移须重新核对作用对象。

实际结果也有正面支持：Table2 中 VP-SDE→SPDM+FA 的 rotated MNIST 全数据 FID 为 2.81→2.64，LYSTO 为 7.88→5.31，ANHIR 为 8.03→7.57。但附录 H 使用群平均后的参考特征统计，不能直接当作本任务标准 FID 口径。其有限群 C4/D4 对应每次输出 4/8 个网络查询；同样的步数不是同样的计算预算。Table6 报告 LYSTO/ANHIR 的 VP-SDE 与 SPDM+FA 都使用 2 张 A40/L40S 等级 GPU 训练约 5 天，未给同推理预算质量比较。多数模型训练还做群数据增强；CT-PET 的 VAE 使用 FA 微调，不能把整张实验表都称作对任意现成 checkpoint 的零训练修改。

作者 [SPDM 实现](https://github.com/watml/SPDM/blob/41e7d9027a71e53459323eecd6b24a8ab050568e/ddbm/karras_diffusion.py#L333)进一步明确了成本：flip/C4/D4 分别按顺序完整执行 2/4/8 次 denoise，再逆变换并等权平均。该文件 sampler 的 `nfe` 只统计外层 denoiser 调用，**没有乘内部群大小**，不能据 NFE 文件名认定同成本。群平均的 \(1/|\mathcal G|\) 没有待调增益，但 bridge CLI 仍有通用 guidance、steps、rho 等参数；它们不由群投影定理决定。

**Chen, Katsoulakis, Zhang，*Robustness and Structure Preservation in Flow-Based Generative Models via Wasserstein Path-Space Divergences*，[arXiv:2410.01244v2](https://arxiv.org/abs/2410.01244v2)，2026-06-28 修订预印本。** 已取得 40 页全文，重点阅读 §3–4 的路径误差界、§7 的群结构、§8 实验，以及 §9 对 Theorem7 和相关命题的证明。该版本题名与 2024 年初版不同，不能把 v2 的 flow 结论标成已在 2024 年正式发表。

对群不变输入分布 \(\rho_t\) 及其等变真实 score，Theorem7 / Eq.80–81 给出

\[
J(\rho,s)=J(\rho,\mathcal S s)
 +\int\|s-\mathcal S s\|^2\,d\rho_t\,dt.
\]

证明的核心是 Haar 平均在不变分布下为正交投影。这不是“让弱模型变差”的启发式；被删除分量与合法目标空间正交，因此平方误差减少量就是它的能量。该结论对任意待修正网络成立，不要求它本来就是真 score。

§7.5 / Corollary5 将此结构接到确定性 flow 的 W1 **上界**：若理想路径等变、参考边缘保持群对称，并有统一空间 Lipschitz 控制，群平均降低路径 L2 误差且不增其全局 Lipschitz 上界，因而改善这类终点误差估计。它没有证明两种实际终点 W1 的大小顺序，更没有 FID 保证。Proposition2 的“直接对终点概率测度群平均会降低 W1”是另一算子；不能把它换成“平均向量场之后生成的终点测度”。

§8 是 10 维八分量 Gaussian mixture，不是 ImageNet：四隐藏层、每层 128 节点，Adam 30000 次、batch32，25 次独立训练重复；等变参数化显式平均 8 个网络输出。Table1 的 Ntrain=1000 对照为非等变/增强 2.74±0.14，等变/未增强 2.49±0.10，指标是两个 5000 样本集合的 EMD/W1。论文未提供实测推理延迟或同总成本图像质量结论。其文字中“增加训练也不能减少非等变误差”不能照搬成普遍不可能定理：固定网络的正交分量能被群平均精确删除，但改变网络参数当然可能改变该分量；严谨的可保留结论是给定网络的分解和结构保证。

## 如何准确对应当前候选

沿用本地约定 \(Z_t=(1-t)X+t\epsilon\)，\(t=1\) 是噪声端。真实 clean latent 属于已知仿射空间 \(H=c+T\)，\(P_N\) 为法向投影，\(P_T=I-P_N\)。令

\[
q=z-(1-t)c,\quad A=I-2P_N,\quad
R_tz=(1-t)c+Aq,
\quad f_t(q)=P_T\{G((1-t)c+q,t)-c\}.
\]

这样 \(A\) 是固定正交反射，\(A^2=I\)。由于 \(f_t\) 已在切空间内，\(Af_t=f_t\)，故两元素群 \(\{I,A\}\) 的向量群平均恰好为

\[
\mathcal S f_t(q)=\tfrac12[f_t(q)+f_t(Aq)]
=G_{\rm sym}(z,t)-c,
\qquad
G_{\rm sym}=\Pi_H\tfrac12[G(z,t)+G(R_tz,t)].
\]

**这里的 1/2 是完整有限群的均匀测度，不是待调 guidance gain。** 它只翻转已知法向 Gaussian，保留真实 clean 样本和类别，既不变换图像语义标签，也不从多个图像中选择输出。现有 IG 仍可作为被作用的 G；以上正交结构不要求将同类 Base/Full 伪装成条件/无条件模型。

更具体地，teacher joint distribution 下 \(q=W+N\)，其中 \(N\sim\mathcal N(0,t^2P_N)\) 独立于 \((X,W)\)。全 Gaussian nuisance 积分是 \(E_N[f_t(W+N)\mid W]\)，而两点反射只是对轨道 \(\{N,-N\}\) 条件平均。它精确删除整体反射的奇分量，保留偶分量，例如 \(\|N\|^2\) 依赖及 \(N_iN_j\) 交叉项；不能称为已经积分掉全部法向噪声。在平方可积条件下，可直接得到

\[
\begin{aligned}
E\|G-X\|^2-E\|G_{\rm sym}-X\|^2
&=E\|P_N(G-c)\|^2\\
&\quad+\tfrac14E\|P_T[G(Z_t,t)-G(R_tZ_t,t)]\|^2.
\end{aligned}
\]

这是已有本地 teacher 风险恒等式在“先投影、再删奇分量”下的细分，不是新训练目标。它不能保证有限 cohort 的经验差值逐项非负，也不能自动用于不保持反射对称的实际 rollout 分布。精确仿射支撑与实数算术是等式条件；存储误差、BF16 舍入下须按实际有限实现审计，不能将理论恒等式称为逐位数值保证。

两点平均相对**一次**随机 nuisance 查询的方差不增；相对同成本的两次独立 nuisance 查询平均，则只有在反射对输出的交叉协方差迹非正时才更优，不能把单次风险保证直接当成最优计算利用率。当前候选每状态要完整查询原输入与反射输入两次，反射输入一般不能复用原 encoder；每次内部仍可共享 Full/Base。这是过程内预测平均，不是两条完整生成轨迹。

在中心化坐标中，本地 clean-prediction ODE 为 \(dq/dt=[q-(G-c)]/t\)。替换为上述切向群平均后，法向速度仍保留解析项 \(P_Nq/t\)，所以不应对全部 velocity 直接做不带输出变换的简单平均。将 clean 风险转成速度风险还带 \(t^{-2}\) 权重，接近端点的稳定性、离散误差和实际轨迹必须另验。

**候选还多了一次输出投影，不能直接继承“相对原官方速度的 Lipschitz 常数不增”。** 对未投影的 \(f=G-c\)，Reynolds velocity 平均对应 clean \([f(q)+Af(Aq)]/2\)；当前候选则进一步把法向 clean 强制为 0。\(\|I-P_TDf\|\) 一般不受 \(\|I-Df\|\) 控制：例如 \(f(q)=q\) 时原速度为 0，投影后速度为 \(P_Nq/t\)，Lipschitz 常数反而为 \(1/t\)。因此可对“已投影速度→再群平均”使用非增结论，或另给两方法共同的稳定性界；不能由 Corollary5 认证当前候选相对未投影官方的整体 W1 上界已经缩紧。文献提供的是有条件连接，而非补齐了本地闭环与质量条件。

## 本轮用途与归档

可用的正向机制是：**从已知 Gaussian channel 的精确无关变量中，删除被真实目标对称性禁止的奇误差分量。** 群、投影与平均系数均由既有结构决定；收益大小由待删除分量的实际能量和下游传播决定。它和 attention 对比的错误假设不同。[原法向审计](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md)记录了较小的 teacher-MSE 收益且没有计算 FID，因此不是质量否证；本轮在闭环机制审核后，固定质量实验已经获得研究准入。文献提供结构依据，质量与公平成本缺口仍须由获准实验检验，不能把文献结论当作达标结果。本文阅读工作没有改变候选协议或自行启动实验。

初筛还看到 *Antithetic Noise in Diffusion Models*（2506.06185）、*Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion*（2502.09890，早版标题不同）及 VRG（2510.21792）；分别侧重成对终点样本、训练目标积分和采样轨迹搜索。因与当前算子不如上述两篇直接，本轮没有将它们凑入精读计数。

来源、版本、哈希及独立 SPDM 审查存于 [reading_reflection_symmetrization_v1](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_reflection_symmetrization_v1/manifest.json>)。Chen v2 PDF 的下载与转文本实测 wall 1.405372 秒、进程 CPU 0.031906 秒（不含 pdftotext 子进程 CPU）；其他阅读/浏览成本未统一计时，未重构总成本。模型/GPU 调用均为 0。

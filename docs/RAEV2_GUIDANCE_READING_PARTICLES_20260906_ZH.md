# Particle Guidance 与 EDDY：从联合目标理解轨迹内粒子相互作用

结论：这是一条真正的 guidance 研究线。原始 PG 在每个生成步骤交换粒子信息，全部粒子都是输出；其方法不依赖生成后选图。最有价值的结构是**先确定联合目标，再由条件期望导出整个时间势**，从而避免逐时手写外推系数。现有论文还没有给出适合当前 RAEv2 的、零额外设计自由度且公平成本下改善 FID ≥5% 的现成方案。不能仅因它需要核或权重而抹去其贡献，也不能把集合多样性改善等同于单图生成分布改善。

本次只读两篇原文及 PG 作者代码，未调用模型、GPU、生成新样本、训练、读取或计算当前实验 FID，未改冻结运行。额外理论审查由 `/root/endpoint_witness/particle_related_screen` 独立完成，并由本文逐项复核。没有为这些问题添加采样试验。

## 1. 来源、版本与实际读取范围

1. Corso, Xu, de Bortoli, Barzilay, Jaakkola，**Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models**。正式版本为 [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html)，[正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/612a7948f3294a02a63d970566ca8536-Paper-Conference.pdf)。arXiv 2310.13102 的最新记录为 2023-11-24 v2，本文以正式 proceedings 为准，未假定两版逐字一致。
2. Vinograd, Achituve, Fetaya，**Diverse Sampling in Diffusion Models with Marginal Preserving Particle Guidance (EDDY)**。[arXiv 2605.06553v1](https://arxiv.org/html/2605.06553v1)，2026-05-07；本次查到的是预印本，没有确认正式会议版。[作者项目页](https://galvinograd.github.io/eddy/) 仍标注 Code (soon)，未取得官方算法实现，故不能判定实验采用了下文哪组有符号差异的公式。

归档：`R/reading_particle_guidance_v1/`，其中 R 为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906`。`paper_sources.json` 保留 URL、字节数及 SHA；`author_code/sources.json` 保留源码来源，`archive_manifest.json` 汇总完整文件身份。

| 材料 | SHA-256 |
|---|---|
| PG 正式 PDF，28,179,798 bytes | `723c3c2019f0646697cc8e51eda521b55ffc3960dc53bb3ff977ead7b92e0821` |
| EDDY v1 PDF，3,991,206 bytes | `24a443dbfc33e050d4cfd6dc0c657e5e10ccf7470e5ceb74c7b6720e5b803f94` |
| PG 作者仓库固定 commit | `ce7de745191168c10b2c125b78d94ad8de119488` |
| PG 核心 pipeline | `20e888d6697df35c0d0dd63746ade65d4654eb9e6b2e25340ff6ed521204e7e2` |

PG 实际读取：正文 §§2–7，附录 A.1–A.4 的 Feynman–Kac、Doob、边缘约束/IPF 与不变性，B.2 成本及限制，E.1/E.2 图像实现/评价；分子实验正文与参数说明。源码读取核心 pipeline 的 latent/feature guidance、生成入口与评价入口；synthetic/demo notebook 已归档，未执行、未逐单元审计；外链 torsional-diffusion 实现未作第二次全仓库审计。EDDY 实际读取：正文 §§2–6、附录 A 证明、B/C 公式与高维假设、D 算法、E score 换算、F/G 结果和参数。另直接查看 PG PDF 第 7、8、17 页及 EDDY 第 20 页图片核对表格；定性大图未逐图评分。

## 2. PG：目标分布先行比“排斥力看起来合理”更有价值

PG 的固定势写作

\[
\log\Phi_t(X)=-\frac{\alpha_t}{2}\sum_{i,j}k_t(x_i,x_j),
\qquad b_i^{PG}=b_i+g_t^2\nabla_i\log\Phi_t(X).
\]

核梯度产生粒子相互排斥；基础 drift 仍由每个粒子自身的 score 给出。作者准确指出：任意手写 \(\Phi_t\) 的输出通常**不等于** \(\Phi_0\prod_i p_0\)，因为任意这样的中间密度不一定构成正确扩散路径。Theorem 1 的 Feynman–Kac 表达式刻画实际重加权，但含未知 guided density/score，是解释性恒等式，不能直接充当可计算的目标保证。[PG §5/A.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/612a7948f3294a02a63d970566ca8536-Paper-Conference.pdf)

更值得吸收的是 §7/A.2–A.3。以下用去噪终点为 0 的统一记号重述：先给定联合密度比 \(\Phi_0\)，再定义

\[
\Phi_t(X_t)=\mathbb E[\Phi_0(X_0)\mid X_t].
\]

它满足相应 backward Kolmogorov 方程，Doob 变换由此决定 drift 修正；给定正确目标和精确势时，前面的自然系数是 1，时间依赖已经包含在条件期望中。这里仍需要正确端点/初始分布、真实条件核、足够光滑性和精确势；不是“任意学一个能量再乘 1”就自动成立。

若要求每个粒子的边缘仍是 p，目标联合密度必须满足约束。对给定正势 \(\Phi'_0\)，作者引入单点平衡函数

\[
q_0(X)\propto\Phi'_0(X)^\beta\prod_i p(x_i)\gamma(x_i),
\qquad
\mathbb E_{X_{-i}\sim p^{\otimes(n-1)}}[\Phi_0(x_i,X_{-i})]=C.
\]

这将“多样性导致边缘偏移”变成需要显式补偿的方程。\(\beta\) 是期望多样性约束的对偶变量，约束强度本身没有被理论唯一指定；\(\gamma\) 在理想求解下由边缘约束确定，论文给出近似 IPF 的训练方案。固定核、约束和边缘后，时间势与补偿不必再逐时手调，但所需学习/条件期望并非免费。

不变性与保边缘还须区分：核尊重旋转、置换等对称性，可以保留这些对称性；这并不推出保持整个 p，也不推出保留类别内部各模态的概率。

## 3. PG 的具体正面结果和实现边界

正式论文的图像实验是 SD1.5、COCO 同 prompt 四图、Euler 30 NFE；图 2 展示更好的 DINO 批内多样性与 CLIP/Aesthetic 折衷，未报告 ImageNet/DiT 或相对真实数据的 FID 改善。分子实验的同样本预算确有实质收益：Torsional Diffusion 的 recall median AMR 0.565→0.520，precision median AMR 0.729→0.594，同时 coverage 改善。它支持“用同一批输出更好地覆盖有效构象”这件事，不能换算成图像 FID。

保边缘 toy 的表 2 也很有启发：10 粒子平均覆盖模式数由 IID 4.9 升至 5.9；加入边缘补偿后为 5.3。**这个 toy 表格用 50,000 个 IID 集合重加权获得**，证明的是目标联合分布可有此性质，不是 learned-PG 轨迹端到端已经在该成本下实现。它与图像实验的轨迹内 PG 不是同一种证据，不能混用。

实现方面，已读取 [固定 commit 的核心 pipeline](https://github.com/gcorso/particle-guidance/blob/ce7de745191168c10b2c125b78d94ad8de119488/stable_diffusion/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_particle.py)：

- 直接 latent 排斥与 DINO 特征排斥都在每步实际更新前计算；没有把生成出的四张图丢弃后重选。
- 论文采用 RBF median heuristic，并设 feature/pixel 的 \(\alpha_t\) 分别为 \(8\sigma_t\)、\(30\sigma_t^2\)。代码还含 latent 的 \(\sigma<1\) 关闭规则和 feature 的 `sigma_break=3`；这不是从 Doob 方程解出的时间势。
- 代码的带宽是逐行 median，移除对角后使用 `log(n−1)`；feature 分支将核力 detach 后，通过 decoder+DINO 的 VJP 拉回 latent，模型预测本身在无梯度采样路径中产生。这是实用近似，不能直接等同于完整 \(\nabla_{X_t}\log\Phi_t(X_t)\)，也不能默认它包含带宽变化的梯度。
- 作者入口固定四图同 prompt；`generate_particle.py` 的 DINO backbone 与正文写法还有版本差异，故复现时应固定实际代码而非只照摘要。本文不据此否定图 2，而是保留实现身份。

PG 的 pairwise 工作量随 n² 增长；跨显存分批时还有同步/搬运。feature 分支有 decoder、特征网络和反向传播，30 denoiser NFE 相同不意味着相同墙钟。原文 B.2 讨论这些开销，但没有我们要求的完整 T/W 成本补足实验；分子参数也经过验证集调参，不能当零参数机制。

## 4. EDDY：合法单粒子 Stein 恒等式与联合律的区别

EDDY 从一个正确且优雅的恒等式出发。对反对称矩阵场 \(A^\top=-A\)，定义 row-wise divergence 的 Stein 场

\[
\psi=p^{-1}\operatorname{div}(pA)=\operatorname{div}A+A\nabla\log p.
\]

由于混合偏导对称，\(\operatorname{div}(p\psi)=\operatorname{div}\operatorname{div}(pA)=0\)。在正确边界和适定性下，这可以改变轨迹而不改变指定单粒子概率路径。仅此恒等式并没有给出排斥场的唯一大小、核、方向或作用区间。

EDDY Claim 2 将其扩展到依赖其他粒子的 \(A_i(X)\)。以下三项是本次独立数学审查，不是作者声称的结论。

### 4.1 IID 初态下，正确精确构造保持完整乘积分布

令 \(Q_t(X)=\prod_i p_t(x_i)\)，\(\psi_i=p_i^{-1}\operatorname{div}_i(p_iA_i(X))\)。即使 \(A_i\) 依赖其他粒子，也有

\[
\operatorname{div}_i(Q_t\psi_i)
=\Big(\prod_{j\ne i}p_j\Big)
\operatorname{div}_i\operatorname{div}_i(p_iA_i)=0.
\]

因此对独立 Brownian 粒子噪声的 SDE 或确定性 ODE，\(Q_t\) 本身满足修改后的 joint FPE。若从 IID 起步且解唯一，**整个联合律仍为乘积**。另一种看法是把各 \(A_i\) 放进联合空间 block-diagonal 反对称矩阵，得到联合 Stein 场。

这不禁止同一随机种子下图像/轨迹发生改变，却意味着正确精确场不能使 IID 集合的期望覆盖率变好。Algorithm 1 正是 IID Gaussian 初态。有限步长、近似 score、公式差异或核导数近似可改变这一结论；但那时不能继续把改善归因于这个精确保分布定理。

### 4.2 一般相关初态下，Claim 2 的证明缺少条件分布项

Claim 2 未要求初态独立。附录 A Step 5 Eq.(32) 将给定全部状态的冻结生成元写成未扰动生成元，遗漏了 \(\psi_i(X)\cdot\nabla f(x_i)\)。单粒子恒等式只能在针对 p 的积分里消去该项，不能逐状态消去。相关联合分布的 \(q(x_i\mid X_{-i})\) 通常不等于 p。

可构造符合光滑和全局 Lipschitz 条件的解析反例。取两个二维粒子 X,Y，各边缘标准高斯，只有 \(\operatorname{Corr}(X_2,Y_1)=\rho>0\)。未扰动 drift/noise 均为 0，令

\[
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad
a(X)=e^{-\|X\|^2/2},\quad A_1=a(X)\tanh(Y_1)J,\quad A_2=0.
\]

正确 Stein 场为 \(\psi_1=-2a(X)\tanh(Y_1)JX\)。于是

\[
\left.\frac{d}{dt}\mathbb E[X_1]\right|_0
=2\mathbb E[a(X)X_2\tanh(Y_1)]>0.
\]

正性来自 \(\mathbb E[\tanh(Y_1)\mid X_2]\) 与 \(X_2\) 同号；全部系数及导数有界或受高斯因子控制。因此目标边缘立即变化。这里的 \(\rho\) 仅用于解析反例，不是任何候选 guidance 参数；没有进行随机采样验证。

### 4.3 公布的矩阵公式存在独立的符号不一致

EDDY Eq.(7) 明确写 \(A=rv^\top-vr^\top\)，\(r=-\nabla k\)，且 v 对当前粒子坐标不变。按 Definition 1 的 row-wise divergence，直接求导应为

\[
\operatorname{div}A=(\mathrm{Jac}\,r)v-v\operatorname{div}r
=(\Delta k\,I-\nabla^2k)v.
\]

而 Eq.(9)–(10) 使用相反的 \((\nabla^2k-\Delta kI)v\)。附录的 score 项写法也有符号不一致。该问题与上面的联合 FPE 论证相互独立；翻转一个符号不会自动得到既保边缘又改变 IID 联合的机制。没有作者实现，不能判断实验到底用了哪个表达式，更不能声称“修正后一定改善/失败”。[EDDY 原文公式与附录](https://arxiv.org/html/2605.06553v1#S4.SS2)

## 5. EDDY 实验的正面信息必须按真实评价目标解读

作者在 FLUX.1-dev/SDXL 的 2048 条 COCO prompt 上各生成四图，研究多样性与输出偏离基础模型的折衷。DINO 核采用轻量 decoder+特征网络，HVP 用有限差分，Laplacian 用 25 个 Hutchinson probes，并常只施加前 20% 步。论文明确承认该近似不再精确保边缘。

有实质正面现象：FLUX 高多样性行，EDDY/PG 的 DINO 相似度为 0.497/0.495，接近相同多样性；EDDY 的 CLIP 24.737 高于 PG 23.792，Aesthetic 5.660 高于 5.362，FID 数字 41.353 小于 69.051。它说明有结构的相互作用比直接排斥更能保住该基础模型的输出特征。SDXL 则并非所有指标占优：高多样性 EDDY/PG 的 FID 为 69.135/53.471。

**这里 FID 的 reference 是 IID 基础模型样本，不是真实 COCO/ImageNet 图像。** 论文从每个 batch 取一个粒子形成 marginal 样本，再对不同 variation 与 IID variation 计算距离；IID 自比 FID 也非零。因此这些数值不能读成“相对真实数据的 FID 改善”，也不是当前 RAEv2 的 5% 目标。

| 方法 | SDXL 秒/次生成 | FLUX 秒/次生成 |
|---|---:|---:|
| IID | 8.0 | 28.1 |
| PG | 9.0 | 28.4 |
| EDDY | 23.0 | 35.1 |

这是原表 4 的 A100 80GB 计时，未做当前研究的相同 T/W baseline 补足。原表 8–12 还明确扫描强度、核带宽、部分作用区间；其 sweep 与 chosen 表的模型/数值范围部分不对应，本文不替作者猜测修正。因此既不能称免调参，也不能称公平成本 FID 证据。[EDDY 实验与完整参数](https://arxiv.org/html/2605.06553v1#A7)

## 6. 对 RAEv2 的可迁移结构与下一层问题

第一，可继续吸收 **PG 的联合目标—边缘平衡—Doob 时间势** 链条。它给出自然系数和动态来源，避免手写 gain schedule；但必须先用可解释的目标固定 \(\Phi_0\)/约束，否则核、带宽、强度只是被移到终点。选定目标后学习一个单点补偿 \(\gamma\) 与时间势，原则上比逐时手扫更自洽；本次没有提出训练授权或具体候选。

这条路线还给出 RAE 时间边界的明确要求：若采用正确的 Gaussian bridge，t=1 时噪声与真实终点 X 完全独立，那么 \(\Phi_1(Z)=\mathbb E\Phi_0(X)\) 为常数、其 guidance 梯度为零；无需人工起始窗口。t=0 则回到指定终点势。当前原生 IG 场与这套精确条件期望之间的差异不能靠替换记号消除。

第二，**“保边缘”需要明确保哪个边缘**。当前 Full/Base/IG 的实际 BF16 场没有被证明是任何真实目标的精确 score；把 clean 预测代入 Tweedie 换算不能自动补齐这个缺口。若一个机制精确保留基础模型的单样本输出边缘，它不会降低该模型相对真实分布的 population FID；它仍可有批次覆盖/估计方差价值，且有限步实现可能改变质量。这不是把局部理论当 FID 上界，而是区分不同研究目标。

第三，RAEv2 当前 1K 协议每类一图，而两篇图像工作针对同条件四图的内部多样性。跨类别直接排斥改变了问题，类别标签不能自动给出合适的语义核。即便每类多图也要把新增模型、decoder、反向与通信成本全部计入；不能只比较输出数量或 denoiser NFE。

EDDY 的高维相对量论证也不证明原始 RAE latent 上存在足够的相互作用：其附录 C 假设有效邻居距离 O(1)、带宽与维度无关。对于彼此独立的高维高斯，距离通常随维度平方根增长；固定带宽 RBF 的全部权重可能一起趋于零。“某项相对占优”与“这一项足以影响轨迹”是两个问题，特征空间核的选择仍承载实际机制。

第四，理论上允许非 IID 且保边缘的联合律，PG 的平衡目标已经示例证明；所以 EDDY 的问题不意味着这条线不可能。真正需要解决的是**让联合律变化而各边缘按目标演化的相容方程**，而非对每个条件场都施加过强的零散度要求。状态相关的噪声交叉协方差等是另一种数学入口，但会更改目前确定性 ODE 的过程，不能未经推导加进 sampler。

本轮得到的可用 taste 是：先对联合律写清楚要保留什么、改变什么，再从其演化方程导出相互作用；不能把“看起来排斥”或“局部散度为零”当成完整机制。尚未形成无需额外机制选择的 RAEv2 新候选，故本轮止于文献与数学结论，保持公平成本 FID ≥5% 的总目标不变。

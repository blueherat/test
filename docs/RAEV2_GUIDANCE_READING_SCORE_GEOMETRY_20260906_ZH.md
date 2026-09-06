# RAEv2 guidance 阅读：score 中的几何、密度与误差尺度

日期：2026-09-06。论文：Xiang Li、Zebang Shen、Ya-Ping Hsieh、Niao He，**When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis**，ICLR 2026。

**核心判断。** 这篇值得保留的机制是：Gaussian smoothing 将“离开数据支持的代价”和“支持内部的概率分配”放在不同尺度上；可以利用这一分离改变采样目标，而且不需要把神经网络当作精确保守场。但“靠近流形”“在流形上均匀”“恢复真实数据分布”是三个不同目标。论文为前两者提供理论和初步实验，不能据此把 RAEv2 的整个 score 当作法向，也不能把均匀化自动当作 FID 改进。

原文归档在 [reading_score_geometry_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_score_geometry_v1)。本次未运行 GPU、模型查询、采样、训练或 FID，未更改实验、状态或阅读索引。

## 1. 来源与实际阅读范围

- 正式依据是 [ICLR proceedings 的 34 页 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/4406cafe40d3ca4a7511a8996b6bd1b6-Paper-Conference.pdf)，不是仅凭摘要或会议信息推定版本。正式 PDF SHA256：8f13956a4aff4ea1c47efdca9ece3aa3c1e3dc1353c0762c3e2ae2bd5d296631。
- [arXiv v2 HTML](https://arxiv.org/html/2509.24912v2) 用于可搜索的公式定位；论文身份以正式稿为准。已读正文 §1–8、附录 A、B.1–B.6、C.1–C.4 和 D.1–D.3；查看了正式稿 p.10 的 Table 2／Fig.3。
- 重点逐步核对 **B.2 的一致 Laplace 展开、B.5 的 Eq.(20) 及其更强积分余项论证**，同时跟读 B.1／B.3 和 B.4 对结论的依赖。B.4 的 WKB 全局存在性是作者假设，并未在本笔记中补成无条件定理；D 引用的外部混合时间定理没有逐篇重新证明。
- 下文明确标识“原文定理”“直接推论”“结构启发”。后两者不充当作者已证明的新 guidance 性能结论。
- 正式稿、标题检索及 [Xiang Li 作者主页](https://shawnli.me/)、[Zebang Shen 作者主页](https://shenzebang.github.io/) 均未确认公开的对应作者代码。本次以 C.2 的算法描述核对实现含义，不声称审过不存在于归档中的代码。

## 2. 真正的分离：距离势、密度和曲率

用 \(D\) 表示环境维数、\(d\) 表示流形维数。原文 Assumption 2.1 是紧致、无边界的 \(C^4\) 嵌入流形 \(\mathcal M\)；Assumption 2.2 要求局部坐标密度 \(p_{\rm data}(u)\) 严格正且为 \(C^1\)。这里的密度相对于坐标 Lebesgue 测度 \(du\)，不是直接相对于内禀体积 \(d\mathrm{vol}_{\mathcal M}\)。

先考虑 VE：\(Y=X+\sigma\epsilon\)，\(\epsilon\sim N(0,I)\)。在一个固定、足够小、具有唯一最近点投影 \(P_{\mathcal M}\) 的管状邻域内，正式稿 Theorem B.2（p.18–19）给出

\[
\log p_\sigma(x)
=-\frac{\|x-P_{\mathcal M}x\|^2}{2\sigma^2}
+\log p_{\rm data}(u_x)
-\frac{D-d}{2}\log(2\pi\sigma^2)
-\frac12\log\det\widehat H(u_x,x)
+r_\sigma(x),
\quad \sup_x|r_\sigma(x)|=o(1),
\]

\[
u_x=\Phi^{-1}(P_{\mathcal M}x),\qquad
\widehat H_{ij}(u,x)
=\langle\partial_{ij}\Phi(u),\Phi(u)-x\rangle
+\langle\partial_i\Phi(u),\partial_j\Phi(u)\rangle .
\]

第一项是到流形的平方距离，第二项是数据密度，行列式项保留曲率和坐标体积。令内禀密度为 \(\rho=p_{\rm data}/\sqrt{\det g}\)，则在 \(x=\Phi(u)\) 上有 \(\widehat H=g\)，两项合成 \(\log\rho(u)\)。因此“均匀”对应 \(\rho\) 常数，不能把坐标密度 \(p_{\rm data}(u)\) 常数当作一般曲面上的均匀。

**证明为什么是一致的。** 作者对 Gaussian 卷积做 Laplace 展开，极小点是 \(u_x\)。局部 Hessian 的 metric 部分有正下界，曲率扰动由管半径和图册导数控制；缩小但固定管半径与局部积分球，可以保持正下界。紧致性允许有限图册，统一控制密度正下界、导数上界和远离极小点的间隙。B.2 的 \(C^1\) 密度版本得到积分相对误差 \(O(\sigma)\)，再得到一致的 log-density 余项。它不是对一个孤立 \(x\) 的形式 Taylor 展开。

VP 定义为 \(Y=\sqrt{1-\sigma^2}X+\sigma\epsilon\)。同一正式定理使用未缩放的 \(P_{\mathcal M}\) 展开，另有
\[
-\tfrac12\langle P_{\mathcal M}x,x-P_{\mathcal M}x\rangle
\]
的有限阶项，不能直接照搬 VE 的全部表达式。对于 RAE 的直线桥 \(Z_t=(1-t)X+t\epsilon\)，可在 \(t<1\) 改成 \(Y=Z_t/(1-t)\)、有效 VE 噪声 \(\sigma=t/(1-t)\) 后再比较；\(\sigma\) 不是不加换算就等于模型时间 \(t\)。

**不要微分 \(o(1)\)。** 一致的函数值余项不保证其梯度小。这一点关系到能否用 log-density 展开宣称“score 只有法向加某个精确切向项”。作者在 B.5 专门重新估计 score，没有直接对 B.2 的余项求导。[正式稿 B.2、B.5](https://proceedings.iclr.cc/paper_files/paper/2026/file/4406cafe40d3ca4a7511a8996b6bd1b6-Paper-Conference.pdf)

## 3. 有限小噪声与模型误差：Eq.(20) 能保证什么

**原文结果。** Theorem 5.2 额外要求 \(p_{\rm data}\in C^2\)。其证明 B.5（p.27–28）建立

\[
\sup_{x\in T_{\mathcal M}(\varepsilon)}
\left\|\nabla\log p_\sigma(x)+
\frac{x-P_{\mathcal M}x}{\sigma^2}\right\|\le C
\tag{原文 Eq.20}
\]

对充分小的 \(\sigma\) 成立，\(C\) 与 \(\sigma\) 无关。这里的加强条件不能省成主文 Assumption 2.2 的 \(C^1\)。

证明先利用 \(\nabla\frac12\mathrm{dist}^2=x-P_{\mathcal M}x\)，把
\[
\sigma^2\nabla\log p_\sigma(x)+x-P_{\mathcal M}x
\]
写成 Gaussian 加权的 \(\Phi(u)-P_{\mathcal M}x\) 的积分比。只使用 B.2 的 \(O(\sigma)\) 相对误差会留下过大的 score 误差，因此改用 \(C^4\) 相位、\(C^2\) 振幅下的 \(O(\sigma^2)\) Laplace 余项。对任意单位方向先加一个常数 \(1\)，避免积分振幅在极小点为零，再减回该常数，得到一致的 \(O(\sigma^2)\) 比值误差。除以 \(\sigma^2\) 才是 Eq.(20)。这一步给出了所需尺度，而不是假定 denoiser 已经是投影。

**直接推论，并非作者单独编号的定理。** 令
\[
D_\sigma(x)=\mathbb E[X\mid Y=x]
=x+\sigma^2\nabla\log p_\sigma(x),\qquad
\widehat D_\sigma=x+\sigma^2\widehat s_\sigma .
\]
若学习误差在同一管状邻域满足
\[
E_\sigma^{\rm oracle}
=\sup_x\|\widehat s_\sigma(x)-\nabla\log p_\sigma(x)\|,
\]
则
\[
\|\widehat D_\sigma(x)-P_{\mathcal M}x\|
\le \sigma^2(C+E_\sigma^{\rm oracle}).
\]

更一般地，\(Z=aX+\sigma\epsilon\)、\(a>0\) 时，以 \(a\mathcal M\) 为支持集，
\[
\widehat D(z)=\frac{z+\sigma^2\widehat s(z)}a,\qquad
\|\widehat D-a^{-1}P_{a\mathcal M}z\|
\le \frac{\sigma^2}{a}(C_a+E).
\]
常数依赖缩放后几何和密度，不能假设随 \(a\to0\) 一致。

这些式子的有限噪声边界是：

1. 保证是“存在 \(\sigma_0,C\)，所有 \(0<\sigma<\sigma_0\)”；原文没有给可直接代入 RAE 的数值常数。管半径、曲率、密度正下界和导数均重要。
2. 它在固定管状邻域内成立；高维典型 Gaussian 位移约为 \(\sigma\sqrt D\)，单看标量 \(\sigma\) 小不足以认证 actual rollout 留在该邻域。
3. \(E_\sigma^{\rm oracle}\) 是一致误差，不是训练平均 MSE，也不是 Full/Base 差的范数。两个近似头相近不代表它们共同接近真 score。
4. \(O(1)\) score 余项包含真实密度与曲率贡献，不应一律当作学习噪声。其 denoiser 尺度为 \(O(\sigma^2)\)，也不等于其对最后图像分布的影响可以忽略。

**一个独立的精确例子。** 在均匀单位圆上取 \(x=(1,0)\)，此点已经位于流形。Gaussian 后验角度密度正比于 \(\exp(\cos\theta/\sigma^2)\)，所以
\[
D_\sigma(x)=
\left(
\frac{\int_{-\pi}^{\pi}\cos\theta\,e^{\cos\theta/\sigma^2}\,d\theta}
{\int_{-\pi}^{\pi}e^{\cos\theta/\sigma^2}\,d\theta},0
\right).
\]
对任何有限 \(\sigma>0\)，第一坐标严格小于 \(1\)。因此真 posterior mean 位于圆内，真 score 在流形上仍可非零；最近点投影却是 \(x\) 本身。这是曲率产生的合法去噪行为，不是必须修正的模型错误。没有为这个代数例子运行数值实验。

## 4. 两种误差对象与两个真正不同的理论结果

**Theorem 4.1（p.6，证明 B.3）：最终分布的 score。** 作者定义实际算法的输出密度为 \(\pi_\sigma\)，比较的是
\[
E_\sigma^{\rm law}
=\|\nabla\log\pi_\sigma-\nabla\log p_\sigma\|_{L^\infty(K)}.
\]
在固定紧集集中、log-density 可微、\(K\) 一致可求长路径连通等条件下，\(o(\sigma^{-2})\) 足以让极限集中在 \(\mathcal M\)；\(o(1)\) 足以恢复真实数据分布；仅有非消失的有限阶差异可以保留支持而改变支持上的密度。

B.3 将 score 差沿有限长度路径积分成 log-density 差，利用归一化固定加法常数，再通过法向 Laplace 积分识别极限。这里真正使用的两档误差是 \(\sigma^{-2}\) 与 \(1\)。它没有沿反向采样轨迹累计神经网络误差，也没有把训练 MSE 自动转成 \(E_\sigma^{\rm law}\)；正式稿 §8 对此有明确限制说明。

**Theorems 5.1／5.2（p.7–8，证明 B.5）：直接使用学习 oracle。** 方法是固定噪声水平的 Tempered Score Langevin：
\[
dX_\tau=\sigma^\alpha s(X_\tau,\sigma)\,d\tau+\sqrt2\,dW_\tau .
\]
这里 \(\tau\) 是 Langevin 运行时间，\(\sigma\) 在该 SDE 内固定。假设
\[
\|s-s^*\|_\infty=o(\sigma^\beta),\quad
\beta>-2,\quad \max\{-\beta,0\}<\alpha<2.
\]
在其余条件成立时，平稳分布随 \(\sigma\to0\) 趋向内禀体积均匀分布。

在上一节 Eq.(20) 的加强光滑条件下，机制可直接写成：
\[
\sigma^\alpha s
=-\sigma^{\alpha-2}(x-P_{\mathcal M}x)
+O(\sigma^\alpha)+o(\sigma^{\alpha+\beta}).
\]
法向约束继续增强，真实密度漂移与允许的学习误差都消失。对保守场，平稳密度直接为 \(p_{\rm learned}^{\,\sigma^\alpha}\) 归一化；在几何管内积分法向以后留下内禀体积。**该方法通过改变目标消除难以可靠学习的密度信息，不是更精确地恢复原密度。**

非保守场 Theorem 5.2 另外假设唯一平稳分布及 Assumption B.2：管内质量集中、局部 WKB 形式、正的 \(C^2\) 前因子及其 \(C^2\) 收敛、相应相位解条件。B.4 将 Fokker–Planck 方程逐阶展开；法向输运方程确定沿法向的前因子，下一阶在流形上产生椭圆算子，借强最大值原理得到常数。这个证明保留了非梯度 score 的重要情形，但不是“任意有小点值误差的网络都已有全域平稳／混合保证”。强最大值原理本身只给各连通分支上的常数；若支持不连通，分支间质量还需全局论证，不能只凭该局部步骤认证全局均匀。这里保守场的显式 Gibbs 论证与非保守场的条件式论证应分开使用。

**理论没有唯一确定 \(\alpha=1\)。** 它确定的是尺度分离的可行区间。正式定理的多项式余量也严格强于摘要式的笼统 \(E=o(\sigma^{-2})\)：例如 \(E=\sigma^{-2}/\log(1/\sigma)\) 虽满足后者，但对任意固定 \(\alpha<2\)，\(\sigma^\alpha E\) 不趋零。不能拿这条较弱口号为一个固定幂次的实际算法背书。

作为机制概括，在保守场及相同集中条件下，把 \(\sigma^\alpha\) 写成 \(a_\sigma\)，需要的是 \(a_\sigma\to0\)、\(a_\sigma/\sigma^2\to\infty\)、\(a_\sigma E_\sigma^{\rm oracle}\to0\)。这是本笔记对证明结构的概括，不是已自动识别了误差率或设计出唯一调度。

## 5. 方法与实验支持到哪里

Theorem 6.1 增加固定观测势 \(v\)，漂移为 \(-\nabla v+\sigma^\alpha s\)，目标极限是 \(e^{-v}\,d\mathrm{vol}_{\mathcal M}\)。保留观测项的系数来自这个目标，而均匀化 prior 仍遵循上一节条件。作者将其启发用于 CFG：只温度缩放无条件项，保留条件增量。真实 CFG 中的增量通常随 diffusion time 变化，且是两个近似网络项；这不自动等于定理中的一个固定、满足其正则性条件的 \(v\)。[正式稿 §6、C.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/4406cafe40d3ca4a7511a8996b6bd1b6-Paper-Conference.pdf)

正式实现描述为
\[
\widetilde s_t(x,c)
=-\frac1{\sigma_t}
\left[\sigma_t^\alpha\epsilon_u+
w(\epsilon_c-\epsilon_u)\right].
\]
它只用于 predictor–corrector 中的 Langevin corrector。C.2 采用 SD 1.5、\(w=7.5\)、30 个 DDPM predictor，每个 predictor 后做 \(n_{\rm corr}\) 个 corrector；最后再做 \(n_{\rm corr}\) 次无 guidance、无随机噪声的无条件 score 更新，作者称为 projection。有限噪声下这组更新不因此成为严格的最近点投影。PC 与 TS 都保留相同末尾步骤。

| 证据 | 正向结果 | 成本和选择边界 |
|---|---|---|
| C.1 的圆／椭圆，训练出的近似 score | 标准 Langevin 可能错失原密度，TS 更接近内禀均匀；椭圆可区分均匀角度与均匀弧长 | \(\sigma=.01,\alpha=1\)；Euler–Maruyama 步长 .1，10,000 步 × 10,000 runs；训练超参按 test loss 选择 |
| C.3 真 score 加确定性 \(O(1)\) 误差 | 在支持仍正确时破坏原密度；TS 仍趋向均匀的现象与尺度分离相符 | 低维构造，目标为 uniform，非自然图像 FID |
| Table 2，固定 \(\alpha=1\)，同 \(n_{\rm corr}\) 比 PC | 三个 prompt × 五种步数的 15 格 P-sim 全提高；I-sim 14/15 降低 | car、20 correctors 的 I-sim 为 88.06→88.07，略差；只有三个创意 prompt，每设置 512 图，无误差条 |
| Table 1，各自 best-results | 家具 P-sim：PC 29.40→TS 30.20；汽车 26.30→26.62 | TS 搜 \(\alpha\in\{.1,.5,1,1.5\}\) 和 \(n_{\rm corr}\in\{5,10,15,20,30\}\)；PC 搜相同 corrector 数。建筑 TS 27.32 仍略低于 DDPM 27.36 |

P-sim 是 CLIP 图文相似度，I-sim 是同 prompt 图像两两 CLIP 相似度；二者不能直接替代 FID、真实多样性或类别分布恢复。Table 2 的同 corrector 数比较是应保留的正面证据，而不是把所有收益都归因于更大的调参预算。

**C.1 的 .1 步长使用了重标时间。** 实际积分的 drift 为 \(\widehat s=\sigma^2s\)；standard 的 noise 为 \(\sqrt{2\sigma^2}\)，TS 的 noise 为 \(\sqrt{2\sigma^{2-\alpha}}\)。相对于主文 \(s+\sqrt2\,\dot W\) 和 \(\sigma^\alpha s+\sqrt2\,\dot W\) 的两个 SDE，生成元分别整体乘了 \(\sigma^2\) 和 \(\sigma^{2-\alpha}\)。因此 Euler–Maruyama 的 .1 不是直接作用于主文奇异 score 的同一时钟；平稳分布一致不等于固定积分时间、有限步误差或墙钟混合成本一致。

**成本要按比较对象说。** 对固定 \(n_{\rm corr}\)，PC 与 TS 的模型调用结构相同，TS 只多标量运算。相对 30 步纯 DDPM，则额外有 \(30n_{\rm corr}=150\)–900 个 corrector 和 5–30 个末尾无条件更新；按每次条件／无条件各算一支，调用量由算法描述可推为 \(2\times30(1+n_{\rm corr})+n_{\rm corr}\)，即 365–1890 次分支计算，具体 batch 融合墙钟未报告。这是从算法描述推导的计数，不是作者实测 NFE。没有 RAE／DiT 同成本 FID 或完整墙钟证据。

附录 D 讨论保守场、单位圆特殊势的连续时间 Poincaré 混合：温度缩放可消除由误差形成的双井势垒，这是有用的机制，并非必然降低计算效率。但它不覆盖 SD 的离散调用成本。D.2 的全域 Holley–Stroock 比较还有需要补足的技术边界：所用 \(\phi(x)=(|x_1|-1)^2\) 在全空间无界，不能直接把全域 oscillation 当成一个小常数；若要将该例升级为严格复杂度认证，需处理尾部／局部化与光滑性。这里不依赖该附录来认证任何实际推理预算。

## 6. 对 RAEv2 真正有用的结构启发

**A. 噪声模型决定尺度；几何近似误差必须和学习误差分开。** 对合法 Gaussian posterior mean，\(\sigma^2s\) 与去噪位移的关系是精确的，最近点解释却有 \(O(\sigma^2)\) 偏差。这样可以先问“误差属于未满足的几何约束、有限噪声曲率，还是支持内的概率分配”，而不是看见残差就加一个任意幅度。自然尺度来源是观测噪声协方差和坐标约定，不是扫出来的 guidance gain。

**B. 共享几何可以先解析分离，但不要扩大已知几何的含义。** 若已知 \(X\in H=c+T\)，且 \(Z=aX+\sigma\epsilon\)，令 \(P_N\) 是 \(T^\perp\) 的正交投影，则 Gaussian 法向因子独立给出
\[
P_N\nabla_z\log p_Z(z)
=-\frac{P_N(z-ac)}{\sigma^2},
\qquad
P_N\mathbb E[X\mid Z=z]=P_Nc.
\]
这是本笔记的直接因子分解，对 affine \(H\) 有限噪声下精确，不需要 \(\sigma\to0\)、曲率估计或新的幅度参数。它能隔离一个可辨识分量；**RAE 的已知 affine 约束并不描述 \(H\) 内部的非线性数据流形**，更不能认证图像语义、密度或 FID。仓库已有 affine 投影／反射路线的事实与成败仍按原审计记录处理；该恒等式不是重命名后重新做一遍的理由。

**C. 两个同支持的精确 score，其共同奇异项会抵消；真实 Full/Base 差仍须识别。** 若两个严格正光滑密度共享同一个 \(\mathcal M\) 和 Gaussian 核，Eq.(20) 给 \(s_F-s_B=O(1)\)，B.2 的 log-density 差中公共距离和曲率项抵消。不能因此微分余项，宣称有限噪声 gap 严格纯切向。更不能把 RAE 同类别共享 Full/Base 自动当成两个已知密度：若它们以同一总体 posterior mean 为训练目标，理想极限中两者应相同；现实 gap 只是两个误差之差，未自带正确密度纠偏方向。

**D. 想保护密度时，不能只证明几何回归；想均匀化时，应把新目标写明。** 论文最有价值的设计顺序是先定极限目标，再通过不同尺度保留或压低不同项。它给“先识别哪一项应被保护，再校正其余项”的研究结构；它没有要求必须抹掉真实密度。我们的目标仍是公平总成本下真实分布质量提升，因而不能直接把 TS 的 uniform prior 或 \(\alpha=1\) 拿来当新 arm。

本次形成的是更准确的机制分解：已知 Gaussian 几何有自然尺度和可辨识分量，非线性几何仍有真实曲率修正，支持内密度需要额外信息才能纠偏。这些结构允许继续设计；具体 correction 的误差来源、闭环目标和成本尚需独立成立，论文没有替 RAEv2 解决这些问题。

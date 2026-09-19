# CFG 中什么应当不变：文献与反方核查

日期：2026-09-13。状态：原始文献核查与独立推导；没有 GPU 实验，没有改采样代码。本文服务于“如何把 identity/idempotence 思想接到 CFG”，不把以前的 guidance 门控、norm 修正或固定点扫描改名为新想法。

## 核心裁决

**先声明条件采样的目标，再讨论应保留的量。** CFG 没有一个普适的“图像应该不动”的要求。图像、latent norm、条件差 \(g\)、无条件未来终点，都不是任意 CFG 必须保持的不变量。

可以严肃讨论三类不同对象：

1. **指定目标分布的结构**：若目标仅根据条件似然重新加权，条件似然相同的样本之间的相对概率应当保留；硬事件条件化确实幂等。
2. **同一概率模型的相容关系**：条件/无条件 score 的 Bayes 混合身份，以及正确随机过程下后验预测的塔式关系。它们是“模型是否自洽”的约束，不保证自洽模型等于真实数据。
3. **同一动力学的表示与数值一致性**：同一向量场的剩余终点映射沿自身轨迹不变；这是流映射半群性质，不是普通 clean denoiser 沿轨迹恒定，也不是条件与无条件终点必须相等。

第 1 类最接近“已经满足同一条件，再做一次不应进一步任意改变”的直觉；第 2 类适合机制诊断；第 3 类主要校验算子或求解器。不能把三类混成一个 guidance loss。

## 1. 四组主要文献证据

### 1.1 Classifier-Free Guidance is a Predictor-Corrector，2024 / TMLR 2025

Bradley–Nakkiran 给出精确模型反例，说明 DDPM-CFG、DDIM-CFG 和终点 power distribution 一般不同；并在 SDE 极限下用条件 DDIM predictor 与 Langevin corrector 解释 CFG。[原文 §3–4](https://arxiv.org/html/2408.09000v1) [TMLR 版本](https://openreview.net/pdf?id=2dZswRE2sD)

对本问题的作用：不能先假定 CFG 的终点是 \(p(x)p(c\mid x)^w\)，再把该目标的某项性质宣布成现行 CFG 的定理。采样器及完整路径都参与定义结果；“瞬时混合 score 对应一个密度”不等于“这一族密度由同一个正向扩散得到”。

### 1.2 Conditional Diffusion Models with Classifier-Free Gibbs-like Guidance，2025

Moufad 等区分“每个噪声层分别 tilt”与“先 tilt 终点，再按同一 corruption 平滑”。两者 score 相差 Rényi 散度梯度项；普通 CFG denoiser 一般不等于指定 tilt 目标的后验均值。小噪声下修正项消失的结论带正则假设。[原文 Proposition 1–2](https://arxiv.org/html/2505.21101v1)

启发：真正的时间相容性要求模型来自同一个终点分布，而不是任意拼接各时刻的合理密度。边界：不能把小噪声光滑结论直接套到严格硬标签的支撑边界；也不能由“缺一个修正项”直接假定低成本 guidance gap 门控已经估计到了它。

### 1.3 Consistent Diffusion Models，NeurIPS 2023

Daras 等定义 denoiser 为其自身反向 SDE 最终输出的条件期望。Lemma 3.1 将它与同一过程下的 reverse-martingale 和终端 identity 边界等价联系；论文明确说相容性及 score 条件只保证对应**某个**数据分布，未保证是正确分布。[原文 §3](https://arxiv.org/html/2302.09057v1)

相关的 [On the Equivalence of Consistency-Type Models](https://arxiv.org/html/2306.00367v1) 解释了 SDE、ODE 和 Fokker–Planck 一致性的关系。要保留的是正确过程下的期望关系，不是将单条随机轨迹上的所有预测强制相等。使用不同 guidance、不同 drift、独立重加噪或任意固定噪声插值后，需要重新检查转移核是否仍是该关系所指定的核。

### 1.4 Analytic Distribution of Classifier-Free Guidance for Schedule Design，2026-07，v2 预印本

Jiang–Ma 对确定性 CFG 给出路径积分形式。其 v2 Theorem 4.2 表明，终点乘积密度之外还有沿 guided characteristic 积分的 score-discrepancy 修正；Theorem 4.3 推广到时间调度，保留有限噪声端的边界比例。理论要求正密度、正则性、可积性及唯一流。[原文 §4 与附录](https://arxiv.org/html/2607.19725v2)

这支持“累积路径不能省略”，但不能推出“移动样本让当前 gap 更小就更接近真实数据”：路径、Jacobian、归一化与抵达的状态都同时改变。本文的实用 schedule 是进一步近似设计，不能把精确分布表示误写成其 schedule 的普适最优性。仍应以已有正式 PCG 分析为基础，将这篇视为较新的预印本补充。

## 2. 可以保留的分布结构：不是像素不动

下面是我们的直接推导，不是四篇论文共同承诺的 CFG 终点定理。设我们**主动指定**目标

\[
\pi_w(dx)=\frac{L(x)^w\,p(dx)}{Z_w},\qquad
L(x)=p(c\mid x),\quad Z_w=\int L(x)^w p(dx)>0.
\]

若 \(R=L(X)\)，则在有定义的条件层上

\[
\boxed{\pi_w(dX\mid R)=p(dX\mid R).}
\]

理由很简单：给定 \(R=r\)，权重 \(r^w\) 是常数，归一化时抵消。离散情况下，若 \(L(x)=L(x')>0\)，两者的概率比保持不变。

这比“无关内容不要变”精确：如果 \(X=(S,N)\)，似然只依赖 \(S\)，则保留的是 \(p(N\mid S)\)。若 \(N\) 和 \(S\) 相关，其边缘 \(p(N)\) 仍可能随 \(S\) 被重新加权而改变；若二者独立，才进一步保持 \(p(N)\)。所以条件变化时，不能任意选一个看起来无关的图像属性并要求它逐样本不变。

它的研究价值是明确哪些“类内筛窄”属于额外偏好；它的实际难点是未知的真实 \(L\)、有限模型校准及观测指标。用同一未校准模型自评 \(L\) 再宣布保留成功，会循环论证。

### 硬事件条件化的真正幂等性

对事件 \(A\) 及 \(\mu(A)>0\)，定义

\[
\mathcal C_A(\mu)(B)=\frac{\mu(B\cap A)}{\mu(A)}.
\]

则 \(\mathcal C_A^2=\mathcal C_A\)。同一事件已经确定后，重复给出这一事实不应再次缩窄类内概率。

但 soft likelihood weighting 一般不幂等：\(\mathcal T_L^2\mu\propto L^2\mu\)，不同于 \(L\mu\)。重复同一证据会产生双重计数；第二次真正独立观测则可能合理地产生第二个 likelihood 因子。扩散的不同采样时刻具有不同状态与条件后验，不能把每一步 CFG 简单解释成对同一固定 posterior 再做一次条件化。

## 3. 最强反例一：硬标签下 terminal power 根本不改变类内分布

令真实标签 \(C=f(X)\) 是确定函数，目标类事件为 \(A=\{x:f(x)=c\}\)。那么

\[
L(x)=\mathbf1_A(x),\qquad
\pi_w(dx)=p(dx\mid A)\quad\text{对所有 }w>0.
\]

这证明：若我们把“条件”严格定义为硬类别归属，终点 power tilt 没有理由继续挑选该类中更典型、更饱和或更狭窄的一部分。实际 CFG 仍可能改变类内分布，因为噪声层后验不是硬指标，且逐时混合 score 不等于该终点目标的正向平滑 score。

**反例排除的是哪种说法？** 它排除“调大 CFG 就是在精确终点类别后验上继续加强同一已确定条件，所以类内收缩在数学上必需”。它不推出额外 guidance 应为零，更不否定有限网络中 CFG 的实用改进。我们可以把 guidance 当误差修正或独立质量偏好，但必须明确那是另外的目标和证据。

如果研究的是文本、软属性或有标签噪声的模型，\(L\) 不再是 indicator，此退化不自动成立。也不能把 ImageNet 的人工标签确定性，直接等同于现实世界语义分类永远无歧义。

## 4. 最强反例二：精确 clean posterior prediction 沿精确 ODE 也会改变

用仓库的噪声→数据时间 \(t\in[0,1]\)，取独立 \(X,\epsilon\sim\mathcal N(0,1)\)，线性桥

\[
Z_t=tX+(1-t)\epsilon,\qquad V_t=t^2+(1-t)^2.
\]

精确 FM 场与 clean 后验均值分别为

\[
v(t,z)=\frac{2t-1}{V_t}z,
\qquad m(t,z)=\mathbb E[X\mid Z_t=z]=\frac{t}{V_t}z.
\]

精确 ODE 从 \(z_0\) 出发的轨迹是 \(z_t=\sqrt{V_t}z_0\)，因此

\[
m(t,z_t)=\frac{t}{\sqrt{V_t}}z_0
\]

从 0 逐渐变成 \(z_0\)，**没有保持不变，尽管网络、流和分布全部精确**。真正的剩余终点映射为

\[
\boxed{F(t,z)=\Phi_{t\to1}(z)=z/\sqrt{V_t},\quad F(t,z_t)=z_0.}
\]

所以“沿路径不断询问 clean head 应得到同一张图”不是普通 FM 训练的 oracle 身份。把它当硬约束会惩罚正确模型。Consistency model 学习的 trajectory endpoint 与普通 denoiser 的后验均值是不同函数，哪怕两者输出张量尺寸完全相同。

### 正确的替代关系

对同一确定性场 \(v_w\) 的精确流，

\[
F_w(t,z)=F_w(s,\Phi_w(t,s,z)),\quad
\partial_tF_w+D_zF_w\,v_w=0.
\]

这是半群关系；若 guidance 带动量或历史 buffer，状态必须包含这些历史变量，不能只传 \(z,t\)。若沿 CFG 轨迹查询 conditional-only 或 unconditional-only 的剩余终点，一般不恒定，因为读取场和推进场不同。

对同一随机 Markov 反向过程，正确对象是

\[
h(t,z)=\mathbb E[X_1\mid X_t=z],\quad
h(t,X_t)=\mathbb E[h(s,X_s)\mid\mathcal F_t],\quad t<s.
\]

这里的等式是条件期望，不是逐次相等。在非嵌套的独立加噪观测之间，或者仅固定一对 \((X,\epsilon)\) 的 FM 插值之间，不能未经检查就套塔式性质。将过程换成 guided drift 后，原 conditional denoiser 也未必仍是该新过程的最终后验均值。

## 5. CFG++、APG、CFG-Ctrl 的“不变”到底是什么

| 方法 | 可成立的局部或条件性陈述 | 不能升级成的必需不变量 |
|---|---|---|
| CFG++ | 对 clean guidance 与已有 unconditional noise 作确定性重组；其 DDS/SDS 推导带局部几何和冻结 Jacobian 近似 | clean 混合点必在任意非凸数据流形上；unconditional 整条轨迹或最终图片不变 |
| APG | 相对条件 clean prediction 分解方向，删平行分量后局部内积为零；结合 rescale、momentum 得到经验收益 | latent norm 精确保留；平行分量必然全是伪影；正交分量必然纯语义 |
| CFG-Ctrl | 在指定连续滑模动力学与方向界下可给稳定性保证；发布采样器用 gap 历史与逐坐标 sign 校正 | 任意 learned field 的 raw gap 必然收敛；gap 变小必然更符合真实条件分布；有限离散算子就是精确投影 |

原始来源：[CFG++](https://arxiv.org/html/2406.08070v2)、[APG](https://arxiv.org/html/2410.02416v1)、[CFG-Ctrl 正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_CFG-Ctrl_Control-Based_Classifier-Free_Diffusion_Guidance_CVPR_2026_paper.pdf)、[CFG-Ctrl 补充](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_CFG-Ctrl_Control-Based_Classifier-Free_CVPR_2026_supplemental.pdf)。

几个必须保留的反方检查：

- APG 的 \(\langle m,d_\perp\rangle=0\) 仅消去一阶能量变化；有限步 \(\|m+ad_\perp\|^2=\|m\|^2+a^2\|d_\perp\|^2\)。因此“APG 就是在保持 norm”不精确。
- 两个各在不同局部 chart 上的点，其插值也可能离开流形；一个仿射子空间上的外推反而仍在子空间内。仅靠 \(w\le1\) 不给普适 manifold 保证。
- 条件与无条件 score gap 为 \(\nabla\log p(c\mid z_t)\)，不是天然的错误。把它全程消成零可以删除必要条件信息；让 scalar guidance 沿 raw gap 调整，本身还满足若干代数换元规律，但这些不证明概率正确。
- CFG-Ctrl 正文的最小奇异值界单独不足以保证 sign 控制方向；补充材料给了更强的方向条件，公平引用时必须一并保留。真实模型及离散发布实现是否满足条件是另外的问题。

## 6. 仓库已有研究：不能换名重启

本次已读并继承以下本地证据：

- [FSG、CFG-Ctrl、CFG-MP 比较](../../FSG_CFG_CTRL_CFG_MP_COMPARISON_20260911_ZH.md)：已有局部 gap、完整未来差及输运积分区别；精确兼容模型里三种一致性误差同降，真实条件分布却变差的反例；CTRL 理论和发布代码边界。
- [CFG++ 正式版精读](../../RAEV2_GUIDANCE_READING_CFGPLUSPLUS_20260906_ZH.md)：已核 flow matching 扩展与一阶重加权等价，不能把重组改称新“流形不变量”。
- [APG 机制及五项延伸](../../APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md)：已做 parallel/orthogonal 对实际未来语义和能量的干预；“平行全无用”已被本地观察反对；有 curl、数值细分和尺度对照。
- [已有条件流随机更新方向](../../FIVE_MECHANISM_IDEAS_REVIEW_20260912_ZH.md)：Gaussian 不变核、重采样稳定性不等于逐图 identity，不能把此类旧构造重命名为本次新 idea。

## 7. 交叉辩论记录与裁决

**构造代理提出：** 保留 \(P(X\mid R)\)，\(R=p(c\mid X)\)，比笼统保持“无关内容”更有根据。**反方：** 仅当主动选择 likelihood-tilt 目标时成立；普通 CFG 不保证实现它，真实 \(R\) 又不可直接获得。**修正：** 把它作为清楚的目标规范与精确玩具模型的验收条件，不能立即作为图像网络的可测损失。

**表示代理提出：** 同场退化、共同项平移协变、预测参数化仿射换元交换，是 CFG 线性对比的可靠代数结构。**反方：** 这些是实现和表示一致性，坐标偏好有时也是有意的归纳偏置；违反它不直接意味着 FID 必差。**修正：** 可用于排查单位依赖或重复施加强度；与概率目标分开报告。固定参考的嵌套 CFG 满足强度相乘，而非通常意义的幂等。

**最初可能的直觉：** 更好的生成器应让自己的 clean 预测沿采样保持不变。**精确 Gaussian 反例：** ordinary denoiser 即使完全准确也不满足。**修正：** ODE 测真实剩余终点映射；SDE 测同一转移核下的后验期望一致性，不混用。

**最初可能的直觉：** 条件已经实现后应要求 \(g=0\)。**反方：** 有限噪声下，条件后验仍可能非平坦；近零噪声时 score gap 与 velocity gap 又有不同的时间换元和端点极限。**修正：** 可以研究何时附加条件信息不再改变指定目标，但不能把某种参数化的 gap=0 当作普适实现标准。

当前可信的研究推进是先选定“精确条件分布”“指定偏好倾斜”或“有限模型误差修正”之一，再用两个 oracle 反例检查所提约束是否惩罚正确模型。只有这一步通过，才能讨论对应的可观测估计量；当前材料不支持立刻恢复大规模 guidance 调参。

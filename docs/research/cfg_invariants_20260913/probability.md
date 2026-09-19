# CFG 中什么应当不变：概率必要关系、可操作诊断与反例

日期：2026-09-13。独立概率数学审查。已阅读仓库两份主线报告及 posterior/null 训练结果；本轮仅做文献、推导与 CPU 解析核对，没有运行 GPU 或启动训练。

**最值得保留的“不变”对象是条件证据相同时的相对概率结构，以及全部类别共同满足的概率归一化关系。它们都不是“同一噪声必须生成同一张图片”。前者能够约束 CFG 的分布目标和状态反馈形式；后者能够检验条件/null 场是否有自洽的概率解释，但都不直接保证图像质量。**

## 1. 先约定精确对象：score 差与速度差不能混写

固定噪声层 `t`，设完整类别集合为 `c=1,...,K`，先验 `pi_c`，严格正且可微的联合模型满足

\[
p_t(z)=\sum_c\pi_c p_t(z\mid c),\qquad
w_c(z,t)=p_t(c\mid z).
\]

记条件和无条件 score 为 `s_c`、`s_u`，则 Bayes 公式直接给出

\[
g_c=s_c-s_u=\nabla_z\log w_c.
\tag{1}
\]

这是 CFG 的标准概率关系，不是本轮提出的新假设。[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

在当前 SiT 的独立 Gaussian 线性桥 `Z_t=tX+(1-t)epsilon` 中，对相容的理想 FM 速度，有

\[
s_c(z,t)=\frac{t v_c(z,t)-z}{1-t},\qquad
g_c(z,t)=\frac{t}{1-t}\,[v_c(z,t)-v_u(z,t)],\quad0<t<1.
\tag{2}
\]

该式是当前坐标下直接推导；不能把速度差直接代入 score 恒等式而漏掉时间尺度。更一般 FM 的任意速度表示未必具有这一规范 Gaussian 关系。[Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2)

学习到的网络可能不是保守场，也不一定与同一联合密度相容。以下“必要”均应理解为：**如果声称这些场精确代表相应概率对象，则必须成立。** 它不是对所有经验上有用控制器的无条件要求。

## 2. 首选不变量：同一条件证据层内的分布应保持

### 2.1 密度重加权有严格的保留结构

固定一个基础图像分布 `P` 和一个可测证据量 `R(x)>0`。若目标仅通过该证据重加权：

\[
Q(dx)=\frac{h(R(x))}{Z}P(dx),\qquad0<Z<\infty,
\tag{3}
\]

则在正权重的证据层上，正则条件分布满足

\[
\boxed{Q(dx\mid R=r)=P(dx\mid R=r).}
\tag{4}
\]

**证明：** 将 `P` 对 `R` 解体为 `P(dx|R=r)P_R(dr)`，重加权只把第二项改为 `h(r)P_R(dr)/Z`，第一项保持。这个表述避免把连续空间中概率为零的等值面误当作普通事件；等式按条件分布的通常几乎处处意义成立。

取 `R(x)=p(c|x)`、`h(r)=r^gamma`，就是常用的终点条件幂重加权目标。它允许改变不同条件证据层的占比，却不能在证据完全相同的一层中任意偏爱一种姿态、纹理或构图。

也可以直接说：对于同一层上的两点，若密度和坐标定义适用，`q(x_1)/q(x_2)=p(x_1)/p(x_2)`。更一般的式 (4) 比单点密度比表述完整。

这是本文由重加权定义直接推出的必要结构，并不声称现有 CFG 采样器已经实现它。它将“不要破坏无关细节”的笼统动机，收紧为一个可证伪的概率命题。

### 2.2 与条件独立的关系：必须保留的是条件分布，不一定是边缘

若图像可表示为 `(S,N)`，并且 `C` 与 `N` 在给定 `S` 后条件独立，即

\[
p(c\mid s,n)=p(c\mid s),
\]

则精确 gap 的 `N` 分量为零，且证据倾斜满足

\[
g_{c,N}=0,\qquad Q(N\mid S)=P(N\mid S).
\tag{5}
\]

这里 `N` 是在指定联合分布和坐标下确证的条件无关量，不是凭视觉直觉选出的“背景”“颜色”或“高频”。如果 `N` 与 `S` 相关，改变 `S` 的占比仍然可以合法改变 `N` 的边缘分布。

**CPU 反例：** 六个等概率状态，证据值分别为 `.2,.2,.2,.8,.8,.8`，另一属性为 `-1,0,2,1,2,4`。按证据平方倾斜后，每个证据层内三个状态仍然等概率，但该属性的总体均值从 `1.33333` 变成 `2.21569`。因此“无关属性的边缘也必须不变”不是式 (4) 的推论。[解析数据](probability_analytic_checks.json)

另一个容易遗漏的边界是加噪。即使 clean 层满足 `C→S→N`，只观察加噪后的 `S_t` 时，`N_t` 可能提供关于未知 `S` 的额外证据，因此 `C` 与 `N_t` 在给定 `S_t` 后未必独立。若想把式 (5) 应用到每个采样时刻，必须验证该时刻的条件独立关系；“clean 语义无关”不能原样搬到高噪声状态。

即使每点的直接 guidance 在 `N` 方向为零，也不保证配对轨迹的 `N` 坐标不变。基础速度的 `N` 分量可能依赖 `S`，而 guidance 已改变 `S`。只有额外存在相应解耦的动力学，才可推出同初值下的逐样本保留。

### 2.3 与 CFG-CTRL 更直接的关系：状态反馈不能任意破坏证据等值层结构

令 `ell=log R`，理想 gap 为 `g=grad ell`。若控制器将它缩放为

\[
\widetilde g(z)=a(z)g(z),
\]

则按 `J_ij=partial_j g_i` 约定，直接微分给出

\[
J_{\widetilde g}-J_{\widetilde g}^{\mathsf T}
=g(\nabla a)^{\mathsf T}-(\nabla a)g^{\mathsf T}.
\tag{6}
\]

因此在 `g≠0` 的光滑局部，若仍要求缩放后的向量场是某个标量势的梯度，必须有 `grad a` 与 `g` 平行。局部而言，`a` 只能随 `ell` 改变、不能沿其连通等值面任意改变；可以写为 `a=f(ell)`，进而 `a grad ell=grad A(ell)`，其中 `A'=f`。全局结论还需处理不连通等值层与区域拓扑。

纯时间调度 `a(t)` 在每个固定时间不引入这种空间非保守性。任意依赖 `||g||` 的反馈一般没有该保证：同一个证据等值层上，证据梯度大小完全可以不同。实现中对系数 `stop_gradient` 也不会改变相邻输入下实际系数发生变化这一事实。

**精确反例：** 取合法 posterior `R(x,y)=.5 exp[-(x²+2y²)/2]`，则 `g=(-x,-2y)`。用 `a=1+lambda ||g||²` 后，二维旋度为

\[
\partial_x\widetilde g_y-\partial_y\widetilde g_x=4\lambda xy.
\]

在 `(1,1)`、`lambda=.2` 时，旋度精确为 `.8`；CPU 有限差分得到 `0.8000000000008`。原始 gap 是精确梯度，非保守性完全由缩放规则引入。[核对数据](probability_analytic_checks.json)

**不能过度推广：** 这仅约束“控制器仍有纯密度梯度/标量证据倾斜解释”的主张。非保守控制场也可以正确运输某个分布，甚至改善图像质量；零旋度也不能证明实际采样获得了预期终点分布。具有缓存或历史依赖的控制器还需要先明确扩展状态上的数学对象，不能直接把历史系数当作固定的 `a(z)`。

### 2.4 最小可操作研究，而非立即加一个投影方法

1. 用已知后验的二维或低维相容 Gaussian FM 模型，明确终点目标为式 (3)，比较普通 CFG、仅时间缩放、状态反馈。分别测目标证据层权重与层内分布误差，不能只测类别成功率。
2. 对状态反馈额外测局部闭环积分或 Jacobian 反对称部分，判断它是否失去所声称的势函数解释；在证据梯度大小不同但证据相同的位置作配对比较。
3. 真实图像中不存在可靠的真实后验时，分类器置信度只能当代理。按窄置信度分箱再看多样性会受到分类器失准、箱宽和选样偏差影响；这只能形成待验证读数，不能宣称已经检验精确等值层定理。

这条路线与仓库“未来语义收益—外观代价”的主张相容，新增的是明确的等证据分布保持关系，以及对反馈缩放能否保留梯度身份的可检验限制。它不是把旧的 gap 范数控制换一个名称。

**文献重叠：** 用置信度自适应指导强度本身已有直接先例。[GFCG 的 WACV 2026 原文](https://openaccess.thecvf.com/content/WACV2026/papers/Shenoy_Gradient-Free_Classifier_Guidance_for_Diffusion_Model_Sampling_WACV_2026_paper.pdf)包含分类器置信度驱动的强度和参考类别选择，因此不能把 `a=f(confidence)` 宣称为新方法。这里的潜在区别只能是：使用的证据是否与当前 CFG gap 的势函数相容，以及保持这种相容性是否有额外作用。如果置信度来自预测 clean 图的分类器，它一般不等于当前 noisy-state posterior；即使系数形式看起来是置信度函数，也不能直接推出式 (6) 为零。

## 3. 第二个不变量：全部类别的归一化必须在一阶和二阶上自洽

### 3.1 一阶关系已在仓库做过，不应重复计为新 idea

由 `sum_c w_c=1` 和式 (1)，有

\[
\boxed{\sum_c w_c g_c=0,\qquad s_u=\sum_c w_c s_c.}
\tag{7}
\]

这要求 null score 位于完整条件 score 集合的凸包中。它要求后验权重，不是均匀权重；类先验必须与实际联合分布一致，不能随机选少数类别后仍称完整必要关系。

仓库 [2026-09-12 主线报告](../../CFG_IG_RESEARCH_DIRECTIONS_20260912_ZH.md) 已经实现完整 100 类凸包/仿射共同分量诊断，并记录中低噪声 5%–10% 量级的共同正交 gap 能量。因此这里保留它作为已知必要关系，不重新包装新颖性。

### 3.2 真正可补充的是二阶归一化兼容性

对式 (7) 再微分，使用 `grad w_c=w_c g_c`，得到

\[
\boxed{\sum_c w_c\left[Jg_c+g_cg_c^{\mathsf T}\right]=0.}
\tag{8}
\]

因此

\[
\sum_c w_c Jg_c=-F,\qquad
F=\sum_c w_c g_cg_c^{\mathsf T}\succeq0,
\tag{9}
\]

以及 trace 版本

\[
\sum_cw_c\operatorname{div}g_c=-\sum_cw_c\|g_c\|^2.
\tag{10}
\]

这是把经典 Fisher 信息等式应用于以状态 `z` 参数化的类别分布 `w_c(z)`；不是新数学定理。此处为有限类别，证明只需对概率之和直接求两次导数。它比单点凸包关系增加了**权重随状态变化也必须与 gap 相容**的要求。

一个一阶检查会遗漏的反例：二类 posterior `w=sigmoid(a z)`，

\[
g_1=(1-w)a,\quad g_0=-wa,\quad
Jg_1=Jg_0=-w(1-w)a^2.
\]

同时将两个 gap 乘 `kappa`，原后验加权的一阶和仍为零；二阶残差却为

\[
\kappa(\kappa-1)w(1-w)a^2.
\]

在 `a=1.7,z=0,kappa=1.4` 时为 `0.4046`，未缩放精确场的 CPU 二阶误差不超过 `5.6e-17`。`kappa=0` 的全零 gap 也会通过两个恒等式，这再次说明兼容性并不是充分的模型质量检验。[解析数据](probability_analytic_checks.json)

### 3.3 一个不需要另训 posterior 头的有限诊断

从真实联合分布采样 `(X,C)`，再独立生成 Gaussian 噪声形成 `(Z_t,C)`，则条件于 `Z_t` 的标签恰遵循真实后验。因此理想场应满足

\[
\mathbb E[g_C(Z_t,t)\mid Z_t]=0,
\]

\[
\mathbb E[Jg_C+g_Cg_C^{\mathsf T}\mid Z_t]=0.
\tag{11}
\]

由此可直接在真实加噪样本上，用其实际配对标签查询 conditional/null 两路，估计总体或预设状态分层的弱形式。无需先用一个未经校准的神经 posterior 估计器替换真实标签，也无需对每个状态穷举类别。

例如用独立随机向量 `u`、`E[uu^T]=I`，单次随机投影

\[
u^{\mathsf T}Jg_Cu+(u^{\mathsf T}g_C)^2
\tag{12}
\]

可用于 trace 的随机估计；JVP/有限差分成本必须单列。预设的标量状态权重 `h(Z_t)` 可以乘在式 (11) 外部做局部弱检验，但不能事后按符号挑选有利区域。

**解释限制：** 它只检验网络场与真实联合数据相容的某些必要矩关系；有限样本高维方差可能很大，多个区域误差也会总体抵消。guided rollout 的指定生成类别不是该状态下从真实后验抽出的标签，不能把同一估计式直接搬过去。

更不能把每点凸包拟合出的任意一组权重，自动当成可微 posterior，再计算式 (8)；权重本身需要满足 `grad w_c=w_c g_c`，这恰是新增的要求。

### 3.4 为什么这不能变成“CFG 的倍数必须等于一”

普通 CFG 故意放大类条件相对 null 的 gap。如果把放大后的所有类别场仍与原始 null 当成同一固定联合分布来套式 (8)，可能违反必要关系；但正确结论是这个**固定联合分布解释**失效，而不是图像控制必然无效。

更一般地，所有类别共用任意可微的状态系数 `a(z)`，令 `g'_c=a(z)g_c`，仍用原后验加权，则

\[
\sum_cw_c\left[Jg'_c+g'_cg_c^{\prime\mathsf T}\right]
=a(a-1)F.
\tag{13}
\]

因为 `grad a` 引起的额外项等于 `(sum_cw_cg_c)(grad a)^T=0`。因此把这一残差直接作为指导后的最小化目标，会偏向 `a=1` 或无条件退化解 `a=0`，即使放大本来就是有意的生成控制。**本轮式 (8–12) 明确只建议作为未放大 conditional/null 原始分支的相容性诊断。**

事实上，对原 posterior 做温度变换 `r_c=w_c^kappa/sum_j w_j^kappa`，真正归一化后的 posterior 梯度为

\[
\nabla\log r_c=\kappa g_c
-\nabla\log\sum_jw_j^\kappa.
\]

这个共有归一化项通常不为零。另一方面，若分别给每类密度作 CFG 式幂重加权，其混合后的无条件密度一般已经改变，不能继续用原 null 充当它的精确无条件 score。

这些关系帮助明确“保持哪一个概率结构”是新增的建模选择。它们不直接提供一个可部署的免费修正：真实 posterior 难以获得，瞬时相容也不等于跨时间运输相容。

## 4. 三个容易误判的边界不变量

### 4.1 纯噪声时 score 相同，FM 速度却可以不同

在 `t=0`，`Z_0=epsilon` 独立于类别，所以 `w_c=pi_c`、所有 score 都是 `-z`，score gap 严格为零。

但理想 FM 速度是

\[
v_c(z,0)=\mathbb E[X\mid C=c]-z,\qquad
v_u(z,0)=\mathbb E[X]-z.
\]

若各类均值不同，速度 gap 非零。式 (2) 的 `t/(1-t)` 恰使 score gap 仍为零；不能在 `t=0` 反除这个退化系数。

**CPU 例子：** 对称两类 `N(-1,1)` 与 `N(1,1)`，在 `z=.7,t=0` 时条件速度 `.3`、null 速度 `-.7`，速度 gap 为 `1`，score gap 为 `0`。因此“纯噪声没有类别信息，所以所有 FM 速度必须相同”会直接处罚精确模型。[数据](probability_analytic_checks.json)

### 4.2 零旋度不意味着正确的 guided 概率路径

固定 `t` 和空间常数 `gamma`，精确 CFG score

\[
s_u+\gamma g_c
=\nabla\log[p_t(z)w_c(z,t)^\gamma]
\]

确实是保守场。但逐时密度幂重加权与“先重加权终点分布、再按训练核加噪”通常不交换，所以该 score 族不一定对应所声称终点目标的加噪边缘。已有原始论文用精确模型反例明确说明 DDIM/DDPM CFG 不普遍采样出终点幂密度。[Classifier-Free Guidance is a Predictor-Corrector，§3](https://arxiv.org/html/2408.09000v2#S3)、[What does guidance do?](https://arxiv.org/abs/2409.13074)

对普通光滑 flow，可存在非保守速度而仍正确运输密度。因此式 (6) 应作为概率梯度解释审查，不能作为所有有效 flow 控制器的硬禁令。

### 4.3 真正的重复条件化幂等，固定软似然倾斜不幂等

对事件 `A`，条件化算子

\[
C_A(P)(B)=\frac{P(B\cap A)}{P(A)}
\]

满足 `C_A(C_A(P))=C_A(P)`。已经知道同一件事，再重复告知它不会增加证据。

但固定似然 `L` 的倾斜 `B_L(P)∝LP` 重复两次得到 `L²P`，一般不幂等。如果 `L=1_A`，所有正次幂相同，类内相对分布确实应该保持；如果 `L` 是软置信度，重复乘权会进一步集中。

不能因此说采样过程中每次 CFG 更新都在错误地重复使用同一条证据：各时刻的状态、条件后验及动力学任务不同。也不能看到当前图片像猫，就把原始 conditional/null 模型之间的所有差都规定为零；两个模型仍然代表原始联合分布的不同条件化，而不是以“当前已经确定猫”为新的完整联合分布。

## 5. 与仓库既有成功/失败的关系

- [2026-09-11 基础假设](../../GUIDANCE_FUNDAMENTAL_HYPOTHESIS_20260911_ZH.md) 已有终点 KL 倾斜、未来价值和逐时幂密度不相容的完整推导。这里的新增重点是式 (4) 的等证据条件分布保持与式 (6) 的状态反馈可积性，而非重述“CFG 不是精确终点 tilt”。
- [2026-09-12 CFG/IG 主线](../../CFG_IG_RESEARCH_DIRECTIONS_20260912_ZH.md) 已测完整类别凸包及共同正交分量。式 (8–12) 是可补充的二阶兼容性诊断，不把旧的一阶几何诊断重命名。
- [posterior/null 读出结果](../../GUIDANCE_REFERENCE_LOSS_RESEARCH_20260912_ZH.md) 已完成，配对 conditional teacher 的 1K FID 为 `45.6688`，同预算 real/FM 为 `45.5388`，APG 为 `44.5038`，没有通过推进规则。本次不据 Bayes 恒等式成立就复活该训练；必要关系成立与一种训练损失提高生成质量，是不同命题。
- 全零 gap、错误但自洽的联合分布都可能通过兼容性关系。修正兼容性也可能删除有益的模型误差抵消。最终生成收益仍须独立检验。

## 6. 本轮建议的优先级

**第一优先：等条件证据下应该保留什么。** 用明确的目标密度和相容解析 flow，测条件证据层内失真；同时判断 CFG-CTRL 类状态反馈是否保留它所声称的梯度结构。它最直接接上“改该改的，保留条件没有要求改变的部分”，但把“该改/不该改”变成概率对象，而非任意图像距离。

**第二优先：未放大条件/null 场的归一化兼容性。** 在真实加噪联合样本上做一阶与二阶必要关系诊断，检查现有共同分量是否只是部分误差。它是机制审查入口，暂不导出新 null 头训练或新采样排名。

本轮 CPU 只核对了四类解析事实：重加权保持层内概率、Bayes/Fisher 恒等式及其缩放反例、纯噪声 score/velocity 区别、范数反馈产生旋度。结果见 [probability_analytic_checks.json](probability_analytic_checks.json)，不构成真实图像质量证据。

## 原始来源

1. Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance*. 2022。[原文](https://arxiv.org/abs/2207.12598)
2. Yaron Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023。[原文](https://arxiv.org/html/2210.02747v2)
3. Arwen Bradley, Preetum Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector*. 本文核对 2024 v2 §3 的精确分布反例，不把其 SDE predictor-corrector 定理直接套到任意 FM ODE。[原文](https://arxiv.org/html/2408.09000v2#S3)
4. Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, Jianfeng Lu. *What does guidance do? A fine-grained analysis in a simple setting*. 2024。[原文](https://arxiv.org/abs/2409.13074)
5. Rahul Shenoy et al. *Gradient-Free Classifier Guidance for Diffusion Model Sampling*. WACV 2026。[正式原文](https://openaccess.thecvf.com/content/WACV2026/papers/Shenoy_Gradient-Free_Classifier_Guidance_for_Diffusion_Model_Sampling_WACV_2026_paper.pdf)

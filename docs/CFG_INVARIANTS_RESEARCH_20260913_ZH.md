# CFG 的不变量：条件证据、概率相容性与反馈控制

日期：2026-09-13。问题：能否把“没有增加要求，重复操作应保留原有内容”的思想引入 CFG？本轮进行了三路独立审查、交叉反驳、原始文献核对和 CPU 解析验证。没有启动新的 GPU 调参。本报告区分已证明的关系、已知文献、研究假设与尚无质量证据的方向。

**结论：有可以严格定义的“不变”，但需要说明保留的对象。普通 CFG 已保留当前预测中两分支共有的分量；理想条件模型必须满足 Bayes 归一化的相容关系；如果主动规定只按条件证据重新加权，则同等证据下的相对分布必须保留。第三条最接近原始直觉，也能对 CFG-CTRL 的反馈形式提出约束。它不意味着一张图片、一个 latent、背景或预测 clean 图应该沿任意生成操作保持不动。**

## 1. 先把三种“必须”分开

| 层次 | 不变量或相容关系 | 何时必须成立 | 能否直接证明生成质量 |
|---|---|---|---|
| 算式与表示 | 共同分量、共同仿射换元、相同输入与一致历史的退化 | 声称实现的是普通 CFG 或同一控制器的另一种表示 | 不能 |
| 概率模型 | conditional/null 的 Bayes 混合关系、概率归一化的一阶与二阶关系 | 两路精确对应同一个联合分布 | 不能；错误分布也可自洽 |
| 指定生成目标 | 等条件证据层内的分布、同一终点目标在不同噪声层的相容性 | 选择了明确的证据倾斜目标和 corruption | 可以检验是否实现目标，但目标本身仍需论证 |
| 动力学与求解 | 剩余终点映射沿其自身精确流不变 | 固定向量场、条件、时间约定和唯一流 | 主要检查算子与数值误差 |

以下采用 `v_w=v_u+w(v_c-v_u)`，`w=1` 是原条件模型；如果实现写成 `v_c+gamma(v_c-v_u)`，则 `gamma=w-1`。score 与 FM velocity 分开记，不能互换端点结论。标准 CFG 的两分支组合和质量／多样性权衡来自 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 2. 最直接的严格不变量：当前两分支共有的分量

在同一 `(z,t,c)`，记 `d=v_c-v_u`。对任意满足 `n^T d=0` 的线性读数，有

\[
\boxed{n^\top v_w=n^\top v_u=n^\top v_c.}
\]

因此普通 CFG 只改变两路有差异的方向。即使 `w>1`，在当前 gap 的正交补上，两路共同预测的部分也不被修改。这是算式本身的定理，不依赖网络正确。

更普遍地，对同一仿射表示变换 `T(v)=Av+b`，

\[
G_w(Av_c+b,Av_u+b)=A G_w(v_c,v_u)+b.
\]

这里 `A,b` 可以依赖当前共享的状态和时间，但必须在两分支间相同。普通 CFG 的这种表示一致性，比“某个 norm 必须不变”更可靠。

### 为什么它不能推出背景不变

该性质只比较**同一状态**下的瞬时预测。用简单 ODE 反例：

\[
v_u(x_1,x_2)=(0,x_1),\qquad v_c(x_1,x_2)=(1,x_1).
\]

第二分量在同一状态完全共有。由 `(0,0)` 积分到时间 1，条件流终点为 `(1,1/2)`，`w=2` 的 CFG 终点却为 `(2,1)`。第一坐标的改变会通过共同动力学影响第二坐标。这个例子只反驳“局部共同分量 ⇒ 配对终点保持”，不声称它就是某个 Gaussian FM 的概率模型。

### 与 CFG-CTRL 的实际联系

逐坐标 `sign` 修正一般不再沿原 gap，因此会改变上述共同分量。此次直接导入仓库保存的作者代码，以默认 `lambda=.05,K=.3`，取 `v_u=(0,0)`、`d=(2,1)`、`w=2`，得到输出 `(3.4,1.4)`；在原 gap 正交方向 `(1,-2)/sqrt(5)` 上出现 `0.268328` 的新分量。相同例子旋转坐标再换回，输出差为 `0.581917`。这说明它与普通 CFG 的代数契约不同，不等于该控制器必然无效。[CFG-Ctrl 原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_CFG-Ctrl_Control-Based_Classifier-Free_Diffusion_Guidance_CVPR_2026_paper.pdf)；[本轮代码与结果](../experiments/audit_cfg_invariants_20260913.py)。

这条性质可以成为受控消融的依据，但单独投影回原 gap 只会得到某种自适应标量 CFG。仓库也已做过 sign 与同幅值 gap 的比较，不能重新命名为新方法。

## 3. “重复同一个条件不应继续改变”：哪里成立，哪里不成立

对同一个硬事件 `A` 条件化，确实幂等：

\[
\mathcal C_A(P)(B)=\frac{P(B\cap A)}{P(A)},\qquad
\mathcal C_A(\mathcal C_A(P))=\mathcal C_A(P).
\]

知道“这是一只猫”后，再知道**同一件事**，没有新信息。但把同一固定似然 `L` 再乘一次，是另外一种操作：

\[
\mathcal B_L(P)\propto LP,\qquad
\mathcal B_L^2(P)\propto L^2P.
\]

软似然重加权通常不幂等。我们的六状态例子中，乘一次和两次的分布 L1 差为 `0.334304`。如果把重复的同一证据当作独立证据使用，就产生重复计数。已经条件化的联合模型与原来的联合模型也不能混用：前者的条件事件已几乎必然成立，后者的固定 conditional/null 网络并没有随之重新定义。

**普通 CFG 组合器同样不幂等。** 固定参考 `u`，令 `C_w(v)=u+w(v-u)`，则

\[
\boxed{C_b(C_a(v))=C_{ab}(v).}
\]

当 `w>1` 且 gap 非零，要求 `C_w^2=C_w` 与其定义冲突。这只分析固定输入和固定参考的组合，不能把真实采样的连续时间步骤说成反复乘同一个似然；沿途状态和噪声层都会改变。

可以把幂等性放到“投影到固定允许集合”的操作上。例如非空闭凸集合的欧氏投影满足 `P_A(P_A(y))=P_A(y)`。但难题是定义哪个集合值得保护，而不是实现投影公式；若第二次更新了集合或历史，已经不是同一个算子的幂等性。

## 4. 最接近原始直觉的不变量：等条件证据下保留相对分布

固定基础分布 `P`、证据量 `R(X)` 和非负权重函数 `h`，主动指定

\[
Q(dX)=\frac{h(R(X))}{Z}P(dX),\qquad 0<Z<\infty.
\]

那么在正权重的证据层上，按正则条件分布的几乎处处意义，有

\[
\boxed{Q(dX\mid R=r)=P(dX\mid R=r).}
\tag{1}
\]

证明：把 `P` 分解为 `P(dX|R=r) P_R(dr)`，重加权只改变第二项为 `h(r)P_R(dr)/Z`，第一项保持。这个写法避免把连续空间的零概率等值面当作普通事件。

取 `R(X)=p(c|X)`、`h(r)=r^w`，则模型可以提高“更符合条件”的区域所占比例，但不能在**条件证据完全相同**的一层中额外偏爱一种构图或纹理。在有密度的地方，若 `R(x_1)=R(x_2)>0`，则

\[
\frac{q(x_1)}{q(x_2)}=\frac{p(x_1)}{p(x_2)}.
\]

这是从所选重加权目标直接推出的结构；不是现行 CFG 采样器自动拥有的定理。

### 这比“无关细节不变”精确在哪里

若 `X=(S,N)` 且 `p(c|s,n)=p(c|s)`，则

\[
Q(N\mid S)=P(N\mid S).
\]

保护的是条件分布。如果背景 `N` 与主体状态 `S` 相关，改变主体状态的占比仍然可以改变背景的边缘分布，更不必逐图保留背景。只有额外独立或动力学解耦条件成立，才有更强的保持结论。

也不能把 clean 层的条件独立直接搬到每个噪声层：观察到的主体含噪后，背景可能为未知主体提供额外证据。高频、颜色、图像范数也没有天然的“条件无关”资格。

因此，我认为可以继承原观察的研究命题是：**在条件证据没有进一步区别的地方，guidance 是否引入了额外选择偏好？** 它允许图像变化，同时明确哪些分布变化属于目标之外。

## 5. 更强的时间相容性：所有噪声层应来自同一个终点目标

仅规定每个时刻的瞬时 score 还不够。假设 `C→X→Z_t`，即加噪只依赖 clean 图 `X`；记

\[
L(X)=p(c\mid X),\qquad
q_c(z,t)=p(c\mid Z_t=z)=\mathbb E[L(X)\mid Z_t=z].
\]

如果真正想采样的是终点目标

\[
Q_w(dX)=Z_w^{-1}L(X)^w P(dX),
\]

那么把**同一个** corruption 应用于该目标，得到的噪声层密度必须为

\[
\boxed{Q_{w,t}(z)=\frac{p_t(z)}{Z_w}
\underbrace{\mathbb E[L(X)^w\mid Z_t=z]}_{h_t(z)}.}
\tag{2}
\]

因此正确目标 score 是 `s_u+∇log h_t`。普通 CFG 的瞬时 score 却是

\[
s_u+w\nabla\log q_c
=s_u+\nabla\log\big(\mathbb E[L\mid Z_t]^w\big).
\]

一般来说，**先取条件期望再乘幂，与先乘幂再取条件期望不相同**。这就是一种有明确内容的跨噪声相容性，而不是要求每次 denoising 的 clean 预测相同。

例如 `w=2`：

\[
h_t=q_c^2+\operatorname{Var}(L\mid Z_t),
\]

\[
\nabla\log h_t-2\nabla\log q_c
=\nabla\log\left(1+\frac{\operatorname{Var}(L\mid Z_t)}{q_c^2}\right).
\tag{3}
\]

缺失项与含噪观察下尚未消除的证据不确定性有关，其方向没有统一符号；它不是凭空加一个减小 gap 的正则项。若标签是 clean 图的确定函数，`L` 为 0/1 指示量，则对所有 `w>0` 都有 `L^w=L`、`h_t=q_c`；终点 power 目标本身退化为原条件分布，仍需普通条件生成而不是取消条件。实际 CFG 在已有反例和实验中可以随 `w` 改变输出，这说明不能把两者不加区分地等同。

本轮还用六状态分布与一个固定、类别无关的两观测 corruption 核验证了式 (2–3)：正确的终点倾斜再加噪，与条件二阶矩公式一致到 `1.11e-16`；误用“条件均值的平方”所得归一化分布，L1 差为 `0.267152`。这是条件期望代数的离散核对，不是 FM 采样质量实验。

上述公式是本报告在指定联合模型下的直接推导，假设正噪声层满足所需的光滑性与微分交换条件。相关文献已经指出普通 CFG 一般不精确采样终点 power 分布：[Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/html/2408.09000v1) 给出反例与特定 SDE 极限下的 predictor–corrector 解释；[Conditional Diffusion Models with Classifier-Free Gibbs-like Guidance](https://arxiv.org/html/2505.21101v1) 分析所需的 Rényi 修正；[Analytic Distribution of Classifier-Free Guidance for Schedule Design](https://arxiv.org/html/2607.19725v2) 则给出确定性 CFG 的路径积分表达。仅凭中间场不同，不能断言某次采样的终点必然不同，因为不同路径可能抵达同一终点；本节识别的是指定加噪边缘与瞬时场的不相容。也不能宣称这个非交换问题是新发现。

研究上的关键选择是：究竟希望精确条件采样、证据幂倾斜，还是允许额外的质量筛选？若不先选目标，很容易把 CFG 有意带来的分布变化当成需要修掉的“不一致”。

## 6. 对 CFG-CTRL 最直接的约束：反馈沿等证据面应保持一致

固定 `t`，假设精确 score gap 为

\[
g=s_c-s_u=\nabla\ell,\qquad \ell=\log p(c\mid z).
\]

考虑无历史的标量反馈 `g_tilde=a(z)g(z)`。取 Jacobian 约定 `J_ij=partial_j g_i`，直接求导得到

\[
J_{\widetilde g}-J_{\widetilde g}^{\mathsf T}
=g(\nabla a)^{\mathsf T}-(\nabla a)g^{\mathsf T}.
\tag{4}
\]

在 `g≠0` 的光滑局部，若修改后的 gap 仍要解释成某个标量概率势的梯度，必须有 `∇a` 与 `g` 平行。局部可写成

\[
\boxed{a=f_t(\ell),\qquad a\nabla\ell=\nabla H_t(\ell),\quad H_t'=f_t.}
\tag{5}
\]

也就是说，系数不能沿同一个连通证据等值面随意改变。全局结论还要处理不连通等值层和区域拓扑。在一维上没有二维旋度障碍；这里真正约束的是多维输入。

**这给出一个可检验的问题：按 gap 范数或控制误差调节强度时，会不会对证据相同、仅梯度大小不同的状态施加不同偏好？** 纯时间系数在每个固定噪声层不引入这种空间问题，但任意依赖 `||g||` 的系数一般会。`stop_gradient(a)` 只改变自动微分路径，不会消除实际向量场随输入变化的系数。

解析反例使用合法 posterior

\[
p(c\mid x,y)=\tfrac12\exp[-(x^2+2y^2)/2],\qquad g=(-x,-2y).
\]

原 gap 是精确梯度。将它归一化为 `g/||g||` 后，在 `(1,1)`，Jacobian 反对称部分的 `(0,1)` 元素为 `2/5^(3/2)=0.1788854382`。CPU 有限差分得到 `0.1788854382206`。同例改用 `a=exp(ell)`，反对称残差仅 `6.94e-13`。

### 一个能说明思想、但还不能称新方法的例子

设 `q=p(c|z,t)`，可以考虑

\[
v_{\rm new}=v_u+[1+\gamma(1-q)](v_c-v_u),\qquad \gamma\ge0.
\tag{6}
\]

额外放大在后验接近 1 时趋于零，在证据不足时保留。其 score 势为 `H(ell)=(1+gamma)ell-gamma exp(ell)`，所以在理想相容模型下保持本节的梯度身份。**它不是幂等算子，也没有终点分布或质量保证。** “看起来已经是猫”不等于真实 noisy posterior 为 1；我们也不能直接从 gap 的大小恢复 posterior 的绝对值。

此外，置信度自适应指导已有直接先例：[Gradient-Free Classifier Guidance for Diffusion Model Sampling，WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Shenoy_Gradient-Free_Classifier_Guidance_for_Diffusion_Model_Sampling_WACV_2026_paper.pdf)。真正可能研究的差异是“反馈证据是否与当前 gap 的势相容”，而不是再次提出按置信度调强度。clean 预测上的分类器置信度一般不是 `p(c|z_t)`，不能套用式 (5) 的证明。

还必须保留两个边界：非保守速度场也可以正确运输分布；零旋度也不保证实现式 (2) 的跨时间目标。故本节是对**概率势解释**的限制，不是对所有有效控制器的禁令。具有历史的 CFG-CTRL 需要把缓存纳入状态后重新定义契约，不能直接当成这里的无历史标量函数。

## 7. 模型本身必须相容的东西：概率归一化的一阶和二阶关系

假设完整有限类别集合、固定先验 `pi_c`、正且光滑的密度满足

\[
p_t(z)=\sum_c\pi_c p_t(z\mid c),\quad
q_c=p_t(c\mid z),\quad g_c=s_c-s_u=\nabla\log q_c.
\]

由 `sum q_c=1`，直接得到

\[
\boxed{\sum_c q_cg_c=0,\qquad s_u=\sum_cq_cs_c.}
\tag{7}
\]

再求一次状态导数：

\[
\boxed{\sum_cq_c\left[Jg_c+g_cg_c^{\mathsf T}\right]=0.}
\tag{8}
\]

因此后验平均的 gap Jacobian 应当等于负的条件 Fisher 信息矩阵：

\[
\sum_cq_cJg_c=-F,\qquad F=\sum_cq_cg_cg_c^{\mathsf T}\succeq0.
\]

这是经典归一化／Fisher 恒等式在这里的应用，不是新定理。二阶条件比“这一点上 null 能被所有条件预测混合出来”更强：它要求邻域中的变化也相容。我们在二维、三类别的精确 Gaussian 混合的 18 个状态上，得到一阶最大残差 `2.17e-16`、二阶 `4.44e-16`，独立有限差分 Jacobian 误差 `1.35e-10`。

在一个点给全部条件 gap 加共同扰动 `B(z-z_*)`，该点的一阶关系仍完全成立，二阶残差却恰为 `B`。CPU 用 `B=diag(.2,-.15)` 验证了这个反例。

### 怎样避免再训练一个不可靠的 posterior 头

从真实配对 `(X,C)` 加独立 Gaussian 噪声得到 `(Z_t,C)`，标签条件于 `Z_t` 正好遵循真实后验。因此可以直接检验弱形式

\[
\mathbb E[Jg_C+g_Cg_C^\top]=0,
\]

或乘预先指定的状态权重／分层。用独立探针 `u`、`E[uu^T]=I`，读数

\[
u^\top Jg_Cu+(u^\top g_C)^2
\]

估计 trace，只需要 conditional/null 两路及额外 JVP 或有限差分。代价、置信区间和方差需要单列；总体正负误差也可能抵消。**每个样本的读数本来就不必为零，不能把逐样本平方残差当作必为零的损失。** guided rollout 上预先指定的类别也不是从真实 posterior 抽出的标签，不能照搬这个估计。

这里检验的是**相对于真实联合分布的必要相容性／正确性**：真实标签使用 `q_data(c|z)`，不一定等于网络自身模型的 `q_theta(c|z)`。所以非零残差可能来自内部不相容，也可能只是一个内部完全自洽的模型不符合数据，不能凭此辨认两者。严格反例：内部模型 `p_theta(z|C=±)=N(±a,1)`、均匀先验，真实标签却与状态独立且均匀。模型内部 Bayes 身份完全成立，但用真实标签平均后，条件二阶读数为 `2a^2 tanh^2(az)>0`（`z≠0`）。若要只诊断网络内部自洽性，还需要其自身 posterior 或额外辨识。

### 只能先诊断原始分支，不能直接惩罚放大后的 CFG

如果所有类别共用系数 `a(z)`，令 `g'_c=a g_c`，仍错误地使用原 posterior 和原 null，则

\[
\sum_cq_c[Jg'_c+g'_c g_c^{\prime\top}]=a(a-1)F.
\tag{9}
\]

`∇a` 项由式 (7) 抵消。因此直接最小化指导后残差会偏向 `a=1` 或 `a=0`，把有意的 guidance 放大一并取消。原始分支的相容性诊断与指导后的目标分布必须分开；所有分支塌缩为同一个场也会通过这些关系，仍需要独立的预测正确性检查。

在当前 canonical Gaussian FM `Z_t=tX+(1-t)epsilon` 中，对理想条件期望速度，

\[
s=\frac{t v-z}{1-t},\qquad
g_{\rm score}=\frac{t}{1-t}(v_c-v_u),\quad 0<t<1.
\tag{10}
\]

这是当前坐标的推导，不能漏时间因子，也不覆盖所有可自由选取的 FM 速度表示。[Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2) 给出 FM 框架。若参考分支实际是 negative prompt、弱模型或 IG 而不是真实 null，式 (7–9) 的全类别混合解释不自动成立。

## 8. 两个需要排除的“看起来必须不变”

### 8.1 普通 clean 预测不必沿轨迹恒定

固定精确流与条件，剩余终点映射 `F_t(z)=Phi_{t→1}(z)` 确实满足

\[
F_t(z_t)=F_s(z_s)
\]

沿自身轨迹不变。这个半群性质对坏模型的流同样成立。普通 posterior mean `m_t(z)=E[X|Z_t=z]` 则不是 `F_t`。

精确 Gaussian FM 反例：`X,epsilon` 独立标准正态，`V_t=t^2+(1-t)^2`，精确流为 `z_t=sqrt(V_t)z_0`，于是

\[
m_t(z_t)=\frac{t}{\sqrt{V_t}}z_0,
\qquad F_t(z_t)=\frac{z_t}{\sqrt{V_t}}=z_0.
\]

取 `z_0=1.2`，在五个时间点 `0,.25,.5,.75,1`，clean mean 为 `0,.37947,.84853,1.13842,1.2`，剩余终点始终 `1.2`。强制两者都恒定会惩罚精确模型。

[Consistency Models](https://arxiv.org/html/2303.01469v2) 使用同一 ODE 轨迹终点的一致性；[Consistent Diffusion Models](https://arxiv.org/html/2302.09057v1) 则涉及 denoiser 与自身反向 SDE 的条件期望／鞅相容性。后者要求的是正确随机过程下的期望关系，不是单条随机轨迹上的值不动。独立回灌噪声、改变 drift、固定 `(X,epsilon)` 的插值桥，都不能未经检查就套同一转移核的鞅结论。

### 8.2 纯噪声时 score gap 为零，不代表 FM velocity gap 为零

在 `t=0`，`Z_0=epsilon` 与类别独立，所以所有 score 都是 `-z`。但 canonical FM 速度为

\[
v_c(z,0)=E[X\mid c]-z,\qquad v_u(z,0)=E[X]-z.
\]

它们可以不同，因为速度描述接下来如何离开纯噪声端点。取 `z=.4`、条件均值 1、总体均值 0，则 score 都是 `-.4`，速度却为 `.6` 和 `-.4`。不能要求 FM 在这个端点“完全看不见条件”。

## 9. 与新论文的直接交集：保持指定密度，不等于总概率不丢失

[Probability-Conserving Flow Guidance，2026-05 预印本](https://arxiv.org/html/2605.20079v1) §3.3 提出：若 `v_t` 已运输参考密度 `p_t`，增加 `u_t` 后仍保持**同一条** `p_t` 的条件是

\[
\nabla\cdot(p_tu_t)=0
\iff \nabla\cdot u_t+u_t\cdot\nabla\log p_t=0.
\tag{11}
\]

AdaMaG 实际用相对条件 score 的分解与后期衰减近似处理，并没有精确执行这个等式。两项只要求相加为零，不要求分别为零；单独去掉 score 平行分量也不自动消除散度。

需要区别：良性的 ODE 推前实际分布仍然归一化，普通 CFG 不是必然把总概率质量“弄丢”；式 (11) 讨论的是是否保持指定参考密度。若同一初始分布下精确保留整条条件密度，最终总体分布和总体 FID 本来就相同，不能把完全不变同时作为改变总体质量分布的机制。

独立例子：标准 Gaussian 上 `u=Jz`、`J` 反对称时，式 (11) 成立，个体样本可以旋转；`u=az` 一般改变该 Gaussian 密度，但其 ODE 推前仍是归一化分布。CPU 已核对两者。这个例子再次说明逐样本不变、指定分布不变和概率总质量守恒是三个对象。

## 10. 仓库已有工作与本轮真正补充的内容

| 既有工作 | 本轮判断 |
|---|---|
| [CFG/IG 主线报告](CFG_IG_RESEARCH_DIRECTIONS_20260912_ZH.md) 已做完整 100 类凸包／共有正交分量检查 | 一阶 Bayes 身份不是新 idea；本轮补充邻域二阶相容性及其使用限制 |
| [posterior reference 结果](CFG_POSTERIOR_REFERENCE_RESULTS_20260912_ZH.md) 与 [null readout 结果](CFG_NULL_READOUT_RESULTS_20260913_ZH.md) 未支持继续扩大训练 | 不因为新的数学表述就恢复旧训练；真实配对标签的弱诊断可先绕过 posterior 头 |
| [FSG / CFG-CTRL / CFG-MP 比较](FSG_CFG_CTRL_CFG_MP_COMPARISON_20260911_ZH.md) 已分析局部 gap 与未来质量目标错位 | 本轮增加等证据层结构和反馈势相容性，保留“局部残差好不代表质量好”的结论 |
| [基本假设报告](GUIDANCE_FUNDAMENTAL_HYPOTHESIS_20260911_ZH.md) 已讨论未来终点与密度路径 | 本轮排除把普通 denoiser 当终点映射，给出严格 Gaussian 反例 |
| [APG 机制报告](APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md) 已讨论投影和参数化 | 不将一般投影、减小 norm 或 clean-space 转换重新计为新贡献 |
| [图像作为条件的 identity pilot](FM_SOURCE_CONDITION_IDENTITY_PILOT_20260913_ZH.md) 构造同场差分为零的精确保留 | 证明了算子可被设计为 identity；不证明模型更好，也不能把变更 CFG 的双场操作称为无操作回灌 |

本轮数学恒等式大多是已知概率与向量分析的直接应用。可能的研究贡献要落在**选定什么保留目标、如何测到实际偏离、这种偏离是否解释独立质量差异，以及如何低成本改进**，而不是给恒等式命名。

## 11. 收敛到两个值得先验证的问题

### 问题 A：反馈是在加强条件证据，还是增加了未声明的额外偏好？

这是我最推荐的概念主线，直接对接 CFG-CTRL。

先用已知 `P(X,C)` 和 canonical Gaussian corruption 的低维分布，选择式 (1–2) 的明确目标；同时包含两类例子：证据等值层存在不同梯度大小，以及主体与其他属性存在相关性。比较普通 CFG、仅时间系数、范数反馈、证据势相容反馈，单列作者 CFG-CTRL 的历史控制作为另一类机制。

预先固定读数：条件证据的总体提升、等证据层内的条件分布偏差、目标终点分布偏差、Jacobian 反对称部分、计算预算。范数反馈与证据反馈还要做有效指导幅度的配平，防止把单纯更弱的 guidance 当成结构收益。幂倾斜指数、图像质量和分类成功率不是同一个目标。

**继续条件：** 保留势结构在相同证据收益和近似相同指导预算下减少额外分布偏移，而不只是令旋度读数更小。否则式 (5) 只是一个解释契约，没有证明它值得控制。迁移到真实图像前，必须面对 noisy posterior 与 gap 相容的估计问题；不能用未校准分类器自证成功。

### 问题 B：原始 conditional/null 场的邻域变化是否与真实联合分布相容？

先做式 (8) 的真实加噪配对标签诊断，以少量预设噪声层、独立探针与置信区间回答：除旧凸包读数外，是否存在稳定、可定位的二阶相容性缺口？与原始预测误差、已有 failure 分层对照，避免总体平均抵消。

**继续条件：** 相容性残差不仅可重复，而且解释已有一阶／预测误差读数之外的失败差异。否则不训练修正头、不直接将残差加到 guided gap 上。即便诊断成立，内部不相容与数据分布误差、conditional 与 null 两路的误差来源，仍要独立辨认。

另外，CFG-CTRL 的单位与记忆一致性值得作为实现检查：canonical FM 有 `gap_epsilon=-t gap_v`、`gap_clean=(1-t)gap_v`。固定数值的 `K` 不能不换单位直接复用，历史也必须运输到当前单位；额外 Heun／探针查询不等于物理时间推进。仓库已经处理过部分求解器和历史更新问题，这适合检查迁移鲁棒性，不作为本轮主要新理论方向。

这两个问题当前都是研究假设，没有图像质量收益数字。本轮只完成了必要的定义、反例和解析审计；尚未运行上述分布比较或真实模型二阶诊断。

## 12. 本轮 CPU 验证与复现

运行：

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/audit_cfg_invariants_20260913.py
```

| 核对内容 | 结果 |
|---|---:|
| 普通 CFG 共同分量误差 | 0 |
| 共同仿射表示误差 | 1.67e-16 |
| 固定参考 CFG 系数相乘关系误差 | 0 |
| CFG 幂等性反例的两次／一次距离 | 4.47214 |
| 硬事件条件化幂等误差 | 5.55e-17 |
| 重复软似然重加权的分布 L1 差 | 0.334304 |
| 等证据层内条件分布误差 | 1.11e-16 |
| 终点倾斜与条件矩加噪公式误差 | 1.11e-16 |
| 误用条件均值平方的噪声层分布 L1 差 | 0.267152 |
| 精确混合模型 Bayes 一阶／二阶最大误差 | 2.17e-16 / 4.44e-16 |
| 二阶公式的独立有限差分核对 | 1.35e-10 |
| 精确 endpoint 映射沿流误差 | 0 |
| 精确 gradient 经 norm 归一化后的反对称项 | 0.1788854382 |
| 证据势相容反馈的反对称残差 | 6.94e-13 |
| Gaussian 旋转的加权散度残差 | 1.33e-17 |

这些数值证明脚本实现与相应解析例子一致，不是网络质量实验，也不能当作 FID、视觉收益或方法排名。

产物：

- [解析脚本](../experiments/audit_cfg_invariants_20260913.py)；[完整 JSON、作者代码哈希和脚本哈希](data/cfg_invariants_20260913/audit.json)。
- [等证据层数据](data/cfg_invariants_20260913/likelihood_fibers.csv)、[Bayes 相容性数据](data/cfg_invariants_20260913/bayes_identities.csv)、[clean mean 与终点映射数据](data/cfg_invariants_20260913/endpoint_vs_posterior.csv)。
- 独立审查：[概率推导](research/cfg_invariants_20260913/probability.md)、[几何与控制器审查](research/cfg_invariants_20260913/geometry.md)、[文献与反方核查](research/cfg_invariants_20260913/literature_debate.md)。

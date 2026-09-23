# 从局部密度响应到概率流保持：self-guidance 的两条理论推进

2026-09-23。接续 [RAM 复核](RAM_INTERNAL_GUIDANCE_REVIEW_20260923_ZH.md)。本轮研究原目标的局部梯度估计，以及非 canonical 概率流如何在改变终点分布时保留下来。推导、文献定位、Gaussian/离散 Heun CPU 核验；没有修改正式训练或启动图像实验。

**本轮最具体的进展：对当前确定性采样器，可以只扰动一个 Heun interval，学习这个局部干预造成的标量密度响应；它与未来收益预测结合后，梯度估计偏差是两个预测误差的乘积。另一个结构是：指数倾斜目标并不必然要求消除原有的保分布流，可以按密度比的倒数搬运它。前者直接服务当前终点目标，后者服务新的分布改进目标。**

这些不是已经验证的图像训练突破。Riesz 回归、双稳健估计和 Tilt Matching 均有直接先例；可研究的空间是将它们落实到受限 internal guidance、实际离散求解器，以及不必学习高维完整 score 的反馈结构。

## 1. 问题应当进一步缩小：我们究竟需要多少未来信息

固定生成器更新所用的 D，取 R 为当前 non-saturating generator loss 的负值。写实际离散采样器为

\[
X_{i+1}=F_{i,\theta}(X_i),\quad J(\theta)=E[R(X_N)].
\]

现有完整反传准确求这个离散目标的梯度。对于参数方向 δθ，它把局部状态变化 c_i=DθF_i δθ 与未来敏感度 ∇V_{i+1} 相乘：

\[
DJ[\delta\theta]=\sum_i E[c_i(X_i)\cdot\nabla V_{i+1}(X_{i+1})].
\tag{1}
\]

这里 V_{i+1}(y) 是从 y 出发，按当前后续 sampler 继续生成所得的终点收益。

完整 ∇V 是一个高维向量。但对固定 weak 的 schedule，每一步只有一个动作方向。我们只关心沿该方向的平均变化。理论上不必先重建整个未来敏感度，再取一次内积。

这并非仅凭“小头参数少”就获得免费梯度。joint weak 的可训练方向仍可能很多；能否压缩必须看实际方向空间。

## 2. 连续形式：终点梯度可以写成局部密度响应

暂用光滑连续流，q_t 为实际 rollout 边缘，b_t=Dθvθ δθ。定义

\[
\eta_{b,t}(x)
=-\frac{\nabla\cdot(q_tb_t)(x)}{q_t(x)}
=-\nabla\cdot b_t-b_t\cdot\nabla\log q_t.
\tag{2}
\]

它表示当前粒子沿 b 移动时，各位置的概率密度怎样一阶增减。适当可积性和零边界通量下，

\[
E[b\cdot\nabla f]=E[\eta_b f].
\]

代入未来 value，

\[
\boxed{
DJ[\delta\theta]=\int E[R(X_T)\eta_{b,t}(X_t)]dt.
}
\tag{3}
\]

这已经把未来 Jacobian 换成了标量终点奖励。但直接计算 Eq. (2) 仍可能很难：它需要实际 q_t 的 score，以及控制方向的散度。不能拿终点重新加噪的 score 直接替代。

这条恒等式来自连续性方程/Stein 分部积分，是已有数学。Newton Matching §8.3.1 的沿流密度比公式也直接相关。[Newton Matching](https://arxiv.org/abs/2609.05727)

它有一个很好的结构性质：若 ∇·(q_tb_t)=0，则 η_b=0，任何终点 reward 都看不到该方向的一阶变化。它自然忽略保分布运动，无需事先把整个速度场改成 canonical FM。

## 3. 不学完整 score：直接学我们需要的标量响应

考虑小网络 e(x,t)，最小化

\[
L_{\rm resp}(e)
=E[\tfrac12e(X_t,t)^2-b_t(X_t)\cdot\nabla_xe(X_t,t)].
\tag{4}
\]

由分部积分，

\[
L_{\rm resp}(e)
=\tfrac12\|e-\eta_b\|_{L^2(q)}^2
-\tfrac12\|\eta_b\|_{L^2(q)}^2.
\tag{5}
\]

无限函数类最优解准确为 η_b；受限类则拟合其投影。训练时 b 的数值 detach，只对小 e 求导，避免计算 b 的输入散度。

这属于 Riesz representer 的直接回归，并非新估计原理。已有自动 Riesz 回归和 RieszNet 文献给出了通用形式与统计理论。[Automatic Debiased ML via Riesz Regression](https://arxiv.org/abs/2104.14737)、[RieszNet](https://arxiv.org/abs/2110.03031)

这里对我们的特殊意义是：

\[
v=S+a(t)B,\quad B=S-W,\quad
\eta_{\delta a(t)B}=\delta a(t)\eta_B.
\]

weak 固定时，一个以时间为条件的标量响应网络可以服务所有 schedule bins，不必为每个像素/latent 坐标估计一份 score。

若控制有 m 个冻结方向 B_j，则学习 m 个响应。响应线性可组合。对状态 gate，准确公式则是

\[
\boxed{\eta_{a(x,t)B}=a\eta_B-B\cdot\nabla a.}
\tag{6}
\]

额外项就是状态 gate 的密度压缩作用。只有时间 gate 才能简单相乘。若 gate/e 的输入导数需要穿过 strong 特征提取器，局部 strong Jacobian 的成本会回来；detach 特征后求的导数不能冒充真正状态导数。

## 4. 两种不完美反馈可以互相纠偏

另训练 h_t≈V_t，用真实轨迹状态和终点回报做回归。把直接 value 反馈与密度响应反馈组合：

\[
\boxed{
G_{\rm DR}[b]
=\int E[b_t\cdot\nabla h_t+(R_T-h_t)e_t]dt.
}
\tag{7}
\]

在 h、e 固定时，准确偏差是

\[
\boxed{
G_{\rm DR}-DJ[b]
=\int E[(V_t-h_t)(e_t-\eta_{b,t})]dt.
}
\tag{8}
\]

因此有

\[
|G_{\rm DR}-DJ|
\le \|V-h\|_{L^2(dt\,q_t)}
\,\|e-\eta_b\|_{L^2(dt\,q_t)}.
\tag{9}
\]

只要两个预测中一个准确，人口估计就正确；两个都不准确时，误差按乘积出现。它不是要求先学出完美 critic 才能开始工作。

与单独用 ∇h 的不同之处在于：最终偏差由 value 的函数值误差控制，而非单独要求 h 的导数近似完整 costate。响应残差项抵消了对应的一阶误差。

这就是经典双稳健/Neyman 正交估计在平均方向导数上的应用。确定性策略的 DR gradient 也有直接文献，不能将乘积误差本身包装为新突破。[Huang/Jiang 2020](https://proceedings.mlr.press/v119/huang20b.html)、[Kallus/Uehara 2020](https://arxiv.org/abs/2006.03900)

跨样本拟合和评估必须控制依赖，例如 held-out split 或 cross-fitting；不能从人口恒等式直接推出同一小 batch 上训练再评估的有限样本无偏性。

## 5. 更适合当前代码的版本：仅前向的单步 Heun 反事实

连续式仍有输入导数，而且不能直接当作实际 Heun 的准确梯度。事实上，Heun predictor 本身也随参数变动，漏掉其导数就改变了更新。

可以在实际离散映射上重新定义整个估计问题。

固定一步输入 X_i、上游分布和全部后续 sampler，仅对该步参数方向做正负扰动：

\[
Y=F_{i,\theta}(X_i),\qquad
Y^\pm=F_{i,\theta\pm\epsilon\delta\theta}(X_i).
\]

这里的 θ±εδθ 只作用于这一步。对 schedule，它就是该 interval 的系数增减；对共享 weak 参数，它表示一次局部干预，不是把后续所有步一起改掉。

记 Y、Y+、Y− 的边缘为 q0、q+、q−。定义有限局部响应

\[
\eta_{i,\epsilon}(y)
=\frac{q_+(y)-q_-(y)}{2\epsilon q_0(y)}.
\tag{10}
\]

只要扰动边缘相对 q0 绝对连续、响应平方可积，对任意测试函数 f：

\[
E_{q_0}[\eta_{i,\epsilon}(Y)f(Y)]
=E\left[\frac{f(Y^+)-f(Y^-)}{2\epsilon}\right]
\equiv E[M_{i,\epsilon}(f)].
\tag{11}
\]

因此可以直接训练

\[
\boxed{
L_{i,\epsilon}(e)
=E[\tfrac12 e(Y)^2-M_{i,\epsilon}(e)].
}
\tag{12}
\]

人口最优解准确为 Eq. (10)。不需要显式密度、score、散度、逆映射，也不需要对 strong 求输入导数。两次局部 Heun 前向均 no-grad，只有小网络 e 的参数反传。

对 Heun，每个正负分支各有两个 field forward，朴素实现增加四个 field forward。当前步与不同分支能否缓存复用，应按实际头与 predictor 依赖分析，不能把两个 Heun forward 误数成两个 field forward。

### 5.1 为什么不用把 Y+、Y− 继续生成到终点

真实局部有限反事实是

\[
\Delta_{i,\epsilon}J
=\frac{E[V_{i+1}(Y^+)]-E[V_{i+1}(Y^-)]}{2\epsilon}.
\]

Eq. (11) 已经把这个量转成

\[
\Delta_{i,\epsilon}J
=E_{q_0}[R_T\eta_{i,\epsilon}(Y)].
\]

右边使用原始轨迹的 Y 与终点 R_T。它把“不同起点的未来收益比较”转换成了“原分布下的密度响应加权”。这解释了为什么不必对每个局部扰动重跑全部未来。

只需用 h 去近似未知未来，得到完全前向的双稳健形式：

\[
\boxed{
\widehat{\Delta}_{i,\epsilon}
=E[M_{i,\epsilon}(h)]
+E[(R_T-h(Y))e(Y)].
}
\tag{13}
\]

准确误差为

\[
\boxed{
\widehat{\Delta}_{i,\epsilon}-\Delta_{i,\epsilon}J
=E[(V_{i+1}-h)(e-\eta_{i,\epsilon})].
}
\tag{14}
\]

这里没有 h 或 e 的状态导数。h、e 可以使用 frozen 特征，只要在 Y、Y+、Y− 上确实计算对应特征，并计入成本；不再需要通过这些特征反传到 strong。

在参数方向三阶导数和换序条件成立时，有限反事实相对该步准确方向导数另有 O(ε²) 偏差。各步贡献相加近似共享参数方向导数，有限差分偏差的常数随各步累积；随机抽 interval 需按抽样概率加权，会增加方差。

整个结构的误差可以明确分成：

\[
\boxed{
\text{原离散梯度估计误差}
=O(\epsilon^2)
+\text{value error}\times\text{response error}
+\text{有限样本/时间抽样误差}.
}
\tag{15}
\]

有限 ε 不能宣称原梯度严格无偏。BF16 舍入也意味着 ε 不能任意缩小；有限精度下的差分与自动微分实现的导数可能不同，需在真实模型上核对。

### 5.2 一个有用的理解

确定性 transition 的 action likelihood 是退化的，不代表当前状态边缘没有光滑密度。Gaussian 初始噪声经过适当流之后仍可产生具有密度的边缘。

这里利用的是边缘分布对局部干预的响应，不是给 action 硬加探索噪声再套 policy likelihood。因此不要求将现有 ODE sampler 改成 SDE。

难点从长链求导转移到了局部分布响应学习。高维密度响应依然可能难学；原始噪声存在也不自动保证所需的绝对连续和 L² 有限条件。

## 6. 再减少一点要求：只匹配 value 误差所在的函数空间

有限函数类并不必然破坏整个估计。令响应模型类为闭线性空间 E（例如有限维空间），e 是 η 在当前 q0 下的人口 L² 正交投影。则

\[
E[(e-\eta)f]=0,\qquad f\in E.
\]

结合 Eq. (14)，若 V−h 恰落在 E 内，双稳健估计仍准确，即使 e 本身没有逐点恢复完整响应。

这给“两个小网络共享什么表示”一个具体标准：**响应模型应能校准 value 剩余误差的测试函数，而不是仅追求一个通用密度模型。**

共享同一表示本身不保证这一点。特别是若 h 也恰为 V 到同一 E 的正交投影，V−h 反而位于 E 的正交补；不能因此期待额外抵消。需要响应空间覆盖或测试实际 value 残差。

当 V−h 不在 E 内时，取其投影 ΠE，有

\[
|\text{bias}|
\le\|(I-\Pi_E)(V-h)\|_2\|e-\eta\|_2.
\tag{16}
\]

这属于 Riesz/Galerkin 正交结构，不是新的普适定理。它在本项目中的作用，是为低容量反馈头的功能分工给出可核验的原则。神经网络非凸训练、正则化和有限样本会产生额外投影误差；不能直接套用精确正交结论。

## 7. 第二条推进：改变终点分布时，不必抹掉旧概率流

这一节切换到分布改进问题，不宣称求当前 GAN/Heun 的同一个梯度。

固定 old 终点分布 q、其解析重加噪边缘 ρ_t，以及对应 canonical velocity m_t。假设 old 实际场的全部边缘正好也是 ρ_t，并写作

\[
v_t=m_t+d_t,\qquad\nabla\cdot(\rho_td_t)=0.
\tag{17}
\]

这个假设比“old 终点正确”更强。d 是保分布流，不要求等于零。

希望终点变成一次指数倾斜：

\[
q^+=q\,w,\quad
w(X)=\frac{e^{\lambda R(X)}}{E_qe^{\lambda R(X)}},\quad
h_t(y)=E[w(X)\mid Y_t=y].
\]

其重加噪密度是 ρt+=ρt ht，canonical velocity 为 mt+。现在定义

\[
\boxed{
v_t^+=m_t^++\frac{d_t}{h_t}.
}
\tag{18}
\]

因为

\[
\rho_t^+\frac{d_t}{h_t}=\rho_td_t,
\qquad
\nabla\cdot\left(\rho_t^+\frac{d_t}{h_t}\right)=0,
\]

Eq. (18) 准确生成新的边缘密度，同时保留旧的无散概率通量。

这提供了一个比“发现 canonical defect 后就消掉它”更温和的选择：**密度改变以后，调整保分布速度的大小，使其搬运的概率通量不变。**

需要 h>0、足够正则，以及控制 d/h 的增长来保证流良定。若 h 很小，修正速度可能很大；不能只凭代数恒等式保证数值稳定。

### 7.1 局部回归可以隐式实现这个变换

记 U 为解析插值的速度 target。考察以 old 场为 anchor 的隐式回归方程：

\[
v^+(y)
=v(y)+E[(w(X)-1)(U-v^+(y))\mid Y_t=y].
\tag{19}
\]

整理：

\[
h_tv^+
=v+E[wU\mid Y]-E[U\mid Y]
=d+h_tm^+,
\]

所以其解正是 Eq. (18)。

这种隐式指数倾斜回归属于 Tilt Matching 的直接近邻，不应将算法框架称为新发明。本轮有用的推导是说明：在 Eq. (17) 条件下，非 canonical 场也可以通过保留概率通量实现准确倾斜，而非必须先把 d 消成零。[Tilt Matching](https://arxiv.org/abs/2512.21829)

权重归一化有一个非常好的性质：奖励整体增加常数时，w 不变；若 reward 本来就是常数，w=1，target 恰为 old 场。

当前 class-conditional 情况应将这些期望理解为给定类别 c，并使用 Zc=E[exp(λR)|c]。否则跨类别的单一归一化不保证“每个类别的常数奖励都不更新”。同 batch 自归一化保持整体常数平移不变，但其统计偏差、类别稀疏和样本耦合仍需处理。

因此只要当前参数能复现 old 场，即使模型容量受限，常数奖励下也不会仅为了 canonicalization 而移动。这直接消除了上一轮 RAM 反例中的一种不必要更新来源。它不保证一般非恒定奖励下的受限最优。

小 λ 展开给出

\[
v^+-v
=\lambda\left[
\operatorname{Cov}(R,U\mid Y)
-(E[R\mid Y]-E[R])d
\right]+O(\lambda^2).
\tag{20}
\]

这里“奖励条件均值×速度缺陷”不是一律应删的污染项，它可以是正确搬运保分布通量所必需的修正。关键在于具体 reference、目标分布和归一化是否一致。

这不否定上一轮 RAM 分解：上一轮分析的是固定 strong anchor 的另一条更新，不能把两者混同。

固定 old 场与端点分布时，这个隐式回归的期望半梯度还准确等于

\[
E_{\rho_t}[J_f^\top h_t(f-v^+)]
=E_{\rho_t^+}[J_f^\top(f-v^+)].
\]

即在目标桥密度下拟合保留概率通量的新场。这为有限容量误差提供了明确度量；在适当 Lipschitz 条件下可以接标准同步耦合的终点 Wasserstein 误差界，但不等于有限容量下的终点 KL 最优。

如果 old 实际边缘不等于旧桥，令其桥连续性残差为

\[
e_t=\partial_t\rho_t+\nabla\cdot(\rho_tv_t).
\]

上述同一代数变换给出新桥残差 e_t^+=e_t。它保留无散通量，也会保留不一致的密度源项。因此实际需要分辨的是 div(ρd)，而不是一律以 ‖d‖ 大小判定错误。绝对残差保持不变，也不代表相对误差 e/ρ+ 或新终点误差不变。

## 8. 仍然存在的障碍，应该变成可测量的问题

对第一条路线：

- 标量输出不等于函数简单。响应可能高频或方差很大，高维状态下的样本复杂度仍需测量。
- Riesz 人口平方结构不保证有限样本训练稳定。很灵活的网络可让扰动点输出极大、base 点平方惩罚较小，导致过拟合；需正则化和独立验证矩条件。
- h 随当前 D 和后续 policy 变化；响应 e 不依赖 reward，可被多个 D 复用，但仍随当前 policy 和状态边缘变化。旧缓存不是自动 on-policy。
- time-only schedule 最直接。对全 weak 头应限制到明确的低维参数方向，或保留原精确梯度；随机方向估计整个高维向量仍有维度方差。
- 每次普通采样和奖励前向依然存在。只省未来反传，不会让整轮训练成本变为常数。
- 使用压缩状态表示时，必须证明它足以表达/校准所需 value 和 response；固定 Inception 特征未必足够描述中间动力学。

对第二条路线：

- 核心 Eq. (17) 假设实际边缘等于解析重加噪边缘；当前 guided 场没有这个保证。
- 无限函数空间固定点能实现，不代表 native 小头能实现。
- 每步更新的样本分布、old anchor 和 target 内的 candidate 场需要严格定义；不能把隐式固定点换成单次显式 target 后仍沿用全部保证。
- Z 的有限样本估计、权重退化及 h 接近零，都产生真实困难。

目前两条路线各有明确位置：局部反事实 Riesz 直接面对现有离散动力学；概率通量保持解释如何改进 RAM 类 surrogate 的几何兼容性。

## 9. 本轮数值核验与研究判断

CPU 的二维非交换线性场、四步 Heun 例子：

- interval=1 的单个 schedule 分量的精确离散导数为 −0.11791125586207858；密度响应形式为 −0.11791125586207860。这不是共享参数在所有 interval 上的总导数。
- 无穷小 Gaussian 密度响应 η 的二次多项式 Riesz 人口拟合，L² 误差约 1.58e−16。
- 局部 ε 从 .1 降到 .05，中心差分偏差从 2.42e−6 降到 6.05e−7；再减半则再缩小约四倍。
- value 或 response 任一精确时，有限反事实双稳健估计误差在 1e−15 左右。
- 两者误差幅度同时减半，估计偏差缩小四倍；这核验乘积关系，不说明原始偏差一定小。
- 用错边缘 score 的旋转例子中，真实梯度为 0，错误 score 却给出 1，确认“actual rollout”不是可省略的条件。

有限 ε 部分使用解析 Gaussian 密度计算 ηε，再核验有限反事实和双稳健等式；没有训练网络去学习这个有限 ε 响应。不能将上述人口积分核验解释为已经解决高维 response learning。

来源：[脚本](../../experiments/theory_self_guidance_20260922/local_riesz_audit.py)、[JSON](../research/self_guidance_ram_20260923/local_riesz_audit.json)、[图](../research/self_guidance_ram_20260923/local_riesz_audit.png)。

概率通量保持的独立二维例子中，目标终点协方差为 diag(1,2)。正确运输 d/h 后，32768 个 Sobol 初始点的终点二阶矩约为 [[1.000044,.000209],[.000209,1.999757]]；直接保留旧旋转 d 的对照场则得到 [[1.179465,−.360070],[−.360070,1.805607]]。后者由协方差 ODE 求得，前者的剩余误差主要来自有限样本积分。局部通量恒等式误差约 2.8e−17，隐式方程残差约 1.8e−15。来源：[脚本](../../experiments/theory_self_guidance_20260922/tilt_flux_audit.py)、[JSON](../research/self_guidance_ram_20260923/tilt_flux_audit.json)。

**优先值得验证的是单步反事实响应能否学得准，而不是马上训练整个生成器。** 对一个冻结 weak 的 checkpoint，用少量真实轨迹收集单步正负样本，检查独立测试函数上的 Riesz 矩条件，以及相对现有精确 schedule 梯度的误差、方差和成本。若这个响应都学不准，就不应继续用更大的框架掩盖问题。

对 joint weak，可先选 RAM 更新方向、当前精确梯度的低维历史子空间等，验证方向导数；子空间的选择和覆盖决定能优化什么。完全替代全头梯度不是本轮已经解决的事。

本轮文献核查也修正了新颖性判断：2026-09 的 Newton Matching 已深入讨论 canonical manifold、RAM 的一致性与 canonicalization；Tilt Matching 已研究指数倾斜动力学。我们应把候选贡献收缩到可操作的受限控制结构、概率通量解释和离散局部反馈效率，而不是泛称发现新的 flow-RL 理论。

## 10. 延伸材料

- [局部响应、Riesz、DR 和离散 Heun 完整推导](../research/self_guidance_ram_20260923/quotient_control_followup.md)。
- [桥反馈与通量保持推导](../research/self_guidance_ram_20260923/bridge_feedback_followup.md)。
- [前一轮 RAM 独立复核](RAM_INTERNAL_GUIDANCE_REVIEW_20260923_ZH.md)。

复现本轮局部响应 CPU 核验：

    /home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/theory_self_guidance_20260922/local_riesz_audit.py

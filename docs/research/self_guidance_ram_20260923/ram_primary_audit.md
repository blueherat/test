# RAM 原始论文与官方实现核查（2026-09-23）

本笔记针对用户贴文推荐的 RAM，区分论文的准确公式、工程近似，以及迁移到当前 guided field 时新增的问题。没有改动正式训练。

## 核查对象

- [RAM 论文 v1](https://arxiv.org/html/2605.10759v1)：重点 §3、§4、§4.1、表 1、附录 A/B/D。
- [作者项目页](https://bergmeister.ai/ram/)。
- [官方训练脚本](https://github.com/AndreasBergmeister/ram/blob/main/scripts/training_sd3.py)：`compute_loss`、`sample_epoch`、EMA 更新。

论文的 Eq. 17 与贴文给出的主要公式一致；它确实不需要采样轨迹的反向传播。但其理论身份是近似固定点回归，不是当前确定性 Heun 终点损失的无偏梯度替代品。原文表 1 直接把基本 RAM 标为 biased；§4.1 将它解释为指数倾斜路径积分的右端点求积。

以下以论文时间约定表示：数据端 t=0，噪声端 t=1。迁移到仓库时必须统一时间方向和预测参数化。

## 1. 必须区分三种速度

令 q 是当前实际 ODE 的终点分布。重新加噪产生

\[
X_t=(1-t)X_0+t\epsilon,\quad X_0\sim q,\quad\epsilon\sim N(0,I).
\]

定义该加噪分布的 canonical FM velocity：

\[
m_q(x,t)=E[\epsilon-X_0\mid X_t=x].
\]

它与生成 q 的任意 ODE 速度 v 不必相等。即使某个速度与 m_q 在所有时刻产生相同密度，还可以包含保分布流，因而不等于 m_q。

论文的桥 score 公式对应 m_q：

\[
\nabla_x\log q_{0|t}(X_0\mid x)
=\frac{1-t}{t}\{m_q(x,t)-(\epsilon-X_0)\}.
\]

将 m_q 换为当前 v 是 plug-in approximation。其逐点桥 score 误差恰好为

\[
\frac{1-t}{t}(v-m_q).
\]

因此当前 guided 速度 S+a(S-W) 能产生好的终点样本，不足以证明它可以作为其终点重加噪分布的精确 score/velocity。

论文使用的随机控制也与用户记号不同：其 SDE 控制 u 满足

\[
\sigma_t u_t=2(v_t-v_t^{ref}),\qquad \sigma_t^2=\frac{2t}{1-t}.
\]

用户的 velocity residual 是 v-S；不能在能量惩罚、KL 或 AM 公式里省略这个尺度转换。

## 2. memoryless 条件具体是什么

论文从固定的 forward noising SDE 出发，参考 backward SDE 是它的精确时间反转。条件于数据终点的整条加噪路径律与数据分布无关，且 t=1 的条件分布是 N(0,I)，完全忘记 X0。

因此按 exp(r(X0)) 倾斜路径律时，起始 Gaussian 不变，改变的仅是数据端分布。正是这个条件让最优 SDE 保留解析的重加噪桥。一般确定性 ODE 的路径律没有这种桥结构；固定其终点并不等于可以任意改变桥而不改变原参数优化问题。

当前设置还有一个初始化区别：以 S 为参考，现有 a=.75 且 W 不等于 S 的 guided 初始化已经有非零控制。把“初始化时 path cost 为零”的结论搬过来不成立。若改用当前 guided 作为参考，虽然控制初值归零，但参考场是否等于其自身终点分布的 m_q 仍需审查。

## 3. 三层近似不能合并成最优性定理

基本 RAM 的实践链条包括：

1. 从完整价值梯度中去掉积分控制成本的梯度。
2. 用当前 ODE 终点＋独立解析加噪，替代当前受控 SDE 的真实联合分布。
3. 用当前网络速度替代重加噪分布的 canonical FM velocity。

其中第二、三项在理想初始化和真正最优的 canonical 场具有相应一致性；第一项一般不在真正最优处消失。RAM 本身没有普遍的“唯一驻点就是 KL optimum”结论。

即便暂时把第二、三项设为精确，第一项仍然留下以下区别。定义指数倾斜族

\[
q_\lambda\propto p_{ref}e^{\lambda r},\qquad m_\lambda=m_{q_\lambda}.
\]

由条件期望直接微分可得

\[
\partial_\lambda m_\lambda
=E_{q_\lambda}[r(X_0)(\epsilon-X_0-m_\lambda)\mid X_t].
\]

准确的 m1-m0 是此式从 λ=0 到 1 的积分；RAM 固定点使用 λ=1 的单次值。只有这条路径足够接近仿射，或奖励足够小时，才能期待其接近完整积分。Gaussian reference＋线性 reward 是精确特例，不代表一般 reward。

## 4. 独立反例：真正 KL optimum 不是 RAM 固定点

取一维参考 N(0,1)，r(x0)=-x0²/2。准确的 KL 倾斜最优分布为 q*=N(0,1/2)。

对于 q=N(0,s)，canonical FM velocity 为

\[
m_s(x,t)=\frac{t-(1-t)s}{(1-t)^2s+t^2}x.
\]

在 t=1/2，有 m_ref=0，m*=2x/3。条件分布为 X0|Xt=x ~ N(2x/3,1/3)。于是

\[
E[r(X_0)(\epsilon-X_0-m_*)\mid X_t=x]=4x/9.
\]

RAM 固定点要求左侧等于 m*-m_ref=2x/3，二者相差 2x/9。这已经是在完全 canonical、自洽、无限容量情形下的偏差，不能归因于 guided 架构或估计噪声。

## 5. 独立推导：reward centering 在非自洽时改变人口更新

记 d=ε-X0，当前场 v，重加噪 canonical 场 m=E[d|Xt]。RAM 条件半梯度残差为

\[
H_r=v-v_{ref}-E[r(d-v)\mid X_t].
\]

将 reward 加常数 C，得到

\[
H_{r+C}-H_r=C(v-m).
\]

准确的指数倾斜分布对奖励常数完全不变。因此这个公式给出一种明确的一致性审计：若改变统一 baseline 会显著改变平均参数更新，就能检测到可训练切空间内的 v-m 缺陷。参数级差异为 2C E[J_v^T(v-m)]，并非必须估计整个高维 score。

同理，状态相关 baseline b(Xt) 的桥 score 无偏性质也需要正确桥 score。用错的 plug-in score 时，减 baseline 的作用不能仅解释为减方差。

实际有限 batch 的 group mean 还含自身样本，group standard deviation 也随机，另有有限样本效应；上式讨论的是确定的平移，不声称已完整分析实际 group normalization。

## 6. 官方实现与最简公式的工程差别

官方 `compute_loss` 中，目标使用 lagged/EMA 的 `old_v`，不是前导可训练速度；终点也由 old adapter 采样。模型主体冻结，更新分布在 Transformer 各层的 LoRA；贴文“原文训练整个大模型”不准确。虽然 LoRA 参数少，其导数仍穿过相应大模型层，当前外挂早期小头确实可能进一步省局部反传。

脚本的实际目标可以准确写为

\[
\widehat v=v_{base}+\rho A\,(\epsilon-X_0-v_{old}),
\qquad L=\operatorname{mean}\|v_\theta-\operatorname{sg}(\widehat v)\|^2.
\]

具体细节为：同 prompt 的 G=24 个奖励减 group mean，再除整个训练步 pooled reward std；ρ 在 GenEval/OCR 为 100，PickScore 为 1000。时间密度为 p(t)=2t；没有再乘控制空间产生的 4/σ²=2(1-t)/t 权重。每张图 K=8 个重新加噪状态；维度、K 和 batch 都取均值。old EMA 的 decay 为 .9、warmup rate .01。上述常数是论文实验配置，不是当前项目应直接复制的推荐值。

实践采用 group-relative rewards、全训练步标准差、任务 reward multiplier 和偏向噪声端的时间抽样。原文训练一次更新包含 48×24 个终点，每个终点 8 个训练状态。所谓最高 50 倍是其任务中达到 Flow-GRPO 峰值奖励所需更新数的比较；论文也报告相近的每步计算量，因此在其设置中有相应时间优势，但不是当前训练的 50 倍加速预测。

它在采样时使用 CFG，局部预测采用单次条件模型输出，这本身也超出了最理想的单个 canonical 场推导。这说明实践可容忍偏离理论；不能据理论不严格便断言方法不可用。

## 7. 对当前项目的结论

可以实现一个 no-grad guided rollout＋终点打分＋独立重加噪＋局部小头回归的 RAM baseline。它明确消除了完整轨迹反向和奖励网络反向，而且比先训练高维 value-gradient predictor 更简单。

但它会替换当前训练的更新规则和隐含目标，不只是对同一梯度进行计算优化。已有完整离散反传应保留作为少量审计依据，分别检查：

- RAM 更新与当前真实终点梯度是否同向，以及沿该方向的小步实际奖励变化。
- 固定 D 与 reward baseline 后的敏感性，避免把 D 更新造成的变化误认作算法效果。
- 参考是 S 还是已有 guided 场时，锚定到底要保留什么。
- 局部目标对当前受限弱头可实现方向的投影，而非默认能实现任意最优 velocity。
- 相同终点数、场计算数、实际时间的比较，而非只比较 optimizer steps。

这些边界不减损贴文最有价值的启发：终点质量反馈确实可以转化为局部回归；需要明确承认所利用的分布结构和由此引入的近似。

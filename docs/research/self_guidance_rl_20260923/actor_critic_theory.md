# 当前 Self-Guidance 的 actor–critic / adjoint 理论审计

日期：2026-09-23。本文只做理论与实现契约分析，没有改动生产训练。全篇使用**最小化 cost** 的记号；对应 RL reward 为 cost 的负数。

## 1. 先固定真正的优化问题

当前速度场为

\[
 f_\theta(x,t,c)=(1+a_\theta(t))S(x,t,c)
 -a_\theta(t)W_\phi(x,t,c),\qquad \theta=(\phi,a).
\]

其中 strong 冻结，但它对输入状态的 Jacobian 没有消失。一次 generator 更新中固定 discriminator \(D\)，令

\[
 J_D(\theta)=\mathbb E_{x_0,c}\ell_D(x_T,c),
 \qquad \ell_D=\operatorname{softplus}[-D(\operatorname{Inception}(\operatorname{Decode}(x_T)),c)].
\]

真实 sampler 的离散转移记为 \(x_{i+1}=F_i^\theta(x_i,c)\)。以下定理对这个**实际离散 sampler**成立，不先换成另一个 ODE/SDE。额外 gap anchor 可以作为独立的局部可微正则继续求导；它不必被包装为随机终点 reward。

梯度恒等式假设相应映射可微或几乎处处可微、所需矩有限，且可以交换梯度与期望。HJB 部分另加该节的光滑和无限函数类假设，不作为当前神经网络全局最优性的保证。

给定初始噪声后轨迹是确定的。随机初始噪声不意味着条件转移已经有可用于 PPO 的动作概率密度。

## 2. 完整 BPTT 已经是确定性 policy gradient

设 \(A_i=\partial_xF_i^\theta(x_i)\)、\(B_i=\partial_\theta F_i^\theta(x_i)\)，其中 \(B_i\) 求导时保持当前状态固定。定义 costate：

\[
 p_T=\nabla_x\ell_D(x_T),\qquad
 p_i=A_i^\top p_{i+1}.
\]

链式法则给出精确梯度

\[
 \boxed{\nabla_\theta J_D=\mathbb E\sum_{i=0}^{T-1}B_i^\top p_{i+1}.}
\]

若把所有可训练部分定义为动作 \(u_i=\mu_\theta(s_i)\)，并使环境 \(s_{i+1}=T_i(s_i,u_i)\) 与 \(\theta\) 无关，则

\[
 V_i^\mu(s)=V_{i+1}^\mu(T_i(s,\mu_\theta(s))),\quad
 Q_i^\mu(s,u)=V_{i+1}^\mu(T_i(s,u)),\quad V_T=\ell_D,
\]

\[
 \nabla_\theta J_D=
 \mathbb E_{s_i\sim d_i^{\mu_\theta}}
 \sum_i (\partial_\theta\mu_\theta(s_i))^\top
 \nabla_u Q_i^\mu(s_i,\mu_\theta(s_i)).
\]

这里 \(\nabla_uQ=T_{i,u}^\top\nabla V_{i+1}\)，而 \(p_i=\nabla V_i(s_i)\)。因此 Bellman、DPG 与 adjoint 是同一条链式法则的不同分解。新计算价值来自**学习并跨轨迹复用未来敏感度**，不是把 BPTT 改名为 RL。

这与 [Deterministic Policy Gradient](https://proceedings.mlr.press/v32/silver14.html) 和 [Stochastic Value Gradients](https://arxiv.org/abs/1510.09142) 的框架直接相连。后者已经给出了 model-free Q、短步模型加 V、完整可微 rollout 之间的连续谱。

## 3. 三种 critic 替代的计算契约不同

### 3.1 学习 \(Q(s,u)\)：完全免环境反传，但需要学动作敏感度

actor 使用 \((\partial_\theta\mu)^\top\nabla_u\widehat Q\)。strong 只做 forward。

这最适合少量 state-dependent guidance 系数、参考方向的混合权重等低维动作。动作若是完整 weak 输出，Q 必须在高维动作上提供可靠梯度，样本需求可能很大。

### 3.2 一步真实模型加 \(V(s')\)：只替代未来，不替代当前转移

actor 使用

\[
 (\partial_\theta\mu)^\top T_u^\top\nabla_{s'}\widehat V(s').
\]

对 Euler 和当前 additive guidance，保持当前 x 固定后 \(T_u\) 很简单，可以不反传 strong。对整个 Heun 区间，第一场改变 predictor，第二场依赖 predictor，因此精确 \(T_u\) 仍含 strong input Jacobian。不能把 predictor detach 后继续声称是当前 Heun 的精确梯度。

另一个陷阱：若 \(V_\psi\) 输入是冻结 strong 的特征 \(H(x)\)，则

\[
 \nabla_xV_\psi(H(x))=J_H(x)^\top\nabla_HV_\psi.
\]

冻结 H 参数仍需要 H 的输入反传。若目标是完全消除这部分成本，可让 V 使用便宜的 latent encoder，或直接预测下一节的向量 costate。将 H detach 后所得 feature 梯度并不是 x 的 costate。

### 3.3 直接学习向量 \(\widehat p(s)\)：把未来梯度作为监督信号

actor 在 detached states 上使用

\[
 \widehat g=\mathbb E\sum_iB_i^\top\operatorname{sg}(\widehat p_{i+1}),
\]

或等价的局部线性 surrogate \(\sum_i\langle\operatorname{sg}(\widehat p_{i+1}),F_i^\theta(x_i)\rangle\)。这里 \(\operatorname{sg}\) 表示本次 actor 更新不对 critic 求导。

向量 predictor 可以吃 detached strong 特征，不需要为了得到 \(\widehat p\) 而反传 strong。代价是输出高维向量，且该向量不自动是某个 scalar value 的梯度；需要验证它在 actor 实际使用方向上的误差。当前区间若仍当作整个 Heun 转移，局部 \(B_i\) 的成本仍在。

## 4. Heun 可以精确成为两阶段 MDP

当前 Heun 一区间内的时间系数 a 共享，令

\[
 f_1=(1+a)S(x,t)-aW_\phi(x,t),\qquad
 y=x+hf_1,
\]
\[
 f_2=(1+a)S(y,t+h)-aW_\phi(y,t+h),\qquad
 x'=\frac{x+y}{2}+\frac h2f_2.
\]

第一阶段的状态是 x，动作是 \((a,w_1)\)，转移到增广状态 \((x,y)\)。第二阶段的状态必须包含 \((x,y)\)，动作是 \((a,w_2)\)，转移到 x'。两个动作中的 a 是同一个参数所生成的值，而不是两个独立的新自由度。

若将 interval 系数随机化而仍保持共享语义，应在第一阶段只采样一次 a，并将它存入中间状态供第二阶段复用；两次独立采样会定义另一个随机 sampler。

假设下一完整区间的精确 costate 是 \(p'=\nabla V_{i+1}(x')\)，定义中间状态 value

\[
 U_i(x,y)=V_{i+1}\left(\frac{x+y}{2}+\frac h2 f_\theta(y,t+h)\right).
\]

对当前时间系数形式，\(\eta_y=\nabla_yU_i\) 满足

\[
 \eta_y=\frac12p'+\frac h2(\partial_y f_2)^\top p'.
\]

注意 \(\partial_yf_2\) 包含 strong 与 weak 对 y 的全部依赖。定义两个场的 cotangent

\[
 \boxed{c_1=h\eta_y,\qquad c_2=\frac h2p'.}
\]

则该区间的直接参数贡献精确等于

\[
 g_\phi= -a\left[(\partial_\phi W_\phi(x,t))^\top c_1
 +(\partial_\phi W_\phi(y,t+h))^\top c_2\right],
\]
\[
 g_a=\langle c_1,S(x,t)-W_\phi(x,t)\rangle
 +\langle c_2,S(y,t+h)-W_\phi(y,t+h)\rangle.
\]

对所有区间求和就是完整 sampler 梯度。计算这两个局部参数 Jacobian 时，可以把 x、y 和 strong 特征视为固定输入，strong 只 forward；被省掉的 strong 输入 Jacobian 信息已经存在于精确 c1、c2 中。

因此，一个严格而有用的研究契约是：**学习两个 solver-stage cotangent，以局部 weak/scale 反传代替完整 strong 链反传。** 用精确 cotangent 时没有改 sampler，也没有截断。用预测 cotangent 时误差被明确集中在 critic 上。

只给中间 critic 一个 y，一般不能保证 Markov 性，因为 \(x'\) 还显式依赖 x。可以给 critic x 与 y，或 x 与 predictor increment；只用强特征摘要需要额外的充分性假设。时间、阶段编号、类别也必须进入 critic。

这个分解并不凭空制造便宜的精确 cotangent。若每次都完整计算精确 c1、c2 再训练一次，计算并未减少。收益依赖少量精确 teacher 轨迹能否支持多次可信的局部更新。

## 5. 应当拟合 actor 用得到的梯度，而不是只拟合 return

对一般离散转移，若 \(e_{i+1}=\widehat p_{i+1}-p_{i+1}\)，则

\[
 \widehat g-g=\mathbb E\sum_iB_i^\top e_{i+1},
\]

\[
 \|\widehat g-g\|
 \leq\sum_i\mathbb E[\|B_i^\top e_{i+1}\|].
\]

这里最相关的是 \(B_i\) 可触及方向上的误差。若只训练 K 个系数，往往只需要 K 个动作敏感度，不必准确重建整个 latent costate；若联合 weak 输出能覆盖更多状态方向，要求也相应增加。不同时间误差可能抵消，逐项范数是保守上界。

对一次小步梯度下降，若 \(\|\widehat g-g\|<\|g\|\)，则
\(g^\top\widehat g>0\)，所以 \(-\widehat g\) 是局部下降方向。该条件还未计入有限步长曲率与不同轨迹估计噪声。

**反例 1：on-policy Q 的值可以全对，动作梯度仍任意。**

确定性 policy 只提供 \(u=\mu(s)\) 上的数据。对任意函数 \(b(s)\)，

\[
 \widehat Q(s,u)=Q(s,\mu(s))+b(s)^\top[u-\mu(s)]
\]

在所有 on-policy 动作上完全拟合 Q，但 \(\nabla_u\widehat Q=b(s)\) 任意。甚至无限多 on-policy state 数据也不能识别离开动作图的方向导数。需要动作探索、小范围分叉 rollout，或精确 adjoint 导数锚点。

**反例 2：很小的全局 value 误差也不控制梯度。**

\[
 \widehat Q(a)=Q(a)+\epsilon\sin(a/\epsilon^2)
\]

满足 \(\|\widehat Q-Q\|_\infty\leq\epsilon\)，但导数误差可达 \(1/\epsilon\)。因此 value MSE、TD residual 或终点 reward 相关系数都不能单独认证 actor 梯度。

拟合 value-gradient 已有长期前身。除 SVG 外，近期 [Distributional value gradients for stochastic environments](https://arxiv.org/abs/2601.20071) 也明确研究梯度信息与 Bellman 学习；这里可主张的研究内容应是本任务的低成本结构、stage 契约和梯度质量，而不是首次学习 value gradients。

## 6. RL 动作必须包含全部要优化的动力学参数

若只令 scale 随机化 \(a_i\sim\pi_\psi(\cdot\mid s_i)\)，而把 \(W_\phi\) 留在环境

\[
 s_{i+1}=T_i^\phi(s_i,a_i),
\]

则 REINFORCE 的 \(\mathbb E[\ell_D\sum_i\nabla_\psi\log\pi_\psi]\) 可以给出 scale policy 的梯度；它不会自动包含 \(\partial_\phi T_i^\phi\) 的动力学变化。即使 policy 网络显式依赖 \(\phi\)，score 项也只覆盖 policy 的依赖，不能补回环境的直接依赖。

最小反例：\(a\sim\mathcal N(0,1)\)、\(x'=\phi+a\)、\(\ell=x'\)。真实 \(\partial_\phi J=1\)，但动作分布与 \(\phi\) 无关，score estimator 给 0。

完整选择只有：

1. weak 固定，只训练低维 gate；此时环境真的固定。
2. 将 weak 的输出，或足以覆盖全部可训练变化的动作坐标，也放入 policy 动作；环境只消费动作与冻结 S。
3. 混合方法：gate 用 stochastic policy gradient / learned Q，weak 或方向字典用局部模型梯度与 learned costate。

把完整 weak 输出作为动作可以恢复形式上的 RL 正确性，但会带来高维动作探索问题。只对一小部分输出坐标加噪声，不能无条件得到其他未随机化方向的 score 梯度。

## 7. 直接 score-function 不是免费的反传替代

令一阶段动作 \(u=\mu+\sigma\varepsilon\in\mathbb R^d\)，\(\varepsilon\sim\mathcal N(0,I)\)，cost 是 \(G(u)=g^\top u\)。使用精确的均值 cost baseline 后，关于动作均值的 REINFORCE estimator 为

\[
 \widehat g=(g^\top\varepsilon)\varepsilon,
\qquad \mathbb E\widehat g=g,
\]

\[
 \boxed{\mathbb E\|\widehat g-g\|^2=(d+1)\|g\|^2.}
\]

这是最简单线性例子中的精确结果，不是任意神经参数空间方差的普适等式。参数空间还受 actor Jacobian、有效秩与共享参数结构影响。但它足以说明：把完整 high-dimensional weak vector 随机化，再给一个终点标量，不会天然比已有可微梯度省样本。减小 \(\sigma\) 也不会消除该线性例子的维度项；若 baseline 不准，还会出现随 \(1/\sigma^2\) 增长的项。

对非线性动力学，随机化优化的是 \(J_\sigma\)，而原来的确定性系统优化 \(J_0\)。在充分光滑和可交换极限条件下，可有 \(J_\sigma-J_0=O(\sigma^2)\) 的局部展开；常数受长程动力学敏感度控制。不能仅凭把探索标准差设小就宣称无偏优化原 sampler。

连续时间极限还依赖噪声标度：给 velocity 加固定方差、再乘步长 h，累计状态方差通常随 \(\sum h^2\) 消失；要得到非退化 SDE，velocity 噪声需随 \(1/\sqrt h\) 缩放。不同离散化会改变目标和 policy likelihood，不能混为同一 RL 实验。

[DDPO](https://arxiv.org/abs/2305.13301) 是将去噪视作序列决策并用 policy gradient 的明确前身。当前确定性 Heun 不能直接继承 stochastic denoising kernel 的 likelihood ratio；需要显式定义随机动作及其密度。

## 8. Q-Prop 型残差校正：critic 用于降方差，而不必完全接管梯度

固定 actor 当次访问的 state，令 policy 为 \(u=\mu_\theta(s)+\sigma\varepsilon\)，\(G_i\) 是从第 i 步起的真实 Monte Carlo cost-to-go。令 \(h(s)\) 是一个固定的动作梯度预测，\(b(s)\) 是 action-independent baseline。下式是关于 policy mean 的无偏校正：

\[
 \widehat g_\theta=
 \sum_i\left\{
 \nabla_\theta\log\pi_\theta(u_i\mid s_i)
 [G_i-b(s_i)-h(s_i)^\top(u_i-\mu_\theta(s_i))]
 +(\partial_\theta\mu_\theta(s_i))^\top h(s_i)
 \right\}.
\]

原因是 \(\mathbb E[\nabla_\theta\log\pi\,h^\top(u-\mu)]=(\partial_\theta\mu)^\top h\)。critic 线性预测被 score 项减去，再解析加回来，所以对任意固定 h 都保持原来的 stochastic-policy gradient 的期望。

在线性 cost 例子中，校正后的动作均值估计为

\[
 \widehat g=h+[(g-h)^\top\varepsilon]\varepsilon,
\qquad
 \mathbb E\|\widehat g-g\|^2=(d+1)\|g-h\|^2.
\]

这给 critic 一个清晰角色：只要预测到一部分真实动作敏感度，残差探索问题就会变容易。它也解释为何低维、结构化动作更适合先试。

边界条件：

- 上式写的是固定协方差的均值参数；学习噪声参数需另加相应项。
- h、b 对当次动作样本应当是事先固定的函数。为严格无偏，可使用旧 critic 或交叉拟合；不能任意用当前动作拟合出的 action-dependent statistic 充当 baseline。
- 使用真实 Monte Carlo tail、正确的 on-policy sampling、参数无关环境时成立。把 G 换成 biased TD target，或把旧 replay 当作当前 occupancy，不再自动无偏。
- 它保持的是 \(\nabla J_\sigma\)，不是原确定性 \(\nabla J_0\)。PPO clipping、组内样本标准化等额外处理也可能改变精确等式。
- 它仍不补回上一节中被留在环境里的 weak 参数梯度。

这是 [Q-Prop](https://arxiv.org/abs/1611.02247) 已有的控制变量思想，不应重新命名成独立理论突破。

## 9. 更贴合当前确定性目标的校正：随机少量精确反传

如果要保留当前确定性 Heun、当前 endpoint GAN objective，又不愿让 actor 完全依赖近似 critic，可以用标准的多精度控制变量。

固定当次 fresh batch \(\mathcal B\)、当前 \(\theta\)、当前 D 与调用前的 critic。设便宜的局部 critic 梯度是 \(\widetilde g_{\mathcal B}\)，完整反传梯度是 \(g_{\mathcal B}\)。先计算前者，再独立抽
\(I\sim\operatorname{Bernoulli}(p)\)，仅当 I=1 时支付完整反传成本。令

\[
 \boxed{\widehat g_{\mathcal B}
 =\widetilde g_{\mathcal B}
 +\frac I p(g_{\mathcal B}-\widetilde g_{\mathcal B}),\qquad p>0.}
\]

则条件于当次 batch 和调用前 critic，

\[
 \mathbb E_I\widehat g_{\mathcal B}=g_{\mathcal B},
\qquad
 \mathbb E_I\|\widehat g_{\mathcal B}-g_{\mathcal B}\|^2
 =\left(\frac1p-1\right)\|g_{\mathcal B}-\widetilde g_{\mathcal B}\|^2.
\]

对 fresh batch 再取期望，得到原确定性 GAN objective 的无偏梯度。总均方误差分解为

\[
 \mathbb E\|\widehat g-\nabla J_D\|^2
 =\mathbb E\|g_{\mathcal B}-\nabla J_D\|^2
 +\left(\frac1p-1\right)
 \mathbb E\|g_{\mathcal B}-\widetilde g_{\mathcal B}\|^2.
\]

因此 critic 可以是有偏的，但校正后的 gradient estimator 仍无偏。越准确的 critic 允许越小的 p；若 critic 很差，额外方差会抵消反传节省。若每次 forward 成本为 F、额外精确 backward 成本为 B、便宜预测与局部更新成本为 C，则理想期望成本约为 \(F+pB+C\)，需要再计入 critic 训练和可能的 replay/recompute 成本。

这条路线不需要探索噪声，不改变 ODE/Heun，直接使用当前 exact backward 作为可抽样的高精度梯度源。p=1 退回精确梯度；p=0 不允许使用无偏公式，而是纯近似 actor–critic。

实现契约与限制：

1. 两个梯度对应同一 batch、相同噪声、同一当前参数与固定 D，同样的 loss 归一化；不能拿上一轮精确梯度纠正本轮 cheap 梯度而仍引用此条件等式。
2. I=1 获得的标签可以更新未来的 critic，但必须先固定本次 \(\widetilde g\) 与校正估计。若 I=1 时先把 critic 更新为新模型再计算 cheap 梯度，I=0 时仍用旧模型，一般会产生选择偏差。
3. p 可由抽 I 前的状态或旧 critic 决定，只要条件概率正确、p 严格为正且使用对应的 \(1/p\)。估错 confidence 不破坏形式无偏性，却可能导致不可接受的尾部方差。
4. gradient clipping、Adam、自适应步长会对估计做非线性变换。这里只保证输入梯度无偏，不保证 optimizer update 是完整反传 update 的无偏版本。
5. 当前精确可计算的 gap regularizer 可一直直接加入，不必随机化；critic 只预测昂贵的 GAN 梯度部分。
6. 必须同时报告 wall-clock、真实精确校正频率与梯度残差分布。节省 backward 次数不自动意味着相同计算预算下效果更好。

这是经典随机校正/控制变量恒等式的任务化应用，不是新数学定理。它的研究价值是把 actor–critic 从“不可控地替代梯度”变为“以预测误差明确控制方差与精确反传频率”，并能与当前正确的离散 sampler 直接对照。

## 10. 终点 GAN 带来三个独立的漂移

**Reward 漂移。** discriminator 每次更新后 \(\ell_D\) 改变；value/costate 学习的是特定 D 下的 tail。可以存 terminal features 并用当前 D 重算 scalar reward，但精确 terminal latent gradient 还涉及 decode/Inception Jacobian。reward 重标注不等于更新了所有导数标签。

**Policy 漂移。** 旧 trajectory 的 Monte Carlo return 仍对应旧的未来 policy。重算 terminal reward 不能把它变成当前 policy 的 return。off-policy TD 可以利用 replay，但有自己的逼近、探索和稳定性条件。

**Occupancy 漂移。** 即使 Q 完全准确，从旧 state distribution 直接取 actor gradient，通常也只是带分布偏差的 surrogate gradient；当前终点目标的精确 DPG 用的是当前 policy 的 state occupancy。

如果 weak 仍作为环境参数更新，还有第四项：旧 transition 本身变化。将完整 weak output 记成动作可使底层环境固定，但不能自动解决前面三项。

一个可审计的训练单元是：短 outer block 内固定 D 与用于造标签的 target policy，采 fresh no-grad trajectories，混合少量精确 cotangent anchors 与较多 critic 局部更新，再用 fresh 完整轨迹检查真实 gradient 对齐和 objective。冻结时长是近似质量与数据复用之间的选择，不是理论保证。

## 11. 与前一轮“互补方向 × 状态系数”的优雅连接

固定一个冻结基场 S 与 K 个参考方向组成的矩阵 \(U(x,t)\)，令

\[
 \dot x=S(x,t)+U(x,t)b(x,t),\qquad
 J=\mathbb E\left[\ell_D(x_T)+\frac12\int b^\top Rb\,dt\right],\quad R\succ0.
\]

在无限函数类、光滑 value、无幅度约束等标准条件下，HJB 方程为

\[
 \partial_tV+\nabla V^\top S
 -\frac12\nabla V^\top U R^{-1}U^\top\nabla V=0,
 \qquad V(T,x)=\ell_D(x),
\]

且最优局部控制满足

\[
 \boxed{b^*(x,t)=-R^{-1}U(x,t)^\top\nabla V(x,t).}
\]

这说明 K 维系数只需要 K 个方向导数 \(U^\top\nabla V\)。方向字典决定能改变什么，value 方向导数决定何处值得改变；能量矩阵 R 规定代价。它给出低维 action critic 的自然对象。

这里的 quadratic running cost 是新定义的控制目标，不能与当前 minibatch gap norm anchor 等同。若没有动作惩罚且没有幅度限制，Hamiltonian 对动作线性，点态最优化通常无有限解；只展示 \(b^*\) 而省略 R/约束会掩盖这个问题。

HJB 与上述反馈公式是经典最优控制。与 [Adjoint Matching](https://arxiv.org/abs/2409.08861) 也有直接关联，但该工作的随机控制、噪声日程与正则假设不能直接移植为当前确定性 GAN 目标的保证。

对当前仓库，值得验证的命题是：少量互补方向能否让需要学习的未来敏感度降到很低维，同时保留有用的生成控制能力；以及 exact Heun cotangent 的间歇监督能否让这种 RL/actor–critic 在同等计算预算下比逐轨迹反传更有效。两者都是待验证假设，不能从以上等式直接推出图像 FID 收益。

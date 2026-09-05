# Semigroup-Consistent Guidance：从端点目标推出整条引导路径

## 1. 为什么要换一个理论起点

此前最优 IG/FMD 条件确实把 ImageNet-100 SiT-S/2 的配对 FID-5K 做到了
`36.5003`，但它依赖经验选择的前瞻距离、查询位置和增益。那些实验说明“未来弱场
变化有用”，却没有回答一个更基本的问题：我们究竟想让最终分布变成什么，以及每个
噪声层的引导场为什么应该对应这个最终目标。

本方向反过来做：

1. 先明确一个有限强度外推所代表的端点分布；
2. 对该端点分布施加与原模型完全相同的加噪 semigroup；
3. 由这个合法概率路径唯一推出每个噪声层的 score；
4. 将普通 AG/IG 与它比较，缺少的项不再是可调技巧，而是一个精确的条件矩缺口。

核心直觉只有一句：

> **先在干净端点上强调 strong-over-weak evidence，再把它加噪；不能等价为先加噪，
> 再强调已经被平均过的 evidence。幂运算与条件平均不对易。**

## 2. 明确端点目标

设弱模型和强模型在干净端点的密度分别为

\[
q_0(x),\qquad p_0(x),
\]

并定义端点似然比

\[
r(x)=\frac{p_0(x)}{q_0(x)}.
\]

普通 weak-to-strong 外推系数记为 \(\gamma\ge 0\)，同时定义

\[
\beta=1+\gamma\ge1.
\]

它最自然对应的端点 power tilt 是

\[
\boxed{
\pi_{\beta,0}(x)
\propto
p_0(x)^\beta q_0(x)^{1-\beta}
=q_0(x)r(x)^\beta .
}
\]

这个定义没有小 \(\gamma\) 假设。\(\beta=1\) 恢复强模型；\(\beta>1\) 强调强模型
相对于弱模型更相信的端点区域。这与 AutoGuidance 的 density-ratio 直觉一致，但这里
它只是端点定义，尚未规定中间噪声层该怎么走。

## 3. 唯一自然的 noisy path

令 \(K_\tau(y\mid x)\) 是方差为 \(\tau\) 的 Gaussian heat kernel。弱、强模型的
noisy marginal 为

\[
q_\tau(y)=\int K_\tau(y\mid x)q_0(x)\,dx,
\qquad
p_\tau(y)=\int K_\tau(y\mid x)p_0(x)\,dx.
\]

如果我们已经指定端点是 \(\pi_{\beta,0}\)，那么与训练噪声过程一致的路径只能是

\[
\pi_{\beta,\tau}(y)
=\int K_\tau(y\mid x)\pi_{\beta,0}(x)\,dx.
\]

把 \(p_0=q_0r\) 代入，并在弱模型后验
\(q_0(x\mid y)\) 下取条件期望，得到

\[
\boxed{
\pi_{\beta,\tau}(y)
\propto
q_\tau(y)\,
\mathbb E_q[r(X)^\beta\mid Y_\tau=y].
}
\]

因此其精确 score 为

\[
\boxed{
s^\star_{\beta,\tau}(y)
=s_{q,\tau}(y)
+\nabla_y\log\mathbb E_q[r(X)^\beta\mid Y_\tau=y].
}
\]

这条路径有一个比“当前点往哪里走”更高层的语义：每个噪声层携带的是**同一个端点
utility \(r(X)^\beta\) 在当前 noisy latent 中的后验条件期望**。

## 4. 普通 AG/IG 实际做了什么

Bayes 恒等式给出

\[
\frac{p_\tau(y)}{q_\tau(y)}
=\mathbb E_q[r(X)\mid Y_\tau=y].
\]

所以普通 score 外推

\[
\widetilde s_{\beta,\tau}
=\beta s_{p,\tau}+(1-\beta)s_{q,\tau}
\]

等价于

\[
\boxed{
\widetilde s_{\beta,\tau}
=s_{q,\tau}
+\nabla_y\log
\left(\mathbb E_q[r(X)\mid Y_\tau=y]\right)^\beta.
}
\]

它先把 endpoint ratio 条件平均，再做幂运算。合法端点路径则先对 ratio 做幂运算，
再条件平均。两者的顺序相反：

\[
\mathbb E[r^\beta\mid y]
\quad\text{vs}\quad
(\mathbb E[r\mid y])^\beta.
\]

除非后验中 \(r(X)\) 没有不确定性，或者 \(\beta=1\)，二者不会相同。

## 5. 缺失项是条件 Jensen / Rényi 势

定义

\[
\boxed{
\delta_{\beta,\tau}(y)
=\log\mathbb E[r^\beta\mid y]
-\beta\log\mathbb E[r\mid y].
}
\]

那么有精确恒等式

\[
\boxed{
s^\star_{\beta,\tau}
=\widetilde s_{\beta,\tau}+\nabla\delta_{\beta,\tau}.
}
\]

对 \(\beta>1\)，由条件 Jensen 不等式，\(\delta\ge0\)（相差一个与 \(y\) 无关的
归一化常数不影响 score）。它衡量 noisy latent 后验中 strong-over-weak evidence 的
不确定性：

- 后验几乎确定时，平均与幂近似对易，修正很小；
- 一个 noisy latent 仍对应很多 evidence 强弱不同的 clean endpoints 时，普通 guidance
  把这些 endpoint 先平均掉，修正不再可以忽略。

因此它不是“再加一点锐化”，而是恢复被条件平均丢失的**高阶证据矩**。

## 6. Tower property：一个更像 fixed point 的不变量

令

\[
h_{\beta,\tau}(y)=\mathbb E[r(X)^\beta\mid Y_\tau=y].
\]

对任意两层噪声 \(\tau_1<\tau_2\)，条件期望的 tower property 要求

\[
\boxed{
h_{\beta,\tau_2}(Y_{\tau_2})
=\mathbb E[
h_{\beta,\tau_1}(Y_{\tau_1})
\mid Y_{\tau_2}].
}
\]

这就是本理论最核心的不变量：

> **不同噪声层不是各自定义一个新的“偏好”；它们必须是同一个端点偏好在不同信息量
> latent 中的 Bayes 投影。**

普通 AG 的 \((\mathbb E[r\mid y])^\beta\) 一般不满足这个 tower consistency。
Semigroup-consistent guidance 的目标正是恢复这个跨噪声层不变量。

## 7. 修正势满足一个唯一的 Bellman/Feynman--Kac 方程

记

\[
h_1=\mathbb E[r\mid y],\quad
h_\beta=\mathbb E[r^\beta\mid y],\quad
u=\log h_1,\quad
\delta=\log h_\beta-\beta u.
\]

对 heat semigroup，条件矩满足

\[
\partial_\tau h_k
=\mathcal L_{q,\tau}h_k,
\qquad
\mathcal L_{q,\tau}
=\frac12\Delta+s_{q,\tau}^{\mathsf T}\nabla.
\]

直接代数消去 \(h_1,h_\beta\)，得到

\[
\boxed{
\partial_\tau\delta
=
(\mathcal L_q+\beta\nabla u^{\mathsf T}\nabla)\delta
+\frac12\|\nabla\delta\|^2
+\frac12\beta(\beta-1)\|\nabla u\|^2,
\qquad
\delta(y,0)=0.
}
\]

其中

\[
\nabla u=s_{p,\tau}-s_{q,\tau}
\]

正是 strong--weak score gap。令 \(k=e^\delta\)，非线性方程又化为线性
Feynman--Kac 方程：

\[
\boxed{
\partial_\tau k
=
(\mathcal L_q+\beta\nabla u^{\mathsf T}\nabla)k
+\frac12\beta(\beta-1)\|\nabla u\|^2k,
\qquad k(y,0)=1.
}
\]

这给出完整的 theory-to-design 链：普通 AG 提供一阶 ratio gradient；它的平方范数是
产生条件 Jensen 修正的 running cost；一个标量 value function 把这些局部 cost 按合法
noising dynamics 聚合起来；最终只需把该 value 的梯度加入 score。

## 8. 不增加自由参数的最低阶实现

在数据端附近做短 heat-time 展开：

\[
\delta_{\beta,\tau}(y)
=\frac12\beta(\beta-1)\tau
\|s_p(y,\tau)-s_q(y,\tau)\|^2+O(\tau^2).
\]

SiT 使用线性路径

\[
z_t=t x+(1-t)\epsilon,
\qquad t:0\to1.
\]

换到 heat 坐标 \(y=z_t/t\)，有 \(\tau=((1-t)/t)^2\)。若 strong/weak velocity
差为

\[
g_v(z,t)=S(z,t)-W(z,t),
\]

则上式化简为

\[
\delta_{\rm local}(z,t)
=\frac12\beta(\beta-1)t^2\|g_v(z,t)\|^2.
\]

score 与 velocity 的关系给出最终修正

\[
\boxed{
\Delta v_{\rm local}
=\beta(\beta-1)t(1-t)
J_{g_v}(z,t)^{\mathsf T}g_v(z,t).
}
\]

它只需要一次 VJP，不形成完整 Jacobian；系数完全由原 IG 的 \(\gamma\) 决定，因为
\(\beta=1+\gamma\)。因此首轮真实模型实验没有新增 scale、horizon、查询位置或时间表。

## 9. 已完成的解析 toy 证据

在一维 Gaussian-mixture 解析实验中，目标是端点 power tilt 的真实分布。稳定后的 v2
使用相同初始样本与相同 probability-flow 数值积分：

| 路径 | W1 | W2 |
|---|---:|---:|
| 普通 static score extrapolation | 0.037459 | 0.043640 |
| 理论最低阶 local-risk 修正 | 0.016156 | 0.019532 |
| learned soft-Bellman value | 0.002139 | 0.003977 |
| 精确 semigroup-consistent score | **0.002110** | **0.002198** |

精确 score identity 的误差约为 `1e-6`。learned soft-Bellman 已接近 exact path，说明
目标对象与跨时间聚合在可解分布中确实成立；它不证明 finite neural weak/strong gap
满足同一个 semigroup 假设。

这同时给出正、负两条预测：

1. 非对易缺口是真实且可显著影响端点分布的；
2. 只用局部项在高噪声区不够，若真实 SiT 首轮失败，正确升级不是扫一个增益，而是
   近似完整 Bellman/Feynman--Kac value。

## 10. RAEv2 finite-model 检验

RAEv2 上使用独立、均衡覆盖 1000 类的 switch bank 训练 4000 step。训练 loss 从约
`4.5e-6` 降到 `4e-8--2e-7`，但 held-out Bellman 审计显示高噪声区失配迅速增大：
`t=.80/.90/.99` 的 residual/target 分别约为 `2.49/6.59/51.34`；真实标签与
permuted-label correction-gradient cosine 约为 `0.978`。这说明 value network 虽能拟合
训练对象，其 correction 已主要退化为 class-agnostic plug-in signal。

同 seed、同 batch size 的配对 1K 正式采样为：

| condition | FID-1K | IS |
|---|---:|---:|
| piecewise IG | **39.370218** | **57.111010** |
| semigroup value | 40.667069 | 55.720657 |

两路 generator 与首批 noise/label 哈希完全一致。semigroup correction 平均 RMS 为
`0.193813`，相对 ordinary clean RMS 的均值约 `0.2881`，并非数值上消失；但 FID
恶化 `1.296851`，IS 同时下降。于是“exact-density 下的合法 semigroup correction”
没有穿过当前 finite internal-head 近似，按预注册规则停止，不扫额外 scale，也不做 5K。

## 11. 与相邻工作的边界

- **AutoGuidance** 给出 local density-ratio force，但不要求各噪声层对应同一个合法端点
  noising path。
- **Feynman--Kac Correctors** 指出几何 score mixing 的 noisy path 一般不合法，并用
  reaction weighting/SMC 纠正指定的 noisy-time mixture。这里指定的是 clean endpoint
  power tilt，再推导其唯一自然 noising path，目标对象不同。
- **Analytic Distribution of CFG for Schedule Design** 刻画 heuristic deterministic CFG
  ODE 实际得到的 path-integral 分布。这里不是描述 heuristic 最终采到了什么，而是先
  指定想要的 endpoint，再构造与之相容的场。
- 此前 **potential AutoGuidance** 的主要问题是 finite neural gap 明显 non-conservative，
  不能任意宣称存在全局 density ratio potential。本推导的严格部分针对合法 score 模型；
  SiT 内部弱头实验是对 finite-model 近似是否仍有用的实证检验，不能把假设偷换成定理。

## 12. 停止标准与实际裁决

首轮 SiT 只检验零新增自由参数的 local correction：

\[
v_{\rm SCG-local}
=S+\gamma(S-W)+\Delta v_{\rm local}.
\]

local correction 的早期检验推动了完整 scalar value approximation；完整 value 随后在
held-out Bellman 审计和配对 1K 质量上同时失败。实际裁决因此是：保留 exact-density
恒等式和解析 toy 作为理论边界，但停止把当前 finite internal-head gap 包装为该
posterior Rényi value 的可靠估计，不再继续扩参。

## 13. 参考

- Karras et al., *Guiding a Diffusion Model with a Bad Version of Itself*,
  <https://arxiv.org/abs/2406.02507>.
- Skreta et al., *Feynman--Kac Correctors in Diffusion*,
  <https://arxiv.org/abs/2503.02819>.
- *Analytic Distribution of Classifier-Free Guided Diffusion Models for
  Schedule Design*, <https://arxiv.org/abs/2607.19725>.

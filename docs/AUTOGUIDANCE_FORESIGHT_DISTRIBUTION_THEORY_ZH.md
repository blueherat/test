# AutoGuidance 的前瞻校准：分布、共轭算子与严格边界

## 1. 研究范围

本研究只讨论 AutoGuidance（AG）。NeurIPS 2025 Spotlight《Towards a
Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations》
提供的是研究方法上的启发：

1. 把 guidance 拆成 calibration 与 transport；
2. 不只查询当前时刻，而是构造未来区间上的子问题；
3. 把迭代算子、区间长度、迭代次数和计算预算分开比较。

CFG 的 conditional/unconditional 一致性目标不能搬到 AG。AG 必须从
strong/weak 模型的概率语义重新定义自己的校准对象。

## 2. 必须区分的三类分布

仓库采用 noise-to-data 的 linear flow：

\[
z_t=(1-t)\epsilon+t x,\qquad t:0\to1.
\]

以下三类分布不能混用。

### 2.1 真实 bridge marginal

由真实数据和高斯噪声定义：

\[
p_t^{\rm data}=\operatorname{Law}((1-t)\epsilon+t x).
\]

这是 flow matching 的训练分布，但生成时的状态不一定仍服从它。

### 2.2 strong/weak 的局部隐含分布

只有在模型场满足 score consistency 时，才可把 strong、weak velocity
解释为某些逐时刻密度 \(\pi_t^S,\pi_t^W\) 的 score-induced velocity。
对 linear flow，population velocity 与 score 的关系是

\[
s_t(z)=-\frac{z-tv_t(z)}{1-t},
\qquad
v_t(z)=\frac{z+(1-t)s_t(z)}{t}.
\]

因此在两个模型都满足这一关系时，

\[
S_t(z)-W_t(z)
=\frac{1-t}{t}\left(s_t^S(z)-s_t^W(z)\right)
=\frac{1-t}{t}\nabla_z
\log\frac{\pi_t^S(z)}{\pi_t^W(z)}.
\]

有限神经网络场未必对应任何合法密度，所以这是有条件的理想化，不是
对任意 checkpoint 自动成立的恒等式。

### 2.3 sampler 的实际 rollout 分布

普通 AG 使用

\[
b_t^\gamma(z)=S_t(z)+\gamma\bigl(S_t(z)-W_t(z)\bigr).
\]

令其实际状态分布为 \(q_t^\gamma\)，则

\[
\partial_tq_t^\gamma+
\nabla\cdot(q_t^\gamma b_t^\gamma)=0.
\]

即使每个固定 \(t\) 上可以形式地写出

\[
\widetilde\pi_t^\gamma
\propto
(\pi_t^S)^{1+\gamma}(\pi_t^W)^{-\gamma},
\]

也一般没有

\[
q_t^\gamma=\widetilde\pi_t^\gamma.
\]

原因是 \(\{\widetilde\pi_t^\gamma\}_t\) 通常不是同一个合法 noising path。
AutoGuidance 原文也明确指出了这个限制。局部 density-ratio score 解释不
等于终端采样分布定理。

## 3. 为什么不能直接复制 CFG 的固定点

FSG 对 CFG 寻找的是 conditional/unconditional 未来生成一致的 latent。
其语义来自外部条件 \(c\)：conditional 与 unconditional 的差值在理想
情况下对应 \(\nabla\log p(c\mid z_t)\)。

AG 中直接要求

\[
\Phi^S_{\tau\leftarrow t}(z)
=\Phi^W_{\tau\leftarrow t}(z)
\]

没有相同语义。它只要求两个近似模型在未来相等；二者可能在错误位置
相等。仓库中 strong-forward/weak-inverse 的长区间固定点实验虽然降低了
相应 discrepancy，却显著恶化 FID，已经构成直接反例。

所以 AG 的目标不应是“消灭 strong/weak 差值”，而应是利用这个差值所
代表的 density-ratio ascent。

## 4. AG-specific 的未来校准

记 strong flow map 为

\[
\Phi^S_{\tau\leftarrow t}:z_t\mapsto z_\tau,
\qquad \tau>t,
\]

并假设它在所研究区间内可逆。未来 AG gap 为

\[
g_\tau(z)=S_\tau(z)-W_\tau(z).
\]

先在未来状态定义一次校准：

\[
C_{\tau,\eta}(z)=z+\eta g_\tau(z).
\]

然后把它通过 strong flow 拉回当前坐标：

\[
\boxed{
T_{t,\tau,\eta}
=(\Phi^S_{\tau\leftarrow t})^{-1}
\circ C_{\tau,\eta}
\circ\Phi^S_{\tau\leftarrow t}.
}
\]

这不是 CFG 算子的替换符号。它表达的是一个 AG-specific 命题：当前
latent 经校准后，在 strong dynamics 下到达的未来状态，恰好是原未来
状态做过 AG density-ratio 校准的位置。

## 5. 两个精确恒等式

### 5.1 粒子层的交换图

只要 flow map 与逆映射精确，便有

\[
\boxed{
\Phi^S_{\tau\leftarrow t}(T_{t,\tau,\eta}(z))
=C_{\tau,\eta}(\Phi^S_{\tau\leftarrow t}(z)).
}
\]

这不要求 \(\eta\) 很小，也不使用 Taylor 展开。

### 5.2 分布层的 pushforward

对当前任意实际 rollout 分布 \(q_t\)，

\[
\boxed{
(\Phi^S_{\tau\leftarrow t})_\#(T_{t,\tau,\eta})_\#q_t
=(C_{\tau,\eta})_\#
(\Phi^S_{\tau\leftarrow t})_\#q_t.
}
\]

因此共轭算子不是在声称当前分布等于某个 product density；它精确控制
的是：由 strong flow 观察到的未来分布，先接受一次未来 AG 校准。

## 6. 在理想 score 假设下，校准能证明什么

定义未来 log density ratio

\[
r_\tau(z)=\log\frac{\pi_\tau^S(z)}{\pi_\tau^W(z)}.
\]

若 score consistency 成立，则

\[
g_\tau(z)=a_\tau\nabla r_\tau(z),
\qquad a_\tau=\frac{1-\tau}{\tau}>0.
\]

于是 \(C\) 是对 \(r_\tau\) 的显式梯度上升。若 \(r_\tau\) 的梯度为
\(L\)-Lipschitz，并令 \(\alpha=\eta a_\tau\)，则 smoothness inequality
给出

\[
r_\tau(z+\alpha\nabla r_\tau(z))
\ge r_\tau(z)
+\alpha\left(1-\frac{L\alpha}{2}\right)
\lVert\nabla r_\tau(z)\rVert^2.
\]

因此当

\[
0<\alpha<\frac{2}{L}
\]

时，一次未来校准不会降低该粒子的 **模型 strong/weak density ratio**。
由交换图，同一个保证也适用于拉回后的当前 latent：

\[
r_\tau(\Phi^S(T(z)))
\ge r_\tau(\Phi^S(z)).
\]

这是当前理论中最明确的正向结论，但它仍不等于：

\[
\text{model ratio 上升}\Longrightarrow\text{真实质量或 FID 改善}.
\]

后一个箭头依赖 AutoGuidance 的 compatible degradation 假设：weak 分布
应比 strong 更弥散，但二者错误结构相容，使高 ratio 区域倾向于 strong
学得较好的区域。

## 7. 固定点与迭代的严格含义

因为 \(T=\Phi^{-1}C\Phi\)，精确映射下

\[
T^K=\Phi^{-1}C^K\Phi.
\]

并且

\[
T(z^*)=z^*
\Longleftrightarrow
g_\tau(\Phi(z^*))=0.
\]

在 score-consistent 情况，这意味着未来 ratio 的驻点。固定点方程本身
不区分最大值、最小值与鞍点；稳定性才有判别力。

设 \(y^*=\Phi(z^*)\) 是 ratio 的严格局部最大点，Hessian 特征值
\(\lambda_i<0\)。梯度上升映射的 Jacobian 为

\[
J_C(y^*)=I+\alpha\nabla^2r_\tau(y^*).
\]

若所有 \(|1+\alpha\lambda_i|<1\)，该最大点局部稳定。由于

\[
J_T(z^*)=J_\Phi(z^*)^{-1}J_C(y^*)J_\Phi(z^*),
\]

二者相似并具有相同特征值，所以 fixed-point 稳定性被共轭映射继承。

当前正式实验只使用 \(K=1\)。它验证的是一次未来校准是否有用，还没有
验证多轮 fixed-point convergence。只有因果对照通过后，才应测试
\(K>1\)。

## 8. 一阶极限：strong-flow pullback metric

虽然有限 \(\eta\) 的定义应使用上面的精确共轭映射，其小步极限提供了
一个有解释力的几何视角。记 \(\Phi=\Phi^S_{\tau\leftarrow t}\)，则

\[
T(z)-z
=\eta J_\Phi(z)^{-1}g_\tau(\Phi(z))+O(\eta^2).
\]

令当前坐标上的未来 ratio objective 为

\[
R_t(z)=r_\tau(\Phi(z)).
\]

则

\[
\nabla_zR_t=J_\Phi^\top\nabla r_\tau,
\]

从而

\[
J_\Phi^{-1}\nabla r_\tau
=(J_\Phi^\top J_\Phi)^{-1}\nabla_zR_t.
\]

所以一阶下，共轭校准等价于在 strong flow 的 pullback metric

\[
G_t=J_\Phi^\top J_\Phi
\]

中，对未来 density ratio 做 natural-gradient ascent。它不会把未来向量
粗暴地当成当前向量，而是通过 strong dynamics 的坐标变换拉回。

这段 natural-gradient 解释只是一阶结论；当前较大的 \(\rho\) 实验必须
以精确共轭映射而非 Taylor 近似解释。

### 8.1 raw future direction 的局部理论

因果实验表明，经验上更好的算子反而是

\[
F^{\rm raw}_{t,\tau}(z)
=z+\eta g_\tau(\Phi(z)),
\]

即用 canonical latent coordinates 直接识别不同时间的向量，不做 inverse
pullback。它没有共轭算子的坐标不变性，也没有无条件的 ratio-ascent
保证，但并非完全无法分析。

仍令

\[
R_t(z)=r_\tau(\Phi(z)),\qquad
g_\tau=a_\tau\nabla r_\tau,qquad J=J_\Phi(z).
\]

沿 raw future direction 的一阶变化是

\[
DR_t(z)[\eta g_\tau]
=\frac{\eta}{a_\tau}g_\tau^\top Jg_\tau
=\frac{\eta}{a_\tau}g_\tau^\top
\operatorname{Sym}(J)g_\tau.
\]

因此只要 strong flow Jacobian 的对称部分在该 gap 方向上为正，raw update
仍是 future ratio 的上升方向。短区间有

\[
J=I+(\tau-t)J_S+O((\tau-t)^2),
\]

所以当区间足够短、strong dynamics 没有在该方向上产生强折叠时，该条件
自然可能成立。这是局部充分条件，不是全局定理。

对比三种未来方向：

\[
\begin{aligned}
\delta_{\rm raw}&=g_\tau,\\
\delta_{\rm conjugate}&=J^{-1}g_\tau,\\
\delta_{\rm adjoint}&=J^\top g_\tau.
\end{aligned}
\]

其中 conjugate 是 pullback-metric natural gradient；adjoint 则与当前
Euclidean objective 的真正梯度同向：

\[
\nabla_zR_t=\frac{1}{a_\tau}J^\top g_\tau.
\]

在固定当前扰动范数时，归一化的 \(J^\top g_\tau\) 最大化 future ratio 的
一阶增长。它是后续可能测试的理论候选，但需要反向传播或 adjoint solve，
成本明显高于 raw future。

当前结果中 raw 优于 conjugate 说明一件重要的事：即使 conjugate 对未来
ratio 的局部上升保证更干净，也不代表它更能改善真实 FID。模型 ratio、
真实质量以及选用的 current-space metric 仍是三个不同对象。

### 8.2 future gap 的精确 Lagrangian 分解

前瞻方向并不是任意取另一个时刻的 gap。定义

\[
h_{t\to\tau}(z)
=g_\tau\!\left(\Phi^S_{\tau\leftarrow t}(z)\right),
\qquad g_s=S_s-W_s.
\]

沿 strong characteristic

\[
y_s=\Phi^S_{s\leftarrow t}(z),
\qquad \dot y_s=S_s(y_s),
\]

链式法则精确给出

\[
\frac{d}{ds}g_s(y_s)
=\left(\partial_s+S_s\cdot\nabla\right)g_s(y_s).
\]

因此

\[
\boxed{
h_{t\to\tau}(z)
=g_t(z)+
\int_t^\tau
\left(\partial_s+S_s\cdot\nabla\right)g_s
\left(\Phi^S_{s\leftarrow t}(z)\right)ds.
}
\]

这是有限区间上的精确恒等式，不要求 \(\tau-t\) 很小。它把 FSG 的
“未来子问题”直觉转换成了 AG 自己的对象：

- 当前 AG 只使用 \(g_t\)；
- future AG 额外积累 strong/weak discrepancy 沿 strong flow 的
  material derivative；
- 因而只调当前 gamma 只能改变 \(g_t\) 的长度，不能恢复积分项提供的
  新方向。

在理想 score-consistent 情况，\(g_s=a_s\nabla r_s\)，所以该积分项同时
包含 ratio 随时间的变化、ratio Hessian 与 strong transport 的耦合。它
不是“未来更清晰”这一口头直觉，而是 discrepancy 为什么会在未来旋转
和重组的具体微分结构。

raw future calibration 对当前分布的作用也可以不借助小步展开来定义。
令

\[
F_{t,\tau,\eta}(z)=z+\eta h_{t\to\tau}(z),
\qquad q_t^+=(F_{t,\tau,\eta})_\#q_t.
\]

若 \(F\) 是微分同胚，则精确的换元公式为

\[
q_t^+(F(z))
=\frac{q_t(z)}
{\left|\det\left(I+\eta J_h(z)\right)\right|},
\qquad
J_h=J_{g_\tau}(\Phi(z))J_\Phi(z).
\]

所以有限强度更新改变分布的方式不仅取决于 gap 指向哪里，还取决于
它的空间 Jacobian。校准后的分布再被后续 guided dynamics 推到终点；
这一完整 pushforward 才决定 FID。由此不能从 pointwise ratio ascent
直接推出终端质量改善。

这也给 horizon tradeoff 一个可检验的结构：增大 \(\tau-t\) 会积累更多
material-derivative 信息，但同时会增大 \(J_\Phi\) 带来的坐标畸变，并使
未来方向对当前状态变得陈旧。因此理论并不预言“越远越好”，而是预言
存在语义成熟与 transport mismatch 之间的竞争。

## 9. 为什么前瞻可能有用

噪声端附近，strong 与 weak 都接近高斯，模型 ratio 可能近似平坦，当前
gap 的可辨识语义有限。经过一段 strong denoising 后：

1. 类别与结构开始显现；
2. compatible weak degradation 造成的扩散程度差异更易观测；
3. future ratio gradient 可能比当前 gap 更能定位 under-fit 区域。

但未来向量位于未来状态的切空间，原则上存在坐标识别问题。共轭拉回是
坐标不变的解法；canonical latent 中直接复用未来方向则是更便宜、但依赖
参数化的解法。二者谁更符合真实质量目标必须由因果对照决定，不能只凭
几何形式判定。

## 10. 必须完成的因果对照

现有 40-step Euler AG 记为 baseline。仅在 step 0 与 step 5 做额外校准。
正式扫描包含四种方法：

### 10.1 当前 gamma 调度

\[
z^+=z+h\left[S_t(z)+\gamma\rho(S_t(z)-W_t(z))\right].
\]

它没有未来信息、没有状态校准、没有 flow pullback。与 40-step AG 调用
数完全相同。

### 10.2 零前瞻校准后 strong response

\[
y=z+h\gamma\rho(S_t(z)-W_t(z)),
\qquad z^+=y+hS_t(y).
\]

它是共轭算子的 \(\tau\to t\) 离散控制，隔离“校准后让 strong model
重新响应”本身。调用数匹配 41-step AG。

### 10.3 未来 gap 直接放回当前坐标

先算 \(y_\tau=\Phi^S_{\tau\leftarrow t}(z)\)，再使用

\[
z_{\rm raw}=z+\eta g_\tau(y_\tau).
\]

它拥有未来信息但没有 inverse pullback，是故意构造的坐标不变性反例。
调用数匹配 61-step AG。

### 10.4 strong-flow 共轭未来校准

\[
z_{\rm conj}=\Phi^{-1}
\left(\Phi(z)+\eta g_\tau(\Phi(z))\right).
\]

它同时具有未来信息和正确的 strong-flow 拉回。调用数匹配 81-step AG。

判别逻辑：

- 当前 gamma 调度若复现收益，前瞻解释不成立；
- 零前瞻校准若复现收益，主要机制是 strong state response；
- raw future 若复现收益，未来信息重要但 pullback geometry 不重要；
- 只有共轭方法稳定胜出，才能支持 future-ratio + strong-flow geometry。

## 11. 当前已完成的数值验证

1. RK4 strong forward/inverse round-trip RMS 约为状态 RMS 的
   `0.009%--0.011%`。
2. 共轭交换图残差约为 intended future calibration RMS 的
   `1.9%--2.5%`；Euler inverse 的误差反而大于校准量，因此已弃用。
3. `scheduled_ag` 在 \(\rho=1\) 时与普通 AG 逐像素完全一致，NPZ
   SHA256 完全相同。
4. 四类方法的模型调用数已分别与 40/41/61/81-step 普通 AG 精确匹配。
5. 单元测试 `17 passed`，端到端 smoke 的 peak CUDA allocation 约
   `1.05 GiB`。

## 12. FID-1K 因果结果

ImageNet-100 SiT-S/2，strong=`v800`，weak=`v500`，EMA，相同 noise、label
与 seed：

### 12.1 共轭前瞻的初步信号

| 条件 | FID | sFID | IS |
|---|---:|---:|---:|
| 普通 AG，\(\gamma=4\)，81 steps | 74.6439 | 220.0474 | 32.9507 |
| 共轭前瞻，\(\rho=1\) | 73.6055 | 219.4505 | 34.4922 |
| 共轭前瞻，\(\rho=2\) | 73.2901 | 218.5866 | 34.7201 |
| 共轭前瞻，\(\rho=4\) | **73.1083** | **216.9331** | 34.2560 |

### 12.2 时间调度、零前瞻与 raw future

每类方法都和模型调用数相同的普通 AG 比较：

| 方法 | 同预算 baseline FID | 最好 FID | 差值 |
|---|---:|---:|---:|
| 只调当前 AG gamma | 74.0992 | 74.0990，\(\rho=1\) | 约 0 |
| 零前瞻校准 + strong response | 74.0782 | 74.1204，\(\rho=1\) | +0.0422 |
| raw future gap | 74.3359 | **72.5348**，\(\rho=4\) | **-1.8011** |
| strong-flow 共轭 | 74.6439 | 73.1083，\(\rho=4\) | -1.5356 |

所以收益不能由“早期把普通 AG 加强”或“校准后让 strong 重新响应”解释。
未来 gap 的确提供了新信息。但更重要的是，raw future 不仅优于自己的
同预算 baseline，也优于计算更多的 inverse-flow 共轭方法。当前证据不
支持“inverse pullback geometry 是收益来源”。

### 12.3 逐样本等范数的方向/幅值对照

在正式样本首批 16 张上：

| 事件 | current/future gap cosine | future/current RMS |
|---|---:|---:|
| \(t=0\to\tau=0.125\) | 0.0630 | 1.9071 |
| \(t=0.125\to\tau=0.25\) | 0.4068 | 1.5558 |

未来 gap 更长，但也明显改变了方向。为严格分离二者，做两种逐样本、逐
事件等 RMS 的对照，并匹配 63-step AG 的实际模型调用：

| 条件 | FID | sFID | IS |
|---|---:|---:|---:|
| 普通 AG，63 steps | 74.4785 | 219.9801 | 33.9972 |
| 未来方向，缩放到当前 gap RMS，\(\rho=1\) | 73.6491 | 219.6315 | 34.5273 |
| 未来方向，缩放到当前 gap RMS，\(\rho=2\) | 73.3973 | 218.7874 | 34.2210 |
| 未来方向，缩放到当前 gap RMS，\(\rho=4\) | **72.8876** | **217.1403** | **35.3846** |
| 当前方向，放大到未来 gap RMS，\(\rho=1\) | 75.0979 | 220.2565 | 34.2716 |
| 当前方向，放大到未来 gap RMS，\(\rho=2\) | 74.9860 | 219.1762 | 35.0319 |
| 当前方向，放大到未来 gap RMS，\(\rho=4\) | 75.4001 | 217.2546 | 33.1898 |

保留未来方向、去掉其额外幅值后，FID 仍改善 `1.5909`，且三项指标同向。
保留当前方向、只复制未来幅值则恶化 FID。FID 不是线性量，不能说保留
了某个精确百分比的“机制”，但这个对照已经明确否定纯幅值解释。

### 12.4 第二个采样 seed

换用 `global_seed=1` 后复验关键条件：

| 方法 | 同 seed baseline | 方法 FID | 改善 |
|---|---:|---:|---:|
| raw future，matched 61 | 73.2225 | **70.3966** | **2.8259** |
| future direction / current RMS，matched 63 | 73.1042 | **71.9769** | **1.1273** |
| current direction / future RMS，matched 63 | 73.1042 | 72.4128 | 0.6914 |
| conjugate future，matched 81 | 72.3552 | 72.1216 | 0.2336 |

第二个 seed 中，放大的 current direction 偶尔也改善，因此不能说它在每
个 sample set 上必然有害。但两次 seed 中 future direction 都改善、都优于
等范数的 current direction；raw future 的收益也都明显大于 conjugate。

两次 seed 相对各自同预算 baseline 的 FID 改变量为：

| 方法 | seed 0 | seed 1 | 平均 |
|---|---:|---:|---:|
| raw future | -1.8011 | -2.8259 | **-2.3135** |
| future direction / current RMS | -1.5909 | -1.1273 | **-1.3591** |
| current direction / future RMS | +0.9216 | -0.6914 | +0.1151 |
| conjugate future | -1.5356 | -0.2336 | -0.8846 |

这仍是同一个训练 run 上的两个采样 seed，并非两个训练 seed；它验证的是
采样/评估稳定性，而不是模型训练稳定性。

### 12.5 两轮 future-residual iteration

定义 raw future operator

\[
F(z)=z+\eta g_\tau(\Phi^S(z)).
\]

比较相近的总校准强度：一次 \(\rho=4\) 与两次 \(\rho=2\)。seed 0 中：

| 条件 | FID | sFID | IS |
|---|---:|---:|---:|
| 普通 AG，matched 83 | 74.6637 | 220.0414 | 32.8629 |
| \(K=2,\rho=1\) | 73.1700 | 217.7725 | 33.9869 |
| \(K=2,\rho=2\) | **71.7913** | 214.9936 | 33.5651 |
| \(K=2,\rho=4\) | 72.8802 | **210.8766** | **34.2567** |

seed 0 中，\(K=2,\rho=2\) 比 \(K=1,\rho=4\) 的 72.5348 进一步改善
0.7435 FID；但 seed 1 中，\(K=2,\rho=2\) 为 71.1077，虽优于 matched-83
baseline 72.4024，却不如同 seed 的 \(K=1,\rho=4\)（70.3966）。

因此“future residual iteration 有益”相对 baseline 成立，但“多轮小步一定
优于单轮大步”尚不稳定。诊断还显示第二轮 move 与第一轮 cosine 为
`0.94--0.99`、RMS 比接近 1；当前并没有观察到明显 contraction。把它称为
已收敛的 fixed-point solver 是不准确的，它更接近在 calibration pseudo-time
上对 future residual dynamics 做两步显式积分。

### 12.6 lookahead horizon

便携结果见 `docs/data/autoguidance_foresight_horizon_fid1k.csv`。

为区分“只要查询未来就行”与“未来区间长度本身有结构”，固定
\(\rho=4\)，只在前两个事件使用 future direction，并将每个未来 gap
逐样本缩放到当前 gap 的相同 RMS。对 \(H=1,2,5,10\)，分别使用
47/51/63/83-step 普通 AG 匹配真实模型 forward 次数。

相对各自同 seed、同预算 baseline 的结果为：

| horizon | seed 0 \(\Delta\)FID | seed 1 \(\Delta\)FID | 平均 |
|---:|---:|---:|---:|
| 1 | -0.3857 | -1.3185 | -0.8521 |
| 2 | -1.0444 | -1.2196 | -1.1320 |
| 5 | **-1.5909** | -1.1273 | **-1.3591** |
| 10 | -1.1647 | +0.2742 | -0.4453 |

sFID 的两 seed 平均改善分别为 `2.53/3.00/2.97/1.60`。因此可以排除
“收益只是多做 forward”：每个条件都与自己的 compute baseline 比较；也
可以排除“越远越好”：\(H=10\) 的 FID 收益不稳定，且平均 sFID 收益
明显回落。

这与第 8.2 节的精确分解一致。非零 horizon 通过 material derivative
引入当前 gamma scaling 无法产生的新方向；但 horizon 继续增大后，
coordinate distortion、state staleness 与有限模型误差也同步累积。当前
数据支持“短到中等前瞻有用、过长前瞻不稳”，但不支持把 \(H=5\) 宣布
为普适最优值。

这些仍是配对 FID-1K 机制筛查，不是正式 benchmark。

## 13. 当前最严格的结论边界

目前可以证明或已验证的是：

1. 共轭算子精确实现“未来校准后拉回当前”的交换图；
2. 在 score-consistent 与 smoothness 假设下，它提高未来 strong/weak
   模型 density ratio；
3. 共轭版本的固定点与稳定性由未来 ratio ascent 映射继承；
4. future gap 的方向而非单纯幅值带来稳定的 FID-1K 正信号；
5. 数值逆映射误差远小于校准量，但 inverse pullback 并非经验收益所必需。
6. 短到中等 lookahead 在两个采样 seed 上均有正信号，而更长 horizon 的
   FID 收益不稳定。

目前不能声称的是：

1. finite neural velocity gap 必然是 conservative density-ratio gradient；
2. 模型 ratio 上升必然改善真实数据似然、FID 或人类质量；
3. future gap 为什么与真实质量方向对齐，以及该现象是否跨训练 seed、
   checkpoint 和模型成立；
4. \(K>1\) 固定点迭代优于单次校准；
5. raw future 更新在一般坐标变换下有不变性或全局 ratio-ascent 保证。

当前最强结论是“沿 strong characteristic 查询到的 future direction 有效”，
而不是“共轭固定点有效”。两个采样 seed 已完成，但它们共享同一训练
checkpoint；下一步若继续，应优先验证训练 seed/checkpoint 稳定性，并
研究 future-direction operator 的局部稳定性，不应直接把 \(K>1\) 当作
必然升级。

## 14. 参考

- Wang et al., [Towards a Golden Classifier-Free Guidance Path via Foresight
  Fixed Point Iterations](https://arxiv.org/abs/2510.21512), NeurIPS 2025
  Spotlight.
- Karras et al., [Guiding a Diffusion Model with a Bad Version of
  Itself](https://arxiv.org/abs/2406.02507), AutoGuidance.

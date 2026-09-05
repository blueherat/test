# PFR 的反事实参考残差理论：从有限弱响应到终点风险

日期：2026-09-03

状态：**新主线；精确代数与一阶终点变分成立，质量符号由实验校准，不宣称无条件 FID 定理。**

对应实现与数据：

- PFR 主代数：`experiments/internal_guidance_path_extrapolation.py`；
  历史主 runner：`experiments/run_imagenet100_sit_path_extrapolated_ig.py`；
  query-control 实现：`experiments/run_imagenet100_sit_pfr_query_controls.py`；
- 本文可执行恒等式与严格反例：
  `experiments/pfr_counterfactual_residual_theory.py`；
- 终点 observable 审计：`experiments/analyze_pfr_terminal_distribution.py`；
- 紧凑新数据：`docs/data/pfr_counterfactual_residual_theory_20260903/`；
- 机制总审计：`docs/PFR_MECHANISM_AUDIT_20260903_ZH.md`。

---

## 一、结论先行

PFR 最有解释力的故事不是“用未来查询做一次更准确的数值积分”，而是：

> **ordinary IG 用当前弱头作 reference；PFR 保留当前弱头作 transport anchor，
> 却把 calibration 中的弱 reference 单独移到更后的 nominal denoising-time coordinate。
> 主效应在固定状态时已经存在；projected PFR 再用 strong–weak calibration proposal
> 对查询状态做一个受约束的空间条件化。PFR 不把未来弱输出当 teacher，而是代数地
> 减去它的 finite response。**

一句更短的话是：

> **一个更差的预测器，仍可以是一个更有用的反事实参考。**

它解释了仓库里最反常、也最有价值的事实：

- query 后的弱头对真实 conserved velocity 的 MSE **更差**，PFR 却更好；
- 沿 ordinary field 的局部 characteristic tangent 查询时，收益反而消失；
- seed0 1K 中 time-only 拿走主要收益，空间 probe 再带来较小增量；
- 提高 solver 阶数、隐式迭代次数或 NFE 都不能复制 PFR；
- 相邻的 characteristic-reference Picard 实验表明，更高的 reference
  self-consistency 不会普遍带来更好 FID。

数学上，这个故事由三层组成：

1. **transport-code 层**：在线性 flow matching 中，跨时间比较的是同一个守恒目标
   \(U=X-E\)，因此 raw velocity 的跨时间差有合法语义；
2. **field 层**：PFR 是一个 finite counterfactual residual，而不是积分器；其时间部分是
   coboundary，空间查询在一阶上添加 cross-Jacobian response；
3. **terminal 层**：场的好坏必须由终点风险的 adjoint 加权，而不是由局部 velocity MSE
   排序。本文给出一个严格、与部署时间 schedule 和 horizon clamp 匹配的 PFR
   反例：time-integrated local MSE 从 1 变为约 4.03，但终点 Gaussian
   Fréchet/Wasserstein 误差从 1 变成 0。

真实 ImageNet-100 的新终点审计又给出一个几乎“正交”的对应关系：

- time-only 相对 ordinary IG 的 FID-1K 改善中，**90.92% 来自 feature mean**；
- projected 相对 time-only 的空间增量中，**90.18% 来自 covariance**。

这给出一个很强的**经验分工假说**，但尚不是由定理推出的因果箭头：

\[
\boxed{\text{temporal query gain}\ \overset{\rm observed}{\longrightarrow}\ \text{mostly mean}}
\qquad+
\qquad
\boxed{\text{conditional spatial increment}\ \overset{\rm observed}{\longrightarrow}\ \text{mostly covariance}}.
\]

下文的 adjoint 分解说明该如何检验这两个箭头，但当前尚未直接测量分项
adjoint action。因此这不是无条件质量定理，而是给出了此前缺失的

\[
\delta G\to\delta\Phi_1\to\delta J_{\rm terminal}
\]

桥梁、正确的可测量量，以及为什么旧理论会做错预测。

---

## 二、整个仓库的研究轨迹实际上在反复说同一件事

从七月到 PFR，仓库里最稳定的科学事实并不是“局部预测越准，生成越好”，而是它的
反面。

| 阶段 | 当时的直觉 | 最终留下的事实 |
|---|---|---|
| EQ-VAE / RAE adapter | 先修局部表示或重建几何 | noisy reconstruction L1 改善，FID 仍恶化；局部代理不决定 pushforward |
| 频谱加权 / teacher rollout | 降低 raw、高频或 decoder proxy MSE | 多个局部指标改善，终点 FID 仍可反号 |
| dual-target closed-loop toy | 追逐 local Bayes oracle | endpoint-safe gate 的局部 Bayes MSE 更差，但终点 SWD 更好 |
| ImageNet-100 internal guidance | 更深头应当是更准 teacher | 任意 weak head 不行；有用的是结构化 strong–weak 对照 |
| foresight / FMD / FWR | 看未来也许是在修积分 | 收紧 solver 不好；有限弱响应有效；更短的导数极限反而变差 |
| fixed point / semigroup / score telescope | 更自洽、更保守或更像 density ratio 应更好 | 数学 toy 可成立，真实 FID 全部失败 |
| PFR | 选一个最小 forward-ray query | time 主导、space 增量；posterior MSE 更差；Euler-tangent query 失败 |

关键文档链是：

- `docs/RESEARCH_STATUS.md`；
- `docs/DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md`；
- `docs/IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md`；
- `docs/IMAGENET100_SIT_FORESIGHT_MATERIAL_IG_RESULTS_ZH.md`；
- `docs/PROJECTED_FUTURE_REFERENCE_IG_THEORY_ZH.md`；
- `docs/INFORMATION_TIME_POSTERIOR_REVISION_IG_ZH.md`；
- `docs/CROSS_TIME_VELOCITY_GEOMETRY_ZH.md`；
- `docs/AFFINE_COUNTERFACTUAL_RATIO_IG_THEORY_ZH.md`；
- `docs/PFR_MECHANISM_AUDIT_20260903_ZH.md`。

因此这次不再为“更准的局部估计”寻找另一层包装。真正需要理论化的是：

> **为什么一个有结构的局部 bias 可以让终点分布更好。**

这也把项目从“数值积分技巧”抬到一个更普遍的问题：生成 ODE 的 vector field 并非按
pointwise regression risk 排序，而是按它经 flow Jacobian 传播后对 terminal observable
做了什么来排序。

---

## 三、PFR 的精确重写：future output 是 negative reference，不是 future teacher

记当前点 \(p=(z,t)\)，强头和弱头分别为

\[
S=S(p),\qquad W=W(p),
\]

普通 internal guidance 为

\[
C=\beta(S-W),\qquad B=W+C=W+\beta(S-W),\qquad \beta=1+\gamma.
\]

令 \(P\) 是 calibration \(C\) 到 sampler forward ray 的投影：

\[
\alpha^\star=\left[\frac{\langle C,B\rangle}{\|B\|^2}\right]_+,
\qquad P=\alpha^\star B,
\]

当 \(B=0\) 时按实现约定 \(\alpha^\star=P=0\)。

记 correction 结束时刻为 \(\tau=0.5\)。部署实现在边界处截断名义 horizon \(h\)：

\[
\delta_h(t)=\min\{h,(\tau-t)_+\},\qquad
q_h=(z+\delta_h(t)P,t+\delta_h(t)).
\]

PFR 为

\[
F_h(p)=W(p)+\beta\,[S(p)-W(q_h)].
\]

最关键的重写是

\[
\boxed{
F_h
=B-\beta\,[W(q_h)-W(p)]
=W(p)+\beta\underbrace{[S(p)-W(q_h)]}_{\text{counterfactual residual}}.
}
\]

这句话里的 \(W(q_h)\) 不承担“更接近 \(U\)”的责任。它承担的是另一种责任：

1. \(S(p)\) 给出当前深层证据；
2. time-only 中 \(q_h=(z,t+\delta_h)\)，只推进 nominal denoising-time coordinate；
   full PFR 才另外加入 calibration-conditioned state displacement；
3. \(W(q_h)\) 给出浅层系统在该 off-channel query 上的有限响应；
4. \(S(p)-W(q_h)\) 代数地减去该响应；它不是已证明正交的 regression residual；
5. 当前 \(W(p)\) 仍作为 transport anchor，故不会像 difference-in-differences 那样把
   shared weak drift 一起删除。

这里的“残差化”是**精确代数残差**，不是已经证明正交的统计回归残差。经验上
\(\lambda\) sweep 在测试点 \(\lambda=1\) 最好，支持单位系数的 calibration；但该 sweep
的历史 `seed=0/1` 实际高度重叠，不能写成独立双 seed 复现，见第九节。

### 3.1 为什么可以跨时间比较 raw velocity

在线性 interpolant

\[
Z_t=tX+(1-t)E
\]

中，每条 conditional path 的速度都是同一个常量

\[
U=\dot Z_t=X-E.
\]

Bayes flow field 是 \(v^\star(z,t)=\mathbb E[U\mid Z_t=z]\)。不同深度、不同时间的
velocity head 都是在对同一个 supervised transport code 做条件估计；这使跨时间 velocity 对照有
语义基础，但**不意味着 off-path 的 \(W(q_h)\) 是更准 posterior**。

相同事实也可由 endpoint 参数化验证：

\[
\widehat X(z,t)=z+(1-t)W(z,t),\qquad
\widehat E(z,t)=z-tW(z,t).
\]

对任意 \(p,q\)，clean revision 与 negative-noise revision 的特定和满足

\[
R_X(p,q)+R_{-E}(p,q)=W(p)-W(q),
\]

且这是消掉人为坐标位移的唯一线性组合。clean 或 negative-noise endpoint
revision 单独使用时均比完整组合差（FID-1K `63.649/64.858` vs `61.909`）。
另一组 score-parameterization 分解中，only-\(P\)/only-\(E\) 更分别崩到
`170.624/271.154`，而精确重组恢复到 `61.969`。两组证据都说明应保留的是
raw transport-code difference，不是某个坐标分量。

### 3.2 它也是一次 inverse-response precompensation，但不是 fixed-point 质量定理

把 weak 对拟议 correction \(C\) 的有限自响应记为

\[
E_h(C)=W(z+\delta_hP(C),t+\delta_h)-W(z,t),
\qquad P(C)=\Pi_{\operatorname{ray}(W+C)}C.
\]

若希望“施加 correction 后再计入 weak 自响应”仍恢复名义 correction \(C_0\)，需要解

\[
F(C)=C+\beta E_h(C)-C_0=0.
\]

从 \(C_0\) 做一次 fixed-point update，正好得到

\[
\boxed{C_1=C_0-\beta E_h(C_0),}
\]

也就是 PFR。严格地说，若存在闭的完备集 \(\mathcal D\)，且
\(T(C)=C_0-\beta E_h(C)\) 将 \(\mathcal D\) 映回自身，\(E_h\) 在其上为
\(L\)-Lipschitz，\(\beta L<1\)，则 Banach 定理给出唯一 fixed point，且

\[
\|F(C_1)\|\le \beta L\,\|F(C_0)\|.
\]

这里已把 projection active-set kink 与 \(W+C=0\) 附近的不稳定性吸收进
Lipschitz/自映射假设。该命题只解释为什么单位系数的 “subtract the anticipated
response” 有一个 inverse-response 解释；它**不推出 FID 下降**，目前也没有实验
直接迭代这个 projected 映射。仓库 K2/K3 来自不同的 characteristic-reference
Picard 映射；它们只是相邻反例：更高的 reference self-consistency 不普遍意味更好 FID。

---

## 四、第一条新命题：时间差是 coboundary，不是 quadrature correction

先看 time-only，即 \(P=0\)。在距 \(\tau\) 远于 \(h\) 的 active 区间，修正项是

\[
r_h(t)=-\beta\,[W(z,t+h)-W(z,t)].
\]

若暂时冻结空间依赖并固定同一 horizon，则在任意时间窗 \([0,L]\) 上有精确恒等式

\[
\int_0^L[W(t+h)-W(t)]\,dt
=\int_L^{L+h}W(s)\,ds-\int_0^hW(s)\,ds.
\]

也就是中间所有项消掉，只留下两个边界带。离散情形更直观：

\[
\sum_{k=0}^{N-1}(W_{k+1}-W_k)=W_N-W_0.
\]

这就是 coboundary。canonical clamp 把最后一个 \(h\)-宽边界层改成
\(\delta_h(t)=\tau-t\)；分段 \(\beta(t)\) 还会产生 schedule-jump contrast。它们不破坏
“这是修改 vector field 而不是修改 quadrature”的结论，但意味着真实累积不能简化为
无权的两个端点。准确地说：

- PFR 的有限时间差在改变 active window 的**入口—出口平衡**；
- 它不是在近似同一个 ODE 的高阶积分；
- 即便外层 ODE 被精确求解，这个 modified field 仍然存在；
- solver step 与 query horizon \(h\) 是两个不同对象，把二者都叫“步长”会制造错误直觉。

更完整地，令 \(K_h^P f=f(z+hP,t+h)\)，令 \(U_h^B\) 是 ordinary field \(B\) 的精确
flow pullback。则有纯代数恒等式

\[
\boxed{
(I-K_h^P)W=(I-U_h^B)W+(U_h^B-K_h^P)W.
}
\]

第一项是 on-policy Koopman coboundary；第二项是 off-policy semigroup defect。小 \(h\)
时

\[
(I-K_h^P)W
=-hD_BW+hJ_W(B-P)+O(h^2).
\]

baseline-tangent query \(P=B\) 会把一阶 defect 删掉；PFR 则保留一个受约束的 defect。
这不预言 FID 符号，但它给出一个可证伪的区分：如果收益来自“更准确的未来”，
删除一阶 semigroup defect 应该有利；实验恰好不支持它。

必须区分两个说法：coboundary 沿它自己的 \(K_h^P\) 反事实轨道会精确 telescope；真实
closed-loop rollout 一般不是这条轨道，所以不能声称完整 PFR 在真实采样中“只改边界”。
真实 rollout 的内部 defect/非交换项正是下一节保留的部分。

仓库已经给出三类与此一致的反证：

1. depth4 IG 将 Dopri5 NFE 从 `8572` 收紧到 `14740/20986`，FID 从 `64.8509`
   变为 `65.1548/65.2041`；
2. 约 64 NFE 下，Euler、Heun、implicit midpoint、implicit trapezoid 都没有得到 PFR
   的收益；最好的 matched-NFE midpoint 仍为 `64.9978`，而 PFR 约为 `61.86`；
   紧凑表见 `docs/data/pfr_counterfactual_residual_theory_20260903/solver_equal_nfe64.csv`；
3. 沿 ordinary field 的局部 characteristic tangent 做 Euler query 的两个 FID-1K 是 `64.8948/64.9809`，
   恰恰比同 bank canonical projected PFR 的 `61.9244/61.8369` 差；这两个历史
   nominal seed 高度重叠，因此这里依赖的是各 bank 内的 paired 对照，不是独立重复。

这三组结果不是逻辑上穷尽所有 numerical-correction 模型，但它们是很强的
disconfirming evidence：单纯提高积分精度不会复制 PFR。

---

## 五、第二条新命题：终点一阶效应的 boundary–cross-transport 分解

上面的冻结空间版本还不够。现在保留完整状态依赖，并把 PFR 看成 baseline field
\(B=W+C\) 的有限扰动：

\[
r_h(z,t)=\beta\,[W(z,t)-W(z+\delta_hP,t+\delta_h)],
\qquad t<\tau.
\]

### 命题 1：有限查询展开

若 \(W\) 二阶可微，则在 active window 内部 \(\delta_h=h\) 处，\(h\to0\) 时

\[
\frac{r_h}{h}
=-\beta(\partial_tW+J_WP)+O(h).
\]

注意方向是 query direction \((P,1)\)，并不是 baseline characteristic \((B,1)\)。
最后一个 \(h\)-宽 boundary layer 使用 \(\delta_h=\tau-t\)；对下面按 \(h\) 归一化的
时间积分，它只改变 \(O(h)\) 余项。

### 命题 2：terminal-adjoint 分解

令最终生成时刻为 \(T=1\)，correction 支持结束于 \(\tau=0.5\)。完整
baseline field 记为 \(H\)：active window 内它是分段的 \(B_j=W+C_j\)，之后是
late strong field。baseline trajectory 满足

\[
\dot z_t=H(z_t,t).
\]

令 \(J=\ell(z_T)\) 是任意可微终点风险。对 perturbed flow

\[
\dot z_t^\varepsilon=H(z_t^\varepsilon,t)
+\varepsilon\mathbf 1_{t<\tau}r_h(z_t^\varepsilon,t)
\]

定义 baseline adjoint

\[
-\dot a_t=J_H(z_t,t)^\top a_t,
\qquad a_T=\nabla\ell(z_T).
\]

令 \(0=\tau_0<\tau_1=0.25<\tau_2=\tau\)，在第 \(j\) 段 \(\beta_j\) 为常数。
假设每段场足够光滑、\(P_j\) 有界，且 baseline flow 与 adjoint 存在，则

\[
\boxed{
\frac1h\frac{dJ}{d\varepsilon}\bigg|_{\varepsilon=0}
=\sum_{j=0}^{1}\beta_j\left\{
-\big[a^\top W\big]_{\tau_j}^{\tau_{j+1}}
+\int_{\tau_j}^{\tau_{j+1}}a^\top
\left[J_W(C_j-P_j)-J_{C_j}W\right]dt
\right\}+O(h).
}
\]

这里 \(a_\tau\) 不是在 \(\tau\) 定义的局部目标，而是从真正终点 \(T\) 穿过后半段
strong flow 回传得到的敏感度。若把 \(\beta\) 写成有界变差函数，等价的首两项为

\[
-\big[\beta a^\top W\big]_0^\tau
+\int_{[0,\tau]}a^\top W\,d\beta,
\]

其中 Stieltjes 积分显式计入 schedule jump；在平滑情形下它就是
\(\int_0^\tau\beta'a^\top Wdt\)。随机初值时对整个右侧取期望。

### 证明

标准 first variation 给出

\[
\frac{dJ}{d\varepsilon}\bigg|_0=\int_0^\tau a^\top r_h\,dt.
\]

代入有限查询展开：

\[
\frac1hJ'(0)
=-\sum_j\int_{\tau_j}^{\tau_{j+1}}
\beta_j a^\top(\partial_tW+J_WP_j)\,dt+O(h).
\]

沿 baseline trajectory，

\[
\frac d{dt}(a^\top W)
=-a^\top J_{B_j}W+a^\top(\partial_tW+J_WB_j).
\]

在 active 第 \(j\) 段有 \(J_H=J_{B_j}\)。解出 \(a^\top\partial_tW\)，逐段分部积分，
再利用

\[
J_B=J_W+J_C,\qquad B=W+C,
\]

便得到

\[
J_W(B-P)-J_BW=J_W(C-P)-J_CW.
\]

证毕。

### 5.1 time 与 space 的精确分工，以及三种 query

先不做任何因果归属。full revision 有精确代数拆分

\[
\begin{aligned}
r_{\rm time}
&=\beta[W(z,t)-W(z,t+\delta_h)],\\
r_{\rm space}
&=\beta[W(z,t+\delta_h)-W(z+\delta_hP,t+\delta_h)],\\
r_{\rm full}&=r_{\rm time}+r_{\rm space}.
\end{aligned}
\]

它们的一阶形式分别是

\[
r_{\rm time}=-\beta\delta_h\partial_tW+O(\delta_h^2),
\qquad
r_{\rm space}=-\beta\delta_h J_WP+O(\delta_h^2).
\]

这是精确的“time/space”定义；它本身不推出前者必然改 mean、后者必然改
covariance。回到 terminal-adjoint 式，采用 Lie bracket 约定

\[
[W,C]=J_CW-J_WC.
\]

- **time-only，\(P=0\)**：内部项为
  \[
  J_WC-J_CW=-[W,C].
  \]
  它同时包含 boundary contrast 与 weak/calibration 两个场的不交换项；因此
  time-only 绝不能被简写成“纯 boundary”。

- **理想 calibration query，\(P=C\)**：内部项化为
  \[
  -J_CW.
  \]
  空间查询恰好消去 \(J_WC\) 这一半交叉响应。

- **baseline-tangent query，\(P=B\)**：此时 \(C-P=-W\)，内部项为
  \(-J_BW\)。它在一阶上更像 ordinary characteristic，但不会把 terminal-adjoint
  action 化为一个必然有利的符号。其 FID 更差是区分证据，不是由公式自动推出。

### 5.2 projection 的 conditional surrogate rationale

PFR 不能任意沿 \(C\) 查询；它把空间 probe 限制在当前 sampler 的 forward ray
\(\{\alpha B:\alpha\ge0\}\)。在这个约束集内，

\[
P=\arg\min_{\alpha B,\,\alpha\ge0}\|C-\alpha B\|^2.
\]

而 terminal-adjoint 公式中由 query mismatch 引入的项满足

\[
|a_t^\top J_W(C-P)|
\le \|a_t\|\,\|J_W\|_{\rm op}\,\|C-P\|.
\]

因此 Euclidean ray projection 精确最小化了这个最直接的 operator-norm 上界。但该上界无符号且
可能很松；\(C\)、forward ray 和 Euclidean metric 也都是设计约束，不是从 terminal risk
唯一推出的。所以这个结论只是：在“只能向前、并把 \(P=C\) 当作代理目标”的条件下，
\(\alpha^\star\) 最小化 worst-case nuisance mismatch。它不声称 projection 必然使 FID 下降，
也不排除 mismatch 本身有益。

这也与实验的真实强度相称：projection 不是主收益来源，time-only 已经很好；它在
两个不重叠 1K bank 上对 time-only 的经验 FID 增量约为 `0.55–0.65`。

还可以把它写成一个 minimax 等价式。记 \(K_B=\{\alpha B:\alpha\ge0\}\)，则

\[
P^*=\arg\min_{P\in K_B}\|C-P\|
=\arg\min_{P\in K_B}\sup_{\|u\|\le1}|u^\top(C-P)|.
\]

真实 nuisance sensitivity 是 \(u=J_W^\top a_t\)。因此在不知道 terminal adjoint 的
情况下，Euclidean 目标可解释为 forward-ray 内的 worst-direction mismatch surrogate，
不是已证明的 robust terminal-risk 规则。若以后能
对固定的局部 \((C,B)\)，若能估计敏感方向 \(u=J_W^\top a_t\) 的条件二阶矩
\(M=\mathbb E[uu^\top\mid z,t]\succeq0\)，则平均平方 nuisance
为 \((C-\alpha B)^\top M(C-\alpha B)\)，自然给出新设计

\[
P_M=\alpha_MB,\qquad
\alpha_M=\left[\frac{B^\top MC}{B^\top MB}\right]_+.
\]

当 \(B^\top MB=0\) 时需要预先规定 fallback；该式也只最小化 mismatch，不是完整 terminal
risk。它是第十二节 adjoint-metric 实验的来源。

### 5.3 为什么 FID 天然分成 mean-adjoint 与 covariance-adjoint

把 FID 看成分布前两矩的 population Gaussian Fréchet functional。设
\(\Sigma,\Sigma_r\succ0\)，并定义 Gaussian optimal-transport map

\[
A=\Sigma^{-1/2}
(\Sigma^{1/2}\Sigma_r\Sigma^{1/2})^{1/2}\Sigma^{-1/2},
\qquad A\Sigma A=\Sigma_r.
\]

对任意 feature 位移 \(Y_\varepsilon=Y+\varepsilon\eta(Y)\)，其 pathwise first variation 是

\[
\left.\frac d{d\varepsilon}{\rm FID}(Y_\varepsilon,Y_r)\right|_0
=\mathbb E[g_{\rm FID}(Y)^\top\eta(Y)],
\]

其中 feature-gradient 方向为

\[
\boxed{
g_{\rm FID}(Y)
=2\left[(\mu-\mu_r)+(I-A)(Y-\mu)\right].
}
\]

第一项对所有样本相同，只计价 feature mean；第二项是 centered、sample-dependent 的
covariance gradient，两者在 population \(L^2\) 中正交。经 decoder 与 feature extractor 拉回，
终端 latent adjoint 为

\[
a_T=J_{\phi\circ D}(z_T)^\top g_{\rm FID}(Y).
\]

这说明 mean/covariance 是 terminal objective 中数学上可分的两个 channel；它**不会**
把 temporal/spatial 分支先验分配给某个 channel。真正需要验证的是各分支经 flow
pullback 后分别与哪个 channel 对齐。另外，当前 \(n=1000,d=2048\) 的经验协方差必然奇异；
直接实作 adjoint 时必须预注册 ridge、PCA 或与现有 FID 实现一致的伪逆/子梯度规则。

---

## 六、第三条新命题：局部速度 MSE 可以严格更差，终点分布却严格更好

仅说“local MSE 与 FID 不完全相关”还不够。下面给一个与部署版本的两段
\(\gamma\) schedule、\(\tau=1/2\) horizon clamp 与 late strong-only switch 都一致的解析反例。

考虑一维目标 flow \(\dot z=0\)，初始与目标分布均为 \(\mathcal N(0,1)\)。令

\[
h=\frac1{32},\qquad
\beta(t)=
\begin{cases}
\frac85,&0\le t<\frac14,\\
\frac{17}{10},&\frac14\le t<\frac12,
\end{cases}
\qquad
\delta_h(t)=\min\left\{h,\frac12-t\right\}.
\]

取

\[
\psi(t)=\left(t-\frac14\right)\left(t-\frac12\right),
\qquad
A=\int_0^{1/2}\beta(t)[\psi(t+\delta_h)-\psi(t)]dt
=-\frac{5573}{983040},
\]

\[
k=A^{-1}=-\frac{983040}{5573},
\qquad W(t)=1+k\psi(t).
\]

在 active window 内定义

\[
S(t)=\frac{1+[\beta(t)-1]W(t)}{\beta(t)},
\]

在 \(t\ge1/2\) 令 strong field \(S(t)=1\)。因为 \(\psi(1/4)=\psi(1/2)=0\)，
\(S\) 在两个 schedule 边界连续。ordinary sampler 在 active window 为

\[
B(t)=W+\beta(t)(S-W)=1,
\]

之后也为 \(S=1\)。由于所有场均与状态无关，time-only 与 projected query 在这个例子中重合。
canonical PFR 场恰为

\[
F_h(t)=
\begin{cases}
1-\beta(t)k[\psi(t+\delta_h)-\psi(t)],&t<1/2,\\
1,&t\ge1/2.
\end{cases}
\]

由 \(kA=1\) 立即得到

\[
\int_0^1B(t)dt=1,
\qquad
\int_0^1F_h(t)dt=1-kA=0.
\]

再记

\[
Q=\int_0^{1/2}\beta(t)^2[\psi(t+\delta_h)-\psi(t)]^2dt
=\frac{4069501}{25165824000},
\]

则对真实 target velocity 0，

\[
\int_0^1|B(t)|^2dt=1,
\qquad
\int_0^1|F_h(t)|^2dt
=-1+k^2Q
=\frac{626052547}{155291645}
\approx4.03146.
\]

因此 time-integrated local regression risk 严格变差，但普通 guidance 将终点分布送到
\(\mathcal N(1,1)\)，PFR 却恰好留在 \(\mathcal N(0,1)\)。两者的 Gaussian FID/
\(W_2^2\) 分别为 1 和 0。这是连续时间闭式结果，不是积分误差。它严格证明：

> **不存在仅由 time-integrated local regression risk 排序推出 terminal distribution
> risk 排序的普遍单调定理。**

可执行复核：

```bash
python experiments/pfr_counterfactual_residual_theory.py \
  --output-dir docs/data/pfr_counterfactual_residual_theory_20260903
```

该反例是部署代数的一个 state-independent 特例，不声称构造的 \(S,W\) 是
Bayes-optimal 网络，也不验证空间 projection 的作用。它的作用是把理论目标纠正过来：
该测量的量是 adjoint-weighted terminal action，不是 local teacher MSE。

---

## 七、真实终点审计：temporal 主收益与 conditional spatial 增量

### 7.1 精确 terminal-mean witness

设 ADM pool-3 feature 的 reference mean、ordinary mean 和候选 mean 分别为
\(\mu_r,\mu_0,\mu_1\)，记

\[
e=\mu_r-\mu_0,
\qquad d=\mu_1-\mu_0.
\]

FID mean 项的改善有精确恒等式

\[
\boxed{
\|e\|^2-\|e-d\|^2=2\langle e,d\rangle-\|d\|^2.
}
\]

右边把“朝正确 terminal residual 的对齐”与“干预能量”分开。获益当且仅当

\[
2\langle e,d\rangle>\|d\|^2.
\]

在固定 seed0 的 paired 1K sample bank 和独立 5K reference 上：

| 条件 | terminal shift 与 residual cosine | \(2\langle e,d\rangle/\|d\|^2\) | mean 项改善 |
|---|---:|---:|---:|
| time-only | 0.6433 | 4.5867 | 2.1322 |
| temporal + parallel spatial | 0.6508 | 4.7495 | 2.1272 |
| temporal + orthogonal spatial | 0.6515 | 4.5642 | 2.1945 |
| full projected PFR | **0.6580** | 4.6871 | **2.1958** |

把 1000 个生成样本与 5000 个 reference 样本分别按偶/奇索引拆成两个不相交半集，四个
候选在两个半集上均保持正 witness。full PFR 的 mean 改善为 `2.2321/2.1343`；time-only
为 `2.1485/2.0856`。

这不是独立的新质量指标：它与 FID mean 项代数等价，只是把它重写成“方向对齐
收益减去干预能量”。偶/奇结果也是 post-hoc split-half stability check，不是 held-out
replication；它只显示符号不由某一个半样本单独驱动。

### 7.2 mean/covariance 的近正交分工

同一个 sample bank 的精确 FID 分解为：

| 比较 | 总 FID 改善 | mean 改善 | covariance 改善 | 主分量占比 |
|---|---:|---:|---:|---:|
| time-only vs ordinary | 2.3451 | 2.1322 | 0.2129 | **mean 90.92%** |
| full projected vs time-only | 0.6481 | 0.0636 | 0.5845 | **covariance 90.18%** |
| full projected vs ordinary | 2.9932 | 2.1958 | 0.7974 | mean 73.36% |

这一个 paired seed0 1K bank 呈现出与第五节相容、但尚未被它证明的经验分工：

- temporal query 相对 ordinary 的收益主要落在 FID mean 项；
- 在 time-only 上加 spatial response 后的增量主要落在 covariance 项；
- full 与 time-only 的逐样本总 feature-shift cosine 为 `0.9552`，只表明 time 分量主导后
  两个总移动很相似；它不证明孤立 spatial increment 同向，也不证明它是二阶机制。

同样地，state-only 相对 ordinary 有害，不与 spatial increment 相对 time-only 有利矛盾。
后者的正确局部问题是

\[
\left.\frac d{d\lambda}J(H_{\rm time}+\lambda r_{\rm space})\right|_{\lambda=0}
=\mathbb E\int_0^\tau a_{\rm time}^\top r_{\rm space}\,dt,
\]

其 adjoint \(a_{\rm time}\) 与 ordinary baseline 上评估 state-only 的 adjoint 不同，符号完全可以相反。

同时 precision 从 `0.427` 升至 `0.455`，recall 从 `0.7328` 降至 `0.7058`。所以当前
PFR 的 FID/precision 改善伴随 coverage trade-off，不是所有分布指标同时占优；在该
1K bank 上其 sFID `206.519` 也略差于 time-only 的 `206.442`。

本节 1K terminal audit 的复核命令为：

```bash
python experiments/analyze_pfr_terminal_distribution.py \
  --root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_distribution_audit_v1/seed0 \
  --reference-stats /home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz \
  --reference-activations /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_distribution_audit_v1/reference_imagenet100_validation_n5000_adm_activations.npz \
  --output-dir docs/data/pfr_counterfactual_residual_theory_20260903/terminal_observable_audit
```

### 7.3 修复 seed 审计后的不重叠 FID-5K 确认

发现第九节的历史 seed-overlap 后，我们立即在与历史 seed0/1 完全不相交的
`seed=1000003` RNG namespace 上重跑 ordinary IG 与代数同式的 projected query-control
implementation：

| 条件 | FID ↓ | mean 项 | covariance 项 | sFID ↓ | IS ↑ |
|---|---:|---:|---:|---:|---:|
| ordinary IG | 40.7983 | 7.7499 | 33.0484 | **69.8480** | 37.5832 |
| projected query-control（PFR 代数同式） | **37.6459** | **5.8276** | **31.8183** | 70.6396 | **40.5603** |

配对 FID 改善为 **3.1524**，其中 mean/covariance 分别贡献 `1.9223/1.2301`；noise 与
label hashes 在两方法间完全相同。该 bank 使用 batch RNG seeds `1000003..1000627`，与
历史 `0..625` 无交集，所以它修复了代数同式 PFR paired evidence 的 RNG-overlap
缺口；它不是 historical main runner 的 byte-identical replication。

这个结果确认的是 **ordinary→PFR 的不重叠 paired 差异**，不是旧 oblique 的新 bank 排名。
而且 `seed=1000003` 的前 1K 此前已用于 query-control，因此这不是 fully blind 的新 seed
选择；它是一次 partially seen 的不重叠 5K 扩展确认。sFID 仍变差，不能隐藏 trade-off。
该 run 在 RNG 代码修复前紧接着完成，原始 manifest 尚没有新 `batch_rng` 字段；compact
JSON 根据实际命令记录为 `legacy_additive_v1`。这里的可用性来自与历史区间完全不交，
不是说 legacy 相邻 seed 设计安全。

计算量也不应隐藏：ordinary/PFR 的 base ODE NFE 分别为 `42662/47270`，PFR 另有
`21339` 次 weak-prefix query。按仓库中 prefix 约为 full forward `39.1%` 的微基准粗略折算，
full-forward-equivalent overhead 约为 `30.4%`。因此这不是 compute-matched 优势证明；
solver 对照只说明把额外计算用于积分精化没有复制收益。

真正重新生成时应使用一个新的空 output root（若改回已存在的 archived root，runner
会直接 reuse 结果）：

```bash
python experiments/run_imagenet100_sit_pfr_query_controls.py fid \
  --output-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_counterfactual_residual_theory_v1/fid5k_reproduction_seed1000003 \
  --gpus 1,2 --conditions ordinary_ig,projected \
  --num-samples 5000 --batch-size 8 --seed 1000003 \
  --batch-seed-schema legacy_additive_v1
```

---

## 八、候选解释与仓库证据的一致性/反证账本

| 候选解释或可检验命题 | 已有区分实验 | 结果 | 当前证据边界 |
|---|---|---|---|
| 收益不是单纯 solver 精化 | tight Dopri5；matched-NFE explicit/implicit solvers | 更多 NFE 或更高阶没有复制收益 | 排除最简单的 quadrature 解释；不是一般 compute-matched 定理 |
| \(W(q)\) 不需更准 | held-out \(U=X-E\) teacher audit | current W `0.7510`；projected Q `0.7534`；time Q `0.7542` | 排除“future teacher 必须更准”；仅与 reference 解释相容 |
| baseline-tangent query 不足以带来收益 | Euler-tangent / coupled-time controls | `64.895/64.981`，远差于 oblique PFR | 区分局部 on-policy tangent；未测 exact characteristic flow |
| nominal time query 是主收益 | seed0 paired ordinary/time/full | `64.853/62.508/61.859`；time 占总 FID 改善 `78.35%` | 在一个 1K bank 上支持；第二 bank 没有 ordinary 行 |
| spatial response 是较小条件增量 | projected vs time-only | 两个 RNG 不重叠 banks 改善 `0.648/0.550` FID | 增量符号重现；尚未完成 adjoint 机制识别 |
| spatial 增量可能主要影响 covariance | ADM mean/covariance audit | projected 相对 time-only 的增量 90.18% 来自 covariance | 单个 seed0 1K 的 post-hoc 机制假说，尚未复现 |
| Euclidean projection 是一个受约束 surrogate | parallel/orthogonal/anti/donor | 多种方向仍有效，projection 只小幅最好 | 相容；不支持“唯一真方向” |
| 仅高 weak-density line integral 不够 | state-only | line integral 为正，但 FID `65.3742` | 排除纯 state-only density 解释 |
| 不能删除全部 shared weak drift | difference-in-differences | `66.59–68.77`，全失败 | 与保留 current weak anchor 相容 |
| 不能写成当前时刻的精确 density ratio | Jacobian antisymmetry pilot | PFR revision antisymmetric ratio `0.2532` | 仅 `n=4`、每点 2 probes；排除过强 score 叙事的先导证据 |

仍未完成的一箭是：直接测量第五节右侧的 exact adjoint action 与 boundary/
schedule-jump/cross-Jacobian
各项，并预注册其对 held-out query/horizon 的排序能力。当前 terminal feature 审计只记录了
终点统计结构，没有反向传播到每个时间点，故不能冒充完整因果闭环。

另有一个诊断口径需避免误用：`run_imagenet100_sit_pfr_query_controls.py` 当前的 cycle
residual 在 per-query 循环外只用 `controls["projected"]` 计算一次，再复制到各 query 行。
因此各 query 相同的 cycle ratio 不是 projected 方向特异性的证据；本文不使用它支持
projection claim。

---

## 九、证据勘误：历史相邻 seed 并不独立

全仓库审计发现一个会影响论文措辞、但不推翻 paired 方法差的 RNG 问题。

历史版本 sampler（现保留为 `legacy_additive_v1` branch）在每个 batch 使用

```python
manual_seed(args.seed + batch_index)
```

现行复现公式见 `experiments/batch_seed_schema.py`，审计函数见
`experiments/pfr_counterfactual_residual_theory.py`。于是：

- `N=5000, batch=8` 时，run seed0 使用 batch RNG seed `0..624`，run seed1 使用
  `1..625`；二者共享 624/625 个 batch，即 **4992/5000 = 99.84%** 的样本；
- `N=1000, batch=8` 时，共享 **992/1000 = 99.2%**；
- 保存的 PFR 5K label bank 满足 `seed0_labels[8:] == seed1_labels[:-8]`，4992 项
  逐项相等；按相同 CUDA RNG 调用重建 noise/label hashes 也与 manifest 完全一致。

因此下列表述必须降级：

- canonical PFR 的 `36.350961/36.343097` 是两个高度重叠 bank 上的测量，不能称为
  独立双 seed 复现；
- lambda sweep 的 `seed0/1` 同样不是独立重复；
- ordinary、old-oblique 和 PFR 在同一 nominal seed 内仍共享完全相同的 noise/label，
  所以 **within-bank paired 排序仍有效**；失效的是跨 bank 方差与重复性论证。

query-control 文档中名为 `fid1k_seed1` 的第二个 bank，其 manifest 实际 seed 是
`1000003`，与 seed0 的 RNG seed 集合不交。因此 projected 相对 time-only 的增量符号
在两个 RNG-disjoint 1K bank 上重现：

\[
61.8592<62.5067,\qquad 62.9425<63.4923.
\]

另一个非重叠 FID-5K bank 已在第 7.3 节补跑：ordinary/PFR 为
`40.7983/37.6459`，配对改善 `3.1524`。因此旧的“两次历史 seed 独立”措辞必须撤回，
但 PFR 相对 ordinary IG 的 paired 差异现在有了一次不重叠的 5K 确认。

本次同时把 RNG schema 升级为 namespaced batch seeds，并保留显式 legacy 模式来复现
历史产物；新 manifest 必须记录 schema。历史产物不覆盖、不重命名。

---

## 十、与相邻 guidance 工作的关系

这条解释与近年的一个共同观察相容：auxiliary model 有价值，并不要求它本身更准确，而
要求它提供结构匹配的差异。例如 [Guiding a diffusion model using sliding windows
(arXiv:2411.10257)](https://arxiv.org/abs/2411.10257) 的核心观察是 auxiliary model 若具有
相似但更强的 generalization error，反而特别适合 guidance；[Guiding Diffusion Models
with Semantically Degraded Conditions (arXiv:2603.10780)](https://arxiv.org/abs/2603.10780) 则把 null negative 换成
“almost good”的语义退化条件。

PFR 与它们的共同点是 **negative reference 的结构比其 standalone quality 更重要**。
区别是 PFR 的 negative 不是固定弱模型或固定退化条件，而是：

\[
\text{同一个 weak head}
+\text{nominal noise-time coordinate advance}
+\text{optional candidate-conditioned state query}.
\]

因此 PFR 更适合被描述成 **dynamic counterfactual reference in transport space**。
这不是因果推断中具有 null/exclusion/completeness 性质的正式 negative control。
本文不据此声称外部工作的理论自动证明 PFR；它们只提供相邻经验原则。

---

## 十一、论文故事线建议

### 11.1 一句话标题

**A Bad Predictor Can Be a Good Reference: Terminal-Risk Analysis of Internal Guidance**

更保守的标题：

**Counterfactual Reference Residualization for Internal Guidance**

### 11.2 四段主故事

1. **反常现象**：现有 guidance 默认 weak reference 越差越该远离，但“差多少、在哪里差”
   没有结构；仓库发现更准的未来估计和更准 solver 都无效。
2. **设计**：先把 weak reference 单独推到 nominal future time coordinate；再用
   calibration proposal 给空间 query 一个 forward-ray 约束。
3. **理论**：守恒 transport code 保证跨时间 velocity 可比较；finite difference 是
   coboundary；terminal adjoint 将一阶效应分成 boundary、schedule-jump 和 cross-Jacobian
   action；projection 有一个条件性 mismatch-bound rationale。
4. **证据**：仓库历史固定协议中的 best recorded FID + solver/Euler-tangent/
   posterior 反证 + 单 bank mean/covariance 假说 + RNG-disjoint 确认。

### 11.3 可以写进摘要的 claim

- PFR constructs a nominally time-advanced weak reference, with an optional proposal-conditioned
  spatial query, rather than predicting a more accurate future state.
- We derive a terminal-adjoint decomposition into boundary, schedule-jump, and cross-Jacobian terms.
- We give a schedule- and clamp-matched exact counterexample where PFR increases time-integrated
  local velocity MSE from 1 to 4.03 while eliminating terminal
  Gaussian Fréchet error.
- On one paired ImageNet-100 SiT-S/2 bank, temporal gain falls predominantly in the FID mean term,
  while the conditional spatial increment falls predominantly in its covariance term.
- PFR has the best recorded FID on the repository's historical fixed internal bank; this is not a
  public benchmark or compute-matched SOTA claim.

### 11.4 现在绝对不能写的 claim

- “PFR 是二阶、隐式或更准确的 ODE solver”；
- “\(W(q)\) 是更准确 posterior/teacher”；
- “PFR 是当前时间三个真实 density 的精确 ratio”；
- “projection 是唯一有用方向”或“Euclidean metric 第一性最优”；
- “fixed-point 收敛保证生成质量”；
- “两个历史 5K seed 独立复现”；
- “公开 ImageNet-1K FID-50K SOTA”或跨模型普遍成立；
- “FID 改善且 diversity 无代价”。

---

## 十二、下一轮最有信息量的实验，而不是继续扫参数

### A. terminal-adjoint 分项检验：最高优先级

在固定 128–256 个 paired trajectories 上，对一个冻结、可微的 terminal feature risk
反传 adjoint \(a_t\)。首先核对有限 \(h\) 的 exact first variation

\[
E_{\rm exact}=\frac1h\mathbb E\int_0^\tau a_t^\top r_h\,dt
\]

与小 \(\lambda\) intervention 的实测差分；然后再验证渐近分解

\[
\begin{aligned}
E_{\rm endpoint}&=-[\beta a^\top W]_0^\tau,\\
E_{\rm jump}&=\sum_{j\ge1}(\beta_j-\beta_{j-1})
a_{\tau_j}^\top W_{\tau_j},\\
E_{\rm mismatch}&=\sum_j\int_{\tau_j}^{\tau_{j+1}}
\beta_j a^\top J_W(C_j-P_j)dt,\\
E_{\rm cross}&=-\sum_j\int_{\tau_j}^{\tau_{j+1}}
\beta_j a^\top J_{C_j}Wdt.
\end{aligned}
\]

预注册：在未参与选择的 seed 与 horizon 上，先要求 \(E_{\rm exact}\) 预测实际一阶变化；
再要求四个渐近项之和在随 \(h\) 收缩的 remainder 内追踪 \(E_{\rm exact}\)。只报告相关性而不报告
sign accuracy 不算通过。

### B. mean-removal / centered-response 干预

把 temporal revision 分成 batch/population mean 与 centered residual，在相同 correction RMS
下分别采样。预注册的经验假说（不是第五节定理的直接推论）是：

- mean branch 保留大部分 FID mean 改善；
- centered branch 更影响 covariance/precision-recall；
- 二者的终点作用不应被 local \(U\)-MSE 排序解释。

需要避免 batch coupling 伪影，最好用独立 calibration bank 预估 time-binned population
mean，而不是在生成 batch 内即时求均值。

### C. projection metric 的真正检验

Euclidean projection 只最小化 Euclidean mismatch bound。用低秩估计的
\(J_W^\top aa^\top J_W\) 构造 adjoint metric，并在**预注册 held-out bank**比较：

\[
\arg\min_{\alpha\ge0}\|C-\alpha B\|_{M_t}^2.
\]

若新 metric 不能稳定超过 Euclidean，保留当前最简单设计，不再添加理论装饰。

### D. 外部效度

在修复 RNG 后依次做：

1. 一个未使用的 5K bank：ordinary/PFR/old-oblique；
2. 至少一个不同训练 seed 或不同 weak depth；
3. 最后才做 ImageNet-1K FID-50K 或第二架构。

在 A 没有通过前，不建议继续做更多 \(h,\gamma,\lambda\) 局部 sweep；那些只会提高结果，
不会提高解释力。

---

## 十三、最终研究判断

两条路线中，应明确选择第二条：**理论—设计—仓库内最佳结果**，但把理论核心从
numerical lookahead 改成 counterfactual reference residualization 的 terminal-risk analysis。

真正令人意外而读完又自然的点不是“我们算得更准”，而是：

> **生成模型里的弱预测器不一定要成为更准的 teacher 才有用。保留它当前的
> transport anchor，但把 calibration 里的参考移到一个 nominally time-advanced query，
> 就能构造一个新 field intervention；该干预的价值应由它对终点风险做了什么来判断，
> 而不是由 query 自己的局部预测误差来判断。**

这条理论既容纳了现有最好结果，也解释了为什么局部 MSE 不必改善；更重要的是，它给出了下一步
能被直接证伪的终点量，而不是再给同一个数值积分直觉换一个名字。

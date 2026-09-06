# RAEv2：以 full score 平衡 guidance 与随机扩散

日期：2026-09-06。状态：**未满足最新方法准入；仅数学审计，真实采样暂停，无待执行调参队列。**

用户最新约束要求理论连接模型误差/生成质量，不能依靠手选外推强度或时间窗。
下文的 `eta` 与 `(0.5,0.95)` 只保留为已完成数值审计的设置，**不再作为待测方法**。
当前保证仅为 oracle marginal preservation 和相对 full 分布的耗散，不能保证有限
RAEv2 head 的质量提高，因而本轮不据此继续扫参或采样。第 7 节列出无自由幅度的
自然版本及其仍未解决的障碍。

本轮改换设计轴：保留现有 IG 漂移，用 full conditional head 给出的 score 配平随机扩散。
不再修改 weak-reference 的跨时间几何，也不以消除 full/base 差距为目标。
其连续机制属于已知的 stochastic interpolants / Langevin 动力学；本轮贡献范围只是
RAEv2 的数值适配、与 guidance 的受控比较，以及尚待验证的质量机制。

## 1. 可识别对象与假设

RAEv2 使用

\[
z_t=(1-t)x+t\epsilon,\qquad t:1\to0.
\]

记 full clean prediction 为 \(F(z,t,c)\)，ordinary guided clean 为 \(G\)。则

\[
s_F(z,t)=\frac{(1-t)F(z,t)-z}{t^2},\qquad
b_{IG}(z,t)=\frac{G(z,t)-z}{t}.
\]

这里 \(b\) 是随生成时间 \(u=1-t\) 增长的漂移。当前作用区间远离 \(t=0\)，
不把官方 velocity 的 \(t_\epsilon\) clamp 偷换进 score 恒等式。

在理想的 Bayes full head 下，\(s_F=\nabla\log p_{F,t}\)，且 full ODE 输运同一
\(p_{F,t}\)。考虑

\[
dz=\left[b_{IG}+d(t)s_F\right]du+\sqrt{2d(t)}dB_u.
\]

若关闭额外 IG，score 漂移与扩散在 Fokker–Planck 方程中严格抵消，因此任意非负
时间函数 \(d\) 都保留 full model 的边缘密度。这个结论允许任意数据均值、协方差
和非高斯形状，不要求 unit isotropic data。

开启 IG 后令 \(g=b_{IG}-b_F\)，实际生成密度为 \(q\)。在同一理想假设、光滑性和
无穷远边界项消失条件下，得到

\[
\frac{d}{du}\operatorname{KL}(q\|p_F)
=\mathbb E_q[g\cdot\nabla\log(q/p_F)]
-d\,\mathbb E_q\|\nabla\log(q/p_F)\|^2.
\]

因此新增项是可识别的 relative-KL 耗散；它不需要把有限网络的 \(F-B\) 解释成
保守 density-ratio score。这比“某个局部 gap 变小”多了明确的分布对象。
但它也**不保证 FID 改善**：IG 有用的分布校准同样可能被削弱；有限 full head 也不是
精确 Bayes score。当前假设只是：在有用外推之外，随机配平可能减轻额外的模式集中
或轨迹误差。不能提前断言 RAEv2 已经过度收缩。

相关原始来源：[Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
给出共享边缘密度的可调 ODE/SDE 家族。[Foresight Guidance](https://arxiv.org/html/2510.21512v1)
在本轮仅提供“先确定校准对象，再选择算子”的启发；其固定点不是当前方法的前提。

## 2. 有限步稳定更新

先**原样执行已有 Euler IG 更新**得到 \(z_E\)。为避免对 Gaussian 收缩和加噪分别
做 Euler 而产生方差失配，选一个正的解析参考方差

\[
c(t)=(1-t)^2+t^2,
\qquad r_t=s_F(z_t,t)+z_t/c(t).
\]

\(c(t)\) 只是数值分裂参考：无论真实 latent 是否已完全白化，
\(-z/c+r_t=s_F\) 都严格成立。**不假设真实 RAE latent 的总体协方差为单位矩阵，
也不假设它是 Gaussian。**

令 \(h=t-t_{next}>0\)，冻结旧时刻的非线性 residual \(r_t\)，把 Gaussian 部分
作精确 OU 更新：

\[
\begin{aligned}
\lambda &= d(t)h/c(t_{next}),\\
z_{next}&=e^{-\lambda}z_E
+c(t_{next})(1-e^{-\lambda})r_t
+\sqrt{c(t_{next})(1-e^{-2\lambda})}\,\xi.
\end{aligned}
\]

展开到一阶为

\[
z_{next}=z_E+d(t)h\,s_F(z_t,t)+\sqrt{2d(t)h}\,\xi+O(h^2)
\]

其中随机项系数相对首项的误差为 \(O(h)\)。该实现是弱一阶分裂，**一般有限步并不
严格保留非高斯密度**。Gaussian unit-data oracle 下 residual 为零，如果 Euler 部分
换成精确 Gaussian transport，则 OU 子步在有限步精确保留目标方差；保留真实 Euler
时，它只会衰减已经存在的 Gaussian 方差误差。非零均值和非单位协方差时，冻结 residual
仍有有限步误差，必须检查步长收敛。

## 3. 已审计的时间与剂量；不作为方法候选

此前 CPU 数值审计采用

\[
d(t)=\eta\frac{t}{1-t}\,\mathbf 1\{0.5<t<0.95\},
\qquad \eta\in\{0.05,0.15\}.
\]

这是常见 linear-flow SDE 扩散形状的区间限制版本，起点和终点均关闭。
代码对跨边界 Euler 步只积分与区间重叠的长度；所有 denominator 均远离奇点。
这些值及窗口均为人为设置，不满足最新准入条件；不继续实施真实模型扫描。

其尺度并非在高维 latent 上任意加噪。归一到参考方差后，每个坐标的噪声标准差是
\(\sqrt{1-e^{-2\lambda}}\)，相对向量范数也不会再随 \(\sqrt D\) 放大，因为参考
latent 与噪声的范数都按 \(\sqrt D\) 增长。仍应记录最大 \(\lambda\)、累计
\(\sum\lambda\) 和逐步扰动 RMS；高维并不意味着任意剂量安全。

100 步均匀网格的解析剂量为：

| eta | 最大 lambda | 累计 lambda |
|---:|---:|---:|
| 0.05 | 0.010708 | 0.137083 |
| 0.15 | 0.032124 | 0.411250 |

真实 RAEv2 使用 shifted grid，不能照搬这两列，必须记录实际网格的剂量。
有利剂量和作用区间尚属待检验设计选择，不是理论推出的最优参数。

## 4. 模块接口与随机数

实现：[`raev2_stochastic_guidance.py`](../experiments/raev2_stochastic_guidance.py)。

```python
refresh_after_euler(
    state, euler_state, full_clean, time, next_time,
    eta=0.15, noise=refresh_noise, mode="balanced", interval=(0.5, 0.95),
)
```

- `state/full_clean` 必须来自旧时刻，`euler_state` 是原采样器的下一状态。
- `eta=0` 或区间外返回同一个 `euler_state` 对象，完全不改变基线算术。
- `mode` 为 `balanced / score_only / noise_only / none`。
- 所有 score/refresh 算术在 fp32 完成，返回 `euler_state` 的 dtype。
- 模块自身不调用 RNG。调用方必须为 refresh 建立独立 generator，不能消耗下一批
  初始噪声或标签的随机流；跨条件固定同一 refresh noise bank。
- 不增加 model forward；full clean 是普通 IG 已有的输出。

## 5. 若质量机制成立时所需的对照；当前不执行

若未来有不依赖手选强度/窗口、且满足质量机制要求的实现，仍需要分离下列两个成分。
这些是必要的可证伪对照，不构成当前启动试验的理由：

| 条件 | 更新 | 要排除的解释 |
|---|---|---|
| ordinary | `z_E` | 配对基线 |
| balanced | 精确 Gaussian score/noise 配平 | 主候选 |
| score-only | 主候选移除 noise | 收益是否只是更强收缩/某种确定性 guidance |
| noise-only | `z_E + noise_std * xi` | 收益是否只是普通扰动 |

固定 reference、100 步、标签、初始噪声和 batch 布局，记录实际 forward 数。
若仅有噪声也同样有效，不能把结果归因于 KL 配平。若仅 score-only 有效，应回到
确定性 distribution calibration 解释。另做 full-only 的 ODE/SDE 对照，可辨别收益是
一般采样正则化还是与 IG 的特定交互。

目标仍为相对当前强 baseline 约 5% 的 FID 降幅。FID-1K 只用于配对筛查；成功条件
应在未参与筛选的独立 bank 复现，并以更多样本检查。尤其保留 KID、IS 和可得的
precision/recall，避免再把旧 PFR 的 IS 上升和 1K 好转误判成正式分布质量改善。

## 6. 已完成的 CPU oracle 验证

入口：[`audit_raev2_stochastic_gaussian.py`](../experiments/audit_raev2_stochastic_gaussian.py)。
数据：[`gaussian_audit.json`](data/raev2_stochastic_guidance_20260906/gaussian_audit.json)。

联合三维 target 为

\[
\mu=(-1.5,0.4,2.0),\qquad\Sigma=\operatorname{diag}(0.25,1,4).
\]

因为 Gaussian oracle 使整个离散 sampler 成为 affine map，直接递推精确均值/方差，
完全没有 Monte Carlo 或 GPU 误差。`eta=.15` 的 balanced 结果：

| 步数 | 三维最大终端协方差相对误差 | 最大终端均值绝对误差 |
|---:|---:|---:|
| 100 | 0.033711 | 0.018958 |
| 400 | 0.008553 | 0.004653 |
| 1600 | 0.002146 | 0.001158 |
| 6400 | 0.000537 | 0.000289 |

误差按约一阶速度趋零；这同时诚实暴露了 100 步的有限步误差。方差为 1 的坐标在
6400 步时，ordinary/balanced 的方差分别为 `0.999598/0.999634`，score-only 为
`0.458031`，noise-only 为 `1.779591`。后两者的失配不会因网格变细消失。

另外通过：`eta=0` 同一 Tensor 对象返回、独立 refresh RNG 不改变初始 RNG 状态、
非零均值各向异性 tensor 更新与解析公式误差 `<1.2e-7`，以及 unit-Gaussian
有限步 OU 方差恒等式。以上验证数值与概率机制，**没有提供真实 RAEv2 的 FID 证据**。

## 7. 无自由系数也不自动解决质量保证

由线性 Gaussian channel 的 \(\alpha(t)=1-t,\sigma(t)=t\) 可以唯一构造 Markov
forward diffusion：

\[
f(t)=\alpha'(t)/\alpha(t)=-1/(1-t),
\qquad
a(t)=(\sigma^2)' -2 f\sigma^2=2t/(1-t).
\]

其精确 reverse SDE 对应本节公式的 \(d(t)=t/(1-t)\)，没有新增 eta 或窗口。
但 \(t=1\) 存在奇点；离散 Gaussian channel transition 可以避免直接计算奇点，
其 reverse conditional 又涉及真实 clean posterior 的不确定性。只把 posterior mean
替成网络 \(F\)，不能自动得到正确的 finite-step reverse distribution。

更根本的障碍是：**full-anchor 耗散不保证比 ordinary IG 更接近真实分布。**
一个明确反例是令真实 Bayes clean field 为 \(F_*\)，full 有非零偏差，而 weak 满足

\[
B=\frac{(1+\gamma)F-F_*}{\gamma},\qquad \gamma>0.
\]

此时 ordinary \(F+\gamma(F-B)=F_*\) 已精确正确，朝有偏 full score 的 stochastic
relaxation 会破坏它。该构造不声称真实 RAEv2 恰为此例；它说明单凭现有 head 的形式
及 oracle 保分布定理，不能获得所要求的无条件质量保证。

因此自然 diffusion 系数只解决“参数是否人为选择”，没有解决“修正是否针对真实模型
误差”。除非建立并独立验证额外误差条件，这一方向当前**不合格，不开展新试验**。

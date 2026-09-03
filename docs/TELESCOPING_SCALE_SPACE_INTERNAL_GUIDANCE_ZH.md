# Telescoping Scale-Space Internal Guidance

## 1. 研究边界

当前最佳 PFR/ITPR 条件不能再解释为隐式数值积分，也不能解释为只在有限模型误差存在时
才出现的 correction：当 strong 与 weak 都等于同一个 population field 时，跨时间项仍然
非零。深度乘信息的 difference-in-differences 条件虽然满足 population diagonal 上归零，
但 FID-1K 全部退化到 `66.59--68.77`，已经否定该质量假设。

因此本轮明确研究一个不同问题：有意改变当前模型分布的有限强度 density-ratio guidance。
这类方法在完美模型极限仍非零不是不一致；它的合法性来自一个明确的新目标分布，而不是
来自“修正模型误差”。

跨噪声 score ratio 已由 Self-Guidance 提出。本轮不把它重新包装成 novelty。需要检验的
新问题是：在 rectified flow 的 internal strong/weak hierarchy 中，深度 ratio 与噪声 ratio
能否按概率代数精确合成，以及现有跨时间 velocity 差是否真的实现了这个对象。

## 2. Rectified flow 的统一 Gaussian-channel 坐标

本仓库使用

\[
Z_t=(1-t)E+tX,\qquad t:0\to1,
\]

其中 `E` 是标准高斯 source，`X` 是数据 endpoint。对 `t>0` 定义 endpoint-normalized
observation

\[
Y_\sigma=\frac{Z_t}{t}=X+\sigma E,
\qquad
\sigma=\frac{1-t}{t}.
\]

这才是标准加性 Gaussian heat channel。记其密度与 score 为

\[
q_\sigma(y),\qquad r_\sigma(y)=\nabla_y\log q_\sigma(y).
\]

若 `v(z,t)` 是 linear-path marginal velocity，则 Tweedie 恒等式给出

\[
\boxed{
r_\sigma(y)
=\frac{t\,[t v(z,t)-z]}{1-t},
\qquad y=\frac zt.
}
\]

反变换为

\[
\boxed{
v(z,t)=\frac zt+\frac{1-t}{t^2}r_\sigma(z/t).
}
\]

因此不同时间的 velocity 不能直接当成不同噪声层的 score 相减；它们包含不同的 affine
identity term 与不同的比例系数。

## 3. 同一个 endpoint coordinate 上的 noisier reference

选择 `t_-<t`，则

\[
\sigma_- = \frac{1-t_-}{t_-}>\sigma_t.
\]

为了在同一个 `y` 上比较两个 heat-channel density，reference state 必须为

\[
\boxed{
z_-=t_-y=\frac{t_-}{t}z.
}
\]

直接把同一个 `z` 输入 `t_-` 虽然也定义了两个 ambient density 的比值，但它不再比较
同一个 endpoint-normalized observation，不能使用 Gaussian heat-flow 的 valley-filling
解释。

## 4. 深度与噪声 ratio 的望远镜消去

在同一个 `(y,t)` 上记：

\[
p_0=q_{S,t},\qquad
p_1=q_{W,t},\qquad
p_2=q_{W,t_-}.
\]

`p0` 是 strong/current，`p1` 是 weak/current，`p2` 是 weak/noisier。两个相邻质量证据
分别为

\[
\nabla\log\frac{p_0}{p_1}=r_{S,t}-r_{W,t},
\]

\[
\nabla\log\frac{p_1}{p_2}=r_{W,t}-r_{W,t_-}.
\]

它们不是两个需要独立调权的 heuristic。用同一个 guidance exponent `gamma` 相加时，
中间 density 精确消去：

\[
\boxed{
\nabla\log\frac{p_0}{p_1}
+\nabla\log\frac{p_1}{p_2}
=\nabla\log\frac{p_0}{p_2}.
}
\]

于是新 score 唯一写为

\[
\boxed{
r_{\rm TSI}
=r_{S,t}+\gamma(t)\,[r_{S,t}-r_{W,t_-}].
}
\]

它对应逐噪声层的明确有限强度目标

\[
\boxed{
\widetilde q_t(y)
\propto
q_{S,t}(y)
\left[
\frac{q_{S,t}(y)}{q_{W,t_-}(y)}
\right]^{\gamma(t)}.
}
\]

这里没有小 `gamma`、Taylor 或数值积分假设。它只是 density-ratio 的代数恒等式。

## 5. 三条精确边界

1. `t_-=t` 时，TSI 严格退化为普通 Internal Guidance；
2. `gamma=0` 时，TSI 严格退化为 strong field；
3. 即使 `S=W`，只要 `t_-<t`，TSI 仍然非零。这不是误差 correction，而是
   Self-Guidance 式的质量-覆盖倾斜。

第三条也说明它可能改善 FID/precision、同时损害 recall/sFID；理论不承诺所有指标同向。

## 6. 与现有 PFR/ITPR 的严格区别

现有方法使用

\[
W_t+(1+\gamma)(S_t-W_{t+h}),
\]

其中 `t+h` 在本仓库方向上更干净，而且三项是各自时间参数化下的 velocity。它一般不等于
任何上述 score density ratio。即使新理论成立，也不能倒推现有 `36.3470` 已经由该理论
解释。

## 7. 第一次 smoke 暴露的 prior-boundary 问题

最初按第 2 节的 endpoint-normalized coordinate 直接实现后，8-sample smoke 即出现明确
坍缩：noisier-reference 两个条件接近常数图，cleaner-reference 变成高频噪声。诊断中
score revision RMS 最高达到 `50.9--313.9`，因此没有进入 FID。

原因不是普通数值积分误差。`Y=Z_t/t` 在 `t->0` 发散，而跨两个不同 Gaussian width
做有限 density tilt 会改变 source prior 的尺度；把 score 再变回 current velocity 时含有
`1/t` 或 `1/t^2` 放大。公式作为坐标变换没有错，但它不是 source-compatible guidance。

所以真正用于筛查的 density 定义改为未缩放的 path marginal `p_t(z)`：

\[
\boxed{s_t(z)=\frac{t v_t(z)-z}{1-t}.}
\]

所有 `p_t` 都在同一个 `z` 空间，且 `t=0` 共享标准高斯 source。为使有限网络误差下的
新增 velocity 在 source boundary 仍有界，temporal ratio 使用无参数边界权重

\[
a(t)=\min(1,t/H).
\]

于是可部署主式为

\[
s_{\rm new}
=s_{\rm IG}
+\gamma a(t)[s_{W,t}-s_{W,t_-}],
\qquad t_-=\max(0,t-H).
\]

当 `t>=H` 时，中间 `W,t` 完全望远镜消去；当 `t->0` 时，新增 temporal ratio 连续关闭，
严格保留原 IG 的 source boundary。

## 8. 预注册筛查条件

固定 checkpoint、depth-4 head、IG schedule、noise/labels、`H=1/32` 和评估代码，只改变
reference 构造：

1. `marginal_score_weak_noisier`：主条件；weak temporal ratio 与 ordinary IG 以相同
   `gamma` 合成；
2. `marginal_score_strong_noisier`：把 temporal branch 换为 strong，判断 internal weak
   hierarchy 是否提供超出普通 Self-Guidance 的信息；
3. `marginal_score_weak_cleaner`：反转 noise orientation；
4. `velocity_noisier_aligned`：直接相减跨时间 velocity，检验收益是否只是参数化效应；
5. 普通 depth-4 IG、现有 PFR/ITPR 作为已知 anchors。endpoint-normalized 三个失败条件
   只保留为 smoke 级反例，不进入 FID。

边界权重由上述 source regularity 固定，不另扫 exponent 或 taper。

预注册判断：

- 若主条件优于普通 IG，且优于 cleaner / raw-velocity controls，则 density-ratio 设计得到
  初步支持；
- 若 raw velocity 好而 score 主条件差，则现有增益是 parameterization-specific，停止使用
  density-ratio 理论；
- 若 cleaner 与 noisier 无稳定顺序，则 heat-flow artifact-suppression 不是本 setup 的主要
  机制；
- FID-1K 未通过时不扩 FID-5K，不用后验调 exponent 挽救理论。

## 9. 配对 FID-1K 结果与结论

固定 seed 0 的同一 noise/label bank，结果为：

| condition | FID-1K |
|---|---:|
| weak + noisier marginal-score ratio | 114.1261 |
| strong + noisier marginal-score ratio | 110.8041 |
| weak + cleaner marginal-score ratio | 198.3493 |
| raw velocity noisier reference | 80.4258 |

四个条件都显著差于普通 depth-4 IG，也远差于现有 PFR/ITPR。它们没有进入 FID-5K，
也没有继续扫描 density-ratio exponent。

因此本轮只保留一个负结论：虽然固定时间上的 score 线性组合确实具有精确的
power-density 解释，但当前 internal weak head 在另一噪声层级的输出并不是一个可直接用作
质量反例的合法 density witness。现有 `36.3470` 的收益不能归因于 Self-Guidance 式的
跨噪声 density ratio。

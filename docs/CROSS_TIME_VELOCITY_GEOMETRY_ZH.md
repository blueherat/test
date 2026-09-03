# 跨信息时间速度差：精确分解、反例与后验压力假设

## 1. 先排除两个错误故事

现有最好方法在普通 Internal Guidance

\[
G=W_t+\beta(S_t-W_t),\qquad \beta=1+\gamma
\]

上加入

\[
\beta(W_t-W_{t+h}).
\]

这里的 `W_{t+h}` 在同一 latent 上查询更干净的 information time。这个操作不是隐式 ODE
求解；隐式积分只能改变离散求解误差，不能把错误的 learned vector field 变成正确场。
它也不是现成的跨噪声 density-ratio guidance：不同时间的 score-to-velocity 映射不同，
直接相减 velocity 不能等同于相减 score。

## 2. 跨时间 velocity 差的精确分解

在线性 bridge

\[
Z_t=(1-t)E+tX
\]

下，边缘 score 与 velocity 的精确关系为

\[
s_t(z)=\frac{t v_t(z)-z}{1-t},
\qquad
\mathcal V_t(s)=\frac{z+(1-t)s}{t}.
\]

令 `r=t+h`。在 `v_t-v_r` 中插入同一个中间量
`\mathcal V_r(s_t)`，得到无近似恒等式

\[
\boxed{
v_t-v_r
=
\underbrace{\big[v_t-\mathcal V_r(s_t)\big]}_{P_{t\to r}:\text{固定 score，只换表示}}
+
\underbrace{\big[\mathcal V_r(s_t)-v_r\big]}_{E_{t\to r}:\text{固定表示，只换 score}}.
}
\]

第一项还能化为

\[
\boxed{
P_{t\to r}
=\frac{h}{(t+h)(1-t)}(v_t-z).
}
\]

第二项为

\[
\boxed{
E_{t\to r}
=\frac{1-t-h}{t+h}(s_t-s_{t+h}).
}
\]

所以 `P` 与 `E` 都依赖选择的 score/velocity 表示；只有总和 `v_t-v_{t+h}` 是实际部署
velocity field 的完整跨时间变化。

## 3. 配对实验：两个分量都不是机制

在同一个 v800、depth-4 weak head、`h=1/32`、相同 noise/labels、相同 IG schedule 下，
分别只把 `P`、只把 `E`、以及精确重组的 `P+E` 加回普通 IG。FID-1K 为：

| revision | FID-1K |
|---|---:|
| only parameterization transport `P` | 170.6241 |
| only score evolution `E` | 271.1539 |
| exact recomposition `P+E` | **61.9691** |

8-sample trajectory diagnostics 中，`P` 与 `E` 的逐次 cosine 均值为 `-0.9606`；两者 RMS
均约 `0.50`，但单独部署分别产生过度平滑和高对比纹理崩溃。精确重组严格等于原来的
time-only weak-reference revision，并恢复正常图像。

因此不能说“有效的是 score 演化”，也不能说“有效的是参数化增益”。二者是一个随时间
变化的表示中产生的两个巨大、近乎反向的坐标分量。把其中任何一个当作独立 correction
都会破坏必要抵消。当前唯一被数据支持的对象是完整 physical velocity difference。

## 4. 一个更高层但仍需验证的统计结构

令 conditional particle velocity 为

\[
U=X-E,
\]

其边缘 Bayes velocity 与条件协方差分别为

\[
v_t(z)=\mathbb E[U\mid Z_t=z],
\qquad
\Sigma_t(z)=\operatorname{Cov}(U\mid Z_t=z).
\]

直线 conditional particles 的联合密度满足自由输运方程。取零阶与一阶速度矩可精确得到

\[
\partial_t p+\nabla\cdot(pv)=0,
\]

\[
\partial_t(pv)+\nabla\cdot\left[p(vv^\top+\Sigma)\right]=0.
\]

消去 `partial_t p` 后：

\[
\boxed{
(\partial_t+v\cdot\nabla)v
=-\frac1p\nabla\cdot(p\Sigma).
}
\]

这说明边缘场的时间变化不是数值误差。即使每个 conditional particle 都以常速度直线
运动，许多不同速度的粒子在同一位置叠加后，条件速度分散仍会让边缘场产生真实曲率。
相邻时间的完整 velocity query 因而可能在不训练 covariance head 的情况下，读出一部分
二阶后验信息。

这个矩方程并非新定理；近期 Flow Matching 的路径直化工作也使用了同一条件协方差结构。
尚未解决、且属于本项目的问题是：internal weak field 的哪一种有限查询能把该统计结构
变成有益的多步 generation correction。

## 5. 下一轮预注册判别

对任意 probe velocity `q`，有

\[
-(\partial_t+q\cdot\nabla)v
=
(v-q)\cdot\nabla v
+\frac1p\nabla\cdot(p\Sigma).
\]

因此有三个不引入新 coefficient 的自然查询：

1. `q=0`：固定 state 的 Eulerian temporal change，包含 convection 与 pressure；已有结果
   约为 FID-1K `62.51`；
2. `q=W`：沿 weak field 自身 characteristic，在 Bayes 极限中隔离 posterior-pressure 项；
3. `q=S`：用更准确的 strong field 运输 weak estimator，检验 weak-model误差是否破坏
   `q=W` 的统计语义。

普通 guided characteristic 只是另一个经验 `q`，不等同于上述 Bayes characteristic。下一轮
只比较 `q=W`、`q=S` 与已知 `q=0`；不扫描空间系数。若 `q=W` 稳定优于 time-only，
posterior-pressure 可以升级为设计原则。若它不优于 time-only，则该矩方程只能解释“为何
存在曲率”，不能解释“为何该曲率提高质量”，这条理论必须降级，不能继续包装现有最优点。

## 6. Posterior-pressure 判别结果

配对 FID-1K 得到：

| query velocity `q` | interpretation | FID-1K |
|---|---|---:|
| `0` | Eulerian time-only，已有对照 | 约 62.51 |
| `W` | weak characteristic / pressure probe | 64.6762 |
| `S` | strong-transported weak probe | 64.9222 |
| projected refinement ray | 当前 PFR 查询 | 约 61.88 |

因此 posterior-pressure hypothesis 没有通过预注册判据。条件协方差动量方程仍是正确的
population identity，也说明 temporal velocity change 不是数值积分误差；但隔离 pressure
项会损失大部分质量收益，不能用它解释当前最好方法。此处停止，不做第二 seed 或 FID-5K。

## 7. 当前最硬的理论落点

对直线 conditional path，

\[
U=X-E
\]

沿整条 path 守恒，而所有时间的 Bayes velocity 都是在不同 observation channel 下估计
同一个随机变量：

\[
v_t(z)=\mathbb E[U\mid Z_t=z].
\]

这给 raw velocity contrast 一个 score contrast 没有的语义：它比较的是同一守恒 transport
code 的两次 posterior estimate。第 3 节的灾难性分解进一步说明，若先把它拆成随时间改变
含义的 score 与参数化分量，这个共同目标结构就被破坏了。

这条结论解释了“为什么必须比较完整 velocity”，却仍不自动解释“为什么对该 posterior
revision 做负向外推会降低 FID”。后者仍是接下来方法理论需要解决的缺口。

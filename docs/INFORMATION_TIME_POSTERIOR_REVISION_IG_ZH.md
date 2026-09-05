# Information-Time Posterior Revision for Internal Guidance

> **2026-09-03 解释与 RNG 勘误。** 后续 held-out \(U=X-E\) 审计显示 future weak
> query 的 posterior MSE 并未改善；本页的 information/posterior 语言只能描述构造，
> 不能作为质量机制。推荐解释见
> [`PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md)。
> 本文表中的 historical `seed=0/1` 使用 additive per-batch seeds，1K bank 共享
> 992/1000 个样本；paired 排序有效，跨 seed 独立性无效。

## 1. 一句话直觉

普通 IG 只问：

> 深层网络比浅层网络多预测了什么？

本方向再问一次：

> **如果把这份深层 refinement 作为反事实信息写进 latent，并让浅层头在更高信息量的
> 时间重新判断，它会怎样修订自己对 clean/noise endpoints 的看法？**

这里的 future query 不是数值积分器对真实未来状态的近似。它是一个只用于读取模型内部
belief 的 counterfactual probe。扩散时间在这里同时具有第二种用途：除了作为 sampler
clock，还可以作为改变 posterior information level 的干预变量。

## 2. 一个 velocity 同时编码两个 endpoint beliefs

考虑 SiT 的线性 bridge

\[
Z_t=tX+(1-t)E,\qquad E\sim\mathcal N(0,I),\qquad t:0\to1.
\]

对任意 velocity predictor `W(z,t)`，定义它诱导的两个 endpoint estimates：

\[
\widehat X_W(z,t)=z+(1-t)W(z,t),
\]

\[
\widehat E_W(z,t)=z-tW(z,t).
\]

如果 `W` 等于 Bayes velocity

\[
v^*(z,t)=\mathbb E[X-E\mid Z_t=z],
\]

那么严格有

\[
\widehat X_W=\mathbb E[X\mid Z_t=z],\qquad
\widehat E_W=\mathbb E[E\mid Z_t=z].
\]

对有限网络，它们至少仍是由同一个 velocity 一致诱导出的 clean/noise estimates；不能
未经验证就称为精确 posterior means。

## 3. 唯一的 query-displacement-invariant posterior revision

取两个任意时空查询

\[
p=(z,t),\qquad q=(z',\tau).
\]

定义 clean belief revision 与 negative-noise belief revision：

\[
R_X=\widehat X_W(p)-\widehat X_W(q),
\]

\[
R_{-E}=\widehat E_W(q)-\widehat E_W(p).
\]

直接代数化简得到对任意 `W,p,q` 都成立的恒等式

\[
\boxed{R_X+R_{-E}=W(p)-W(q).}
\]

更重要的是它具有唯一性。一般线性组合 `a R_X+b R_-E` 中，人工坐标位移 `z-z'`
的系数为 `a-b`。因此若要求 correction 不直接依赖我们为了 probe 而人为选择的坐标
位移，则必须且只需

\[
a=b.
\]

所以除整体尺度外，

\[
\boxed{W(p)-W(q)}
\]

是唯一的线性 endpoint revision invariant。它不是任意 material derivative，也不是从
FID 网格中挑出来的方向。

已有同协议 FID-1K 因果实验符合这一唯一性预测：

| correction | FID-1K |
|---|---:|
| 完整 invariant contrast `W(p)-W(q)` | **61.9093** |
| clean revision | 63.6491 |
| clean revision，RMS matched | 63.5848 |
| negative-noise revision | 64.8581 |
| negative-noise revision，RMS matched | 66.0669 |

在 `R_X+lambda R_-E` 的扫描中，理论点 `lambda=1` 也位于经验最低区域；偏离两侧均
恶化。该结果说明收益不能由最大能量的 clean component 单独解释。

## 3.1 为什么扩散时间严格是一条 information axis

对任意 `t>0`，把 latent 做可逆缩放

\[
Y_t=Z_t/t=X+\sigma(t)E,
\qquad
\sigma(t)=\frac{1-t}{t}.
\]

当 `t_2>t_1` 时，`sigma(t_2)<sigma(t_1)`，并且可以构造独立高斯噪声 `xi` 使

\[
Y_{t_1}
=Y_{t_2}
+\sqrt{\sigma(t_1)^2-\sigma(t_2)^2}\,\xi.
\]

所以较小 `t` 的 observation 可以由较大 `t` 的 observation 再加噪得到。按 Blackwell
ordering，`t_2` 对 clean endpoint `X` 至少和 `t_1` 一样有信息；对任意 Bayes decision
problem，更高 `t` 的最优 Bayes risk 不会更差。

这使“information time”成为严格的统计概念，而不是视觉上“晚一点更清楚”的比喻。
不过，对一个指定的当前样本，`(z,t+h)` 只是更高信息 channel 下的 counterfactual
observation，不是该样本真实 forward characteristic 上的未来点。

## 4. Internal hierarchy 唯一决定已有 correction 的尺度

记当前完整模型和 depth-4 internal head 为

\[
S=S(z,t),\qquad W=W(z,t),
\]

并令

\[
\Delta=S-W,\qquad \beta=1+\gamma.
\]

普通 Internal Guidance 精确等于

\[
G=S+\gamma(S-W)=W+\beta\Delta.
\]

因此浅层 `W` 是 internal base，`beta Delta` 是网络 suffix 提供并被 IG 放大的
refinement。若遵守 **no-new-gain principle**，即只运输已有 refinement/reference、
不再引入一个独立修正强度，那么 future revision 的系数必须沿用 `beta`。

这给出

\[
\boxed{G_{\rm new}=G+\beta[W(p)-W(q)]}
\]

或等价地

\[
\boxed{G_{\rm new}=W(p)+\beta[S(p)-W(q)].}
\]

weak-base 与 strong-base 并非纯代数等价的语义选择。已有 FID-1K 对照为：

| anchor | FID-1K |
|---|---:|
| internal weak base，系数 `beta` | **61.9244** |
| strong base，系数 `gamma` | 63.5911 |

这支持 internal computational hierarchy，而不能自动推广为任意两个独立模型的定理。

## 5. Query 不是未来状态，而是最小空间干预的信息探针

需要让 weak head 在更高 information time `t+h` 重新读取 refinement。定义当前
calibration

\[
C=\beta(S-W).
\]

为避免在 latent 中任意发明一个方向，只允许空间 probe 位于当前 deployed field `G`
的正向射线，并选择最接近 calibration 的点：

\[
\boxed{
a^*=\arg\min_{a\ge0}\|aG-C\|^2
=\left[\frac{\langle C,G\rangle}{\|G\|^2}\right]_+.
}
\]

最终 query 为

\[
\boxed{q=(z+h a^*G,t+h).}
\]

必须准确理解这个式子：当 `a*` 不等于 1 时，它不是同一 characteristic 上的 Euler
future。空间与 information time 被有意解耦；前者只负责把 refinement 的可表达部分
写入 latent，后者负责询问更高 SNR 下的 internal belief。

这个投影还有一个直接的 intervention 解释。当前场已经分成

\[
G=\underbrace{W}_{\text{base transport}}
+\underbrace{C}_{\text{deep refinement}}.
\]

完整 characteristic `z+hG` 同时把 base transport 和 refinement 写入 query，因此无法
判断 weak belief 的变化究竟来自它自己的基础运动，还是来自 deep suffix 提供的新信息。
`a*G` 是在当前可用 forward direction 中对 `C` 的最小失真表示；它只编码总运动中可
归因于 refinement 的部分。于是 query 近似回答：

> 如果只把 deep refinement 的可实现部分作为 intervention 写进 latent，并给 weak head
> 一个更有信息的 observation level，它会怎样修订 reference？

这不是统计因果识别定理，但它是由网络计算图给出的明确 structural intervention；四种
query 对照可以直接证伪该语义。

正式 FID-5K 运行中 `a*` 的均值为 `0.1318`，最大值约 `0.875`。它明确反对“主要收益
来自更准确 future integration”的解释，并支持“以时间为信息干预、以空间为小幅
calibration probe”的解释。

## 6. 三条原则唯一推出当前算法

当前方法可以由三个彼此独立、可证伪的原则推出：

1. **Endpoint-coordinate invariance**：probe 不应把人为坐标位移直接混入 belief
   revision；这唯一给出 `W(p)-W(q)`。
2. **No-new-gain hierarchy**：只运输普通 IG 已有的 suffix refinement；这固定系数为
   `beta=1+gamma`。
3. **Minimum spatial intervention**：在当前 forward ray 中以最小欧氏失真编码
   calibration；这闭式确定 `a*`。

于是

\[
\boxed{
\begin{aligned}
G&=W+\beta(S-W),\\
C&=\beta(S-W),\\
a^*&=\left[\langle C,G\rangle/\|G\|^2\right]_+,\\
q&=(z+h a^*G,t+h),\\
G_{\rm ITPR}&=G+\beta[W(z,t)-W(q)].
\end{aligned}
}
\]

暂称 **Information-Time Posterior Revision IG (ITPR-IG)**。代码中的
`weak_calibration_projected` 与这组公式完全一致；名称改变不代表偷偷改变方法。

## 7. 当前质量结果

ImageNet-100、SiT-S/2、同一 depth-4 head、EMA、FID-5K：

| method | FID | sFID | IS |
|---|---:|---:|---:|
| depth-4 IG | 39.9468 | 69.9252 | 36.8522 |
| 旧三参数 oblique 方法 | 36.5003 | **69.0649** | 40.0216 |
| ITPR-IG / 当前 PFR 实现 | **36.3470** | 70.2208 | **40.6690** |

两个高度重叠的历史 nominal bank 的 FID 为 `36.35096/36.34310`。这是当前协议的 FID/IS 最优，
不是 sFID 最优，也不是跨模型 benchmark SOTA。

## 8. 预注册的语义判别实验

固定所有训练、guidance、horizon 与 coefficient，只改变 query：

1. `time-only`: `(z,t+h)`；
2. `oblique`: `(z+h a*G,t+h)`，当前方法；
3. `spacetime-coupled`: `(z+h a*G,t+h a*)`；
4. `characteristic`: `(z+hG,t+h)`。

若这是隐式积分/真实未来机制，3 或 4 应优于 2。若这是 information-time posterior
probe，则 1 应保留显著收益，2 应在加入最小 calibration-aware spatial information 后
最好，而 3 会因 `a*` 很小而同时削弱最关键的 information-time advance。

该实验在查看结果前写入本报告。

### 8.1 两个高度重叠的历史 nominal bank

| query | seed 0 FID | seed 1 FID | mean FID |
|---|---:|---:|---:|
| `oblique`，当前方法 | **61.9244** | **61.8369** | **61.8807** |
| `time-only` | 62.5605 | 62.4532 | 62.5069 |
| `spacetime-coupled` | 64.4475 | 64.3128 | 64.3802 |
| `characteristic` | 64.8948 | 64.9809 | 64.9379 |

每个 bank 内的 paired 排序都一致；但两个 bank 共享 992/1000 个样本，这不构成独立
复现。`time-only` 相对普通 depth-4 IG 保留显著收益，加入 refinement-aligned 的小空间
intervention 后再改善约 `0.63 FID`；反过来，把 time advance 也乘以很小的 `a*`，收益
大部分消失。最接近真实 Euler future 的 characteristic query 甚至略差于普通 IG。

该结果直接否定“更准确预测未来状态/隐式积分”是主要机制，并支持预注册的
information-time interpretation。三项 ADM 指标并非完全同序：例如 coupled query 的
sFID/IS 也明显较差；因此结论不依赖只挑 FID 排名。

## 9. 与 semigroup-consistent density theory 的边界

上述 endpoint-invariance 定理对任意 finite `W` 成立，但它只决定 correction 的形式，
不保证 FID 下降。更强的 population 理论是先指定 clean endpoint power tilt

\[
\pi_0\propto p_0^\beta q_0^{1-\beta},
\]

再由同一个 noising semigroup 唯一推出整条 score path。普通 score extrapolation 缺少
一个 conditional-Jensen correction。其最低阶 SiT velocity 形式为

\[
\beta(\beta-1)t(1-t)J_{S-W}^{\mathsf T}(S-W).
\]

该方向没有新增 scale。真实 SiT FID-1K 结果为 `75.8041`，相对普通 depth-4 IG 的
约 `64.85` 明显恶化；sFID/IS 也同时恶化到 `220.90/31.86`。因此 internal weak head
不能在当前证据下被宣称为合法 strong/weak density pair，最低阶 population correction
也没有穿过 finite-model error。该失败还可能来自高噪声区的 local heat-time truncation，
所以它不是对完整 semigroup 理论的反证；但已经足以停止用这条未验证的 population
故事解释 ITPR-IG。

## 10. 当前尚未解决的边界

1. `h` 仍是 information-time probe radius；当前取 `1/32`，尚未由质量定理唯一决定。
2. endpoint-coordinate invariance 决定 correction 形式，不决定目标数据分布。
3. `a*` 的欧氏 metric 可能不是唯一合理的 latent metric；当前只证明它是给定 metric 下
   的闭式最小失真解。
4. 仍需更大模型、precision/recall 与正式 FID-50K，才能把内部最好结果升级为 SOTA。

## 11. 理论、设计与证据的严格分层

为了避免用一个好听的故事替代证明，当前结论应分成四层。

### 11.1 两个精确结论

第一，对任意有限网络、任意两个 query，

\[
R_X+R_{-E}=W(p)-W(q)
\]

严格成立；在所有 endpoint-revision 线性组合中，它也是唯一消除人为 query 坐标位移的
方向（差一个整体常数）。这不依赖 Bayes 最优、连续极限或小步长近似。

第二，在线性高斯 bridge 下，较大 `t` 对 clean endpoint 构成 Blackwell 意义下更有信息的
observation channel。这是分布层面的 information ordering；它不声称任意固定 `(z,t+h)`
都是同一样本的真实未来 observation。

### 11.2 一个闭式优化结论

给定“空间 probe 只能位于当前 deployed field 的正向射线”与欧氏 latent metric，

\[
a^*=[\langle C,G\rangle/\|G\|^2]_+
\]

是编码 suffix refinement `C` 的唯一最小二乘解。该结论固定了空间 intervention，但“选择
这条射线和这个 metric”仍是结构设计，而不是数据分布定理。

### 11.3 一个明确的设计公理

`no-new-gain hierarchy` 要求新方法只重读普通 IG 已有的 suffix refinement，不再为
revision 引入第二个自由强度，因此复用 `beta=1+gamma`。它让方法从多参数搜索缩减为一个
information radius `h`，但它是可证伪的设计原则，不是假装成数学必然性。

### 11.4 一个被预注册消融支持的经验命题

真正反直觉的经验发现不是“future query 有用”，而是：

> **越像真实数值未来的 query，效果反而越差。真正有用的是在时间上进入更有信息的
> channel，同时只在空间上写入最小量的 deep-refinement evidence。**

如果方法来自隐式积分误差修正，characteristic query 应最好；实际它在两个历史 nominal banks
都最差。若空间和时间共享同一个很小的投影系数，主要收益也消失；纯 time-only 已保留大部
分收益，而加入最小 refinement-aware 空间 intervention 后达到最优。该排序正是
information-time hypothesis 在看结果前给出的预测。

因此当前完整故事不是“更准确地预测未来”，而是：

\[
\boxed{
\text{内部深层 refinement}
\to
\text{最小证据干预}
\to
\text{更高信息层级的 weak belief revision}
\to
\text{坐标不变的 endpoint correction}
}
\]

这个链条解释并预测了方法结构；它尚未证明 correction 必然降低真实分布距离。后者目前由
配对 FID/sFID/IS 实验支持，并保留第 10 节所列边界。

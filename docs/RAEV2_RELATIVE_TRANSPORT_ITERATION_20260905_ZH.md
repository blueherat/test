# RAEv2 Relative Transport Iteration：有限 Flow-Map 外推与严格对照

记录日期：2026-09-05。

本实验检验一个不依赖 score conservativity 的候选：不再把 finite internal-head gap
解释成 density-ratio score，而是直接在两个模型实际定义的离散 flow map 上做一次
relative transport iteration。它没有无条件质量定理；实验的目的正是判断 map 复合是否
提供超出普通 guidance 加倍的高阶收益。

## 1. 定义

在 RAEv2 的反向噪声坐标上，只看 guidance-active 前缀。令

\[
R:z_1\mapsto z_s^R
\]

为 full-head 的离散 Euler map，令

\[
G:z_1\mapsto z_s^G
\]

为同一网格上 `beta=1.78` 的 piecewise-IG map。二者使用同一个 frozen checkpoint、
同一个初始 noise 和同一个 class label。relative map 定义为

\[
T_{R\to G}=G\circ R^{-1}.
\]

它满足一个精确但有限的事实：

\[
T_{R\to G}\circ R=G.
\]

因此，如果输入确实来自 reference map 的像，`T` 会把它搬到 guided map 的对应像。
本实验把同一个 `T` 再作用到已经 guided 的状态：

\[
\boxed{
z_s^{\mathrm{relative}}
=G\circ R^{-1}\circ G(z_1).
}
\]

这一步不是上述恒等式的推论，也没有保证更接近真实数据分布。它只是一个无新增连续
gain 的有限 map iteration。

## 2. 为什么必须与普通加倍比较

写作

\[
G=R+\varepsilon H.
\]

对可逆且光滑的 `R`，在一阶有

\[
R^{-1}(G(z))
=z+\varepsilon [D R(z)]^{-1}H(z)+O(\varepsilon^2).
\]

再作用一次 `G`：

\[
G(R^{-1}(G(z)))
=R(z)+2\varepsilon H(z)+O(\varepsilon^2)
=2G(z)-R(z)+O(\varepsilon^2).
\]

所以 relative iteration 的一阶部分就是把原始 map displacement 加倍。对于
`beta=1.78`，局部 guidance 系数加倍对应

\[
\beta_{\mathrm{double}}
=1+2(1.78-1)=2.56.
\]

因此只有 relative iteration 相对以下两个 control 的差异，才可能属于 map 复合带来的
高阶信息：

1. `local_double`：直接用 `beta=2.56` 重跑前缀；
2. `switch_linear`：在 switch state 上使用 `2G(z)-R(z)`。

若 relative 与二者相当，不能把结果解释成新的 transport 原理；若更差，则说明有限
复合的高阶项有害。

## 3. 数值逆与 cycle controls

`R` 是 89 个显式 Euler step 的离散复合。每一步

\[
y=x+\Delta t\,R_t(x)
\]

通过 fixed-point equation

\[
x=y-\Delta t\,R_t(x)
\]

逆解。`inverse_tolerance=1e-3` 只是一项 BF16 数值容差，不是科学自由参数；若任一步
在 16 次迭代内不收敛，整个 run 直接失败。

两条 cycle control 与候选共享相同数值路径：

\[
\mathrm{cycle\_null}=R\circ R^{-1}\circ R(z_1),
\]

\[
\mathrm{cycle\_guided}=R\circ R^{-1}\circ G(z_1).
\]

它们分别应复现 `full` 与 `piecewise_ig`。若 cycle control 的 FID 已显著漂移，就不能把
relative 分支的差异归因于 transport iteration。

## 4. 配对协议

- RAEv2 DINOv3-L K7，step `100080` EMA；
- 官方 stage-1 decoder 与 normalization statistics；
- shifted 100-step Euler，实际 switch time `0.4971751571`；
- 前缀 89 steps，后缀 11 steps；
- 1000 类各一张，sampling seed `20260903`，原始 RNG batch size `8`；
- 四个 shard 只按完整原始 batch index 分工；每个 worker 仍消费并哈希全部 RNG batch；
- BF16 autocast，所有 clean/velocity/map arithmetic 为 float32；
- 七个分支共享 noise、labels、checkpoint、decoder、后缀 full-head flow 和官方 evaluator。

分支为：`full`、`piecewise_ig`、`local_double`、`switch_linear`、
`relative_iterate`、`cycle_null`、`cycle_guided`。

## 5. 预注册裁决

只有 `relative_iterate` 同时满足以下条件，才值得继续：

1. 超过配对的 strongest `piecewise_ig`；
2. 超过两个一阶 scale-doubling controls；
3. 提升不能由 cycle drift 解释；
4. FID 与 IS 不出现明显相反方向；
5. 1K 先通过后才考虑更大样本，不在失败后追加自由 scale。

这个门槛把“map 复合可定义”与“map 复合能提高生成质量”严格分开。

## 6. 正式配对 1K 结果

四个 shard 完整生成 `1000` 个样本；所有分支共享同一批 initial noise、labels、
checkpoint、decoder 和 evaluator。合并产物的 noise/label SHA256 分别为
`dd51673a5f594ecab78ea3433786fbb36a71bbb006d5873f4348e3718b03a8c6` 与
`702746827e553786bb026ac120cb58745fef3d3f554c33891809001cc37639f0`。

| condition | FID-1K | IS |
|---|---:|---:|
| full | 38.767251 | 55.087851 |
| piecewise IG, `beta=1.78` | **38.126650** | 58.991963 |
| local double, `beta=2.56` | 38.638198 | **60.247493** |
| switch linear, `2G-R` | 38.705508 | 57.338654 |
| relative iterate, `G o R^-1 o G` | 38.828115 | 58.144827 |
| cycle null | 38.701944 | 54.949987 |
| cycle guided | 38.087313 | 59.037864 |

relative iteration 相对 strongest `piecewise_ig` 恶化 `0.701466` FID；它也分别比
`local_double` 和 `switch_linear` 差 `0.189918/0.122607`。因此结果不支持“有限
flow-map 复合的高阶项优于普通 guidance 加倍”。

## 7. 数值逆与 cycle 审计

正式运行中，逐步 fixed-point inverse 最多使用 `3` 次迭代，平均/最大逆解残差为
`0.0009555/0.0009617`。nominal switch displacement RMS 为 `0.227674`，relative
second increment RMS 为 `0.143250`。cycle 误差相对这些信号较小：

- null cycle 的 switch-state RMS 为 `0.033908`；
- guided cycle 的 switch-state RMS 为 `0.010209`；
- guided cycle error / nominal signal 为 `0.042319`；
- guided cycle error / second increment 为 `0.059708`。

更直接地，`cycle_guided` 相对 `piecewise_ig` 的 FID 只变化 `-0.039337`，远小于
relative iteration 的 `+0.701466` 恶化。数值逆误差因此不足以解释候选失败。

## 8. 裁决

该构造未通过预注册门槛：它没有超过 strongest baseline，没有超过任一一阶对照，
FID 与 IS 也都落后于 `piecewise_ig`。正式结论为**阴性**；不追加自由 scale，不做
5K，也不把 `T(R(z))=G(z)` 的精确恒等式误写成 `T(G(z))` 的质量保证。

机器可读结果归档在
`docs/data/raev2_guidance_exploration_20260905/raw/raev2_relative_transport_20260905/`。

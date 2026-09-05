# RAEv2 Guidance 理论探索归档：从 PFR 迁移到半群 Value

日期：2026-09-05

状态：**阶段性收束。** 本报告记录 `HEAD=91f87b0` 之后围绕 SiT PFR 理论、
RAEv2 跨模型迁移、强基线校准、后验/投影结构、Feynman--Kac 可行性和
semigroup-consistent value 所做的完整实验链。报告区分解析恒等式、机制相关性、
配对 1K 筛查和正式 5K 结论；没有通过相应证据门槛的结果均明确记为阴性或未完成。

机器可读数据位于：

- `docs/data/raev2_guidance_exploration_20260905/`；
- SiT PFR 反事实残差：`docs/data/pfr_counterfactual_residual_theory_20260903/`；
- SiT OU 概率小波：`docs/data/pfr_ou_probability_wavelet_20260904/`；
- RAEv2 PFR 正式迁移：`docs/data/raev2_pfr_ou_transfer_20260904/`。

## 1. 本轮最短结论

本轮得到的结论不是“找到一个可直接迁移到 RAEv2 的新 guidance”，而是五条边界：

1. **SiT 上的 PFR/OU certificate 是真实且强的配对结果。** 在两个独立 5K bank 上，
   strong OU direction + raw norm 相对 ordinary IG 的 FID 改善为 `11.54%/9.82%`。
2. **同一构造不跨 representation 自动成立。** RAEv2 raw PFR 的两个 1K bank 均为正，
   但正式 5K 从 ordinary IG `7.034546` 恶化到 `7.224213`；OU 投影也未能救回。
3. **目前唯一稳定的 RAEv2 轻量正信号是更强的时间窗基线。** 只在反向时间
   `t>0.5` 使用原系数 `1.78`，两个配对 1K bank 分别由 `38.4235/38.1755`
   改为 `38.1974/37.9748`。幅度约 `0.5%`，尚没有 5K 结论，更没有达到 5% 目标。
4. **新增的理论驱动修正均未通过配对 1K。** semigroup value 将 FID 从
   `39.3702` 恶化到 `40.6671`；future flow pullback 从 `38.7351` 恶化到
   `39.2757`；零新增强度的 radial 分量在两个 bank 中一次改善 `0.0983`、一次
   恶化 `0.0848`，发生符号翻转。
5. **有限 relative flow-map 复合也没有产生额外价值。** `G o R^-1 o G` 的 FID
   为 `38.8281`，落后 `piecewise IG=38.1266`，也落后两个一阶加倍对照；cycle
   drift 只有 `-0.0393` FID，不能解释候选的 `+0.7015` 恶化。

后续的 posterior projection、consensus、orthogonal innovation、characteristic
foresight、full-dimensional FKC、semigroup value、radius decomposition 与 flow
pullback 都没有形成可放行的 RAEv2 质量结果。它们的价值是排除了多条看似优雅但
不充分的理论捷径。

## 2. 证据等级与协议边界

本报告使用四种标签：

| 标签 | 含义 |
|---|---|
| `exact` | 解析恒等式或可解分布中数值误差内成立 |
| `mechanism` | teacher/trajectory/geometry 审计，不直接代表终点质量 |
| `screen-1K` | 同噪声、同标签的配对 1000 样本筛查，可能发生选择反转 |
| `formal-5K` | 同一 evaluator/reference 的配对 5000 样本结果 |

不同目录、batch size、world size 或算术路径的绝对 FID 不直接横比。历史相邻 seed
曾因旧 `seed + batch_index` 规则高度重叠；本轮新增代码采用 namespaced batch RNG，
旧结果只保留其 within-bank 配对含义。

## 3. SiT：PFR 从现象到可证伪理论

### 3.1 反事实 weak-reference residual

PFR 不是更准确的未来状态估计，也不是高阶积分器。在线性 flow 中 raw velocity
编码同一个守恒端点差，因此跨时间 weak output 可以作为 finite counterfactual
reference。解析反例进一步证明：一个修正可以让 time-integrated local MSE 从 `1`
恶化到约 `4.03`，同时把终点 Gaussian Fréchet/Wasserstein 误差从 `1` 降到 `0`。

不重叠 5K bank 上，ordinary IG/PFR 为 `40.7983/37.6459`，改善 `3.1524`；
FID mean/covariance 分量分别贡献 `1.9223/1.2301`，但 sFID 反向变化，且 PFR
多约 `30.4%` full-forward-equivalent。它是强 paired 证据，不是全指标支配或公开
benchmark SOTA。

详见 `docs/PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`。

### 3.2 指数重定时

Gaussian-centered odds 坐标给出唯一 affine retiming operator。PFR 可以精确分解为
same-time depth contrast 与 probability-path e-geodesic defect。把两个大项拆开后，
FID-1K 分别恶化到 `170.624/271.154`，正确配对重组则为 `61.969`；这说明有用对象是
配对差，不是某个分项的大范数。

等计算 5K 中 ordinary Heun-32、PFR Heun-27、stage-reuse 的 FID 分别为
`40.912/37.530/37.626`。额外 query 或积分精度不能解释主要收益。

详见 `docs/PFR_EXPONENTIAL_RETIMING_THEORY_ZH.md`。

### 3.3 OU probability wavelet

linear flow 可精确重写为 OU channel；relative score 的跨尺度条件期望恒等式与
degree-1 消去器是精确结果。由此构造的 strong certificate + raw norm 在两个独立
5K bank 得到 `36.1902/35.7588`，对应 ordinary IG 的 `40.9122/39.6538`。
方向/幅值交换说明主要判别信息来自证书方向，而不是简单缩小 correction。

这是本轮最完整的 SiT 正结果，但 RAEv2 反例证明它不是跨 representation 的充分不变量。
详见 `docs/PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md`。

## 4. RAEv2：PFR 迁移与补救变体

### 4.1 raw PFR

两个配对 1K bank 都显示小幅改善：

| seed | ordinary IG | raw PFR | 差值 |
|---:|---:|---:|---:|
| 20260903 | 38.9193 | 38.4169 | -0.5024 |
| 20260904 | 38.3974 | 37.8900 | -0.5074 |

但正式 5K 为：

| condition | FID | IS |
|---|---:|---:|
| ordinary IG | **7.034546** | 154.5394 |
| raw PFR | 7.224213 | **167.8552** |

因此这是一例清楚的 1K 选择反转和质量/覆盖折衷，正式裁决为阴性。

### 4.2 补救构造

随后依次测试 shared retiming、potential-preserving/angular、information-matched、
OU common/polar、weak certificate、anchor consistency、first-half 限制和 bridge
counterfactual。它们没有任何一个形成“双 bank 后再通过 5K”的结果：

| 构造 | 代表性配对结果 | 裁决 |
|---|---:|---|
| shared retiming, mild | `38.4872 -> 38.0412` | 单 bank 正信号 |
| shared retiming, `rho=1` | `38.4872 -> 50.4565` | 强度崩溃 |
| information-matched | `38.4854 -> 38.9559` | 阴性 |
| angular/potential-preserving | `38.4872 -> 39.2790/39.2953` | 阴性 |
| OU common/polar，双 bank 均值 | raw PFR `38.1535`，OU `38.3600/38.3290` | 均落后 raw |
| first-half boundary | `52.9043/54.4061` | 明确失败 |
| direct foresight weak reference | `39.2135 -> 42.3453`、`39.1415 -> 42.6729` | 双 bank 失败 |

这组实验排除了“只要让跨时刻 operator 更保守/一致，就会改善 FID”的强说法。

## 5. 先把最强 RAEv2 基线校准清楚

对 high-noise/medium/low-noise 三段做配对扫描后，原 constant `1.78` 并非最强 1K
基线。关闭数据侧 `t<=0.5` 的 guidance 更好：

| seed | constant `1.78` | `early=1.78, middle=1.78, late=off` |
|---:|---:|---:|
| 20260903 | 38.423521 | **38.197351** |
| 20260904 | 38.175549 | **37.974836** |
| mean | 38.299535 | **38.086094** |

改进只有约 `0.56%`，但两个 bank 同符号。后续 RAEv2 新方法必须与这条基线比较，
不能把一个隐式时间调度包装成新的机制。

within-time 审计显示 token-local gap 在相邻步高度持久（典型 cosine `0.99`），但跨较长
区间会旋转（例如 `t=.8` 的跨区间 persistence 约 `0.38`）。因此 gap 既不是白噪声，
也不是一条全程固定方向。

## 6. Posterior / projection 路线

### 6.1 几何审计

teacher-forced projection hierarchy 中 residual-gap correlation 均值约 `-0.0110`，
MSE-optimal scale 均值约 `0.9576`。这说明 base/full 具有近似正交投影结构，但只是一阶
回归几何。

### 6.2 生成反事实

| condition | FID-1K |
|---|---:|
| ordinary | **39.3242** |
| tangent | 40.4887 |
| posterior I-projection | 45.7511 |
| reflected | 39.6328 |

结论很明确：projection-like teacher geometry 成立，不推出把同一几何写进闭环 sampler
会改善分布；I-projection 方法被直接否定。

## 7. 条件结构、consensus 与 innovation

将 depth 与 class conditioning 分解后，interaction guidance 在两个 bank 一次改善
`0.3970`、一次恶化 `1.2385`，不复现。minimum-norm Bayes consensus 在两个 bank
都比 ordinary 差约 `0.15`；midpoint 也稳定更差。

orthogonal innovation 的正号、反号和 donor control 在两个 seed 中发生符号翻转，
没有得到“正确符号”或 sample/class specificity 的稳定证据。这排除了通过单次几何
投影就能识别 useful innovation 的简单理论。

## 8. Characteristic 与 fixed-point 风格查询

characteristic reference 相对 ordinary 一次改善 `0.5285`，另一次恶化 `0.0761`；
直接 foresight weak reference 在两个 bank 都恶化约 `3.1--3.5` FID。Foresight
Fixed Point 提供了“先定义跨时间一致对象，再推导 operator”的研究品味，但它在 CFG
中的质量语义不能直接搬到 RAEv2 IG。更 self-consistent 的 query 不是质量定理。

## 9. Feynman--Kac 粒子权重审计

对 full-dimensional RAE latent 直接使用 plug-in FKC 增量权重时，权重快速退化：

| setting | ESS |
|---|---:|
| BF16, K=16, `t≈.95` | 1.632 / 16 |
| BF16, K=16, `t≈.90` | 1.023 / 16 |
| BF16, K=16, switch | 1.000 / 16 |
| FP32, K=8, `t≈.90` | 1.002 / 8 |

K=16 的累计 log-weight range 均值/最大值约 `2501/3883`。FP32 复核排除了精度
假象。这只否定当前 full-dimensional、未裁剪 plug-in 路径，不等于证明所有 FKC/SMC
设计都不可能；但它足以停止把该路径当作低成本 RAEv2 解法。

## 10. Semigroup-consistent value

### 10.1 精确理论对象

在 exact weak/strong endpoint densities \(q,p\) 下，令 \(r=p/q\)、
\(\beta=1+\gamma\)。端点 power tilt 为

\[
\pi_0(x)\propto q_0(x)r(x)^\beta.
\]

其合法 noisy path 的缺失修正势为

\[
\delta(y)=\log E_q[r^\beta\mid y]-\beta\log E_q[r\mid y]
          =(\beta-1)D_\beta(P(X\mid y)\|Q(X\mid y)).
\]

这说明 ordinary geometric guidance 丢掉的是 hidden endpoint explanations 的条件高阶矩。
对应的 (h=\exp\delta) 满足 tower/semigroup 一致性，也可写成 HJB/Feynman--Kac
value equation。该恒等式本身已与 CFG Gibbs-like guidance、FKC 和 generalized
h-transform 文献相邻，不能作为新的宽泛贡献。

### 10.2 解析 toy

同初始样本下：

| method | W1 | W2 |
|---|---:|---:|
| static extrapolation | 0.037459 | 0.043640 |
| local-risk correction | 0.016156 | 0.019532 |
| learned soft-Bellman | 0.002139 | 0.003977 |
| exact semigroup path | **0.002110** | **0.002198** |

解析 score identity 误差约 `1e-6`。这证明目标对象和 Bellman 聚合在可解分布中成立。

### 10.3 RAEv2 balanced value

第一版 bank 只覆盖 128 类，正式 screen 无效并废弃。第二版使用独立、均衡覆盖全部
1000 类的 switch bank，训练 4000 step；loss 从约 `4.5e-6` 降到 `4e-8--2e-7`。
训练收敛不等于理论前提成立，因为 finite full/depth fields 并不保证是同一 OU
semigroup 下的合法 conservative scores。

held-out 1000 类审计给出：

| noise time | Bellman residual / target increment | correction / gap | true/permuted-label gradient cosine |
|---:|---:|---:|---:|
| .60 | .0747 | .330 | .9780 |
| .70 | .8237 | .827 | .9783 |
| .80 | 2.490 | 1.298 | .9783 |
| .90 | 6.594 | 1.634 | .9786 |
| .95 | 14.318 | 1.925 | .9786 |
| .98 | 33.762 | 2.668 | .9779 |
| .99 | 51.342 | 3.781 | .9772 |

它数值有限、边界 `t=.5` 修正确为零，但在高噪声区不再满足自己的 held-out
Bellman increment，并且梯度主要是 class-agnostic。当前实现因此只是 plug-in controller，
不能当成 posterior Rényi correction 的可靠近似。

同 seed、同 batch size 的正式配对 1K 结果为：

| condition | FID-1K | IS |
|---|---:|---:|
| piecewise IG | **39.370218** | **57.111010** |
| semigroup value | 40.667069 | 55.720657 |

两路 initial/final generator、first noise 与 first labels 的 SHA256 全部一致。value
correction 的平均 RMS 为 `0.193813`，相对 ordinary clean RMS 的均值约 `0.2881`；
但 FID 恶化 `1.296851`，IS 同时下降。它没有超过 strongest piecewise IG，因此按
预注册规则停止，不增加自由 scale，也不做 5K。

## 11. 半径/方向筛查

另一个预注册的零新增强度对照，将 ordinary clean prediction 相对 full prediction
分成 radial/tangential 分量，并测试保持 full token norm 的 retraction。它与 APG 相邻，
只检验“ordinary IG 的径向变化是否有害”，不声称数据位于球面。代码与协议见
`docs/RAEV2_RADIUS_DIRECTION_PLAN_20260905_ZH.md`。

首个配对 bank 中，ordinary/radial/retracted/tangent 的 FID-1K 分别为
`38.735080/38.636796/39.689400/39.449084`。radial 只改善 `0.098283`
（`0.254%`），且 IS 更低。独立确认 bank 中，ordinary/radial 为
`38.773441/38.858200`，符号反转为恶化 `0.084759`。因此 radial 没有可复现收益，
retraction 与 tangent 在首个 bank 已明显失败；整条构造不追加 5K。

## 12. Future query 与 flow pullback

另一个候选先把 early-time future gap 当作 covector，再通过 short full-flow 的 input VJP
拉回当前状态，并逐样本匹配 ordinary gap RMS。FP32 有限差分与 `K=4/8` 一致性支持
短区间 pullback 数值实现；它并不证明方向具有终点质量语义。

严格配对的 1K 结果为：

| condition | FID-1K | IS |
|---|---:|---:|
| ordinary | **38.735080** | **57.284344** |
| raw future | 38.864229 | 57.000472 |
| pullback | 39.275690 | 56.718486 |

raw future 与 pullback 分别相对 ordinary 恶化 `0.129149/0.540610` FID；pullback
相对 raw future 也更差 `0.411461`。因此 Jacobian-transpose transport 虽有清楚的局部
优化定义，却不是当前 RAEv2 上足够的质量原则。详见
`docs/RAEV2_FLOW_PULLBACK_20260905_ZH.md`。

## 13. 被排除的捷径

这轮实验共同排除了以下推理：

1. local MSE 更低，所以 rollout 一定更好；
2. vector cosine/energy 更漂亮，所以该分量更有效；
3. 更接近 characteristic、fixed point 或 semigroup，所以 FID 一定更低；
4. 某种 OU/score 恒等式在 exact density 成立，所以 finite internal head 自动满足前提；
5. 两个 1K bank 同符号，所以 5K 必然保持；
6. SiT 上的有效 certificate 可以不经验证迁移到 RAEv2 latent。

这些不是“理论没有价值”，而是要求理论必须同时说明：对象是否可识别、有限网络是否满足
前提、以及终点质量为何受益。只证明第一项不够。

## 14. Relative flow-map extrapolation

当前 semigroup 路线最根本的困难，是把有限的 non-conservative `S-W` 当作 density-ratio
score。一个避开该假设的候选，是直接使用两个模型在 guidance-active 前缀上定义的
离散 flow maps：

\[
R:z_1\mapsto z_s^R,
\qquad
G:z_1\mapsto z_s^G.
\]

其中 `R` 是 full-head reference map，`G` 是 `beta=1.78` 的 piecewise-IG map，
实际 switch time 为 `0.4971751571`。数值可逆范围内的相对 transport 为

\[
T_{R\to G}=G\circ R^{-1}.
\]

将同一个相对作用再应用到 guided switch state，并通过共享的 full-head 后缀 \(\Psi\)
生成终点：

\[
\boxed{
z_0^{\rm map}
=\Psi\circ G\circ R^{-1}\circ G(z_1).
}
\]

它的优点是：

- 不要求 `S-W` conservative，也不声明存在 density ratio；
- 外推发生在可逆映射的复合群中，不是逐点向量相加；
- reference/guided 前缀的有限 transport 被显式包含，和此前 replay 失败、frozen 成功的
  事实相容；
- 第一版只有“应用一次相对升级”这个离散选择，不做多参数扫描。

它仍没有无条件质量定理：`T` 只保证 `T(R(z))=G(z)`，不保证 `T(G(z))` 更好。其一阶
项又与 `beta=2.56` 或 switch-space `2G-R` 等价，因此只有超过这两个对照的部分才可能
属于 map 复合的高阶信息。

正式配对 1K 结果为：

| condition | FID-1K | IS |
|---|---:|---:|
| full | 38.767251 | 55.087851 |
| piecewise IG | **38.126650** | 58.991963 |
| local double | 38.638198 | **60.247493** |
| switch linear | 38.705508 | 57.338654 |
| relative iterate | 38.828115 | 58.144827 |
| cycle null | 38.701944 | 54.949987 |
| cycle guided | 38.087313 | 59.037864 |

relative iteration 比 `piecewise_ig` 恶化 `0.701466` FID，也没有超过两个一阶对照。
guided cycle 相对原 guided 分支只漂移 `-0.039337` FID；逆解平均残差约
`9.56e-4`，因此数值逆误差不足以解释失败。该候选按预注册规则停止，不追加 scale
或 5K。完整推导、遥测与裁决见
`docs/RAEV2_RELATIVE_TRANSPORT_ITERATION_20260905_ZH.md`。

## 15. 归档边界

Git 只保存源码、测试、报告、紧凑 CSV/JSON 和少量解释性图片。以下内容留在
`/home/zhoushunyu/data/eqvae/experiments/`，不提交：

- `samples.npz` 与逐卡生成图片；
- RAEv2 checkpoint；
- 约 500 MiB 的 training/held-out switch banks；
- 可由 manifest 与代码重建的临时 cache。

`docs/data/raev2_guidance_exploration_20260905/archive_manifest.csv` 记录被归档小文件的
来源、大小与 SHA256，便于以后从外部实验目录核对。

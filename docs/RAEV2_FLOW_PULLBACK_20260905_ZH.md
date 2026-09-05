# RAEv2 有限区间 Flow Pullback：理论、数值门槛与配对采样协议

记录日期：2026-09-05。

**当前结果是数值机制门槛与采样实现检查，不是生成质量改进。** 短区间
`h=1/32` 的 early-time pullback 提供了不同于 raw future gap 的方向；冻结
covector 的 VJP 通过缩小扰动后的有限差分检查。`K=4/8` 的总体方向一致性通过
预设的总体门槛，但最早两个时间点的分组中位数略低于门槛。长区间 `h=1/4`
在当前粗积分配置下未通过，不用于本轮质量采样。

本文随后完成了 `raw_future` 与 `pullback` 的配对 1K 质量筛查。结果均未超过
ordinary IG，且 pullback 比只查询 future gap 更差；该构造按预注册门槛停止，不追加
自由强度扫描或 5K。

## 1. 从已有机制选择研究对象

RAEv2 采用反向时间坐标，噪声在 `t=1`，数据在 `t=0`。完整预测器记为 `F`，
depth-8 base predictor 记为 `B`，二者均输出 clean latent。ordinary IG 为

\[
\widehat x_{IG}=F+(\beta-1)(F-B),\qquad \beta=1.78.
\]

已有单步脉冲、等能量窗口与 replay 实验显示：高噪声早期的微小 IG 扰动，经后续
full dynamics 可放大数百倍；相同注入能量下，最早窗口的终点影响约为最晚窗口
的 `368` 倍。在已测试的早期窗口、`gamma=0.05` 下，重新计算 IG gap 的额外
反馈仅约 `0.83%`，replay/recursive 方向 cosine 为 `0.99988`。这支持把
**full dynamics 如何传播一个方向**作为独立研究对象，而不是只研究局部 gap
的范数或重算反馈。[已有机制终验](INTERNAL_GUIDANCE_MECHANISM_PHASE1_RESULTS_ZH.md)

该证据并不证明官方 `gamma=0.78` 下反馈也可忽略。选择 full flow 是本轮有依据
的设计假设，不是已获证明的全部 guided dynamics 近似。

[Foresight Guidance 原文](https://arxiv.org/html/2510.21512v1) 先用跨区间的生成映射
一致性定义校准对象，再构造 fixed-point 方法。这里借鉴的是这个研究顺序：
先定义未来状态上的局部校准作用，再推导怎样将它传回当前状态。本文没有求解
FSG 的 golden-path 方程，也没有把 fixed point 收敛或局部校准作用等同于 FID 改善。

## 2. 冻结 future covector 与唯一的局部最优方向

令 \(\Phi_{t\to\tau}\) 为 frozen full model 定义的有限区间 Euler 映射，
\(\tau<t\)。当前 gap、未来状态与未来 gap 分别为

\[
g_t=F(z,t)-B(z,t),\qquad y=\Phi_{t\to\tau}(z),
\]

\[
q_\tau=\operatorname{stopgrad}\big[F(y,\tau)-B(y,\tau)\big].
\]

未来 gap 只在当前基点提供一个冻结的 covector。令 \(J=D\Phi_{t\to\tau}(z)\)，
pullback 为

\[
u_t=J^\top q_\tau.
\]

对当前的小扰动 \(\delta\)，定义未来局部线性作用

\[
\mathcal W_z(\delta)
=q_\tau^\top\big[\Phi(z+\delta)-\Phi(z)\big]
=q_\tau^\top J\delta+O(\|\delta\|^2).
\]

在 \(u_t\ne0\) 且当前扰动范数固定时，Cauchy–Schwarz 唯一给出

\[
\arg\max_{\|\delta\|\le\|g_t\|}q_\tau^\top J\delta
=\|g_t\|\frac{u_t}{\|u_t\|}.
\]

因此本轮两个候选只改变方向，均保持逐样本 current-gap 范数：

\[
\widetilde g_t^{raw}=\|g_t\|\frac{q_\tau}{\|q_\tau\|},\qquad
\widetilde g_t^{pullback}=\|g_t\|\frac{J^\top q_\tau}{\|J^\top q_\tau\|}.
\]

最终 clean prediction 为 \(F+(\beta-1)\widetilde g_t\)。退化零方向回退到
current gap，并记录 fallback，不让数值退化意外取消 ordinary guidance。

上述局部最优性针对 \(\mathcal W_z\)，**不针对 FID、真实数据密度或完整生成
分布风险**。冻结 \(q_\tau\) 很关键；若让梯度穿过 future gap，就会引入
\(Dq_\tau\) 的附加项，变成另一个目标。

RAEv2 的 gap 已表现出显著非保守性，因此不能直接称为某个全局 scalar potential
的梯度。但非闭合 differential 1-form 仍有合法的 pullback：

\[
q_\tau^\top dy=q_\tau^\top Jdz=(J^\top q_\tau)^\top dz.
\]

这个坐标变换恒等式不要求该 1-form 为恰当微分。本文只是把实际网络 gap 用作
局部、可能具有路径依赖性的校准参照，不额外假设存在全局 log-density ratio。

## 3. 与既有 RAEv2 查询的区别及实现边界

既有 pathwise PFR 沿 ordinary IG 做 Euler future query，再更换 weak reference。
posterior tangent 分支则在同一时间计算 full predictor 的正向 Jacobian action，
随后匹配 scalar progress。本轮在 future full flow 上计算的是转置 Jacobian action，
不同于这两者。

有限区间由 \(K\) 步组成时，\(J=A_{K-1}\cdots A_0\)，保留不同时间
Jacobian 的次序与非交换性。实现明确计算的是离散 Euler map 的导数；在积分
收敛之前，不能把它报告为精确连续 flow 的导数。

代码位于 [`raev2_flow_pullback.py`](../experiments/raev2_flow_pullback.py)。
它用固定 float32 time vector、RAEv2 的 `max(t,0.05)` 分母和非重入 activation
checkpoint。future gap 在无梯度上下文中计算，随后只执行一次关于 detached
current state 的 `autograd.grad`；不累积模型参数梯度。

[`test_raev2_flow_pullback.py`](../tests/test_raev2_flow_pullback.py) 的 CPU
验证为 `10 passed`，覆盖非正规且随时间变化的 affine map、非交换 Jacobian
顺序、有限差分、checkpoint 重算时的原始时间、frozen covector、逐样本范数、
inference/no_grad 上下文以及已有参数梯度不变。

## 4. 数值审计协议与逐时间结果

原始结果目录：

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_flow_pullback_20260905/
```

全部机制审计使用官方 step100080 EMA、`seed=202609053`、8 个样本、每个样本
3 个时间点、FP32、关闭 TF32。当前状态来自 ordinary IG `beta=1.78` rollout，
future map 使用 full predictor，比较均匀 raw-time Euler `K=4` 与 `K=8`。

有限差分在单位逐样本 RMS 的 pullback 方向 \(d\) 上计算

\[
\frac{\langle q_\tau,\Phi(z+\epsilon d)-\Phi(z-\epsilon d)\rangle}
{2\epsilon}
\quad\text{与}\quad
\langle u_t,d\rangle
\]

的相对差异。两边使用同一冻结 \(q_\tau\)。下表中位数沿用原始
`summary.json` 的 `torch.median` 口径，即偶数样本取下中位数；逐时间行均为
`n=8`，总体行为 `n=24`。

### 4.1 首轮短区间：`h=1/32, epsilon=1e-3`

来源：`audit_local_h1over32/{request.json,rows.csv,summary.json}`。

| actual noise time | pullback/raw cosine | K4/K8 cosine | 有限差分相对误差 |
|---:|---:|---:|---:|
| 0.969696999 | 0.783746804 | 0.995890563 | 0.065771433 |
| 0.900212348 | 0.966336543 | 0.999890313 | 0.001725171 |
| 0.747404814 | 0.997598808 | 0.999999307 | 0.000000421 |
| 总体 | 0.966336543 | 0.999890313 | 0.001307407 |

方向差异主要集中在高噪声侧；`t≈0.7474` 时短区间 pullback 与 raw gap 几乎
共线。`t≈0.9697` 的 `epsilon=1e-3` 相对误差超过 5%，因此不能只拿总体值
宣称该时间点已经通过线性检查。

### 4.2 最早区间的 epsilon 收敛复核

来源：`audit_early_eps3e4/` 和 `audit_early_eps1e4/`。两路共用相同状态、
future flow 与 VJP，只改变有限差分扰动大小，方向指标应一致。

| actual noise time | pullback/raw cosine | K4/K8 cosine | 误差 eps=3e-4 | 误差 eps=1e-4 |
|---:|---:|---:|---:|---:|
| 0.994818687 | 0.832506907 | 0.989030445 | 0.001439768 | 0.000155698 |
| 0.984785616 | 0.757867938 | 0.988180220 | 0.004104001 | 0.000566424 |
| 0.969696999 | 0.783746804 | 0.995890563 | 0.010334041 | 0.001279179 |
| 总体 | **0.796627684** | **0.993600889** | **0.007428293** | **0.000649456** |

总体门槛预设为：pullback/raw cosine `<0.99`，K4/K8 cosine `>0.99`，
有限差分相对误差 `<0.05`。缩小 epsilon 后，总体相对误差从 `0.0074283`
降到 `0.0006495`，支持实现计算了预期的局部 VJP。逐样本 norm ratio 的总体
中位数为 `1.0`。

**总体门槛通过不等于每个时间点都收敛。** 最早两个时间点的 K4/K8 cosine
分组中位数分别为 `0.98903/0.98818`，略低于 `0.99`。因此本轮把 K4 当作
一个明确、可复现的离散候选进入小规模质量筛查，不能称其已充分近似连续 flow。
epsilon 收敛与 K 收敛是不同检查，也不能互相替代。

### 4.3 长区间反例：`h=1/4, epsilon=1e-3`

来源：`audit_long_h1over4/{request.json,rows.csv,summary.json}`。

| actual noise time | pullback/raw cosine | K4/K8 cosine | 有限差分相对误差 |
|---:|---:|---:|---:|
| 0.969696999 | 0.327258306 | 0.606639747 | 0.130150039 |
| 0.900212348 | 0.579140874 | 0.935247198 | 0.102577249 |
| 0.747404814 | 0.880126295 | 0.999307848 | 0.000123126 |
| 总体 | 0.579140874 | 0.935247198 | 0.075314036 |

当前 K4/K8 粗积分在长区间高噪声侧不稳定，方向变化不能直接当成有效新信息；
有限 epsilon 检查也未通过。**本轮质量采样不用 `h=1/4`。** 这否定的是当前
长区间离散配置的可用性，不是更细积分下所有有限区间拉回方法。

同样，某个短区间的 pullback 与 raw gap 共线，也只能否定该区间可用的方向
创新，不能否定完整剩余轨迹中的瞬态放大。

## 5. 已固定的配对 1K 质量筛查

实现入口：[`sample_raev2_flow_pullback.py`](../experiments/sample_raev2_flow_pullback.py)。

- 官方 DINOv3-L K7，step100080 EMA，官方 stage-1 decoder/statistics；
- shifted 100-step Euler，clean-space `beta=1.78`，IG 区间 `[0.1,1]`；
- 仅 solver index `0..19` 使用新方向，共前 20 步；其余步骤保留 ordinary IG；
- future full flow 固定 `h=1/32`，均匀 raw-time Euler `K=4`；
- 两个候选为 `raw_future` 与 `pullback`，逐样本匹配当前 full-minus-base gap 范数；
- 采样 `seed=202609051`、均衡 1000 类、原始 batch size `8`；
- 质量筛查使用 bf16 autocast、TF32 enabled，clean arithmetic 为 float32。

机制审计是 FP32/no-TF32，质量采样是上述 bf16 配置，两者精度口径不同；不能
把 FP32 的有限差分误差原样宣称为 bf16 下的数值精度。

本轮匹配的 ordinary 基线来自新 radius/direction sampler，而非任意历史同 seed
artifact，其官方 FID-1K 为 **38.735079617949**，IS 为 `57.28434410095215`。
官方 evaluator 为 `nanogen-evals`，reference 为 `imagenet_256_fid_stats`。
若以该 bank 筛查相对 5% FID 降幅，数值门槛约为 `36.798326`。

基线来源：

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_radius_direction_20260905/
  screen_seed202609051/ordinary/official_metrics.json
```

新 flow-pullback sampler 的 ordinary smoke 已与匹配基线的 8 张图逐像素核对：
`ordinary_parity.json` 记录 `equal=true`、`differing_pixels=0`、
`max_pixel_difference=0`。这支持当前 smoke 的 sampler parity；正式分片 1K
仍应保留原始 batch 顺序、noise/label hashes 和合并完整性检查。

主要对比先看 `pullback` 相对 `raw_future`，以区分 future query 本身与 transpose
transport 的贡献；两者各自也必须对比匹配的 ordinary 基线。官方配对结果为：

| condition | FID-1K | delta vs ordinary | IS |
|---|---:|---:|---:|
| ordinary | **38.735080** | 0 | **57.284344** |
| raw future | 38.864229 | +0.129149 | 57.000472 |
| pullback | 39.275690 | +0.540610 | 56.718486 |

三路的 noise SHA256 均为
`1b5671d1759797211611b71c96d6ca82292a0579a0059adf05f0d54f579c3222`，label
SHA256 均为
`702746827e553786bb026ac120cb58745fef3d3f554c33891809001cc37639f0`。
因此这里不是随机 bank 差异。pullback 相对 raw future 还恶化 `0.411461` FID；局部
covector transport 的数值成立，并没有转化成更好的生成分布。

## 6. 已测成本与当前结论边界

相同 8 张样本、batch size 8 的 smoke 结果：

| 条件 | elapsed seconds | 实际 model forward 入口次数 | input VJP次数 | peak allocated |
|---|---:|---:|---:|---:|
| ordinary | 5.738694 | 100 | 0 | 5,507,942,400 bytes |
| pullback | 22.341863 | 280 | 20 | 11,425,637,888 bytes |

pullback 的 280 次入口包括 ordinary solver 的100次、forecast/future-query
primal的100次和checkpoint重算的80次。checkpoint重算可能提前停止；forward
入口、一次VJP与一次完整forward不是等FLOP单位，不能直接相加后称为等计算预算。
该 smoke wall time 约为 ordinary 的 `3.89x`，也不是稳定吞吐或正式 1K wall-time
估计。

来源：`smoke_ordinary/summary.json`、`smoke_pullback/summary.json`。两路8张
样本的 noise 与 labels SHA 均相同，费用表只报告该实际运行。

正式分片运行中，raw future 与 pullback 的最大 shard wall time 分别为
`371.685 s` 与 `1197.890 s`；峰值显存分别约 `5.53 GB` 与 `11.43 GB`。这些 wall
time 仍受并行机器负载影响，但足以说明 pullback 不是低成本替代。

最终可以说：实现了一个有明确局部优化对象的未来 covector 拉回；early-time
方向创新可测，frozen-covector VJP 具有有限差分收敛证据，且正式 1K 协议严格配对。
质量结论则为阴性：raw future 与 pullback 均未降低 FID，pullback 还同时降低 IS，
没有达到 5% 目标，也没有理由追加 5K。该结果说明“把 future covector 通过局部
Jacobian transpose 拉回”不是当前 RAEv2 上足够的质量原理。

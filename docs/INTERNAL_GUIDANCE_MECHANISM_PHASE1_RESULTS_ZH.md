# Internal Guidance 机制研究：第一阶段结果

日期：2026-08-03

## 1. 研究问题

本阶段不训练模型，只研究一个问题：

> Internal Guidance 使用的 `full - base`，究竟是在逐时刻修正监督误差，还是通过整条采样动力学把终点分布推向更高质量但不一定更真实的位置？

记：

- `Y`：监督目标；RAEv2 中是 clean RAE latent，SiT 中是目标 velocity；
- `F`：full head 预测；
- `B`：base/early head 预测；
- `d = F - B`：IG 方向；
- `s`：代码中的 IG scale；
- `gamma = s - 1`：相对 full head 的外推系数。

于是：

```text
guided = F + gamma * d
```

配对平方误差满足：

```text
R(gamma) - R(0)
  = 2 gamma E[<F-Y, d>] + gamma^2 E[||d||^2]
```

总体局部最优系数为：

```text
gamma* = E[<d, Y-F>] / E[||d||^2]
```

只有 `gamma* > 0`，正向 IG 才是局部监督误差修正。

## 2. RAEv2 局部方向审计

配置：

- 官方 RAEv2 DINOv3-L K7 checkpoint；
- EMA state；
- ImageNet validation；
- 512 张图；
- 5 个随机种子；
- 8 个官方 shifted Euler solver 时刻；
- 模型冻结，bf16 推理；
- 每个种子独立抽图和噪声；
- 总计 40 个“种子 × 时刻”组合。

结果：

- 40/40 个组合的总体 `gamma*` 都为负；
- 跨种子 `gamma*` 标准差很小；
- 官方 `gamma=+0.78` 在所有时刻都增大配对 clean-latent MSE；
- base head 大多数时刻确实比 full head 更差，但 `full-base` 不朝向 full 的剩余误差。

跨种子平均 `gamma*`：

| solver time | gamma* |
|---:|---:|
| 0.0748 | -0.1055 |
| 0.1404 | -0.1128 |
| 0.1983 | -0.1010 |
| 0.3380 | -0.0573 |
| 0.4972 | -0.0408 |
| 0.6524 | -0.0384 |
| 0.7976 | -0.0424 |
| 0.9492 | -0.0962 |

结论：

> RAEv2 的正向 IG 不是 forward-noised 训练分布上的局部 clean-target MSE 纠错器。

结果目录：

```text
~/data/eqvae/experiments/raev2_ig_direction_audit/ema_solvergrid_n512_5seed_v1/
```

## 3. 原始 SiT-IG 跨模型复核

配置：

- 官方 `SiT-XL/2+IG` 800 epoch checkpoint；
- EMA state；
- 677,786,144 参数；
- ImageNet validation 的 SD-VAE latent；
- 512 张图；
- 7 个 flow 时刻；
- fp32，关闭 TF32；
- 模型冻结，不做 teacher-path rollout。

总体最优系数：

| time | gamma* | scale*=1+gamma* |
|---:|---:|---:|
| 0.10 | -0.0553 | 0.9447 |
| 0.20 | -0.0707 | 0.9293 |
| 0.35 | -0.0935 | 0.9065 |
| 0.50 | -0.1341 | 0.8659 |
| 0.65 | -0.2616 | 0.7384 |
| 0.80 | -0.5167 | 0.4833 |
| 0.95 | -0.4632 | 0.5368 |

同时，`scale=1.2/1.5/1.8` 在 7/7 个时刻都增大局部 velocity MSE。

结论：

> “正向 IG 不是逐时刻监督 MSE 纠错”同时出现在 RAEv2 和原始 SiT-IG，因而不是 RAE latent 独有现象。

结果目录：

```text
~/data/eqvae/experiments/internal_guidance_sit_audit/n512_seed20260803_v1/
```

## 4. RAEv2 单窗口终点响应

将 100 步采样分为五段：

```text
[0.8, 1.0]
[0.6, 0.8]
[0.4, 0.6]
[0.2, 0.4]
[0.1, 0.2]
```

只在一个窗口使用 `gamma=+0.05` 或 `-0.05`，其余步骤保持 full head。所有条件共享 checkpoint、类别和初始噪声。为避免小扰动被 bf16 舍入污染，采样和 endpoint 存储均使用 fp32。

正负扰动被分解为：

```text
odd  = (z(+gamma) - z(-gamma)) / 2
even = (z(+gamma) + z(-gamma)) / 2 - z(0)
```

- `odd/gamma` 近似一阶终点响应；
- `even/odd` 衡量二阶和非线性作用。

64 样本结果：

| window | direct injected norm | odd endpoint RMS | odd/gamma | even/odd |
|---|---:|---:|---:|---:|
| [0.8, 1.0] | 0.000339 | 0.05733 | 1.1466 | 0.7350 |
| [0.6, 0.8] | 0.000774 | 0.00522 | 0.1044 | 0.2377 |
| [0.4, 0.6] | 0.001400 | 0.00367 | 0.0735 | 0.3288 |
| [0.2, 0.4] | 0.002154 | 0.00359 | 0.0718 | 0.3360 |
| [0.1, 0.2] | 0.003487 | 0.00406 | 0.0811 | 0.2991 |

需要注意，official shifted solver grid 在这五个 `t` 窗口中分别包含 `67/18/8/4/2` 个步骤。所以下表测到的是整个时间窗口的累计响应，不能直接解释成单个高噪声步骤的传播算子更强。真正的逐时刻 terminal leverage 还需要单步脉冲或等步数窗口实验。

关键现象：

1. `h/t` 使直接注入能量向低噪声末端集中；
2. 最高噪声时间段的累计终点响应最大，但其中同时包含更多 active steps；
3. 高噪声窗口的单位 `gamma` 累计终点响应约为其余窗口的 7–16 倍；
4. 高噪声窗口 `even/odd` 较大，说明其作用不是干净的线性传播，而包含明显轨迹弯曲或分叉。

因此：

> “哪个窗口累计的平方 impulse 最大”和“哪个窗口最终最能改变样本”是两个不同问题；但尚需用单步脉冲排除 active-step 数量和跨步方向相干性。

结果目录：

```text
~/data/eqvae/experiments/raev2_ig_window_response/fp32_n64_seed20260809_g005_v1/
```

## 5. 64 样本解码筛选

同一批 endpoint 经官方 RAEv2 decoder 解码，再用 ImageNet Inception-v3 分类器筛选语义置信度。

高噪声窗口的主要变化：

| condition | max probability delta | true-class log-prob delta | entropy trend |
|---|---:|---:|---|
| gamma=+0.05 | +0.00910 | -0.00224 | 下降 |
| gamma=-0.05 | -0.00832 | +0.01343 | 上升 |

top-1 accuracy 在所有条件中均为 0.8125，没有变化。

配对 bootstrap 进一步表明，这些 N=64 趋势尚未达到统计显著：

- 正向最大概率变化 `+0.00910`，95% CI 约为 `[-0.00120, +0.02032]`；
- 正向 true-class log-prob 变化 `-0.00224`，95% CI 约为 `[-0.03518, +0.02934]`；
- 正向 entropy 变化 `-0.03372`，95% CI 约为 `[-0.09627, +0.02762]`。

这个小样本结果支持：

> 正向 IG 在高噪声阶段呈现“更自信、更集中”的候选趋势，而没有提高 top-1；该趋势仍需要更大样本确认。

这与原始 IG 论文在二维例子中观察到分布变窄是一致的。但 N=64 只能作为筛选信号，不能报告为 FID 结论。

## 6. 当前能说与不能说的结论

已经有可靠证据：

1. RAEv2 与原始 SiT-IG 的正向 `full-base` 都不是局部监督 MSE 纠错方向；
2. RAEv2 的终点响应不能由 raw head gap 或 `h/t` 单独解释；
3. 高噪声小扰动具有异常高的终点杠杆和明显非线性；
4. 小样本解码信号给出分布集中/置信度提高的候选方向，但尚未达到统计显著。

目前不能声称：

1. 高噪声窗口一定改善 FID；
2. 固定 scale 的最优值可以由当前局部量直接计算；
3. decoder 是全部排名反转的原因；
4. 已经得到可在线使用的反馈控制器；
5. 当前结果足以形成通用 IG 理论。

## 7. 大规模质量实验及验收标准

在第 9 节列出的 N=64 机制实验通过后，大规模质量实验只保留三路：

```text
full baseline
high-noise gamma=+0.05
high-noise gamma=-0.05
```

先做 1k 配对样本，报告：

- decoded FID/KID，仅作为方向筛选；
- precision/recall 或等价覆盖指标；
- Inception true-class log-prob、confidence、entropy；
- endpoint latent 分布指标；
- 两个独立 seed。

扩到 5k 的门槛：

1. 正向与负向条件在两个 seed 上给出稳定相反的 concentration 信号；
2. 正向条件的 FID/KID 方向优于 baseline，且不是仅靠类别置信度提高；
3. endpoint 与 decoded 指标的排序能复现；
4. 结果大于 1k 评估噪声。

Replay/recursive、真正 equal-energy 和 curl 诊断应先在 N=64 上完成；它们是进入本节 1k/5k 质量实验的前置机制检查。

## 8. 文献定位

- [Internal Guidance, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Guiding_a_Diffusion_Transformer_with_the_Internal_Dynamics_of_Itself_CVPR_2026_paper.html)
- [RAEv2, arXiv:2605.18324](https://arxiv.org/abs/2605.18324)
- [The Unreasonable Effectiveness of Guidance](https://arxiv.org/abs/2411.10257)
- [Limited Guidance Interval, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html)

现有论文说明“强模型减弱模型”常常有效，也说明 guidance 的时间窗口很重要；本阶段的新证据集中在它们之间缺失的一环：

> 局部 head difference 并不修正监督误差，但其时间局部扰动可被后续动力学以高度不均匀、非线性的方式传播到终点。

## 9. 对照理论计划后的剩余工作

### 9.1 已完成的部分

- 局部误差对齐审计已经完成。每个 seed 内都做了样本 bootstrap；40 个“seed x time”的 `gamma*` 置信区间上界全部小于 0。
- equal-gamma 单窗口响应已经完成，并使用 `+gamma/-gamma` 分离了一阶响应和偶次非线性。
- 已经验证 fp32 推理和 fp32 endpoint 存储是小扰动实验的必要条件；bf16 会污染小差分。
- 已做 64 样本 decoder 后筛选，但它只能说明置信度和熵的变化，不能替代 FID/KID 或 precision/recall。

### 9.2 仍必须完成的部分

1. **真正的 equal-injected-energy 重采样。** 当前 N=64 的 energy-normalized 数值是事后用 endpoint response 除以预测注入范数，并不是为每个窗口重新选择 `gamma` 后采样。高噪声窗口已经明显非线性，两者不能等价。应使用对称 `+gamma_i/-gamma_i`，令 baseline 轨迹上的一阶注入能量相同，并记录实际 guided 轨迹上的注入能量作为偏离检查。
2. **单步脉冲和等步数窗口。** 当前五个等 `t` 区间包含 `67/18/8/4/2` 个 solver steps。应先在若干代表 solver index 上只扰动一步，再做每段相同 step 数的窗口。否则不能把累计窗口响应称为逐时刻传播增益。
3. **同批样本上的真实注入统计。** 当前 predicted energy 来自另一项 baseline 参数化统计，且 `sum ||b_k||^2` 忽略跨步方向相干或抵消。新采样应同时保存逐样本 `sum ||b_k||^2`、`||sum b_k||^2` 和 active-step 数。
4. **Replay 与 recursive 对照。** Replay 固定使用无 guidance 轨迹记录的 `g_k^0=F-B`，但 full field 仍在新状态上重新计算；recursive 每一步都在新状态上重算 `g(x_k,t_k)`。二者之差隔离 guidance 方向自身的状态反馈。
5. **终点质量的有符号响应。** 目前主要测到 endpoint 变化的范数，只知道“影响多大”，不知道“对质量有利还是有害”。应对 `+gamma/-gamma` 计算 FID/KID、precision/recall、类别 log-prob 和熵的中心差分。
6. **两窗口交互。** 在 replay 结果明确后，先只测“高噪声窗口 + 其余窗口”，计算 `z_ij-z_i-z_j+z_0` 及质量指标的交互项，不应一开始穷举全部调度。
7. **保守性/curl 否证。** 用多个有限差分步长估计 `u^T J_g v-v^T J_g u`，并使用已知梯度场作为数值正对照。若反对称成分稳定显著，就停止把 `F-B` 解释为 density-ratio score。
8. **跨模型终点复核。** SiT-IG 目前只完成局部方向审计。若 RAEv2 的 replay/curl/窗口结论稳定，必须在原始 SiT-IG 上复核至少一个关键终点现象，才能区分通用 IG 机制与 RAEv2 的 `x`-prediction、`h/t` 特性。

### 9.3 一个重要的结论边界

局部误差审计使用真实图像构造的 forward-noised 训练路径 `p_t`，因此有配对 clean target；实际采样运行在模型生成轨迹 `q_t` 上，那里没有配对 clean target。于是当前结果严格支持：

> 正向 IG 不是训练 marginal 上的局部监督误差修正。

但当前结果还不支持：

> 正向 IG 在实际生成轨迹的任何位置都不可能具有纠错作用。

若没有额外 oracle、teacher 或可验证的密度比，不能把 `q_t` 上的 `full-base` 直接命名为“真实误差”。Replay、窗口响应和 curl 可以研究它的动力学性质，但不会凭空提供这个不可观测目标。

### 9.4 严格执行顺序

1. N=64，fp32：补同批逐样本注入统计，并做单步脉冲与等步数窗口。
2. N=64，fp32：在上述控制下做真正 equal-energy 的对称重采样。
3. N=64，fp32：对最敏感的单步/窗口做 replay/recursive `+0.05/-0.05` 对照。
4. 小样本、多有限差分尺度：curl/保守性诊断。
5. 只有上述机制稳定后，做两个 seed、每路 1k 的 baseline/high-positive/high-negative 质量实验。
6. 只有 1k 的方向超过评估噪声，才扩到 5k，并随后测高噪声窗口与其他窗口的二阶交互。

这个顺序把便宜的机制否证放在昂贵的分布指标之前，也避免在尚未知道执行方向性质时直接设计 response-aware schedule。

## 10. 单步、等步数、Replay 与 Curl 新结果

本节实验全部冻结官方 RAEv2 DINOv3-L K7 EMA checkpoint，使用官方 float32 shifted Euler grid、关闭 TF32，并自动对照显式 baseline 与官方 sampler。两者 endpoint RMS 差异约 `1.6e-6`，相对 endpoint RMS 约 `0.9` 可以忽略。

### 10.1 单步脉冲排除了 active-step 数量混杂

在两个独立 seed、每个 N=64 上，只在一个 solver step 使用 `gamma=+0.05/-0.05`。下表报告 `odd endpoint RMS / gamma` 再除以该步单位注入 RMS，即后续 full-field 的传播归一化增益：

| solver step | time | seed 20260810 | seed 20260811 |
|---:|---:|---:|---:|
| 10 | 0.9863 | 540.7 | 352.5 |
| 30 | 0.9492 | 134.7 | 72.5 |
| 50 | 0.8889 | 41.8 | 18.4 |
| 70 | 0.7742 | 2.82 | 2.97 |
| 90 | 0.4706 | 0.995 | 1.008 |
| 98 | 0.1404 | 0.944 | 0.947 |

这说明早期高杠杆不是原来 `67/18/8/4/2` 个 active steps 导致的。单次极小注入也会被后续 baseline full-field 放大数百倍；晚期步骤的终点变化主要来自直接注入大，后续传播增益约为 1。

早期响应明显重尾：

- step 10 的两个 seed 中位数分别为 `0.0261/0.0227`，均值为 `0.0885/0.0587`；
- step 10 的 q95 分别为 `0.405/0.231`，最大值为 `1.280/0.540`；
- step 70 以后均值与中位数接近，尾部明显收缩。

因此更准确的机制描述是：

> 高噪声早期 full-field 对特定 IG 方向存在强烈、样本依赖且重尾的瞬态放大；少数轨迹会发生远大于典型样本的终点跳变。

### 10.2 等步数窗口复现同一排序

将 100 步按 solver index 等分为五个 20 步窗口。两个 seed 的传播归一化增益依次为：

```text
seed 20260810: 731, 274, 89, 13, 2.09
seed 20260811: 758, 285, 72, 12, 2.09
```

排序和量级跨 seed 稳定。

### 10.3 真正 equal-energy 的独立 seed 验证

使用 seed 20260810 校准每个窗口的固定 gamma：

```text
0.050000, 0.019822, 0.009737, 0.003858, 0.000271
```

随后在未参与校准的 seed 20260811、N=64 上重采样。五个窗口的实际注入能量都约为 `1.66e-9`，终点 odd RMS 为：

```text
0.031302, 0.012785, 0.003301, 0.000490, 0.000085
```

最早窗口对终点的影响约为最晚窗口的 368 倍。因此早期优势不能由 active-step 数量或 `h/t` 注入能量解释。

### 10.4 Replay/recursive 隔离状态反馈

对早期 `step 0:20` 窗口、N=64：

| metric | value |
|---|---:|
| recursive response / gamma | 0.5868 |
| replay response / gamma | 0.5871 |
| feedback difference / recursive | 0.83% |
| feedback fraction 95% CI | [0.52%, 1.22%] |
| recursive/replay direction cosine | 0.99988 |

Replay 固定使用 baseline 轨迹记录的 `F-B`，但 full field 仍在新状态上重算。结果说明持续重算 guidance gap 的额外反馈很小；早期巨大响应主要是 baseline full-field 对已注入偏移的传播。

### 10.5 Curl/保守性否证

先做两个独立 seed、各 N=16；随后又用第三个独立 seed 扩大到 N=64。全部实验都覆盖六个时刻、四个随机方向对和三个有限差分 epsilon：

- 两个 N=16 seed 的标准化反对称 Jacobian 能量约为 `0.96-1.39`；
- N=64 终验在六个时刻得到 `0.978-1.305`，完整复现同一量级；
- N=64 的逐样本比值均值约为 `1.04-1.29`，bootstrap 95% CI 始终远离 0；
- `epsilon=0.01/0.003/0.001` 的结果几乎重合；
- 扩大样本后没有出现向 0 收缩的趋势。

`F-B` 的反对称部分与总体交叉响应同量级，因而不能合理解释为某个 scalar potential、energy 或 density-ratio score 的梯度。对当前 RAEv2，更合适的表述是一般非保守干预方向。

### 10.6 当前最可靠的机制结论

目前可以较可靠地说：

1. `F-B` 不是训练 marginal 上的局部 MSE 纠错方向；
2. `F-B` 也不是近似保守的 density/energy gradient；
3. IG 的早期巨大作用主要来自 baseline full-field 的后续瞬态放大；
4. 这种放大在早期具有明显重尾和强非线性；
5. 反复重算 `F-B` 的状态反馈在 `gamma=0.05`、早期20步窗口中只贡献约 1%。

仍然不能声称这种动力学一定改善 FID。正在运行的 N=256、两个 gamma 单步实验用于确认重尾概率和局部线性区；只有机制终验后才决定是否投入 1k/5k decoded quality 对照。

## 11. 官方 SiT-XL/2+IG 跨参数化复核

为排除 RAEv2 的 shifted grid、clean prediction 和 `h/t` 参数化造成假象，新增官方 `SiT-XL/2+IG` 复核。使用作者公开的 800 epoch EMA checkpoint、第 8 层 auxiliary head、`4x32x32` SD-VAE latent；模型冻结、关闭 CFG、关闭 TF32。机制实验采用与官方代码完全一致的 50 步 deterministic Euler ODE。显式 full-head baseline 与官方 sampler 的 RMS 和最大绝对误差均为 0。

两个独立 seed、每个 N=128，共 256 个样本。只在一个 solver step 注入 `gamma=+0.01/-0.01` 或 `+0.05/-0.05`，随后恢复 frozen full-head flow。下表是两个 gamma 合并后几乎相同的传播增益量级；置信区间取 `gamma=0.01`：

| solver time | propagation gain mean | 95% CI | median | q95 |
|---:|---:|---:|---:|---:|
| 0.90 | 24.17 | [20.25, 28.93] | 13.61 | 83.75 |
| 0.70 | 12.49 | [11.13, 13.95] | 8.84 | 34.61 |
| 0.50 | 4.29 | [3.92, 4.78] | 3.72 | 6.95 |
| 0.30 | 1.66 | [1.61, 1.75] | 1.62 | 1.88 |
| 0.10 | 1.019 | [1.016, 1.021] | 1.022 | 1.051 |
| 0.02 | 1.000 | [1.000, 1.000] | 1.000 | 1.000 |

主要结果：

1. SiT 直接预测 velocity，单步状态注入是 `dt * gamma * (full-base)`，没有 RAEv2 的 `1/t` 因子；但早期高杠杆仍完整复现。
2. `gamma=0.01` 与 `0.05` 的中心差分方向在典型样本上几乎一致。各时刻平均 cosine 接近 1、amplitude ratio 接近 1；少数早期重尾样本会抬高均值相对误差。
3. 五个等 10 步窗口的传播增益均值依次约为 `52.06/37.09/13.36/5.26/3.04`，与单步排序一致。
4. 两窗口一阶响应基本可加。最强的早期与中期窗口组合，其 derivative relative error 均值为 `0.0088`，direction cosine 为 `0.9996`；但 mixed-over-joint 均值仍约 `0.024`，且少数样本存在明显尾部。

因此可以把 RAEv2 结论收紧为：

> 早期 IG 扰动的高终点杠杆不是 `h/t` 的数值伪影，而是至少同时存在于 RAEv2 clean-prediction flow 和 SiT velocity flow 中的有限时间传播现象。

这仍只说明“IG 在什么时候最能改变终点”，不说明“改变是否改善图像质量”。原论文在 `SiT-B/2` 上报告 `scale=2.3, [0.3,1)` 优于全程 IG，并发现关闭低噪声区间有利。当前公开可复用 checkpoint 是 `SiT-XL/2` 第 8 层版本，因此后续 5k 区间实验只作为趋势复现，不能与论文 `SiT-B/2` 的 50k FID 绝对值直接比较。

方向特异性控制已经完成。对两个独立 seed、每个 N=64，共 128 个样本，在每个时刻构造四个逐样本正交于 IG gap、且与 IG gap 同 RMS 的确定性随机方向。实际构造审计确认随机方向与 IG gap 的 cosine 约为数值零、RMS 比为 1。传播增益结果为：

| solver time | IG gain | matched random gain | IG/random | bootstrap 95% CI | IG 高于随机的样本比例 |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 32.26 | 2.42 | 10.20 | [8.87, 11.85] | 100% |
| 0.70 | 11.14 | 1.56 | 6.65 | [6.10, 7.30] | 100% |
| 0.50 | 4.31 | 1.15 | 3.68 | [3.42, 3.99] | 100% |
| 0.30 | 1.64 | 0.96 | 1.71 | [1.67, 1.76] | 100% |
| 0.10 | 1.02 | 0.97 | 1.06 | [1.05, 1.07] | 100% |
| 0.02 | 1.00 | 1.00 | 1.00 | [1.00, 1.00] | 42% |

它排除了下面这个简单解释：

```text
任何早期扰动都会被放大
```

更符合数据的描述是：IG gap 在高、中噪声阶段明显对齐了 frozen full flow 的高有限时间增益方向；这种方向特异性随接近终点而消失。这里的“高增益”只表示同样大小的局部扰动会造成更大的终点 latent 改变，仍不表示该改变一定改善 FID。

下一步补齐论文未报告的两个连续区间 `[0,0.3)` 与 `[0.7,1)`，并保留论文锚点做 5k SDE/ADM-FID 趋势复现。评估实现已经拆成官方 ADM Inception 特征提取和 PyTorch/MKL 精确 FID 两步，避免通用 `sqrtm` 在小样本、秩亏协方差上的异常耗时；随机小维测试与标准 FID 公式一致。

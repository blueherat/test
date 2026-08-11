# 双预测目标闭环动力学：连续螺旋 Toy 实验报告

## 技术结论

这组实验复现并拆解了真实 SiT 双头实验中的核心反常现象：

> 在训练路径上的逐点速度场误差更低，并不保证闭环采样后的终点分布更好。

最干净的证据来自固定同一组 `D0` 的 `x`/`epsilon` 双头、只替换 gate 的公平对照。在高维环境 `D=512` 中：

| 同一组 D0 双头上的策略 | full-D SWD，均值 +/- seed 标准差 | intrinsic SWD | 螺旋弧长覆盖 TV |
|---|---:|---:|---:|
| 只使用 `x` 头 | 0.2213 +/- 0.0577 | 0.1545 | 0.3102 |
| `D1` scaled-loss gate | 0.1759 +/- 0.0307 | 0.1241 | 0.2723 |
| `D2` velocity-loss gate | 0.1449 +/- 0.0172 | 0.1032 | 0.2285 |
| `D3` 逐状态 Bayes oracle gate | 0.1429 +/- 0.0168 | 0.1015 | 0.2186 |
| `D4` asymptotic-safe gate | **0.1270 +/- 0.0145** | **0.0900** | **0.1853** |

`D4` 相对同头 `x` 分支降低 full-D SWD 约 `42.6%`，并相对逐状态 Bayes oracle 降低约 `11.2%`。后一个结果在三个 seed 上都成立：

| seed | D3 oracle | D4 safe |
|---:|---:|---:|
| 20260831 | 0.1612 | **0.1416** |
| 20260901 | 0.1281 | **0.1125** |
| 20260902 | 0.1395 | **0.1268** |

这不能解释为 `D4` 的局部预测更准确。恰恰相反，在 `D=512, t=0.99`：

| 指标 | D3 oracle | D4 safe |
|---|---:|---:|
| teacher state 上到精确 Bayes 场的 MSE | **0.7453** | 0.9762 |
| rollout state 上到精确 Bayes 场的 MSE | **3.6131** | 49.1262 |

因此目前最可靠的结论不是“`D4` 学到了更准确的速度场”，而是：

> 局部 `L2` 最优的向量场组合与闭环终点分布最优的组合并不等价；满足端点渐近约束的时间调度，可以牺牲局部场误差并改善最终分布覆盖。

这给真实 SiT 实验中“dynamic validation velocity MSE 更低但 FID 更差”的现象提供了一个机制级、可直接测量的对应物。

## 研究问题

实验只回答下面的问题：

1. `x` 与 `epsilon` 两个预测头换算成 velocity 后，逐点最优组合是否也会产生最优闭环分布？
2. 原来的 scaled gate loss、真实 velocity gate loss、解析 oracle gate 和端点安全 gate 的区别在哪里？
3. 失败来自双头共享、端点除法、高维 ambient geometry，还是闭环 rollout 本身？
4. `x + epsilon` 与更适合 flow matching 的 `x + v` 双头是否表现不同？

本实验不尝试证明真实图像模型中的最终机制，也不把 toy 的最优 gate 直接当作图像生成方法。

## 数据与流

数据沿用 v4/v10 的连续两圈螺旋：

```text
s ~ Uniform(0, 1)
u = spiral(s) + Gaussian jitter
x = linear_embed(u) in R^D
```

flow matching 路径为：

```text
x_t = (1 - t) * epsilon + t * x
v*  = x - epsilon
t: 0 -> 1
```

其中：

- `D=2` 是无高维 ambient effect 的控制组。
- `D=512` 保留 JiT/v4/v10 中 prediction target 难度随环境维数拉开的现象。
- 嵌入使用 `unit_rms`，curvature 固定为 `0`，从而可用已知连续分布计算任意状态上的精确 Bayes conditional velocity。
- v4 会在加 jitter 后整体乘 `1.6`，因此 intrinsic jitter 标准差是 `1.6 * 0.015 = 0.024`。正式评估和 NLL 均使用这个值。

## 模型与方法

所有 MLP 使用 `hidden=128`、`depth=4`。正式实验训练 `15,000` steps，batch size 为 `2,048`。

### 单头基线

- `B0_v`：直接预测 velocity。
- `B1_x`：预测 clean endpoint，再用 `(x_hat - x_t) / (1 - t)` 转为 velocity。
- `B2_eps`：预测 noise endpoint，再用 `(x_t - eps_hat) / t` 转为 velocity。

### x + epsilon 双头

- `D0_xeps`：共享 trunk，同时训练 `x` 与 `epsilon` 两个头，不训练 gate。
- `D1_scaled`：用原 SiT 迁移版本的 scaled residual 训练 gate。这个目标等于真实 velocity residual 乘以 `[t(1-t)]^2`，会弱化两端。
- `D2_velocity`：直接最小化混合 velocity 的误差。
- `D3_oracle_bayes_gate`：不训练 gate；在每个访问状态上，用已知 Bayes velocity 解析求 `[0,1]` 内的最优 scalar gate。
- `D4_safe`：仍训练 velocity gate，但参数化为

```text
r = sigmoid(log((1 - t) / t) + h_theta(x_t, t))
```

当 `h_theta=0` 时，`r=1-t`。该形式自然满足：

```text
1 - r = O(t)       as t -> 0
r     = O(1 - t)   as t -> 1
```

从而避免 gate 本身让 `(1-r)/t` 或 `r/(1-t)` 发散。

### x + v 对照

- `S0_xv`：共享 trunk，预测 `x` 和原生 `v`，使用固定切换。
- `S1_xv`：在 `S0` 上增加 `x_hat` 与 `x_t + (1-t)v_hat` 的 consistency loss。

### 公平的 same-head gate transfer

`D1/D2/D4` 各自训练时，双头表征也会不同。为排除这个混杂，核心对照把三个已训练 gate 全部应用到同一组 `D0_xeps` 头：

```text
D1_gate_on_D0
D2_gate_on_D0
D4_gate_on_D0
```

因此这张表中的差异来自 gate，而不是来自不同的 `x`/`epsilon` 预测头。

## 指标定义

- `intrinsic SWD`：把样本投影回真实二维子空间后，与真实螺旋分布的 sliced Wasserstein distance；越低越好。
- `full-D SWD`：在完整 ambient space 中比较生成分布与真实分布；越低越好。
- `ridge distance`：样本到螺旋中心线的平均距离；它单独不能判断分布好坏，因为过度锐化也会让它变小。
- `ridge width / reference`：条件横向宽度相对真实数据宽度的比例；接近 `1` 较好。
- `arc coverage TV`：沿螺旋弧长的离散分布与参考分布的 total variation；越低表示模式覆盖越完整。
- `off-subspace RMS`：样本在真实二维线性子空间之外的均方根；它是几何诊断，不是单调的生成质量指标。
- `teacher Bayes MSE`：在训练 bridge 采样的状态上，到精确 Bayes velocity 的误差。
- `rollout Bayes MSE`：在模型真实闭环轨迹访问的状态上，到精确 Bayes velocity 的误差。

`Reference_resample` 是两份有限参考样本之间的 Monte Carlo 比较，只用于给出有限样本噪声尺度，不是严格数学下界。

## 主要发现

### 1. 高维设置满足预期的 target 难度顺序

在 `D=512` 中，clean 数据只占二维子空间，而 `v/epsilon` 需要传递约 510 个法向噪声方向。模型最后的 128 维 hidden bottleneck 无法完整表达这些方向，所以：

```text
x prediction 明显优于 v prediction，epsilon prediction 最差。
```

这正是选择该高维设置的目的。`D=2` 中没有这个 rank bottleneck，`v` 可以优于 `x`，它只作为低维控制组，不能与高维结论混用。

### 2. 逐点 oracle 能显著改善差头，但不是终点分布最优

`D3` 在每个状态都选择使当前 Bayes velocity MSE 最小的 gate。它显著优于任何单独的 `D0` 分支，证明两个头确有互补信息，mixing 本身不是无效的。

但 `D4` 在 `D=512` 上的 full-D SWD 和 intrinsic SWD 都优于 `D3`，且三个 seed 一致。这说明逐点最优并没有给出全局 transport 最优。

### 3. D4 的优势来自闭环调度，不来自更低的局部误差

`D4` 在大部分轨迹中保持较高的 `x` 权重，接近数据端时再快速切换到 `epsilon`；同时渐近参数化限制了端点除法放大。

在 `t=0.99`，同一组 D0 头上大致有：

```text
D4: r/(1-t) ~= 2.26
D3: r/(1-t) ~= 26.56
D2: r/(1-t) ~= 21.34
D1: r/(1-t) ~= 99.76
```

这支持“endpoint-safe switching”解释。不过它目前仍是机制一致性证据，不是唯一因果证明；还需要固定 gate 的时间均值、移除 state dependence 等干预。

### 4. 法向误差不是可以直接最小化的质量代理

在同头高维对照中，`D4` 相比 `D3`：

- intrinsic SWD 更低：约 `0.0900` 对 `0.1015`；
- arc coverage TV 更低：约 `0.1853` 对 `0.2186`；
- ridge distance 略高：约 `0.1621` 对 `0.1592`；
- off-subspace RMS 更高：约 `0.0672` 对 `0.0154`。

因此不能说“越贴近已知子空间，生成质量越好”，也不能反过来说“法向偏离导致了改善”。当前数据只证明：更好的终点覆盖可以伴随更大的法向误差，法向误差不是单调质量指标。

### 5. x + v consistency 在本设置中没有解决问题

`S0/S1` 的 intrinsic SWD 不差，但 full-D SWD 约为 `0.199`，主要被较大的 off-subspace deviation 拉高；当前 consistency weight `0.1` 没有带来可见改善。这是一个有用负结果，但不能据此否定 SC-Flow 的完整方法，因为这里没有复现它的完整架构、训练预算与采样协议。

### 6. 当前没有证据支持“共享 trunk 梯度强冲突”

最终 gradient audit 中，`x` 与 `epsilon` 梯度整体接近正交，没有看到稳定的强负 cosine。这个检查样本和时间都有限，所以结论只能是：当前数据不支持把失败主要归因于明显的梯度冲突。

## 可靠性检查

- 三个训练 seed：`20260831/20260901/20260902`。
- 两个 ambient dimensions：`2/512`。
- 每个 setting 的行数、键唯一性、必需列、NaN/Inf 均由独立 validator 检查。
- 相关测试共 `27` 项，全部通过。
- 精确 Bayes sampler 的终点接近 reference，ridge width/reference 约为 `1`，说明 Bayes 场实现本身健康。
- Heun `200` 与 `400` steps 的 full-D SWD 排序稳定：

| D | 200 steps | 400 steps |
|---:|---:|---:|
| 2 | 0.02945 | 0.02946 |
| 512 | 0.02679 | 0.02677 |

- 正式评估发现并修复了一个只影响 Bayes/NLL 评估、不影响训练 checkpoint 的尺度错误：v4 的 jitter 会随整体 `1.6` 缩放。修复后重新评估了全部 6 个 setting。
- `D3_oracle_pair_teacher_only` 只用于 teacher-forced 配对上限诊断，从未用于闭环采样；闭环 `D3_oracle_bayes_gate` 只读取已知分布的 Bayes velocity，不读取当前样本对应的 clean/noise pair，因此不存在 paired-target 泄露。

## 局限

1. 只有 3 个 seed，足以确认方向一致，但不适合做细微差异的强统计声明。
2. 数据流形是已知的连续螺旋，且 curvature 固定为 `0`；不能直接外推到自然图像或非线性 ambient manifold。
3. gate 为 scalar；没有研究 channel/token/spatial gate。
4. 训练预算和网络很小；这里的目标是机制可识别性，不是最佳 toy 分数。
5. `D4` 的改善尚未被分解为纯时间调度、state-dependent correction、端点正则和随机优化四个独立因素。
6. SWD、coverage、ridge 与 off-subspace 指标是互补描述，不存在一个已证明等价于图像 FID 的单一 toy 指标。

## 下一步

最有信息量的下一组实验不是扩大模型，而是把 `D4` 的有效成分拆开：

1. 用 `D4` 的跨样本平均 gate `r_bar(t)` 替换 state-dependent gate；若效果保持，主要机制是时间调度。
2. 扫描满足端点渐近约束的解析 switch family，分离“何时切换”和“是否自适应”。
3. 固定同一组 D0 双头，对 gate 做 tangent-only、normal-only 和 endpoint-weighted 干预，检查何种误差分量真正改变终点覆盖。
4. 在上述因果拆分通过后，再迁移到小型图像 flow matching；不直接在大 SiT 上盲调 learned gate。

## 复现与产物

正式入口：

```bash
bash experiments/run_dual_target_closed_loop_spiral_toy_formal.sh
```

结果默认写到：

```text
/home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1
```

Git 中的精简结果包位于：

```text
dual_target_closed_loop_spiral_toy_v1/
```

其中不包含 checkpoint 和原始 sample tensor，只包含复核所需的 CSV/JSON、无损多页 PDF、页码 manifest 与说明。

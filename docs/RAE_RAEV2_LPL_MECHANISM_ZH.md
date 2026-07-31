# 旧 RAE 与 RAEv2 的 LPL 分化机制

## 结论

现有证据支持一个两层结论。

第一层解释原始 strict LPL 为什么在 RAEv2 上失败：

> 旧 RAE 中，LPL 同时改善了真实 decoder-feature 对齐并恢复了不足的特征
> 方差；RAEv2 中，internal guidance 已经恢复了大部分方差，strict LPL
> 主要继续增大 prediction-variance 分母，而固定目标统计下的真实对齐反而
> 变差。

第二层解释为什么去掉该分母捷径仍不能超过 Flow：

> 单个真实插值配对上的 clean-latent 或 decoder-feature 代理，不足以控制成熟
> RAEv2 在自生成 ODE 轨迹上的最终分布。修正代理可以被正确优化，但它的
> 非 Flow 切向更新仍会扰动已经校准较好的向量场。

因此，目前没有证据支持继续做 low-noise-only strict LPL。低噪声恰好是
RAEv2 guided 输出方差已经最接近目标、又最容易被 LPL 推成过校准的区域。

## 严格比较协议

- 数据：相同的 16 张 ImageNet-1k validation 图像。
- 图像索引：复用旧 RAE cache 的 `test_indices.npy`。
- 预处理：相同的 ADM 256 中心裁剪，不做随机翻转。
- 固定时刻：`t = 0.25, 0.40, 0.55, 0.70`。
- 固定随机种子：`20260730`。
- 旧 RAE：DINOv2-B、DiTDH-S ep20、配对的 Flow500/LPL500 EMA。
- RAEv2：DINOv3-L K=7、配对的 Flow10/full+base LPL10 online model。
- RAEv2 额外分别审计 `full`、`base` 和实际采样使用的
  `base + 1.78 * (full - base)`。

两个 tokenizer 的 latent 形状不同，不能比较绝对 loss 或梯度范数。可比较的
量是各自配对分支内的相对变化、归一化误差、方差比例和梯度余弦。

## 指标含义

设 decoder feature 残差平方和为 `N`：

```text
strict loss       = N / prediction feature variance
target-normalized = N / fixed target feature variance
variance ratio    = prediction variance / target variance
```

`strict` 下降可能来自两条路：

1. 分子 `N` 真正减小；
2. prediction variance 分母增大。

`target-normalized` 的分母固定，因此更适合判断真实 feature alignment 是否
改善。

## 跨版本固定时刻结果

### 旧 RAE：LPL500 相对 Flow500

| t | clean latent error | feature variance | strict | target-normalized | raw |
|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.067% | +0.011 | -2.552% | -1.057% | -0.217% |
| 0.40 | +0.068% | +0.021 | -4.075% | -1.232% | -2.041% |
| 0.55 | +0.063% | +0.028 | -5.025% | -1.193% | -1.823% |
| 0.70 | +0.044% | +0.032 | -4.928% | -0.409% | +0.902% |

旧 RAE 的 LPL 不是只增大分母。它在四个时刻都降低固定目标统计的误差，同时
提高原本不足的 feature variance。它允许 clean-latent MSE 极小幅恶化，换取
decoder 看来更有意义的 feature 修正；这与其生成质量改善同向。

先在每张图内平均四个时刻，再跨 16 张图统计：target-normalized 改善为
`-0.961% +/- 0.137% SEM`，`16/16` 张图同向；feature variance 增加
`0.023 +/- 0.0014 SEM`，同样为 `16/16` 张图同向。

### RAEv2：full+base LPL10 相对 Flow10

下表报告实际采样的 guided 输出。

| t | clean latent error | feature variance | strict | target-normalized | raw |
|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.106% | +0.027 | -1.642% | +1.587% | +9.169% |
| 0.40 | +0.157% | +0.035 | -2.255% | +1.852% | +6.369% |
| 0.55 | +0.174% | +0.039 | -2.353% | +2.326% | +2.941% |
| 0.70 | +0.153% | +0.035 | -2.193% | +2.037% | +0.509% |

这里 strict 比值虽然下降，但固定目标统计误差和 clean-latent 误差全部恶化。
因此，原始 RAEv2 strict LPL 的改善主要是 prediction variance 分母增长，而
不是更好的目标 feature 对齐。

同样先按图平均四个时刻：guided target-normalized 恶化为
`+2.031% +/- 0.403% SEM`，`16/16` 张图同向；feature variance 增加
`0.034 +/- 0.0032 SEM`，也是 `16/16` 张图同向。该符号翻转不是少数异常
样本驱动。

## Internal guidance 为什么关键

官方 RAEv2 起点的 feature variance ratio：

| t | full | base | guided 1.78 |
|---:|---:|---:|---:|
| 0.25 | 0.945 | 0.913 | 0.978 |
| 0.40 | 0.919 | 0.876 | 0.965 |
| 0.55 | 0.882 | 0.831 | 0.941 |
| 0.70 | 0.837 | 0.774 | 0.910 |

只看 full/base 会误以为 RAEv2 和旧 RAE 一样严重欠方差。但采样时的 internal
guidance 已经将方差明显推回目标。strict LPL10 后，guided ratio 在
`t=0.25/0.40` 进一步变为约 `1.005/1.001`，即低噪声处已经轻微过校准。

这解释了为什么把 LPL 限制到更低噪声不是合理补救：该区间没有明显可恢复的
guided variance 缺口。

## Decoder 逐层结果

将四个时刻合并后，比较 LPL 相对 Flow 的逐层变化：

| decoder layer | 旧 RAE target-normalized | 旧 RAE variance | RAEv2 guided target-normalized | RAEv2 guided variance |
|---:|---:|---:|---:|---:|
| 6  | -2.678% | +0.015 | +0.561% | +0.015 |
| 11 | -2.116% | +0.019 | +1.100% | +0.029 |
| 17 | -1.803% | +0.019 | +1.722% | +0.035 |
| 22 | -1.258% | +0.024 | +2.549% | +0.045 |
| 28 | +0.629% | +0.039 | +2.773% | +0.049 |

旧 RAE 在 decoder 早中层产生真实对齐改善，最后一层主要表现为方差恢复。
RAEv2 从第一层开始就没有该改善，且越靠后越明显地变成“增加方差、恶化固定
目标误差”。所以问题不是只选错了最后一层 LPL feature。

## 去掉分母捷径后的控制

在 RAEv2 guided 输出上：

- `target-normalized10` 将自己的固定目标误差降低约 `2.4%--3.1%`；
- `symmetric10` 将 symmetric proxy 降低约 `0.5%--1.0%`；
- 两者的 clean-latent error 仍分别增加约 `0.04%--0.06%` 和
  `0.15%--0.21%`。

对应的 1k 采样结果为：

| branch | FID |
|---|---:|
| Flow10 | 40.7326 |
| strict full+base10 | 40.9203 |
| symmetric10 | 41.5457 |
| target-normalized10 | 41.8816 |

这说明“增大分母”是原 strict LPL 的直接失败机制，但不是所有 decoder-aware
endpoint loss 失败的完整解释。即使显式优化固定目标误差，生成分布仍可能变差。

### Prediction-denominator detach 因果消融

进一步训练了严格的 `prediction_detach` 分支：

```text
prediction_full   = N / V_prediction
prediction_detach = N / stop_gradient(V_prediction)
```

两者前向数值完全相同，区别仅是 `prediction_detach` 不让梯度穿过预测特征
方差分母。本轮没有重新调 LPL 权重，继续使用原 full+base 分支的
`2.9384045033942286e-5`，从而只改变分母的反向路径。源点审计中 detach/full
的 clean-latent 梯度 RMS 比约为 `0.99`，因此该固定权重对照也没有引入明显
梯度量级失配。

训练从同一个官方 step `100080` checkpoint 出发，恢复相同的 online model、
EMA、GMuon、scheduler 和学习率 `2e-5`。四个 rank 的第一批图像、标签、
DINO latent、噪声、时间和 CFG mask 哈希与原 full+base 分支逐项相同。
训练使用 ImageNet-1k train、四卡全局 batch `1024`，运行 10 个 optimizer
updates。

在 16 张未见 ImageNet validation 图像、三档 noise/signal
`0.5/1.0/3.0`、base/full/guided 三种输出上，将每个条件下的相对变化再平均，
得到：

| metric | strict full+base10 vs Flow10 | detach10 vs Flow10 |
|---|---:|---:|
| latent error | +0.067% | +0.016% |
| raw feature error | -5.592% | +12.154% |
| strict feature error | -3.964% | +1.830% |
| target-normalized error | +2.307% | -2.184% |
| symmetric error | -0.902% | -0.203% |
| prediction/target variance | +5.628% | -2.851% |
| centered feature cosine | +0.088% | +0.344% |
| normalized feature mean error | -10.072% | +11.718% |

detach 的行为符合其梯度定义：它不再通过增大预测方差降低 strict 比值，固定
目标方差下的相对对齐也确实改善。但它没有让所有 feature 误差共同下降，而是
将更多绝对误差移到真实方差较大的通道，并明显增加 feature mean error。
因此 `prediction_detach` 仍不是普通 raw matching；其
`1 / V_prediction` 通道权重仍会改变误差预算。

严格同噪声、non-EMA、官方 `IG=1.78`、100-step ODE 的生成结果为：

| branch | 1k FID | 5k FID | 5k KID | 5k IS |
|---|---:|---:|---:|---:|
| Flow10 | **40.7326** | **10.8278** | 0.000167 | **153.03** |
| strict full+base10 | 40.9203 | 10.9289 | 0.000213 | 152.77 |
| prediction-detach10 | 40.8780 | 10.8518 | **0.000161** | 152.13 |

detach 将原 strict LPL 相对 Flow 的 5k FID 退化从 `+0.1010` 缩小到
`+0.0240`，即消除约 `76%` 的 FID 损失。这支持直接分母梯度是原 strict
LPL 退化的重要来源。但 detach 仍未超过 Flow；其 KID 微小优势小于 KID
估计标准差，FID 和 IS 也不支持净生成收益。

所以更准确的结论是：

> 直接 prediction-variance gradient 解释了原 strict LPL 相对 Flow 的大部分
> 额外退化；切断它可以把生成质量拉回接近 Flow，却不能把 decoder-feature
> 对齐转化成超过 Flow 的收益。

### 完全删除方差分母

为了继续区分 inverse-variance channel weighting 与普通 feature matching，
又训练了：

```text
raw = sum(mask * (F_prediction - F_target)^2)
```

该分支保留原 LPL 的五层 feature、mask、noise gate、`full+base` 目标和数据，
只删除所有 feature-variance 分母。参数梯度重新校准为 Flow 的 `20%`，正式
权重为 `8.1396425e-8`；10-step 总梯度范数为 `0.37--0.47`，没有复用早期
raw 实验中会导致 `77.4%` clipping 的无效大权重。

validation16 上，raw10 相对 Flow10：

| metric | 全部 base/full/guided 条件 | guided-only |
|---|---:|---:|
| latent error | +0.009% | +0.016% |
| raw feature error | **-8.647%** | **-0.996%** |
| strict feature error | -1.770% | -0.545% |
| target-normalized error | +0.825% | +0.516% |
| prediction/target variance | +2.147% | +0.823% |
| normalized feature mean error | -11.068% | -0.462% |

所以 raw 确实在未见图像上降低了自己声明的绝对 feature MSE，不存在 loss
没有接通的问题。但无归一化求和由大尺度 feature/channel 主导，它改善绝对
差值的同时仍会牺牲部分按目标方差衡量的相对误差。

严格同噪声的最终生成对照为：

| branch | 1k FID | 5k FID | 5k KID | 5k IS |
|---|---:|---:|---:|---:|
| Flow10 | **40.7326** | **10.8278** | **0.000167** | 153.03 |
| strict full+base10 | 40.9203 | 10.9289 | 0.000213 | 152.77 |
| prediction-detach10 | 40.8780 | 10.8518 | 0.000161 | 152.13 |
| raw full+base10 | 40.9333 | 10.8414 | 0.000176 | **153.84** |

raw 比 strict 和 detach 的 5k FID 都好，将 strict 相对 Flow 的退化消除约
`86.5%`，但仍比 Flow 高 `0.0136`。其 KID 略差于 Flow，IS 略高；这组
互有胜负的小差异不足以称为生成改善。

这个结果把机制边界进一步收紧：

> prediction-variance 分母不是必需的，删除它能消除大部分额外伤害；但
> decoder-feature 代理无论采用 strict、detach 还是 raw，都没有提供超出
> Flow 的稳定信息。问题不只在分母，而在 paired endpoint feature proxy
> 与 rollout distribution quality 本身不等价。

## 梯度与 rollout 证据

- 官方起点处，旧 RAE 的 LPL/Flow 输出梯度正向分量约从 `6.2%` 增至
  `8.4%`；RAEv2 full 约为 `4.7%--5.2%`，guided 约为
  `4.5%--4.9%`。
- 两代模型的 LPL 梯度都以 Flow 切向分量为主。因此“近乎正交”本身不能解释
  成败；区别在于旧 RAE 的切向更新改善了固定目标 feature alignment，而
  RAEv2 的切向更新没有。
- RAEv2 的 flow-parallel LPL 5k FID 为 `10.828043`，与 Flow10 的
  `10.827822` 几乎相同。去掉切向分量能消除退化，但也消除全部新增收益。
- 递归模型状态审计中，strict proxy 继续改善，raw/target-normalized 与 latent
  error 却恶化。这排除了“LPL 只在第一步或真实路径上有效”的解释。

## LPL 训练误差从哪里来

训练时先构造：

```text
x_t = (1 - t) * z + t * noise
```

模型从 `x_t` 预测 clean latent `z_hat`。LPL 比较的是：

```text
decoder_features(z_hat) 与 decoder_features(z)
```

所以误差来自有限模型在随机 bridge 状态上的 conditional prediction error。
它不是实际 ODE 采样结束后得到的 endpoint latent 与 decoder 之间的直接错配。
真实采样 endpoint 没有逐样本配对的 ground-truth `z`，两者不能混为一谈。

## 当前判断

1. `internal-guidance variance redundancy + prediction-denominator shortcut`
   解释了原 strict LPL 相对 Flow 的大部分额外退化；detach 5k 消融将该
   FID 退化缩小约 `76%`。
2. 更一般的失败机制是 `paired endpoint proxy` 与 `rollout distribution
   quality` 之间存在断层；现有 detach、flow-parallel、recursive rollout 和
   替代目标已共同支持这一点。
3. 不启动 low-noise-only strict LPL。它不满足机制门槛。
4. 若继续研究，应该直接使用真实 prior rollout endpoint：
   - 冻结成熟 Stage-2；
   - 只微调 decoder；
   - 同时保持 clean-latent reconstruction；
   - 评价 rollout endpoint 的 FID/KID 与 clean reconstruction。

这条 decoder-adaptation 实验才真正测量“prior 生成出来的 latent 与 decoder
不匹配”，不会再把 bridge 上的 conditional prediction error 当成实际采样
错配。

## 结果位置

```text
~/data/eqvae/experiments/rae_raev2_lpl_time_mechanism/
  rae_official_ep14_n16/
  raev2_official_valmatched_heads_n16/
  rae_flow_lpl_ep20_seed4101_ema_n16/
  raev2_flow_lpl_valmatched_heads_n16/
  raev2_objectives_guided_valmatched_n16/
  cross_system_valmatched_n16/
```

Prediction-denominator detach 的 checkpoint、1k/5k 样本、FID/KID/IS 和
validation16 审计位于：

```text
~/data/eqvae/experiments/raev2_prediction_detach_10step/
```

无方差归一化 raw 分支位于：

```text
~/data/eqvae/experiments/raev2_raw_10step/
```

仓库中的统一入口：

```text
experiments/run_rae_lpl_component_audit.py
experiments/run_raev2_lpl_component_audit.py
experiments/summarize_rae_raev2_lpl_time_mechanism.py
```

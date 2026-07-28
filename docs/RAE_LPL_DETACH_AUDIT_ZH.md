# RAE-LPL prediction statistics detach 审计

## 研究问题

当前 strict LPL 对每个 decoder feature channel 使用预测分支的空间方差：

\[
L_{\mathrm{full}}
=
\frac{N}{V},
\qquad
N=\sum_i(\hat f_i-f_i)^2,
\qquad
V=\operatorname{Var}(\hat f)+\epsilon.
\]

它的梯度可以精确拆成：

\[
\nabla L_{\mathrm{full}}
=
\underbrace{\frac{1}{V}\nabla N}_{g_{\mathrm{detach}}}
-
\underbrace{\frac{N}{V^2}\nabla V}_{g_{\mathrm{stats}}}.
\]

本实验比较三条梯度路径：

- `raw`：只使用 \(N\)，测量有限半径 decoder feature matching。
- `prediction-detach`：使用 \(N/\operatorname{sg}(V)\)，保留动态 inverse-variance weighting，但不通过方差分母反向传播。
- `prediction-full`：使用 \(N/V\)，再加入直接的 prediction-statistics gradient。

`detach` 不改变前向数值，只切断分母的梯度。因此：

- `raw -> detach` 反映动态 channel reweighting；
- `detach -> full` 反映直接分母梯度；
- 本审计不能替代三分支训练和生成 FID，只用于确认三种梯度在真实 RAE 状态上的大小和方向。

## 配置

- 模型：RAE DINOv2-B，冻结 ViT-XL decoder。
- Stage-2：DiT-H/S。
- checkpoint：训练前 `source`、Flow continuation 500 step、LPL continuation 500 step。
- 权重：`model`，不是 EMA；目的是检查训练梯度机制。
- 数据：ImageNet-1k validation 的 held-out latent cache。
- 样本：64 张固定图像。
- 每张图使用同一份固定高斯噪声。
- noise-to-signal ratio：`0.5, 1.0, 2.0, 3.0`。
- decoder hidden states：第 `6, 11, 17, 22, 28` 层。
- 总观测：`3 x 64 x 4 = 768`。
- 数值：全程 fp32，关闭 TF32。
- 运行：4 张 RTX 4090，约 51 秒。

结果位于：

`~/data/eqvae/experiments/rae_lpl_detach_audit/dinov2b_seed0_step500_n64_model_v1`

## 完整性检查

- 768 个预期观测全部存在。
- 每个 checkpoint、每个 noise ratio 都有 64 个样本。
- 所有数值均为有限值，无 NaN/Inf。
- 所有观测中 `prediction-full` 与 `prediction-detach` 的前向相对差都严格为 `0`。
- 单元测试验证 `prediction-full` 与现有 strict LPL 前向一致。
- `g_stats` 按定义精确取为 `g_full - g_detach`。

## 主要结果

下表先对四个 noise ratio 和 64 个样本取平均：

| checkpoint | token std ratio | centered cosine | raw feature loss | normalized loss | stats/full RMS | full-detach cosine |
|---|---:|---:|---:|---:|---:|---:|
| source | 0.9220 | 0.7911 | 27363 | 644.6 | 0.409 | 0.893 |
| Flow-500 | 0.9231 | 0.7901 | 27962 | 647.1 | 0.405 | 0.895 |
| LPL-500 | 0.9898 | 0.7933 | 32120 | 565.4 | 0.367 | 0.924 |

这里的 `token std ratio` 是单张图、单 channel、跨空间 token 的标准差比，再做 channel/layer 几何平均。它不是跨数据集方差，也不是生成样本 covariance。

### 1. 直接分母梯度不是可忽略的小扰动

在 source checkpoint 上：

- `stats gradient RMS / full gradient RMS = 0.409`；
- `cos(g_full, g_detach) = 0.893`。

因此直接分母梯度具有实质量级，也确实改变了 clean-latent 输出梯度方向。但 `0.409` 不是“最终参数更新的 40.9%”：Stage-2 Jacobian、Adam、clipping 和梯度累积都会进一步变换它。

### 2. 直接分母梯度稳定推动预测 feature 方差增大

定义：

\[
\cos(-g_{\mathrm{stats}}, \nabla_{\hat z}\log V).
\]

它在 source 上平均为 `0.588`，在全部 768 个观测中有 766 个为正。分 noise ratio 的 source 均值为：

| noise ratio | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---:|---:|---:|---:|
| stats descent vs log-variance cosine | 0.570 | 0.590 | 0.595 | 0.598 |

这确认了数学推导在真实 decoder/latent 上不是退化效应：`g_stats` 确实提供了稳定的直接反收缩方向。

### 3. detach 本身没有同样的直接方差推动

source 上：

- detach 下降方向与 log-variance 梯度的平均余弦为 `-0.015`；
- full 的对应值为 `+0.223`。

也就是说，在 clean-latent 输出空间的一阶近似下，加入分母梯度把整体 LPL 更新从“平均近似中性”翻成了“倾向增加 token feature variance”。detach 仍可能通过 feature residual、decoder Jacobian和共享参数间接改变方差，所以不能把它解释成“完全不校准方差”。

### 4. inverse-variance weighting 本身也明显改变梯度

source 上：

\[
\cos(g_{\mathrm{raw}}, g_{\mathrm{detach}})=0.693.
\]

因此 `raw -> detach` 不是简单缩放。不同 channel 的 \(1/V\) 权重显著改变了更新方向。即使后续训练证明 full 与 detach 的 FID 接近，也只能说明直接分母梯度不是主要原因，不能否定 prediction normalization 整体的作用。

### 5. LPL-500 主要表现为尺度校准，不是 raw feature matching 改善

相对 source，LPL-500：

- token standard-deviation ratio：`0.9220 -> 0.9898`；
- normalized loss：`644.6 -> 565.4`；
- centered cosine：`0.7911 -> 0.7933`，变化很小；
- raw feature loss：`27363 -> 32120`，反而更高；
- clean latent MSE：`0.1121 -> 0.1170`，也略高。

按样本聚类的 paired bootstrap 显示：

- std ratio 增量 `+0.0677`，95% CI `[+0.0644, +0.0712]`；
- normalized loss 增量 `-79.3`，95% CI `[-90.8, -68.1]`；
- raw feature loss 增量 `+4757`，95% CI `[+2345, +7424]`。

因此当前证据不支持“LPL 只是让预测 latent 的 decoder features 在普通欧氏误差下更接近 target”。它更像在优化预测尺度相关的 normalized objective。该变化是否造成已观察到的 FID 改善，仍需要训练干预。

### 6. 校准具有层间差异

LPL-500 的五层 variance ratio 分别为：

`0.947, 0.937, 0.954, 1.002, 1.069`。

前四层总体靠近 1，最后一层略有 overshoot。不能把当前结果描述成所有层都被统一拉到完全相同的尺度。

## 当前结论

已经确认：

1. strict LPL 中存在非平凡的直接 prediction-variance gradient。
2. 该分量在真实 DINOv2-B decoder 上稳定推动单图、单 channel 的空间 feature variance 增大。
3. inverse-variance weighting 和直接分母梯度都显著改变 clean-latent 输出梯度。
4. LPL-500 的主要可见变化是 token feature scale calibration；它并未改善 raw decoder feature error。

尚未确认：

1. 直接分母梯度是否是 FID 改善的主要原因。
2. detach 分支经过 Stage-2 参数 Jacobian、Adam 和 gradient clipping 后的实际更新如何。
3. token-level feature variance calibration 是否会改善最终生成分布的 covariance、FID 或 recall。
4. raw、detach、full 三条训练分支各自贡献多少生成收益。

## 下一步判别实验

从同一个 source checkpoint、optimizer state、数据顺序和噪声开始，分别训练：

1. Flow continuation；
2. Flow + raw feature loss；
3. Flow + prediction-detach；
4. Flow + prediction-full。

先做 100/500 step 单 seed 筛选，并记录实际 Adam update、clipping 前后梯度、function drift 和 token variance；只有候选差异清楚后才做多 seed 5k/50k FID。这个实验才真正回答：

- `raw -> detach` 的动态 channel budgeting 是否贡献生成收益；
- `detach -> full` 的直接方差梯度是否贡献生成收益。

## 大规模 DINOv2-B detach 训练

### 实验设计

在零训练审计通过后，进行了大规模 `prediction-detach` 训练：

- 模型：DINOv2-B RAE + DiTDH-S；
- 数据：ImageNet-1k train；
- seeds：`0, 1, 2, 3`；
- 每个 seed：2,000 optimizer updates；
- 每个分支：2 GPU，micro batch 2，gradient accumulation 4；
- global batch：16；
- raw/full/detach 共用相同的 time gate、五层 decoder features 和样本上限；
- detach 权重：`0.0006749372833287125`，与原 full LPL 完全相同；
- source：同一个官方权重对应的 full-state checkpoint；
- encoder/decoder 冻结，只有 Stage-2 velocity model 更新；
- fp32，关闭 TF32。

两个 detach seed 各使用两张 GPU 并行训练，从而同时使用四张 GPU，并保持原
DINOv2-B 主实验的 `world_size=2` 数据划分和随机数规则。

结果目录：

`~/data/eqvae/experiments/rae_lpl_detach_factorial`

### 校准与 smoke

detach 在 source 上的推荐权重为：

`0.0006749372812837192`

原 full LPL 权重为：

`0.0006749372833287125`

两者只相差约 \(2\times10^{-12}\)。1-step smoke 中：

- detach 与旧 full 的总前向 loss 都为 `0.5230721421539783`；
- detach gradient norm 为 `0.4188`；
- full gradient norm 为 `0.4878`。

新分支与旧 Flow/full 的第一批 image、label、time、noise、noisy latent、
target velocity、initial prediction 和 flow loss 哈希全部一致。

### 完整性

四个 detach seed 均满足：

- `metrics.jsonl` 恰有 2,000 条连续记录；
- step 范围为 `1...2000`；
- 所有训练数值有限；
- step-2000 checkpoint 可完整加载；
- checkpoint 中有 190 个 model tensors；
- model 和 EMA tensors 均无 NaN/Inf；
- `checkpoint.step=2000` 且 `branch_start_step=0`；
- 单个完整 checkpoint 约 2.93 GiB。

### 训练指标

旧 seed 0 的 Flow/full 曾从本地 step-500 checkpoint 恢复，而新 detach seed 0
是连续训练。当前恢复逻辑不会恢复 epoch 内的 dataloader cursor，因此 seed 0
不能作为严格相同训练流的 detach/full 机制对照。下面的主表只使用从 source
连续训练到 2,000 steps 的 seeds `1, 2, 3`：

| objective | mean flow loss | mean feature contribution | mean grad norm | clip rate |
|---|---:|---:|---:|---:|
| Flow | 0.43680 | 0 | 0.3676 | 0.15% |
| prediction-detach | 0.43859 | 151.52 | 0.5704 | 3.15% |
| prediction-full | 0.43954 | 130.09 | 0.5206 | 1.75% |

这个结果呈现出清楚的训练侧分工：

- detach 比 full 更接近原 Flow loss；
- full 比 detach 更有效降低 prediction-normalized decoder feature loss；
- full 和 detach 前向目标相同，差异只能来自 prediction variance denominator
  是否参与反向传播及其后续 optimizer 动态。

### 最终参数更新几何

对 seeds `1, 2, 3`，相对同一个 source 比较 step-2000 参数更新：

- model update norm 的 `detach/full` 比例约为 `0.9903`；
- model update cosine 约为 `0.8341`；
- EMA update norm 的 `detach/full` 比例约为 `0.9880`；
- EMA update cosine 约为 `0.8294`。

因此，detach 与 full 的总参数漂移大小几乎相同，但方向相差约 33 度。直接
denominator gradient 的作用不是简单增加训练步长，而是实质改变 optimizer
最终走向。

seed 0 的 detach/full update cosine 只有 `0.11`，已确认其旧 full 分支从
step-500 恢复，而新 detach 连续运行。它被保留作训练稳定性补充，不纳入上述
严格配对均值。

### raw 分支为何暂停

raw feature loss 在相同 source calibration protocol 下得到：

- mean unweighted contribution：`6455.07`；
- loss-matched weight：`1.6455225e-5`。

但该权重在前 84 steps 中产生：

- mean grad norm：`2.08`；
- median grad norm：`1.63`；
- maximum grad norm：`7.35`；
- clip rate：`77.4%`。

同期 full 的 clip rate 约 `1.2%`。因此继续训练该 raw 分支会把“去掉
inverse-variance weighting”的效果与大规模 gradient clipping 混在一起。该
分支在 step 84 主动停止；后续必须先做 optimizer-update matched calibration，
不能把当前 raw 结果用于生成机制结论。

## 实现与协议终审

为排除统一训练脚本本身造成的混杂，使用历史 seed 1 full LPL 的同一 source
checkpoint、随机种子、2-rank 数据流、超参数和目标，重新运行当前 `full`
实现 10 步。新旧 step-10 checkpoint 的比较结果为：

- `model` 的 190 个 tensors 全部 bitwise identical；
- `EMA` 的 190 个 tensors 全部 bitwise identical；
- optimizer 和 scheduler state 完全一致。

这比只比较 loss 更严格，说明当前统一脚本精确复现了历史 full 分支。当前
full/detach 差异可以归因于 prediction variance denominator 是否参与反向传播。

采样固定为：

- EMA checkpoint；
- Euler 50 steps，CFG 1；
- seed `20260715`；
- ImageNet 1000 类 fixed equal-label sampling mode；
- 4 processes，每 process batch 4；
- fp32，关闭 TF32。

相同 seed 但改变 process 数或 per-process batch 会改变 label/noise 的分配，
因此不属于严格配对。评估入口现已拒绝其他 process/batch 组合。若采样目录中
只有部分 PNG、尚无最终 NPZ，也会拒绝继续，因为旧采样器跳过已有文件时不会
同步恢复被跳过样本对应的 CUDA RNG 消耗；严格实验必须清空该部分目录后重跑。

训练恢复同样增加了保护：本地中途 checkpoint 未保存 dataloader cursor 和
每-rank RNG，默认拒绝把它当作 exact resume；只有显式
`--allow-nonexact-resume` 才允许诊断性恢复。

这里还有一个不会破坏配对、但必须准确记录的旧采样器细节：5,000 不能被
固定 global sampling batch 16 整除，因此采样器先生成 5,008 张，再按文件
编号取前 5,000 张组成 NPZ。旧实现先按 rank 切分 label pool、再交错写盘，
所以 5k NPZ 不是每类严格 5 张：实际有 12 个类别偏离目标计数，6 个样本
相当于从某些类别替换到另一些类别，占 `0.12%`。所有 Flow/detach/full
分支使用完全相同的 label/noise pairing，因此该细节不影响分支间因果比较。
50,000 可被 16 整除，50k 终验是每类严格 50 张。评估结果现会显式记录
`class_balance_exact` 和 `samples_generated_before_trim`，避免以后把 5k
误写成严格类别均衡。

## 三种子 5k 生成结果

严格配对的 step-2000 结果如下。FID/KID 越低越好，IS 越高越好：

| seed | Flow FID | detach FID | full FID | detach 保留的 full FID 收益 |
|---:|---:|---:|---:|---:|
| 1 | 22.0543 | 21.1901 | 19.4700 | 33.4% |
| 2 | 21.9830 | 21.1322 | 19.5437 | 34.9% |
| 3 | 22.4084 | 21.2199 | 20.0234 | 49.8% |
| mean | 22.1486 | 21.1808 | 19.6790 | 39.2% |

三种子均满足 `full < detach < Flow`，不存在只由单个 seed 驱动的排序。

| objective | FID mean ± sd | KID mean ± sd | IS mean ± sd |
|---|---:|---:|---:|
| Flow | 22.1486 ± 0.2278 | 0.007280 ± 0.000222 | 69.0761 ± 0.4565 |
| prediction-detach | 21.1808 ± 0.0446 | 0.006659 ± 0.000047 | 69.9130 ± 0.6271 |
| prediction-full | 19.6790 ± 0.3005 | 0.006006 ± 0.000198 | 71.7810 ± 1.5160 |

相对 Flow：

- detach 平均改善 FID `0.9678`；
- full 平均改善 FID `2.4696`；
- detach 保留 full 总收益的 `39.2%`；
- full 的 differentiable denominator 额外贡献 `1.5017` FID，即总收益的
  `60.8%`。

因此两部分都有效，但贡献不同：

1. `N / sg(V)` 的 inverse-variance channel weighting 本身有稳定收益；
2. `-N/V^2 * grad(V)` 不是无关副作用，在当前 DINOv2-B RAE 上贡献了多数
   full LPL 收益；
3. 这仍是因果消融意义上的“分母梯度贡献”，不能进一步偷换成它单独改善了
   某一种生成统计量的结论。

完整汇总：

`~/data/eqvae/experiments/rae_lpl_detach_factorial/generation_5k_seed1_seed2_seed3_detach.json`

## 当前大规模结论

已经确认：

1. differentiable denominator 显著改变最终参数更新方向，而不是只改变范数；
2. full 更有效优化 normalized decoder-feature objective；
3. detach 略少损害原 flow objective；
4. detach 和 full 均改善生成质量，但 full 在三个独立 seed 上都明显更强；
5. 当前证据支持“inverse-variance weighting 有用，直接 denominator gradient
   贡献更大”，而不是二选一。

## DINOv2-B 50k 终验

预先固定的 seed 3 生成了 50,000 张类别均衡样本。NPZ 已核验为精确的
`(50000, 256, 256, 3)`、`uint8`，像素范围为 `[0, 255]`。

| objective | torch FID ↓ | ADM FID ↓ | ADM sFID ↓ | ADM IS ↑ |
|---|---:|---:|---:|---:|
| Flow | 16.7410 | 13.5043 | 12.9084 | 93.7710 |
| prediction-detach | 15.6331 | 12.4540 | 12.2836 | 96.5625 |
| prediction-full | 14.3156 | 11.1027 | 9.2378 | 98.1204 |

50k ADM 上仍满足 `full < detach < Flow`。以 ADM FID 计：

- detach 相对 Flow 改善 `1.0503`；
- full 相对 Flow 改善 `2.4016`；
- detach 保留 full 收益的 `43.7%`；
- differentiable denominator 额外贡献 `1.3513`，约占 full 收益的
  `56.3%`。

这个比例与三种子 5k 的 `39.2% / 60.8%` 接近。因此，小样本 FID 方差不能
解释 DINOv2-B 上的结论：inverse-variance weighting 和直接 denominator
gradient 都有效，后者贡献更大。

ADM 结果：

`~/data/eqvae/experiments/rae_lpl_detach_factorial/generation_50k_seed3_detach_adm.json`

## MAE-B 跨 tokenizer 扩展

使用官方 MAE-B RAE、ViT-XL decoder 和 DiT-XL epoch-80 model-only 权重。
三分支从相同模型参数和相同的新 optimizer/scheduler 状态出发；每个 seed
训练 500 updates，global batch 16。detach 与 full 使用完全相同的
`0.001310218123057881` 权重。

首批 image、label、time、noise、noisy latent、target velocity 和 initial
prediction 指纹与历史 Flow/full 逐项一致。detach 与 full 的首步
flow/LPL/total forward loss 也完全一致，差异只在反向梯度。

三种子 5k 结果：

| seed | Flow FID ↓ | detach FID ↓ | full FID ↓ |
|---:|---:|---:|---:|
| 3407 | 17.6496 | 17.5372 | 16.9699 |
| 3408 | 17.7907 | 17.6810 | 17.0663 |
| 3409 | 17.8310 | 17.7009 | 17.0908 |
| mean | 17.7571 | 17.6397 | 17.0424 |

| objective | mean FID ↓ | mean KID ↓ | mean IS ↑ |
|---|---:|---:|---:|
| Flow | 17.7571 | 0.003933 | 77.7073 |
| prediction-detach | 17.6397 | 0.003890 | 78.1980 |
| prediction-full | 17.0424 | 0.003579 | 79.0248 |

MAE-B 三个 seed 同样全部满足 `full < detach < Flow`，且 KID、IS 也同向。
但作用比例与 DINOv2-B 不同：

- detach 平均仅改善 FID `0.1174`；
- full 平均改善 FID `0.7147`；
- detach 只保留 full 收益的 `16.4%`；
- differentiable denominator 额外贡献 `0.5974`，占 full 收益的 `83.6%`。

因此，直接 denominator gradient 的重要性跨 tokenizer 泛化，但其贡献比例
不是常数。当前 MAE-B 证据甚至比 DINOv2-B 更依赖该梯度路径。

MAE 汇总：

`~/data/eqvae/experiments/rae_lpl_detach_cross_tokenizer/generation_5k_mae_seed3407_seed3408_seed3409_detach.json`

## SigLIP2-B 跨 tokenizer 扩展

SigLIP2-B 使用官方 ViT-XL decoder 和 DiT-XL epoch-80 model-only 权重。
detach 与 full 共享同一新 optimizer/scheduler、数据流和
`0.0009191518565962038` 权重。三个 seed 的 13 个非 objective 首批指纹
均与历史 full 分支精确一致；每个分支都有 500 条连续且有限的 metrics、
step-500 model/EMA checkpoint，所有参数有限。

三种子 5k 结果：

| seed | Flow FID ↓ | detach FID ↓ | full FID ↓ |
|---:|---:|---:|---:|
| 3407 | 13.7162 | 13.6237 | 13.3887 |
| 3408 | 13.7370 | 13.6633 | 13.4065 |
| 3409 | 13.6789 | 13.6335 | 13.3509 |
| mean | 13.7107 | 13.6402 | 13.3821 |

| objective | mean FID ↓ | mean KID ↓ | mean IS ↑ |
|---|---:|---:|---:|
| Flow | 13.7107 | 0.001418 | 112.0641 |
| prediction-detach | 13.6402 | 0.001405 | 112.9537 |
| prediction-full | 13.3821 | 0.001325 | 114.0753 |

SigLIP2-B 仍是 3/3 seeds 的 FID/KID/IS 同向改善，并全部满足
`full < detach < Flow`。以平均 FID 计：

- detach 相对 Flow 改善 `0.0706`；
- full 相对 Flow 改善 `0.3287`；
- detach 保留 full 收益的 `21.5%`；
- differentiable denominator 额外贡献 `0.2581`，占 full 收益的
  `78.5%`。

SigLIP2 汇总：

`~/data/eqvae/experiments/rae_lpl_detach_cross_tokenizer/generation_5k_siglip2_seed3407_seed3408_seed3409_detach.json`

## 官方原模型的严格同协议基线

为把官方原模型提升到与 Flow/detach/full 相同的比较层级，对三个未经继续
训练的官方 Stage-2 checkpoint 重新执行了固定 5k 采样：

- Euler 50 steps，CFG 1；
- seed `20260715`；
- 4 processes，每 process batch 4；
- fp32，关闭 TF32；
- 相同 ImageNet reference、metric batch 64 和 KID RNG seed。

旧 DINOv2 官方样本与严格重采样的 5,008 张 PNG、5k NPZ 均逐字节一致。
旧 MAE/SigLIP2 官方目录各有 5,056 张 PNG，与当前 global batch 16 协议
不一致；因此为两者生成了新的 5,008 张严格样本并重算全部指标。

下表将官方零更新模型与三种子均值并列。`ΔFID` 定义为
`branch FID - official FID`，负数为改善：

| tokenizer | objective | FID ↓ | KID ↓ | IS ↑ | ΔFID | 超过官方的 seeds |
|---|---|---:|---:|---:|---:|---:|
| DINOv2-B | 官方原模型 | 21.6122 | 0.006439 | 71.8754 | 0 | baseline |
| DINOv2-B | Flow | 22.1486 | 0.007280 | 69.0761 | +0.5363 | 0/3 |
| DINOv2-B | detach | 21.1808 | 0.006659 | 69.9130 | -0.4315 | 3/3 |
| DINOv2-B | full | **19.6790** | **0.006006** | 71.7810 | **-1.9332** | **3/3** |
| MAE-B | 官方原模型 | 17.5508 | 0.003593 | **80.2416** | 0 | baseline |
| MAE-B | Flow | 17.7571 | 0.003933 | 77.7073 | +0.2063 | 0/3 |
| MAE-B | detach | 17.6397 | 0.003890 | 78.1980 | +0.0889 | 1/3 |
| MAE-B | full | **17.0424** | **0.003579** | 79.0248 | **-0.5085** | **3/3** |
| SigLIP2-B | 官方原模型 | 13.5717 | 0.001379 | **117.8009** | 0 | baseline |
| SigLIP2-B | Flow | 13.7107 | 0.001418 | 112.0641 | +0.1390 | 0/3 |
| SigLIP2-B | detach | 13.6402 | 0.001405 | 112.9537 | +0.0685 | 0/3 |
| SigLIP2-B | full | **13.3821** | **0.001325** | 114.0753 | **-0.1896** | **3/3** |

这个基线带来两个比 `full < detach < Flow` 更严格的结论：

1. 普通 Flow 的九个 seed 没有一个超过官方 FID；短程 latent MSE 继续训练
   本身整体上使生成质量退化。
2. full LPL 的九个 seed 全部超过官方 FID，而 detach 只有 4/9。特别是
   MAE/SigLIP2 上，detach 主要是在缓解 Flow 退化，只有可微 denominator
   gradient 加回后，三种子均值和每个 seed 才都超过官方起点。

因此，直接 denominator gradient 的证据不再只是“full 比 detach 好”，还
包括“它使 MAE/SigLIP2 从未能稳定超过官方，跨过到 3/3 seed 超过官方”。
但 full 并非全面支配官方：MAE/SigLIP2 的 IS 仍低于官方，当前绝对比较也
只是 5k proxy。

严格官方结果：

`~/data/eqvae/experiments/rae_strict_lpl_official_source_5k/official_source_strict_5k.json`

统一汇总：

`~/data/eqvae/experiments/rae_lpl_detach_cross_tokenizer/official_flow_detach_full_strict_5k_summary.json`

## 跨 tokenizer 参数更新几何

相对每个 tokenizer 的同一官方 source，比较各自预注册终点的 full 与
detach 参数更新：DINOv2-B 为 step 2000，MAE-B/SigLIP2-B 为 step 500。
下面对每个 tokenizer 的三个严格配对 seed 取平均：

| tokenizer | state | detach/full update norm | update cosine |
|---|---|---:|---:|
| DINOv2-B | model | 0.9903 | 0.8341 |
| DINOv2-B | EMA | 0.9880 | 0.8294 |
| MAE-B | model | 0.9690 | 0.8985 |
| MAE-B | EMA | 0.9662 | 0.8866 |
| SigLIP2-B | model | 0.9856 | 0.8070 |
| SigLIP2-B | EMA | 0.9825 | 0.8080 |

三种 tokenizer 上，detach/full 的总漂移范数都很接近，但方向稳定分开。
同时：

- MAE detach 的平均 gradient norm 为 `0.3133`，full 为 `0.3040`；
- SigLIP2 detach 的平均 gradient norm 为 `0.5652`，full 为 `0.4983`；
- SigLIP2 detach 的 clip rate 为 `3.8%`，full 为 `2.13%`。

因此 full 的更好生成质量不能归因为更大的 raw gradient norm、更多 clipping
或更大的总参数漂移。更符合证据的解释是：prediction variance denominator
的直接梯度改变了更新方向，并更有效地优化了冻结 decoder 所定义的归一化
feature geometry。

## 最终机制结论

当前因果证据支持把原始 LPL 分成两项：

\[
L_{\mathrm{detach}} = \frac{N}{\operatorname{sg}(V)},
\qquad
g_{\mathrm{stats}} = -\frac{N}{V^2}\nabla V.
\]

在三种公开 RAE tokenizer、九个严格配对 seed 中：

1. detach 相对 Flow 的 FID 全部改善，说明 inverse-variance weighting 本身
   有稳定生成价值；
2. full 相对 detach 的 FID 全部进一步改善，说明直接 denominator gradient
   也有稳定生成价值；
3. DINOv2-B、MAE-B、SigLIP2-B 中，detach 分别保留 full FID 收益的约
   `39%`、`16%`、`21%`；
4. 50k DINOv2 ADM 复验给出 `43.7%`，与其 5k 比例一致；
5. full 的额外收益跨 tokenizer 都占多数，但强度依赖 tokenizer；
6. 对严格同协议官方起点，Flow、detach、full 分别有 `0/9`、`4/9`、
   `9/9` 个训练 seed 改善 FID。

可以可靠地说：

> 原始 LPL 不只是 decoder feature matching，也不只是固定的
> inverse-variance channel weighting。其可微 prediction-statistics 路径是
> 跨 RAE tokenizer 可复现的主要有效组成部分。

这里的“保留比例”和“额外占比”都是嵌套干预
`Flow -> detach -> full` 上的 FID contrast ratio。由于两种梯度会通过
Adam、clipping 和后续轨迹相互作用，它不是两个可独立相加机制的 Shapley
归因，也不表示单独加入 `g_stats` 必然获得完全相同的 FID 差值。

暂时仍不能说：

1. “增大 feature variance”本身就是 FID 改善的唯一原因；
2. 分母梯度在所有训练阶段、所有 decoder 层贡献相同；
3. 当前比例可直接外推到更大 prior、不同训练时长或其他生成目标；
4. full LPL 已是最优形式。

最有价值的下一项机制实验不再是重复更多同配置 seed，而是对
`g_stats` 做受控方向干预：保持相同 update norm，比较完整
`g_stats`、仅保留其 feature-variance 平行分量、以及去掉该平行分量的正交
分量。这样才能区分“方差校准”与“分母梯度携带的其他 decoder-Jacobian
信息”。

# RAE 上 LPL 实验全量台账

## 1. 文档范围

本文整理截至 `2026-07-26` 仓库中所有与“把 LPL 应用于 RAE”直接相关的
实验、对照、诊断和未运行方案。

本文只记录：

- 实验做了什么；
- 使用了什么模型、数据、优化器、种子和评价协议；
- 哪些实验已经完成，哪些只是 smoke，哪些尚未运行；
- 实验中实际出现的数值和现象；
- 对应代码、checkpoint、表格和图所在位置。

本文不讨论原因，不给出机制判断，也不提出下一步建议。

状态定义：

| 状态 | 含义 |
|---|---|
| `完成` | 预定训练或评价已经结束，结果文件存在 |
| `诊断完成` | 不训练新生成模型，只对已有 checkpoint 做测量 |
| `Smoke` | 只验证接口、显存、梯度或单步运行 |
| `未运行` | 只有设计、协议或代码入口，没有正式结果 |

本机路径说明：

- `/home/zhoushunyu/data` 解析到 `/data/users/zhoushunyu`。
- 本文优先使用 `/home/zhoushunyu/data/...` 书写实验资产路径。
- ImageNet-1K 数据位于 `/data/shared/imagenet-1k`。
- ADM ImageNet 256 reference 位于
  `/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz`。

## 2. 实验总索引

| ID | 状态 | 模型或内容 | 规模 | 主要产物 |
|---|---|---|---|---|
| LPL-00 | `Smoke` | DINOv2-B 原始权重 `3` 与校准权重 | 1 update | 单步 loss、梯度和显存验证 |
| LPL-01 | `完成` | RAE-DINOv2-B 严格 Flow/LPL 对照 | 4 seeds，2,000 updates | held-out、5k、50k |
| LPL-02 | `完成` | 官方 Stage-2 权重直接采样 | 3 tokenizer，各 5k | 官方起点 5k |
| LPL-03 | `完成` | RAE-MAE-B 严格 Flow/LPL 对照 | 3 seeds，500 updates | 5k、50k |
| LPL-04 | `完成` | RAE-SigLIP2-B 严格 Flow/LPL 对照 | 3 seeds，500 updates | 5k、50k |
| LPL-05 | `完成` | Flow 额外计算预算对照 | 2 tokenizer，各 1 seed | 600-update 5k |
| LPL-06 | `诊断完成` | DINOv2-B 局部误差方向 | 4 seeds，512 observations | 局部放大、有限差分 |
| LPL-07 | `诊断完成` | DINOv2-B 有限半径扫描 | 4 seeds，5 个半径 | raw/strict/归一化分解 |
| LPL-08 | `诊断完成` | 常数 decoder metric 代理 | train 1,024 / val 2,048 | Spearman、梯度 cosine |
| LPL-09 | `诊断完成` | predictability、atlas、decoder prefix 代理 | 32 score / 8 gradient | proxy 对完整 LPL 的接近程度 |
| LPL-10 | `完成，无 FID` | RAE-DINOv2-L 短程 pilot | 3 seeds，250+100 updates | 96 observations、幅度扫描 |
| LPL-11 | `未运行` | target/symmetric normalization 训练消融 | 无 | 只有设计 |
| LPL-12 | `未运行` | variance-only LPL 训练消融 | 无 | 只有设计 |
| LPL-13 | `未运行` | Decoder Metric Preconditioning | 无 | 只有设计和静态代理诊断 |
| LPL-14 | `未运行` | 联合训练 encoder/decoder、rollout LPL | 无 | 未启动 |

## 3. LPL 的统一实现

### 3.1 被更新与冻结的模块

正式 Flow/LPL 训练中：

- RAE encoder：冻结，`eval()`；
- RAE decoder：冻结，`eval()`；
- RAE decoder 为确定性前向，`noise_tau=0`；
- 只更新 Stage-2 latent velocity model；
- LPL target decoder features 停止梯度；
- LPL prediction decoder features 保留对输入 latent 的梯度；
- 梯度经过冻结 decoder 回传到 Stage-2，decoder 参数本身不更新。

### 3.2 Flow 路径和 clean latent 预测

采用线性 OT 路径：

```text
z_t = (1 - t) z_0 + t epsilon
target_velocity = epsilon - z_0
z_hat_0 = z_t - t * v_theta(z_t, t)
```

Flow 分支只使用 velocity MSE。LPL 分支使用：

```text
total_loss = flow_loss + lpl_weight * gated_lpl
```

### 3.3 Decoder feature loss

- decoder 深度：28；
- 取层：`[6, 11, 17, 22, 28]`；
- 对应约 `20% / 40% / 60% / 80% / 100%` decoder 深度；
- 五层 feature map 都是 `16 x 16`；
- 五层权重均为 `1.0`；
- target 和 prediction 共用 prediction 分支的 channel mean/std；
- channel normalization epsilon：`1e-6`；
- outlier quantile：`0.02`；
- morphology opening kernel：`5`；
- morphology closing kernel：`3`。

### 3.4 时间门控

门控条件：

```text
t / (1 - t) <= 3
```

训练时每个 rank 最多选择 1 个 eligible 样本计算 LPL。若一个 local batch
中存在多个 eligible 样本，代码使用比例修正保持 gated LPL 的期望贡献。

### 3.5 统一数值设置

- 精度：fp32；
- autocast：关闭；
- TF32：关闭；
- flash SDP：关闭；
- memory-efficient SDP：关闭；
- math SDP：开启；
- deterministic algorithms：开启；
- gradient clipping：`1.0`；
- EMA：`0.9995`。

### 3.6 统一优化器

- optimizer：AdamW；
- learning rate：`2e-5`；
- betas：`(0.9, 0.95)`；
- weight decay：`0`；
- epsilon：`1e-8`。

DINOv2-B 配置的 scheduler：

- warmup：`0`；
- decay end：`6255`；
- base learning rate：`2e-5`；
- final learning rate：`2e-5`。

MAE-B 和 SigLIP2-B 配置的 scheduler：

- warmup：`0`；
- decay end：`500`；
- base learning rate：`2e-5`；
- final learning rate：`2e-5`。

由于 base 和 final learning rate 相同，上述运行中的实际 learning rate 保持
`2e-5`。

### 3.7 数据和图像变换

- 训练数据：ImageNet-1K train parquet；
- held-out：ImageNet-1K validation；
- 训练图像：center crop 到 `256 x 256`；
- augmentation：random horizontal flip；
- 输入：`ToTensor()`；
- LPL 权重校准只读取 train；
- held-out 和生成评价读取 validation/reference；
- validation 不参与权重更新。

### 3.8 统一采样

- 使用 EMA 参数；
- Euler ODE；
- 50 sampling steps；
- CFG：`1.0`；
- sampling seed：`20260715`；
- ImageNet 类别均衡采样；
- fp32；
- TF32 关闭。

5k 的 FID、KID、IS 使用 `torch-fidelity`。50k 同时记录：

- `torch-fidelity` 的 FID、KID、IS；
- ADM/guided-diffusion TensorFlow Inception 的 FID、sFID、IS。

## 4. LPL-00：初始 Smoke 和权重校准

### 4.1 DINOv2-B 原始权重 smoke

状态：`Smoke`。

直接使用 LPL 论文中的权重 `3.0`，运行 1 update：

| 项目 | 数值 |
|---|---:|
| flow loss | 0.42213 |
| raw LPL | 101.69379 |
| total loss | 305.50348 |
| eligible fraction | 0.125 |
| gradient norm | 1663.879 |

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/smoke_lpl_seed0_s1
```

### 4.2 DINOv2-B 校准权重 smoke

使用固定 train 样本校准后的权重：

```text
lpl_weight = 0.0006749372833287125
```

同一首批数据运行 1 update：

| 项目 | 数值 |
|---|---:|
| flow loss | 0.42213 |
| raw LPL | 101.69379 |
| total loss | 0.49076 |
| eligible fraction | 0.125 |
| gradient norm | 0.478 |

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/smoke_lpl_calibrated_seed0_s1
```

### 4.3 正式校准记录

| 模型 | 校准样本 | mean flow | raw gated LPL | eligible | selected | 固定权重 |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2-B | 256 | 0.424879 | 157.3771 | 20.3125% | 52 | 0.0006749373 |
| MAE-B | 256 | 0.635879 | 121.3308 | 19.9219% | 51 | 0.0013102181 |
| SigLIP2-B | 256 | 0.569890 | 155.0044 | 19.9219% | 51 | 0.0009191519 |
| DINOv2-L | 128 | 1.362761 | 730.3512 | 17.9688% | 23 | 0.0004664744 |

前三组权重的目标是让初始 `weighted LPL / Flow = 0.25`。

校准文件：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/calibration_source_seed0_n64/lpl_calibration.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl_cross_tokenizer/mae_calibration_train256_seed3407/lpl_calibration.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl_cross_tokenizer/siglip2_calibration_train256_seed3407/lpl_calibration.json
/home/zhoushunyu/data/eqvae/experiments/rae_dinov2_large/pilot/calibration_step250/lpl_calibration.json
```

## 5. LPL-01：DINOv2-B 严格 Flow/LPL 对照

### 5.1 模型配置

- Stage-1：DINOv2-with-registers-base encoder；
- encoder input：`224 x 224`；
- latent：`[B, 768, 16, 16]`；
- latent 使用 RAE normalization；
- decoder：ViT-XL n08，28 层；
- Stage-2：`DiTwDDTHead`，记录名 `DiTDH-S`；
- Stage-2 input channels：768；
- patch size：1；
- hidden sizes：`[384, 2048]`；
- depths：`[12, 2]`；
- heads：`[6, 16]`；
- MLP ratio：4；
- class dropout：0.1；
- classes：1,000；
- rope、RMSNorm、SwiGLU、position embedding：开启。

源 full-state checkpoint：

```text
/home/zhoushunyu/data/eqvae/stage2_training/finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu/checkpoints/ep-0000000.pt
```

该 checkpoint 中 190 个 model tensors 与以下官方权重逐项相同：

```text
/home/zhoushunyu/data/eqvae/models/RAE/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep14/stage2_model.pt
```

### 5.2 训练配置

- train seeds：`0, 1, 2, 3`；
- 每个 seed：Flow 和 LPL 各 2,000 updates；
- world size：2；
- micro batch：2；
- gradient accumulation：4；
- global batch：16；
- 每对实验从相同完整状态和相同随机数据流开始。

每个 seed 的 Flow/LPL pair fingerprint 均核对：

- image；
- class label；
- time；
- noise；
- noisy latent；
- target velocity；
- initial prediction。

主目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl
```

### 5.3 Held-out decoder feature 结果

固定 128 张 ImageNet validation 图像，固定噪声，使用 EMA。

| step | noise/signal | Flow decoder LPL | LPL decoder LPL | Flow latent MSE | LPL latent MSE |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.5 | 281.5501 | 270.5122 | 近似相同 | 近似相同 |
| 500 | 1.0 | 502.8955 | 476.1477 | 近似相同 | 近似相同 |
| 500 | 2.0 | 815.2498 | 768.1400 | 近似相同 | 近似相同 |
| 500 | 3.0 | 1063.2202 | 1006.2104 | 近似相同 | 近似相同 |
| 2000 | 0.5 | 281.0503 | 255.7828 | 0.0568701 | 0.0574161 |
| 2000 | 1.0 | 501.7172 | 439.6366 | 0.0904670 | 0.0914707 |
| 2000 | 2.0 | 813.5828 | 696.4300 | 0.1377714 | 0.1392187 |
| 2000 | 3.0 | 1061.1430 | 914.2799 | 0.1766395 | 0.1776900 |

outlier mask 的 keep fraction 约为 `0.999995` 到 `0.999998`。

结果文件：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/heldout_ema_128.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/heldout_ema_128_step2000.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/heldout_model_128.csv
```

### 5.4 Seed 0 的 500/2,000-update 5k 结果

| endpoint | branch | FID | KID | IS |
|---:|---|---:|---:|---:|
| 500 | Flow | 21.3610 | 0.006237 | 71.8230 |
| 500 | LPL | 20.1467 | 0.005596 | 73.9966 |
| 2000 | Flow | 21.8395 | 0.006645 | 68.0297 |
| 2000 | LPL | 19.6806 | 0.005708 | 70.7485 |

### 5.5 四 seed 的 2,000-update 5k 结果

| seed | Flow FID | LPL FID | Flow KID | LPL KID | Flow IS | LPL IS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.8395 | 19.6806 | 0.006645 | 0.005708 | 68.0297 | 70.7485 |
| 1 | 22.0543 | 19.4700 | 0.007198 | 0.005834 | 69.6032 | 72.7129 |
| 2 | 21.9830 | 19.5437 | 0.007111 | 0.005961 | 68.8070 | 72.5984 |
| 3 | 22.4084 | 20.0234 | 0.007532 | 0.006224 | 68.8182 | 70.0317 |
| mean | 22.0713 | 19.6794 | 0.007121 | 0.005932 | 68.8145 | 71.5229 |

记录现象：

- 4/4 seeds 的 LPL FID 低于配对 Flow；
- 4/4 seeds 的 LPL KID 低于配对 Flow；
- 4/4 seeds 的 LPL IS 高于配对 Flow。

### 5.6 50k 结果

终验训练 seed：`3`。

`torch-fidelity`：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| Flow | 16.7410 | 0.007405 | 93.7008 |
| LPL | 14.3156 | 0.006247 | 98.1091 |

ADM：

| branch | FID | sFID | IS |
|---|---:|---:|---:|
| Flow | 13.5043 | 12.9084 | 93.7710 |
| LPL | 11.1027 | 9.2378 | 98.1204 |

两份采样数组均核对为：

```text
shape = (50000, 256, 256, 3)
dtype = uint8
```

结果文件：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/generation_50k_seed3_flow_torch.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/generation_50k_seed3_lpl_torch.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/generation_50k_seed3_flow_adm.json
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl/generation_50k_seed3_lpl_adm.json
```

### 5.7 训练日志汇总

2,000 updates：

| branch | mean Flow loss | mean grad norm | clip rate | wall time |
|---|---:|---:|---:|---:|
| Flow | 0.43447 | 0.36066 | 0.10% | 801.9 s |
| LPL | 0.43753 | 0.51638 | 1.85% | 985.3 s |

新增 seeds：

| seed | Flow branch loss | LPL branch Flow loss | Flow grad | LPL grad | LPL clip |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.43728 | 0.44004 | 0.36646 | 0.52134 | 1.55% |
| 2 | 0.43663 | 0.43935 | 0.36847 | 0.51721 | 1.65% |
| 3 | 0.43648 | 0.43922 | 0.36777 | 0.52337 | 2.05% |

## 6. LPL-02：官方 Stage-2 权重直接采样

状态：`完成`。

这组实验不创建 optimizer，不执行训练，不更新 EMA，直接对官方 Stage-2
权重按统一协议生成 5,000 张图。

结果文件：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl_official_source_5k/official_source_strict_5k.json
```

| tokenizer | checkpoint | updates | FID | KID | IS |
|---|---|---:|---:|---:|---:|
| DINOv2-B | 官方 epoch 14 | 0 | 21.6122 | 0.006439 | 71.8754 |
| MAE-B | 官方 epoch 80 | 0 | 17.5508 | 0.003593 | 80.2416 |
| SigLIP2-B | 官方 epoch 80 | 0 | 13.5717 | 0.001379 | 117.8009 |

2026-07-27 的严格复验固定为 4 processes、每 process batch 4。DINOv2
新旧 5,008 张 PNG 和最终 NPZ 均逐字节一致。旧 MAE/SigLIP2 官方目录各有
5,056 张 PNG，使用过不同 global batch；上表已替换为新生成的 5,008 张
严格同协议样本。原 `official_source_5k.json` 只作为旧采样历史记录，不再
进入主表。

与相应单 seed LPL checkpoint 并列记录：

| tokenizer | checkpoint | FID | KID | IS |
|---|---|---:|---:|---:|
| DINOv2-B | LPL-500 seed 0 | 20.1467 | 0.005596 | 73.9966 |
| DINOv2-B | LPL-2000 seed 0 | 19.6806 | 0.005708 | 70.7485 |
| MAE-B | LPL-500 seed 3409 | 17.0908 | 0.003624 | 78.6336 |
| SigLIP2-B | LPL-500 seed 3409 | 13.3509 | 0.001311 | 114.0744 |

## 7. LPL-03：MAE-B 跨 tokenizer 对照

### 7.1 模型与训练

- encoder：MAE-B/16；
- input：`256 x 256`；
- latent：`[B, 768, 16, 16]`；
- decoder：官方 ViT-XL；
- Stage-2：官方 DiT-XL epoch 80；
- hidden size：1,152；
- depth：28；
- heads：16；
- MLP ratio：4；
- class dropout：0.1；
- rope、RMSNorm、SwiGLU、gembed：开启；
- source：
  `/home/zhoushunyu/data/eqvae/models/RAE/DiTs/MAE/b16/ImageNet256/DiT-XL-ep80/stage2_model.pt`；
- seeds：`3407, 3408, 3409`；
- 每个分支：500 updates；
- world size：4；
- micro batch：1；
- gradient accumulation：4；
- global batch：16；
- 每个分支共读取 8,000 个 train samples；
- 折合约 `0.0062` ImageNet epoch；
- LPL weight：`0.001310218123057881`。

官方 Stage-2 只提供 model-only 权重。本实验 Flow/LPL 使用相同模型权重，
并分别新建内容一致的 optimizer、scheduler 和 EMA 状态。

### 7.2 5k 结果

| seed | Flow FID | LPL FID | Flow KID | LPL KID | Flow IS | LPL IS |
|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 17.6496 | 16.9699 | 0.003793 | 0.003492 | 77.5089 | 78.7342 |
| 3408 | 17.7907 | 17.0663 | 0.003998 | 0.003620 | 78.1699 | 79.7067 |
| 3409 | 17.8310 | 17.0908 | 0.004009 | 0.003624 | 77.4432 | 78.6336 |
| mean | 17.7571 | 17.0424 | 0.003933 | 0.003579 | 77.7073 | 79.0248 |

记录现象：

- 3/3 seeds 的 LPL FID 更低；
- 3/3 seeds 的 LPL KID 更低；
- 3/3 seeds 的 LPL IS 更高。

### 7.3 50k 结果

终验 seed：`3409`。

`torch-fidelity`：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| Flow | 11.9943 | 0.004107 | 109.3436 |
| LPL | 11.3350 | 0.003792 | 111.8263 |

ADM：

| branch | FID | sFID | IS |
|---|---:|---:|---:|
| Flow | 8.6148 | 5.6520 | 109.2572 |
| LPL | 7.9498 | 5.2172 | 111.7411 |

## 8. LPL-04：SigLIP2-B 跨 tokenizer 对照

### 8.1 模型与训练

- encoder：SigLIP2-B/16；
- input：`256 x 256`；
- latent：`[B, 768, 16, 16]`；
- decoder：官方 ViT-XL；
- Stage-2：官方 DiT-XL epoch 80；
- Stage-2 结构与 MAE-B 对照相同；
- source：
  `/home/zhoushunyu/data/eqvae/models/RAE/DiTs/SigLIP2/b16/ImageNet256/DiT-XL-ep80/stage2_model.pt`；
- seeds：`3407, 3408, 3409`；
- 每个分支：500 updates；
- world size：4；
- micro batch：1；
- gradient accumulation：4；
- global batch：16；
- 每个分支读取 8,000 个 train samples；
- 折合约 `0.0062` ImageNet epoch；
- LPL weight：`0.0009191518565962038`。

### 8.2 5k 结果

| seed | Flow FID | LPL FID | Flow KID | LPL KID | Flow IS | LPL IS |
|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 13.7162 | 13.3887 | 0.001419 | 0.001328 | 112.0353 | 113.7402 |
| 3408 | 13.7370 | 13.4065 | 0.001423 | 0.001337 | 111.8065 | 114.4114 |
| 3409 | 13.6789 | 13.3509 | 0.001412 | 0.001311 | 112.3505 | 114.0744 |
| mean | 13.7107 | 13.3821 | 0.001418 | 0.001325 | 112.0641 | 114.0753 |

记录现象：

- 3/3 seeds 的 LPL FID 更低；
- 3/3 seeds 的 LPL KID 更低；
- 3/3 seeds 的 LPL IS 更高。

### 8.3 50k 结果

终验 seed：`3409`。

`torch-fidelity`：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| Flow | 8.2955 | 0.001454 | 174.5569 |
| LPL | 8.0143 | 0.001375 | 177.2613 |

ADM：

| branch | FID | sFID | IS |
|---|---:|---:|---:|
| Flow | 4.9550 | 7.2129 | 174.8867 |
| LPL | 4.6811 | 6.1272 | 177.5565 |

MAE-B 和 SigLIP2-B 的主目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl_cross_tokenizer
```

## 9. LPL-05：Flow 额外计算预算对照

状态：`完成`。

抽取 seed `3409`，从与原 Flow-500 精确一致的 checkpoint 继续训练到
600 updates。

核对内容：

- 1,759 个 tensor；
- 373 个 scalar/state entries；
- model、EMA、optimizer、scheduler、CPU RNG、CUDA RNG；
- step-500 新旧 checkpoint 的核对差值为 0。

记录的 LPL 单 update 墙钟开销：

| 模型 | LPL 相对 Flow 的额外墙钟 |
|---|---:|
| MAE-B | 8.7% |
| SigLIP2-B | 14.7% |

600 updates 相对 500 updates 增加 20% updates 和 train samples。

| tokenizer | branch | updates | FID | KID | IS |
|---|---|---:|---:|---:|---:|
| MAE-B | Flow | 500 | 17.8310 | 0.004009 | 77.4432 |
| MAE-B | Flow extra | 600 | 17.7512 | 0.004115 | 77.1313 |
| MAE-B | LPL | 500 | 17.0908 | 0.003624 | 78.6336 |
| SigLIP2-B | Flow | 500 | 13.6789 | 0.001412 | 112.3505 |
| SigLIP2-B | Flow extra | 600 | 13.8835 | 0.001442 | 110.6498 |
| SigLIP2-B | LPL | 500 | 13.3509 | 0.001311 | 114.0744 |

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_strict_lpl_compute_control
```

## 10. LPL-06：DINOv2-B 局部误差方向诊断

状态：`诊断完成`。

### 10.1 配置

- 使用 seeds `0, 1, 2, 3` 的 Flow/LPL step-2000 checkpoints；
- checkpoint state：EMA；
- validation latent cache：
  `/home/zhoushunyu/data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32`；
- 每个 seed：32 张图；
- noise/signal：`0.5, 1, 2, 3`；
- 总观测：`4 x 32 x 4 = 512`；
- local displacement：clean latent RMS 的 `1%`；
- symmetric finite difference displacement：clean latent RMS 的 `0.1%`；
- Flow/LPL 方向在共同半径
  `min(RMS(e_flow), RMS(e_lpl))` 上比较；
- random Gaussian 和 channel/token shuffle 作为方向对照；
- decoder hidden layers：`[6, 11, 17, 22, 28]`；
- fp32，TF32 关闭。

### 10.2 汇总观测

下表为 LPL/Flow 比值：

| 测量 | LPL / Flow | LPL 数值更低的观测比例 |
|---|---:|---:|
| latent MSE | 1.0092 | 0.0% |
| 1% raw local amplification | 1.1217 | 15.4% |
| symmetric FD raw amplification | 1.1364 | 15.2% |
| 1% strict local amplification | 1.0614 | 15.8% |
| symmetric FD strict amplification | 1.0750 | 14.5% |
| matched full-radius strict feature error | 0.8768 | 100.0% |

其他记录：

- actual error RMS / clean latent RMS：约 18% 到 41%；
- 中位 actual error ratio：约 28%；
- local quadratic 与 raw full error Spearman：`0.7987`；
- latent MSE 与 raw full error Spearman：`0.8019`；
- strict quadratic 与 strict full error Spearman：`0.8218`；
- random local gain / LPL local gain：`0.5850`；
- shuffled local gain / LPL local gain：`0.5164`。

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_lpl_error_geometry/paired_4seed_n32_v1
```

## 11. LPL-07：DINOv2-B 有限半径和归一化分解

状态：`诊断完成`。

### 11.1 幅度设置

沿每条配对误差方向测试：

```text
fraction = [0.04, 0.16, 0.32, 0.64, 1.00]
```

`1.00` 表示 Flow/LPL 两个方向使用相同的完整配对半径。

主误差几何 manifest 明确记录 `state_key=ema`。幅度扫描脚本默认
`state_key=ema`，最终幅度 manifest 没有单独持久化这一字段。

### 11.2 汇总结果

LPL/Flow：

| fraction | raw | target-normalized | symmetric-normalized | prediction-normalized |
|---:|---:|---:|---:|---:|
| 0.04 | 1.1106 | 1.0563 | 1.0547 | 1.0531 |
| 0.16 | 1.1031 | 1.0516 | 1.0450 | 1.0383 |
| 0.32 | 1.0806 | 1.0363 | 1.0226 | 1.0087 |
| 0.64 | 1.0302 | 1.0070 | 0.9772 | 0.9461 |
| 1.00 | 0.9869 | 0.9766 | 0.9285 | 0.8768 |

记录现象：

- 512/512 条轨迹存在被采样到的 crossover；
- first sampled LPL-better fraction 的中位数为 `0.64`；
- full radius prediction feature variance relative to clean：
  Flow `0.848626`，LPL `0.932225`；
- full radius centered channel cosine：
  Flow 约 `0.795`，LPL 约 `0.804`；
- raw local layer ratios：
  `[0.938, 0.968, 1.036, 1.089, 1.126]`；
- full-radius 五层 strict ratios 均约为 `0.87` 到 `0.88`；
- mask keep fraction 接近 1；
- mask-free prediction-normalized 与 strict 最大相对差小于 `2.4e-7`；
- fraction 1.0 与独立 error-geometry strict 结果精确一致；
- raw 最大差 `7.6e-6`；
- 无 NaN 或 Inf。

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_lpl_amplitude_sweep/paired_4seed_n32_v3_norm_decomp
```

主要图：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_lpl_amplitude_sweep/paired_4seed_n32_v3_norm_decomp/mechanism_curve.png
```

## 12. LPL-08：常数 Decoder Metric 代理

状态：`诊断完成`。

这组实验不做 LPL 生成训练，只比较 latent MSE gradient、真实 decoder
gradient 和若干静态 proxy。

### 12.1 数据与计算

- 模型：RAE-DINOv2-B；
- calibration：ImageNet train 1,024；
- test：ImageNet validation 2,048；
- exact gradient subset：128；
- time bins：5；
- fp32；
- TF32 关闭；
- 4 GPUs。

### 12.2 Exact gradient 记录

五个 time bins 中，`grad_x0` 与 `grad_decoder` 的 median cosine：

```text
[0.0327, 0.0200, 0.0119, 0.0047, 0.0018]
```

局部 step 取当前 latent error norm 的 `0.1%`：

| 更新方向 | decoder error 变化 |
|---|---:|
| x0 gradient | -0.109% |
| decoder gradient | -9.335% |

五个 time bin 内，x0 loss 与 decoder loss 的 Pearson 约为 `0.11` 到 `0.22`。

### 12.3 静态 proxy 记录

| proxy | calibration Spearman | test Spearman | gradient cosine |
|---|---:|---:|---:|
| decoder embedding metric | 0.1512 | 0.2055 | 0.0285 |
| randomized Gauss-Newton | 0.1970 | 0.1795 | 0.0224 |
| GN x4 DCT bands | 0.0858 | 0.0731 | 0.0159 |

预设 threshold：

```text
test Spearman >= 0.80
gradient cosine >= 0.70
calibration/test gap <= 15%
```

上述三个 proxy 均未达到前两个 threshold。

目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_decoder_risk_phase0
```

## 13. LPL-09：Predictability、Atlas 和 Decoder Prefix 代理

状态：`诊断完成`。

### 13.1 协议

- 数据：ImageNet-1K validation latent cache；
- score ranking：32 张图；
- exact gradients：8 张图；
- times：
  `[0.57217896, 0.87386799, 0.97289228]`；
- predictability basis：独立 ImageNet train split 拟合；
- decoder oracle：独立 held-out ImageNet train latent 拟合；
- fp32；
- TF32 关闭。

### 13.2 汇总

| proxy | median-time Spearman | median-time gradient cosine |
|---|---:|---:|
| latent MSE | 0.0656 | 0.0126 |
| predictability static | 0.0652 | 0.0210 |
| decoder-atlas oracle static | 0.0363 | 0.0286 |
| decoder prefix 1 | 0.1822 | 0.1166 |
| decoder prefix 2 | 0.4205 | 0.2339 |
| decoder prefix 3 | 0.7276 | 0.4928 |

预设 threshold：

```text
median-time Spearman >= 0.80
median-time gradient cosine >= 0.70
```

汇总文件中 `pass=false`。

结果文件：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_lpl_proxy_gate_v1/summary.json
/home/zhoushunyu/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_lpl_proxy_gate_v1/proxy_scores.csv
/home/zhoushunyu/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_lpl_proxy_gate_v1/proxy_gradients.csv
```

## 14. LPL-10：DINOv2-L 短程 Pilot

状态：`完成，无 FID`。

### 14.1 模型

- Stage-1：`nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08`；
- encoder：DINOv2-L/14 with registers；
- encoder layers：24；
- channels：1,024；
- latent：`[B, 1024, 16, 16]`；
- decoder：ViT-XL，28 层；
- Stage-2：DiTDH-S，input channels 1,024；
- 没有匹配该 latent 的官方 Stage-2 prior。

### 14.2 训练

- 从随机初始化训练共同 Flow source：250 updates；
- Flow/LPL 从同一个完整 step-250 checkpoint 分叉；
- 每个分支继续 100 updates，到 step 350；
- seeds：`3407, 3408, 3409`；
- world size：4；
- micro batch：1；
- gradient accumulation：4；
- global batch：16；
- optimizer 与统一设置相同；
- fp32；
- TF32 关闭；
- 使用 online `model` 权重做短程诊断；
- 100-step EMA 只吸收约 4.9% 的分支参数变化。

三个 seeds 的首批 fingerprint 均匹配 image、label、time、noise、noisy
latent、target velocity 和 source prediction。

### 14.3 测试

- validation images：每 seed 8 张；
- noise/signal：`0.5, 1.0, 2.0, 3.0`；
- 每 seed 32 observations；
- 总计 96 observations；
- amplitude fractions：
  `[0.04, 0.16, 0.32, 0.64, 1.00]`。

### 14.4 幅度扫描

三个 seed 几何平均的 LPL/Flow：

| fraction | raw | strict | target-normalized | symmetric-normalized | feature variance |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 0.932 | 0.919 | 0.922 | 0.921 | 1.004 |
| 0.16 | 0.898 | 0.870 | 0.888 | 0.879 | 1.019 |
| 0.32 | 0.873 | 0.807 | 0.856 | 0.833 | 1.056 |
| 0.64 | 0.937 | 0.681 | 0.888 | 0.793 | 1.278 |
| 1.00 | 1.089 | 0.545 | 1.019 | 0.807 | 1.683 |

full-radius strict feature LPL/Flow：

| seed | ratio |
|---:|---:|
| 3407 | 0.5441 |
| 3408 | 0.5467 |
| 3409 | 0.5449 |

其他记录：

- latent MSE 与 raw full error Spearman：`0.8253`；
- local quadratic 与 raw error Spearman：`0.5044`；
- strict quadratic 与 strict error Spearman：`0.6911`；
- random direction / LPL local gain：`0.4942`；
- shuffled direction / LPL local gain：`0.4713`；
- mechanism gate：`false`；
- observed crossover fraction：`1.0`；
- first sampled LPL-better fraction 中位数：`0.04`；
- 未计算 FID、gFID 或 IS。

主目录：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_dinov2_large/pilot
```

## 15. 未运行或只完成设计的 LPL 变体

以下内容不属于已完成实验结果。

| 方案 | 当前状态 | 已有内容 | 尚无内容 |
|---|---|---|---|
| target-normalized feature loss | `未运行` | loss 定义和消融设想 | 多 seed 训练、FID |
| symmetric-normalized feature loss | `未运行` | loss 定义和消融设想 | 多 seed 训练、FID |
| variance-only LPL | `未运行` | 目标定义 | 正式训练、FID |
| Decoder Metric Preconditioning | `未运行` | 常数 metric 和 proxy 诊断 | Stage-2 训练 |
| 仅使用 decoder prefix 的廉价 LPL | `未运行` | prefix 1/2/3 proxy 数值 | 配对生成训练 |
| joint encoder/decoder training | `未运行` | 无正式运行 | checkpoint、指标 |
| rollout LPL | `未运行` | 无正式运行 | checkpoint、指标 |
| 完整 80/800 epoch LPL | `未运行` | 无 | 长训练结果 |
| DINOv2-L 生成终验 | `未运行` | 250+100 update pilot | 成熟 prior、FID |

## 16. 数据隔离和配对记录

### 16.1 Train/validation 用途

| 内容 | 数据 split |
|---|---|
| Stage-2 Flow/LPL 更新 | ImageNet-1K train |
| LPL 权重校准 | ImageNet-1K train |
| held-out latent/decoder metrics | ImageNet-1K validation |
| 5k/50k 生成 reference | ImageNet validation/ADM reference statistics |
| DINOv2-B error geometry | ImageNet validation latent cache |
| DINOv2-L pilot evaluation | 未参与训练的 ImageNet validation images |

### 16.2 配对控制

DINOv2-B、MAE-B、SigLIP2-B 和 DINOv2-L 的配对实验均保存
`pair_fingerprint.json`。同 seed 的 Flow/LPL 首批比较项目包括：

```text
image
label
time
noise
noisy_latent
target_velocity
initial/source_prediction
```

正式生成比较固定：

```text
sampling seed = 20260715
sampling steps = 50
solver = Euler
CFG = 1.0
class sequence = balanced ImageNet classes
```

## 17. 代码索引

| 文件 | 内容 |
|---|---|
| `experiments/rae_strict_lpl.py` | clean estimate、时间门控、mask、cross-normalization、LPL |
| `experiments/train_rae_strict_lpl.py` | DDP 训练、校准、pair fingerprint、checkpoint |
| `experiments/evaluate_rae_strict_lpl.py` | held-out latent/decoder feature 评价 |
| `experiments/evaluate_rae_strict_lpl_generation.py` | 5k/50k 生成和指标入口 |
| `experiments/rae_lpl_error_geometry.py` | 误差方向、幅度、归一化分解和汇总函数 |
| `experiments/run_rae_lpl_error_geometry.py` | 多 checkpoint 局部/有限半径诊断 |
| `experiments/run_rae_lpl_amplitude_sweep.py` | 多幅度扫描、crossover、图表 |
| `experiments/rae_decoder_risk_phase0.py` | 常数 decoder metric proxy |
| `experiments/run_rae_predictability_lpl_proxy_gate.py` | predictability/atlas/prefix proxy |
| `experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml` | DINOv2-B 配置 |
| `experiments/configs/rae_strict_lpl_dit_xl_mae.yaml` | MAE-B 配置 |
| `experiments/configs/rae_strict_lpl_dit_xl_siglip2.yaml` | SigLIP2-B 配置 |
| `experiments/configs/rae_strict_lpl_ditdh_s_dinov2_large.yaml` | DINOv2-L 配置 |

## 18. 既有专题文档

| 文档 | 记录内容 |
|---|---|
| `docs/RAE_DETERMINISTIC_LPL_REPRODUCTION_ZH.md` | DINOv2-B 严格复现、held-out、5k、50k |
| `docs/RAE_LPL_CROSS_TOKENIZER_PROTOCOL_ZH.md` | MAE/SigLIP2 预注册协议 |
| `docs/RAE_LPL_CROSS_TOKENIZER_RESULTS_ZH.md` | 跨 tokenizer、官方起点、算力对照、50k |
| `docs/RAE_LPL_FINITE_RADIUS_MECHANISM_ZH.md` | DINOv2-B 误差几何和幅度扫描 |
| `docs/RAE_DINOV2_L_LPL_PILOT_ZH.md` | DINOv2-L pilot |
| `docs/RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md` | 常数 decoder metric proxy |
| `docs/RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md` | predictability 和 decoder prefix proxy |
| `docs/RAE_LPL_IMPROVEMENT_RESEARCH_ZH.md` | 已设计但未完成的 LPL 变体 |
| `docs/RAE_LPL_RESEARCH_AUDIT_ZH.md` | 既有研究审计和解释性文字 |

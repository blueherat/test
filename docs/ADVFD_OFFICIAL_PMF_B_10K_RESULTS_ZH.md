# AdvFD 官方 pMF-B 10K 中程结果

日期：2026-08-23

## 1. 协议

本轮遵循公开代码实际行为，而不是反向改写 paper-only 实现：

- pMF-B 官方预训练权重；
- ImageNet-1K packed training data；
- full SIM static loss：SigLIP2 + MAE-L + Inception；
- adaptive Inception-2048、real whitening、FP64 moments；
- 3 张 RTX 4090，local batch 24，global batch 72；
- 10,000 generator steps，generator warmup 6,250 steps；
- adaptive 从 step 1,000 启用，4,000 steps warmup，之后权重为 0.05；
- 官方 `official_avg` 分布式梯度语义；
- online generator，1 NFE，CFG 8.5，interval `[0.1, 0.7]`；
- 官方 continuous Inception/ADM stats，5,000 张图。

该实验只有约 72 万张训练样本暴露，远小于论文正式 pMF 训练的约 1.28 亿，属于
official-code-faithful scaled reproduction，不是论文完整数值复现。

本轮 10K 还把 cosine 调度压缩到 10K 终点，因此它是匹配 AdvFD/static 的短预算实验，
不是官方 125K 学习率曲线的前 10K。后续 25K 阶段检查已改为保持官方 125K 调度
horizon，只在 25K 暂停评估，使 checkpoint 可以无缝续训。

## 2. 生成质量

| checkpoint | FID(ADM) | IS | 同 step `AdvFD - static` FID |
|---|---:|---:|---:|
| 原始 pMF-B | 8.9752 | 153.73 | - |
| static SIM 5K | **7.6439** | 162.17 | - |
| AdvFD 5K | 7.8468 | **165.87** | +0.2029 |
| static SIM 10K | 7.4971 | 165.05 | - |
| AdvFD 10K | **7.4395** | **171.47** | -0.0575 |

从 baseline 到 5K、再到 10K，AdvFD 自身的 FID 与 IS 同向持续改善，因此 10K
不构成饱和证据。但 matched static 对照改变了因果解释：5K 时 AdvFD 的 FID 反而比
static 差 0.2029；10K 时才领先 0.0575。原始模型到 AdvFD 10K 的 1.5356 FID 总改善中，
static 10K 已经解释 1.4781，adaptive branch 的同 step 增量只占很小一部分。这里的
10K static/AdvFD 使用相同评估协议，但旧评估分别由 4/3 个 rank 生成，因此不是严格
逐噪声配对；下面的重新生成实验统一为 4 个 rank，并取代它作为小差值的主要证据。

因此当前可以确认的是：static FD 后训练有效；AdvFD 在 10K 的 ADM-FID 和 IS 上出现
小幅额外正信号。严格配对复验表明该信号不跨 FID 参考统计和 held-out representation
稳定成立，因而不能概括成整体分布质量稳定提升。为检查压缩 10K
cosine 调度混杂，曾从原始 pMF-B 重新启动使用官方 125K 学习率日程的 25K 前缀；在
保存 `step_0001999.pth` 后，因现有 clean-room 与 official-code 两组结果已经复现出
“adaptive 晚于 static 起效并带来小幅额外收益”的核心现象，按研究范围主动停止扩训。

## 3. 严格配对的 held-out FDr3

为避免训练 AdvFD 所用的 Inception 表示参与结论，并消除旧评估 4/3 rank 的随机样本
差异，本轮对 static 10K 和 AdvFD 10K 重新生成各 5,000 张图。两个条件使用相同 seed、
4 个 rank、batch、类别顺序、CFG、NFE 和采样参数；每个 ImageNet 类别均生成 5 张。
随后在同一批 PNG 上统一评估论文 FDr3 的三个 held-out 表示：ConvNeXtV2-Base、
DINOv2-L/14 和 CLIP-L/14。FDr 为 raw FD 除以论文固定的 ImageNet validation FD，
FDr3 是三项 FDr 的算术平均，均为越低越好。

| condition | ConvNeXt FD / FDr | DINOv2 FD / FDr | CLIP FD / FDr | FDr3 |
|---|---:|---:|---:|---:|
| static 10K | 169.2190 / 2.9755 | 148.1873 / 10.4431 | 82.6236 / 14.7542 | **9.3909** |
| AdvFD 10K | 180.8146 / 3.1794 | 148.7045 / 10.4795 | 83.5569 / 14.9209 | 9.5266 |
| AdvFD - static | +11.5956 / +0.2039 | +0.5172 / +0.0365 | +0.9334 / +0.1667 | **+0.1357** |

三种 held-out 表示全部同向判定 AdvFD 更差，FDr3 相对 static 增加 0.1357。为确认这
不是重新生成批次本身与旧 FID 不一致造成的假象，同一批图片还补算了两套 Inception
reference：

| condition | FID(JiT) | FID(ADM) | IS |
|---|---:|---:|---:|
| static 10K | **7.5202** | 7.5027 | 164.88 |
| AdvFD 10K | 7.5444 | **7.3991** | **167.91** |
| AdvFD - static | +0.0241 | -0.1037 | +3.03 |

因此，AdvFD 在当前短预算复现中的增量具有明显的 metric dependence：ADM-FID 与 IS
改善，JiT-FID 略差，三个 held-out representation FD 全部变差。5K 样本的绝对 FDr3
受有限样本协方差估计偏差影响，不能与论文 50K 的绝对 FDr3 横比；但 static 与 AdvFD
完全同协议、同样本规模，方向性对照是有效的。当前证据不支持“adaptive branch 已经
带来跨表示一致的生成分布改善”。

## 4. Critic 尺度和谱

使用与 FID 相同的 5K 生成图，以及训练未使用的 ImageNet validation 图，对保存的
adaptive Inception 做 fresh forward：

| checkpoint | real RMS | fake RMS | fake/real | real effective rank | fake effective rank |
|---|---:|---:|---:|---:|---:|
| 5K | 291.95 | 293.94 | 1.0068 | 5.70 | 5.38 |
| 10K | 4236.08 | 4238.89 | 1.0007 | 1.90 | 1.81 |

原始 feature 出现极强的共同尺度漂移和低秩化，但 real/fake RMS 几乎同步。因此当前
证据不支持“只放大 fake artifact”的 selective-amplification 机制；它更接近共同 gauge
漂移与少数协方差方向的强化。

逐样本分位数进一步排除了“少数异常 fake 拖高总体 RMS”的解释：

| checkpoint | split | adaptive RMS q50 | q90 | q99 | max |
|---|---|---:|---:|---:|---:|
| 5K | real | 292.58 | 302.42 | 309.55 | 320.90 |
| 5K | fake | 294.12 | 303.15 | 310.10 | 323.42 |
| 10K | real | 4227.45 | 4473.90 | 4705.30 | 5061.40 |
| 10K | fake | 4233.34 | 4459.18 | 4665.21 | 4921.38 |

两边从中位数到尾部都高度重叠，且 10K 的 fake q99/max 甚至略低于 real。故在当前
checkpoint 上，selective amplification 是一个成立的总体理论反例，但不是已观测到的
训练机制。

## 5. Train 与 held-out 校准 FD

`train EMA` 使用 checkpoint 内保存的 real/fake EMA moments。`held-out` 固定同一
critic，用 checkpoint 的训练期 real EMA whitening 全新 validation real 与生成图：

| checkpoint | train EMA FD | held-out FD | held-out mean | held-out covariance |
|---|---:|---:|---:|---:|
| 5K | 102.76 | 155.03 | 12.29 | 142.74 |
| 10K | 257.14 | 276.31 | 5.94 | 270.37 |

到 10K，held-out FD 没有塌缩，且与 train EMA FD 接近；二者均几乎由 covariance
分支主导。这不符合“critic 只记住当前训练 batch”的简单解释。它说明 critic 找到的
低秩协方差 witness 能跨到独立样本，但是否对生成器有额外因果收益仍需 static 对照。

作为口径检查，同一 held-out moments 若分别使用 fresh validation-real whitening 和
fresh pooled whitening，5K 为 `958.98 / 563.23`，10K 为 `1510.28 / 703.64`。这些量
使用不同的参考几何，不能与 checkpoint-whitened FD 横向比较为“谁更好”；它们用于
后续 real-only/pooled 机制消融。

## 6. 产物

- 机器可读汇总：`docs/data/advfd_official_pmf_b_10k_results.csv`
- 严格配对 FDr3/FID：`docs/data/advfd_official_pmf_b_10k_paired_fdr3.csv`
- 逐样本 feature 分布：`docs/data/advfd_official_pmf_b_samplewise_feature_audit.csv`
- 配对生成与原始评估：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_10k_fdr3_paired5k_v1`
- 官方 FID：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/official_eval5k`
- 校准 critic 审计：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/fresh_feature_audit5k_calibrated_v2`
- 逐样本 critic 审计：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/fresh_feature_audit5k_samplewise_v3`
- checkpoint EMA FD：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/checkpoint_fd_audit_5k_10k`

## 7. 当前边界

- 没有完成官方 global batch 1024、125K steps 和 FID-50K 的完整数值复现；
- 没有做多训练 seed，也没有论文规模的 FDr3/FDr6-50K；当前补充的是严格配对的
  FDr3-5K 机制筛查；
- fresh 25K/62.5K 扩训已主动取消，不把 2K checkpoint 当作质量结论；
- mean-only、covariance-only、real/pooled calibration 与已有 critic 约束仍属于候选机制对照。

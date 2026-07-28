# RAE Encoder-Decoder 反向层级诊断结果

## 研究问题

本实验检验两个必须分开的命题：

1. frozen RAE-DINOv2 decoder 是否自然地反向恢复 encoder 的表征层级；
2. 这种层级对应是否是生成 latent 解码失败的主要瓶颈，从而值得逐层训练对齐。

第一项成立并不自动推出第二项成立。

## 数据与控制

- ImageNet-1K parquet validation/test split，seed `20260718`。
- 512 张真实图像：前 256 张只用于 calibration，后 256 张只用于 test。
- 第二批独立 512 张图像使用 seed `20260719` 做复现。
- encoder：frozen RAE-DINOv2，13 个 hidden states，每层 256 tokens。
- decoder：官方 frozen ViT-XL decoder，29 个 hidden states，每层 256 tokens。
- 所有模型和指标使用 fp32，关闭 TF32。
- 四条 generated path 各使用已有的 256 个 paired endpoint；四路共享初始 noise 和 labels。
- generated latent 没有配对到真实图像。generated-cycle 只比较
  `D_l(z_gen)` 与 `E_k(clamp(D(z_gen)))`，因此测量的是自洽层级，不是到真实图像的距离。
- 5k FID 来自此前固定配置，atlas/probe 没有参与路径训练或 checkpoint 选择。

## 指标

### Basis-invariant cross-layer atlas

对每层 token 去掉该图的 token mean，计算 centered token Gram。encoder 层 `k` 与
decoder 层 `l` 的分数为 linear CKA。通道维度可以不同，且该指标不受正交通道换基影响。

为了排除所有图像共享的位置编码和网格模板，主分数使用：

```text
excess CKA = paired-image CKA - mismatched-image CKA
```

### Held-out linear projector

从 calibration atlas 固定五个 anchor：

```text
D0 -> E12, D7 -> E8, D14 -> E6, D21 -> E4, D28 -> E2
```

只在 256 张 calibration 图像上拟合 ridge projector `D_l W_l ~= E_k`，随后在 test、
latent perturbation 和 generated-cycle 上固定评估。

## 结果一：反向层级真实存在

第一批 clean calibration/test：

| 指标 | 数值 |
|---|---:|
| atlas Pearson | 0.9979 |
| atlas RMS difference | 0.00723 |
| 完全相同的 best-layer mapping | 93.10% |
| test soft reverse Spearman | 0.9995 |
| test argmax reverse Spearman | 0.9730 |
| test 非递增相邻比例 | 100% |
| test mean peak excess CKA | 0.5049 |

decoder 的自然轨迹非常清楚：

```text
D0-D6   -> E12
D7-D10  -> E8
D11-D14 -> E6
D15-D23 -> E4
D24-D28 -> E2
```

第二批独立 512 张图像与第一批 test atlas 的 Pearson 为 `0.9984`；29 个 decoder 状态中
`96.55%` 的 best encoder layer 完全相同，soft reverse Spearman 为 `0.9911`。

因此可以可靠声称：

> 官方 RAE decoder 已自然形成从 DINOv2 深层语义到浅层空间表征的反向层级，而不是
> 29 个彼此无关的重构 block。

## 结果二：普通 latent 扰动几乎不破坏该结构

相对 clean-source test atlas：

| condition | atlas Pearson | RMS distance | mapping exact |
|---|---:|---:|---:|
| Gaussian sigma 0.10 | 0.999995 | 0.00072 | 100% |
| Gaussian sigma 0.30 | 0.999709 | 0.00589 | 100% |
| latent scale 0.75 | 0.992810 | 0.01389 | 96.55% |

这说明用 isotropic Gaussian 或简单方差收缩训练“层级 CKA 对齐”，大部分时间只会监督一个
decoder 本来已经保持的结构。它们不是当前 generated endpoint 的充分替代。

## 结果三：生成路径偏离 clean，但偏离不等于 FID

所有 generated-cycle 都保留强反向顺序：soft reverse Spearman 为 `0.983-0.989`。
它们主要表现为 sample-specific excess CKA 整体变弱，而不是层级顺序消失。

| path | 5k FID | atlas vs clean Pearson | RMS distance | mean peak excess CKA |
|---|---:|---:|---:|---:|
| static | 123.53 | 0.7628 | 0.1299 | 0.3926 |
| random | 128.99 | 0.7473 | 0.1371 | 0.3861 |
| annealed | 143.54 | 0.6221 | 0.1902 | 0.3415 |
| reverse | 159.05 | **0.7893** | **0.1178** | **0.4035** |

atlas distance 与 FID 的 Spearman 只有 `-0.20`。最重要的反例是：FID 最差的 reverse
反而最接近 clean atlas。因此“保持反向层级形状”不是生成质量的充分条件。

## 结果四：线性 projector 能泛化，但仍不能识别最差路径

五个 projector 的平均结果：

| source | relative error | cosine |
|---|---:|---:|
| clean calibration | 0.5690 | 0.7827 |
| clean test | 0.6001 | 0.7575 |
| clean cycle test | 0.6512 | 0.7554 |
| Gaussian sigma 0.30 | 0.6671 | 0.7299 |
| static generated-cycle | 0.8774 | 0.5416 |
| random generated-cycle | 0.8798 | 0.5364 |
| annealed generated-cycle | 0.9084 | 0.4977 |
| reverse generated-cycle | 0.8818 | 0.5230 |

clean test 只比 calibration 高 `5.47%`，证明 channel projector 的确跨图像泛化。生成
endpoint 也确实明显离开 clean linear relation。

但 projector error 与四路 FID 的 Spearman `0.80` 只有描述意义；它把 annealed 判断为
最差，而真正 FID 最差的是 reverse。anchor 拆分进一步显示：

- `D0 -> E12` 在 clean source 上误差仅 `0.147`，本质上接近 final-latent/cycle closure；
- generated 上该误差约 `1.27-1.33`，但 reverse 反而是四路最低；
- 中后层 generated error 与 clean 的差距快速缩小；
- `D28 -> E2` 的 generated error `0.687-0.713`，并不比 clean-source test 的 `0.730` 更差。

因此平均 projector signal 主要包含已有 cycle 信息，没有定位出 reverse 的关键生成失败。

## 决策

### Gate A：是否存在可复现的反向层级

**通过。** 该结构强、可复现、具有样本特异性，并非位置模板或通道坐标造成的假象。

### Gate B：反向层级偏离是否解释生成质量

**未通过。** CKA atlas 无法排序 FID；linear probe 只能粗略区分 clean/generated，不能识别
最差路径；普通 Gaussian/scale perturbation 又几乎不破坏 atlas。

最终建议：

> 现在不训练“所有 decoder 层对齐 encoder 层”。官方 decoder 已经自然做了这件事，且
> 最差生成路径仍能保留甚至强化该层级。全面施加辅助损失很可能是冗余约束，并可能损害
> decoder 自身的通道 gauge 与重构自由度。

当前更接近生成断层的信号仍是此前验证过的：cycle residual、decoder local sensitivity、
以及 reverse 路径的早层 activation amplification。下一项低成本研究若继续，应检查
**层级轨迹的幅值/条件数/Jacobian amplification**，而不是只检查层级表示相似性。

## 相关工作边界

- [Reverse Distillation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)
  已让 student decoder 恢复 teacher encoder 的多尺度特征，但用于异常检测。
- [REPA](https://arxiv.org/abs/2410.06940) 对齐 diffusion hidden states 与外部视觉表征，
  不是 RAE encoder-decoder 反向对齐。
- [LPL](https://arxiv.org/abs/2411.04873) 用 decoder hidden features 监督 diffusion，方向相反。
- [RAE](https://arxiv.org/abs/2510.11690) 的 decoder 使用 L1、LPIPS、GAN 和 latent noise，
  没有显式中间层对齐。

所以“逐层 feature reconstruction”本身不是新方法；若以后重新打开，贡献必须来自对
generated-latent decoder dynamics 的新机制，而不能只是在 RAE 上增加 distillation loss。

## 代码与产物

代码：

- `experiments/rae_encoder_decoder_atlas.py`
- `experiments/run_rae_encoder_decoder_atlas.py`
- `experiments/run_rae_encoder_decoder_probe.py`
- `tests/test_rae_encoder_decoder_atlas.py`

主产物：

- `~/data/eqvae/experiments/rae_encoder_decoder_atlas/dinov2_clean512_generated256_seed20260718/`
- 复现：`~/data/eqvae/experiments/rae_encoder_decoder_atlas/dinov2_clean512_seed20260719_replicate/`
- 图：`atlas_heatmaps.png`
- 全维 probe：`full_linear_probe/probe_decision.json`

复现命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_encoder_decoder_atlas.py

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_encoder_decoder_probe.py
```

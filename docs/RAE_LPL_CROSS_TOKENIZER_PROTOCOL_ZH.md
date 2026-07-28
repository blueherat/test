# RAE LPL 跨 Tokenizer 生成验证协议

## 问题

已有严格实验只证明：在官方 `RAE-DINOv2-B + DiTDH-S` 上，从同一
checkpoint 继续训练 2,000 step 时，LPL 在四个训练 seed 的 5k
FID/KID/IS 上同向改善，并在一个预先固定 seed 的 50k ADM FID 上改善。

这不能推出 LPL 对其他 RAE tokenizer 也有效。本实验单独检验：

> 冻结 tokenizer 和 decoder 后，decoder-feature LPL 对 latent prior
> 的生成改善能否跨 MAE-B 和 SigLIP2-B 复现？

## 官方资产

| tokenizer | encoder | decoder | latent | 官方 prior |
|---|---|---|---|---|
| MAE-B | `facebook/vit-mae-base` 的固定本地 snapshot | ViT-XL, 28 layers | `[768,16,16]` | DiT-XL, epoch 80 |
| SigLIP2-B | `google/siglip2-base-patch16-256` 的固定本地 snapshot | ViT-XL, 28 layers | `[768,16,16]` | DiT-XL, epoch 80 |

官方只提供 model-only prior 权重。因此每个配对实验都从相同官方模型参数、
相同新建 AdamW/scheduler 状态和相同 RNG 数据流开始。optimizer 重置是协议
限制，但在同一 tokenizer 内不会偏向 Flow 或 LPL 分支。

## 固定实验

- 数据：ImageNet-1K train parquet；validation 不参与训练或权重选择。
- 数值：四张 RTX 4090，fp32，关闭 TF32。
- encoder/decoder：冻结且 `eval()`。
- prior：只更新官方 DiT-XL。
- 目标：
  - Flow：原始 latent velocity MSE。
  - LPL：Flow 加冻结 decoder 的五层 feature loss。
- LPL 权重：每个 tokenizer 在官方源模型上用固定 train batch 校准，使
  初始加权 LPL/Flow 为 `0.25`；不能根据 FID 调权重。
- 训练：`500` paired updates，seeds `3407, 3408, 3409`。
- 采样：EMA、Euler 50 steps、CFG 1.0、固定 seed `20260715`、类别均衡
  5,000 张。
- 主指标：配对 5k FID；辅指标：KID、IS。

## 预先固定的判据

对每个 tokenizer 分别判断：

- **强复现**：`3/3` seeds 的 FID 均改善，平均 KID 不恶化。
- **弱复现**：`2/3` seeds 的 FID 改善，平均 FID 改善且平均 KID 不恶化。
- **失败**：平均 FID 不改善，或仅 `1/3` seed 改善。

只有 MAE-B 和 SigLIP2-B 都至少弱复现，才能说 LPL 的收益不是
DINOv2-B 单一 tokenizer 的偶然现象。若任一模型失败，应报告异质性，不能
使用跨模型平均数掩盖失败。

若两个 tokenizer 都通过 5k 门槛，预先固定 `seed=3409` 做各自 50k ADM
终验。5k 只负责晋级，不用于挑选最好的 seed。

## 解释边界

该实验能回答“LPL 是否跨公开 RAE tokenizer 改善生成”，不能单独证明
具体 Jacobian 机制，也不能证明所有 RAE、所有 decoder 或所有 prior 尺度都
受益。MAE/SigLIP2 使用官方 DiT-XL，而既有 DINOv2 终验使用 DiTDH-S；
每个 tokenizer 内的 Flow/LPL 因果比较严格，但不同 tokenizer 之间的绝对
FID 不作排名解释。

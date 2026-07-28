# RAE floor path：5k checkpoint 持久性门控结果

## 结论

`floor=0.2,p=2` 没有通过事前门控，停止该 schedule 路线，不进入多 seed 或更长训练。
2k 时它接近 static 是早期瞬态；到 5k，static 明显加速学习，而 floor candidate 基本退回
original annealed 的水平。

## 实验口径

- 三路均为 seed `3407`、相同 fp32 latent stream、相同模型与训练配置。
- floor candidate 从原 step-2000 checkpoint 连同 optimizer、scheduler、EMA、RNG 和 cache
  offset 原位续训到 step 5000。
- 统一采样 seed `20260718`、1000 图、50 Euler steps、fp32、关闭 TF32。
- 这是单训练 seed、1k 样本门控，绝对 FID/KID 不能作为正式论文数字。

## 5k 结果

| condition | FID 1k | KID 1k |
|---|---:|---:|
| **static** | **229.672** | **0.196240** |
| original annealed | 267.851 | 0.240071 |
| floor 0.20, p=2 | 267.030 | 0.242542 |

floor candidate 相对 static 的 FID 劣化 `16.27%`，KID 劣化 `23.59%`，平均劣化
`19.93%`。它只在 FID 上比 annealed 好 `0.82`，KID 反而更差，因此“同时改善 FID/KID”
和“距 static 不超过 2%”两项门控都失败。

## 2k 到 5k

| condition | FID 改善 | KID 改善 |
|---|---:|---:|
| static | 16.95% | 35.07% |
| original annealed | 6.02% | 22.74% |
| floor 0.20, p=2 | 3.42% | 18.13% |

关键不是 floor 在 5k 变坏，而是它的学习速度没有保持：static 在语义开始形成的阶段进步
明显更快。5k normalized velocity loss 中，floor (`0.3164`) 甚至略低于 annealed
(`0.3182`)，但 KID 更差。这再次说明单个训练 loss 不足以判断生成质量。

## 解释边界

当前证据否定的是：给 annealed detail coefficient 加正 floor 足以恢复 static 的生成学习。
它并不否定原路径存在条件数问题。后续 held-out error atlas 专门区分“局部条件数被修复”
与“真正的生成瓶颈被修复”。

结果与图位于：

```text
~/data/eqvae/experiments/rae_path_schedule_train/checkpoint5k_gate/
```

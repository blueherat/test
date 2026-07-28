# RAE path 2k->5k 交叉分叉：事前协议

## 问题

已有独立确认显示，`floor/annealed` 路径的低噪声 basis/semantic 梯度在训练后期由协同
翻为冲突，而 `static` 保持稳定。该相关性仍有两种解释：

1. **后期目标效应**：2k 之后继续使用 time-dependent path 导致翻转；切回 static 应恢复。
2. **早期状态滞后**：前 2k 已把模型带入不同吸引域；之后换目标也难以恢复。

本实验使用一个 2x2 crossover 区分二者，不扫描超参数。

## 固定设计

四路均从已有 online step-2000 checkpoint 分叉，训练到 step 5000：

| 名称 | 前 2k checkpoint | 后 3k 路径 |
|---|---|---|
| `floor_to_floor` | floor 0.20, p=2 | floor 0.20, p=2 |
| `floor_to_static` | floor 0.20, p=2 | static |
| `static_to_static` | static | static |
| `static_to_floor` | static | floor 0.20, p=2 |

- seed `3407`，rank-16 basis，fp32，关闭 TF32。
- 使用相同 latent cache；从原 `branch_start_step=0` 计算消费偏移，后 3k 从第 64000 个
  cache 样本继续。
- 恢复 checkpoint 中的 model、EMA、optimizer、scheduler、CPU/CUDA RNG 和 epoch。
- 四路各占一张 4090，均为单进程；不改变原 checkpoint 的 world size。
- `floor_to_floor` 与 `static_to_static` 是 replay control。其 step-5000 model/EMA 必须与
  旧同路径 checkpoint 完全一致；否则停止解释 crossover。

## 指标与比较

主要比较使用固定 held-out latent、固定 noise/label 和固定采样 seed：

1. 低噪声 `t=0.1` last-block semantic descent ratio。
2. 1k 固定样本 FID/KID，只作单 seed 相对筛选。
3. generated-latent cycle residual 与 decoder local sensitivity。

定义晚期 floor 效应：

```text
within early-floor: floor_to_floor - floor_to_static
within early-static: static_to_floor - static_to_static
```

对 FID、KID、closure，正值表示晚期 floor 更差；对 semantic descent ratio，负值表示
晚期 floor 更冲突。

## 事前判据

先检查可解释性：

- replay model/EMA tensor 必须逐元素一致。
- replay optimizer/scheduler step 和 cache offset 必须一致。
- 四路训练必须有限，无 NaN/Inf。

若可解释性通过，判断如下：

### H1：后期目标主导

- `floor_to_static` 在 `t=0.1` 的 descent ratio 比 `floor_to_floor` 至少高 `0.05`；
- `static_to_floor` 比 `static_to_static` 至少低 `0.05`；
- 两个 switch 的 FID/KID 和 closure 都朝其**后期路径**的 replay control 移动。

四项趋势中至少三项成立，才支持后期目标主导。

### H2：早期状态滞后

- `floor_to_static` 仍更接近 `floor_to_floor`；
- `static_to_floor` 仍更接近 `static_to_static`；
- 上述接近关系在 gradient、generation、closure 至少两个层面一致。

### H3：交互

若两个切换方向明显不对称，或 gradient 恢复而 generation/closure 不恢复，则结论是早期
状态与后期目标存在交互，局部梯度翻转不是 rollout gap 的充分原因。

## 停止规则

- replay 不一致：只修恢复逻辑，不解读实验结果。
- H1/H2 均不成立：停止该路径调度路线，不追加 floor/power/rank 搜索。
- 即使 H1 成立，本实验也只支持机制因果，不支持优于从头 static 或跨 seed 泛化。

## Replay 审计修正记录（看 crossover 指标之前）

首次 full-state fork 后，`floor_to_floor` 的 model/EMA/optimizer/scheduler/RNG 与旧 5k
全部逐 tensor 一致；`static_to_static` 不一致。排查确认：

- 旧 static 是 0→10k 单进程连续训练，2k 后没有重建 DataLoader iterator；
- 旧 floor 是 0→2k 后恢复到 5k，2k 后重建过 iterator；
- RAE transport 的 `t` 使用 CPU `torch.rand`，重建 iterator 会额外消耗全局 CPU RNG；
- checkpoint 保存了模型 RNG，却没有保存 DataLoader iterator 状态。

因此首次结果尚未计算、也不解释。修正仅作用于恢复机制，不改模型、目标、数据或判据：

- static-origin 两路为 DataLoader 提供独立 generator，使 iterator 重建不消耗恢复出的全局
  CPU RNG，从而复现旧连续 static；
- floor-origin 两路保留原重启语义，从而复现旧 resumed floor；
- 四路重新写入新目录 `rae_path_crossover_train_v2`；仍要求两个 replay 全部逐 tensor 一致。

## Online/EMA 事后机制确认协议

在 v2 指标计算前未预料到，step-5000 EMA 仍保留 `0.9995^3000=22.3%` 的 step-2000
历史。首轮 EMA 结果显示梯度已恢复、生成只部分恢复，因此新增一个不训练的替代解释检查：

- 对同一四个 step-5000 checkpoint，分别采样 checkpoint `model` 与 `ema`；
- closure 使用同一 64 noise/label，分别评估 online 与 EMA；
- 不改变模型权重，不选择 checkpoint，不根据结果调 sampling 参数。

首轮 1k online 发现 `floor_to_static` 同时优于 `floor_to_floor` 与 `static_to_static`。在扩大
样本前固定确认判据：使用相同 50-step sampler、fp32、seed `20260718`，扩大到 5k；只有
以下三项同时成立，才认为“前期 floor、后期 static”具有进入多训练 seed 的资格：

1. `floor_to_static` 的 FID/KID 都优于 `floor_to_floor`；
2. `floor_to_static` 的 FID/KID 都优于 `static_to_static`；
3. `static_to_floor` 的 FID/KID 都差于 `static_to_static`。

5k 仍不是论文结果，也不支持跨训练 seed 泛化；任一条件失败即停止 curriculum 方法线。

# ImageNet-100 SiT CAFM tangent predictivity v1

## 实验范围

- 强模型：SiT-S/2，velocity prediction，800K EMA。
- 数据：ImageNet-100 SD-VAE latent cache。
- critic：CAFM 风格 SiT tangent critic，FP32。
- 训练：4 seeds，1000 optimizer steps，batch 32，gradient accumulation 8，effective batch 256。
- 优化器：AdamW，learning rate `1e-5`，betas `(0.0, 0.95)`，weight decay `0`。
- 条件 dropout：`0.1`。
- 验证：每 100 step 固定验证；step 1000 使用 2048 个固定样本。
- 审计：每个 critic 使用相同的 4096 个 teacher-path 样本，对 26 个候选方向计算 A/B 分数。
- 本轮没有更新生成器，也没有重新计算 FID。

## Seed 0 延长训练

在原 seed 0 的 step 1000 checkpoint 上，使用 2-GPU DDP、保持 global effective batch 256，继续训练到 step 5000。全程最佳 validation loss 为 `2.000394999`（step 4000），step 5000 为 `2.001162964`；两者仍接近恒定输出 0 的 `2.0` 参考值。完整配置、连续曲线和原始验证文件见 `critic_training_ddp2_seed0_5000/`。

## 最终验证数据

LSGAN loss 定义下，恒定输出 0 的参考值为 `2.0`。

| seed | step | validation loss | margin | real sign accuracy | fake sign accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 2.021037959 | 0.000038455 | 0.485352 | 0.521973 |
| 1 | 1000 | 2.030064411 | -0.000131831 | 0.357422 | 0.637207 |
| 2 | 1000 | 2.016702503 | 0.000417118 | 0.582031 | 0.415527 |
| 3 | 1000 | 2.014850106 | -0.000773813 | 0.532227 | 0.453125 |
| mean | 1000 | 2.020663745 | -0.000112518 | - | - |

机器可读版本见 `critic_final_validation.csv`。

## 文件

- `critic_training/seed*/run.json`：训练配置、模型与 checkpoint provenance。
- `critic_training/seed*/train_log.csv`：训练日志。
- `critic_training/seed*/validation_*.json`：固定验证集上的逐 checkpoint 指标。
- `critic_training_ddp2_seed0_5000/`：seed 0 从 step 1000 延长至 5000 的 DDP 配置、日志与验证数据。
- `audit/seed*/manifest.json`：审计配置、输入哈希与 checkpoint 哈希。
- `audit/seed*/direction_scores.csv`：每个 critic 的方向和时间分段结果。
- `audit/all_critic_scores.csv`：四个 critic 的完整合并表。
- `audit/aggregate_scores.csv`：跨 critic 聚合表。
- `audit/predictions_with_historical_quality.csv`：与已有 FID 数据的键连接结果。
- `audit/summary.json`：聚合统计。

模型 checkpoint、逐样本 NPZ、缓存和 smoke 产物未收入 Git。

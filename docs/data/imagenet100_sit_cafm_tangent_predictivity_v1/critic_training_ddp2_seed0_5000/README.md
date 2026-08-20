# Seed 0 DDP continuation to 5000 steps

## 配置

- 起点：原四 seed 实验的 seed 0、step 1000 optimizer checkpoint。
- 强模型：SiT-S/2 velocity，800K EMA，训练期间冻结。
- critic：CAFM 风格 SiT tangent critic，FP32。
- 设备：2 GPUs，DDP。
- local batch：16/GPU；gradient accumulation：8；global effective batch：256。
- 优化器：AdamW，learning rate `1e-5`，betas `(0.0, 0.95)`，weight decay `0`。
- 条件 dropout：`0.1`；centering scale：`1e-3`。
- 验证：每 250 optimizer steps，2048 个分布式固定验证样本。
- 保存：每 500 optimizer steps。
- 本实验没有更新强生成器，也没有进行采样或 FID 评估。

LSGAN loss 下，恒定输出 0 的参考 loss 为 `2.0`。

## 结果

| step | validation loss | margin | real sign accuracy | fake sign accuracy |
|---:|---:|---:|---:|---:|
| 1000 | 2.021037959 | 0.000038455 | 0.485352 | 0.521973 |
| 2000 | 2.007631450 | -0.000180099 | 0.666992 | 0.330078 |
| 3000 | 2.003478680 | 0.000113265 | 0.594727 | 0.411621 |
| 4000 | **2.000394999** | 0.000352751 | 0.749512 | 0.260254 |
| 5000 | 2.001162964 | 0.000617154 | 0.167969 | 0.837402 |

- 全程最佳 validation loss：`2.000394999`，位于 step 4000。
- step 5000 validation loss：`2.001162964`。
- step 5000 checkpoint SHA256：`42b2b0b662d1ebacfdb318593a10b5d8ac00773094a25a2ced7600c9dc3cdc86`。
- 3000 到 5000 阶段耗时 `4164.74` 秒；单卡 PyTorch peak allocated memory 为 `10000.23 MiB`。

## 文件

- `validation_curve.csv`：step 1000 到 5000 的连续验证曲线。
- `validation_*.json`：DDP 接续阶段的逐次原始验证结果。
- `train_log.csv`：step 1010 到 5000 的训练日志；每次接续的 elapsed time 会重新计时。
- `run.json`：最终 3000 到 5000 阶段的完整运行配置和模型 provenance。
- `complete_003000.json`、`complete_005000.json`：对应阶段的完成标记。
- `checkpoint_manifest.csv`：未收入 Git 的模型 checkpoint 哈希。

模型 checkpoint、数据缓存和控制台日志未收入 Git。

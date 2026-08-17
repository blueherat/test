# 官方 SiT-S/2 近期冻结头实验复现

本报告只整理实验配置与结果，不作机制解释。大权重、生成样本和运行日志保留在本机 `/data`，不进入 Git。

## 官方来源与适配

- 官方仓库：`nyu-visionx/SiT-collections`
- 官方文件：`SiT-S-2-256.pt`
- 原始权重 SHA256：`a245dc6330cd0d5906a5da00718b7d348a417d740e6b6cfeeb504e9d1448d070`
- ImageNet-100 子集权重 SHA256：`2f1cedb41a721ea85b910b7087480770dcb5b2646e62cc1a495fd250a0a5e8e8`
- 类别与 unconditional 输出等价审计：`True`，最大绝对误差分别为 `0.0` 和 `0.0`。
- 官方最终权重没有发布训练 step；实验记录中的 `source_step=0` 仅是带来源校验的哨兵值。

## 复现范围

- 起始提交：`3901741617bdae33e31c9dca04c89b3d7bec8696`
- 代码提交范围终点：`b5a5a3bbaa072135f113da1cf205f779b0542a6c`
- 正式流程开始：`2026-08-17T02:40:28+0800`
- 正式流程结束：`2026-08-17T15:46:45+0800`
- 所有 FID 横向条件使用相同初始噪声、类别、VAE、ODE 和 ADM reference。
- 指标为单 seed、1,000 张样本的配对 FID，不等同于正式 FID-50K。
- 官方只发布一个最终 SiT-S/2 state dict，因此同一训练轨迹的 EMA 权重外推不可识别，未构造替代实验。

## 训练结果

| 实验 | 目标 | 深度 | 可训练参数 | step | 首个 loss | 最终 loss | EMA native MSE | EMA velocity MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final_linear_x | clean | 12 | 6160 | 50000 | 0.659531 | 0.251047 | 0.249752 | 1.672872 |
| depth8_v | velocity | 8 | 301840 | 50000 | 1.626833 | 0.874851 | 0.863490 | 0.863490 |
| depth8_x | clean | 8 | 301840 | 50000 | 0.637316 | 0.250869 | 0.273681 | 1.235707 |
| depth8_epsilon | epsilon | 8 | 301840 | 50000 | 0.964604 | 0.336130 | 0.342885 | 1.135969 |
| depth12_x_full | clean | 12 | 301840 | 50000 | 0.651055 | 0.233088 | 0.230219 | 1.106460 |

## 配对 FID-1000

| 实验 | baseline | auxiliary | 最佳外推 | 相对 baseline | 最佳内插 |
|---|---:|---:|---:|---:|---:|
| final_linear_x | 100.5173 | 151.0918 | extrap_gamma_0p01: 100.5827 | +0.0654 | - |
| depth8_v | 100.5105 | 218.6584 | extrap_gamma_0p4: 90.3528 | -10.1577 | - |
| depth8_x | 101.0902 | 224.7488 | extrap_gamma_0p15: 95.2320 | -5.8582 | - |
| depth8_epsilon | 101.0880 | 258.5801 | extrap_gamma_0p15: 93.1182 | -7.9698 | - |
| hidden_state_mixing | 101.0904 | 235.4708 | output_gamma_0p001: 101.1734 | +0.0829 | hidden_alpha_0p0175: 100.8747 |
| depth12_x_full | 101.0869 | 127.3345 | extrap_gamma_0p08: 100.9901 | -0.0969 | - |

## 文件

- 全部逐条件指标：`docs/data/imagenet100_official_sit_s2_recent_replication/fid1k_all.csv`
- 训练摘要：`docs/data/imagenet100_official_sit_s2_recent_replication/training_summary.csv`
- 结构化结果：`docs/data/imagenet100_official_sit_s2_recent_replication/results.json`
- FID 曲线：`docs/data/imagenet100_official_sit_s2_recent_replication/fid1k_sweeps.png`
- 配对预览：`docs/data/imagenet100_official_sit_s2_recent_replication/preview_comparison.png`
- hidden gap 审计：`docs/data/imagenet100_official_sit_s2_recent_replication/hidden_state_gap_audit.csv`

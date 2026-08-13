# ImageNet-100 SiT 400K guidance 对照实验数据

该目录保存
[`IMAGENET100_SIT_400K_GUIDANCE_MECHANISM_AUDIT_ZH.md`](../../IMAGENET100_SIT_400K_GUIDANCE_MECHANISM_AUDIT_ZH.md)
使用的便携汇总数据和图片。

| 文件 | 内容 |
|---|---|
| `mechanism_audit.csv/json` | floor、时间窗口、平行/正交、同目标 AutoGuidance 的 28 条 FID-5K 汇总 |
| `mechanism_audit.png` | floor 与同目标 AutoGuidance 总览 |
| `direction_component_fid5k.csv` | `x400/v270` full、parallel、orthogonal 闭环指标 |
| `direction_comparison_summary.json` | 方向分量 FID 与平行系数比较 |
| `direction_comparison.png` | 两类 guidance 的 FID 分量和方向比较 |
| `direction_geometry_by_time.csv` | 512 个样本在 11 个时刻的聚合方向统计 |
| `direction_geometry_summary.json` | teacher/rollout 两种状态的总体统计 |
| `direction_geometry.png` | 方向 cosine、幅值与正交能量随时间的变化 |

所有 FID 条件使用相同的 5000 个初始噪声与类别标签。该目录不包含 checkpoint、生成
样本、数据集、FID reference 或逐样本方向表。

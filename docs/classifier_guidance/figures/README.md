# 训练图表与固定快照

静态评估图：[JiT / SiT checkpoint FID-5K](gan_checkpoint_fid_20260923.png)，
对应[评估报告](../GAN_CHECKPOINT_FID_20260923_ZH.md)，直接纳入 Git。

## 2026-09-23 固定快照

| 内容 | Git 快照 | 本地实时文件 |
|---|---|---|
| JiT 系数、EMA 与历史 | [PNG](snapshots/20260923/jit_gan_schedule.png) | `jit_gan_schedule.png` |
| JiT GAN 诊断 | [PNG](snapshots/20260923/jit_gan_health.png) | `jit_gan_health.png` |
| SiT 系数与历史 | [PNG](snapshots/20260923/sit_joint_schedule.png) | `sit_joint_schedule.png` |
| SiT 系数及尺度 | [PNG](snapshots/20260923/sit_joint_guidance.png) | `sit_joint_guidance.png` |
| SiT GAN 诊断 | [PNG](snapshots/20260923/sit_joint_gan_health.png) | `sit_joint_gan_health.png` |
| SiT 监控状态 | [JSON](snapshots/20260923/sit_joint_status.json) | `sit_joint_status.json` |

[快照清单](snapshots/20260923/manifest.json)记录复制时间、各来源文件的修改时间、字节数和 SHA-256。
监控器逐个原子更新图表，快照也逐个读取；这些文件不保证对应完全相同的训练步。

## 本地实时预览

上表第三列文件继续由 [`monitor_jit_schedule.py`](../../../classifier_guidance/monitor_jit_schedule.py)
和 [`monitor_sit_joint.py`](../../../classifier_guidance/monitor_sit_joint.py) 在本目录刷新，
可直接通过编辑器打开。实时文件及写入临时文件由 `.gitignore` 排除，固定快照保留在
`snapshots/<日期>/`。仅 clone 仓库时请查看固定快照；实时文件需要本机训练数据与监控程序。

历史报告里的实时 PNG 链接仍指向本地预览。此次整理保留正在运行的训练、监控、checkpoint
和既有停止标记，不改变训练状态。

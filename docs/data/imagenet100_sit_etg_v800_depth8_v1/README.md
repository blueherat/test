# SiT v800 Error-Triangulated Guidance 验证数据

该目录保存 `v800` depth-8 velocity/clean/epsilon 三头 ETG 验证的便携数据和图。完整中文说明见：

- `docs/IMAGENET100_SIT_ETG_V800_RESULTS_ZH.md`

## 结果摘要

- 三个现成弱头严格共享冻结的 `v800` EMA backbone、depth-8 feature、训练步数与数据协议，无需重新训练。
- 三角估计在 12/12 个时间 bin 都产生负的 velocity 私有方差；正则化后平均最大 head 权重为 `85.45%`。
- ETG channel gap 与 velocity gap 的平均夹角仅 `3.57` 度，非平行能量约 `0.255%`。
- 最佳配对 FID-1K：ETG channel `74.2426`，single velocity `74.3389`，只差 `0.0963`。
- 该版本未通过预注册的 `1.0` FID 与非平凡方向差异保留门。

## 文件

| 文件 | 内容 |
|---|---|
| `calibration.json` | checkpoint/protocol 审计、两 seed rollout 校准、三角估计、teacher 诊断 |
| `geometry_audit.json` | ETG 与 velocity gap 的整体和逐时刻几何统计 |
| `geometry_audit.csv` | 可直接分析的逐时刻几何表 |
| `etg_fid1k.csv` | 46 个严格配对条件的 FID/sFID/IS/NFE 与样本指纹 |
| `fid_summary.json` | baseline、各 mode 最佳点及全局最佳点 |
| `weights_and_geometry.png` | 三头权重与 ETG/velocity 夹角随时间变化 |
| `fid1k_sweep.png` | 独立 scale 扫描的 FID 曲线 |

生成的图像 NPZ、预览、逐条件日志和 checkpoint 均保留在数据盘，没有复制进仓库。完整实验目录约 `8.5 GB`。

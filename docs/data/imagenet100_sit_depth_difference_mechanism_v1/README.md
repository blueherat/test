# SiT 深度有限差分机制便携数据

本目录对应
[`IMAGENET100_SIT_DEPTH_DIFFERENCE_MECHANISM_RESULTS_ZH.md`](../../IMAGENET100_SIT_DEPTH_DIFFERENCE_MECHANISM_RESULTS_ZH.md)。

## 文件

- `depth_difference_geometry_per_sample.csv`：512 个样本、9 个时刻、teacher 与
  rollout 两种 context 的逐样本几何；
- `depth_difference_geometry_by_time.csv`：逐时刻聚合；
- `depth_difference_geometry_summary.json`：公式、checkpoint hash 和总体汇总；
- `depth_difference_geometry.png`：cosine、投影系数和残差能量时序；
- `v800_projection_metrics.csv`、`v800_projection_summary.json`：
  `W`、`W_parallel`、`W_orthogonal` 的配对 FID-1K；
- `x800_transfer_metrics.csv`、`x800_transfer_summary.json`：x800 上的完整系数扫描；
- `depth_difference_fid.png`：两组 FID 结果图。

## 复现入口

```bash
python experiments/analyze_imagenet100_sit_depth_difference_geometry.py

python experiments/run_imagenet100_sit_depth_difference_mechanism.py \
  --phase v_projection --gpu 2

python experiments/run_imagenet100_sit_depth_difference_mechanism.py \
  --phase x_transfer --gpu 3
```

本目录不包含 checkpoint、生成样本、ImageNet、latent cache 或日志。

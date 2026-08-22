# 数据说明

本目录归档两组 ImageNet-100 SiT 实验的便携结果，不包含模型权重、生成样本或数据集。

## 主要文件

- `best_results.csv`：四个实验族的最优点与基线。
- `fid1k_sweeps.png`：所有正式扫描曲线。
- `weak_head_difference_fid1k.csv`：`depth8-depth4` 弱头差值的 13 点正式扫描。
- `x800_depth4_readout_fid1k.csv`：三个 depth4 读出目标的 13 点扫描；另保留早期 9 点弱头差值粗扫。
- `training_validation_step50k.csv`：三个读出头在 50K 的验证指标。
- `*_protocol.json` 与 `*_summary.json`：协议和原始汇总。
- `*_run_config.json` 与 `*_train_metrics.jsonl`：三个读出头的训练配置和完整标量日志。

完整配置与结果表见 `docs/IMAGENET100_SIT_WEAK_DIFFERENCE_X800_READOUT_RESULTS_ZH.md`。

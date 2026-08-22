# ImageNet-100 SiT 对角矩残差 20K 终验

本目录只保存可复核的小型指标。checkpoint、SD-VAE latent cache、ADM reference、生成
样本和 train statistics 均留在本机数据目录，不进入 Git。

## 协议

- model：SiT-S/2；
- dataset：ImageNet-100 SD-VAE latent cache；
- step：20,000；
- weights：raw `model`，不使用 EMA；
- conditions：native velocity 与 train-only diagonal-LMMSE residual velocity；
- sampling：无 guidance、5000 张、Dopri5、相同 noise/label；
- evaluation：同一 ImageNet-100 validation 5K ADM reference；
- paired validation：5000 个固定样本与随机量。

## 结果

| condition | validation velocity MSE | FID-5K | sFID | IS |
|---|---:|---:|---:|---:|
| native raw | 0.8283545 | 155.6983 | 84.2994 | 9.0112 |
| diagonal residual raw | 0.8286877 | 161.0571 | 89.0443 | 8.2944 |

`paired_time_bins.csv` 保存逐时间段的配对风险；`summary.json` 保存协议、hash、总体结果
和停止判断。正式解释见
`docs/MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md` 第 21 节。

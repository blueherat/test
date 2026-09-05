# RAEv2 PFR / OU 迁移的便携结果

本目录只保存可审计的小型指标表，不保存 `samples.npz`、checkpoint 或逐 rank 预览图。

- `official_fid1k.csv`：两个独立配对 seed 的官方 `nanogen-evals` 指标，以及采样调用数；
- `official_fid5k_raw_pfr.csv`：预注册 raw PFR 与 ordinary IG 的正式 5K 结果。

共同设置：官方 RAEv2 DINOv3-L K7 checkpoint step `100080`、EMA、100-step shifted Euler、
CFG `1.0`、IG scale `1.78`、ImageNet-256 官方 FID reference。OU 条件原样沿用 raw PFR 的
`h=1/32,rho=0.05`，没有针对 RAEv2 重新调参。

原始大文件位于：

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_pfr_retiming_v1/
/home/zhoushunyu/data/eqvae/experiments/raev2_pfr_ou_polar_v1/
```

结论边界：两个 FID-1K bank 中 OU-common / OU-polar 均优于 ordinary IG，但均不如 raw PFR；
raw PFR 的 FID-1K 正信号又未通过正式 FID-5K。因此当前 RAEv2 迁移判为阴性。

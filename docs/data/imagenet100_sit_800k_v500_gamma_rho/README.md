# SiT 800K v500 gamma-rho 扫描便携数据

该目录保存 `v800` strong + `v500` weak 的 factorized response 扫描聚合结果。模型权重、生成样本、逐条件日志和 ADM 中间特征仍保存在本机数据盘，没有写入 Git。

## 实验定义

```text
b' = S(b,t)
g_b = S(b,t) - W(b,t)
z' = S(b,t) + rho * [S(z,t) - S(b,t)] + gamma * g_b
```

- dataset：ImageNet-100；
- model：SiT-S/2；
- strong：native velocity `v800` EMA；
- weak：native velocity `v500` EMA；
- sampler：Dopri5，FP32/TF32，CFG=1；
- metric：ADM FID/sFID/IS；
- screening：1,000 samples，sample seed 0；
- 所有条件使用相同 initial noise、class labels 和 reference statistics。

## 文件

| 文件 | 内容 | 行数 |
|---|---|---:|
| `v500_gamma_rho_fid1k.csv` | 去重后的全部 gamma-rho 网格，包含粗扫与细扫 | 63 |
| `v500_gamma_rho_best_by_gamma_fid1k.csv` | `rho=1.00...1.50` 内每个 gamma 的最低 FID 条件 | 5 |
| `v500_gamma_rho_duplicate_audit.csv` | `rho=1.50` 重复端点的数值复现检查 | 4 |
| `v500_gamma_rho_headline_fid1k.csv` | factorized 最优点与 tuned closed AutoGuidance 对照 | 2 |
| `v500_gamma_rho_fid1k.png` | 细扫 FID 曲线及 closed 基准线 | 1 figure |

`v500_gamma_rho_fid1k.csv` 每个 `(gamma,rho)` 只保留一个 canonical artifact；若同一条件被重复运行，则选择更新时间更晚的完整结果，同时保留 `replicate_count` 与 `fid_replicate_spread`。四个重复端点的最大 FID spread 为 `0.0033`。

本地完整结果来源：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/
  factorized_guidance_800k_v1/v500_gamma_rho_grid_n1000_seed0/
```

报告正文：[`docs/IMAGENET100_SIT_800K_RESPONSE_AMPLIFICATION_RESULTS_ZH.md`](../../IMAGENET100_SIT_800K_RESPONSE_AMPLIFICATION_RESULTS_ZH.md)

默认细扫启动脚本：[`experiments/launch_imagenet100_sit_800k_v500_gamma_rho_grid.sh`](../../../experiments/launch_imagenet100_sit_800k_v500_gamma_rho_grid.sh)

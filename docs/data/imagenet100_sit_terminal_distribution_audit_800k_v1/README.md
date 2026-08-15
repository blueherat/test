# SiT 800K 终端分布控制审计便携数据

该目录保存 `v800` strong + `v500` weak 的终端分布控制审计结果。模型权重、逐样本 latent 轨迹、ADM 激活、解码样本和日志仍保存在本机数据盘，没有写入 Git。

## 实验定义

baseline trajectory：

```text
b' = S(b,t)
g_b = S(b,t) - W(b,t)
```

factorized control：

```text
z' = S(b,t) + rho * [S(z,t) - S(b,t)] + gamma * g_b
```

相对于当前 strong field `S(z,t)` 的控制为：

```text
u = gamma * g_b + (rho - 1) * [S(z,t) - S(b,t)]
```

控制 action 按下式记录：

```text
A = 0.5 * integral E[mean(u^2)] dt
```

同时保留正常 closed AutoGuidance：

```text
z' = S(z,t) + gamma * [S(z,t) - W(z,t)]
```

## 协议

- dataset：ImageNet-100；
- model：SiT-S/2；
- strong：native velocity `v800` EMA；
- weak：native velocity `v500` EMA；
- sampler：Dopri5，FP32/TF32，CFG=1；
- factorized solver：`atol=1e-7, rtol=1e-4`；
- closed `gamma=3` 独立积分：`atol=1e-8, rtol=1e-5`；
- evaluation：每个条件 1,000 张图，sample seeds `0,1`；
- metric：官方 ADM FID/sFID/IS；
- 所有条件在每个 seed 内使用相同 initial noise、class labels 和 reference statistics；
- 轨迹记录 20 个时间点，端点保存 FP32；
- 终端图像分布使用 ADM pool-3 features、PCA-128、SWD/MMD/Frechet 和线性 C2ST；
- C2ST cross-condition 与 split-null 均固定为每类 500 个 group。

## 条件

| 名称 | 类型 | gamma | rho |
|---|---|---:|---:|
| `factorized_g1_r1p5` | factorized | 1.0 | 1.50 |
| `factorized_g1p5_r1p35` | factorized | 1.5 | 1.35 |
| `factorized_g2_r1p35` | factorized | 2.0 | 1.35 |
| `factorized_g2p5_r1p35` | factorized | 2.5 | 1.35 |
| `factorized_g3_r1` | factorized/frozen | 3.0 | 1.00 |
| `closed_g3` | closed AutoGuidance | 3.0 | 1.00 |

## 文件

`combined_*.csv` 保留两个采样 seed 的逐 seed 结果，`aggregate_*.csv` 给出跨 seed mean/std。

| 文件组 | 内容 |
|---|---|
| `*_condition_summary.csv` | FID/sFID/IS、action、path length 与相对 baseline 位移的联合主表 |
| `*_action.csv` | forcing、strong-response control、cross term 和总 action |
| `*_diagnostics.csv` | 20 个时间点的逐条件动力学统计 |
| `*_path.csv` | snapshot path length 和 paired endpoint RMS |
| `*_feature_pairwise.csv` | 解码后 ADM 特征的 pairwise 分布检验与 split-null 校准 |
| `*_latent_pairwise.csv` | 轨迹各时间点的随机投影 latent pairwise 诊断 |
| `*_paired_feature.csv` | 同 noise/class 下 ADM 特征的逐样本差异 |
| `*_equivalence.csv` | factorized 合批积分与逐条件独立积分的数值等价检查 |
| `seed*_sampling_manifest.json` | 每个采样 seed 的完整模型、采样和 fingerprint 配置 |
| `seed*_analysis_manifest.json` | 分析维数、bootstrap/C2ST 设置和数据规模 |
| `summary_manifest.json` | 跨 seed 汇总清单 |
| `terminal_distribution_audit_summary.{png,pdf}` | 质量/action、action 分解、轨迹差异和终端 C2ST 汇总图 |

本机完整结果来源：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/
  terminal_distribution_audit_800k_v1/
```

正文报告：[`docs/IMAGENET100_SIT_TERMINAL_DISTRIBUTION_CONTROL_AUDIT_ZH.md`](../../IMAGENET100_SIT_TERMINAL_DISTRIBUTION_CONTROL_AUDIT_ZH.md)

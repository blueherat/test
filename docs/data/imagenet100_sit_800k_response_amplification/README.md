# SiT 800K strong-response amplification 便携数据

该目录只保存本轮报告需要的轻量聚合表。模型权重、原始 latent、生成样本 NPZ、FID 临时特征和日志均保存在本机数据盘，没有写入 Git。

## 文件

| 文件 | 内容 | 条件数 |
|---|---|---:|
| `formal_fid5k.csv` | x800 response `rho=1.3` 与 tuned closed `gamma=1.125`，2 个采样 seed | 4 |
| `response_screen_fid1k.csv` | x800/v500 的粗 response 与 closed sweep | 17 |
| `response_refinement_fid1k.csv` | x800/v500 的局部 response/closed refinement | 14 |
| `factorized_screen_fid1k.csv` | nominal scale 与 online orthogonal scale 的首轮二维筛查 | 40 |
| `joint_response_direction_fid1k.csv` | x800 的 response scale x online direction scale | 12 |
| `frozen_low_gain_control_fid1k.csv` | x800 frozen `gamma={0.5,0.75}` | 2 |
| `frozen_high_gain_control_fid1k.csv` | x800 frozen `gamma={1.25,...,3}` | 6 |
| `headline_summary.csv` | 正式结果、历史配对 baseline 与均值/差值 | 8 |

所有 screen CSV 由 `experiments/summarize_imagenet100_sit_factorized_guidance_screen.py` 从逐条件 `nominal_intervention_fid5k.json` 生成。脚本会按 `family + sample seed + sample count` 检查 noise/label fingerprint 一致性。

正式结果与 1K sweep 没有画在同一张图中：正式结果只有两个离散采样 seed，而 1K sweep 是选参与机制筛查，混合成连续曲线容易让有限样本精度看起来比实际更高。主报告因此使用精确表格，并完整保留原始聚合行供复算。

CSV 中的 `result` 列是本机完整 artifact 的 provenance 路径；在其他机器上可能不可访问，不影响其余聚合字段使用。

## 协议

- dataset：ImageNet-100；
- strong：native velocity SiT-S/2 `v800` EMA；
- weak：JiT-style x-output `x800` EMA，主要正式比较；`v500` EMA 用于跨 weak-family 对照；
- sampler：Dopri5，FP32/TF32，CFG=1；
- metric：ADM FID/sFID/IS；
- formal：5000 samples x 2 sample seeds；
- screening：1000 samples x 1 sample seed；
- 同一个 seed 内所有条件使用相同 initial noise 与 class labels。

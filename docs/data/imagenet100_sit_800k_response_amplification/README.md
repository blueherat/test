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
| `v500_response_direction_dense_fid1k.csv` | v500 的 `gamma x rho x lambda` 完整细扫 | 231 |
| `v500_response_direction_formal_controls_fid5k.csv` | v500 closed 与不补在线方向的正式对照 | 4 |
| `v500_response_direction_candidate1_fid5k.csv` | 细扫中途最优参数的正式结果 | 2 |
| `v500_response_direction_final_fid5k.csv` | 完整细扫最终最优参数的正式结果 | 2 |
| `v500_response_direction_summary.csv` | 四种正式条件的两 seed 均值与 FID 差值 | 4 |

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

## v500 response-direction 细扫

本轮没有训练模型，只在 `v800` strong flow 与 `v500` weak flow 上改变推理动力学。记 baseline strong trajectory 为 `z_b`，weak-to-strong gap 为

```text
g_b(t) = S(z_b,t) - W(z_b,t)
```

实际筛查的 factorized field 为

```text
S(z_b,t)
+ rho * [S(z,t) - S(z_b,t)]
+ gamma * [g_b(t) + lambda * Orth_gb(g(z,t))]
```

其中 `rho` 控制 strong model 对偏移状态的响应，`lambda` 只恢复 current-state gap 相对 nominal gap 的方向变化分量。

完整 FID-1K 网格为：

| 参数 | 取值 |
|---|---|
| `gamma` | `1.25, 1.5, 1.75` |
| `rho` | `1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50` |
| `lambda` | `0, 0.125, ..., 1.25` |

共 `3 x 7 x 11 = 231` 个条件，全部使用同一个 seed 0 initial-noise/class pairing。最终 FID-1K 最优条件为：

```text
gamma=1.75, rho=1.30, lambda=1.25, FID-1K=72.4043
```

该点位于 `gamma` 和 `lambda` 的扫描上边界。x800 没有进入本轮 dense sweep。

## 正式 FID-5K

| 条件 | `gamma` | `rho` | `lambda` | FID seed 0 | FID seed 1 | FID 均值 | sFID 均值 | IS 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed AG | 3.00 | 1.00 | 1.00 | 48.5135 | 47.9738 | 48.2437 | 67.1667 | 34.9816 |
| Response，不补在线方向 | 1.50 | 1.35 | 0.00 | 48.0841 | 47.7515 | 47.9178 | 66.8418 | 33.3637 |
| 中途最优 | 1.75 | 1.25 | 1.125 | 46.3754 | 46.1226 | 46.2490 | 67.2159 | 34.4622 |
| 最终最优 | 1.75 | 1.30 | 1.25 | **46.0697** | **46.0003** | **46.0350** | 67.8223 | 33.9644 |

最终参数在两个严格配对 seed 上均优于 closed：

- FID 均值相对 Closed AG 降低 `2.2086`，约 `4.58%`；
- FID 均值相对不补在线方向降低 `1.8828`，约 `3.93%`；
- sFID 相对 Closed AG 增加 `0.6556`，IS 减少 `1.0172`。

因此本轮结果支持“response amplification 与在线方向修正组合可以稳定降低 FID”，但不支持“所有生成指标同步改善”。

## 一致性检查

- 231 个细扫条件的 `(gamma, rho, lambda)` 组合唯一且齐全；
- 所有 FID-1K 条件共享相同 noise/label fingerprint；
- 正式 seed 0 与既有 seed 0 对照逐样本配对，正式 seed 1 同样逐样本配对；
- seed 0 使用 batch 8，seed 1 使用 batch 2，以复用既有正式对照的准确随机流；
- `total_nfe` 是逐 batch 累加的 solver evaluation 次数，因此不同 batch size 的 seed 之间不能直接比较 NFE。

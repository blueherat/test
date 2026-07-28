# RAE well-conditioned path：2k tiny gate 结果

## 一句话结论

> 给 annealed detail path 加正的 observation floor，确实能修复一部分生成退化；但
> `1/k` 越小并不代表生成越好。最好的候选是保留原 `p=2` 形状、只加入适中
> `floor=0.2`，说明生成需要同时兼顾可逆条件数和 detail 的分阶段学习。

## 实验正确性

- 四条候选共享 source checkpoint、seed `3407`、latent cache、标签、time/noise 序列、
  optimizer 和 2k updates。
- 首批 `images/labels/time/noise/clean` 哈希四路完全一致，state/target 哈希按设计不同。
- 四个 checkpoint 均为 step 2000，含 EMA；每路 200 条训练记录全部有限，无 NaN/Inf，
  clip rate 为 0。
- 采样统一使用 seed `20260718`、50 Euler steps、fp32、TF32 关闭、ImageNet 1000 类各
  1 张。
- 绝对 FID 很高是因为模型只训练 2k steps 且只采 1k 图。本表仅用于同配置筛选。

## 生成结果

| condition | offline risk | FID 1k | KID 1k |
|---|---:|---:|---:|
| static | - | 276.556 | 0.302254 |
| original annealed | 1.000 | 284.999 | 0.310749 |
| floor 0.05, p=1 | 0.691 | 283.591 | 0.307563 |
| floor 0.15, rational alpha=0.5 | 0.705 | 279.565 | 0.301989 |
| floor 0.30, p=2 | 0.716 | 281.001 | 0.302673 |
| **floor 0.20, p=2** | **0.748** | **276.499** | **0.296254** |

四个候选的 FID/KID 都优于 original annealed。`floor=0.2,p=2` 最好：FID 与 static
几乎相同，KID 在该 1k screen 上更低。但差异来自单 seed、单 sampling seed，不能写成
“超过 static”的正式结论。

## 事前预测判定

1. 数值稳定：**成立**。四路无 NaN/Inf/clip。
2. 至少两个候选同时改善 FID/KID：**成立，实际为 4/4**。
3. static 最好或接近最好：**成立**。static FID 与最佳候选只差 `0.057`。
4. 候选按 offline risk 排序：**明确失败**。候选 risk 与 FID/KID 的 Spearman 均为
   `-0.8`，方向与预期相反。

## 机制解释

离线 risk 的价值与局限现在可以分开：

- 有效部分：它正确识别了原始 `k(t)->0` 是有害的。所有正 floor 候选都优于原
  annealed，说明去掉严重逆放大值得做。
- 无效部分：它假设换路径后模型的 raw observation error 不变，因此错误地预测
  `floor005_p1` 最好。真实训练会同时改变 state、velocity target 和各时间段的学习难度。

`floor005_p1` 在中段较早放回 detail，例如 `t=0.5` 时 `c=0.525`；
`floor020_p2` 此时只有 `c=0.4`。前者虽然解析条件更好，却削弱了“先学粗结构、后补
detail”的作用。后者做的是更小的干预：

```text
c_new(t) = 0.2 + 0.8(1-t)^2
k_new(t) = 0.2 + 0.8 k_old(t)
```

它把 `min k` 从 0 提到 0.2，同时保留原路径 80% 的 delay area。当前证据支持的是一个
sweet spot，而不是“conditioning 越强越好”。

## 当前研究价值

这轮把一个大而模糊的想法收缩成了更贴近生成模型的问题：

> 数据路径中的分阶段信息暴露可能帮助生成学习，但只有在 endpoint observation
> 保持良好条件数时才有效；最优路径由 curriculum benefit 与 inverse-conditioning
> cost 的平衡决定。

这比单纯做频带加权或追求 latent 群结构更贴近 flow/diffusion 的训练路径设计。不过，
目前仍只是 tiny empirical signal，还不是论文结论。

## 下一步门控

同索引 montage 显示 2k 输出仍以纹理为主、语义不足。因此在多 seed 前先做一个更便宜的
checkpoint 持久性门控：只把 `floor=0.2,p=2` 续训到 5k，并对比现成 static/annealed
5k。若 5k 仍通过，再用 seeds `3407/4211/5821` 对比：

- static；
- original annealed；
- floor-annealed `0.2,p=2`。

仍训练 2k、采样 1k。只有候选相对 original annealed 在至少 `2/3` seeds 同时改善
FID/KID，且平均不差于 static 超过 2%，才进入 10k/5k。否则停止，不再追加大实验。

## 结果位置

```text
~/data/eqvae/experiments/rae_path_schedule_screen/offline_oracle_n32_v3/
~/data/eqvae/experiments/rae_path_schedule_train/tiny_gate_1k/
```

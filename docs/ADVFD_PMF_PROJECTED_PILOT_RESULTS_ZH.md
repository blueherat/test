# AdvFD pMF-B 64维投影 Pilot 结果

日期：2026-08-23

状态：**paper-only clean-room 的缩放筛查，不是论文数值复现。** 本轮没有查看或
导入 AdvFD 官方实现。

## 1. 实验目的

本轮只验证以下链路能否端到端工作：

- pMF-B/16 官方 checkpoint 的单步生成；
- 连续生成图像到可微 Inception-2048；
- FD-Loss 式 EMA 一、二阶矩；
- G-step 后用相同 noise/class 重新生成的 D-step；
- raw adaptive FD 与 AdvFD real-feature whitening；
- checkpoint、严格配对采样和在线模型评估。

为了快速筛查，Inception-2048 被固定正交投影到 64 维，warm-start 只有 2048 张，
global batch 为 8，完整 125k-step schedule 被压缩到 120 步。adaptive branch 在第 10
步启动并用 40 步 warm-up。所有产物均标记
`paper_reproduction_metric=false`。

## 2. 配对 5K 结果

所有条件使用相同的 5000 个 noise、class labels、ImageNet reference stats 和在线模型
权重。adaptive critic 的 current-encoder 诊断仍只使用训练时严格预留的最后 512 张
ImageNet 图片。

| variant | step | FID-5K | 相对 base | 64维 FD | adaptive held-out FD |
|---|---:|---:|---:|---:|---:|
| base | 0 | 10.827492 | 0 | 0.046689 | - |
| static | 40 | 10.944540 | +0.117048 | 0.047419 | - |
| static | 80 | 10.929372 | +0.101880 | 0.046147 | - |
| static | 120 | 10.948130 | +0.120638 | 0.047022 | - |
| raw | 40 | 10.863531 | +0.036039 | 0.046778 | 0.382390 |
| raw | 80 | 10.866584 | +0.039092 | 0.045340 | 0.371532 |
| raw | 120 | 10.831783 | +0.004292 | 0.045834 | 0.389296 |
| real | 40 | 10.865733 | +0.038241 | 0.047190 | 5.732953 |
| real | 80 | 10.868413 | +0.040921 | 0.045681 | 5.695726 |
| real | 120 | 10.875604 | +0.048112 | 0.045427 | 5.767668 |

完整数据见
[`docs/data/advfd_pmf_projected_pilot_v1_fid5k.csv`](data/advfd_pmf_projected_pilot_v1_fid5k.csv)。
原始 JSON、训练 CSV、checkpoints 和样本图位于：

`/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_projected_pilot_v1`

## 3. 当前结论

1. 训练和评估链路通过；G/D 梯度有限，生成图 clamp 比例约 0.6%--0.7%，没有发现
   OOM、NaN、checkpoint 接续错误或采样不配对。
2. 本轮三个训练目标均未改善完整 Inception FID。static 是稳定的轻微恶化；real
   whitening 也是轻微恶化；raw 到 120 步接近 base，但没有正收益。
3. 64维 projected FD 与完整 Inception FID 不同步。例如若干 checkpoint 的投影 FD
   下降，但 FID 仍恶化。这说明固定低维目标本身已经可能产生 representation hacking，
   不能用该缩放结果否定完整 AdvFD。
4. raw critic 的 adaptive FD 从约 0.08 增至 0.13；real-whitened critic 从约 0.08
   增至 1.74，说明 critic 最大化方向确实生效。不同 calibration 下 FD 绝对值不可横比。
5. `adaptive_train_ema_fd` 与 held-out FD 的差异混合了 changing-encoder EMA、当前
   encoder、有限样本偏差和数据泛化，不能直接解释为 train/held-out overfitting。

## 4. 为什么在这里停止延长

论文正式 pMF 配方使用完整 SIM（SigLIP、Inception、MAE）、每个表示完整维数、50K
warm-start、global batch 1024、step 1000 启动 AdvFD、4000 步 warm-up 和 125K
generator steps。当前 64 维、120 步压缩 schedule 同时改变了目标可观测空间和优化时间
尺度。继续延长它只会回答一个缩放后的新问题。

因此下一阶段恢复完整 Inception-2048 和 50K warm-start，先复现论文最小
Inception-only FD-Loss 行为；通过后再加入完整 real-whitened adaptive branch，最后扩到
正式 SIM。AdvFD 论文的 pMF-B 正式结果是 FD-Loss FID 0.85、AdvFD FID 0.81，均基于
50K 样本正式评估，不能与本表的 FID-5K 绝对值直接比较。

一手来源：

- [AdvFD](https://arxiv.org/abs/2608.11205)
- [FD-Loss](https://arxiv.org/abs/2604.28190)
- [pMF](https://arxiv.org/abs/2601.22158)

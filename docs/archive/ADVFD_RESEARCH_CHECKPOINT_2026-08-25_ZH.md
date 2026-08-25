# AdvFD 研究阶段归档（2026-08-25）

## 状态

AdvFD 主线暂时冻结。本页记录当前证据边界、可恢复训练状态和代码入口，避免以后把
机制探针、理论反例或未完成长训误当作正式方法结论。

仓库只保存代码、测试、文档和精简表格/图片；ImageNet、checkpoint、feature bank、
生成图片及大体积逐样本数据均留在 `/data`，不进入 Git。

## 已经成立的结果

### 1. 官方代码缩放复现

[`ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md`](../ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md)
记录了 official-code-faithful 的 pMF-B 缩放复现。严格配对的 5K 结果表明：

- static FD 后训练解释了相对原始 pMF-B 的绝大部分 FID 改善；
- adaptive branch 在 ADM-FID 和 IS 上有小幅正信号；
- JiT-FID 略差，三个 held-out representation FD 也全部变差；
- 因而当前证据不支持“AdvFD 带来跨表示一致的分布质量提升”。

这不是论文 global batch 1024、125K steps、50K evaluation 的完整数值复现。

### 2. 时序坐标不一致

[`ADVFD_TEMPORAL_GAUGE_AUDIT_RESULTS_ZH.md`](../ADVFD_TEMPORAL_GAUGE_AUDIT_RESULTS_ZH.md)
确认：critic 持续变化时，直接 EMA 其历史 feature moments 会产生可测量的
temporal-coordinate mismatch。该差异超过同规模 fresh-bank 抽样噪声，并会旋转传给
当前生成图像的梯度；协方差项是主导项。

这是一条机制结论，不等于已经证明 mismatch 必然损害最终 FID，也没有证明某个修复
能够提高生成质量。

### 3. Witness 梯度的非识别性反例

[`ADVFD_SCORE_COUNTEREXAMPLE_AUDIT_ZH.md`](../ADVFD_SCORE_COUNTEREXAMPLE_AUDIT_ZH.md)
给出一个可复现反例：real-feature whitening 约束 feature value 的真实分布统计，但不
识别 fake support 上的输入 Jacobian。相同 AdvFD 值可以对应零、相反或任意放大的
generator correction field。Gaussian-noised score difference 在同一构造上则给出
noised reverse-KL 的严格下降方向。

严格边界是“AdvFD scalar objective 不控制 sample-space correction field”，而不是
“官方 AdvFD 一定失败”或“score 方法天然更新颖”。直接 score-difference 后训练已经
处在 Diff-Instruct、DMD、Denoising Fisher Training 等工作的覆盖范围内。

机器可读反例结果位于：

```text
docs/data/advfd_score_counterexample_v1/
```

### 4. 残差 score 后训练 pilot

[`RESIDUAL_SCORE_POSTTRAIN_PILOT_RESULTS_ZH.md`](../RESIDUAL_SCORE_POSTTRAIN_PILOT_RESULTS_ZH.md)
记录了冻结 shared-DSM 残差场的 pMF-B 短后训练。正方向的三采样 seed FID 都轻微改善，
反方向都恶化，但平均改善只有约 `0.045` FID，且 IS 不同向。它是方向性 pilot，不是
完整质量提升，也不足以形成独立方法贡献。

## 本轮保留但尚未形成结论的工具

以下代码作为研究资产保留，不能单凭其存在声称实验结论：

- critic mean/covariance 梯度分解、surrogate direction 和 generator cross-play；
- mode coverage、co-adaptation 和 fresh-critic tournament；
- quotient、Fisher/Pearson、pullback、divergence transport 与 smoothing toys；
- moment-tangent feature/parameter projection；
- mean-D/mean-G、decoupled generator 更新及其 5K 评估入口；
- official-schedule AdvFD/static FD-Loss 配对续训流水线。

对应实现集中在：

```text
experiments/advfd_cleanroom/
experiments/advfd_*.py
experiments/frechet_residual_score_toy.py
experiments/run_advfd_*.py
experiments/run_frechet_residual_score_toy.py
tests/test_advfd_*.py
tests/test_frechet_residual_score_toy.py
```

## 已停止的 125K 日程前缀训练

四卡、global batch 96 的 AdvFD pMF-B 训练使用官方 `125K` 学习率日程，按用户要求在
完成 `13,969` 次更新后安全抢占。保存状态包含 model、EMA、optimizer、static FD queues
和 adaptive critic：

```text
/data/users/zhoushunyu/eqvae/experiments/
  advfd_pmf_b_schedule125k_to20k_pipeline_v1/
  preserved_checkpoints/advfd_preempt_step_0013968.pth
```

checkpoint 内记录：

```text
step = 13968
current_step = 13969
samples_seen = 1341024
```

static FD-Loss 的公平 `20K` 对照尚未启动，配对 `10K/20K` 5K 评估也尚未执行。因此该
中断训练只有内部 surrogate 曲线，不能用于 AdvFD 与 static FD-Loss 的最终比较。

可恢复入口：

```text
experiments/advfd_cleanroom/run_pmf_b_schedule125k_to20k_pair.sh
```

## 恢复这条线时的最低要求

1. 从上述完整 checkpoint 恢复，不把内部 `fid_siglip/mae/inception` 当作图像 FID。
2. static 与 AdvFD 必须使用相同 pMF-B 起点、global batch、数据暴露和 `125K` 日程前缀。
3. 使用完全配对的噪声、类别和采样参数，同时报告 ADM/JiT FID 与 held-out FDr3。
4. 探索性机制脚本只有在对应结果和边界写入文档后，才能升级为正式结论。

# AdvFD clean-room：pMF-B Inception-2048 前缀实验

日期：2026-08-23

状态：**paper-only clean-room 的中间结果。** 本实验未查看或导入 AdvFD 官方实现；
尚未使用论文完整 SIM、global batch 1024、125K steps 或 FID-50K，因此不是论文数值
复现。

## 1. 为什么从 64 维恢复到 2048 维

先前 64 维正交投影 pilot 中，训练目标有时下降，但完整 Inception FID-5K 没有改善。
该结果证明低维代理本身会产生表示目标与正式指标的错位，不能用来判断完整 FD-Loss
或 AdvFD 是否有效。

本轮因此恢复：

- 完整 Inception-2048 static feature；
- 50,000 个 base generator 样本的 EMA warm-start；
- pMF-B/16 官方生成器 checkpoint；
- 论文的 AdamW、generator LR、warm-up 和 125K cosine horizon；
- online generator 权重与 pMF-B 论文配置（CFG 8.5、区间 `[0.1, 0.7]`）的配对
  FID-5K。

缩放差异仍包括 global batch 32、仅 Inception 而非完整 SIM，以及短训练前缀。

## 2. static FD-Loss 的首个可信正信号

同一批 5,000 个 noise/class、同一 real stats、同一 eval batch size 下：

| variant | step | FID-5K | 相对 base |
|---|---:|---:|---:|
| base | 0 | 10.989063 | 0 |
| static | 250 | 10.921381 | -0.067682 |
| static | 500 | 10.797668 | -0.191396 |
| static | 750 | 10.616869 | -0.372194 |
| static | 1000 | 10.460898 | -0.528165 |

FID 在四个连续 checkpoint 上单调改善。训练 static FD 同时从约 3.77 降到 3.42，
没有 NaN、OOM 或 clamp 异常。相较 64 维 pilot，这说明恢复完整表示后，FD-Loss 的
优化方向至少在短前缀内与完整 Inception FID 一致。

完整表见
[`docs/data/advfd_pmf_inception2048_static_prefix1000_fid5k.csv`](data/advfd_pmf_inception2048_static_prefix1000_fid5k.csv)。
原始 checkpoint、CSV、JSON 和样本图位于：

`/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_prefix_v1`

## 3. 5,000-step AdvFD 前缀结果

独立流水线从同一 base checkpoint、同一 seed 分别训练 static 与 real-whitened
adaptive 版本到 5,000 steps。AdvFD 按论文规则在 step 1,000 启动，并在后续 4,000
steps 把权重升至 0.05。

为控制单卡成本：

- static feature 保持完整 2048 维；
- adaptive critic 暂用固定 2048→512 正交投影；
- warm-start 仍来自同一批 50,000 个完整 Inception features；
- real/fake adaptive moments、训练 batch 与 held-out 评估严格使用同一投影；
- 正式配对 sweep 的 adaptive held-out 诊断使用完整 5,000 fake 和训练未见的最后
  5,000 real，避免 class-major fake 的前 512 张只覆盖少数类别；
- batch 32 峰值显存约 17.63 GiB，热身后平均约 0.5 s/step（critic 每两步更新）。

为避免 2048 维小样本协方差在 GPU FP32 下产生约 `0.01--0.60` 的数值漂移，最终
配对曲线使用相同生成样本、CPU FP64 矩和 eigendecomposition。原 GPU FP32 数值仍
保留在每个 evaluation JSON 中。

| step | static FID-5K | AdvFD FID-5K | AdvFD - static |
|---:|---:|---:|---:|
| 1,000 | 8.372117 | 8.377445 | +0.005328 |
| 2,000 | 8.071232 | 8.127349 | +0.056117 |
| 3,000 | 7.799712 | 7.794971 | -0.004741 |
| 4,000 | 7.694586 | 7.587229 | -0.107357 |
| 5,000 | 7.783402 | **7.573232** | **-0.210171** |

base FID-5K 为 `8.740201`。static 在 4,000 steps 达到最好值后于 5,000 steps
回升 `0.088817`；AdvFD 在 4,000--5,000 steps 维持改善。5,000-step AdvFD 相对
同 step static 改善 `0.210171`，约为 1,000-step 分支数值漂移 `0.005328` 的 39 倍。
因此该缩放实验已经给出一个可信的额外收益信号，但仍不能冒充论文数值复现。

完整表位于
[`docs/data/advfd_pmf_inception2048_adaptive512_prefix5000_fid5k.csv`](data/advfd_pmf_inception2048_adaptive512_prefix5000_fid5k.csv)，
原始结果位于：

`/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_adaptive512_prefix5000_v1`

### 3.1 同时发现的稳定性问题

adaptive held-out FD 在 1k--5k 分别为：

`31.69, 44.14, 52.06, 47.65, 54.80`。

它没有随 generator FID 单调变化。更重要的是，当前 critic 的 held-out 原始 feature
RMS 从 1k 的约 `0.39` 增长到：

`41.3 (2k), 175.5 (3k), 440.7 (4k), 640.7 (5k)`。

FP64 重新计算确认这些幅值增长是真实模型现象，不是诊断误差；real whitening 后真实
feature RMS 始终约为 1。训练态 FP32 EMA covariance FD 则在 step 4,771 首次塌到
0，随后 critic 未裁剪梯度一度超过 100。这说明当前小 batch、投影 critic 的训练统计
在后段已经数值失稳。5k FID 的正收益是真实配对结果，但不能据此断言该 critic 动力学
健康；必须在查看官方实现后核对 adaptive moments、精度、归一化和更新顺序。

该 512 维 adaptive 分支仍是缩放 pilot，不是论文正式 AdvFD。下一轮须恢复论文实现
细节，并使用明显长于 5k 的训练来判断收益是否持续。

## 4. selective amplification 候选的当前证据边界

解析反例已经验证：artifact mass 为 0.05、fake-only feature amplification
`M=10000` 时，real-whitened FD 约为 4,995,643，而 pooled-whitened FD 约为
2.0495，并趋于解析极限 2.0513。

这证明 real-only whitening 在 population 层面没有排除 distribution-selective
amplification；它**尚未证明**真实 AdvFD critic 会实际利用该通道，也尚未证明 pooled
calibration 能改善生成。真实 artifact-critic 生死实验只会在 paper-only AdvFD 结果冻结
后进行，避免方法改动污染复现。

## 5. 当前可说与不可说

可以说：完整 Inception static FD 在 pMF-B 上给出稳定正信号；在统一 FP64 配对评估
下，缩放 AdvFD 到 4k/5k 后额外优于同 step static。

不能说：论文完整 AdvFD 已数值复现、当前 adaptive critic 动力学稳定、512 维 adaptive
等价于论文 2048 维 critic、FID-5K 可与论文 FID-50K 绝对比较，或 pooled
calibration 已成为有效方法。

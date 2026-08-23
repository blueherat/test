# AdvFD pMF-B 纯速度续训对照

日期：2026-08-23

## 1. 实验问题

本实验只回答一个问题：此前 static FD / AdvFD 的改善，是否可能只是预训练 pMF-B
继续看了相同数量的 ImageNet batch 所致？

为此，从同一个公开 pMF-B checkpoint 出发，关闭所有 feature-distance、adaptive critic
和生成图训练，只做 10,000 step 的局部 flow-matching 速度 MSE 续训。

## 2. 训练目标

沿用公开 pMF 的数据到噪声路径：

```text
z_t = (1 - t) x_0 + t epsilon
t ~ sigmoid(N(0.8, 0.8^2))
```

目标和预测为：

```text
v_target = (z_t - x_0) / clamp(t, 0.05, 1)
v_pred   = pMF.u_fn(z_t, t, h=0, omega=1, interval=[0,1])[0]
L_v      = mean((v_pred - v_target)^2)
```

模型内部仍使用公开 pMF 的 clean-output parameterization，再按同一个 capped denominator
转换到 velocity space。训练使用公开 `cond_drop`，不包含 FD feature extractor、critic、
generated-image rollout 或额外 loss。

训练协议与 10K static/AdvFD 短预算对照匹配：同一 pMF-B 起点、ImageNet-1K packed data、
ADM center crop、无 horizontal flip、global batch 72、AdamW、`lr=1e-6`、6,250-step warmup
和压缩到 10K 的 cosine schedule。4 张 GPU 各 batch 18，共处理 720,000 张训练图；5K
和 10K 均保存 online checkpoint。

## 3. 适用边界

公开 pMF-B checkpoint 只有 `u` branch，共 238 个 tensor，没有可恢复的独立 `v` head。
因此本实验是：

> 公开 pMF-B 上的 `h=0` 局部 flow-matching 速度续训。

它不是原始双头 pMF mean-flow objective 的完整 continuation。公开网络只条件于
`h=t-r`，不直接条件于 `t`；正式 one-step 生成调用 `h=1`，而本对照监督 `h=0`。
这一区别是解释结果时必须保留的边界，不得把本实验写成“官方 pMF 全目标续训”。

## 4. 审计

- 公开 base、velocity-5K、velocity-10K 均以 238/238 keys 精确加载，无 missing 或
  unexpected key；三者均无 `v_head` key。
- 5K/10K checkpoint 分别记录 360,000/720,000 samples，step 语义与公开 evaluator 一致。
- 相对 base 的参数 L2 改变量为 `5.555e-4` 和 `7.749e-4`；238/238 tensors 均发生变化。
- 5K 生成使用完全相同的 seed、类别顺序、CFG 8.5、1 NFE、interval `[0.1,0.7]` 和
  online 权重。每个 condition 恰有 5,000 张 PNG。
- feature 评估使用 2 张 GPU，5,000 可整除，不存在 DistributedSampler padding。一次
  3-GPU、5,001-entry 的初始 feature 评估被明确排除，不进入下表。
- base/static/AdvFD 在该流程中复现此前配对结果，ADM-FID 偏差仅约 `0.01--0.03`，说明
  生成和评估入口没有漂移。
- 局部 MSE bank 从第一轮 continuation 的 sampler permutation 中显式排除前 720,000
  个已消费 index，再取 1,024 张；五个 checkpoint 共用相同图、标签、时刻和噪声。
- 速度公式、端点 clamp、`h=0` conditioning、checkpoint step 语义均有单元测试覆盖。

## 5. 结果

| condition | step | continuation-unseen velocity MSE | 相对 base | FID(ADM) | FID(JiT) | IS | FDr3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 0.068388 | - | 8.9849 | 8.8429 | 152.55 | 14.1066 |
| static FD | 10K | 0.070715 | +3.40% | 7.5227 | 7.5318 | 168.07 | 9.2661 |
| AdvFD | 10K | 0.070577 | +3.20% | **7.4266** | 7.5503 | **170.58** | 9.4724 |
| velocity MSE | 5K | 0.067560 | -1.21% | 13.3880 | 13.5642 | 109.94 | 23.8202 |
| velocity MSE | 10K | **0.067517** | **-1.27%** | 13.2255 | 13.4132 | 109.97 | 23.5884 |

纯速度续训确实改善了它被要求优化的局部目标。10K 相对 base 的未见样本速度 MSE
下降 `1.27%`，且 `t in [0.2,1]` 的四个有足够样本的时间段均下降。

但生成质量呈完全相反的方向：

- ADM-FID 从 `8.9849` 恶化到 `13.2255`，增加 `47.20%`；
- JiT-FID 同样从 `8.8429` 恶化到 `13.4132`；
- IS 从 `152.55` 降到 `109.97`；
- ConvNeXt/DINOv2/CLIP 组成的 FDr3 从 `14.1066` 恶化到 `23.5884`，增加 `67.22%`；
- 5K 已经出现几乎全部恶化，继续到 10K 没有恢复。

同噪声样图也显示纯速度模型更平滑，局部纹理和对象结构更弱；这不是仅由某一套
Inception reference 造成的数值反转。

另一方面，static FD 和 AdvFD 都显著改善图像指标，却令同一局部速度 MSE 分别上升
`3.40%` 和 `3.20%`。

## 6. 结论

当前证据明确否定：

> static FD / AdvFD 的 10K 收益只是相同步数的普通 ImageNet 续训收益。

纯局部速度续训不但不能复现收益，而且在降低自身 teacher-forced MSE 的同时显著破坏
one-step 生成。最直接的结构原因是目标错位：局部 FM 对照监督 `h=0`，而 pMF-B 的
one-step sampler 使用 `h=1` 的 average-velocity output；二者共享参数，但不是同一个
函数切片。优化局部速度风险并不保证保持已经训练好的 one-step mean-flow 映射。

这项结果也不能反过来证明 AdvFD adaptive branch 独有地解决了该错位。此前严格配对
实验已表明，static FD 解释了大部分改善，AdvFD 相对 static 的增量很小且依赖评估表示。
本实验只负责排除“普通局部速度训练也会同样变好”这一解释。

按照反证停止标准，不继续扩大该纯速度续训分支。

## 7. 产物

- 汇总：`docs/data/advfd_pmf_b_velocity_control_10k_results.csv`
- 时间分段 MSE：`docs/data/advfd_pmf_b_velocity_control_unseen_mse_bins.csv`
- 训练 checkpoint：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_velocity_control_10k_v1`
- 配对生成和精确评估：`/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_velocity_control_w3_paired5k_v1`
- 正式 feature 结果位于上述目录的 `exact_w2/`；根目录早期 3-GPU padded feature CSV
  不属于正式结果。

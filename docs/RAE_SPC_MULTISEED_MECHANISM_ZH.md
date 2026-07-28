# RAE 子空间路径课程：五种子机制结果

## 结论先行

这轮实验否定了原先的容量重分配解释，也明确否定了当前 SPC 作为生成方法：

> SPC 并没有让主体部分更早学好；它在高噪声阶段选择性降低了模型对 guided rank-16
> 子空间的依赖，而且这种方向选择性在切回标准路径 3000 步后仍有残留。但该 guided
> 子空间实际接近 top PCA，SPC 压低了最强信号方向，最终使五种子平均 FID 恶化 4.85%。

后续白化诊断已把“绝对能量”与“跨层可预测比例”分开，并发现 stage-2 的方向信任会随噪声时间变化。
完整结果见 `docs/RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md`。

## 实验协议

- 模型：RAE-DINOv2 latent 上的相同小型 stage-2 模型。
- 条件：`static` 与 `SPC(floor=0.2, power=2, rank=16)`。
- SPC 在 step 0--1999 使用子空间路径，step 2000--4999 切回标准路径。
- seed：`1201, 2309, 3413, 4517, 5623`。
- 每个 seed 的初始化、latent 顺序、time、noise 各自变化；同 seed 两条件逐值配对。
- step 2000 重置 EMA，使后续 EMA 只累计共同的标准路径阶段。
- 五个 seed 的 manifest、首 batch、RNG、scheduler、路径行数和 EMA 重置审计全部通过。

## 1. 预注册机制被否定

预注册预测是：SPC 早期暂时卸载 rank-16 任务后，step 2000 应表现为：

- basis loss 更高；
- complement loss 更低。

实际在固定 standard-static probe、`t={0.3,0.1}` 平均后：

| seed | 2k complement 相对变化 | 2k basis 相对变化 | P1 |
|---:|---:|---:|:---:|
| 1201 | +4.38% | +36.54% | 否 |
| 2309 | +4.68% | +42.98% | 否 |
| 3413 | +5.29% | +51.86% | 否 |
| 4517 | +3.85% | +37.05% | 否 |
| 5623 | +3.12% | +30.62% | 否 |

五个 seed 中没有一个满足 P1。SPC 在 2k 时不是“细节更差、主体更好”，而是两个部分都更差。
因此不能再声称 SPC 释放了容量去优先学习主体。

到 5k 时，basis 相对差距均缩小超过 97%，complement 差距缩小到约 0.18%--0.97%。
这说明切回标准路径后模型能够追赶，但追赶本身不解释生成质量。

## 2. 切换后的训练流严格一致

step 2000 之后，两条件的 target energy、semantic energy、detail energy、mean time 和学习率
逐记录完全相同。尽管如此，SPC 的标准训练 MSE 在每个 seed、每个 1000-step 窗口都更高：

- 2k--3k：高约 `0.0155--0.0233`；
- 3k--4k：高约 `0.0074--0.0133`；
- 4k--5k：高约 `0.0053--0.0089`。

所以 SPC 不是标准目标上的普通加速器。如果最终 FID 改善，只能来自损失没有直接衡量的轨迹、
鲁棒性或解空间选择，而不是更低的训练误差。

## 3. 输出正交不等于学习独立

SPC 只改变 guided basis 内的 state 和 target。按投影代数，两个路径的 complement target
完全一致，数值检查的 target shift 为零。但模型是共享参数的非线性条件预测器，最优速度场为：

```text
v*(x_t, t) = E[target | x_t, t]
```

改变 guided detail 的输入状态会改变条件信息，因此也可能改变 complement 预测。输出空间的正交
分解并不保证统计独立，更不保证网络的参数或条件速度场按同样方式分解。这正是线性频域 LMS
直觉迁移到生成模型时缺失的一步。

## 4. 交叉路径结果

对同一个 checkpoint、同一批 clean/noise/label，分别输入 static state 与 SPC state：

- 在 step 2000，SPC 模型使用自己的 SPC state 时，complement loss 仍比 static 模型使用
  static state 更高；因此 P1 失败不是单纯的评估路径 OOD。
- 在 `t=0.85`，static 模型的 complement 预测会随 guided path 改变而明显变化；SPC 模型的
  变化小得多。
- 在 step 5000，两个模型的标准路径损失基本追平，但高噪声方向敏感度仍有差异。

## 5. 等能量随机方向对照

为排除“SPC 只是整体没学好，所以对任何输入都迟钝”，构造一个与 guided basis 严格正交的
随机 rank-16 basis。对每个样本把随机方向扰动缩放到与真实 guided path state shift 完全相同
的 Frobenius norm，再比较：

```text
gain = ||prediction(x + delta) - prediction(x)||^2 / ||delta||^2
```

### 高噪声 `t=0.85`

| step | SPC/static guided gain | SPC/static control gain | 方向选择性 |
|---:|---:|---:|---:|
| 2000 | 12.08% | 92.23% | 5/5 seed |
| 5000 | 81.45% | 100.38% | 5/5 seed |

step 2000 时，static 模型在 guided 方向的平均 gain 约为随机正交方向的 26 倍。SPC 把这项
比值从约 `22--30x` 压到约 `3x`，而对随机方向的响应基本保留。到 step 5000，普通方向响应
已经完全追平，但 guided 方向仍低约 18.5%。

随后把随机正交 basis 扩展为五个 control seed。`5 training seeds × 5 control seeds` 的 25 个
配对中：

| step | guided SPC/static | control SPC/static | guided 更低 |
|---:|---:|---:|---:|
| 2000 | `0.1208 +/- 0.0156` | `0.9191 +/- 0.0445` | 25/25 |
| 5000 | `0.8145 +/- 0.0306` | `1.0002 +/- 0.0177` | 25/25 |

因此选择性降敏不依赖某一个幸运的随机 control basis。

这个结果支持“选择性去依赖”，不支持“全局欠训练导致统一变平”。它也提示当前称为 detail
的子空间可能更准确地暂称为：

> 由中间层可预测、能被生成模型高增益利用的低秩空间捷径。

这里的“捷径”仍是候选解释，而不是价值判断。guided 高增益也可能说明它是在高噪声阶段最有用的
少量真实信号。只有当降敏改善生成，或该方向误差在 rollout 中被异常放大时，才能称为有害捷径。

## 6. 当前最合理的机制

标准模型在高噪声阶段会强烈利用 guided 子空间中的微弱信号来改变大范围输出。SPC 在训练早期
把这部分信号沿时间压弱，迫使模型减少对该方向的依赖。切回标准路径后，模型重新学习完整任务，
但高噪声区的部分方向鲁棒性仍然保留。

这更像一个 **transient directional robustness curriculum**，而不是：

- 频段独立学习率；
- 主体先学、细节后学；
- 标准损失收敛加速；
- 简单的路径能量缩放。

若它改善生成，可能的原因是采样早期的 guided 坐标本身还不可靠，降低速度场对这些坐标的高增益
依赖可减少 rollout 误差放大。这个因果链目前还缺少 trajectory perturbation 和最终 FID 证据。

还有一个方向相反、目前同样合理的解释：guided rank-16 具有高能量、中间层可预测性和远高于
随机方向的模型响应，可能恰恰是高噪声阶段最有用的信号锚点。当前 SPC 优先隐藏了这些方向，
这与 Spectral Forcing “保留信号频段、隐藏噪声频段”的原则相反。P1 中 complement loss 也同步
变差，符合模型失去有用条件信息的预测。若最终 FID 不改善，应优先接受这个解释，并改为按
subspace SNR 保留高信号方向、削弱低信号方向，而不是继续调当前 floor/power。

1024 个 held-out latent 的直接统计进一步支持该解释：guided 每个 active dimension 的 residual
方差为 `4.245`，随机正交 basis 为 `0.582`，complement 为 `0.575`，即 guided 是普通方向的
约 `7.3x`。在线性 static state 中，`t=0.85` 的每维 SNR 分别为：

```text
guided=0.1322, random-control=0.0181, complement=0.0179
```

SPC 在同一时刻把 guided state SNR 乘以 `0.0475`，降至 `0.0063`。它把原本 SNR 最高的方向
压到了随机方向的约三分之一。这不是“隐藏无效细节”，而是“优先隐藏最可见信号”，能同时解释
2k complement loss 变差、方向 gain 被选择性压低以及当前多数 seed 的 FID 劣化。

### Guided basis 实际上接近 top PCA

在同一批 1024 个 held-out latent 上重新计算 residual channel covariance：

- 最优 top-16 PCA 捕获 `14.35%` residual energy；
- guided basis 捕获 `13.58%`，达到 PCA 最优值的 `94.65%`；
- guided 与 top-PCA 的平均平方 principal cosine 为 `0.8201`；
- 随机正交 control 与 top-PCA 的同一指标只有 `0.0039`。

因此 guided basis 并不是与 PCA 清楚分离的“中间层细节”。当前构造对 predicted covariance
直接做 eigendecomposition，最大化的是 **绝对可预测能量**，会天然选中高方差 final directions。
这解释了为什么它近似 top PCA，也说明此前把该子空间命名为 detail 过于武断。

若真正想找“中间层特有的可预测结构”，应最大化可预测能量占该方向总能量的比例，例如使用
regularized generalized eigenproblem 或 CCA/whitened predictability。若目标是生成课程，则还应
单独按 state SNR 决定哪些方向在各时刻应保留，而不能把 layer predictability 直接等同于学习顺序。

## 7. 可信边界

- 方向敏感度是一次有限扰动比值，不是无穷小 Jacobian 谱。
- 随机对照已经覆盖五个正交 basis，但都来自同一种随机正交补构造；尚未覆盖 PCA 低能量方向、
  DCT 方向或不同 guided basis 拟合 seed。
- probe 使用 32 个 held-out latent，训练 seed 有五个；还需扩大 probe 样本。
- guided basis 在所有训练 seed 间固定，因此只能证明训练随机性的稳定性，不能证明 basis 拟合的稳定性。
- `complement` 包含 token mean 和 rank-16 正交补，不能直接称为纯语义。
- 若 FID 不改善，这个方向性现象仍真实，但没有证明它对生成有益。

## 7.1 Decoder closure 辅助结果

step-5000 online model 各采样 64 个 latent 后，SPC 相对 paired static 的 cycle residual 在
`4/5` seed 变差，local decoder sensitivity 在 `2/5` seed 变差。cycle 差值依 seed 为：

```text
+0.0073, +0.0237, -0.0090, +0.0222, +0.0003
```

因此按事前协议，closure 的“不得 4/5 系统变差”条件已经失败。幅度本身不大，不能说 decoder
发生严重退化；但也没有证据说明训练早期的方向降敏改善了最终 latent 的 decoder closure。

## 7.2 五种子生成质量

step-5000 online model、50-step Euler、固定 sampling seed、每模型 5000 张 ImageNet-1k
类别均衡样本的结果如下：

| seed | static FID | SPC FID | FID 差 | static KID | SPC KID |
|---:|---:|---:|---:|---:|---:|
| 1201 | 136.688 | 141.844 | +5.156 | 0.12188 | 0.12565 |
| 2309 | 125.292 | 130.186 | +4.895 | 0.11712 | 0.11028 |
| 3413 | 144.334 | 142.726 | -1.607 | 0.12367 | 0.12858 |
| 4517 | 127.292 | 144.201 | +16.909 | 0.12171 | 0.13855 |
| 5623 | 139.552 | 145.683 | +6.132 | 0.12925 | 0.13494 |

- `0/5` seed 同时改善 FID 和 KID；
- 平均 FID 差为 `+6.30`，即 SPC 平均相对恶化 `4.85%`；
- paired FID bootstrap 95% CI 为 `[+1.29,+12.16]`，完整位于零以上；
- 平均 KID 差为 `+0.00488`，其 CI 跨零；
- closure 条件也失败。

所以生成闸门明确失败。按事前停止规则，不运行 EMA 扩展、不延长到 10k/20k，也不继续扫描
当前 floor/power/rank。seed-3407 crossover 的改善应视为发现集上的偶然或弱不稳定效应。

## 8. 与现有工作的关系

- [Spectral Forcing](https://arxiv.org/abs/2606.15236) 在 pixel diffusion 的 noisy input 前使用随时间变化的 DCT 低通，核心也是高噪声时
  隐藏信号不足的频段。因此“按时间隐藏频率”本身不新。
- [Variational Trajectory Optimization of Anisotropic Diffusion Schedules](https://arxiv.org/abs/2602.19512)
  用矩阵值 schedule 在训练和采样中分配不同子空间的噪声，也覆盖了广义
  anisotropic path。
- [Curriculum Sampling](https://openreview.net/forum?id=LpmuITZLAk) 用两阶段 timestep 分布说明早期和后期可以使用不同训练任务，也与当前
  `2k switch` 现象接近。
- [Atrous Learning](https://openreview.net/forum?id=EKMKPwOAuY) 通过训练期空间 masking 促进模型使用
  更广范围上下文，也与“抑制局部捷径依赖”的解释相邻。

如果这条线最终成立，区别只能建立在以下组合上：

1. 子空间来自 RAE 中间层到最终 latent 的可预测性，而不是固定 DCT/PCA；
2. anisotropic path 只用于训练早期，最终训练目标与推理采样器完全标准；
3. 方法作用可由高噪声 guided-direction sensitivity 的选择性下降解释；
4. 相同能量的随机方向、学习率课程和普通 loss reweighting 不能复现生成收益。

## 9. 下一步闸门

1. 当前 SPC 生成闸门已失败，停止 EMA 扩展、长训练和 `floor/power/rank` 扫描。
2. 使用白化广义特征问题分离 latent 方差与跨层可预测比例。
3. 当前 128-sample、五种子 probe 已发现高噪声方向响应由方差和可预测性共同解释。
4. 下一步只做 teacher-path rollout perturbation，验证局部方向 gain 是否真的传播成 endpoint/decoder 风险。
5. 只有 rollout 因果门槛通过，才允许做一个单 seed、5000-step 的反向 trust curriculum 小试。

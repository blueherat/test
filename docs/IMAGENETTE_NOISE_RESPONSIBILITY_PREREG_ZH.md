# Imagenette-64 噪声阶段责任曲线：受控实验预注册

## 1. 唯一研究问题

本实验不预设 latent 应该只在某个噪声阶段使用，也不训练时间门控。唯一问题是：

> 当一个带合法空条件分支的随机 decoder 从头训练后，它对样本级 latent 的依赖
> 是否随噪声阶段、空间频率和 bottleneck 容量呈现稳定且可区分的自然曲线？

PiD 公开 checkpoint 只用于证明该量可测。这里使用统一架构和训练目标，排除
不同预训练表征、蒸馏方式和 decoder 容量造成的混杂。

## 2. 固定设置

- 数据：`/data/shared/imagenette2-320` 官方 train/val split。
- 输入：训练时随机裁剪到 `64x64` 并水平翻转；评估使用确定性中心裁剪。
- encoder：从头训练的小卷积网络，不使用类别标签、预训练权重或重构损失。
- decoder：从头训练的条件卷积 velocity U-Net。
- 路径：`x_t = (1-t)x_0 + t epsilon`，`v_target = epsilon - x_0`；`t=1`
  为高噪声端。
- latent 始终允许输入 decoder；主实验没有 `high_noise/low_noise` 门函数。
- 每个训练 batch 独立地以 `10%` 概率把 latent 精确替换为零，因此 null 分支
  位于训练支持内。
- bottleneck 维数固定扫描 `16/64/256`。
- 正式实验至少使用 seed `0/1/2`；若某项门槛只在一个 seed 上反向或落在阈值
  附近，再补 seed `3/4`，不得只挑有利 seed。
- encoder 输出做无仿射 LayerNorm；decoder 的 condition projection 再做固定
  RMS 归一化。该设计控制 latent 尺度和条件投影尺度，但不限制其信息内容。
- 全程 fp32，关闭 TF32，不使用类别标签参与生成模型训练。
- 所有结果只写到 `~/data/eqvae/imagenette_noise_responsibility/`。

## 3. 责任指标

对同一张验证图像、同一份像素噪声、同一 `x_t`、同一时刻和同一 prediction
target，仅替换 latent：

```text
real:    z = E(x)
null:    z = 0
shuffle: z = E(x_pi), pi(i) != i
```

主指标为：

```text
Delta_null(t)    = MSE(v_null, v_target) - MSE(v_real, v_target)
Delta_shuffle(t) = MSE(v_shuffle, v_target) - MSE(v_real, v_target)
```

`Delta_shuffle` 是主要证据；`Delta_null` 只作为训练支持内的辅助对照。同样的
误差在二维 Fourier 域按归一化半径划分为低频 `[0,.25)`、中频 `[.25,.5)`、
高频 `[.5,1]`。额外报告同类别 shuffle，区分类别语义和实例级信息。

噪声区域固定为：

- 低噪声：`t < 1/3`
- 中噪声：`1/3 <= t < 2/3`
- 高噪声：`t >= 2/3`

曲线形状使用正值面积归一化后的区域占比，不用绝对幅度代替形状差异。

## 4. 预注册预测

P1：三个容量在中高噪声至少有稳定正的 `Delta_shuffle`，说明 decoder 使用的
是配对样本信息，而不只是 latent 总体统计。

P2：容量会改变归一化曲线形状。预期 `16d` 更集中于高噪声语义，`256d` 在
中低噪声和较高频率仍保留更多责任；三个 seed 的容量排序应一致。

P3：同类别 shuffle 在高容量模型中仍应产生正差值；否则观测到的主要是类别
条件，而不是更细粒度的样本信息。

P4：曲线形状差异在固定 condition projection RMS 后仍存在，且不能由 raw
latent norm、condition embedding norm 或模型参数量的变化解释。decoder 主干
在所有容量间完全相同，只允许 encoder 最后一层和 condition 输入层随维数改变。

P5：以真实验证 latent 做条件 rollout 时，责任曲线特征对条件生成 feature FID
的跨配置预测力应强于单一验证 velocity MSE 或配对像素重构误差。该比较只称为
小模型机制证据，不外推为 ImageNet FID 结论。

## 5. 实现与数据护栏

任何预测不符时，必须先完成以下审计：

1. real/null/shuffle 是否复用完全相同的 `x_t`、噪声、时刻和 target；
2. shuffle 是否无固定点，且同类别 shuffle 是否确实保持类别；
3. null 是否与训练 dropout 使用完全相同的零张量；
4. 实测 dropout 比例是否在 `[0.08, 0.12]`；
5. velocity 符号、Euler 采样方向和 `x0` 反推公式是否通过解析 toy test；
6. train/val 官方 split 是否严格分离，标签是否只进入评估；
7. 三种容量是否使用相同 decoder、batch 顺序、像素噪声、时刻和优化预算；
8. identity 重复 forward 的最大绝对 RMS 是否不超过 `1e-7`；
9. 结论是否由未收敛、NaN、单个时刻、少数异常样本或频带系数数量造成；
10. 至少三个 seed 是否方向一致，并同时查看均值、中位数、正值率和置信区间。

实现 smoke 和性能 benchmark 不进入科学结论，允许在不查看正式指标的前提下
修正 OOM、数据吞吐和明显数值错误。正式 run 启动后不得按结果调整容量、频带、
时刻、dropout、模型宽度或训练步数。

## 6. 进入 latent prior 的硬门槛

以下条件必须同时满足：

1. P1：每个 seed 至少两个容量的中高噪声 `Delta_shuffle` 均值为正，95% CI
   下界大于零，且逐样本正值率大于 `0.6`。
2. P2：最大与最小容量的高噪声责任占比相差至少 `0.10`，方向在所有正式 seed
   一致；或整体容量效应通过预注册的置换检验 `p < 0.05`。
3. P3：`256d` 的同类别 shuffle 在至少两个噪声区域具有正均值，三个 seed
   方向一致。
4. P4：condition embedding RMS 的容量间差异小于 `2%`；使用归一化曲线后
   P2 仍成立。
5. P5：leave-one-seed-out 预测中，曲线特征预测条件 rollout FID 的 RMSE 低于
   仅用验证 velocity MSE 或像素重构误差的两个基线。

任一门槛失败，先执行第 5 节审计。若实现正确且补足 seed 后仍失败，则记录为
假设反证并停止，不训练 latent prior。只有全部通过，才以相同 latent 维数、网络、
step、batch 和采样 NFE 比较三个 latent prior；prior 通过后才考虑端到端联合训练。

## 7. 解释边界

- 责任曲线是 encoder、decoder、当前像素状态和训练目标共同产生的系统属性，
  不是 latent 单独的互信息曲线。
- teacher-path 结果不自动代表自由 rollout；两者必须分别命名。
- 条件 rollout 使用真实 encoder code，不是完整无条件生成，不能称为 gFID。
- `10%` dropout 只使 null 对照合法，不保证 classifier-free guidance 最优。
- 当前 MNIST `high_noise` 时间门控实验是独立旁支，不用于替代本协议。

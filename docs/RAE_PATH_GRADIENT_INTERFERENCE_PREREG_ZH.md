# RAE path 子空间梯度干扰：事前预测

## 问题

`floor=0.2,p=2` 在 5k 时修复了 annealed 的 rank-16 endpoint 逆放大，却没有恢复
static 的生成学习。当前实验检验一个更具体的解释：

> floor 增大了 rank-16 任务在共享参数中的梯度压力，但该梯度没有帮助、甚至干扰了
> semantic-complement 的学习。

本轮不训练模型，不修改 checkpoint，只审计已有训练状态下的局部梯度几何。

## 固定设置

- 路径：`static / annealed / floor020_p2`。
- checkpoint：online model 的 step `2000 / 5000`；不用 EMA，因为问题是训练更新如何分配。
- 数据：latent cache logical indices `[100128,100160)`。5k 训练只消费 `[0,80000)`；该
  切片也不与上一轮 atlas 的 `[100000,100128)` 重叠。
- 三路共享 latent、label、noise；seed `20260724`。
- 时间：`0.97 / 0.85 / 0.70 / 0.50 / 0.30 / 0.10`。
- 参数组：最后一个 DiT block、output head，以及二者合并。
- 全程 fp32、关闭 TF32；模型 eval，不发生 optimizer update。

## 梯度定义

把输出误差正交拆成：

```text
L = L_sem + L_basis
g_sem   = grad L_sem
g_basis = grad L_basis
```

报告：

```text
cosine = <g_sem,g_basis> / (||g_sem|| ||g_basis||)
basis pressure = ||g_basis|| / ||g_sem||
semantic descent ratio
  = <g_sem,g_sem+g_basis> / ||g_sem||^2
  = 1 + <g_sem,g_basis> / ||g_sem||^2
```

`semantic descent ratio < 1` 表示加入 basis loss 会削弱 semantic 的一阶下降；小于 0
表示完整更新会直接增大 semantic loss。另将 32 样本对半拆分，用 calibration 的 basis
gradient 与 test semantic gradient 计算 cross-split cosine，排除只在同一小批次出现的
偶然冲突。

## 事前预测

1. **P1 pressure**：在 `t=0.97/0.85`，floor 的 `basis pressure` 相对 annealed 至少
   `2x`，在四个 `时间 x 主要参数组` 条件中至少三个成立。
2. **P2 interference**：step 5000 时，static 的 semantic descent ratio 至少比 floor
   高 `0.05`，在六个时间点中至少四个成立，且 last block 与 output head 同方向。
3. **P3 reversal**：`static - floor` semantic descent ratio 的时间中位差在 5k 大于 2k；
   这才与 2k 接近、5k 分叉的生成曲线一致。
4. **P4 generalization**：aggregate cosine 与 cross-split cosine 的符号一致率至少 75%，
   否则局部梯度结论不稳定。

只有 P1--P4 全部成立，才把“basis 梯度挤占/干扰 semantic 更新”称为当前生成分叉的受支持
机制。只通过 P1 只能说明风险重分配，不能说明干扰或生成因果。

## 停止规则

- 本审计之后不续训 floor candidate。
- 若 P2/P3 失败，不再用梯度冲突解释 2k--5k 反转。
- 即使全部成立，也只授权先做小图像五 seed 因果干预，不直接增加 RAE 训练。

# RAE path 低噪声梯度翻转：独立确认协议

## 探索发现与确认边界

首轮 32 张预注册审计没有通过原“高噪声 basis pressure 导致全程 semantic 干扰”假设。
查看失败结构后，出现一个更窄的 post-hoc 现象：最后一个共享 DiT block 在 `t=0.1` 上，
floor 和 annealed 的 basis gradient 对 semantic gradient 从 step 2000 的同向变成 step 5000
的反向；static 保持同向。这与 2k 生成接近、5k 分叉的时间顺序一致。

本轮只确认这个已经固定的翻转，不重新修改阈值，不把首轮 32 张混入确认数据。

## 固定设置

- 新 cache indices `[100288,100416)`，128 张；未参与训练，也不与首轮或 smoke 重叠。
- 新 noise seed `20260725`。
- checkpoint：online model step `500 / 1000 / 2000 / 5000`。
- 路径、时间、参数组、fp32 设置与首轮一致。
- batch size 8；calibration/test 各 64 张。
- 主检验只看 last block 的 `t=0.1`；`t=0.3` 为次要复现，其余时间和 output head 是
  定位对照。

## 固定预测

1. **C1 主翻转**：floor 与 annealed 在 `t=0.1` 的 aggregate 和 cross-split semantic
   descent ratio 均满足 `step2000 > 1`、`step5000 < 1`；static 两步均 `>1`。
2. **C2 批次稳健性**：step 5000 的 paired batch `static-floor` 与
   `static-annealed` descent-ratio 均值差 95% bootstrap CI 全部大于 0；step 2000 的
   `static-floor` CI 小于 0。
3. **C3 时间定位**：`t=0.3` 上 5k 的 static 仍比 floor/annealed 更少干扰，但效应小于
   `t=0.1`；在 `t>=0.5` 不要求存在明显差异。
4. **C4 参数定位**：5k、`t=0.1` 的 `static-floor` gap 在 last block 至少是 output head
   gap 的 `3x`，说明翻转主要发生在共享表征，而非输出线性层的几何必然结果。
5. **C5 split 稳定**：主条件 aggregate cosine 与 cross-split cosine 符号完全一致。

C1--C5 全部成立，才称“低噪声共享梯度随训练发生动态翻转”获得独立确认。即使通过，仍只
是局部机制证据：它没有证明该翻转单独造成 FID 分叉，也不授权直接修改 RAE 训练。

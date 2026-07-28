# RAE floor path：5k checkpoint 持久性门控

## 为什么做这一轮

2k/1k-screen 中，`floor=0.2,p=2` 的 FID/KID 最好，并接近 static；但同索引 montage
显示所有 2k 模型仍主要生成相似的纹理团块，语义尚未形成。因此先检验训练时间上的持久
性，比立刻重复多个 early-training seed 更有信息量。

## 固定设置

- 只保留事前 tiny gate 选出的 `floor=0.2,p=2`，不再换候选。
- 从其原 step-2000 checkpoint、optimizer、scheduler、EMA、RNG 和 latent-cache offset
  原位续训到 step 5000。
- 对照直接使用同 seed、同训练流的既有 static/annealed step-5000 checkpoint。
- 三路统一采样 seed `20260718`、1000 图、50 Euler steps、fp32、关闭 TF32。
- 1k FID/KID 仍只是筛选指标，不作为正式生成数字。

## 事前预测与门控

1. static 的 FID/KID 继续优于 original annealed，复现已知方向。
2. floor candidate 的 FID/KID 同时优于 original annealed；若任一指标反向，则 2k 改善
   视为早期瞬态，停止该路径。
3. floor candidate 相对 static 的 FID 与 KID 平均劣化不超过 2%；若满足，才说明它有
   资格进入多 seed，而不是只修复了一点 annealed 的病态。
4. 不要求 floor candidate 显著超过 static。单 seed、1k 样本不足以支持这种结论。

## 停止规则

- 未同时通过预测 2 和 3：停止，不做 3-seed/10k。
- 同时通过：下一轮仅做 `static/annealed/floor020_p2` 的 3-seed 复验，不再扫 schedule。

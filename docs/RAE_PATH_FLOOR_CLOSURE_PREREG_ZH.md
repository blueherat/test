# RAE floor path：generated-latent closure 外推检验

## 问题

旧 Phase-0 在 `static/random/annealed/reverse` 四条路径上发现，generated latent 的
cycle residual 与 decoder local sensitivity 和 5k FID 完全同序。`floor=0.2,p=2` 是
一个后来才训练的新路径，因此可检验旧 closure 机制能否外推，而不是继续看 teacher loss。

已知 5k/1k screen：

```text
static FID 229.67
annealed FID 267.85
floor FID 267.03
```

floor 与 annealed 的生成质量接近，且都明显差于 static。

## 固定设置

- step-5000 EMA：`static / annealed / floor020_p2`，每张 GPU 一路。
- 共享 64 个初始噪声、ImageNet label、sampling seed `20260726`。
- 50-step official shifted Euler、fp32、关闭 TF32。
- frozen official RAE-DINOv2 encoder 与 ViT-XL decoder。
- clean reference 128，clean query 64，来自已有独立 ImageNet validation cache。
- cycle：`E(clamp(D(z_gen),0,1))` 相对原 latent 的 RMS。
- local sensitivity：对 latent 加相对 RMS `1e-3` 的固定各向同性扰动后，decoder hidden
  feature deviation 除以扰动 RMS。

## 事前预测

1. static 的 cycle residual 和 local sensitivity 都低于 annealed 与 floor。
2. floor/annealed 的两个指标比值均位于 `[0.95,1.05]`。
3. 对每个指标定义
   `(floor-static)/(annealed-static)`；该相对位置位于 `[0.75,1.25]`，即 floor 跟随已知
   FID 聚类在 annealed 附近。
4. 三个 generated source 的 cycle residual 与 local sensitivity 均高于 clean query。

四项全部成立，才称旧 decoder-closure 机制成功外推到未参与旧四路径分析的新 floor
candidate。它仍不证明 decoder 是唯一原因，但会排除“floor 已把生成 latent 拉回 static
decoder 区域，只是 1k FID 没看出来”的解释。

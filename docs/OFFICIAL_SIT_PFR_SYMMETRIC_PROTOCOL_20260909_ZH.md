# 官方 SiT-XL：PFR 与普通 IG 的对称强度比较

2026-09-09，在任何本轮 PFR 质量结果产生前固定。用途是补齐 [普通 IG 对照](OFFICIAL_SIT_GUIDANCE_BASELINE_RESULTS_20260909_ZH.md) 的不对称调优，并区分小模型完整 projected PFR 与旧 XL time-only PFR。仍是探索实验，目标尚未达成。

## 固定比较

官方 SiT-XL/2、ImageNet-1K、encoder_depth=8、EMA、同一 VAE、FP32 网络、FP64 状态、TF32 关闭、B4。沿用已核对的 1000 图噪声/标签银行，seed=202609428；每类一图。所有方法使用相同四点 scale 网格：1.35、1.40、1.50、1.75。保留全部结果，不据端点最低值声称全局最优。

- 普通 IG：Euler115，四个点均复用已完成结果。
- time-only PFR：Euler100 + 每图 50 次 depth-8 前缀；复用 1.35，新增 1.40、1.50、1.75。
- 完整 projected PFR：同样的 Euler100 + 50 前缀；新增四个点。

新增共七臂，顺序为 projected1.35、time1.40、projected1.40、time1.50、projected1.50、time1.75、projected1.75。每次四张卡共同采同一臂，按全局 B4 批次 round-robin 分片，按索引恢复顺序。比较计算时同时报告调用数、block 数与实测 GPU 秒。普通 115×28=3220 block/图；PFR 100×28+50×8=3200，另有头和投影开销，不能称完全等时。

## 与小模型 PFR 一致的时间方向

官方 noise-time `t` 从 1 降到 0。定义 `F,B` 为官方 Full/Base 速度，`b` 为 scale，`G=B+b(F-B)`。PFR 只在 `t>.5` 激活，`h=min(1/32,t-.5)`，未来噪声时间 `t'=max(.5,t-1/32)`。

time-only 查询 `B'=B(z,t')`。projected 查询：

```
C = b * (F - B)
a = max(<C,G> / ||G||², 0)        # 每图；零方向取 a=0
z' = z - h * a * G
B' = B(z', t')
G_pfr = G + b * (B - B')
z_next = z + (t_next - t) * G_pfr
```

小模型用 data-time `τ=1-t`，速度 `S_d=-F, W_d=-B, G_d=-G`。投影系数在同时取负后不变，故小模型 `z+h*a*G_d` 正好对应上式的减号。使用已有 `project_to_forward_ray`，不新增裁剪或可调投影系数。未来查询用 FP32 输入；保持历史 XL time-only 的算术顺序，便于逐像素核对。

## 运行前与运行后验证

运行前核对既有源码与数据哈希、完整 checkpoint 哈希、VAE 状态哈希；CPU 合成张量验证时间反向变换下投影及状态查询等价，检查反向射线截断和零方向。四张卡各复现四张历史 time-only PFR=1.35 图像，合计 16 张必须逐像素一致；同点前缀输出必须等于 joint Base 输出。任何失败均不进入质量队列。

记录真实完整网络/前缀调用数、样本身份、像素文件、原始 Inception 特征。每个新增点都生成 1000 张真实图像，用既有 ADM 兼容评估器与同一 ImageNet 参考统计计算 FID/IS，再用 float64 样本空间特征值复算 FID。保留每图在预定时间点的投影系数、查询位移 RMS 与修正 RMS，作为描述性诊断；这些不能充当质量或机制证据。

## 解释边界

两种 PFR 形态分别与同网格普通 IG 比较，不把从两个 PFR 家族中择优隐去。相同探索银行上的最低 FID 只是选择结果；后续若要主张收益，必须先冻结配置，再以新噪声确认，并检查质量/覆盖率与更强采样基线。历史小模型正结果与旧 XL 5K 不被本轮重新命名为独立确认。

本轮不引入新方法。无论结果正负，单次 1K、均值项改善、投影代数或低 MSE 均不满足用户的核心 idea、因果机制与 ICLR 论文目标。后续方案应依据实测曲线决定，不能将普通 IG 或 PFR 未充分调优造成的差异包装为创新。

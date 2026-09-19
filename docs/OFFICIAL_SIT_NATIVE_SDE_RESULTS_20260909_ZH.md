# 官方 SiT-XL 原生 SDE 基线结果

原生 SDE250＋CFG1.35＋IG1.4 已完成 1000 张图像生成及双 VAE 解码，来源、像素和 FID 算术复核通过。EMA VAE 的 FID-1K 为 **38.05281**，MSE VAE 为 **38.21555**。这是强基线的受控实现，尚未与同预算的 SDE 候选比较，不构成新方法或研究成功。

| 同一组终点潜变量 | FID-1K ↓ | IS ↑ | 均值项 ↓ | 协方差项 ↓ |
|---|---:|---:|---:|---:|
| EMA VAE（作者配置） | 38.05281 | 60.73418 | 0.23249 | 37.82035 |
| MSE VAE（本地 Euler 对照配置） | 38.21555 | 59.64388 | 0.24913 | 37.96645 |

两行使用完全相同的 FP32 终点，只有解码器变化。EMA 相对 MSE 的 FID 差为 −0.16275；这是本批样本下的解码器干预结果，不能把两次解码计作两次独立采样。均值及协方差项使用同一特征独立重算，合计与评估器存在约 2.5e-5 数值差。

直接调用作者 `euler_maruyama_ig_sampler`，250 步、CFG 在 noise-time ≤ .7 激活、IG 全区间激活。官方 800EP SiT-XL/2 EMA、联合 depth-8 弱头；FP32 网络、原生 FP64 SDE 状态、TF32 关闭、B4。沿用已看过的平衡 1K 探索银行，每类一图；各批次 Brownian seed 由独立命名空间的哈希固定。它与发布脚本的默认 TF32、随机标签和大批量执行有区别，**不是作者 FID-50K 的复现**。

CFG 的额外分支使每图实际需要 422 次完整网络前向，即 11816 个 transformer block 计算；此前 Euler115 普通 IG 为 3220，PFR100＋50 次前缀为 3200。原生 SDE 轨迹合计 3519.86 GPU 秒，EMA / MSE 解码分别 16.35 / 15.92 秒。包括一次解码的推理约 3536 GPU 秒，是此前 PFR1.75 实测 1022.16 秒的约 3.46 倍。四卡控制器耗时 976.79 秒，另有其启动前的资产核对。

因此，即使本行 FID 低于此前 Euler PFR 的 38.77339，也不能称原生 SDE 在同成本上获胜。它确定了一个带 CFG 的官方强工作点，后续 PFR 要在相同 SDE、相同 VAE、配对随机增量下运行，并补充预算对照。

验证包含四卡共 16 个原生 FP64 终点重跑逐位相同；模型计数包装不改变输出；所有 1000 个索引恰好覆盖一次；两个 VAE 实际状态与原始文件都有哈希；合并终点与片段相同。两行 FID 用 float64 样本空间谱公式复算，最大误差 2.50331e-5。该复算验证算术，未提供新的图像样本或特征提取器证据。

审阅入口：[冻结协议](OFFICIAL_SIT_NATIVE_SDE_PROTOCOL_20260909_ZH.md)、[完整数值](data/guidance_goal_20260909/official_sit_native_sde.csv)、[校验记录](data/guidance_goal_20260909/official_sit_native_sde_audit.json)。原始图像、特征与终点在 `/home/zhoushunyu/data/eqvae/experiments/official_sit_native_sde_20260909`；运行和复算入口分别为 `experiments/run_official_sit_native_sde_20260909.py`、`experiments/analyze_official_sit_native_sde_20260909.py`。

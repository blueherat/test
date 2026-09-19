# 官方 SiT-XL 普通 IG 对照：结论范围与后续公平比较

本轮四个普通 IG 对照已完成；原始输出与独立数值复算通过校验。**它发现 XL 上旧的 IG=1.35 固定工作点调优不足，不能据此否定 PFR。** 这轮只给普通 IG 增加了强度选择，且旧 XL PFR 是 time-only，不是小模型的完整 projected PFR。用户于 2026-09-09 明确提醒了这一差异。

配置为官方 SiT-XL/2、ImageNet-1K、联合训练的 depth-8 弱头、官方 EMA、FP32 网络与 FP64 Euler 状态、固定 B4。所有行共享历史 1K 探索噪声与标签，普通 IG 用 Euler115；time-only PFR 用 Euler100 加每图 50 次 depth-8 前缀，计算量接近。

| 方法 | IG scale | FID-1K ↓ | IS ↑ |
|---|---:|---:|---:|
| 普通 IG，历史点 | 1.35 | 42.02391 | 48.11077 |
| time-only PFR，历史点 | 1.35 | 40.56543 | 52.14112 |
| 普通 IG | 1.40 | 41.32023 | 48.14857 |
| 普通 IG | 1.50 | 40.49286 | 49.24908 |
| 普通 IG | 1.75 | 39.27349 | 52.63829 |
| 普通 IG，仅 noise-time ≤ .7 | 1.40 | 42.31480 | 47.00606 |

IG=1.75 比固定 PFR=1.35 低 1.29194 FID，但这是不对称选择后的比较。尚未观察到普通 IG 曲线的内部最优点，也没有新的独立 5K 确认。`.7` IG 窗口是研究者设置的辅助控制，作者脚本中的同名窗口实际用于 CFG，不能称作官方推荐 IG 窗口。

小 SiT-S/2 历史固定工作点里，完整 projected PFR 的 FID-1K 约 61.859，普通 IG 约 64.851；历史 RNG 独立 5K 也有正结果。上述 XL 对照不改写这些结果。详细限制见 [PFR 机制审计](PFR_MECHANISM_AUDIT_20260903_ZH.md)。旧 XL 5K 的 10.15590 → 9.29188 同样保留为固定强度的迁移证据，而非调优后最优比较。

验证包括：四张卡共 16 张历史普通 IG 图像逐像素复现；1000 个索引各覆盖一次；噪声、标签、权重、VAE、源码与像素文件哈希核对；六行 FID 使用同一 Inception 特征进行 float64 样本空间谱复算，与评估器最大差异 2.80e-5。数值复算不是独立抽样复现，也不是另一特征提取器。

审阅入口：[CSV](data/guidance_goal_20260909/official_sit_baseline_controls.csv)、[校验记录](data/guidance_goal_20260909/official_sit_baseline_audit.json)、[事先固定的协议](OFFICIAL_SIT_GUIDANCE_BASELINE_CONTROL_20260909_ZH.md)。原始数据位于 `/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909`。新增四个对照的推理时间合计 4096.44 GPU 秒，不含加载、预检和评估。

下一步按[对称 PFR 协议](OFFICIAL_SIT_PFR_SYMMETRIC_PROTOCOL_20260909_ZH.md)补 time-only 与完整 projected PFR 在同一强度网格上的结果。补对照属于必要证据建设，不构成新方法或研究成功。

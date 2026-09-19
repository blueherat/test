# 官方 SiT：普通 IG 强度和区间对照

状态：运行前冻结，2026-09-09。属于新研究目标的基线验证，不是新的 guidance 方法。

已有 ImageNet-1K SiT-XL/2 联合头 PFR 在固定独立 5K 上相对普通 Euler115/IG1.35 改善约 8.51%，但原报告明确没有与调优的 IG 比较。当前先检验简单的强度或区间变化能否达到或超过该 PFR 工作点，避免将欠调优基线的差距作为核心新意证据。

固定模型、VAE、FP64 状态/FP32 前向/noTF32、Euler115、B4、seed202609428、每类一张，共 1000 张；复用原探索 bank。旧 ordinary115 与 PFR100 的完整像素和特征仍在原位置。四张卡共同处理同一对照，按全局 B4 批次号模 4 分片，合并时恢复原顺序。

| 新对照 | IG scale | 生效噪声时间 t |
|---|---:|---|
| ig140_all | 1.40 | (0,1] |
| ig150_all | 1.50 | (0,1] |
| ig175_all | 1.75 | (0,1] |
| ig140_lig | 1.40 | (0,.7] |

记主头 F、弱头 B，活动时速度为 B+scale(F−B)，非活动时为 F。这里是原脚本的 scale 约定，额外外推量为 scale−1。1.40 来自[作者 gen.sh](https://github.com/CVL-UESTC/Internal-Guidance/blob/main/SiT/gen.sh)的 IG_SCALE。逐项追踪 generate.py 后确认该 shell 的 GUIDANCE_HIGH=.7 传给 CFG，IG 使用独立的 ig_guidance_high；因此本次 IG 上限 .7 是研究者预先固定的区间控制，不能称为作者推荐的 IG 上限。shell 的运行行还有未定义 sg_val/sg_guidance_low，generate.py 调用处也存在 args.sg_scale 与 ig_scale 命名差异，不声称原脚本可直接无修改执行。本次是 Euler/no-CFG 对照，不是复现作者 SDE250+CFG 的最终质量。

全部新对照每图 115 Full，无额外前缀。旧 PFR 每图 100 Full+50 depth8 prefix，按 Transformer block 工作量近似匹配（3220 对 3200），并记录实际时间。这也不是完整的速度质量 Pareto 比较。

先由原 CUDA generator 按原 B4 调用方式重建噪声，要求全量噪声/标签哈希与旧实验一致。四卡在各自首个全局批次复现 ordinary115/scale1.35，共前16张，要求像素逐元素等于旧 quality 样本；通过后才运行四项对照。固定源码哈希、模型和 VAE 身份，保存实际输入、原图、分片索引、计数和评测缓存。评测使用同一 nanogen/ADM Inception 参考，复用旧基线指标并核验文件身份。

全部四项完成后报告整表。若普通 IG 达到或超过 PFR，不再把历史差值解释为已验证的独特方向收益；这不证明两个算子等价，也不证明所有 PFR 配置无效。若 PFR 仍占优，仍须进一步做幅度匹配干预、强基线/求解器比较与独立确认，不能将本次 1K 排名直接提升为论文结论。不按本次结果自动扩大网格。

入口：`python experiments/run_official_sit_baseline_control_20260909.py`。本次实验不训练模型。

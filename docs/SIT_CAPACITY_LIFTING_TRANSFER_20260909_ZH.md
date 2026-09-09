# RAEv2 lifting 先迁移到已有小 SiT

用户新增授权：把 RAEv2 上当前机制补到 SiT，验证有没有效果。
这里的机制指 finite capacity lifting 及低噪关闭 guidance 对照。
最新优先级：用户要求 JiT 暂放一旁，小 SiT 第一优先。
JiT 已暂停，保留 24000 步检查点；现在实现并启动本协议，
RAEv2 低噪关闭对照后置，XL 暂缓。

用户最新修正：先跑我们的小模型，XL 暂缓，不自动启动 XL。
首轮采用 ImageNet-100 SiT-S/2 velocity 800K EMA 与已有 depth4 velocity
readout（50K），与 sample_sit_ou_output_control.py 的 v 配置一致。
沿用小模型原生 data-time t=0 到 1、100 步 Euler 和 FP32 配置，
不是先前 XL 的 FP64 状态。参考 baseline manifest 冻结精度、VAE 和网格。
不得照搬 RAEv2 shift8 或方向：用噪声坐标 u=1-t 表述开关，
u<=.2/.5 分别对应小模型 t>=.8/.5。速度沿原生时间方向推进。

迁移算子为 L=(Phi_W)^(-1) Phi_S，z'=z+alpha*(L(z)-z)，
随后 Strong Euler 推进同一细网格；Full 目标与 Base 逆流用 Heun，
每 block 最多四步，不额外叠加 native IG，不引入 PFR/OU 混合。
逆流是数值近似，不声称精确离散逆。

首轮四组固定 1K：
1. alpha=.35，持续 lifting。
2. alpha=.78，持续 lifting，检查 RAEv2 原强度的直接迁移。
3. alpha=.35，噪声坐标 u<=.2 关闭 lifting，仅 Strong 推进。
4. alpha=.35，噪声坐标 u<=.5 关闭 lifting，仅 Strong 推进。

采用 .35 的理由：算子小步极限为 F+alpha*(F-B)，与现有 SiT
ordinary 的 F+.35*(F-B) 在一阶引导系数上相同。
持续组遵循 SiT 已有全程 IG 窗口，不复制 RAEv2 的 t<.1 原生关闭。
关闭组沿用按步起点判断 cutoff、截断跨界 block 的约定，记录实际节点。
SiT 与 RAE 的时间网格和表示不同，不能把同一数值 t 解释为相同感知噪声。

复用小模型 seed202609417、B8、continuous RNG、原有 100 类标签序列
的既有 ordinary/PFR 样本和评价，目录为
/home/zhoushunyu/data/eqvae/experiments/sit_pfr_output_control_20260908/v。
已有 FID1K：ordinary 71.16979062046266，PFR 68.35781145881765。
评估沿用 ImageNet-100 ADM reference，不使用 XL 的 ImageNet-1K reference。
不把这些已有 baseline 称作与 lifting 成本匹配。
冻结现有 VAE、像素转换和评价器；分片必须保持原连续 RNG 批次顺序，
验证重组后的全局噪声/标签哈希。只跑新方法，四卡共同分片一组，组间顺序。

运行前检查 alpha=0 恢复 Strong、短步一阶符号/系数、prefix 接口、
block 边界及调用计数；JiT 已释放四张 GPU，先做小批 GPU 检查。
比较 FID、IS、配对图片和实际开销。持续 .35 是相对 native IG 的主要
方法检验，关闭组回答尾部写入的必要性，.78 是直接强度迁移检查。
此次四组不能建立完整的系数稳健性结论；不能用两个不同架构的绝对 FID
比较优劣，也不能据此单独归因 latent 表示。明确改善后再安排独立验证。

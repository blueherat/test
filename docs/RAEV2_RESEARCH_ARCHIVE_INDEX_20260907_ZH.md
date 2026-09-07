# RAEv2 guidance：当前结果与新旧研究总入口

2026-09-07，第 3/8 轮。3% 目标未达到；[最后八轮台账](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md)约束后续工作，不因等待或调试重置轮次。本文连接当前实验证据与已经归档的旧研究，不把准备、分类验证或解析定理当作图像质量成功。

## 当前协议的完整质量结果

固定官方 EMA、K7 decoder、native BF16 IG1.78、100 步 shift8 Euler、B8、nanogen 官方 Inception/reference。1K 用 seed202609071、每类一张；5K 用 seed202609072、每类五张。同一规模的方法初始噪声和类别配对；两个规模不是同一批图像的嵌套子集。所有改善都须与相应规模的可靠控制比较，不能跨行更换协议。

| 方法 | 1K FID | 5K FID | 理论、限制与状态入口 |
|---|---:|---:|---|
| official | 38.486774 | 6.949768 | [固定目标与协议](RAEV2_GUIDANCE_GOAL_20260907_ZH.md) |
| 历史 interval | 38.335024 | 7.011577 | [历史调度来源](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)，仅对照 |
| ancestral | 38.894423 | 未做 | [Gaussian 通道协议](RAEV2_GUIDANCE_GOAL_20260907_ZH.md) |
| partial ancestral | 38.506604 | 未做 | 同上；η=.5 固定 |
| guided reverse 方差 | 38.541200 | 6.944997 | [方差最优性与实际边界](RAEV2_GUIDED_REVERSE_VARIANCE_20260907_ZH.md) |
| 全图速度投影 | 38.478746 | 未做 | [两种投影](RAEV2_TRANSPORT_PROJECTION_20260907_ZH.md) |
| 全图物理噪声投影 | 38.483936 | 未做 | 同上；没有实用 1K 改善 |
| stochastic weak | 38.270118 | 7.027320 | [随机弱参考](RAEV2_STOCHASTIC_WEAK_20260907_ZH.md)，5K 阴性 |
| mean weak | 38.566323 | 未做 | 同上；非线性随机平均与均值门控有别 |
| 图像 critic / 球形 | 38.535461 | 未做 | [图像判别器](RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md)，约 5.56 倍推理成本 |
| 图像 critic / 空间交换 | 38.434218 | 未做 | 同上；[高噪声线性化问题](RAEV2_CRITIC_JACOBIAN_AUDIT_20260907_ZH.md) |
| two-mode | 38.518130 | 6.933352 | [有限 Euler 矩反解](RAEV2_TWO_MODE_RATIO_20260907_ZH.md)，5K +.2362%，不足目标 |
| semantic add | 37.746546 | 未做 | [类别差补充](RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md) |
| semantic orthogonal | 37.704792 | 7.189965 | [结果](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)，1K +2.0318% 但 5K −3.4562% |
| paired ratio | 38.442135 | 未做 | [共享噪声分类](RAEV2_PAIRED_NOISE_RATIO_20260907_ZH.md)，实际 q 与再加噪 q 不同 |
| paired ratio calibrated | 38.567599 | 6.938002 | 同上；唯一原全局倍率 1.5041111779 |
| actual prefix ratio64K | 38.576570 | 6.926133 | [冻结前缀头](RAEV2_PREFIX_RATIO_20260907_ZH.md)，1K −.2333%、5K +.3401%，推理约2.35倍；未达标 |
| conditional variance | 38.547115 | 运行中 | [条件Gaussian方差](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md)，留出NLL通过，1K −.1568%，同头固定5K继续 |

精确数值、实际样本 SHA、每条源码快照、输入配对、成本和对应理论文档在[机器可读总索引](../experiments/results/raev2_guidance_20260907/research_evidence_index.json)。构建器为 [index_raev2_guidance_research_20260907.py](../experiments/index_raev2_guidance_research_20260907.py)。当前 26 条完整质量结果的样本 SHA 与冻结源码都已复核；运行中项目单列，状态文件中的 PID 本身不被当作存活证明。

独立指标复算仍由各自审计提供：[早期 1K](../experiments/results/raev2_guidance_20260907/fid_audit.json)、[旧 ratio 双 1K](../experiments/results/raev2_guidance_20260907/paired_ratio_screens.json)、[semantic 及两个 5K 控制](../experiments/results/raev2_guidance_20260907/semantic_confirm5k.json)、[新前缀 1K](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_screen1k_audit.json)、[新前缀 5K](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_confirm5k_audit.json)和[两条旧 5K](../experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json)。索引并不声称替代所有数值、全像素及成本审计。

## 没有生成质量结果的研究也完整保留

| 家族/阶段 | 已知证据 | 理论与原始数据入口 |
|---|---|---|
| 实际轨迹小 Transformer | 5K train/1K validation；原准入未过，没有 FID | [实际 q 诊断与拟合](RAEV2_ACTUAL_RATIO_20260907_ZH.md)，`actual_ratio_fit/` |
| 小数据冻结前缀头 | 单配对与全部五个真实配对分别失败；没有 FID | [前缀理论](RAEV2_PREFIX_RATIO_20260907_ZH.md)，`prefix_ratio_fit/`、`prefix_ratio_rb_fit/` |
| 64K native/real 数据 | 64K train/8K heldout；真实图不重叠，初始噪声配对；native 准备 3,590,360 逐样本主模型调用 | `actual_ratio_bank64k/`、`real_ratio_bank64k/` 的完整 execution 与 shard summaries；[native 紧凑记录](../experiments/results/raev2_guidance_20260907/actual_ratio_bank64k.json) |
| 64K 特征与固定凸拟合 | 特征通过 full-prefix 逐位对齐；唯一 2881 参数解的独立风险上界 −.780459 | [特征完成记录](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_features_complete.json)、[唯一拟合](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_fit.json) |
| 64K 概率校准检查 | 一个偶数类似然最优倍率 1.300557，在奇数类上损失 +.210390，准入失败；不采样该变体 | [诊断结果](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_probability_calibration.json)，[代码](../experiments/calibrate_raev2_prefix_ratio64k.py) |
| 两次工程失败与恢复 | 特征局部变量遮蔽；前向精度上下文未覆盖反向。都保留原失败记录，未重训或再生成已完成数据 | [第一次恢复](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_recovery.json)、[反向问题定位](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_backward_precision_diagnostic.json)、[第二次恢复](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_backward_recovery.json) |

上述目录均相对 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/`。大特征、latent、模型、图像及完整日志保留原址；表中及机器索引中的路径与 SHA 将其连接至 Git 中的轻量证据。native 与 real 编码曾共享 GPU，它们的 inclusive worker 秒不能相加后冒称独占 GPU 总算量。新前缀 1K 的推理成本为 1487.015 GPU 秒，训练准备另列；当前质量已差于更便宜原基线，不靠追加更贵且变差的控制制造成功。

## 之前各个理论与想法

[2026-09-06 全家族档案](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)是此前完整导航，保留每条理论的条件、反例、代码、解析检查、生成结果、失败和未完成状态。它覆盖：

- 真实 bridge 上的势函数/残差投影、proximal 校准、噪声边界、二阶压力与曲率、endpoint adjoint、decoder 推送与线性化。
- 实际 q 的密度反馈、Sobolev 输运、有限配对桥流、Moser source、score PDE 自洽性，以及它们缺少的边界/刷新/有限步条件。
- 仿射反射和群平均、空间能量、attention 及 strong/weak 兼容性、语义和内部 guidance 分解。
- SiT/PFR/OU 的模型内正结果、RAEv2 迁移失败、fixed-point、relative transport、半群 value、posterior consensus、径向和切向方法。
- LPL 与 decoder 后训练、AdvFD、DiT bad/good、e-process、Fisher 和早期 prior-decoder/latent-trust 背景研究；明确区分不同模型、参考与数据规模。

旧正结果不改写成当前 RAEv2 成果，旧“未做 5K”不改写成已证伪，历史 seed 重叠和数据复用勘误也随原记录保留。用户排除的完整图像生成后挑选/接受拒绝方法继续排除。

[52 篇一手文献总索引](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)连接原文阅读范围、证明核对、官方代码版本、PDF/源码资产 manifest 和实验成本限制。142 份文档是理论/实验文档数量，不是新增论文数量；本次只对这些文档计算 SHA，未把整个历史大型资产库重新散列或复制进 Git。当前重读 DG、改进 DG、Sobolev 与有限样本 FID 的用途，是判断训练对象、梯度、实际分布与最终指标之间缺什么，不能以论文篇数替代实验证据。

新的[条件方差协议](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md)由 ICML2022 covariance 原文进一步推导，只在既定试验失败后使用剩余额度；第3轮原生64K/8K特征、唯一拟合和完整8图验证完成；1K38.547115略差，原头5K运行中。[原文获取身份与阅读范围](../experiments/results/raev2_guidance_20260907/extended_analytic_dpm_reading.json)已归档。

## 复现与收束

当前已冻结运行的源文件不要在进程结束前修改。先按 execution 的真实 PID/starttime 检查存活，随后核对输入和 source snapshots；不能因为一次观察超时就重启。模型、配置、环境、解码精度与评价 reference 身份均见 [64K 环境](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_environment.json)和各 sampling request。

新 5K 完成后重新运行索引构建器并更新本文，保留此前快照的 Git 历史。最后一轮无论是否达到目标，都须提交理论、文献导航、方法与数据索引、全部正负结果及尚缺证据，明确哪些检查已实际完成。仅有该归档不代表 FID 目标达成。

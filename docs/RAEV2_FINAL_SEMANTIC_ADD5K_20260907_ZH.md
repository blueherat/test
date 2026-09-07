# 第5轮：最后一个原设置的 semantic_add 5K

此项是最后八轮内最后一个新增质量候选。只补做旧 `semantic_add` 的完整 5K，没有修改 .15 强度、时间区间、层、精度、seed 或训练模型。完成后不再开启质量新路线；若达到门槛，只做必要的独立指标与成本对照。否则整理全部研究并按八轮上限收束。

## 为什么仍值得补这一次

原 `semantic_add` 的 1K FID 为 37.74654580265707，相对 official 38.486773927092486 改善 1.9233%。它是当前尚无自己 5K 结果的较强旧候选。此前完整 5K 失败的是 `semantic_orthogonal`，不是直接补充版本；不能把一个结构变体的失败冒称为另一个版本已做过实验。

但预期应保持克制：正交版本的 1K 只比直接版本好约 .1106%，其 5K 却较官方恶化 3.4562%。已有 80 条真实模型 CPU 状态检查中，大部分时刻的类别差几乎与内部双头差正交，因此两种更新可能很接近。该检查没有覆盖全部高噪声时间点，更不能证明完整轨迹或最终 FID 相同。补做理由是补齐证据，不能承诺一定翻转。

## 文献依据与本地推论分开

本轮重读 Karras 等 [Guiding a Diffusion Model with a Bad Version of Itself，arXiv v2](https://arxiv.org/html/2406.02507v2) 的 §5.2、§6、附录 B.1–B.2。B.2 的式(6)允许同时加条件弱模型差和无条件模型差；原文的 DeepFloyd 实验是这一组合的直接来源。原文也讨论 CFG 的条件一致性与多样性权衡；它没有证明本地 RAEv2 加 .15 必然改善 FID。B.1 的大规模搜索不迁移到本实验。本次重读不增加原 52 篇文献计数。

本地取 F 为完整条件预测、B 为较浅条件预测、U 为同一完整模型的 null-label 预测。原生 IG 在活动区间保持 BF16 表达式 `B + 1.78*(F-B)`，再以 FP32 加 `.15*(F-U)`；区间外用原 F。对真实 Gaussian 通道 `Z_t=(1-t)X+tε`，若 F/U 都是对应的精确 posterior 均值，才有 `(F-U) = t²/(1-t) ∇_z log p_t(c|z)`，且此写法只用于 `0<t<1`。这解释类别差的 score 含义，不适用于把 B 冒充 U，也不保证近似预测器产生终点密度的精确幂重加权。

.15 是原协议的一次工程选择，不是理论唯一最优值。本次不再选值。活动区间沿用原 IG 的 `[.1,1]`；t=1 处代码直接使用模型差，未对含 `1-t` 分母的解释公式求值。B8 条件与 B8 null 分开调用，null label 固定 1000，使用已有 class dropout 训练能力。

## 固定执行和成功判定

- 5000 张，seed202609072，每类五张，100 步 shift8 Euler，实际 B8，四张 GPU；输入噪声和类别与原 official 5K 逐 batch 配对。
- 唯一分支 `semantic_add`；主模型逐样本调用 995000，其中 null 调用 495000，没有新的学习头、反向或数据准备。
- 启动前对原 official8、原 semantic_add8 做全像素一致性检查，两项均已通过；新程序冻结源码与原 1K request/summary/metrics 身份。
- 固定对照 official FID 6.9497684777115865，3% 阈值 6.741275423380239；历史 interval 7.011576542127955 继续保留。
- 若先过质量门槛，再按旧 1K 实测成本比 1.9913447187653919 的既定公式 `ceil((100*r+1)/2)`，审核 101 步 Heun、最后 Euler 的原 IG 成本控制，201 次调用/图。还须报告实测成本，不以 NFE 相近替代时间测量。仅在需要时基于当前源码构造补丁，禁止应用此前已过期的 Heun 补丁。

源代码：[顺序控制器](../experiments/continue_raev2_final_semantic_add5k.py)、[独立审核器](../experiments/audit_raev2_final_semantic_add5k.py)、[原公式](../experiments/raev2_semantic_complement.py)。固定机器协议：[plan.json](../experiments/results/raev2_guidance_20260907/final_semantic_add5k_plan.json)。旧结果和完整机制诊断见 [semantic 原结果](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md) 与 [CPU 原诊断](RAEV2_SEMANTIC_COMPLEMENT_CPU_20260907_ZH.md)。

大数据目录为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/final_semantic_add_confirm5k/`。第5轮执行中的状态只表示正在采样，不是质量结论；最终结果以独立全像素、输入身份、冻结源码和 FID 审核为准。

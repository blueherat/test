# Inverse-noise prior 与阶段融合：有界历史审计

2026-09-13。结论：**不把 inverse-noise prior 列为本轮快速备选。** 它有明确的数据目标，但基本构造已由 INC 覆盖；当前缺少 SiT 强基线对应的可靠真实图像逆噪声和低成本可学习结构。阶段 CFG/APG/CTRL 切换在本次检索中未发现已完成记录，但不能据此称为新机制。已确认最强相关融合仍是旧 embedding secant + APG + rescale。

本次只读旧文件、原始论文及结果 JSON，未启动 GPU、未修改公共代码。下列“未找到”限于本次检索到的仓库源码、目录、报告和相关已提交配置，不声称穷尽所有未归档尝试。

## 1. INC 已经做了什么

原文题名为 **Fine-Tuning Diffusion Models via Intermediate Distribution Shaping**，首版 2025-10-03，v3 为 2026-03-03；arXiv 注明 ICLR 2026 接收，INC 是其中 §4 的方法。它反演真实数据，另训小 flow 学习逆噪声，再接固定原模型；实验使用无条件像素 flow，校正器 16M、原模型 65M。§G 按主模型步数分别训练校正器，亦报告反演不稳定。其离散保证要求步长乘 Lipschitz 常数小于 1。[原始条目](https://arxiv.org/abs/2510.02692)；[§4、§5.3、Appendix B/G](https://arxiv.org/html/2510.02692v3)。

实现需核查一个原文索引不一致：对前向 Euler 的第 k 步 `x_next=x+h*v(x,t_k)`，逆方程应在同一 `t_k` 查询。Appendix B Algorithm 6 的 `1-eta*(j+1)` 符合这一配对；正文式 (5) 写的时间索引与之不同。不能按“负速度”字面省略状态或物理时间核对。[Algorithm 5–6](https://arxiv.org/html/2510.02692v3#A2)。

## 2. 迁到当前 SiT 的三个障碍

以下是针对仓库算子的独立推断，而非 INC 的原文结论。设一个**全部参数、时间表、求解器和历史规则均固定**的采样映射为 T_c。如果它可逆，数据目标对应

\[
Q_c=(T_c^{-1})_\#P_{\rm data}(\cdot\mid c),\qquad (T_c)_\#Q_c=P_{\rm data}(\cdot\mid c).
\]

这给出了明确目标，但没有给出一个低成本拟合器。

1. **目标必须对应实际生成映射。** 用 conditional w=1 反演，再用调优 APG 或高 CFG 生成，通常是另一个 pushforward；不能继承上面的目标恒等式。对普通 CFG 还必须匹配 cutoff、总强度与离散网格。当前 baseline 使用 Heun64；INC 的 Euler 单步逆方程不是 Heun 的精确离散逆。
2. **APG/CTRL 不是仅依赖当前 z,t 的固定场。** 当前 [baselines.py](../../../experiments/cfg_transport_search_20260913/baselines.py) 有 accepted-step 动量/modified-gap 历史。只拿图像端点、清空历史后反走，通常不会得到原采样器的逆。即使改在扩展状态上求逆，也须处理终点历史未知及初始历史固定的约束；这不是当前 stateless CFG inverse 的直接复用。
3. **低阶噪声统计不等于足够的数据目标。** 将逆噪声重新归一化成 Gaussian，可能恰好抹掉希望学习的偏移；只匹配均值/协方差也没有保证恢复图像分布。若先拟合新的条件 noise adapter，就已进入 INC 的迁移与压缩实现问题，需要新的监督集、留出检查和实际生成成本，不能包装成免训练 CFG 改进。

因此，本轮没有发现一个同时满足“未被 INC 覆盖、已有可用真实标签、成本低、能立即对照最强 APG”的 prior 子问题。**建议否决本轮建逆噪声银行、训练辅助 flow、按噪声范数直接修正的队列。** 未来若重启，应先在固定、无历史 CFG 映射上验证真实图像的离散反演，再问逆分布是否存在可交叉验证的低维结构；该准入诊断本身不算生成方法，也不因通过就自动训练。

## 3. 旧 RAE 记录是什么证据

|已做检查|实际设置与结果|能支持的结论|
|---|---|---|
|单步离散逆|RAEv2 原生 Full/IG、100步 shift8 Euler、8条既有轨迹，固定位置 0/47/89/99；32次 Picard。末步残差平方比中位数 3.05e−7，但前像误差平方比 .001139。1216次 B1 Full、28.477秒。|小残差不能单独认证逆噪声。|
|完整离散往返|8条轨迹、每步32次 Picard；噪声相对 MSE .00150–.00195。7条终点接近恢复，label0 终点相对 MSE .007859。28000次 B1 Full、639.719秒。|已知模型样本的端点重建好，不等于正确恢复噪声标签。|

来源：[单步记录](../../RAEV2_DISCRETE_INVERSE_PROTOCOL_20260908_ZH.md)、[完整往返记录](../../RAEV2_INVERSE_ROUNDTRIP_PROTOCOL_20260908_ZH.md)；脚本 `experiments/audit_raev2_discrete_inverse.py`、`experiments/audit_raev2_inverse_roundtrip.py`；完整数据 `/home/zhoushunyu/data/eqvae/experiments/raev2_inverse_roundtrip_20260908/`。两轮均未建立真实数据逆噪声银行、未训练 noise flow、未跑新 FID。

不能把这些 RAE 数值问题外推为 SiT 不可逆。也不能用本轮 SiT 的短区间 8 状态 probe 替代从真实图像端点开始的全程逆标签验证。另，[RESEARCH_STATUS.md](../../RESEARCH_STATUS.md:612)旧条目仍有“全程未检查”的当时状态；其前面的新条目和完整往返报告已经更新为完成后暂停，应按更新顺序读。

## 4. 实际跑过的融合与阶段切换

统一记 `alpha` 为 `v_c+alpha*(v_c-v_u)` 的附加强度，即普通 CFG 总 w=1+alpha。以下三组原始目录的 request/config/result/commit 存在且本次逐项核对；未重算 FID 或重新校验全部大 batch。

### 同阶段融合：已有独立 5K 确认

原始目录：`/home/zhoushunyu/data/eqvae/experiments/sit_guidance_fusion_20260910/selected_5k/`。每行是同一全新 5K bank，全部 Heun64、CFG cutoff=.75；t≥.75 继续 conditional，而非 null。

|真实 arm|冻结配置|FID5K↓|每图 Full|
|---|---|---:|---:|
|`cfg_native_02`|alpha=1.25|22.512481|224|
|`cfg_apg_04`|alpha=2，beta=−.5|21.651659|224|
|`cfg_secant_apg_rescale_11`|alpha=3，embedding condition_scale=.5，APG beta=−.5，channel_rescale=.75|**21.228058**|320|

融合相对同 bank APG 降低 .423600 FID，采样加解码秒数比约 1.377；算术 Full 比为 320/224≈1.429。它是**每个 active step 内依次做条件 embedding 割线、APG、通道方差回缩**，不是 CFG/APG 的时间交接。1K 筛选还实际完成 APG+rescale、secant+APG、secant+rescale 各12配置；只有上述三组件配置进入该轮 5K。详见 [1K结果](../../SIT_GUIDANCE_FUSION_TUNING_RESULTS_20260910_ZH.md)、[5K结果](../../SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md)、实现 `experiments/sit_guidance_fusion_20260910.py` 的 `FAMILIES/configurations/evaluate`。

另一独立 portfolio 5K bank 上，原 CFG=22.478488、APG=21.533588、embedding secant=22.180625；不能把不同 bank 的最低数字直接排成一张强弱榜。[Portfolio确认](../../SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_RESULTS_20260910_ZH.md)。

### 同一旧 1K bank 的强对照与交接

强对照原始目录：`/home/zhoushunyu/data/eqvae/experiments/sit_control_output_50ideas_20260910/control_screen_1k/`。

|真实 arm|参数|FID1K↓|
|---|---|---:|
|`cfg_native_04`|alpha=1.25|45.707341|
|`cfg_apg_07`|alpha=2，beta=−.5|**44.503871**|
|`cfg_smc_control_07`|alpha=2.75，K=.2，lambda=5|44.725613|

交接原始目录：`/home/zhoushunyu/data/eqvae/experiments/sit_fsg_ctrl_hypothesis_20260911/hypothesis_1k/`。其配置明确列出 `handoff` 和 `tail`，不是推测某段关闭条件。

|真实 arm|交接操作|FID1K↓|
|---|---|---:|
|`cfg_tuned_null_48`|alpha1.25 CFG 在 t=.75 转真正 null|45.841086|
|`cfg_tuned_conditional_32`|alpha1.25 CFG 在 t=.5 转纯 conditional|47.417862|
|`smc_high_null_48`|上述 CTRL 在 t=.75 转真正 null|45.296063|
|`smc_high_conditional_32`|上述 CTRL 在 t=.5 转纯 conditional|47.641292|

还实际完成 t=.25/.5 转 null，完整表见 [FSG/CTRL机制结果](../../SIT_FSG_CTRL_HYPOTHESIS_RESULTS_20260911_ZH.md)。上述交接都未超过同 bank 的完整 APG；CTRL 最佳 null 交接也弱于完整 CTRL。实现为 `experiments/sit_fsg_ctrl_hypothesis_20260911/pipeline.py` 的 `HANDOFF_METHODS/configurations`；前缀方法列表不包含 APG，tail 只取 null/conditional。

### 本次未找到的组合

在 `experiments` / `docs` 的 CFG/APG/CTRL/SMC、mix/switch/early/late/window/schedule/handoff 搜索，以及 portfolio、fusion、FSG-CTRL、APG extension 的相关目录中，**未找到显式 CFG→APG、APG→CFG、CTRL→APG 或 APG→CTRL 的已提交时间切换配置与结果**。固定 cutoff、CTRL/CFG→null/conditional、以及同阶段多组件 fusion 均不属于这些组合。

因此可以将真正阶段切换视为“尚未在所查记录中完成的低额外 NFE 基线调参”，不能称已被旧实验否定，也不能称为已具备真实数据目标的新机制。如果后续选择这项调参，应先固定切换时刻、两侧强度和 inactive 阶段历史是否更新；APG/CTRL 的历史不能在实现中被隐式继承或清空。其对照至少包含各自调优的完整方法、单方法分阶段强度表；质量仍须留出噪声确认。

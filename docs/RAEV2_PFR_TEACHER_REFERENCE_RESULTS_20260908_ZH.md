# 教师拟合参考的 PFR：1K微小正向，不扩样

方法和参数在 RAEV2_PFR_TEACHER_REFERENCE_20260908_ZH.md 冻结。直接复用既有10噪声/图的teacher affine权重，没有新训练；原生Full/Base/IG不改，PFR差值来自teacher参考，h1/32、rho.05、原IG活动区间。

四卡共同完成1000张，四worker与评估17242全部exit0。噪声/标签hash、模型/参考权重/源码hash、全样本覆盖与调用预算检查通过。只有候选新采样和评估，所有原生数据复用。

| 方法 | FID1K | IS |
|---|---:|---:|
| 原生IG（已有） | 38.264238947 | 59.077367 |
| 教师参考PFR | 38.199288964 | 62.034629 |

FID改善 0.1697%，绝对仅 0.064950。多次研究共用的探索1K，不是独立确认；IS上升也不能替代FID证据。无同bank同rho的原Base raw PFR比较，不能把改善归因于teacher拟合目标。

每图100 Full+99 prefix+198小读出调用。最长分片 263.172秒，batch GPU时间合计 1016.511秒，模型加载/评估及既有拟合成本另计。样本SHA256 e2a75a87407329364e625ed4116095d2d08f98cd022f1063077dff749550285d。

结论：这份此前未通过MSE准入的现成资产现已完成质量检验，但没有得到足够的生成收益。按预定规则不扩5K，不扫参考权重/强度。不能由此证明所有独立弱头训练都无效，更不能声称解决PFR在RAE的迁移。论文暂停，核心研究目标未完成。

机器结果 experiments/results/terminal_defect_20260908/ig_pfr_teacher_reference_quality.json。

# RAEv2 开放问题备忘

归档日期：2026-08-11

以下内容是当时的研究问题，不是已经验证的实验结论。

1. RAEv2 continuation flow 是否使用了与 internal guidance 对应的训练目标？如果 LPL 也使用相同目标，它的最佳 scale 不一定是 `1.78`，可能存在其他更优系数。
2. 如果 clean `x` 可以视为位于数据流形上，能否构造一个更接近流形、或更偏向流形内部的可控预测方向？

后续相关证据应优先查阅：

- `docs/FREQUENCY_PREDICTION_EXTRAPOLATION_AUDIT_ZH.md`
- `docs/DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md`

3. 如果让模型只预测v和eps，然后使用 v 和 eps 来计算 x 的预测值，是否会得到更好的 extrapolation 结果？（即只预测 v 和 eps，而不是直接预测 x）
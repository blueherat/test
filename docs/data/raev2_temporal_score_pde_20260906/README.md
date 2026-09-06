# 第3轮：时间score PDE的目标可辨识性

本包保存固定CPU解析检查的请求、结果、75点表及原论文获取manifest。它不包含RAEv2样本、模型、Inception特征或实际FID。

- [机制与结果](../../RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md)
- [冻结协议](../../RAEV2_TEMPORAL_SCORE_PDE_PROTOCOL_20260906_ZH.md)
- [检查脚本](../../../experiments/audit_raev2_temporal_score_pde.py)
- [完整摘要](summary.json)、[逐点数据](grid.csv)、[来源与复制核验](manifest.json)
- [原论文获取记录](reading_manifest.json)：PDF、HTML、文本保留外部原址，本次实际核对其字节SHA；没有将大论文原文声称为Git数据。

固定案例由解析推导选定，不是盲测；一般参数下的符号残差恒等式比有限网格检查更强。不同Gaussian目标均满足同一个score PDE及噪声端点，因此该结构独自不能识别真实目标。有数据边界的PDE正则化不被这个反例否定。

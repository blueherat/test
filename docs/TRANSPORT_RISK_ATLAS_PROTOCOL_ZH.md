# Transport Risk Atlas：回顾性校准与前瞻验证协议

## 目标

当前实验已经能解释“为什么某些谱加权改善 teacher MSE 却损害生成”，但还不能在
训练前预测伤害。Transport Risk Atlas 的目标是把机制压缩成一个低成本、无 endpoint
泄露的风险分数，再决定是否值得开发优化器约束方法。

## 两类证据必须分开

### Prospective 特征

只允许使用 baseline checkpoint 上计算的量：

- `allocation_collapse = max(1 - coarse_descent_ratio, 0)`；
- `gradient_decoupling = clip((1 - coarse_detail_cosine) / 2, 0, 1)`；
- `detail_pressure = max(log(allocation_multiplier), 0)`。

固定风险分数为：

```text
optimization_risk_score =
    allocation_collapse
    * gradient_decoupling
    * log1p(detail_pressure)
```

该公式不读取 weighted checkpoint、FID、SWD、teacher ratio 或 splice 结果。

### Post-hoc 证据

失败模型的 field splice、individual-band endpoint delta 和 shared-state drift 只能用于
解释风险为什么兑现，不能作为 prospective score 的输入。后续可以用 baseline 的受控
小扰动估计 endpoint leverage，但在该 probe 完成前不能宣称完整风险已经可预测。

## 当前阶段的性质

现有 MNIST/FashionMNIST 结果已经被观察过，因此第一轮 Atlas 只能称为
`retrospective calibration`，不能称为独立预测。它的用途是检查：

1. 固定公式能否区分 DCT/PCA 与等谱 random；
2. baseline 梯度几何能否对 endpoint damage 排序；
3. 哪个缺失变量最可能解释剩余误差。

## 进入下一阶段的门槛

- aggregate Spearman `>= 0.70`；
- endpoint harm ROC-AUC `>= 0.80`；
- 固定阈值 `0.05` 的 harm sign accuracy `>= 0.80`；
- DCT/PCA 与等谱 random 必须被稳定区分；
- 所有 prospective 特征均通过 baseline-only 泄露审计。

即使回顾性门槛全部通过，也只能进入“设计独立前瞻验证”的阶段，不能直接进入大型
RAE 训练。独立验证至少需要未参与公式设计的新 seed、时间权重形状或新小数据集。

除原始 `FID ratio > 1` 标签外，同时报告与此前预注册一致的 `FID ratio > 1.02`
practical-harm 标签。前者对有限样本 FID 的微小波动敏感，后者只作为稳健性分析；
不得在看到结果后选择更有利的标签。评估还必须拆分：

- `between_basis_spearman`：能否解释 DCT/PCA/random 的平均机制差异；
- `within_basis_seed_spearman`：能否解释同一基底内部的 seed 波动；
- `structured_vs_random_pair_accuracy`：同一数据集、同一 seed 下能否正确排序结构基底
  与等谱 random 对照。

## 执行

```bash
python -m experiments.transport_risk_atlas --save
```

结果默认保存到 `$HOME/data/eqvae/experiments/transport_risk_atlas/`，不写入仓库内的
`outputs/` 或 `artifacts/`。

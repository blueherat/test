# Train-only Rollout Checkpoint Selection 前瞻协议

## 研究问题

已有 hash-matched 路径重放表明，teacher loss 相近的模型会在晚期进入不同 endpoint
盆地。当前最后一个低成本假设是：固定的 train-only rollout distribution proxy 能否
选择更好的 checkpoint，从而阻止晚期 calibration drift。

本协议写于首次训练 seeds `5--9` 之前。旧 seeds `0--4` 只用于提出假设，不进入前瞻
统计。

## 固定设置

- 数据集：MNIST、FashionMNIST；
- 新训练 seeds：`5,6,7,8,9`；
- 模型：同一初始化和 batch/noise stream 下的 baseline、DCT、PCA、等谱 random；
- 训练：原配置 1,000 steps、fp32、AdamW；
- checkpoint candidates：`200,300,400,500,600,700,800,900,1000`；
- selection reference：256 张未参与任何 minibatch update 的训练集图像；
- proxy rollout：50 Euler steps，固定独立 noise；
- proxy：canonical-DCT 8 bands 的平均平方 log-energy gap；
- 每个模型独立选择 proxy 最小的 checkpoint，相同值选更早 checkpoint；
- test：4,096 张独立测试图像，3 个新 rollout seeds；
- checkpoint 选择不读取 test image、label、classifier feature、FID 或 SWD。

## 两个可证伪假设

### H1：rollout-aware selection 能阻止晚期漂移

对 `baseline/dct/pca`，分别比较 selected checkpoint 与自身 final checkpoint。

H1 只有在以下条件全部满足时成立：

1. MNIST 和 FashionMNIST 上三个模型的 mean feature-FID ratio 均 `<=0.98`；
2. 每个数据集/模型至少 `3/5` training seeds 的 ratio `<1`；
3. 任一数据集/模型 mean ratio 不得 `>1.02`。

任一条件失败，则 H1 在当前设置下被否定。

### H2：checkpoint selection 能把谱加权变成绝对收益

比较 `dct_selected/pca_selected` 与 `baseline_selected`。

H2 只有在以下条件全部满足时成立：

1. DCT 和 PCA 在两个数据集上的 mean feature-FID ratio 均 `<=0.98`；
2. 每个数据集/基底至少 `3/5` training seeds 的 ratio `<1`；
3. 不能依赖单个异常 seed 才得到 aggregate 改善。

任一条件失败，则 H2 被否定，不进入 RAE 或新谱优化器实验。

random 只作为等谱控制，不参与 H1/H2 的通过条件。

## 输出与完整性

结果只写入：

```text
$HOME/data/eqvae/experiments/small_image_rollout_checkpoint_selection/
```

必须保存 selection history、raw rollout metrics、paired ratios、seed-level summary、
aggregate summary、数据索引 hash 和 gate 判定。仓库内不写 `outputs/`。

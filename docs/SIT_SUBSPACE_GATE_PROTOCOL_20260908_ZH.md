# 真实 SiT 的 state-dependent 分区 gate pilot

状态：预先冻结的探索设置，尚无生成质量结果。

- 固定 ImageNet-100 SD-VAE SiT-S/2 dual-output 450K EMA，原模型参数不更新。
- 使用训练集已估计的rank32 PCA basis；补空间保留，不将其当作噪声删除。
- 从未用于PCA或此前time-only检查的训练图像中，固定抽取32768 fit、4096 holdout，
  每图像一个独立posterior样本、Gaussian噪声和uniform(.001,.999)时间；不使用validation。
- 缓存原生BF16网络输出对应的FP32状态及两头误差的正交二次充分统计量。
  训练gate时只访问状态、标签、时间和配对监督统计，不访问Bayes oracle。
- scalar与PCA双gate：相同hidden128/depth2/time32/label8/双输出网络、相同初始化。
  scalar使用两logit平均；二者统一sigmoid(log((1-t)/t)+logit)安全参数化。
- Adam2e-4、固定5000updates、batch256，相同样本索引序列；不按holdout选checkpoint。
  目标是完整velocity平方误差的依赖gate部分，去掉与参数无关常数，不改样本权重。
- holdout只检验过拟合及实现，不用MSE代替生成质量，也不凭MSE变化小结束质量验证。
- 后续采样比较native空间gate、新scalar、新PCA、pure x、pure epsilon。
  使用同noise/label/solver/precision/decoder/reference；至少将候选与最强单头比较。
- 首次生成实验属于探索；需确认solver端点处理、原生路径parity以及独立采样后，
  才能谈方法有效。若只胜较弱scalar而不胜native或单头，不认定真实模型成功。

缓存/训练种子202609311；生成bank固定为202609321，CPU生成Gaussian初态和随机排列的
平衡100类标签。固定5000样本、batch32、原生DOPRI5（atol=1e-6、rtol=1e-3）、
BF16主模型、FP32 gate/projection和同一SD-VAE decoder。先32样本检查实现和吞吐，
不据此选择gate或改采样器。自适应NFE另记，不能将“不加单次主模型前向”误作零采样开销。
数据保存在
`~/data/eqvae/experiments/sit_subspace_gate_20260908/`，原研究资产只读。

## 固定训练结果

两个gate各547106参数。32768 fit / 4096 holdout的velocity MSE如下：

| split | native | scalar | PCA |
|---|---:|---:|---:|
| fit | 0.782601833 | 0.782218510 | 0.781901863 |
| holdout | 0.779284201 | 0.779254329 | 0.779992859 |

PCA训练误差下降但holdout变差，存在过拟合迹象；不能认定真实数据有效。
仍按预先固定协议检查最终生成分布，不能以局部MSE替代FID结论。

## 一次预先固定的原生训练权重对照

首轮5K：native FID71.265543、scalar71.223144、PCA71.132263，尚未达到实质收益。
配对holdout分解显示PCA相对scalar MSE差+0.00073853（SE0.00045010），其中主空间
+0.00070325、补空间+0.00003528。0–0.1和0.9–1两个时间区间贡献几乎全部恶化，
但总体不足两个SE；因此只能定位不稳定来源，不能断言已证明统计显著过拟合。

为区分控制类无收益与训练时间权重不匹配，增加一次原生DDO权重对照：训练每个样本
损失乘以[t(1-t)]²，完全复用原SiT dual_output_flow_losses的gate时间权重。
在固定t的无限函数类下正权重不改变条件最优解，但会改变有限网络/数据的优化；
这是已知的训练控制，不是新方法。缓存、初始化、scalar/PCA容量、学习率、5000更新
和采样bank均保持不变；不搜索权重指数、端点窗口或停止点。结果单独保存在
`gates_native_weight`与`fid5k_native_weight`，首轮结果完整保留。
若训练可用，两个gate都完成同一5K评估，不按holdout择优。

## 首轮完整生成结果

所有5000-sample输入noise和label逐数组相等检查通过。FID使用原始ImageNet-100
validation图像和仓库ADM评估器；后续行复用同一reference统计。以下都是同一探索bank，
不是独立重复或统计置信结论。

| 方法 | ADM FID5K↓ | sFID↓ | 平均NFE | 生成秒数 |
|---|---:|---:|---:|---:|
| native | 71.265543 | 69.443699 | 70.6880 | 164.24 |
| scalar | 71.223144 | 69.431517 | 67.1264 | 166.66 |
| PCA | 71.132263 | 69.326348 | 73.3088 | 169.59 |
| pure epsilon | **70.512364** | 69.654932 | 306.2912 | 469.51 |
| pure x | 71.908922 | 70.262389 | 284.1440 | 427.24 |

PCA相对native FID仅降0.187%，且比pure epsilon差0.879%，未通过最强基线比较。
三个混合版本比纯头便宜，但这是原生双头也具备的优势，不能归给PCA。
全部采用相同自适应容差，NFE不同，不是固定计算量比较；秒数来自同时运行的单GPU任务，
包括解码/输出写入，不能当作严格隔离环境benchmark。完整精度CSV为`sit_fid5k.csv`。

原生权重对照训练已完成：holdout scalar MSE0.779809820、PCA0.780522798，
也未见局部误差改善；按照冻结协议，两个对应的5K生成仍执行，不以MSE选择结果。

该对照5K现已完成：scalar FID71.034772，PCA71.223001。PCA落后同设置scalar，
且两者都落后pure epsilon70.512364。时间权重对照没有挽救迁移结果。
分区gate暂不作为论文主线，不再增加权重、rank或端点窗口扫描。
根据用户最新指示，转向[PFR迁移与条件留存](PFR_CONDITION_RETENTION_PROTOCOL_20260908_ZH.md)。

## 近期相关工作边界

[Balancing Frequencies and Pixels in Flow Matching](https://arxiv.org/html/2609.02748v1)
（2026-09-02）已提出频域误差重加权与向像素损失的训练日程，并报告像素生成收益。
其公式3–4不属于这里的预测差值分区控制；但不能把频谱不均衡、早晚训练目标不同
或频域权重本身作为本项目的新颖性。该论文也不能为我们的真实SiT迁移提供证据。

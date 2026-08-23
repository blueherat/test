# 残差分数估计与 pMF 后训练实验整理

## 1. 整理范围

本记录归档以下四段已经完成的实验：

1. 在解析高斯混合 toy 上比较多种残差分数估计器；
2. 在冻结的 pMF-B 生成样本上建立真实/生成 Inception 特征库；
3. 比较估计器输出经过图像与生成器参数 VJP 后的一致性；
4. 使用冻结的 shared-DSM 残差估计器，对 pMF-B 做 100 step 短后训练并进行正式 5k 评估。

刚取消的多步 SiT `v800 -> v850`、FD-loss 和 AdvFD-loss 对照不在本记录中：该流程没有产生 checkpoint，临时日志也已清理。

## 2. 问题定义

设真实分布为 `p`，当前生成分布为 `q`。实验估计共同噪声尺度下的残差分数：

```text
r_sigma(y) = score_p_sigma(y) - score_q_sigma(y)
```

随后把它作为冻结的特征空间向量场，通过特征提取器和生成器的 VJP 传回参数。pMF 后训练使用：

```text
loss = -sign * E[stopgrad(r_sigma)^T feature(image_theta)]
```

其中 `sign=+1` 是朝估计的 `score_real-score_fake` 方向更新，`sign=-1` 是严格的反向对照。残差估计器在 pMF 更新期间保持冻结。

## 3. 解析 toy

真实与生成分布共享 12 个二维高斯分量，只改变混合权重。失配幅度为：

```text
epsilon in {0.03, 0.10, 0.30}
```

每组使用 8192 个真实样本、8192 个生成样本、3 个训练 seed、3000 个训练 step。比较：

- density-ratio classifier；
- 带输入梯度正则的 Sobolev ratio；
- 独立 DSM；
- shared-domain DSM；
- factorized-domain DSM；
- 解析 oracle。

在代表性的 `sigma=0.7` 上，估计场与解析残差分数的 global cosine 为：

| epsilon | ratio | Sobolev ratio | shared DSM | factorized DSM |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 0.279 | 0.342 | 0.082 | 0.185 |
| 0.10 | 0.633 | 0.741 | 0.155 | 0.246 |
| 0.30 | 0.820 | 0.898 | 0.584 | 0.431 |

这些结果只说明有限样本残差估计的可辨识程度：失配变小时，所有方法都更难稳定恢复解析向量场。有限步 pushforward 的 KL 变化也已保存在数据中，但它可能受场幅度和有限步 overshoot 影响，不能替代向量场真值指标。

## 4. pMF 特征与 VJP 审计

特征协议：

- 生成器：官方 pMF-B 256 checkpoint；
- 表示：torch-fidelity Inception-2048；
- 固定投影：64 维；
- whitening：仅由真实 warm-start 统计量确定；
- 训练特征：真实/生成各 8192；
- held-out 特征：真实/生成各 4096；
- 估计器：3 个 seed。

不同 seed 的 shared-DSM 在 `sigma=0.3` 上，特征场 pairwise cosine 约为 `0.623-0.668`，经过共享仿射 VJP 后约为 `0.671-0.736`。

在生成器 VJP 审计中，三 seed、三 trial 的汇总为：

| 方法 | 特征场 cosine | 图像 VJP cosine | 参数 VJP sketch cosine |
| --- | ---: | ---: | ---: |
| shared DSM | 0.653 | 0.716 | 0.810 |
| Sobolev ratio | 0.225 | 0.229 | 0.431 |
| zero-noise ratio | 0.187 | 0.183 | -0.039 |

batch 8 的 shared-DSM 复验得到：

```text
feature field cosine = 0.647
image VJP cosine = 0.713
parameter VJP sketch cosine = 0.755
```

这里比较的是估计器 seed 一致性，不是真实图像分布上的解析真值。

## 5. pMF-B 短后训练

正式协议：

```text
steps = 100
batch size = 8
gradient accumulation = 9
effective batch size = 72
learning rate = 1e-7
sigmas = {0.1, 0.3, 0.7, 1.5}
estimator ensemble = 3 shared-DSM seeds
training precision = FP32
```

投影特征 FD 的训练曲线：

| 条件 | step 0 | step 100 | 变化 |
| --- | ---: | ---: | ---: |
| 正方向 | 0.88512 | 0.87113 | -0.01399 |
| 反方向 | 0.88512 | 0.90595 | +0.02083 |

官方 5k、三采样 seed 的结果：

| 条件 | 平均 FID | 相对 baseline | 平均 IS |
| --- | ---: | ---: | ---: |
| baseline | 8.89198 | 0 | 152.2926 |
| 正方向 | 8.84715 | -0.04483 | 150.9867 |
| 反方向 | 8.93417 | +0.04219 | 153.4605 |

正方向三个 seed 的 FID 变化分别为：

```text
-0.06041, -0.05176, -0.02231
```

反方向三个 seed 的 FID 变化分别为：

```text
+0.03260, +0.03165, +0.06231
```

单 sigma、seed 1 的 FID 为：

| sigma | FID |
| ---: | ---: |
| 0.1 | 8.86965 |
| 0.3 | 8.85913 |
| 0.7 | 8.84896 |
| 1.5 | 8.87476 |

该结果是 pMF-B 一步生成器上的短后训练 pilot。FID 改善很小但三 seed 同号，反向控制也三 seed 同号恶化；同时 IS 没有与 FID 同向改善，因此不能把它表述为全面质量提升，更不能外推为多步 diffusion/flow 的结论。

## 6. 代码索引

- `experiments/residual_score_toy.py`：解析混合分布、解析 score 与各估计器。
- `experiments/run_residual_score_estimator_toy.py`：toy 训练、真值评估与有限 pushforward。
- `experiments/advfd_cleanroom/build_pmf_residual_feature_bank.py`：配对特征库。
- `experiments/advfd_cleanroom/audit_pmf_residual_estimators.py`：pMF 特征空间估计器审计。
- `experiments/advfd_cleanroom/audit_pmf_residual_generator_vjp.py`：特征、像素和参数三级 VJP 审计。
- `experiments/advfd_cleanroom/run_pmf_residual_score_posttrain.py`：冻结估计器的 pMF 短后训练。
- `experiments/advfd_cleanroom/generators.py`：pMF checkpoint 格式转换。
- `tests/test_residual_score_toy.py`：解析公式和方向性测试。
- `tests/test_pmf_residual_generator_vjp.py`：VJP sketch 与分组测试。
- `tests/test_advfd_cleanroom_generators.py`：checkpoint 转换测试。

## 7. 数据索引

轻量数据位于：

```text
docs/data/residual_score_posttrain_pilot_v1/
```

其中包含：

- 三个 epsilon 的 toy aggregate CSV 与配置；
- pMF 特征库元数据，不含二进制 feature bank；
- pMF 估计器三 seed metrics/interventions/pairwise CSV；
- 两组生成器 VJP 汇总；
- pMF 后训练配置、proxy 曲线、三 seed 5k 结果和单 sigma 结果。

未提交内容包括 ImageNet、pMF/估计器 checkpoint、feature bank、生成图片、逐样本二进制场和采样 `.npz`。

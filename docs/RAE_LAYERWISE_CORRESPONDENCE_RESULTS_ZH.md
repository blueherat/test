# RAE-DINOv2 Layerwise Correspondence 大样本结果

## 实验

- 数据：ImageNet-1k parquet `test` split。
- 样本：5,196，seed 0；索引与此前三模型 layerwise direct-error 实验相同。
- 模型：frozen RAE-DINOv2。
- 变换：`rot90`、`flip_h`。
- 设备：4 张 RTX 4090，fp32，TF32 disabled。
- 输出：`~/data/eqvae/artifacts/layerwise_correspondence/dinov2_imagenet_test_n5196_seed0`。

现有旧 artifact 没有缓存 token，只保存了 direct-error CSV，因此新增指标必须重新
编码。新实现在线累计统计量，不保存高维 latent。

## 验收结果

| 变换 | residual error 下降 | aligned 相对随机优势 | exact correspondence / 随机 | 结论 |
|---|---:|---:|---:|---|
| `rot90` | 37.62% | 40.48% | 92.18x | 通过 |
| `flip_h` | 65.61% | 69.23% | 212.63x | 通过 |

因此 `H1` 通过：DINOv2 深层 direct error 下降不能只用 token mean 或空间同质化
解释，真实位置 correspondence 的确明显增强。

## 最重要的非单调现象

### rot90

| 层 | direct error | diagonal cosine | exact match | within-1 | best displacement |
|---|---:|---:|---:|---:|---:|
| patch pre-pos | 1.3657 | 0.2105 | 1.77% | 6.75% | 6.95 |
| hidden 6 | 1.1068 | 0.3919 | 49.44% | 54.25% | 3.12 |
| hidden 9 | 1.0069 | 0.4981 | **67.97%** | 74.07% | **1.67** |
| final raw | **0.8519** | **0.6257** | 43.00% | 53.75% | 2.78 |

### flip_h

| 层 | direct error | diagonal cosine | exact match | within-1 | best displacement |
|---|---:|---:|---:|---:|---:|
| patch pre-pos | 1.2633 | 0.3236 | 16.13% | 26.34% | 5.42 |
| hidden 6 | 0.8401 | 0.6434 | 86.22% | 88.85% | 0.90 |
| hidden 9 | 0.5854 | 0.8226 | **92.22%** | 95.15% | **0.37** |
| final raw | **0.4345** | **0.8960** | 83.77% | 88.93% | 0.67 |

从 patch 到 hidden 9，direct alignment、cosine 和精确定位共同改善，属于真实几何
correspondence 的形成。hidden 9 到 final 则出现分叉：direct error 和 diagonal
cosine 继续改善，但 exact match 下降、位移上升。

这不是“越深越等变”的单调故事。更准确的机制是：

> 中层逐步形成强空间对应；末层进一步形成语义相似和低维组织，使对应位置的
> cosine 更高，却降低 token 的空间唯一性。

## 低秩压缩控制

| 层 | residual energy fraction | spatial effective rank | rank / 256 |
|---|---:|---:|---:|
| patch pre-pos | 0.599 | 226.0 | 0.883 |
| hidden 6 | 0.755 | 83.0 | 0.324 |
| hidden 9 | 0.775 | 81.9 | 0.320 |
| hidden 12 | 0.472 | 37.8 | 0.148 |
| final raw | 0.468 | 38.7 | 0.151 |

final spatial residual 并没有消失，能量仍占 46.8%；但空间 effective rank 相对 patch
下降约 83%。这解释了为什么 final diagonal cosine 可以继续提高，而 exact spatial
match 从 hidden 9 回落。随机置换 error 在各层仍约为 1.41，而 final aligned error
明显更低，因此结果不是完全同质化造成的假等变。

## 位置编码干预

在固定 256 张图像上：

| 变换 | normal position | zero position | rotated position for `gx` |
|---|---:|---:|---:|
| `rot90` | **0.847** | 0.995 | 1.376 |
| `flip_h` | **0.433** | 0.635 | 0.932 |

直接去掉或旋转 absolute position embedding 都让 final alignment 更差。位置编码确实
破坏架构层面的朴素等变，但在预训练完成的 DINOv2 中，它已和 attention/content
共同适配，不能在推理时把它当作可独立删除的 nuisance。

## 研究决策

- 保留“DINOv2 存在深层弱几何响应”的结论。
- 停止使用“等变性随深度单调增强”的说法。
- final latent 同时包含语义低秩化和残余空间 correspondence；最强精确几何层约在
  hidden 9，而不是 final。
- 不据此强制 final latent 实现全局 `D4`。后续 adapter 若方法化，应优先考虑
  spatial residual 或中层几何监督。
- `H1` 已通过，可以进入无训练 transport 审计；该结果不直接证明等变性有利于生成。

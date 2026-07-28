# RAE 生成路径差分方向：机制筛查结果

## 一句话结论

逐样本路径差分能预测“同一个样本会怎样变化”，但不能预测“质量会怎样变化”。
在当前四组 stage-2 endpoint 上，它不是可用的 latent 质量方向；直线插值反而系统性地
穿过更差的区域。

## 实验口径

- endpoint：`static / random / annealed / reverse`，每种 256 个样本；四种路径使用相同
  初始噪声、ImageNet label、采样 seed 和 sample index。
- calibration：indices `[0,128)`，仅用于拟合 global mean direction。
- held-out test：indices `[128,256)`，所有主结果均在这 128 对样本上计算。
- clean reference：独立 ImageNet validation clean-latent cache `[1024,1152)`。
- 模型全冻结：官方 RAE-DINOv2 ViT-XL decoder 与 torchvision Inception-v3，fp32，
  TF32 关闭。
- 主实验在查看 held-out 结果前固定；RMS 保持和球面插值是看到中点坏化后增加的
  post-hoc 鉴别实验，不能混作预注册证据。

原始结果位于：

```text
~/data/eqvae/experiments/rae_path_difference/paired_geometry_cal128_test128_seed20260719/
```

## 主结果

### 1. 质量 proxy 通过了最重要的有效性检查

四个原始 endpoint 的 projected Frechet 为：

| endpoint | 已知 5k FID | projected Frechet | SWD |
|---|---:|---:|---:|
| static | 123.53 | 3.800 | 0.436 |
| random | 128.99 | 3.808 | 0.439 |
| annealed | 143.54 | 4.104 | 0.471 |
| reverse | 159.05 | 4.651 | 0.502 |

两种 proxy 与已知 FID 的 Spearman 都是 `1.0`。这不能把 proxy 变成正式 FID，
但说明它们在当前四个 endpoint 上没有把质量顺序看反。

### 2. 差分包含稳定的样本特异变化信息

在 good endpoint 走 25% 时，own direction 的 median feature progress 为
`0.195--0.248`，而 shuffled/random 通常为 `0.119--0.153`。逐样本单侧 Wilcoxon
检验在四条路径、两种主要对照上均显著，own 胜率约为 `70.3%--96.9%`。

因此 `z_bad(i)-z_good(i)` 不是纯噪声。它确实编码了第 `i` 个样本在两个生成器之间
如何改变。

### 3. 但它不是质量方向

预注册 gate 的结果为：

| 检查 | 结果 |
|---|---:|
| endpoint proxy 有效 | 通过 |
| 3/4 路径有剂量响应 | 0/4，失败 |
| 3/4 路径能从 bad endpoint 修正 | 0/4，失败 |
| 3/4 路径有足够样本特异性 margin | 1/4，失败 |

最反常也最关键的结果是：四条路径的线性中间点普遍比两个 endpoint 的线性质量预期
更差；从 bad endpoint 沿成对方向回退 25% 后，四条路径也全部继续变差。换句话说，
“连接好坏 endpoint 的箭头”并不等于“局部质量梯度”。

## 为什么中间点会坏

### 1. 有一部分来自 latent 半径收缩

两个高维 endpoint 做普通线性插值时，中间向量会相互抵消。例如：

| 路径 | endpoint cosine 中位数 | endpoint delta RMS 中位数 | 中间 RMS 的现象 |
|---|---:|---:|---|
| static -> reverse | 0.759 | 0.653 | `1.000 -> 0.847 -> 0.808` |
| static -> random | 0.922 | 0.394 | 仅轻微收缩 |
| random -> annealed | 0.885 | 0.471 | `0.997 -> 0.955 -> 0.975` |
| annealed -> reverse | 0.736 | 0.665 | `0.975 -> 0.829 -> 0.808` |

RMS-preserving lerp 在 12 个中间条件中改善 11 个，球面插值改善 10 个，证明半径
收缩确实是一个真实因素。

### 2. 但半径收缩只解释了小部分

相对于 endpoint 指标的线性插值，普通 latent 直线的额外坏化中位数为 `0.269`。
RMS-preserving lerp 对这部分的中位恢复比例约 `21.9%`，球面插值约 `20.3%`；而且
两者都没有让任意一条路径的 bad-side correction 成功。

decoder 特征也没有沿 endpoint feature 直线前进：中间点相对端点 feature chord 的
垂直偏离中位数约为 `0.50--0.65` 个 chord 长度。这说明主因不是一个标量 norm，
而是 decoder 映射和生成 latent 支撑集具有明显弯曲。

## 当前最合理的机制解释

当前证据支持下面这个较窄、但可靠的结论：

1. 不同 stage-2 训练路径会给同一噪声和类别产生有配对意义的样本变化。
2. endpoint 的总体质量差异是分布级、动力学级性质，不能降成逐样本欧氏差分向量。
3. 两个合法 endpoint 之间的欧氏线段不一定仍处于 decoder 熟悉的 latent 区域。
4. 简单 norm 修正只能缓解，不能修复这条直线离开生成支撑集的问题。

这不是在证明 decoder 单独导致一切，也没有证明真实生成流形的唯一形状；它排除的是
更简单的说法：“好坏生成器 endpoint 的成对差分就是可迁移的质量方向”。

## 研究判断

这条 endpoint-difference 路线不应直接进入 adapter 训练，也不应在当前 seed 上继续
调 alpha 或换 proxy。预注册 gate 已明确失败，继续调参会把机制验证变成结果追逐。

仍有价值的发现是“真实采样轨迹与 endpoint 欧氏弦之间存在断层”。它比抽象的 latent
平滑度更贴近生成模型：生成器可能沿一条 decoder-compatible 的弯曲路径运动，而直线
捷径会离开该路径。

## 下一步最小实验

优先做一个不训练的 sampler trajectory 实验：

1. 对同一 stage-2 模型、相同噪声和 label，保存 50-step sampler 的若干中间 latent。
2. 比较真实轨迹点与“起点到终点直线插值点”，并令两者具有相同进度或相同 RMS。
3. 对每一步报告 clean-reference proxy、decoder feature progress、latent RMS 和轨迹曲率。
4. 加一个 time-shuffled trajectory 作为负对照。

验收标准：真实 sampler 轨迹在至少 3 个 seed 上显著优于匹配的直线捷径，且优势不能
被单纯 RMS matching 消除。若成立，下一步才值得研究 decoder-aware path regularizer；
若不成立，就停止“几何路径”叙事，回到 stage-2 训练目标和分布建模本身。

## 证据边界

- 当前只有一个 sampling seed、每个 endpoint 128 个 held-out 样本。
- projected Frechet/SWD 是 torchvision feature 的小样本筛查量，不是正式 ADM-FID。
- endpoint 已用于此前路径实验，所以本轮不是完全独立发现集。
- post-hoc 几何对照只用于解释主结果，不能用于改变主 gate。
- 因主 gate 失败，当前结果不授权 fresh-seed endpoint-direction 扩展；新的 sampler
  trajectory 假设必须重新固定协议。

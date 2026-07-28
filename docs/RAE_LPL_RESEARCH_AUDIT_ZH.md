# RAE 上 LPL 后续研究审计

## 当前结论

当前最可靠的生成结论是：

> 在冻结 RAE encoder/decoder、只继续训练 latent prior 时，完整 decoder
> feature LPL 相对等步数 Flow，在 DINOv2-B、MAE-B、SigLIP2-B 上均稳定
> 改善生成指标。

这并不等于已经证明：

> LPL 把 latent 预测误差移动到了 decoder 的局部低敏感方向。

后一个解释已被有限半径实验直接反证。当前更符合数据的解释是：完整 LPL
在 stage-2 实际误差尺度上改善 decoder feature 的内容和尺度匹配，而这种
修正依赖样本、token、时间和当前状态，不能由 clean latent 处的一个固定
局部二次型概括。

## 已完成的研究

| 研究 | 做了什么 | 结果 | 当前作用 |
|---|---|---|---|
| DINOv2-B 严格 LPL | 四 seed、2,000 updates、预固定 50k 终验 | 5k 三指标 4/4 同向；50k ADM FID `13.5043 → 11.1027` | 确认完整 LPL 在确定性 RAE decoder 上有效 |
| MAE-B/SigLIP2-B 跨 tokenizer | 各三 seed、500 updates、各自 50k 终验 | 6/6 seed 三指标同向；50k ADM FID 分别改善 `7.72%/5.53%` | 排除 DINOv2 和小 prior 特例 |
| 官方起点 5k | 三个官方权重按严格 4×4 协议直接采样，不做训练 | 对官方 FID，Flow/detach/full 分别有 `0/9`、`4/9`、`9/9` 个 seed 改善；full 的三 tokenizer 平均 KID 也改善 | 排除“只比同样退化的 Flow 好”，并显示 denominator gradient 对跨过官方起点很关键 |
| Flow 计算预算控制 | 同一训练轨迹在 step 500 精确复现，再让 Flow 训练到 600 updates | 两个模型都未追上 LPL-500 | 排除多 updates 或约 8%--15% 单步额外计算的简单解释 |
| 局部误差方向实验 | 四 seed、32 张 held-out 图，比较同范数 Flow/LPL 误差方向 | LPL 局部敏感性更高，而不更低 | 反证“搬到局部低敏感方向” |
| 有限半径幅度扫描 | 沿同一误差方向从 4% 扫到 100% 真实半径 | 大多数轨迹从局部 LPL 更差反转为真实半径 LPL 更好 | 发现 local-to-finite-radius 非线性断层 |
| feature 归一化分解 | 分离 raw、target/symmetric/prediction normalization、方差和 cosine | LPL 更好保持 feature 方差，同时略改善内容方向 | 给出当前最强的描述性机制 |
| 常数 decoder metric proxy | decoder embed、randomized Gauss-Newton、DCT-band GN | test Spearman 最高仅 `0.2055`，gradient cosine 最高仅 `0.0285` | 否定全局常数 channel/frequency metric |
| 静态 predictability/oracle proxy | 用可预测性及 decoder atlas sensitivity 近似完整 LPL | 即使 oracle 静态 metric 的 gradient cosine 也只有 `0.029` | 说明失败不是 metric 估计不准，而是修正本身高度输入相关 |
| decoder prefix proxy | 用 decoder 前 1/2/3 段近似完整 LPL | prefix 3 的 Spearman `0.728`、cosine `0.493`，但已不够便宜且仍未达门槛 | 暂未得到可替代完整 decoder 的廉价方法 |
| DINOv2-L pilot | 官方大 encoder/decoder；共同 prior 250 步后 Flow/LPL 各分叉 100 步 | 三 seed 的有限半径 strict feature 对齐同向改善，但未做 FID | 支持几何现象可扩展，不构成生成结论 |

尚未完成的项目不能当作实验结果：

- `target-norm`、`symmetric-norm` feature loss 目前只有因果消融设计，尚未
  进行正式多 seed 生成训练。
- Decoder Metric Preconditioning 没有进入训练阶段，因为它依赖的局部
  低敏感机制先被实验反证。
- 没有联合训练 encoder/decoder，也没有证明完整训练 80/800 epoch 后仍有
  相同比例收益。

## 为什么局部敏感性更高却仍能改善

设冻结 decoder feature map 为 `phi`，clean latent 为 `z`，prior 的预测
误差为 `e`。

局部敏感性测量的是：

```text
||J_phi(z)e|| / ||e||
```

真正训练和生成在意的是有限距离后的误差：

```text
||phi(z + e) - phi(z)||
```

二者只有在 `e` 足够小时才近似相同。严格地说：

```text
phi(z + e) - phi(z) = integral_0^1 J_phi(z + alpha e)e d alpha
```

局部实验只测积分起点 `alpha=0`，而实际结果由整条路径上的 Jacobian 决定。
当前 stage-2 误差 RMS 是 clean latent RMS 的约 `18%--41%`，中位数约
`28%`，并不是无穷小扰动。

实验数值正好显示了这个断层：

| 指标 | LPL / Flow | 解释 |
|---|---:|---|
| latent MSE | 1.0092 | LPL 的平均 latent 误差略大 |
| 1% clean RMS 的 raw 局部放大 | 1.1217 | LPL 局部方向更敏感 |
| 对称有限差分 raw JVP | 1.1364 | 不是单边差分误差 |
| 真实半径 raw hidden error | 0.9869 | 到真实尺度后已略优 |
| 真实半径 strict feature error | 0.8768 | LPL 在实际目标上低 `12.3%` |

512 条真实半径配对观测全部是 LPL 更好；但其中约 `84%` 在最小的 4%
半径处仍是 LPL 更差。优劣通常到共同真实半径的约 `64%` 才发生反转。

这不是矛盾，而是说明 decoder 深层具有显著曲率。LPL 优化的是
`phi(z_hat)` 与 `phi(z)` 的有限距离，不是让 `J_phi(z)` 的方向增益尽量小。

## 为什么“高敏感方向”不必一律压低

敏感度只表示单位 latent 改动会带来多大 feature 响应，没有说明响应是否
朝正确目标移动。高敏感方向可能承载物体结构、位置或语义；把误差机械旋转
到 decoder 的近零空间，虽然局部输出变化小，也可能让 prior 丢失必须恢复
的信息。

实验也支持这一点：同范数随机或打乱方向的局部放大只有真实 LPL 方向的约
`0.59/0.52`，看起来“更安全”，但它们不是 stage-2 实际会产生、也不是
能够恢复目标 feature 的方向。因此，低敏感不是生成质量的充分目标。

## 当前最合理的机制

在真实同范数误差半径上，decoder feature 方差相对 clean feature 为：

```text
Flow = 0.8486
LPL  = 0.9322
```

Flow 更容易使 decoder 内部表征收缩。LPL 使 feature 方差更接近 clean，
同时 centered channel cosine 从约 `0.795` 提高到 `0.804`。使用固定
clean 方差归一化时 LPL 仍然更好，因此它不是只利用 prediction
normalization 的分母取巧。

所以目前最稳妥的机制表述是：

> 普通 Flow 对所有 latent 坐标使用均匀 MSE，而完整 LPL 直接观察冻结
> decoder 在当前预测状态、真实误差半径上的 feature 偏差。它没有降低
> clean latent 处的局部敏感性，而是减少了整段非线性传播后的 feature
> 收缩和内容错位。

这解释了生成改善，也解释了为什么常数 Jacobian、静态 channel 权重和
固定频带权重都不能替代完整 LPL。它仍是强一致的机制解释，不是已经由
单一因果消融唯一确认的最终机制。

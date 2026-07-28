# RAE layerwise path 逆条件数：修正后的事前预测

## 推导

训练路径写成：

```text
z_t = (1-t) s(t) + t epsilon
v_t = epsilon - s(t) + (1-t) s'(t)
```

因此普通线性公式实际得到：

```text
z_t - t v_t = s(t) - t(1-t)s'(t)
```

对任一系数 `c(t)`，endpoint component 的观测系数是：

```text
k(t) = c(t) - t(1-t)c'(t)
```

当前 `power=2` 的 fading component 有：

```text
k_fade(t) = (1+2t)(1-t)^2
```

它在高噪声端接近 0。反演 endpoint 时，该子空间中的模型误差会被约 `1/|k|`
放大。

## 为什么四条路径不同

- `static`：semantic/detail 的 `k=1`，不存在路径引入的逆放大。
- `annealed`：只有 rank-16 middle-guided detail 使用 `k_fade`。
- `reverse`：semantic、token mean 和 rank-16 detail 之外的大部分通道使用
  `k_fade`，受影响维度最大。
- `random`：detail 使用 `scale=2.568`，basis 上的有效系数为
  `1-scale+scale*k_fade`；它会在 `t≈0.44` 附近过零，形成局部奇异点。

## 修正实验

- 继续使用同一四模型、seed、noise、label 和 held-out indices `[128,160)`。
- 先在 synthetic teacher path 上验证修正反演能以 `<=2e-5` 误差恢复 endpoint。
- 模型 rollout 记录 steps `8,16,24,32,40,44,46,48` 和最终 endpoint。
- 对 `z_t-t v_t` 按实际 basis、path mode、power、detail scale 做解析反演。
- 分别报告 basis component 与其余 semantic component 对最终生成 endpoint 的误差。
- 只在数值相对稳定的后半程解码 corrected endpoint estimate；所有指标仍是探索 proxy。

## 事前预测

1. `reverse` 的 semantic relative error 在 steps `32/40` 明显高于 `annealed`，并在
   最后几步快速下降。
2. `annealed` 的主要放大集中在 rank-16 detail，semantic error 不出现 reverse 式
   长时间滞后。
3. `random` 的 basis error 在 step `44` 附近出现局部峰值，与有效系数过零对应。
4. corrected estimate 的 decoder proxy 在后半程总体排序为
   `static/annealed` 优于 `reverse`；若不成立，逆条件数只能解释 latent 数值，不能
   解释生成差异。
5. 组件 oracle 干预中，`reverse` 替换 semantic 后的 decoded endpoint-feature
   距离小于替换 basis；`annealed` 则相反，替换 rank-16 basis 更有效。该预测固定于
   首轮 32 样本条件数实验之后、oracle 解码之前，属于第二阶段探索预测。

## 停止条件

若 component error 不随解析系数的病态程度变化，或 decoder proxy 与 latent error
完全脱钩，就停止“路径逆条件数解释生成差异”的路线，不训练新模型补故事。

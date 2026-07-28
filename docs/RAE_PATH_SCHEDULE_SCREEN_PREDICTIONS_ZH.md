# RAE decoder 加权路径调度：离线筛选事前预测

## 目的

本轮不训练新模型。我们只复用已有 annealed rollout 的逐样本误差和 component
oracle 解码特征，筛掉解析上明显不值得训练的 coefficient schedule。

对 detail coefficient `c(t)`，endpoint observation factor 为：

```text
k(t) = c(t) - t(1-t)c'(t)
```

候选路径必须同时保留“高噪声时延迟 detail”的作用，并避免 `k(t)` 接近或穿过 0。

## 候选族

### 带下限的幂函数

```text
c(t) = floor + (1-floor)(1-t)^p
k(t) = floor + (1-floor)(1+p*t)(1-t)^p
```

### 带下限的有理函数

```text
c(t) = floor + (1-floor)(1-t)/(1+alpha*t)
```

两族均满足 `c(0)=1`、`c(1)=floor`，且 `k(t)>=floor>0`。

## 风险估计

已有 annealed 模型是在原始 `c(t)=(1-t)^2` 下训练的。设其修正后的 basis
relative error 为 `e_old`，则先反推 observation-space raw error：

```text
e_raw = e_old * k_old
```

再做候选反事实：

```text
e_candidate = e_raw / k_candidate
```

decoder component sensitivity 由 oracle 估计：

```text
w_sem   ~= feature_error(basis_oracle)^2 / semantic_error^2
w_basis ~= feature_error(semantic_oracle)^2 / basis_error^2
```

报告两种风险：

- 总风险：`w_sem*e_sem^2 + w_basis*e_basis^2`；
- 路径额外风险：相对 `k=1` 的 basis amplification 增量。

30% 门槛只用于路径额外风险。因为候选不改变 semantic error，用总风险要求下降 30%
会把一个不可由路径系数控制的量写进验收条件。

## 固定预测

1. 任意 `floor>=0.05` 的候选都不会过零，理论最大逆放大不超过 `20x`。
2. 至少两个、且来自至少两个不同 power/alpha 设置的候选，会同时满足：
   - detail delay retention `>=70%`；
   - 最小 observation factor `>=0.05`；
   - 相对原始 annealed 的路径额外 decoder-weighted risk 下降 `>=30%`。
3. 总 decoder-weighted risk 的改善会明显小于路径额外风险的改善，因为已有 oracle
   表明 semantic 方向通常更敏感，而且本轮不延迟或修改 semantic。
4. 最有希望的 Pareto 区域是 `floor=0.10--0.20`、`power=2--3`；过大的 floor
   会牺牲延迟效果，过小的 floor 仍会留下较强逆放大。
5. 原 random control 的 `rank=16, scale=2.568` 会继续被判定为不干净。无缩放时：
   - `rank=16` 是干净的同-rank control，但不匹配能量；
   - 约 `rank=106` 是近似同能量 control，但不匹配 rank；
   - 因而后续必须同时报告两者，不能再用一个放大后的 rank-16 control 代表二者。

## 验收与停止条件

候选通过离线 gate 需要：

- 连续 `t in [0,1]` 上无过零，`min k>=0.05`；
- 均匀时间积分定义的 detail delay retention `>=0.70`；
- 路径额外 decoder-weighted risk ratio `<=0.70`；
- 至少两个不同形状参数的候选通过，避免结论依赖单点调参。

这只是筛选，不是生成质量证明。即使通过，也只能进入 tiny、短程、多 seed 训练；若
没有稳定通过者，就停止继续训练这种 time-dependent data path。

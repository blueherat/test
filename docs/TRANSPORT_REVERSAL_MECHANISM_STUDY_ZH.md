# 方向加权为何改善 Teacher MSE 却损害生成：机制研究结论

## 最终判断

低成本机制实验已经把核心原因从宽泛的“exposure bias”收紧为：

> **生成损伤不是由频域权重单独决定，而来自四项乘积：高噪声风险预算偏移、coarse/detail 共享梯度弱耦合或冲突、被低权重方向具有高 endpoint leverage、以及缺少受保护通路。DCT/PCA 同时满足四项；等谱 random 只有预算偏移，但组间梯度几乎同向、group 0 leverage 低，因此不系统恶化。**

四个条件缺一不可：

1. **风险方向与数据几何对齐。** 同一个权重特征值谱换成随机正交基底后，系统性损伤基本消失。
2. **共享梯度不能替低权重方向代偿。** DCT/PCA 的 coarse/detail 梯度弱耦合或冲突；random 的组梯度余弦约 `0.9-0.95`，detail 更新同时也训练 coarse。
3. **低权重方向具有高 endpoint leverage。** DCT/PCA group-0 high-window splice 解释约 `90%-101%` 的损伤；random group 0 近似中性。
4. **高杠杆方向没有受保护通路。** exact linear skip 阻止 analytic toy 反转；移除 skip 后反转在 5/5 seeds 出现。小图像定向 guardrail 与 split path 也能移除大部分相对损伤。

off-path state shift 可以进一步放大 RAE 的中段误差，但新实验表明它不是简单图像反转的必要根因：从真实 teacher marginal restart，DCT/PCA 的 middle-window 损伤仍然存在，有时反而更大。

## 为什么权重会系统性牺牲粗结构方向

当前方向权重使用：

```text
R_b(t) = m_b / ((1-t)^2 m_b + t^2)
w_b(t) proportional to R_b(t)^(-gamma), gamma=0.5.
```

其中 `m_b` 是 clean data 在方向 `b` 上的二阶矩。

- 当 `t -> 0`，`R_b(t) ~= 1/(1-t)^2`，方向差异很小，权重趋近均匀。
- 当 `t -> 1`，`R_b(t) ~= m_b/t^2`，所以 `w_b(t) ~ m_b^(-gamma)`。
- 高能量方向因此在高噪声阶段获得最低权重，低能量方向获得最高权重。

FashionMNIST 中，DCT/PCA group 0 仅占 `1/8` 系数，但每系数实际 clean second moment 约为 `7.0`；其余组通常低于 `0.4`。随机正交基底中，各权重组的实际能量都约为 `1.0`。这直接预测：

1. DCT/PCA group 0 的 teacher error 应在中高噪声恶化。
2. groups 1-7 的 error 应大多改善。
3. 高噪声 group 0 field splice 应造成主要 endpoint 损伤。
4. 随机基底不应出现稳定 group-0 损伤。

四条均被实验验证。

## 跨基底、跨数据的配对结果

所有结果使用 width-24 tiny U-Net、train `8192`、test/sample `1024`、`1000` updates、`5` paired seeds。DCT、PCA、random 在每个时间拥有相同的权重特征值谱；同一 seed 的三种基底 baseline checkpoint 哈希完全相同。

### FashionMNIST

| basis | low/mid teacher ratio | high teacher ratio | endpoint feature FID ratio | FID 变差 seeds |
|---|---:|---:|---:|---:|
| DCT | 0.950 | 1.046 | 1.461 | 5/5 |
| PCA | 0.971 | 1.057 | 1.787 | 5/5 |
| random orthogonal | 0.997 | 0.999 | 1.003 | 3/5，方向不稳定 |

### MNIST

| basis | low/mid teacher ratio | high teacher ratio | endpoint feature FID ratio | FID 变差 seeds |
|---|---:|---:|---:|---:|
| DCT | 0.967 | 1.019 | 1.184 | 4/5 |
| PCA | 0.974 | 1.028 | 1.422 | 4/5 |
| random orthogonal | 1.009 | 1.003 | 1.014 | 3/5，幅度很小 |

MNIST seed 4 是重要反例：DCT/PCA endpoint 都明显改善。因此当前机制提高的是反转发生概率，不是“只要对齐就必然变差”的决定论。

## 时间因果定位

FashionMNIST 从同一 baseline 噪声开始，只在指定窗口换入 weighted field：

| basis | high only | middle only | low only | high+middle | full weighted |
|---|---:|---:|---:|---:|---:|
| DCT | 1.312 | 1.097 | 1.010 | 1.443 | 1.459 |
| PCA | 1.463 | 1.241 | 1.035 | 1.738 | 1.785 |
| random | 0.967 | 1.019 | 1.000 | 0.998 | 1.004 |

数值为 feature FID 相对 baseline 的 ratio，低于 1 更好。

这验证了权重公式给出的时间预测：high-window 是首要损伤来源，middle-window 继续积累，low-window 很弱。high+middle 已接近 full weighted，因此损伤不是采样最后几步产生的。

## 子空间因果定位与反事实修复

只换入 weighted-baseline field difference 的指定子空间：

### High-window feature FID

| dataset | basis | full | group 0 only | groups 1-7 only | group 0 解释比例 |
|---|---|---:|---:|---:|---:|
| FashionMNIST | DCT | 1.312 | 1.288 | 1.021 | 91.7% |
| FashionMNIST | PCA | 1.463 | 1.417 | 1.042 | 89.6% |
| MNIST | DCT | 1.139 | 1.114 | 1.011 | 92.0% |
| MNIST | PCA | 1.300 | 1.285 | 1.006 | 100.7% |

随机基底的 group 0-only 在 FashionMNIST 和 MNIST 分别为 `0.992` 和 `1.001`，没有稳定损伤。

这既是原因定位，也是一个 post-hoc rescue：在 high-window 保留 baseline 的 group 0，仅使用 weighted groups 1-7，FID 基本回到 baseline。因而 group 0 不是和坏结果偶然相关，而是对 high-window endpoint degradation 具有主要因果贡献。

middle-window 更复杂：

- Fashion PCA 的 group 0 与 groups 1-7 分别解释约 `44%` 与 `45%`。
- Fashion DCT 两部分均有贡献和非线性交互。
- 不能把“group 0 解释约 90%”外推到 middle-window，更不能外推到 RAE 的所有频带。

## Teacher Restart 否定了简单 exposure-bias 解释

只在 `0.3<=t<0.7` 换入 weighted field，比较两种 `t=0.7` 起点：

| basis | baseline rollout start | true teacher-marginal restart |
|---|---:|---:|
| DCT | 1.097 | 1.159 |
| PCA | 1.241 | 1.758 |
| random | 1.019 | 1.059 |

若中段损伤主要来自 off-path generalization，teacher restart 应明显减轻损伤；实验结果相反。因此对 FashionMNIST：

> weighted middle field 在真实 `p_0.7` 上已经不能正确完成多步边缘输运，state shift 不是损伤成立的必要条件。

这不否定已有 RAE shared-state sign reversal。更准确的边界是：

- **图像 toy：** on-path risk/leverage mismatch 已足以造成损伤。
- **RAE：** high-time on-path mismatch 之后，还观察到 mid-time off-path amplification。

## Dense-Time Audit 排除了部分测量漏洞

DCT 在 `0.3<=t<0.7` 的 20 个 dense time points 上：

- 5/5 seeds 的 ratio-of-mean teacher MSE 仍小于 1。
- 每个 seed 有 `65%-85%` 的中段时间点改善。
- 但 teacher-restart FID 在 4/5 seeds 变差。

这是真正的 pointwise teacher-risk / multi-step transport 分离。

PCA 的结果不同：4/5 seeds 的 dense middle MSE 轻微大于 1，说明原先只测 `0.3/0.5/0.7` 确实漏掉了小幅中间峰值。因此 PCA 不能作为“aggregate teacher MSE 改善”的最干净证据。不过约 `1%` 的 dense MSE 退化对应 `32%-120%` 的 teacher-restart FID 退化，仍说明不同方向和时间的 endpoint leverage 极不均匀。

## Exact Linear Skip 的预测性消融

8 维 analytic Gaussian-mixture flow 提供整个状态空间上的 exact conditional velocity。

### 有 exact best-linear skip

网络只拟合 nonlinear residual：

| hidden | teacher exact MSE ratio | endpoint coordinate W1 ratio |
|---:|---:|---:|
| 24 | 0.862 | 0.827 |
| 96 | 0.899 | 0.843 |

加权改善 teacher 和 endpoint，未出现稳定反转。

### 移除 skip，直接预测 raw velocity

| hidden | teacher exact MSE ratio | 高方差方向 MSE ratio | endpoint coordinate W1 ratio | W1 变差 seeds |
|---:|---:|---:|---:|---:|
| 24 | 1.101 | 1.397 | 1.055 | 5/5 |
| 96 | 0.997 | 1.158 | 1.066 | 5/5 |

hidden-96 尤其关键：aggregate exact teacher MSE 均值几乎不变，3/5 seeds 还改善，但高方差方向 5/5 恶化，endpoint W1 5/5 恶化。P12 的预测因此命中：exact linear skip 把高能量线性输运从方向竞争中隔离出来，是原 analytic toy 不反转的关键保护条件。

## 训练侧预测性干预

sampling-time splice 只能说明哪个已训练 field difference 有害。P13-P17 进一步
在看到结果前指定怎样修改训练；这组实验既命中了关键方向，也暴露了尚未解决的
Pareto 约束。

### 恢复 coarse 权重能够救 endpoint

`coarse_protected` 把 group 0 权重固定为 `1`，并重新归一化 details；
`additive_guardrail` 则保留原 detail weights，只额外把 group 0 补到 `1`。
后者另有一个每个样本总 loss mass 完全相同的 `time_scale_control`。

| dataset | basis | raw weighted | normalized protect | additive guardrail | matched scale control |
|---|---|---:|---:|---:|---:|
| Fashion | DCT | 1.461 | 0.988 | 1.053 | 1.405 |
| Fashion | PCA | 1.787 | 0.949 | 0.923 | 1.709 |
| Fashion | random | 1.003 | 1.027 | 0.979 | 0.976 |
| MNIST | DCT | 1.184 | 1.341 | 0.907 | 1.210 |
| MNIST | PCA | 1.422 | 0.971 | 1.043 | 1.382 |
| MNIST | random | 1.014 | 1.021 | 1.018 | 1.112 |

数值是 feature FID 相对各自 raw baseline。Fashion DCT/PCA 中 additive
guardrail 在 10/10 paired seeds 同时优于 raw weighted 和 matched-scale
control。MNIST DCT 也在 5/5 seeds 优于 raw；PCA 均值改善但有一个反例。
所以 endpoint rescue 来自定向补回 coarse 风险，不是整体增大 high-time loss。

但 P15 的 Pareto 预测失败：Fashion DCT/PCA 的 high-time nonzero MSE gain
只保留原 weighted 的约 `9%/15%`，MNIST 约 `2%/40%`。在共享网络中把梯度
还给 coarse，会拿走大部分 detail gain。

### 参数隔离保护 coarse，但固定容量仍有代价

两个 width-17 通路合计 `231,678` 参数，与 raw width-24 的 `229,897` 只差
`0.8%`。显式 branchwise loss 下，split baseline 与 split additive 的 coarse
state hash 在 15/15 runs 逐位相同；DCT/PCA endpoint ratio 为 `1.031/1.056`。
然而 DCT detail gain 完全消失，PCA 只保留约 `24%`。P16 因此只证明了保护
coarse 足以移除相对损伤，没有实现 Pareto。

P17 给 detail 恢复完整 width-24，再增加 width-12 coarse branch；参数匹配控制
是单个 width-27 shared network：

| basis | asym weighted / asym baseline | wide weighted / wide baseline | asym weighted / raw baseline |
|---|---:|---:|---:|
| DCT | 1.012 | 1.465 | 1.243 |
| PCA | 1.030 | 1.558 | 1.228 |

普通扩宽仍牺牲 group 0，说明不是简单参数量阈值。asymmetric 架构相对自身
baseline 稳定，但 width-12 coarse branch 的绝对 MSE 比 raw baseline 高约
`9%-10%`，使绝对 FID 仍高 `23%-24%`；它也未通过预注册方法门槛。

绝对 detail MSE 同时揭示了重要的天花板效应：不加权的 asym baseline 已比 raw
baseline 改善 DCT/PCA detail 约 `3%/10%`，等于或超过 raw weighted。也就是
detail 专门化本身复现了“细节收益”，代价是小 coarse branch underfit。这支持
共享表示资源重新分配机制，但不是一个可直接部署的修复。

## 梯度机制：Conflict 与 Neglect 分型

P18/P19 不再训练模型，而在 held-out teacher states 的 `t=0.7/0.9` 上对现有
checkpoint 求 group-0 与 groups-1-7 梯度。以下是 raw baseline checkpoint 的
shared trunk 五 seed 均值：

### FashionMNIST

| basis | coarse/detail cosine | allocation multiplier | weighted coarse descent / MSE descent | high group-0 splice FID ratio |
|---|---:|---:|---:|---:|
| DCT | +0.165 | 4.12x | 0.213 | 1.288 |
| PCA | -0.318 | 3.71x | 0.161 | 1.417 |
| random | +0.947 | 5.65x | 0.992 | 0.992 |

### MNIST

| basis | coarse/detail cosine | allocation multiplier | weighted coarse descent / MSE descent |
|---|---:|---:|---:|
| DCT | +0.332 | 3.38x | 0.238 |
| PCA | +0.234 | 4.39x | 0.224 |
| random | +0.909 | 5.61x | 0.968 |

这组结果解释了等谱 random 反例：random 的 loss budget 同样大幅转向 groups 1-7，
但这些组的梯度与 group 0 几乎同向，所以 detail 更新仍能替 coarse 提供约完整的
一阶下降。DCT/PCA 中两组梯度弱耦合；Fashion PCA 还出现真实负冲突，因此
weighted coarse 有效下降预算只剩 ordinary MSE 的 `16%-24%`。

P18 的“DCT 也应负冲突”预测失败，不能统一使用 gradient conflict。最终分型为：

- **PCA/Fashion：** conflict + neglect。
- **DCT 及 MNIST 主体：** weak coupling 导致的 neglect，不要求负余弦。
- **random：** strong positive coupling，其他组梯度可以代偿显式低权重。

P18 还预测 local detail descent 必须增大，该项没有稳定命中；final-checkpoint
一阶 detail 指标不能解释完整训练轨迹。因此可靠结论限于 coarse budget collapse，
不能声称每一步都主动加速 detail。

## 最终机制方程

当前证据支持把风险写成定性乘积：

```text
endpoint degradation risk
  ~= output-risk allocation gap
     x gradient decoupling/conflict
     x endpoint leverage of the neglected subspace
     x absence of a protected transport path.
```

四项中的任一项缺失都可阻断反转：random 缺少后两项；exact skip 缺少第四项；
matched-scale control 保留第一至第四项而继续恶化；定向 guardrail 消除第一项中
对 coarse 的缺口并救回 endpoint。这个机制能预测干预方向，但当前实验没有找到
同时保留 raw baseline 绝对 FID 和全部 detail gain 的低成本 Pareto 方法。

## 冻结 RAE 的跨模型桥接

P20/P21 不训练 RAE，也不加载 encoder/decoder。它们复用三组 step-10000 tiny
RAE baseline/partial EMA、缓存的 ImageNet validation latents 和官方 50-step
Euler，只审计 stage-2 最后 transformer block/output linear 及 latent endpoint。

### RAE 中同样存在 basis-specific gradient neglect

high-time `t=0.85/0.95` 三 seed 均值：

| checkpoint | basis | parameter group | band0/nonzero cosine | allocation multiplier | coarse descent ratio |
|---|---|---|---:|---:|---:|
| baseline | DCT | last block | 0.098 | 4.67x | 0.244 |
| baseline | random | last block | 0.836 | 4.87x | 0.952 |
| baseline | DCT | output head | 0.034 | 4.67x | 0.234 |
| baseline | random | output head | 0.670 | 4.88x | 0.900 |
| partial | DCT | last block | 0.073 | 4.68x | 0.240 |
| partial | random | last block | 0.877 | 4.87x | 0.962 |

DCT 的 coarse-descent ratio 在 baseline/partial、两个高时间、last block/output
head 的 `24/24` 个 seed-level 条件全部 `<0.5`。random 使用完全相同的八个
weight eigenvalues，却由其余 bands 的强同向梯度把 coarse descent 保持在
`0.90-0.96`。每个 seed 的 random-DCT cosine gap 至少 `0.63`，descent gap
至少 `0.66`。因此小图像发现的 basis-specific neglect 不是卷积 U-Net 特例，
它在冻结 RAE DiT 中同样存在。

### Individual-band leverage 按梯度预算排序

P21 对 `t>=0.85` 分别只换入一个 partial DCT band。主指标是 32 个严格配对
validation latents 的 summary SWD 相对 baseline：

| band | high-time gradient descent ratio | endpoint SWD ratio | mean delta |
|---:|---:|---:|---:|
| 0 | 0.244 | 1.01908 | 0.02542 |
| 1 | 0.511 | 1.00719 | 0.00959 |
| 2 | 0.760 | 1.00645 | 0.00860 |
| 3 | 0.901 | 1.00593 | 0.00790 |
| 4 | 1.022 | 1.00451 | 0.00601 |
| 5 | denominator near zero | 1.00481 | 0.00640 |
| 6 | denominator near zero | 1.00317 | 0.00422 |
| 7 | denominator near zero | 1.00225 | 0.00300 |

band 0 在 `3/3 seeds` 都是最有害 individual band。稳定 bands 0-4 的
`1-descent_ratio` 与 endpoint delta 在每个 seed 的 Spearman 都为 `1.0`。
八个 individual deltas 的和与 high-all delta 的相对误差仅 `-1.0%` 到
`+2.4%`；band0/nonzero 份额为 `35.7%/64.3%`，复现旧 64-latent aggregate
probe 的约 `34%/66%`。

所以 RAE 的 nonzero 损伤不是某个高频 band 推翻了 coarse neglect：band 0 是
单位 band 最大杠杆，七个更小的 nonzero 损伤近似线性累积。P20 的优化预算排序
在 P21 的冻结 endpoint 干预中准确预测了 bands 0-4 的因果排序。

### RAE 的两阶段机制

RAE 证据现可分为：

1. **High window：** DCT risk allocation 与 shared gradient decoupling 使低频
   bands 的有效下降预算塌缩；逐 band endpoint leverage 按该塌缩排序并累积。
2. **Middle window：** 已有 shared-state audit 显示 partial 在 teacher states
   可更好、在同一 baseline-rollout states 上反而更差，属于 off-path response
   amplification，不能由静态 high-time 梯度排序替代。

这完成的是 tiny RAE latent proxy 的机制桥接，不是 50k ImageNet FID 证明，也
不是成功的新 loss。固定 inverse-variance weighting 的失败原因得到解释；怎样在
不牺牲 detail/absolute baseline 的情况下修复，仍是独立方法问题。

## 预测审计：哪些命中，哪些失败

| 预测 | 结果 | 对机制的影响 |
|---|---|---|
| P1：有 skip 的 analytic toy 稳定反转 | 失败 | 反转不是一般 FM 必然现象 |
| P2：off-path dose response 足以预测 endpoint | 部分；斜率常为正，但 endpoint 可改善 | off-path amplification 非充分条件 |
| P4：off-path/transport 指标比 aggregate teacher MSE 更相关 | 部分命中 | 相关性有提升，但单步指标仍不够敏感 |
| P5：DCT/PCA 比等谱 random 更容易反转 | Fashion 强命中 | 证明风险方向与数据几何对齐很关键 |
| P6：容量只解释 teacher reallocation | 命中 | width 不能单调消除 endpoint penalty |
| P7：high/middle 有害、low 弱 | 命中 | 时间权重机制成立 |
| P8：teacher restart 应减轻 middle 损伤 | 失败且反向 | off-path 不是图像 toy 的必要根因 |
| P9：dense-time DCT MSE 仍改善 | 命中；PCA 支持 sparse-time 替代解释 | DCT 是最干净的 risk/transport 断层证据 |
| P10：high group 0 解释多数损伤 | DCT/PCA、两数据集均命中 | 高能量低权重子空间得到因果确认 |
| P11：MNIST 复现 basis 排序 | 达到预注册 4/5 门槛，但有 seed-4 反例 | 机制可跨数据复现但非决定论 |
| P12：移除 exact skip 后 analytic toy 反转 | 5/5 命中 | 参数化保护条件得到预测性验证 |
| P13：归一化 coarse protection 实现 Pareto rescue | endpoint 命中；detail retention 失败，MNIST 有大方差反例 | coarse 是损伤开关，但简单重归一化不是方法 |
| P14：参数匹配 split-path 强于 loss protection | endpoint 部分命中；预期排序与 detail retention 失败 | 固定总容量隔离会缩小每个任务的表示 |
| P15：additive guardrail 优于 matched-scale control | DCT/PCA 10/10 endpoint 命中；detail `50%` 门槛失败 | 定向风险是因果变量，但共享模型存在 Pareto 竞争 |
| P16：split-additive 同时保住 coarse/detail | coarse hash 与 endpoint 命中；detail 失败 | 严格保护成立，完整 Pareto 不成立 |
| P17：full-width detail + small coarse 实现 Pareto | wide control 命中；绝对 baseline 与 detail 相对门槛失败 | 普通扩宽无效；专门化收益与 coarse underfit 并存 |
| P18：DCT/PCA 都是 gradient conflict | allocation/coarse descent 命中；仅 Fashion PCA conflict，DCT 失败 | 修正为 conflict/neglect 分型 |
| P19：MNIST 复现梯度分型 | 全部门槛命中 | 梯度弱耦合与 random 强耦合可跨两数据集复现 |
| P20：冻结 RAE 复现 basis-specific neglect | 全部门槛命中，basis gap 远超阈值 | 小图像梯度机制桥接到 RAE stage-2 |
| P21：individual-band endpoint 按下降预算排序 | band0 3/3 第一；bands0-4 Spearman 1.0 | nonzero 66% 是多 band 累积，不是否定 coarse leverage |
| P22：frozen projected residual 实现 Pareto | 精确保护命中；DCT 回到 1.009，但 detail retention 42%；PCA 1.208；rollout target 反向 | 机制可消除大部分损伤，但未带来绝对质量收益 |
| P23：offpath paired MSE 是反向来源 | 全部命中；offpath-only 1.131，drift-only 1.034 | 原 pair target 不能直接监督 self-generated states |
| P24：validation trust region 获得至少 2% 提升 | 失败；3/5 退回 scale 0，无中间 scale，test 均值 0.9978 | residual correction 没有稳定正 endpoint leverage |

失败预测没有被删除，它们把最终机制从“teacher forcing 导致一切”修正为更精确的风险方向、数据几何、参数化和因果杠杆共同作用。

## 当前可以和不可以声称什么

### 可以声称

1. 固定 inverse-variance output weighting 不是有限网络中的无害优化预条件器。
2. 当权重基底与数据能量/结构方向对齐时，它会系统性低权重高能量方向。
3. 这些方向的普通 MSE 占比与 endpoint feature leverage 不匹配。
4. high-window 高能量子空间 field difference 对两种小图像数据的损伤具有主要因果贡献。
5. matched-scale control 证明 endpoint rescue 来自定向 coarse 风险，而不是总 loss scale。
6. 两个数据集都复现 DCT/PCA 的 coarse/detail 梯度弱耦合与 random 的强正耦合；这解释了等谱基底为何结果不同。
7. random-basis、time-window、group-splice、teacher-restart、exact-skip、training guardrail 和 gradient audit 共同支持四因素机制。
8. 冻结 tiny RAE 中，同谱 DCT/random 对照复现了 basis-specific gradient
   neglect，而不依赖卷积 U-Net。
9. RAE high-window 的逐 band endpoint 损伤由 gradient descent collapse
   预先预测；band 0 单项最大，nonzero 总量更大来自七项近似累积。

### 仍不能声称

1. 所有 spectral loss 都会伤害生成。
2. 高能量方向必然比低能量方向重要。
3. radial group 0 可以解释 RAE 的完整退化；现有 RAE 中非零 bands 在中段贡献更大。
4. diagonal energy drift 或 one-step SWD 是充分预警指标；本轮它们有多次漏检。
5. 已经得到适用于大 RAE 的最终训练修复。
6. gradient conflict 是统一原因；DCT 主要是 weak-coupling neglect，只有部分 PCA 配置是真冲突。
7. 小图像 guardrail 已实现 Pareto；所有训练侧保护都牺牲了部分 detail gain 或改变了 architecture baseline。
8. P20/P21 已证明修改 RAE 训练梯度一定改善 FID；它们是冻结梯度和 latent
   field-splice 证据，尚未完成 paired RAE training intervention。
9. 8-sample P20、单个 random basis 和 32-sample P21 proxy 可以替代全网络或
   50k ImageNet 审计。
10. 已确认的失败机制本身保证生成质量提升；P22-P24 只证明它能指导避免大部分
    退化，未通过绝对 baseline 质量 gate。

## 对 RAE 下一步的直接含义

不应再按固定 DCT variance 调 `gamma`。P20/P21 已无训练地估计了 high-window
radial DCT directions 的 gradient allocation 与 endpoint leverage：

```text
leverage(direction, time)
  = endpoint distribution change caused by a controlled field splice
    / teacher MSE mass of that field difference.
```

在当前 radial DCT 分解中，band 0 是最大单 band 杠杆，但 bands 1-7 的小损伤
会近似累积并占总量约 `64%`。所以训练侧不能只保护 band 0，也不能按 radial
energy 一项决定保护集合；必须同时检查 loss allocation、shared-parameter 梯度
耦合和 endpoint leverage。PCA/semantic/cross-band 子空间仍是未检验的推广方向。

方法候选也因此变得具体：

1. 保留原始 MSE，不再全局替换成方向加权风险。
2. 为高-leverage 且与其他任务梯度弱耦合的 branch 设置 no-regression guardrail。
3. 或显式提供 learned linear/coarse skip，让细节方向预条件不再与关键输运竞争。
4. 只在 groups 1-7 或低-leverage residual branch 使用预条件。

小图像 group-splice 已证明第 2/4 条在采样侧能移除约 90% 的 high-window 损伤；下一阶段需要在不访问 baseline field 的训练侧实现同样保护。

但 P13-P17 也证明，保护本身不等于 Pareto：固定容量 split 会缩小单任务表示，
小 coarse branch 又会 underfit。任何 RAE 方法在大训练前必须先同时报告绝对
coarse/detail risk、gradient coupling 和 endpoint proxy，不能只展示相对 ratio。

## 复现与数据质量

代码入口：

- `experiments/nonlinear_transport_gap.py`
- `experiments/nonlinear_raw_velocity_gap.py`
- `experiments/small_image_basis_transport.py`
- `experiments/small_image_basis_mechanism.py`
- `experiments/small_image_time_switch.py`
- `experiments/small_image_teacher_restart.py`
- `experiments/small_image_dense_teacher.py`
- `experiments/small_image_group_splice.py`
- `experiments/small_image_training_rescue.py`
- `experiments/small_image_additive_guardrail.py`
- `experiments/small_image_split_guardrail.py`
- `experiments/small_image_asymmetric_guardrail.py`
- `experiments/small_image_gradient_allocation.py`
- `experiments/rae_frozen_gradient_bridge.py`
- `experiments/run_rae_individual_band_leverage.py`
- `experiments/small_image_residual_adapter.py`
- `experiments/small_image_residual_trust_region.py`

预测按实验顺序写在 `docs/SMALL_TRANSPORT_GAP_PREREG_ZH.md`。

正式结果位于：

- `$HOME/data/eqvae/experiments/nonlinear_transport_gap/preregistered_v1_20260716_141255/`
- `$HOME/data/eqvae/experiments/nonlinear_raw_velocity_gap/preregistered_v1_20260716_180622/`
- `$HOME/data/eqvae/experiments/small_image_basis_transport/fashion_mnist_preregistered_v2_deterministic_20260716_171615/`
- `$HOME/data/eqvae/experiments/small_image_basis_transport/mnist_preregistered_v2_deterministic_20260716_175304/`
- `$HOME/data/eqvae/experiments/small_image_training_rescue/`
- `$HOME/data/eqvae/experiments/small_image_additive_guardrail/`
- `$HOME/data/eqvae/experiments/small_image_split_guardrail/`
- `$HOME/data/eqvae/experiments/small_image_asymmetric_guardrail/`
- `$HOME/data/eqvae/experiments/small_image_gradient_allocation/`
- `$HOME/data/eqvae/experiments/rae_frozen_gradient_bridge/preregistered_20260716_200310/`
- `$HOME/data/eqvae/experiments/rae_individual_band_leverage/preregistered_20260716_200958/`
- `$HOME/data/eqvae/experiments/small_image_residual_adapter/fashion_mnist_residual_adapter_preregistered_20260716_215425/`
- `$HOME/data/eqvae/experiments/small_image_residual_adapter/fashion_mnist_residual_adapter_preregistered_20260716_220534/`
- `$HOME/data/eqvae/experiments/small_image_residual_trust_region/residual_trust_region_preregistered_20260716_221738/`

审计结果：

- 所有正式 CSV 行数符合设计，复合主键无重复，数值列无非有限值。
- 预期空值仅为 PCA 专属 metadata 和非-teacher context 不定义的 field MSE。
- 同 seed 的 DCT/PCA/random baseline checkpoint 哈希完全一致。
- 三种基底的权重谱差异仅为 fp32 舍入，最大约 `1.9e-6`。
- high-window time-switch 与 group-splice 的独立实现最大 ratio 差异为 `0`。
- middle teacher-restart 与 group-splice 的独立实现最大 ratio 差异为 `0`。
- 审计分类器未保存原 checkpoint，重训精度与原运行最多相差 `0.68` 个百分点；所有因果 ratio 均在同一个审计分类器内部严格配对。
- P13-P17 新结果共 `405` 个 variant-summary rows，复合主键无重复且数值全部有限；重新评估的 raw weighted ratio 与原正式 CSV 逐 seed 差异为 `0`。
- P16/P17 的受保护 coarse state hash 分别在 `15/15`、`10/10` runs 完全一致。
- P20 的 aggregate/band/cosine 表为 `72/576/4,608` 行；P21 的
  metrics/paired/band 表为 `120/120/240` 行，均无重复主键或非有限值。
- P21 三 seed 使用相同的 32 个唯一 validation indices；individual-band 和
  high-all 的加和残差为 `-1.03%` 到 `+2.37%`。
- P22/P23/P24 顶层结果为 `40/20/25+15` 行，无重复键；P24 每 seed 的
  source/validation/test 各 1,024 个 indices 完全不相交。
- Fashion/MNIST gradient audit 各 `360` 行，复合主键无重复、数值全部有限；两个数据集均使用相同 `t=0.7/0.9` 和 held-out batch 定义。

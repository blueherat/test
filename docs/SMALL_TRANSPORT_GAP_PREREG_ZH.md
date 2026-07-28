# 小型 Flow Transport Gap：机制预测预注册

## 冻结时间与证据边界

本文件在运行新的低维解析 flow 实验和 FashionMNIST / 跨基底实验之前写入。已有证据包括：

- width-24 MNIST 的 3-seed teacher/rollout 反转。
- MNIST 的 solver、能量校准和 time x band splice。
- RAE 的 3-seed time x band splice 与 shared-state drift。
- 已有 width 12/24/48 初步结果已经表明 rollout penalty 不随容量单调消失。

因此“增加容量会单调消除反转”不再是新预测，而是一条已被否定的简单机制。

## 当前候选机制 H2

候选机制包含三个缺一不可的环节：

1. **有限模型投影改变。** 方向加权改变共享参数网络在 teacher marginal `p_t` 上的有限函数类投影，而不只是改变到同一有限模型解的收敛速度。
2. **高噪声输运启动误差。** 某些点对点 MSE 很小的误差对 `E[<z,v>]`、cross-covariance 或粗结构 transport 有较大影响，令自生成 marginal `q_t` 提前偏离 `p_t`。
3. **中段状态反馈。** 同一个加权 field 在 teacher states 上可能不差，但在偏移后的 `q_t` states 上误差放大，继续推动轨迹偏离。

如果只有第 1 项，机制只能称为 finite-capacity reallocation；如果没有第 2 和第 3 项，就不能解释稳定的 endpoint reversal。

## 新实验前冻结的预测

### P1：低维 exact-oracle 反转不是必然，但应可在受限模型中出现

在具有解析 conditional velocity 的 8 维独立 Gaussian-mixture flow 中：

- 可表达模型趋近 oracle 后，baseline 与正定方向加权应收敛到相同 field 和 endpoint。
- 共享瓶颈或有限训练下，方向加权可以改善一部分 teacher directions，同时损害 endpoint distribution。
- 若 MLP 与 mini-DiT 在所有容量和 5 个 seed 上都不出现这种分离，则当前现象依赖图像局部结构或更高维流形，不能宣称为一般 Flow Matching 机制。

判据：至少一个预先声明的模型容量中，5 个 paired seeds 至少 4 个满足“部分 teacher direction 改善，但 endpoint mean-coordinate W1 或 covariance error 恶化”。

### P2：off-path 剂量响应必须可见

用解析 oracle velocity 定义：

```text
z_alpha(t) = (1-alpha) z_teacher(t) + alpha z_weighted_rollout(t)

Delta(alpha,t) = log(
    MSE(v_weighted(z_alpha,t), v_oracle(z_alpha,t)) /
    MSE(v_baseline(z_alpha,t), v_oracle(z_alpha,t))
).
```

对于出现 endpoint reversal 的配置，预测：

- 在 `t in [0.3, 0.7]`，`Delta(alpha,t)` 对 `alpha` 的平均斜率为正。
- 至少 4/5 paired seeds 的中段平均斜率为正。
- 强预测是 `Delta(0,t) < 0` 而 `Delta(1,t) > 0`，即从 teacher 优势跨越到 rollout 劣势；若只满足正斜率而不跨零，只能称为 off-path amplification，不能称为完整符号翻转。

若 endpoint 明显恶化但 `Delta` 不随 `alpha` 增大，则“中段 off-path field failure”不是主要机制，需要转向 cross-time accumulation 或非 MSE transport moments。

### P3：输运统计应先于 endpoint 预警

对每个时间与状态，比较模型和 oracle 的：

```text
field MSE
mean radial/coordinate drift: E[z * v]
cross-covariance drift: E[z_i v_j + v_i z_j]
```

预测：高噪声阶段的 drift/cross-covariance error 会在 endpoint 分布明显分开前出现，并且其 paired effect 与最终 endpoint penalty 同号。

若 teacher MSE 已经足以稳定预测 endpoint，而 drift 没有增加解释力，则当前“pointwise risk 与 marginal transport 错配”的核心表述被削弱。

### P4：off-path 指标应比 teacher MSE 更能跨配置预测 endpoint

跨 architecture、capacity、gamma 和 seed，计算：

- teacher exact field MSE ratio。
- middle-time rollout-state exact field MSE ratio。
- high-time drift/cross-covariance error ratio。
- endpoint W1/covariance error ratio。

预测：middle rollout-state error 或 high-time transport error 与 endpoint penalty 的 Spearman 相关性，至少比 aggregate teacher MSE 高 `0.15`。

若差值小于 `0.15` 或跨 seed 不稳定，当前 diagnostic 尚不能作为可预测机制，只能作为事后描述。

### P5：方向基底决定的是发生概率，而非唯一原因

在 MNIST/FashionMNIST 中匹配权重均值和条件数后：

- DCT 与 PCA-aligned weighting 应比随机正交 weighting 更容易产生 teacher/rollout 分离。
- 若随机正交基底同样稳定地产生分离，则机制应改写为一般的 output-risk reweighting + state feedback，而不是频率机制。
- 若只有 DCT 产生分离，则机制应限制为图像局部性与频率方向的特殊耦合。

此项必须在独立 FashionMNIST test split 和 5 个 paired seeds 上判断，不能只看单个 MNIST seed。

### P6：容量只解释 teacher reallocation，不充分解释 endpoint

这是根据现有 width sweep 修订后的预测：

- width 增大应减小 baseline/weighted 在 teacher states 上的平均差异。
- endpoint penalty 不要求随 width 单调下降。
- 若 off-path sensitivity 随 width 的变化能够解释这种非单调性，则支持 H2；否则还缺少 architecture-specific dynamics。

### P7：FashionMNIST 时间窗因果切换

这组预测写于确定性 FashionMNIST 等谱实验和 shared-state audit 之后、首次
time-window field switch 之前。已知 DCT/PCA 在高噪声 teacher drift 上更差，
但中段 one-step off-path SWD ratio 接近 1。

从 baseline 噪声和轨迹开始，只在指定时间窗换入 weighted field。预测：

- DCT/PCA 的 `t>=0.7` high-only switch 应在至少 4/5 seeds 上恶化 feature FID。
- 若 H2 的第二阶段在 FashionMNIST 也必要，则 `0.3<=t<0.7` middle-only
  switch 也应在至少 4/5 seeds 上恶化 feature FID。
- 若 high-only 稳定恶化而 middle-only 不恶化，则机制必须修订为：简单图像
  上主要是 high-time on-path transport mismatch；RAE 的 mid-time off-path
  amplification 是高维语义 latent 的附加机制，不是普遍必要环节。
- 随机正交基底的 high/middle switch 应明显弱于 DCT/PCA；否则等谱 endpoint
  对照可能只是训练噪声，而不是几何对齐。
- low-only 预期最弱。如果 low-only 反而稳定解释主要损伤，当前时间因果推断失败。

### P8：middle-window teacher restart 与 rollout start

这组预测写于 P7 time-window switch 完成之后、首次 teacher-restart 实验之前。
P7 已知：DCT/PCA 的 high-only 在 5/5 seeds 伤害 feature FID，middle-only
分别在 4/5 和 5/5 seeds 伤害，而 low-only 很弱。

从两种相同时间 `t=0.7` 的状态开始，只在 `0.3<=t<0.7` 使用 weighted field：

1. `teacher restart`：直接从 held-out `(1-t)x+t epsilon` 采样真实 `p_0.7`。
2. `rollout start`：从 baseline 自噪声积分到 `t=0.7` 得到 `q_0.7^B`。

H2 的 off-path amplification 预测：

- DCT/PCA 的 middle penalty 在 rollout start 上应大于 teacher restart。
- 强预测是 teacher restart 接近无害、rollout start 明显有害。
- 若两种 start 的 penalty 相当，则 FashionMNIST 的 middle 损伤主要也是
  on-path pointwise-risk / marginal-transport mismatch，off-path feedback 不是必要解释。
- 若 teacher restart 反而更差，则当前把 middle 损伤归于 state shift 的解释被否定。

比较使用同一审计分类器、同一 held-out target 和同样 `dt=0.02`，只把
`t=0.7` 起始 marginal 作为干预变量。

### P9：dense-time teacher metric 排除稀疏采样假象

这组预测写于 P8 完成之后、首次 dense-time audit 之前。P8 已否定
“FashionMNIST middle 损伤必须由 off-path state shift 触发”：teacher restart
的 FID penalty 对 PCA/DCT 没有减小，PCA 反而明显增大。

替代解释是原 teacher audit 只测 `t=0.3/0.5/0.7`，漏掉 `0.32-0.68`
之间的 MSE 峰值。现在固定每 `0.02` 一个时间点，预注册两种互斥判断：

- **H_sparse-time：** DCT/PCA 在 middle window 的 dense raw teacher MSE
  累计 ratio 会大于 1；之前的“teacher 改善”只是稀疏取点假象。
- **H_transport-risk（当前主预测）：** DCT/PCA 在至少 4/5 seeds 上的
  middle dense teacher MSE 累计 ratio 仍小于 1，但 teacher-restart endpoint
  FID ratio 大于 1；同时 component drift error 不随 MSE 同向改善。

选择 H_transport-risk 的理由是网络对时间连续，且 `t=0.3/0.5` 的改善跨两个
相距较远的锚点稳定出现。若 H_sparse-time 命中，则核心机制应降级为时间采样
不足，而不是 pointwise risk 与 marginal transport 的断层。

### P10：高能量低权重子空间的因果 splice

这组预测写于稀疏/稠密 teacher 分带审计之后、首次 group-selective field
splice 之前。已知：

- DCT/PCA 的 group 0 占 `1/8` 系数，但每系数实际 clean second moment
  约为 `7`，其余组通常小于 `0.4`。
- DCT/PCA group 0 在 `t=0.5/0.7/0.9` 的 teacher error 选择性恶化，
  groups 1-7 大多改善。
- 随机基底中每个权重组的实际能量都约为 `1`，没有这种对齐。

原因预测来自权重公式：

```text
R_b(t) = m_b / ((1-t)^2 m_b + t^2)
w_b(t) proportional to R_b(t)^(-gamma).
```

当 `t -> 0`，`R_b` 近似与 `m_b` 无关；当 `t -> 1`，高能量 `m_b`
对应更低权重。因此固定 inverse-variance weighting 会在高/中噪声系统性牺牲
少数高能量粗结构/主成分方向。

冻结预测：

- 在 high-window rollout 和 middle-window teacher restart 中，只换入 group 0
  的 weighted-baseline field difference，应在 DCT/PCA 至少 4/5 seeds 上伤害
  feature FID。
- group 0-only 应解释 full splice FID 差值的多数；groups 1-7-only 应明显弱，
  甚至可能改善，因为这些组的 teacher error 普遍下降。
- 随机基底 group 0-only 不应稳定伤害，因为其 group 0 不承载特殊高能量结构。
- 若 DCT/PCA 的损伤主要由 groups 1-7 产生，则“高能量低权重子空间”机制
  被否定，即使 band-0 teacher error 的相关性很好看。

### P11：MNIST 跨数据复现

这组预测写于 FashionMNIST P10 完成之后、确定性 MNIST 等谱三基底实验之前。
模型、训练步数、样本数、gamma、权重谱和 5 个 seeds 与 FashionMNIST 完全
相同，不根据 MNIST 结果调参。

冻结预测：

- MNIST 的 DCT/PCA 实际能量比 FashionMNIST 更集中，因此 inverse-variance
  weighting 同样会低权重高能量 group 0。
- DCT/PCA 应在至少 4/5 seeds 上同时出现：低中噪声 aggregate teacher MSE
  改善、高噪声 MSE 恶化、endpoint feature FID 恶化。
- 随机正交基底的 teacher 与 endpoint ratio 应更接近 1，且不能稳定复现
  DCT/PCA 的同向伤害。
- high-window group 0-only splice 应解释 DCT/PCA high-window FID delta 的
  多数，预注册阈值为平均 `>=70%`。

若 MNIST 只复现 DCT 而不复现 PCA，则机制应限制为卷积网络与空间频率的
特殊耦合；若随机基底也同样稳定恶化，则 FashionMNIST 的方向对齐解释失效。

### P12：解析 toy 中移除 exact linear skip

这组预测写于 MNIST P11 完成之后、首次 raw-velocity analytic toy 训练之前。
原 8 维 analytic toy 没有稳定复现 endpoint reversal，但它给网络外接了 exact
best-linear skip，网络只拟合 nonlinear residual；图像 U-Net 则必须自己学习
完整 raw velocity，包括承载大部分能量的粗结构线性项。

保持同一 mixture、seeds、gamma、steps 和 endpoint audit，仅移除 oracle linear
skip，让 `raw_mlp` 直接预测 microscopic velocity。冻结预测：

- raw model 的高方差方向 teacher error 应比 residual+skip 模型更容易被
  inverse-variance weighting 牺牲。
- 至少一个预注册容量 `hidden=24/96` 应在 4/5 seeds 上出现 endpoint W1 或
  covariance degradation；若出现，则 exact skip 是原 analytic toy 阻断反转的
  关键保护机制。
- 若 raw model 仍主要改善 endpoint，则高能量低权重仍不是充分条件，必须再加
  “高能量方向承载离散语义/空间结构”或卷积局部归纳偏置，不能把图像结论推广
  成一般低维 Flow Matching 定理。

### P13：训练侧保护高杠杆粗结构方向

这组预测写于 P12 完成之后、首次训练 `coarse_protected` 模型之前。P10 的
sampling-time splice 已说明 high window 的 group 0 field difference 对
DCT/PCA endpoint 损伤有主要因果贡献，但这种事后替换使用了 baseline field，
还不能证明机制能够指导训练修复。

`coarse_protected` 仍使用同一个 width-24 U-Net 和完全相同的训练 batch，唯一
变化是把 group 0 的 coefficient weight 固定为 `1`。groups 1-7 保留原
inverse-variance 相对权重，再统一缩放，使全部 784 个 coefficient 的平均权重
严格等于 `1`。因此它不是减小总 loss scale，也没有增加模型容量。

冻结预测：

- FashionMNIST 的 DCT/PCA 上，`coarse_protected` 相对 full `weighted` 的
  endpoint feature FID 至少在 `4/5 seeds` 改善，且平均 FID ratio 更接近
  baseline。
- high teacher path 的 group 0 MSE ratio 应比 full `weighted` 更接近或低于
  `1`；若 group 0 仍同等恶化，说明“低 loss 权重直接造成粗方向退化”不足。
- groups 1-7 的 high-time teacher MSE 改善量应至少保留 full weighting 的
  `50%`。否则 endpoint 修复可能只是退回普通 MSE，不能视为机制性 rescue。
- MNIST 应复现方向相同的平均改善；考虑已有 seed-4 反例，不要求 5/5 单调。
- 等谱 random basis 没有特殊高能量 group 0，因此保护其 group 0 不应产生
  DCT/PCA 同等稳定的 endpoint rescue。

若保护 group 0 不改善 endpoint，P10 只能解释两个已训练 field 的差异，不能
推出训练风险如何形成该差异，“高杠杆方向被降权”机制必须降级。

### P14：参数通路隔离应强于单纯 loss 保护

这组预测与 P13 同时写入、首次训练 split-path 模型之前。当前机制还包含一个
更强条件：有限共享参数使粗结构与大量细节方向发生梯度竞争。只把 group 0
权重恢复为 `1` 仍允许共享 U-Net 的参数被 groups 1-7 梯度改变，因此它应是
不完全修复。

`split_baseline` 和 `split_weighted` 都使用两个独立 width-17 U-Net：一个输出
只投影到 group 0，另一个只投影到 groups 1-7。两者总输出相加。两条 width-17
通路合计 `231,678` 个参数，单个 raw width-24 U-Net 为 `229,897`，参数量只差
`0.8%`。两个变体架构、初始化和 batch 完全配对：

- `split_baseline` 对两个通路都用普通 coefficient MSE；
- `split_weighted` 的 coarse 通路同样用普通 MSE，只有 detail 通路使用
  inverse-variance 相对权重。

冻结预测：

- DCT/PCA 的 `split_weighted / split_baseline` endpoint feature FID ratio 应
  显著小于 raw `weighted / baseline`，至少在 FashionMNIST `4/5 seeds` 成立。
- split coarse 通路的 teacher group-0 MSE 不应出现 raw weighted 中的系统性
  high-time 恶化；detail 通路仍应保留 groups 1-7 的多数改善。
- 若 P13 只部分修复，split-path 应进一步改善；预注册排序为
  `raw weighted` 最差、`coarse_protected` 居中、`split_weighted` 相对其各自
  baseline 最接近 `1` 或更好。
- random basis 的 `split_weighted / split_baseline` 不应显示 DCT/PCA 特有的
  大幅 rescue。由于总参数量已匹配，该对照也不能用“容量翻倍”解释。

若 split-path 仍稳定复现相同 endpoint 退化，则“共享容量竞争”不是必要条件；
应转向目标本身的多步输运偏差、局部卷积归纳偏置或 classifier feature
敏感性解释。

### P15：additive coarse guardrail 的 Pareto 修复

这组预测写于 FashionMNIST P13/P14 完成之后、MNIST P13/P14 尚在运行时，且
首次训练 additive guardrail 之前。P13/P14 已给出一个不能回避的失败：

- `coarse_protected` 和 split-path 都把 DCT/PCA endpoint 拉回 baseline 附近；
- 但 `coarse_protected` 为保持全部 coefficient 平均权重为 `1`，在把 group 0
  从约 `0.2` 提到 `1` 的同时，也把 groups 1-7 的平均权重从约 `1.11` 缩回
  `1`；
- 结果 DCT high-time nonzero MSE ratio 从 `0.966` 退到 `1.003`，PCA 从
  `0.899` 退到 `0.996`，没有满足“至少保留一半细节收益”的预注册标准。

因此 P13 证明了粗方向是损伤开关，却还没有得到 Pareto rescue。下一项干预固定：

```text
additive_guardrail:
  group 0 weight = 1
  groups 1-7 weights = 原 inverse-variance weights，不再重新归一化

time_scale_control:
  所有原 inverse-variance weights 乘以 additive_guardrail 的样本均值
```

两者在每个样本、每个时间的 coefficient weight 总和完全相同。区别仅在于：
`additive_guardrail` 把新增 loss mass 定向给 group 0，`time_scale_control` 按原
weighted 比例分配。两者仍是同一个 width-24 U-Net、相同初始化、相同 batch，
不增加参数。

冻结预测：

- FashionMNIST DCT/PCA 的 additive guardrail high-time group-0 MSE ratio
  均值应 `<=1.02`，且至少 `4/5 seeds` 低于 time-scale control。
- additive guardrail 应保留 full weighted 的 high-time nonzero improvement
  至少 `50%`：定义
  `(1-r_guard_nonzero)/(1-r_weighted_nonzero)`，在 DCT/PCA 分别按五 seed
  聚合后应 `>=0.5`。
- additive guardrail endpoint feature FID ratio 均值应 `<=1.10`，并至少
  `4/5 seeds` 同时优于 full weighted 与 time-scale control。
- time-scale control 应更接近原 full weighted，而不是 additive guardrail；
  若两个变体同样被救回，说明关键是 time-dependent 总 loss scale，不是粗方向。
- random basis 没有特殊高能量 group 0，因此 additive guardrail 不应出现
  DCT/PCA 特有的稳定 Pareto rescue。

若 endpoint 被救回但 nonzero 收益仍消失，则当前 weighting 与生成质量之间没有
在该模型上的简单 Pareto 改进，机制只能用于解释失败，不能支持训练方法。若
time-scale control 同样有效，则 P13 的定向解释必须改写成高噪声整体训练强度。

### P16：split-additive 应解除 coarse/detail Pareto 竞争

这组预测写于 FashionMNIST P15 完成之后、MNIST P15 正在运行时，且首次训练
split-additive 模型之前。P15 命中了方向因果预测，但否定了简单 Pareto 修复：

- DCT/PCA endpoint ratio 从 `1.461/1.787` 降到 `1.053/0.923`；
- matched-total-loss scale control 仍为 `1.405/1.709`，证明定向保护而非总
  loss scale 是救援原因；
- 但 nonzero teacher-MSE improvement 只保留原 weighted 的约 `9%/15%`。

这正是“共享参数竞争”机制的下一条生死线。固定两个参数匹配的独立 width-17
通路，并训练：

```text
split_baseline:
  coarse path: group-0 ordinary MSE
  detail path: groups-1-7 ordinary MSE

split_additive_guardrail:
  coarse path: group-0 ordinary MSE
  detail path: groups-1-7 original inverse-variance weights

split_time_scale_control:
  coarse/detail paths: original weights * matched total-loss scale
```

三者初始化和 batch 完全配对。本实验将用显式 branchwise coefficient loss
重训 split baseline，避免“先合成像素输出、再投影回 coefficient”产生的 fp32
子空间泄漏；同时审计其 endpoint 是否复现 P14 split baseline。新 guardrail 的
coarse path 与它参数独立于 detail path，接收逐 batch 完全相同的 group-0
梯度，并独立进行 gradient clipping。因此在确定性 fp32 运行中，两者 coarse
参数应逐位一致；这既是机制预测，也是实现审计。

冻结预测：

- `split_additive_guardrail` 与 `split_baseline` 的 coarse state hash 必须完全
  相同，high-time group-0 teacher MSE ratio 应为 `1`（只允许浮点打印误差）。
- FashionMNIST DCT/PCA 的 high-time nonzero improvement retention 应分别
  `>=50%`，使用与 P15 相同定义、以 raw weighted gain 为分母。
- endpoint feature FID ratio 均值应 `<=1.10`，且至少 `4/5 seeds` 优于 raw
  weighted；若 detail gain 与 endpoint 同时满足，才称为 Pareto rescue。
- matched scale control 不应同时满足 coarse hash、detail retention 和 endpoint
  三项；random basis 不应呈现 DCT/PCA 特有的大幅 rescue。

若 coarse hash 一致但 detail gain 仍无法保留，则 raw weighted 的细节改善并非
简单来自 detail 子空间自身的加权拟合，可能依赖共享 coarse features、跨频耦合
或更宽的单通路表示。此时“有限容量竞争”只能解释 endpoint rescue，不能解释
原始 teacher gain 的来源，机制仍未闭合。

### P17：完整 detail 容量 + 小 coarse 支路

这组预测写于 FashionMNIST P16 完成之后、首次 asymmetric-path 训练之前。P16
的三项结果为：

- coarse state hash 在 15/15 runs 完全相同，DCT/PCA endpoint ratio 为
  `1.031/1.056`，保护粗方向足以移除大部分生成损伤；
- 但 DCT detail gain 完全消失，PCA 只保留约 `24%`，未达到 `50%` 门槛；
- 参数匹配的两个 width-17 通路虽然总容量不变，但 detail 单支路的宽度从 24
  降到 17。

替代机制因此收紧为：raw weighted 的细节收益需要使用完整 width-24 表征容量；
固定总参数拆分会保护 coarse，却缩小 detail 的可用表示。要实现 Pareto，可能
必须在完整 detail 网络之外增加一个小的受保护 coarse branch。

固定比较：

```text
asym_baseline / asym_weighted:
  detail path = width 24，只输出 groups 1-7
  coarse path = width 12，只输出 group 0
  weighted 只改变 detail path；coarse 始终 ordinary MSE
  total parameters = 229,897 + 58,069 = 287,966

wide_baseline / wide_weighted:
  单个共享 width-27 U-Net
  total parameters = 290,629
```

两种架构参数量只差 `0.9%`。每一对内部初始化和 batch 完全配对。第一轮只跑
FashionMNIST DCT/PCA 五 seeds；若未通过，不再花成本跑 MNIST/random。

冻结预测：

- asym weighted 与 asym baseline 的 coarse state hash 必须完全一致。
- asym weighted 的 DCT/PCA high-time nonzero gain retention 相对 raw weighted
  均应 `>=75%`；full width-24 detail path 应恢复 P16 丢失的收益。
- asym weighted endpoint feature FID ratio 均值应 `<=1.10`，且 asym baseline
  的绝对 FID 不应超过 raw baseline 的 `1.2x`，避免用一个坏 reference 制造好 ratio。
- 参数匹配的 wide weighted 仍应出现 group-0 MSE 恶化；其 endpoint ratio 应
  至少在 `4/5 seeds` 高于 asym weighted。若单纯 width-27 同样实现 Pareto，
  机制应改写为普通容量阈值，而不是受保护通路。

若 asymmetric path 保住 coarse 和 endpoint，却仍不能保住 detail gain，则
raw weighted 的 detail 改善依赖跨频共享 features 或 coarse/detail 输出耦合，
而非单纯 detail 宽度。此时当前机制仍只能解释损伤，不能给出成功修复。

### P18：shared-gradient allocation 与 conflict/neglect 分型

这组预测写于 FashionMNIST P17 完成之后、首次读取 gradient audit 结果之前。
P17 相对指标未通过，但绝对量揭示：

- asym baseline 的 DCT/PCA detail MSE 已比 raw baseline 低约 `3%/10%`，等于
  或优于 raw weighted；相对 detail gain 小有明显天花板效应；
- width-12 coarse branch 的绝对 group-0 MSE 又比 raw baseline 高约 `9%-10%`，
  导致 asym weighted 的绝对 FID 仍高约 `18%-19%`；
- 参数匹配 width-27 shared model 继续牺牲 group 0，说明普通扩宽不改变风险分配。

当前机制因此分成两个可区分版本：

```text
H_conflict:
  detail 与 coarse 在 shared parameters 上梯度负相关；提高 detail 风险会主动
  增大 coarse loss。

H_neglect:
  两者梯度未必冲突，但 weighted objective 把绝大多数更新预算分给 detail，
  coarse 收敛不足；当 coarse 恰好是高 endpoint-leverage 方向时生成恶化。
```

固定在已训练 raw baseline/weighted checkpoints 上，不再训练模型。使用 held-out
teacher states、`t=0.7/0.9`，分别计算 group-0 与 groups-1-7 loss 对 shared trunk
和 output head 的梯度、余弦、范数和一阶下降量。

冻结预测：

- 三种基底使用同一权重谱，因此 weighted/unweighted 的
  `detail/coarse gradient-norm ratio` 在高时间应至少增加 `3x`；这是风险预算
  重新分配的必要实现检查。
- DCT/PCA 上，weighted 总梯度对 coarse loss 的一阶下降量应小于 ordinary MSE
  总梯度的 `0.5x`，而对 detail loss 的下降量应更大。
- **冲突是待检验的强预测：** DCT/PCA 的 coarse/detail gradient cosine 在 shared
  trunk 上应比 random 更负，并在至少 `3/5 seeds` 的高时间平均值小于 `0`。
  若该项失败，但前两项成立，则机制必须明确归类为 H_neglect，不能再使用
  “梯度冲突”措辞。
- random 同样会有 loss-budget shift，但其 group 0 sampling-time endpoint
  leverage 已知接近零；因此只有“allocation gap × endpoint leverage”共同存在
  才应预测生成损伤。

若 DCT/PCA weighted 梯度并未减少 coarse 的一阶下降预算，则当前 loss-risk
解释与实际优化动力学矛盾，机制应被否定。若只是 cosine 不负，则删去 conflict，
保留高杠杆任务被低权重忽视的更窄机制。

### P19：MNIST 梯度机制跨数据复现

这组预测写于 FashionMNIST P18 完成之后、首次读取 MNIST gradient audit 结果
之前。FashionMNIST 已把统一的 H_conflict 修正为分型机制：

- DCT shared-trunk coarse/detail cosine 为 `+0.165`，不是负冲突，但耦合很弱；
- PCA 为 `-0.318`，存在真实冲突；
- random 为 `+0.947`，其 detail 梯度几乎同时训练 coarse；
- 对应 weighted coarse 一阶下降量相对 ordinary MSE 为 `0.213/0.161/0.992`。

这说明同一 weight spectrum 是否有害，取决于 group 划分是否把共享梯度也分成
弱耦合/冲突任务。现在不训练新模型，只把相同审计原样跑到 MNIST。

冻结预测：

- MNIST DCT/PCA shared-trunk cosine 五 seed 均值应 `<0.4`，random 应 `>0.8`。
- DCT/PCA weighted coarse-descent ratio 均值应 `<0.5`，random 应 `>0.8`。
- 三个基底 allocation multiplier 仍应 `>3`，证明差异来自梯度几何而非权重谱。
- DCT 可以是弱正相关而不要求负；只有 PCA 若均值小于 0 才称 conflict。
- 不预注册逐 seed endpoint 相关性，因为已有 MNIST seed-4 反例说明多步采样和
  训练随机性会改变 endpoint；P19 只检验 basis-level optimization mechanism。

若 MNIST random 也出现低 coarse-descent ratio，Fashion 的梯度解释不能跨数据；
若 DCT/PCA ratio 不低，则小图像机制仍是数据集特例，不能作为一般结论。

### P20：冻结 RAE checkpoint 的低成本梯度桥接

这组预测写于 P19 完成之后、首次计算当前三 seed RAE tiny EMA 的新梯度结果
之前。它不是新的 RAE 训练：只复用 seed `3407/4211/5821` 已有 step-10000
baseline/partial EMA、缓存的 ImageNet validation latents 和已有 endpoint splice。
不加载 decoder，不反传 RAE encoder；仅对 stage-2 DiTDH-S 的最后 transformer
block 与 final output linear 求解析/自动微分梯度。

已知但不作为新结果的事实：

- `t=0.85/0.95` 时 DCT band-0 权重为 `0.239/0.218`，其余 bands 平均约
  `1.11`；三 seed 配置完全一致。
- 现有 64-latent sampling splice 中，high-window band 0 约解释总 summary-SWD
  损伤的 `34%`，nonzero bands 约解释 `66%`，所以 RAE 不能预设“band 0
  解释一切”。
- 较早的官方 ep14 final-head sketch 中，DCT band-0 与其他 bands 的相关仅约
  `0.03`；当前 tiny checkpoints 尚未做同定义审计。

固定设计：同一误差分别投影到 DCT 与一个固定等维 random orthogonal basis；
两者使用完全相同的八个 band weights。验证样本固定为 cache 前 `8` 个，时间固定
`0.95/0.85/0.70`，batch size `2`，fp32、TF32 off。报告每个 band 的 shared-
parameter gradient cosine、weighted/unweighted 一阶下降比，以及 band0-vs-nonzero
聚合指标。

冻结预测：

- DCT 在 `t=0.85/0.95` 的 band-0 weighted coarse-descent ratio 应 `<0.5`，
  baseline EMA 至少 `2/3 seeds` 在 last block 与 output head 都满足；否则小图像
  risk-allocation 机制不能桥接到 RAE。
- DCT band0/nonzero cosine 在 last block 应 `<0.30`；允许正的 weak coupling，
  不要求 conflict。
- **basis-specific 桥接门槛：** random basis 的 high-time band0/nonzero cosine
  应比 DCT 高至少 `0.10`，coarse-descent ratio 高至少 `0.15`。若该项失败，
  只能说 RAE 的 DCT band0 被低权重，不能说数据对齐基底导致特殊梯度 neglect。
- partial EMA 应保留同方向的 DCT budget collapse；若只有 baseline 存在，说明
  梯度几何随训练发生反转，机制不稳定。
- 即便 P20 全部命中，也只解释 high-window band0 部分；必须结合 per-band
  endpoint leverage 才能解释 nonzero bands 的 `66%`，不能宣称 RAE 完整闭环。

若 DCT budget collapse 命中但 random 对照失败，最终结论必须保留为“小图像
机制得到部分 RAE 一致性证据”；只有梯度基底对照与 endpoint leverage 同时命中，
才可称跨 RAE 机制证据。

### P21：RAE high-window 逐 band endpoint leverage

这组预测写于 P20 完成之后、首次运行 RAE individual-band endpoint splice 之前。
P20 三 seed 正式结果完整命中：baseline EMA high-time DCT last-block 的
band0/nonzero cosine 为 `0.098`、coarse-descent ratio 为 `0.244`；等维 random
分别为 `0.836/0.952`。partial EMA 与 output head 同方向。

P20 的 DCT baseline per-band high-time descent ratio 为：

```text
band:       0      1      2      3      4
ratio:    .244   .511   .760   .901  1.022
```

bands 5-7 的 ordinary-MSE descent denominator 接近零，ratio 不稳定，不按其
大数值排序。现有 64-latent aggregate splice 已知 band0 约占 high-window 总损伤
`34%`，七个 nonzero bands 合计约 `66%`；这并不表示任一 nonzero band 比 band0
更有杠杆。

固定新实验：不训练，只对现有 baseline/partial EMA 做 `t>=0.85` field splice；
分别只换 band `0..7`，另含 baseline 与 high-all。使用三个 paired seeds、cache
前 `32` 个 validation latents、相同 evaluation seed、50-step official Euler、
fp32。主 endpoint proxy 是 `summary_swd_to_validation` 相对 baseline 的配对增量。

冻结预测：

- band 0 应是八个 individual bands 中平均 endpoint degradation 最大者；至少
  `2/3 seeds` 排名第一。
- band 1 应比 bands `3-7` 的平均损伤更大；bands `4-7` 单独应接近中性。该排序
  来自下降预算，而不是频率标签本身。
- individual-band degradation 与稳定 bands `0-4` 的
  `1 - coarse_descent_ratio` Spearman 相关应 `>=0.5`。
- high-all 可以大于 band0，因为七个 nonzero field differences 会累积并产生
  交互；不要求 individual delta 可加。
- 若任一 band `4-7` 稳定比 band0 更有害，或相关为负，则“gradient neglect ×
  endpoint leverage”不足以解释 RAE high-window，必须加入 decoder/semantic
  band sensitivity 或跨 band interaction。

P21 只针对 high window。middle-window 已有 shared-state sign reversal，不能用
同一静态梯度排序直接外推。

### P22：冻结 baseline 的投影 residual adapter 能否把机制变成质量收益

这组预测写于首次训练或评估当前 residual-adapter 方案之前。P13-P17 已知结果
说明：在同一共享模型中补 coarse 权重会拿走 detail gain；参数匹配 split 会缩小
每条路径容量；full-detail + small-coarse 又因新 coarse branch 欠拟合而失去绝对
baseline。尚未检验的是保留已经训练好的 raw baseline vector field，并只学习一个
不能改动保护子空间的增量。

固定方法：

- 从正式 FashionMNIST DCT/PCA 五 seed raw baseline checkpoint 开始，冻结其全部
  参数，不继续微调。
- 增加一个零初始化 `TinyVelocityUNet(width=12, depth=2)` residual adapter；其
  输出在固定训练基底中投影到 groups `1-7`，group 0 始终精确为零。
- `teacher_residual` 只在原 teacher interpolation states 上优化 groups 1-7 的
  weighted residual MSE。
- `rollout_drift_residual` 使用相同初始化和 batch stream；目标为 teacher residual
  MSE，加权 `0.5` 的单步 self-generated-state residual MSE，以及加权 `0.25` 的
  归一化 per-band marginal-drift mismatch。单步沿采样方向使用
  `dt=min(0.1, 0.5t)`，生成状态 stop-gradient，不反传 frozen baseline。
- 两个 adapter 都训练 `1000` steps、AdamW `2e-4`、batch `128`，沿用原
  `8192/1024` train/test split、50-step rollout、同 seed classifier 和 fp32
  deterministic 设置。第一轮只跑 FashionMNIST DCT/PCA 五 seed，不调 width、
  loss coefficient 或时间步。

冻结预测：

1. 两个 residual variants 在任意同一输入状态上的 group-0 field delta 最大绝对值
   必须 `<1e-5`；否则“保护路径”实现无效，结果作废。
2. 因 baseline coarse 不再欠拟合，两个 residual variants 的 mean feature-FID
   ratio 应显著优于 P15 additive guardrail 和 P17 asymmetric weighted；至少不能
   复现 raw weighted 的 `1.46/1.79` 大幅恶化。
3. `teacher_residual` 应保留 raw weighted high-time nonzero-MSE gain 的至少
   `50%`，同时 DCT/PCA mean feature-FID ratio 相对 raw baseline `<=1.02`。
4. `rollout_drift_residual` 应在两个基底上达到 mean feature-FID ratio `<=0.98`，
   且至少 `4/5 seeds` 不差于 raw baseline；mean latent-SWD 与 feature-SWD ratio
   均应 `<=1.02`。
5. small-image 的主要根因是 protected-path 缺失，已有 teacher-restart 说明
   off-path 不是必要条件。因此不强制 rollout-drift variant 显著胜过
   teacher-only；但它不能比 teacher-only mean FID 恶化超过 `2%`。若它更差，
   说明当前单步 off-path target 或 drift 正则定义错误。

方法 gate：只有第 1、3、4 条同时成立，才能说机制产生了低成本生成质量收益，
并允许在 MNIST 复现；MNIST 也达到至少 `4/5 seeds` 不差于 baseline 后，才允许
设计一次 tiny RAE residual-adapter screen。若只移除退化但不能优于 baseline，
结论必须写成“机制可用于避免伤害，尚不能提高质量”。若 detail gain 与 FID 再次
冲突，则说明低 teacher error 本身没有正 endpoint leverage，方向加权方法线关闭。

### P23：错误来自 off-manifold paired MSE，还是 marginal drift 项

这组预测写于 P22 完成后、首次训练两个新消融之前。P22 的 FashionMNIST 结果为：

- DCT `teacher_residual` feature-FID ratio 为 `1.009`，combined rollout/drift 为
  `1.147`；PCA 分别为 `1.208/1.368`。
- 两者 group-0 teacher MSE ratio 都精确为 `1.000`，投影误差小于 `7e-7`，所以
  恶化不能归因于 coarse 保护失效。
- combined 最终 offpath detail loss 约 `0.20-0.25`，而 normalized drift loss
  只有约 `3.5e-4-4.6e-4`；乘固定系数后，drift 对标量 objective 的贡献比
  offpath MSE 小三个数量级。

固定消融：只跑 FashionMNIST DCT 五 seed，复用 P22 完全相同的 frozen baseline、
width-12 adapter 初始化、batch stream 和 1000 steps。

- `offpath_mse_residual = teacher_detail + 0.5 * offpath_detail_mse`
- `drift_only_residual = teacher_detail + 0.25 * offpath_normalized_drift`

冻结预测：

1. `offpath_mse_residual` 的 mean feature-FID ratio 应比 teacher-only 至少差
   `0.08`，并与 P22 combined 相差不超过 `0.03`。
2. `drift_only_residual` 应与 teacher-only 的 mean feature-FID ratio 相差不超过
   `0.03`，且明显优于 offpath-only。
3. 三者 high-time teacher nonzero MSE 应相近；若 endpoint 分离而 teacher 指标
   不分离，就进一步证明 off-manifold supervision 改变的是 vector-field extension，
   不是 teacher fit。
4. 所有 variant 的 group-0 ratio 仍必须为 `1.000`、投影误差 `<1e-5`。

若命中，当前 off-path 处理原则修正为：没有 oracle conditional field 时，不得把
原 pair target 直接监督 self-generated states；只能使用边缘约束、baseline trust
region 或真正 rollout-level distribution objective。P23 是机制定位，不是方法 gate。

### P24：validation-selected residual trust region 能否获得绝对质量收益

这组预测写于 P23 完成后、首次评估任何非单位 residual scale 之前。P23 正式结果
命中：DCT offpath-only FID ratio 为 `1.131`，比 teacher-only `1.009` 差
`0.122`，与 combined `1.147` 只差 `0.016`；drift-only 为 `1.034`，与
teacher-only 只差 `0.025`。因此错误 paired off-manifold target 已被定位，不能
继续作为方法。P22 teacher-only 在五 seed 中两次优于 baseline、均值接近 `1`，
提示 residual correction 可能具有局部正 leverage，但单位强度过大或不稳定。

固定无训练实验：

- 只使用 P22 FashionMNIST DCT `teacher_residual` adapters，不更新任何参数。
- 对 residual field 使用固定 scale grid `{0, 0.25, 0.5, 0.75, 1.0}`；baseline
  field 始终完整保留，group 0 仍严格不变。
- 原 P22 已看过的 1,024 个 official-test indices 全部排除。沿同一固定 permutation
  取接下来的 1,024 张作为 validation，再取 1,024 张作为 final test；两组索引
  不重叠。validation 与 test 使用不同、预先固定的新 Gaussian noise/direction seed。
- 每个训练 seed 仅按 validation feature FID 选择 scale；并列时选择更小 scale。
  final test 只报告一次，不根据 test 改 scale。

冻结预测：

1. 至少 `3/5` seed 的 validation 最优 scale 应位于 `0.25-0.75`，而不是退回
   `0`；否则 residual detail correction 没有可复现的正 endpoint leverage。
2. validation-selected scale 在独立 final test 上 mean feature-FID ratio 应
   `<=0.98`，至少 `4/5` seed 不差于 baseline。
3. final-test mean feature-SWD 和 latent-SWD ratio 都应 `<=1.02`，避免只针对
   一个 classifier metric 过拟合。
4. selected scale 应优于同一新 test 上的 full-scale adapter；否则 P22 的问题
   不是 trust-region overshoot。

只有第 1-3 条全部命中，才能说 protected residual + validation trust region 在
小图像上提高了生成质量，并进入 MNIST 复现。若 validation 选择非零但 final test
不能泛化，则 endpoint proxy 方差过大；若多数选择 0，则“降低 teacher detail
error”没有正生成价值，本方向停止，不进入 RAE。

## 预先声明的否定条件

以下任一情况出现，都不能说“机制已经完全说清楚”：

1. exact-oracle toy 中 endpoint reversal 与 off-path slope 没有稳定共现。
2. 预测只在一个 seed、一个 gamma 或一个 DCT 基底成立。
3. teacher、rollout 和 endpoint 指标的关系需要每个数据集重新挑时间窗才能成立。
4. shared-state field error 不预测 endpoint，只有事后 hard splice 能解释结果。
5. 低维和小图像结果与 RAE 的主要损伤频带或时间段方向相反，且没有统一的状态反馈解释。
6. 训练侧 coarse protection 与 split-path 都不能按 P13/P14 预测救回 endpoint。

## 本轮完成标准

第一轮 exact-oracle 配置固定为：8 维默认 Gaussian mixture，`mlp` 与
`mini_dit`，hidden size `24/96`、depth `2`、gamma `0/0.5`、batch size
`128`、`800` updates、seed `0-4`。teacher 与 endpoint 各用 `4096` 个
独立样本，Heun `80` 步，并用 oracle `80/160` 步同噪声差异审计 solver。
target times 固定为 `0.9/0.7/0.5/0.3/0.1`，插值系数固定为
`0/0.25/0.5/0.75/1`。在看到结果前不因效果大小修改这些设置。

本轮研究只在以下证据同时存在时结束：

1. 预测文件先于新结果落盘。
2. exact-oracle toy 至少 5 个 paired seeds。
3. teacher、rollout、插值状态和 endpoint 使用独立随机样本。
4. oracle rollout 与 step convergence 证明 solver floor 足够低。
5. 报告每条预测命中、部分命中或失败，不删除反例。
6. 根据结果给出一个比 H2 更精确的机制，或明确判定 H2 不成立。

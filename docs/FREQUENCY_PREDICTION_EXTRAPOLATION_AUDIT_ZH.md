# 频率外推与预测目标外推：代码、实验和研究判断

更新时间：2026-08-08

## 1. 当前结论

目前有两条相关但不能混为一谈的路线：

1. **频率外推**：在同一个 RAEv2 clean predictor 周围，用可控模糊构造一个局部响应方向，再沿负模糊时间外推。小规模实验已经证明这个方向数值上稳定、相邻尺度一致，而且不同于 `full-base` internal-guidance 方向；但它没有对齐逐样本 clean target 误差。13 个条件、每个 5000 张图的正式闭环采样已在 49/52 个 rank shard 处安全暂停，最终必须以 FID/KID 和路径分布指标判断，不能依据 teacher-forced MSE 提前宣布有效。
2. **预测目标外推**：分别训练直接输出 `x`、`v` 和 `epsilon` 的网络，把三者都换算成 clean estimate，再研究它们的差值是否提供有用方向。v4 弯曲流形容量扫描和原始 v3 `ReLU` 精确复跑均已完成。两套四种子结果一致显示：**正 `gamma` 的真正外推恶化完整 rollout 分布；负 `gamma`，即从 `x` predictor 向 `v/epsilon` predictor 插值，有时改善 SWD/MMD。**插值的流形误差取决于 regime：大幅插值常泄漏到流形外，但 `D=16` 的小幅 `x-v` 插值没有泄漏。由于该设置下 `v` baseline 的 intrinsic SWD 本来就优于 `x`，这更像两个有不同闭环偏差的 estimator 之间的校准，不是“外推找到更准确的数据流形”。

最重要的理论修正是：在相同的平方 `v-loss` 和足够模型容量下，三种参数化换算后的最优 clean estimate 都是

\[
\mathbb E[x\mid x_t,t].
\]

因此，`x/v/epsilon` 之间的差值不是三种真实动力学的差值，而是**有限容量、优化难度、参数化条件数和递归采样误差所产生的 estimator gap**。实验真正研究的是：这个有限模型差值能否成为有用的推理时控制方向。

## 2. 最近提交做了什么

### `a767f17`：latent mismatch 诊断

- 对 RAEv2 的 official IG scale 做逐点误差、latent 分布、decoder feature 和图像 FID 的统一扫描。
- 不同指标的最优 scale 明显不同：逐样本 clean error 在 `scale=0.92/1.0` 附近最好，projected latent SWD/FID 在 `1.2` 最好，解码后相对真实图像的 gFID 在 `1.5` 最好，相对 clean reconstruction 的 FID 在 `1.78` 最好，decoder 中间特征 FID 则在 `1.85` 最好。
- 这说明不能用一个 latent 欧氏距离替代最终生成质量，也为之后研究“有用偏差”提供了直接动机。

### `9694eb9`：IG 时间阶段消融

- 把 IG 的作用拆成 early/middle/late 区间及其组合。
- early 区间的终点杠杆很高，但不同区间存在明显非线性交互；全程收益不是各窗口收益的简单相加。
- 这排除了“只看某一步方向大小就能预测最终质量”的简单解释。

### `788efeb`：IG 频谱诊断

- 将 `full-base` gap 分频段分析，并尝试 static 和 learned spectral controller。
- 1000 样本结果中，scalar IG 的 FID 为 `40.987`，static spectral 为 `40.988`，learned spectral 为 `41.160`；没有证据说明更复杂的频段控制优于统一 scale。
- 这条负结果很重要：频率分解本身不是方法，必须找到一个与闭环质量有关的新方向。

### `f08faef`：输入频率轴

- 不再切分 IG gap，而是主动模糊当前 clean prediction，并重新查询 frozen predictor，构造其输入频率响应。
- 小规模结果：预测轴在相邻模糊尺度上的 cosine 约 `0.925`，不同估计方式得到的轴 cosine 约 `0.883`，说明轴本身稳定；它和 `full-base` gap cosine 约 `0.294`，说明不是旧 IG 的简单重写。
- 但它与真实逐样本 paired residual 的 cosine 约 `-0.003`，正向局部外推平均不降低 teacher MSE。它可能影响分布质量，但不是逐点监督纠错方向。

### `26f9303`：预测目标外推

- 复现 JiT 风格的低维数据嵌入高维空间 toy，分别训练 `x/v/epsilon` 输出头。
- v3 初步显示输出参数化在高维、有限容量时差异巨大，但不同条件曾使用不同 metric seed，微小差值不能作同随机性分布比较。
- 当前 v4 已改为共享参考样本、共享 SWD 投影、共享采样噪声，并加入弯曲流形、切向/法向分解、多宽度、checkpoint 和断点续跑。

## 3. 预测目标实验到底在预测什么

仓库采用

\[
x_t=(1-t)x+t\epsilon,
\]

其中 `t=0` 是干净数据，`t=1` 是高斯噪声。它与 JiT 论文的时间方向相反，但只是变量替换，不改变问题。

三个网络直接输出：

- `x-head`：干净样本 `x`；
- `v-head`：速度 `v=epsilon-x`；
- `epsilon-head`：噪声 `epsilon`。

代码把它们统一换算成 clean prediction：

\[
\hat x_x=f_x(x_t,t),
\]

\[
\hat x_v=x_t-t f_v(x_t,t),
\]

\[
\hat x_\epsilon=\frac{x_t-t f_\epsilon(x_t,t)}{1-t}.
\]

所有网络统一使用速度空间平方损失。此时三个 loss 分别等价于：

\[
L_x=\frac{\|f_x-x\|^2}{t^2},\qquad
L_v=\|f_v-(\epsilon-x)\|^2,\qquad
L_\epsilon=\frac{\|f_\epsilon-\epsilon\|^2}{(1-t)^2}.
\]

所以虽然监督批次、损失空间和最优 clean estimate 相同，参数化给不同时间区间施加了不同数值条件：

- `x` 输出在小 `t` 处受到很强权重，并且输出目标本身位于低维数据流形；
- `epsilon` 输出在接近噪声端时条件很差，必须精确抵消高维噪声才能恢复 `x`；
- `v` 介于两者之间，输出目标占据环境空间，但 clean conversion 没有 `1/(1-t)` 的直接爆炸。

这正是 JiT toy 想说明的核心：**相同理论最优解不代表有限网络同样容易学。**这不等于“epsilon 的理想终点分布应该留在全空间”；理想情况下三者仍应生成相同的数据分布。

## 4. “外推”符号必须固定

当前 `x-v` 方向定义为

\[
g_{xv}=\hat x_x-\hat x_v,
\]

推理时使用

\[
\hat x_\gamma=\hat x_x+\gamma g_{xv}.
\]

因此：

- `gamma > 0`：从 `v` 指向 `x` 后继续向外走，是真正的预测目标**外推**；
- `gamma < 0`：`(1-|gamma|) x + |gamma| v`，是从 `x` 向 `v` 的**插值**；
- `x-epsilon` 完全同理。

以后报告必须同时写出符号和“interpolation/extrapolation”，不能只写一个容易误读的 `gamma`。

## 5. 如何判断“是否改进”

这里必须区分三层指标。它们回答不同问题，不能互相替代。

### 5.1 局部预测层：只解释机制，不判定生成成功

- teacher-forced clean/velocity MSE：给定训练路径上的真实 `x_t`，预测是否更接近与它配对的真实 `x` 或 `v`；
- gap/residual cosine：外推方向是否与逐样本剩余误差同向；
- tangent/normal 分解：方向主要沿数据流形移动，还是把样本推离流形。

这层能解释“方向在做什么”，但不能决定 rollout 后的生成分布是否更好。RAEv2 已经直接证明：逐样本误差最优点并不等于 latent 分布或解码图像的最优点。

### 5.2 闭环 latent 分布层：当前 toy 的主要判据

每个候选 `gamma` 都在 ODE 的每一步递归使用，并从相同初始噪声完整 rollout 到终点。当前主要比较：

- intrinsic SWD：生成样本在已知二维内在坐标中的分布与真实分布有多远，越低越好；
- fixed-bandwidth MMD：使用另一类分布距离交叉验证 SWD，越低越好；
- manifold consistency RMS：生成点离已知完整流形有多远，越低越好。

所谓“同随机性”是指不同条件共享初始噪声、真实参考样本、SWD 投影、MMD 带宽和 bootstrap 重采样索引，从而降低条件差值的 Monte Carlo 噪声。它不是“用逐样本配对误差判断生成质量”。

SWD 也不是单独的真值：它只检查流形内的二维分布，因此必须与 MMD 和 manifold RMS 一起报告。只有 SWD 改善而 manifold RMS 明显恶化时，只能说投影分布变好，不能说完整生成分布变好。

### 5.3 解码图像层：真实 RAEv2 的最终判据

真实 latent 模型最终必须以相同采样协议下的 decoded FID/KID、precision/recall 和跨 seed 稳定性判断。latent SWD 可以用于筛选和解释，却不能代替 decoder 后的图像指标。已有 official scale sweep 的最优点分离正是这一原则的直接证据。

因此，当前 toy 的成功门槛是“完整 rollout 后 SWD 与 MMD 跨 seed 改善，同时没有明显的流形外泄漏”；若以后迁移到 RAEv2，还必须再通过 decoded image 指标，不能因为 toy SWD 下降就宣布方法有效。

## 6. 已有预测目标结果

### 6.1 v3 提供的方向性线索

- `D=2/8` 时，`v` 可以不差于 `x`；
- `D=512` 时，`x` 的 SWD 约 `0.032`，`v` 约 `0.303`，`epsilon` 约 `39.2`，JiT 的高维参数化断层被明显复现；
- 但 v3 的条件间 metric seed 不完全共享，不能依据很小的 SWD 差异判断外推是否有效。

本轮直接读取 v3 已保存的 10000 个 intrinsic samples，用同一个新 reference、512 个固定投影、固定 MMD bandwidth 和同随机性 bootstrap 重算了 `D=16`：

| 条件 | 原报告 SWD | 同随机性 SWD | 相对同随机性 `x` | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| `x` | 0.02514 | 0.030798 | 0 | - |
| `x-v, gamma=+0.1` | 0.02459 | 0.030605 | -0.000193 | [-0.000986, 0.001390] |
| `x-epsilon, gamma=+0.1` | 0.02181 | 0.031392 | +0.000594 | [-0.003343, 0.004542] |

因此原先最醒目的 `x-epsilon` 13.2% 改善没有通过同随机性分布复核；`x-v` 只剩约 0.6% 的点估计优势，而且置信区间跨 0。`D=16` 仍可作为多种子候选，但不能再称为已有正证据。

同一协议重算 `D=2/8/512` 后也没有发现可信的正外推收益：`D=8` 的正 `x-v` 从 `gamma=0.1` 起稳定变差，`D=512` 明显变差；`D=2` 的个别极小点估计变化均落在 bootstrap 不确定性内。换言之，v3 保存结果在同随机性分布重评估后支持 JiT 参数化断层，但不再提供统计上可信的 prediction extrapolation 正证据。

### 6.2 v4 线性流形，一种子容量扫描

下表统一为 `D=512`、`curvature=0`、`unit_rms`、共同 `v-loss`。SWD 越低越好；manifold RMS 越低表示越接近已知二维流形。

| hidden | 条件 | SWD | manifold RMS |
|---:|---|---:|---:|
| 64 | `x` | 0.088798 | 0.003426 |
| 64 | `xeps, gamma=-0.03` | 0.083257 | 0.003542 |
| 64 | `xeps, gamma=+0.03` | 0.095787 | 0.003462 |
| 256 | `x` | 0.056472 | 0.001890 |
| 256 | `xeps, gamma=-0.03` | 0.055244 | 0.002064 |
| 256 | `xeps, gamma=+0.03` | 0.060885 | 0.001944 |
| 512 | `x` | 0.047410 | 0.001354 |
| 512 | `xeps, gamma=-0.03` | 0.037468 | 0.001558 |
| 512 | `xeps, gamma=+0.03` | 0.062396 | 0.001443 |
| 1024 | `x` | 0.056670 | 0.001594 |
| 1024 | `xeps, gamma=-0.03` | 0.056614 | 0.001759 |
| 1024 | `xeps, gamma=+0.03` | 0.056699 | 0.001697 |

目前最稳妥的解释是：

1. `x` predictor 很擅长把样本压回低维流形，但有限容量时可能有均值收缩或覆盖不足；
2. `v/epsilon` predictor 含有更多法向误差，单独采样可能很差；
3. 向它们少量插值会给 `x` 结果补回一些分散度，因此流形内 SWD 可能改善；
4. 同时法向噪声也被带回，manifold RMS 稳定恶化；
5. 网络足够宽时差值和收益趋近消失，符合“有限 estimator gap”而非新 population vector field 的解释。

这不是正向外推成功。真正的正 `gamma` 在绝大多数已完成设置中恶化，尤其 `H=512` 的 `x-epsilon` 正向外推使 SWD 从 `0.0474` 变成 `0.0624`。

### 6.3 v4 弯曲流形，四种子 `D=512/H=256`

弯曲流形让 clean data 的内在维数仍为 2，但固定线性 span 可以接近 512，因此 direct `x` predictor 也不再能只靠二维输出子空间解决问题。四个独立 seed 的结果为：

| 方向 | gamma | 平均 SWD 相对 `x` | 改善 seed 比例 | 平均 manifold RMS 变化 |
|---|---:|---:|---:|---:|
| `x-v` 插值 | -0.01 | -0.19% | 3/4 | -0.000023 |
| `x-v` 插值 | -0.03 | -0.74% | 3/4 | -0.000061 |
| `x-v` 插值 | -0.10 | -1.29% | 3/4 | +0.000326 |
| `x-v` 外推 | +0.01 | +0.23% | 0/4 | +0.000038 |
| `x-v` 外推 | +0.03 | +0.83% | 0/4 | +0.000129 |
| `x-v` 外推 | +0.10 | +3.32% | 0/4 | +0.000473 |
| `x-epsilon` 外推 | +0.03 | +4.21% | 0/4 | -0.000242 |
| `x-epsilon` 外推 | +0.10 | +28.23% | 0/4 | -0.000777 |

正 `x-v` 外推随 `gamma` 增大呈稳定恶化，且 4/4 seed 同号；正 `x-epsilon` 外推虽然把样本拉得更接近已知流形，却同时让 intrinsic SWD 明显变差，说明“更靠近流形”也不等于“流形上的分布更正确”。负 `x-v` 插值有很小的点估计收益，但各 seed 的 bootstrap 区间仍跨 0，暂时只应称为弱趋势。

teacher-forced 诊断与生成结果一致：`x-v` gap 和大 `x` predictor residual 的平均 cosine 在 `t=0.1/0.3/0.5/0.7/0.9` 分别约为 `-0.065/-0.034/-0.040/-0.050/-0.068`。也就是说，即使让 `x` 自身 capacity-limited，继续沿 `x-v` 正方向仍没有对齐 `x` 的剩余误差。

### 6.4 新四种子 `D=16` 复验

新复验使用 `D=16`、`hidden=256`、30000-step、每条件 5000 个完整 rollout 样本；所有条件共享初始噪声、参考样本、SWD 投影、MMD 带宽和 bootstrap 重采样。结果非常稳定：

| 条件 | 平均 SWD 相对 `x` | SWD 改善 seed | MMD 改善 seed | 联合通过 seed |
|---|---:|---:|---:|---:|
| `x-v, gamma=-0.03` 插值 | -0.87% | 4/4 | 4/4 | 4/4 |
| `x-v, gamma=+0.03` 外推 | +0.89% | 0/4 | 0/4 | 0/4 |
| `x-epsilon, gamma=-0.03` 插值 | -1.52% | 4/4 | 4/4 | 2/4 |
| `x-epsilon, gamma=+0.03` 外推 | +1.56% | 0/4 | 0/4 | 0/4 |

这里“联合通过”要求 SWD bootstrap 95% 区间确认改善、MMD 同向下降，并且 manifold RMS 不超过 `x` baseline 的 1.1 倍。所有 `gamma>0` 候选从 `0.01` 到 `0.5` 都在 4/4 seed 上恶化 SWD/MMD，且逐 seed bootstrap 区间在最小正步长时也已位于 0 以上。因此旧的“D=16 可能存在正外推甜点区”不再成立。

但负插值的解释也必须克制：本轮单独 `v` 的平均 SWD 为 `0.0587`，本来就优于 `x` 的 `0.0708`。向 `v` 插值改善分布，不足以证明存在超越两个 predictor 的新 guidance 方向。它说明两个同 population optimum 的有限估计器具有不同的 rollout bias，混合可以校准闭环动力学。

更重要的是，这个 setting 不具备“从较弱 `v` 经过较强 `x` 再向外走”的顺序。因为 `v` 的 rollout 分布本来就好于 `x`，`x+gamma(x-v)` 在 `gamma>0` 时实际是在从更好的 `v` 穿过更差的 `x` 后继续远离；它的恶化不能作为 strong-over-weak extrapolation 的关键反例。

这组结果还直接复现了“局部误差与闭环分布不等价”：在 `t=0.1/0.3/0.5/0.7/0.9`，`x` 的逐样本 MSE 均略低于 `v`，分别约为 `0.001363/0.015999/0.040778/0.056556/0.061628`，而 `v` 为 `0.001365/0.016011/0.040833/0.056768/0.062474`；但完整 rollout 后反而是 `v` 的 intrinsic SWD 更低。这里判定插值改善所依据的是终点 SWD/MMD 与流形闭合，不是这组 teacher-forced MSE。

这次复验使用 v4 的 `SiLU` MLP 和新的随机种子派生方式；旧 v3 使用 `ReLU`。因此它是同问题、同随机性指标下的多 seed 复验，不是逐位复现。原始 v3 代码路径的精确多 seed 对照见下一节。

### 6.5 原始 v3 `ReLU` 路径四种子终验

为排除 v4 的 `SiLU`、新种子派生和新实现路径造成结论偏移，本轮直接使用提交 `26f9303` 的原始 `ReLU` 网络、30000-step 训练、200-step ODE、每条件 10000 个样本，并在采样结束后统一用共享参考、共享 SWD 投影、固定 MMD 带宽和同一组 bootstrap 索引重评估。

| 条件 | 平均 SWD 相对 `x` | SWD 改善 seed | bootstrap 确认改善 | MMD 改善 seed |
|---|---:|---:|---:|---:|
| `x-v, gamma=+0.01` | +0.78% | 0/4 | 0/4 | 1/4 |
| `x-v, gamma=+0.03` | +2.34% | 0/4 | 0/4 | 0/4 |
| `x-v, gamma=+0.10` | +7.88% | 0/4 | 0/4 | 0/4 |
| `x-epsilon, gamma=+0.01` | +1.99% | 0/4 | 0/4 | 1/4 |
| `x-epsilon, gamma=+0.03` | +6.52% | 0/4 | 0/4 | 1/4 |
| `x-epsilon, gamma=+0.10` | +25.54% | 0/4 | 0/4 | 1/4 |

即使最小的正外推步长也没有一个 seed 的 SWD 优于 `x`；步长增加后恶化近似单调。原先单 seed、不同 metric randomness 下看到的 `D=16` 正外推改善，已经被“旧保存样本重评估 + v4 多 seed + 原始 v3 路径多 seed”三重证据否定。激活函数和实现路径不是结论翻转的原因。

与 v4 `SiLU` 的 `D=16` 不同，这组原始 v3 `ReLU` 复跑中 `x` 的 rollout SWD 在 4/4 seed 上优于 `v`，所以强弱顺序是正确的；但它仍是固定二维线性子空间，direct `x` predictor 很容易把法向分量几乎清零。当前保存样本只包含 intrinsic 坐标，不能重新精确恢复 endpoint off-manifold RMS。因此它说明“仅有 x 优于 v 仍不足够”，但没有覆盖“x 优于 v 且自身仍明显离开弯曲数据流形”的目标窗口。

### 6.6 弯曲高维流形的容量扫描

`D=512/curvature=0.5` 的四种子容量扫描已经完整结束。随着 hidden width 增大，三个参数化之间的有限估计器差值明显缩小：

| hidden | 平均 `x-v` gap RMS | `x-v, gamma=-0.1` SWD | 严格联合通过 | `x-v, gamma=+0.1` SWD |
|---:|---:|---:|---:|---:|
| 64 | 0.501 | -6.03% | 4/4 | +6.34% |
| 256 | 0.465 | -1.29% | 0/4 | +3.32% |
| 1024 | 0.250 | -1.13% | 0/4 | +0.67% |

`H=64` 的小模型上，向 `v` 插值确实同时改善 SWD、MMD 和流形误差；正向外推则显著恶化。到 `H=256/1024`，点估计变化更小且不再通过 bootstrap/MMD/流形联合门槛。`H=1024, gamma=+0.03` 有约 `0.48%` 的 SWD 点估计改善，但 bootstrap 区间跨 0，MMD 只有 1/4 seed 同向，不能视为正外推证据。

这一容量依赖支持“有限估计器校准”而非“新 population vector field”：模型越受限，`x/v` 的递归 rollout bias 差异越大，凸混合越可能有用；容量增加后差值与可靠收益一起消失。

### 6.7 强弱交叉带与越过 `v` 的对称终验

为避免把 `x` 较差、`v` 较好的 setting 用于“越过较强 `x`”的错误检验，新增了 `D=512/curvature=0.5`、`H=96/128/160/192/224` 的四种子扫描。基线强弱顺序随容量发生了预期翻转：

| hidden | `x` SWD | `v` SWD | `x` 优于 `v` 的 seed | `x` manifold RMS |
|---:|---:|---:|---:|---:|
| 96 | 0.05368 | 0.04368 | 1/4 | 0.03837 |
| 128 | 0.04674 | 0.05442 | 3/4 | 0.02441 |
| 160 | 0.04522 | 0.06171 | 3/4 | 0.01907 |
| 192 | 0.03456 | 0.04190 | 3/4 | 0.01386 |
| 224 | 0.02896 | 0.04420 | 4/4 | 0.01232 |

`H=224` 最符合“`x` 较强但仍有非零终点缺陷”的要求。然而最小正外推 `gamma=0.003` 的平均 SWD 只变化 `-0.025%`，仅 2/4 seed 同号，0/4 seed 的 bootstrap 置信区间确认改善；更大步长没有形成稳定收益。`H=128/160/192` 同样没有任何正步长通过跨 seed、bootstrap、MMD 和流形联合门槛。因此，补齐强弱交叉区后仍没有可靠的 beyond-`x` 证据。

对称实验在 v4 `D=16/H=256` 中把较好的 `v` 当 strong predictor，测试

\[
v+\alpha(v-x),\qquad \alpha>0.
\]

四种子、每条件 10000 样本、200-step rollout 的结果为：

| `alpha` | 平均 SWD 相对 `v` | SWD 改善 seed | bootstrap 确认 | MMD 改善 seed |
|---:|---:|---:|---:|---:|
| 0.003 | +0.027% | 1/4 | 0/4 | 0/4 |
| 0.010 | +0.093% | 1/4 | 0/4 | 0/4 |
| 0.030 | +0.303% | 1/4 | 0/4 | 0/4 |
| 0.100 | +1.342% | 1/4 | 0/4 | 0/4 |
| 0.250 | +5.671% | 1/4 | 0/4 | 0/4 |

`lambda=0.75` 的插值点与 `v` 形成一个几乎水平的平台，但差异没有通过 bootstrap；一旦真正越过 `v`，平均 SWD、MMD 和 manifold RMS 均朝更差方向变化，且随步长总体增大。由此可见，先前“拿错 strong predictor”确实是设计问题；纠正后，两个方向仍都没有显示稳定外推收益。

### 6.8 对外部分析的客观评价

外部分析中最可靠、应保留的部分是：

- 在线性流形 toy 中，最后一层 `R^hidden -> R^D` 确实把所有输出限制在一个至多 `hidden` 维的固定仿射子空间；当 `D-2 > hidden` 时，`v/epsilon` 无法完整传递所有正交噪声方向，而 `x` 只需表示二维 clean 子空间。
- D=512 的 `v/epsilon` clean error 几乎完全由 normal component 主导，因此 `epsilon -> v -> x` 的共线性主要是“逐渐删除同一批 normal error”，不是“逐渐逼近 x predictor 的剩余 intrinsic error”。
- “有序 axis”不是 guidance 有效的充分条件；还必须检查 axis 是否与 strong predictor 的剩余误差或闭环分布缺陷有关。
- 统一 `v-loss` 不等于优化条件完全相同；`x` 和 `epsilon` 的 direct-output error 分别受到 `1/t^2` 和 `1/(1-t)^2` 权重。

需要修正的部分是：

- `D=16` 的 13.2% 外推改善来自不同 metric 随机性的结果，同随机性重算后消失；不能把它作为已确认的“中间甜点区”。
- AutoGuidance 的 weak model 与 prediction target predictor 只有类比关系。前者通常是同目标、不同容量或训练程度；后者还混入参数化条件数和 target geometry，不能直接共享理论。
- “x 已到流形，所以 beyond-x 必然失败”只在线性子空间 toy 中近似成立。弯曲流形的固定线性 span 可以很高，`x` predictor 本身也可能 under-capacity；这正是当前 curvature/capacity sweep 的目的。
- `D=512, hidden=256` 的最后一层 rank bottleneck 是这个 MLP toy 特有的强机制。在真实 DiT/UNet 中，hidden width、每 token 输出维度和跨 token mixing 的关系不同，可能不存在同样的硬 rank 下界。真实模型只能检验定性预测，不能直接照搬 toy 的 `D-hidden` 数值阈值。

### 6.9 模型能力是否处在有效区间

当前实验清楚复现了 JiT 的定性现象，但尚未严格证明模型能力落在最适合检验外推的中间区间：

- 在 `D=512` 且 `H<512` 时，plain MLP 的最后 hidden bottleneck 使输出局部秩至多为 `H`。`v/epsilon` clean conversion 需要保留随 `x_t` 变化的全维 identity-like 分量，因此这些模型对 `v` 存在硬性欠表达；不能把 `H=96...256` 当成普通的“适度弱”。
- `H=1024` 虽消除了 `H<D` 的硬瓶颈，但 seed 20260817 的 v-loss 在 8000 到 15000 step 仍从约 `0.52` 降到 `0.37`，没有充分平台化；其 `v` endpoint manifold RMS 仍为 `0.418`，而 `x` 为 `0.009`。
- 所有 `D=512` 宽度的 `x-v` gap 约有 `99%` RMS 位于法向分量，说明 weak/strong 差值主要仍是 `v` 的无关 ambient error，而不是 strong `x` 尚未修正的同类偏差。
- 当前所谓 large `x-oracle` 只是更宽、训练更久的另一个 x-network。它的 rollout SWD 在若干 setting 中反而差于普通 `x`，所以不能作为真实 Bayes oracle 来划分“太强/太弱”。
- plain MLP 没有显式 residual identity path，而真实 Transformer 通常有残差结构。当前 toy 因此可能额外放大了 noisy-target 学习恒等分量的优化困难。

所以现有结果能说明“这些已测有限网络上，静态直线外推无效”，却还不能证明“在训练充分、无硬瓶颈、两者都不灾难的中等能力模型上也必然无效”。严格能力判断应改用可计算的真实 Bayes clean predictor，并同时要求：无硬 rank bottleneck、held-out excess risk 已收敛、`x/v` 都能合理 rollout、strong 仍有显著 excess risk、weak/strong gap 不被法向垃圾主导。

## 7. 频率轴的精确定义

对当前状态 `x_t`，先用 scale-1 full head 得到 clean prediction `y_0`，再对 `y_0` 做高斯模糊 `H_tau`，并构造保持当前噪声不变的反事实状态：

\[
x_t^{(\tau)}=x_t+(1-t)(H_\tau(y_0)-y_0).
\]

重新查询同一 frozen predictor 得到 `y_tau`，定义

\[
g_{freq}=\frac{y_0-y_\tau}{\tau},\qquad \tau=\sigma^2.
\]

最终使用

\[
y_{guided}=y_{model-guided}+\eta g_{freq}.
\]

`eta>0` 近似沿负 blur-time 外推，倾向恢复 predictor 对高频衰减的响应；`eta<0` 倾向平滑。这里的方向经过模型重新评估，所以它不是直接给图像乘一个频域增益，也不是简单锐化滤镜。

正式实验使用自适应 `sigma`，使反事实状态变化约为原状态 RMS 的 `1%`，避免不同时间和样本的有限差分尺度不可比。当前 13 个条件同时包含：

- 无 IG、官方 scalar IG；
- 正负频率 `eta`；
- early/middle/late 时间窗；
- scalar IG 与频率轴相加；
- model-gap 与 frequency response 的双线性交互项。

## 8. 本轮代码审计与修正

### 已验证正确

- toy 时间约定和 `x/v/epsilon -> clean/velocity` 公式做了代数与数值双重检查；
- 人工构造同一个 clean estimate 时，三个参数化的预测目标 gap 为零；
- 三个 predictor 使用相同初始参数、相同样本、相同时间和相同噪声；
- 弯曲流形的 `embed/decode`、Jacobian 及切向/法向正交分解通过单元测试；
- 所有条件共享参考样本、SWD 投影和初始采样噪声；
- `gamma` 的正负标签由测试固定；
- CPU 端完整 smoke、结果写出和 `--resume` 断点续跑均通过；
- 当前测试结果为 `6 passed`，`py_compile`、shell 语法和 `git diff --check` 均通过。

### 已做性能优化

- `x/v/epsilon` 单头条件只执行所需网络；`x-v` 和 `x-epsilon` 只执行两个网络，不再每一步固定跑三个头；
- 生成结果改为逐条件流式统计和落盘，不再把全部高维 ambient samples 同时留在内存；
- 每完成一个条件写 `generation_metrics.partial.csv`，中断后可从已完成条件继续；
- oracle 和 triplet 支持 checkpoint/resume；
- 多种子实验每张 GPU 独立输出，最终统一聚合，避免并发覆盖。

### 仍需谨慎

- v4 中所谓 `oracle` 只是更宽、训练更久的 `x` 网络，不是真实 Bayes oracle；它可用于观察估计器差值，不应作为生成质量真值。
- 线性流形的一种子结果只用于提出假设。主要结论必须由弯曲流形、多种子和同随机性 bootstrap 决定。
- `epsilon` 参数化靠近高噪声端存在 `1/(1-t)` 条件数放大；报告必须同时给训练区间和 conversion clip。
- SWD 只看已知二维 intrinsic distribution，必须与 manifold RMS 配对，不能把“投影后更像”误称为完整分布更好。

## 9. 与已有工作的边界

- [JiT / Back to Basics](https://arxiv.org/html/2511.13720) 已系统说明：相同 `v-loss` 下，直接 `x` 输出在低维流形嵌入高维环境时更容易学习。当前 toy 是在复核并研究有限预测器差值，而不是重新发现这一现象。
- [Dynamic Dual-Output Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Benny_Dynamic_Dual-Output_Diffusion_Models_CVPR_2022_paper.html) 已联合预测 `x0` 和噪声，并学习时间相关混合。
- [k-Diff](https://arxiv.org/html/2601.21419) 已研究连续预测目标，并从内在/环境维数推出中间目标。
- [Self-Consistent Flow](https://arxiv.org/html/2607.12171) 已联合 endpoint 和 velocity 预测，并通过一致性及分段/混合采样利用两者。

因此，“同时预测 `x` 和 `v` 再凸组合”本身已经不新。尚有区分度的问题只剩下：

> 两个独立、有限容量、偏差不同的 predictor 之间，是否存在可跨 seed、跨流形和跨模型泛化的**区间外推**收益？

目前证据偏向否定正向外推，而支持一个更普通的轻微插值/噪声注入权衡。若正式实验继续如此，应停止把它包装成新方法。

## 10. 执行状态

### 频率轴正式闭环

- 模型：官方 RAEv2 DINOv3-L，EMA，fp32；
- 每个条件：5000 个同噪声、同标签样本；
- 共 13 个条件；
- 输出：latent path metrics、pulse response、解码图像 FID/KID/IS 和最终报告；
- 状态：已在完整 rank-shard 边界安全暂停。52 个 `condition × rank` shard 已完成 49 个；rank 1–3 全部完成，rank 0 只剩 3 个 joint 条件。`--resume` 会保留这 49 个 shard，不会重算或覆盖；当前优先让 GPU 运行预测目标验证。

### 预测目标四种子弯曲流形 screen

- Phase A 多 seed 复验：`D=16`、线性流形、`hidden=256`、30000 step、每条件 5000 样本、200-step ODE；
- Phase B 限制 `x` predictor：`D=512`、`curvature=0.5`、`hidden in {64,256,1024}`、15000 step、每条件 3000 样本、100-step ODE；
- 两阶段均使用 4 个 seed、相同初始噪声、固定 SWD projection、固定 MMD bandwidth 和同随机性 bootstrap；
- Phase A 同时覆盖 `gamma=+0.1` 以直接核验旧结果，Phase B 同时覆盖正外推与负插值；
- 多个 `gamma` 轨迹在同一个 GPU batch 中同步积分，数学更新不变；
- 状态：连同 `H=96/128/160/192/224` 交叉带，共 36 个 setting 已全部完成并汇总。强弱顺序约在 `H=128` 翻转；最合格的 `H=224` 仍没有跨 seed/metric 的正外推收益。

### 越过较强 `v` 的对称实验

- 设置：v4 `D=16/H=256` 四种子 checkpoint，每条件 10000 样本、200-step rollout；
- 路径：`x -> v -> v+alpha(v-x)`，以 `v` 作为 SWD/MMD/bootstrap 基线；
- 状态：全部完成。`alpha=0.003` 起平均指标已经轻微恶化，之后随步长总体增大；任何步长均为 0/4 bootstrap、0/4 MMD 确认改善。

### 原始 v3 精确复跑

- 直接使用提交 `26f9303` 的 `ReLU` 代码路径、旧种子派生、30000-step 和 200-step ODE；
- 四个 seed 各使用一张 GPU，每条件 10000 样本；
- 结束后用共享参考、SWD 投影、MMD 带宽和 bootstrap 重评估；
- 状态：四个 seed 已全部完成。最小 `gamma=+0.01` 起，`x-v` 和 `x-epsilon` 的平均 SWD 均恶化，且 0/4 seed 获得点估计改善；原始 v3 架构混杂已排除。

## 11. 预注册式判断标准

### 继续“预测外推”所需证据

在比较任何 `gamma>0` 之前，setting 必须先满足以下资格条件：

1. 完整 rollout 后 `x` baseline 的 SWD/MMD 明确优于 `v`，否则负 `gamma` 的改善只是向更好的 `v` 插值，不能检验“越过较强 x”的外推；
2. `x` 的终点分布仍有可测缺陷，包括非零 SWD/MMD 和高于数值误差的 manifold RMS，否则已没有可供外推修正的空间；
3. `v` 应当是适度较差，而不是其流形误差比 `x` 高几十至上百倍。后者会让 `x-v` gap 被无关 weak-model normal error 主导；
4. 最好还能观察到正小步长使完整 rollout 指标在 `gamma=0` 附近朝改善方向变化，而不是只看 teacher-forced residual cosine。

现有 `D=16/H=256` 和 `D=512/H=64` 不满足第 1 条；`D=512/H=256/1024` 虽然 `x` 的 SWD/MMD 优于 `v` 且 `x` manifold RMS 仍约为 `0.011/0.009`，但 `v/x` manifold RMS 比约为 `85/47`，不满足第 3 条。因此已有容量点没有一个干净覆盖“x 较强、仍不完美、v 适度较弱”的目标窗口。

满足资格条件后，正外推还必须同时满足以下成功标准，才值得移到真实图像模型：

1. 某个 `gamma>0` 条件在至少 `3/4` seed 上优于 `x` baseline；
2. 平均 SWD 改善超过 bootstrap 不确定性；
3. manifold RMS 相对恶化不超过 `10%`，或者存在另一个完整分布指标证明不是法向泄漏；
4. 现象在弯曲流形成立，不依赖线性子空间的特殊投影；
5. 随模型容量增加，收益规律可解释，而不是仅一个宽度偶然最优。

当前已完成实验没有任何 `gamma>0` 条件满足上述门槛。强弱交叉带已经补齐，对称的 beyond-`v` 实验也已完成，两侧均无正证据。因此在当前 toy 与静态线性 predictor-gap 形式下，结论是：

> 不同输出参数化在有限容量下产生不同的递归 rollout bias；凸混合有时可以校准这种偏差，但把差值反向外推没有稳定收益。

这时停止“预测外推”方法线，只保留为 prediction parameterization 的机制结果。

`D=512/curvature=0.5`、`H=96/128/160/192/224` 的四种子交叉带扫描已经完成。它确认了 `x/v` 强弱顺序会随容量翻转，但没有发现翻转后继续越过 strong predictor 的稳定收益。

还需做对称检查：当完整 rollout 中 `v` 优于 `x` 时，真正的 strong-over-weak 外推应写成

\[
v+\alpha(v-x),\qquad \alpha>0.
\]

在当前 `x+gamma(x-v)` 记号下，它对应 `gamma=-(1+alpha)<-1`。实验已覆盖 `lambda=0/0.25/0.5/0.75/1` 的完整轴和 `alpha=0.003/0.01/0.03/0.1/0.25`；结果同样不支持越过较强 `v`。

### 继续“频率外推”所需证据

1. 至少一个预先定义的 `eta/window` 在 5000 样本上稳定优于 `no_ig`；
2. 若声称补充 scalar IG，则必须优于 `scalar_ig`，而不是只优于无 guidance；
3. 改善不能只来自单个 FID 点，KID、precision/recall 或跨种子/配对重采样需给出一致方向；
4. 正负 `eta` 和时间窗应形成可解释响应，而不是 13 个条件中偶然挑中一个。

若频率轴数值稳定但所有闭环图像指标不改善，则它只是一个可辨识的模型响应方向，不是生成质量方向。

## 12. 当前研究判断

这个阶段并非没有发现，但发现与最初直觉不同：

- **预测空间确实改变有限网络的误差几何。**JiT 断层可复现。
- **两个 predictor 的差值确实能改变最终分布。**但严格确认的收益来自凸混合，不是外推，并且集中在容量受限的模型。
- **强弱顺序必须先验证。**早期把 `v` 较好的 setting 用于 beyond-`x` 检验确实不合理；补齐交叉带并对称测试 beyond-`v` 后，无论 `x/v` 谁更强，越过 strong predictor 都没有稳定收益。
- **局部 MSE 更低不保证 rollout 分布更好。**`D=16` 中 `x` 的逐时刻 MSE 略优于 `v`，完整采样的 SWD 排序却相反。
- **更好的 intrinsic SWD 可以与更差或更好的流形闭合同时发生。**方向的作用依赖参数化与容量，必须使用多指标而非固定故事解释。
- **频率轴是稳定且独立的新扰动轴，但未证明是质量方向。**正式闭环结果是它的生死线。

所以基于当前 plain-MLP 与固定训练预算的静态直线外推已经达到停止标准，不应继续在同一实现上增加步长、容量或符号变体。但若要对 prediction-target extrapolation 本身下最终结论，还差一个带真实 Bayes oracle、无硬 rank bottleneck且训练到收敛的能力控制实验。频率轴则是另一个尚未闭环的问题：frequency-response extrapolation 是否能改善实际生成，而不只是形成一个漂亮的局部向量。

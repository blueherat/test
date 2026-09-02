# Foresight Fixed Point：CFG 复验与 AutoGuidance 迁移实验

## 1. 结论

本轮研究得到两个方向相反、但彼此一致的结果：

1. 在 ImageNet-100 SiT-S/2 上，Foresight Guidance（FSG）的核心现象可以复现。严格匹配实际模型调用数时，closed CFG-50 的 FID-1K 为 `60.8032`，FSG-40 的 FID-1K 为 `57.8860`；sFID 和 IS 也同时改善。
2. FSG 的固定点算子不能直接迁移到 AutoGuidance（AG）。即使把算子修正为语义上最合理的 `strong forward + weak inverse`，它仍会恶化生成；并且从零开始增大算子松弛系数时，FID 呈干净的单调恶化曲线。

因此，当前最可靠的结论是：

> CFG 与 AG 可以共享“算子、迭代、固定点”的形式语言，但不能共享“该固定点为何高质量”的语义。固定点收敛、strong/weak gap 变小和生成质量提高是三个不同命题。

本轮是配对 FID-1K 机制筛查，不是 FID-50K 正式 benchmark。

## 2. 论文的核心直觉与 taste

论文：[Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b56d827a2b8433517e722e0272c7f464-Abstract-Conference.html)，NeurIPS 2025 Spotlight。官方实现位于 [Ka1b0/Foresight-Guidance](https://github.com/Ka1b0/Foresight-Guidance)。

这篇工作的漂亮之处不只是提出了一个 sampler，而是把 guidance 重新拆成两件事：

1. **Calibration**：先修改当前 latent，使 conditional 与 unconditional 的未来预测更一致。
2. **Denoising**：再使用 reference（unconditional）分支向下一时刻推进。

这个分解把 CFG、CFG++、Z-Sampling、Resampling 放入同一组设计轴：

- 一致性区间有多长；
- 使用什么固定点算子；
- 每次更新多强；
- 每个区间迭代几次。

FSG 的关键设计不是“每一步都多算”，而是把有限额外计算集中到生成早期的少数长区间，并在那里做多轮 fixed-point iteration。官方 NFE-50 配置使用：

```text
step 0:  lookahead 5, K=2
step 5:  lookahead 5, K=2
step 15: lookahead 5, K=1
```

这套设计的 taste 是：先把已有 heuristic 变成可比较的算子设计空间，再用收缩性与预算分配指导 sampler，而不是直接再发明一个无来源的 guidance 项。

## 3. CFG 固定点的数学含义

记区间上的 conditional、unconditional flow map 为

\[
\Phi_C^{b\leftarrow a},\qquad \Phi_U^{b\leftarrow a}.
\]

FSG 的一轮往返可抽象为

\[
T_{CFG}
=
(\Phi_U^{b\leftarrow a})^{-1}
\circ
\Phi_G^{b\leftarrow a},
\]

其中 \(\Phi_G\) 是 guided forward map。若 inverse 有效，则

\[
T_{CFG}(x)=x
\quad\Longrightarrow\quad
\Phi_G^{b\leftarrow a}(x)
=
\Phi_U^{b\leftarrow a}(x).
\]

在单步、prediction 线性组合的情形，非零 guidance scale 可进一步推出 conditional/unconditional prediction 一致。对有限非线性区间，\(\Phi_G=\Phi_U\) 并不纯代数地保证 \(\Phi_C=\Phi_U\)；论文把相应的 fixed-point equivalence 作为其理论假设之一。

在精确 score 的理想化下，

\[
s_C-s_U
=
\nabla_x\log p_t(x\mid c)-\nabla_x\log p_t(x)
=
\nabla_x\log p_t(c\mid x).
\]

所以 CFG 差值至少有一个外部语义量：它提高条件 \(c\) 在当前 latent 上的可辨识性。论文把 conditional/unconditional 未来一致的 latent 称为 golden path。

这里仍然没有无条件的质量定理。\(\nabla\log p(c\mid x)=0\) 也可能是最小值或鞍点；论文的 golden-path 质量含义来自经验观察与额外假设，而不是单靠 fixed-point 等式推出。

## 4. 论文定理真正证明了什么

论文 Theorem 1 在以下类型的假设下工作：

- 状态与预测有界；
- conditional/unconditional predictor 光滑；
- 区间 fixed-point operator 存在正确固定点；
- 算子在所选度量下收缩。

定理控制的是 calibrated trajectory 上的平均 conditional/unconditional prediction gap，并说明在固定迭代预算下，短区间单迭代通常不是最优预算分配。

它没有证明：

\[
\text{gap 下降}
\Longrightarrow
\text{FID 下降}.
\]

本轮 AG 实验正好给出了这个缺口的反例。

## 5. 为什么 AG 只有形式相似

记 strong、weak velocity 为 \(S,W\)。普通 AG 为

\[
G_\gamma
=
S+\gamma(S-W).
\]

局部一步存在两个完全等价的写法：

\[
G_\gamma
=
W+(1+\gamma)(S-W)
=
S-\gamma(W-S).
\]

但长区间往返算子不再唯一：

\[
\Phi_W^{-1}\circ\Phi_{G_\gamma}
\neq
\Phi_S^{-1}\circ\Phi_{G_\gamma}.
\]

这两个算子在 infinitesimal 一步上可以给出同一局部 AG 方向，有限区间内却会产生不同闭环动力学。本轮把两种 reference 都做了对照，两者都明显恶化。

若真正想把“strong 与 weak 的未来一致”定义成 fixed point，语义正确的算子应是

\[
\boxed{
T_{AG}
=
(\Phi_W^{b\leftarrow a})^{-1}
\circ
\Phi_S^{b\leftarrow a}
}
\]

因为

\[
T_{AG}(x)=x
\quad\Longleftrightarrow\quad
\Phi_S^{b\leftarrow a}(x)
=
\Phi_W^{b\leftarrow a}(x).
\]

对短区间 \(h=b-a\)，

\[
T_{AG}(x)-x
=
h[S(x,a)-W(x,a)]+O(h^2),
\]

所以它确实与 AG 的局部 strong-minus-weak 方向一致。本轮的修正版实现使用的就是这个算子，而不是把高强度 AG 再用于远视 forward。

## 6. 为什么 AG 固定点没有质量语义

即便理想化地假设 strong/weak 都对应合法 score，

\[
s_S-s_W
=
\nabla\log\frac{p_S}{p_W}.
\]

那么 \(S=W\) 只表示 strong/weak density ratio 的驻点。它可能是：

- strong 相对 weak 更偏好的局部最大点；
- 局部最小点；
- 鞍点；
- 两个模型都很差但恰好一致的位置；
- weak 在某些区域追上 strong 的位置。

与 CFG 的 \(p(c\mid x)\) 不同，\(p_S/p_W\) 没有外部真实语义。更现实地，有限神经 velocity field 甚至未必对应某个合法 score-induced density，因此 product-density 解释还会更弱。

## 7. 对粘贴分析的修正

粘贴分析中以下部分是正确且重要的：

- fixed-point calibration 是 transport，product density 是 reweighting，两者通常不同；
- 真正的边缘分布由连续 pushforward 复合得到；
- AG 的未来一致固定点只是 hypothesis，不是 theorem；
- fixed-point/transport 语言不要求差值场严格 conservative。

需要收紧的部分有三处。

### 7.1 Product density 只是逐时刻形式代数

\[
s_S+\gamma(s_S-s_W)
=
\nabla\log(p_S^{1+\gamma}p_W^{-\gamma})
\]

在每个固定时刻可以成立，但这些 tilted marginals 未必组成同一个合法 forward/noising path，因此实际 ODE 不保证追踪它们。

### 7.2 “稳定 fixed point 是 ratio 最大点”并不适用于一般 FSG 算子

只有当更新真的是足够小步的 gradient ascent，且场确实为某个势函数梯度时，才能用 Hessian 判定最大点。有限区间的 forward/inverse 复合既可能非保守，也可能因离散误差而改变 fixed point 与稳定性。

### 7.3 Pushforward 公式正确，但本身只是描述

\[
\rho_{k+1}=(D_k\circ T_k^{K_k})_\#\rho_k
\]

准确描述了算法把质量搬到哪里，却不会自动解释“为什么那里更好”。要成为质量理论，还必须把该 pushforward 与真实数据分布或明确质量 functional 联系起来。

## 8. 实验设置

- 模型：ImageNet-100 SiT-S/2。
- strong：`step_00800000.pt`。
- weak：同 target 的 `step_00500000.pt`。
- 权重：EMA。
- 采样：显式 Euler，40 或 50 个 base steps。
- FSG schedule：官方 NFE-50 的 `0:5:2,5:5:2,15:5:1`。
- 样本：1000，所有条件使用相同 noise、label 与 seed。
- 评估：ImageNet-100 validation 5000-reference ADM FID/sFID/IS。
- 结果定位：`/home/zhoushunyu/data/eqvae/imagenet_sit_flow/foresight_fixed_point_v1/fid1k`。

本实现是 linear-flow/SiT analogue，不是 SDXL 原配置的逐数值复现。尤其 CFG scale 与 flow-time 参数化不同。它验证的是固定点设计现象能否跨到现有 SiT 资产，而不是复刻论文表格的绝对数字。

## 9. CFG 复验结果

| 条件 | 实际模型调用 | FID ↓ | sFID ↓ | IS ↑ |
|---|---:|---:|---:|---:|
| CFG closed-40 | 5040 | 61.1162 | 215.6880 | 46.1463 |
| CFG closed-50 | 6300 | 60.8032 | 215.4915 | 46.1819 |
| FSG-40，matched-50 | 6300 | **57.8860** | **210.6319** | **55.0116** |

FSG 与 closed-50 的实际模型调用数完全相同。FID 改善 `2.9172`，约 `4.80%`；sFID 和 IS 同时改善。因此核心收益不能用“只是多算了几次网络”解释。

## 10. AG 迁移结果

### 10.1 直接搬运完整 guided forward

| 条件 | FID ↓ | sFID ↓ | IS ↑ |
|---|---:|---:|---:|
| AG closed，\(\gamma=1\) | 86.7012 | 222.7643 | 30.6226 |
| FSG，weak reference | 137.2288 | 229.5278 | 18.8956 |
| FSG，strong reference | 159.8942 | 239.5855 | 13.2634 |
| AG closed，\(\gamma=3\) | 76.7258 | 221.4389 | 33.3352 |
| FSG，weak reference | 98.7850 | 217.7570 | 26.6316 |
| FSG，strong reference | 111.5790 | 221.9383 | 23.9390 |

两种局部等价分解在长区间都失败，说明 reference 的选择确实改变动力学，但并不存在一个简单的“选对 reference 就能迁移”的答案。

### 10.2 修正为 strong-forward / weak-inverse

机制 n=64 时，算子在三个事件点的局部 L2 比例为约 `0.985–0.991`；迭代 move 与 strong/weak 当前 velocity gap 代理也下降约 `1.5%–8%`。所以它在局部确实表现为弱收缩，并在降低当前实现所记录的一致性代理；这里没有把 velocity gap 偷换成严格的完整未来路径距离。

然而 FID 为：

| 局部 AG | closed FID | 加完整未来一致校准后的 FID |
|---|---:|---:|
| \(\gamma=0\) | 92.5706 | 168.1445 |
| \(\gamma=1\) | 86.7012 | 141.6736 |
| \(\gamma=3\) | 76.7258 | 107.9857 |

这是“fixed-point residual 下降但生成质量恶化”的直接反例。

### 10.3 松弛扫描

定义

\[
x\leftarrow x+\rho[T_{AG}(x)-x].
\]

固定局部 \(\gamma=3\)：

| \(\rho\) | FID ↓ | sFID ↓ | IS ↑ |
|---:|---:|---:|---:|
| 0 | **76.7258** | **221.4389** | **33.3352** |
| 0.05 | 77.8461 | 222.5703 | 33.0680 |
| 0.10 | 79.1428 | 223.9576 | 32.9387 |
| 0.25 | 83.0458 | 227.8056 | 31.8447 |
| 0.50 | 91.9885 | 228.9168 | 28.9403 |
| 1.00 | 107.9857 | 220.7395 | 24.7440 |

FID 从 \(\rho=0\) 开始严格单调恶化，IS 也严格下降。sFID 在 \(\rho=1\) 有非单调波动，但不改变整体结论。

这排除了“固定点目标正确，只是原步长太大”这一最直接解释。至少在当前 same-target v800/v500 pair、当前 early schedule 与 FID-1K 筛查下，朝 strong/weak future consistency 移动本身就是负方向。

## 11. 统一解释

本轮最重要的逻辑链是：

\[
\text{算子局部收缩}
\centernot\Rightarrow
\text{定义的 gap 对质量有意义}
\centernot\Rightarrow
\text{终端分布更接近真实分布}.
\]

FSG 对 CFG 有效，是因为 fixed-point machinery 与 conditional semantics 恰好配合。AG 的 strong-minus-weak 差值在普通 closed-loop 外推中可以有用，但这不表示应把差值消到零；实际上 AutoGuidance 的作用正依赖 strong/weak 的结构化不一致。把未来差异强行收敛到零，会消除或扭曲原本有用的 bias contrast。

所以，AG 的“好点”若存在，不能仅由

\[
\Phi_S(x)=\Phi_W(x)
\]

定义。它必须额外引用终端质量、真实分布、reward 或另一种外部准则。没有这个准则，fixed-point iteration 只是一个数值求解器，它无法告诉我们应该求哪个 fixed point。

## 12. 产物与验证

代码：

- `experiments/foresight_fixed_point_flow.py`
- `experiments/sample_imagenet100_sit_foresight_fixed_point.py`
- `experiments/run_imagenet100_sit_foresight_fixed_point_study.py`
- `tests/test_foresight_fixed_point_flow.py`

便携数据：

- `docs/data/foresight_fixed_point_cfg_ag_fid1k.csv`

验证：

```text
30 tests passed
py_compile passed
git diff --check passed
```

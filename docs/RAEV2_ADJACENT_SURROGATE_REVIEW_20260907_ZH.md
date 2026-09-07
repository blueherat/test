# 极小 readout surrogate 采样加速：终止归档

日期：2026-09-07。状态：**user-rejected；停止研究，不进入 16 轨迹 gate，不推荐变体。**

用户已明确拒绝这个采样加速方向。本文件仅保存中止前完成的反方检查，避免以后重新包装同一思路。没有 GPU 实验、训练或新 sampler 实现；既定 5K 补样不受影响。后续新方向应面向生成能力，而非采样加速。

## 被拒绝的方案

冻结完整 IG 场 \(F\)，以原生 Base velocity 加 rank-8 最终 readout 增量形成便宜场 \(b_\phi\)，令 \(R_\phi=F-b_\phi\)。宏步精确更新残差，微步计算便宜场；沿完整场轨迹训练残差的时间差分，而不是只拟合残差值。拟议增量为 19,712 参数，该数字来自方案说明，本次没有独立核对实现。

## 中止前确认的前作覆盖

- **降低轨迹高阶总时间导数以降低积分成本，已有直接先例。** Kelly 等的 NeurIPS 2020 论文将轨迹总导数的平方范数积分作为训练正则，讨论正则阶数与求解器阶数的关系。因此，不能把“由局部截断误差导出 material-derivative loss”本身宣称为新原则。[Learning differential equations that are easy to solve，§3、§6](https://papers.nips.cc/paper/2020/file/2e255d2d6bf9bb33030246d31f1a79ca-Paper.pdf)
- **便宜 surrogate 加精确残差的多速率结构也已有系统理论。** Roberts 等的 SISC 2022 工作给出了这种加性拆分及多速率构造；局部误差常数涉及 surrogate 误差及其导数。文章同时指出，全场、surrogate 和两者之差都可能带来刚性限制。[A fast time-stepping strategy for dynamical systems equipped with a surrogate model](https://arxiv.org/html/2011.03688)
- **扩散中依据 guidance 时间变化较慢而稀疏更新，已有直接先例。** THG 测量条件预测与额外 guidance 沿轨迹的时间导数，并据此设计多速率方法。冻结完整场、专门训练拆分的差分损失，与以上文献的精确重合程度尚未完成查证；不能据此宣称首创。[Tortoise and Hare Guidance，NeurIPS 2025](https://openreview.net/pdf?id=3cYcUmcDhU)

## 独立推导：局部依据成立，稳定性保证并不成立

以下只是对被拒绝方案的数学审计，不构成重启建议。考虑宏步内将残差固定在起点，并精确积分便宜场：

\[
\dot v=b_\phi(v,t)+R_\phi(z_n,t_n),\qquad v(t_n)=z_n.
\]

若场足够光滑，则完整解与近似解的宏步差满足

\[
z(t_n+H)-v(t_n+H)
=\frac{H^2}{2}(\partial_tR_\phi+J_{R_\phi}F)(z_n,t_n)+O(H^3).
\]

所以沿完整轨迹的残差差分，对应这个一阶构造的领先局部误差项。它不自动代表其他阶数多速率求解器的完整误差，也不控制数值稳定性。

一个精确反例说明问题。令

\[
F(x,y)=(1,-\lambda y),\qquad b(x,y)=(1,Ky),\qquad R=(0,-(K+\lambda)y),
\]

其中 \(K,\lambda>0\)。在训练轨迹 \(y=0\) 上，残差值和 material derivative 均为零；完整场的横向动力学稳定。然而上述宏步即便微积分完全精确，横向更新仍为

\[
y_{n+1}=\left[1-\frac{\lambda}{K}(e^{KH}-1)\right]y_n.
\]

其放大因子可以任意大。故零训练差分损失甚至零训练值损失，都不能推出横向稳定性、真实 rollout 改善或 FID 改善。这个反例限制的是理论保证，并不声称真实 RAEv2 已出现同样故障。

**终止结论：** 方向已被用户拒绝；本归档不安排训练、16 轨迹 gate、FID 或任何延伸变体。前作检索随拒绝中止，未完成全面 novelty 审查。

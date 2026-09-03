# 仿射反事实密度比 Internal Guidance

## 1. 先把已经被实验否掉的解释删除

当前最好条件为

\[
G_{\rm PFR}=W+\beta(S-Q),\qquad \beta=1+\gamma,
\]

其中 `S` 是 v800 完整速度场，`W` 是 depth-4 internal head，`Q` 是同一弱头在
投影反事实查询点上的输出。这个方法不是隐式数值积分：隐式方法只能改变给定向量场的
离散求解误差，不能把 learned field error 自动变成正确场。它也不是一次更准确的
teacher posterior update。held-out 线性 bridge 审计显示，`Q` 对守恒目标
`U=X-E` 的 MSE 在所有测试时间都略差于 `W`，而 PFR 的单步 MSE 也始终差于普通 IG。

因此，接下来唯一诚实的问题是：为什么一个有意构造的、略差但结构兼容的参考场可以
改善最终生成分布？

## 2. 线性 Flow Matching 的仿射闭包

对线性路径

\[
Z_t=(1-t)E+tX,
\]

任意当前时刻边缘 score 与 velocity 满足

\[
s_i(z,t)=\frac{t v_i(z,t)-z}{1-t},
\qquad
v_i(z,t)=\frac zt+\frac{1-t}{t}s_i(z,t).
\]

取任意系数 `c_i`，记 `C=sum_i c_i`。则

\[
\sum_i c_i v_i
=C\frac zt+\frac{1-t}{t}\sum_i c_i s_i.
\]

把这个组合重新解释成当前时间的 score，得到精确恒等式

\[
\boxed{
s_{\rm mix}
=\sum_i c_i s_i+\frac{C-1}{1-t}z.
}
\]

所以只有当

\[
\boxed{\sum_i c_i=1}
\]

时，velocity 的线性组合才严格等于 score 的同系数组合。若系数和不为一，会凭空多出
一个径向场 `(C-1)z/(1-t)`；它对应二次势，且在数据端被 `1/(1-t)` 放大。这个条件
不是审美偏好，而是保持 linear-flow kinematic term 不变的精确闭包条件。

普通 IG、AutoGuidance 与 PFR 都满足它：

\[
G_{\rm IG}=S+\gamma(S-W)
=(1+\gamma)S-\gamma W,
\]

\[
G_{\rm PFR}=W+\beta(S-Q)
=W+\beta S-\beta Q.
\]

两式的系数和都为一。此前把跨时间 velocity change 拆成 parameterization transport
与 score evolution 后分别部署，FID-1K 分别崩到 `170.62` 与 `271.15`；只有两个巨大、
近乎反向的坐标分量完整相消后才恢复 `61.97`。该反例与仿射闭包完全一致：单独的
跨参数化分量保留了不应独立存在的径向项，完整 physical velocity difference 才恢复
正确的共同 kinematic term。

仿射闭包是必要条件，不是质量充分条件。所有 difference-in-differences 对照同样可以
满足系数和为一，却仍然生成较差。

## 3. PFR 是有限强度的三密度比，而不是 Taylor 修正

理想化地假设 `S/W/Q` 分别是同一当前时刻三个正密度 `p_S/p_W/p_Q` 的 score-induced
velocity。由上一节仿射闭包，PFR 对应的 score 为

\[
s_{\rm PFR}=s_W+\beta(s_S-s_Q).
\]

因此

\[
\boxed{
s_{\rm PFR}
=\nabla_z\log\left[
p_W\left(\frac{p_S}{p_Q}\right)^\beta
\right].
}
\]

也就是当前时刻的隐式密度

\[
\boxed{
p_{\rm PFR}(z,t)
\propto
p_W(z,t)\left[
\frac{p_S(z,t)}{p_Q(z,t)}
\right]^\beta.
}
\]

这是代数恒等式，不要求 `gamma` 或 `beta` 很小。普通 IG 是特殊情形 `Q=W`：

\[
p_{\rm IG}\propto p_W(p_S/p_W)^\beta.
\]

所以 PFR 的实质变化不是“多做一次积分”，而是把 density-ratio 的 denominator 从 stale
weak reference `W` 换成一个结构化 counterfactual weak reference `Q`。

和 AutoGuidance 一样，这些逐时间隐式密度通常不构成某个终端密度的合法 forward
diffusion path。因此该等式解释的是每个当前时刻施加了什么有限强度 density tilt，不能
直接推出最终 FID。

## 4. 等价的 KL 正则分布更新

上一式还能从一个精确变分问题推出。定义 counterfactual log-ratio reward

\[
R(z,t)=\log p_S(z,t)-\log p_Q(z,t).
\]

在所有归一化密度 `rho` 上考虑

\[
\boxed{
\rho^*
=\arg\max_\rho
\left\{
\beta\,\mathbb E_\rho[R]
-D_{\rm KL}(\rho\|p_W)
\right\}.
}
\]

第一项奖励 strong 相对 counterfactual weak 更偏好的区域；第二项要求新分布不要任意
离开 weak base。加入归一化拉格朗日乘子并对 `rho` 作变分：

\[
\beta R-\log\rho+\log p_W+\text{constant}=0.
\]

于是唯一闭式解为

\[
\boxed{
\rho^*\propto p_W e^{\beta R}
=p_W(p_S/p_Q)^\beta.
}
\]

这给当前公式一个比 fixed point 或隐式积分更自洽的解释：它是一次有限强度、
KL-regularized 的反事实比率提升。

## 5. 为什么要构造反事实 `Q`

普通 IG 使用 `W` 同时承担两个角色：

1. 它是层级模型的 base transport；
2. 它又是 strong/weak density ratio 的 denominator。

但被放大的 suffix refinement 会改变随后访问的 latent。PFR 保留 `W` 作为 KL base，
只把 ratio denominator 换成弱头在“refinement 已被最小程度写入 latent”后的响应：

\[
C=\beta(S-W),\qquad G=W+C,
\]

\[
a^*=\left[\frac{\langle C,G\rangle}{\|G\|^2}\right]_+,
\qquad
q=(z+h a^*G,t+h),
\qquad Q=W(q).
\]

`a*G` 是正向 sampler ray 中距离 `C` 最近的向量，因此这一步没有新增空间系数。它让
`Q` 成为与当前 weak head、当前样本和当前 refinement 绑定的 structured hard negative。
对应 reward 的直觉是：

> 只放大 strong 相对“已经看过所提 refinement 的 weak reference”仍然保留的优势。

这个解释不要求 `Q` 比 `W` 更准确。相反，AutoGuidance 的 denominator 本来就应是与
strong 错误结构兼容、但更弱的参考。teacher MSE 审计证明当前 `Q` 正是一个更差的参考，
而不是一个 oracle future estimate。

目前尚无定理证明上述最小 forward-ray 查询一定构造出最优 hard negative。它是方法中
仍需跨模型验证的结构假设。

## 6. 多反事实参考的唯一无新强度扩展

若有多个反事实参考 `Q_j`，取固定权重

\[
\pi_j\ge0,\qquad\sum_j\pi_j=1,
\]

并把 reward 定义为平均 log-ratio：

\[
R_\pi=\log p_S-\sum_j\pi_j\log p_{Q_j}.
\]

同一变分问题的解为

\[
\boxed{
\rho_\pi^*
\propto
p_W
\left[
\frac{p_S}{\prod_j p_{Q_j}^{\pi_j}}
\right]^\beta.
}
\]

分母是反事实密度的几何平均，其 score 恰好是参考 score 的凸平均。因此 velocity 实现
无需估计任何 log density：

\[
\boxed{
G_\pi
=W+\beta\left(S-\sum_j\pi_jQ_j\right).
}
\]

当前预注册两个等权、无额外强度的候选：

1. time-only 与 projected counterfactual 的几何平均；
2. 半 horizon 与整 horizon projected counterfactual 的几何平均。

若它们优于单参考 PFR，说明多 negative 减少参考噪声；若更差，说明收益依赖一个足够
强的单一 hard negative，几何平均会稀释 ratio。两种结果都会直接约束理论。

## 7. 已有证据与明确边界

支持：

1. PFR 系数满足 current-time affine closure；
2. 单独破坏跨时间坐标抵消会灾难性失败；
3. `Q` 的 teacher MSE 更差，排除了 posterior-accuracy 故事，却符合 hard-negative 角色；
4. 输出正交化 FID-5K 为 `36.4408`，完整三密度比为 `36.3470`，说明完整 ratio 中沿旧
   IG 轴的部分也有用，不能只保留欧氏“创新”分量。

不能声称：

1. 有限神经 velocity 必然是某个正规化密度的保守 score；
2. 逐时间的 `p_PFR` 组成合法 probability path；
3. KL tilt 自动降低真实数据 KL 或 FID；
4. 当前一个训练 seed、ImageNet-100 SiT-S/2 的内部最优等于跨模型 SOTA；
5. 反事实 ratio 解释已经唯一确定了 `h=1/32`。

因此这套理论解决的是“公式为何是一个合法、有限强度的分布组合，以及哪些组合被禁止”，
尚未完整解决“哪个 counterfactual denominator 必然提高真实生成质量”。后一个问题必须
由多模型因果验证继续收紧。

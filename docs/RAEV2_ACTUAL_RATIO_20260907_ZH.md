# 实际轨迹密度比：已有诊断与固定数据准备

2026-09-07。当前3%目标仍未达成。此前单次共享噪声判别器与一次分类概率校准的1K均失败；这里围绕同一条Discriminator Guidance路线定位一个具体分布假设，不追加温度/时间窗搜索。实际轨迹数据正在准备，尚未训练这个新目标的判别器，也没有它的FID。

## 已经观察到的分布迁移失败

此前训练负例是历史生成终点重新加噪后的 q_bar_t。其共享噪声训练目标具有正确边缘，但 q_bar_t 不自动等于原官方Euler采样的实际 q_native_t。固定64个均匀散布类别、seed202609089、所有99个正信号query time，用同一个已冻结critic作如下对照：

| 分类负例 | real-vs-fake 的 scaled logistic loss |
|---|---:|
| 历史生成终点重新加噪 | −7.511846 |
| 原官方实际轨迹状态 | +3.566283 |

两种计算共用对应真实endpoint、类别、时间及独立于历史endpoint的新初始Gaussian噪声。差为11.078128，按64类聚合后的标准误1.275236。实际轨迹上的值已高于零判别器的0。全部时刻都计入，不选有利区间；没有解码、图像、FID、拟合或参数调整。这是该固定critic在指定诊断上的风险迁移失败，不是最优密度比、所有类别的距离估计或因果分解。完整身份见 [诊断JSON](../experiments/results/raev2_guidance_20260907/paired_ratio_actual_marginal_audit.json)。

## 为什么共享初始噪声仍可消除边界训练奇点

以下推导针对光滑的理想连续场，假设可交换微分、积分，边界项消失；实际BF16场和有限Euler网格仅作数值近似，不宣称满足这些精确假设。

令生成方向的时间 a=1−t。真实前向状态为 z_p=a X_p+t epsilon，原生ODE从同一个epsilon出发，其初始速度为 G(epsilon,1,c)−epsilon。因此

    z_native = epsilon + a [G(epsilon,1,c)−epsilon] + O(a²),
    z_native − z_p = a [G(epsilon,1,c)−X_p] + O(a²).

仍设 logit d=a f，使用此前不变的共享噪声scaled logistic loss。在边界，其总体目标成为

    E[ (1/4) grad f(epsilon)·(G(epsilon,1,c)−X_p) + (1/8) f(epsilon)² ].

对Gaussian epsilon作分部积分，最优边界函数现在是

    f*(epsilon,1,c) = div G(epsilon,1,c)
                     + [E(X_p|c)−G(epsilon,1,c)]·epsilon.

它与之前重新加噪终点所产生的线性均值差边界不同。一个独立的标量Gaussian解析测试已通过：非恒定初始G的精确密度导数与该公式一致，该函数同时使配对边界弱损失的参数梯度小于1e−12。直接从两条连续性方程对 log(p_a/q_native_a) 在 a=0 求导，也得到同一表达式。训练不显式计算高维div G；真实/实际轨迹的配对状态差提供其弱形式信号。有限数据、FP32抵消与有限网格仍需检查。

## score反馈能证明什么，仍缺什么

生成方向的native速度为 U_G=(G−z)/t。若对当前实际 q 有精确 d=log(p/q)，添加

    delta U = (t/a) grad d = t grad f

在原clean坐标中对应 delta G=t² grad f。对时间变化的参考p，KL导数为

    d KL(q||p)/da = E_q[(U_G−U_p)·grad log(q/p)]
                   − (t/a) E_q[||grad log(q/p)||²].

第二项给出清楚的局部耗散结构；第一项仍在，不能因此宣称整体KL单调，更不能宣称FID必然下降。若训练负例只来自冻结的原官方 q_native，而采样时已变成 q_guided，则耗散项变成交叉内积，符号也未必保持。本次实际状态数据仅修正 q_bar 与 q_native 的混淆，没有解决 q_native 与 q_guided 的后续偏移。后者可能需要实际反馈更新；当前尚未启动反馈拟合循环。

这沿用 [Discriminator Guidance](https://proceedings.mlr.press/v202/kim23i.html) 的密度比思想，并用本地连续性方程明确当前迁移的边界。上述共享噪声/初始速度展开为本次局部推导，不归为该论文原公式，不作新颖性声明。

## 已固定、正在执行的数据准备

- 训练5000条，seed202609090，query时间排列seed202609092；验证1000条，seed202609091，时间排列seed202609093。类别为global id mod1000，仍B8。
- 每个全局batch在原100步shift8网格的索引1..99中取得一个预先平衡后shuffle的停止点；保存每个id的这个实际query状态和原始Gaussian epsilon。它是训练数据采样，未定义任何分时guidance参数。
- 原EMA、native BF16 IG1.78及官方区间、FP32 Euler均不变。状态和epsilon都以FP32保存，避免把高噪声端O(a)的配对差异量化掉。不加载decoder，不生成图像，不计算FID。
- 真实endpoint沿用已验证无重叠的5000 train / 1000 validation真实编码数据；配对真实图像与生成初始noise在类内独立。实际状态bank准备完成后才接入新的拟合；旧的最终checkpoint不会被覆盖或重新命名为新目标的模型。
- 每个分片保存source/checkpoint/config SHA、每batch noise/state SHA、准确query index、global ids。合并检查覆盖且无重复；源文件在运行期间冻结。

实现：[状态缓存](../experiments/cache_raev2_actual_ratio_states.py)、[固定执行器](../experiments/prepare_raev2_actual_ratio_bank.py)。数据目录为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/actual_ratio_bank`。该准备本身不是质量突破；下一步是否有效仍需独立分类验证、实际采样与固定FID评测。

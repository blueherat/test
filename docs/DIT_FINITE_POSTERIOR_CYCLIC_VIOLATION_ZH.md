# DiT 有限尺度后验循环单调违反（FPCV）

## 0. 一句话结论

Projected Tweedie-cone Violation（PTCV）只在当前状态的无穷小邻域检查 posterior-mean
Jacobian 是否像一个对称半正定矩阵。它在现有 discovery pool 中对全局模糊有窄信号，
却把两张结构错位图排到了反方向。

下一条结构候选不再叠加轨迹分支，也不再挑一个 Jacobian 分量，而是检查一个严格的
有限尺度事实：

> 精确高斯后验均值是一个凸函数的梯度。因此，在同一状态周围放置一组确定性探针时，
> 每个输入与自己的输出所形成的 identity matching 必须是所有重新指派中的最优解。

若 Hungarian assignment 找到一个更优的交叉重配，learned denoiser 在该有限邻域内就
不可能是精确后验均值。本文把归一化的 assignment regret 称为
**Finite-scale Posterior Cyclic-monotonicity Violation（FPCV）**。

它与视觉错位的关系仍是待证伪假说，不是定理。能严格证明的是 posterior realizability
违反；不能严格证明的是“所有肢体错位必定违反”或“违反就一定生成坏图”。

## 1. 每个字母的含义

- `d`：latent 总维数；DiT ImageNet-256 中 `d=4*32*32=4096`。
- `c`：固定 ImageNet 类别条件。
- `t`：前向扩散噪声时刻。
- `X_0 in R^d`：随机的干净 latent。
- `X_t in R^d`：时刻 `t` 的 noisy latent。
- `alpha_t=sqrt(alpha_bar_t)>0`：前向过程中保留干净信号的系数。
- `sigma_t=sqrt(1-alpha_bar_t)>0`：前向高斯噪声的标准差。
- `epsilon ~ N(0,I_d)`：标准高斯噪声。
- `m_t^c(x)`：精确 raw class-conditional 后验均值
  `E[X_0 | X_t=x,c]`。
- `mhat_t^c(x)`：冻结 DiT 给出的 raw class-conditional、unclipped clean prediction。
- `P_0^c`：类别 `c` 下 clean latent 的概率分布。
- `Z_t^c(x)`：把高斯 likelihood 与 `P_0^c` 积分后的 tilted partition function。
- `Psi_t^c(x)`：其缩放 log-partition potential；精确后验均值等于它的梯度。
- `x_star`：真实 CFG=4 采样轨迹在检查点保存的中心状态。
- `Q=[q_1,...,q_r]`：事前固定的 `d*r` 正交投影基，`Q^T Q=I_r`。
- `r=16`：投影子空间维数。
- `z_i in R^r`：子空间中的第 `i` 个确定性 probe coordinate。
- `x_i=x_star+Q z_i`：实际喂给冻结 DiT 的 probe latent。
- `y_i=Q^T mhat_t^c(x_i)`：probe 输出的投影 clean prediction。
- `n=2r+1=33`：一个半径下的 probe 数。
- `C_ij=y_i^T z_j`：把第 `i` 个输出配给第 `j` 个输入位置的 assignment score。
- `A_id`：保持原输入输出配对时的总 assignment score。
- `A_star`：所有排列中最大的 assignment score。
- `V_h=A_star-A_id`：半径 `h` 下未归一化的 cyclic assignment regret。
- `H=I_n-11^T/n`：对 probe 行去均值的矩阵。
- `D_h`：归一化到 `[0,1]` 的单检查点 FPCV 分数。
- `D_path^(h)`：三个冻结检查点的整条路径 FPCV 分数。

## 2. 从 VP 高斯通道推到凸梯度

### 2.1 前向通道

固定类别 `c`，VP 前向过程写作

\[
X_t=\alpha_tX_0+\sigma_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I_d).
\]

给定 clean latent `x_0`，noisy state `x` 的 likelihood 为

\[
p(x\mid x_0,c)
\propto
\exp\left[-\frac{\lVert x-\alpha_tx_0\rVert^2}{2\sigma_t^2}\right].
\]

展开平方项：

\[
\begin{aligned}
p(x\mid x_0,c)
&\propto
\exp\left(-\frac{\lVert x\rVert^2}{2\sigma_t^2}\right)\\
&\quad\times
\exp\left(
\frac{\alpha_t}{\sigma_t^2}x^\top x_0
-\frac{\alpha_t^2}{2\sigma_t^2}\lVert x_0\rVert^2
\right).
\end{aligned}
\]

第一项与 `x_0` 无关，在 posterior 归一化时会抵消。定义

\[
\mathcal Z_t^c(x)
=
\int
\exp\left(
\frac{\alpha_t}{\sigma_t^2}x^\top x_0
-\frac{\alpha_t^2}{2\sigma_t^2}\lVert x_0\rVert^2
\right)P_0^c(dx_0).
\]

### 2.2 后验均值是 log-partition 的梯度

对 `x` 求导：

\[
\nabla_x\log\mathcal Z_t^c(x)
=\frac{\alpha_t}{\sigma_t^2}
\mathbb E[X_0\mid X_t=x,c].
\]

定义缩放势函数

\[
\Psi_t^c(x)
=\frac{\sigma_t^2}{\alpha_t}\log\mathcal Z_t^c(x).
\]

于是

\[
\boxed{
\nabla_x\Psi_t^c(x)=m_t^c(x)
=\mathbb E[X_0\mid X_t=x,c].
}
\]

`log Z` 是 affine functions 的 log-Laplace/log-sum-exp，因此是凸函数；
`sigma_t^2/alpha_t` 为正数，所以 `Psi_t^c` 仍然凸。由此得到比单点 Jacobian 更完整的
结论：

\[
\boxed{m_t^c=\nabla\Psi_t^c,\qquad \Psi_t^c\text{ 是凸函数}.}
\]

再求一次导数，便回到 PTCV 使用的 Tweedie covariance identity：

\[
\boxed{
\nabla_xm_t^c(x)
=\frac{\alpha_t}{\sigma_t^2}
\operatorname{Cov}(X_0\mid X_t=x,c)
\succeq0.
}
\]

因此理想 map 的 Jacobian 必须对称半正定；更全局地，整个 map 必须循环单调。

### 2.3 DiT 中实际查询什么

精确 conditional epsilon target 满足

\[
\varepsilon_t^{c,*}(x)
=\mathbb E[\varepsilon\mid X_t=x,c]
=\frac{x-\alpha_tm_t^c(x)}{\sigma_t}.
\]

所以 learned surrogate 为

\[
\boxed{
\widehat m_t^c(x)
=\frac{x-\sigma_t\widehat\varepsilon_\theta^c(x,t)}{\alpha_t}.
}
\]

必须查询 raw class-conditional epsilon，并关闭 `clip_denoised`。不能把 CFG 合成后的

\[
\widehat\varepsilon^{\rm cfg}
=\widehat\varepsilon^u+w(\widehat\varepsilon^c-\widehat\varepsilon^u)
\]

代入，因为 `w>1` 的线性外推一般不对应任何真实 conditional posterior mean。

真实生成状态 `x_star` 可以来自 CFG=4 轨迹：上述恒等式对 raw conditional density 支持
中的每个 `x` 都逐点成立，而高斯卷积在有限噪声下具有全空间支持。这里检验的是 raw
conditional map 在 CFG 访问点附近的 realizability，不是 CFG map 自身的后验解释。

## 3. 降到固定 16 维子空间

令

\[
Q=[q_1,\ldots,q_r]\in\mathbb R^{d\times r},
\qquad Q^\top Q=I_r,
\qquad r=16.
\]

在中心状态 `x_star` 周围写

\[
x(z)=x_\star+Qz,\qquad z\in\mathbb R^r.
\]

把凸势限制到这个仿射子空间：

\[
\phi(z)=\Psi_t^c(x_\star+Qz).
\]

其梯度为

\[
g(z)=\nabla_z\phi(z)
=Q^\top m_t^c(x_\star+Qz).
\]

凸函数限制到任意仿射子空间仍然凸，所以 `g` 仍是凸梯度、仍循环单调。这让我们只用
16 维固定查询便能得到一个严格的 projected certificate。低分仍不能证明完整 4096 维
map 合法，因为违反可能落在 `span(Q)` 之外。

本实验复用 PTCV 的同一基，禁止看到标签后另挑空间位置：

- 4 个 normalized Hadamard channel modes；
- 4 个二维 DCT spatial modes `(0,1),(1,0),(1,1),(2,2)`；
- tensor product 后得到 16 个单位正交方向；
- basis raw SHA-256 为
  `698fa3fcf6a67265ccdb618f3d1c6642affd03aa41dbcb5ffce8d6f36529d179`。

这些方向只是固定的低/中频 latent 结构探针，不等于具体的腿、尾巴、果柄或物体边界。

## 4. Cross-polytope probes

在 `R^r` 中固定一组中心对称 probes：

\[
z_0=0,
\]

\[
z_{2a-1}=h e_a,\qquad z_{2a}=-h e_a,
\qquad a=1,\ldots,r,
\]

其中 `e_a` 是第 `a` 个标准基，`h>0` 是 finite probe radius。总数为

\[
n=2r+1=33.
\]

实际网络输入和投影输出为

\[
x_i=x_\star+Qz_i,
\qquad
y_i=Q^\top\widehat m_t^c(x_i).
\]

这里有 33 次模型查询，但没有 33 条生成轨迹：probe 不加随机 innovation、不 rollout、
不解码、不成为候选终点，也不做投票或选枝。它是一条轨迹在一个时刻周围的确定性
finite stencil。

## 5. 循环单调为什么等价于 identity assignment 最优

精确 posterior map 下，`y_i=grad phi(z_i)`。凸性给出任意 `i,j`：

\[
\phi(z_j)\ge
\phi(z_i)+y_i^\top(z_j-z_i).
\]

任取一个排列 `pi`，代入 `j=pi(i)` 并对所有 `i` 相加：

\[
\sum_i\phi(z_{\pi(i)})
\ge
\sum_i\phi(z_i)
+\sum_i y_i^\top(z_{\pi(i)}-z_i).
\]

排列不会改变 `phi(z_i)` 的总和，所以左右第一项完全抵消：

\[
\sum_i y_i^\top z_i
\ge
\sum_i y_i^\top z_{\pi(i)}.
\]

定义 assignment score matrix

\[
C_{ij}=y_i^\top z_j.
\]

原配对总分为

\[
A_{\rm id}=\sum_i C_{ii},
\]

所有排列中的最优总分为

\[
A_\star=\max_{\pi\in S_n}\sum_iC_{i,\pi(i)}.
\]

于是精确 posterior mean 必须满足

\[
\boxed{A_\star=A_{\rm id}.}
\]

identity 不一定是唯一最优。合法收缩、低秩投影甚至常数 map 都可能产生 ties；ties 不是
违反。

定义未归一化 assignment regret

\[
\boxed{V_h=A_\star-A_{\rm id}\ge0.}
\]

精确 posterior mean 的硬结论是 `V_h=0`。Hungarian algorithm 一次便同时搜索所有
二循环、三循环和更长 permutation cycles，不需手工拼接 pairwise secant、curl 和负
特征值。

### 二次匹配解释

因为排列保持 `sum ||z_i||^2` 不变，

\[
\boxed{
2V_h
=\sum_i\lVert y_i-z_i\rVert^2
-\min_\pi\sum_i\lVert y_i-z_{\pi(i)}\rVert^2.
}
\]

也就是说：若把 clean responses 重新错配到别的 noisy probes 后，整体平方匹配成本反而
更低，原 input-output graph 就违反循环单调。

这给“结构错配”提供了一个直觉，但不能把 latent assignment 直接解释成可见物体部件
assignment。

## 6. 无量纲归一化与 `[0,1]` 上界

把所有 `y_i^T` 堆成 `Y in R^(n*r)` 的行矩阵，把所有 `z_i^T` 堆成 `Z`。令

\[
H=I_n-\frac1n\mathbf1\mathbf1^\top
\]

为行去均值矩阵。任意排列都保持 `sum z_i`，所以 assignment gap 对输出整体平移不敏感：

\[
A_\pi-A_{\rm id}
=\langle HY,P_\pi HZ-HZ\rangle_F.
\]

由 Cauchy--Schwarz 和 permutation matrix 保持 Frobenius norm：

\[
\begin{aligned}
A_\pi-A_{\rm id}
&\le \lVert HY\rVert_F
\lVert P_\pi HZ-HZ\rVert_F\\
&\le 2\lVert HY\rVert_F\lVert HZ\rVert_F.
\end{aligned}
\]

因此定义

\[
\boxed{
D_h=
\frac{V_h}
{2\lVert HY\rVert_F\lVert HZ\rVert_F+\epsilon}
}
\]

就有严格界

\[
\boxed{0\le D_h\le1.}
\]

若 response denominator 为零，所有 assignment 必然同分，定义 `D_h=0`，同时单独记录
unresolved-response flag，不能悄悄删样。

对冻结检查点集合 `K={99,149,199}`，整条路径分数为

\[
\boxed{
D_{\rm path}^{(h)}
=
\frac{\sum_{k\in K}V_{k,h}}
{\sum_{k\in K}
2\lVert HY_{k,h}\rVert_F\lVert HZ_{k,h}\rVert_F+\epsilon}.
}
\]

它仍在 `[0,1]`。这是按每个时刻可解析 input-response energy 加权的块分数，不是时刻
等权平均。

## 7. 它与 PTCV 的精确关系

令 probe 写成 `z_i=h u_i`，并在中心展开 learned projected map：

\[
g(hu_i)=g(0)+hBu_i+O(h^2),
\]

其中

\[
B=Q^\top\nabla_x\widehat m_t^c(x_\star)Q
\]

正是 PTCV 的 projected Jacobian。常数 `g(0)` 在所有 assignment gap 中抵消，所以

\[
V_h
=h^2\max_\pi
\sum_i(Bu_i)^\top(u_{\pi(i)}-u_i)+O(h^3).
\]

分母也为 `O(h^2)`。因此当 `h -> 0` 时，FPCV 退化为 `B` 决定的一组 finite-cycle
inequalities。

必须主动承认：

1. FPCV 没有发现新的 infinitesimal posterior law；
2. 在无穷小尺度，它与 PTCV 检查同一对称 PSD 几何；
3. 固定 cross-polytope 不保证覆盖 PTCV 的所有 Jacobian 缺陷；
4. 唯一可能的增量是中心 Jacobian 尚正常、但 finite neighborhood 的离中心位置已经
   folding、折返或交叉。

一维玩具例子

\[
g(z)=z-2z^3
\]

在中心有 `g'(0)=1>0`，无穷小 PTCV 通过；但半径足够大后 derivative 变负，有限 probes
能看到 mapping 折返。

因此 FPCV 实验的关键增量预测不是“自己 AUC 看起来还行”，而是

\[
\operatorname{AUC}(D_{h_L})
\ge \operatorname{AUC}(D_{h_S})
\]

且

\[
\operatorname{AUC}(D_{h_L})
\ge \operatorname{AUC}(\mathrm{PTCV}).
\]

否则它只是用 assignment 重新实现失败的局部 PTCV。这里不能做 Richardson；
Richardson 会主动外推回中心 Jacobian，消灭 finite-scale 部分。

## 8. 第一个冻结历史 discovery 设计

### 8.1 数据角色

使用 seed 50--129 的 `fresh_eval240` 历史池：

- 80 个互不重叠 seed；
- 每个 seed 固定 class 207/602/795；
- 240 条 baseline CFG=4、DDPM-250 trace；
- label lock identity：
  `21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a`；
- 216 `clean_good`、8 `clear_bad`、16 `mild_or_disputed`；
- 主结构 endpoint 固定为 class602 的 5 张 fusion/topology bad 与 class795 的 2 张
  misalignment/fusion bad，对应两个类的 145 张 clean-good；
- class207 的一张仅 global-blur bad 不进入结构主终点。

这些历史 labels 在 FPCV 以前锁定、seed 与 targeted100/expansion 不重叠，但研究过程
已经接触过该池的标签组成，所以本实验只能叫 historical discovery，不是 confirmation。

### 8.2 检查点与半径

固定检查点：

| array index | internal timestep | 角色 |
|---:|---:|---|
| 99 | 150 | 早中期 |
| 149 | 100 | 中期 |
| 199 | 50 | 较晚期、仍未完成 |

令

\[
s(x,t)=\sqrt d\max\{\operatorname{RMS}(x),\sigma_t\}.
\]

固定

\[
\boxed{h_L=s(x,t)/32,\qquad h_S=s(x,t)/64.}
\]

因为 `q_a` 是单位向量，两者对应每坐标 RMS 约为当前状态/噪声标尺的 3.125% 与
1.5625%。`h_L` 是唯一主半径；`h_S` 只做数值与机制对照。标签解封后禁止交换。

### 8.3 无标签数值门

在任何 FPCV score-label join 前必须封存并通过：

1. 所有中心 raw conditional outputs 与保存的 conditional epsilon 重放 bitwise exact；
2. `Q^TQ=I` 的 float64 误差低于冻结阈值；
3. 两个半径的 probes、outputs、cost、denominator 与 assignment 均 finite；
4. 两个半径复用完全相同的中心 output；
5. 成本和 Hungarian 重算使用 float64；
6. `A_star>=A_id`，且归一化分数处于 `[0,1]`；
7. analytic affine-symmetric-PSD toy 给 `D≈0`；
8. rotation 与 negative-slope toys 给严格正违反；
9. 输出整体平移和正比例缩放不改变 `D`；
10. 固定无标签 smoke 重复查询给出 numerical floor；只有 regret 超过 floor 100 倍才
    记为 resolved violation；
11. 若任一主半径路径不能解析，或 unresolved 比例超过 1%，整个实验标成 inconclusive，
    不允许删掉路径后继续。

不要求 `h_L/h_S` 分数高度相关：finite folding 本来可能只在大半径出现。若只有小半径
高，不能事后把它换成主分数。

### 8.4 冻结科学门

主分数只用 `D_path^(h_L)`，高为 structural bad。至少报告：

- class-stratified AUC；
- exact class-conditioned rank-sum 单侧检验；
- 每个 bad 的类内 rank；
- class602 与 class795 分别的 AUC；
- leave-one-checkpoint-out；
- small-radius FPCV、原 PTCV、response denominator/energy controls。

建议 discovery advance gates：

1. class-stratified AUC 至少 `0.65`；
2. exact 单侧 `p<=0.10`；
3. 至少 `5/7` structural bad 高于各自类别 clean median；
4. class602 与 class795 AUC 均严格大于 `0.5`；
5. 每个 leave-one-checkpoint-out AUC 至少 `0.55`；
6. large-radius AUC 不低于 small-radius、冻结 PTCV 和 response-energy controls；
7. 禁止按 sign、radius、checkpoint、cycle length、class 或 basis 事后救援。

通过只授权至少含 15 个 structural clear-bad、覆盖至少三个类的新池确认；不直接授权
guidance、rollback 或 rejection。失败则停止当前 FPCV probe/radius/basis 组合。

## 9. 计算量

每个 checkpoint、每个半径有 `2r+1=33` 个计分 points。数学上两个半径共享同一个
中心；但实现中为了让每个 GPU query batch 的形状完全相同、避免不同 TF32/GEMM kernel
制造假的方向不对称，两个半径各自查询一次中心。固定 `chunk=3` 后，每个半径恰好是
11 个 `3 classes × 3 points = 9 samples` 的等形状 batch。

所以每个 seed 的正式成本是：

- cross-polytope：`3 checkpoints × 3 classes × 2 radii × 33 = 594`
  个 sample-equivalent forward；
- 血缘 replay：`3 checkpoints × 3 classes × 2 = 18` 个 sample-equivalent forward；
- 合计 `612` 个 sample-equivalent、`69` 次模型调用。

80 个 seed 的全池为 5,520 次调用、48,960 个 sample-equivalent；四卡各 20 个 seed 时，
每卡约 1,380 次调用。不需要 backward、JVP、随机 suffix 或 endpoint decode；33×33
Hungarian 的 CPU 成本可忽略。额外 replay 与第二次中心查询都不重复进入 FPCV 计分。

## 10. 必须保留的失败边界

以下错误可以严格漏掉：

1. 模型已经把六肢、融合或错误附着学成一个局部自洽的错误模态；
2. learned map 内容错误，但仍是某个凸函数的梯度；
3. 纯收缩、低秩 projection 或常数 map；
4. folding 位于固定 16 维基没有覆盖的方向；
5. 错误来自知识、训练数据或模型容量不足，而不是 posterior map 局部不一致。

非保守也不是任意 sampler vector field 采样错误的必要充分条件。这里的 hard defect 只
相对于 raw conditional MSE target 的 Bayes posterior realizability，不是通用生成正确性
定理。

## 11. 先验工作与可守的新颖性边界

不能声称首创的部分包括：

- 凸函数次梯度与 cyclic monotonicity、permutation 形式：
  [Rockafellar, 1966](https://msp.org/pjm/1966/17-3/pjm-v17-n3-p08-p.pdf)；
- 高斯 posterior mean 的凸梯度/variational 结构：
  [Gribonval & Nikolova, 2018](https://arxiv.org/abs/1807.04021)；
- posterior covariance 与 denoiser derivative：
  [Manor & Michaeli, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/d6adf82fa531dcb8bfa53c224ce5e466-Paper-Conference.pdf)；
- denoiser Jacobian 的 PSD 谱结构：
  [Kadkhodaie et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/cbaf319a4712385b5ba8a414808b5713-Paper-Conference.pdf)；
- score conservativity 与 gauge freedom：
  [QCSBM, ICML 2023](https://proceedings.mlr.press/v202/chao23a.html)、
  [Horvat & Pfister, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/d553f0e0abb80e2a60328d634583bd2e-Paper-Conference.pdf)；
- posterior mean overshrink，且一般不会把 `p_t` 直接 push 到 `p_0`：
  [García Trillos & Sen, 2023](https://arxiv.org/abs/2312.08135)；
- pre-completion temporal artifact detection/correction：
  [ASCED, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_Temporal_Score_Analysis_for_Understanding_and_Correcting_Diffusion_Artifacts_CVPR_2025_paper.html)。

`m_t^c=grad Psi_t^c` 只说明它是从 `p_t^c` 到自身 pushforward `m_#p_t^c` 的 Brenier
map；一般 `m_#p_t^c != p_0^c`。不能把 posterior mean 错写成从 noisy marginal 到 clean
marginal 的最优传输。

目前可守的潜在贡献只能是：

> 在真实生成轨迹的 preterminal 状态周围，用固定 finite-neighborhood probes 的
> cyclic-monotonicity identity-assignment regret，检验结构伪影是否富集于 raw
> conditional denoiser 的 off-center folding 区域。

assignment、凸梯度和 cyclic monotonicity 本身都是经典结果。初步原始文献检索尚未发现
这个特定 per-trajectory artifact detector，但正式投稿前仍需系统检索，不能先写“首次”。

## 12. 当前评级

**Conditional GO。** 值得做一次严格冻结、仅 discovery 的数值实验。它的成败标准不是
“挑到一个 cycle 看起来相关”，而是 large-radius full assignment score 必须同时超过
small-radius、原 PTCV 和简单 response controls，并在两个结构阳性类别中同向。否则立即
停止，不通过换基、换半径或挑 checkpoint 延长路线。

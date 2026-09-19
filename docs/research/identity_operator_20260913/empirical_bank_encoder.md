# 用真实数据 bank 反演：可计算，但还不是免费的真实数据锚

日期：2026-09-13。此文继续原始 no-op/复制问题，不讨论有限映射外推。完成了一个小型 CPU 核覆盖审计；未运行生成模型或 GPU。

**裁决：给定有限原子 bank 的 Bayes 速度有闭式解，但其终点没有唯一反演。把 bank 平滑成高斯混合可得到合法编码器，却只锚定该估计分布；当前 SiT 的 4096 维 latent 实测有很强的单核集中和 bank 依赖，不宜直接作为真实图像 oracle。可先走不需要 oracle 的一维样本 CDF 编码路线，在独立数据上验证 no-op 能否可靠校准 CFG。**

## 1. 原子公式正确，终点反演不成立

对固定类别 c，冻结 B={x1,…,xM}，令 I 均匀，ε~N(0,I)，二者独立：

\[
Z_t=t x_I+(1-t)\epsilon,\qquad
w_i(z,t)=
\frac{\exp[-\|z-tx_i\|^2/(2(1-t)^2)]}
{\sum_j\exp[-\|z-tx_j\|^2/(2(1-t)^2)]}.
\]

\[
m_B(z,t)=\sum_iw_i(z,t)x_i,\qquad
v_B(z,t)=\frac{m_B(z,t)-z}{1-t}.
\]

这是对经验分布的精确条件期望速度，不是对总体真实分布的 oracle。Gaussian 混合经验 score 的精确最优解会导致记忆，已有直接理论与实验；FM 的终点数据几何和记忆也已有分析。[Memorization and Regularization, §3–4](https://arxiv.org/html/2501.15785v2#S3), [FM ODE Dynamics](https://arxiv.org/abs/2412.18730)

当 t→1，连续 prior 被压到 M 个原子，生成映射多对一。最简单 M=1 时

\[
z_t=t\mu+(1-t)e.
\]

所有 e 都在 t=1 到达 μ；不能从 μ 唯一恢复 e。从未出现在 bank 的 x 也没有该终点映射的原像。直接从 t=1 启动数值反演，选中的分支来自端点扰动或实现约定。

截断到 b<1 虽使流可逆，但不能把原始 x 无声当成合法 zb。单中心例子的反演为

\[
e=\frac{x-b\mu}{1-b}.
\]

当 heldout x 与 μ 有固定距离、b→1，编码可以任意大。此时目标应是平滑状态分布 pb，不再是原始图像分布。

局部雅可比还能说明刚性：

\[
D_zv_B=\frac{t\,\mathrm{Cov}_{w}(x_i)}{(1-t)^3}
-\frac{I}{1-t}.
\]

单核主导区有强前向收缩、强反向放大；核切换区域还有协方差项。增加求解精度可以控制数值误差，不能修复端点可逆性或参考目标错误。

## 2. 一个定义完整的平滑版

令真实数据估计为

\[
P_{B,\sigma}=\frac1M\sum_i\mathcal N(x_i,\sigma^2I),\quad \sigma>0.
\]

要保持同一 independent Gaussian FM coupling，应令 clean X=xI+ση，且 η、ε、I 独立：

\[
Z_t=tX+(1-t)\epsilon,\qquad
s_t^2=(1-t)^2+t^2\sigma^2.
\]

权重的方差改为 st²，仍记其中心后验平均为 mB。解析、无端点消减误差的速度是

\[
\boxed{
v_{B,\sigma}(z,t)
=\frac{[t\sigma^2-(1-t)]z+(1-t)m_B(z,t)}{s_t^2}.}
\]

由于 min_t st²=σ²/(1+σ²)>0，有限 bank 下速度全局 Lipschitz 且至多线性增长，确实可以定义可逆流及 E_B,σ=Φ[v_B,σ]^{0←1}；类别不变，bank 与 σ 跨轮冻结。这一正则性针对固定 σ，不能保证 σ→0 时仍有良好的统一条件数。

注意 mB 是中心 xI 的后验均值，不是平滑后 clean X 的后验均值。把 st² 改掉后继续用 (mB−z)/(1−t)，是错误的速度。常见线性最小噪声路径 Zt=t xI+[1−(1−σ)t]ε 虽有相同终点混合分布，却是另一种 coupling，不能在“同 canonical FM 目标”论证中静默替换。

平滑版不需要每轮给输入图片新加随机噪声；σ 是冻结参考密度的模型参数。实际 copy 仍是 x[r+1]=Gθ(E_B,σ(x[r]))。

## 3. 平滑没有自动恢复“正确模型应复制真实图”

精确反例：真实 P=N(0,1)，单中心平滑 bank 为 N(0,σ²)。参考编码 Eref(x)=x/σ。完全正确的生成器 G*(z)=z 给出

\[
T_*(x)=x/\sigma;
\]

σ<1 时多轮越走越远。错误的 Gbad(z)=σz 却完美保持 x。这里没有求解误差、模型合谋或原子奇点；唯一问题是参考密度错了。

因此平滑以后允许提出复制任务，但其“正确”首先是对 P_B,σ 的 canonical coupling 而言。真实总体 fixed point 只能作为估计一致时的近似目标。不可用某个被测模型的重建误差选 σ，然后反过来宣称它最稳定。

外部条件仍是 [此前笔记](canonical_copy_anchor.md) 的两项：

\[
W_2(G_\#\nu,P_{\rm real})
\le \mathrm{Lip}(G)W_2(\nu,E_\#P_{\rm real})
+\sqrt{\mathbb E_{\rm real}\|G(E(X))-X\|^2}.
\]

“编码器将真实分布映成 prior”本身已是密度学习问题。若 E 可逆且可微，

\[
p_{\rm real}(x)=\nu(E(x))|\det DE(x)|.
\]

所以 bank 不是绕开生成建模难题的免费 oracle，而是另一个可以独立检验、也可能很差的密度估计器。

## 4. 高维、泄露与不同 bank

对 bank 中某图 xi 的合法加噪状态 z=t xi+(1−t)ε，

\[
\log\frac{w_j}{w_i}
=-\frac{t^2\|x_i-x_j\|^2+2t(1-t)\epsilon^\top(x_i-x_j)}
{2(1-t)^2}.
\]

距离增大使 self kernel 很容易支配。若复制测试包含 bank 图、同源裁剪、近重复图，结果可主要来自查表。每轮把输入 xr 临时加入 bank 也会改变任务并制造自匹配，必须禁止。Source ID 应按原图分组隔离；近重复另检查。

没有 self kernel 的 heldout 状态也可能由某个最近核独占，此时速度对参考 bank 的偶然近邻极敏感。σ 增大可改善某些数据的覆盖，却增加目标分布的平滑偏差；在 d 维，各向同性平滑噪声的 RMS 总范数是 σ√d。单看每坐标 σ 小会低估改动。

换独立 bank、增加 bank 大小、在独立数据上选择带宽，能够量化估计误差，但不能通过“两个有共同偏差的 bank 得出相似结果”来证明正确。相关研究已将记忆与局部覆盖联系起来；神经去噪器的跨数据子集泛化也可能来自其归纳偏置，而非逐步逼近有限原子经验最优。[Local Coverage](https://arxiv.org/abs/2606.14390), [Geometry-adaptive Generalization](https://arxiv.org/abs/2310.02557)

## 5. 当前 SiT 的 CPU 覆盖检查

缓存 mean/std 对应 4×32×32=4096 维 latent，采样单位为 (mean+std·noise)×0.18215。[训练源](../../../experiments/train_imagenet100_sit_flow.py)

选类别 0、1，各取两个不相交的 512 图 bank 及 16 张额外 heldout 图；两类别各固定一个 seed。检查 source ID 零交集。在固定 posterior mean 表示上构造合法 Gaussian bridge probes，计算核责任权；没有运行模型、没有执行实际反演。

| 诊断 | 类别 0 | 类别 1 |
|---|---:|---:|
| heldout 至 bank 最近图 RMS / 坐标 | 0.9133 | 0.8770 |
| t=.10，bank 内 self 图本核后验中位数 | .999995 | .999699 |
| t=.25，heldout 有效核数 ESS 中位数 | 1.0278 | 1.0038 |
| t=.50，heldout ESS 中位数 | ≈1 | ≈1 |
| t=.25，换 bank 后中心后验均值差 RMS / 坐标 | .8354 | .8140 |
| 平滑 σ=1 时终点 heldout ESS 中位数 | ≈1 | ≈1 |

σ=1 的平滑总噪声 RMS 已达 64，终点责任权仍几乎集中到一个核。责任权集中本身不证明一个 Gaussian mixture 密度一定错误；但结合 heldout 距离、自匹配差异和 bank 敏感性，本轮结果没有支持把该全维 KDE 当成稳定外部锚。尤其这些是 bridge probes，不得写成“真实 inverse 轨迹已坍缩”。

另一个仓库细节：实际训练会从 VAE posterior 采样，因此直接用缓存分量 N(μi,Σi) 可得到更贴训练的经验混合：

\[
S_i(t)=(1-t)^2I+t^2\Sigma_i,\quad
v_i=\mu_i+[t\Sigma_i-(1-t)I]S_i^{-1}(z-t\mu_i).
\]

混合权重必须包含 logdet Si。其非零协方差从数学上避免原子，但本轮缩放后的 posterior std 中位数只有 1.38×10⁻⁴ 和 1.76×10⁻⁴；它不是足以覆盖新图的自然平滑尺度。这里未执行各向异性混合反演。

脚本：[empirical_bank_anchor_audit.py](../../../experiments/cfg_transport_search_20260913/empirical_bank_anchor_audit.py)。

结果：[coverage.json](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/empirical_bank_anchor/coverage.json)。

复现命令：

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/cfg_transport_search_20260913/empirical_bank_anchor_audit.py

## 6. 一条不需要 oracle 的实际起点：一维样本 CDF 公共编码

先验证原假设是否真的能指导 CFG，避免把高维密度估计误差混进来。一维有一个很有用的特殊性：所有正则 ODE 生成映射保持次序，因此目标分布指定唯一的单调生成映射，不再有任意旋转 gauge。

仅从真实样本 bank 估计平滑 CDF：

\[
F_{B,\sigma}(x)=\frac1M\sum_i\Phi((x-x_i)/\sigma),\qquad
E_{B,\sigma}(x)=\Phi^{-1}(F_{B,\sigma}(x)).
\]

这等于该混合分布的一维 FM 终点逆，无需知道总体真实密度，也无需 ODE 反演。数值上需要稳定处理极端尾部；截断 CDF 应记录其阈值和饱和比例，不能悄悄压掉难例。

最小协议如下：

1. A 仅拟合参考密度；带宽使用预先固定、仅依赖 A 的规则，或另设 B 按 heldout likelihood 选择；C 用于 no-op 校准；D 最终测试。另取独立 A′ 做参考敏感性。不得用 C/D 构建 bank。
2. 使用有限数据拟合或已有近似 conditional/unconditional 场；对固定 CFG 候选 G_w，计算 C 上的单轮重建损失 Lcopy(w)=mean|G_w(E_A(X))−X|²，冻结选出的 w。此处只是验证原始 no-op 信号能否校准 guidance，不宣称新 guidance 公式。
3. D 上报告单轮和五轮相对原始 X 的漂移，同时从标准 prior 独立生成，报告经验 W2、覆盖和类别收益。另检验 E_A(D) 与 N(0,1) 的完整一维分布相容性，而非只看均值/方差。
4. 先放完全正确 conditional 场作为反证控制：no-op 校准应趋向普通 conditional。再测有限模型误差下，copy 选择是否能改善独立质量；若只能把模型拉向坏 bank，则原假设在该估计设置下失败。

正向理由是：若 F 是总体真实 CDF，E*=Φ⁻¹F，任意单调 G 都满足

\[
\mathbb E_{X\sim P}|G(E^*(X))-X|^2
=W_2^2(G_\#\nu,P).
\]

这是同序最优耦合的直接结论。有限样本 encoder 近似这一量；真实密度只需作为可选的独立评估器，不参与编码或参数选择。参考 bank 增大后能否收敛、换 bank 是否稳定、holdout prior 是否相容，都可实际测量。

该低维路线已由负责一维实验的代理实际完成：使用 1024 样本 bank、三个固定 bank seed、预定 Silverman 带宽、256 个 bank 外复制校准样本。尾部使用 log-CDF/log-survival 与 inverse-normal 的稳定实现，没有裁剪。三个 bank 均选 CFG α=1、AG α=1.25；独立生成 W2² 从 strong 的 .10861 降至 .05672 和 .00237。此为预设低维模型的校准结果，不是 SiT 图像收益。

同一次实验也暴露了偏差：heldout E_B#P 到高斯的 W2² 为 .0154–.0246，编码标准差为 .866–.914；oracle 编码选择的 α 则为 1.5。因此这是“不需要 oracle 也能获得有用信号”的正例，同时不是“样本 bank 已无偏”的证据。[非 oracle 实验结果](../identity_copy_toy_20260913/nonoracle_results.json)

高维 PCA 投影可做类似低维诊断，但仅保留投影信息；把剩余像素硬传回只能证明那部分由结构保持，不能升级为全图 generative no-op。当前不启动 SiT 全维 KDE 反演、不据其残差修改 CFG。另行进行的 SiT 公共 checkpoint 参考回灌仍可作为相对重建和参考敏感性观察，不能因为引用了数据 bank 理论就改称已有外部真值锚。

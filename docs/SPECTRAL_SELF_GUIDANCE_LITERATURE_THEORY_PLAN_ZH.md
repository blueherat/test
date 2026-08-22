# 频谱退化自引导：文献边界、严格形式化与研究计划

更新时间：2026-08-22

## 1. 结论先行

这条想法有一个有价值的理论母题，但用户给出的直接算法不能作为新方法：

1. **“预测 clean、模糊 clean、保持噪声重加噪、再次查询同一模型、从退化预测向完整
   预测外推”已经是 SAG 论文中的 global blur guidance。** SAG 进一步只模糊自注意力
   选中的区域，以减轻全局模糊造成的背景和光滑区域退化。
2. **“扩散是频谱自回归”只在数据集平均和软 SNR 前沿意义下成立。** 它不意味着单张
   图的频率严格单调，也不意味着每个采样步只负责一个硬频带。
3. 二次查询产生的差值一般不是“纯高频增量”。它的精确对象是 denoiser 对频带删除的
   非线性响应；模型 Jacobian 可以旋转和跨频混合输入扰动。
4. “浅层就是低频、深层就是高频”在 DiT 上是需要测量的经验命题，不是架构定理。
   仓库现有结果已经说明，仅看 gap 的频谱质心或高频能量不能区分 useful 和 failed
   guidance。
5. “回看过去预测”的轻量变体与 HiGS、频域 moving-average sampling 等历史预测
   guidance 高度重合，并且原描述混淆了两种相反的时间约定，不宜作为主线。

因此，真正还有研究空间的问题应收缩为：

> **频带删除后，生成模型产生的响应能否分解成普通线性锐化与非高斯、跨尺度的语义
> innovation；后者能否在不使用 FID 标签的情况下，预测哪些 internal/auto guidance
> 弱头及时间调度真正有用？**

这与 SAG 的差别不是换一个 blur kernel，而是从“直接使用 blur gap”转向：

- 建立可验证的频带响应算子；
- 给出线性平稳高斯情况下的闭式零假设；
- 分离普通频率增益与非线性跨频响应；
- 用该响应无监督选择 IG/AG provider，而不是靠生成 FID 扫出 `4 -> 8 -> 10`。

只有这个版本同时通过机制诊断和 held-out 调度选择，才值得继续发展成方法。

## 2. 文献边界

### 2.1 频谱自回归的来源

- Sander Dieleman 的博客
  [Diffusion is spectral autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)
  用自然图像平均功率谱与白噪声平坦功率谱解释 diffusion 的 coarse-to-fine 现象。
- 该博客明确指向 ICLR 2023 的
  [Generative Modelling With Inverse Heat Dissipation](https://openreview.net/pdf?id=4PJUBT9f2Ol)。
  IHDM 直接把热方程作为逐渐抹除细尺度结构的 forward process，并分析标准高斯 diffusion
  隐式具有的 coarse-to-fine 频谱偏置。
- 2025 年的
  [A Fourier Space Perspective on Diffusion Models](https://arxiv.org/abs/2505.11278)
  进一步分析标准白噪声过程为何更早破坏高频 SNR，以及这种频率不均匀腐化可能造成的
  high-frequency reverse approximation 问题。它是 arXiv 工作，不能当作 ICLR 录用论文。

这些工作支持“存在统计意义下的频率顺序”，但不支持“网络每一层或每一步严格对应一个
傅里叶频带”。

### 2.2 直接的新颖性碰撞：SAG

[Self-Attention Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.html)
在 ICCV 2023 已经提出 blur guidance：

1. 从当前 noisy state 得到 `x0` 与预测噪声；
2. 高斯模糊 `x0`；
3. 使用同一预测噪声把模糊结果重新扩散到当前时间；
4. 对退化 state 再运行同一 denoiser；
5. 从退化预测向完整预测外推。

官方代码
[cvlab-kaist/Self-Attention-Guidance](https://github.com/cvlab-kaist/Self-Attention-Guidance)
的 `attention_masking()` 先 blur `pred_xstart`，再用 `prev_noise` 调用 `q_sample()`；采样
代码使用：

\[
\epsilon_{guided}
=
\epsilon_{deg}+s(\epsilon_{full}-\epsilon_{deg}).
\]

本次审计的官方仓库 commit 为
`b4f16d8e09cf0b93a325b5ad94970d30deef3994`。SAG 相比全局 blur guidance 的主要新增
是利用 self-attention mask 只退化当前关注区域。

所以，用户给出的“谱退化自引导”主体应被准确标记为 **global blur guidance / SAG
baseline 的重新表述**，不能声称算法新颖性。

### 2.3 AG、IG 与内部弱模型

- [AutoGuidance, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/5ee7ed60a7e8169012224dec5fe0d27f-Paper-Conference.pdf)
  使用更小或训练更少的同类模型作为 weak predictor，并沿 strong-minus-weak 外推。
- [S2-Guidance, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d97a16cb83d74195b76e0bf1e85bf072-Abstract-Conference.html)
  在推理时随机 drop blocks 构造同一模型的 sub-network 作为 weak predictor。
- Internal Guidance 使用中间层 readout 作为 weak predictor。当前仓库已经严格研究了
  depth-specific readout、频谱 atlas、time-band intervention 和 depth schedule。

这些工作说明“无需另训完整 weak model”是活跃方向；仅仅换成 blur、截层或 block drop
都已经很拥挤。新的贡献必须解释 **怎样选退化、怎样选层、为什么该差值有用**。

### 2.4 近两年 ICLR 中的频率与 guidance 工作

- [Warm Diffusion, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/22a25fc3da528794d52664dacc7bd470-Abstract-Conference.html)
  把 blur 与 noise 同时放进训练 forward process，并用 denoising/deblurring 双分支利用
  低频结构与高频细节之间的 spectral dependency。它不是推理时 SAG，但已经系统研究了
  “模糊强度、噪声强度、流形偏离”的权衡。
- [FreCaS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/12564fe3900e4e2cefb52d4b4ad40625-Abstract-Conference.html)
  在高分辨率级联采样中逐步扩展频带，并对 CFG 的低/高频分量使用不同强度。
- [K-Flow, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5faf0f518f39e4d6b2b595e3ca2fe0c4-Abstract-Conference.html)
  直接在 Fourier、wavelet 或 PCA 的 scale/amplitude 参数上定义 flow。它覆盖了把频率
  层级直接变成 generative path 的训练式路线。
- [W-Edit, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f049728dc51a393a6d8c7d6e25198ee5-Abstract-Conference.html)
  在编辑任务中报告 DiT block-wise frequency progression，并用 wavelet 分解控制结构与
  细节。这为“层深与频率可能相关”提供实证，但不是通用层频率定理。
- [Stage-wise Dynamics of CFG, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f9e2800a251fa9107a008104f47c45d1-Abstract-Conference.html)
  从 multimodal conditional dynamics 分析 early direction shift、middle mode separation
  与 late concentration，说明 guidance 时间调度会改变质量与多样性。
- [Guidance Matters, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html)
  指出很多 guidance 改善可以被有效 guidance scale 或评价偏好解释，要求校准与基线方向
  平行/正交的作用后再比较。
- [HiGS, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6768f8953c485bf115d82a84e61668bd-Abstract-Conference.html)
  使用当前预测与历史预测 moving average 的差作为近零额外 NFE 的 guidance。

[Frequency-Decoupled Guidance](https://arxiv.org/abs/2506.19713) 直接对 CFG 的低、高频
分量应用不同 scale，但截至本报告只应按 arXiv/ICLR 投稿稿件描述，不能写成已录用
ICLR 结论。

## 3. Sander 频谱前沿的准确推导

采用统一的线性 noisy observation：

\[
Z_t=\alpha_t X+\sigma_t E,
\qquad E\sim\mathcal N(0,I).
\]

记数据在频率 `f` 的平均功率为 `P_X(f)`。白噪声的平均功率为常数 1，则该频率的
信噪比为：

\[
\boxed{
\operatorname{SNR}_t(f)
=
\frac{\alpha_t^2P_X(f)}{\sigma_t^2}
}.
\]

设可检测阈值为 `tau`，频率 `f` 仍可辨识的近似条件为：

\[
P_X(f)>\tau\frac{\sigma_t^2}{\alpha_t^2}.
\]

如果自然图像平均谱满足：

\[
P_X(f)=C\lVert f\rVert^{-\beta},
\]

则软频谱前沿为：

\[
\boxed{
f_*(t)
=
\left(
\frac{C\alpha_t^2}{\tau\sigma_t^2}
\right)^{1/\beta}
}.
\]

Sander 博客使用 data-to-noise 的 rectified-flow 约定：

\[
\alpha_t=1-t,\qquad\sigma_t=t,
\]

因此 `f_*(t)` 随 `t` 下降。当前 ImageNet-100 SiT 代码使用相反约定：

\[
Z_t=(1-t)E+tX,
\qquad
\alpha_t=t,\quad\sigma_t=1-t,
\]

采样从 `t=0` 噪声走到 `t=1` 数据，所以：

\[
f_*(t)
=
\left(
\frac{Ct^2}{\tau(1-t)^2}
\right)^{1/\beta}
\]

随采样推进而增大。

这个推导有三条边界：

1. `P_X(f)` 是数据集平均量，单张图像可能不单调；
2. SNR 从信号主导到噪声主导存在宽过渡带，不是硬 cutoff；
3. SD-VAE latent 的 2D FFT 不自动等价于 decoded pixel 频率，必须单独验证。

## 4. 谱退化自引导的精确形式

令：

\[
m_t(z)=\widehat X_0
\]

表示把任意 model parameterization 转换到 clean space 后的 predictor。令 `L_tau` 为
低通算子，`H_tau=I-L_tau` 为被删除的部分。

从当前状态估计噪声：

\[
\widehat E_t
=
\frac{Z_t-\alpha_tm_t(Z_t)}{\sigma_t}.
\]

保持该噪声不变，把 clean prediction 换成其低通版本：

\[
\begin{aligned}
\widetilde Z_t
&=\alpha_tL_\tau m_t(Z_t)+\sigma_t\widehat E_t\\
&=Z_t+\alpha_t(L_\tau-I)m_t(Z_t)\\
&=\boxed{Z_t-\alpha_tH_\tau m_t(Z_t)}.
\end{aligned}
\]

这条公式也给出一个实现审计规则：

- 旧 RAEv2 频率轴使用 data-to-noise 约定，clean coefficient 是 `1-t`；
- 当前 SiT 使用 noise-to-data 约定，clean coefficient 是 `t`；
- 将旧代码中的 `(1-t)` 原样搬到 SiT 会构造错误的反事实状态。

再次查询同一模型并定义：

\[
\boxed{
d_{t,\tau}(z)
=
m_t(z)-m_t\big(z-\alpha_tH_\tau m_t(z)\big)
}.
\]

直接外推为：

\[
m_t^{guided}(z)=m_t(z)+\gamma d_{t,\tau}(z).
\]

除 model output space 和 scale 记号外，这就是 global blur guidance 的主体。

## 5. 为什么 `d` 一般不是纯高频

令：

\[
\delta=\alpha_tH_\tau m_t(z).
\]

由微积分基本定理，有精确恒等式：

\[
\boxed{
d_{t,\tau}(z)
=
\left[
\int_0^1J_{m_t}(z-s\delta)\,ds
\right]\delta
}.
\]

这不是 Taylor 近似。它说明二次预测差是 denoiser Jacobian 沿有限退化线段对删除频率
的运输。即使 `delta` 完全位于高频，`J_m delta` 也可能产生低频、语义或跨通道变化。

若 Gaussian blur 写成 heat semigroup：

\[
L_\tau=e^{\tau\Delta},
\qquad H_\tau=I-e^{\tau\Delta},
\]

则小 `tau` 时：

\[
H_\tau=-\tau\Delta+O(\tau^2),
\]

因此：

\[
\boxed{
\lim_{\tau\to0}
\frac{d_{t,\tau}(z)}{\tau}
=
-\alpha_tJ_{m_t}(z)\Delta m_t(z)
}.
\]

仓库 RAEv2 所谓 frequency derivative 正是在有限差分估计这个对象。它不是简单的
`-Delta m` 锐化，而是 `J_m` 作用后的模型响应。

## 6. 一个闭式零假设：平稳高斯数据

假设 `X` 是零均值平稳高斯场，频率 `f` 的功率为 `P_X(f)`。此时 posterior-mean
denoiser 是 Fourier-diagonal Wiener filter：

\[
\widehat{m_t(z)}(f)
=
K_t(f)\widehat z(f),
\]

其中：

\[
\boxed{
K_t(f)
=
\frac{\alpha_tP_X(f)}
{\alpha_t^2P_X(f)+\sigma_t^2}
}.
\]

由于 `K_t` 与任何 Fourier low/high-pass filter 可交换：

\[
\begin{aligned}
d_{t,\tau}
&=K_tz-K_t(z-\alpha_tH_\tau K_tz)\\
&=\boxed{\alpha_tK_tH_\tau K_tz}\\
&=\boxed{\alpha_tK_tH_\tau m_t(z)}.
\end{aligned}
\]

于是 guided predictor 只是：

\[
\widehat m_t^{guided}(f)
=
\left[1+\gamma\alpha_tK_t(f)H_\tau(f)\right]
\widehat m_t(f).
\]

这给出一个很重要的结论：

> 在线性平稳高斯极限下，谱退化自引导不会发现新的 high-frequency information；它
> 只是一个由 denoiser confidence 调制的频率增益/锐化算子。

它可能改变 FID 或 precision，但本质上仍可能通过压缩方差、降低 recall 或过锐化获得
收益。若真实模型的 response 也几乎满足这个零假设，继续包装成“模型提取了语义高频”
是不成立的。

## 7. 真正值得测量的对象：跨频响应矩阵

令 `{P_b}` 为互相正交、和为 identity 的平滑频带投影。对第 `b` 个输入频带做小幅删除：

\[
z_t^{(-b)}
=
z_t-\eta\alpha_tP_bm_t(z_t).
\]

定义从输入频带 `b` 到输出频带 `a` 的响应：

\[
\boxed{
r_{a\leftarrow b}(z,t)
=
\frac{1}{\eta}
P_a\left[
m_t(z)-m_t(z_t^{(-b)})
\right]
}.
\]

当 `eta -> 0`：

\[
r_{a\leftarrow b}
\to
\alpha_tP_aJ_{m_t}(z)P_bm_t(z).
\]

由此定义归一化 response energy matrix：

\[
\boxed{
M_{ab}(t)
=
\frac{
\mathbb E\|r_{a\leftarrow b}(Z_t,t)\|^2
}{
\mathbb E\|P_bm_t(Z_t)\|^2+\varepsilon
}
}.
\]

- `M` 近对角：模型主要把频带扰动留在原频带；方法接近 Wiener sharpening/SAG。
- `M` 有稳定的 off-diagonal 结构：存在真实的跨尺度非线性依赖，但它是否有益仍需
  因果采样验证。
- `M` 随 `eta` 大幅变化：当前 perturbation 不在线性响应区，不能用 Jacobian故事解释。

为了进一步分离普通增益，可在每个 `(t,f)` 上拟合最优 Fourier-diagonal transfer：

\[
\kappa_t(f)
=
\frac{
\mathbb E[\widehat r_t(f)\overline{\widehat m_t(f)}]
}{
\mathbb E|\widehat m_t(f)|^2+\varepsilon
},
\]

并定义：

\[
\boxed{
r_t^{nonlinear}
=
r_t-
\mathcal F^{-1}
\left[\kappa_t(f)\widehat m_t(f)\right]
}.
\]

在线性平稳高斯零假设下，足够细频带中的 `r_nonlinear` 应趋近于零。它非零只说明偏离
该零假设，不自动说明它提高质量。

## 8. 为什么网络深度不能直接等同于频率

对内部 readout `W_l`，clean/velocity 统一后定义：

\[
g_l(z,t)=S(z,t)-W_l(z,t).
\]

“浅层输出更模糊”只说明某些样本上存在视觉相关性。它不推出：

\[
g_l=P_{high}g_l,
\]

更不推出：

\[
\text{high-frequency energy large}
\Rightarrow
\text{guidance useful}.
\]

当前仓库已经有直接反例：

- `depth8_v` 与 failed `final_linear_x` 的频谱质心、高频占比几乎相同；
- `depth8_x` 与 failed `depth12_x` 也相近；
- static/learned spectral controller 没有超过 scalar IG；
- 但 time-band causal intervention 显示 useful gap 的主要收益集中在 `mid x high`；
- `depth 4 -> 8 -> 10` 在 FID-5K 上达到 `42.6254`，明显优于 static heads 与旧 AG。

所以频率可以是**因果干预坐标**，却不是仅靠静态能量就能读取的 utility 标签。

## 9. 仓库里已经做过的同构实验

`experiments/run_raev2_frequency_axis_extrapolation_screen_v1.py` 与 suite v1--v3 已经实现：

\[
x_t^{(\tau)}
=
x_t+(1-t)(L_\tau\widehat x_0-\widehat x_0),
\]

保留预测噪声后再次查询 frozen RAEv2 clean predictor，并用：

\[
g_{freq}
=
\frac{\widehat x_0-\widehat x_0^{(\tau)}}{\tau}
\]

做闭环外推。小规模诊断得到：

- 相邻 blur scale 的 direction cosine 约 `0.925`；
- 两种估计方式 cosine 约 `0.883`；
- 与 IG `full-base` gap cosine 约 `0.294`；
- 与 paired clean residual cosine 约 `-0.003`。

这说明该模型响应轴稳定且不同于旧 IG，但没有证明它是逐样本误差修正方向。正式 13 条件
FID-5K suite 仍停在 49/52 shards，不能把它当成已完成的正结果。

另外，`docs/IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md` 已经完成 99 条配对
FID-1K time-band/depth 条件和一个 FID-5K 确认。新研究必须复用这些 useful/failed provider
作为 held-out 判别题，不能重新从 blur scale 扫参开始。

## 10. 唯一值得优先检验的新假设

### H1：存在可重复的 model spectral response frontier

用 empirical latent power 定义：

\[
\operatorname{SNR}_{t,b}
=
\frac{\alpha_t^2P_b}{\sigma_t^2N_b},
\]

并用平滑权重选取 `SNR` 接近 1 的 frontier band：

\[
w_b(t)
\propto
\exp\left[
-\frac{(\log\operatorname{SNR}_{t,b})^2}{2h^2}
\right].
\]

需要验证 `M_ab(t)` 的主响应是否随这个 frontier 从低频向高频移动，而不是事后画出一条
看起来合理的曲线。

### H2：useful guidance 与非平凡 spectral innovation 对齐

对每个 internal/weak provider，统一到相同 output space 得到 `g_l`。定义无监督
alignment score：

\[
\boxed{
A_l(t)
=
\sum_b w_b(t)
\mathbb E
\left[
\cos\big(g_l(Z_t,t),r_{t,b}^{nonlinear}(Z_t,t)\big)
\right]
}.
\]

最关键的测试不是该分数是否和 FID 数字相关，而是它能否在**不读取生成指标**时：

1. 把 useful `depth4/6/8/10_v`、`external_v500` 与 failed `depth12_x`、
   `final_linear_x` 分开；
2. 在 early/mid/late 分别选出接近 `4/8/10` 的顺序；
3. 在 held-out noise、class 和 checkpoint 上保持排序。

如果做不到，频谱响应仍只是另一个漂亮但不预测质量的诊断量，应停止。

### H3：频谱响应可无标签校准便宜的 IG schedule

若 H2 通过，可在少量 unguided trajectories 上离线估计：

\[
l^*(t)=\arg\max_l A_l(t),
\]

再在真实采样中使用：

\[
\dot z_t
=
S(z_t,t)
+\gamma\left[S(z_t,t)-W_{l^*(t)}(z_t,t)\right].
\]

频谱 perturb-and-requery 只用于离线、无标签的 schedule calibration；推理仍使用内部
readout，不必像 SAG 一样每个 NFE 多跑一次完整模型。

这条方法线真正可能有贡献的地方是：

> 用可推导的 spectral response probe 取代 FID sweep，自动发现 internal-guidance 的
> layer-time schedule，并在未参与校准的模型上迁移。

它不是“频谱外推本身”，而是“频谱响应校准 IG/AG”。

## 11. 最小实验计划

### Phase A：无生成训练、无 FID 的机制筛查

模型固定为现有 SiT-v800 strong；不训练任何新权重。使用 512 个完全配对的 baseline
trajectory states：

1. 时间：`t={0.05,0.15,0.3,0.5,0.7,0.85,0.95}`；
2. 频带：4--6 个平滑 Fourier band，并补一个 2-level wavelet control；
3. perturbation：`eta={0.005,0.01,0.02}`，先验证线性稳定性；
4. provider：useful `depth4/6/8/10_v, external_v500` 与 failed
   `depth12_x, final_linear_x, raw-depth proxy`；
5. 输出：`M_ab(t)`、diagonal concentration、off-diagonal energy、`r_nonlinear` RMS、
   provider alignment 和 bootstrap CI；
6. 同时 decode clean responses，检查 latent band 与 pixel band 的关系。

这一阶段不允许用现有 FID utility map调 score；现有 utility 只能在 score 完成后作为
held-out 标签揭盲。

### Phase B：完全配对 FID-1K 因果筛查

只有 Phase A 能区分 useful/failed provider 时才进行。条件至少包括：

1. unguided baseline；
2. static depth-6 与现有 `4 -> 8 -> 10` 参考；
3. spectral-response 自动选出的 schedule；
4. reverse schedule 与 random schedule；
5. global blur guidance/SAG baseline；
6. 只保留 Fourier-diagonal gain 的 control；
7. 只保留 `r_nonlinear` 的 direct upper-bound condition；
8. scalar gamma sweep 与 RMS-matched control。

所有条件共享 noise/label，报告 FID、sFID、IS、precision、recall、NFE、wall time 和
decoded frequency statistics。参考 Guidance Matters，还需比较 guidance 相对 native
field 的平行/正交成分，避免“只是更大 effective scale”的伪进步。

### Phase C：迁移确认

只有自动 schedule 在 Phase B 超过 static head、global blur guidance 与 matched scalar
IG 后才进行：

1. 至少两个 sampling seeds 的 FID-5K；
2. 从 v800 校准的规则迁移到 v400 或另一 official checkpoint，不重新扫 FID；
3. 再迁移到一个不同 architecture/representation；
4. 最后才考虑正式 50K 与更大模型。

## 12. 停止标准

出现任一情况即停止该频谱主线：

1. response 对 `eta` 不稳定，无法定义局部算子；
2. latent SNR frontier 不呈稳定 coarse-to-fine 顺序；
3. latent response 与 decoded pixel frequency 没有可解释关系；
4. alignment score 无法把已知 useful 与 failed heads 分开；
5. 无标签选择不能恢复或迁移 `4 -> 8 -> 10` 一类有效顺序；
6. direct method 不优于 SAG/global blur 或 matched scalar IG/AG；
7. FID 改善伴随明显 recall/class coverage 下降或只是频谱过锐化；
8. 收益只在调 score 的同一 checkpoint 上存在，迁移后消失。

## 13. 为什么暂不做“噪声反照”

该变体当前有三个问题：

1. 时间符号不清。SiT 从 `t=0` 噪声走到 `t=1` 数据，`t-delta` 是更早、更 noisy 的
   state；在传统 data-to-noise 记号中含义相反。
2. 不同时间的 prediction 差同时混入 state drift、time embedding 与 output conversion
   conditioning，不能直接称为高频增量。
3. 当前预测减历史预测已经由 HiGS 与 frequency-domain moving-average sampling 系统
   研究；它并非真正“零概念开销”的空白方向。

因此它只应作为 HiGS 风格 baseline，不应和 spectral-response 主问题并行扩张。

## 14. 最终研究目标

当前目标不是证明“高频外推一定有效”，而是回答一个可以被杀死的问题：

\[
\boxed{
\text{模型的非平凡 spectral response}
\stackrel{?}{\Longrightarrow}
\text{可迁移的 IG/AG provider 与时间调度选择}
}
\]

如果答案为否，plain blur guidance 已被 SAG 覆盖，history variant 已被 HiGS 覆盖，这条线
应干净结束。

如果答案为是，真正的贡献将是：

1. 一个关于 denoiser band-deletion response 的线性高斯零假设与非线性偏离诊断；
2. 一个无需 image label、reward 或 FID sweep 的 internal-guidance schedule selector；
3. 一个由理论前沿预测并在多个模型上迁移的质量改进方法。

这个版本比“模糊一下再外推”更窄、更难，但也更可能形成 solid 的机制与方法闭环。

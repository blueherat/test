# 公共反演下的五轮图像复制：实测与下一步依据

2026-09-13。研究主线仍是用户提出的“输入图像、要求不变、把输出作为下一轮输入”。本轮没有恢复 CFG/APG/CTRL 组合或调制搜索。

**已把操作落实到真实图像的 FM latent，并完成三个公共参考、四个生成器、每图五轮，共 960 个输出。实验表明该操作确实能产生可观察的多轮漂移，也表明漂移强烈依赖反演参考。另一个有独立数据锚的一维实验则确认：在明确条件下，可以用复制目标选择有益的 guidance。尚未得到超过调优生成 baseline 的新方法。**

## 1. 本次实际执行的“回灌”

固定参考模型 R 及类别 c，每轮执行

\[
z_r=E_{R,c}(x_r),\qquad x_{r+1}=G_{\theta,c}(z_r).
\]

E_R 从图像端 t=1 积分到噪声端 t=0；G 从 0 积分到 1。这里反演使用负时间步，并在每个新位置、新时刻重新查询网络。每个方向 128 步 Heun。

- 原始输入是验证集随机选取的 16 类各一张真实图的缓存 latent，所有组使用同一批图。
- VAE posterior 只在固定 R0 时采样一次。整个 R1–R5 没有新噪声，没有重新采样 VAE。
- 每轮重新反演**上一轮输出**，没有复用最初编码，也没有始终输入 R0。
- 轮间传递 latent，逐轮解码、保存完整 PNG 和 latent/逆编码 NPZ。它是 latent 复制实验，不能称为每轮上传 RGB 的完整多模态接口。
- 每轮同时比较原图及上一轮；不把“最后两轮接近”误当作保留原图。

两个公共参考分别为同训练轨迹的 700K 模型，以及另一次采用 logit-normal 时间采样训练的 300K 模型。后者仍与被测模型共享架构、数据及 seed。**两者都不是 oracle；前者尤其可能偏向同训练轨迹。**

被测生成器为 800K conditional、400K conditional、800K CFG，以及 800K/400K autoguidance。CFG 为 S+1.25(S−U)，AG 为 S+0.5(S−W)，两者均在 t<.75 的接受区间启用。这里 AG 使用两个 checkpoint，不是此前 depth-4 内部头。系数是固定观察条件，未用本批图调参。

## 2. 五轮实测

下表为相对 R0 的每坐标 latent MSE，16 图平均；不是 FID。

| 公共反演器 | 正向生成器 | R1 | R5 |
|---|---|---:|---:|
| 700K | 800K conditional | 0.02242 | 0.21942 |
| 700K | 400K conditional | 0.03572 | 0.28830 |
| 700K | CFG | 0.17169 | 0.75037 |
| 700K | AG | 0.03265 | 0.28311 |
| 独立训练 300K | 800K conditional | 0.03858 | 0.69255 |
| 独立训练 300K | 400K conditional | 0.04146 | 0.74748 |
| 独立训练 300K | CFG | 0.17123 | 1.12149 |
| 独立训练 300K | AG | 0.04661 | 0.75332 |

可直接检查同一个 800K 生成器的两套完整图册：

- [700K 公共反演，R0–R5](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2/ref700/strong/rounds.png)：可见细节消失、物体简化及部分结构改变。例如叠碗、两只狗、建筑细节逐轮改变。
- [另一次训练的 300K 公共反演，R0–R5](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2/ref_alt300/strong/rounds.png)：同一生成器出现了另一种明显漂移，边缘增强、颜色失真和纹理伪影逐轮增加。
- [全部曲线](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2/copy_drift.png)、[逐轮核验结果](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2/analysis.json)。各实验臂目录均有完整 R0–R5 图册。

800K 在这两个参考下的 latent MSE 都低于 400K，但这不足以证明“越强越会复制”的一般关系。相对次序会受指标影响：独立训练参考下，400K 的平均像素 PSNR 略高于 800K。参考模型及配对规则仍然显著参与了任务本身。

### 积分精度复核

预先指定的首四图完成 128→256 步、完整五轮的加密复算。下面同时给第五轮相对原图的误差、两套求解结果之间的误差；这两个量不能相减当成因果分解。

| 参考 | 生成器 | 256 步 R5 复制 MSE | 128/256 步 R5 输出差 MSE |
|---|---|---:|---:|
| 700K | 800K | 0.20435 | 0.00000541 |
| 700K | 400K | 0.25789 | 0.00000436 |
| 700K | CFG | 0.94511 | 0.006888 |
| 700K | AG | 0.31070 | 0.0000452 |
| 独立训练 300K | 800K | 0.63108 | 0.0000127 |
| 独立训练 300K | 400K | 0.66545 | 0.00000669 |
| 独立训练 300K | CFG | 1.28873 | 0.005973 |
| 独立训练 300K | AG | 0.70528 | 0.0003148 |

在这四图上，复制漂移远大于加密带来的输出变化；尤其两个 conditional 分支的现象不能主要用这一级步长误差解释。CFG 的数值敏感性较大，不能由此宣称精确图像结果已收敛。此复核只覆盖四图，不冒充全 16 图的精度认证。

自反 R∘R⁻¹ 的检查也已执行，但仅用于实现/数值检查，不拿其小误差给参考模型评优。每轮 conditional 往返计 512 次模型调用，CFG/AG 计 704 次；本实验不是低成本部署方法。

## 3. 正向理论检验：不变性目标确实可以提供有用信息

[一维可复现实验](research/identity_copy_toy_20260913/README.md)采用双峰真实分布、可精确积分的 Gaussian FM、具有明确欠拟合偏差的 strong/weak 模型。原图来自真实目标分布；公共反演被冻结。

在一维、正则单调 ODE 映射和正确公共逆下，用同一噪声坐标配对就是最优运输配对，因此

\[
\mathbb E_{X\sim P}\lvert G_\alpha(E^*X)-X\rvert^2
=W_2^2(G_\alpha\#\nu,P).
\]

这是一种确实成立的“逐图复制目标连接生成分布”的情形。两边是同一个数学量，不能计成两份独立证据。在高维一般只保留相应上界，不能直接推广等号或模型排序。

还完成了**不使用 oracle 反演**的版本：仅从 1024 个独立真实样本拟合平滑 CDF，以固定带宽规则构造 E_B；用 bank 外的 256 个真实样本，按单轮重建 MSE 选 CFG/AG 强度。三个 bank seed 都选出 CFG extra=1、AG extra=1.25。独立分位点积分得到：

| 生成器 | 生成分布 W₂² |
|---|---:|
| 未校准 strong | 0.108611 |
| 真实图复制校准 CFG | 0.056716 |
| 真实图复制校准 AG | 0.002373 |

这说明无需在校准时掌握真实分布解析式，复制目标也可能选到有益 guidance。这里的模型误差是人工构造的、具有可利用的相关性，不能把这个幅度外推到 SiT。它仍只是校准现有 CFG/AG 的可行性，不是超过充分调参 baseline 的新采样器。

相反，换成 strong 自己的精确逆，同一套复制目标就选 extra=0，保留已有分布偏差。五轮稳定性也不能独立给生成器排序：报告包含“错误收缩器多轮更稳定，但单次生成更差”的反例。

## 4. 下一步为何需要更好的外部参考

真实实验说明：只把一个普通 checkpoint 指定为逆，仍会把它的偏差写进复制任务。理论实验说明：真实数据锚定的逆可以让复制目标有正确方向。因此下一步集中在**如何获得足够可靠的公共编码，使‘保留原图’成为有价值的学习目标**，而不是继续盲调循环次数。

直接使用全维真实图 kernel bank 没有通过可行性检查：4096 维 latent 中，在两个类别、互不重叠的样本 bank 下，核责任权几乎落在单张参考图上，换 bank 显著改变估计。平滑虽能解除原子分布的反演奇点，却不能消除参考偏差。[公式与缓存实测](research/identity_operator_20260913/empirical_bank_encoder.md)。因此没有启动把该 bank 当作真值的 SiT 反演。

独立训练、更大架构的 **SiT-XL 公共反演已完成**，使用同一批 16 图、相同四个被测生成器及五轮协议。XL 原生时钟与当前 SiT-S 相反，已显式使用 v_aligned(z,t)=−v_XL(z,1−t)，并按 manifest 映射 100 类到原 ImageNet-1000 标签。仅用 XL 的 full conditional 输出；不用其内部弱头或 CFG。[训练路径、类别及 VAE 编码器兼容性核查](research/identity_operator_20260913/xl_reference_compatibility.md)已完成。

| SiT-XL 公共反演后的生成器 | R1 latent MSE | R5 latent MSE |
|---|---:|---:|
| 800K conditional | 0.09796 | 0.76942 |
| 400K conditional | 0.10088 | 0.93547 |
| CFG | 0.21524 | 1.77437 |
| AG | 0.10708 | 0.87839 |

[SiT-XL 参考下的 800K 五轮图册](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_xl/ref_xl/strong/rounds.png)中，第一轮已有明显结构变化，后续仍出现颜色和纹理漂移。该结果不支持“只换更大参考，就取得零编辑真值”。首四图单轮 128→256 步的 800K 输出差 MSE 为 1.14e−6，而对应复制 MSE 约 .0919；不能主要归于这一档积分误差。XL 本次只加密检查第一轮，未冒充五轮累计收敛检查。新增 320 个输出已与原始 NPZ 独立复算，见[核验结果](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_xl/analysis.json)。

基于这组结果，后续候选不再把参考模型当零编辑真值，而是**用原始真实图作为修复目标**：S800 仅反演到 .75/.5，再由 W400 返回图像端，形成两种 C_tau(x)→x 配对；另训练 x→x。学习轻量 P，部署只运行一次 P。该候选的监督、单一逆映射捷径和既有方法边界见[具体理由与反方](research/identity_operator_20260913/inverse_copy_refiner_rationale.md)。这是明确增加的生成修复假说。两版训练、P 自身五轮递归及 CFG/APG 1K 评估现已完成：单向版选回恒等，双向版四个设置均未改善 FID，见[最终结果](INVERSE_COPY_REFINER_RESULTS_20260913_ZH.md)；不扩大本候选到 5K。

若最终用复制损失校准 guidance，还必须关注 E_ref#P 与实际采样 prior 的相容性。仅看每条噪声的均值和范数不能确认这一点。重建与编码分布共同约束的思想已有 [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558)；[iCD](https://arxiv.org/abs/2406.14539)也将双向 preservation 与独立的 teacher consistency 训练结合，不能把 cycle 单独说成生成质量目标。

## 5. 可复现记录与当前边界

- [主实验代码](../experiments/fm_common_inverse_copy_20260913/run.py)、[输出核验/作图](../experiments/fm_common_inverse_copy_20260913/analyze.py)、[累计精度复核](../experiments/fm_common_inverse_copy_20260913/refine.py)、[XL 参考接口](../experiments/fm_common_inverse_copy_20260913/xl_reference.py)。
- [主实验冻结请求](/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2/request.json)。640 个输出的形状、标签、有限性及逐轮 MSE 已独立复算；R0 与输入缓存逐位一致。
- 最初启动在首次模型查询时遇到标签 int16 类型问题，尚无复制输出即退出；修正入口类型后在 `_v2` 新目录完成上述结果。原目录未覆盖，不混入结果。
- 这轮完成的是原始想法的操作实测、参考敏感性和正向可行性验证。没有 FID 收益结论，没有将课题目标缩减成“只需复制成功”，目标仍是获得有独立生成收益的方法。

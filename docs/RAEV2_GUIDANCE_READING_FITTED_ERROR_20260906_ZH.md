# Guidance 的数值误差：终端精确系数与 ERK 误差方向

日期：2026-09-06。**有限检索没有找到同时满足“无需新增幅度/手选窗口、机制成立、RAEv2 或 ImageNet 强基线同成本收益”的新方法。** 本轮只深读两篇最近的原始工作：fitted CFG 的系数确由指定结构唯一决定，但生成实证尚不覆盖 ImageNet/RAE；ERK-Guid 有 ImageNet 同 NFE 的正面结果，但实际增益和阈值仍手选。两篇计入阅读索引，但不进入本地新实验，也不改变正在执行的反射实验。

先核对了总表、旧 CFG++、密度去污染及极值曲率审计。ERK-Guid 已在旧 [literature/repo audit](INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md)被简要提及，本轮补正文、证明和作者实现；不能称作首次发现数值误差问题。

## 1. Fitted CFG：具体终端结构决定一个有限步系数

Shiheng Zhang，[arXiv:2607.07665v1](https://arxiv.org/abs/2607.07665v1)，*Guidance Breaks the Fitted Operator: A Terminal-Fitted Repair for Classifier-Free Guidance*，2026-07-08，14 页。全文及全部附录已读。随后取得作者 [官方仓库](https://github.com/shzhang3/fitted-cfg/tree/13899e5d1303dce03e89cf84e67891cce9d20915)的 [2026-07-15、18 页修订快照](https://github.com/shzhang3/fitted-cfg/blob/13899e5d1303dce03e89cf84e67891cce9d20915/paper/fitted-cfg.pdf)，题名改为 *Classifier-Free Guidance Breaks DDIM’s Terminal Exactness: A One-Coefficient Repair*；也读完其正文、证明及新增实验附录。以下保证和数值以该较新作者快照为准，不把两版本算作两篇，不称该快照为会议正式发表。

论文区分两类误差：连续 guided ODE 相对真实目标的分布误差，以及有限步 sampler 相对**该 guided ODE 自身**的数值误差。它只尝试修复后者，并不推导新的 clean 目标分布。

在 VE 坐标 \(x=X+\sigma\epsilon\) 下，设条件/边缘分布为中心 Gaussian，协方差可同时正交对角化，共同方向方差满足 \(0\le a\le b\)。定义

\[
A=\frac{\sigma^2}{a+\sigma^2},\quad B=\frac{\sigma^2}{b+\sigma^2},\quad
D_w=(1+w)D_c-wD_u,
\quad\frac{d\log|\xi|}{d\log\sigma}=\mu_w=(1+w)A-wB.
\]

这里实际常用 guidance scale 是 \(g=1+w\)。它导出三个不同结构：共享法向 \(a=b=0\) 的指数始终为 1；两方差都正的方向，终端指数趋向 0；**类内已塌缩、边缘仍有方差的方向 \(a=0<b\)**，指数趋向 \(1+w\)。最后一种才是本文要修复的额外刚性，不是所有“法向”或“大范数”方向。

在纯判别法向极限，精确一步收缩是 \(r^{1+w}\)，\(r=\sigma_{n+1}/\sigma_n\)。普通 DDIM+CFG 却给 \((1+w)r-w\)，粗步时可能换号。若保留相同两次 denoiser 查询，并限定更新族为

\[
x_{n+1}=D_c+r(x_n-D_c)+\alpha(r,w)(D_u-D_c),
\]

该极限下总倍率为 \(r+\alpha\)，故终端精确性唯一要求

\[
\boxed{\alpha(r,w)=r^{1+w}-r.}
\]

这是实质性的设计依据：**先给出一个应被有限步更新精确保持的动力学模式，再由它解系数。** 不需要估计方差大小、不增加 guidance 幅度或启动时刻。它与普通系数 \(w(r-1)\) 相差二阶项，保留同一连续 guided ODE 的一阶一致性。这里的唯一性仅限所写的标量更新族和终端模式，不是全部 sampler 中的唯一最优算法。

更强的条件性保证也确实存在。Hypothesis1 下

\[
G_{\rm fit}=(1-A)+rB+r^{1+w}(A-B)\ge0,
\]

因此不翻转这些共同特征坐标；严格正性取 \(0<r<1\)，零端点可以收缩到 0。在 \(a=0<b\) 的完整过渡段，累计归一化残差放大不超过 \(1+w\)。附录 A 的 Theorem2 对固定 log-noise 步长 \(0<h\le1\) 证明

\[
0\le\log(\xi_K^{\rm fit}/\xi_K^{\rm exact})
\le\frac{1+h}{4}w(w+2)h,
\]

并得到相对该精确 guided flow 的 W₂ 界。常数不依赖 \(\sigma_{\min}\)，却依赖 w。较新稿明确修正了初版较宽的 AP 措辞：固定 h 到达更低 noise floor 会增加步数，故**该 floor-uniform 误差界不等于完整采样配置的渐近保持定理**。一般非线性网络、非交换 covariance 或未满足方差顺序的方向，不继承这些保证。

正面实证与成本边界：

- 较新 Table2 在 CIFAR-10、\(g=9,N=8\)、50K 中，CFG→fitted 的 FID 为 **27.87→20.88**，target-class accuracy **95.12%→94.83%**；但 KID **.0108→.0177** 恶化。两套独立 5K 网格均报告 9/9 个设置的点估计 FID 改善，未声称每格统计显著。这是高强度修复，未与该模型最优低强度 CFG 作完整公平质量比较。
- 较新附录 H 已超过初版的小 smoke：两组固定 5K SD1.5，共享同一个条件/无条件 U-Net，\((g,N)=(12,12),(7.5,20)\)。saturation p95 为 **.10402→.00728、.06554→.02318**；CLIP 为 **.31452→.30638、.31361→.30964**，下降但满足作者预定的 .01 非劣界。它没有给出这两组 T2I FID 或 ImageNet/RAE 结果。
- 同 NFE 的 APG 对照使用从 SD2.1 转来的固定配置，两个设置各有优劣，不能称全面胜过 APG。相对 512 步 CFG 数值参考，fitted 的终点图像误差反而更大；低残差或低饱和度并不自动意味着更准确地积分了真实网络的 CFG。
- 附录 E 的五个系数插值仅作控制实验，没有将最小 FID 对应系数当主方法。越过理论系数可把一组 FID **25.22→21.27**，却使类别准确率 **94.84%→2.32%**。它说明目标不能仅由一个无条件质量标量识别，不是建议本地重做系数扫描。

作者代码身份为 `13899e5d1303dce03e89cf84e67891cce9d20915`。`coefficients.py` 直接使用 `r**guidance_scale-r`，`core.py` 对应上式，`ddim.py` 明确归一化 VP 坐标并限制 epsilon prediction、确定性 DDIM、无 thresholding。SD 示例每步一次 B2 U-Net 调用包含两支；README 的 12 NFE 实际是 12 次 B2 查询，不能与 12 个单支工作量混用，但 CFG/fitted 两臂确实复用相同调用结构。未发现额外 gain/window；原 guidance scale、网格与步数仍是输入。完整 CIFAR/5K 实验 ledger 未包含在这个小型作者仓库或 arXiv 源包中，因此已核对核心实现，未独立复核全部实证产物、总 wall 成本或所有公开表格的执行身份。

## 2. ERK-Guid：局部误差提供方向，但没有唯一确定部署系数

Kong 等，[*Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance*](https://arxiv.org/abs/2603.03692v1)，ICLR 2026 作者版，23 页。独立协作审阅覆盖正文 §3–5、附录 A/B/C、Heun 主 sampler、DPM/DEIS 扩展及复现实验脚本。OpenReview 下载受 403/浏览器验证限制，归档的是带 ICLR2026 页眉的 arXiv v1，不宣称与正式页面 PDF 字节一致。作者代码固定于 `ef4f1a7d41c74cac753159c4d10dc4fdf2e019ca`。

核心正向观察是：若局部 drift 近似自治线性系统，且误差被一个主特征方向支配，Euler/Heun 的已有差分可以指出 LTE 的主要方向，无需额外求完整 Jacobian。用同一噪声时刻的两个近邻状态及 drift 差，定义 \(e=\Delta x,d=\Delta f\)、\(\widehat\rho=\|d\|/\|e\|\)、\(v=d/\|d\|\)，向 Heun 解加

\[
-h\,\mathbf1[\widehat\rho>w_{\rm con}]
(w_{\rm stiff}h\widehat\rho)^2\langle f,v\rangle v.
\]

理想单主方向的精确 LTE 有指数函数表达式，其小步首项中的 **1/6** 来自 Taylor 展开；实际论文选择二次缩放、将幅度吸收进 \(w_{\rm stiff}\)，并另加门控阈值 \(w_{\rm con}\)。两者都不是定理给出的唯一值。作者 [reproduce.sh](https://github.com/mlvlab/ERK-Guid/blob/ef4f1a7d41c74cac753159c4d10dc4fdf2e019ca/scripts/reproduce.sh)实际扫描幅度；附录称 Table2 的阈值 .5，脚本在 16/8 步却设 .3/.1，复现身份存在具体差异。

命题 1 **预先要求** e 与主特征方向对齐，才推出 secant 商近似主特征值模；不能由“大 secant 商”反过来证明对齐。对非保守网络，一般 \(\|Jv\|\) 也不是谱半径；非自治 drift 的显式时间导数、DPM/DEIS 不同时刻差分还会带来额外项。保留的有用设计原则是“从已发生的有限步误差中估计纠偏信息”，不是把任何大曲率都认定为错误。

作者实际结果不能只取最醒目一行：

| ImageNet-512 配置 | FID 基线→ERK-Guid |
|---|---:|
| EDM2-S，Heun 32 步，最佳 FID 行 | 2.58→2.56 |
| 同上，8 步 | 7.06→4.91 |
| CFG，32 步 | 2.27→2.27 |
| AG，32 步 | 1.36→1.36 |
| CFG，16 步 | 3.61→3.20 |
| AG，16 步 | 2.32→1.93 |

各设置 50K；低步数有明显正向结果，强 32 步 CFG/AG 没有 FID 提升。32 步最佳 FD-DINOv2 行的 FID 反而为 2.74，不能把 DINO 变化当作 FID 收益。单 RTX3090、B1 的墙钟 **2.777→2.794 秒/图（约 +.61%）**，显存相同。对已使用 Heun 的 sampler，可复用两阶段查询而不增加 NFE；当前 RAEv2 是 Euler100，引入两阶段查询本身有成本，不能按“零额外 NFE”直接迁移。PixArt DiT 在该文为定性图，而非本表的定量底座。详见[独立原文与代码短审](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_fitted_error_guidance_v1/erk_guid/REVIEW_ZH.md>)。

## 3. 能给 RAEv2 什么启发，尚不能给什么

**最值得保留的是先确定需要修复的动力学模式。** Fitted CFG 用一个明确模式导出系数；ERK-Guid 尝试从已有两阶段误差识别模式。二者都比“弱分支更差，所以外推应该有用”更具体，但当前本地证据尚未接上。

1. **公共法向不等于判别法向。** RAE 真实 clean 共享仿射支撑的法向，在理想条件和无条件分布中都零方差，应对应 fitted 论文的 \(a=b=0\)，其指数仍为 1，gap 在该方向为零。不能据本地已知 affine normal 自动得到 \(1+w\) 指数。Full/Base 更是同类同目标双头，不能直接改名为 \(D_c,D_u\)。还缺的是预测误差的特定结构，而不是再证明存在法向。
2. **已知曲率结果不能代替缺失假设。** [极值曲率审计](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md)找到正对称 Rayleigh 方向，并测得这些方向上的显著非对称响应；它没有证明同时对角化、LTE 单主方向支配或该方向应被抑制。只用该缓存正曲率宣称 ERK 条件成立，会把不同对象混同。
3. **坐标决定有限步系数。** 在 \(0<s<t<1\) 且未触发 floor 时，RAE 可用 \(y=z/(1-t),\sigma=t/(1-t)\) 转为 VE，故论文的 ratio 应为 \(r=s(1-t)/[t(1-s)]\)，不是直接 \(s/t\)。形式上 fitted 更新仍落在当前 F−B 方向，等效外推系数为 \(w_{\rm eff}=(r-r^{1+w})/(1-r)\)。它趋向原 w 于小步，零 ratio 时趋于 0。这是有限步结构推导，不是新的误差方向；若缺少上述指数机制，照搬仍只是在减弱旧 gap。精确噪声端 t=1 的归一化坐标奇异，不能直接把作者有限 \(\sigma_{\max}\) 初态搬过来。

本轮因此不选择新幅度、时间窗口、求解器或候选实验。可迁移的问题是：**实际 guided drift 的哪种误差模式被当前有限步更新错误处理，而结构是否足以确定修复？** “更小 LTE”“保住潜空间符号”与“公平成本下至少 5% FID 降低”仍需要不同证据。本次也未用文献结论解释正在运行的反射实验结果。

来源、两版本全文、作者代码、局部审计快照及 SHA 见 [reading_fitted_error_guidance_v1/manifest.json](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_fitted_error_guidance_v1/manifest.json>)。仅执行下载、文本提取、阅读和归档；GPU、模型前向、训练、采样及作者测试执行均为 0。下载/转文本实际可测成本另记于来源日志，未统一计时的阅读与检索成本如实留空。

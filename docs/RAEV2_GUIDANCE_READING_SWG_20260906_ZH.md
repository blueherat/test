# SWG 阅读：可控信息缺失，不等于兼容的预测误差

日期：2026-09-06。本轮只阅读论文、作者代码并归档；没有运行作者代码、模型、噪声模拟、GPU、采样或 FID，也没有修改已冻结的 decoder query-mean runner/tests。

**值得保留的是“负参考应放大同一种误差”的问题意识；SWG 没有证明 crop、权重衰减或当前 decoder JQ 必然产生这种误差。** 它提供解析 toy model、真实图像实验和可解释的空间干预，但实际指导强度、组合权重与时间窗口仍依赖实验选择。不能把其 FDD 最优结果当成当前任务的公平成本 FID 保证。

## 版本与实际读取范围

- [arXiv v1](https://arxiv.org/abs/2411.10257v1)，2024-11-15，19 页，题为 *The Unreasonable Effectiveness of Guidance for Diffusion Models*；作者为 Tim Kaiser、Nikolas Adaloglou、Markus Kollmann，前两位标注共同贡献。
- [arXiv v3](https://arxiv.org/abs/2411.10257v3)，2025-08-29，30 页。摘要页短标题为 *Guiding a diffusion model using sliding windows*；PDF/HTML 正文标题包含 *with itself*。
- [BMVC 2025 正式记录](https://bmvc2025.bmva.org/proceedings/26/)和[正式正文](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)：*Guiding a diffusion model with itself using sliding windows*，Nikolas Adaloglou、Tim Kaiser、Damir Iagudin、Markus Kollmann。正文 14 页、[独立 supplementary](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/supplementary.zip) 18 页。正式记录 BibTeX 为第 36 届 BMVC 2025；页面小标题残留“35th”，不据此改变会议身份。
- 作者链接明确指向 [HHU-MMBS/swg_bmvc2025_official](https://github.com/HHU-MMBS/swg_bmvc2025_official)。本次固定 commit `68b141675d3705710595b086e86615ebe127c539`，提交时间 2026-01-05；这是本次取得的代码版本，不保证等于论文实验当时的代码。

读取：v3 §§1–6、Appendix A 的全部实验/成本/WD 设置与 SWG 伪代码、Appendix B 的 posterior/score 推导和 Euler 算法；逐项核对正式正文对应机制、主表及正式 supplement §§1–2。v1 只读标题摘要、机制段、WD 与 SWG 方法及 §5.3 消融，未声称重新精读其全部旧版附录。视觉检查 v3 PDF p4 的 toy 几何、p26 的 oracle 轨迹图，以及正式 supplementary p3 的成本/WD 表。其余大幅图像主要读 caption，未做人类偏好复评。

作者代码实际阅读的是 toy posterior/score/solver 与绘图调用、EDM2/DiT crop/mask/组合分支、位置编码插值、采样入口/Optuna 参数段、WD optimizer 设置及 README/有关启动脚本；notebook 提取 source cells 后阅读，没有执行输出。旧仓库两篇文档曾引用此 arXiv，本次是有正式版本和源码核对的补充，不能记成此前从未提及的论文。

## 理论成立在哪里

论文先规定一个人为但解析可解的错误族：真实分布 P 经各向同性 Gaussian convolution 得到 Pδ。令 σ̃²=σ²+δ²，则 posterior clean prediction 满足

\[
y_\delta(x,\sigma)=\frac{\delta^2x+\sigma^2y^*(x,\tilde\sigma)}{\sigma^2+\delta^2},\qquad
\epsilon_\delta(x,\sigma)=\frac{\sigma}{\tilde\sigma}\epsilon^*(x,\tilde\sigma).
\]

这来自 Gaussian 卷积和条件均值恒等式；正式 supplementary §2 / v3 Appendix B Eq.19–23 是推导主体。它证明该规定分布对应怎样的 denoiser，**没有证明训练较短、WD 较强或感受野较小的神经网络等价于增加 δ**。三点数据/点云上的轨迹与 oracle weight 图是这个解析族的数值示例，并非任意 diffusion 网络的误差消除定理。[v3 原文](https://arxiv.org/html/2411.10257v3)

下面是独立补全的向量条件，避免把正文“正模型更接近最优”读成充分条件。记 e=εpos−ε*，d=εpos−εneg，则

\[
\|e+wd\|^2-\|e\|^2=2w\langle e,d\rangle+w^2\|d\|^2.
\]

存在正的小 w 能改善同状态平方误差，当且仅当 ⟨e,d⟩<0；仅有 ‖εneg−ε*‖>‖e‖ 不够。例如 ε*=0、εpos=(1,0)、εneg=(0,2)，负模型误差更大，但 e·d=1，任何 w>0 都使误差增加。

若 εneg−ε*=αe 且 α>1，才有 d=−(α−1)e，以及能精确消除误差的 w=1/(α−1)。此时论文 Eq.4 的 ‖e‖/‖d‖ 正确。一般向量的平方误差最优无约束标量应为 −⟨e,d⟩/‖d‖²，需限制 w≥0 时再取非负部分；范数比缺少夹角。论文紧接 Eq.4 明确限定“误差只差乘法因子”，也承认固定常数指导强度通常是 ad hoc；不应把作者这个限制删除后批评其声称了一般定理。[正式正文 §3](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)

另一个源码细节：当前 `toy_math.py:compute_score` 的 oracle 分支对整个 `[n,D]` 数组调用无 axis 的 `numpy.linalg.norm`，所得权重为整批共享标量，再广播给各条轨迹；它不是一般逐样本的 w*(x,t)。`utils.py:optimal_weights` 对已广播的 weight history 求平均不会恢复逐轨迹 oracle。这限制的是当前发布实现与点态符号的一致性，不据此断言论文当时的图一定由该 commit 产生。[固定源码](https://github.com/HHU-MMBS/swg_bmvc2025_official/blob/68b141675d3705710595b086e86615ebe127c539/toy_example/toy_math.py#L185)

## stronger weight regularization 的证据强度

WD 负模型保持架构，额外 fine-tune 或重新训练。正式 supplementary Tables 2–4 给了训练量、λ 和最终生成指标；未报告真实高维 Bayes error 的夹角、共线比例或逐时刻风险导数。由 WD 增加 bias 推出“同方向 bias 放大”仍缺一步：参数正则并不一般保留每个输入处的函数误差方向。

EDM2-S ImageNet-512 的对照如下；每列分别选其最优 guidance 权重，**同一行 FID/FDD 不一定属于同一次采样设置**。

| 负参考方式 | FDD | FID |
|---|---:|---:|
| 无 guidance | 112.2 | 2.92 |
| Reduced training | 46.5 | 1.79 |
| Reduced capacity + training | 42.1 | 1.67 |
| WD fine-tuning | 52.1 | 2.46 |
| WD + reduced training | 43.9 | 2.25 |

WD fine-tune 使用 λ=2e−5，正模型已见 2147M 图；FDD 选出的负模型见到 2233M、FID 选到 2243M，即追加约 86M/96M 图。WD+RT 的负模型重新训练到 63M。小规模 CIFAR/FFHQ 还展示 WD retrain，但那是新训练负模型。v1 §5.3/Table 3c 另扫 λ=1e−5…5e−4，并对各 λ 优化 fine-tune 时长和 guidance weight；因此“WD 也有效”有实证依据，“无需调参且必然制造兼容误差”没有。[正式补充材料](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/supplementary.zip)、[旧版消融](https://arxiv.org/pdf/2411.10257v1)

## SWG 与 decoder JQ 的本质区别

SWG 对当前 noisy image/latent 取若干不缩放的 crops，每个 crop **从整网输入开始独立前向**，再按原位置累加、对重叠计数平均。M-SWG 只在覆盖次数>1的位置应用正负差。默认四个 crops、约 5H/8 边长来自 U-Net 下采样整除约束和经验；mask 本身也是实验发现，不是由兼容误差定理导出。DiT 额外插值位置编码以容纳裁剪输入。[正式正文 §4](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)

| 对象 | SWG/M-SWG | 当前固定 decoder JQ |
|---|---|---|
| 删除的信息 | 每次 crop 前向看不到 crop 外的 noisy latent | 每个 head 不再按空间 query 位置选择不同 attention 行 |
| 全局计算 | 每个 crop 的所有层重新计算 | 28 层全局 encoder 已共享完成，仅重放两层 decoder |
| keys/values | 仅来自当前 crop | 每个弱 block 仍使用全部 256 个空间 keys/values |
| 位置作用 | 裁剪边界与位置编码插值改变输入几何 | 对 post-RoPE Q 取 JQ；保留全图 key 位置结构 |
| 严格可说的性质 | 当前弱预测只依赖覆盖该输出位置的 crops 所包含的输入；远处影响按这些 crops 的并集决定 | Q 的行常量 Frobenius 投影；无 mask/bias 时 attention 为 reverse-KL 重心 |
| 不能直接说的性质 | “远程依赖正好是 Full 的主要误差”或“crop 放大相同误差” | “截断全网远程感受野”或“JQ 放大 Full 的同方向误差” |

特别是重叠区域的多个 crop 并集可能覆盖很大范围，SWG 的每个输出也不都具有统一的 k×k 感受野。JQ 的平均 query、全部 keys，以及全局 encoder 产生的 conditioning 则都还能携带远程信息。两者同属构造结构弱参考，但不能互换其解释。当前 JQ 的局部保证及执行范围见[本地冻结协议](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md)。

## 成本、调参与质量证据

主表用 50K 样本，**论文主指标是 FDD**。正式正文 Table 1 的 EDM2-XXL：无 guidance FID=2.29，SWG=2.61，M-SWG=2.46；FDD 则从 49.7 改善到 37.7/36.8。其“强基座上仍有价值”的证据主要落在 FDD/IS，并未兑现同幅度 FID 改善。DiT 的无 guidance 到 M-SWG 有较大改善，不能等同于在当前已强 IG 的 RAEv2 上再改善 5%。[正式正文 Tables 1–2](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)

正文与附录公开做了窗口大小/数目消融、分别针对 FID/FDD 的 gain search、两种 guidance 的权重组合搜索；额外 interval 实验在 32 步里手选 15–20（SWG）或 13–23（M-SWG）等，并重新选 gain。作者代码也有 Optuna 入口（默认 10 trials、离散步长 .025）。这不抹去结构的简洁性，但不符合本研究“不能靠大量 gain/window 搜索获胜”的直接部署约束；本次没有迁移任何数值参数。[源码搜索入口](https://github.com/HHU-MMBS/swg_bmvc2025_official/blob/68b141675d3705710595b086e86615ebe127c539/edm2/sample_edm2.py#L580)

正式 supplementary Table 1：2 张 A100 40GB、生成 512 样本的报告时间为 EDM2-S 无指导 1.36 min、CFG† 1.61、SWG(r=.4) 2.10；DiT 无指导 4.61、CFG 8.45、SWG 11.37。由表内时间计算，SWG 比 CFG†/CFG 分别慢 **30.4%/34.6%**；比无指导慢 54.4%/146.6%。前者与正文“少于 30%”有轻微不一致，不应取口号覆盖表值。没有同 wall time 增加官方采样步数的质量对照，也没给组合方法的完整吞吐表。窗口虽共享权重，仍是额外前向。[正式成本表](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/supplementary.zip)

该表的 images/s/GPU 列还有计数口径不明：若“512 samples”指全体两卡合计，则通常每卡吞吐是 512/(分钟×60×2)，表中数值约是它的四倍；若样本数另有所指，正文未在该表说明。因此这里只用同表时间比，不转述其每卡绝对吞吐为独立核准值。没有原始 timing logs，不能修订作者表格。

## 作者代码不能静默当作正式实验真值

以下均为固定 commit 的静态事实；未运行、未修改，也不据此判原论文结果无效。

1. `edm2/sample_edm2.py:138–146` 的 `wmg-swg/cfg-swg` 在进入 mask 分支前已 return；`scale_window_guide=1` 不会让该组合变成 M-SWG。`cfg-swg` 还把 labels 改为 None 后传给 positive crop。README 称该入口可复现组合 M-SWG，与目前代码不完全一致。独立 `swg` 分支确实有 `(overlap>1)` mask。[固定代码](https://github.com/HHU-MMBS/swg_bmvc2025_official/blob/68b141675d3705710595b086e86615ebe127c539/edm2/sample_edm2.py#L117)
2. DiT 的 mask 在独立/组合 crop 分支中都确实实现了。但 vanilla CFG 按原代码默认只指导三个 ε 通道；`forward_with_cfg_crop` 指导四个 ε 通道、保留 variance 部分；独立 `forward_with_crop` 则混合整个模型输出。不能把这些默认入口当成只有弱参考不同的消融。[固定 DiT 实现](https://github.com/HHU-MMBS/swg_bmvc2025_official/blob/68b141675d3705710595b086e86615ebe127c539/edm2/diffusion/models.py#L324)
3. 作者 README 的复现表也保留了差异：DiT M-SWG w=.5 的 FID=5.97，正式正文对应最优 FID=3.30；组合结果行还把 DiT CFG 组合标成 RCT。不能静默任选更好的数字。当前代码/配置和正式评估对应关系需要作者原始运行记录才能进一步闭合。[作者复现表](https://github.com/HHU-MMBS/swg_bmvc2025_official/blob/68b141675d3705710595b086e86615ebe127c539/edm2/README.md#L157)

## 留下的一个可证伪问题

**结构删除产生的 Δ=F−W，是否针对 Full 的预测偏差，而不只是删除了 Full 本来正确提取的信息？** 这比检查弱参考 standalone 质量差、Δ 非零或与 F−B 有正余弦更接近本文的兼容误差机制。

独立推导：在固定真实 bridge joint distribution 下，令 X 为 clean latent，μ(z,t,c)=E[X|z,t,c]，F、W 为输入可测的确定预测，则

\[
\left.\frac{d}{dw}E\|F+w\Delta-X\|^2\right|_{w=0}
=2E\langle F-X,\Delta\rangle
=2E\langle F-\mu,\Delta\rangle.
\]

若 Full 已等于 μ，则该一阶项为零，任何非零 Δ 的固定正 gain 只增加 w²E‖Δ‖² 的同分布 denoising risk。特别地，即使 W 是更少信息下的精确条件均值，也不能凭“信息较少”保证 guidance 降低该风险。这不是否定 guidance 的生成价值：条件均值风险、轨迹动力学、终点分布和 FID 是不同对象；它说明保证必须落在明确的误差机制或分布目标上。

当前固定 JQ pilot 不读真实 X，**其输出方向几何无法识别上式符号**。若今后研究该命题，需独立冻结适当 joint distribution 与身份，teacher bridge 的配对 clean X 才有直接解释；历史 rollout 不能把最初那张 X 不加论证地当成当前状态的 posterior target。本条是机制问题，不是本轮新增试验授权，也没有推导部署 gain 或时间调度。

## 归档

目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_swg_v1/`。保留下载 UTC、URL、bytes、SHA，PDF/HTML/正式 supplementary、arXiv TeX source、固定 commit 完整源码包及解包文件、文本提取、notebook source cells 和上述三张 PDF 渲染。

| 文件 | SHA256 |
|---|---|
| `arxiv_v1.pdf` | `2e9d13ae980a1f0504a646694c5b42bbbe3af192294233e5bdaece7fd9b2ae0d` |
| `arxiv_v3.pdf` | `5a844348436b8814ab5d622126e5f0358ce04d673c264c686e92befce51e19d2` |
| `bmvc_paper.pdf` | `419e316bca322a9d2d05300cb32452d7789ce6dd15d6f80b804436072cbcc1d7` |
| `bmvc_supplementary.zip` | `4f0a578a363ebdc71b9d9f4a213269ced3eeeb6730d71c8e27f20263f346b45d` |
| `arxiv_v3_source.tar` | `76182129bae87755de7770520e9c88b70e67db64eda918f49c031df2caebaa6a` |
| `author_code.tar.gz` | `f4143e6089733e563164f94881f59f81111837f5624860fc05662eac970d7812` |

逐个阅读源码 SHA 在 `read_author_sources.json`；所有归档文件身份及读取边界在 `manifest.json`。本轮未用额外实验为论文补写没有给出的保证。

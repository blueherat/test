# RAEv2 水平翻转平均：必要前提 CPU 审计

日期：2026-09-06。状态：**当前证据不足以放行具有对称性风险保证的 guidance 候选；未测试生成质量。**

本次只读取真实 DINOv3 权重、官方 K7 normalization statistics 和源码，并在 CPU 上
计算代数残差。没有运行 encoder/model forward、GPU、采样或 FID 评估。机器可读结果、
完整路径、文件大小及源码哈希见
[symmetry_prerequisite_audit.json](data/raev2_guidance_restart_20260906/symmetry_prerequisite_audit.json)。

## 1. 理论保证需要什么

令 $P$ 为水平空间翻转，$P^2=I$。如果 $P$ 是正交作用，真实联合 clean/noisy 分布
对 $P$ 不变，则真实 Bayes denoiser $m_t$ 等变，Reynolds 平均

$$
\Pi f_t(z)=\tfrac12\left[f_t(z)+P f_t(Pz)\right]
$$

是对应 $L^2$ 空间的正交投影，并满足

$$
\|f_t-m_t\|_{L^2}^2-\|\Pi f_t-m_t\|_{L^2}^2
=\|f_t-\Pi f_t\|_{L^2}^2\ge0.
$$

系数 $1/2$ 由有限群 Haar 平均确定。但上述前提不能仅由图像翻转看起来合理、
预训练使用了数据增广，或模型存在空间 token 而推出。相关扩散理论明确区分群作用、
不变分布和等变 Gaussian kernel；见
[Target Symmetrization](https://arxiv.org/html/2502.09890v1) 与
[Diffusion Models under Group Transformations](https://proceedings.mlr.press/v258/lu25a.html)。
降低 denoising/score 误差或其导出的分布误差上界，也不保证当前有限步 Euler+IG 的实际
FID 单调改善。

## 2. 官方训练与 encoder 的前提边界

当前 `dataset.type=hf`。`external/RAEv2/src/train.py` 虽定义包含
`RandomHorizontalFlip` 的 `stage2_transform`，但对 `hf/wds` 不传入它；HF loader
默认只有 `Resize + ToTensor`。此前官方 Arrow 像素审计也确认这一点，见
[严格续训协议](RAEV2_LPL_STRICT_CONTINUATION_ZH.md)。因此官方 Stage-2 augmentation
不能在这里作为目标 latent 分布具有翻转不变性的证明。

K7 使用 `DINOv3MultiLayerSimpleAddEncoder`：七层归一化 patch tokens 取均值，再加
最后一层的 token 空间均值广播。该聚合保持已经存在的等变性，不会自动创造等变性。
真实 `patch_embed.proj.weight` 形状为 `[1024,3,16,16]`，其水平核翻转残差为

$$
\frac{\|W-\operatorname{flip}_h(W)\|_F}{\|W\|_F}=1.3662688732.
$$

所以“图像翻转只导致 patch-grid 置换”不在 patch projection 层精确成立。该结果
**不单独否定最终 DINOv3 encoder 可能学到近似等变性**；本次没有测量最终 encoder 输出。

## 3. 空间相关标准化不交换翻转

官方 `stats.pt` 的 mean、variance 都是 `[1024,16,16]`，并非空间共享的逐通道常量。
记 $\sigma=\sqrt{\mathrm{var}+10^{-5}}$，$N(z)=(z-\mu)/\sigma$。
原始 latent 空间翻转在标准化坐标中的共轭作用为

$$
T=N\circ P\circ N^{-1},\qquad
T(y)=\frac{P\sigma}{\sigma}\,Py+\frac{P\mu-\mu}{\sigma}=Ay+b.
$$

它精确满足 $T^2=I$，但不因此成为正交线性作用。实际 CPU 结果如下。

| 检查量 | 数值 |
|---|---:|
| $\|\mu-P\mu\|_F/\|\mu\|_F$ | 0.1293376684 |
| $\|\mathrm{var}-P\mathrm{var}\|_F/\|\mathrm{var}\|_F$ | 0.0301094260 |
| $\operatorname{RMS}(b)$ | 0.4239512384 |
| $\max|b|$ | 23.6162281036 |
| $(P\sigma)/\sigma$ 最小值、最大值 | 0.7827230692、1.2775911093 |
| $\|A^\top A-I\|_F/\sqrt d$，$d=262144$ | 0.0239148289 |

因此 $T$ 不保持标准高斯；平移项还导致 $T(az)-aT(z)=(1-a)b$，不能直接把它
作为当前标量线性 bridge 的等距群作用。朴素标准化 spatial flip $P$ 虽保持标准高斯，
却不等于原始 latent flip 的共轭作用。

这排除了从 raw spatial flip 与官方 normalization 直接得到所需定理前提的路径。
它**不逻辑排除**标准化 latent 分布偶然对朴素 $P$ 不变；若重新研究，应独立检验
该分布及对应风险，不应先假定对称性。

## 4. 历史证据与本次裁决

旧 DINOv2 的 5196 图实验中，final `flip_h` direct relative error 为 `0.4345`，
证明有空间对应，但不是等变恒等式；详见
[layerwise correspondence 结果](RAE_LAYERWISE_CORRESPONDENCE_RESULTS_ZH.md)。旧等变
adapter 与 inverse-adapter 没有建立生成收益，见 `docs/RESEARCH_STATUS.md`。
这些结果不能代替本次 DINOv3 K7 的前提检查，也不是 RAEv2 inference-only Reynolds
averaging 的质量反证。

本次裁决仅为：**现有必要前提不足，暂不启动这一 guidance 候选。** 若继续，只做
真实 paired encoder/denoising 风险审计，并固定完整官方时间网格与群平均系数。
不能把本记录写成“水平翻转平均已使 FID 失败”。

## 5. 文件校验

| 文件 | SHA-256 |
|---|---|
| DINOv3-L16 pretrained checkpoint | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| 官方 K7 `stats.pt` | `40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77` |
| 本地 official-compatible YAML | `3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342` |

全部数值重新从上述文件计算；其他相关源码的完整 SHA-256 保存在配套 JSON。

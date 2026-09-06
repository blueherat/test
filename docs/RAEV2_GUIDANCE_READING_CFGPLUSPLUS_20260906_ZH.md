# CFG++ 精读：重组、时间权重，以及迁移 RAEv2 时必须补上的机制

日期：2026-09-06。状态：完成正文、正式版附录和作者代码核对；只做一次 CPU 代数审计，模型调用、GPU、训练均为 0。本文不修改已冻结实验、质量结论或研究总状态。

**最有用的发现是：ICLR 2025 正式版已经把 CFG++ 延伸到 flow matching，并明确承认其一阶形式等价于随时间重加权的 CFG。** 对 RAEv2 的 clean 双头直接照搬，不会自动产生一个新的方向或流形投影。值得保留的机制问题是：条件修正被下一步残差重组抵消了多少，承担运输的参考头是否具有更小的时间误差，以及这个差异是否与真实目标误差一致。以下将论文结论、代码事实与我们的推导分开。

## 1. 版本和证据来源

仓库已有材料引用 CFG++，本次检索未发现独立的完整正文精读。新增阅读计为一篇，不把版本、附录或作者代码重复计数。

| 材料 | 本次固定版本与用途 |
|---|---|
| [arXiv:2406.08070v2](https://arxiv.org/abs/2406.08070v2) | 2024-09-12，25 页；核对原始主文及版本差异 |
| [ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/4c9477b9e2c7ec0ad3f4f15077aaf85a-Paper-Conference.pdf) | 27 页，PDF 元数据生成时间为 2025-02-28；本笔记以此为准。新增附录 B（flow matching）和 D（reweighted CFG）是本次重点 |
| [OpenReview 入口](https://openreview.net/forum?id=E77uvbOTtp) | 浏览器校验/403；未取得评审正文，不声称读过评审。正式论文由 ICLR proceedings 获取 |
| [作者代码](https://github.com/CFGpp-diffusion/CFGpp/tree/8035352527eb01ac69f65feaab374ba50616ce2c) | commit `8035352527eb01ac69f65feaab374ba50616ce2c`，提交时间 2025-03-21 01:09:11 UTC；下载指定原始文件，没有运行作者采样器 |

档案目录：[reading_cfgplusplus_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_cfgplusplus_v1)。包含 PDF、提取文本、作者源码、下载日志、CPU 审计及 `reading_manifest.json`。关键 SHA256：

```text
arxiv_v2.pdf
5c4bc720e2e4e8f7b36d8b17af5b0805d1b925fb6738be274b2c9327ffda5e2b
iclr2025_proceedings.pdf
c1da9f768ed271db4d164fa2b1859d1b223f3a9e10782766a634d114331202ef
author_source/latent_diffusion.py
c90053b1565314b0530a1a43c0e04fc74288c45cb688d9cc791efffe8d1509d5
author_source/latent_sdxl.py
883c433d5977af2ebb27af5968f05e671790f9af1e3f47817366f9b28de4c219
```

## 2. 主公式：修改的是哪一项

令当前噪声时刻为 $t$，下一步为 $s<t$，VP forward bridge 为 $z_t=a_tX+b_t\epsilon$，其中 $a_t=\sqrt{\bar\alpha_t},b_t=\sqrt{1-\bar\alpha_t}$。定义

$$
D_u=(z_t-b_t\epsilon_u)/a_t,\quad D_c=(z_t-b_t\epsilon_c)/a_t,
\quad \Delta=D_c-D_u,\quad D_w=D_u+w\Delta.
$$

标准确定性 DDIM 与 CFG++ 分别是

$$
z_s^{\rm CFG}=a_sD_\omega+b_s\epsilon_\omega,
\qquad z_s^{++}=a_sD_\lambda+b_s\epsilon_u,
\qquad \epsilon_w=\epsilon_u+w(\epsilon_c-\epsilon_u).
$$

这里的“加回噪声”是已有预测量的确定性重组，不是再抽取高斯随机数；两个分支在同一步复用。论文采用 $D_u+\omega(D_c-D_u)$ 的约定，故 $\omega=1$ 是 conditional；若另一实现写成 $D_c+g(D_c-D_u)$，则 $\omega=1+g$。来源：正式版 §3.1、Algorithm 1–2；[作者 DDIM CFG++](https://github.com/CFGpp-diffusion/CFGpp/blob/8035352527eb01ac69f65feaab374ba50616ce2c/latent_diffusion.py#L618)。

**以下等价式是我们由上式直接推导。** 设 $r=b_s/b_t$，$q=a_s-ra_t$，则

$$
z_s^{\rm CFG}=rz_t+qD_u+q\omega\Delta,
\qquad z_s^{++}=rz_t+qD_u+a_s\lambda\Delta.
$$

因此同一状态、同一网格上的有效 CFG 系数为

$$
\boxed{\omega_{\rm eff}(t,s)=\lambda a_s/q}.
$$

它不是一个与步数、scheduler 无关的常数对应。论文中的 $\lambda=.6\leftrightarrow\omega=7.5$ 是 50 步 DDIM 上用同 seed 图像的 LPIPS 搜索得到的经验对应，不能移植成 RAEv2 的参数。正式版 §5、附录 D 也明确讨论了重加权解释；上式保留 $a_t,b_t$，避免附录排版中 $\alpha_t$ 与 $\bar\alpha_t$ 混用造成的歧义。

## 3. 理论给了什么，哪些条件仍未给出

正式版 §2–3 把 text guidance 看成在 clean manifold 上降低条件去噪损失，并借助 DDS 将 noisy-space 梯度改成 clean-space 修正。原损失写为

$$
L_t(x)=\|\epsilon_\theta(a_tx+b_t\epsilon,c)-\epsilon\|^2
=\frac{a_t^2}{b_t^2}\|x-D_c(a_tx+b_t\epsilon,t)\|^2.
$$

把条件目标在当前点冻结后，从 $D_u$ 做一次二次 surrogate 梯度步得到 $D_u+\lambda(D_c-D_u)$，其中

$$
\lambda=2(a_t^2/b_t^2)\gamma_t.
$$

这解释了优化步长 $\gamma_t$ 到混合系数 $\lambda$ 的换算。它没有唯一确定 $\lambda$，更没有推导主表中的经验 $\lambda\leftrightarrow\omega$ 对应。原文的 MCG→DDS 迁移需要局部线性流形及进一步条件；没有计算这些模型的流形投影算子。来源：正式版式 (7)–(13)。

我们核对得到的适用边界：

- Tweedie 是 Gaussian corruption 下的条件均值恒等式；实际网络需要接近相应分布的正确 score/denoiser。条件均值本身不必在非凸数据支撑上，低密度处或多模态之间尤其如此。
- “两点之间插值不离开流形”只对共享的凸/仿射局部区域成立。两个不同 piecewise-linear chart 上的点，线段可离开二者；例如两条坐标轴上的 $(1,0)$、$(0,1)$，中点不在轴的并集上。反过来，单个完整仿射子空间上的外推也仍在子空间内。
- 从上面的原损失求精确梯度时，$D_c$ 依赖 $x$，必须保留 Jacobian。论文主动采用绕开 Jacobian 的 DDS/SDS 近似，不能把其更新当成任意网络原损失的严格下降保证。CPU 反例取 $D_c(x)=2x$：原损失是 $x^2$，而冻结目标的梯度是 $-2x$，正步长使原损失上升。这个例子说明额外假设必要，不声称训练好的模型必然具有该坏例子。
- $\lambda\in[0,1]$ 只保证混合在两个估计之间；它不证明输出分布正确、无 mode loss 或 FID 下降。20 NFE 实验还允许 $\lambda\ge1$，因此主文的插值几何不能覆盖所有报告设置。

论文的可取之处是从 denoise/transport 两个角色组织设计，再测量轨迹与 inversion；它没有给出 RAEv2 或通用 FID 的定理。

## 4. 迁移到 RAEv2：两种自然读法并不相同

本仓库配置为 $z_t=(1-t)X+t\epsilon$，预测 clean；$F$ 为 full head，$B$ 为同一类别的 base head，$\Delta=F-B$。官方 IG 在 $t\in[.1,1]$ 使用 $G=B+1.78\Delta$，其他时刻用 $F$。100 步 shift=8 的最小正查询时刻约 .074766，大于分母 floor .05，故该网格上实际 Euler 恰为

$$
z_s=z_t-h\frac{z_t-G}{t}=rz_t+q(B+\omega\Delta),
\quad h=t-s,\ r=s/t,\ q=h/t.
$$

来源：[配置](/home/zhoushunyu/eqvae/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml)、[官方 transport](/home/zhoushunyu/eqvae/external/RAEv2/src/stage2/transport/transport.py:67)、[官方 IG](/home/zhoushunyu/eqvae/external/RAEv2/src/utils/guidance_utils.py:43)、[官方 Euler](/home/zhoushunyu/eqvae/external/RAEv2/src/stage2/transport/sampler.py:13)。

### 4.1 按正式版附录 B 的 flow 公式移植

原文由 $D_\theta=z_t-tv_\theta$ 把 flow ODE 写成 $(z_t-D_\theta)/t$，然后沿用 VE Euler 的重组。换成 $F,B$ 后直接得到

$$
z_s^{++\!,\mathrm{flow}}=(B+\lambda\Delta)+\frac{s}{t}(z_t-B)
=rz_t+qB+\lambda\Delta.
$$

因此

$$
\boxed{\omega_{\rm eff}=\lambda t/h;\quad \lambda_k=\omega_kh/t\ \Longrightarrow\ \text{恢复原 Euler IG}.}
$$

附录 B 的 $1/t^2$ clean loss 若沿用冻结目标优化，对应 $\lambda=2\gamma_t/t^2$；它依然没有决定全局增益或保证其适合 F/B。

这里留下的不是额外投影，而是随步长改变的增益。固定 $\lambda$ 时，非零 $\Delta$ 的每步增量不随 $h\to0$ 缩小；一般不能解释为一个固定有限 guidance ODE 的一致离散化。100 步 shift=8 取 $\lambda=.6$，首步 $\omega_{\rm eff}=475.8$，末步 .6，显然不是可直接照抄的“温和 guidance”。

反过来，按官方区间令 $\lambda_k=\omega_kq_k$，全部 100 步都有 $\lambda_k\le1$（末步 IG 关闭，$\lambda=1$）。**同一原始采样器已经可写成附录 B 的插值式。** 这说明只观察公式里出现插值系数不足以区分有效机制。该换算是推导出的恒等关系，不是建议手调时间表。

### 4.2 按真正 Gaussian bridge 的 clean/noise 重组移植

若“保留 prior noise”指物理 forward noise 估计，应先定义

$$
\epsilon_B=\frac{z_t-(1-t)B}{t}.
$$

随后按 DDIM 的角色分工重组，得到另一个公式：

$$
z_s^{++\!,\mathrm{bridge}}=(1-s)(B+\lambda\Delta)+s\epsilon_B
=rz_t+qB+(1-s)\lambda\Delta.
$$

故 $\omega_{\rm eff}=\lambda(1-s)t/h$。它与附录 B 相差因子 $1-s$，原因是 flow velocity $(z_t-B)/t=\epsilon_B-B$ 并不是 Gaussian noise $\epsilon_B$。也可在 $t<1$ 通过 $y=z/(1-t),\sigma=t/(1-t)$ 的 VE 坐标推导此式；端点用上面的有限 bridge 公式。

对当前 100 步网格，以剩余步数 $n=100,\dots,1$ 表示，

$$
\frac{(1-s)t}{h}=\frac{n(101-n)}{100}.
$$

因此固定 $\lambda=.6$ 的有效系数是首末 .6、中间最高 15.3。**两种移植都完全落在当步 $F-B$ 的标量 span 内，只是诱导不同的时间权重；二者的差别也说明“自然移植”还需要固定坐标和 metric。** 若改变步数到触发 .05 floor，必须使用真实分母重推，本文不把该恒等式无条件外推到其他网格。实数恒等也不等于 BF16 不同括号次序的逐位相等。

### 4.3 $B$ 不是 unconditional

F/B 共享相同类别条件及同一真实 clean 训练目标；B 是更浅的内部估计。若二者都达到同一 Bayes 条件均值，$\Delta\to0$。这与 conditional/unconditional 的 gap 不同：后者在两个正确模型之间仍包含真实类别信息。不能把弱头天然称作无条件 prior，亦不能由论文推断其残差适合运输。

一个直接的可检验限制是：对真实 forward 样本 $(X,\epsilon,z_t)$，

$$
\epsilon_B-\epsilon=-\frac{1-t}{t}(B-X),\qquad
\epsilon_F-\epsilon=-\frac{1-t}{t}(F-X).
$$

在 $0<t<1$ 的同一时刻，clean MSE 与 noise MSE 只差正标量；$t=1$ 时两者的重建 noise 都等于输入，不能比较其 clean 精度。于是“F 更会估计 clean、B 却在同一点天然更会估计真实 noise”并非两个独立优势；内部时刻的相同 L2 目标上两者排序相同。若 B 真有用途，理由须来自时间一致性、误差相关结构或受保护的分布量，而不是给同一误差换名字。作者示例默认还把 `null_prompt` 设成负面描述，说明实际参考分支连严格无条件语义也不能仅凭变量名认定。[作者示例](https://github.com/CFGpp-diffusion/CFGpp/blob/8035352527eb01ac69f65feaab374ba50616ce2c/examples/text_to_img.py#L15)

## 5. 保留下来的机制：预测 clean 的时间抵消与运输误差

附录 C 试图解释为何 CFG 需要较大增益：连续两步的条件 gap 在 posterior mean 演化中互相抵消，而 CFG++ 每步直接加入当前 gap。该观察可证伪，但原文 Proposition 1 不能原样照用。我们从 DDIM 重算如下，令 $\sigma_t=b_t/a_t$，所有预测在各自实际状态上求值，先考虑常数 $\omega,\lambda$：

$$
D_\omega(z_s,s)-D_\omega(z_t,t)
=\sigma_s[\epsilon_u(z_t,t)-\epsilon_u(z_s,s)]
+\omega\left[\Delta_s-\frac{\sigma_s}{\sigma_t}\Delta_t\right],
$$

$$
D_\lambda(z_s,s)-D_\lambda(z_t,t)
=\sigma_s[\epsilon_u(z_t,t)-\epsilon_u(z_s,s)]+\lambda\Delta_s
\quad(\mathrm{CFG++}).
$$

原文定义 $dz=z(\mathrm{new})-z(\mathrm{old})$，但式 (35)、(36) 中 noise 差分符号与其推导相反；CFG 历史 gap 还需要 $\sigma_s/\sigma_t$ 比例。已查看正式 PDF 第 16 页原图，且一次 CPU 仿射预测器检查确认：修正式成立，印刷公式一般不成立。符号修正不抹去“历史 gap 抵消当前 gap”的机制；它要求在真实网格上量化抵消幅度，不能直接把差分叫作不良振荡。

**有价值且可失败的假说**：若参考分支的 noise 随轨迹足够平滑，而条件/双头 gap 的时间变化放大局部 inversion 或多步运输误差，则把历史运输项交给参考分支、把新的语义修正放在当前 clean 项，可能降低该数值误差。反例很清楚：参考分支自身时间误差大，或被抵消的部分恰好是错误修正，去掉抵消反而更差。即便误差降低，也尚未等于 unconditional FID 改善。

最低证据应先是同一批预定轨迹相邻时刻的 $z,t,F,B$，从而计算真实 $\Delta$、$\sigma_s/\sigma_t$、参考 noise 的差分，以及 gap 差分与它的内积；只有范数不足以判断相加后的误差。若要把该机制连到真实质量，还至少需要独立的真实 clean/noise 对或真实特征误差，对预定干预测量误差交叉项。不能用同一个 gap 自定义 reward 后上涨作为证据。只有 endpoint 特征 bank 无法恢复这些时间量。

前向一阶重组必须先与精确系数换算后的 Euler 做 parity；若相同，就没有额外的流形效应可归因。高阶形式的参考历史项可能包含与当前 gap 不共线的方向，值得从结构上理解，但需要先与同成本的标准高阶 solver 对照，而不是把改变 solver 的收益归给 guidance。本文没有据此启动新 sampler、时间系数搜索、训练或 FID sweep。

## 6. 论文 FID 与成本对照

以下抄录数值事实并另算相对 FID 降幅，正数表示改善。来源均为正式版 Table 1–4、§4 及附录 E–F；没有复现图像或重算作者 FID。

**Table 1：SD v1.5，COCO 10K，DDIM 50 NFE。** $\omega/\lambda$ 按同 seed 的 LPIPS 搜索匹配；没有报告独立重复或误差条。

| $\omega\leftrightarrow\lambda$ | FID CFG → ++ | 相对改善 | CLIP CFG → ++ | ImageReward CFG → ++ |
|---|---:|---:|---:|---:|
| 2 ↔ .2 | 13.84 → 12.75 | 7.876% | .298 → .303 | −.235 → −.218 |
| 5 ↔ .4 | 15.08 → 14.95 | .862% | .310 → .310 | .068 → .071 |
| 7.5 ↔ .6 | 17.71 → 17.47 | 1.355% | .312 → .312 | .152 → .156 |
| 9 ↔ .8 | 20.01 → 19.34 | 3.348% | .312 → .313 | .170 → .208 |
| 12.5 ↔ 1 | 21.23 → 20.88 | 1.649% | .313 → .313 | .192 → .194 |

第一组以及表中各方法最佳 FID 的比较超过 5%，但不证明所有强度或全局优化后的 CFG 都有 5% 优势。LPIPS 匹配不是理论强度等价，搜索成本也未单列。

**Table 2：加速采样。** 蒸馏两组为相同 5K prompts、6 NFE；DPM++ 2M 为 20 NFE，正文未同样明确单列其样本数，不能擅自补成 10K。DPM 低步数区间有 $\omega=5\leftrightarrow\lambda=1$ 的经验对应。

| 报告模型/solver | FID CFG → ++ | 相对改善 | CLIP CFG → ++ | ImageReward CFG → ++ |
|---|---:|---:|---:|---:|
| SDXL-Turbo | 59.67 → 59.21 | .771% | .320 → .325 | .777 → .968 |
| SDXL-Lightning | 56.11 → 55.19 | 1.640% | .322 → .324 | .691 → .829 |
| SD v1.5 / DPM++ 2M | 32.72 → 32.58 | .428% | .313 → .312 | .086 → .023 |

附录 Figure 12–13 明确主表及图片分别用了 DreamShaper XL 与 Leosam’s HelloWorld XL 的蒸馏变体；不能把这两行直接说成官方原版权重的可复现实验。DPM 结果是小幅 FID 变化，同时 CLIP/IR 下降；论文也把该设置描述为大致相当。

**Table 3：FFHQ 512、前 1000 样本、PSLD inverse problems。** 这是有测量条件的重建，不是无条件生成。

| 任务 | FID PSLD / +CFG / +CFG++ | LPIPS 三者 | PSNR 三者 |
|---|---|---|---|
| SR ×8 | 46.24 / 41.24 / 36.58 | .413 / .394 / .385 | 24.41 / 24.91 / 24.87 |
| Motion deblur | 97.51 / 91.90 / 65.67 | .500 / .493 / .482 | 21.83 / 22.29 / 21.93 |
| Gaussian deblur | 41.65 / 41.52 / 39.85 | .388 / .390 / .400 | 26.88 / 26.94 / 26.90 |
| Inpaint | 10.27 / 9.36 / 9.78 | .053 / .055 / .052 | 30.15 / 30.27 / 30.31 |

PSLD 和 +CFG 使用 $\eta=1,\gamma=.1$；+CFG++ 按任务改变 $\eta=(1.3,.4,1,1)$、$\gamma=(.1,.025,.12,.1)$、$\lambda=(.1,.2,.2,.6)$。故不是保持其他优化超参固定的一行 ablation。此处的 PSLD $\gamma$ 也不应与 §3 的 DDS 步长混用。Inpaint 在遮罩外回填真实图像。正文/该附录未给这张表独立明确的 NFE，不能从表格反推运行成本。

**Table 4：COCO 1K，归一化重加权函数后在 noise 分支系数间插值。** noise 系数 $0,.4,.6,.8,1$ 对应 FID $66.52,66.60,66.78,67.50,68.10$，IR 为 $−.112,−.125,−.126,−.153,−.420$。表中从 CFG 向 CFG++ 改善约 2.320%；正文的方向描述恰好相反，应以数值为准。该表不含重复不确定性，具体 NFE 也未在这项 ablation 中单列。归一化总权重使它比仅对比 $\lambda\in[0,1]$ 更接近“权重放在哪里”的机制证据，但仍是小样本特定模型结果。

**成本边界。** 非蒸馏普通 CFG 的两个分支已在同一步计算，CFG++ 复用它们，因而可以保持相同模型调用结构。作者 `predict_noise` 在两分支存在时拼成 batch=2：50 NFE 是 50 次双分支批量调用，等价样本前向工作约 100 个分支；不要与 RAEv2 单次前向免费带出的 F/B 直接按数字对齐。[SD 实现](https://github.com/CFGpp-diffusion/CFGpp/blob/8035352527eb01ac69f65feaab374ba50616ce2c/latent_diffusion.py#L131)

蒸馏设置即使 $\lambda=1$，CFG++ 仍需参考 noise。相对于可只算 conditional 的标准蒸馏采样，这未必零额外成本；作者示例调用路径仍提供两个 embeddings。论文没给分支 FLOPs、峰值显存或 wall-time，不能由同为 6 NFE 推断严格同算力。RAEv2 的真实 unconditional 分支同样不能免费替代现有 same-class B。

## 7. 作者代码核对与可复现边界

- DDIM `BaseDDIMCFGpp` 的 clean 混合和 unconditional noise 重组与主公式相符；Euler CFG++ 使用 VE/Karras 形式 $D_\lambda+\sigma_s(z-D_u)/\sigma_t$。示例默认是负面 prompt 参考，复现理论 unconditional 设定需明确改为空文本，不能默认论文测量用了示例默认值。
- 正式版附录 A 的 DPM++ 2M 历史项为 $D_u(t)-D_u(t_{\rm prev})$。固定 commit 的 [SDXL 实现](https://github.com/CFGpp-diffusion/CFGpp/blob/8035352527eb01ac69f65feaab374ba50616ce2c/latent_sdxl.py#L911) 与之相符；[SD1.5 实现](https://github.com/CFGpp-diffusion/CFGpp/blob/8035352527eb01ac69f65feaab374ba50616ce2c/latent_diffusion.py#L864) 却使用 $D_\lambda(t)-D_u(t_{\rm prev})$。两者当步差为 $(1-e^{-h})\lambda\Delta/(2r)$，不是完全相同的离散算法。本次没有改作者文件，也没有证据把主表差异归因给这一处。
- 代码里的 `NFE` 配置不自动等于每条路径的实测前向次数：例如 SDXL DPM 循环遍历 `timesteps[:-1]`，其他路径又会追加零 sigma；复现时需实际计数，不能只抄命令行参数。
- 论文 DDIM inversion 的分析依赖相邻状态 noise 预测变化小。两种方法的实际轨迹不同，$\lambda<\omega$ 乘同一个差分的局部比较不能直接升级为所有轨迹的误差序或全局可逆性定理。重建 PSNR/RMSE 与无条件生成 FID 是不同证据。

## 8. 本次交付和下一步边界

一次 [CPU 审计](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_cfgplusplus_v1/algebra_audit.json) 检查了：两种 flow 重组与对应 Euler 的等价、当前网格的系数范围、posterior-mean 修正式及原式反例、冻结目标下降的反例、SD/SDXL 2M 差异。没有加载已有图像或重新读取任何质量试验 bank；这些代数检查不是 RAEv2 的实测效果。

可以继承的是分离“当前修正”和“历史运输”的设计视角，以及一个明确可失败的时间误差假说。当前不能据此提出“把 B 的 noise 加回去就能涨 FID”的方法，更不能搜索 $\lambda(t)$ 来补理论缺口。若之后发现已有相邻轨迹缓存，可先审计 §5 的差分、比例与交叉项；若没有这些量，阅读结论就是尚缺机制证据，而不是默认继续训练。既有 endpoint native 响应失败及旧候选 5K 阴性结论独立成立。

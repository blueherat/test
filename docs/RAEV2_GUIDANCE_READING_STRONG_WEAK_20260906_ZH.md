# RAEv2：strong/weak 原文阅读与目标冲突

日期：2026-09-06。只深读下列两篇论文；已读方法、实验和相关附录。RAEv2 本体仅核对[官方项目页](https://raev2.github.io/)及本地实现。本文没有训练、GPU 实验或新的性能结论。

**会改变设计的洞见：先区分要消除的有限模型误差与要保留的 guidance 偏置，再定义辅助训练标签。直接拟合 `X−G` 会把二者一起作为残差；扩大这个求解器的能力，并不天然等于加强有用的 strong/weak guidance。** 这是本文结合原文与本地目标所得的推论，不是两篇论文已证明的 RAEv2 结论。

**1. Karras et al., “Guiding a Diffusion Model with a Bad Version of Itself”，NeurIPS 2024。** 阅读 [arXiv v2 原文](https://arxiv.org/html/2406.02507v2)：§2–6、Appendix B.1、C。

同任务、同条件下，方法使用

\[
D_w=D_1+(w-1)(D_1-D_0).
\]

关键假设是弱模型放大强模型已有的同类误差。最有辨别力的证据在 §4：EDM2-S 原 FID 2.56；强/弱各加入 5%/10% dropout 后，强模型 FID 4.98，经 guidance 回到 2.55。若强、弱分别使用 dropout 与 input-noise 两种错配退化，最优值是 `w=1`，即关闭 guidance。

在合法 score 的解释中，附加项为 `(w−1)∇log(p1/p0)`；但 §2 明确说明这些逐噪声隐含密度通常不构成同一热扩散，不能据此声称终点恰为 `p1^w p0^(1−w)`。Appendix C 为画密度图采用标量 log-density 后求梯度的 toy；该结构没有自动赋予真实 F/B。论文未给一般受益定理；Appendix B.1 的约 30,000 次指标评估也不能迁移为本任务的调参方案。[原文](https://arxiv.org/html/2406.02507v2)

**2. Zhou et al., “Guiding a Diffusion Transformer with the Internal Dynamics of Itself”，2025/2026。** 阅读 [arXiv v2 原文](https://arxiv.org/html/2512.24176v2)：§3–6、Tables 1–5。官方 CVPR PDF 本次返回 403，采用完整 arXiv 正文核对。

训练为 `L=||Df−X||²+λ||Di−X||²`，采样为 `Di+w(Df−Di)`。Table 1 的 SiT-B/2：原始 FID 33.02，第 4 层中间监督后的无引导值 30.60，加固定 `w=1.5` 后 19.02，区分了训练收益与外推收益。§6 进一步研究非原图训练标签：

\[
X_{\mathrm{target},f}=X+\omega\,\operatorname{sg}(D_f-D_i).
\]

这里借 EMA 输出产生偏置。它支持研究目标错位，不能证明任何固定偏置都改善质量。§4 的机制是 toy 与经验解释；强头欠拟合时，大 IG 仍可制造 outlier。Table 3 搜索窗口、主结果组合 CFG 的做法不符合本任务的直接迁移条件。§5.1 使用 50K 随机类别评估，SiT 为 250-step SDE、LightningDiT 为 125-step Heun；LightningDiT 还改过优化器和 EMA，成绩不能视为当前冻结 RAEv2 的同成本收益。[原文](https://arxiv.org/html/2512.24176v2)

**有限模型下的具体推导：它决定训练目标，而不假设 F/B 是 exact score。**

令 `m(z,t,y)=E[X|z,t,y]`，`d=F−B`，`τ=G−F`。在精确算术的官方活动区间内 `τ=.78d`，区间外 `τ=0`；真实 BF16 复现应直接保存 `G−F`，不擅自重排原生算术。`F,B,G` 均为冻结、同 class 的当前状态函数。

对任意固定场 `Q`，无限函数类中的平方损失残差回归满足

\[
\arg\min_c\;\mathbb E[\tfrac12\|c\|^2-(X-Q)\cdot c\mid z,t,y]=m-Q.
\]

于是两种目标并不相同：

| 校正训练的残差 | 理想无限类校正 | 加到现有 G 后的输出 |
|---|---|---|
| `X−G`，当前 observable potential 的目标 | `m−G` | `m` |
| `X−F`，保留原引导偏置的对照目标 | `m−F` | `m+τ` |

第二行可等价写成相对 G 拟合标签 `X+τ`。它只使用同次 forward 已有的 F/B/G，不需要空条件分支、额外主模型或手工时间调度。这不是推荐立即换目标：`τ` 的有用性仍是假设，保留它也可能保留错误。“保留”只指同一状态上的加性项；改变采样轨迹后，原 IG 的实际终点作用并未因此得到保留。两行仅在相同真实 bridge 上解释条件均值，不能外推为 rollout 或 FID 定理。

若限制到固定测度、相同权重下的完整闭合梯度子空间 `H`，总体解变为正交投影；两种目标的解之差为 `Π_H τ`。实际有限神经势函数类并非线性子空间，优化也未精确收敛，不能把投影恒等式写成两次真实训练结果的保证。

由此得到一个具体、可证伪的后续问题：**现有 `X−G` 校正是否在有用方向上撤销了原生 IG，而不是主要补上 Full 的剩余误差？** 若考察旧校正 `c`，应在所有预定时间与同图配对数据上分别计算 `E[(X−G)·c]`、`E[(X−F)·c]` 和 `E[τ·c]`；后两者与第一项的差有严格恒等关系。它能检验“撤销原引导”的方向事实，但不能单凭负相关认定撤销有害，更不能据此搜索时间窗口。任何新训练对照必须先补充偏置为何值得保留的生成机制及可证伪预测。

**与当前 RAEv2 及已做工作的关系。**

| 现有事实或约束 | 迁移判断与重复风险 |
|---|---|
| 同 class 原生 F/B；Base 来自第 8 层、Full 另含 DDT decoder | 符合“同条件”这一个必要条件；不证明误差兼容。不能把 F−B 当类别 posterior 梯度。已核对 [DDT.py](../external/RAEv2/src/stage2/models/DDT.py) 和 [guidance_utils.py](../external/RAEv2/src/utils/guidance_utils.py)。 |
| [冻结深度×读出检查](RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md) | 冻结 head-swap 的交互过大，不再制造一个“更差”的交叉读出后靠范数匹配补救；Autoguidance 的错配退化实验给出直接理由。 |
| [密度去污染检查](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md) | 原文密度比解释没有修复有限 F/B 的可积性或跨时间一致性；不重启相同 mixture/ratio 候选。 |
| [Common adapter 账本](RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md) | 保持 contrast 的 common correction 已做；`X−F` 对照与这条线高度邻近。旧 flow-common 同时优化 Full/Base，目标不自动等于此处 Full-only，但差异不足以构成新方法贡献；保持 gap 本身也未解决质量断层。 |
| [Observable potential 筛查](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md) | bridge coupling MSE 降低与 FID 未达 5% 已发生。该结果与目标冲突假说相容，也与容量、覆盖、优化、递归误差等解释相容，尚未识别因果。 |
| [当前目标](RAEV2_GUIDANCE_GOAL_20260906_ZH.md) | 不采用原文扫系数/窗口，不选图或生成后拒绝采样。辅助训练若有机制依据可以使用；训练与每图推理成本分开，新增成本要有可比预算基线。最终仍需 `1−FID(method)/FID(cost-matched baseline)≥.05` 并独立噪声确认。 |

本轮的设计裁决是：**不把“更完整地消除 `X−G` 残差”直接设成下一代 strong/weak guidance 的质量目标；先识别偏置与误差，再决定要拟合哪一个残差。** 论文给出了有用反例和目标设计启发，尚没有把这个问题转化为已通过机制门槛的新候选。

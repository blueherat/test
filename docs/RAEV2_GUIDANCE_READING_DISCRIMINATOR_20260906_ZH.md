# RAEv2：判别器 guidance 的证明、残差等价性与轨迹边缘

日期：2026-09-06。阅读两篇原始论文的方法、相关证明和实验成本。本轮只写此笔记，没有训练、采样或 GPU 使用。

**结论：直接把“真实图／模型终点再加噪”的判别器梯度加到当前 RAEv2 场，尚缺少决定性的自洽条件。2025 新损失的纯 MSE 部分属于已经做过的 `X−G` 梯度残差回归；不同时间权重、CE 混合与参数化使完整方法并不完全相同。一个有建设性的变化是把负例换成实际 rollout 边缘，并将作用解释为可检验的 KL 输运校正，而非未经核对的 missing-score 恒等式。**

**原文 1：Kim et al., ICML 2023。** [论文正式 PDF](https://proceedings.mlr.press/v202/kim23i/kim23i.pdf)，[PMLR 页面](https://proceedings.mlr.press/v202/kim23i.html)，[arXiv](https://arxiv.org/abs/2211.17091)。已读 §2–3、§5、Appendix A.1–A.3、D.1、Tables 3/5/6/8。OpenReview 验证页不可用，改读正式 PDF。

Algorithm 1 从模型生成终点建假样本库，然后用相同前向核加噪；Bayes 判别器的 logit 为 `log(p_t/bar_q_t)`。Theorem 1 要求终端 score 与先验一致、`log pθ=ELBO`。Appendix A.1 经 `S_sol` 明确推出 `sθ=∇log bar_q_t`，才得到 `sθ+∇log(p_t/bar_q_t)=∇log p_t`；这是实质条件。Theorem 2 对基模型 KL 给等式，对 guided 终点给上界；CE 训练不保证其 gain 为正。A.3 只在 CIFAR-10 检查 NLL/NELBO 近似接近。

Table 3 报告 CIFAR-10 无条件 FID：EDM 的 manual-seed 原报告值 1.97、随机 seed 重算值 2.03、DG 默认随机 seed 值 1.77，不能当同噪声配对。Table 5 的 DiT 2.27→1.83，D.1 同时使用分段 CFG。Table 6 的 CIFAR 准备为约 1 A100-hour 生成假数据、10 分钟训练判别器；Table 8 还使用冻结分类器、25K 假／50K 真数据和 60 epochs。ImageNet-DiT 使用约 128 万假样本。主模型 NFE 未计全判别器输入反传，不能当总成本相同。[正式 PDF](https://proceedings.mlr.press/v202/kim23i/kim23i.pdf)

**原文 2：Vérine et al., ECML/PKDD 2025。** [arXiv v2 全文](https://arxiv.org/html/2503.16117v2)，[会议原文](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_1033.pdf)。已读 §3–6、Appendix A.1–A.5。

Eq.6/11 和 Theorem 4.1 直接假设基模型及校正场分别等于自身边缘 score，Eq.7/12 才采用终点 KL 等式。CE 反例的核心是小幅、高频振荡：接近最优的函数值不保证梯度准确。Appendix A.4 讨论重叠分布上的训练集过拟合。注意正文将 CE 写成绝对小于 ε，A.2 实际使用距最优 CE 的误差；对重叠分布应采用后者，不能照抄绝对零 CE 的说法。

Eq.15 为 `E||s_cond−sθ−∇d||²`，实用版本混入 CE；Eq.17 与 Algorithm 1 对 γ 的位置不一致，复现须核对代码。Table 1 在优化过 w/γ 后：

| 数据 | EDM | CE-DG | 新损失 | 相对 CE-DG 降低 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 1.96 | 1.94 | 1.91 | 1.55% |
| FFHQ-64 | 2.54 | 2.42 | 2.41 | 0.41% |
| AFHQv2-64 | 2.57 | 2.47 | 2.44 | 1.21% |

前两组 FID 用 50K，AFHQ 用 15K。Table 1 训练计时比约 3.39/4.51/4.51，且显存增加；只更新 2.88M 参数不代表无需经过完整冻结主干。未提供本任务所需的完整同成本基线。2023 的 CIFAR 1.77 与此处 CE-DG 1.94 协议不同，不能跨表宣称超过既有最优 DG。[原文](https://arxiv.org/html/2503.16117v2)

**迁移时必须分开的三个边缘。以下为针对当前 RAEv2 的推导。**

定义 `K_t(X,ε)=(1−t)X+tε`：

| 记号 | 实际含义 |
|---|---|
| `p_t` | 真实数据 `p_0` 经 K_t 的边缘 |
| `q_t` | 当前冻结官方 IG／候选 Euler 采样器实际访问的状态分布 |
| `bar_q_t` | 该采样器的最终 `q_0` 另取独立噪声、经 K_t 得到的边缘 |

即使 `q_0=bar_q_0` 且纯噪声端匹配，也不推出中间 `q_t=bar_q_t`；一般有限 F/B、IG、ODE 离散误差都不被端点身份约束。把实际 `q_t` 换作 CE 的负例也只得到 `log(p_t/q_t)`，仍不自动推出当前向量场是 `∇log q_t`。

若用 `bar_q_t` 训练出完美 logit，实际校正结果为

\[
s_G+\nabla\log(p_t/\bar q_t)
=\nabla\log p_t+\underbrace{[s_G-\nabla\log\bar q_t]}_{\text{未被判别器修复的自洽误差}}.
\]

2023 Theorem 1 的假设恰好消除方括号。单纯提高 CE 精度无法消除它。2025 的两个 score 身份也不是“网络定义了反向过程”的自动推论；一般情况下 Girsanov 控制的是路径 KL，端点由 data processing 得到上界。降低一个非紧上界，不能直接推出相对基线的端点 KL 降低；这些 SDE 结论更不能直接等同于当前确定性 Euler 的 FID 保证。

**新 MSE 与现有势函数：同一种残差，何时完全等价。**

在 `0<t<1`，只作代数定义而不假设合法 score：

\[
s_G(z,t)=\frac{(1-t)G(z,t)-z}{t^2},\qquad
s_{\mathrm{cond}}(z,t;X)=\frac{(1-t)X-z}{t^2}.
\]

令 clean 修正 `c=∇Φ`，则相应 score 修正为

\[
\nabla d=\frac{1-t}{t^2}c,\qquad d=\frac{1-t}{t^2}\Phi,
\]

从而论文纯 MSE 项正比于

\[
\sum_k h_k\lambda(t_k)\frac{(1-t_k)^2}{t_k^4}
\mathbb E\|X-G-c\|^2.
\]

现有 [observable potential](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md) 是 `Σ(h/t²)E[.5||c||²−(X−G)·c]`，即相同 clean 残差平方损失去掉常数。匹配 `λ(t)∝t²/(1−t)²`、样本、函数类和优化时才得到同一加权问题；论文的实用权重、CE 项和共享参数化不能略掉。当前 grid 包含 `t=1`，此时 score 对 G 的系数为零，逆转换奇异，不能声称全部 100 点无条件精确等价。

更进一步，对 K_t 对应的线性前向 SDE，`f=−z/(1−t)`、`g²=2t/(1−t)`；若真采用 SDE 路径 KL 权重 `g²/2`，转换后的 clean 权重是 `(1−t)/t³`，并非当前 `1/t²`。因此应把纯 MSE 视为同一类残差投影，不能给旧结果换个 discriminator 名称后重跑。

条件期望恒等式仍给出 `E||s_cond−s_G−∇d||² = E||∇log p_t−s_G−∇d||²+C`，无需 `s_G=∇log bar_q_t`。它保证所优化的是 data-score 残差；只有自洽时才能同时称为所选密度比的梯度。此区分避免对论文中的记号直接套用。

**建设性设计启发：实际边缘上的密度输运。**

取 dataward 时间 `s=1−t`。假设 p_s、q_s 有光滑正密度、相应速度为 v_p、v_q，边界项消失。直接从连续性方程可得

\[
\frac{d}{ds}\mathrm{KL}(q_s\|p_s)
=\mathbb E_{q_s}[(v_q-v_p)\cdot\nabla\log(q_s/p_s)].
\]

在同一个当前快照 q_s 上，给原速度添加 `a∇log(p_s/q_s)`，`a>0`，其对该导数的贡献严格为

\[
-a\,\mathbb E_{q_s}\|\nabla\log(p_s/q_s)\|^2\le0.
\]

这不要求原速度或 F/B 是自身 score。因而可研究：以**实际 rollout 状态**为负例、真实 bridge 为正例，跟踪密度比，将其空间梯度作为轨迹内输运方向。原理与“终点假样本再加噪后修补 score”不同，也与只用真实 bridge 标签的 `X−G` 回归不同。CE 的梯度失配问题仍需解决；冻结旧 q_s 的判别器随分布改变会失效，必须审计这种时效性，不能只靠旧 bank 分类准确率。

若实际方向为 `∇r_hat`，令 `r=log(p_s/q_s)`，导数改善条件为 `E_q[∇r_hat·∇r]>0`；例如 `||∇r_hat−∇r||_{L²(q)}<||∇r||_{L²(q)}` 足够。这个条件控制梯度而非 CE。它只保证同快照、连续时间的一阶贡献，不保证含原 drift 的总 KL 单调、有限 Euler 步下降或终点 FID。本文没有指定或搜索 a、窗口或额外步数；成为候选前仍需机制导出的有限步控制及成本协议。

**一个精确、可证伪的 Gaussian 试验规格，无需网络自洽假设。**

设真实 clean 方差为 3，模型 endpoint `q_0=N(0,1)`。构造合法确定性采样路径 `Z_t=[1+sin²(πt)]ε`，从 `q_1=N(0,1)` 到上述 q_0。在 `t=.5`：

\[
p_t=N(0,1),\quad q_t=N(0,4),\quad\bar q_t=N(0,.5).
\]

冻结这个时刻的目标与状态，真实比值梯度为 `−.75z`，endpoint 再加噪比值梯度为 `+z`。让快照粒子分别沿两种场走一个无穷小输运步 τ，`d KL(q||p)/dτ` 分别为 **−2.25** 与 **+3**。这是直接解析计算：`∇log(q/p)=.75z`、`E_q z²=4`。它展示完美 CE 比值也会因为选错参考边缘而给出反方向；同时给出实际 q_t 跟踪方向的可验证正例。后续实现首先应复现这些导数，再讨论有限数据与离散步，不以该 toy 充当 RAEv2 性能证据。

**任务边界与重复风险。** 上述 DG 在尚未结束的轨迹状态上加梯度；不生成多张完整图后排序、接受或拒绝，因而与已禁的 [终端 acceptance](RAEV2_POSTERIOR_ACCEPTANCE_GUIDANCE_20260906_ZH.md) 不同。任何辅助训练、假样本库、实际轨迹刷新、完整冻结特征提取器的输入反传、额外步数都需计费；不能只报可训练参数或主模型 NFE。纯 residual 分支已有 [阴性筛查](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)，不自动重启。真实 q_t 的梯度跟踪尚未完成机制准入；最终仍须符合 [当前目标](RAEV2_GUIDANCE_GOAL_20260906_ZH.md) 的无手工 schedule、无选图、公平成本下至少 5% FID 改善及独立噪声确认。

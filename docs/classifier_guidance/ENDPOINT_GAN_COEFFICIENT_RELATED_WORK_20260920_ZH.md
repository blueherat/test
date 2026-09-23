# 终点分布反馈学习采样系数：相关工作核对

核对日期：2026-09-20。结论：已有很接近的论文。不能把“冻结扩散主干，用 GAN 的终点反馈优化少量采样系数”本身作为新贡献。

| 工作 | 学什么 | 判别器/质量目标监督哪里 | 与当前实验的关系 |
|---|---|---|---|
| GAS，ICLR 2026；预印本 2025-10 | 求解器混合系数、时间网格和模型查询时间修正 | 完整学生采样终点，对照教师采样终点；蒸馏损失加 GAN | 最直接的“终点 GAN + 可训练采样系数 + 反传”先例 |
| AdaGen，2026-03 | 冻结生成模型的自适应采样策略，包含 guidance 和时间步 | 最终图像与真实图像的判别奖励，用 RL 更新策略 | 最接近“真实数据终点反馈学习 guidance”的先例 |
| Adversarial Learning of Classifier-Free Guidance Schedules，2026-08 | 非负、依赖状态/条件/时间的 CFG 系数网络 | 从真实数据加噪状态出发的一次 guided 跳跃，与目标时刻真实加噪边缘分布比较 | 最接近“GAN 学 guidance 系数”；训练对象不是完整自由采样轨迹 |
| DDSS，ICLR 2022 | 冻结扩散模型的采样器参数 | 完整生成样本的 KID，通过采样过程反传 | 终点质量直接优化系数的早期先例，目标不是 GAN |

上述事实分别见 [GAS §3.1–3.2](https://arxiv.org/html/2510.17699v2#S3)、[AdaGen §III-C–D](https://arxiv.org/html/2603.06993v1)、[对抗 CFG §3 / Algorithm 1](https://arxiv.org/html/2608.14038v1#S3)、[DDSS 论文](https://arxiv.org/abs/2202.05830)。GAS 的会议身份已由 [ICLR 正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/42d6cda0d721d41ea5751b4bf4d53616-Abstract-Conference.html)核实。

## GAS：已有公开代码，区别在目标与控制方向

GAS 将历史状态与历史模型预测做可训练线性组合。其 GAN 正样本是高质量教师生成结果，并保留逐样本蒸馏约束；它主要研究少步求解器逼近教师。当前实验用真实训练图像作正样本，固定 64 步 Heun，控制方向限定为同一强模型与原生弱头的差。

已读取作者仓库的 `gs_wrapper.py`、`generalized_solver.py`、`synt_data.py`、`adversarial_module/dist_adv_loss.py` 和 `training.py`：可训练求解器产生 `student_images`，与合成教师数据构造对抗/回归目标，再通过 `loss.backward()` 更新。不能仅把它归入“GAN 微调整个 diffusion 网络”而略过。[作者实现](https://github.com/3145tttt/GAS)

## AdaGen：终点 GAN 奖励与 RL 调度

其 MDP 仅在终点给奖励，生成主干冻结；动作可包含 CFG 强度。判别器与策略交替学习，策略用 PPO 类更新，避免完整轨迹的路径导数。系数可依赖样本状态，而当前实验是所有图像共享的 64 个有符号参数。这也是之前讨论“RL 替代长轨迹反传”的直接参考。[论文](https://arxiv.org/html/2603.06993v1)、[作者仓库](https://github.com/LeapLabTHU/AdaGen)

作者 README 声明已发布 MaskGIT、DiT、SiT、VAR 的评估代码与策略 checkpoint；本轮不将其称为已验证可直接复现训练的完整代码。

## 对抗 CFG：不能误读成完整 endpoint GAN

Algorithm 1 的负例由真实图像加噪到源时刻，再做一次 Euler 跳跃产生；正例是另一真实图像直接加噪到目标时刻。它学习非负系数，另有系数正则与 CLIP 奖励。其小模型主表 FID 为 31.31，常数 CFG 为 29.73，不能把文本对齐的收益转述为 FID 全面改善。本轮未在论文中找到作者代码链接。[论文 §3、Table 1](https://arxiv.org/html/2608.14038v1)

## 对当前故事的判断

本轮实际学习的是如下受限生成器：

\[
G_a(z,c)=\operatorname{Decode}\!\left(\operatorname{Heun}_{64}
 [S+a_i(S-W)](z,c)\right),
\qquad \min_a \mathbb E\,\operatorname{softplus}[-D_\phi(F(G_a(z,c)),c)].
\]

强模型与弱头冻结，`a_i` 可正可负，判别器持续更新。这里 `F` 是冻结 Inception，因此判别训练直接约束的是图像特征分布；不能由此直接声称完整像素分布已经一致。

结合上述工作，我的判断是：

- “只学系数”适合作为受控实验，证明已有强弱差值方向中存在可利用的终点改善空间；单靠这个机制，独立方法的新意较弱。
- 冻结主干、终点 GAN、少量参数、推理时不使用判别器，这些属性都有先例。
- 我们原本更核心的问题仍是：弱头是否必须拟合 diffusion 目标，还是应通过终点质量学习更合适的控制方向。它与本轮冻结弱头、只学系数是不同实验。
- 新的两档纯 diffusion MLP 是容量对照：需要先知道增大常规弱头本身能到哪里，之后才能判断 GAN 目标改变是否有额外价值。

目前没有核对到与“原生 IG 双头均冻结 + 64 个无符号限制的全局时间系数 + 真实图像特征终点 GAN + 完整 Heun 反传”逐项完全相同的实验；这不构成不存在其他相关工作的证明，也不足以单凭配置差异认定方法新颖。

## 本地证据

论文 HTML、提取文本与 GAS 相关源码只读保存在实验目录 `literature/`，并记录内容哈希。未安装或运行第三方训练代码。

实验入口：[MLP 容量对照](MLP_CAPACITY_DIFFUSION_20260920_ZH.md)。

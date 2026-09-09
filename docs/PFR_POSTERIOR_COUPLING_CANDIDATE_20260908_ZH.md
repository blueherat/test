# 后验耦合的未来弱预测：候选算子与限制

本轮从原始 PFR 在精确 Gaussian 场也会改动目标分布这一已知问题出发，
定义一个有明确零模型的候选。尚无真实模型质量结果，不能称为新方法突破。
原始 PFR 在官方 SiT 的固定独立 5K 收益仍成立，不被此分析否定。

## 既有理论与目标限制

理想 denoiser 的反向鞅关系是既有知识。
[Consistent Diffusion Models，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/831406cfe7e4a0aed5ac5c8a8389d1f5-Paper-Conference.pdf)
将此一致性用于训练正则化，并要求额外的目标信息来识别正确分布。
本轮读到其引言中的条件期望定义、定理条件及训练用途，未复核全部证明和代码。
相关 score PDE 关系见
[FP-Diffusion](https://proceedings.mlr.press/v202/lai23d/lai23d.pdf)，
仓库已有详细阅读与反例，不重新声称该理论新颖。

`RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md` 已说明：任意错误的
Gaussian 数据分布也可以满足自洽性及同一个纯噪声端点。
本候选不能消除这个不可辨识性，零残差不是正确数据分布或好 FID 的认证。
是否存在有意义的新推理算子及实证机制，仍待验证。

## 有限时刻查询

沿线性 bridge，记 Z_t=(1-t)X+t epsilon，r<t，a=1-t、b=1-r。
选取具有这些边缘的 Markov 加噪耦合。给定 Z_t=z 和 clean X，
Z_r 的条件分布为 Gaussian，其参数为

`A = (a/b) r²/t²`

`B = b - A a`

`V_noise = r² [1 - a² r²/(b² t²)]`。

故条件均值恰为 `A z + B D_t(z)`。完整条件分布还包含 clean 后验的
不确定性；用后验均值替代 X 得到的单 Gaussian 并非精确逆转移。
候选以该均值为中心，执行两次成对查询

`q_± = A z + B D_t(z) ± sqrt(V_noise) eta`

`D_bar = [D_r(q_+) + D_r(q_-)]/2`。

拟研究的残差是 `D_t(z)-D_bar`。它目前仅是一个可计算对象，尚未
据此确定强度、方向符号、部署窗口或保证改善的控制律。
若用于 PFR 弱参考，需要明确在 clean 坐标回到当前时刻再转换速度；
不能直接混用两个时刻的 native velocity。

该表达式在 t=1 仍有限：A=0、B=1-r、V_noise=r²。
没有用除以 1-t 的 Brownian 坐标在纯噪声点执行数值计算。

## 零模型及误差

任意多元 Gaussian 数据目标的 D_r 都是仿射函数，成对噪声恰好相消。
因为查询中心是准确的 E[Z_r|Z_t]，条件期望塔式恒等式给出
`D_bar=D_t`。这说明算子对该族精确为零，包括错误的 Gaussian 目标。
此性质是仿射性与条件期望的直接推论，不是新理论。

一般非 Gaussian 目标下，替换后验 X 会遗漏 `B² Cov(X|Z_t)` 及更高阶
信息。即使 denoiser 完全准确，残差也可能来自转移近似；有限对数查询
还引入 Monte Carlo 随机误差。两者必须与模型自身的时间不一致区分。

## CPU 检查

`audit_pfr_posterior_coupling_toy.py` 已完成（会话 62422，exit 0），
运行约 .155 秒，无 GPU、训练或 decoder。
固定 3 个均值、4 个方差、4 个当前时刻、3 个状态，每点 16 对噪声，
总计 2304 次 Gaussian 检查，最大误差 **1.776e-15**。

非 Gaussian 检查使用固定双 Gaussian 混合，不根据结果选择分布。
128 点 Gauss–Hermite 积分的精确后验混合转移满足条件期望关系，
全部 36 行误差小于 1e-10。以下是近似转移的误差：

| h | 最大绝对偏差 | 单对噪声估计的标准差中位数 |
|---|---:|---:|
| 1/32 | .01121230 | .00685658 |
| 1/64 | .00316708 | .00299152 |
| 1/128 | .00084533 | .00141343 |

该表是固定 toy 的数值结果，不是对一般真实模型的渐近保证。
结果保存在 `experiments/results/terminal_defect_20260908/pfr_posterior_coupling_toy.json`。

下一步仅检查真实弱头中跨噪声重复查询的均值与随机波动，先判断这个
两次 prefix 的估计是否具有足够稳定的信号。尚未决定扩大质量实验，
不因 Gaussian 零残差就直接进入训练、1K 或论文写作。

## 后续真实质量结果

重复性检查完成后进行了另行固定的RAE1K探索，现已判为阴性：
候选42.801481，普通100步38.264239，普通150步38.458631。
Gaussian零残差性质和真实弱头重复性没有带来质量收益。
该具体有限查询替换不继续扩大，见RAEV2_POSTERIOR_REFERENCE_1K_RESULTS_20260908_ZH.md。

# RAEv2 guidance 阅读：ADG 与范数约束的对象

日期：2026-09-06。**ADG 提供的有用直觉是分开研究方向与长度；当前证据不足以把 RAE 的引导设计为保持 Full 范数。** 本次进一步辨认了三个不同对象：真实潜变量、条件 posterior mean，以及有限模型 Full/IG 的预测。它们不能共用一个未经验证的球壳半径。没有运行 ADG 采样、调角度或计算新 FID。

一手阅读为 [ADG，ICML 2025，arXiv v1](https://arxiv.org/html/2506.11039v1) 的正文、B–F 证明/算法与 G 实验，以及[作者源码](https://github.com/jinc7461/ADG/tree/86a651bbb990220c5c65cab40e7ebe2e9fc53bd4)。PDF SHA256 为 `bfee6f3b3de2a9e283da3e0d04285589fa51a18efaa090b57b25470c123b5c3e`；不是 v2。材料位于 [reading_angle_covariance_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_angle_covariance_v1)，其中 covariance 项目页仅是早期浏览，另篇全文由独立笔记记录。

**论文中实际得到的保证。** 在共同单位协方差 Gaussian mixture 和 surface class 假设下，Theorem 3.2 给出沿支撑超平面法向的向外投影增大；其公式不是任意原点下逐样本欧氏范数的比较。Theorem 3.3 找到 guided score 与 conditional score 内积非正的局部区域；这也不是完整概率流漂移的密度变化公式。ADG 随后约束 clean prediction 的转角。论文 Proposition 4.1 保证范数至多为 conditional prediction 的 √2 倍，而非严格保持范数。上述前提针对 conditional/unconditional，不自动覆盖同类 Full/Base。

**独立展开算法几何。** 令 F、B 为两份非零 clean prediction，夹角 γ∈(0,π)，沿用论文 ω>1 的前提，a=min((ω−1)γ,π/3)。理想实数算法为

\[
q=\frac{F-\operatorname{proj}_B F}{\sin\gamma},\qquad
A=\cos(a)F+\sin(a)q.
\]

直接计算得到

\[
\|q\|=\|F\|,\quad q^\top B=0,\quad
q^\top F=\|F\|^2\sin\gamma,
\]

\[
\boxed{\|A\|^2/\|F\|^2=1+\sin(2a)\sin\gamma.}
\]

因而 q 通常不垂直于 F，这不是围绕 F 的等长旋转。在论文的角度范围内，该范数比处于 [1,√2]。固定二维例子 F=(1,0)、B=(1/2,√3/2)、ω=1.78，输出范数为 **1.36539976**；直接计算与公式相差 `2.22e−16`。这验证范数界与“严格保长”的区别，不反驳论文的 Proposition 4.1，也不是 RAE 的实测夹角。

作者当前代码的主函数还使用 `(ω−1)_+ + 1e−3`，默认 angle 为 `3.14/3`，小 sin 分支阈值 `1e−3`；utils 的投影分母加 `1e−8`。本文几何计算采用论文实数公式，没有宣称复现这些浮点细节。Flow 附录先从速度恢复 clean prediction，再执行相同操作；在 RAE 中该对象就是 F/B，不能直接旋转原 velocity 而称为同一方法。源码快照仅用于核读，没有导入运行整个 pipeline。

**实证怎样影响采用判断。** SD3.5、COCO10K、10 NFE 主表的最佳 ADG FID 为 16.6，细扫后的 CFG 也是 16.6，APG 为 16.1。ADG 在很强 guidance 下较稳健，但这组数据不证明超过最佳常数 CFG 的 5% 收益。论文还比较不限制角度与归一化版本，说明角度界和归一化选择有实证作用，不能声称全由定理唯一确定。[主表与附录](https://arxiv.org/pdf/2506.11039v1)

**由阅读独立导出的关键约束：posterior mean 不在原始数据球壳上。** 对任意平方可积 X，记 m(z)=E[X|Z=z]、C(z)=Cov(X|Z=z)，有

\[
E[\|X\|^2\mid z]=\|m(z)\|^2+\operatorname{tr}C(z).
\]

即使 X 的范数高度集中，也不推出 m(z) 应有同样范数。以完全可解的 RAE 线性 Gaussian bridge 为例，c>0、X∼N(0,cI)、Z_t=(1−t)X+tε，ε∼N(0,I) 且与 X 独立，令 Q_t=(1−t)^2c+t^2，则

\[
m_t(z)=\frac{(1-t)c}{Q_t}z,\quad
C_t=\frac{ct^2}{Q_t}I,\quad
\frac{E\|m_t(Z_t)\|^2}{D}=\frac{(1-t)^2c^2}{Q_t}.
\]

它与真实每维能量 c 的缺口恰为 `ct²/Q_t`，是正常 posterior 不确定性。在 t=1，m 恒为零，真实 X 却仍集中于半径 √(Dc) 附近。这个例子并不说 ADG 一定失败：ADG 的主要参照是有限 F 的长度。它说明“高维原始 latent 球壳”不足以证明 F、G 应满足哪一种范数约束；还须识别 F 的偏差和 guidance 真正改变的部分。

**现有 RAE 数据的有限检验。** 复用[噪声端点审计](RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md)的全部 1000 张真实图、每类一张、D=262144 个归一化潜变量坐标。该库为 FP32 encoder 输出再存为 FP16；本次只读已有 FP64 能量数组，不重编码。旧审计已经公开能量均值和 SEM，因此这属于后续描述性分析，不是对未见数据的预注册假说检验。新的统计定义先写入 `radius_request.json`，SHA `db54f2857f4a63b19dcb58458a4f63217bb143b33022742921b82f1b91e7bf58`。

| 量 | 全部固定 1K 真图 | D 维中心各向同性 Gaussian 参照 |
|---|---:|---:|
| 每维能量均值 | 1.00028636 | 可匹配相同标量均值 |
| 能量变异系数 | 18.5477% | 0.276214% |
| 半径变异系数 | 8.83504% | 0.138107% |
| 半径/√D 的 5%–95% 分位数 | 0.890428–1.170356 | 未用随机样本估计 |

实测半径 CV 是这个名义维数 Gaussian 参照的 **63.9725 倍**。不能因为维数大就假定当前数据近乎固定球壳；空间/通道相关、非 Gaussian 性和类间异质性均可能贡献差异。按中心 Gaussian 公式反算的矩等效维数为 58.1365，**不是本数据的真实秩、内在维度或已估计 covariance participation ratio**。本检验也不包含给定 z 的 posterior mean，不否定条件或局部的其他几何近似。

**因此改变的下一步。** 不从这些数据挑一个半径、截断角度或时间窗去跑 1K。更有辨别力的问题是：整体能量接近真实值时，IG 是否仍在不同空间/通道子空间保留可重复的二阶结构偏差？这不是从本次半径统计推断出的偏差，而是一个独立待检验问题；例如各向同性的随机径向缩放也能产生很大的半径 CV。若另有证据，应从实际偏差和 bridge 的方向性不确定性推导纠正，而非统一归一化。均值/协方差代理的改进仍需以真实生成质量验证，不能直接认领 FID 保证。

可复核结果在 `radius_and_geometry_results.json`，脚本为 `audit_radius_and_geometry.py`。本次计算 wall `0.279800 s`、CPU `12.122974 s`（含多线程库导入，至计算完成，不含最终序列化/退出）；零 GPU、模型、随机噪声或 FID 调用。全文与源码下载时间没有记录为精确计算预算；没有候选性能或公平成本达标结论。

另一代理的只读 CPU 复核重新检查源数组/request SHA、1000 类身份和全部汇总，数值一致；以 60 位精度计算 Gaussian 半径 CV，与记录的相对差约 `9.10e−13`。复核 wall `0.101580 s`、CPU `5.555719 s`，无 GPU；它不是新数据或独立质量实验。复核提出的 ω、c 与噪声独立性前提已补充。

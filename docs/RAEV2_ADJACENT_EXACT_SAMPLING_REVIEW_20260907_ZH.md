# 精确 speculative sampling 初读归档（2026-09-07，停止）

**状态：用户已明确否决采样加速方向：『这个方向不好，换一个』。本记录只保存停止前已读内容，不建议继续，不提出其他加速变体。没有运行 GPU 实验、训练 head、生成或筛选图像；正在进行的固定 5K 扩展不受影响。**

## 已访问的一手文献与阅读范围

1. De Bortoli, Galashov, Gretton, Doucet，[*Accelerated Diffusion Models via Speculative Sampling*，ICML 2025，PMLR 267:12590–12631](https://proceedings.mlr.press/v267/de-bortoli25a.html)。已读会议入口摘要，以及 [arXiv v1](https://arxiv.org/html/2501.05370v1) 的引言、目标随机转移核、draft 和 adjusted rejection 的初步说明。目录列出了 reflection-maximal coupling 和 Medusa-like correction；**尚未核对这些章节的完整推导或会议最终 PDF，不能据此判断小 head 方案的新颖性**。
2. [*Continuous Speculative Decoding for Autoregressive Image Generation*，arXiv:2411.11925](https://arxiv.org/abs/2411.11925)。已打开 [v1 HTML](https://arxiv.org/html/2411.11925v1)，停止前尚未完成正文阅读。不能把这篇计为已精读，不能声称完成与第一篇的算法或速度比较。

## 初读所得的边界

第一篇把目标定义为既定模型与离散随机 sampler 所产生的 Markov 链；同协方差 Gaussian draft 与 target 可以进行 maximal coupling。draft 样本 $X\sim p$ 按 $\min(1,q(X)/p(X))$ 接受；拒绝时必须从与 $(q-p)_+$ 成比例的修正分布取样，不能随意改成重新从 $q$ 抽样。正确修正保留的是**该目标采样链的分布**，不是对真实数据分布的恢复保证。这个基本机制见上述 [arXiv v1 的初始方法部分](https://arxiv.org/html/2501.05370v1)。

由同协方差 Gaussian 的全变差距离可直接算出一步最优接受率

$$
\int\min(p,q)=2\Phi\!\left(-\frac{\|\mu_p-\mu_q\|}{2\sigma}\right).
$$

这是对机制的解析推论，不是本仓库的实测结果。它说明高维均值差相对于噪声的大小会限制接受率。确定性 ODE 转移是 Dirac 核；两种均值不同时其重叠为零。因此不能把随机核接受机制直接套在当前确定性 ODE 上，再声称保持原 sampler 的分布。也尚未验证目标模型批量校验能否带来实际 wall-time 收益。

即使精确保留分布的 speculative 算法有效，其研究指标也应是**同一采样分布下的实际延迟/吞吐**；它没有单图 population FID 改善机制。这类局部转移提议与修正也不是先完成多张图再按质量挑选，但这一概念区别不改变用户已否决该方向的决定。

## 收束

未形成新颖性结论、RAEv2 可行性结论或性能结果。停止小 head、Gaussian overlap、accepted-prefix 和其他采样加速变体的后续研究；本文件仅作研究轨迹归档。

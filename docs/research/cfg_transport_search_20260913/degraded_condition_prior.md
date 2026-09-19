# 两查询弱条件参照：有界历史审计与实现

2026-09-13。可以作为已知 CDG 思路的经验基线测试；不认领新方法。本次检索到的仓库采样实现中，未发现精确的 vc + alpha * (vc - vhalf) 臂。“未发现”限于已查源码、目录和报告，不证明所有历史尝试都没有做过。尚无本构造的生成质量结果。

## 1. 与旧三查询构造的准确区别

令 \(v_0=v(e_u)\)、\(v_1=v(e_c)\)，\(v_r=v(e_u+r(e_c-e_u))\)。

- 仓库旧 #38 使用 \(v_1+\alpha\,\mathrm{normmatch}(v_r-v_0,v_1-v_0)\)，需要三次 full 查询。实现位于 experiments/sit_guidance_portfolio_20260910/operators.py 的 cfg_condition_secant 分支。
- 新经验基线使用 \(v_1+\alpha(v_1-v_r)\)，仅两次 full 查询。没有隐藏的原 null 查询，也没有恢复原 CFG gap 范数。
- APG 变体把新 gap 送入原 APG 的负动量、范数上限和 clean 方向正交投影；上限是新 gap 范数的两倍。两 Heun stage 共享旧历史，只提交左端点提出的新历史。

半条件时，记 \(g=v_1-v_0\)、\(k=v_1-2v_{1/2}+v_0\)，则

\[
v_{1/2}-v_0=(g-k)/2,\qquad v_1-v_{1/2}=(g+k)/2.
\]

两种割线在原 CFG 正交方向上的曲率修正**符号相反**。因此旧割线的质量收益不能当作新构造的收益证据。若模型对该 embedding 线段近似仿射，则新 gap 约为 \(g/2\)，主要效果只是把 alpha 减半。APG 的历史、cap 和投影对 gap 正齐次，所以同样存在这个强度解释。

已有质量记录：

| 已有构造 | 同 bank FID | 同 bank 对照 | 成本 |
|---|---:|---:|---:|
| 旧 #38 单组件，独立 5K | 22.180625 | CFG 22.478488；APG 21.533588 | 320 full 对 224；约 1.400 倍耗时 |
| 旧割线 + APG + channel rescale，另一独立 5K | 21.228058 | CFG 22.512481；APG 21.651659 | 320 full 对 224；约 1.377 倍 APG 耗时 |

来源：[portfolio 5K](../../SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_RESULTS_20260910_ZH.md)、[fusion 5K](../../SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md)。不同 bank 的数字不能直接排序。

## 2. 最近原始文献

[Condition-Degradation Guidance，2026-03-11，§4 Eq.5 与 §5 Eq.11](https://arxiv.org/html/2603.10780v1) 已直接提出用 degraded condition 替换 null，计算 conditional + alpha * (conditional - degraded)，并通过原条件与 null 的 masked interpolation 构造退化条件。论文的实际构造依赖文本内容 token 与上下文 token 的选择性替换；SiT 的单类别 embedding 均匀减弱没有该 token 结构，不能称完整 CDG 复现，也不能继承其语义解耦结论。

[CADS，ICLR 2024，§3.1](https://studios.disneyresearch.com/app/uploads/2024/05/CADS-Unleashing-the-Diversity-of-Diffusion-Models-Through-Condition-Annealed-Sampling-Paper.pdf) 则给 conditioning vector 加入随时间退火的 Gaussian noise，并可重标定均值/方差。它改变条件支的 conditioning，不是本例保持原 conditional 锚、使用确定性半条件负支的同一个算子。

## 3. 旧缓存能否不查询模型就预判方向

旧 portfolio/fusion 的 diagnostics 只保存归一化后修正相对原 gap 的幅度比、正交能量比例、clean 范数比等，并沿查询时间累计平均。它没有保留原始三场、未归一化半条件割线的范数和带符号内积，不能反推出 full-to-half gap 的同状态方向及时间变化。

本次检查的 #38 臂 i38_cfg_condition_secant_g2_p0 已只剩 manifest/summary/results；manifest 引用的 samples.npz 与原 batch 文件当前不在目录中。即便有最终样本，也不足以恢复三个速度。

本轮 transport_calibration/fields_t*_b*.npz 仅保存 q/raw/source_ids/t；计算时曾查询三分支，但没有将原始三场落盘。因此不能拿已有这些文件伪称已经完成 half-gap 的角度检查。下一步应由独立小 probe 在相同状态、相同时间显式查询。

## 4. 私有实现与已完成检查

实现：[degraded_condition.py](../../../experiments/cfg_transport_search_20260913/degraded_condition.py)。

配置为 kind=degraded_cfg 或 degraded_apg，condition_scale=.5（别名 scale）、alpha 为额外 guidance 系数；APG 默认 beta=-.5。采样采用原 64/96 步 Heun、cutoff=.75；64 步 alpha 非零时为 224 full、0 prefix。query_pair(rt,z,t,labels,r) 恰好返回两次 full 查询的 conditional/degraded 预测。

端点通过原生运算实现：r=0 直接查询原 null 标签；r=1 再查询一次原 conditional。只有中间 r 才覆写 y_embedder 输出，finally 清理 hook 并恢复标签。

已运行 CPU fake-runtime 检查：r=0 的 CFG/APG 完整轨迹及快照逐位恢复原基线；r=1 guidance 消失且完整轨迹逐位等于 conditional；alpha=0 回到 128 calls；half 确实改变轨迹；异常清理及参数边界检查通过。没有启动 GPU 或真实质量采样。

    /home/zhoushunyu/miniconda3/envs/myenv/bin/python \
      -m experiments.cfg_transport_search_20260913.degraded_condition \
      --output /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/degraded_condition_cpu.json

CPU 检查不是 checkpoint 上的逐位核验。真实 8 图检查与同状态 gap 幅度诊断由主任务安排；冻结参数后才比较生成质量，强度对照应覆盖近似 alpha/2 的原 CFG/APG。

## 5. 本轮停止的水平翻转方向

当前 runtime 使用 step_00800000.pt，训练直接读取 center-crop VAE moments；实际 cache manifest 明确 horizontal_flip=false，训练采样循环也没有在线翻转。加上 VAE 编解码与简单 latent 空间翻转不交换，不能把 latent 水平翻转称作此 checkpoint 训练分布必须满足的等变约束。

9 月 12 日的同预算 whole-field ABBA 和 conditional/null 错配视图已失败：[记录](../../REFLECTION_GUIDANCE_RESULTS_20260912_ZH.md)。[Visual Anagrams，§3.2 与 §3.5](https://arxiv.org/html/2311.17919v2) 已覆盖预测的变换平均、按步交替和 latent 变换伪影。本轮不再扩展该候选。

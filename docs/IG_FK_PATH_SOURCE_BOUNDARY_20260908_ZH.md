# 当前 FK 实验的已知来源与解释边界

本说明只记录方法边界，不是论文稿，也不改变已冻结采样协议。

1. 用户附件和本仓库 SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md 中的
   endpoint power tilt / posterior moment / Feynman–Kac 势为同一个理想对象。
   旧实际实现是拟合 normalized-HJB value，当前为随机路径直接输入梯度。
   不能仅凭换估计器就宣称有论文级贡献。
2. Conditional Diffusion Models with Classifier-Free Gibbs-like Guidance,
   https://arxiv.org/html/2505.21101v1 ，Proposition1 已给出 posterior Rényi
   梯度缺项；Proposition2 的小噪声消失依赖其假设，不能直接套到不一致内部头。
3. Feynman-Kac Correctors in Diffusion,
   https://arxiv.org/html/2503.02819 ，使用加权 SDE/SMC 来跟踪 prescribed
   noisy-time mixture。当前尝试固定 clean endpoint 的合法加噪路径之 score
   修正，目标路径不同；不能把两者直接当成同一数值算法。
   其 Appendix F.5 明确报告 SDXL GenEval 未获得稳定提升，并提出目标对指标
   未必更好或高维权重方差等可能原因，并未确定单一失败机制。
4. 当前 2 粒子、2 步、tau→.75tau 的估计是有偏截断近似；末端 C_s=1，
   没有求出完整 C_0=1 的全未来解。主采样仍使用已有 IG 时间窗口，不能据此
   声称主轨迹在所有时间精确具有理想 q_t 边缘分布。
5. 首批低噪声处 logweight 大且两粒子 ESS 接近1，属于估计限制。
   不通过除维度、温度或 clipping 隐藏该问题，也不将此诊断代替最终 FID。

若本轮质量失败，结论是这一个 direct plug-in estimator 无效；不等同于否定
精确密度下的恒等式，也不自动授权无穷粒子/截断/强度扫描。先报告完整结果。

## 追加：核查旧 FK 审计的实际代码

`experiments/audit_raev2_fkc_weight_degeneracy.py` 明确是 feasibility audit，
不是 FKC sampler：同类多个独立初始噪声沿普通 IG probability-flow ODE
推进，梯形累计正 running potential，再计算组内 softmax ESS；不包含
逐步 Brownian 创新、不从同一 z 分叉、不反传条件 log moment，也不重采样。
旧 ESS1.632/16（t约.95）、1.023/16（t约.90）说明该 ODE 上的累计势
跨轨迹差异很大，不能直接视为本轮条件 reverse-SDE 估计的重复实验，
更不能当成完整 FKC 算法的质量失败。此前摘要中的“粒子权重审计”须按此理解。

旧 `train_raev2_semigroup_value.py` 的 fitted Bellman target 则确实含
ordinary IG OU reverse-SDE 漂移和随机创新，但通过 value 网络 bootstrap，
并使用 normalized HJB 显式梯度平方项；与当前 direct pathwise estimator
仍是估计方式之别，而不是新的理论目标。

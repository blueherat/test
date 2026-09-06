# Guidance 的输运坐标投影：固定两种物理坐标

前两轮随机构造没有改善：official38.486774、ancestral38.894423、partial38.506604、MSE-calibrated38.541200。这里转向不同机制，不继续调随机强度。

来源：[CFG-Zero⋆](https://arxiv.org/html/2503.18886v2)、[仓库原文及代码核查](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md)。采用 optimized-scale 部分，**不采用 zero-init**，不添加起始窗口或剪裁。原论文中的普遍误差上界有已归档的反例；本次只依靠正确的最小二乘/正交约束，并将收益视为实证假说。

F/B 是同类 clean estimates。对指定物理预测 V_F/V_B，最小二乘系数 s*=<V_F,V_B>/||V_B||²，修正 V_F+.78(V_F−s*V_B)，保留 V_F 的弱预测平行分量，仅外推正交分量。可能修复的误差是 strong/weak 共享输运方向上被误当作质量差异放大的速度或噪声幅值。若该分量实际是有用的尺度校准，则投影会伤害 FID，这正是实验需要判定的地方。

只固定两个有物理定义的坐标：

- velocity：V_F=(F−z)/t，V_B=(B−z)/t，忠实迁移 CFG-Zero⋆。
- Gaussian noise：V_F=[z−(1−t)F]/t，V_B=[z−(1−t)B]/t，使用原加噪通道的噪声预测。t=1 在 clean-space 公式中取连续极限。

二者转换回 clean-space 都为 G' = G−.78 Proj_r(F−B)，其中 r 分别为 B−z 和 z−(1−t)B。每图全部 262144 维一个投影，没有逐 token 或逐频带系数。使用原生 BF16 G 加 FP32 修正，γ固定官方 .78，区间固定官方 [.1,1]。这和旧的 clean-token APG（相对 F、每个1024维token投影）有明确区别。

预定 1K：seed202609071、B8、相同噪声/标签、原模型/decoder/100步。两种坐标各一个候选，无超参扫描，无新增模型调用。对官方及历史 piecewise 基线比较；有显著方向信号再在独立 seed 及 5K 确认。保存失效结果，不宣称几何恒等式就是质量保证。

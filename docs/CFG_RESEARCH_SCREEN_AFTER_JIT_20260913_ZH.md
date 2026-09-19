# CFG文献筛选与当前实验选择

当前没有据此确立一个新的纯CFG方法。JiT实验首先确认既有MLP读出的迁移，并用接近同预算的官方CFG检查实际差距；随后的小规模组合应用也不被计作新的纯CFG原理。没有因文献中出现理论叙述就启动另一轮系数或窗口搜索。

重新核对了[Analytic Distribution of Classifier-Free Guidance for Schedule Design](https://arxiv.org/html/2607.19725v1)，尤其§4–5。它区分瞬时乘幂密度与实际ODE采样分布，给出路径积分修正，并据此提出DG-CFG调度。该文此前已经在仓库[原文](../readings/fsg_followup_20260910/2607.19725v1.txt)中留存，本轮复核不是新的文献发现。路径积分恒等式与某个调度的最优性是不同命题；我们不将又一条中间增强、两端减弱的时间曲线包装成新方法。

下面是独立的参数化检查，不是该文声称的结论：若做单调时间替换t=t(r)，VP噪声率变为β̃(r)=β(t(r))dt/dr，概率流路径积分中的β(t)dt保持不变。只规定“让(ω−1)β对所选时间坐标恒定”，得到的ω却会随坐标选择改变。因此，这类均衡准则还需要说明为什么采用该时钟；恒等式本身并不唯一推出一个坐标无关的最优调度。这不否定论文在其固定DDIM设置中的实验结果。

另读了Zhao与Schwing的[Studying Classifier(-Free) Guidance from a Classifier-Centric Perspective](https://ojs.aaai.org/index.php/AAAI/article/view/38329)，AAAI 2026，§3.2及其原始PDF。文章从决策边界分析引导，并用额外流匹配后处理检验经验观察。其关于条件前向核与无条件前向核是否相同的讨论，需要与真实联合分布和分别近似的网络严格区分。

对于按共同核构造的真实链C→X_t→X_{t+1}，条件独立就是构造性质，所以直接由Bayes公式有

\[
p(x_t\mid x_{t+1},c)
=p(x_t\mid x_{t+1})\frac{p(c\mid x_t)}{p(c\mid x_{t+1})}.
\]

它不要求正反向过程具有同一个条件依赖结构。分别训练的近似转移、分类器、有限步Gaussian近似以及放大引导不必满足这一恒等式；它们的轨迹不一致也不能单独反证真实链的条件独立性。这是本轮的代数核对，未据此把全部经验结果判为无效，更没有把“更远离决策边界”当作高质量的充分条件。

分裂查询、隐式固定点与逆无条件流也与仓库[FSG/CFG-MP比较](FSG_CFG_CTRL_CFG_MP_COMPARISON_20260911_ZH.md)及[既有Strang实验](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)重叠。仅交换调用顺序或提高ODE近似精度，没有提供新的生成质量保证。本轮不重启这些旧路线。

此次具体应用选择是：在完全相同的官方CFG求解器下，比较单独CFG、匹配附加强度的CFG、CFG+原IG、CFG+MLP IG。它直接检验新增读出在强基线中的价值，避免以单独IG相对较弱基线的收益替代实际用途。条件与弱模型引导的配合已有[IG](https://github.com/CVL-UESTC/Internal-Guidance)和[SGG](https://arxiv.org/html/2603.20584v1)等先例；[冻结协议](JIT_READOUT_CFG_APPLICATION_PROTOCOL_20260913_ZH.md)明确这是一项有失败门槛的应用检验。

这不改变“理想分布+共同误差分布+结构错配+加性误差”的建模起点。它要求解释性的假说最终产生新的受控质量收益；当前读出结果本身尚未鉴定误差分布、卷积核或误差共线性。[已保留的推导](AG_IG_STRUCTURED_MISMATCH_20260912_ZH.md)继续作为有条件模型使用。

本轮下载副本及SHA见[来源清单](../readings/jit_readout_confirm_20260913/source_manifest.json)。

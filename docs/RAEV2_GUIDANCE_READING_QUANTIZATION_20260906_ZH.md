# Guidance 的有限精度误差：PTQD 与 Q-Diffusion

日期：2026-09-06。只读两篇此前阅读集中未覆盖的一手论文、补充材料与官方源码；没有运行量化、GPU、模型或新实验。**最值得借鉴的是在实际外推后的算子上定义误差，并把相关误差、偏置和随机残差分开。修复这些误差本身属于数值实现校正，不能直接作为新 guidance 或 ≥5% 公平成本 FID 改善。**

[PTQD，NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)研究均匀 INT4/INT8 量化。主文 §4.1–4.2、式 (6)–(12) 的机制链为：归一化使量化误差可能与预测相关；写成 `e=k f+r`，先消除乘性偏差，再校正残差均值，并从已有采样噪声预算中扣掉残差带来的方差。令采样式为 `x_next=A(x)+c f+σz`，`r` 的均值/方差为 `μ_r,v_r`，则模型下使用

`f_corrected=(f_quant−μ_r)/(1+k)`，

`σ_corrected²=max(σ²−c² v_r/(1+k)²,0)`。

这是由误差模型导出校正量的优点；确定性 `σ=0` 没有可吸收的方差。§4.3 又以量化预测 SNR 与前向 SNR 比较选最低可用 activation bitwidth，权重固定；该规则源于噪声预算，仍不是质量最优定理。§5.1 用 FP/quant 的 1024 个样本估计统计，50K 评价；表 2 的 250 步、η=1、scale=1.5、W4A8 下，Q-Diffusion / PTQD / FP 的 FID 为 5.37 / 5.11 / 5.05。应把它读作量化损失的恢复证据。[主文 pp.4–9](https://papers.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Paper-Conference.pdf)

[补充 §A–B](https://papers.neurips.cc/paper_files/paper/2023/file/2aab8a76c7e761b66eccaca0927787de-Supplemental-Conference.pdf)给出 DDIM 系数和残差正态性检验。正态性未被拒绝只支持被检验样本的分布近似；回归残差与 `f` 不相关，不保证它独立于 `x`、条件类别或其他时刻。即使暂不考虑量化混合算术，在 `A(x)+cf` 中，只知道 `Cov(r,f)=0` 仍不足以删除 `Cov(r,A(x))`。因此方差扣减不能被升格为任意条件转移核、任意 ODE 或 FID 的保证。

官方实现提供了更直接的 guidance 细节。固定 commit `6b190f94b459ca869bf165509454c8676c3a62be` 的 [ddim.py:376–393](https://github.com/ziplab/PTQD/blob/6b190f94b459ca869bf165509454c8676c3a62be/ldm/models/diffusion/ddim.py#L376) 在**同一 x,t** 上分别算 FP 与 quant 的完整 CFG 输出，再记录两者差；[612–657 行](https://github.com/ziplab/PTQD/blob/6b190f94b459ca869bf165509454c8676c3a62be/ldm/models/diffusion/ddim.py#L612) 在 CFG 后减偏置、除 `1+k`，然后处理噪声方差。它没有只校准单支、也没有把两条不同轨迹的差冒充同状态量化误差。

[analyze_error.py:34–103](https://github.com/ziplab/PTQD/blob/6b190f94b459ca869bf165509454c8676c3a62be/quant_scripts/analyze_error.py#L34) 实际逐 t 做带截距回归、保存 slope；用于残差时将负 slope 置零。bias 表由**原始 error** 的逐通道均值生成，std 则为逐样本残差标准差再平均。因此源码并不逐项等于“残差均值＋全体 pooled variance”的理想表述；复制时必须重建量的定义。[量化校准脚本](https://github.com/ziplab/PTQD/blob/6b190f94b459ca869bf165509454c8676c3a62be/quant_scripts/quantize_ldm_brecqA.py#L109)还有权重 10000、activation 5000 次 block/layer 优化设置。估计这些 nuisance statistics 与校准的成本需要计入，不能因论文称 PTQ 就视为零训练开销。

一个由归一化机制得到的独立检查：设总体均值为零，`Cov(Y,e)=0`，`Var(Y)=s²`、`Var(e)=v>0`。将 FP 与扰动版本各自除以总体标准差，则

`Z=Y/s`，`Z_quant=(Y+e)/sqrt(s²+v)`，

`Cov(Z_quant−Z,Z)=s/sqrt(s²+v)−1<0`。

这个简化模型说明“归一化会产生相关误差”不推出“增益误差必为正”；PTQD 的 `k≥0` 是额外实现选择。这里没有证明 RAEv2 的 k 为负，只有不能未经测量就迁移其符号限制。类似地，“不相关”不等于“独立高斯”。

[Q-Diffusion，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)的主文 §3 把误差累积与输入分布变化连接起来：一次量化误差会改变后续输入，因此用模型自身多时刻轨迹做校准；UNet shortcut 拼接的两组 activation 范围不同，则拆开量化，而不是共用一个尺度。它的设计优点是校准数据覆盖实际计算分布，量化结构对齐张量来源。[补充 A.1–A.3](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Li_Q-Diffusion_Quantizing_Diffusion_ICCV_2023_supplemental.pdf)明确：CFG 每个 `(x,t)` 同时放条件/无条件两支；Stable Diffusion W/A 校准含 12800 个分支输入，attention scores 用 INT16，归一化与非线性仍保留全精度。因此 W4A8 标签不意味着整个网络都在 INT4/INT8 上计算。

固定 commit `715783da70baa267321d6700ceb8941400c309d1` 的 [get_train_samples:325–347](https://github.com/Xiuyu-Li/q-diffusion/blob/715783da70baa267321d6700ceb8941400c309d1/qdiff/utils.py#L325) 确认复制相同 x,t 并拼接两种条件。但 [block_recon:125–151, 204–205](https://github.com/Xiuyu-Li/q-diffusion/blob/715783da70baa267321d6700ceb8941400c309d1/qdiff/block_recon.py#L125) 优化的是缓存 block 输出的重建误差；**成对入集不等价于最小化外推后误差**。[txt2img 的默认校准参数](https://github.com/Xiuyu-Li/q-diffusion/blob/715783da70baa267321d6700ceb8941400c309d1/scripts/txt2img.py#L265)含权重 20000、activation 5000 次优化。量化器先 round/clamp 再反量化，后接普通 `F.conv2d/F.linear`，见 [quant_layer:66–89, 248–276](https://github.com/Xiuyu-Li/q-diffusion/blob/715783da70baa267321d6700ceb8941400c309d1/qdiff/quant_layer.py#L66)；这份模拟量化路径不能作为 INT4 实测延迟证据。它的 shortcut 分组也不自动适用于 DDT 的同类 Full/Base 输出。

以下是结合两篇得到的**本地代数推论**，不归为论文已经证明的 RAEv2 结果。固定同一状态和时间，令 FP32 参考为 `F,B`，低精度两支为 `F+e_F,B+e_B`，同类外推为

`G=B+w(F−B)`。

暂把额外混合算术舍入 `e_mix` 单列，则

`e_G=w e_F+(1−w)e_B+e_mix`。

忽略 `e_mix` 的协方差项仅为展示两支结构，中心化后有

`Σ_G=w²Σ_F+(1−w)²Σ_B+w(1−w)(C_FB+C_FBᵀ)`。

这是精确线性组合恒等式，不需要误差为高斯；完整核算还需加 `e_mix` 的方差及交叉项。在等方差、相关系数 ρ 的标量简化下，方差倍率为 `1+2w(w−1)(1−ρ)`。固定 `w=1.78` 时，ρ=0 给 3.7768，ρ=1 给 1：共享主干的相关误差可能抵消外推放大，不能套用“两支独立噪声”的结论。反过来，若 `e_F=e_B=d≠0`，guidance gap 完全正确而 `e_G=d` 仍然存在。只检查 `F−B` 会漏掉共同漂移；只检查单支 MSE 会漏掉协方差。

对当前 Euler `T(z)=z−h(z−G)/max(t,.05)`，同状态的 clean 输出误差转为 `h*e_G/max(t,.05)`，再叠加状态乘加舍入；完整轨迹还要经过后续动力学。故“输出误差小”“gap 准”“终点影响小”是三个不同命题。这提供自然的机制设计标准：**需要校正时，应在实际复合算子/更新单位下拟合可解释误差，而不是凭图片质量给 guidance gain 配一条曲线。** 这里没有批准拟合、增加参数或新实验。

迁移判断是：BF16 是浮点舍入，论文的 INT4/INT8 是带尺度/截断的均匀量化；共享 DDT 主干的 Full/Base 又不是条件/无条件两次 UNet。可以借用“同状态配对误差、相关增益与偏置、联合协方差、实际更新单位”这四个机制；不能照搬 k 的符号、各时刻误差表、独立高斯假设、INT 混合精度方案或 SDE 方差抵消。当前确定性 Euler 与已经失败的 8 图终点控制，都不因这些论文自动获得新准入。

如果某项改动只逼近原 FP32 算子，它回答的是数值保真；要成为本研究的新 guidance，还需从条件生成机制导出不同的有效向量场，并在固定算术、完整成本和独立质量评价下验证。两篇给的是错误建模的直觉，不是“改精度就是 SOTA”的故事。

一手全文、补充、14 份关键源码及 commit/tree 均归档到 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_quantization_v1`。逐项阅读范围/源行号见 [claim_sources.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_quantization_v1/claim_sources.json)，URL 与下载成本见 retrieval/code_sources/downloaded_code/supplement_retrieval；完整 SHA 见 [manifest.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_quantization_v1/manifest.json)。只记录实际测得的下载、转换和归档成本，不把网络工具延迟、未计时人工阅读或子进程 CPU 补成总研究耗时。没有模型/GPU/量化实验调用。

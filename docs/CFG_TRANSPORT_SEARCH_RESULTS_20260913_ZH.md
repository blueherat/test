# CFG 运输与控制机制：当前实验结果

目标仍是超过充分调参、计算量可比的基线。目前完成了基线复测、反演算子验证和一次真实图像监督筛选，**尚未找到通过独立确认的新方法**。本文件随后续实验更新。

## 已完成的强基线复测

SiT-S/2 ImageNet100 EMA 800K，SD-VAE，256px，同一套新 balanced 1K 噪声和类别。全部 12/12 配置完成采样及 ADM 评估。表中 alpha 是附加系数：普通 CFG 为 `v_c + alpha*(v_c-v_u)`；t≥.75 使用纯条件场。不同样本数、不同 bank 的绝对 FID 不直接横向排名。

|方法|alpha|Heun 步数|FID↓|sFID↓|IS↑|R18 目标 top1↑|单分支求值/图|
|---|---:|---:|---:|---:|---:|---:|---:|
|CFG 最优扫描点|1.25|64|44.910314|207.350780|62.237225|85.0%|224|
|APG 最优扫描点|2|64|43.464164|210.747747|63.177006|85.4%|224|
|历史调优 CTRL|2.75|64|43.481451|208.712332|62.198566|84.5%|224|
|加密 APG 最优扫描点|2|96|43.485220|210.775483|62.654457|85.4%|336|

APG 相对普通 CFG 的 FID 点估计下降 3.22%，CTRL 为 3.18%。这是已有方法的配对复测，不能记为新候选的收益。APG 和 CTRL 的 FID 差仅 .0173；在这个规模上不能据此声称可靠胜负。APG 的饱和像素比例 .01683，CTRL 为 .03109，空间 FID 则是 CTRL 较低。这提示可以检验互补性，并不证明组合必然更好。APG 96 步未改善本轮 FID，也说明增加模型调用本身不足以超过强对照。

来源：[独立核验后的完整结果](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k/verified_results.csv)、[请求及源文件身份](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k/request.json)。`verified_results` 分别检查主样本数、输入及参考身份；原始 controller 的 CSV 有 `samples` 键被评估文件路径覆盖的输出问题，不以该列作为样本数。

![完整基线扫描](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k/baseline_curves.png)

全部图像和初始/四分之一/二分之一/四分之三/终点 latent 已保存。五个时间点是一次采样的轨迹，**不是五次图片回灌**。

## 反演方向存在，但校准未通过

真实 SiT 探针验证了同场往返、异场运输和末腿抵消。小的同场往返误差支持数值实现正确；它不能证明生成质量。异场往返的大部分中间位移在完成最后一段生成后，会被等效 CFG 强度和输运所解释。完整结果见[算子探针](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/operator_probe/report.md)及[加密检查](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/operator_probe_refine/report.md)。

唯一保留的监督候选先计算精细条件逆流带回的方向，再去除现有 gap、无历史 APG 和两种便宜割线的张成部分。剩余 q 只依赖模型查询，不读取真实 clean 图像或本次噪声。200 张拟合图和 200 张留出图来自不重叠训练图像 ID，每个集合每类两张，FID 验证参考不参与拟合；模型冻结，仅拟合三个时间桶的标量。监督目标是线性 flow matching 的真实速度 `X-epsilon`。

|时刻|拟合系数 lambda|留出集 FM 风险变化，负值较好|
|---|---:|---:|
|.25|.165026|−3.39e−5|
|.5|.216282|−1.41e−5|
|.7|.000554|+5.63e−8|

联合变化为 −1.60e−5，95% 区间 [−4.76e−5,+1.40e−5]，各时间点的区间也均跨零。额外方向在几何上非零，**目前没有稳定的真实误差修正证据**。同一系数下加密逆流未扭转坏子集的符号。按事前门槛，该候选暂不进入 5K 图像实验。

这项阴性结果限于当前特征、时距、模型及监督桥分布；不能推出所有反演 guidance 都不可能有效。反过来，微小均值改善也不构成 FID 改善，更不能把理想模型也存在的异场回环当成模型错误。

推导、监督数据身份、数值检查和正反例见[完整机制报告](research/cfg_transport_search_20260913/mechanism_debate.md)。

## 下一项可检验问题

当前 CTRL 适配遵循递推 `m_k=g_k-K*sign(g_k+(lambda-1)*m_(k-1))`。恒定标量 gap、lambda>1 时，阈值为 `(lambda-1)*K/lambda`：阈值内是均值为 g 的两周期，阈值外是 `g-K*sign(g)` 的固定点。因此“CTRL 的全部收益来自振荡”并未被现有瞬时缩减对照证明。下一组消融将区分平均作用、交替作用和阶段内数值响应，使用相同求值预算。

这属于对 [CFG-Ctrl](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CFG-Ctrl_Control-Based_Classifier-Free_Diffusion_Guidance_CVPR_2026_paper.html) 离散适配的机制研究。[APG](https://arxiv.org/html/2410.02416v2) 已提出反向动量与 clean 方向投影；组合现有模块本身不当作方法新颖性。任何候选只有在本轮搜索胜过强基线，再冻结参数并完成独立噪声确认后，才可报告方法收益。

# 真实反演回灌修复器：训练与首轮评估复核

2026-09-13。只读检查训练日志、摘要、冻结评估源码及保存的 R0–R5 状态；重新计算百分比、逐轮误差和状态哈希。未新增训练、采样或模型前向。

**当前没有足够的生成收益证据。** 单向训练按预先指定目标选回初始恒等模型；双向训练仅获得 0.1424% 的选模目标下降，四个 CFG/APG 后处理臂的 FID 均略高于原基线。不能将它称为已改善生成的 guidance，也没有理由将五轮图像基本相似解释为已达到固定点。

## 1. 单向为什么选择零残差

[单向摘要](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/training/summary.json)与 rawmetrics 一致：完整训练 1500 步，逐 100 步评估，step 0 参与 \(L_{repair}+2L_{identity}\) 的最小值选择。

|单向检查点|repair MSE|identity MSE|选模目标|
|---|---:|---:|---:|
|step 0，恒等|.001836872427|0|.001836872427|
|最好的非零步，step 100|.001836867654|4.83361×10⁻⁷|.001837834375|
|step 1500|.001919722185|8.32725×10⁻⁵|.002086267210|

step 100 的修复改善仅 4.77303×10⁻⁹，远小于两倍 identity 代价 9.66721×10⁻⁷。其余已评检查点的总目标更高，因此选择 step 0 正确；不是训练没有执行，也不是检查点保存故障。它意味着这次单向配对训练未击败“保持不动”的预定基线。

## 2. 双向的收益与代价

[双向摘要](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_training/summary.json)选择 step 1000。四档顺序为 C(.75)、C(.5)、C⁻¹(.75)、C⁻¹(.5)，等权计算 repair。

|量|恒等基线|step 1000|变化|
|---|---:|---:|---:|
|repair MSE|.001776463236|.001772089629|降低 **0.246197%**|
|identity MSE|0|9.22150×10⁻⁷|新增绝对代价；不能对零基线报相对百分比|
|repair + 2 identity|.001776463236|.001773933928|降低 **0.142379%**|

修复误差减少 4.37361×10⁻⁶，其中 42.17% 被两倍 identity 代价抵消。identity MSE 为原始平均 corruption MSE 的 0.05191%；这是归一化量，不是 identity 相对零增加的百分比。step 1500 的 repair 虽更低，identity 代价更高，所以总目标仍输给 step 1000。

**改善并非覆盖全部回灌类型：** C(.75) 的 repair 变差 0.343661%，C(.5) 改善 0.015329%，C⁻¹(.75) 改善 0.678047%，C⁻¹(.5) 改善 0.521847%。原始 C 的两档平均误差反而增加 **0.031680%**；总体微小收益主要来自新增的逆向配对。M2 和 M4 的平均 repair 任务不同，也不能直接用两个绝对 loss 排名。

这 100 个 heldout source 被用于 15 个训练检查点的选择，不是独立质量测试；这里没有置信区间或统计显著性结论。

## 3. 四个后处理臂均未改善 FID

同一已有 1K bank、同一 5K ImageNet100 reference，原始生成端点直接复用。\(P_\beta(z)=z+\beta(P(z)-z)\)，每图只后处理一次。β=0 的 latent 和解码像素与缓存逐位一致；全部原始图片的解码复核也通过，避免把重新解码的差异误算成修复收益。

|原采样器|β|原 FID|后处理 FID|差值，越低越好|
|---|---:|---:|---:|---:|
|CFG extra α=1.25|.5|44.91031432|44.91722652|+ .00691220|
|CFG extra α=1.25|1|44.91031432|44.92149553|+ .01118120|
|APG extra α=2|.5|43.46416357|43.46536026|+ .00119669|
|APG extra α=2|1|43.46416357|43.47123883|+ .00707526|

四个 sFID 都有约 .03–.075 的微小下降，但 FID 未改善；没有独立 bank 或统计检验支持将这些小变化解释为稳健质量收益。β=1 的全 1K 平均 latent 改动 MSE 仅约 8.7–8.8×10⁻⁷，接近恒等的输出也是指标变化很小的直接背景。

来源：[CFG β=.5](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/infer/cfg_a1.25_s64_beta0.5/fid.json)、[CFG β=1](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/infer/cfg_a1.25_s64_beta1/fid.json)、[APG β=.5](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/infer/apg_a2_s64_beta0.5/fid.json)、[APG β=1](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/infer/apg_a2_s64_beta1/fid.json)。

## 4. 五轮是真实递归，但递归的是 Pβ

冻结源码每轮将上次 latent 送入 Pβ，重新生成下一 latent；R0–R5 均分别解码、保存 NPZ、图片和统计。四条链各 8 个输入，包含真实 heldout 与 CFG 生成输入、β=.5/1。已逐个读取全部 24 个状态文件，核对 round index、latent 哈希，重新计算相邻轮及对初始输入的 MSE，均与摘要一致；推理及 rounds 的源哈希、model 哈希也与 request 一致。

这里**没有反复执行原回灌 C、没有每轮重加噪，也没有将解码图片重新编码**。它检查的是修复器 Pβ 自身是否继续改动输入，不等同于原图经过生成模型的 no-op 回灌测试。

|输入，β|R1 对原 MSE|R5 对原 MSE|
|---|---:|---:|
|真实 heldout，.5|2.49462×10⁻⁷|6.21162×10⁻⁶|
|真实 heldout，1|9.97847×10⁻⁷|2.47472×10⁻⁵|
|CFG 生成，.5|2.42410×10⁻⁷|6.03530×10⁻⁶|
|CFG 生成，1|9.69642×10⁻⁷|2.40419×10⁻⁵|

前五轮的对原 MSE 约按 k² 累积，相邻轮改动只略减小，表现更接近持续的小偏移；不能称为已经收敛到固定点。这些 8 图链是诊断样本，未提供全体样本或无限迭代的性质。

来源：[五轮完整目录](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/rounds/summary.json)。本次结论限定在当前配对来源、592,100 参数修复器、损失权重与这组复用 1K 评估；它否定不了所有真实图锚定训练，也尚未支持扩大这一个候选。

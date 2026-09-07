# 最后三轮 guidance 研究：理论、实验与数据收束

2026-09-07。**整理稿：第3轮的唯一配对5K仍在采样，尚未形成最终质量结论。** 用户要求再做约三轮，未成功则收束并Git。第1轮已在预定机制门槛前止步，第2轮完整5K仅改善0.34595%，第3轮结束后不增加想法或超参数试验。执行前公式、门槛与后续结果见[三轮原始记录](RAEV2_FINAL_THREE_ROUNDS_20260907_ZH.md)。

## 从论文到这三轮的机制主线

[FSG原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b56d827a2b8433517e722e0272c7f464-Paper-Conference.pdf)的关键启发是：先形成带有条件信息的未来提案，再把它写回当前latent，使后续生成能够使用它。固定点为这种校准与反馈提供解释。附录A的CFG/CFG++重写使用校准前缓存的reference；重新读取校准后的latent会改变算法。这个区别直接决定最后一轮的设计。

对RAEv2，Full/Base接受相同类别并估计同一clean目标，因此不能把它们的预测一致自动当作质量目标。PFR保留当前弱推进参照，改动校准里的反事实负参考；其SiT正结果仍保留，但本次没有得到足以统一解释SiT成功与RAEv2迁移不佳的质量机制。此前的[时间求积](RAEV2_GUIDANCE_QUADRATURE_20260907_ZH.md)和[空间协方差](RAEV2_SPATIAL_GUIDANCE_COVARIANCE_20260907_ZH.md)均未达到3%，已结束。

本次收回了一个过强的局部论证：原Euler的写入分解对任意IG强度成立，而一次Jacobian给出的写入方向只在局部问题中最优。大外推下不能把后一结论当作实际有限响应的保证。解析反例、写入位置修正和完整推导保存在[方向回顾](RAEV2_LATENT_GUIDANCE_REVIEW_20260907_ZH.md)。

原生有限干预给出的正面证据是：在128个固定样本中，只额外写入一次原生IG、随后由Full推进，终点差对冻结消息均有正投影。早期响应更会改变方向，晚期更接近原消息；不同时间使用不同cohort，不能把组均值看成同批样本的时间曲线。这否定了“RAEv2普遍擦除这类一次写入”的强猜测，未证明正交分量有害或投影越大FID越好。完整[诊断与图表](RAEV2_FINITE_GUIDANCE_RETENTION_20260907_ZH.md)保留所有查询时刻。

## 保留下来的有限关系

令F为Full，G为原生BF16混合后转FP32的IG，M=G−F，q=(t−s)/t。在实数算术下：

    原生写入：z_s = Euler_F(z_t) + qM。
    后续读取：Δz_(j+1) = (t_(j+1)/t_j)Δz_j + (1−t_(j+1)/t_j)ΔF_j。

其中ΔF_j是两条实际latent轨迹经过Full的完整有限差，不用J_FΔz替换。若一路撤去后续IG，最后的原Euler步在实数下直接输出Full预测；终点影响因而必须通过后续模型响应实现，不能只靠原始写入被动搬运到终点。实际BF16/FP32舍入另行测量，不把数学等式冒充逐位等价。

三轮分别改变这条链条的一个环节：

| 轮次 | 问题与设计 | 理论保证的对象 | 不能由此推出 |
|---|---|---|---|
| 1 | 在原生latent与Full读取两项范数预算内，非线性搜索编码方向 | 实际预算不超出、冻结消息投影增加；原投影非负时正交响应能量不增 | 全局最优、完整语义留存、FID改善 |
| 2 | 用有/无本次IG写入的两次未来Base读取，构成ΔB；G_new=G−1.78ΔB | 相同未来时刻的有限反事实差，保留当前弱推进参照 | Base变得更准、减少该响应一定有益 |
| 3 | 把M前置到当前输入，再让Full重新读取；G_new=G+ΔF | 缓存读取的精确重写及真实有限Full反馈 | 反馈方向正确、收缩、质量提高 |

第3轮的前置写入是z_hat=z+(t−s)M/s，仅s>0有定义；缓存Full时(s/t)z_hat+(1−s/t)F(z,t)等于原IG一步。新动作是读取Full(z_hat,t)，其额外变化为q[Full(z_hat,t)−Full(z,t)]。公式保留完整外推幅度；“小时间步的修正阶数”与“IG幅度的一阶近似”是不同问题。实际实现保留原生运算顺序与零写入逐像素控制。

## 质量结果与成本

<!-- FINAL_RESULTS_START -->
第3轮结果待完成并独立复核。当前不能把归档进度称作质量目标完成。
<!-- FINAL_RESULTS_END -->

第1轮没有FID：128样本中15个发生改变、共20次样本更新被接受，终点投影仅4/8时刻组增加，未通过预定6/8门槛。全部有限预算与单调性由独立NumPy快照复算通过。实际904次B8前向、28次B8输入反向，四worker计算耗时合计52.00秒；这个固定求解器失败不等于不存在更好的可行解。

质量比较统一使用官方EMA100080、DINOv3-L K7、原decoder/stat、100步shift8、IG1.78与原活动区间、B8、nativeBF16/FP32/TF32on。5K seed202609072按每类五图，逐batch核对原始噪声与类别；指标为官方nanogen/ImageNet256 FID，另以FP64对称PSD公式独立重算。发现bank已反复用于探索，不能把小幅获胜点当作独立确认，也不能与论文50K指标直接比较。

成本表中的比值来自轨迹与解码worker耗时之和；基线与候选在相同硬件协议下先后运行，不靠千分位时间差宣称加速。准备、模型加载、机制诊断和输入反向成本分别保留。第2轮500000次主模型样本调用外加990000次Base-prefix调用；第3轮500000次主调用外加495000次完整Full读取。最后三轮没有训练模型权重。

## 可复核材料与复现入口

[证据目录](../experiments/results/raev2_final_three_rounds_20260907/INDEX.md)保存逐时刻统计、完整正负结果、审计、CSV、图表和状态；各`archive_manifest.json`记录实际读取计算的文件SHA、字节数和原始数据位置。大数组、5K图像与Inception特征留在数据盘，Git保存代码、协议及轻量证据。此处没有重新运行所有历史研究；更早轨迹通过[研究索引](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)和[前一阶段收束](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md)保留。

当前数据根目录：

- 原生有限读取诊断：`/home/zhoushunyu/data/eqvae/experiments/raev2_finite_guidance_retention_20260907`。
- 第1轮：`/home/zhoushunyu/data/eqvae/experiments/raev2_finite_read_budget_writer_20260907`。
- 第2轮：`/home/zhoushunyu/data/eqvae/experiments/raev2_causal_reference_20260907`。
- 第3轮：`/home/zhoushunyu/data/eqvae/experiments/raev2_full_read_after_write_20260907`。

保留以下复现入口，采样时使用新的输出目录；原始目录不会被覆盖。这些是手动复现说明，收束后不会自动重启。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m pytest -q \
  tests/test_raev2_causal_reference.py tests/test_raev2_full_read_after_write.py

/home/zhoushunyu/miniconda3/envs/myenv/bin/python \
  experiments/run_raev2_full_read_after_write.py \
  --output /home/zhoushunyu/data/eqvae/experiments/reproduce_full_read_after_write \
  --samples 5000 --seed 202609072 --modes full_read_after_write --gpus 0 1 2 3
```

第2轮对应`run_raev2_causal_reference.py`与`--modes causal_reference`。两个`audit_raev2_*`脚本接受`--folder`和`--output`重算配对身份与FID；`summarize_raev2_final_three_rounds.py`从完成的审计生成汇总表，`plot_raev2_final_three_rounds.py`生成可导出的PNG/PDF。原生诊断和第1轮机制程序提供已记录的cohort及有限求解协议，见三轮原始记录和数据内PLAN。

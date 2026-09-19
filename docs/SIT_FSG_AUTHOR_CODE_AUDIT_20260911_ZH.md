# FSG 作者代码与当前 SiT 适配：执行核对

**不能把当前 SiT 适配称为完整的作者代码复现。** 它的局部前向—反演公式已通过回放和直接计算检查；但普通推进、迭代调度、裁剪都与公开 SDXL pipeline 有差别。本页以实际代码及可执行检查为依据，不把论文的机制解释当作代码已经实现的性质。

检查的作者仓库为 `Ka1b0/Foresight-Guidance`，提交 `012398fae56912f88fd8fec588b4ceb92800d9d6`。pipeline 的 SHA256 为 `6ef11518bc116724437ebf6064b56ab7bd8a0bf92f5aac3c0610798582032d0a`；本地 checkout 无改动，与当天归档文件一致。

公开实现的执行顺序是：在指定步骤进行若干次 guided 前向与NULL inverse；随后查询当前latent的两个噪声预测，做CFG++状态校准；最后把校准前查询的NULL噪声预测传入DDIM，推进校准后的latent。它没有计算完整的 \(U(x)-C(x)\) 残差，没有按该残差接受或拒绝更新，也没有求到相等后自动停止。[作者主循环](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/utils/pipeline_stable_diffusion_xl.py#L995)、[CFG++函数](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/utils/pipeline_stable_diffusion_xl.py#L1114)

| 操作 | 作者公开SDXL配置/代码 | 本轮SiT算子适配 | 对解释的影响 |
|---|---|---|---|
| 普通推进 | CFG++校准后调用DDIM，使用校准前查询的NULL噪声 | Heun CFG，额外guidance在t≥0.75关闭 | 两者不是同一基础采样器 |
| 前瞻调度 | NFE50配置使用40个普通步；索引0/5/15执行2/2/1次，名义间隔5步 | Heun64，每4步一次，H=0.125 | 迭代数和分配不同 |
| 前瞻强度 | guidance_scale=5.5，inverse scale=0 | 前向权重w=1+a，主对照w=3.75 | 强度不可直接混称 |
| 状态位移裁剪 | 该公开函数没有本轮所用的相对半径裁剪 | r=(4/64)a‖g‖ | 会改变有限转移目标的实现程度 |
| 时间推进 | 使用真实DDIM scheduler及inverse scheduler | 显式Euler使用指定的H | 公开函数的实际步距需要检查scheduler实现 |

作者配置来源：[NFE50 YAML](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/configs/NFE-50.yaml)。它是SDXL示例；这些参数并不是在当前SiT-S/2上选出的参数。

2026-09-11重新执行了作者 `foresight_sampling_update` 的函数体：从源码AST直接提取，使用真实本地DDIM与DDIMInverse调度器，测试输入是合成CPU latent和相同的常量噪声预测。没有把作者函数改写成我们认为它应该实现的公式。当前 diffusers 为0.30.0，40个推理步、1000训练时间刻度，关闭clipping。此项合成测试使用调度器默认beta配置，没有加载SDXL的scheduler配置；下表RMS不是SDXL模型的实测误差。

| 名义前瞻 | 实际forward系数区间 | 实际inverse系数区间 | 作者函数往返RMS | 严格对齐往返最大误差 |
|---|---|---|---:|---:|
| 975→975，控制 | 975→950 | 950→975 | 2.23e−8 | 5.96e−8 |
| 975→950 | 975→950 | 925→950 | 0.003875 | 5.96e−8 |
| 975→850 | 975→950 | 825→850 | 0.019600 | 5.96e−8 |
| 975→725 | 975→950 | 700→725 | 0.039770 | 5.96e−8 |

原因是公开函数只替换了调度器的 `timesteps`，没有同步调整 `num_inference_steps`；该版本DDIM按后者计算系数步距。inverse接口的时间参数还对应其输出侧系数。于是，模型查询时间、状态实际推进区间、函数最终声明的时间没有完全对齐。每一行作者函数输出都与单独写出的实际DDIM系数公式逐元素一致，最大差为0。

这个复核说明，在上述软件组合中，公开代码的操作不能直接当作理想的长区间ODE反演。作者的[requirements](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/requirements.txt)没有固定diffusers版本；本次没有恢复其论文实验环境，也没有执行SDXL图像复现。因此不能由这个CPU结果断定作者的论文指标有误。

相应地，本轮数据能够分别回答三个问题：当前SiT适配实际怎样改变图片；高精度流反演是否实现给定未来目标；直接降低同一状态的条件/NULL完整未来差，是否改善图片。第三个问题使用独立的完整latent优化实验回答，目标由可执行代码显式定义，不依赖给FSG的操作附加“已经写入”或“已经一致”的解释。

本次原始重放：[summary.json](/home/zhoushunyu/data/eqvae/experiments/sit_fsg_ctrl_hypothesis_20260911/author_code_clock_replay_20260911/summary.json)；执行脚本：[audit_fsg_scheduler_clocks.py](../experiments/audit_fsg_scheduler_clocks.py)。当前SiT计算的独立检查见 [Jacobian与目标复核](SIT_FSG_JACOBIAN_AUDIT_RESULTS_20260911_ZH.md)，直接一致性检验见 [固定协议](SIT_GOLDEN_PATH_DIRECT_PROTOCOL_20260911_ZH.md)。

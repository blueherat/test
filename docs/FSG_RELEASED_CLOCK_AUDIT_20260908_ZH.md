# FSG公开实现的执行时间与查询时间检查

状态：CPU调度器行为已验证；未复现作者训练/采样环境，未得到新的生成质量结果。

## 确认范围

官方仓库[Ka1b0/Foresight-Guidance](https://github.com/Ka1b0/Foresight-Guidance)
commit `012398fae56912f88fd8fec588b4ceb92800d9d6`，pipeline源SHA256
`6ef11518bc116724437ebf6064b56ab7bd8a0bf92f5aac3c0610798582032d0a`。
已读infer.py、NFE-50配置、pipeline主循环与foresight_sampling_update。
requirements未固定diffusers版本。本地实测版本0.30.0；另检查上游v0.35.1源码。

公开方法临时替换scheduler.timesteps，但没有同步修改num_inference_steps。
DDIM step使用num_train_timesteps // num_inference_steps决定alpha索引跨度。
Inverse DDIM的参数t表示系数输出侧：其系数从t−stride到t，而不是t到t+stride。
因此“传入网络的t”“输入状态被调度器解释的t”“最终声明的t”需要分别记录。

## 最小数值证据

从公开文件AST直接提取该方法；使用真实本地DDIM调度器、合成CPU latent和常数epsilon，
禁用clipping，40 inference steps/1000 train steps。这里没有加载SDXL。

| interval索引差 | 名义前瞻目标 | 实际forward系数 | 实际inverse系数 | 往返RMS |
|---:|---:|---|---|---:|
| 0（控制） | 975 | 975→950 | 950→975 | 2.23e-8 |
| 1 | 950 | 975→950 | 925→950 | .003875 |
| 5 | 850 | 975→950 | 825→850 | .019600 |
| 10 | 725 | 975→950 | 700→725 | .039770 |

各行都与独立显式DDIM系数公式逐元素一致。真正对齐的975→目标→975常epsilon
往返误差均小于6e-8。首个测试版本误把inverse接口解释为t→t+stride而断言失败；
读实际step源码后修正为t−stride→t，失败目录保留，不能隐藏这个验证过程。

## 研究含义及不能推出的结论

至少在上述明确的软件组合中，公开函数不是其名义时间端点之间的精确forward/inverse
算子。它确实分离了状态推进跨度和未来模型查询坐标，与仓库PFR的反事实时间问题有关。
这不证明作者论文实验使用了这个组合，更不证明论文质量结果无效。
不得将此CPU检查称为FSG图像复现、PFR质量解释或新方法成功。

对于线性flow的独立研究，一个短状态步长h与独立查询间隔H产生的简化往返可写为

    z' = z + h G(z,t)
    T(z) = z' - h W(z',t+H)
         = z + h[G(z,t)-W(z,t)] - h[W(z',t+H)-W(z,t)]。

这是恒等式，不是公开DDIM函数的逐系数等价式。它将普通contrast与有限反事实reference
响应分开，给出下一项值得检验的区别：收益来自真正长区间输运，还是参考查询坐标的
改变。仓库已有Euler/未来参考反例必须同时作为约束；不能仅靠新命名声称创新。

脚本：`experiments/audit_fsg_scheduler_clocks.py`；完整记录位于
`~/data/eqvae/experiments/pfr_condition_retention_20260908/fsg_clock_audit_retry/`。

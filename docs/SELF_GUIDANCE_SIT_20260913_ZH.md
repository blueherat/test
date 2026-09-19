# Self-Guidance：官方代码核查与小 SiT 迁移实验

**已完成：官方实现存在；本次按官方预测组合规则适配小SiT，完成12组×1K筛选和4组×1K独立种子复核，共16000张生成与评测。未观察到稳定质量收益，保留代码作为可复用基线，不建议据此替换现有采样默认设置。**

2026-09-13。论文是 [Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/abs/2412.05827)，作者 Tiancheng Li 等。已找到 [MAPLE Lab 官方实现](https://github.com/maple-research-lab/Self-Guidance)，README 标注 2025 年 7 月发布，包含 SD、SD3、FLUX、CogVideoX。2026-09-13 实际 `git ls-remote origin HEAD` 与本地干净 checkout 一致：`843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d`。

本地官方仓库：`/home/zhoushunyu/data/eqvae/baselines/Self-Guidance`。本项目原有的 SD1.4 COCO1K 配置停在等待模型阶段；本次不把这些旧配置当成已完成结果。当前已具备 SiT 权重和 ADM 评测资产，因此实际执行的是现有 ImageNet100 SiT-S/2 的迁移实验，不是 SD3/FLUX 论文表格复现。

## 方法对应与边界

官方 FLUX `pipeline_flux.py:310–340` 和 SD3 `pipeline_sd_3.py:351–403` 实际直接组合模型预测。采用此发布版规则：

`v_out = v_c + alpha*(v_c-v_u) + omega*(v_c-v_ref)`。

`alpha` 是 CFG 额外系数；常规 CFG scale 为 `1+alpha`。SG 的参考是在**同一个 latent、同一类别**上查询更噪时间；SG-prev 则是前一个已接受步骤、前一个状态的原始条件预测。缓存不会混入 CFG 或 SG 的输出。

SiT 使用 `z(t)=t*x+(1-t)*epsilon`，从 0 到 1 去噪。官方 `t_noise/1000=1-t`，故官方加 10 的时间偏移转换成 `t_ref=max(0,t-.01)`，默认 `sg_prev_max_t=500` 转换成严格 `t>.5`。SG 首步偏移被裁剪到同一时间，复用当前预测，省掉一次完全重复前向。SG-prev 每条采样链重新清空历史，前半程仍缓存原始预测，首个激活步骤使用紧邻上一步。

这里实现的是官方 flow 预测组合；论文密度比在 score 空间写出，跨时间 velocity 与 score 的转换包含不同系数，因此不能声称当前直接 velocity 外推严格等于论文密度比的精确 score。SG-prev 使用 Euler，以保持“上一个接受步”的明确含义。

## 结果产生前固定的评测

- 模型：ImageNet100 SiT-S/2，800K EMA；现有 SD-VAE 解码；FP32 和现有 runtime TF32 设置。
- 12 组，每组 1,000 张，全部共享 seed `2026091317` 的噪声和类别；100 类各 10 图；batch 16。
- 主对照：64 步 Euler 的条件 strong 和 CFG（额外 alpha=1.25，t<.75 启用）。SG、SG-prev 分别在这两条主线上测试 omega=1/3；SG shift=.01，SG-prev t>.5。
- 额外对照：CFG Euler100 与原生 CFG Heun64。Euler64 CFG 为112次单分支前向/图；其 SG 为175次，SG-prev仍112次。Euler100 CFG为175次，为SG提供相同前向预算对照；Heun64 CFG为224次。
- 指标：同一 ImageNet100 验证集5K参考的 ADM FID、sFID、IS；保存完整生成像素、终点latent、Inception特征、固定前32图预览、输入/源码/资产哈希和实际前向计数。FID越低越好。
- 1K结果用于观察迁移趋势；不与论文绝对FID比较，不把同一bank挑选的最优值当独立确认。

## 运行

适配代码：[sampler.py](../experiments/self_guidance_20260913/sampler.py)。先进行独立公式/边界检查以及真实模型零强度与原生Heun逐元素一致性检查：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.check --model
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.run prepare --phase screen_1k --config experiments/self_guidance_20260913/configs.json --sampler experiments.self_guidance_20260913.sampler --samples 1000 --seed 2026091317 --batch 16
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.run controller --phase screen_1k --gpus 0,1,2,3
```

输出目录：`/home/zhoushunyu/data/eqvae/experiments/self_guidance_20260913/screen_1k`。首轮12组均已完成；结果与证据见下文。

## 独立种子复核：结果产生前固定

首轮 CFG 基线 FID=44.8835；SG omega=1/3 分别为44.5363/44.5356，差距不到0.001，因此采用更小的 omega=1，不把这点差距当作可靠参数排序。SG-prev omega=1 为44.8174，omega=3 为45.9762，因此固定 omega=1。首轮不加CFG的四个候选全部差于基线，暂不扩大。

复核固定4组：`cfg_e64`、`cfg_sg_w1`、`cfg_sg_prev_w1`、`cfg_e100`。参数不变；使用独立 seed=2026091318 的1000张配对噪声与类别，沿用相同资产、参考、精度和batch，目录为 `confirm_1k`。仅观察独立bank上是否保持改善及是否胜过等前向预算Euler100，不再在此bank调参；两个bank都保留原始结果。

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.run prepare --phase confirm_1k --config experiments/self_guidance_20260913/confirm_configs.json --sampler experiments.self_guidance_20260913.sampler --samples 1000 --seed 2026091318 --batch 16
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.run controller --phase confirm_1k --gpus 0,1,2,3
```

## 首轮完整结果

全部12组成功退出，每组1000张，共12000张。以下FID/sFID越低越好，IS越高越好；时间为各batch采样与解码累计的GPU秒，包含生成全部1000张，不含模型加载、预热和指标计算。各组采用相同计时口径。

|配置|FID ↓|相对同源Euler64 ΔFID|sFID ↓|IS ↑|前向/图|采样+解码GPU秒|
|---|---:|---:|---:|---:|---:|---:|
|`strong_e64`|87.941|+0.000|221.040|28.902|64|41.1|
|`strong_sg_w1`|89.543|+1.602|225.718|27.767|127|66.4|
|`strong_sg_w3`|90.875|+2.933|232.622|26.289|127|66.2|
|`strong_sg_prev_w1`|89.665|+1.724|225.274|27.470|64|40.6|
|`strong_sg_prev_w3`|96.323|+8.382|234.972|24.127|64|41.0|
|`cfg_e64`|44.884|+0.000|206.876|60.992|112|60.2|
|`cfg_sg_w1`|44.536|-0.347|209.549|62.086|175|85.4|
|`cfg_sg_w3`|44.536|-0.348|214.981|60.559|175|85.4|
|`cfg_sg_prev_w1`|44.817|-0.066|208.705|60.637|112|60.7|
|`cfg_sg_prev_w3`|45.976|+1.093|213.460|58.832|112|60.2|
|`cfg_e100`|44.832|-0.051|206.917|61.773|175|85.6|
|`cfg_h64`|44.763|-0.120|207.104|62.008|224|105.2|

不加CFG的4个SG/SG-prev候选均使FID、sFID、IS变差。加CFG时，SG omega=1使FID下降0.347（0.77%）、IS略升，但sFID上升2.673，采样加解码耗时增加41.78%；omega=3并未进一步改善FID，sFID恶化更多。SG-prev omega=1只改善0.066 FID，而omega=3恶化1.093。按首轮结果，尚无充分理由用SG替换当前默认采样。

[首轮固定前6个样本、全部12组对比图](data/self_guidance_20260913/screen_1k/comparison.png)；[完整CSV](data/self_guidance_20260913/screen_1k/results.csv)；[请求与源码/资产哈希](data/self_guidance_20260913/screen_1k/request.json)；[审计记录](data/self_guidance_20260913/screen_1k/results.json)。每组自己的grid.png还保留固定前32图，完整像素与latent位于数据盘原输出目录。

实现检查通过：独立解析模型的同状态更噪时刻、CFG+SG加法、原始条件历史、严格半程门控、历史重置、label恢复和实际调用计数；真实SiT上两种方法的零强度与Euler基线逐元素一致，Heun对照与仓库既有实现逐元素一致，非零强度生成有限且确实改变输出。归档另核对源码/资产/输入/生成像素/每批哈希、类别配对、每组数量、终点与Inception特征有限性。

## 独立种子复核结果与结论

复核4组全部完成，无配置失败；新增4000张生成和ADM评测。两个seed的每组FID都各用1000张计算，没有把不同样本数FID混为同一口径。

|配置|FID ↓|相对同源Euler64 ΔFID|sFID ↓|IS ↑|前向/图|采样+解码GPU秒|
|---|---:|---:|---:|---:|---:|---:|
|`cfg_e64`|44.870|+0.000|209.505|62.481|112|60.6|
|`cfg_sg_w1`|44.893|+0.023|212.424|62.048|175|86.2|
|`cfg_sg_prev_w1`|44.975|+0.105|211.443|62.125|112|60.4|
|`cfg_e100`|44.848|-0.022|209.569|63.142|175|85.7|

首轮SG omega=1相对CFG Euler64的ΔFID=-0.347，在独立bank变为+0.023；SG-prev omega=1从-0.066变为+0.105。两种方法的sFID在两个bank上均差于CFG Euler64；复核中IS也下降。SG复核耗时约86.2 GPU秒，对照60.6秒，增加约42%；相同前向预算Euler100为44.848 FID，略优于SG的44.893。

因此当前证据不支持稳定的FID或综合质量改进。不加CFG时，本轮测试的两种方法及两个强度均明显变差。这里的结论只适用于冻结的小SiT、时间偏移.01、SG-prev默认后半程窗口和有限参数；没有做更大样本显著性检验，也没有复现论文原SD3/FLUX模型，不能据此否定论文在那些模型上的实验。

独立bank没有保留首轮正收益，故本次有界测试结束，不追加大规模搜索或训练。官方仓库与本地适配、复现实验命令、原始生成和指标全部保留。

[复核同种子对比图](data/self_guidance_20260913/confirm_1k/comparison.png)；[复核CSV](data/self_guidance_20260913/confirm_1k/results.csv)；[复核请求](data/self_guidance_20260913/confirm_1k/request.json)；[复核审计](data/self_guidance_20260913/confirm_1k/results.json)。两个阶段的所有源/资产哈希和逐batch校验均通过，控制器退出码全部为0。

生成轻量归档与全部方法的固定样本对比图：

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.report --phase screen_1k
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_20260913.report --phase confirm_1k
```

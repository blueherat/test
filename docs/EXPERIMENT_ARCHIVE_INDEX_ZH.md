# 实验归档索引

更新时间：2026-08-11

## 归档目的

本索引登记 2026 年 8 月前后尚未提交的探索代码和结论记录。它们包含成功结果、负结果和中途原型，保留的目的主要是：

1. 防止后续重复进行已经做过的实验；
2. 保存可复核的机制证据和失败边界；
3. 明确哪些代码是当前主线，哪些仅供历史追溯；
4. 让实验数据继续留在 `/home/zhoushunyu/data` 或 `/data/shared`，Git 只保存代码、测试和小型文档。

本批归档不包含 checkpoint、样本张量、数据集、缓存或 FID 特征文件。

## A. 预测目标、AutoGuidance 与 Internal Guidance

### 结论性文档

- `docs/FREQUENCY_PREDICTION_EXTRAPOLATION_AUDIT_ZH.md`
  汇总频率外推、预测目标外推、v3/v4 公平性修正和当前研究判断。核心结论是：预测目标之间的差异属于有限模型 estimator gap；正向 beyond-x 外推不是普遍有效方法。
- `docs/GUIDANCE_TOY_AG_IG_PTG_RESULTS_ZH.md`
  汇总 AutoGuidance、Internal Guidance 和 prediction-target guidance 在二维/低维机制实验中的定量比较。该文档同时登记官方复现与论文规格重建之间的边界。

### 仍可复现的核心入口

- `experiments/run_prediction_target_extrapolation_toy_v3.py`
- `experiments/run_prediction_target_toy_v3_exact_multiseed.sh`
- `experiments/reanalyze_prediction_target_toy_v3.py`
- `experiments/summarize_prediction_target_toy_v3_replays.py`
- `experiments/run_guidance_fractal_toy.py`
- `experiments/train_prediction_target_internal_guidance.py`
- `experiments/evaluate_prediction_target_autoguidance.py`
- `experiments/evaluate_guidance_toy_benchmark.py`
- `experiments/analyze_guidance_sharpness_tradeoff.py`
- `experiments/analyze_prediction_target_cluster_separation.py`
- `experiments/summarize_internal_guidance_multiseed.py`

### 历史机制扫描

下面的 v5/v6/v7 文件保留用于追溯 Bayes oracle、宽 gamma、轨迹和连续分布检查。它们不再是当前推荐入口，也不应替代后来经过严格公平性审计的 v4/v10 和闭环螺旋实验。

- `experiments/run_prediction_target_bayes_oracle_v5.py`
- `experiments/run_prediction_target_bayes_oracle_v5_multiseed.sh`
- `experiments/summarize_prediction_target_bayes_oracle_v5.py`
- `experiments/run_prediction_target_bayes_oracle_v6_trajectory.py`
- `experiments/run_prediction_target_bayes_oracle_v6_confirmation.sh`
- `experiments/run_prediction_target_bayes_oracle_v6_multiseed.sh`
- `experiments/run_prediction_target_bayes_oracle_v6_wide_gamma.sh`
- `experiments/reevaluate_prediction_target_bayes_oracle_v6_wide_gamma.py`
- `experiments/summarize_prediction_target_bayes_oracle_v6.py`
- `experiments/run_prediction_target_bayes_oracle_v7_continuous_multiseed.sh`

归档判断：这些版本有诊断价值，但不能单独作为论文结论；后续引用数字时应优先查对应文档的限制说明和原始配置。

## B. RAEv2 频率轴探索

入口：

- `experiments/run_raev2_frequency_axis_extrapolation_suite_v3.py`
- `experiments/resume_raev2_frequency_axis_suite_v3.sh`

状态：探索性、非当前主线。

这组代码研究 frozen RAEv2 predictor 对输入频率扰动的局部响应，并与 internal-guidance gap 比较。已有证据说明该轴数值稳定且不等同于旧 IG 方向，但没有证明它能改善闭环 FID。正式判断记录在 `docs/FREQUENCY_PREDICTION_EXTRAPOLATION_AUDIT_ZH.md`。

## C. 双预测目标闭环 Toy

### 32 分量 Gaussian spiral mixture

- 报告：`docs/DUAL_TARGET_CLOSED_LOOP_TOY_ZH.md`
- 核心代码已在前一批提交中登记为通用双头闭环诊断模块。

该版本使用已知二维子空间中的 32 分量 Gaussian mixture，适合检查 oracle gate、端点放大和 shared-head 公平对照。

### 连续螺旋

- 正式报告：`docs/DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md`
- 精简结果包：`dual_target_closed_loop_spiral_toy_v1/`

连续螺旋版本是当前更严格的后续实验。阅读闭环动力学结论时，应优先使用连续螺旋报告，同时把 32 分量版本作为独立控制和历史证据。

## D. Imagenette SD-VAE / Latent Diffusion 原型

状态：代码和单元测试可用，但没有在本次归档中附带正式生成结果，因此属于探索性原型。

### 单头 Stable-Diffusion 风格基线

- `experiments/imagenette_sdvae_latent_diffusion.py`
  使用 frozen `stabilityai/sd-vae-ft-mse`、class-conditional U-Net、SD v1 scaled-linear noise schedule、epsilon prediction、EMA、DDIM 和 DDP。
- `experiments/run_imagenette_sdvae_ldm256_4gpu.sh`
  256 分辨率四卡 benchmark/train 入口。
- `experiments/evaluate_imagenette_sdvae_ldm128_baseline.sh`
  128 分辨率 checkpoint 的 CFG 预览、5K 采样和匹配评估入口。
- `experiments/evaluate_imagenette_sdvae_samples.py`
  使用与真实 Imagenette 匹配的图像处理流程计算生成指标。

### 双预测目标 latent 原型

- `experiments/imagenette_dual_target_latent.py`
  frozen SD-VAE latent 上的共享 U-Net 双头实验，可比较 clean/noise 头及其 clean-space extrapolation。
- `experiments/run_imagenette_dual_target_latent_4gpu.sh`
  四卡多 seed 启动入口。
- `experiments/run_imagenette_dual_target_direct_after_v.sh`
  等待既有任务后接续 direct-loss 对照的历史启动脚本。

### ImageNet-100 tokenizer reconstruction floor

- `experiments/build_imagenet100_sdvae_reconstruction_npz.py`
  把缓存的 SD-VAE posterior mean 解码为 ADM FID 格式，用于测 tokenizer reconstruction floor。它不是生成 FID。

这些脚本的默认路径遵循本机约定：只从 `/data/shared` 读取数据，把生成资产写入 `~/data/eqvae`。Git 不保存这些资产。

## E. 开放问题

- `docs/archive/RAEV2_OPEN_QUESTIONS_ZH.md`

该文件只是当时的研究备忘，不是实验结论。后续若问题已被实验回答，应在新报告中引用证据，而不是直接修改历史记录的含义。

## 验证状态

归档提交前完成：

```text
Python tests: 55 passed
Python entry-point compilation: passed
Shell syntax checks: passed
Secret-pattern scan: no credential found in this archive scope
```

测试覆盖：

- guidance fractal toy；
- sharpness/distribution trade-off；
- AG/IG/PTG benchmark；
- Bayes oracle v5/v6；
- prediction-target v3 protocol；
- Imagenette SD-VAE latent diffusion；
- Imagenette dual-target latent；
- Imagenette sample evaluator。

## 使用原则

1. 文档中的负结果也应保留，不能只挑有效的设置。
2. v5/v6/v7 是历史机制扫描；新的主张应使用后续公平对照或重新运行。
3. Imagenette 文件目前只能证明接口和训练流程可运行，不能据此宣称生成质量改善。
4. 所有默认绝对路径都是本机复现便利项；迁移机器时应通过命令行参数或环境变量覆盖。
5. 任何进入论文的数字都必须回到对应 CSV/JSON、正式配置和多 seed 结果再次核验。

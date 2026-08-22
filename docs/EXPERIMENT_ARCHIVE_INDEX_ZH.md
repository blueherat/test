# 实验归档索引

更新时间：2026-08-22

## 归档目的

本索引登记 2026 年 8 月前后尚未提交的探索代码和结论记录。它们包含成功结果、负结果和中途原型，保留的目的主要是：

1. 防止后续重复进行已经做过的实验；
2. 保存可复核的机制证据和失败边界；
3. 明确哪些代码是当前主线，哪些仅供历史追溯；
4. 让实验数据继续留在 `/home/zhoushunyu/data` 或 `/data/shared`，Git 只保存代码、测试和小型文档。

本批归档不包含 checkpoint、样本张量、数据集、缓存或 FID 特征文件。

## 0. 已终止：ImageNet-100 SiT 对角矩残差

- 理论与终验：`docs/MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md`
- 便携指标：`docs/data/imagenet100_sit_moment_residual_pilot20k/`
- 核心实现：
  - `experiments/imagenet100_sit_moment_residual.py`
  - `experiments/estimate_imagenet100_sit_diagonal_moments.py`
  - `experiments/analyze_imagenet100_sit_moment_residual_pair.py`
  - `experiments/run_imagenet100_sit_moment_residual_pilot20k.sh`
  - `tests/test_imagenet100_sit_moment_residual.py`

结论：moment-exact affine decomposition 的数学部分成立，但同预算 raw-only SiT-S/2
终验中，diagonal residual 模型的 paired validation velocity MSE 略高，FID-5K 从
`155.6983` 恶化到 `161.0571`。该路线按负结果归档，不再继续扩展。

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

## F. ImageNet-100 SiT FID 总表

- `docs/IMAGENET100_SIT_FID_INVENTORY_ZH.md`

该文档统一登记原生 velocity 单头、单 prediction-target、JiT-style x/velocity、动态
双头及静态双头混合的全部正式 FID-5K 结果。它同时标记 700K baseline 异常点、800K
重复复评和单 seed/FID-5K 的使用边界。引用 SiT 数字时应先查该总表，再回到对应的本地
JSON、manifest 和 checkpoint SHA256 复核。

### 独立 SiT-v / JiT-x 静态速度场扫描

- `docs/IMAGENET100_SIT_V_JIT_X_STATIC_SWEEP_ZH.md`
- `docs/data/imagenet100_sit_v_jit_x_static_sweep_400k.csv`

该报告登记两个独立 400K 单头 checkpoint 的 15 点配对内插/外推实验。当前最明确的
现象是：从 SiT-v 朝 JiT-x 内插会持续恶化，而沿反方向越过 SiT-v 会持续改善至已测
边界 `scale=-1`。报告同时保留单 seed、FID-5K、最佳点在边界和缺少 precision/recall
等限制，不能与同一双头 checkpoint 内部的静态 head mixing 混为一谈。

### 400K 训练方向与共同/独有分量记录

- `docs/IMAGENET100_SIT_400K_FUTURE_COMMON_UNIQUE_RESULTS_ZH.md`
- `docs/data/imagenet100_sit_400k_future_common_unique/`

该记录包含 `v800-v400` 与两条 400K guidance 的逐时刻对齐统计，以及 `x400`、
`v270` 正交 guidance 的双向 common/unique 投影 FID-5K。仓库保存汇总 CSV/JSON 和
两张结果图，不保存逐样本表、生成样本或 checkpoint。

### 400K 有限强度 Guidance 动力学审计

- `docs/IMAGENET100_SIT_400K_FINITE_GUIDANCE_DYNAMICS_ZH.md`
- `docs/data/imagenet100_sit_400k_finite_guidance_dynamics/`

该记录包含 finite-gamma 线性度、frozen-gap 与 closed-loop 的配对 FID-5K、
conservativity、局部 density-action、跨方向终点响应和 finite-strength exact gauge
toy。正式结论是：基线轨迹上的有限幅度 guidance action 保留了大部分 FID 收益，
闭环反馈有次要增益；field norm、cosine、local action 和 curl 均不能单独预测终点质量。

### 800K nominal-path frozen guidance

- `docs/IMAGENET100_SIT_NOMINAL_GUIDANCE_TRANSFER_800K_RESULTS_ZH.md`
- `docs/data/imagenet100_sit_800k_nominal_guidance_transfer/`

该记录使用 `v800` strong model 和 `x800`、`v500` 两类 weak model，比较 baseline、
frozen、replay 与 closed AutoGuidance。结果表明 nominal gap 必须经过当前状态上的 strong
field 响应才能产生主要收益；closed 在线重算的额外作用主要来自相对 nominal gap 的方向
变化，而非单纯强度重标定。

### 800K tangent transport 首轮审计

- `docs/IMAGENET100_SIT_800K_TANGENT_TRANSPORT_RESULTS_ZH.md`
- `docs/data/imagenet100_sit_800k_tangent_transport/`

该记录用中心差分验证 strong-flow JVP，并比较 gamma-zero variational tangent、exact
frozen 与 closed endpoint response。tangent 在 `gamma<=0.1` 时可以定量预测 frozen；
实际使用的 `gamma=1` 仍保留约 `0.846` 的方向 cosine，但非线性残差达到 `0.62-0.74`，
因此不能单独解释有限强度 frozen response。

### 800K tangent endpoint 投影审计

- `docs/IMAGENET100_SIT_800K_TANGENT_PROJECTION_RESULTS_ZH.md`
- `docs/data/imagenet100_sit_800k_tangent_projection/`

该记录把 `gamma=1` exact frozen endpoint response 逐样本分解为 transported tangent
方向上的投影与正交余量，并分别解码 5000 张严格配对样本。`x800` 与 `v500` 两组中，
平行投影分别保留约 `93.4%` 和 `87.1%` 的数值 FID 收益；正交余量单独使用时均使
FID 恶化。该实验说明 tangent 方向承载主要有益 endpoint 作用，但投影系数依赖 exact
frozen endpoint，因此当前属于机制 oracle，而不是可部署方法。

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

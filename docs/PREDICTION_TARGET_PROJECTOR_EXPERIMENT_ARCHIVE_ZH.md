# Prediction Target、Projector 与多头实验归档

## 1. 归档目的

本文件登记 2026-08-22 工作区中尚未提交的机制实验。它们属于已经执行过的
探索或实现储备，不构成当前新方法的证据，也不应和矩精确残差 Flow Matching
的理论命题混写。

本次只提交代码、测试、小型 CSV 和文字结论。ImageNet、latent cache、官方模型、
训练 checkpoint、生成样本和大型中间数组均未加入 Git。

## 2. Rank 与 operator toy

主要入口：

- `experiments/run_prediction_target_rank_symmetry_toy.py`
- `experiments/audit_prediction_target_rank_spectra.py`
- `experiments/run_prediction_target_operator_k_toy.py`
- `experiments/run_prediction_target_operator_hybrid_from_rank_baseline.py`
- `experiments/run_prediction_target_spectral_preconditioning_toy.py`
- `experiments/prediction_target_spectral_projector.py`
- `prediction_target_rank_operator_toy_v1/README_ZH.md`

已经确认的边界：

1. 固定输出 rank 时，`x/v/eps` 的 teacher-risk 排序可以随 rank 翻转。
2. 在线性 oracle 子空间上，解析处理法向速度显著改善 teacher MSE 与 SWD。
3. oracle 子空间 `P` 使用真实线性几何，只是 upper bound，不能直接迁移到图像。
4. 由训练分布均值和协方差得到的 LMMSE residual 不使用真实 `P`，属于可实现的
   非 oracle 对照；其三 seed 数据已归档到
   `docs/data/prediction_target_spectral_preconditioning_toy_v1/`。
5. 曲面 toy 中 LMMSE residual 的 velocity MSE 和 SWD 总体改善，但 MMD 变差，
   因此不能宣称它已普遍改善生成分布。

## 3. Learned/local posterior projector

主要入口：

- `experiments/posterior_response_projector.py`
- `experiments/audit_posterior_response_projector_toy.py`
- `experiments/run_prediction_target_learned_projector_toy.py`
- `experiments/run_prediction_target_local_projector_rollout_toy.py`
- `experiments/imagenet100_sit_posterior_response_head.py`
- `experiments/train_imagenet100_sit_posterior_response_head.py`
- `experiments/audit_imagenet100_sit_posterior_response_projector.py`

这些实验研究由 clean estimator Jacobian 近似 posterior response operator。现有
toy 与 SiT 结果没有给出足够稳定、足够便宜的质量收益。它们保留为机制代码，
但 local projector 不再作为当前主线。

`posterior_response` 采样控制已经接入静态 pair sampler，并有单测覆盖。该模式
需要额外 finite-difference forwards，正式使用时必须单独报告推理成本。

## 4. 多深度累计头与 progressive innovation

主要入口：

- `experiments/imagenet100_sit_joint_cumulative_heads.py`
- `experiments/train_imagenet100_sit_joint_cumulative_heads.py`
- `experiments/sample_imagenet100_sit_joint_cumulative_fid.py`
- `experiments/audit_imagenet100_sit_joint_cumulative_heads.py`
- `experiments/audit_imagenet100_sit_depth_convergence.py`
- `experiments/imagenet100_sit_progressive_innovation.py`
- `experiments/train_imagenet100_sit_progressive_innovation.py`
- `experiments/run_imagenet100_sit_progressive_innovation_2gpu.sh`

这些实现用于检验“多个中间头形成显式收敛序列”。现有扫描没有证明深度序列
本身是收益来源：early-window 测试存在宽阔的强度平台，把第二段改为 depth 10
后最优条件几乎会关闭第二段。因此代码归档，但不把多层弱头当作 solid 主线。

## 5. Spectral selector 与 continuation

主要入口：

- `experiments/imagenet100_sit_spectral_target_selector.py`
- `experiments/train_imagenet100_sit_spectral_target_continuation.py`
- `experiments/run_imagenet100_sit_internal_early_two_segment_gamma_sweep.py`

这些脚本保留了 operator-valued target、DCT/channel basis 和 early-window 对照的
完整实现。它们不能替代真实生成指标，也不能由局部 teacher loss 单独选择。

## 6. 测试状态

本次归档对应的 12 个测试文件与静态 pair 回归测试共同运行：

```text
65 passed, 18 warnings
```

warnings 均来自 PyTorch `torch.jit` deprecation，不是数值或逻辑失败。测试命令
需要从仓库根目录以 `PYTHONPATH=.` 运行。

## 7. 当前研究边界

- 已归档不等于已证明有效。
- oracle `P`、真实 tangent 或测试分布统计不能进入真实图像方法。
- 局部 projector、多层收敛头和 prediction-target selector 暂不继续扩参。
- 当前可继续严格审计的是“训练集低阶矩的解析输运 + 神经非高斯残差”；其理论、
  公平性和泄露边界单列于
  `docs/MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md`。

# Foresight Material-Derivative IG 数据说明

本目录保存 ImageNet-100 SiT depth4 Internal Guidance 的便携汇总，不包含模型权重、
生成样本 NPZ 或原始大文件。

- `fid1k_horizon_strength.csv`：短 horizon/strength 网格，`kappa=H*eta`。
- `fid1k_signed_control.csv`：固定 `H=0.125` 的正负号和过冲对照。
- `fid5k_paired_results.csv`：两个 sampling seed 的 depth4 IG 与 FMD-IG 正式配对结果。
- `compute_controls.csv`：求解器 NFE 对照和 prefix 等价运行。
- `fid1k_decomposition.csv`：future weak drift 的 gap-change/strong-change 精确拆分。
- `fid1k_lookahead_alpha.csv`：current/future weak reference 混合系数扫描。
- `fid1k_lookahead_horizon.csv`：`alpha=1` 的单参数 future-weak horizon 扫描。
- `fid5k_future_weak_reference.csv`：两套配对 5K bank 上的一参数 FWR 与原 FMD 对照。

完整解释见 `docs/IMAGENET100_SIT_FORESIGHT_MATERIAL_IG_RESULTS_ZH.md`。

# Self-guidance 理论与审计脚本

报告入口：[理论与审计目录](../../docs/classifier_guidance/README.md#理论与审计目录)。
脚本保留原模块名和路径；对应 JSON、CSV、PNG、PDF 与推导笔记作为紧凑研究证据一并纳入 Git。

| 脚本 | 输出目录（相对 `docs/research/`） | 内容 |
|---|---|---|
| [`gauge_reference_witness.py`](gauge_reference_witness.py)、[`deep_theory_audit.py`](deep_theory_audit.py) | [self_guidance_breakthrough_20260922](../../docs/research/self_guidance_breakthrough_20260922/) | gauge 反例、局部响应及深入理论检查 |
| [`ram_transfer_audit.py`](ram_transfer_audit.py)、[`local_riesz_audit.py`](local_riesz_audit.py)、[`tilt_flux_audit.py`](tilt_flux_audit.py) | [self_guidance_ram_20260923](../../docs/research/self_guidance_ram_20260923/) | RAM 转移、局部 Riesz 与 tilt/flux 检查 |
| [`rl_adjoint_audit.py`](rl_adjoint_audit.py) | [self_guidance_rl_20260923](../../docs/research/self_guidance_rl_20260923/) | RL 与完整伴随对照 |
| [`schedule_shape_snapshot.py`](schedule_shape_snapshot.py) | [self_guidance_schedule_shape_20260923](../../docs/research/self_guidance_schedule_shape_20260923/) | 读取本地训练材料并导出曲线快照 |
| [`signed_schedule_control_audit.py`](signed_schedule_control_audit.py)、[`input_shaping_audit.py`](input_shaping_audit.py)、[`second_order_control_audit.py`](second_order_control_audit.py) | 同上 | 有符号控制、输入整形与二阶控制检查 |
| [`latest_tail_backward_cpu_audit.py`](latest_tail_backward_cpu_audit.py) | 同上 | 最新系数范围下的小模型 CPU float64 离散反传检查 |

从仓库根目录使用模块名运行，例如：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  "$HOME/miniconda3/envs/myenv/bin/python" -m experiments.theory_self_guidance_20260922.deep_theory_audit
```

脚本通常直接写入表中的固定输出目录；复跑会更新已有证据文件。读取训练快照的脚本还需要
`$HOME/data/eqvae/projects/classifier_guidance/` 中的本地资产。小模型审计不加载完整 JiT/SiT
权重，不替代真实模型的 CUDA 梯度检查或 FID 质量评估。整理工作区时只保留既有证据，不重跑研究。

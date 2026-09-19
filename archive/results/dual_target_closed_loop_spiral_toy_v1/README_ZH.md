# 连续螺旋双目标闭环实验包

这个目录是 `dual_target_closed_loop_spiral_toy_v1` 正式实验的 Git 精简包。

## 包含内容

- `all_experiment_figures.pdf`：全部正式图像的无损多页 PDF。
- `figures_manifest.tsv`：每一页对应的原始相对路径和中文说明。
- `package_manifest.json`：打包数量、来源和明确排除项。
- `data/aggregate/`：跨 seed 汇总表、机制表、求解器收敛表和验证报告。
- `data/seed*/D*_H128/`：每个 setting 的配置、训练历史和逐条件指标表。

没有纳入 Git：

- `checkpoint.pt`；
- 原始生成 sample tensor；
- 可由代码重新生成的临时缓存。

完整 checkpoint 和原始实验目录保留在本机：

```text
/home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1
```

## 看图方法

先打开 `figures_manifest.tsv`。它的 `page` 列与 PDF 页码一一对应：

```text
page -> source_path -> description
```

例如 aggregate 图用于跨 seed 阅读；`seed*/D*_H128/endpoint_scatter.png` 用于检查单个 seed 的实际生成散点；`mechanism_summary.png` 同时显示该 setting 的终点与动力学诊断。

PDF 通过 `img2pdf` 直接嵌入 PNG，没有 JPEG 重编码或降采样。大图请放大查看坐标与图例。

## 先读哪些表

1. `data/aggregate/spiral_mechanism_seed_summary.csv`
   给出最精简的跨 seed 结论，包括 reference、同头最佳分支、oracle 以及 D1/D2/D4 gate。
2. `data/aggregate/cross_gate_seed_summary.csv`
   固定 D0 双头、只替换 gate 的公平终点对照。
3. `data/aggregate/cross_gate_teacher_seed_summary.csv`
   同头 gate 在 teacher states 上的局部场误差。
4. `data/aggregate/cross_gate_rollout_seed_summary.csv`
   同头 gate 在真实 rollout states 上的场误差与中间分布。
5. `data/aggregate/solver_step_convergence.csv`
   Heun 200/400 steps 收敛检查。
6. `data/aggregate/validation_report.json`
   6 个正式 setting 的完整性和有限值验证结果。

## 指标方向

- `swd_2d` / `swd_fullD`：越低越好。
- `arc_hist_tv`：越低表示螺旋弧长覆盖越接近真实分布。
- `ridge_width_ratio`：接近 `1` 较好，过小通常表示过度锐化，过大表示横向过宽。
- `ridge_distance_mean`：需和 ridge width、coverage 一起看，不能单独作为质量结论。
- `off_subspace_rms`：几何诊断，不是越小就必然生成越好。
- `bayes_velocity_mse`：局部速度场误差；它不等价于最终闭环分布质量。

## 结论报告与复现代码

中文技术报告：

```text
docs/DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md
```

正式运行：

```bash
bash experiments/run_dual_target_closed_loop_spiral_toy_formal.sh
```

重新汇总、校验和打包：

```bash
python experiments/summarize_dual_target_closed_loop_toy.py \
  --root /home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1

python experiments/summarize_dual_target_closed_loop_spiral_toy.py \
  --root /home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1

python experiments/validate_dual_target_closed_loop_spiral_results.py \
  --root /home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1

python experiments/package_dual_target_closed_loop_spiral_results.py \
  --source-root /home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1 \
  --destination-root dual_target_closed_loop_spiral_toy_v1
```

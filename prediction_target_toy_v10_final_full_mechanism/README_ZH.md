# Prediction-target toy v10 完整机制诊断归档

## 实验范围

本归档是 v10 的完整机制诊断，不是先前仅含 17 个条件的
`final_first_check`。

- 数据维数：`D=512`
- 曲率：`1.0`
- 网络宽度：`H=1024`
- loss space：`v`
- 独立训练 seed：`20260807, 20260808, 20260809`
- 每个 seed 的诊断条件：`187`
- 每个条件：`10,000` 个样本、`200` 个递归 ODE step
- 指标使用固定且匹配的样本子集、SWD 投影和 MMD 带宽

## 权重来源

本实验没有重新训练模型。v10 通过 `--reuse-v4-setting` 读取 v4 已归档的
三个 `D512 / curv1 / H1024 / loss_v` checkpoint。

这是有意的实验设计：v10 新增的是递归外推条件、几何分解和分布诊断，
不是新的训练目标。v10 与 v4 的共享采样条件已经做过数值回归；关闭合批时
逐元素一致，启用合批后仅有 fp32 GEMM 批形状导致的微小舍入差。

需要区分：v10 的“从头 fixed 训练”使用了新的随机种子派生方式，因此不会
逐参数复现 v4 checkpoint。本归档没有使用该分支，也不能把二者描述为
bitwise 相同的重新训练结果。

## 187 个条件

| 条件族 | 数量 |
|---|---:|
| `x / v / eps` baseline | 3 |
| raw `x-eps / x-v` 外推 | 36 |
| 显式 prediction-target path | 16 |
| 固定绝对 RMS 动作 | 8 |
| 相对 `x` RMS 动作 | 16 |
| shuffle / Gaussian-covariance / random controls | 18 |
| curve / ridge / ambient raw 几何分量 | 30 |
| curve / ridge / ambient 相对动作 | 24 |
| curve / ridge / ambient 绝对动作 | 12 |
| early / mid-high / mid-low / late 时间窗口 | 24 |
| **合计** | **187** |

## 文件说明

- `generation_metrics_v10_all_seeds.csv`：三个 seed 的全部 `561` 行原始端点指标。
- `generation_metrics_v10_seed_summary.csv`：按条件汇总的 mean / std / SEM。
- `aggregate_manifest_v10.json`：三 seed 聚合协议与来源文件。
- `seed*/generation_metrics_v10.csv`：各 seed 的 187 条件指标。
- `seed*/trajectory_mechanism_v10.csv`：递归采样期间的 gap/action 几何诊断。
- `seed*/setting_manifest_v10.json`：条件列表、模型来源和指标协议。
- `seed*/embedding_observability.json`：embedding 局部可观测性检查。
- `seed*/manifest_v10.json`：该 worker 的完整命令参数。
- `archive_manifest.json`：Git 归档文件的大小与 SHA-256。
- `../all_experiment_figures.pdf`：v4 与 v10 的重要图片合并成的单个无损 PDF。
- `../figures_manifest.tsv`：PDF 每一页的原始来源、seed 和中文说明。

## 未提交内容

以下内容仍保留在本机实验目录，不进入 Git：

- v4/v10 模型 checkpoint；
- 每个条件的高维 `.npz` 样本；
- `.partial.csv` 临时文件；
- seed `20260807` 的确定性 replay 副本。

replay 只用于验证端到端确定性；其正式指标文件必须与对应 worker 的
SHA-256 完全一致，但不能伪装成第四个独立 seed。

## 复现命令

```bash
bash experiments/run_v10_final_full_mechanism.sh
python experiments/package_prediction_target_v10_results.py
```

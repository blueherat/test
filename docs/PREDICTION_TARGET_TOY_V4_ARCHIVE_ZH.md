# Prediction Target Toy v4 实验归档

这份归档保存 2026-08-08 至 2026-08-09 运行的 prediction-target extrapolation toy v4 结果。目标是让主要数值、配置和视觉证据可以随 Git 审计，同时避免提交模型权重和可重新生成的大型中间产物。

## 归档内容

| 目录 | 实验作用 | 完成规模 |
| --- | --- | ---: |
| [`prediction_target_toy_v4_main`](../prediction_target_toy_v4_main) | `D=512`、四种曲率、五种 hidden width、三 seed 的主实验 | 60 settings |
| [`prediction_target_toy_v4_constant_norm`](../prediction_target_toy_v4_constant_norm) | constant-total-norm 缩放对照 | 9 settings |
| [`prediction_target_toy_v4_direct_loss`](../prediction_target_toy_v4_direct_loss) | direct loss 对照，用于区分 target geometry 与 v-loss conditioning | 9 settings |
| [`prediction_target_toy_v4_curved_screen`](../prediction_target_toy_v4_curved_screen) | 曲面、受限容量条件下的四 seed 筛查 | 4 settings |
| [`prediction_target_toy_v4_multiregime_screen`](../prediction_target_toy_v4_multiregime_screen) | `D=16` 中间区间和 `D=512` curved-capacity 的联合筛查 | 36 settings |
| [`prediction_target_toy_v4_reverse_from_v`](../prediction_target_toy_v4_reverse_from_v) | 当 v 强于 x 时，从 x 朝 v 以及越过 v 的反向外推对照 | 40 metric rows |

每组目录均保留：

- 全部 CSV 定量结果，包括 teacher-forced、generation、matched-randomness contrast 和 bootstrap 汇总；
- manifest、setting summary 等 JSON 配置与元数据；
- 自动生成的 `aggregate/final_report.txt`；
- 聚合 trade-off 图、轻量 phase diagram，以及少量能代表容量、缩放和损失对照的 generation scatter。

主入口是各目录下的：

- `aggregate/final_report.txt`
- `aggregate/settings.csv`
- `aggregate/aggregate_contrasts.csv`
- `aggregate/swd_manifold_tradeoff.png`

其中 reverse-from-v 使用 `aggregate/aggregate_metrics.csv` 和 `aggregate/final_report.txt`。

## 未提交内容

以下大型或可重建产物仍只保存在本机 `/home/zhoushunyu/data/eqvae/experiments/`：

- `models.pt`、`oracle_x.pt` 等 checkpoint；
- rollout/sample 的 `.npz` 数组；
- worker 日志和缓存；
- 大量同构的逐 setting scatter 图。

仓库 `.gitignore` 已覆盖 `*.pt`、`*.pth`、`*.ckpt`、`*.npy`、`*.npz`、`*.safetensors` 和 `*.log`，防止后续误提交。

## 对应代码

- [`run_prediction_target_extrapolation_toy_v4.py`](../experiments/run_prediction_target_extrapolation_toy_v4.py)：训练、teacher-forced 诊断和 rollout 评估主程序。
- [`summarize_prediction_target_toy_v4.py`](../experiments/summarize_prediction_target_toy_v4.py)：多 setting、多 seed 聚合。
- [`run_prediction_target_toy_v4_worker.sh`](../experiments/run_prediction_target_toy_v4_worker.sh)：主实验、缩放对照和 direct-loss 对照入口。
- [`run_prediction_target_toy_v4_recommended.sh`](../experiments/run_prediction_target_toy_v4_recommended.sh)：最初规划的三组推荐实验入口。
- [`run_prediction_target_toy_v4_curved_screen.sh`](../experiments/run_prediction_target_toy_v4_curved_screen.sh)：多区间初筛。
- [`run_prediction_target_toy_v4_eligibility_screen.sh`](../experiments/run_prediction_target_toy_v4_eligibility_screen.sh)：capacity eligibility 补点。
- [`evaluate_prediction_target_reverse_from_v.py`](../experiments/evaluate_prediction_target_reverse_from_v.py)：反向外推评估。

归档仅保存实验事实，不把某个单点最优解释成稳定方法收益。正式判断应优先读取多 seed aggregate、matched randomness、SWD bootstrap、MMD 和 manifold RMS，而不是只看单张散点图。

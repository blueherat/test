import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/rae_teacher_rollout_gap.ipynb"


def _id(prefix: str, source: str) -> str:
    return f"{prefix}-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]}"


def markdown(source: str):
    source = dedent(source).strip()
    cell = nbf.v4.new_markdown_cell(source)
    cell["id"] = _id("md", source)
    return cell


def code(source: str):
    source = dedent(source).strip()
    cell = nbf.v4.new_code_cell(source)
    cell["id"] = _id("code", source)
    return cell


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        markdown(
            r"""
            # RAE Teacher-Forced 到 Rollout 的断层

            ## TL;DR

            这个 notebook 研究 tiny 频谱实验里最反常的事实：`gamma=0.5` 在未见验证图像的
            velocity/latent 代理上稳定改善，却没有转化为 5k FID/KID 的有效改善。

            诊断严格分成两侧：

            - **teacher-forced：** 在已知线性插值上用 $\hat z_0=z_t-t\hat v$，允许逐样本比较；
            - **rollout：** ODE 样本与任意验证图像没有配对，只比较边缘分布、频带能量和轨迹一致性。

            当前证据指向的不是普通过拟合，也不是简单高频幅度收缩。加权模型在 teacher 轨道上的
            高频能量、方向余弦、回归斜率都更接近真实 target，但自生成轨道从早期开始出现更强的
            高频方差亏损。最合理的机制是：**单步条件 MSE/方向准确性与多步 transport 的边缘方差、
            体积变化和感知质量之间存在断层。**

            速度侧也出现反直觉结果：尽管实际 state update 集中在低噪声后段，把采样点搬到后段反而
            明显劣于保留官方 shifted time warp。25-step 官方网格在 64 样本 endpoint proxy 上看似温和：
            约 `2.02x` 加速、latent 相对误差约 `5.03%`；但同采样口径的正式 5k FID 仍恶化 `2.50%`，
            KID 恶化 `3.94%`。这正是本 notebook 要强调的第三类断层：**endpoint 距离也不等于生成
            分布质量。**
            """
        ),
        markdown("## Context & Methods\n\n### 1. Load and verify artifacts"),
        code(
            """
            from pathlib import Path
            import json

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
            paths = {
                "teacher": RESULTS / "teacher_rollout_gap_teacher.csv",
                "teacher_bands": RESULTS / "teacher_rollout_gap_teacher_bands.csv",
                "rollout": RESULTS / "teacher_rollout_gap_rollout.csv",
                "bands": RESULTS / "teacher_rollout_gap_bands.csv",
                "steps": RESULTS / "teacher_rollout_gap_steps.csv",
                "generation": RESULTS / "generation_metrics.csv",
                "generation25": RESULTS / "generation_metrics_25steps_seed3407_baseline_from_s5000.csv",
                "switch": RESULTS / "vector_field_switch_metrics.csv",
                "switch_bands": RESULTS / "vector_field_switch_bands.csv",
            }
            missing = [str(path) for path in paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError("缺少正式诊断结果:\\n" + "\\n".join(missing))

            teacher = pd.read_csv(paths["teacher"])
            teacher_bands = pd.read_csv(paths["teacher_bands"])
            rollout = pd.read_csv(paths["rollout"])
            bands = pd.read_csv(paths["bands"])
            steps = pd.read_csv(paths["steps"])
            generation = pd.read_csv(paths["generation"])
            generation25 = pd.read_csv(paths["generation25"])
            switch = pd.read_csv(paths["switch"])
            switch_bands = pd.read_csv(paths["switch_bands"])

            plt.rcParams.update({
                "figure.dpi": 130,
                "font.size": 10,
                "axes.titlesize": 12,
                "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                "axes.unicode_minus": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
            COLORS = {"baseline": "#111827", "partial": "#2563EB", "bad": "#DC2626", "good": "#059669"}

            for name, frame in {"teacher": teacher, "rollout": rollout, "bands": bands}.items():
                assert sorted(frame.seed.unique()) == [3407, 4211, 5821], name
                assert set(frame.treatment) == {"baseline", "partial"}, name
            metadata = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(RESULTS.glob("seed*_*_from_s5000/gap_study/metadata.json"))
            ]
            assert len(metadata) == 6
            assert {item["count"] for item in metadata} == {64}
            assert {item["perceptual_count"] for item in metadata} == {12}
            assert len({tuple(item["validation_indices"]) for item in metadata}) == 1
            display(pd.DataFrame({
                "检查": ["paired seeds", "validation latent", "perceptual latent", "数值", "rollout 比较"],
                "值": ["3", "64 / branch", "12 / branch", "fp32, TF32 off", "仅分布；不伪造样本配对"],
            }))
            """
        ),
        markdown(
            r"""
            ### Key Assumptions

            - 两个 treatment 从同一个 step-5000 full-state checkpoint 出发，每个 seed 内数据流严格配对。
            - 这里加载 step-10000 EMA；采样时间网格和官方 50-step shifted Euler 完全一致，终点约为 $t=0.0069$。
            - torchvision Inception-v3 pre-logit cosine 只是配对感知代理，不等同于 FID 的 Inception 网络。
            - 训练 seed 是独立重复单位；64 张图用于降低同一 seed 内测量噪声，不被当成 64 个训练重复。
            - rollout 的 same-noise coupling 只用于 vector-field secant 敏感度，不能解释成生成样本“对应”验证图像。
            - summary SWD 只作用于随机投影的 token mean 与 8 个 log-band energies，是低维分布探针，
              不是完整 latent Wasserstein 距离或 FID。
            """
        ),
        markdown("## Results\n\n### 2. Teacher-forced 改进在哪里传递、在哪里翻转"),
        code(
            """
            def paired_gain(frame, metrics):
                selected = frame[frame.metric.isin(metrics)]
                paired = selected.pivot_table(
                    index=["seed", "time_index", "time", "metric"],
                    columns="treatment",
                    values="value",
                ).reset_index()
                paired["gain_pct"] = 100 * (paired.baseline - paired.partial) / paired.baseline
                return paired

            teacher_metrics = [
                "latent_mse", "pixel_mse", "inception_cosine_distance",
                "decoder_hidden_input_cosine_distance",
                "decoder_hidden_middle_cosine_distance",
                "decoder_hidden_final_cosine_distance",
            ]
            teacher_gain = paired_gain(teacher, teacher_metrics)
            summary = teacher_gain.groupby(["time", "metric"], as_index=False).agg(
                gain=("gain_pct", "mean"),
                low=("gain_pct", "min"),
                high=("gain_pct", "max"),
                positive_seeds=("gain_pct", lambda values: int((values > 0).sum())),
            )
            labels = {
                "latent_mse": "latent MSE",
                "pixel_mse": "decoder pixel MSE",
                "inception_cosine_distance": "Inception distance",
                "decoder_hidden_input_cosine_distance": "decoder input",
                "decoder_hidden_middle_cosine_distance": "decoder middle",
                "decoder_hidden_final_cosine_distance": "decoder final",
            }
            figure, axis = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
            for metric, group in summary.groupby("metric"):
                group = group.sort_values("time", ascending=False)
                axis.plot(group.time, group.gain, marker="o", lw=2, label=labels[metric])
            axis.axhline(0, color="#6B7280", lw=1)
            axis.invert_xaxis()
            axis.set(xlabel="ODE time（采样方向：1 → 0）", ylabel="partial 相对 baseline 改善（%）", title="同一个 teacher 改进在 latent / pixel / semantic 空间发生符号翻转")
            axis.grid(alpha=.2)
            axis.legend(frameon=False, ncol=3)
            display(summary.pivot(index="time", columns="metric", values="gain").sort_index(ascending=False).round(2))
            """
        ),
        markdown(
            """
            最反常的局部例子是 `t≈0.69`：latent MSE 与 Inception distance 分别改善约 `3.8%` 和 `8.9%`，
            但 decoder pixel MSE 反而变差约 `3.2%`。这说明 decoder 的像素欧氏几何与语义/latent 几何并不一致，
            不能再把 pixel proxy 当生成质量的单调替代量。
            """
        ),
        markdown("### 3. 高频 MSE 变好是方向更准，还是幅度收缩？"),
        code(
            """
            calibration_mean = teacher_bands.groupby(
                ["treatment", "time", "metric", "band"], as_index=False
            ).value.mean()
            calibration_delta = calibration_mean.pivot_table(
                index=["time", "metric", "band"], columns="treatment", values="value"
            ).reset_index()
            calibration_delta["delta"] = calibration_delta.partial - calibration_delta.baseline

            metrics = [
                "prediction_target_cosine",
                "prediction_target_slope",
                "prediction_energy_log_ratio_to_target",
            ]
            titles = ["方向余弦增量", "回归斜率增量", "log 能量比增量（向 0 更好）"]
            figure, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
            for axis, metric, title in zip(axes, metrics, titles):
                matrix = calibration_delta.query("metric == @metric").pivot(index="time", columns="band", values="delta").sort_index(ascending=False)
                limit = float(np.abs(matrix.to_numpy()).max())
                image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
                axis.set_xticks(range(8), range(8))
                axis.set_yticks(range(len(matrix)), [f"{value:.3f}" for value in matrix.index])
                axis.set(xlabel="DCT band（低 → 高）", ylabel="time", title=title)
                figure.colorbar(image, ax=axis, shrink=.78)
            """
        ),
        markdown(
            """
            结果排除了“靠把高频预测缩到 0 来降低 MSE”：除最高噪声处的 band 0 外，partial 在 3/3 seed 上
            同时提高 target cosine、把 prediction/target 能量比推近 1、把回归斜率推近 1。teacher 监督是真的改善了，
            因而失败发生在**多步 transport 与边缘分布**，不是单步目标的伪改善。

            还有一个重要理论边界：这里的频带权重 $W(t)$ 固定、正定且不依赖样本。若模型函数空间无限，
            加权和未加权平方损失的 population minimizer 都是 $E[u\mid z_t,t]$；因此它不创造新的理想
            vector field。当前变化来自有限模型容量、共享参数和优化路径上的资源重分配，不能称为保持原
            目标的纯 preconditioning。
            """
        ),
        markdown("### 4. 一进入 rollout，改善是否还存在？"),
        code(
            """
            rollout_metrics = [
                "state_summary_swd",
                "rollout_clean_summary_swd",
                "state_energy_log_ratio_absolute_mean",
                "clean_energy_log_ratio_absolute_mean",
                "predicted_clean_to_endpoint_inception_cosine_distance",
            ]
            rollout_gain = paired_gain(rollout, rollout_metrics)
            rollout_summary = rollout_gain.groupby(["time", "metric"], as_index=False).agg(
                gain=("gain_pct", "mean"),
                positive_seeds=("gain_pct", lambda values: int((values > 0).sum())),
            )
            rollout_labels = {
                "state_summary_swd": "state marginal SWD",
                "rollout_clean_summary_swd": "rollout clean-estimate SWD",
                "state_energy_log_ratio_absolute_mean": "state band-energy gap",
                "clean_energy_log_ratio_absolute_mean": "clean band-energy gap",
                "predicted_clean_to_endpoint_inception_cosine_distance": "clean→endpoint Inception",
            }
            figure, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
            for axis, metric in zip(axes.flat, rollout_metrics):
                if metric == "state_energy_log_ratio_absolute_mean":
                    values = rollout.query("metric == @metric").groupby(
                        ["treatment", "time"], as_index=False
                    ).value.mean().pivot(index="time", columns="treatment", values="value").reset_index()
                    values["plot_value"] = values.partial - values.baseline
                    ylabel = "partial - baseline（绝对 gap）"
                else:
                    values = rollout_summary.query("metric == @metric").copy()
                    values["plot_value"] = values.gain
                    ylabel = "相对改善（%）"
                values = values.sort_values("time", ascending=False)
                axis.plot(values.time, values.plot_value, marker="o", lw=2.2, color=COLORS["bad"])
                axis.axhline(0, color="#6B7280", lw=1)
                axis.invert_xaxis()
                axis.set(xlabel="time", ylabel=ylabel, title=rollout_labels[metric])
                axis.grid(alpha=.2)
            axes.flat[-1].axis("off")
            figure.suptitle("teacher 改进在 rollout 中立即反转（0 以下为 partial 更差）", fontsize=15)
            display(rollout_summary.pivot(index="time", columns="metric", values="gain").sort_index(ascending=False).round(2))
            """
        ),
        markdown(
            """
            `t≈0.95` 的 state SWD 相对恶化约 `70%`，但其 baseline 绝对值只有约 `0.155`；这个百分比
            受小分母放大。可信结论是 3/3 seed 的方向一致，以及中后段约 `15%--21%` 的持续差距，
            不是把最早时刻的 `70%` 当成效应量。
            """
        ),
        markdown("### 5. 频带边缘能量：断层具体落在哪些 band"),
        code(
            """
            state_energy = bands.query("metric == 'state_energy_log_ratio'").groupby(
                ["treatment", "time", "band"], as_index=False
            ).value.mean()
            matrices = {
                treatment: state_energy.query("treatment == @treatment").pivot(index="time", columns="band", values="value").sort_index(ascending=False)
                for treatment in ["baseline", "partial"]
            }
            matrices["partial - baseline"] = matrices["partial"] - matrices["baseline"]

            figure, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
            settings = [
                ("baseline", -1.1, .1), ("partial", -1.1, .1), ("partial - baseline", -.2, .2)
            ]
            for axis, (name, lower, upper) in zip(axes, settings):
                matrix = matrices[name]
                image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=lower, vmax=upper)
                axis.set_xticks(range(8), range(8))
                axis.set_yticks(range(len(matrix)), [f"{value:.3f}" for value in matrix.index])
                axis.set(xlabel="DCT band（低 → 高）", ylabel="time", title=name)
                figure.colorbar(image, ax=axis, shrink=.78)
            print("log energy ratio < 0 表示 rollout state 相对真实 interpolation marginal 方差不足。")
            """
        ),
        markdown(
            """
            partial 从 `t≈0.95` 就比 baseline 更偏离真实 marginal，之后高频 deficit 逐步累积；终点 band 7 的
            log-energy ratio 约从 `-0.30` 降到 `-0.49`。同时 band 0 在后段略接近真实能量。
            这与训练目标“牺牲低频、改善高频 MSE”的方向并不矛盾：MSE 更准并不保证 deterministic transport
            保持正确的 batch covariance / volume。
            """
        ),
        markdown("### 6. 时间区间 vector-field swap：断层到底从哪里进入"),
        code(
            """
            switch_order = [
                "baseline",
                "partial_high_ge_085",
                "partial_mid_030_085",
                "partial_low_lt_030",
                "baseline_high_partial_below_085",
                "partial",
            ]
            switch_summary = switch.groupby(["schedule", "metric"], as_index=False).value.mean()
            switch_table = switch_summary.pivot(index="schedule", columns="metric", values="value").reindex(switch_order)
            display(switch_table.round(5))

            switch_energy = switch_bands.groupby(["schedule", "band"], as_index=False).value.mean()
            energy_matrix = switch_energy.pivot(index="schedule", columns="band", values="value").reindex(switch_order)
            figure, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
            swd = switch_table["summary_swd_to_validation"]
            axes[0].barh(range(len(swd)), swd, color=["#111827"] + ["#DC2626"] * (len(swd) - 1))
            axes[0].set_yticks(range(len(swd)), swd.index)
            axes[0].invert_yaxis()
            axes[0].set(xlabel="summary SWD to validation（低更好）", title="每个 partial 时间窗都伤害 rollout marginal")
            axes[0].grid(axis="x", alpha=.2)
            image = axes[1].imshow(energy_matrix, aspect="auto", cmap="RdBu_r", vmin=-1.1, vmax=.1)
            axes[1].set_xticks(range(8), range(8))
            axes[1].set_yticks(range(len(energy_matrix)), energy_matrix.index)
            axes[1].set(xlabel="DCT band（低 → 高）", title="endpoint log-energy ratio to validation")
            figure.colorbar(image, ax=axes[1], shrink=.8)
            """
        ),
        markdown(
            r"""
            这个实验推翻了“主要由 `t>=0.85` 的 band-0 牺牲触发”的简单因果故事。只在高噪声、中段、
            低噪声使用 partial，summary SWD 分别恶化约 `5.5% / 11.7% / 2.5%`，且全部是 3/3 seed
            同方向；中段累计贡献最大，低噪声虽然只有最后两次求值，单步杠杆却最大。更反常的是，三个
            区间都会让 band 0 略接近真实能量、同时让 band 1--7 更欠方差。这说明损害来自跨频耦合的
            transport contraction，而不是某个错误频带幅度被直接搬运到终点。现有 partial 没有可直接
            拼接成收益的时间窗。

            数学上，ODE 边缘协方差满足
            $\frac{d\Sigma}{dt}=\operatorname{Cov}(z,v)+\operatorname{Cov}(v,z)$，沿轨道的体积变化满足
            $\frac{d}{dt}\log p(z_t)=-\nabla\!\cdot v(z_t,t)$。逐样本 velocity MSE 并不直接控制这两个量；
            time-swap 的一致负结果正是这个断层的功能性证据。
            """
        ),
        markdown("### 7. Vector-field 敏感度与 50 步预算分布"),
        code(
            """
            secant = rollout[rollout.metric.isin([
                "coupled_state_rms", "coupled_velocity_rms", "velocity_state_secant_gain"
            ])].groupby(["treatment", "time", "metric"], as_index=False).value.mean()
            figure, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
            for treatment, group in secant.query("metric == 'velocity_state_secant_gain'").groupby("treatment"):
                group = group.sort_values("time", ascending=False)
                axes[0].plot(group.time, group.value, marker="o", lw=2.2, color=COLORS[treatment], label=treatment)
            axes[0].invert_xaxis()
            axes[0].set(xlabel="time", ylabel="||Δv|| / ||Δz||", title="same-noise secant sensitivity")
            axes[0].grid(alpha=.2)
            axes[0].legend(frameon=False)

            step_mean = steps.groupby(["treatment", "time_index", "time", "metric"], as_index=False).value.mean()
            for treatment, group in step_mean.query("metric == 'euler_update_relative_rms'").groupby("treatment"):
                group = group.sort_values("time", ascending=False)
                axes[1].plot(group.time, group.value, lw=2.2, color=COLORS[treatment], label=treatment)
            axes[1].invert_xaxis()
            axes[1].set(xlabel="time", ylabel="relative Euler update RMS", title="实际状态更新集中在低噪声后段")
            axes[1].grid(alpha=.2)
            axes[1].legend(frameon=False)

            curvature = step_mean.query("metric == 'clean_estimate_relative_change'")
            display(curvature.groupby("treatment").value.agg(["mean", "max"]).round(5))
            """
        ),
        markdown("### 8. 减少采样步数的数值代理"),
        code(
            """
            schedule_metric_paths = sorted(RESULTS.glob("seed*_baseline_from_s5000/step_schedule_probe/metrics.csv"))
            schedule_dist_paths = sorted(RESULTS.glob("seed*_baseline_from_s5000/step_schedule_probe/distribution.csv"))
            if len(schedule_metric_paths) != 3:
                print("三 seed schedule probe 尚未全部完成。")
                schedule_metrics = pd.DataFrame()
            else:
                schedule_metrics = pd.concat([pd.read_csv(path) for path in schedule_metric_paths], ignore_index=True)
                key_metrics = ["endpoint_latent_relative_rms", "endpoint_inception_cosine_distance", "endpoint_pixel_mse"]
                schedule_summary = schedule_metrics[schedule_metrics.metric.isin(key_metrics)].groupby(
                    ["schedule", "model_evaluations", "theoretical_speedup", "metric"], as_index=False
                ).value.mean()
                short_names = {
                    "official_50": "official 50",
                    "official_numsteps_25": "num_steps 25",
                    "official_numsteps_16": "num_steps 16",
                    "shifted_subsample_25": "shifted 25",
                    "shifted_subsample_16": "shifted 16",
                    "uniform_actual_t_20": "uniform-t 20",
                    "hybrid_early5_uniformlate_20": "hybrid 20",
                    "hybrid_early4_uniformlate_16": "hybrid 16",
                }
                family_colors = {
                    "official_50": "#111827",
                    "official_numsteps_25": "#0F766E",
                    "official_numsteps_16": "#5EEAD4",
                    "shifted_subsample_25": "#2563EB",
                    "shifted_subsample_16": "#60A5FA",
                    "uniform_actual_t_20": "#DC2626",
                    "hybrid_early5_uniformlate_20": "#F59E0B",
                    "hybrid_early4_uniformlate_16": "#FBBF24",
                }
                figure, axes = plt.subplots(1, 3, figsize=(20, 7.2), constrained_layout=True)
                for axis, metric, title in zip(
                    axes, key_metrics, ["latent endpoint error", "Inception endpoint distance", "pixel endpoint MSE"]
                ):
                    group = schedule_summary.query("metric == @metric").sort_values(
                        ["model_evaluations", "schedule"], ascending=[False, True]
                    )
                    positions = np.arange(len(group))
                    axis.barh(
                        positions,
                        group.value,
                        color=[family_colors[name] for name in group.schedule],
                    )
                    axis.set_yticks(positions, [short_names[name] for name in group.schedule])
                    axis.invert_yaxis()
                    axis.set(xlabel=metric, title=title)
                    axis.grid(axis="x", alpha=.2)
                figure.suptitle("减少 Euler 求值次数的 endpoint 数值误差（3-seed mean）", fontsize=15)
                display(schedule_summary.pivot(index=["schedule", "model_evaluations", "theoretical_speedup"], columns="metric", values="value").round(6))
            """
        ),
        markdown("### 9. 与真实 5k generation 结论对齐"),
        code(
            """
            kid = "kernel_inception_distance_mean"
            fid = "frechet_inception_distance"
            paired_generation = generation.pivot(index="seed", columns="treatment", values=[kid, fid])
            result_rows = []
            for metric in [kid, fid]:
                baseline = paired_generation[(metric, "baseline")]
                partial = paired_generation[(metric, "partial")]
                gain = 100 * (baseline - partial) / baseline
                result_rows.append({
                    "metric": metric,
                    "mean_gain_pct": gain.mean(),
                    "positive_seeds": int((gain > 0).sum()),
                    "min_gain_pct": gain.min(),
                    "max_gain_pct": gain.max(),
                })
            display(pd.DataFrame(result_rows).round(3))

            baseline50 = generation.query(
                "branch == 'seed3407_baseline_from_s5000'"
            ).iloc[0]
            baseline25 = generation25.query(
                "branch == 'seed3407_baseline_from_s5000'"
            ).iloc[0]
            formal_metrics = [
                "inception_score_mean",
                "frechet_inception_distance",
                "kernel_inception_distance_mean",
            ]
            formal_rows = []
            for metric in formal_metrics:
                value50 = float(baseline50[metric])
                value25 = float(baseline25[metric])
                formal_rows.append({
                    "metric": metric,
                    "50 steps": value50,
                    "25 steps": value25,
                    "25 - 50": value25 - value50,
                    "relative change (%)": 100 * (value25 / value50 - 1),
                })
            formal_speed = pd.DataFrame(formal_rows)
            display(formal_speed.round(5))

            from PIL import Image
            paired_pixel_mse = []
            folder50 = Path(baseline50.sample_folder)
            folder25 = Path(baseline25.sample_folder)
            for index in range(64):
                image50 = np.asarray(Image.open(folder50 / f"{index:06d}.png"), dtype=np.float32) / 255
                image25 = np.asarray(Image.open(folder25 / f"{index:06d}.png"), dtype=np.float32) / 255
                paired_pixel_mse.append(np.square(image50 - image25).mean())
            print(f"前 64 个严格配对样本 pixel MSE: {np.mean(paired_pixel_mse):.6f}")

            figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
            for axis, row in zip(axes, formal_speed.itertuples(index=False)):
                axis.bar(["50 steps", "25 steps"], [row[1], row[2]], color=["#111827", "#0F766E"])
                axis.set_title(row.metric.replace("_", " "))
                axis.grid(axis="y", alpha=.2)
            figure.suptitle("正式 5k：2x 数值加速并非无损", fontsize=15)
            """
        ),
        markdown(
            """
            这里必须把两条结论分开：25 步是一个可部署的 **quality-speed trade-off**，不是无损加速。
            单 seed、5k 样本只足以否定“proxy 小，所以真实质量基本不变”的乐观推断；若要精确估计
            Pareto 曲线，仍需更多 sampling seeds 或 50k 指标。它不影响前面的机制结论，但再次证明
            paired endpoint error 不能替代生成分布指标。
            """
        ),
        markdown(
            r"""
            ## Takeaways

            ### 10. 断层机制与可利用方向

            **已验证：**

            1. partial 不是只在训练集变好；在固定 ImageNet validation interpolation 上，多数中高频 velocity
               MSE、方向余弦、能量比和 slope 都稳定改善。
            2. 改进并不单调穿过 decoder：pixel、decoder hidden 和 Inception 会随时间发生符号翻转。
            3. rollout marginal 从很早的 `t≈0.95` 就变差，最终表现为除 band 0 外更强的频带方差亏损；
               这与 5k FID 在 2/3 seed 变差一致。
            4. partial 早期 vector field 更平滑，但中后期 secant sensitivity 反而提高，呈现“早期过度收缩、
               后期补偿更硬”的动力学形态。
            5. hard time-swap 显示高/中/低三个 partial 区间都使 rollout marginal 变差；中段累计损害最大，
               末段单步最敏感。因而不能靠直接拼接现有两个 checkpoint 获益。

            **质量方向，优先级从高到低：**

            - 不再继续调固定 `gamma`，也不把“只在中段加现有 weighted loss”当主候选；time-swap 已显示
              当前 partial vector field 在每个区间都伤害边缘分布。
            - 下一候选应直接约束短 rollout 的 marginal，重点覆盖中段和最后两步：按时间/频带匹配
              `z_{t-Δ}` 或 `z0_hat` 的 batch covariance / log-energy，而不是只重加权逐样本 velocity MSE。
            - 加一个 1–2 步 differentiable rollout consistency loss，约束 teacher state 与 rollout state 的
              频带统计差；这正面覆盖当前 exposure gap。
            - 一个无需训练的先行试验是 train-only spectral variance calibration：用独立生成校准集估计
              固定频带 scale，再用于 held-out 5k latent。它可能补回细节，也可能放大 decoder artifact，
              所以必须用 KID/FID 而不是像素图判断。
            - decoder pixel loss 不适合作主目标。若加 perceptual 项，应使用冻结的 latent/decoder hidden 或
              DINO/Inception feature，并同时保留 covariance guardrail，避免“感知点估计更好但分布更窄”。
            - 若继续 FxLMS 式加速，优先研究不改变原始 MSE stationary points 的参数空间 optimizer
              preconditioner；当前 output-band weighted loss 在有限容量模型中会改变折中解，已经被证实会
              牺牲 transport covariance。

            **速度方向：**

            - 50-step 网格的实际 state update 虽集中于低噪声最后约 10 步，但高噪声 `t>0.97` 的单位时间
              velocity curvature 最高。把点搬到后段的 `uniform_actual_t/hybrid` 明显变差，说明“更新小”
              不等于“可跳步”。
            - 最稳的候选仍是保持官方 shifted warp 的 `num_steps=25`：实测约 `2.02x` 加速，3-seed
              endpoint latent relative RMS `5.03%`、Inception cosine distance `0.0296`。但正式 5k
              检查显示它不是无损近似；只能作为 quality-speed Pareto 点，不能包装成质量提升。

            **尚未证明：** 当前结果不能证明 covariance loss 或 optimizer preconditioner 一定提升 FID；25 步正式结果也只有一个
            checkpoint 和一个 sampling seed，足以否定“无损”，但不足以精确估计其平均质量代价。
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)


if __name__ == "__main__":
    build_notebook()

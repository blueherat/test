import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/rae_spectral_tiny_screen.ipynb"


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
            # RAE 频谱方向重加权：Tiny 生死实验

            ## TL;DR

            这个 notebook 只回答一个预注册问题：从同一个 `DiTDH-S step-5000` full-state checkpoint
            出发，在相同 5k update 预算、相同数据/增强/时间/噪声/标签下，`gamma=0.5` 的固定 DCT
            direction-only loss 是否比 `gamma=0` 改善未见数据和 5k 生成分布。

            它**不把方法称为优化加速或纯 preconditioner**。每个时间点的系数加权平均权重虽然严格为 1，
            参数梯度范数仍可能变化；因此这里检验的是有限容量下的目标/预算重分配。

            notebook 会自动识别正在训练、只完成 validation 或已完成 generation 的状态，不会把不完整结果写成结论。

            **最终筛选结论：FAIL。** `gamma=0.5` 在 3/3 seed 上都让 KID 略降，但相对改善仅
            `0.65%–1.89%`，0/3 达到预注册的 `5%` 实际阈值；FID-5k 在 2/3 seed 上变差。
            因此保留“真实 RAE 存在频谱目标错位”的机制结论，停止把这个固定频带重加权当作候选生成方法。
            """
        ),
        markdown("## Context & Methods\n\n### 1. Load protocol and artifacts"),
        code(
            """
            from pathlib import Path
            import json
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from PIL import Image

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
            AUDIT = Path.home() / "data/eqvae/experiments/rae_spectral_gradient_audit_full"
            protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))

            plt.rcParams.update({
                "figure.dpi": 130,
                "font.size": 10,
                "axes.titlesize": 12,
                "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                "axes.unicode_minus": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
            COLORS = {"baseline": "#111827", "partial": "#2563EB"}
            display(pd.DataFrame({
                "字段": ["起点", "主终点", "paired seeds", "比较", "继续门槛"],
                "预注册值": [
                    f"step {protocol['start']['checkpoint_step']}",
                    f"step {protocol['primary_endpoint_update']}",
                    str(protocol["paired_seeds"]),
                    "gamma=0 vs gamma=0.5",
                    protocol["continue_rule"],
                ],
            }))
            """
        ),
        markdown(
            r"""
            ### Key Assumptions

            - 8 个 band 是固定的等基数径向 DCT band；二阶矩只从 2,048 个 ImageNet train latent 估计。
            - 每个样本、每个 $t$ 上按 band 系数数目归一化：$sum_b d_bw_b/\sum_b d_b=1$。
            - 两分支使用同一 model/EMA/optimizer/scheduler checkpoint；checkpoint 不含原 dataloader 游标，
              所以这是从共同起点开始的**新配对随机流**，不是原轨迹的逐 batch 无缝续跑。
            - fp32、TF32 关闭；flash/memory-efficient attention 关闭，使用 deterministic math attention。
            - 5k KID/FID 是 screening proxy；训练 seed 才是统计单位，生成图片数不是独立训练重复。
            - sampler 为整除全局 batch 会产生 5,008 个 PNG；指标严格读取 `.npz` 中预注册的前 5,000 张。
            - 这里的 FID 是 `torch-fidelity` 对 10k virtual ImageNet reference 的 5k proxy，不能与 ADM 50k gFID 横比。
            """
        ),
        markdown("## Data\n\n### 2. Verify the real-RAE mechanism premise"),
        code(
            """
            residual = pd.read_csv(AUDIT / "residual_table.csv")
            decoder = pd.read_csv(AUDIT / "decoder_sensitivity.csv")
            residual_band = residual.groupby("spatial_band", as_index=False).agg(
                residual_mse=("linear_residual_mse", "mean"),
                predictability=("rho_lower_clipped", "mean"),
            ).merge(decoder[["spatial_band", "pixel_sensitivity_mean"]], on="spatial_band")

            normalized = residual_band.copy()
            for column in ["residual_mse", "predictability", "pixel_sensitivity_mean"]:
                normalized[column] /= normalized[column].mean()

            figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
            axis.plot(normalized.spatial_band, normalized.residual_mse, "o-", lw=2.2, label="FM residual energy")
            axis.plot(normalized.spatial_band, normalized.predictability, "s-", lw=2.2, label="teacher predictability")
            axis.plot(normalized.spatial_band, normalized.pixel_sensitivity_mean, "^-", lw=2.2, label="decoder sensitivity")
            axis.axhline(1, color="#9CA3AF", lw=0.8)
            axis.set(xlabel="DCT band（低频 → 高频）", ylabel="相对各自均值", title="真实 RAE 的 predictability–observability mismatch")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, ncol=3)
            display(residual_band.round(4))
            """
        ),
        markdown("### 3. Verify pairing and fixed weights"),
        code(
            """
            smoke = {
                treatment: RESULTS / f"seed3407_{treatment}_s5000_to_5002_smoke_v3"
                for treatment in ["baseline", "partial"]
            }
            fingerprints = {
                treatment: json.loads((path / "pair_fingerprint.json").read_text())
                for treatment, path in smoke.items()
            }
            hash_fields = [key for key in fingerprints["baseline"] if key.endswith("sha256")]
            pairing = pd.DataFrame({
                "field": hash_fields + ["raw_mse", "band_mse"],
                "exact_match": [
                    fingerprints["baseline"][key] == fingerprints["partial"][key]
                    for key in hash_fields + ["raw_mse", "band_mse"]
                ],
            })
            assert pairing.exact_match.all()
            display(pairing)

            manifests = {
                treatment: json.loads((path / "manifest.json").read_text())
                for treatment, path in smoke.items()
            }
            times = manifests["partial"]["weight_times"]
            weights = np.asarray(manifests["partial"]["weights"])
            figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
            for time, row in zip(times, weights):
                axis.plot(np.arange(8), row, marker="o", lw=2, label=f"t={time:.2f}")
            axis.axhline(1, color="#111827", lw=1)
            axis.axhspan(0.2, 2.0, color="#E5E7EB", alpha=0.25)
            axis.set(xlabel="DCT band（低频 → 高频）", ylabel="loss weight", title="冻结的 direction-only 权重")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, ncol=4)
            print("通过：图像、标签、t、噪声、target、更新前 prediction 和 raw MSE 均严格配对。")
            """
        ),
        markdown("## Results\n\n### 4. Training status and paired curves"),
        code(
            """
            def load_training_runs():
                rows = []
                for branch in sorted(RESULTS.glob("seed*_*_from_s5000")):
                    manifest_path = branch / "manifest.json"
                    metrics_path = branch / "metrics.jsonl"
                    if not manifest_path.exists() or not metrics_path.exists():
                        continue
                    manifest = json.loads(manifest_path.read_text())
                    frame = pd.read_json(metrics_path, lines=True)
                    frame["seed"] = int(manifest["global_seed"])
                    frame["treatment"] = "baseline" if float(manifest["gamma"]) == 0 else "partial"
                    frame["complete"] = (branch / "checkpoints/step-0010000.pt").exists()
                    rows.append(frame)
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

            training = load_training_runs()
            if training.empty:
                print("训练尚未产生 structured metrics。")
            else:
                status = training.groupby(["seed", "treatment"], as_index=False).agg(
                    last_update=("branch_update", "max"), complete=("complete", "max")
                )
                display(status)
                figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
                for (seed, treatment), group in training.groupby(["seed", "treatment"]):
                    group = group.sort_values("branch_update")
                    style = "-" if treatment == "partial" else "--"
                    label = f"{treatment} / seed {seed}"
                    axes[0].plot(group.branch_update, group.raw_mse.rolling(5, min_periods=1).mean(), style, color=COLORS[treatment], alpha=.45, label=label)
                    axes[1].plot(group.branch_update, group.high_frequency_mse if "high_frequency_mse" in group else group[[f"band_mse_{i}" for i in range(4, 8)]].mean(axis=1), style, color=COLORS[treatment], alpha=.45)
                    axes[2].plot(group.branch_update, group.grad_norm, style, color=COLORS[treatment], alpha=.45)
                axes[0].set_title("训练 batch raw velocity MSE")
                axes[1].set_title("高频 band MSE（训练 batch）")
                axes[2].set_title("参数梯度范数")
                for axis in axes:
                    axis.set_xlabel("branch update")
                    axis.grid(alpha=.2)
                handles, labels = axes[0].get_legend_handles_labels()
                figure.tight_layout(rect=(0, .16, 1, 1))
                figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
            """
        ),
        markdown("### 5. Fixed held-out latent diagnostics"),
        code(
            """
            validation_path = RESULTS / "fixed_validation_metrics.csv"
            if validation_path.exists():
                validation = pd.read_csv(validation_path)
                figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
                for treatment, group in validation.groupby("treatment"):
                    mean = group.groupby("branch_update", as_index=False)[["raw_mse", "decoder_weighted_mse", "high_frequency_mse"]].mean()
                    for axis, metric, title in zip(
                        axes,
                        ["raw_mse", "decoder_weighted_mse", "high_frequency_mse"],
                        ["Raw MSE", "Decoder-sensitive MSE", "High-frequency MSE"],
                    ):
                        axis.plot(mean.branch_update, mean[metric], marker="o", lw=2.2, color=COLORS[treatment], label=treatment)
                        axis.set_title(title)
                        axis.set_xlabel("branch update")
                        axis.grid(alpha=.2)
                axes[0].legend(frameon=False)
                display(validation.sort_values(["seed", "treatment", "branch_update"]).round(6))
                endpoint = validation.query("branch_update == 5000")
                endpoint_summary = []
                for metric in ["raw_mse", "decoder_weighted_mse", "low_frequency_mse", "high_frequency_mse", "band_mse_0", "band_mse_7"]:
                    paired_metric = endpoint.pivot(index="seed", columns="treatment", values=metric)
                    gain = (paired_metric["baseline"] - paired_metric["partial"]) / paired_metric["baseline"]
                    endpoint_summary.append({
                        "metric": metric,
                        "mean_relative_improvement": gain.mean(),
                        "min_seed": gain.min(),
                        "max_seed": gain.max(),
                    })
                display(pd.DataFrame(endpoint_summary).style.format({
                    "mean_relative_improvement": "{:.2%}", "min_seed": "{:.2%}", "max_seed": "{:.2%}",
                }))
            else:
                validation = pd.DataFrame()
                print("固定 validation 诊断尚未运行；训练完成后执行 experiments/evaluate_rae_spectral_tiny.py。")
            """
        ),
        markdown("### 6. Fixed-noise 5k generation"),
        code(
            """
            generation_path = RESULTS / "generation_metrics.csv"
            if generation_path.exists():
                generation = pd.read_csv(generation_path)
                kid_column = "kernel_inception_distance_mean"
                metrics = [kid_column, "frechet_inception_distance", "inception_score_mean"]
                titles = ["KID-5k（低）", "FID-5k proxy（低）", "Inception Score（高）"]
                figure, axes = plt.subplots(1, 4, figsize=(21, 5.2), constrained_layout=True)
                x = np.arange(len(generation.seed.unique()))
                width = .34
                for offset, treatment in enumerate(["baseline", "partial"]):
                    group = generation.query("treatment == @treatment").sort_values("seed")
                    for axis, metric, title in zip(axes[:3], metrics, titles):
                        axis.bar(x + (offset - .5) * width, group[metric], width, color=COLORS[treatment], label=treatment)
                        axis.set_xticks(x, group.seed)
                        axis.set_title(title)
                        axis.set_xlabel("training seed")
                        axis.grid(axis="y", alpha=.2)
                axes[0].legend(frameon=False)
                kid_paired = generation.pivot(index="seed", columns="treatment", values=kid_column).sort_index()
                kid_gain = 100 * (kid_paired["baseline"] - kid_paired["partial"]) / kid_paired["baseline"]
                axes[3].bar(x, kid_gain, width=.55, color=COLORS["partial"])
                axes[3].axhline(5, color="#DC2626", lw=2, ls="--", label="预注册门槛 5%")
                axes[3].set_xticks(x, kid_paired.index)
                axes[3].set(xlabel="training seed", ylabel="相对改善（%）", title="KID 改善未达到门槛")
                axes[3].set_ylim(0, max(6, float(kid_gain.max()) * 1.25))
                axes[3].grid(axis="y", alpha=.2)
                axes[3].legend(frameon=False)
                display(generation.round(6))
            else:
                generation = pd.DataFrame()
                print("5k generation 尚未完成；这里不会用训练 loss 代替生成结论。")
            """
        ),
        markdown("### 7. Paired image panel"),
        code(
            """
            def V_pair(seed=3407, indices=(0, 1, 2, 3, 1000, 2000)):
                folders = {}
                for treatment in ["baseline", "partial"]:
                    branch = RESULTS / f"seed{seed}_{treatment}_from_s5000"
                    folders[treatment] = branch / "generation" / f"fixed_seed20260715_5000_step10000"
                if not all(folder.exists() for folder in folders.values()):
                    print("该 seed 的配对样本尚未生成。")
                    return None
                figure, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6.2), constrained_layout=True)
                for row, treatment in enumerate(["baseline", "partial"]):
                    for column, index in enumerate(indices):
                        axes[row, column].imshow(Image.open(folders[treatment] / f"{index:06d}.png"))
                        axes[row, column].axis("off")
                        if row == 0:
                            axes[row, column].set_title(f"sample {index}")
                        if column == 0:
                            axes[row, column].annotate(
                                treatment,
                                xy=(-.04, .5),
                                xycoords="axes fraction",
                                ha="right",
                                va="center",
                                rotation=90,
                                fontsize=12,
                                fontweight="bold",
                            )
                return figure

            _ = V_pair()
            """
        ),
        markdown("## Takeaways\n\n### 8. Apply the preregistered decision rule"),
        code(
            """
            if generation.empty:
                decision = "PENDING：真实 5k 生成尚未完成，不能判活或判死。"
                decision_table = pd.DataFrame()
            else:
                kid = "kernel_inception_distance_mean"
                paired = generation.pivot(index="seed", columns="treatment", values=kid).dropna()
                paired["relative_gain"] = (paired["baseline"] - paired["partial"]) / paired["baseline"]
                paired["passes_5pct"] = paired["relative_gain"] >= .05
                wins = int((paired["relative_gain"] > 0).sum())
                practical = int(paired["passes_5pct"].sum())
                if wins >= 2 and practical >= 2:
                    decision = "PASS SCREEN：至少 2/3 seed 的 KID 同方向且达到 5%；补到 5 seeds，但暂不做大训练。"
                else:
                    decision = f"FAIL SCREEN：KID 同方向 {wins}/3，但达到 5% 的只有 {practical}/3；停止 frequency-weighting 方法线，不调 gamma 追结果。"
                decision_table = paired.reset_index()
            print(decision)
            if not decision_table.empty:
                display(decision_table.style.format({"relative_gain": "{:.2%}"}))
                fid = generation.pivot(index="seed", columns="treatment", values="frechet_inception_distance")
                fid_gain = (fid["baseline"] - fid["partial"]) / fid["baseline"]
                score = generation.pivot(index="seed", columns="treatment", values="inception_score_mean")
                score_gain = (score["partial"] - score["baseline"]) / score["baseline"]
                print(f"KID 平均相对改善：{decision_table.relative_gain.mean():.2%}")
                print(f"FID 平均相对改善：{fid_gain.mean():.2%}（负值表示变差）")
                print(f"Inception Score 平均相对改善：{score_gain.mean():.2%}")
            print("无论结果如何，参数梯度范数不匹配意味着结论属于目标/容量分配，而不是纯优化加速。")
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_notebook()

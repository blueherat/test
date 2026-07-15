import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/stochastic_fm_whitening_toy.ipynb"


def markdown(source: str):
    normalized = dedent(source).strip()
    cell = nbf.v4.new_markdown_cell(normalized)
    cell["id"] = f"md-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:12]}"
    return cell


def code(source: str):
    normalized = dedent(source).strip()
    cell = nbf.v4.new_code_cell(normalized)
    cell["id"] = f"code-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:12]}"
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
            # Stochastic FM 协方差白化：精确解析 toy

            这个 notebook 只回答一个机制问题：

            > 当完整 residual whitening 已经把确定性曲率修到最优时，microscopic velocity
            > target 的不可约噪声是否仍会让有限 batch、有限预算训练偏好分数次 whitening？

            全部动力学都有闭式递推，不训练神经网络、不下载数据，也不使用 FID。默认同时比较：

            - `decoupled`：8 个参数，作为可分离线性负对照；
            - `shared`：4 维共享瓶颈，模拟有限网络中的参数共享。

            日常只需修改“配置”和“最简操作区”两个单元格。主要接口是
            `Run(...)`、`V_curve(...)` 和 `V_sweep(...)`。
            """
        ),
        markdown(
            r"""
            ## 1. 模型与可证伪假设

            在总体最优速度场附近，令参数误差为 $e$，原始 velocity 误差局部线性化为

            $$\delta v = Je.$$

            每个输出方向的总 residual 协方差、可预测协方差和不可约噪声分别为

            $$R_i=S_i+N_i,\qquad \rho_i=S_i/R_i.$$

            分数次 whitening 使用

            $$W_\gamma=\operatorname{diag}(R_i^{-\gamma}),\qquad 0\leq\gamma\leq1.$$

            其中 `gamma=0` 是原始 MSE，`gamma=1` 是完整 whitening。局部 Hessian 和 batch
            target-noise covariance 为

            $$H_\gamma=\frac1dJ^\top W_\gamma J,$$

            $$G_\gamma=\frac1{d^2B}J^\top W_\gamma N W_\gamma J.$$

            于是均值和协方差严格满足

            $$\mu_{k+1}=(I-\eta H_\gamma)\mu_k,$$

            $$\Sigma_{k+1}=(I-\eta H_\gamma)\Sigma_k(I-\eta H_\gamma)^\top
            +\eta^2G_\gamma.$$

            ### 关键控制

            - 构造 $J=\operatorname{diag}(\sqrt R)B$，其中 $B$ 列正交。因此
              `gamma=1` 时 $H_1=I/d$，完整 whitening 在确定性条件数上被刻意设为最优。
            - 每个 `gamma` 使用相同的最大稳定步长比例，而不是用同一个绝对学习率惩罚某个方法。
            - 评价始终回到原始 velocity/decoder 坐标，分解为 optimization bias 和 stochastic
              misadjustment。
            - 该 toy 是局部二次/NTK 机制实验，不证明深层 DiT、AdamW 或完整 ODE 上一定成立。
            """
        ),
        markdown("## 2. 设置\n\n### 2.1 导入依赖"),
        code(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from experiments.stochastic_fm_whitening_toy import (
                analytic_operators,
                analytic_trajectory,
                make_toy,
                mechanism_checks,
                mode_table,
                monte_carlo_validation,
                optimal_gamma,
                run_sweep,
            )

            np.set_printoptions(precision=5, suppress=True)
            plt.rcParams.update({
                "figure.dpi": 120,
                "figure.figsize": (10, 5),
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "font.size": 10,
                "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                "axes.unicode_minus": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
            """
        ),
        markdown("### 2.2 配置\n\n默认配置故意让低能量方向较难预测，从而构成 whitening 的压力测试。"),
        code(
            """
            SEED = 0
            STEPS = 500
            LEARNING_RATE_FRACTION = 0.40

            RESIDUAL_VARIANCE = np.logspace(-2.0, 2.0, 8)
            PREDICTABILITY = np.array([0.02, 0.05, 0.10, 0.40, 0.80, 0.98, 0.999, 0.99999])
            DECODER_GAIN = np.array([8.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0])

            GAMMAS = np.linspace(0.0, 1.0, 21)
            DISPLAY_GAMMAS = (0.0, 0.5, 0.75, 1.0)
            BATCH_SIZES = (4.0, 16.0, 64.0, 256.0, np.inf)

            TOYS = {
                "decoupled": make_toy(
                    parameter_dim=8,
                    residual_variance=RESIDUAL_VARIANCE,
                    predictability=PREDICTABILITY,
                    decoder_gain=DECODER_GAIN,
                    seed=SEED,
                ),
                "shared": make_toy(
                    parameter_dim=4,
                    residual_variance=RESIDUAL_VARIANCE,
                    predictability=PREDICTABILITY,
                    decoder_gain=DECODER_GAIN,
                    seed=SEED,
                ),
            }

            def batch_label(batch_size):
                return "full" if np.isinf(batch_size) else str(int(batch_size))

            display(mode_table(TOYS["decoupled"]).round(5))
            """
        ),
        markdown("### 2.3 最简 `Run / V` 接口"),
        code(
            """
            def Run(gamma=0.5, batch_size=16, architecture="decoupled", steps=STEPS):
                return analytic_trajectory(
                    TOYS[architecture],
                    gamma=gamma,
                    batch_size=batch_size,
                    steps=steps,
                    learning_rate_fraction=LEARNING_RATE_FRACTION,
                )

            def V_curve(
                gammas=DISPLAY_GAMMAS,
                batch_size=16,
                architecture="decoupled",
                steps=STEPS,
                ax=None,
            ):
                if ax is None:
                    _, ax = plt.subplots(figsize=(9, 5))
                for gamma in gammas:
                    result = Run(gamma, batch_size, architecture, steps)
                    ax.plot(
                        result["step"],
                        np.maximum(result["risk"], 1e-12),
                        label=f"gamma={gamma:g}",
                        linewidth=2,
                    )
                ax.set_yscale("log")
                ax.set_xlabel("训练步数")
                ax.set_ylabel("期望 decoder-weighted risk")
                ax.set_title(f"{architecture} | batch={batch_label(batch_size)}")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, ncol=2)
                return ax

            def Sweep(architecture="decoupled", steps=STEPS):
                return run_sweep(
                    TOYS[architecture],
                    gammas=GAMMAS,
                    batch_sizes=BATCH_SIZES,
                    steps=steps,
                    learning_rate_fraction=LEARNING_RATE_FRACTION,
                )

            def V_sweep(architecture="decoupled", steps=STEPS):
                _, summary = Sweep(architecture, steps)
                table = summary.copy()
                table["batch"] = table["batch_size"].map(batch_label)
                heat = table.pivot(index="gamma", columns="batch", values="risk_auc")
                heat = heat.reindex(columns=[batch_label(batch) for batch in BATCH_SIZES])

                fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
                image = axes[0].imshow(
                    np.log10(heat.to_numpy()),
                    origin="lower",
                    aspect="auto",
                    cmap="viridis_r",
                )
                axes[0].set_xticks(range(heat.shape[1]), heat.columns)
                axes[0].set_yticks(
                    range(0, heat.shape[0], 4),
                    [f"{value:.1f}" for value in heat.index[::4]],
                )
                axes[0].set_xlabel("batch size")
                axes[0].set_ylabel("gamma")
                axes[0].set_title(f"{architecture}: log10(AUC risk)")
                fig.colorbar(image, ax=axes[0], shrink=0.85)

                best = optimal_gamma(summary)
                axes[1].plot(
                    [batch_label(value) for value in best["batch_size"]],
                    best["gamma"],
                    marker="o",
                    linewidth=2,
                )
                axes[1].set_ylim(-0.03, 1.03)
                axes[1].set_xlabel("batch size")
                axes[1].set_ylabel("AUC 最优 gamma")
                axes[1].set_title("随机噪声减小时，最优 whitening 强度")
                axes[1].grid(alpha=0.25)
                plt.show()
                return summary, best
            """
        ),
        markdown(
            """
            ## 3. 最简操作区

            只改下面一格即可快速查看某个设置。`batch_size=np.inf` 表示 full-batch；
            `architecture` 可选 `decoupled` 或 `shared`。
            """
        ),
        code(
            """
            architecture = "decoupled"
            batch_size = 16
            gammas = (0.0, 0.5, 0.75, 1.0)

            result = Run(gamma=0.5, batch_size=batch_size, architecture=architecture)
            V_curve(gammas=gammas, batch_size=batch_size, architecture=architecture)
            plt.show()
            """
        ),
        markdown("## 4. 结果\n\n### 4.1 受控的 residual 谱、可预测性与 decoder 权重"),
        code(
            """
            factors = mode_table(TOYS["decoupled"])
            figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

            axes[0].plot(factors["direction"], factors["residual_variance_R"], marker="o")
            axes[0].set_yscale("log")
            axes[0].set_title("总 residual variance R")
            axes[0].set_xlabel("方向")

            axes[1].plot(factors["direction"], factors["predictable_fraction_rho"], marker="o")
            axes[1].plot(factors["direction"], factors["irreducible_fraction"], marker="s")
            axes[1].set_title("可预测 / 不可约比例")
            axes[1].set_xlabel("方向")
            axes[1].legend(["rho", "1-rho"], frameon=False)

            axes[2].bar(factors["direction"], factors["decoder_gain"])
            axes[2].set_title("decoder gain")
            axes[2].set_xlabel("方向")

            for axis in axes:
                axis.grid(alpha=0.2)
            plt.show()
            """
        ),
        markdown(
            """
            ### 4.2 Full-batch：只看确定性条件数

            这里没有 target noise。按构造，`gamma=1` 的条件数应精确等于 1；如果它仍然不比
            baseline 快，说明 toy 或实现有错误。
            """
        ),
        code(
            """
            figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for architecture, axis in zip(("decoupled", "shared"), axes):
                V_curve(batch_size=np.inf, architecture=architecture, ax=axis)
            plt.show()

            condition_rows = []
            for architecture, toy in TOYS.items():
                for gamma in DISPLAY_GAMMAS:
                    operators = analytic_operators(
                        toy,
                        gamma=gamma,
                        batch_size=np.inf,
                        learning_rate_fraction=LEARNING_RATE_FRACTION,
                    )
                    condition_rows.append({
                        "architecture": architecture,
                        "gamma": gamma,
                        "condition_number": operators.condition_number,
                    })
            display(pd.DataFrame(condition_rows).round(5))
            """
        ),
        markdown(
            """
            ### 4.3 Minibatch：收敛速度与 misadjustment 同时出现

            这里加入 microscopic target noise。完整 whitening 仍最快消除 mean bias，但它会让
            原本低能量、低可预测的方向以接近单位强度注入随机梯度。
            """
        ),
        code(
            """
            figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
            for row, architecture in enumerate(("decoupled", "shared")):
                for column, selected_batch in enumerate((4, 64)):
                    V_curve(
                        batch_size=selected_batch,
                        architecture=architecture,
                        ax=axes[row, column],
                    )
            plt.show()
            """
        ),
        markdown(
            """
            ### 4.3b 因果负对照：保持曲率不变，只移除不可约 target noise

            将所有 `rho` 设为 1 后，`J`、`R`、decoder metric 和 batch size 都不变，仅令
            $N=0$。如果有限 batch 的分数次最优确实来自 microscopic target noise，完整
            whitening 应恢复为最优。
            """
        ),
        code(
            """
            NOISELESS_TOY = make_toy(
                parameter_dim=8,
                residual_variance=RESIDUAL_VARIANCE,
                predictability=np.ones_like(PREDICTABILITY),
                decoder_gain=DECODER_GAIN,
                seed=SEED,
            )

            figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for axis, (title, toy) in zip(
                axes,
                (("原始 stochastic target", TOYS["decoupled"]), ("N=0 负对照", NOISELESS_TOY)),
            ):
                for gamma in DISPLAY_GAMMAS:
                    rows = analytic_trajectory(
                        toy,
                        gamma=gamma,
                        batch_size=4,
                        steps=STEPS,
                        learning_rate_fraction=LEARNING_RATE_FRACTION,
                    )
                    axis.plot(
                        rows["step"],
                        np.maximum(rows["risk"], 1e-12),
                        linewidth=2,
                        label=f"gamma={gamma:g}",
                    )
                axis.set_yscale("log")
                axis.set_xlabel("训练步数")
                axis.set_ylabel("期望 risk")
                axis.set_title(f"{title} | batch=4")
                axis.grid(alpha=0.25)
                axis.legend(frameon=False, ncol=2)
            plt.show()

            _, noiseless_summary = run_sweep(
                NOISELESS_TOY,
                gammas=GAMMAS,
                batch_sizes=(4.0,),
                steps=STEPS,
                learning_rate_fraction=LEARNING_RATE_FRACTION,
            )
            display(optimal_gamma(noiseless_summary).round(6))
            """
        ),
        markdown("### 4.4 预注册扫描：`gamma × batch size`"),
        code(
            """
            curve_frames = []
            summary_frames = []
            for architecture in TOYS:
                curves, summary = Sweep(architecture)
                curve_frames.append(curves)
                summary_frames.append(summary)

            ALL_CURVES = pd.concat(curve_frames, ignore_index=True)
            SUMMARY = pd.concat(summary_frames, ignore_index=True)
            OPTIMUM = optimal_gamma(SUMMARY)
            display(OPTIMUM.round(6))
            """
        ),
        code(
            """
            figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for architecture, axis in zip(("decoupled", "shared"), axes):
                rows = OPTIMUM[OPTIMUM["architecture"] == architecture]
                axis.plot(
                    [batch_label(value) for value in rows["batch_size"]],
                    rows["gamma"],
                    marker="o",
                    linewidth=2.5,
                    label=architecture,
                )
                axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="full whitening")
                axis.set_ylim(-0.03, 1.05)
                axis.set_xlabel("batch size")
                axis.set_ylabel("AUC 最优 gamma")
                axis.set_title(f"{architecture}: finite-budget optimum")
                axis.grid(alpha=0.25)
                axis.legend(frameon=False)
            plt.show()

            # 单独调用 V_sweep("decoupled") 或 V_sweep("shared") 可查看完整热图。
            _, _ = V_sweep("decoupled")
            _, _ = V_sweep("shared")
            """
        ),
        markdown("### 4.5 将最终风险拆成 optimization bias 与 stochastic misadjustment"),
        code(
            """
            decomposition_rows = []
            for architecture in TOYS:
                for gamma in DISPLAY_GAMMAS:
                    trajectory = Run(gamma, 16, architecture)
                    last = trajectory.iloc[-1]
                    decomposition_rows.append({
                        "architecture": architecture,
                        "gamma": gamma,
                        "bias": last["bias"],
                        "misadjustment": last["misadjustment"],
                    })
            decomposition = pd.DataFrame(decomposition_rows)

            figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
            for architecture, axis in zip(("decoupled", "shared"), axes):
                rows = decomposition[decomposition["architecture"] == architecture]
                width = 0.10
                bias_plot = np.maximum(rows["bias"].to_numpy(), 1e-12)
                noise_plot = np.maximum(rows["misadjustment"].to_numpy(), 1e-12)
                axis.bar(
                    rows["gamma"] - width / 2,
                    bias_plot,
                    width=width,
                    label="optimization bias",
                )
                axis.bar(
                    rows["gamma"] + width / 2,
                    noise_plot,
                    width=width,
                    label="misadjustment",
                )
                axis.set_yscale("log")
                axis.set_xlabel("gamma")
                axis.set_ylabel("step 500 risk")
                axis.set_title(f"{architecture} | batch=16")
                axis.grid(alpha=0.2)
                axis.legend(frameon=False)
            plt.show()
            display(decomposition.round(8))
            """
        ),
        markdown(
            """
            ## 5. 数值核验

            ### 5.1 Monte Carlo 与闭式协方差递推

            显式模拟 1024 条随机优化轨迹，验证解析期望不是公式实现中的假象。
            """
        ),
        code(
            """
            MC = monte_carlo_validation(
                TOYS["shared"],
                gamma=0.5,
                batch_size=16,
                steps=200,
                runs=1024,
                learning_rate_fraction=LEARNING_RATE_FRACTION,
                seed=1234,
            )

            figure, axis = plt.subplots(figsize=(10, 5))
            axis.plot(MC["step"], MC["analytic_risk"], linewidth=2.5, label="解析期望")
            axis.plot(MC["step"], MC["monte_carlo_risk"], linewidth=1.8, label="Monte Carlo")
            axis.fill_between(
                MC["step"],
                np.maximum(MC["monte_carlo_risk"] - 2 * MC["monte_carlo_se"], 1e-12),
                MC["monte_carlo_risk"] + 2 * MC["monte_carlo_se"],
                alpha=0.2,
                label="Monte Carlo +/- 2 SE",
            )
            axis.set_yscale("log")
            axis.set_xlabel("训练步数")
            axis.set_ylabel("期望 risk")
            axis.set_title("闭式递推与显式随机轨迹一致性")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
            plt.show()

            mc_selected = MC.iloc[[20, 50, 100, 200]].copy()
            mc_selected["z_error"] = (
                mc_selected["absolute_error"] / np.maximum(mc_selected["monte_carlo_se"], 1e-12)
            )
            display(mc_selected.round(7))
            assert mc_selected["z_error"].max() < 4.0
            """
        ),
        markdown("### 5.2 自动机制检查"),
        code(
            """
            CHECKS = mechanism_checks(
                TOYS["decoupled"],
                TOYS["shared"],
                steps=STEPS,
                learning_rate_fraction=LEARNING_RATE_FRACTION,
            )
            display(CHECKS)
            assert CHECKS["passed"].all(), CHECKS.loc[~CHECKS["passed"]].to_string(index=False)
            """
        ),
        markdown("## 6. 运行后结论"),
        code(
            """
            full_condition = analytic_operators(
                TOYS["decoupled"], gamma=1.0, batch_size=np.inf
            ).condition_number
            finite_optimum = OPTIMUM[np.isfinite(OPTIMUM["batch_size"])]
            full_optimum = OPTIMUM[np.isinf(OPTIMUM["batch_size"])]

            print(f"1. 完整 whitening 的 full-batch 条件数：{full_condition:.6f}。")
            for architecture in ("decoupled", "shared"):
                finite = finite_optimum[finite_optimum["architecture"] == architecture]
                full = full_optimum[full_optimum["architecture"] == architecture]
                pairs = ", ".join(
                    f"B={batch_label(row.batch_size)} -> gamma={row.gamma:.2f}"
                    for row in finite.itertuples()
                )
                print(f"2. {architecture} 的有限 batch 最优值：{pairs}。")
                print(f"   full-batch 最优 gamma={float(full['gamma'].iloc[0]):.2f}。")
            print("3. 因而该 toy 支持的是条件命题：完整 whitening 改善确定性条件数，")
            print("   但有限 batch 下的最佳强度还取决于不可约 target noise 与参数共享。")
            print("4. 这仍不是 RAE/DiT 证据；下一关应是受控非线性 toy 与 AdamW。")
            """
        ),
        markdown(
            r"""
            ## 7. 边界与下一步

            - 这里固定 Hessian，只隔离 microscopic target noise；没有模拟输入采样导致的随机 Hessian。
            - `J` 是局部线性化，不能替代真实 DiT 的 LayerNorm、attention、AdamW 和特征学习。
            - 默认 `rho` 与 residual 能量相关，是预先声明的压力测试；应主动修改
              `PREDICTABILITY` 为全高、全低和反向相关，确认结论条件而不是追求单一胜负。
            - decoder gain 只用于终点风险，不等于包含完整 ODE 灵敏度的 filtered-x secondary path。
            - 只有当相同 crossover 在非线性 toy 和真实 RAE gradient-SNR 审计中仍存在，才值得训练模型。
            """
        ),
    ]
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_notebook()

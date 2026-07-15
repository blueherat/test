from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def build_playground():
    cells = [
        markdown(
            r"""
            # Architecture-aware Gauge Playground

            ## Goal

            在固定 codec 下只改变严格正交 latent 坐标 `y=A(z)`。这个 notebook 暴露
            最简接口 `A(z)`、`A_inv(y)`，先验证重建和扩散噪声路径严格等价，再用
            paired tiny probe 快速检查 identity 附近是否可能存在 headroom。

            这里的 quick probe 只是 go/no-go smoke，不是 FID 或论文结论。
            """
        ),
        markdown(
            """
            ## Setup

            ### 1. Imports

            数据集从 `/data/shared` 读取；模型与本机结果由仓库软链接指向
            `$HOME/data/eqvae`。默认全程 fp32。
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            import torch
            from IPython.display import display

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from experiments.architecture_gauge import (
                CodecDataConfig, GaugeSpec, ProbeConfig, ProbeTrainingConfig,
                add_identity_ratios, exact_equivalence_table, finite_difference_headroom,
                make_gauge, plot_learning_curves, prepare_codec_data,
                reconstruction_equivalence_table, run_probe_grid,
                visualize_gauge_roundtrip, visualize_latent_pca,
            )
            """
        ),
        markdown(
            """
            ### 2. Parameters

            日常只改这个单元。`MODEL_KEY` 可选 `rae_dinov2`、`rae_mae`、
            `rae_siglip2`、`sdvae`、`eqvae`。`A_KIND` 可选 `identity`、`roll`、
            `channel_givens`、`fourier_allpass`、`block_haar`。
            """
        ),
        code(
            """
            MODEL_KEY = "rae_dinov2"
            DATASET_NAME = "imagenet_parquet"
            DATASET_PATH = "/data/shared/imagenet-1k"
            TRAIN_SPLIT, VAL_SPLIT = "train", "validation"
            TRAIN_COUNT, VAL_COUNT = 16, 8
            IMAGE_SIZE = 256
            DEVICE = "cuda:0"
            SEED = 0

            # 当前 A。all-pass 的 radius 控制主要混合距离，strength 控制强度。
            A_SPEC = GaugeSpec(
                name="my_A",
                kind="fourier_allpass",
                strength=0.45,
                radius=2,
                seed=SEED,
            )

            # identity 邻域的对称方向，用于估计 directional gradient / curvature。
            HEADROOM_DELTA = 0.25
            COMPARE_GAUGES = [
                GaugeSpec("identity"),
                GaugeSpec("allpass_plus", kind="fourier_allpass", strength=+HEADROOM_DELTA, radius=1),
                GaugeSpec("allpass_minus", kind="fourier_allpass", strength=-HEADROOM_DELTA, radius=1),
                GaugeSpec("channel_0.5", kind="channel_givens", strength=0.5, seed=SEED),
            ]

            RUN_QUICK_PROBE = True
            QUICK_STEPS = 8
            """
        ),
        markdown(
            """
            ## Steps

            ### 3. Load frozen codec and disjoint data

            ImageNet 默认直接使用 train/validation 两个官方 split，因此 quick probe
            不会把 validation 图像用于更新。
            """
        ),
        code(
            """
            data = prepare_codec_data(CodecDataConfig(
                dataset_name=DATASET_NAME,
                dataset_path=DATASET_PATH,
                train_split=TRAIN_SPLIT,
                val_split=VAL_SPLIT,
                train_count=TRAIN_COUNT,
                val_count=VAL_COUNT,
                image_size=IMAGE_SIZE,
                model_key=MODEL_KEY,
                device=DEVICE,
                seed=SEED,
            ))

            print("train images:", tuple(data.train_images.shape), "train z:", tuple(data.train_latents.shape))
            print("val images:  ", tuple(data.val_images.shape), "val z:  ", tuple(data.val_latents.shape))
            print("latent RMS scale:", data.latent_scale)
            """
        ),
        markdown("### 4. Minimal `A` interface"),
        code(
            """
            active_gauge = make_gauge(A_SPEC)

            def A(z):
                return active_gauge.forward(z)

            def A_inv(y):
                return active_gauge.inverse(y)

            x = data.val_images
            z = data.val_latents.to(data.device)
            y = A(z)
            z_roundtrip = A_inv(y)

            print("A:", A_SPEC)
            print("max |A_inv(A(z))-z|:", float((z_roundtrip - z).abs().max()))
            """
        ),
        markdown(
            """
            ### 5. Exact equivalence gate

            `inverse/norm/distance/noise` 应接近 fp32 数值误差。all-pass 还应保持总 PSD；
            `block_haar` 是正交的，但故意不保持逐频率 PSD。
            """
        ),
        code(
            """
            exact = exact_equivalence_table(data.val_latents, [A_SPEC, *COMPARE_GAUGES], device=DEVICE)
            recon_exact = reconstruction_equivalence_table(data, [A_SPEC, *COMPARE_GAUGES], count=2)
            display(exact.round(8))
            display(recon_exact.round(8))

            assert exact[["inverse_rel_l2", "norm_rel_error", "paired_noise_rel_error"]].to_numpy().max() < 1e-5
            assert recon_exact["image_mean_abs_error"].max() < 1e-4
            """
        ),
        markdown(
            """
            ### 6. Visualize the same information in two coordinates

            `D(Az)` 只用于展示 decoder 收到错误坐标时会发生什么，不属于 AGF 方法。
            方法中的 decoder 输入始终是 `A_inv(y)`。
            """
        ),
        code(
            """
            visualize_gauge_roundtrip(data, A_SPEC, count=min(3, VAL_COUNT));
            visualize_latent_pca(data.val_latents, A_SPEC, count=min(4, VAL_COUNT));
            """
        ),
        markdown(
            """
            ### 7. Optional paired headroom probe

            同一 probe 下的 `identity/+delta/-delta` 使用相同样本、噪声、时间、初始化
            和步数。`g != 0` 只表示该 tiny probe 的 identity 可能不是 stationary point；
            至少 3 个 seed 与正式生成迁移成功后才能称为 H2。
            """
        ),
        code(
            """
            if RUN_QUICK_PROBE:
                quick_training = ProbeTrainingConfig(
                    steps=QUICK_STEPS,
                    eval_steps=(0, QUICK_STEPS // 2, QUICK_STEPS),
                    batch_size=min(4, TRAIN_COUNT),
                    eval_batches=2,
                    time_bins=4,
                    seed=SEED,
                )
                quick_probes = [
                    ProbeConfig("local_rf9", kind="local", hidden=16, depth=2),
                    ProbeConfig("global_attn", kind="global", hidden=16, depth=1, heads=4),
                ]
                quick_runs, quick_history, quick_bins = run_probe_grid(
                    data.train_latents,
                    data.val_latents,
                    COMPARE_GAUGES[:3],
                    quick_probes,
                    quick_training,
                    seeds=(SEED,),
                    device=DEVICE,
                    latent_scale=data.latent_scale,
                )
                display(add_identity_ratios(quick_history).round(5))
                display(finite_difference_headroom(
                    quick_history,
                    plus_gauge="allpass_plus",
                    minus_gauge="allpass_minus",
                    delta=HEADROOM_DELTA,
                ).round(5))
                plot_learning_curves(quick_history);
            else:
                print("RUN_QUICK_PROBE=False: exact and visual checks completed.")
            """
        ),
        markdown(
            """
            ## Checks

            - Exact gate 失败：先修 `A/A_inv`、noise pairing 或 codec normalization。
            - 只有坏坐标、没有坐标优于 identity：只支持 H1，不支持质量方法。
            - 单 seed tiny probe 的小差异：只能用于排查代码和决定下一组方向。
            - `D(Az)` 很差但 `D(A_inv(Az))` 精确：说明 decoder 只接受原坐标，符合设计。

            ## Next Steps

            用 `gauge_mechanism_routing.ipynb` 检查差异究竟来自 locality、有限训练预算、
            decoder 放大还是 time-bin 冲突。
            """
        ),
    ]
    return notebook(cells)


def build_routing():
    cells = [
        markdown(
            """
            # Step 2: Gauge Mechanism Routing

            ## tl;dr

            这个 notebook 不预设 LocalGauge 正确。它用同一批严格正交 gauge 和 paired
            tiny probes，依次检查 exactness、local/global 与感受野、短训/长训、二阶
            统计、time-bin 偏好和 decoder 对真实 probe residual 的放大，并输出下一步
            分流表。

            默认设置是可运行 smoke。只有 `SEEDS>=3`、更长预算和正式生成实验才能形成
            研究结论。
            """
        ),
        markdown(
            """
            ## Context & Methods

            ### Key assumptions

            - `A` 是固定、低容量、严格正交的，不联合训练。
            - train 只更新 probe；validation 只比较 gauge。
            - local/global 不是严格 matched-FLOPs，因此它是机制路由证据；最终必须加
              matched-FLOPs attention 和 cross-architecture crossover。
            - decoder 分支使用 probe 实际预测出的 `z0` residual，不使用各向同性随机噪声。
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            from IPython.display import display

            ROOT = Path.cwd()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from experiments.architecture_gauge import (
                CodecDataConfig, GaugeSpec, ProbeConfig, ProbeTrainingConfig,
                add_identity_ratios, decoder_residual_table, exact_equivalence_table,
                finite_difference_headroom, mechanism_routing_table,
                plot_decoder_residuals, plot_learning_curves, plot_locality_comparison,
                plot_second_order_control, plot_time_bin_heatmap,
                prepare_codec_data, probe_configs, reconstruction_equivalence_table,
                run_probe_grid,
            )
            """
        ),
        markdown("### Experiment parameters"),
        code(
            """
            MODEL_KEY = "rae_dinov2"
            DATASET_NAME = "imagenet_parquet"
            DATASET_PATH = "/data/shared/imagenet-1k"
            TRAIN_COUNT, VAL_COUNT = 24, 12
            DEVICE = "cuda:0"
            SEED = 0

            # Smoke: 12 steps, one seed. Research run: >=500 steps and SEEDS=(0,1,2).
            STEPS = 12
            SEEDS = (0,)
            HIDDEN = 16
            HEADROOM_DELTA = 0.25

            GAUGES = [
                GaugeSpec("identity"),
                GaugeSpec("allpass_plus", kind="fourier_allpass", strength=+HEADROOM_DELTA, radius=1),
                GaugeSpec("allpass_minus", kind="fourier_allpass", strength=-HEADROOM_DELTA, radius=1),
                GaugeSpec("allpass_r3", kind="fourier_allpass", strength=0.65, radius=3),
                GaugeSpec("channel_0.5", kind="channel_givens", strength=0.5, seed=SEED),
                GaugeSpec("haar_2x2", kind="block_haar"),
            ]
            PROBES = probe_configs(hidden=HIDDEN)
            TRAINING = ProbeTrainingConfig(
                steps=STEPS,
                eval_steps=(0, max(1, STEPS // 3), max(2, 2 * STEPS // 3), STEPS),
                batch_size=min(4, TRAIN_COUNT),
                eval_batches=3,
                time_bins=6,
                seed=SEED,
            )
            """
        ),
        markdown(
            """
            ## Data

            ### 1. Frozen codec and disjoint ImageNet splits
            """
        ),
        code(
            """
            data = prepare_codec_data(CodecDataConfig(
                dataset_name=DATASET_NAME,
                dataset_path=DATASET_PATH,
                train_split="train",
                val_split="validation",
                train_count=TRAIN_COUNT,
                val_count=VAL_COUNT,
                model_key=MODEL_KEY,
                device=DEVICE,
                seed=SEED,
            ))
            print("train z:", tuple(data.train_latents.shape), "val z:", tuple(data.val_latents.shape))
            print("train/val source splits: train / validation")
            """
        ),
        markdown(
            """
            ## Results

            ### 2. Exact gate and second-order controls

            all-pass 若在 probe 上产生差异，但 `total_psd_rel_error` 仍接近零，说明差异
            不能由总功率谱解释。它仍不自动证明 higher-order locality 是唯一原因。
            """
        ),
        code(
            """
            exact = exact_equivalence_table(data.val_latents, GAUGES, device=DEVICE)
            recon_exact = reconstruction_equivalence_table(data, GAUGES, count=2)
            display(exact.round(8))
            display(recon_exact.round(8))
            assert exact[["inverse_rel_l2", "norm_rel_error", "paired_noise_rel_error"]].to_numpy().max() < 1e-5
            """
        ),
        markdown(
            """
            ### 3. Paired probe grid

            `local_rf5` 与 `local_rf9` 在同一卷积族内改变感受野；`global_attn` 提供全局
            读取对照。表中同时显示参数量，避免把容量差异藏起来。
            """
        ),
        code(
            """
            runs, history, time_rows = run_probe_grid(
                data.train_latents,
                data.val_latents,
                GAUGES,
                PROBES,
                TRAINING,
                seeds=SEEDS,
                device=DEVICE,
                latent_scale=data.latent_scale,
            )

            probe_sizes = history[["probe", "probe_kind", "receptive_field", "parameter_count"]].drop_duplicates()
            final_ratios = add_identity_ratios(history)
            final_ratios = final_ratios[final_ratios["step"] == STEPS]
            display(probe_sizes)
            display(final_ratios[["probe", "gauge", "seed", "relative_mse", "loss_ratio_to_identity"]].round(5))
            """
        ),
        markdown(
            """
            ### 4. Finite-horizon and locality views

            左图回答差异是否随训练预算消失；右图回答同一 gauge 对不同感受野/全局
            probe 的影响是否系统不同。identity 水平线固定为 1。
            """
        ),
        code(
            """
            plot_learning_curves(history);
            plot_locality_comparison(history);
            """
        ),
        markdown("### 5. Identity-neighborhood directional check"),
        code(
            """
            headroom = finite_difference_headroom(
                history,
                plus_gauge="allpass_plus",
                minus_gauge="allpass_minus",
                delta=HEADROOM_DELTA,
            )
            display(headroom.round(6))
            """
        ),
        markdown(
            """
            ### 6. Can second-order statistics explain probe sensitivity?

            这张散点图是控制图，不做线性因果推断。关键读法是：PSD/Gram 误差接近
            数值零时，probe ratio 是否仍明显偏离 1。
            """
        ),
        code("plot_second_order_control(exact, history);"),
        markdown(
            """
            ### 7. Noise-level preference

            同一 gauge 在低噪声和高噪声 bin 中若稳定出现相反方向，才有理由考虑
            time-dependent gauge。单 seed 的局部翻转只算提示。
            """
        ),
        code(
            """
            plot_time_bin_heatmap(time_rows, probe="local_rf5");
            plot_time_bin_heatmap(time_rows, probe="global_attn");
            """
        ),
        markdown(
            """
            ### 8. Decoder amplification from actual probe residuals

            只解码 identity 与强 all-pass 的 local/global 结果，避免 notebook 被 decoder
            可视化拖慢。横轴是预测 `z0` 的 latent RMSE，纵轴是对应 decoded image L1。
            """
        ),
        code(
            """
            decoder_rows = decoder_residual_table(
                data,
                runs,
                TRAINING,
                count=min(3, VAL_COUNT),
                run_filter={
                    ("local_rf5", "identity"),
                    ("local_rf5", "allpass_r3"),
                    ("global_attn", "identity"),
                    ("global_attn", "allpass_r3"),
                },
            )
            display(decoder_rows.round(6))
            plot_decoder_residuals(decoder_rows);
            """
        ),
        markdown(
            """
            ### 9. Mechanism routing table

            路由阈值是项目管理启发式，不是统计检验。`exploratory (<3 paired seeds)`
            的任何 supported/candidate 都必须升级为多 seed 长预算后再使用。
            """
        ),
        code(
            """
            routing = mechanism_routing_table(
                history,
                time_rows,
                exact,
                decoder_rows,
                improvement_threshold=0.02,
                locality_threshold=0.03,
            )
            display(routing)
            """
        ),
        markdown(
            """
            ## Takeaways

            - exact gate 失败：没有机制结论，先修实现。
            - 只有 non-identity 变差：支持 H1，不支持 Architecture-Optimal Gauge。
            - all-pass 二阶统计不变，且 local penalty 随 RF/global 明显收缩：进入 static
              LocalGauge 与跨架构交换实验。
            - 差异随训练消失：只主张 finite-horizon efficiency。
            - latent error 接近但 decoded error 分离：将 decoder-aware 作为独立方向。
            - time-bin 方向稳定反转：才评估 moving gauge。
            - 多方向、多 seed、长预算均无法 beat identity：停止质量方法。

            当前 notebook 的默认 smoke 只验证管线和可视化，不自动证明上述任一机制。
            """
        ),
    ]
    return notebook(cells)


def main():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nbf.write(build_playground(), NOTEBOOKS / "architecture_gauge_playground.ipynb")
    nbf.write(build_routing(), NOTEBOOKS / "gauge_mechanism_routing.ipynb")
    print("wrote architecture_gauge_playground.ipynb and gauge_mechanism_routing.ipynb")


if __name__ == "__main__":
    main()

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/buried_local_manifold_toy.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        markdown(
            r"""
            # Token 空间埋藏流形：精确解析玩具实验

            ## 核心结论

            这个自包含 notebook 将标准化八高斯变量的两个坐标放在**同一个 token 的两个通道**中，
            随后只在 $8\times8$ token 网格的空间维度施加正交混合。其精确机制由感受野捕获的
            gauge 能量决定：当某个感受野捕获能量 $e_i$ 时，它等价于以有效信噪比
            $\gamma e_i$ 观测原始变量。

            实际运行结果建立了预期的因果链：全部 **25** 项数值检查均通过；恒等变换、全局感受野、
            空间负对照以及 oracle 解混的局部性代价均为零；全部 **128** 个显式 latent 验证点都落在
            独立理论预测的三个 Monte Carlo 标准误以内。在 32 个留出的 Givens 变换上，局部风险的
            平均相对误差为 **0.745%**，按变换 bootstrap 得到的 95% 置信区间为
            **0.402%-1.157%**。
            """
        ),
        markdown(
            r"""
            ## 背景与方法

            令 token 数 $N=64$、通道数 $C=2$。标准化八高斯样本
            $s\in\mathbb{R}^2$ 被存放在一个锚点 token 中，因此在恒等坐标下，$1\times1$ 感受野
            已经足够。空间正交矩阵 $A\in\mathbb{R}^{64\times64}$ 以
            $G=A\otimes I_2$ 的形式作用。记 $q=Ae_{j_0}$，则
            $y_{0,i}=q_i s$ 且 $\sum_iq_i^2=1$。

            对环面感受野 $N_r(i)$，定义其捕获能量为

            $$e_i(r)=\sum_{j\in N_r(i)}q_j^2.$$

            完整的局部观测等价于有效信噪比为 $\gamma e_i$ 的二维 AWGN 信道，因此

            $$R_{\rm local}=\sum_iq_i^2\,\mathrm{mmse}_s(\gamma e_i),\qquad
            R_{\rm global}=\mathrm{mmse}_s(\gamma).$$

            ### 关键假设

            - 所有数值推断均使用 FP64。
            - 主要感受野为环面上的 $1\times1$、$3\times3$、$5\times5$、$7\times7$ 以及全局感受野。
            - 积分分数预先固定 $t\sim U(0,1)$，并令
              $\gamma(t)=\cot^2(\pi t/2)$；不使用事后选择的 SNR 权重。
            - 图 1 展示单步 $x_0$ 去噪，而不是完整生成过程或 FID。
            - 本 notebook 只实现解析阶段 1A，不训练神经网络。
            """
        ),
        markdown("## 实验设置\n\n### 1. 导入依赖并固定实验配置"),
        code(
            """
            from dataclasses import asdict, dataclass
            import hashlib
            import json
            import math

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from numpy.polynomial.hermite import hermgauss
            from scipy.special import logsumexp

            np.set_printoptions(precision=5, suppress=True)
            plt.rcParams.update({
                "figure.dpi": 110,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "font.size": 10,
                "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                "axes.unicode_minus": False,
            })

            @dataclass(frozen=True)
            class ToyConfig:
                grid_side: int = 8
                channels: int = 2
                anchor_row: int = 3
                anchor_col: int = 3
                ring_radius_raw: float = 2.0
                component_std_raw: float = 0.1
                component_count: int = 8
                design_seeds: int = 32
                heldout_seeds: int = 32
                mc_blocks: int = 256       # 256 x 16 个分层对偶样本，共 4096 个样本
                gh_order: int = 24
                seed: int = 0

                @property
                def token_count(self):
                    return self.grid_side**2

                @property
                def anchor(self):
                    return self.anchor_row * self.grid_side + self.anchor_col

                @property
                def prior_scale(self):
                    return math.sqrt(self.component_std_raw**2 + self.ring_radius_raw**2 / 2.0)

                @property
                def ring_radius(self):
                    return self.ring_radius_raw / self.prior_scale

                @property
                def component_std(self):
                    return self.component_std_raw / self.prior_scale

            CONFIG = ToyConfig()
            RF_WIDTHS = (1, 3, 5, 7, 8)
            LOGSNR_GRID = np.arange(-8.0, 8.0 + 0.25, 0.5)
            DEBUG_LOGSNR = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
            CLEAN_VARIANCE = 2.0
            CONFIG_HASH = hashlib.sha256(
                json.dumps(asdict(CONFIG), sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]

            config_display = pd.DataFrame([{**asdict(CONFIG), "config_hash": CONFIG_HASH}]).rename(columns={
                "grid_side": "网格边长",
                "channels": "通道数",
                "anchor_row": "锚点行",
                "anchor_col": "锚点列",
                "ring_radius_raw": "原始环半径",
                "component_std_raw": "原始分量标准差",
                "component_count": "高斯分量数",
                "design_seeds": "设计种子数",
                "heldout_seeds": "留出种子数",
                "mc_blocks": "MC 分块数",
                "gh_order": "Gauss-Hermite 阶数",
                "seed": "随机种子",
                "config_hash": "配置哈希",
            })
            display(config_display)
            """
        ),
        markdown("### 2. 标准化八高斯先验与精确后验"),
        code(
            r"""
            def make_component_means(config=CONFIG):
                angles = 2.0 * np.pi * np.arange(config.component_count) / config.component_count
                return config.ring_radius * np.column_stack([np.cos(angles), np.sin(angles)])

            COMPONENT_MEANS = make_component_means()
            TAU = CONFIG.component_std

            def sample_gmm(count, seed, means=COMPONENT_MEANS, tau=TAU):
                rng = np.random.default_rng(seed)
                labels = rng.integers(len(means), size=count)
                samples = means[labels] + tau * rng.standard_normal((count, 2))
                return samples.astype(np.float64), labels

            def gmm_posterior(observation, eta, means=COMPONENT_MEANS, tau=TAU):
                '''计算 u=sqrt(eta)s+eps 下的后验均值、方差迹与分量权重。'''
                observation = np.asarray(observation, dtype=np.float64)
                eta = np.broadcast_to(np.asarray(eta, dtype=np.float64), observation.shape[:-1])
                sqrt_eta = np.sqrt(np.maximum(eta, 0.0))
                observation_variance = 1.0 + eta * tau**2

                residual = (
                    observation[..., None, :]
                    - sqrt_eta[..., None, None] * means
                )
                logits = -np.square(residual).sum(axis=-1) / (
                    2.0 * observation_variance[..., None]
                )
                logits -= logsumexp(logits, axis=-1, keepdims=True)
                weights = np.exp(logits)

                posterior_variance = tau**2 / (1.0 + eta * tau**2)
                component_posterior_means = posterior_variance[..., None, None] * (
                    means / tau**2
                    + sqrt_eta[..., None, None] * observation[..., None, :]
                )
                posterior_mean = np.sum(
                    weights[..., :, None] * component_posterior_means, axis=-2
                )
                variance_trace = (
                    2.0 * posterior_variance
                    + np.sum(
                        weights * np.square(component_posterior_means).sum(axis=-1),
                        axis=-1,
                    )
                    - np.square(posterior_mean).sum(axis=-1)
                )
                return posterior_mean, np.maximum(variance_trace, 0.0), weights

            def sample_stratified_antithetic(block_count, seed, token_count=64):
                '''每个分块覆盖全部 8 个分量，并使用正负成对的残差与噪声。'''
                rng = np.random.default_rng(seed)
                sample_rows, label_rows, noise_rows, block_rows = [], [], [], []
                for block in range(block_count):
                    residual = rng.standard_normal((8, 2))
                    noise = rng.standard_normal((8, token_count, 2))
                    for sign in (1.0, -1.0):
                        sample_rows.append(COMPONENT_MEANS + sign * TAU * residual)
                        label_rows.append(np.arange(8))
                        noise_rows.append(sign * noise)
                        block_rows.append(np.full(8, block, dtype=int))
                return (
                    np.concatenate(sample_rows),
                    np.concatenate(label_rows),
                    np.concatenate(noise_rows),
                    np.concatenate(block_rows),
                )

            # 该构造具有精确标准化性质：Cov(s)=I_2。
            prior_covariance = TAU**2 * np.eye(2) + COMPONENT_MEANS.T @ COMPONENT_MEANS / 8.0
            display(pd.DataFrame(prior_covariance, index=["s1", "s2"], columns=["s1", "s2"]))
            """
        ),
        markdown("### 3. 仅作用于空间的 gauge 与二维环面感受野"),
        code(
            r"""
            def haar_orthogonal(size, rng):
                matrix = rng.standard_normal((size, size))
                q, r = np.linalg.qr(matrix)
                signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
                return q * signs

            def block_haar_gauge(block_side, seed, config=CONFIG):
                '''在互不重叠的空间块内独立进行 Haar 正交混合。'''
                side = config.grid_side
                if side % block_side:
                    raise ValueError("空间块边长必须整除网格边长")
                rng = np.random.default_rng(seed)
                gauge = np.zeros((side * side, side * side), dtype=np.float64)
                for top in range(0, side, block_side):
                    for left in range(0, side, block_side):
                        indices = [
                            (top + row) * side + left + col
                            for row in range(block_side)
                            for col in range(block_side)
                        ]
                        gauge[np.ix_(indices, indices)] = haar_orthogonal(len(indices), rng)
                return gauge

            def givens_heldout_gauge(seed, layers=12, config=CONFIG):
                '''留出变换族：交替堆叠局部水平与垂直 Givens 旋转。'''
                side = config.grid_side
                rng = np.random.default_rng(seed)
                gauge = np.eye(config.token_count, dtype=np.float64)
                for layer in range(layers):
                    horizontal = layer % 2 == 0
                    shift = int(rng.integers(2))
                    pairs = []
                    if horizontal:
                        for row in range(side):
                            for pair in range(side // 2):
                                col0 = (shift + 2 * pair) % side
                                col1 = (col0 + 1) % side
                                pairs.append((row * side + col0, row * side + col1))
                    else:
                        for col in range(side):
                            for pair in range(side // 2):
                                row0 = (shift + 2 * pair) % side
                                row1 = (row0 + 1) % side
                                pairs.append((row0 * side + col, row1 * side + col))
                    for first, second in pairs:
                        angle = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
                        cosine, sine = np.cos(angle), np.sin(angle)
                        old_first = gauge[first].copy()
                        old_second = gauge[second].copy()
                        gauge[first] = cosine * old_first + sine * old_second
                        gauge[second] = -sine * old_first + cosine * old_second
                return gauge

            def torus_rf_mask(width, config=CONFIG):
                side, token_count = config.grid_side, config.token_count
                if width >= side:
                    return np.ones((token_count, token_count), dtype=bool)
                if width < 1 or width % 2 == 0:
                    raise ValueError("有限环面感受野的宽度必须为正奇数")
                radius = width // 2
                mask = np.zeros((token_count, token_count), dtype=bool)
                for output in range(token_count):
                    row, col = divmod(output, side)
                    for delta_row in range(-radius, radius + 1):
                        for delta_col in range(-radius, radius + 1):
                            input_row = (row + delta_row) % side
                            input_col = (col + delta_col) % side
                            mask[output, input_row * side + input_col] = True
                return mask

            RF_MASKS = {width: torus_rf_mask(width) for width in RF_WIDTHS}

            def gauge_column(gauge, anchor=CONFIG.anchor):
                return gauge[:, anchor].copy()

            def captured_energy(q, width):
                return RF_MASKS[width].astype(np.float64) @ np.square(q)

            def orthogonality_error(gauge):
                identity = np.eye(gauge.shape[0])
                return np.linalg.norm(gauge.T @ gauge - identity, ord="fro") / np.sqrt(len(gauge))

            def translate_q(q, row_shift, col_shift, config=CONFIG):
                return np.roll(
                    np.roll(q.reshape(config.grid_side, config.grid_side), row_shift, axis=0),
                    col_shift,
                    axis=1,
                ).reshape(-1)

            def rotate_q(q, turns=1, config=CONFIG):
                return np.rot90(q.reshape(config.grid_side, config.grid_side), turns).reshape(-1)

            def reflect_q(q, config=CONFIG):
                return np.fliplr(q.reshape(config.grid_side, config.grid_side)).reshape(-1)
            """
        ),
        markdown("### 4. 独立计算的 Gauss-Hermite MMSE 曲线"),
        code(
            r"""
            def mmse_gmm_gauss_hermite(eta, order=CONFIG.gh_order):
                '''在 GMM 观测密度上积分 E[tr Var(s|u)]。'''
                eta = float(eta)
                nodes, weights = hermgauss(order)
                standard_nodes = np.sqrt(2.0) * nodes
                grid_x, grid_y = np.meshgrid(standard_nodes, standard_nodes, indexing="ij")
                standard_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                quadrature_weights = np.outer(weights, weights).ravel() / np.pi
                observation_std = math.sqrt(1.0 + eta * TAU**2)
                total = 0.0
                for component_mean in COMPONENT_MEANS:
                    observations = (
                        math.sqrt(eta) * component_mean
                        + observation_std * standard_grid
                    )
                    _, variance_trace, _ = gmm_posterior(observations, eta)
                    total += np.dot(quadrature_weights, variance_trace) / 8.0
                return float(total)

            MMSE_LOGETA_GRID = np.arange(-20.0, 20.0 + 0.0625, 0.125)
            MMSE_ETA_GRID = np.exp(MMSE_LOGETA_GRID)
            MMSE_VALUES = np.array([
                mmse_gmm_gauss_hermite(eta) for eta in MMSE_ETA_GRID
            ])

            def mmse_lookup(eta):
                eta = np.asarray(eta, dtype=np.float64)
                result = np.empty_like(eta)
                zero = eta <= 0.0
                result[zero] = CLEAN_VARIANCE
                positive = ~zero
                if np.any(positive):
                    result[positive] = np.interp(
                        np.log(eta[positive]),
                        MMSE_LOGETA_GRID,
                        MMSE_VALUES,
                        left=CLEAN_VARIANCE,
                        right=MMSE_VALUES[-1],
                    )
                return result

            mmse_preview = pd.DataFrame({
                "logsnr": DEBUG_LOGSNR,
                "gmm_mmse": mmse_lookup(np.exp(DEBUG_LOGSNR)),
                "matched_gaussian_mmse": 2.0 / (1.0 + np.exp(DEBUG_LOGSNR)),
            })
            display(mmse_preview.rename(columns={
                "logsnr": "log-SNR",
                "gmm_mmse": "八高斯 MMSE",
                "matched_gaussian_mmse": "匹配高斯 MMSE",
            }))
            """
        ),
        markdown("### 5. 捕获能量理论与显式 128 维 latent Monte Carlo"),
        code(
            r"""
            def theory_risk(q, width, gamma):
                q_squared = np.square(q)
                energy = captured_energy(q, width)
                gamma = np.asarray(gamma, dtype=np.float64)
                local = np.sum(
                    q_squared * mmse_lookup(gamma[..., None] * energy), axis=-1
                )
                global_risk = mmse_lookup(gamma)
                return local, global_risk, local - global_risk

            def block_mean_and_se(values, block_ids):
                block_count = int(block_ids.max()) + 1
                block_means = np.array([
                    np.mean(values[block_ids == block]) for block in range(block_count)
                ])
                return float(block_means.mean()), float(block_means.std(ddof=1) / np.sqrt(block_count))

            def full_latent_mc(gauge, width, gamma, samples, base_noise, block_ids):
                '''路径 A：显式构造 [样本, 64 个 token, 2 个通道]。'''
                q = gauge_column(gauge)
                clean = q[None, :, None] * samples[:, None, :]
                gauge_noise = np.einsum("ij,mjc->mic", gauge, base_noise, optimize=True)
                noisy = math.sqrt(gamma) * clean + gauge_noise

                active = np.flatnonzero(np.square(q) > 1e-15)
                local_weights = RF_MASKS[width][active].astype(np.float64) * q[None, :]
                energy = np.square(local_weights).sum(axis=1)
                sufficient = np.einsum(
                    "ij,mjc->mic", local_weights, noisy, optimize=True
                ) / np.sqrt(energy)[None, :, None]
                local_posterior, _, _ = gmm_posterior(
                    sufficient, gamma * energy[None, :]
                )
                local_losses = np.sum(
                    np.square(q[active])[None, :]
                    * np.square(samples[:, None, :] - local_posterior).sum(axis=-1),
                    axis=1,
                )
                projected_prediction = np.sum(
                    np.square(q[active])[None, :, None] * local_posterior,
                    axis=1,
                )

                global_sufficient = np.einsum("i,mic->mc", q, noisy, optimize=True)
                global_posterior, _, _ = gmm_posterior(global_sufficient, gamma)
                global_losses = np.square(samples - global_posterior).sum(axis=-1)
                tax_losses = local_losses - global_losses

                local_mean, local_se = block_mean_and_se(local_losses, block_ids)
                global_mean, global_se = block_mean_and_se(global_losses, block_ids)
                tax_mean, tax_se = block_mean_and_se(tax_losses, block_ids)
                return {
                    "local_risk": local_mean,
                    "local_se": local_se,
                    "global_risk": global_mean,
                    "global_se": global_se,
                    "tax": tax_mean,
                    "tax_se": tax_se,
                    "projected_prediction": projected_prediction,
                    "global_prediction": global_posterior,
                    "noisy": noisy,
                }

            def oracle_rescue(gauge, gamma, samples, base_noise):
                '''执行 G^T -> 恒等坐标局部去噪器 -> G，并返回公共 s 空间中的预测。'''
                q = gauge_column(gauge)
                clean = q[None, :, None] * samples[:, None, :]
                gauge_noise = np.einsum("ij,mjc->mic", gauge, base_noise, optimize=True)
                noisy = math.sqrt(gamma) * clean + gauge_noise
                unmixed = np.einsum("ij,mic->mjc", gauge, noisy, optimize=True)
                anchor_observation = unmixed[:, CONFIG.anchor, :]
                prediction, _, _ = gmm_posterior(anchor_observation, gamma)
                return prediction, unmixed

            def integrated_uniform_t_score(q, width, point_count=256):
                t = (np.arange(point_count) + 0.5) / point_count
                gamma = np.square(np.cos(0.5 * np.pi * t) / np.sin(0.5 * np.pi * t))
                local, global_risk, tax = theory_risk(q, width, gamma)
                return float(np.mean(tax) / CLEAN_VARIANCE), float(np.sum(np.square(q) * captured_energy(q, width)))
            """
        ),
        markdown("### 6. 构造开发集 gauge 与固定粒度的理论表"),
        code(
            r"""
            def make_designs():
                designs = [{
                    "gauge_family": "identity",
                    "design_seed": 0,
                    "gauge_seed": 0,
                    "mixing_support": 1,
                    "anchor_token": CONFIG.anchor,
                    "gauge": np.eye(CONFIG.token_count),
                }]
                for block_side in (2, 4, 8):
                    for design_seed in range(CONFIG.design_seeds):
                        gauge_seed = 10_000 * block_side + design_seed
                        designs.append({
                            "gauge_family": f"block_{block_side}x{block_side}",
                            "design_seed": design_seed,
                            "gauge_seed": gauge_seed,
                            "mixing_support": block_side**2,
                            "anchor_token": CONFIG.anchor,
                            "gauge": block_haar_gauge(block_side, gauge_seed),
                        })
                return designs

            DEVELOPMENT_DESIGNS = make_designs()

            theory_rows = []
            for design in DEVELOPMENT_DESIGNS:
                gauge = design["gauge"]
                q = gauge_column(gauge, design["anchor_token"])
                for width in RF_WIDTHS:
                    energy = captured_energy(q, width)
                    captured_summary = float(np.sum(np.square(q) * energy))
                    for logsnr in LOGSNR_GRID:
                        gamma = math.exp(logsnr)
                        local, global_risk, tax = theory_risk(q, width, gamma)
                        theory_rows.append({
                            "prior": "standardized_8gmm",
                            "design_seed": design["design_seed"],
                            "gauge_family": design["gauge_family"],
                            "gauge_seed": design["gauge_seed"],
                            "mixing_support": design["mixing_support"],
                            "anchor_token": design["anchor_token"],
                            "rf": width,
                            "topology": "2d_torus",
                            "logsnr": logsnr,
                            "mc_seed": np.nan,
                            "sample_count": 0,
                            "global_risk": float(global_risk),
                            "local_risk_mc": np.nan,
                            "local_risk_theory": float(local),
                            "tax_mc": np.nan,
                            "tax_theory": float(tax),
                            "clean_variance": CLEAN_VARIANCE,
                            "normalized_tax": float(tax / CLEAN_VARIANCE),
                            "mc_se": np.nan,
                            "captured_energy_summary": captured_summary,
                            "orthogonality_error": orthogonality_error(gauge),
                            "config_hash": CONFIG_HASH,
                        })
            theory_table = pd.DataFrame(theory_rows)
            print("理论表行数：", len(theory_table))
            THEORY_COLUMNS_CN = {
                "prior": "先验",
                "design_seed": "设计种子",
                "gauge_family": "gauge 族",
                "gauge_seed": "gauge 种子",
                "mixing_support": "混合支撑大小",
                "anchor_token": "锚点 token",
                "rf": "感受野宽度",
                "topology": "拓扑",
                "logsnr": "log-SNR",
                "mc_seed": "MC 种子",
                "sample_count": "样本数",
                "global_risk": "全局风险",
                "local_risk_mc": "局部风险（MC）",
                "local_risk_theory": "局部风险（理论）",
                "tax_mc": "局部性代价（MC）",
                "tax_theory": "局部性代价（理论）",
                "clean_variance": "干净数据方差",
                "normalized_tax": "归一化局部性代价",
                "mc_se": "MC 标准误",
                "captured_energy_summary": "加权捕获能量",
                "orthogonality_error": "正交性误差",
                "config_hash": "配置哈希",
            }
            DISPLAY_VALUES_CN = {
                "standardized_8gmm": "标准化八高斯",
                "identity": "恒等坐标",
                "block_2x2": "2x2 分块 Haar",
                "block_4x4": "4x4 分块 Haar",
                "block_8x8": "8x8 分块 Haar",
                "local_givens_12layer": "12 层局部 Givens",
                "2d_torus": "二维环面",
                "development": "开发集",
                "heldout_givens": "留出 Givens 集",
            }

            def chinese_display_table(table, column_names):
                '''只翻译展示副本，不改变后续计算使用的内部表。'''
                return table.replace(DISPLAY_VALUES_CN).rename(columns=column_names)

            display(chinese_display_table(theory_table.head(8), THEORY_COLUMNS_CN))
            """
        ),
        markdown(
            r"""
            ## 数值检查

            ### 7. 必需的数值、后验、拓扑、oracle 与路径独立性检查

            这些检查被有意保留在 notebook 内部。只要任一不变量失败，执行就会在绘图之前停止，
            避免解读建立在错误计算上的图像。
            """
        ),
        code(
            r"""
            check_rows = []
            def record_check(name, value, threshold, passed):
                check_rows.append({"check": name, "value": float(value), "threshold": threshold, "passed": bool(passed)})
                assert passed, f"检查失败：{name}，数值={value}"

            # 检查正交性、q 的范数、捕获能量范围以及感受野嵌套关系。
            max_orthogonality = max(orthogonality_error(d["gauge"]) for d in DEVELOPMENT_DESIGNS)
            record_check("最大归一化正交性误差", max_orthogonality, "<1e-12", max_orthogonality < 1e-12)
            q_test = gauge_column(block_haar_gauge(8, 80_007))
            record_check("|q|^2 误差", abs(np.dot(q_test, q_test) - 1.0), "<1e-12", abs(np.dot(q_test, q_test) - 1.0) < 1e-12)
            energies = np.stack([captured_energy(q_test, width) for width in RF_WIDTHS])
            record_check("捕获能量下界违反量", max(0.0, -energies.min()), "0", energies.min() >= -1e-12)
            record_check("捕获能量上界违反量", max(0.0, energies.max() - 1.0), "0", energies.max() <= 1.0 + 1e-12)
            record_check("感受野嵌套违反量", max(0.0, -np.diff(energies, axis=0).min()), "0", np.diff(energies, axis=0).min() >= -1e-12)
            torus_count_error = max(
                np.max(np.abs(RF_MASKS[width].sum(axis=1) - width**2))
                for width in RF_WIDTHS[:-1]
            )
            record_check("环面感受野 token 数误差", torus_count_error, "0", torus_count_error == 0)

            # 恒等变换与全局感受野的理论局部性代价必须精确为零。
            identity_rows = theory_table.query("gauge_family == 'identity'")
            identity_max = identity_rows["tax_theory"].abs().max()
            global_max = theory_table.query("rf == 8")["tax_theory"].abs().max()
            record_check("恒等变换最大局部性代价", identity_max, "<1e-10", identity_max < 1e-10)
            record_check("全局感受野最大局部性代价", global_max, "<1e-10", global_max < 1e-10)

            # 随着感受野增大，精确风险不应上升。
            monotonic_violation = 0.0
            for _, group in theory_table.groupby(["gauge_family", "design_seed", "logsnr"]):
                risks = group.sort_values("rf")["local_risk_theory"].to_numpy()
                monotonic_violation = max(monotonic_violation, float(np.diff(risks).max(initial=-np.inf)))
            record_check("感受野风险单调性违反量", max(monotonic_violation, 0.0), "<1e-10", monotonic_violation < 1e-10)

            # 对完全随机的 gauge，局部性代价在低/高 SNR 两端都应趋于零。
            endpoint_gamma = np.exp(np.array([-20.0, 20.0]))
            _, _, endpoint_tax = theory_risk(q_test, 1, endpoint_gamma)
            endpoint_max = np.max(np.abs(endpoint_tax)) / CLEAN_VARIANCE
            record_check("log-SNR 端点归一化局部性代价", endpoint_max, "<2e-4", endpoint_max < 2e-4)

            # 检查后验权重、eta=0、高 eta，以及分量重合时的高斯恒等关系。
            posterior_samples, _ = sample_gmm(512, 91)
            posterior_noise = np.random.default_rng(92).standard_normal((512, 2))
            eta_probe = np.exp(np.linspace(-6.0, 6.0, 512))
            observations = np.sqrt(eta_probe)[:, None] * posterior_samples + posterior_noise
            posterior_mean, posterior_var, posterior_weights = gmm_posterior(observations, eta_probe)
            record_check("后验权重和误差", np.max(np.abs(posterior_weights.sum(axis=1) - 1.0)), "<1e-12", np.max(np.abs(posterior_weights.sum(axis=1) - 1.0)) < 1e-12)
            finite_ok = np.isfinite(posterior_mean).all() and np.isfinite(posterior_var).all()
            record_check("后验非有限值计数", 0 if finite_ok else 1, "0", finite_ok)
            zero_mean, _, _ = gmm_posterior(np.random.default_rng(93).standard_normal((32, 2)), 0.0)
            record_check("eta=0 时后验均值范数", np.max(np.abs(zero_mean)), "<1e-12", np.max(np.abs(zero_mean)) < 1e-12)
            eta_high = 1e7
            high_obs = math.sqrt(eta_high) * posterior_samples + posterior_noise
            high_mean, _, _ = gmm_posterior(high_obs, eta_high)
            high_mse = np.mean(np.square(high_mean - posterior_samples))
            record_check("高 SNR 后验 MSE", high_mse, "<1e-5", high_mse < 1e-5)
            zero_components = np.zeros((8, 2))
            tau_probe = 0.7
            u_probe = np.random.default_rng(94).standard_normal((64, 2))
            eta_scalar = 2.5
            collapsed_mean, _, _ = gmm_posterior(u_probe, eta_scalar, means=zero_components, tau=tau_probe)
            gaussian_mean = tau_probe**2 * math.sqrt(eta_scalar) / (1.0 + eta_scalar * tau_probe**2) * u_probe
            record_check("重合 GMM 与高斯后验误差", np.max(np.abs(collapsed_mean - gaussian_mean)), "<1e-12", np.max(np.abs(collapsed_mean - gaussian_mean)) < 1e-12)

            # 在三个预先固定的 SNR 上比较 Gauss-Hermite MMSE 与直接 GMM Monte Carlo。
            validation_s, _, validation_noise, validation_blocks = sample_stratified_antithetic(256, 95, token_count=1)
            gmm_zscores = []
            for logsnr in (-3.0, 0.0, 3.0):
                eta = math.exp(logsnr)
                u = math.sqrt(eta) * validation_s + validation_noise[:, 0, :]
                estimate, _, _ = gmm_posterior(u, eta)
                losses = np.square(validation_s - estimate).sum(axis=1)
                mean, se = block_mean_and_se(losses, validation_blocks)
                gmm_zscores.append(abs(mean - float(mmse_lookup(eta))) / se)
            record_check("GH 与 GMM-MC 的最大 z 分数", max(gmm_zscores), "<3", max(gmm_zscores) < 3.0)

            # 比较匹配高斯 Monte Carlo 与解析结果 2/(1+eta)。
            rng_gaussian = np.random.default_rng(96)
            gaussian_s = rng_gaussian.standard_normal((131_072, 2))
            gaussian_noise = rng_gaussian.standard_normal((131_072, 2))
            gaussian_zscores = []
            for logsnr in (-3.0, 0.0, 3.0):
                eta = math.exp(logsnr)
                u = math.sqrt(eta) * gaussian_s + gaussian_noise
                estimate = math.sqrt(eta) / (1.0 + eta) * u
                losses = np.square(gaussian_s - estimate).sum(axis=1)
                mean, se = losses.mean(), losses.std(ddof=1) / np.sqrt(len(losses))
                gaussian_zscores.append(abs(mean - 2.0 / (1.0 + eta)) / se)
            record_check("高斯解析结果与 MC 的最大 z 分数", max(gaussian_zscores), "<3", max(gaussian_zscores) < 3.0)

            # 环面平移、旋转/反射以及通道旋转均作为负对照。
            gamma_control = math.exp(0.7)
            base_control = np.array([theory_risk(q_test, width, gamma_control)[2] for width in RF_WIDTHS])
            transformed_controls = [
                translate_q(q_test, 2, 3), rotate_q(q_test), reflect_q(q_test)
            ]
            control_error = max(
                np.max(np.abs(np.array([theory_risk(q, width, gamma_control)[2] for width in RF_WIDTHS]) - base_control))
                for q in transformed_controls
            )
            record_check("空间对称负对照的局部性代价误差", control_error, "<1e-10", control_error < 1e-10)
            channel_rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
            record_check("通道旋转正交性误差", np.linalg.norm(channel_rotation.T @ channel_rotation - np.eye(2)), "<1e-12", np.linalg.norm(channel_rotation.T @ channel_rotation - np.eye(2)) < 1e-12)
            channel_u = np.random.default_rng(941).standard_normal((256, 2))
            channel_base, _, _ = gmm_posterior(channel_u, gamma_control)
            channel_rotated, _, _ = gmm_posterior(channel_u @ channel_rotation.T, gamma_control)
            channel_equivariance_error = np.max(
                np.abs(channel_rotated - channel_base @ channel_rotation.T)
            )
            record_check("通道旋转的后验等变误差", channel_equivariance_error, "<1e-12", channel_equivariance_error < 1e-12)

            # 检查 gauge 的逐比特可复现性。
            repeated_a = block_haar_gauge(4, 12345)
            repeated_b = block_haar_gauge(4, 12345)
            record_check("同种子可复现性不匹配计数", int(not np.array_equal(repeated_a, repeated_b)), "0", np.array_equal(repeated_a, repeated_b))

            # 检查成对的完整 latent 路径、oracle 解混与全局等变性。
            paired_s, _, paired_noise, paired_blocks = sample_stratified_antithetic(128, 97, CONFIG.token_count)
            paired_gauge = block_haar_gauge(8, 80_003)
            paired_result = full_latent_mc(paired_gauge, 3, 1.0, paired_s, paired_noise, paired_blocks)
            q_paired = gauge_column(paired_gauge)
            local_theory, global_theory, tax_theory = theory_risk(q_paired, 3, 1.0)
            tax_z = abs(paired_result["tax"] - tax_theory) / paired_result["tax_se"]
            record_check("完整 latent MC 与捕获能量理论的 z 分数", tax_z, "<3", tax_z < 3.0)
            identity_result = full_latent_mc(np.eye(CONFIG.token_count), 1, 1.0, paired_s, paired_noise, paired_blocks)
            oracle_prediction, unmixed = oracle_rescue(paired_gauge, 1.0, paired_s, paired_noise)
            oracle_error = np.max(np.abs(oracle_prediction - identity_result["projected_prediction"]))
            record_check("oracle 解混与恒等坐标预测误差", oracle_error, "<1e-12", oracle_error < 1e-12)
            global_prediction_error = np.max(np.abs(paired_result["global_prediction"] - identity_result["global_prediction"]))
            record_check("成对全局预测的等变误差", global_prediction_error, "<1e-12", global_prediction_error < 1e-12)
            gauge_global_latent = q_paired[None, :, None] * paired_result["global_prediction"][:, None, :]
            identity_global_latent = np.zeros_like(gauge_global_latent)
            identity_global_latent[:, CONFIG.anchor, :] = identity_result["global_prediction"]
            transformed_identity_global = np.einsum(
                "ij,mjc->mic", paired_gauge, identity_global_latent, optimize=True
            )
            full_map_error = np.max(np.abs(gauge_global_latent - transformed_identity_global))
            record_check("成对完整 latent 全局映射的等变误差", full_map_error, "<1e-12", full_map_error < 1e-12)

            checks = pd.DataFrame(check_rows)
            display(
                checks.rename(columns={
                    "check": "检查项",
                    "value": "数值",
                    "threshold": "阈值",
                    "passed": "通过",
                }).style.format({"数值": "{:.3e}"})
            )
            print(f"全部 {len(checks)} 项 notebook 检查均已通过。")
            """
        ),
        markdown(
            """
            ## 实验结果

            ### 8. 图 1：恒等坐标、埋藏 gauge 与 oracle 解混

            所有子图都在预先固定的 log-SNR 0 下使用相同的标准化 GMM 样本和成对基础噪声。
            埋藏坐标中的预测器通过指定感受野直接读取变换后的 latent。Oracle 显式应用 `A.T`，
            在恒等坐标中执行局部去噪，再应用 `A` 映射回来；这是因果检查，而不是提出的新方法。
            """
        ),
        code(
            r"""
            figure_s, figure_labels, figure_noise, figure_blocks = sample_stratified_antithetic(
                75, 201, CONFIG.token_count
            )
            figure_gamma = 1.0
            identity_gauge = np.eye(CONFIG.token_count)
            buried_gauge = block_haar_gauge(8, 80_011)
            buried_q = gauge_column(buried_gauge)
            oracle_prediction, _ = oracle_rescue(
                buried_gauge, figure_gamma, figure_s, figure_noise
            )

            fig, axes = plt.subplots(4, 3, figsize=(10.5, 13), sharex=True, sharey=True)
            column_titles = ("恒等坐标", "8x8 埋藏 gauge", "oracle A.T 解混")
            for column, title in enumerate(column_titles):
                axes[0, column].set_title(title)
            for row, width in enumerate((1, 3, 5, 8)):
                identity_output = full_latent_mc(
                    identity_gauge, width, figure_gamma, figure_s, figure_noise, figure_blocks
                )
                buried_output = full_latent_mc(
                    buried_gauge, width, figure_gamma, figure_s, figure_noise, figure_blocks
                )
                predictions = (
                    identity_output["projected_prediction"],
                    buried_output["projected_prediction"],
                    oracle_prediction,
                )
                taxes = (
                    0.0,
                    float(theory_risk(buried_q, width, figure_gamma)[2] / CLEAN_VARIANCE),
                    0.0,
                )
                for column, (prediction, tax) in enumerate(zip(predictions, taxes)):
                    axis = axes[row, column]
                    axis.scatter(
                        prediction[:, 0], prediction[:, 1], c=figure_labels,
                        cmap="tab10", s=7, alpha=0.5, rasterized=True,
                    )
                    axis.scatter(
                        COMPONENT_MEANS[:, 0], COMPONENT_MEANS[:, 1], marker="x",
                        c="#27313d", s=22, linewidths=0.9,
                    )
                    axis.set_aspect("equal")
                    axis.set_xlim(-2.2, 2.2)
                    axis.set_ylim(-2.2, 2.2)
                    axis.grid(color="#e5e7eb", linewidth=0.6)
                    axis.text(
                        0.04, 0.95, f"精确代价={100*tax:.2f}%",
                        transform=axis.transAxes, va="top",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                    if row == 3:
                        axis.set_xlabel("s1")
                    if column == 0:
                        label = "全局 8x8" if width == 8 else f"{width}x{width} 感受野"
                        axis.set_ylabel(f"{label}\ns2")
            fig.suptitle("图 1 | 空间埋藏造成局部困难，oracle 解混将其消除")
            fig.tight_layout()
            """
        ),
        markdown(
            """
            ### 9. 图 2：混合支撑大小 × 感受野相图

            分数在预先固定的均匀 $t$ 网格上取平均。每个非平凡块大小的随机 Haar 矩阵均使用
            32 个独立设计种子。右侧子图报告 $\sum_iq_i^2e_i$，即由输出能量加权的感受野
            捕获比例。
            """
        ),
        code(
            r"""
            integrated_rows = []
            for design in DEVELOPMENT_DESIGNS:
                q = gauge_column(design["gauge"])
                for width in RF_WIDTHS:
                    normalized_tax, captured_summary = integrated_uniform_t_score(q, width)
                    integrated_rows.append({
                        "gauge_family": design["gauge_family"],
                        "design_seed": design["design_seed"],
                        "mixing_support": design["mixing_support"],
                        "rf": width,
                        "integrated_normalized_tax": normalized_tax,
                        "captured_energy": captured_summary,
                    })
            integrated_table = pd.DataFrame(integrated_rows)
            phase = integrated_table.groupby(["mixing_support", "rf"], as_index=False).agg(
                tax_mean=("integrated_normalized_tax", "mean"),
                tax_std=("integrated_normalized_tax", "std"),
                captured_mean=("captured_energy", "mean"),
            )

            supports = [1, 4, 16, 64]
            tax_matrix = phase.pivot(index="rf", columns="mixing_support", values="tax_mean").reindex(index=RF_WIDTHS, columns=supports)
            captured_matrix = phase.pivot(index="rf", columns="mixing_support", values="captured_mean").reindex(index=RF_WIDTHS, columns=supports)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            tax_image = axes[0].imshow(100.0 * tax_matrix, origin="lower", aspect="auto", cmap="magma")
            captured_image = axes[1].imshow(captured_matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            for axis, title in zip(axes, ("均匀 t 积分的局部性代价（占干净方差 %）", "加权捕获的 gauge 能量")):
                axis.set_xticks(range(len(supports)), supports)
                axis.set_yticks(range(len(RF_WIDTHS)), ["全局" if rf == 8 else f"{rf}x{rf}" for rf in RF_WIDTHS])
                axis.set_xlabel("混合支撑大小（token 数）")
                axis.set_ylabel("环面感受野")
                axis.set_title(title)
            fig.colorbar(tax_image, ax=axes[0], fraction=0.046, pad=0.04)
            fig.colorbar(captured_image, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle("图 2 | 相图由感受野捕获的 gauge 能量控制")
            fig.tight_layout()
            display(phase.rename(columns={
                "mixing_support": "混合支撑大小",
                "rf": "感受野宽度",
                "tax_mean": "局部性代价均值",
                "tax_std": "局部性代价标准差",
                "captured_mean": "捕获能量均值",
            }).round(6))
            """
        ),
        markdown(
            """
            ### 10. 图 3：完整 log-SNR 曲线

            曲线在 32 个设计种子上取平均。绘制的指标为
            $\Delta R/\mathbb{E}\|s\|^2$，因此当全局 MMSE 趋近于零时仍保持良好数值性质。
            对所有 log-SNR，全局感受野与 oracle 解混都位于零线上。
            """
        ),
        code(
            r"""
            curve_summary = (
                theory_table.groupby(["mixing_support", "rf", "logsnr"], as_index=False)
                .agg(mean_tax=("normalized_tax", "mean"), std_tax=("normalized_tax", "std"))
            )
            supports = [1, 4, 16, 64]
            rf_colors = {1: "#b91c1c", 3: "#e85d04", 5: "#2563eb", 7: "#0891b2", 8: "#27313d"}
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True)
            for axis, support in zip(axes, supports):
                for width in RF_WIDTHS:
                    rows = curve_summary.query("mixing_support == @support and rf == @width").sort_values("logsnr")
                    label = "全局/oracle 解混" if width == 8 else f"{width}x{width}"
                    axis.plot(rows["logsnr"], 100.0 * rows["mean_tax"], color=rf_colors[width], label=label, linewidth=2)
                    if rows["std_tax"].notna().any():
                        std = rows["std_tax"].fillna(0.0)
                        axis.fill_between(rows["logsnr"], 100.0 * (rows["mean_tax"] - std), 100.0 * (rows["mean_tax"] + std), color=rf_colors[width], alpha=0.10)
                axis.axhline(0.0, color="#27313d", linestyle="--", linewidth=0.8)
                axis.set_title(f"混合支撑大小={support}")
                axis.set_xlabel("log-SNR")
                axis.grid(color="#e5e7eb", linewidth=0.6)
            axes[0].set_ylabel("局部性代价 / 干净方差（%）")
            axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
            fig.suptitle("图 3 | 局部性代价在 SNR 两端消失，并在中间区域达到峰值")
            fig.tight_layout()
            """
        ),
        markdown(
            """
            ### 11. 构造独立仿真验证集

            开发集使用三个 block-Haar 变换族，每族包含 32 个变换种子；留出集使用 32 个多层局部
            Givens 变换。每个变换作为一个统计单位，并分配一个预先指定的感受野/log-SNR 条件。
            所有变换共享同一组分层对偶 GMM 样本和成对基础噪声。
            """
        ),
        code(
            r"""
            validation_s, validation_labels, validation_base_noise, validation_block_ids = sample_stratified_antithetic(
                CONFIG.mc_blocks, 301, CONFIG.token_count
            )
            validation_designs = []
            condition_rfs = (1, 3, 5, 7)
            condition_logsnr = (-2.0, 0.0, 2.0)
            for block_side in (2, 4, 8):
                for design_seed in range(CONFIG.design_seeds):
                    validation_designs.append({
                        "split": "development",
                        "gauge_family": f"block_{block_side}x{block_side}",
                        "design_seed": design_seed,
                        "gauge_seed": 20_000 * block_side + design_seed,
                        "gauge": block_haar_gauge(block_side, 20_000 * block_side + design_seed),
                        "rf": condition_rfs[design_seed % len(condition_rfs)],
                        "logsnr": condition_logsnr[(design_seed // len(condition_rfs)) % len(condition_logsnr)],
                        "mixing_support": block_side**2,
                    })
            for design_seed in range(CONFIG.heldout_seeds):
                validation_designs.append({
                    "split": "heldout_givens",
                    "gauge_family": "local_givens_12layer",
                    "design_seed": design_seed,
                    "gauge_seed": 90_000 + design_seed,
                    "gauge": givens_heldout_gauge(90_000 + design_seed),
                    "rf": condition_rfs[design_seed % len(condition_rfs)],
                    "logsnr": condition_logsnr[(design_seed // len(condition_rfs)) % len(condition_logsnr)],
                    "mixing_support": np.nan,
                })

            validation_rows = []
            for design in validation_designs:
                gamma = math.exp(design["logsnr"])
                gauge, width = design["gauge"], design["rf"]
                q = gauge_column(gauge)
                simulation = full_latent_mc(
                    gauge, width, gamma,
                    validation_s, validation_base_noise, validation_block_ids,
                )
                local_theory, global_theory, tax_theory = theory_risk(q, width, gamma)
                support = int((np.square(q) > 1e-14).sum())
                validation_rows.append({
                    "prior": "standardized_8gmm",
                    "split": design["split"],
                    "design_seed": design["design_seed"],
                    "gauge_family": design["gauge_family"],
                    "gauge_seed": design["gauge_seed"],
                    "mixing_support": support,
                    "anchor_token": CONFIG.anchor,
                    "rf": width,
                    "topology": "2d_torus",
                    "logsnr": design["logsnr"],
                    "mc_seed": 301,
                    "sample_count": len(validation_s),
                    "global_risk": float(global_theory),
                    "global_risk_mc": simulation["global_risk"],
                    "local_risk_mc": simulation["local_risk"],
                    "local_risk_theory": float(local_theory),
                    "tax_mc_raw": simulation["tax"],
                    "tax_mc_raw_se": simulation["tax_se"],
                    "tax_mc": simulation["local_risk"] - float(global_theory),
                    "tax_theory": float(tax_theory),
                    "clean_variance": CLEAN_VARIANCE,
                    "normalized_tax": float(tax_theory / CLEAN_VARIANCE),
                    "mc_se": simulation["local_se"],
                    "captured_energy_summary": float(np.sum(np.square(q) * captured_energy(q, width))),
                    "orthogonality_error": orthogonality_error(gauge),
                    "config_hash": CONFIG_HASH,
                })
            validation_table = pd.DataFrame(validation_rows)
            validation_table["z_error"] = (
                (validation_table["tax_mc"] - validation_table["tax_theory"]).abs()
                / validation_table["mc_se"]
            )
            validation_table["relative_local_risk_error"] = (
                (validation_table["local_risk_mc"] - validation_table["local_risk_theory"]).abs()
                / validation_table["local_risk_theory"]
            )

            def bootstrap_mean_ci(values, seed=0, draws=2_000):
                values = np.asarray(values, dtype=np.float64)
                rng = np.random.default_rng(seed)
                means = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
                return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

            heldout = validation_table.query("split == 'heldout_givens'")
            heldout_error, heldout_low, heldout_high = bootstrap_mean_ci(
                heldout["relative_local_risk_error"], seed=302
            )
            validation_summary = pd.DataFrame([{
                "heldout_design_seeds": len(heldout),
                "heldout_mean_relative_local_risk_error": heldout_error,
                "bootstrap_ci95_low": heldout_low,
                "bootstrap_ci95_high": heldout_high,
                "fraction_all_points_within_3se": float((validation_table["z_error"] <= 3.0).mean()),
                "max_normalized_absolute_tax_error": float(
                    ((validation_table["tax_mc"] - validation_table["tax_theory"]).abs() / CLEAN_VARIANCE).max()
                ),
            }])
            assert len(heldout) == 32
            assert (validation_table["z_error"] <= 3.0).all(), "存在理论/MC 验证点超过 3 个标准误"
            assert validation_summary.loc[0, "max_normalized_absolute_tax_error"] < 0.01
            validation_summary_display = validation_summary.rename(columns={
                "heldout_design_seeds": "留出设计种子数",
                "heldout_mean_relative_local_risk_error": "留出集局部风险平均相对误差",
                "bootstrap_ci95_low": "bootstrap 95% CI 下界",
                "bootstrap_ci95_high": "bootstrap 95% CI 上界",
                "fraction_all_points_within_3se": "落在 3SE 内的验证点比例",
                "max_normalized_absolute_tax_error": "最大归一化绝对代价误差",
            })
            display(validation_summary_display.style.format({
                "留出集局部风险平均相对误差": "{:.3%}",
                "bootstrap 95% CI 下界": "{:.3%}",
                "bootstrap 95% CI 上界": "{:.3%}",
                "落在 3SE 内的验证点比例": "{:.3%}",
                "最大归一化绝对代价误差": "{:.3%}",
            }))
            """
        ),
        markdown(
            r"""
            ### 12. 图 4：捕获能量理论与显式 latent 仿真对比

            横轴数值只来自捕获能量公式与独立 Gauss-Hermite 积分得到的 MMSE 曲线；纵轴数值则显式
            构造全部 $64\times2=128$ 个带噪 latent 坐标。为避免在每个成对设计中重复使用同一份
            全局 MC 噪声，图中的仿真局部性代价定义为显式局部 MC 减去经独立 Gauss-Hermite 验证的
            全局 MMSE；原始的 `局部 MC - 全局 MC` 仍保留在内部结果表中。误差棒表示三个分块级
            局部风险标准误。留出集置信区间以完整的 Givens 变换种子为单位重采样。
            """
        ),
        code(
            r"""
            fig, axis = plt.subplots(figsize=(7.2, 6.2))
            styles = {
                "development": {"color": "#2563eb", "marker": "o", "label": "开发集 block-Haar"},
                "heldout_givens": {"color": "#e85d04", "marker": "^", "label": "留出集局部 Givens"},
            }
            for split, style in styles.items():
                rows = validation_table.query("split == @split")
                axis.errorbar(
                    100.0 * rows["tax_theory"] / CLEAN_VARIANCE,
                    100.0 * rows["tax_mc"] / CLEAN_VARIANCE,
                    yerr=300.0 * rows["mc_se"] / CLEAN_VARIANCE,
                    linestyle="none", marker=style["marker"], color=style["color"],
                    alpha=0.65, markersize=5, capsize=1.5, label=style["label"],
                )
            limits = np.array([
                min(0.0, 100.0 * validation_table[["tax_theory", "tax_mc"]].min().min() / CLEAN_VARIANCE),
                100.0 * validation_table[["tax_theory", "tax_mc"]].max().max() / CLEAN_VARIANCE,
            ])
            padding = 0.05 * max(limits[1] - limits[0], 1.0)
            limits += np.array([-padding, padding])
            axis.plot(limits, limits, color="#27313d", linestyle="--", linewidth=1.2, label="y=x")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("捕获能量理论代价 / 干净方差（%）")
            axis.set_ylabel("显式 128 维 latent MC 代价 / 干净方差（%）")
            axis.grid(color="#e5e7eb", linewidth=0.7)
            axis.legend(frameon=False)
            axis.set_title(
                "图 4 | 理论能够预测独立仿真的留出 gauge 变换族\n"
                f"留出集局部风险平均误差={heldout_error:.2%} "
                f"（按设计 bootstrap 的 95% CI：{heldout_low:.2%}-{heldout_high:.2%}）"
            )
            fig.tight_layout()

            VALIDATION_COLUMNS_CN = {
                **THEORY_COLUMNS_CN,
                "split": "数据划分",
                "global_risk_mc": "全局风险（MC）",
                "tax_mc_raw": "原始局部性代价（MC）",
                "tax_mc_raw_se": "原始局部性代价标准误",
                "z_error": "z 误差",
                "relative_local_risk_error": "局部风险相对误差",
            }
            display(
                chinese_display_table(
                    validation_table.sort_values("z_error", ascending=False).head(12),
                    VALIDATION_COLUMNS_CN,
                ).round(6)
            )
            """
        ),
        markdown(
            r"""
            ## 结论

            1. **修正后的 token 构造消除了旧的混杂因素。** $s_1$ 与 $s_2$ 都从同一个 token 出发，
               因此恒等坐标加 $1\times1$ 感受野已经足够。
            2. **其机制就是精确的局部 SNR 损失。** 一个窗口捕获能量 $e_i$，等价于信噪比
               $\gamma e_i$ 下的 AWGN 观测。
            3. **因果对照全部成立。** 恒等坐标、全局感受野、环面对称、通道旋转，以及显式
               $G^\top$ oracle 解混都会回到零代价基线。
            4. **SNR 依赖并不平凡，但能被理论预测。** 局部性代价在纯噪声端和干净数据端消失，
               在中间 SNR 区域形成峰值；感受野越大，峰值越低。
            5. **该公式不是代码对自身的重复表述。** 全部 128 个显式 128 维 latent Monte Carlo
               条件均落在 3 个标准误内；在 32 个留出的 Givens 变换上，局部风险平均误差为
               0.745%（按设计 bootstrap 的 95% CI 为 0.402%-1.157%）。
            6. **当前结论仍只针对局部架构。** 通过阶段 1A 可以支持后续构造可学习的 masked-
               attention 玩具实验，但尚不能解释全局 DiT 中残留的 gauge 敏感性。
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    build_notebook()

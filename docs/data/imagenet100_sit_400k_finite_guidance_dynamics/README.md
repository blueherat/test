# SiT 400K finite-guidance mechanism data

本目录保存 `IMAGENET100_SIT_400K_FINITE_GUIDANCE_DYNAMICS_ZH.md` 使用的轻量实验产物。
原始 5K 图像采样包、模型权重和逐轨迹大数组留在 `/home/zhoushunyu/data/eqvae/`，不进入 Git。

## 关键文件

| 文件 | 内容 |
|---|---|
| `feedback_fid5k.csv` | 同一噪声、类别和 ADM reference 下的 baseline / frozen-gap / closed-loop FID-5K |
| `*_fid5k.json`, `*_manifest.json`, `closed_*_rank00.json` | 上述 FID 数值、采样配置、NFE 和配对哈希的原始轻量记录 |
| `linearity_*.csv` | latent endpoint 对 `gamma=0` 变分切向的一致性 |
| `feedback_latent_*.csv` | frozen-gap 与 closed-loop latent endpoint 的配对比较 |
| `feedback_feature_*.csv` | 解码后 Inception feature 中的 frozen/closed 比较 |
| `density_action_*.csv` | `div(u) + u^T score` 的主实验和独立 probe-seed 重复 |
| `component_conservativity.csv` | common/unique 分量的 Jacobian 对称性审计 |
| `cross_direction_response.*` | `x400` 与 `v270` guidance 的 endpoint response 比较 |
| `exact_gauge_*` | 有限强度 exact gauge toy 的指标与可视化 |

所有正式图像 FID 均使用 5000 张样本；`frozen-gap` 与对应闭环条件的噪声 SHA256、标签 SHA256 和类别直方图逐项一致。

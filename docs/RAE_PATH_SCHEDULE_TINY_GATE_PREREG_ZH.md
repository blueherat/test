# RAE well-conditioned path：2k tiny gate 事前预测

## 问题

离线筛选假设“更换 coefficient 后，模型在 observation space 的 raw error 不变”。
这不是训练事实。下面用相同初始 checkpoint、相同 seed `3407`、相同 latent cache 和
相同 2k updates，直接训练四个候选，检验该反事实是否有预测价值。

## 条件

复用现成 2k 对照：

- `static`：标准线性 flow path；
- `annealed`：原始 `(1-t)^2` detail path。

新训练四条 annealed detail path：

| 名称 | family | floor | shape | 离线 total-risk ratio |
|---|---|---:|---:|---:|
| `floor005_p1` | power | 0.05 | `p=1` | 0.691 |
| `floor015_rat05` | rational | 0.15 | `alpha=0.5` | 0.705 |
| `floor030_p2` | power | 0.30 | `p=2` | 0.716 |
| `floor020_p2` | power | 0.20 | `p=2` | 0.748 |

训练后只做固定 seed 的 1k 图快速 gate。1k FID/KID 只用于同配置筛选，不作为论文数字。

## 固定预测

1. 四个候选都不会出现原始 annealed 的高噪声 `k->0`，训练数值稳定，无 NaN/Inf。
2. 至少两个候选在 1k FID 和 KID 中同时优于原始 annealed 2k；否则“逆条件数改善能转化
   为生成改善”的核心反事实不成立。
3. `static` 仍应最好或与最好候选接近，因为现有 10k 结果中 static 明显优于 annealed。
4. 候选内部预期大致按离线 total-risk 排序：`floor005_p1`、`floor015_rat05`、
   `floor030_p2`、`floor020_p2`。若完全不相关，说明 raw-error 不变假设过强。
5. 原始训练 loss 不用于横向排序，因为不同 path 的 target energy 不同；生成指标才是
   本 gate 的判断依据。

## 停止条件

- 若没有至少两个候选同时改善 FID/KID，不扩展到 3 seeds 或 10k。
- 若候选改善但 static 仍明显最好，结论只能是“修复了 annealed 的病态”，不能声称
  layerwise path 优于标准生成路径。
- 只有候选在 2k 明确逼近或超过 static，才值得做 3-seed 复验。

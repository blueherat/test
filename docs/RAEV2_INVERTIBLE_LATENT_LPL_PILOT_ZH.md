# RAEv2 可逆 latent adapter + LPL 试验记录

## 1. 研究问题

本试验检查一个很具体的假设：不修改 RAEv2 encoder、decoder 和 Stage-2
生成模型，只学习一个小型可逆映射，能否重新协调 frozen latent prior 与
frozen decoder。

设真实 latent 为 `z = E(x)`，可逆 adapter 为 `A`。训练时使用新坐标：

```text
u = A(z)
u_t = (1 - t) u + t epsilon
```

冻结的 RAEv2 Stage-2 在数值上把 `u_t` 当成原 latent 坐标并预测 clean
endpoint `u_hat`。进入冻结 decoder 前，使用精确逆映射：

```text
z_hat = A^{-1}(u_hat)
x_hat = D(z_hat)
```

因此 clean autoencoder 路径严格不变：

```text
D(A^{-1}(A(E(x)))) = D(E(x))
```

推理时 Stage-2 的 ODE 轨迹本身不变。若其缓存 endpoint 为 `q`，新输出是
`D(A^{-1}(q))`。所以这不是修改采样动力学，而是 post-hoc latent transport。

## 2. 与已有工作的关系

- [LPL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/204fee94c982a19230c39045aa54f977-Abstract-Conference.html)
  使用冻结 decoder 的内部特征训练 diffusion/flow 模型。原方法不在 frozen
  prior 两端学习可逆坐标变换。
- [Preconditioned Flow Matching, 2026](https://arxiv.org/abs/2603.02337)
  与本想法最接近：在 flow 前学习可逆或近似可逆预条件器，在变换后的目标空间
  训练 flow，再施加逆映射。它通常重新训练主 flow；本试验冻结已训练 RAEv2
  Stage-2，只做 post-hoc adapter。
- [Parametrized Diffusion Models, ICLR 2022](https://openreview.net/forum?id=1v1N7Zhmgcx)
  在 diffusion 前联合训练 normalizing flow，属于更早的可逆预处理先例。
- [Prior-Aligned Autoencoders, 2026](https://arxiv.org/abs/2605.07915)
  通过 tokenizer 训练目标塑造 diffusion-friendly latent，但不属于 frozen
  prior 上的可逆后处理。

结论：可逆 latent 预条件本身不是空白。当前可能不同的点仅是“冻结现有
Stage-2 prior 后，用 decoder-feature LPL 学习 post-hoc transport”。这个差异
是否足以构成方法贡献，要由更大规模、跨 seed 的质量改善决定。

## 3. 实现与审计边界

核心文件：

- `experiments/raev2_invertible_latent_lpl.py`
- `experiments/train_raev2_invertible_latent_lpl.py`
- `experiments/evaluate_raev2_invertible_latent_lpl_pairing.py`
- `experiments/evaluate_raev2_invertible_latent_endpoints.py`
- `tests/test_raev2_invertible_latent_lpl.py`

训练边界：

| 项目 | 配置 |
|---|---:|
| Stage-2 权重 | 官方 checkpoint 的非 EMA `model` |
| RAE encoder | 冻结 |
| RAE decoder | 冻结 |
| Stage-2 | 冻结 |
| 可训练参数 | 593,024 |
| 冻结参数 | 1,594,016,244 |
| adapter | 2 个 identity-initialized additive coupling block |
| global batch | 16 |
| optimizer | AdamW, lr `1e-5`, weight decay 0 |
| 训练步数 | 500，即 8,000 张 train 图 |
| 数据 | ImageNet-1k train；固定验证使用 validation |
| precision | bf16 前向；adapter 参数为 fp32 |

LPL 权重 `5.121181994940295e-5` 来自 32 样本梯度审计，使 LPL 加权梯度约为
Flow 梯度的 25%。审计得到 Flow/LPL 梯度 cosine 约 `-0.014`，说明 LPL 提供
近似正交、非零的新方向。

严格对照为：

```text
Flow-only: official transformed-coordinate Flow loss + identity regularization
Flow+LPL:   完全相同 + decoder-feature LPL
```

两条分支使用相同初始化、数据顺序、噪声、时间和 CFG dropout。续训恢复 raw
adapter、adapter EMA、AdamW、各 rank RNG，并跳过已经消费的数据。训练未使用
ImageNet validation、FID reference 或生成 endpoint。

## 4. 固定配对结果

使用 256 张未参与训练的 ImageNet validation 图。对每个 checkpoint 固定图像、
标签、噪声和时刻，仅替换 adapter。

500 步相对 identity 的结果：

| noise/signal | 分支 | LPL 变化 | 相对变化 | 配对 z 值 | Flow loss 变化 |
|---:|---|---:|---:|---:|---:|
| 0.5 | Flow | -0.030 | -0.007% | -0.22 | -0.000107 |
| 0.5 | LPL | -3.195 | -0.705% | -13.53 | -0.000009 |
| 1.0 | Flow | -0.078 | -0.010% | -0.50 | -0.000090 |
| 1.0 | LPL | -6.351 | -0.851% | -16.84 | -0.000031 |
| 3.0 | Flow | -0.276 | -0.020% | -1.38 | -0.000101 |
| 3.0 | LPL | -20.075 | -1.489% | -21.29 | -0.000095 |

这部分结论明确：LPL 分支确实学会降低 decoder-feature objective；Flow 对照
没有相同趋势；同时没有观察到 Flow loss 恶化。

## 5. 同 endpoint FID

缓存 1,000 个非 EMA Stage-2 endpoint。所有 adapter 分支复用同一 endpoint，
只改变 decoder 前的 `A^{-1}`。identity 分支精确复现原采样流程的 FID：

```text
s=1.00: 40.205794
s=1.78: 39.964221
```

完整结果：

| scale | step | Flow FID | LPL FID | identity FID |
|---:|---:|---:|---:|---:|
| 1.00 | 100 | 40.222449 | 40.203395 | 40.205794 |
| 1.00 | 200 | 40.210768 | 40.218495 | 40.205794 |
| 1.00 | 300 | 40.215234 | 40.217151 | 40.205794 |
| 1.00 | 400 | 40.220484 | 40.196073 | 40.205794 |
| 1.00 | 500 | 40.219009 | 40.212509 | 40.205794 |
| 1.78 | 100 | 39.961987 | 39.958664 | 39.964221 |
| 1.78 | 200 | 39.949185 | 39.946828 | 39.964221 |
| 1.78 | 300 | 39.955271 | 39.953198 | 39.964221 |
| 1.78 | 400 | 39.961901 | 39.947507 | 39.964221 |
| 1.78 | 500 | 39.963445 | 39.953735 | 39.964221 |

`s=1.78` 下 LPL 在五个 checkpoint 都优于 identity，并且都略优于同 step
Flow；但差值只有约 `0.002-0.014`。1,000-sample FID 的统计噪声足以覆盖这种
量级，当前不能宣称质量改善。`s=1` 下结果也不是单调的。

## 6. 当前结论

1. 实现逻辑通过了可逆循环、冻结边界、严格续训、identity FID 复现等检查。
2. adapter 训练很快不是少算 step：一个 step 是 global batch 16 的一次参数
   更新；主模型只传播输入梯度，只有 59 万 adapter 参数累积和更新梯度。
3. LPL 对其训练目标的改善明确且随 step 增强。
4. 这种改善尚未可靠转化为 FID。当前是“值得做一次复现”，不是“方法成功”。
5. 若后续更大样本仍只有训练目标改善而 FID 不改善，应停止该路线；这将说明
   decoder-feature one-step objective 仍不足以定义有生成价值的 post-hoc transport。

## 7. 下一道验收门槛

最小确认实验应保持非 EMA、官方 `s=1.78`、同 endpoint 对照：

- 追加一个独立 endpoint seed，至少 1,000 样本；或直接做一个 5,000 样本 seed。
- 预先固定比较 `identity / Flow-200 / LPL-200 / Flow-400 / LPL-400`。
- 只有 LPL 相对同 step Flow 的 FID 改善在独立 seed 上方向一致，才扩大到 5k/多 seed。
- 若独立 seed 反向，先复核 endpoint、checkpoint hash、state key 和 identity FID；
  复核无误后停止，不通过继续调 step 或挑 checkpoint 挽救结论。

本机结果目录：

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_invertible_latent_lpl/
```

合并后的 FID 表和曲线：

```text
summary_model_u0_500_n1000_v1/endpoint_fid.csv
summary_model_u0_500_n1000_v1/endpoint_fid_curves.png
```

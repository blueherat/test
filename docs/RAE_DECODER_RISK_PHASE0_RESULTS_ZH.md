# RAE Decoder-Aware Phase 0 结果

## 结论

Phase 0 的最终决定是：**不进入 Phase 1，不训练 decoder-aware continuation。**

原因不是 decoder-aware 信号不存在。恰恰相反，0A 表明 decoder hidden risk 与
latent MSE 的局部几何差异很强；停止原因是 0C 表明这种信号不能被预注册的廉价、
常数、channel-shared metric 可靠近似。

因此当前证据支持：

> Frozen RAE decoder 对 latent error 有强烈、随样本、token 和时间变化的响应；
> 但一个离线提取的常数二次型不足以替代 decoder perceptual supervision。

这不是新的 latent manifold 结论，也不能由跨 path raw MSE 推出。

## 实验控制

- 0A/0C 只使用 static linear flow：
  `z_t=(1-t)z0+t*epsilon`、`z0_hat=z_t-t*v_hat`。
- 同时报告 `L_v`、`L_x0=t^2*L_v` 和 decoder hidden LPL。
- 五个时间点是 shifted logit-normal 的 5%、27.5%、50%、72.5%、95% 分位点。
- calibration：ImageNet train 1024 张，排除 10k continuation latent cache 使用过的
  160k 原图索引。
- test：ImageNet validation 2048 张。
- exact decoder-gradient test subset：128 张，每张覆盖五个时间点。
- fp32、关闭 TF32、四张 RTX 4090。

## 0A：Loss-Space Audit

0A 三个预注册门槛全部通过：

- 五个时间段的 `median cosine(grad_x0, grad_dec)` 为
  `0.0327 / 0.0200 / 0.0119 / 0.0047 / 0.0018`，均远低于 `0.80`。
- gate 使用的局部步长仅为当前 latent error norm 的 `0.1%`。
- x0 梯度方向平均降低 decoder error `0.109%`。
- decoder 梯度方向平均降低 decoder error `9.335%`。
- 五个时间段全部满足 decoder 方向至少多降低 `15%` 的要求。

同一时间段内，`L_x0` 与 `L_dec` 的 test Pearson 只有约 `0.11-0.22`。
这里按时间段分别计算，避免共同的 `t^2` 尺度制造虚假相关。

梯度能量分布进一步显示：

- 最早时间段的 decoder/x0 channel effective support 约为 `175/713`。
- 最早时间段的 decoder/x0 token effective support 约为 `51/243`。
- decoder token support 随时间增加到约 `144`，不是固定稀疏 mask。
- decoder 梯度主要落在中高 DCT band；x0 梯度随时间逐渐转向最低频 band。

这说明 decoder-aware 风险不仅是频带权重，也不是固定通道重标定。

## 0B：Generated Latent Closure

四条 10k path 各用相同 noise、label seed 保留 256 个 50-step ODE latent。
没有把生成样本错误地逐样本配对到真实图像。

`z_cycle=E(clamp(D(z_gen),0,1))` 的 relative RMS 中位数：

| source | cycle residual | local decoder sensitivity | 5k FID |
| --- | ---: | ---: | ---: |
| clean test | 0.4003 | 1.3175 | - |
| static | 1.0355 | 1.5625 | 123.53 |
| random | 1.0382 | 1.5773 | 128.99 |
| annealed | 1.0759 | 1.6268 | 143.54 |
| reverse | 1.2732 | 1.8958 | 159.05 |

四个生成 path 上，cycle residual、local sensitivity 与 FID 的排序完全一致，
Spearman 都为 `1.0`。由于只有四个 path，这只能作为机制一致性，不能当作统计因果证据。

Projected nearest-clean distance 对生成 latent 反而比 clean query 小。结合 hidden response，
更合理的解释是生成 latent 向参考分布中心收缩；不能把最近邻距离较小误读成更 on-manifold。

## 0C：廉价 Metric Proxy

三个候选全部未通过：

| proxy | calibration Spearman | test Spearman | gradient cosine |
| --- | ---: | ---: | ---: |
| decoder embed | 0.1512 | 0.2055 | 0.0285 |
| randomized GN | 0.1970 | 0.1795 | 0.0224 |
| randomized GN x 4 DCT bands | 0.0858 | 0.0731 | 0.0159 |

门槛是 held-out Spearman `>=0.80`、gradient cosine `>=0.70`、split gap `<=15%`。
Spearman 在每个时间段内独立计算后取中位数，不能依赖时间尺度混合。

Randomized GN 已在 calibration `z0_hat` 的五个真实时间分布上估计，而不只在 clean
latent 上线性化。四 DCT band 是预注册允许的唯一一次结构扩展；它仍失败，因此按协议
停止 proxy 路线，不再追加第二个结构。

## 研究含义

当前最稳妥的机制结论是：

1. Latent MSE 确实遗漏了对 frozen RAE decoder 很重要的局部方向。
2. 生成 latent 的 closure/sensitivity 异常与四条 path 的 FID 排序一致。
3. 有用的 decoder geometry 高度 state-dependent，常数 channel/DCT metric 无法压缩它。
4. 直接使用完整 LPL 已不是新方法；在廉价 proxy 失败后，不值得按原计划花费 Phase 1
   算力验证一个已知方法。

如果以后重新打开这个方向，需要改变核心方法假设，例如研究低成本、输入相关的
decoder Jacobian sketch；这属于新计划，不能作为本 Phase 0 的临时补丁。

## 复现与产物

代码：

- `experiments/rae_decoder_risk_phase0.py`
- `experiments/cache_rae_decoder_risk_phase0.py`
- `experiments/run_rae_decoder_risk_phase0.py`
- `tests/test_rae_decoder_risk_phase0.py`

本机产物：

- `/home/zhoushunyu/data/eqvae/experiments/rae_decoder_risk_phase0/`
- 完整报告：`phase0_gate_report_zh.md`
- 最终门控：`phase0_decision.json`

运行顺序：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/cache_rae_decoder_risk_phase0.py
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_decoder_risk_phase0.py 0a
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_decoder_risk_phase0.py 0c
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_decoder_risk_phase0.py 0b
python experiments/run_rae_decoder_risk_phase0.py report
```

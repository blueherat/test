# ImageNet-100 SiT flow baseline

## 官方来源

- 模型与训练定义来自官方仓库 `willisma/SiT`。
- 本地固定 commit：`cbde832a40b153ccc79603412409da9c9b0c568c`。
- `models.py` 固定 SHA256：`677cfa2cf5a7db122abd014d6d92d7ac5f39745f571773a6adcc26c7d2f33d89`。
- 训练入口默认拒绝加载哈希不一致的模型源码，避免实验过程中无意改变 backbone。

## 对齐项

- SD-VAE posterior sampling：`z = 0.18215 * (mean + std * noise)`。
- Linear interpolant：`x_t = (1-t) * noise + t * z`。
- Velocity target：`u_t = z - noise`。
- `t ~ Uniform[0, 1)`。
- AdamW：`lr=1e-4`、`betas=(0.9, 0.999)`、`weight_decay=0`。
- Classifier-free label dropout：`0.1`。
- EMA：`0.9999`，每个 optimizer step 更新。
- 全局 batch：`256`，常数学习率。

## 有意的系统优化

- 预先抽取连续 `float32` ImageNet-100 SD-VAE posterior moments，训练时不再运行 VAE 或解析 Arrow。
- BF16 autocast；loss、参数和 EMA 仍为 FP32。
- `torch.compile(fullgraph=True, dynamic=False)`。
- fused AdamW、Flash SDPA、static-graph DDP、pinned-memory persistent workers。
- 日志区间内不做逐步 `.item()` 或跨卡同步。
- checkpoint 保存每个 rank 的 Python、NumPy、CPU Torch 和 CUDA RNG 状态。

公开 latent cache 只有每张图的确定性 center-crop 编码，没有官方在线训练中的随机水平翻转。该差异明确写入数据 manifest，不把这条 ImageNet-100 baseline 冒充成官方 ImageNet-1K 数字的严格复现。

## 准备数据

```bash
python experiments/prepare_imagenet100_sdvae_index.py
python experiments/prepare_imagenet100_sdvae_cache.py
```

## 四卡训练

先跑 S/2 验收协议：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  experiments/run_imagenet100_sit_4gpu.sh
```

正式 B/2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SIT_MODEL='SiT-B/2' \
OUTPUT_DIR=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-b-2_seed0 \
  experiments/run_imagenet100_sit_4gpu.sh
```

可通过环境变量修改 `GLOBAL_BATCH_SIZE`、`MAX_STEPS` 和 `SEED`；额外命令行参数会继续传给 Python 入口。

## 本机稳定态 benchmark

四张 RTX 4090、全局 batch 256、完整 forward/backward/optimizer/EMA：

| 配置 | 吞吐 | 单步 | 单卡峰值显存 |
|---|---:|---:|---:|
| SiT-S/2，BF16，compile | 4809 img/s | 53 ms | 3.59 GiB |
| SiT-S/2，BF16，eager | 3818 img/s | 67 ms | 4.32 GiB |
| SiT-S/2，FP32，compile | 3722 img/s | 69 ms | 5.79 GiB |
| SiT-B/2，BF16，compile | 2265 img/s | 113 ms | 8.55 GiB |

这些是 synthetic-latent 稳定态吞吐，不包含第一次编译、checkpoint、validation 和最终采样时间。实际缓存数据 smoke 会另行记录。

真实 memmap 数据的一整个 500-step smoke 中，去掉首次编译区间后的稳定吞吐为 `4480-4702 img/s`，即 synthetic 上限的约 93%-98%；单卡峰值为 `3.47 GiB`。末 50 步平均训练 loss 为 `0.9652`。因此数据读取不是当前主要瓶颈。

四卡 resume 对照中，连续跑到 step 3 与 `step 2 checkpoint -> step 3` 的 loss 完全一致，所有 rank 的 RNG state 完全一致；模型相对 L2 差为 `8.7e-13`，EMA 相对 L2 差为 `1.6e-21`，属于 fused AdamW 在重新装载状态后的浮点舍入量级。

## 300K checkpoint 曲线

2026-08-10 使用固定的无 guidance 5K 协议完成了 60K 到 300K 的 EMA checkpoint 评估：

- sampler：官方 SiT Dopri5；
- `cfg_scale=1.0`，`guidance=False`；
- SD-VAE decoder；
- 每个 checkpoint 使用相同的 5000 个全局采样 seed；
- reference 固定为 ImageNet-100 validation 的 5000 张图；
- FID 使用独立 ADM TensorFlow 环境计算。

| step | FID | sFID | IS |
|---:|---:|---:|---:|
| 60K | 130.7967 | 73.1066 | 12.4126 |
| 120K | 99.4306 | 70.1554 | 17.4834 |
| 180K | 86.9341 | 69.4012 | 20.3537 |
| 240K | 79.9044 | 69.0897 | 22.6320 |
| 300K | 73.9829 | 68.7939 | 24.3396 |

FID 在所有相邻 checkpoint 上都改善。240K 到 300K 仍下降 `5.92`，约为 `7.4%`，因此满足预注册的续训条件：最新 FID 下降，且末三个 checkpoint 的线性斜率为负。

结果文件位于：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k/
  sit-s-2_unguided_fid5k_curve.json
  sit-s-2_unguided_fid5k_curve.csv
```

## 300K 到 800K 自动续训

持久任务使用 `experiments/run_imagenet100_sit_to800k.sh`：

1. 先重新验证 60K--300K 曲线仍满足续训条件。
2. 从 `step_00300000.pt` 精确恢复模型、EMA、optimizer、每个 rank 的 RNG 和 dataloader 位置。
3. 依次训练到 400K、500K、600K、700K、800K。
4. 每 100K 保存 checkpoint，暂停训练进程后执行一次同协议 5K FID，再恢复下一段训练。
5. 任一训练、采样或 FID 环节失败时，shell 的 `set -euo pipefail` 会终止流水线，不会跳过失败继续运行。

当前 tmux 名为：

```text
sit-s2-to800k-with-fid
```

训练稳态约为 `17.8--18.2 step/s`，即 100K steps 约 `1.5` 小时；从 300K 到 800K 的纯训练时间约 `7.6--7.8` 小时，另加五次采样与 FID。

## 固定低显存采样与 FID 流程

后续 SiT checkpoint 的采样评估统一使用：

```bash
STEP=800000 NUM_SAMPLES=5000 \
  experiments/run_imagenet100_sit_fid5k.sh
```

该入口固定执行同一协议：

- 四卡采样，每卡 batch `64`，保持此前正式曲线的采样 RNG 顺序；
- SD-VAE 以 batch `4` 分块解码；
- 每个 PyTorch rank 的 allocator 硬上限为 `7.5 GiB`；
- ADM-FID 只使用 GPU 0，batch `8`，TensorFlow 显存比例上限为 `0.30`；
- 外层每 `0.25` 秒检查一次 `nvidia-smi`，任一参与 GPU 达到 `9216 MiB` 时终止整个进程组；
- 每次分别保存 `sampling_resource_audit.json` 与 `fid_resource_audit.json`，没有有效资源审计的旧结果不会被该流程复用。

2026-08-11 使用 800K EMA checkpoint 完成了正式 5000 张端到端验收：

| 阶段 | 使用 GPU | 峰值显存 | 结果 |
|---|---:|---:|---:|
| Dopri5 采样 + SD-VAE 解码 | 4 | `2778 MiB/卡` | 5000 张约 46 秒 |
| ADM-FID | 1 | `7678 MiB` | FID `61.0016` |

同一 checkpoint 先前高显存流程得到 FID `61.0080`，差值仅 `0.0064`。低显存流程的 sFID 为 `68.9871`，IS 为 `30.2042`。正式结果位于：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_lowmem_v1/
  sit-s-2_step800000_seed0/
    sampling_manifest.json
    sampling_resource_audit.json
    fid5k_adm_results.json
    fid_resource_audit.json
```

## Dynamic Dual-Output 复现入口

双输出实现是在同一套 ImageNet-100、SD-VAE latent 和 SiT-S/2 baseline 上进行的
`DDO-inspired linear-flow adaptation`。它不是 CVPR 2022 原始 VP-DDPM/DDIM 公式的
严格复现；完整方法边界、400K/450K 指标和结论见
[`DYNAMIC_DUAL_OUTPUT_DIFFUSION_LITERATURE_REPORT_ZH.md`](DYNAMIC_DUAL_OUTPUT_DIFFUSION_LITERATURE_REPORT_ZH.md)。

四卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=450000 \
SAVE_EVERY=100000 \
OUTPUT_DIR=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_dual-output_seed0 \
  experiments/run_imagenet100_sit_dual_output_4gpu.sh
```

对同一 checkpoint 的 `x / epsilon / dynamic` 三条路径做配对 5K 采样和 ADM-FID：

```bash
python experiments/run_imagenet100_sit_dual_fid5k.py \
  --checkpoint /home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/\
sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt \
  --output-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/\
fid5k_dual-output_step450000_seed0 \
  --modes x epsilon dynamic
```

主要代码分工：

| 文件 | 作用 |
|---|---|
| `imagenet100_sit_dual_output.py` | `2C+1` 输出拆分、三项损失、gate 和速度转换 |
| `train_imagenet100_sit_dual_output.py` | 四卡训练、EMA、validation、精确 checkpoint 恢复 |
| `sample_imagenet100_sit_dual_output.py` | 少量同 seed 可视化比较 |
| `sample_imagenet100_sit_dual_fid.py` | 四卡正式 NPZ 采样 |
| `run_imagenet100_sit_dual_fid5k.py` | 三路径采样、显存审计、ADM-FID 与汇总 |

Git 仓库只保存代码、配置、测试和报告。ImageNet、latent cache、官方 SiT checkout、
checkpoint、生成样本和 TensorFlow ADM 资产继续保存在 `/data/shared` 或用户数据目录，
不得加入 Git。

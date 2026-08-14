# SiT 800K tangent endpoint 投影便携数据

本目录保存
[`IMAGENET100_SIT_800K_TANGENT_PROJECTION_RESULTS_ZH.md`](../../IMAGENET100_SIT_800K_TANGENT_PROJECTION_RESULTS_ZH.md)
使用的轻量实验产物。

## 范围

- strong model：`v800` EMA；
- weak models：`x800` EMA 与 `v500` EMA；
- 5000 个严格配对的 ImageNet-100 initial noise 与 class label；
- fixed Heun 100 steps，FP32，TF32 关闭；
- 比较 baseline、raw tangent、exact frozen response 的 tangent 投影、正交余量
  和完整 frozen endpoint。

## 文件

- `tangent_projection_fid5k.csv`：十组 FID、sFID、IS、FID gain 和样本 hash；
- `tangent_projection_summary.json`：完整结果与 endpoint geometry summary；
- `tangent_projection_fid5k.png`：FID 与相对 baseline 收益图；
- `x800_sampling_manifest.json`、`v500_sampling_manifest.json`：checkpoint hash、
  sampler、输入配对与资源信息。

## 复现入口

四卡一 rank 一卡时可以直接使用默认参数。下面是本轮实际使用的双卡恢复命令，
四个逻辑 rank 仍保持原有 seed 和全局样本顺序：

```bash
python experiments/run_imagenet100_sit_tangent_projection_fid5k.py \
  --directions x800,v500 \
  --sampling-gpus 2,3 \
  --sampling-processes 4 \
  --fid-gpu-index 2 \
  --num-samples 5000 \
  --per-rank-batch-size 32 \
  --vae-decode-batch-size 4 \
  --heun-steps 100
```

## 未提交内容

本目录不包含 checkpoint、5K 图像 NPZ、逐样本 projection geometry 或日志。
完整本机产物位于：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/tangent_projection_800k_v1/
```

# SiT 800K tangent transport 便携数据

本目录保存 [`IMAGENET100_SIT_800K_TANGENT_TRANSPORT_RESULTS_ZH.md`](../../IMAGENET100_SIT_800K_TANGENT_TRANSPORT_RESULTS_ZH.md) 使用的轻量实验产物。

## 范围

- strong model：`v800` EMA；
- weak models：`x800` EMA 与 `v500` EMA；
- 128 个严格配对的 ImageNet-100 latent 初始噪声和类别；
- fixed Heun 100 steps；
- `gamma=0.01,0.02,0.05,0.1,0.2,0.5,0.75,1.0`；
- 比较 gamma-zero variational tangent、exact frozen 与 closed response。

## 文件

- `tangent_frozen_metrics.csv`：每个 family、gamma 和 response 的完整汇总；
- `tangent_frozen_summary.json`：中心差分、gamma=1 统计和 20,000 次 paired bootstrap；
- `tangent_frozen_gamma_sweep.png`：endpoint cosine、非线性残差和幅值比曲线；
- `x800_manifest.json`、`v500_manifest.json`：模型哈希、输入哈希和数值协议。

## 未提交内容

本目录不包含 checkpoint、逐批 trajectory tensor、日志或生成样本。完整本机产物位于：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/tangent_frozen_800k_v1/
```

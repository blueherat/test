# ImageNet-100 SiT 800K nominal guidance transfer

本目录保存 800K nominal-path frozen AutoGuidance 机制实验的便携汇总数据与图。完整说明见：

- `docs/IMAGENET100_SIT_NOMINAL_GUIDANCE_TRANSFER_800K_RESULTS_ZH.md`

## 实验范围

- strong model：`v800`
- weak models：prediction-target `x800` 与 same-target `v500`
- guidance scale：`gamma=1`
- geometry：2 个采样 seed，每个 seed 512 个配对样本
- formal causal FID：2 个采样 seed，每个条件 5,000 张
- donor screen：1 个采样 seed，每个条件 1,000 张
- 本轮不包含 400K 复验

## 文件说明

- `nominal_transfer_compact.csv`：按 family、trajectory relation 和时间汇总的 gap 几何。
- `nominal_transfer_all_runs_by_time.csv`：每个 800K geometry run 的逐时间汇总。
- `segment_transfer_all_runs_by_time.csv`：baseline 到 frozen 位移线段上的逐时间、逐 alpha 汇总。
- `endpoint_latent_geometry.csv`：frozen 与 closed 终点 latent 位移关系。
- `replay_fid5k.csv`：baseline、frozen、fully replay、closed 的配对 FID-5K。
- `projection_fid5k.csv`：baseline、frozen、gain-only、direction-only、closed 的正式配对 FID-5K。
- `donor_fid1k.csv`：替换 nominal gap 的 noise/class 来源消融。
- `geometry_summary_manifest.json`、`causal_summary.json`：完整汇总元数据与来源记录。
- PNG 文件：上述表格对应的可视化。

## 未提交的大型产物

逐样本几何表、生成的 5K 图像、checkpoint、日志和资源监控明细保存在：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/nominal_guidance_transfer_800k_v1
```

这些文件未放入 Git。汇总包总计约 1.8 MB，不包含模型权重或生成样本。

## 验证

- 61 项相关测试通过。
- 67,584 行几何记录行数完整、主键无重复、projection valid rate 为 100%。
- 正式投影表 20 个条件的 noise/label fingerprint 在同 seed 内完全配对。
- 8 组新增正式采样均为 5,000 张，100 类全部覆盖。
- 56 份 sampling/FID resource audit 均成功且无显存违规。

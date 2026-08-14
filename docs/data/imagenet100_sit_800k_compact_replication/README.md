# ImageNet-100 SiT 800K 紧凑复验数据

本目录保存 800K finite-guidance 稳定性复验的轻量、可审计结果，不包含 checkpoint、采样 NPZ、FID reference 或其他大文件。

## 内容

- `quality_match.json`：`x800` 端点和 `v400/v500/v600/v700` 候选的配对 FID；记录锁定 `v500` 的规则与结果。
- `compact_replication_rows.csv`：两个采样 seed、五个条件的 FID、收益、哈希、NFE 和资源审计长表。
- `compact_replication_summary.json`：汇总统计、验收条件、原始结果来源和完整审计元数据。
- `compact_replication.png`：FID 与 frozen-gap 收益保留率图。

正式说明见 [`docs/IMAGENET100_SIT_800K_COMPACT_REPLICATION_ZH.md`](../../IMAGENET100_SIT_800K_COMPACT_REPLICATION_ZH.md)。

原始实验目录为：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/finite_guidance_800k_compact_replication_v1
```

原始目录约 14GB，仅保留在本机数据盘。

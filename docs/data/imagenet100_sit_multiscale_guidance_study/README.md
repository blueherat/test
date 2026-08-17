# ImageNet-100 SiT multiscale guidance study

本目录保存多尺度 internal guidance 正式流程的便携数据、审计结果与图。完整中文报告见：

- `docs/IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md`

## 实验范围

- strong model：ImageNet-100 `SiT-S/2 v800 EMA`
- 新训练 heads：depth `4/6/10/12` velocity readout，各 50K step
- 已有对照：depth-8 `v/x/epsilon`、depth-12 x、final x、external `v500`
- 条件数：99
- 评估：每条件 1,000 张，所有条件共享 noise/label fingerprint
- 本轮按用户要求不做 FID-5K confirmation

## 主要文件

- `condition_metrics.csv`：全部 99 个正式条件的原始指标。
- `named_method_comparison.csv`：方法级条件及 matched-integrator baseline 差值。
- `causal_map_fid1k.csv`：三种 provider 的 `time × latent-band` 干预结果。
- `causal_order_fid1k.csv`：coarse-to-fine 与逆序对照。
- `spectral_router_selection_counts.csv`：router 在 adaptive ODE 调用中的 depth 选择次数。
- `latent_spectrum_atlas.csv`：14 种 gap 的逐时间 latent FFT 统计。
- `latent_spectrum_atlas_compact.csv`：按 provider 汇总的 atlas。
- `band_delay_fit.csv`：band-wise delay fitting 全部候选结果。
- `study_protocol.json`、`study_summary.json`、`atlas_summary.json`：原始协议和汇总元数据。
- `audit_summary.json`：数据完整性、fingerprint、stage 状态与文件 SHA256。

## 图

- `method_overview_fid1k.png`：主要方法相对 matched integrator baseline 的 FID-1K gain。
- `causal_map_fid1k.png`：成功/失败 gap 的 causal utility map。
- `depth_schedule_and_router_fid1k.png`：静态深度、正反调度与 router。
- `depth_gap_latent_spectrum_atlas.png`：各 depth gap 的 RMS 与频谱质心时间曲线。
- `paired_preview_montage.png`：相同 noise/label 下的 baseline、正反调度与 router 预览。

## 复现汇总包

```bash
python experiments/summarize_imagenet100_sit_multiscale_guidance_study.py \
  --study-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1 \
  --output-dir docs/data/imagenet100_sit_multiscale_guidance_study
```

完整 checkpoint、逐条件原始 preview 和运行日志保留在本机 data 目录，不进入 Git。

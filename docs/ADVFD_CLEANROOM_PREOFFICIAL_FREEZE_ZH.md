# AdvFD paper-only 首轮结果冻结记录

冻结时间：2026-08-23，官方实现差异审计之前。

## 边界

- 独立实现依据 AdvFD、FD-Loss 论文和公开 pMF 生成器编写；没有复制或导入 AdvFD
  官方实现。
- AdvFD 官方仓库此前曾被初步浏览，因此本实验不能声称 blind reproduction；本轮开始
  后一直保持源码封存，直到本记录完成。
- 当前结果是 Inception-2048 static + 投影到 512 维的 adaptive critic、batch 32、
  5k steps、FID-5K 的缩放实验，不是完整 SIM/global-batch-1024/125k/FID-50K 复现。
- 训练结束后只改过评估诊断：adaptive moments 改为 FP64，并新增 CPU FP64 FID。
  训练 checkpoint 的算术路径未被追溯修改。

## 关键结果

- base：`8.740200744744389`
- static@5000：`7.783402471202352`
- AdvFD@5000：`7.573231804664732`
- AdvFD - static@5000：`-0.2101706665376204`
- AdvFD 的训练态 covariance FD 在 step 4,771 首次数值塌到 0；5k held-out raw
  critic feature RMS 约为 `640`，因此收益与稳定性问题必须同时保留。

## 代码与数据哈希

- `run_pmf_pilot.py`: `e2c601ba12cf7dfb1e08c1c0df3551903ae78e405a8b9cb7b1486aa42dacb973`
- `core.py`: `f647298f1d58e68c6822fa95150d905c5bc9e5c57f6f4cd315fafc105f7d92ce`
- `feature_extractors.py`: `3104e1d80e302644035285b248ffd0f564fac47de4bf9bc0f38348b9c74507f9`
- `generators.py`: `b15c9d48ee33d9e11ca57ebbf16b7b1d0ff922e902f99ef335d5b002c6de2127`
- paired CSV: `1721b74e5bad5d2da26b9c3a1b01e6e7f94409d76674f1eca5387bc4e6bb4660`
- paired JSON: `71850fc20e27d062e57077496abe15045a1b2dbaa65497e881fdbab6187c4f6e`
- static config: `05300ecd18c92232b1d591b9abc79ee76d53246bb2a0c2014fd06e3b9ae8173d`
- AdvFD config: `e25e5ccadcfc702de88d8bdf3b58fca5f02cdfb73d4bb15addb768e5f5c2f2b1`
- static@5000 checkpoint: `654edba27676ab6cd1d42a2377ca4139c7c66b2bdc1509265ac4c1a618525b1e`
- AdvFD@5000 checkpoint: `5ecdd9ae9c9e692c9c4d97e011fa2f55e5bc1d5ab3be7f5dd19138034dd2ca57`
- pMF-B base checkpoint: `aa95e128b3378e25285f9c2470d1e074d73a8d9d7846a7c9ab59e851c2ed64f8`
- warm-start: `9a5f62159de7d0fedbf1e449d038ce40546a56e258e7f91f1ae3d7f881b3fe2f`
- AdvFD PDF: `ce128da0304e50e7a103e8e82a72f343d0a284d77b3649f2e29c668225db3b3f`
- FD-Loss PDF: `eadbfa2f5e4431dab81fa20857bfc54c8ba9e97704e6048757503c5e1d5f1529`

仓库状态：eqvae HEAD `2b7674019ae30104649b9e5cd7814f43a8d0d308`；独立文件在
该时刻尚未提交。封存的 AdvFD 官方仓库 commit 为
`4e4cfed944e4fc38a75fae3ea7701ae9e5587060`。

**61.9471 的旧 MLP 记录存在；它与当前实验都关闭后半程 guidance，不能用开关窗口解释两组数字的差异。**

2026-09-15 核对原始请求、checkpoint 哈希、输入数组与实际采样代码：

|记录|旧 Context MLP 1K FID|同组原生 IG|噪声与标签|
|---|---:|---:|---|
|`context_reference_confirm_20260912` 独立核查|61.947099|64.809030|seed 2026121321，独立 1K|
|`guidance_loss_50k_20260914` 的 Context 对照|64.907565|68.462790|另一组 1K；当前完整数据实验沿用这组输入|

两份冻结请求中的旧头 SHA256 都是 `c20c09c251aea8b89a5711515d17f08cc546e196089869256d8c3f05808ec3fd`，对应 `guidance_distribution_20260912/sit_small/input_local_training/head.pt` 的 3K 最终 EMA。不是一个用了 3K、另一个用了 50K。

旧 `context_base` 经 `experiments/context_reference_confirm_20260912/core.py::field` 调用 `guidance_distribution_20260912/local_head.py::field`，引导幅度来自 `guidance_pasted_20260912/common.py::amount`。当前 `guidance_dynamic_50k_20260915/sampling.py::field` 的窗口相同：

\[
\beta(t_{\rm left})=\lambda
\begin{cases}
6/7,&0\leq t_{\rm left}<1/4,\\
1,&1/4\leq t_{\rm left}<1/2,\\
0,&1/2\leq t_{\rm left}\leq1.
\end{cases}
\]

这里时间从噪声 0 走向数据 1；旧 Context 的 \(\lambda=0.8\)。两边都是 Heun 64 步、每图 128 次完整主干调用、0 次额外前缀调用。Heun 第二次场查询继承该步左端点的窗口和幅度。旧名字里的 `native_half`/`context_half` 是幅度减半，不是轨迹只走一半。

两组输入的实际数据哈希不同：

|输入|旧独立 1K|当前沿用的 1K|
|---|---|---|
|noise 数组|`43aa7ade8ea9950bda8035909f0aca260c2c674a42985e348a45fc93a00d1946`|`7b556b26df0c4ff0c2480a73f656c8b4c774b75f5d0e4c0c948026345b3423ab`|
|labels 数组|`6b96abafcd32165df4a61e11e3b471016ab9e761944fc36fdbcaa259abcd6559`|`f127fa252873a95997fc297cd3837f1556653e02755cb0b91345a5b9879f3aef`|

旧 noise 文件名为 `inputs/first.npy`；当前为 `inputs/noise.npy`。本表哈希针对数组数据而不是 `.npy` 文件头。当前 5K 输入的前 1K 与上述当前 1K 逐元素一致。

因此 61.9471 是有效的历史记录，也说明旧 MLP 当时相对同组原生 IG 有改善；它不是当前噪声组上的基线。窗口没有变化，输入样本组发生变化，不能把两个数直接比较后归因于训练、窗口或某一个实现细节。

后续补查发现，旧队列已做过两个采样入口的完整轨迹逐位对照：`guidance_loss_50k_20260914/worker.py::gpu_preflight` 用同一组两张噪声/标签，对 `components.guided_field` 和旧 `local_head.field` 运行完整 Heun 积分，要求 `torch.equal`。原始 `gpu_preflight.json` 中 Context 与 Native 均记录 `zero_or_legacy_trajectory_exact=true`。因此此前“没有做过逐位输出复现”的表述过宽，现予更正；已有两条测试轨迹的精确等价证据，但没有把两个实际 1K 输入组全部进行跨入口复现。

用户追问外推系数后，补齐真正同输入、同系数的模型比较：

|头|1K 输入|峰值系数|FID|
|---|---|---:|---:|
|旧 3K Context MLP|旧独立组 A|0.8|61.947099|
|同一个旧 3K Context MLP|当前组 B|0.8|64.907565|
|新完整数据标准 loss 50K MLP|当前组 B|0.8|64.967917|
|同一个新 50K MLP，扫描后|当前组 B|1.0|64.140582|

旧头和新头在当前输入组、同系数下只相差 0.060352，而不是约 3 点。新头把系数从 0.8 调到 1.0 后下降 0.827335，但它不能解释旧头 61.947099 与 64.907565 之间的差异，因为后两个数使用完全相同的旧头和 0.8。原生 IG 也从组 A 的 64.809030 变为组 B 的 68.462790，支持这是跨采样组的变化，而非只发生于 MLP 的退化。两组都是每类 10 张、100 类，FID 使用同一 ADM 参考路径与评估设置；没有依据把差异归因于类别计数不平衡。

当前完整数据、动态 VAE posterior、50K 标准 diffusion loss 已完成自己的扫描：系数 1.0 的 1K FID 64.140582、5K FID 36.909452；系数 0.975 的 1K FID 64.206670、5K FID 36.924053。与旧 MLP 在当前 1K 的 64.907565 相比，还同时改变了训练协议和所选系数，不能将差距单独解释为新 loss 收益。

可检查的原记录：[旧独立 1K 报告](CONTEXT_REFERENCE_CONFIRM_RESULTS_20260912_ZH.md)、[旧独立 5K 报告](CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)、[此前小数据训练协议审计](GUIDANCE_REAL_50K_VS_CONTEXT_3K_AUDIT_20260915_ZH.md)。原始实验文件均保留于 `/home/zhoushunyu/data/eqvae/experiments/`。

# SiT CFG / IG 强基线与历史证据审计

日期：2026-09-13。本文件依据仓库报告、原始 `result.json` 和采样源码核对；没有重跑历史 FID。按后续授权补了新旧采样器各三类、每类 8 个输入的实现等价检查，见第 5 节。`docs/RESEARCH_STATUS.md` 首条状态仍停在 Sep11，部分队列进度已经过期，下文采用对应结果报告及已提交结果。审计聚焦现有 SiT-S/2 与 Sep12/13 进展，不宣称遍历整个仓库的所有历史配置。

**当前最该超过的低成本 CFG 对照是仓库 APG 适配：extra alpha=2、beta=-.5、64 步 Heun、left t<.75，在两组独立 5K 都比标准 CFG 好约 4%。标准 CFG 的保留工作点是 extra alpha=1.25（总 w=2.25），相同时间窗口。历史调优 CTRL 的正确参数是 extra alpha=2.75、lambda=5、K=.2；它曾超过标准 CFG，但仍未超过同一 bank 的 APG。**

另有一次独立 5K 确认的 CFG 条件割线 + APG + 通道回缩，FID 21.2281，优于当批 APG 21.6517，但需要 320 次 full，而 APG 是 224 次。这是额外预算下必须认识到的历史对照。仅超过 vanilla CFG，或只超过 Sep13 失配参数的 CTRL，均不足以超过仓库强基线。

## 1. 强度、时钟和调用口径

令 `c=v_cond(z,t)`、`u=v_null(z,t)`、`d=c-u`。本仓库 `strength/amount/alpha` 一般指**额外系数**：

`CFG = c + alpha*d = u + (1+alpha)*d`。

SiT-S/2 为 ImageNet100、SiT velocity EMA，时间从噪声 `t=0` 走向数据 `t=1`。当前标准 CFG 使用 `alpha=1.25`，在接受步的 `left t<.75` 启用，之后是 conditional 单分支。固定 64 步 Heun 的 48 个活跃区间，各有两个条件、两个 null 调用；其余 16 区间只调用条件，因此 **224 full / 0 extra prefix 每图**。同一区间的右 Heun stage 沿用左端开关，不能在右端 `.75` 提前关闭；切换时刻是实现的一部分。

这是反复保留、经独立噪声确认的标准 CFG 工作点，不是证明所有时间调度和求解器中的全局最优。此前 fusion 网格明确搜索过 vanilla extra `.75,1,1.25,1.5,1.75,2,2.5,3`，但 cutoff 固定 `.75`。新搜索若改变窗口或步数，须同步调优基线，不能宣称旧工作点已穷尽所有 schedule。

APG 的本地规则为：`M=d+beta*M_previous`；先截到 `2*||d||`，再移除与 conditional clean 预测 `m=z+(1-t)c` 平行的分量；输出 `c+alpha*direction`。两次 Heun 查询读同一旧 `M_previous`，接受后只提交**左 stage** 的 `M`。它是既有 APG 的 velocity/相对范数适配，不冒充论文所有原始实现细节逐项复现，也不是新贡献。

CTRL 的本地强对照为：`S=d-previous+lambda*previous`，`modified=d-K*sign(S)`，输出 **`u+(1+alpha)*modified`**；首次 previous=d，随后 previous 是上个接受步左 stage 的 **modified gap**。它与 `c+alpha*modified` 不同；也不能将历史改为原始 gap、每 stage 立即更新，或把 lambda 改成物理时间率却仍标为同一基线。源码 [portfolio operators](/home/zhoushunyu/eqvae/experiments/sit_guidance_portfolio_20260910/operators.py:264)、[原 CTRL sampler](/home/zhoushunyu/eqvae/experiments/sit_fsg_pasted_20260910/core.py:247)、[Sep11 方法参数](/home/zhoushunyu/eqvae/experiments/sit_fsg_ctrl_hypothesis_20260911/core.py:15)。

## 2. CFG：真正可比的历史数值

| 同一确认 bank | 标准 CFG | APG | 额外预算的历史候选 | 证据边界 |
|---|---:|---:|---:|---|
| Portfolio 新 5K，noise 202610060 / labels 202610061 | 22.478488 | **21.533588** | 条件割线 22.180625；通道回缩 22.370812 | 所有配置由此前 1K 固定；每类 50 图 |
| Fusion 新 5K，noise 202610064 / labels 202610065 | 22.512481 | **21.651659** | **割线+APG+回缩 21.228058** | 另一组 1K 选参后冻结，另一组 5K 确认 |
| Control53 新 1K，noise 202610100 / labels 202610101 | 45.707341 | **44.503871** | **CTRL 44.725613** | 同一 bank 的调参结果；没有对应 CTRL 独立 5K |

5K 对照的 CFG alpha=1.25、APG alpha=2/beta=-.5，均 224/0。Fusion 配置 `cfg_secant_apg_rescale_11` 为 alpha=3、condition_scale=.5、apg_beta=-.5、channel_rescale=.75，320/0；顺序是条件割线并配平原 gap 长度 → APG → conditional clean 通道标准差回缩。相比当批 APG，FID 降 `.423600`（1.956%），累计采样/解码 batch GPU 秒从 850.609 到 1171.200（1.377 倍），IS 从 65.970993 降至 64.420044，sFID 64.621146 → 64.650615。因此不是所有指标或同成本占优。[Portfolio 5K 报告](/home/zhoushunyu/eqvae/docs/SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_RESULTS_20260910_ZH.md)、[Fusion 5K 报告](/home/zhoushunyu/eqvae/docs/SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md)。

Control53 的三个数已经直接对照各自原始 `result.json`：

- `cfg_native_04`: alpha=1.25。
- `cfg_apg_07`: alpha=2、beta=-.5。
- `cfg_smc_control_07`: **alpha=2.75**、K=.2、lambda=5；224/0；FID 44.7256126594，sFID 207.7539577952，IS 62.5293922424。

三者 noise SHA256 都是 `ce4f48e589a64f6bf8570a11496791763d6485299e7cb3d8bbcd3741ee327dca`，labels SHA256 都是 `1a00060ee2bdc21fc6951f392854bfe88ffd832279dbe2b5122c68becc2da8e0`。CTRL 相比标准 CFG 降 `.981728`，但比 APG 高 `.221742`。不能把它的 alpha 误读成标准 CFG 的 1.25。原始目录：[control_screen_1k](/home/zhoushunyu/data/eqvae/experiments/sit_control_output_50ideas_20260910/control_screen_1k)，CTRL 请求 SHA256 `073b1c9b69f3294556aaf8d8b0c32868af4e25c48448222dca8dcdd5b8a1c631`。

Fusion 5K 的 noise SHA256 为 `7d176fe58ddf2478dacade9d7d25a08be66020503b7dd2fbbb19c09d6abce0a3`，labels SHA256 为 `a35777ffeb7971266348ec20741851f76f359c2ee62f84514d37df59bafe0bff`，请求 SHA256 `916bb26a690edf944ec533e6ebea01c26530049b62441fd2a4e04c4af2d6fdcf`。原始目录：[selected_5k](/home/zhoushunyu/data/eqvae/experiments/sit_guidance_fusion_20260910/selected_5k)。历史所有累计 GPU 秒不含加载/ADM 评估，并不等于多 GPU 墙钟。两组确认和缓存特征复算支持“值得保留”的排序，但未建立训练种子泛化或正式置信区间。

## 3. IG：当前最可信的低额外成本正信号

**Sep12 Context 弱读出是这条线当前最值得复用的结果。** 独立 400 筛选 → 独立 1K → 参数和 3000 步 EMA 固定后的新 5K（seed 2026121331，每类 50 图）：

| 方法 | FID5K | full / extra prefix | 解释 |
|---|---:|---:|---|
| 原 IG | 40.281731 | 128 / 0 | extra peak .8；早段乘 6/7，t<.5；64 Heun |
| ADG | 39.577502 | 128 / 0 | 既有方法强对照 |
| 原 IG 66 步 | 40.273430 | 132 / 0 | 额外主干预算对照；固定物理时间窗口，离散落点略变 |
| Context 弱读出 | **37.234255** | 128 / 0 | 比 ADG 降 2.343247，比原 IG 降 3.047476 |

后续把原捕获方式改成直接替换 depth4 读出，24 条完整轨迹的 latent/像素逐项一致。新头 304,528 参数、旧头 301,840，净增 2,688；单独一组同卡 B8 三次轮换计时，直接替换比原 IG 中位数高约 `.29%`。这组同卡计时与并行 5K 采样计时必须分开，不能跨批相除。它通过固定效应门槛，但报告明确没有统计显著性、跨模型成功、新颖性或误差机制证明。[5K 结果](/home/zhoushunyu/eqvae/docs/CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)、[协议](/home/zhoushunyu/eqvae/docs/CONTEXT_REFERENCE_5K_PROTOCOL_20260912_ZH.md)、[直接替换入口](/home/zhoushunyu/eqvae/docs/CONTEXT_REFERENCE_DIRECT_READOUT_20260913_ZH.md)。

Sep13 等训练控制进一步显示：原架构头重训 3000 步也有收益（同 bank FID 67.0089 → 65.7192），MLP 为 63.7752。因此不能把全部改善都归于 MLP 非线性；这项后补 1K 复用已有对照，不是第二次独立确认。[等训练报告](/home/zhoushunyu/eqvae/docs/IG_READOUT_MATCHED_CONTROL_RESULTS_20260913_ZH.md)。

还有独立 5K 的额外 prefix 路线：native weak-prefix 36.173207 vs 同批原 IG 重算 prefix 的实现 40.431806，表中两者都用了 128 full + **64 extra prefix**；高效共享的原 IG 可以省下这 64 次。它数值更低，却违背此前不加弱前向的成本取舍，不能用这个数充当 128/0 的低成本基线。[prefix 确认](/home/zhoushunyu/eqvae/docs/SIT_PREFIX_CONFIRMATION_RESULTS_20260912_ZH.md)。早期 IG 局部注意力及局部+残差也有独立 5K 正信号，但都需额外 64 prefix，且 sFID 变差；不是未做过的新方向。

上述 IG 正信号**没有转化为更强 CFG 的证据**。Sep13 JiT 的 CFG+MLP 应用未胜过匹配附加强度的普通 CFG（39.4083 vs 38.6909，同次 1K）；模型不同，这只限制迁移主张，不能直接作为 SiT 数值排序。[JiT 应用](/home/zhoushunyu/eqvae/docs/JIT_READOUT_CFG_APPLICATION_RESULTS_20260913_ZH.md)。

## 4. 已失败、已覆盖及尚未完成的边界

| 路线 | 已有证据 | 对新搜索的含义 |
|---|---|---|
| FSG 二十项一致性/未来残差构造 | **256/256 完成**；200 候选最佳 44.346356 vs APG 44.116467；最佳成本 2.378× vanilla | 这批具体参数未过强对照；旧状态页的 203/256 已过期。不能把类似未来一致性重新命名成未试方法 |
| 控制输出 53 个方向 | **317/709 已完成后暂停**；完成部分未胜 APG 44.503871 | 没跑的参数/方向不能写成全部排除；原队列不是本轮应自动恢复的任务 |
| 高强度 FSG SiT adapter | FID1K 68.1885，强 CFG 54.6481；同长度方向/数值往返等控制已做；不少早期残差读数改善未转化为末端收益 | 强前/弱逆写入、去数值回环漂移、局部终点拟合等已是历史实验，需指出新的区别 |
| Sep12 CFG posterior/null 替代 | 同探索 1K 最好 data-head 45.0569，APG 44.5038；posterior teacher 45.6688 | 没有超强基线；仅证明某个头相对 vanilla 的小改进不足以推进 |
| Sep13 完整 null 读出训练 | 新 400：MLP 77.9967、vanilla 77.4335、APG 76.6516；三种重训头均未过门槛 | 已停止，不存在独立 1K/5K 正信号 |
| Sep12 shared-push 双路径 CFG | 新 400：vanilla 77.8542，shared-independent 84.3138，opposite 87.7784 | 两路径共享算子没通过 gate；224 次/分支不等于 224 次/独立主图（主图只取首分支，实际生成两条） |
| Sep13 invariant / CTRL 投影 | 新 400：vanilla 79.7772；证据反馈 78.567 vs 时间均值强度 78.657，仅差 .091；K=.03 作者版 83.446 | 暂无可信结构性 FID 收益；语义反馈有独立分类器辅助信号，但额外解码/分类成本约 46%，q 也不是真实 noisy posterior |
| Sep13 K=.3、lambda=.05 CTRL | FID400 208.137；其 gap 投影/物理历史也退化 | 这是特定 velocity 量纲/参数的移植失败，不能据此否定 lambda=5、K=.2、alpha=2.75 的历史调优 CTRL |
| 多轮加新高斯噪声递归 | 暂停时只完成 9/48 个配置轮次；同旧 bank 亚 0.2 FID 差，未独立确认 | 用户已否定其作为 no-op / identity 的任务语义；不能把新噪声重采样包装成保持原图的证据 |

来源：[FSG20](/home/zhoushunyu/eqvae/docs/SIT_FSG_20_IDEAS_RESULTS_20260910_ZH.md)、[Control53](/home/zhoushunyu/eqvae/docs/SIT_CONTROL_53_IDEAS_RESULTS_20260910_ZH.md)、[FSG/CTRL 机制实验](/home/zhoushunyu/eqvae/docs/SIT_FSG_CTRL_HYPOTHESIS_RESULTS_20260911_ZH.md)、[posterior reference](/home/zhoushunyu/eqvae/docs/CFG_POSTERIOR_REFERENCE_RESULTS_20260912_ZH.md)、[null 读出](/home/zhoushunyu/eqvae/docs/CFG_NULL_READOUT_RESULTS_20260913_ZH.md)、[shared push](/home/zhoushunyu/eqvae/docs/CFG_IG_PASTED_EXPERIMENTS_20260912_ZH.md)、[invariants](/home/zhoushunyu/eqvae/docs/CFG_INVARIANTS_RESULTS_20260913_ZH.md)、[递归暂停结果](/home/zhoushunyu/eqvae/docs/RECURSIVE_GUIDANCE_FOCUS_RESULTS_20260913_ZH.md)。这些具体失败都不构成普适不可能性证明。

## 5. 可直接复用的采样与评价入口

模型由 [common.runtime](/home/zhoushunyu/eqvae/experiments/guidance_pasted_20260912/common.py:22) → [Runtime](/home/zhoushunyu/eqvae/experiments/lifting_scale_sweep_20260909.py:88) 加载：

- `/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0/checkpoints/step_00800000.pt`，EMA，SiT-S/2。
- runtime 还加载旧 depth4_v head，但本次 CFG/APG/CTRL 均只用条件/null full 输出，不使用该弱读出。
- `stabilityai/sd-vae-ft-mse` 本地 VAE，4×32×32 latent、256×256 RGB；保留 `rt.decode` 的量化，不自行换后处理。
- null label=100；不是 ImageNet1K 的 1000。runtime 将可见 GPU 的逻辑 0 设为 device；物理卡由 `CUDA_VISIBLE_DEVICES` 控制。
- SiT runtime 使用 FP32、TF32 enabled、matmul precision high，不开 autocast；本轮需固定它，不能在不同 arm 无意改变 dtype/TF32。

按父任务追加的私有 [baselines.py](/home/zhoushunyu/eqvae/experiments/cfg_transport_search_20260913/baselines.py) 已实现：

```python
from experiments.cfg_transport_search_20260913 import baselines
rt = baselines.make_runtime()
result = baselines.sample(rt, noise, labels, {
    'kind': 'ctrl', 'alpha': 2.75, 'steps': 64, 'cutoff': .75,
    'lambda_ctrl': 5., 'K': .2,
}, snapshots=True)
pixels = rt.decode(result['latents'])
# counts = {'full': 224, 'prefix': 0}; per output, not batch-multiplied
```

其他 kind 为 `cfg` / `apg`，APG 用 `beta=-.5`；steps 支持 64/96。96 步与相同 cutoff 对应 336/0。`alpha=0` 三类都退回纯 conditional，64/96 步分别 128/192 次。snapshots 为初始、1/4、1/2、3/4、终点的真实 latent；trace 为 `(steps,batch,6)`，列名由 `TRACE_COLUMNS` 给出。这是单次生成轨迹，不是回灌轮次。

**源码冻结 SHA256：`40468591b9cbbf90b01ebf5bf5c142c3b82b7ca93944929c6f690beb8cf66130`。** 在 CPU 非线性假模型上，FP32/FP64 的 CFG/APG/CTRL 三种完整 64 步轨迹均与旧实际 sampler 逐元素相同；96 步计数、三种零系数退化、schedule 落点及 snapshot 覆盖均通过。[CPU 检查记录](baseline_cpu_check.json)。

随后授权使用 GPU0 补验：新 bank 前 8 个输入、相同真实 SiT 权重，`rt.field` 与 `rt.pair` 的 full 输出在 t=0/.375/.75、条件/null 六组均逐元素相同；CFG alpha=1.25、APG alpha=2/beta=-.5、CTRL alpha=2.75/lambda=5/K=.2 的**完整 64 步轨迹 latent 与解码 uint8 像素也全部逐元素相同**，新旧均 224/0。核查脚本使用真实旧 `rt.pair` 和训练弱头，只为三个算子不读取的旧诊断 `Queries.h` / `portfolio_assets` 提供空占位，未改模型/算子。含加载 16.6 秒，进程结束后立即释放 GPU0；不将核查耗时当作性能测量。[真实模型检查记录](baseline_model_check.json)。该检查没有改冻结 sampler，也不新增 FID 或质量收益结论。

历史融合可用 [fusion.configurations/sample](/home/zhoushunyu/eqvae/experiments/sit_guidance_fusion_20260910.py:68)，选择 `cfg_secant_apg_rescale_11`；它依赖 portfolio runtime 的资产/捕获与进程内 dispatcher 安装，应在独立进程调用其原 runtime，不把普通 common runtime 不经适配直接传入。Context IG 可用 [reference.install/sample](/home/zhoushunyu/eqvae/experiments/context_reference_5k_20260912/reference.py:12)，其接口会验证保留 3000 步头的训练请求与 hash，不需要重训。

统一 ADM 参考为 `/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz`；真实 reference 固定 **ImageNet100 validation 5000**，不是 ImageNet1K/50K。Inception graph 为 `/data/shared/adm_refs/classify_image_graph_def.pb`。推荐直接使用 [compute_adm_fid.py](/home/zhoushunyu/eqvae/experiments/compute_adm_fid.py)，避免旧 guidance_pasted evaluator 的固定 ROOT 和双路径约定：

```bash
/data/shared/envs/adm-fid/bin/python experiments/compute_adm_fid.py \
  --reference /home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz \
  --samples /absolute/new_arm/samples.npz \
  --output /absolute/new_arm/fid.json \
  --activations-output /absolute/new_arm/inception_activations.npz \
  --batch-size 32
```

NPZ `arr_0` 为 uint8 NHWC。图像采样环境为 `/home/zhoushunyu/miniconda3/envs/myenv/bin/python`。**400、1K、5K 的 FID 不可直接排名**；参考相同也不能消除生成样本量偏差。旧 baseline 相同像素重提 Inception 曾产生约 `.002` 的差异；缓存特征 FP64 复算检查的是算术/聚合，不是独立特征抽取器验证。

## 6. 新一轮怎样才能可靠地称为超过强 baseline

1. 在新的配对、100 类均衡 1K bank 上，同时运行 vanilla CFG alpha `1/1.25/1.5/1.75`，APG alpha `1.5/2/2.5/3`，CTRL alpha `1.25` 和历史最佳 `2.75`（均 lambda=5、K=.2）。当前计划中的 APG96 alpha=2/2.5 为额外预算对照。所有配置用一图一条路径，保存相同输入的 hash。
2. 对新 transport 候选给相同选参机会并报告完整表。超过这些标准工作点才有初步价值；如果候选成本接近/超过 320 full，还要在同一新 bank 对照保留的割线+APG+回缩。仅超过 APG64 不能自动推出更好计算效率；96 步比较也不是精确等墙钟。
3. 在 1K 看完后固定方法、强度、窗口、模型、求解器、checkpoint，再运行**全新** 5K 配对 bank 的胜者与最强对照，不在该 5K 调参。若窗口是候选自由参数，vanilla/APG 也应得到公平的窗口搜索预算。
4. 记录 FID/sFID/IS、无筛选固定 seed 图、类别符合度/多样性，以及实际 full/prefix/解码/分类调用。把单卡轮换完整采样+decode 计时单独完成；多 GPU 总 batch 秒和并行墙钟不能混报。
5. 需要可靠误差范围时，使用同类配对的 feature bootstrap 报 FID 差区间；bootstrap 固定当前 reference，不代表 reference 不确定性或跨训练种子。更有说服力的是第二组独立 5K 噪声复验。不要把小差值、局部 residual 下降、分类器分数或“理论保持”单独当成质量成功。

这轮新结果应以自己的固定请求/source/asset hash 落在新目录，不能覆写旧实验。当前最强支持是 APG 的重复 5K 收益、额外预算 CFG fusion 的一次 5K 收益，以及 SiT Context IG 的一次独立 5K 收益；尚没有证据把此次 transport 构造列入其中。

## 7. 相关工作短注：Round-Trip Consistency 的时钟区别

Scheinker 的 [Round-Trip Consistency, arXiv:2608.00675](https://arxiv.org/abs/2608.00675) 首次提交于 2026-08-01，当前摘要页列出 v2 修订 2026-08-13。本文核读的 [v1 第 4–5 节](https://arxiv.org/html/2608.00675v1#S4) 以同一模型的方向标志学习 `p(z_(t±1) | z_t,z_(t∓1),direction)`：**这里 t 是物理过程/视频帧时间，k 才是 diffusion denoising step**。正向预测 i 帧后，从末端预测帧对反向预测 i 帧，比较返回帧对与真实起始帧对；每一帧转换内部采用确定 DDIM。它不是把同一去噪 ODE 的时间积分反过来，也没有定义“高 CFG 正向、低 CFG 反演”。

论文把回环差作为误差检测/选择性预测信号，并未用它修正生成图像。其第 5.1 节界限为 `sqrt(E_i) <= sqrt(2)*mu^(-i)*(sqrt(C_i)+delta_i)`；依赖反向模型在访问区域的 co-Lipschitz 下界及其从真实终点返回的误差 delta。**我们的推论**：不能去掉这些条件，把任意 CFG 回环差下降当作质量提升；两个错误但相互抵消的映射仍可有零回环差。对本仓库，它提供的是“先独立验证回环读数是否预测外部误差”的方法学先例，不直接证明 CFG transport 候选，也不构成去噪时间双向 guidance 的同题先例。[原文第 5 节](https://arxiv.org/html/2608.00675v1#S5)。

# RAEv2 guidance 的时间求积：重新启动后的固定机制实验

2026-09-07。用户本次明确要求继续 guidance 研究，目标为配对 5K 的 FID 相对下降约 3%。旧研究的停止记录和阴性结果保留。本轮首先检验一个尚缺生成结果的机制分解，不声称已达到目标，也不把数值积分公式宣称为新颖性。

## 从 fixed-point 论文获得的设计判断

重新阅读 [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/html/2510.21512v1) 的方法、假设和局限：将校准与去噪拆开，才可能判断跨时间操作改变的是哪一部分。论文给出的收缩性与平均预测差控制并不是 FID 定理，且 conditional/unconditional 的语义不直接转移到 RAEv2 的 Full/Base。

现有 PFR 在 RAEv2 的失败也不能单独区分“新参考方向没有质量意义”与“有限时间操作改变了原指导场”。本实验不求 strong/weak 一致，而在原 100 次当前状态查询中分开检查 Full 和 guidance 的时间变化。它是质量机制对照，不减少调用或复活旧多速率加速/代理训练提案。

## 有限步公式及其边界

RAE 的桥为 z_t=(1-t)X+tε，数据方向为 t 下降，原场 dz/dt=(z-G)/t。令 λ=log((1-t)/t)，则在内点

    d(z/t)/dλ = exp(λ) G。

从 t 到 s<t 的精确变常数公式为

    z_s = (s/t) z_t + (1-s) ∫₀ʰ exp(u-h) G(λ_t+u) du，
    h = λ_s-λ_t。

把 G 沿轨迹保持为常量，得到原 Euler：z_s=(s/t)z_t+(1-s/t)G_t。用两个已有查询估计 G 对 λ 的一阶导数，可以精确积分该线性插值：

    G_eff = G_t + k (G_t-G_prev)，
    k = [h/(1-exp(-h))-1] / (λ_t-λ_prev)。

这正是 [DPM-Solver++](https://arxiv.org/abs/2211.01095) 的数据预测二阶多步 Taylor 型求积在此桥上的写法（diffusers 对应 Heun 型系数；作者源码命名为 `solver_type='taylor'`，区别于其推荐默认 midpoint 型 `dpmsolver`）。系数来自积分核，不由 FID、局部回归或时间表拟合。完整 G 的平滑内点方法为二阶；只补一个分量的两条实验臂一般仍为一阶，不能宣称它们有完整二阶收敛性。

原生 G 使用 BF16 的 `B+1.78*(F-B)`，之后转 FP32；定义实际 guidance 增量 A=G-F_float，包括原有舍入残差。逐步保持 F+A=G，避免用另一种混合精度冒充新机制。

四个预定实验臂为：

| 名称 | G_eff 的新增项 | 用途 |
|---|---|---|
| official | 0 | 原始 100 步控制 |
| guidance_2m | k(A_t-A_prev) | 只补 guidance 的时间变化 |
| full_2m | k(F_t-F_prev) | 只补 Full 的时间变化，机制控制 |
| exponential_2m | k(G_t-G_prev) | 既有二阶求积，数值控制 |

第一步 t=1、没有有限前一 λ 的第二步、末步 s=0、跨原始 t=.1 窗口的步均使用原 Euler，不对无穷 λ 或不连续函数外推。全部保持原时间网格及 t_eps=.05。这个边界规则使完整轨迹不能直接套用一个全程平滑的二阶误差定理。

可证伪预测：若 guidance 的时间滞后有实际质量影响，guidance_2m 应在配对 5K 上产生有意义的改善，而且与 full_2m 的效果有区别；若仅完整二阶积分有效，则得到的是已有 solver 的迁移结果。若各臂均无收益，停止本组固定求积，不增加时间系数搜索。更准确积分网络 ODE 不必产生更好的图像，FID 必须实际测量。

## 冻结评测与复现

- 官方 EMA100080、DINOv3-L K7 decoder、原 100-step shift8 Euler 网格、原 IG1.78 区间 [.1,1]、B8、native BF16 mix / FP32 state、TF32 on。与原 `weak_confirm5k` 一致。
- 首次 5K 用 seed202609072、每类五图，沿用按全局 batch 的 SHA256 初始噪声命名空间。这个 bank 以前已用于探索，本次只作发现集；不能称为独立确认。
- 先执行 8 图全部四臂 smoke，检查解析积分测试和官方 8 图对历史 5K 首八图的像素一致。通过后运行三条新臂全 5K，复用已完整保存的原 official 5K。合并后逐 batch 对历史噪声/标签 SHA，并核对配置、模型、decoder、统计、网格与算术。
- 目标分数仍是 1-FID_new/FID_official≥.03，对当前原 FID6.9497684777 的门槛为6.7412754234。不以 1K 选择或子样本挑选替代 5K。
- 预先保留 seed202609073 作为之后的独立 5K 确认：只有发现候选达到或接近目标并有进一步验证价值时，冻结同一方法，与新 official 成对生成。已有二阶 solver 比候选更好时必须披露，不以弱对照制造新的方法贡献。
- 每图100次主模型调用、一次图像decoder、无训练或额外网络。仍实测时间和历史缓存开销，NFE相同不冒称速度完全相同。
- 官方 nanogen evaluator、ImageNet256 reference、B64；同时保存 IS、图片/特征、完整负结果、源码与输入身份。

代码：[求积](../experiments/raev2_guidance_quadrature.py)、[采样](../experiments/sample_raev2_guidance_quadrature.py)、[四 GPU 驱动](../experiments/run_raev2_guidance_quadrature.py)、[解析测试](../tests/test_raev2_guidance_quadrature.py)。

数据根目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_quadrature_20260907/`。大样本留在数据盘；Git 保存理论、代码及轻量证据。

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_raev2_guidance_quadrature.py
    python experiments/run_raev2_guidance_quadrature.py --output /home/zhoushunyu/data/eqvae/experiments/raev2_guidance_quadrature_20260907/discovery5k --samples 5000 --seed 202609072 --modes guidance_2m full_2m exponential_2m

## 执行记录

解析测试10项通过，包括独立积分参照、制造解二阶收敛、原生BF16算术、端点和窗口处理、分量可加性。它们验证实现与所写公式，不构成质量证据。真实模型 smoke 与5K结果继续追加于此。

真实四臂 smoke8 已完成；official 首八图与旧 5K 首八图逐像素一致，配置/模型/decoder/统计/版本/网格及噪声类别身份全部一致。三条新臂的完整5K及独立审计均已完成。`smoke8_parity.json` 保存核对结果，全部源码在每个 run 的 `frozen_source/` 中保留。

正式 FSG PDF（37页）、DPM-Solver++ 原文及作者采样源码、TPG 正式原文已保存至 `readings/` 并记录来源和SHA。下载不等于全部读完：FSG 本次已核读§3、附录B和C的假设/主定理；DPM 作者 `multistep_dpm_solver_second_update` 的实系数逐式核对。其余阅读范围另行追加，不按文件数声称完成论文数。

## 等待采样时的再读：直觉与证据各自承担什么

再次逐段核读 FSG §3.1–3.2、附录A.5（式31、表9）与附录B。其设计出发点是：如果当前 latent 在参考分支下已能走向与条件相容的图像，后续生成便不需要大幅改道。先构造较长区间的条件原型，再借参考分支的逆向过程把信息带回当前状态，是这一判断的算法化。固定点提供了组织算子与分配计算的语言；理想路径为何值得靠近，仍然需要另外的语义依据。

附录B的实验证据来自同一张条件生成图像的无条件反演，再比较匹配与不匹配的 prompt。它支持条件相容性的直觉，不能直接证明同一条件下 Full/Base 一致就是质量目标。

附录A.5测量的是随机噪声扰动下的平均平方距离比。局部线性化为矩阵 J 时，等方差小扰动的平均方向比对应 ||J||_F²/d，而保证所有方向收缩需要 ||J||_op<1。例：J=diag(2,.99,…,.99)，d=256，平均平方比约.9919<1，但首方向放大两倍。这只是说明两种证据不同，并不是断言论文实际算子存在该坏方向。对高维 RAEv2，不能把整体 gap 或随机方向平均收缩直接当成语义方向稳定、乃至 FID 改善的充分证据。

用户进一步明确一个想法一个想法地推导与试验。本轮先完成求积机制对照；关于空间相关残差的想法仅留作待讨论假设，未拟合、未启动采样。

## 完整配对5K结果：本机制实验结束，目标未达到

| 方法 | FID5K ↓ | 相对原始FID改善 | IS ↑ | 实测推理时间比 |
|---|---:|---:|---:|---:|
| official（历史配对基线） | 6.94976848 | — | 157.57570 | 1.00000 |
| guidance_2m | 6.94955272 | +0.00310% | 158.23040 | 1.00043 |
| full_2m | 6.95026566 | −0.00715% | 155.36181 | 1.00843 |
| exponential_2m | 6.96545914 | −0.22577% | 155.74308 | 1.00658 |

每臂5000张、500000次主模型调用。时间为四个shard实际轨迹与decoder耗时之和，基线来自此前同硬件协议运行，不能据千分位差别声称速度优势。

独立审计已逐图核对合并与分片、逐batch核对全部625组噪声/类别、核对模型/配置/decoder/统计/网格/精度以及冻结源码。用对称PSD特征值公式独立复算的FID与官方结果最大差小于1e−11。审计与完整指标保存在 `experiments/results/raev2_guidance_quadrature_20260907/`，原始图像与特征保存在上述数据目录。

结论限于本次固定公式和原100步协议：guidance时间变化校正的FID与基线几乎相同，完整二阶积分也未改善，不支持把此类时间求积误差视作达到3%目标的突破口。该结果不证明所有时间机制无效，但足以结束这一个想法；不追加步数、强度、窗口搜索，也不消耗独立seed202609073确认一个近零效应。保持完整阴性结果，继续推导下一个机制。

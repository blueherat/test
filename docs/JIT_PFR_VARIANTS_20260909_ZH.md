# JiT PFR 固定IG工作点的替代配置

用户要求继续尝试其他PFR配置，不能据单个迁移参数严重恶化判整条路线失败。
固定depth4 50K EMA、IG前半gamma=.3后半0、原1K噪声/标签/求解器/精度。
复用IG55.4803259及原projected rho1 h1/32的110.2309827，不重跑全量基线。

只改变未来弱查询或修订强度：G_new=G+rho*(1+gamma)*(W-W_future)。
九组：projected h1/32 rho=.05/.1/.25/.5；time-only h1/32 rho=.1/.25/1；
projected和time-only h1/128 rho1。所有h均截断在data-time .5，不改变IG窗口。
time-only保持原z，只把弱查询时间推进；projected沿原正向射线构造q。
两种都将未来clean输出在实际查询点(q,t+h)转换为velocity，floor=.05，
不能在当前z,t上错误换算未来clean，也不能照搬RAE相反时间方向。

四卡协同一组，九组顺序执行，每组1K并官方评价。保存参数/源码/输入/输出
哈希和实测成本。rho0必须等于原IG；保守rho/h是预先固定的搜索，不预设能救回。
原完整PFR前4张按冻结公式复现，核对旧像素；沿IG轨迹记录修订/IG速度的RMS比，
查询位移RMS和投影系数，作为尺度线索，不用机制指标替代生成评价。
此轮不自动扩展其他depth、CFG、lifting或额外训练。1K改善需独立确认。

## 完成结果

九组均已完成，队列状态为 complete；检查时没有 JiT 训练/采样进程，四卡计算利用率为 0%。所有九组的噪声和标签哈希相同，也与复用的 IG 基线相同；九个 samples.npz 均存在。以下为原评测记录，未重新生成基线或重算 FID。

| 查询方式 | 修订强度 rho | 前瞻 h | FID-1K，越低越好 |
|---|---:|---:|---:|
| 原 IG（复用） | 0 | — | **55.480326** |
| projected | .05 | 1/32 | 56.855438 |
| projected | .10 | 1/32 | 58.403123 |
| projected | .25 | 1/32 | 64.284249 |
| projected | .50 | 1/32 | 78.504181 |
| projected（旧结果复用） | 1 | 1/32 | 110.230983 |
| time-only | .10 | 1/32 | 58.551562 |
| time-only | .25 | 1/32 | 64.664540 |
| time-only | 1 | 1/32 | 112.508794 |
| projected | 1 | 1/128 | 64.630479 |
| time-only | 1 | 1/128 | 65.248406 |

最保守的新变体也比 IG 高 1.375112 FID，约恶化 2.48%。projected 在已测正强度上随 rho 增大持续恶化。去掉空间位移、缩短前瞻距离均未产生正结果，因此当前失败不能仅归因于旧 rho=1 太激进。这不证明所有 PFR 定义、训练或工作点都无效。

同 rho 下 projected 与 time-only 的 FID 接近，只是终点评分的描述；不能由此断言两条轨迹接近或空间位移无贡献。rho=.25,h=1/32 与 rho=1,h=1/128 的 rho*h 相同，结果也接近，符合局部时间响应占比较大的假说，但不能单靠 FID 验证向量 Taylor 近似。

新变体每张图均为 100 次 full 加 50 次 prefix 调用。记录的 batch GPU 时间合计为每组 478.11–504.15 秒，原 IG 为 411.85 秒，即约 1.16–1.22 倍；此比值不是含评测与启动开销的端到端墙钟比。该组配置不扩至 5K。

原件：[九组结果](/home/zhoushunyu/data/eqvae/experiments/jit_pfr_variants_20260909/results.json)、[完成状态](/home/zhoushunyu/data/eqvae/experiments/jit_pfr_variants_20260909/status.json)。原 IG 与此前 PFR/lifting 的逐组 result.json 在 `/home/zhoushunyu/data/eqvae/experiments/jit_fine_sweep_20260909/`。

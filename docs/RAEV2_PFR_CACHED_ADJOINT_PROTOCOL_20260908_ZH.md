# 使用已有完整伴随检查 PFR 的类别对齐响应

固定复用 20260906 endpoint_adjoint_response_v1 的全部 8 条轨迹、
类别原型终点量和完整后续伴随。旧控制器的有限响应失败不撤销，
本次不重新部署它，不训练或扩大质量实验。

在每个缓存状态重新计算原生 FP32/无 TF32 Full/Base，要求下一步
状态哈希与缓存逐位一致。缓存 states/adjoints 文件按原 summary
逐个验证。任一不一致即停止，不放宽阈值继续作结论。

只对 noise-time t>.5 计算原始 time-only PFR，future=max(.5,t−1/32)，
速度修正 d=1.78[W(z,t)−W(z,future)]。Euler 写入位移为 −dt*d，
对每张图计算 Σ_k <a_post,k,−dt*d>，保留全部时刻带符号贡献。
这是固定有限 horizon、修正幅度在零点的一阶导数，不是小 horizon
近似，也不是 rho=1 时的真实终点变化或 FID。终点量来自独立真实类原型；
不使用本次生成样本或评价 FID 均值拟合新目标，不据结果调整参数。

预计 800 次 B1 Full 和约 712 次 B1 prefix，另有权重/缓存哈希开销；
不加载 decoder，不新增反向传播。成本和调用数按实际运行记录。
实现 `experiments/audit_raev2_pfr_cached_adjoint.py`。

## 后续导数校验（运行前固定）

缓存诊断完成：800 个下一状态逐位一致；800 Full、712 prefix，
计算段 25.30 秒。8 张图的一阶响应符号为正5/负3，幅度差异很大。
仅凭它不解释有限修正的收益。

固定在相同 8 个初始噪声上重放 rho=0、±2^-12、±2^-13，
每次都在当前实际状态重新计算 PFR。比较两组中心差分与缓存伴随，
不继续搜索更小或更大的幅度，不选择 rho 用于部署。原生 FP32 基准
逐步哈希检查保持不变；终点使用相同连续 FP32 decoder/Inception
和原始独立类原型。预计 4000 Full、2848 prefix、40 次终点评价，
无反向传播。这仅检验有限精度下该方向的导数可靠性。
实现 `experiments/audit_raev2_pfr_adjoint_difference.py`。

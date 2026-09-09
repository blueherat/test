# RAEv2 后验耦合弱参考：固定 1K 探索

在真实弱头重复性检查后决定执行此探索。不是独立验证，不将估计器
重复性当作质量保证。保留初始 seed202609413/B4、1000类各一张、原模型、
decoder、BF16/TF32、IG1.78 原时间区间和 shift8 网格。

两组：posterior 为原生100步 + 前半程每步一对 antithetic prefix；
ordinary150 为150步原生 IG。两组共享初始噪声与标签。后者增加普通
求解计算，以防仅因额外计算宣称收益；不是精确 FLOPs 相等，必须报告
实际耗时。也与已完成100步原生 1K FID38.264239比较。

posterior 在 noise t>.5 时固定 r=max(.5,t-1/32)，不根据重复性结果
挑后段。A、B、V_noise 沿候选公式，center=A*z+B*base_clean，
q_±=center±sqrt(V_noise)*eta，future_mean=(base(q_+,r)+base(q_-,r))/2。
额外速度为 `-(1.78/t)*(base_clean-future_mean)`，叠加原生 IG velocity。
不恢复 raw PFR 范数、不做投影，不挑符号或强度。
query RNG固定独立 seed202609434，不改变生成初始噪声流。

每样本posterior100 Full+178 prefix，ordinary150为150 Full。
RAE Full 含不同宽度的 DDT 后缀，不能只按 block 数宣称等计算。
预计两个1K合计约.7 GPU小时采样，实际计算、加载/smoke/评估分别说明。

每组先native8对既有ordinary100精确像素复现，再目标8张smoke。
posterior 在实际 q_± 上各做一次prefix/Full一致性检查；smoke须2次全头
检查均通过，再运行1000张。最终保留样本、特征、随机数和输入/模型/
源码哈希、实际调用数，独立重算FID。方法8图应等于正式前8图。

如果没有超过原生100步基准，不扩大5K或做参数补救搜索。
若有收益，先判断与150步对照的质量和耗时关系，再决定独立确认。
CDM鞅理论、祖先采样公式和Gaussian零修订均不作为新颖性声明；
此次仅检验一个具体有限查询弱参考替换是否具有方法价值。论文保持暂停。

运行状态：两个driver的native8逐像素复现和方法8均通过，posterior
实际正负query的两次prefix一致性检查通过。两组1K已开始，尚无质量
结果。独立审计脚本analyze_raev2_posterior_reference.py已准备，检查
正式前8图、配对输入、捕获源码、query RNG、实际调用和FP64 FID。

现已完成：posterior FID42.801481，ordinary15038.458631，原生10038.264239。
独立审计通过。按本协议停止质量扩展和参数补救；完整结果见
RAEV2_POSTERIOR_REFERENCE_1K_RESULTS_20260908_ZH.md。

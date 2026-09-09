# 直接标准化未来弱查询：固定1K

源码核对：现有OU-common/polar查询标准化状态构造证书，再投影raw修正；
本次直接将标准化状态用于唯一future Base查询。使用既有坐标变换函数，
不把坐标变换重新声明为新理论。

相对已完成canonical time-only rho1，只改查询状态：
`q=sqrt((r²+(1-r)²)/(t²+(1-t)²))*z`，r=max(.5,t-1/32)。
使用q和r将future clean转换velocity，按原PFR公式更新当前状态。
保持IG1.78及原区间、rho1、前半程、100步shift8、BF16/TF32、
seed202609413/B4和1000类各一张。无新时间/强度搜索，不恢复额外范数。

每图100 Full+89 prefix，与已完成raw和projected查询组相同。
预计采样约.28 GPU小时，加载/校验/评估另计。比较原生100 FID38.264239、
raw52.942740、projected52.069028及已有普通150步38.458631。
超过坏的raw对照不构成成功，至少需要超过原生，再考虑独立确认。

先native8复现原生像素，再方法8在真实future query处验证prefix/Full
一致性。所有query均检查平方范数缩放比例。然后固定1000张并独立FID
审计，核对正式前8图与smoke、源码/模型/输入/像素hash及实际预算。
如果不超过原生，不扩大5K或补救调参。论文保持暂停。

原生8图逐像素复现及方法8图检查通过，实际future prefix与Full输出
一致，逐query范数比例检查通过。会话78592完成固定1K。
独立审计已生成raev2_norm_query.csv，结果为阴性：53.458707，
原生38.264239。按预定规则关闭该试验，不扩大5K或补救调参。
详细结果见RAEV2_NORM_QUERY_1K_RESULTS_20260908_ZH.md。

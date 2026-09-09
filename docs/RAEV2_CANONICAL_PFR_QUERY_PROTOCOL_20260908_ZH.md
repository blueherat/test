# RAEv2 canonical PFR 查询补齐

## 已核清的比较缺口

SiT canonical PFR为rho=1、h=1/32、前半程修订，并将beta(S−W)投影至原生G的
正向射线后形成q。历史RAE正式5K 7.034546→7.224213是raw time-only、rho=.05，
覆盖原IG区间；它仍是有效负结果，但不能当成同一投影算子已经被测过。
`sample_raev2_pfr_retiming.py`的pathwise_first_half分支沿整个ordinary G做Euler
位移，不是上述射线投影。对experiments/*raev2*及docs/RAEV2*的源码/记录检索未
发现同一project_to_forward_ray构造。不能把“未找到”等同于任意外部实验不存在。

## 冻结对照

RAE native IG=1.78、区间[.1,1]、100步shift8 Euler、官方100080EMA/decoder、
BF16/TF32、B4、CUDA seed202609413、1000类各一张，与近期原生baseline同bank。
两组1K：time_only和projected。均在noise t>.5处使用h=min(1/32,t−.5)、rho=1。
生成方向S=−v_full、W=−v_base、G=−v_IG；beta=1.78。

    q_time = z
    q_projected = z + h * proj_forward_ray(beta*(S−W), G)
    future_time = t−h
    v_new = v_base + beta*((v_full−v_base)+(v_base−v_weak(q,future_time)))

future使用原生廉价depth8 prefix。同一次对照仅改变query state，prefix调用数
相同。保留RAE原生IG时间表，不声称整个SiT控制器（.6/.7/0）与RAE完全一致。
这也不复现旧rho=.05条件，更不以其结果反向修改旧结论。

先新native8对旧ordinary100精确像素复现；每组8张finite与输入/调用检查，
并检查第一个实际future query上廉价prefix与完整forward的base逐元素相等。
投影正向内积检查，代码/模型/输入hash留存；然后原样1K与官方FID评价。
预计两个1K约.6–.8 GPU小时采样，实测记录；无训练、不据预览调参数。
任何1K正面仍需独立5K，不将补齐基线称作创新。

## 技术检查状态

两个driver的native8对原ordinary100逐像素复现通过。两种实际future query的
廉价depth8 prefix与完整forward的base逐元素相等，两个8图smoke均完成。
1K已进入正式采样。最终审计代码`experiments/analyze_raev2_canonical_pfr_query.py`
覆盖配对输入、来源、模型、prefix预算、评价版本、像素hash与独立FID重算；
尚未执行最终审计，不将技术检查称为质量确认。

## 完成结果

| arm | 1K FID | IS | 采样秒数 | full/样本 | prefix/样本 |
|---|---:|---:|---:|---:|---:|
| ordinary | 38.264239 | 59.077 | 802.983 | 100 | 0 |
| time_only rho=1 | 52.942740 | 64.033 | 977.389 | 100 | 89 |
| projected rho=1 | 52.069028 | 64.946 | 983.175 | 100 | 89 |

新增采样共1960.564秒（.5446 GPU小时），加载/评价另计。投影相对time-only
改善.8737，但二者均远差于原生。此结果否定“只是缺少正向射线投影，所以RAE
迁移失败”的充分解释；不否定SiT自身的有效结果，也不覆盖任意RAE强度/调度。
不对这两组追加5K，不进行rho或h补救扫描。
来源、模型、配对输入、实际查询预算、评价版本与像素hash按独立审计核对，
结果表为`experiments/results/terminal_defect_20260908/raev2_canonical_pfr_query.csv`。

同时完成的同表示SiT-v/x固定对照在两主模型均获得PFR改善，见
SIT_PFR_OUTPUT_PARAMETERIZATION_PROTOCOL_20260908_ZH.md。因此clean输出不是
本次证据支持的单一障碍；仍未识别RAE参考场时间响应为何不同的完整因果机制。

# PFR条件交互与无条件时间响应：固定1K干预

用户最新澄清：PFR与fixed point是两条独立路线，开始不要混合。
当前选择继续PFR；暂停逆噪声训练扩展以及内部表征信息迁移设想，
只研究SiT有效、RAE失败的核心差异。以下分解只针对PFR自身，
不引入fixed point机制；若后续决定退出PFR，再独立研究第二条路线。

定义同一z下弱头的velocity W_c(t),W_u(t)，r=max(.5,t-1/32)：
R_c=W_c(t)-W_c(r)；R_u=W_u(t)-W_u(r)；R_inter=R_c-R_u。
原raw PFR的附加漂移为beta*R_c。两新组分别只保留beta*R_inter和
beta*R_u。普通IG场完全保留，beta分别1.35/1.78，rho1、t>.5不变。
四个输出都是同一weak head；不混用Full证书，不OU输运、不投影、不恢复
原范数，不做强度/窗口搜索。这是精确分解后的生成干预，不是新增理论。

理想兼容密度下，W_c(t)-W_u(t)与后验score有关，但velocity与score
之间存在时间因子；两时刻的差不能直接称为未经加权的后验score差。
有限神经网络更不保证密度兼容，也没有“保留条件项必然改善FID”的定理。
旧strong×condition双差阅读及条件OU投影失败继续有效。旧OU实验改变
方向后恢复raw范数，不能代替本次原幅度、原weak查询的直接交互项干预。

固定每模型两组1K，分别与自己的旧native/raw对照比较。SiT使用官方XL
checkpoint、100步Euler/FP64状态、FP32模型/noTF32、seed202609428；
RAE使用原官方模型、100步shift8、BF16/TF32、seed202609413。连续B4，
1000类各一张，官方原评价器、decoder和参考。绝不跨模型比较绝对FID。

每方法每图100 Full；SiT另150 prefix，RAE另267 prefix。两分量组即使
不使用某些输出也执行全部查询，使组间计算相同。预计四组采样合计约
1.5–2 GPU小时，加载/检查/评价另计。相较native和raw额外查询更多，
若出现质量正信号，仍需补充相应实测时间的普通步数对照和独立确认；
不能直接宣布同成本优势。本轮固定1K，不自动扩展5K或根据该bank选参数。

每组先native8逐像素复现，再方法8在实际当前/未来空标签与未来条件
查询上验证prefix/Full一致（3次额外Full，仅smoke），之后原样1000张。
保存源码、模型、噪声/标签/像素哈希、实际调用预算和评价特征；正式
前8图必须复现smoke。最终独立重算FID，并补充均值/协方差项。

脚本sample_{sit,raev2}_pfr_condition_component.py和run_pfr_condition_component.py。
这是PFR方法研究的下一项干预，不是对“信息搬入内部特征”已完成验证。
论文保持暂停。

## 当前执行状态

四组驱动26723/63668/38474/63897均exit0，正式1K和评价已完成。
独立审计42541 exit0：源码、native8、方法8/正式前8像素、实际查询
prefix/Full、噪声/标签/checkpoint、调用数、官方评价器和FP64 FID均通过。
结果见PFR_CONDITION_COMPONENT_1K_RESULTS_20260908_ZH.md。

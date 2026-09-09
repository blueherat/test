# 用户否决：未执行的蒸馏方案

用户指出PFR在目标RAEv2上未取得有效收益，蒸馏不是当前质量突破问题的解法。本方案停止；仅准备过脚本，没有启动训练或学生采样。下文保留未执行方案，不作为待启动任务。

# PFR 时间响应蒸馏：训练与质量方案

SiT官方XL已有独立5K PFR收益，但未证明对最优IG工作点领先。本方法目标是在当前同一Full前向的浅层特征上预测PFR修正，省去未来prefix，保留已有效teacher的生成行为。普通蒸馏/仿射拟合本身不作新颖性主张；需真实采样检验，而非仅报告拟合MSE。

教师：现成SiT-XL/2+IG 800epoch，depth8，FP32模型/FP64状态/noTF32，Euler100，IG1.35，t>.5追加1.35*(Base_t-Base_future)，h1/32，future=max(.5,t-h)。

学生：从当前final_layer_xr.linear输入phi读取，拟合A[phi;1]≈patchify(Base_t-Base_future)。部署时Full、Base与phi由同一完整forward得到；v=Base+1.35*(Full-Base)+1.35*unpatchify(A[phi;1])，仅t>.5激活。每图100Full，无未来prefix，一次小矩阵读出/活动步。

训练使用新seed202609601、B4、1000类各一条PFR教师轨迹，四卡协同分片。只生成高噪50步，无需完整终点或解码；共50000状态，不等于50000独立图像。目标由当前/未来head token直接相减，当前head输出反patchify与Base逐位检查。累加FP64充分统计量，单批矩阵乘FP32/noTF32。所有主模型冻结，无反传。每batch保存累计统计可恢复。

统一求解ridge：lambda=1e-4*trace(G)/d，G包括bias列；不根据验证选lambda。固定只拟合一个头。训练输入不得与已有1K(seed202609428)或5K重叠；用请求、输入hash和seed构造检查身份，不以训练误差冒充泛化。

拟合后直接对既有1K协议进行新学生方法四卡协同采样，复用ordinary100、ordinary115与PFR100图像和指标。即使拟合误差很低，也不能据此宣布生成行为等价。若质量不够，不通过扫维度/正则/窗口挽救。若质量保留，应在既有独立5K噪声上只采样学生确认成本与质量。不得声称已超过最优IG；当前不写论文。

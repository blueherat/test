# 输入局部参考：补完已冻结的生成筛选

本轮补完[原协议](IG_INPUT_LOCAL_PROTOCOL_20260912_ZH.md)中已经训练、尚未生成验证的备用方案。共同污染＋结构化错配＋加性残差仍是用户提出的主要分布假设；本试验独立检验一种信息限制，不把局部参考当作已识别的共同错误分布，也不将它重命名为新idea。纯CFG的研究目标继续保留。

旧原始目录及 `generation_deferred.json` 保留其历史状态，生成输出写入新目录 `input_local_completion_20260912`。直接加载旧3000步最终EMA的Local与Context读出，训练数据、checkpoint、结构、种子、loss、窗口、强度和判据均不改变。原训练请求与全部源数据SHA在本轮准备阶段重验。新目录的请求关联旧训练产物，禁止重新训练或按生成结果选择checkpoint。

两个模型固定各8臂、每臂400张，种子2026121307：Local原／半幅度，Context原／半幅度，原IG原／半／双幅度，Strong。Local候选需比全部6个非Local对照至少低2 FID，IS至少为原IG的90%，才进入新的1K确认；1K还必须纳入已有ADG／APG强对照，再决定是否5K。两模型可独立淘汰或晋级，400样本及RAE类别覆盖的限制照实报告。

采样调用仍为SiT每图128次full、RAEv2每图100次full，额外prefix为0。旧弱头会随原共享前向一并返回，新读出只有被当前候选使用时才求值。捕获embedding及中间特征的hook不得改变Strong／原Weak输出；Local的固定patch输出不得受外部patch变化影响；零引导须逐位恢复原Strong。安装hook并不意味着推理严格零开销，完成同卡同批量的原IG与Local／Context计时，报告额外参数和训练成本。

本轮不改变原信息约束，也不做patch大小、深度、loss或宽度搜索。Local仅胜过Context不足以晋级；若全部Local候选失败，则结束该冻结构造。旧大队列、蒸馏及反射构造均不恢复。

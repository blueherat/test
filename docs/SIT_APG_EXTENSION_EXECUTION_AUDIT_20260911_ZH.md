# APG延伸队列执行核对

核查时间：2026-09-11 13:56:35，北京时间。机制研究、实现、预检与自动排队均已完成；新增正式1K FID尚未开始。

机制实验使用32个独立新种子、3档引导、5个时间点，共480个状态观察，四个worker正常完成。全部原始响应、终点latent和汇总带有哈希。新增180个参数配置的CPU和真实GPU轨迹均通过，16个新family另完成B8整段采样；内部引导、原生CFG和旧APG三个历史参考的latent逐元素一致。第55项B8代表配置24次样本事件中接受14次，并通过实际语义和moment条件核验；这个计数只表示控制器按规则工作，不是FID提升证据。

原队列已从191组的完整提交边界恢复，核查时为193/709，四个worker均在生成当前配置，最近批次更新距核查约0.8–4.9秒。原请求及40个冻结源文件保持一致，没有遗留停止标记。

新队列共有201组，每组1K：第54–58项各12组，共60个候选配置，另有141个对照。噪声和标签与原53-idea队列逐字节一致。控制器处于`waiting_for_control53_final`，没有采样worker；必须等待原709组全部提交和最终审计通过后才自动启动。不会自动追加5K。

|进程|状态|tmux|控制器PID|
|---|---|---|--:|
|原53-idea队列|正在采样|sit_control53_0911|2636531|
|新增APG延伸|等待原队列最终完成|sit_apg_extension_0911|2656291|

新请求冻结51个源文件、12个资产和8项引用。

请求SHA256：`ae614bf3b730445db287cbdfa66df46d94a49eebbee65ba84dd04a2610ccdc6e`。

GPU预检SHA256：`5ced8e6b286087c207e1f7e6d06a413eba9cf8d0b709aee5680a8974d807479a`。

机制摘要SHA256：`e640bee139ff715bf443f5eed91587a255c2280fb743132b930a48057faabc06`。

[研究与五项推导](APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md) · [冻结协议](SIT_APG_EXTENSION_PROTOCOL_20260911_ZH.md) · [实时筛选结果](SIT_APG_EXTENSION_RESULTS_20260911_ZH.md) · [机制原始数据](data/apg_mechanism_extension_20260911/effect_observations.csv)。

完整机器记录：`/home/zhoushunyu/data/eqvae/experiments/sit_apg_mechanism_extension_20260911/readiness_audit.json`。根目录下的`mechanism_pause.json`和`preflight_pause.json`分别保存两次配置边界暂停、正常退出和自动恢复记录；已提交结果未丢失。

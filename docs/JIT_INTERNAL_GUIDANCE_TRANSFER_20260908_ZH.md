# JiT 上复现 SiT IG/PFR/OU 路线

用户要求在真正JiT上复现SiT的一套，检验是否保持效果；这不是旧SiT
JiT-style x参数化实验。真正官方代码在/data/users/zhoushunyu/research_repos/JiT，
commit cbc743a2ada5e9762697da2c83f8c4f8379e8c17。
源码/官方README：https://github.com/LTH14/JiT。

旧28GB ZIP下载中断，完整B/16成员已按local-header/descriptor恢复，CRC通过，
SHA256 4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695。
只恢复完整成员，未覆盖原包、未重新下载。模型model_ema1，checkpoint不含epoch，
不能凭文件名编造训练步数。

## 复现顺序

1. 保留正在完成的RAE lifting .9/.78，结束后释放四卡。
2. 验证JiT官方Full输出与复现接口逐位一致，生成真正JiT的配对基线
   （此前没有此模型配对样本，不属于重跑已有baseline）。
3. 冻结官方JiT-B/16主干，独立训练depth4与8的FinalLayer，同一次prefix8
   获取两层特征，depth4不含尚未插入的context tokens，depth8正确移除context。
   头输出clean RGB，在velocity空间监督，t_eps=.05，logitnormal(-.8,.8)，
   label dropout=.1；真实ImageNet训练数据[-1,1]，中心裁剪/随机水平翻转。
   每类首张1000图固定留出，不进入头训练。
4. 四GPU DDP，globalbatch256，AdamW lr1e-4，EMA.9999，50000steps；
   对应已有小SiT冻结头50000steps方案，不重训主干。每1000steps存续训状态，
   每5000steps记录固定留出prediction MSE；它不是生成质量成功的替代。
5. 头训练后按SiT顺序评估IG、PFR、OU-PFR，保留无额外引导Full和官方CFG
   的对照；新方法必须优于同协议IG，不能只优于无guidance就算PFR有效。
   先配对1K，存在清晰质量信号再独立5K及计算成本确认。

本阶段首要是JiT-B/16的可复核迁移；不能预先声称代表JiT-L/H或SOTA。
纯pixel JiT与latent SiT/RAEv2的差异不是单一表示的因果实验。
没有现成IG弱头，不能用raw hidden/最终头直接套中间层冒充训练好的弱模型。

不写论文，不复活PFR蒸馏、FK或已取消的lifting系数。
